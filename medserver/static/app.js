/**
 * MedServer — Clinical AI Frontend Application
 *
 * Handles:
 * - Server health polling & status display
 * - Chat with SSE streaming
 * - Medical image upload & analysis
 * - Markdown rendering
 * - Clinical disclaimer flow
 * - Chat persistence (localStorage)
 * - Multi-conversation history with switching
 */

(function () {
    'use strict';

    // ── Safe LocalStorage ────────────────────────────────
    const safeStorage = {
        get(key, fallback = null) {
            try {
                return localStorage.getItem(key) || fallback;
            } catch { return fallback; }
        },
        set(key, value) {
            try {
                localStorage.setItem(key, value);
                return true;
            } catch { return false; }
        },
        remove(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch { return false; }
        }
    };

    // ── Chat Storage ─────────────────────────────────────
    const STORAGE_KEY = 'medserver_chats';
    const ACTIVE_CHAT_KEY = 'medserver_active_chat';

    /**
     * Chat storage manager using localStorage.
     * Each chat: { id, title, messages[], createdAt, updatedAt }
     */
    const ChatStore = {
        _getAll() {
            try {
                return JSON.parse(safeStorage.get(STORAGE_KEY, '{}')) || {};
            } catch { return {}; }
        },

        _saveAll(chats) {
            safeStorage.set(STORAGE_KEY, JSON.stringify(chats));
        },

        getActiveChatId() {
            return safeStorage.get(ACTIVE_CHAT_KEY);
        },

        setActiveChatId(id) {
            safeStorage.set(ACTIVE_CHAT_KEY, id);
        },

        list() {
            const chats = this._getAll();
            return Object.values(chats)
                .sort((a, b) => b.updatedAt - a.updatedAt);
        },

        get(id) {
            return this._getAll()[id] || null;
        },

        create(title) {
            const id = 'chat_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
            const chat = {
                id,
                title: title || 'New Conversation',
                messages: [],
                createdAt: Date.now(),
                updatedAt: Date.now(),
            };
            const chats = this._getAll();
            chats[id] = chat;
            this._saveAll(chats);
            this.setActiveChatId(id);
            return chat;
        },

        update(id, messages) {
            const chats = this._getAll();
            if (!chats[id]) return;
            chats[id].messages = messages;
            chats[id].updatedAt = Date.now();
            // Auto-title from first user message
            if (!chats[id]._titled && messages.length > 0) {
                const firstUser = messages.find(m => m.role === 'user');
                if (firstUser) {
                    chats[id].title = firstUser.content.slice(0, 60) + (firstUser.content.length > 60 ? '…' : '');
                    chats[id]._titled = true;
                }
            }
            this._saveAll(chats);
        },

        delete(id) {
            const chats = this._getAll();
            delete chats[id];
            this._saveAll(chats);
            if (this.getActiveChatId() === id) {
                const remaining = this.list();
                this.setActiveChatId(remaining.length > 0 ? remaining[0].id : null);
            }
        },

        clearAll() {
            safeStorage.remove(STORAGE_KEY);
            safeStorage.remove(ACTIVE_CHAT_KEY);
        },
    };

    // ── State ──────────────────────────────────────────────
    const state = {
        activeChatId: null,
        messages: [],
        isStreaming: false,
        serverReady: false,
        modelInfo: null,
        attachedImage: null,
        attachedImageData: null,
        healthPollTimer: null,
        disclaimerAccepted: safeStorage.get('medserver_disclaimer') === 'accepted',
    };

    // ── DOM References ────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    let els = {};

    function setupElements() {
        els = {
            loadingScreen: $('#loadingScreen'),
            loadingSubtext: $('#loadingSubtext'),
            disclaimerModal: $('#disclaimerModal'),
            disclaimerAccept: $('#disclaimerAccept'),
            sidebar: $('#sidebar'),
            sidebarToggle: $('#sidebarToggle'),
            sidebarOverlay: $('#sidebarOverlay'),
            statusDot: $('#statusDot'),
            statusText: $('#statusText'),
            networkBadge: $('#networkBadge'),
            modelName: $('#modelName'),
            modelId: $('#modelId'),
            modelBadges: $('#modelBadges'),
            gpuName: $('#gpuName'),
            gpuVram: $('#gpuVram'),
            serverUptime: $('#serverUptime'),
            imageSection: $('#imageSection'),
            imageUploadArea: $('#imageUploadArea'),
            imageInput: $('#imageInput'),
            uploadBtn: $('.upload-btn'),
            imagePreviewContainer: $('#imagePreviewContainer'),
            imagePreview: $('#imagePreview'),
            analyzeBtn: $('#analyzeBtn'),
            removeImageBtn: $('#removeImageBtn'),
            newChatBtn: $('#newChatBtn'),
            clearHistoryBtn: $('#clearHistoryBtn'),
            chatHistoryList: $('#chatHistoryList'),
            chatContainer: $('#chatContainer'),
            welcomeScreen: $('#welcomeScreen'),
            chatInput: $('#chatInput'),
            sendBtn: $('#sendBtn'),
            tokenCounter: $('#tokenCounter'),
        };
    }

    // ── Initialization ────────────────────────────────────
    function init() {
        setupElements();
        bindEvents();
        loadChatHistory();
        startHealthPolling();
    }

    function bindEvents() {
        // Chat input
        if (els.chatInput) {
            els.chatInput.addEventListener('input', onInputChange);
            els.chatInput.addEventListener('keydown', onInputKeydown);
            els.chatInput.addEventListener('input', autoResizeInput);
        }
        if (els.sendBtn) els.sendBtn.addEventListener('click', onSend);

        // Sidebar toggles
        if (els.sidebarToggle && els.sidebar) {
            els.sidebarToggle.addEventListener('click', toggleSidebar);
        }
        if (els.sidebarOverlay) {
            els.sidebarOverlay.addEventListener('click', closeSidebar);
        }

        // Quick prompts
        $$('.quick-prompt').forEach(btn => {
            btn.addEventListener('click', () => {
                if (els.chatInput) {
                    els.chatInput.value = btn.dataset.prompt;
                    onInputChange();
                    onSend();
                    closeSidebar();
                }
            });
        });

        // New chat
        if (els.newChatBtn) els.newChatBtn.addEventListener('click', startNewChat);

        // Clear history
        if (els.clearHistoryBtn) {
            els.clearHistoryBtn.addEventListener('click', () => {
                if (confirm('Delete all chat history? This cannot be undone.')) {
                    ChatStore.clearAll();
                    startNewChat();
                }
            });
        }

        // Image upload drag & drop
        if (els.chatContainer && els.imageUploadArea) {
            els.chatContainer.addEventListener('dragover', (e) => {
                e.preventDefault();
                els.imageUploadArea.classList.add('drag-over');
            });
            els.chatContainer.addEventListener('dragleave', (e) => {
                if (e.relatedTarget === null || !els.chatContainer.contains(e.relatedTarget)) {
                    els.imageUploadArea.classList.remove('drag-over');
                }
            });
            els.chatContainer.addEventListener('drop', (e) => {
                e.preventDefault();
                els.imageUploadArea.classList.remove('drag-over');
                if (e.dataTransfer.files.length) handleImageFile(e.dataTransfer.files[0]);
            });
        }

        if (els.imageInput) els.imageInput.addEventListener('change', onImageSelected);
        if (els.removeImageBtn) els.removeImageBtn.addEventListener('click', removeImage);
        if (els.analyzeBtn) els.analyzeBtn.addEventListener('click', onAnalyzeImage);

        // Disclaimer
        if (els.disclaimerAccept) els.disclaimerAccept.addEventListener('click', acceptDisclaimer);
    }

    function toggleSidebar() {
        if (els.sidebar) {
            const isOpen = els.sidebar.classList.toggle('open');
            if (els.sidebarOverlay) {
                els.sidebarOverlay.style.display = isOpen ? 'block' : 'none';
            }
        }
    }

    function closeSidebar() {
        if (els.sidebar) {
            els.sidebar.classList.remove('open');
            if (els.sidebarOverlay) {
                els.sidebarOverlay.style.display = 'none';
            }
        }
    }

    // ── Chat History Management ───────────────────────────
    function loadChatHistory() {
        const activeId = ChatStore.getActiveChatId();
        const chats = ChatStore.list();

        if (activeId && ChatStore.get(activeId)) {
            switchToChat(activeId);
        } else if (chats.length > 0) {
            switchToChat(chats[0].id);
        } else {
            // No history — create fresh session (don't persist until first message)
            state.activeChatId = null;
            state.messages = [];
            showWelcomeScreen();
        }
        renderChatHistory();
    }

    function renderChatHistory() {
        if (!els.chatHistoryList) return;

        const chats = ChatStore.list();
        els.chatHistoryList.innerHTML = '';

        if (chats.length === 0) {
            els.chatHistoryList.innerHTML = '<div class="chat-history-empty" style="padding: 20px; text-align: center; color: var(--text-dim); font-size: 0.85rem;">No conversations yet</div>';
            return;
        }

        chats.forEach(chat => {
            const item = document.createElement('div');
            item.className = 'chat-history-item' + (chat.id === state.activeChatId ? ' active' : '');
            item.dataset.chatId = chat.id;

            const msgCount = chat.messages.length;
            const timeAgo = getRelativeTime(chat.updatedAt);

            item.innerHTML = `
                <div class="chat-history-item-content" title="${escapeHtml(chat.title)}">
                    <div class="chat-history-title">${escapeHtml(chat.title)}</div>
                    <div class="chat-history-meta">${msgCount} msg${msgCount !== 1 ? 's' : ''} · ${timeAgo}</div>
                </div>
                <button class="chat-history-delete" title="Delete conversation">✕</button>
            `;

            // Click to switch
            item.querySelector('.chat-history-item-content').addEventListener('click', () => {
                if (state.isStreaming) return;
                switchToChat(chat.id);
                renderChatHistory();
                closeSidebar();
            });

            // Delete button
            item.querySelector('.chat-history-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                if (state.isStreaming) return;
                ChatStore.delete(chat.id);
                if (state.activeChatId === chat.id) {
                    const remaining = ChatStore.list();
                    if (remaining.length > 0) {
                        switchToChat(remaining[0].id);
                    } else {
                        state.activeChatId = null;
                        state.messages = [];
                        showWelcomeScreen();
                    }
                }
                renderChatHistory();
            });

            els.chatHistoryList.appendChild(item);
        });
    }

    function switchToChat(chatId) {
        const chat = ChatStore.get(chatId);
        if (!chat) return;

        state.activeChatId = chatId;
        state.messages = [...chat.messages];
        ChatStore.setActiveChatId(chatId);

        // Render messages
        if (els.chatContainer) {
            els.chatContainer.innerHTML = '';

            if (state.messages.length === 0) {
                showWelcomeScreen();
            } else {
                state.messages.forEach(msg => {
                    addMessage(msg.role, msg.content, null, false);
                });
                scrollToBottom();
            }
        }

        removeImage();
    }

    function startNewChat() {
        if (state.isStreaming) return;

        // Create and switch
        state.activeChatId = null;
        state.messages = [];

        // Clear UI
        if (els.chatContainer) {
            els.chatContainer.innerHTML = '';
            showWelcomeScreen();
        }
        removeImage();
        renderChatHistory();
        closeSidebar();
    }

    function ensureActiveChatExists() {
        // Lazily create the chat in storage only when the first message is sent
        if (!state.activeChatId) {
            const chat = ChatStore.create('New Conversation');
            state.activeChatId = chat.id;
        }
    }

    function persistMessages() {
        if (state.activeChatId) {
            ChatStore.update(state.activeChatId, state.messages);
            renderChatHistory();
        }
    }

    function getRelativeTime(timestamp) {
        const diff = Date.now() - timestamp;
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return 'just now';
        if (mins < 60) return `${mins}m ago`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        if (days < 7) return `${days}d ago`;
        return new Date(timestamp).toLocaleDateString();
    }

    // ── Health Polling ────────────────────────────────────
    function startHealthPolling() {
        checkHealth();
        state.healthPollTimer = setInterval(checkHealth, 5000);
    }

    let healthFailCount = 0;

    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            if (!res.ok) throw new Error('Health check failed');
            const data = await res.json();

            state.serverReady = data.status === 'ready';
            state.modelInfo = data;
            healthFailCount = 0;

            updateStatusUI(data);
            hideLoadingScreen();

        } catch (err) {
            healthFailCount++;
            if (els.loadingSubtext) els.loadingSubtext.textContent = 'Waiting for server...';
            if (healthFailCount > 3 && els.loadingScreen) {
                const retryBtn = els.loadingScreen.querySelector('.btn-retry');
                if (retryBtn) retryBtn.style.display = 'block';
            }
            updateStatusOffline();
        }
    }

    function hideLoadingScreen() {
        if (els.loadingScreen && !els.loadingScreen.classList.contains('hidden')) {
            els.loadingScreen.classList.add('hidden');

            // Show disclaimer if not accepted
            if (els.disclaimerModal && !state.disclaimerAccepted) {
                els.disclaimerModal.classList.remove('hidden');
            }
        }
    }

    function updateStatusUI(data) {
        if (els.statusDot) {
            els.statusDot.className = 'status-dot' + (data.status === 'ready' ? ' ready' : '');
        }
        if (els.statusText) {
            els.statusText.textContent = data.status === 'ready' ? 'Model Ready' : 'Loading Model...';
        }

        if (els.networkBadge) els.networkBadge.textContent = `${data.host}:${data.port}`;
        if (els.modelName) els.modelName.textContent = data.model_name || '--';
        if (els.modelId) els.modelId.textContent = data.model_id || '--';

        if (els.modelBadges) {
            els.modelBadges.innerHTML = '';
            if (data.modality) {
                const cls = data.modality === 'multimodal' ? 'multimodal' : 'text-only';
                els.modelBadges.innerHTML += `<span class="badge ${cls}">${data.modality}</span>`;
            }
            if (data.supports_images) {
                els.modelBadges.innerHTML += '<span class="badge multimodal">vision</span>';
            }
        }

        if (els.gpuName) els.gpuName.textContent = data.gpu_name || 'No GPU';
        if (els.gpuVram) els.gpuVram.textContent = data.gpu_vram_gb ? `${data.gpu_vram_gb} GB` : '--';
        if (els.serverUptime) els.serverUptime.textContent = formatUptime(data.uptime_seconds);

        if (els.uploadBtn) {
            els.uploadBtn.style.opacity = data.supports_images ? '1' : '0.3';
            els.uploadBtn.style.pointerEvents = data.supports_images ? 'all' : 'none';
        }
    }

    function updateStatusOffline() {
        if (els.statusDot) els.statusDot.className = 'status-dot error';
        if (els.statusText) els.statusText.textContent = 'Disconnected';
    }

    function formatUptime(seconds) {
        if (!seconds) return '--';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    }

    // ── Disclaimer ────────────────────────────────────────
    function acceptDisclaimer() {
        state.disclaimerAccepted = true;
        safeStorage.set('medserver_disclaimer', 'accepted');
        if (els.disclaimerModal) els.disclaimerModal.classList.add('hidden');
    }

    // ── Chat Input ────────────────────────────────────────
    function onInputChange() {
        if (!els.chatInput || !els.sendBtn) return;
        const hasText = els.chatInput.value.trim().length > 0;
        els.sendBtn.disabled = !hasText && !state.isStreaming;
        updateTokenCounter();
    }

    function onInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (els.sendBtn && !els.sendBtn.disabled) onSend();
        }
    }

    function autoResizeInput() {
        if (!els.chatInput) return;
        els.chatInput.style.height = 'auto';
        els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 150) + 'px';
    }

    function updateTokenCounter() {
        if (!els.chatInput || !els.tokenCounter) return;
        const text = els.chatInput.value;
        const tokens = Math.ceil(text.length / 4);
        els.tokenCounter.textContent = tokens > 0 ? `~${tokens} tokens` : '';
    }

    // ── Sending Messages ──────────────────────────────────
    async function onSend() {
        if (state.isStreaming || !els.chatInput) return;

        const text = els.chatInput.value.trim();
        if (!text) return;

        ensureActiveChatExists();
        hideWelcomeScreen();
        closeSidebar();

        addMessage('user', text, state.attachedImageData);
        state.messages.push({ role: 'user', content: text });
        persistMessages();

        els.chatInput.value = '';
        els.chatInput.style.height = 'auto';
        onInputChange();

        if (state.attachedImage) removeImage();

        await streamChat();
    }

    async function streamChat() {
        if (!els.sendBtn) return;
        state.isStreaming = true;
        els.sendBtn.innerHTML = '■';
        els.sendBtn.classList.add('stop');
        els.sendBtn.disabled = false;

        const msgEl = addMessage('assistant', '');
        const contentEl = msgEl.querySelector('.message-content .content-text');

        contentEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: state.messages,
                    max_tokens: 2048,
                    temperature: 0.3,
                    stream: true,
                }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Server error' }));
                contentEl.innerHTML = `<span style="color:var(--status-alert)">Error: ${err.detail || 'Unknown error'}</span>`;
                state.isStreaming = false;
                resetSendButton();
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const payload = line.slice(6).trim();

                    if (payload === '[DONE]') continue;

                    try {
                        const data = JSON.parse(payload);
                        if (data.error) {
                            contentEl.innerHTML += `<span style="color:var(--status-alert)">\n\nError: ${data.error}</span>`;
                            continue;
                        }
                        if (data.token) {
                            fullText += data.token;
                            contentEl.innerHTML = renderMarkdown(fullText);
                            scrollToBottom();
                        }
                    } catch (_) { }
                }
            }

            state.messages.push({ role: 'assistant', content: fullText });
            persistMessages();

        } catch (err) {
            contentEl.innerHTML = `<span style="color:var(--status-alert)">Connection error: ${err.message}</span>`;
        }

        state.isStreaming = false;
        resetSendButton();
    }

    function resetSendButton() {
        if (!els.sendBtn) return;
        els.sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
        els.sendBtn.classList.remove('stop');
        els.sendBtn.disabled = true;
    }

    // ── Image Upload & Analysis ───────────────────────────
    function onImageSelected(e) {
        if (e.target.files.length) handleImageFile(e.target.files[0]);
    }

    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            alert('Image too large. Max 20MB.');
            return;
        }

        state.attachedImage = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            state.attachedImageData = e.target.result;
            if (els.imagePreview) els.imagePreview.src = e.target.result;
            if (els.imagePreviewContainer) els.imagePreviewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    function removeImage() {
        state.attachedImage = null;
        state.attachedImageData = null;
        if (els.imageInput) els.imageInput.value = '';
        if (els.imagePreviewContainer) els.imagePreviewContainer.classList.add('hidden');
    }

    async function onAnalyzeImage() {
        if (!state.attachedImage || !state.serverReady) return;

        const promptText = (els.chatInput ? els.chatInput.value.trim() : '') ||
            'Analyze this medical image and provide detailed clinical findings.';

        ensureActiveChatExists();
        hideWelcomeScreen();
        closeSidebar();

        addMessage('user', promptText, state.attachedImageData);

        const formData = new FormData();
        formData.append('image', state.attachedImage);
        formData.append('prompt', promptText);
        formData.append('max_tokens', '2048');
        formData.append('temperature', '0.3');

        if (els.chatInput) els.chatInput.value = '';
        removeImage();

        state.isStreaming = true;
        if (els.sendBtn) {
            els.sendBtn.innerHTML = '■';
            els.sendBtn.classList.add('stop');
        }

        const msgEl = addMessage('assistant', '');
        const contentEl = msgEl.querySelector('.message-content .content-text');
        contentEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Server error' }));
                contentEl.innerHTML = `<span style="color:var(--status-alert)">Error: ${err.detail || 'Unknown error'}</span>`;
                state.isStreaming = false;
                resetSendButton();
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const payload = line.slice(6).trim();
                    if (payload === '[DONE]') continue;

                    try {
                        const data = JSON.parse(payload);
                        if (data.token) {
                            fullText += data.token;
                            contentEl.innerHTML = renderMarkdown(fullText);
                            scrollToBottom();
                        }
                    } catch (_) { }
                }
            }

            state.messages.push(
                { role: 'user', content: `[Image Analysis] ${promptText}` },
                { role: 'assistant', content: fullText }
            );
            persistMessages();

        } catch (err) {
            contentEl.innerHTML = `<span style="color:var(--status-alert)">Connection error: ${err.message}</span>`;
        }

        state.isStreaming = false;
        resetSendButton();
    }

    // ── Welcome Screen ────────────────────────────────────
    function showWelcomeScreen() {
        if (!els.chatContainer) return;
        els.chatContainer.innerHTML = '';

        const welcome = document.createElement('div');
        welcome.className = 'welcome-screen';
        welcome.id = 'welcomeScreen';
        welcome.innerHTML = `
            <div class="welcome-overlay">
                <div class="welcome-title">MedGemma AI</div>
                <p class="welcome-desc">Specialized clinical intelligence for research and decision support.</p>
                <div class="quick-actions">
                    <div class="action-card quick-prompt" data-prompt="Common differential diagnoses for acute chest pain?">
                        <div style="font-size: 1.5rem; margin-bottom: 8px;">🫀</div>
                        <div style="font-weight: 600; font-size: 1rem;">Cardiovascular</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted);">Chest pain differential diagnosis</div>
                    </div>
                    <div class="action-card quick-prompt" data-prompt="Summarize management guidelines for T2DM in elderly.">
                        <div style="font-size: 1.5rem; margin-bottom: 8px;">📋</div>
                        <div style="font-weight: 600; font-size: 1rem;">Metabolic</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted);">T2DM management guidelines</div>
                    </div>
                </div>
            </div>
        `;
        els.chatContainer.appendChild(welcome);

        // Bind quick prompts
        welcome.querySelectorAll('.quick-prompt').forEach(btn => {
            btn.addEventListener('click', () => {
                if (els.chatInput) {
                    els.chatInput.value = btn.dataset.prompt;
                    onInputChange();
                    onSend();
                    closeSidebar();
                }
            });
        });
    }

    function hideWelcomeScreen() {
        const ws = $('#welcomeScreen');
        if (ws) ws.style.display = 'none';
    }

    // ── Message Rendering ─────────────────────────────────
    function addMessage(role, text, imageDataUrl, animate = true) {
        if (!els.chatContainer) return;
        
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        if (!animate) msgDiv.style.animation = 'none';

        const avatarIcon = role === 'assistant' ? 'M' : '👤';

        let imageHtml = '';
        if (imageDataUrl) {
            imageHtml = `<img class="message-image" src="${imageDataUrl}" alt="Attached medical image">`;
        }

        msgDiv.innerHTML = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-content">
                ${imageHtml}
                <div class="content-text">${role === 'assistant' ? renderMarkdown(text) : escapeHtml(text)}</div>
            </div>
        `;

        els.chatContainer.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function scrollToBottom() {
        if (els.chatContainer) {
            els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
        }
    }

    // ── Markdown Rendering ────────────────────────────────
    /**
     * Basic Markdown Parser with Code-Block Protection
     */
    function renderMarkdown(text) {
        if (!text) return '';
        
        const codeBlocks = [];
        let html = escapeHtml(text);

        // 1. Protect multi-line code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const id = `__CB_${codeBlocks.length}__`;
            codeBlocks.push(`<pre><code class="language-${lang}">${code.trim()}</code></pre>`);
            return id;
        });

        // 2. Protect inline code
        html = html.replace(/`([^`]+)`/g, (_, code) => {
            const id = `__CB_${codeBlocks.length}__`;
            codeBlocks.push(`<code>${code}</code>`);
            return id;
        });

        // 3. Regular Markdown
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
        html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        // 4. Paragraphs & Line Breaks
        html = html.replace(/\n\n/g, '</p><p>');
        html = `<p>${html}</p>`;
        html = html.replace(/\n/g, '<br>');

        // 5. Restore Protected Code Blocks
        codeBlocks.forEach((block, i) => {
            html = html.replace(`__CB_${i}__`, block);
        });

        // Cleanup empty paragraphs or misaligned tags
        html = html.replace(/<p>\s*<\/p>/g, '');
        html = html.replace(/<p>\s*(<h[1-3]|<ul>|<pre|<blockquote>)/g, '$1');
        html = html.replace(/(<\/h[1-3]|<\/ul>|<\/pre>|<\/blockquote>)\s*<\/p>/g, '$1');

        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Boot ──────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
