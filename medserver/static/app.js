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
                return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
            } catch { return {}; }
        },

        _saveAll(chats) {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
        },

        getActiveChatId() {
            return localStorage.getItem(ACTIVE_CHAT_KEY);
        },

        setActiveChatId(id) {
            localStorage.setItem(ACTIVE_CHAT_KEY, id);
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
            localStorage.removeItem(STORAGE_KEY);
            localStorage.removeItem(ACTIVE_CHAT_KEY);
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
        disclaimerAccepted: localStorage.getItem('medserver_disclaimer') === 'accepted',
    };

    // ── DOM References ────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        loadingScreen: $('#loadingScreen'),
        loadingSubtext: $('#loadingSubtext'),
        disclaimerModal: $('#disclaimerModal'),
        disclaimerAccept: $('#disclaimerAccept'),
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

    // ── Initialization ────────────────────────────────────
    function init() {
        bindEvents();
        loadChatHistory();
        startHealthPolling();
    }

    function bindEvents() {
        // Chat input
        els.chatInput.addEventListener('input', onInputChange);
        els.chatInput.addEventListener('keydown', onInputKeydown);
        els.sendBtn.addEventListener('click', onSend);

        // Quick prompts
        $$('.quick-prompt').forEach(btn => {
            btn.addEventListener('click', () => {
                els.chatInput.value = btn.dataset.prompt;
                onInputChange();
                onSend();
            });
        });

        // New chat
        els.newChatBtn.addEventListener('click', startNewChat);

        // Clear history
        if (els.clearHistoryBtn) {
            els.clearHistoryBtn.addEventListener('click', () => {
                if (confirm('Delete all chat history? This cannot be undone.')) {
                    ChatStore.clearAll();
                    startNewChat();
                }
            });
        }

        // Image upload
        els.imageUploadArea.addEventListener('click', () => els.imageInput.click());
        els.imageInput.addEventListener('change', onImageSelected);
        els.removeImageBtn.addEventListener('click', removeImage);
        els.analyzeBtn.addEventListener('click', onAnalyzeImage);

        // Drag & drop
        els.imageUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            els.imageUploadArea.classList.add('drag-over');
        });
        els.imageUploadArea.addEventListener('dragleave', () => {
            els.imageUploadArea.classList.remove('drag-over');
        });
        els.imageUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            els.imageUploadArea.classList.remove('drag-over');
            if (e.dataTransfer.files.length) handleImageFile(e.dataTransfer.files[0]);
        });

        // Disclaimer
        els.disclaimerAccept.addEventListener('click', acceptDisclaimer);

        // Auto-resize textarea
        els.chatInput.addEventListener('input', autoResizeInput);
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
            els.chatHistoryList.innerHTML = '<div class="chat-history-empty">No conversations yet</div>';
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
        els.chatContainer.innerHTML = '';

        if (state.messages.length === 0) {
            showWelcomeScreen();
        } else {
            state.messages.forEach(msg => {
                addMessage(msg.role, msg.content, null, false);
            });
            scrollToBottom();
        }

        removeImage();
    }

    function startNewChat() {
        if (state.isStreaming) return;

        // Create and switch
        state.activeChatId = null;
        state.messages = [];

        // Clear UI
        els.chatContainer.innerHTML = '';
        showWelcomeScreen();
        removeImage();
        renderChatHistory();
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

    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            if (!res.ok) throw new Error('Health check failed');
            const data = await res.json();

            state.serverReady = data.status === 'ready';
            state.modelInfo = data;

            updateStatusUI(data);

            // Hide loading screen on first successful health check
            if (!els.loadingScreen.classList.contains('hidden')) {
                els.loadingScreen.classList.add('hidden');

                // Show disclaimer if not accepted
                if (!state.disclaimerAccepted) {
                    els.disclaimerModal.classList.remove('hidden');
                }
            }
        } catch (err) {
            els.loadingSubtext.textContent = 'Waiting for server...';
            updateStatusOffline();
        }
    }

    function updateStatusUI(data) {
        if (data.status === 'ready') {
            els.statusDot.className = 'status-dot ready';
            els.statusText.textContent = 'Model Ready';
        } else {
            els.statusDot.className = 'status-dot';
            els.statusText.textContent = 'Loading Model...';
        }

        els.networkBadge.textContent = `${data.host}:${data.port}`;
        els.modelName.textContent = data.model_name || '--';
        els.modelId.textContent = data.model_id || '--';

        els.modelBadges.innerHTML = '';
        if (data.modality) {
            const cls = data.modality === 'multimodal' ? 'multimodal' : 'text-only';
            els.modelBadges.innerHTML += `<span class="badge ${cls}">${data.modality}</span>`;
        }
        if (data.supports_images) {
            els.modelBadges.innerHTML += '<span class="badge multimodal">vision</span>';
        }

        els.gpuName.textContent = data.gpu_name || 'No GPU';
        els.gpuVram.textContent = data.gpu_vram_gb ? `${data.gpu_vram_gb} GB` : '--';
        els.serverUptime.textContent = formatUptime(data.uptime_seconds);

        if (data.supports_images) {
            els.imageSection.style.display = '';
            els.imageUploadArea.classList.remove('disabled');
        } else {
            els.imageSection.style.display = '';
            els.imageUploadArea.classList.add('disabled');
            els.imageUploadArea.querySelector('.upload-text').textContent = 'Text-only model (no image support)';
        }
    }

    function updateStatusOffline() {
        els.statusDot.className = 'status-dot error';
        els.statusText.textContent = 'Disconnected';
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
        localStorage.setItem('medserver_disclaimer', 'accepted');
        els.disclaimerModal.classList.add('hidden');
    }

    // ── Chat Input ────────────────────────────────────────
    function onInputChange() {
        const hasText = els.chatInput.value.trim().length > 0;
        els.sendBtn.disabled = !hasText && !state.isStreaming;
        updateTokenCounter();
    }

    function onInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!els.sendBtn.disabled) onSend();
        }
    }

    function autoResizeInput() {
        els.chatInput.style.height = 'auto';
        els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 150) + 'px';
    }

    function updateTokenCounter() {
        const text = els.chatInput.value;
        const tokens = Math.ceil(text.length / 4);
        els.tokenCounter.textContent = tokens > 0 ? `~${tokens} tokens` : '';
    }

    // ── Sending Messages ──────────────────────────────────
    async function onSend() {
        if (state.isStreaming) return;

        const text = els.chatInput.value.trim();
        if (!text) return;

        // Ensure chat exists in storage
        ensureActiveChatExists();

        // Hide welcome screen
        hideWelcomeScreen();

        // Add user message
        addMessage('user', text, state.attachedImageData);
        state.messages.push({ role: 'user', content: text });
        persistMessages();

        // Clear input
        els.chatInput.value = '';
        els.chatInput.style.height = 'auto';
        onInputChange();

        // Remove attached image after sending
        const hadImage = !!state.attachedImage;
        if (hadImage) removeImage();

        // Stream response
        await streamChat();
    }

    async function streamChat() {
        state.isStreaming = true;
        els.sendBtn.innerHTML = '■';
        els.sendBtn.classList.add('stop');
        els.sendBtn.disabled = false;

        const msgEl = addMessage('assistant', '');
        const contentEl = msgEl.querySelector('.message-content');

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
                contentEl.innerHTML = `<span style="color:var(--status-error)">Error: ${err.detail || 'Unknown error'}</span>`;
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
                            contentEl.innerHTML += `<span style="color:var(--status-error)">\n\nError: ${data.error}</span>`;
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

            // Save assistant response & persist
            state.messages.push({ role: 'assistant', content: fullText });
            persistMessages();

        } catch (err) {
            contentEl.innerHTML = `<span style="color:var(--status-error)">Connection error: ${err.message}</span>`;
        }

        state.isStreaming = false;
        resetSendButton();
    }

    function resetSendButton() {
        els.sendBtn.innerHTML = '▶';
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
            els.imagePreview.src = e.target.result;
            els.imagePreviewContainer.classList.add('has-image');
            els.imageUploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    function removeImage() {
        state.attachedImage = null;
        state.attachedImageData = null;
        els.imageInput.value = '';
        els.imagePreviewContainer.classList.remove('has-image');
        els.imageUploadArea.style.display = '';
    }

    async function onAnalyzeImage() {
        if (!state.attachedImage || !state.serverReady) return;

        const promptText = els.chatInput.value.trim() ||
            'Analyze this medical image and provide detailed clinical findings, observations, and any relevant differential diagnoses.';

        ensureActiveChatExists();
        hideWelcomeScreen();

        addMessage('user', promptText, state.attachedImageData);

        const formData = new FormData();
        formData.append('image', state.attachedImage);
        formData.append('prompt', promptText);
        formData.append('max_tokens', '2048');
        formData.append('temperature', '0.3');

        els.chatInput.value = '';
        removeImage();

        state.isStreaming = true;
        els.sendBtn.innerHTML = '■';
        els.sendBtn.classList.add('stop');

        const msgEl = addMessage('assistant', '');
        const contentEl = msgEl.querySelector('.message-content');
        contentEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Server error' }));
                contentEl.innerHTML = `<span style="color:var(--status-error)">Error: ${err.detail || 'Unknown error'}</span>`;
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
            contentEl.innerHTML = `<span style="color:var(--status-error)">Connection error: ${err.message}</span>`;
        }

        state.isStreaming = false;
        resetSendButton();
    }

    // ── Welcome Screen ────────────────────────────────────
    function showWelcomeScreen() {
        els.chatContainer.innerHTML = '';

        const welcome = document.createElement('div');
        welcome.className = 'welcome-screen';
        welcome.id = 'welcomeScreen';
        welcome.innerHTML = `
            <div class="welcome-icon">🧬</div>
            <div class="welcome-title">MedGemma Clinical AI</div>
            <div class="welcome-subtitle">
                Ask clinical questions, request differential diagnoses,
                analyze medical images, or get evidence-based medical information.
            </div>
            <div class="quick-prompts">
                <button class="quick-prompt" data-prompt="What are the common differential diagnoses for acute chest pain in a 55-year-old male with hypertension?">
                    <span class="quick-prompt-icon">🫀</span>
                    Differential diagnosis for acute chest pain
                </button>
                <button class="quick-prompt" data-prompt="Summarize the current evidence-based guidelines for managing Type 2 Diabetes in elderly patients.">
                    <span class="quick-prompt-icon">📋</span>
                    T2DM management guidelines for elderly
                </button>
                <button class="quick-prompt" data-prompt="Explain the pathophysiology of sepsis and the recommended treatment approach following the Surviving Sepsis Campaign guidelines.">
                    <span class="quick-prompt-icon">🩺</span>
                    Sepsis pathophysiology and treatment
                </button>
                <button class="quick-prompt" data-prompt="What laboratory findings and imaging would you expect in a patient presenting with acute pancreatitis?">
                    <span class="quick-prompt-icon">🔬</span>
                    Lab findings in acute pancreatitis
                </button>
            </div>
        `;
        els.chatContainer.appendChild(welcome);
        els.welcomeScreen = welcome;

        // Bind quick prompts
        welcome.querySelectorAll('.quick-prompt').forEach(btn => {
            btn.addEventListener('click', () => {
                els.chatInput.value = btn.dataset.prompt;
                onInputChange();
                onSend();
            });
        });
    }

    function hideWelcomeScreen() {
        const ws = $('#welcomeScreen');
        if (ws) ws.style.display = 'none';
    }

    // ── Message Rendering ─────────────────────────────────
    function addMessage(role, text, imageDataUrl, animate = true) {
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
                ${role === 'assistant' ? renderMarkdown(text) : escapeHtml(text)}
            </div>
        `;

        els.chatContainer.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function scrollToBottom() {
        els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
    }

    // ── Markdown Rendering ────────────────────────────────
    function renderMarkdown(text) {
        if (!text) return '';
        let html = escapeHtml(text);

        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
        });

        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

        html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        html = html.replace(/\n\n/g, '</p><p>');
        html = `<p>${html}</p>`;

        html = html.replace(/<p>\s*<\/p>/g, '');
        html = html.replace(/<p>\s*(<h[1-3]>)/g, '$1');
        html = html.replace(/(<\/h[1-3]>)\s*<\/p>/g, '$1');
        html = html.replace(/<p>\s*(<pre>)/g, '$1');
        html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');
        html = html.replace(/<p>\s*(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');
        html = html.replace(/<p>\s*(<blockquote>)/g, '$1');
        html = html.replace(/(<\/blockquote>)\s*<\/p>/g, '$1');

        html = html.replace(/\n/g, '<br>');

        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Boot ──────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
})();
