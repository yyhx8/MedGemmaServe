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
    const SYSTEM_PROMPT_KEY = 'medserver_system_prompt';
    const LOOP_STOP_SENSITIVITY_KEY = 'medserver_loop_stop_sensitivity';
    const SAMPLING_TEMPERATURE_KEY = 'medserver_sampling_temperature';
    const SAMPLING_TOP_P_KEY = 'medserver_sampling_top_p';
    const DEFAULT_LOOP_STOP_SENSITIVITY = 5;
    const LOOP_STOP_MIN = 0;
    const LOOP_STOP_MAX = 10;
    const LOOP_STOP_STEP = 1;
    const DEFAULT_SAMPLING_TEMPERATURE = 0.3;
    const DEFAULT_SAMPLING_TOP_P = 0.95;
    const DEFAULT_TEMPERATURE_MIN = 0.0;
    const DEFAULT_TEMPERATURE_MAX = 2.0;
    const DEFAULT_TOP_P_MIN = 0.1;
    const DEFAULT_TOP_P_MAX = 1.0;
    const TEMPERATURE_STEP = 0.05;
    const TOP_P_STEP = 0.01;

    function clampInt(value, min, max, fallback) {
        const n = Number.parseInt(value, 10);
        if (!Number.isFinite(n)) return fallback;
        return Math.max(min, Math.min(max, n));
    }

    function clampFloat(value, min, max, fallback) {
        const n = Number.parseFloat(value);
        if (!Number.isFinite(n)) return fallback;
        return Math.max(min, Math.min(max, n));
    }

    function snapToStep(value, min, step, decimals = 2) {
        const snapped = min + (Math.round((value - min) / step) * step);
        return Number(snapped.toFixed(decimals));
    }

    /**
     * Helper to extract text from message content (which can be a string or array of objects).
     */
    function getMessageText(content) {
        if (typeof content === 'string') return content;
        if (Array.isArray(content)) {
            const textItem = content.find(item => item.type === 'text');
            return textItem ? textItem.text : "";
        }
        return String(content || "");
    }

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

        getSystemPrompt() {
            return safeStorage.get(SYSTEM_PROMPT_KEY, '');
        },

        setSystemPrompt(prompt) {
            safeStorage.set(SYSTEM_PROMPT_KEY, prompt);
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
                    const text = getMessageText(firstUser.content);
                    chats[id].title = text.slice(0, 60) + (text.length > 60 ? '…' : '');
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
        attachedImages: [], // Array of File objects
        attachedImagesData: [], // Array of Base64 strings
        healthPollTimer: null,
        disclaimerAccepted: safeStorage.get('medserver_disclaimer') === 'accepted',
        abortController: null,
        expandedThoughts: new Set(), // Track which thoughts are manually expanded: "msgIdx-thoughtIdx"
        manualScroll: false, // Track if user has manually scrolled up during streaming
        currentStreamText: '', // Track raw text for reliable finalization/tags
        currentStreamAssistantIdx: null, // Exact assistant slot being streamed
        currentStreamContentEl: null, // Exact DOM node for current streamed assistant content
        leftSidebarOpen: false,
        rightSidebarOpen: false,
        loopStopSensitivity: clampInt(
            safeStorage.get(LOOP_STOP_SENSITIVITY_KEY, String(DEFAULT_LOOP_STOP_SENSITIVITY)),
            LOOP_STOP_MIN,
            LOOP_STOP_MAX,
            DEFAULT_LOOP_STOP_SENSITIVITY
        ),
        samplingTemperature: clampFloat(
            safeStorage.get(SAMPLING_TEMPERATURE_KEY, String(DEFAULT_SAMPLING_TEMPERATURE)),
            DEFAULT_TEMPERATURE_MIN,
            DEFAULT_TEMPERATURE_MAX,
            DEFAULT_SAMPLING_TEMPERATURE
        ),
        samplingTopP: clampFloat(
            safeStorage.get(SAMPLING_TOP_P_KEY, String(DEFAULT_SAMPLING_TOP_P)),
            DEFAULT_TOP_P_MIN,
            DEFAULT_TOP_P_MAX,
            DEFAULT_SAMPLING_TOP_P
        ),
        defaultSamplingTemperature: DEFAULT_SAMPLING_TEMPERATURE,
        defaultSamplingTopP: DEFAULT_SAMPLING_TOP_P,
        samplingTemperatureMin: DEFAULT_TEMPERATURE_MIN,
        samplingTemperatureMax: DEFAULT_TEMPERATURE_MAX,
        samplingTopPMin: DEFAULT_TOP_P_MIN,
        samplingTopPMax: DEFAULT_TOP_P_MAX,
        allowClientSamplingConfig: true,
        lightboxTransform: {
            scale: 1, x: 0, y: 0,
            isDragging: false, startX: 0, startY: 0,
            lastTouchDistance: 0, lastTouchCenter: { x: 0, y: 0 }
        },
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
            configSidebar: $('#configSidebar'),
            configSidebarToggle: $('#configSidebarToggle'),
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
            removeImageBtn: $('#removeImageBtn'),
            newChatBtn: $('#newChatBtn'),
            clearHistoryBtn: $('#clearHistoryBtn'),
            chatHistoryList: $('#chatHistoryList'),
            chatContainer: $('#chatContainer'),
            welcomeScreen: $('#welcomeScreen'),
            chatInput: $('#chatInput'),
            sendBtn: $('#sendBtn'),
            tokenCounter: $('#tokenCounter'),
            systemPromptInput: $('#systemPromptInput'),
            systemPromptCounter: $('#systemPromptCounter'),
            temperatureInput: $('#temperatureInput'),
            temperatureValue: $('#temperatureValue'),
            temperatureNumberInput: $('#temperatureNumberInput'),
            temperatureMinusBtn: $('#temperatureMinusBtn'),
            temperaturePlusBtn: $('#temperaturePlusBtn'),
            topPInput: $('#topPInput'),
            topPValue: $('#topPValue'),
            topPNumberInput: $('#topPNumberInput'),
            topPMinusBtn: $('#topPMinusBtn'),
            topPPlusBtn: $('#topPPlusBtn'),
            samplingSummary: $('#samplingSummary'),
            samplingPolicyHint: $('#samplingPolicyHint'),
            loopStopInput: $('#loopStopInput'),
            loopStopValue: $('#loopStopValue'),
            loopStopNumberInput: $('#loopStopNumberInput'),
            loopStopMinusBtn: $('#loopStopMinusBtn'),
            loopStopPlusBtn: $('#loopStopPlusBtn'),
            loopStopSummary: $('#loopStopSummary'),
            chatInputCounter: $('#chatInputCounter'),
            jumpToBottomBtn: $('#jumpToBottomBtn'),
            inputArea: $('.input-area'),
        };
    }

    // ── Toasts ───────────────────────────────────────────
    function getToastContainer() {
        let container = document.getElementById('toastContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'toast-container';
            container.setAttribute('aria-live', 'polite');
            container.setAttribute('role', 'status');
            document.body.appendChild(container);
        }
        return container;
    }

    function showToast(message, type = 'info', duration = 3200) {
        const container = getToastContainer();
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        const maxToasts = 3;
        while (container.children.length > maxToasts) {
            container.removeChild(container.firstChild);
        }

        window.setTimeout(() => {
            toast.classList.add('fade-out');
            toast.addEventListener('animationend', () => toast.remove(), { once: true });
        }, duration);

        // Safety: remove even if animations are disabled.
        window.setTimeout(() => {
            if (toast.parentNode) toast.remove();
        }, duration + 600);
    }

    function updateToastOffset() {
        const inputAreaHeight = els.inputArea ? els.inputArea.offsetHeight : 0;
        document.documentElement.style.setProperty('--toast-offset', `${Math.min(inputAreaHeight, 120)}px`);
    }

    function syncSidebarState(side, isOpen) {
        const isLeft = side === 'left';
        const sidebarEl = isLeft ? els.sidebar : els.configSidebar;
        const toggleEl = isLeft ? els.sidebarToggle : els.configSidebarToggle;

        if (isLeft) {
            state.leftSidebarOpen = isOpen;
        } else {
            state.rightSidebarOpen = isOpen;
        }

        if (sidebarEl) {
            sidebarEl.classList.toggle('open', isOpen);
        }
        if (toggleEl) {
            toggleEl.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        }

        const anyOpen = state.leftSidebarOpen || state.rightSidebarOpen;
        if (els.sidebarOverlay) {
            els.sidebarOverlay.classList.toggle('visible', anyOpen);
            els.sidebarOverlay.setAttribute('aria-hidden', anyOpen ? 'false' : 'true');
        }
        document.body.classList.toggle('sidebar-open', anyOpen);
    }

    // ── Initialization ────────────────────────────────────
    function init() {
        setupElements();
        syncSidebarState('left', false);
        syncSidebarState('right', false);

        if (els.systemPromptInput) {
            els.systemPromptInput.value = ChatStore.getSystemPrompt();
            els.systemPromptInput.addEventListener('input', () => {
                ChatStore.setSystemPrompt(els.systemPromptInput.value);
                updateCharCounters();
            });
        }
        updateSamplingControls();
        updateLoopStopControls();

        bindEvents();
        loadChatHistory();
        startHealthPolling();
        updateToastOffset();
        window.addEventListener('resize', () => {
            updateToastOffset();
            if (window.innerWidth > 1024) {
                closeSidebar();
            }
        });

        // Handle page reload/close during streaming
        window.addEventListener('beforeunload', () => {
            if (state.isStreaming) {
                // Try to finalize and save whatever we have right now
                const targetIdx = state.currentStreamAssistantIdx !== null ? state.currentStreamAssistantIdx : state.messages.length;
                const contentEl = state.currentStreamContentEl || $$('.message.assistant .content-text')[$$('.message.assistant').length - 1];
                finalizeStreamingResponse(null, targetIdx, contentEl);

                // Also abort the fetch to trigger server-side stop if possible
                if (state.abortController) {
                    state.abortController.abort();
                }
            }
        });
    }

    function bindEvents() {
        // Chat input
        if (els.chatInput) {
            els.chatInput.addEventListener('input', onInputChange);
            els.chatInput.addEventListener('keydown', onInputKeydown);
            els.chatInput.addEventListener('input', autoResizeInput);
        }
        if (els.sendBtn) els.sendBtn.addEventListener('click', onSend);
        if (els.temperatureInput) {
            els.temperatureInput.addEventListener('input', () => onSamplingInput('slider'));
        }
        if (els.topPInput) {
            els.topPInput.addEventListener('input', () => onSamplingInput('slider'));
        }
        if (els.temperatureNumberInput) {
            els.temperatureNumberInput.addEventListener('change', () => onSamplingInput('number'));
            els.temperatureNumberInput.addEventListener('blur', () => onSamplingInput('number'));
        }
        if (els.topPNumberInput) {
            els.topPNumberInput.addEventListener('change', () => onSamplingInput('number'));
            els.topPNumberInput.addEventListener('blur', () => onSamplingInput('number'));
        }
        if (els.temperatureMinusBtn) {
            els.temperatureMinusBtn.addEventListener('click', () => adjustSamplingTemperature(-1));
        }
        if (els.temperaturePlusBtn) {
            els.temperaturePlusBtn.addEventListener('click', () => adjustSamplingTemperature(1));
        }
        if (els.topPMinusBtn) {
            els.topPMinusBtn.addEventListener('click', () => adjustSamplingTopP(-1));
        }
        if (els.topPPlusBtn) {
            els.topPPlusBtn.addEventListener('click', () => adjustSamplingTopP(1));
        }
        if (els.loopStopInput) {
            els.loopStopInput.addEventListener('input', () => onLoopStopInput('slider'));
        }
        if (els.loopStopNumberInput) {
            els.loopStopNumberInput.addEventListener('change', () => onLoopStopInput('number'));
            els.loopStopNumberInput.addEventListener('blur', () => onLoopStopInput('number'));
        }
        if (els.loopStopMinusBtn) {
            els.loopStopMinusBtn.addEventListener('click', () => adjustLoopStopSensitivity(-1));
        }
        if (els.loopStopPlusBtn) {
            els.loopStopPlusBtn.addEventListener('click', () => adjustLoopStopSensitivity(1));
        }

        // Sidebar toggles
        if (els.sidebarToggle && els.sidebar) {
            els.sidebarToggle.addEventListener('click', toggleSidebar);
        }
        if (els.configSidebarToggle && els.configSidebar) {
            els.configSidebarToggle.addEventListener('click', toggleConfigSidebar);
        }
        if (els.sidebarOverlay) {
            els.sidebarOverlay.addEventListener('click', closeSidebar);
        }
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeSidebar();
        });

        // Global click to deselect messages
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.message')) {
                $$('.message.selected').forEach(m => m.classList.remove('selected'));
            }
        });

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

        if (els.newChatBtn) els.newChatBtn.addEventListener('click', startNewChat);

        // Jump to bottom
        if (els.jumpToBottomBtn) {
            els.jumpToBottomBtn.addEventListener('click', () => {
                if (els.chatContainer) {
                    els.chatContainer.scrollTo({
                        top: els.chatContainer.scrollHeight,
                        behavior: 'smooth'
                    });
                }
            });
        }

        // Scroll monitoring for jump button
        if (els.chatContainer) {
            els.chatContainer.addEventListener('scroll', () => {
                updateJumpToBottomVisibility();
            });
        }

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
                if (e.dataTransfer.files.length) {
                    Array.from(e.dataTransfer.files).forEach(file => handleImageFile(file));
                }
            });
        }

        if (els.imageInput) els.imageInput.addEventListener('change', onImageSelected);
        if (els.removeImageBtn) els.removeImageBtn.addEventListener('click', removeImage);

        // Disclaimer
        if (els.disclaimerAccept) els.disclaimerAccept.addEventListener('click', acceptDisclaimer);

        // Lightbox Zoom/Pan
        const lightbox = $('#lightbox');
        const img = $('#lightboxImg');
        if (lightbox && img) {
            img.addEventListener('wheel', handleLightboxWheel, { passive: false });
            img.addEventListener('mousedown', handleLightboxDragStart);
            window.addEventListener('mousemove', handleLightboxDragMove);
            window.addEventListener('mouseup', handleLightboxDragEnd);

            // Touch support
            img.addEventListener('touchstart', handleLightboxTouchStart, { passive: false });
            img.addEventListener('touchmove', handleLightboxTouchMove, { passive: false });
            img.addEventListener('touchend', handleLightboxTouchEnd);
        }
    }

    function toggleSidebar() {
        if (!els.sidebar) return;
        const isOpen = !els.sidebar.classList.contains('open');
        syncSidebarState('right', false);
        syncSidebarState('left', isOpen);
    }

    function toggleConfigSidebar() {
        if (!els.configSidebar) return;
        const isOpen = !els.configSidebar.classList.contains('open');
        syncSidebarState('left', false);
        syncSidebarState('right', isOpen);
    }

    function closeSidebar() {
        syncSidebarState('left', false);
        syncSidebarState('right', false);
    }

    function updateJumpToBottomVisibility() {
        if (!els.chatContainer || !els.jumpToBottomBtn) return;

        // Very strict threshold for auto-scroll logic (within 15px of the absolute bottom)
        const isAtBottom = (els.chatContainer.scrollHeight - els.chatContainer.scrollTop - els.chatContainer.clientHeight) <= 15;

        if (state.isStreaming && !isAtBottom) {
            state.manualScroll = true;
        } else if (isAtBottom) {
            state.manualScroll = false;
        }

        // Slightly looser threshold for showing the button (50px from bottom)
        const isNearBottom = (els.chatContainer.scrollHeight - els.chatContainer.scrollTop - els.chatContainer.clientHeight) < 50;
        if (isNearBottom) {
            els.jumpToBottomBtn.classList.add('hidden');
        } else {
            els.jumpToBottomBtn.classList.remove('hidden');
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
                if (state.isStreaming) {
                    showToast('Please wait for the current response to finish or stop it before switching conversations.', 'warning');
                    return;
                }
                switchToChat(chat.id);
                renderChatHistory();
                closeSidebar();
            });

            // Delete button
            item.querySelector('.chat-history-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                if (state.isStreaming) {
                    showToast('Please wait for the current response to finish or stop it before deleting conversations.', 'warning');
                    return;
                }
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

        // Correctly check if we are in the same chat BEFORE updating state.activeChatId
        const isSameChat = (state.activeChatId === chatId);
        state.activeChatId = chatId;
        state.messages = [...chat.messages];

        if (!isSameChat) {
            state.expandedThoughts.clear();
        }
        ChatStore.setActiveChatId(chatId);

        // Render messages
        if (els.chatContainer) {
            els.chatContainer.innerHTML = '';

            if (state.messages.length === 0) {
                showWelcomeScreen();
            } else {
                state.messages.forEach((msg, idx) => {
                    addMessage(msg.role, msg.content, msg.imageData || null, false, idx);
                });
                renderRegenerateButton();
                scrollToBottom();
            }
        }
    }

    function startNewChat() {
        if (state.isStreaming) {
            showToast('Please wait for the current response to finish or stop it before starting a new conversation.', 'warning');
            return;
        }

        // Create and switch
        state.activeChatId = null;
        state.messages = [];
        state.expandedThoughts.clear();

        // Clear UI
        if (els.chatContainer) {
            els.chatContainer.innerHTML = '';
            showWelcomeScreen();
        }
        // removeImage(); // Keep attached images when starting new chat
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

    /**
     * Cleans up conversation history to remove orphaned prompts or broken states.
     * Ensures strict User -> Assistant -> User alternation.
     * Allows the LAST message to be a User message (waiting for response).
     */
    function sanitizeHistory(messages) {
        if (!messages || messages.length === 0) return [];
        
        let clean = [];
        for (let i = 0; i < messages.length; i++) {
            const msg = messages[i];
            
            // Skip empty assistant messages
            if (msg.role === 'assistant') {
                const text = getMessageText(msg.content).trim();
                if (!text) continue;
            }
            
            if (clean.length > 0) {
                const last = clean[clean.length - 1];
                
                // If two user messages in a row, the previous one was orphaned/failed. Drop it.
                if (last.role === 'user' && msg.role === 'user') {
                    clean.pop();
                    clean.push(msg);
                }
                // If two assistant messages in a row (rare), keep latest
                else if (last.role === 'assistant' && msg.role === 'assistant') {
                    clean.pop();
                    clean.push(msg);
                }
                else {
                    clean.push(msg);
                }
            } else {
                // First message usually user, but if assistant is first, keep it if non-empty
                clean.push(msg);
            }
        }
        return clean;
    }

    function persistMessages() {
        if (!state.activeChatId) return;

        // Cleanup: remove any messages that break the User-Assistant alternating flow
        // or that resulted in errors.
        if (!state.isStreaming) {
            state.messages = sanitizeHistory(state.messages);
        }

        ChatStore.update(state.activeChatId, state.messages);
        renderChatHistory();
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

    function getMaxTextLength() {
        return state.maxTextLength || 50000;
    }

    function getContentTextLength(content) {
        if (typeof content === 'string') return content.length;
        if (Array.isArray(content)) {
            return content.reduce((sum, item) => {
                if (item && item.type === 'text') {
                    return sum + String(item.text || '').length;
                }
                return sum;
            }, 0);
        }
        return 0;
    }

    function validateTurnLengths(messages) {
        const maxLen = getMaxTextLength();
        for (const msg of messages) {
            if (msg.role !== 'user' && msg.role !== 'system') continue;
            const len = getContentTextLength(msg.content);
            if (len > maxLen) {
                return {
                    ok: false,
                    role: msg.role,
                    length: len,
                    max: maxLen,
                };
            }
        }
        return { ok: true, max: maxLen };
    }

    function getEffectiveSamplingConfig() {
        if (state.allowClientSamplingConfig) {
            return {
                temperature: state.samplingTemperature,
                topP: state.samplingTopP,
            };
        }
        return {
            temperature: state.defaultSamplingTemperature,
            topP: state.defaultSamplingTopP,
        };
    }

    function applySamplingPolicyFromHealth(data) {
        const temperatureMin = clampFloat(data.temperature_min, -100, 100, DEFAULT_TEMPERATURE_MIN);
        const temperatureMax = clampFloat(data.temperature_max, -100, 100, DEFAULT_TEMPERATURE_MAX);
        const topPMin = clampFloat(data.top_p_min, 0, 1, DEFAULT_TOP_P_MIN);
        const topPMax = clampFloat(data.top_p_max, 0, 1, DEFAULT_TOP_P_MAX);

        state.samplingTemperatureMin = Math.min(temperatureMin, temperatureMax);
        state.samplingTemperatureMax = Math.max(temperatureMin, temperatureMax);
        state.samplingTopPMin = Math.min(topPMin, topPMax);
        state.samplingTopPMax = Math.max(topPMin, topPMax);

        state.defaultSamplingTemperature = clampFloat(
            data.default_temperature,
            state.samplingTemperatureMin,
            state.samplingTemperatureMax,
            DEFAULT_SAMPLING_TEMPERATURE
        );
        state.defaultSamplingTopP = clampFloat(
            data.default_top_p,
            state.samplingTopPMin,
            state.samplingTopPMax,
            DEFAULT_SAMPLING_TOP_P
        );
        state.allowClientSamplingConfig = data.allow_client_sampling_config !== false;

        if (!state.allowClientSamplingConfig) {
            state.samplingTemperature = state.defaultSamplingTemperature;
            state.samplingTopP = state.defaultSamplingTopP;
        } else {
            state.samplingTemperature = clampFloat(
                state.samplingTemperature,
                state.samplingTemperatureMin,
                state.samplingTemperatureMax,
                state.defaultSamplingTemperature
            );
            state.samplingTopP = clampFloat(
                state.samplingTopP,
                state.samplingTopPMin,
                state.samplingTopPMax,
                state.defaultSamplingTopP
            );
            safeStorage.set(SAMPLING_TEMPERATURE_KEY, String(state.samplingTemperature));
            safeStorage.set(SAMPLING_TOP_P_KEY, String(state.samplingTopP));
        }
        updateSamplingControls();
    }

    function updateSamplingControls() {
        const effective = getEffectiveSamplingConfig();
        if (els.temperatureInput) {
            els.temperatureInput.min = String(state.samplingTemperatureMin);
            els.temperatureInput.max = String(state.samplingTemperatureMax);
            els.temperatureInput.step = String(TEMPERATURE_STEP);
            els.temperatureInput.value = String(state.samplingTemperature);
            els.temperatureInput.disabled = !state.allowClientSamplingConfig;
        }
        if (els.topPInput) {
            els.topPInput.min = String(state.samplingTopPMin);
            els.topPInput.max = String(state.samplingTopPMax);
            els.topPInput.step = String(TOP_P_STEP);
            els.topPInput.value = String(state.samplingTopP);
            els.topPInput.disabled = !state.allowClientSamplingConfig;
        }
        if (els.temperatureNumberInput) {
            els.temperatureNumberInput.min = String(state.samplingTemperatureMin);
            els.temperatureNumberInput.max = String(state.samplingTemperatureMax);
            els.temperatureNumberInput.step = String(TEMPERATURE_STEP);
            els.temperatureNumberInput.value = state.samplingTemperature.toFixed(2);
            els.temperatureNumberInput.disabled = !state.allowClientSamplingConfig;
        }
        if (els.temperatureMinusBtn) {
            els.temperatureMinusBtn.disabled = !state.allowClientSamplingConfig;
        }
        if (els.temperaturePlusBtn) {
            els.temperaturePlusBtn.disabled = !state.allowClientSamplingConfig;
        }
        if (els.topPNumberInput) {
            els.topPNumberInput.min = String(state.samplingTopPMin);
            els.topPNumberInput.max = String(state.samplingTopPMax);
            els.topPNumberInput.step = String(TOP_P_STEP);
            els.topPNumberInput.value = state.samplingTopP.toFixed(2);
            els.topPNumberInput.disabled = !state.allowClientSamplingConfig;
        }
        if (els.topPMinusBtn) {
            els.topPMinusBtn.disabled = !state.allowClientSamplingConfig;
        }
        if (els.topPPlusBtn) {
            els.topPPlusBtn.disabled = !state.allowClientSamplingConfig;
        }
        if (els.temperatureValue) {
            els.temperatureValue.textContent = effective.temperature.toFixed(2);
        }
        if (els.topPValue) {
            els.topPValue.textContent = effective.topP.toFixed(2);
        }
        if (els.samplingSummary) {
            const lockSuffix = state.allowClientSamplingConfig ? '' : ' (locked)';
            els.samplingSummary.textContent = `T ${effective.temperature.toFixed(2)} • P ${effective.topP.toFixed(2)}${lockSuffix}`;
        }
        if (els.samplingPolicyHint) {
            els.samplingPolicyHint.classList.toggle('hidden', state.allowClientSamplingConfig);
        }
    }

    function adjustSamplingTemperature(direction) {
        if (!state.allowClientSamplingConfig) return;
        state.samplingTemperature = snapToStep(
            clampFloat(
                state.samplingTemperature + (direction * TEMPERATURE_STEP),
                state.samplingTemperatureMin,
                state.samplingTemperatureMax,
                state.defaultSamplingTemperature
            ),
            state.samplingTemperatureMin,
            TEMPERATURE_STEP,
            2
        );
        safeStorage.set(SAMPLING_TEMPERATURE_KEY, String(state.samplingTemperature));
        updateSamplingControls();
    }

    function adjustSamplingTopP(direction) {
        if (!state.allowClientSamplingConfig) return;
        state.samplingTopP = snapToStep(
            clampFloat(
                state.samplingTopP + (direction * TOP_P_STEP),
                state.samplingTopPMin,
                state.samplingTopPMax,
                state.defaultSamplingTopP
            ),
            state.samplingTopPMin,
            TOP_P_STEP,
            2
        );
        safeStorage.set(SAMPLING_TOP_P_KEY, String(state.samplingTopP));
        updateSamplingControls();
    }

    function onSamplingInput(source = 'slider') {
        if (!state.allowClientSamplingConfig) {
            updateSamplingControls();
            return;
        }

        const tempRaw = source === 'number'
            ? (els.temperatureNumberInput ? els.temperatureNumberInput.value : state.samplingTemperature)
            : (els.temperatureInput ? els.temperatureInput.value : state.samplingTemperature);
        const topPRaw = source === 'number'
            ? (els.topPNumberInput ? els.topPNumberInput.value : state.samplingTopP)
            : (els.topPInput ? els.topPInput.value : state.samplingTopP);

        state.samplingTemperature = snapToStep(
            clampFloat(
                tempRaw,
                state.samplingTemperatureMin,
                state.samplingTemperatureMax,
                state.defaultSamplingTemperature
            ),
            state.samplingTemperatureMin,
            TEMPERATURE_STEP,
            2
        );
        state.samplingTopP = snapToStep(
            clampFloat(
                topPRaw,
                state.samplingTopPMin,
                state.samplingTopPMax,
                state.defaultSamplingTopP
            ),
            state.samplingTopPMin,
            TOP_P_STEP,
            2
        );

        safeStorage.set(SAMPLING_TEMPERATURE_KEY, String(state.samplingTemperature));
        safeStorage.set(SAMPLING_TOP_P_KEY, String(state.samplingTopP));
        updateSamplingControls();
    }

    function updateLoopStopControls() {
        if (els.loopStopInput) {
            els.loopStopInput.min = String(LOOP_STOP_MIN);
            els.loopStopInput.max = String(LOOP_STOP_MAX);
            els.loopStopInput.step = String(LOOP_STOP_STEP);
            els.loopStopInput.value = String(state.loopStopSensitivity);
        }
        if (els.loopStopNumberInput) {
            els.loopStopNumberInput.min = String(LOOP_STOP_MIN);
            els.loopStopNumberInput.max = String(LOOP_STOP_MAX);
            els.loopStopNumberInput.step = String(LOOP_STOP_STEP);
            els.loopStopNumberInput.value = String(state.loopStopSensitivity);
        }
        if (els.loopStopValue) {
            els.loopStopValue.textContent = String(state.loopStopSensitivity);
        }
        if (els.loopStopSummary) {
            const status = state.loopStopSensitivity <= 0
                ? 'Disabled'
                : `Sensitivity ${state.loopStopSensitivity}`;
            els.loopStopSummary.textContent = status;
        }
    }

    function onLoopStopInput(source = 'slider') {
        const raw = source === 'number'
            ? (els.loopStopNumberInput ? els.loopStopNumberInput.value : state.loopStopSensitivity)
            : (els.loopStopInput ? els.loopStopInput.value : state.loopStopSensitivity);
        state.loopStopSensitivity = clampInt(raw, LOOP_STOP_MIN, LOOP_STOP_MAX, state.loopStopSensitivity);
        safeStorage.set(LOOP_STOP_SENSITIVITY_KEY, String(state.loopStopSensitivity));
        updateLoopStopControls();
    }

    function adjustLoopStopSensitivity(direction) {
        state.loopStopSensitivity = clampInt(
            state.loopStopSensitivity + (direction * LOOP_STOP_STEP),
            LOOP_STOP_MIN,
            LOOP_STOP_MAX,
            state.loopStopSensitivity
        );
        safeStorage.set(LOOP_STOP_SENSITIVITY_KEY, String(state.loopStopSensitivity));
        updateLoopStopControls();
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
            state.maxTextLength = data.max_text_length || 50000;
            applySamplingPolicyFromHealth(data);
            healthFailCount = 0;

            updateStatusUI(data);
            updateCharCounters();
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

        const hwContainer = document.getElementById('hwStatsContainer');
        if (hwContainer) {
            if (data.gpu_name) {
                hwContainer.classList.remove('hidden');
                if (els.gpuName) els.gpuName.textContent = data.gpu_name;
                
                if (els.gpuVram) {
                    if (data.gpu_vram_used_gb !== undefined && data.gpu_vram_total_gb !== undefined) {
                        els.gpuVram.textContent = `${data.gpu_vram_used_gb} / ${data.gpu_vram_total_gb} GB`;
                    } else if (data.gpu_vram_gb) {
                        els.gpuVram.textContent = `${data.gpu_vram_gb} GB`;
                    } else {
                        els.gpuVram.textContent = '--';
                    }
                }
            } else {
                hwContainer.classList.add('hidden');
            }
        }

        if (els.serverUptime) els.serverUptime.textContent = formatUptime(data.uptime_seconds);

        const uploadBtns = $$('.upload-btn, .attach-btn');
        uploadBtns.forEach(btn => {
            btn.style.opacity = data.supports_images ? '1' : '0.3';
            btn.style.pointerEvents = data.supports_images ? 'all' : 'none';
        });
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
        // Clear any failed message UI states if user starts typing to fix them
        const failedPairs = document.querySelectorAll('.conversation-pair.failed');
        if (failedPairs.length > 0) {
            failedPairs.forEach(pair => pair.remove());
        }

        if (!els.chatInput || !els.sendBtn) return;
        const hasText = els.chatInput.value.trim().length > 0;
        const hasImages = state.attachedImagesData.length > 0;
        const isTextTooLong = els.chatInput.value.length > getMaxTextLength();

        // During streaming, the button acts as a STOP button and should NEVER be disabled
        if (state.isStreaming) {
            els.sendBtn.disabled = false;
        } else {
            els.sendBtn.disabled = (!hasText && !hasImages) || isTextTooLong;
        }
        updateTokenCounter();
        updateCharCounters();
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
        updateToastOffset();
    }

    function updateTokenCounter() {
        if (!els.chatInput || !els.tokenCounter) return;
        const text = els.chatInput.value;
        const tokens = Math.ceil(text.length / 4);
        els.tokenCounter.textContent = tokens > 0 ? `~${tokens} tokens` : '';
    }

    function shouldAutoStopLoop(text) {
        const sensitivity = state.loopStopSensitivity || 0;
        if (sensitivity <= 0 || !text) return false;

        const normalized = text.toLowerCase().replace(/\s+/g, ' ').trim();
        if (normalized.length < 280) return false;

        // Higher sensitivity => shorter segment + fewer repeats required.
        const segmentLen = Math.max(40, 180 - (sensitivity * 12));
        const repeatsNeeded = Math.max(2, 7 - Math.ceil(sensitivity / 2));
        if (normalized.length < segmentLen * (repeatsNeeded + 1)) return false;

        const tail = normalized.slice(-segmentLen);
        let repeats = 0;
        let cursor = normalized.length - segmentLen;

        while (cursor - segmentLen >= 0 && repeats < repeatsNeeded) {
            const prev = normalized.slice(cursor - segmentLen, cursor);
            if (prev !== tail) break;
            repeats += 1;
            cursor -= segmentLen;
        }
        return repeats >= repeatsNeeded;
    }

    function updateCharCounters() {
        const maxLen = getMaxTextLength();
        if (els.chatInput && els.chatInputCounter) {
            els.chatInputCounter.textContent = `${els.chatInput.value.length} / ${maxLen}`;
            els.chatInputCounter.classList.toggle('over-limit', els.chatInput.value.length > maxLen);
        }
        if (els.systemPromptInput && els.systemPromptCounter) {
            els.systemPromptCounter.textContent = `${els.systemPromptInput.value.length} / ${maxLen}`;
            els.systemPromptCounter.classList.toggle('over-limit', els.systemPromptInput.value.length > maxLen);
        }
    }

    // ── Sending Messages ──────────────────────────────────
    async function onSend() {
        if (state.isStreaming) {
            stopGeneration();
            return;
        }

        if (!els.chatInput) return;

        const text = els.chatInput.value.trim();
        if (!text && state.attachedImagesData.length === 0) return;
        const maxLen = getMaxTextLength();
        const systemPrompt = ChatStore.getSystemPrompt() || '';
        if (systemPrompt.length > maxLen) {
            showToast(`System prompt exceeds ${maxLen} characters.`, 'warning');
            if (els.systemPromptInput) els.systemPromptInput.focus();
            return;
        }
        if (text.length > maxLen) {
            showToast(`Prompt exceeds ${maxLen} characters.`, 'warning');
            if (els.chatInput) els.chatInput.focus();
            return;
        }

        ensureActiveChatExists();
        hideWelcomeScreen();
        closeSidebar();

        const msgIdx = state.messages.length;
        const currentImagesData = [...state.attachedImagesData];

        // Structured content for the API
        const structuredContent = [];
        currentImagesData.forEach(() => {
            structuredContent.push({ type: 'image' });
        });
        if (text) {
            structuredContent.push({ type: 'text', text: text });
        }

        addMessage('user', text, currentImagesData, true, msgIdx);
        state.messages.push({
            role: 'user',
            content: structuredContent,
            imageData: currentImagesData
        });

        persistMessages();

        els.chatInput.value = '';
        els.chatInput.style.height = 'auto';
        onInputChange();

        if (state.attachedImages.length > 0) removeImage();

        state.manualScroll = false; // Reset on new send
        await streamChat();
    }

    async function streamChat(insertIndex = -1) {
        if (!els.sendBtn) return;

        state.isStreaming = true;
        state.abortController = new AbortController();

        els.sendBtn.innerHTML = '■';
        els.sendBtn.classList.add('stop');
        els.sendBtn.disabled = false;

        removeRegenerateButton();

        const assistantMsgIdx = insertIndex !== -1 ? insertIndex : state.messages.length;
        const targetUserIdx = insertIndex !== -1 ? insertIndex - 1 : null;
        const msgEl = addMessage('assistant', '', null, true, assistantMsgIdx, targetUserIdx);
        const contentEl = msgEl.querySelector('.message-content .content-text');
        state.currentStreamAssistantIdx = assistantMsgIdx;
        state.currentStreamContentEl = contentEl;

        const recoverFromSendFailure = (reason = '') => {
            const fallbackNotice = 'Unable to send right now. Your prompt was restored.';
            const notice = reason ? `${fallbackNotice} (${reason})` : fallbackNotice;

            if (insertIndex === -1) {
                const lastMsg = state.messages[state.messages.length - 1];
                const failedText = (lastMsg && lastMsg.role === 'user')
                    ? getMessageText(lastMsg.content)
                    : '';

                if (lastMsg && lastMsg.role === 'user') {
                    state.messages.pop();
                    persistMessages();
                }

                if (state.activeChatId) {
                    switchToChat(state.activeChatId);
                }

                if (els.chatInput) {
                    els.chatInput.value = failedText;
                    autoResizeInput();
                    onInputChange();
                    els.chatInput.focus();
                }
                showToast(notice, 'warning');
                return;
            }

            // Regeneration path: restore previously saved state/assistant output.
            if (state.activeChatId) {
                switchToChat(state.activeChatId);
            }
            showToast(reason || 'Unable to regenerate right now.', 'warning');
        };

        contentEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

        let fullText = '';
        let receivedToken = false;
        try {
            // Determine which messages to send
            const messagesToSend = insertIndex !== -1
                ? state.messages.slice(0, insertIndex)
                : state.messages;
            const systemPrompt = ChatStore.getSystemPrompt() || '';
            const maxLen = getMaxTextLength();
            if (systemPrompt.length > maxLen) {
                recoverFromSendFailure(`System prompt exceeds ${maxLen} characters`);
                return;
            }
            const turnLengthValidation = validateTurnLengths(messagesToSend);
            if (!turnLengthValidation.ok) {
                const roleLabel = turnLengthValidation.role === 'system' ? 'System prompt' : 'User message';
                recoverFromSendFailure(`${roleLabel} exceeds ${turnLengthValidation.max} characters`);
                return;
            }
            const sampling = getEffectiveSamplingConfig();

            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: state.abortController.signal,
                body: JSON.stringify({
                    messages: messagesToSend.map(m => ({
                        role: m.role,
                        content: m.content,
                        image_data: m.imageData || []
                    })),
                    system_prompt: systemPrompt || undefined,
                    max_tokens: 2048,
                    temperature: sampling.temperature,
                    top_p: sampling.topP,
                    stream: true,
                }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Server communication error' }));
                const errorMessage = err.detail || err.error || err.message || `Request failed (${res.status})`;
                throw new Error(errorMessage);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            fullText = '';
            let buffer = '';
            state.currentStreamText = '';

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

                    let data;
                    try {
                        data = JSON.parse(payload);
                    } catch (parseErr) {
                        // Ignore malformed/incomplete SSE JSON chunks.
                        if (parseErr instanceof SyntaxError) continue;
                        throw parseErr;
                    }

                    if (data.error) {
                        throw new Error(data.error || 'Streaming error');
                    }

                    if (data.token) {
                        fullText += data.token;
                        state.currentStreamText = fullText;
                        receivedToken = true;

                        if (shouldAutoStopLoop(fullText)) {
                            showToast(`Auto-stopped repetitive loop (sensitivity ${state.loopStopSensitivity})`, 'warning', 4200);
                            stopGeneration();
                            return;
                        }

                        // Proactive scroll check before DOM update
                        const isAtBottom = (els.chatContainer.scrollHeight - els.chatContainer.scrollTop - els.chatContainer.clientHeight) <= 15;

                        try {
                            contentEl.innerHTML = renderMarkdown(fullText, assistantMsgIdx, state.isStreaming);
                        } catch {
                            // Rendering should not abort generation; keep plain text visible.
                            contentEl.textContent = fullText;
                        }

                        // Incrementally save every ~80 tokens
                        if (fullText.length % 80 === 0) {
                            if (insertIndex !== -1) {
                                state.messages[assistantMsgIdx] = { role: 'assistant', content: fullText };
                            } else {
                                const lastMsg = state.messages[state.messages.length - 1];
                                if (lastMsg && lastMsg.role === 'assistant') {
                                    lastMsg.content = fullText;
                                } else {
                                    state.messages.push({ role: 'assistant', content: fullText });
                                }
                            }
                            ChatStore.update(state.activeChatId, state.messages);
                        }

                        // Sticky scroll
                        if (isAtBottom && !state.manualScroll) {
                            els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
                        }
                    }
                }
            }

            // Finalize
            let finalizedText = fullText;
            const lastUnused94 = finalizedText.lastIndexOf('<unused94>');
            const lastUnused95 = finalizedText.lastIndexOf('<unused95>');
            if (lastUnused94 !== -1 && (lastUnused95 === -1 || lastUnused95 < lastUnused94)) {
                finalizedText += '<unused95>';
                contentEl.innerHTML = renderMarkdown(finalizedText, assistantMsgIdx, false);
            }

            if (insertIndex !== -1) {
                state.messages[assistantMsgIdx] = { role: 'assistant', content: finalizedText };
            } else {
                const lastMsg = state.messages[state.messages.length - 1];
                if (lastMsg && lastMsg.role === 'assistant') {
                    lastMsg.content = finalizedText;
                } else {
                    state.messages.push({ role: 'assistant', content: finalizedText });
                }
            }
            persistMessages();

        } catch (err) {
            if (err.name === 'AbortError') {
                finalizeStreamingResponse(fullText, assistantMsgIdx, contentEl);
            } else {
                const reason = err && err.message ? err.message : 'Request failed';
                if (receivedToken || fullText.trim().length > 0) {
                    finalizeStreamingResponse(fullText, assistantMsgIdx, contentEl);
                    showToast(`Response interrupted: ${reason}`, 'warning');
                } else {
                    recoverFromSendFailure(reason);
                }
            }
        } finally {
            state.isStreaming = false;
            state.abortController = null;
            state.currentStreamAssistantIdx = null;
            state.currentStreamContentEl = null;
            state.currentStreamText = '';
            resetSendButton();
            renderRegenerateButton();
        }
    }

    /**
     * Helper to finalize a stream that was interrupted (stop button or reload)
     */
    function finalizeStreamingResponse(text, msgIdx, contentEl) {
        // Use provided text, then state.currentStreamText, then fallback to DOM (least reliable)
        const currentText = text !== null ? text : (state.currentStreamText || (contentEl ? contentEl.innerText.replace('(stopped)', '').trim() : ''));
        const targetIdx = msgIdx !== undefined ? msgIdx : state.messages.length;

        let processedText = currentText;

        // Close thinking trace only if it was actually in progress
        const lastTagIdx = processedText.lastIndexOf('<unused94>');
        const closeTagIdx = processedText.lastIndexOf('<unused95>');
        if (lastTagIdx !== -1 && (closeTagIdx === -1 || closeTagIdx < lastTagIdx)) {
            processedText += '<unused95>';
        }

        if (contentEl) {
            contentEl.innerHTML = renderMarkdown(processedText, targetIdx, false) + ' <span style="color:var(--text-dim); font-size: 0.8rem;">(stopped)</span>';
        }

        // Persist into the exact target slot to avoid overwriting newer turns.
        if (processedText.trim().length > 0) {
            const assistantMsg = { role: 'assistant', content: processedText };
            if (targetIdx >= 0 && targetIdx < state.messages.length) {
                if (state.messages[targetIdx] && state.messages[targetIdx].role === 'assistant') {
                    state.messages[targetIdx] = assistantMsg;
                } else {
                    state.messages.splice(targetIdx, 0, assistantMsg);
                }
            } else {
                const lastMsg = state.messages[state.messages.length - 1];
                if (lastMsg && lastMsg.role === 'assistant') {
                    lastMsg.content = processedText;
                } else {
                    state.messages.push(assistantMsg);
                }
            }
            persistMessages();
        } else if (targetIdx >= 0 && targetIdx < state.messages.length && state.messages[targetIdx]?.role === 'assistant') {
            // If no token arrived (e.g. immediate stop during regenerate), keep the existing stored assistant.
            if (contentEl) {
                const priorText = getMessageText(state.messages[targetIdx].content);
                contentEl.innerHTML = renderMarkdown(priorText, targetIdx, false);
            }
        }
    }

    function stopGeneration() {
        if (state.abortController) {
            state.isStreaming = false; // Set this first to affect the final render

            // Provide immediate feedback
            if (els.sendBtn) {
                els.sendBtn.innerHTML = '<span style="font-size: 0.7rem;">STOPPING</span>';
                els.sendBtn.disabled = true;
            }

            const msgIdx = state.currentStreamAssistantIdx !== null ? state.currentStreamAssistantIdx : state.messages.length;
            const contentEl = state.currentStreamContentEl || $$('.message.assistant .content-text')[$$('.message.assistant .content-text').length - 1];
            let text = state.currentStreamText || '';
            // Close thinking tags if they are open so renderMarkdown treats them as finished.
            if (text.includes('<unused94>') && !text.includes('<unused95>')) {
                text += '<unused95>';
            }
            finalizeStreamingResponse(text, msgIdx, contentEl);

            state.abortController.abort();
        }
    }

    function resetSendButton() {
        if (!els.sendBtn) return;
        els.sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
        els.sendBtn.classList.remove('stop');
        els.sendBtn.disabled = false; // Force enable so user can send again
        onInputChange();
    }

    // ── Image Upload & Analysis ───────────────────────────
    function onImageSelected(e) {
        if (e.target.files.length) {
            Array.from(e.target.files).forEach(file => handleImageFile(file));
            e.target.value = ''; // Reset for next selection
        }
    }

    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please select an image file.', 'warning');
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            showToast('Image too large. Max 20MB.', 'warning');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            state.attachedImages.push(file);
            state.attachedImagesData.push(e.target.result);
            renderImagePreviews();
        };
        reader.readAsDataURL(file);
    }

    function renderImagePreviews() {
        if (!els.imagePreviewContainer) return;

        els.imagePreviewContainer.innerHTML = '';

        if (state.attachedImagesData.length === 0) {
            els.imagePreviewContainer.classList.add('hidden');
            updateToastOffset();
            return;
        }

        els.imagePreviewContainer.classList.remove('hidden');

        state.attachedImagesData.forEach((data, index) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'preview-item';

            const img = document.createElement('img');
            img.src = data;
            img.style.cursor = 'pointer';
            img.addEventListener('click', () => window.openLightbox(data));

            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-preview-btn';
            removeBtn.textContent = '✕';
            removeBtn.addEventListener('click', () => window.removeImage(index));

            wrapper.appendChild(img);
            wrapper.appendChild(removeBtn);
            els.imagePreviewContainer.appendChild(wrapper);
        });

        updateToastOffset();
    }

    window.removeImage = function (index) {
        state.attachedImages.splice(index, 1);
        state.attachedImagesData.splice(index, 1);
        renderImagePreviews();
    };

    function removeImage() {
        state.attachedImages = [];
        state.attachedImagesData = [];
        if (els.imageInput) els.imageInput.value = '';
        renderImagePreviews();
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

    // ── Message Actions ───────────────────────────────────
    async function editMessage(index) {
        if (state.isStreaming) return;
        const msg = state.messages[index];
        const msgEl = $$('.message')[index];
        if (!msgEl) return;

        const contentText = msgEl.querySelector('.content-text');
        const currentText = getMessageText(msg.content);

        contentText.innerHTML = `
            <textarea class="edit-textarea"></textarea>
            <div class="edit-controls">
                <button class="btn-small btn-cancel">Cancel</button>
                <button class="btn-small btn-save">Save & Submit</button>
            </div>
        `;

        const textarea = contentText.querySelector('.edit-textarea');
        textarea.value = currentText; // Set via .value to avoid HTML entity double-encoding
        textarea.style.height = textarea.scrollHeight + 'px';
        textarea.focus();

        contentText.querySelector('.btn-cancel').addEventListener('click', () => {
            const text = getMessageText(msg.content);
            contentText.innerHTML = msg.role === 'assistant' ? renderMarkdown(text, index, false) : escapeHtml(text);
        });

        contentText.querySelector('.btn-save').addEventListener('click', async () => {
            const newContent = textarea.value.trim();
            if (newContent !== currentText) {
                if (msg.role === 'user' && newContent.length > getMaxTextLength()) {
                    showToast(`Prompt exceeds ${getMaxTextLength()} characters.`, 'warning');
                    textarea.focus();
                    return;
                }

                // Update message content in state
                if (Array.isArray(msg.content)) {
                    state.messages[index].content = msg.content.map(item =>
                        item.type === 'text' ? { ...item, text: newContent } : item
                    );
                } else {
                    state.messages[index].content = newContent;
                }

                if (msg.role === 'user') {
                    const assistantIdx = index + 1;
                    const hasPairedAssistant = state.messages[assistantIdx] && state.messages[assistantIdx].role === 'assistant';
                    const hasFollowingMessages = assistantIdx < state.messages.length;

                    // Preserve newer messages; only regenerate this specific turn.
                    persistMessages();
                    switchToChat(state.activeChatId);

                    // Avoid overwriting a later user turn when history is malformed.
                    if (!hasPairedAssistant && hasFollowingMessages) {
                        showToast('Cannot regenerate this turn because it has no paired assistant response.', 'warning');
                        return;
                    }

                    await regenerateResponse(index);
                } else {
                    // If editing an assistant reply, just save and re-render that message
                    persistMessages();
                    const text = getMessageText(state.messages[index].content);
                    contentText.innerHTML = renderMarkdown(text, index, false);
                    renderRegenerateButton();
                }
            } else {
                const text = getMessageText(msg.content);
                contentText.innerHTML = msg.role === 'assistant' ? renderMarkdown(text, index, false) : escapeHtml(text);
            }
        });
    }

    function deleteMessage(index) {
        if (state.isStreaming) return;

        const scrollPos = els.chatContainer.scrollTop;
        let startIdx = index;
        let count = 1;

        if (state.messages[index].role === 'user') {
            if (state.messages[index + 1] && state.messages[index + 1].role === 'assistant') {
                count = 2;
            }
        } else {
            // If deleting an assistant message, also remove paired user message before it
            if (index > 0 && state.messages[index - 1] && state.messages[index - 1].role === 'user') {
                startIdx = index - 1;
                count = 2;
            }
            // If assistant is at index 0 with no user before, just delete it alone
        }

        state.messages.splice(startIdx, count);
        persistMessages();

        if (state.messages.length === 0) {
            showWelcomeScreen();
        } else {
            switchToChat(state.activeChatId);
            // RequestAnimationFrame ensures DOM is ready before we set scroll
            requestAnimationFrame(() => {
                if (els.chatContainer) els.chatContainer.scrollTop = scrollPos;
            });
        }
    }

    async function regenerateResponse(targetUserIdx = null) {
        if (state.isStreaming || state.messages.length === 0) return;

        const parsedTarget = typeof targetUserIdx === 'string'
            ? parseInt(targetUserIdx, 10)
            : targetUserIdx;
        let userIdx = Number.isInteger(parsedTarget) ? parsedTarget : null;

        if (userIdx === null) {
            // Default behavior: regenerate last user turn.
            userIdx = -1;
            for (let i = state.messages.length - 1; i >= 0; i--) {
                if (state.messages[i].role === 'user') {
                    userIdx = i;
                    break;
                }
            }
        }

        if (userIdx === -1 || !state.messages[userIdx] || state.messages[userIdx].role !== 'user') return;

        const assistantIdx = userIdx + 1;
        const isAssistantExists = state.messages[assistantIdx] && state.messages[assistantIdx].role === 'assistant';
        const hasFollowingMessages = assistantIdx < state.messages.length;

        if (!isAssistantExists && hasFollowingMessages) {
            showToast('Cannot regenerate this turn because it has no paired assistant response.', 'warning');
            return;
        }

        // Find the pair box for this user message.
        const allPairs = Array.from(els.chatContainer.querySelectorAll('.conversation-pair'));
        const targetPair = allPairs.find(p => {
            const userMsg = p.querySelector('.message.user');
            return userMsg && parseInt(userMsg.dataset.index, 10) === userIdx;
        });

        if (targetPair && isAssistantExists) {
            const oldAssistantMsg = targetPair.querySelector('.message.assistant');
            if (oldAssistantMsg) {
                oldAssistantMsg.remove();
            }
        }

        // Always stream into the target assistant slot so newer messages remain intact.
        await streamChat(assistantIdx);
    }

    function renderRegenerateButton() {
        removeRegenerateButton();
        if (state.messages.length === 0 || state.isStreaming) return;

        const lastMsg = state.messages[state.messages.length - 1];
        if (lastMsg.role !== 'assistant' && lastMsg.role !== 'user') return;

        const container = document.createElement('div');
        container.className = 'regenerate-container';
        container.id = 'regenerateContainer';
        container.innerHTML = `
            <button class="btn-regenerate">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                Regenerate response
            </button>
        `;

        els.chatContainer.appendChild(container);
        container.querySelector('.btn-regenerate').addEventListener('click', () => regenerateResponse());
    }

    function removeRegenerateButton() {
        const el = $('#regenerateContainer');
        if (el) el.remove();
    }

    // ── Message Rendering ─────────────────────────────────
    function addMessage(role, content, imageDataUrls, animate = true, index, targetUserIndex = null) {
        if (!els.chatContainer) return;

        const text = getMessageText(content);

        // Handle Pairing
        let pairDiv;
        const allPairs = Array.from(els.chatContainer.querySelectorAll('.conversation-pair'));

        if (role === 'user') {
            pairDiv = document.createElement('div');
            pairDiv.className = 'conversation-pair';
            if (!animate) pairDiv.style.animation = 'none';
            const regenBtn = $('#regenerateContainer');
            if (regenBtn) {
                els.chatContainer.insertBefore(pairDiv, regenBtn);
            } else {
                els.chatContainer.appendChild(pairDiv);
            }
        } else {
            // Regeneration path: attach assistant to the exact target user turn.
            if (targetUserIndex !== null && targetUserIndex !== undefined) {
                pairDiv = allPairs.find((p) => {
                    const userMsg = p.querySelector('.message.user');
                    return userMsg && parseInt(userMsg.dataset.index, 10) === targetUserIndex;
                });
                if (pairDiv) {
                    const existingAssistant = pairDiv.querySelector('.message.assistant');
                    if (existingAssistant) existingAssistant.remove();
                }
            }

            // Default path: latest pair with a user message but no assistant message.
            if (!pairDiv) {
                pairDiv = allPairs.reverse().find(p => p.querySelector('.message.user') && !p.querySelector('.message.assistant'));
            }

            if (!pairDiv) {
                pairDiv = document.createElement('div');
                pairDiv.className = 'conversation-pair';
                if (!animate) pairDiv.style.animation = 'none';
                const regenBtn = $('#regenerateContainer');
                if (regenBtn) {
                    els.chatContainer.insertBefore(pairDiv, regenBtn);
                } else {
                    els.chatContainer.appendChild(pairDiv);
                }
            }
        }

        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        msgDiv.dataset.index = index;

        const avatarIcon = role === 'assistant' ? 'M' : '👤';
        const resolvedPromptIndex = role === 'user'
            ? index
            : (
                targetUserIndex !== null && targetUserIndex !== undefined
                    ? targetUserIndex
                    : (index > 0 && state.messages[index - 1] && state.messages[index - 1].role === 'user'
                        ? index - 1
                        : null)
            );
        const hasPromptActions = Number.isInteger(resolvedPromptIndex) && resolvedPromptIndex >= 0;
        if (hasPromptActions) {
            msgDiv.dataset.promptIndex = String(resolvedPromptIndex);
        }

        let imagesHtml = '';
        const imageUrlsToRender = [];
        if (imageDataUrls && Array.isArray(imageDataUrls) && imageDataUrls.length > 0) {
            imageDataUrls.forEach((url, i) => imageUrlsToRender.push({ url, id: `msg-img-${index}-${i}` }));
        } else if (imageDataUrls && typeof imageDataUrls === 'string') {
            imageUrlsToRender.push({ url: imageDataUrls, id: `msg-img-${index}-0` });
        }
        if (imageUrlsToRender.length > 0) {
            imagesHtml = '<div class="message-images-grid">';
            imageUrlsToRender.forEach(({ url, id }) => {
                const escapedUrl = url.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
                imagesHtml += `<img class="message-image" id="${id}" src="${escapedUrl}" alt="Attached medical image" data-lightbox-url>`;
            });
            imagesHtml += '</div>';
        }

        msgDiv.innerHTML = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-content">
                <div class="message-actions">
                    <button class="action-btn copy-btn" title="Copy">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                    ${hasPromptActions ? `
                    <button class="action-btn regen-btn" title="Regenerate from here">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                    </button>
                    <button class="action-btn edit-btn" title="${role === 'user' ? 'Edit' : 'Edit Prompt'}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                    </button>
                    ` : ''}
                    ${role === 'user' ? `
                    <button class="action-btn delete-btn" title="Delete">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                    </button>
                    ` : ''}
                </div>
                ${imagesHtml}
                <div class="content-text">${role === 'assistant' ? renderMarkdown(text, index, state.isStreaming) : escapeHtml(text)}</div>
            </div>
        `;

        pairDiv.appendChild(msgDiv);

        // Bind lightbox click for message images
        imageUrlsToRender.forEach(({ url, id }) => {
            const imgEl = msgDiv.querySelector(`#${id}`);
            if (imgEl) imgEl.addEventListener('click', () => window.openLightbox(url));
        });

        // Selection Toggle
        msgDiv.addEventListener('click', (e) => {
            // Don't toggle selection if clicking an action button, image, or thinking header
            if (e.target.closest('.action-btn') || e.target.closest('.message-image') || e.target.closest('.thinking-header')) return;

            const isSelected = msgDiv.classList.contains('selected');
            $$('.message.selected').forEach(m => m.classList.remove('selected'));
            if (!isSelected) {
                msgDiv.classList.add('selected');
            }
        });

        // Bind actions
        msgDiv.querySelector('.copy-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            const contentEl = msgDiv.querySelector('.content-text');
            // Prefer state.messages for clean text, fallback to DOM if state is out of sync or streaming
            const textToCopy = (state.messages[index] ? getMessageText(state.messages[index].content) : '') || contentEl.innerText.replace('(stopped)', '').trim();

            navigator.clipboard.writeText(textToCopy);
            const btn = msgDiv.querySelector('.copy-btn');
            const oldHtml = btn.innerHTML;
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
            setTimeout(() => btn.innerHTML = oldHtml, 2000);
        });

        const regenBtn = msgDiv.querySelector('.regen-btn');
        if (regenBtn && hasPromptActions) {
            regenBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await regenerateResponse(resolvedPromptIndex);
            });
        }

        const editBtn = msgDiv.querySelector('.edit-btn');
        if (editBtn && hasPromptActions) {
            editBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                editMessage(resolvedPromptIndex);
            });
        }

        if (role === 'user') {
            msgDiv.querySelector('.delete-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteMessage(index);
            });
        }

        if (animate) scrollToBottom();
        return msgDiv;
    }

    function scrollToBottom(force = false) {
        if (!els.chatContainer) return;

        // Strict threshold: only auto-scroll if already at the literal bottom (within 15px)
        const isAtBottom = (els.chatContainer.scrollHeight - els.chatContainer.scrollTop - els.chatContainer.clientHeight) <= 15;

        // If force is true, scroll regardless.
        // If not forced, only scroll if we are already at the bottom AND user hasn't manually scrolled up.
        if (force || (isAtBottom && !state.manualScroll)) {
            els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
        }
    }

    // ── Markdown Rendering ────────────────────────────────
    /**
     * Basic Markdown Parser with Code-Block Protection
     */
    function renderMarkdown(text, msgIdx = -1, isStreaming = false, isInner = false) {
        if (!text) return '';

        const thoughts = [];
        let processedText = text;

        if (!isInner) {
            processedText = text.replace(/<unused94>([\s\S]*?)(?:<unused95>|$)/g, (match, thought) => {
                const isClosed = match.includes('<unused95>');
                const id = `__THOUGHT_${thoughts.length}__`;
                thoughts.push({
                    content: thought.trim(),
                    isClosed: isClosed
                });
                return id;
            });
        }

        const codeBlocks = [];
        let html = escapeHtml(processedText);

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

        // 3. Regular Markdown (Block level)
        const rawLines = html.split('\n');
        let listStack = []; // Elements: { type: 'ol' | 'ul', indent: number }
        const processedLines = [];

        rawLines.forEach(line => {
            const trimmed = line.trim();
            if (trimmed === '' && listStack.length > 0) {
                processedLines.push(line);
                return;
            }

            const olMatch = trimmed.match(/^(\d+)\. /);
            const ulMatch = trimmed.match(/^([-*]) /);
            const indent = line.match(/^\s*/)[0].length;

            if (olMatch || ulMatch) {
                const type = olMatch ? 'ol' : 'ul';
                const marker = olMatch ? olMatch[0] : ulMatch[0];

                while (listStack.length > 0 && listStack[listStack.length - 1].indent > indent) {
                    const popped = listStack.pop();
                    processedLines.push(popped.type === 'ol' ? '</ol>' : '</ul>');
                }

                const current = listStack[listStack.length - 1];

                if (!current || current.indent < indent) {
                    processedLines.push(type === 'ol' ? '<ol>' : '<ul>');
                    listStack.push({ type, indent });
                } else if (current.type !== type) {
                    processedLines.push(current.type === 'ol' ? '</ol>' : '</ul>');
                    listStack.pop();
                    processedLines.push(type === 'ol' ? '<ol>' : '<ul>');
                    listStack.push({ type, indent });
                }

                processedLines.push(`<li>${trimmed.substring(marker.length)}</li>`);
            } else {
                while (listStack.length > 0) {
                    const popped = listStack.pop();
                    processedLines.push(popped.type === 'ol' ? '</ol>' : '</ul>');
                }
                processedLines.push(line);
            }
        });
        while (listStack.length > 0) {
            const popped = listStack.pop();
            processedLines.push(popped.type === 'ol' ? '</ol>' : '</ul>');
        }
        html = processedLines.join('\n');

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Blockquotes
        html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

        // Inline
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // 4. Paragraphs & Line Breaks
        // Split by newlines and wrap non-block lines in <p>
        const blocks = ['h1', 'h2', 'h3', 'ul', 'ol', 'pre', 'blockquote', 'div', 'li'];
        const lines = html.split('\n');
        html = lines.map(line => {
            const trimmed = line.trim();
            if (!trimmed) return '';
            const isBlock = blocks.some(tag => trimmed.startsWith(`<${tag}`));
            const isPlaceholder = /^<p>__(?:THOUGHT|CB)_\d+__<\/p>$/.test(`<p>${trimmed}</p>`) || /^__(?:THOUGHT|CB)_\d+__$/.test(trimmed);
            return (isBlock || isPlaceholder) ? trimmed : `<p>${trimmed}</p>`;
        }).join('\n');

        // Restore Protected Code Blocks
        codeBlocks.forEach((block, i) => {
            html = html.replace(`__CB_${i}__`, block);
        });

        // 5. Restore Thinking Traces
        if (!isInner) {
            thoughts.forEach((thought, i) => {
                const thoughtKey = `${msgIdx}-${i}`;
                const isManuallyExpanded = state.expandedThoughts.has(thoughtKey);

                // It only glows if the thought is NOT closed AND we are currently streaming
                const isProcessing = !thought.isClosed && isStreaming;

                // It should be collapsed if:
                // 1. It is closed (finished) AND not manually expanded
                // 2. We are NO LONGER streaming AND not manually expanded
                let isCollapsed = (thought.isClosed || !isStreaming) && !isManuallyExpanded;

                // Exception: If it's still processing but the user manually collapsed it
                if (isProcessing && state.expandedThoughts.has(thoughtKey + '-collapsed')) {
                    isCollapsed = true;
                }

                const renderedThought = renderMarkdown(thought.content, -1, false, true);

                const thoughtHtml = `
                    <div class="thinking-trace ${isProcessing ? 'processing' : ''} ${isCollapsed ? 'collapsed' : ''}" data-thought-key="${thoughtKey}">
                        <div class="thinking-header" onclick="window.toggleThought('${thoughtKey}', this.parentElement)">
                            Thinking Process
                        </div>
                        <div class="thinking-content">${renderedThought}</div>
                    </div>
                `;
                html = html.replace(`__THOUGHT_${i}__`, thoughtHtml);
            });
        }

        return html;
    }

    window.toggleThought = function (key, el) {
        if (el.classList.contains('collapsed')) {
            el.classList.remove('collapsed');
            state.expandedThoughts.add(key);
            state.expandedThoughts.delete(key + '-collapsed');
        } else {
            el.classList.add('collapsed');
            state.expandedThoughts.delete(key);
            // If it's a processing thought, we need a special marker to keep it collapsed during stream
            if (el.classList.contains('processing')) {
                state.expandedThoughts.add(key + '-collapsed');
            }
        }
        // If we were at the bottom before toggling, stay at the bottom.
        // But do not force a jump if the user is looking at history.
        scrollToBottom(false);
    };

    // Runtime helper for loop-stop tuning without server config.
    window.setLoopStopSensitivity = function (value) {
        const next = clampInt(value, LOOP_STOP_MIN, LOOP_STOP_MAX, state.loopStopSensitivity);
        state.loopStopSensitivity = next;
        safeStorage.set(LOOP_STOP_SENSITIVITY_KEY, String(next));
        updateLoopStopControls();
        showToast(`Loop auto-stop sensitivity set to ${next}`, 'info');
    };

    // Runtime helper for sampling controls.
    window.setSamplingConfig = function (temperature, topP) {
        if (!state.allowClientSamplingConfig) {
            showToast('Sampling is locked by server policy.', 'warning');
            return;
        }
        state.samplingTemperature = snapToStep(
            clampFloat(
                temperature,
                state.samplingTemperatureMin,
                state.samplingTemperatureMax,
                state.samplingTemperature
            ),
            state.samplingTemperatureMin,
            TEMPERATURE_STEP,
            2
        );
        state.samplingTopP = snapToStep(
            clampFloat(
                topP,
                state.samplingTopPMin,
                state.samplingTopPMax,
                state.samplingTopP
            ),
            state.samplingTopPMin,
            TOP_P_STEP,
            2
        );
        safeStorage.set(SAMPLING_TEMPERATURE_KEY, String(state.samplingTemperature));
        safeStorage.set(SAMPLING_TOP_P_KEY, String(state.samplingTopP));
        updateSamplingControls();
        showToast(`Sampling updated (T=${state.samplingTemperature.toFixed(2)}, P=${state.samplingTopP.toFixed(2)})`, 'info');
    };

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Lightbox ──────────────────────────────────────────
    window.openLightbox = function (url) {
        const lightbox = $('#lightbox');
        const lightboxImg = $('#lightboxImg');
        if (lightbox && lightboxImg) {
            state.lightboxTransform = {
                scale: 1, x: 0, y: 0,
                isDragging: false, startX: 0, startY: 0,
                lastTouchDistance: 0, lastTouchCenter: { x: 0, y: 0 }
            };
            updateLightboxTransform();
            lightboxImg.src = url;
            lightbox.classList.remove('hidden');
        }
    };

    window.closeLightbox = function () {
        const lightbox = $('#lightbox');
        if (lightbox) {
            lightbox.classList.add('hidden');
        }
    };

    function updateLightboxTransform() {
        const img = $('#lightboxImg');
        if (!img) return;
        const { scale, x, y } = state.lightboxTransform;
        img.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    }

    function handleLightboxWheel(e) {
        e.preventDefault();
        const img = $('#lightboxImg');
        if (!img) return;

        const rect = img.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const delta = -Math.sign(e.deltaY);
        const factor = 0.1;
        const oldScale = state.lightboxTransform.scale;
        let newScale = oldScale + delta * factor;
        newScale = Math.max(0.1, Math.min(newScale, 10));

        if (newScale !== oldScale) {
            const scaleRatio = newScale / oldScale;
            const dx = mouseX - rect.width / 2;
            const dy = mouseY - rect.height / 2;

            state.lightboxTransform.x -= dx * (scaleRatio - 1);
            state.lightboxTransform.y -= dy * (scaleRatio - 1);
            state.lightboxTransform.scale = newScale;
        }

        // When zooming back to original or smaller, reset position to center
        if (newScale <= 1) {
            state.lightboxTransform.x = 0;
            state.lightboxTransform.y = 0;
        }

        updateLightboxTransform();
    }

    function handleLightboxDragStart(e) {
        // Allow panning always, but it's most useful when zoomed in
        e.preventDefault();
        state.lightboxTransform.isDragging = true;
        state.lightboxTransform.startX = e.clientX - state.lightboxTransform.x;
        state.lightboxTransform.startY = e.clientY - state.lightboxTransform.y;
        e.target.style.cursor = 'grabbing';
    }

    function handleLightboxDragMove(e) {
        if (!state.lightboxTransform.isDragging) return;
        state.lightboxTransform.x = e.clientX - state.lightboxTransform.startX;
        state.lightboxTransform.y = e.clientY - state.lightboxTransform.startY;
        updateLightboxTransform();
    }

    function handleLightboxDragEnd(e) {
        state.lightboxTransform.isDragging = false;
        const img = $('#lightboxImg');
        if (img) img.style.cursor = state.lightboxTransform.scale > 1 ? 'grab' : 'default';
    }

    function handleLightboxTouchStart(e) {
        if (e.touches.length === 1) {
            state.lightboxTransform.isDragging = true;
            state.lightboxTransform.startX = e.touches[0].clientX - state.lightboxTransform.x;
            state.lightboxTransform.startY = e.touches[0].clientY - state.lightboxTransform.y;
        } else if (e.touches.length === 2) {
            state.lightboxTransform.isDragging = false;
            const dist = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );
            state.lightboxTransform.lastTouchDistance = dist;
            state.lightboxTransform.lastTouchCenter = {
                x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
                y: (e.touches[0].clientY + e.touches[1].clientY) / 2
            };
        }
    }

    function handleLightboxTouchMove(e) {
        if (e.touches.length === 1 && state.lightboxTransform.isDragging) {
            e.preventDefault();
            state.lightboxTransform.x = e.touches[0].clientX - state.lightboxTransform.startX;
            state.lightboxTransform.y = e.touches[0].clientY - state.lightboxTransform.startY;
            updateLightboxTransform();
        } else if (e.touches.length === 2) {
            e.preventDefault();
            const img = $('#lightboxImg');
            if (!img) return;

            const rect = img.getBoundingClientRect();
            const dist = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );
            const center = {
                x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
                y: (e.touches[0].clientY + e.touches[1].clientY) / 2
            };

            const oldScale = state.lightboxTransform.scale;
            const factor = dist / state.lightboxTransform.lastTouchDistance;
            let newScale = oldScale * factor;
            newScale = Math.max(0.1, Math.min(newScale, 10));

            if (newScale !== oldScale) {
                const scaleRatio = newScale / oldScale;
                const mouseX = center.x - rect.left;
                const mouseY = center.y - rect.top;
                const dx = mouseX - rect.width / 2;
                const dy = mouseY - rect.height / 2;

                state.lightboxTransform.x -= dx * (scaleRatio - 1);
                state.lightboxTransform.y -= dy * (scaleRatio - 1);
                state.lightboxTransform.scale = newScale;
            }

            // Pan based on center movement
            state.lightboxTransform.x += center.x - state.lightboxTransform.lastTouchCenter.x;
            state.lightboxTransform.y += center.y - state.lightboxTransform.lastTouchCenter.y;

            state.lightboxTransform.lastTouchDistance = dist;
            state.lightboxTransform.lastTouchCenter = center;

            updateLightboxTransform();
        }
    }

    function handleLightboxTouchEnd(e) {
        state.lightboxTransform.isDragging = false;
        if (e.touches.length === 1) {
            // Transition back to single-touch dragging
            state.lightboxTransform.isDragging = true;
            state.lightboxTransform.startX = e.touches[0].clientX - state.lightboxTransform.x;
            state.lightboxTransform.startY = e.touches[0].clientY - state.lightboxTransform.y;
        }
    }

    // ── Boot ──────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
