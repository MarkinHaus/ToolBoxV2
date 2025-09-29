// ToolBox Pro - Main Popup Controller
// Advanced browser extension with AI, voice, and password management

class ToolBoxPopup {

    constructor() {
        this.currentTab = 'isaa';
        this.isVoiceActive = false;
        this.recognition = null;
        this.chatHistory = [];
        this.synthesis = null;
        this.apiBase = 'http://localhost:8080';
        this.isConnected = false;
        this.currentPageContext = null;
        this.lastSelectedInput = null;
        this.voiceLanguage = 'en-US'; // Default language
        this.supportedLanguages = {
            'en-US': 'English (US)',
            'en-GB': 'English (UK)',
            'de-DE': 'German',
            'fr-FR': 'French',
            'es-ES': 'Spanish',
            'it-IT': 'Italian',
            'pt-PT': 'Portuguese',
            'ru-RU': 'Russian',
            'ja-JP': 'Japanese',
            'ko-KR': 'Korean',
            'zh-CN': 'Chinese (Simplified)'
        };

        this.init();
    }

    async init() {
        console.log('🚀 ToolBox Pro initializing...');

        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            await new Promise(resolve => {
                document.addEventListener('DOMContentLoaded', resolve);
            });
        }

        console.log('📄 DOM ready, initializing components...');

        // Initialize components
        await this.loadSettings();
        this.setupEventListeners();
        this.initializeVoiceEngine();
        this.initializeTabSystem();
        this.checkConnection();
        this.loadCurrentPageInfo();
        this.loadPageContext();

        // Initialize panels
        this.initializeISAAPanel();
        this.initializeSearchPanel();
        this.initializePasswordPanel();

        // Make sure the app is visible
        const app = document.getElementById('app');
        if (app) {
            app.style.display = 'flex';
            console.log('📱 App container made visible');
        } else {
            console.error('❌ App container not found!');
        }

        console.log('✅ ToolBox Pro initialized');

        // Add error handler for the popup
        window.addEventListener('error', (e) => {
            console.error('❌ Popup error:', e.error);
            this.showErrorFallback(e.error?.message || 'Unknown error');
        });
    }

    showErrorFallback(errorMessage) {
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = `
                <div class="error-fallback">
                    <h2>ToolBox Pro</h2>
                    <p>Extension loaded but encountered an error.</p>
                    <p>Error: ${errorMessage}</p>
                    <button id="reloadBtn" class="reload-btn">Reload</button>
                </div>
            `;

            // Add event listener for reload button
            document.getElementById('reloadBtn')?.addEventListener('click', () => {
                window.location.reload();
            });
        }
    }

    async loadSettings() {
        try {
            const settings = await chrome.storage.sync.get(['voiceLanguage']);
            if (settings.voiceLanguage) {
                this.voiceLanguage = settings.voiceLanguage;
            }

            // Update language selector
            const languageSelect = document.getElementById('languageSelect');
            if (languageSelect) {
                languageSelect.value = this.voiceLanguage;
            }
        } catch (error) {
            console.warn('Failed to load settings:', error);
        }
    }

    async saveSettings() {
        try {
            await chrome.storage.sync.set({
                voiceLanguage: this.voiceLanguage
            });
        } catch (error) {
            console.warn('Failed to save settings:', error);
        }
    }

    async loadPageContext() {
        try {
            if (typeof chrome !== 'undefined' && chrome.tabs) {
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                if (tab && tab.url && !tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
                    // Wait a bit for content script to be ready
                    await this.waitForContentScript(tab.id);

                    // Get page index data from content script
                    const response = await chrome.tabs.sendMessage(tab.id, {
                        type: 'GET_PAGE_CONTEXT'
                    });

                    if (response && response.success) {
                        this.currentPageContext = {
                            url: tab.url,
                            title: tab.title,
                            pageIndex: response.pageIndex,
                            summary: response.summary
                        };
                        console.log('📄 Page context loaded:', this.currentPageContext);
                    }
                } else {
                    console.log('📄 Skipping page context for system page');
                }
            }
        } catch (error) {
            console.warn('Failed to load page context:', error);
        }
    }

    async waitForContentScript(tabId, maxAttempts = 3) {
        for (let i = 0; i < maxAttempts; i++) {
            try {
                const response = await chrome.tabs.sendMessage(tabId, { type: 'PING' });
                if (response && response.success) {
                    return true;
                }
            } catch (error) {
                if (i < maxAttempts - 1) {
                    await new Promise(resolve => setTimeout(resolve, 100 * (i + 1)));
                }
            }
        }
        return false;
    }

    setupEventListeners() {
        // Voice button
        const voiceBtn = document.getElementById('voiceBtn');
        voiceBtn?.addEventListener('click', () => this.toggleVoice());

        // Track last selected input for voice targeting
        document.addEventListener('focusin', (e) => {
            if (e.target.matches('input[type="text"], input[type="search"], textarea')) {
                this.lastSelectedInput = e.target;
                console.log('📝 Text input selected for voice targeting');
            }
        });

        // Double-click detection for voice activation
        document.addEventListener('dblclick', (e) => {
            if (e.target.matches('input[type="text"], input[type="search"], textarea')) {
                this.lastSelectedInput = e.target;
                this.toggleVoice();
                e.preventDefault();
            }
        });

        // Language selector
        const languageSelect = document.getElementById('languageSelect');
        languageSelect?.addEventListener('change', (e) => {
            this.voiceLanguage = e.target.value;
            if (this.recognition) {
                this.recognition.lang = this.voiceLanguage;
            }
            this.saveSettings();
            console.log('🌐 Voice language changed to:', this.voiceLanguage);
        });

        // Search input
        const searchField = document.getElementById('searchField');
        const searchSubmit = document.getElementById('searchSubmit');
        searchField?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleSearch();
        });
        searchSubmit?.addEventListener('click', () => this.handleSearch());

        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.currentTarget.dataset.tab;
                this.switchTab(tab);
            });
        });

        // Chat input
        const chatInput = document.getElementById('chatInput');
        const chatSend = document.getElementById('chatSend');
        chatInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendChatMessage();
            }
        });
        chatSend?.addEventListener('click', () => this.sendChatMessage());

        // Live search
        const liveSearchInput = document.getElementById('liveSearchInput');
        liveSearchInput?.addEventListener('input', (e) => {
            this.debounce(() => this.performLiveSearch(e.target.value), 300);
        });

        // Password actions
        document.getElementById('autofillBtn')?.addEventListener('click', () => this.autofillPassword());
        document.getElementById('generateBtn')?.addEventListener('click', () => this.generatePassword());
        document.getElementById('importBtn')?.addEventListener('click', () => this.importPasswords());
        document.getElementById('indexPageBtn')?.addEventListener('click', () => this.indexCurrentPage());

        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => this.closeModal(e.target.closest('.modal')));
        });

        // TOTP copy button
        document.getElementById('copyTotpBtn')?.addEventListener('click', () => this.copyTotpCode());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }

    initializeVoiceEngine() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();

            this.recognition.continuous = false;
            this.recognition.interimResults = true;
            this.recognition.lang = this.voiceLanguage;

            this.recognition.onstart = () => {
                this.isVoiceActive = true;
                this.updateVoiceUI(true);
            };

            this.recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0].transcript)
                    .join('');

                this.updateVoiceTranscript(transcript);

                if (event.results[event.results.length - 1].isFinal) {
                    this.processVoiceInput(transcript);
                }
            };

            this.recognition.onend = () => {
                this.isVoiceActive = false;
                this.updateVoiceUI(false);
            };

            this.recognition.onerror = (event) => {
                console.error('Voice recognition error:', event.error);
                this.isVoiceActive = false;
                this.updateVoiceUI(false);
            };
        }

        // Initialize speech synthesis
        if ('speechSynthesis' in window) {
            this.synthesis = window.speechSynthesis;
        }
    }

    initializeTabSystem() {
        // Set initial active tab
        this.switchTab(this.currentTab);
    }

    async checkConnection() {
        const indicator = document.getElementById('connectionIndicator');
        const text = document.getElementById('connectionText');

        try {
            // Just check if the server is reachable
            const response = await fetch(this.apiBase);
            this.isConnected = response.ok;
            indicator?.classList.remove('connecting', 'error');
            indicator?.classList.add('connected');
            if (text) text.textContent = 'Connected';
        } catch (error) {
            this.isConnected = false;
            indicator?.classList.remove('connecting', 'connected');
            indicator?.classList.add('error');
            if (text) text.textContent = 'Disconnected';
        }
    }

    async loadCurrentPageInfo() {
        try {
            // Check if we're in extension context
            if (typeof chrome !== 'undefined' && chrome.tabs) {
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                this.currentTab = tab;

                const pageTitle = document.getElementById('pageTitle');
                if (pageTitle && tab) {
                    pageTitle.textContent = tab.title || 'Unknown Page';
                }
            } else {
                // Fallback for when popup is opened standalone
                const pageTitle = document.getElementById('pageTitle');
                if (pageTitle) {
                    pageTitle.textContent = 'ToolBox Pro Extension';
                }
            }
        } catch (error) {
            console.error('Failed to load page info:', error);
            // Fallback
            const pageTitle = document.getElementById('pageTitle');
            if (pageTitle) {
                pageTitle.textContent = 'ToolBox Pro Extension';
            }
        }
    }

    // Voice Methods
    toggleVoice() {
        if (this.isVoiceActive) {
            this.stopVoice();
        } else {
            this.startVoice();
        }
    }

    startVoice() {
        if (this.recognition && !this.isVoiceActive) {
            try {
                this.recognition.start();
            } catch (error) {
                console.error('Failed to start voice recognition:', error);
            }
        }
    }

    stopVoice() {
        if (this.recognition && this.isVoiceActive) {
            this.recognition.stop();
        }
    }

    updateVoiceUI(active) {
        const voiceBtn = document.getElementById('voiceBtn');
        const statusDot = voiceBtn?.querySelector('.voice-status-dot');

        if (active) {
            voiceBtn?.classList.add('active');
            if (statusDot) {
                statusDot.classList.add('listening');
                statusDot.classList.remove('ready');
            }
            this.showVoiceTranscript();
        } else {
            voiceBtn?.classList.remove('active');
            if (statusDot) {
                statusDot.classList.remove('listening');
                statusDot.classList.add('ready');
            }
            this.hideVoiceTranscript();
        }
    }

    showVoiceTranscript() {
        // Create transcript display if it doesn't exist
        let transcriptEl = document.getElementById('voiceTranscript');
        if (!transcriptEl) {
            transcriptEl = document.createElement('div');
            transcriptEl.id = 'voiceTranscript';
            transcriptEl.className = 'voice-transcript-display';
            transcriptEl.innerHTML = `
                <div class="transcript-content">
                    <span class="transcript-label">🎤 Listening...</span>
                    <span class="transcript-text"></span>
                </div>
            `;
            document.body.appendChild(transcriptEl);
        }

        transcriptEl.classList.remove('hidden');
        transcriptEl.classList.add('visible');
    }

    hideVoiceTranscript() {
        const transcriptEl = document.getElementById('voiceTranscript');
        if (transcriptEl) {
            transcriptEl.classList.remove('visible');
            transcriptEl.classList.add('hidden');

            // Clear transcript after hiding
            setTimeout(() => {
                const textEl = transcriptEl.querySelector('.transcript-text');
                if (textEl) textEl.textContent = '';
            }, 300);
        }
    }

    updateVoiceTranscript(transcript) {
        const transcriptEl = document.getElementById('voiceTranscript');
        const textEl = transcriptEl?.querySelector('.transcript-text');

        if (textEl) {
            textEl.textContent = transcript;

            // Update label based on transcript content
            const labelEl = transcriptEl.querySelector('.transcript-label');
            if (labelEl) {
                if (transcript.trim()) {
                    labelEl.textContent = '🎤 You said:';
                } else {
                    labelEl.textContent = '🎤 Listening...';
                }
            }
        }

        // Also show live transcript in target input field (if available)
        if (this.lastSelectedInput && this.lastSelectedInput.isConnected && transcript.trim()) {
            // Store original placeholder to restore later
            if (!this.lastSelectedInput.dataset.originalPlaceholder) {
                this.lastSelectedInput.dataset.originalPlaceholder = this.lastSelectedInput.placeholder || '';
            }

            // Show transcript as placeholder
            this.lastSelectedInput.placeholder = `🎤 "${transcript}"`;
            this.lastSelectedInput.classList.add('voice-preview');
        }
    }

    clearVoicePreview() {
        if (this.lastSelectedInput && this.lastSelectedInput.isConnected) {
            // Restore original placeholder
            if (this.lastSelectedInput.dataset.originalPlaceholder !== undefined) {
                this.lastSelectedInput.placeholder = this.lastSelectedInput.dataset.originalPlaceholder;
                delete this.lastSelectedInput.dataset.originalPlaceholder;
            }
            this.lastSelectedInput.classList.remove('voice-preview');
        }
    }

    async processVoiceInput(transcript) {
        console.log('Processing voice input:', transcript);

        // If there's a selected text input, route voice input there
        if (this.lastSelectedInput && this.lastSelectedInput.isConnected) {
            this.lastSelectedInput.value = transcript;
            this.lastSelectedInput.focus();

            // Trigger input event for any listeners
            this.lastSelectedInput.dispatchEvent(new Event('input', { bubbles: true }));

            console.log('📝 Voice input routed to text field');
            return;
        }

        // Determine intent and route to appropriate handler
        if (transcript.toLowerCase().includes('search')) {
            this.switchTab('search');
            const searchInput = document.getElementById('liveSearchInput');
            if (searchInput) {
                searchInput.value = transcript.replace(/search\s*/i, '');
                this.performLiveSearch(searchInput.value);
            }
        } else if (transcript.toLowerCase().includes('password')) {
            this.switchTab('passwords');
            if (transcript.toLowerCase().includes('fill') || transcript.toLowerCase().includes('autofill')) {
                this.autofillPassword();
            }
        } else {
            // Default to ISAA chat
            this.switchTab('isaa');
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = transcript;
                this.sendChatMessage();
            }
        }
    }

    async speak(text, options = {}) {
        if (this.synthesis) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = options.rate || 1.0;
            utterance.pitch = options.pitch || 1.0;
            utterance.volume = options.volume || 1.0;

            this.synthesis.speak(utterance);
        }
    }

    // Tab System
    switchTab(tabName) {
        this.currentTab = tabName;

        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.tab === tabName) {
                btn.classList.add('active');
            }
        });

        // Update panel content
        document.querySelectorAll('.panel-content').forEach(panel => {
            panel.classList.remove('active');
        });

        const activePanel = document.getElementById(`${tabName}PanelContent`);
        activePanel?.classList.add('active');

        // Initialize panel if needed
        if (tabName === 'passwords') {
            this.loadPasswords().catch(console.error);
        }
    }

    // Search Methods
    handleSearch() {
        const searchField = document.getElementById('searchField');
        const query = searchField?.value.trim();

        if (query) {
            // Show search input
            const searchInput = document.getElementById('searchInput');
            searchInput?.classList.remove('hidden');

            // Switch to appropriate tab based on query
            if (query.toLowerCase().includes('password')) {
                this.switchTab('passwords');
            } else {
                this.switchTab('search');
                const liveSearchInput = document.getElementById('liveSearchInput');
                if (liveSearchInput) {
                    liveSearchInput.value = query;
                    this.performLiveSearch(query);
                }
            }
        }
    }

    // Utility Methods
    debounce(func, wait) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(func, wait);
    }

    handleKeyboardShortcuts(e) {
        if (e.ctrlKey && e.shiftKey) {
            switch (e.key) {
                case 'V':
                    e.preventDefault();
                    this.toggleVoice();
                    break;
                case 'F':
                    e.preventDefault();
                    this.switchTab('search');
                    document.getElementById('liveSearchInput')?.focus();
                    break;
                case 'P':
                    e.preventDefault();
                    this.autofillPassword();
                    break;
            }
        }

        if (e.altKey && e.key === 't') {
            e.preventDefault();
            window.close();
        }
    }

    closeModal(modal) {
        modal?.classList.add('hidden');
    }

    // ISAA Panel Methods
    initializeISAAPanel() {
        // Panel is initialized with welcome message in HTML
        console.log('ISAA panel initialized');
    }

    addMessageToHistory(role, content) {
        this.chatHistory.push({ role, content });
    }

    async sendChatMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput?.value.trim();

        if (!message) return;

        // Clear input
        chatInput.value = '';

        // Add user message to chat
        this.addChatMessage('user', message);

        // Show typing indicator
        this.showTypingIndicator();

        if (this.chatHistory.length > 10) {
            this.chatHistory.shift();
        }

        try {
            // Prepare context with page information
            const contextData = {
                mini_task: message,
                user_task: 'Browser extension chat with page context '+ `User is on ${JSON.stringify(this.currentPageContext)}`,
                agent_name: 'speed',
                task_from: 'browser_extension',
                message_history: this.chatHistory,
            };

            const response = await this.makeAPICall('/api/isaa/mini_task_completion', 'POST', contextData);

            this.hideTypingIndicator();

            const responseText = response.result?.data || response.data || 'I understand. How can I help you further?';
            this.addMessageToHistory('user', message);
            this.addChatMessage('isaa', responseText);

            // Check if this looks like an action request
            if (this.isActionRequest(message)) {
                this.addChatMessage('isaa', "action");
                try {
                    const action = await this.executeStructuredAction(message+" Last agent response: "+responseText);
                    if (action) {
                        this.addChatMessage('isaa', `🎯 I've planned to ${action.action_type} on "${action.target_selector}"`);
                    }
                } catch (actionError) {
                    console.warn('Action execution failed:', actionError);
                }
            }

            // Handle TTS if requested
            if (response.speak) {
                this.speak(response.speak);
            }

        } catch (error) {
            this.hideTypingIndicator();
            this.addChatMessage('isaa', 'Sorry, I encountered an error. Please try again.');
            console.error('ISAA API error:', error);
        }
    }

    isActionRequest(message) {
        const actionKeywords = [
            'click', 'press', 'tap', 'select',
            'fill', 'enter', 'type', 'input',
            'navigate', 'go to', 'visit', 'open',
            'scroll', 'find', 'search for',
            'extract', 'get', 'copy', 'download',
            // de key words
            "klicke", "drücke", "tippe", "wähle",
            "fülle", "gebe", "trage", "eingabe",
            "navigiere", "gehe zu", "öffne", "besuche",
            "scroll", "finde", "suche nach",
            "extrahiere", "hole", "kopiere", "lade herunter"
        ];

        const lowerMessage = message.toLowerCase();
        return actionKeywords.some(keyword => lowerMessage.includes(keyword));
    }

    async executeStructuredAction(message) {
        try {
            // Define action schema for format_class
            const actionSchema = {'properties': {'action_type': {'enum': ['click',
                            'fill_form',
                            'navigate',
                            'scroll',
                            'extract_data'],
                           'title': 'Action Type',
                           'type': 'string'},
                          'target_selector': {'title': 'Target Selector', 'type': 'string'},
                          'data': {'anyOf': [{'additionalProperties': true, 'type': 'object'},
                            {'type': 'null'}],
                           'default': null,
                           'title': 'Data'},
                          'confirmation_needed': {'title': 'Confirmation Needed', 'type': 'boolean'}},
                         'required': ['action_type', 'target_selector', 'confirmation_needed'],
                         'title': 'ActionSchema',
                         'type': 'object'};

            const contextInfo = this.currentPageContext ?
                `Current page: "${JSON.stringify(this.currentPageContext)}"` :
                'Current page: Unknown';

            const response = await this.makeAPICall('/api/isaa/format_class', 'POST', {
                format_schema: actionSchema,
                task: `Analyze this user request and determine the web page action needed: "${message}".
                      ${contextInfo}

                      Action Guidelines:
                      - For navigation: Use "navigate" action_type, put URL/path in target_selector (e.g., "/blog", "https://example.com", "about.html")
                      - For clicking: Use "click" action_type, put element selector in target_selector
                      - For form filling: Use "fill_form" action_type, put input selector in target_selector, value in data.value
                      - For scrolling: Use "scroll" action_type, put direction in data.direction ("up"/"down")
                      - For data extraction: Use "extract_data" action_type, put element selector in target_selector

                      Examples:
                      - "go to blog" → {"action_type": "navigate", "target_selector": "/blog", "confirmation_needed": false}
                      - "click login button" → {"action_type": "click", "target_selector": "button[type='submit']", "confirmation_needed": false}`,
                agent_name: 'speed',
                auto_context: true
            });

            if (response.result?.data || response.data) {
                const action = response.result?.data || response.data;
                console.log('🎯 Structured action planned:', action);

                // Execute the action via content script
                await this.executePageAction(action);

                return action;
            }
        } catch (error) {
            console.error('Structured action error:', error);
            throw error;
        }
    }

    async executePageAction(action) {
        try {
            if (typeof chrome !== 'undefined' && chrome.tabs) {
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

                // Check if tab is valid for action execution
                if (!tab || tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
                    this.addChatMessage('isaa', `❌ Cannot execute actions on system pages`);
                    return { success: false, error: 'System page not supported' };
                }

                // Wait for content script to be ready
                const isReady = await this.waitForContentScript(tab.id);
                if (!isReady) {
                    this.addChatMessage('isaa', `❌ Content script not ready. Please refresh the page.`);
                    return { success: false, error: 'Content script not ready' };
                }

                console.log('🎯 Executing action:', action);

                const response = await chrome.tabs.sendMessage(tab.id, {
                    type: 'EXECUTE_ACTION',
                    action: action
                });

                if (response && response.success) {
                    this.addChatMessage('isaa', `✅ ${response.message || `Action completed: ${action.action_type}`}`);
                } else {
                    this.addChatMessage('isaa', `❌ ${response?.error || 'Action failed'}`);
                }

                return response;
            }
        } catch (error) {
            console.error('Page action execution error:', error);

            // Provide more specific error messages
            if (error.message.includes('Receiving end does not exist')) {
                this.addChatMessage('isaa', `❌ Page not ready for actions. Please refresh the page and try again.`);
            } else {
                this.addChatMessage('isaa', `❌ Could not execute action: ${error.message}`);
            }

            return { success: false, error: error.message };
        }
    }

    addChatMessage(sender, message) {
        const chatMessages = document.getElementById('chatMessages');
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${sender}-message`;

        if (sender === 'isaa') {
            this.addMessageToHistory('assistant', message);
            messageEl.innerHTML = `
                <div class="isaa-avatar">🤖</div>
                <div class="message-content">
                    <p>${this.escapeHtml(message)}</p>
                </div>
            `;
        } else {
            messageEl.innerHTML = `
                <div class="message-content user-content">
                    <p>${this.escapeHtml(message)}</p>
                </div>
                <div class="user-avatar">👤</div>
            `;
        }

        chatMessages?.appendChild(messageEl);
        chatMessages?.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        const typingEl = document.createElement('div');
        typingEl.className = 'chat-message isaa-message typing-indicator';
        typingEl.id = 'typingIndicator';
        typingEl.innerHTML = `
            <div class="isaa-avatar">🤖</div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;

        chatMessages?.appendChild(typingEl);
        chatMessages?.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator?.remove();
    }

    // Search Panel Methods
    initializeSearchPanel() {
        console.log('Search panel initialized');
    }

    async performLiveSearch(query) {
        if (!query.trim()) {
            this.showNoResults();
            return;
        }

        try {
            // Send search request to content script
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await chrome.tabs.sendMessage(tab.id, {
                type: 'SEARCH_PAGE',
                query: query
            });

            this.displaySearchResults(response.results || []);

        } catch (error) {
            console.error('Search error:', error);
            this.showNoResults('Search failed. Please try again.');
        }
    }

    displaySearchResults(results) {
        const searchResults = document.getElementById('searchResults');

        if (!results.length) {
            this.showNoResults();
            return;
        }

        searchResults.innerHTML = results.map(result => `
            <div class="search-result-item" data-element-id="${result.id}">
                <div class="result-title">${this.escapeHtml(result.title)}</div>
                <div class="result-snippet">${this.escapeHtml(result.snippet)}</div>
                <div class="result-actions">
                    <button class="result-action" onclick="toolboxPopup.scrollToElement('${result.id}')">
                        Scroll to
                    </button>
                    <button class="result-action" onclick="toolboxPopup.askISAAAboutSection('${result.id}')">
                        Ask ISAA
                    </button>
                </div>
            </div>
        `).join('');
    }

    showNoResults(message = 'No results found. Try a different search term.') {
        const searchResults = document.getElementById('searchResults');
        searchResults.innerHTML = `
            <div class="no-results">
                <p>${message}</p>
            </div>
        `;
    }

    async scrollToElement(elementId) {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, {
                type: 'SCROLL_TO_ELEMENT',
                elementId: elementId
            });
        } catch (error) {
            console.error('Scroll error:', error);
        }
    }

    async askISAAAboutSection(elementId) {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await chrome.tabs.sendMessage(tab.id, {
                type: 'GET_ELEMENT_CONTENT',
                elementId: elementId
            });

            // Switch to ISAA tab and ask about the section
            this.switchTab('isaa');
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = `Tell me about this section: "${response.content}"`;
                await this.sendChatMessage();
            }
        } catch (error) {
            console.error('ISAA section query error:', error);
        }
    }

    async indexCurrentPage() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, {
                type: 'INDEX_PAGE'
            });

            // Show success message
            this.showNoResults('Page indexed successfully! You can now search through all content.');
        } catch (error) {
            console.error('Page indexing error:', error);
            this.showNoResults('Failed to index page. Please try again.');
        }
    }

    // Password Manager Methods
    initializePasswordPanel() {
        console.log('Password panel initialized');
    }

    async loadPasswords() {
        const passwordList = document.getElementById('passwordList');

        try {
            const response = await this.makeAPICall('/api/PasswordManager/list_passwords', 'POST', {});
            const passwords = response.data || [];

            if (passwords.length === 0) {
                passwordList.innerHTML = `
                    <div class="no-passwords">
                        <p>No passwords found. Import from your browser or add new ones.</p>
                    </div>
                `;
                return;
            }

            passwordList.innerHTML = passwords.map(pwd => `
                <div class="password-item" data-id="${pwd.id}">
                    <div class="password-title">${this.escapeHtml(pwd.title || pwd.url)}</div>
                    <div class="password-url">${this.escapeHtml(pwd.url)}</div>
                    <div class="password-meta">
                        <span>Username: ${this.escapeHtml(pwd.username)}</span>
                        ${pwd.has_totp ? '<span class="totp-badge">2FA</span>' : ''}
                    </div>
                </div>
            `).join('');

            // Add click handlers
            document.querySelectorAll('.password-item').forEach(item => {
                item.addEventListener('click', () => {
                    const passwordId = item.dataset.id;
                    this.usePassword(passwordId);
                });
            });

        } catch (error) {
            console.error('Failed to load passwords:', error);
            passwordList.innerHTML = `
                <div class="error-message">
                    <p>Failed to load passwords. Please check your connection.</p>
                </div>
            `;
        }
    }

    async usePassword(passwordId) {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            // Get password details
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', {
                url: tab.url
            });

            if (response.data) {
                // Send to content script for autofill
                await chrome.tabs.sendMessage(tab.id, {
                    type: 'AUTOFILL_PASSWORD',
                    data: response.data
                });

                // Show TOTP if available
                if (response.data.totp_code) {
                    this.showTOTPModal(response.data.totp_code, response.data.title, response.data.time_remaining);
                }

                // Close popup after successful autofill
                setTimeout(() => window.close(), 1000);
            }

        } catch (error) {
            console.error('Password autofill error:', error);
        }
    }

    async autofillPassword() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', {
                url: tab.url
            });

            if (response.data) {
                await chrome.tabs.sendMessage(tab.id, {
                    type: 'AUTOFILL_PASSWORD',
                    data: response.data
                });

                if (response.data.totp_code) {
                    this.showTOTPModal(response.data.totp_code, response.data.title, response.data.time_remaining);
                }

                setTimeout(() => window.close(), 1000);
            } else {
                this.speak('No password found for this website.');
            }

        } catch (error) {
            console.error('Autofill error:', error);
            this.speak('Failed to autofill password.');
        }
    }

    async generatePassword() {
        try {
            const response = await this.makeAPICall('/api/PasswordManager/generate_password', 'POST', {
                length: 16,
                include_symbols: true,
                include_numbers: true,
                include_uppercase: true,
                include_lowercase: true,
                exclude_ambiguous: true
            });

            if (response.data && response.data.password) {
                // Copy to clipboard
                await navigator.clipboard.writeText(response.data.password);

                // Show notification
                this.speak('Password generated and copied to clipboard.');

                // Send to content script for filling
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                await chrome.tabs.sendMessage(tab.id, {
                    type: 'FILL_GENERATED_PASSWORD',
                    password: response.data.password
                });
            }

        } catch (error) {
            console.error('Password generation error:', error);
            this.speak('Failed to generate password.');
        }
    }

    async importPasswords() {
        // Create file input
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.csv,.json';

        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            try {
                const content = await file.text();
                const format = file.name.endsWith('.json') ? 'json' : 'chrome';

                const response = await this.makeAPICall('/api/PasswordManager/import_passwords', 'POST', {
                    file_content: content,
                    file_format: format,
                    folder: 'Imported from Browser'
                });

                if (response.data && response.data.imported_count) {
                    this.speak(`Successfully imported ${response.data.imported_count} passwords.`);
                    this.loadPasswords(); // Refresh the list
                }

            } catch (error) {
                console.error('Import error:', error);
                this.speak('Failed to import passwords.');
            }
        };

        input.click();
    }

    showTOTPModal(code, account, timeRemaining) {
        const modal = document.getElementById('totpModal');
        const codeEl = document.getElementById('totpCode');
        const accountEl = document.getElementById('totpAccount');
        const timerEl = document.getElementById('totpTimer');
        const timeLeftEl = document.getElementById('totpTimeLeft');

        if (codeEl) codeEl.textContent = code;
        if (accountEl) accountEl.textContent = account || 'Account';
        if (timeLeftEl) timeLeftEl.textContent = timeRemaining || 30;

        modal?.classList.remove('hidden');

        // Start countdown
        this.startTOTPCountdown(timeRemaining || 30);
    }

    startTOTPCountdown(seconds) {
        const timerEl = document.getElementById('totpTimer');
        const timeLeftEl = document.getElementById('totpTimeLeft');

        const interval = setInterval(() => {
            seconds--;
            if (timeLeftEl) timeLeftEl.textContent = seconds;

            // Update visual timer
            if (timerEl) {
                const percentage = (seconds / 30) * 360;
                timerEl.style.background = `conic-gradient(var(--marine-blue) ${percentage}deg, transparent ${percentage}deg)`;
            }

            if (seconds <= 0) {
                clearInterval(interval);
                this.closeModal(document.getElementById('totpModal'));
            }
        }, 1000);
    }

    async copyTotpCode() {
        const codeEl = document.getElementById('totpCode');
        if (codeEl) {
            try {
                await navigator.clipboard.writeText(codeEl.textContent);
                this.speak('TOTP code copied to clipboard.');
            } catch (error) {
                console.error('Failed to copy TOTP code:', error);
            }
        }
    }

    async makeAPICall(endpoint, method = 'POST', data = null) {
        const url = `${this.apiBase}${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`API call failed: ${response.status}`);
        }

        return await response.json();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.toolboxPopup = new ToolBoxPopup();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolBoxPopup;
}
