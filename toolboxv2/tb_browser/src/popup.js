// ToolBox Pro - Main Popup Controller
// Advanced browser extension with AI, voice, and password management

class ToolBoxPopup {
    constructor() {
        this.currentTab = 'isaa';
        this.isVoiceActive = false;
        this.recognition = null;
        this.synthesis = null;
        this.apiBase = 'http://localhost:8080';
        this.isConnected = false;

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
        this.setupEventListeners();
        this.initializeVoiceEngine();
        this.initializeTabSystem();
        this.checkConnection();
        this.loadCurrentPageInfo();

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
    }

    setupEventListeners() {
        // Voice button
        const voiceBtn = document.getElementById('voiceBtn');
        voiceBtn?.addEventListener('click', () => this.toggleVoice());

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
        document.querySelectorAll('.modal-close, .voice-close').forEach(btn => {
            btn.addEventListener('click', (e) => this.closeModal(e.target.closest('.modal, .voice-overlay')));
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
            this.recognition.lang = 'en-US';

            this.recognition.onstart = () => {
                this.isVoiceActive = true;
                this.updateVoiceUI(true);
                this.showVoiceOverlay();
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
                this.hideVoiceOverlay();
            };

            this.recognition.onerror = (event) => {
                console.error('Voice recognition error:', event.error);
                this.isVoiceActive = false;
                this.updateVoiceUI(false);
                this.hideVoiceOverlay();
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
        const voiceIndicator = document.getElementById('voiceIndicator');

        if (active) {
            voiceBtn?.classList.add('active');
            voiceIndicator?.classList.remove('hidden');
        } else {
            voiceBtn?.classList.remove('active');
            voiceIndicator?.classList.add('hidden');
        }
    }

    showVoiceOverlay() {
        const overlay = document.getElementById('voiceOverlay');
        overlay?.classList.remove('hidden');
    }

    hideVoiceOverlay() {
        const overlay = document.getElementById('voiceOverlay');
        overlay?.classList.add('hidden');
    }

    updateVoiceTranscript(transcript) {
        const transcriptEl = document.querySelector('.voice-transcript');
        if (transcriptEl) {
            transcriptEl.textContent = transcript;
        }
    }

    async processVoiceInput(transcript) {
        console.log('Processing voice input:', transcript);

        // Hide voice overlay
        this.hideVoiceOverlay();

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
            this.loadPasswords();
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

        try {
            const response = await this.makeAPICall('/api/isaa/mini_task_completion', 'POST', {
                mini_task: message,
                user_task: 'Browser extension chat',
                agent_name: 'self',
                task_from: 'browser_extension'
            });

            this.hideTypingIndicator();
            this.addChatMessage('isaa', response.result.data || 'I understand. How can I help you further?');

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

    addChatMessage(sender, message) {
        const chatMessages = document.getElementById('chatMessages');
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${sender}-message`;

        if (sender === 'isaa') {
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
                this.sendChatMessage();
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
            const response = await this.makeAPICall('/api/call/PasswordManager/list_passwords', 'POST', {});
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
            const response = await this.makeAPICall('/api/call/PasswordManager/get_password_for_autofill', 'POST', {
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
            const response = await this.makeAPICall('/api/call/PasswordManager/get_password_for_autofill', 'POST', {
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
            const response = await this.makeAPICall('/api/call/PasswordManager/generate_password', 'POST', {
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

                const response = await this.makeAPICall('/api/call/PasswordManager/import_passwords', 'POST', {
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
