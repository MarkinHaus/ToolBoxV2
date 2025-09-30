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
        this.currentAudio = null;
        this.currentlyPlayingButton = null;
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
        this.currentPageContext = await this.getFreshPageContext();
        if (this.currentPageContext) {
            console.log('📄 Initial page context loaded:', this.currentPageContext);
        }
        return;
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

        const chatMessages = document.getElementById('chatMessages');
        chatMessages?.addEventListener('click', (e) => {
            const speakButton = e.target.closest('.speak-btn');
            if (speakButton) {
                const text = speakButton.dataset.text;
                this.playTTS(text, speakButton);
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
        document.getElementById('importBtn')?.addEventListener('click', () => {
            const modal = document.getElementById('importHelperModal');
            modal?.classList.remove('hidden');
        });

        document.getElementById('addPasswordBtn')?.addEventListener('click', () => {
            const modal = document.getElementById('addPasswordModal');
            const urlInput = document.getElementById('add-url');
            const titleInput = document.getElementById('add-title');

            // Setze die Felder mit den Daten der aktuellen Seite vor
            if (this.currentPageContext && this.currentPageContext.url) {
                const currentUrl = this.currentPageContext.url;
                urlInput.value = currentUrl;

                // Extrahiere einen sauberen Titel (Hostname) aus der URL
                try {
                    const hostname = new URL(currentUrl).hostname;
                    // Entferne "www.", falls vorhanden, für einen saubereren Titel
                    titleInput.value = hostname.replace(/^www\./, '');
                } catch (e) {
                    // Fallback, falls die URL ungültig ist
                    titleInput.value = this.currentPageContext.title || '';
                }
            } else {
                // Leere die Felder, falls kein Seitenkontext verfügbar ist
                urlInput.value = '';
                titleInput.value = '';
            }

            modal?.classList.remove('hidden');
        });

        // NEU: Listener für das Absenden des Formulars
        document.getElementById('addPasswordForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            console.log('Form submitted');
            this.handleAddNewPassword();
        });

        // NEUER LISTENER FÜR DEN "DATEI AUSWÄHLEN"-BUTTON IM MODAL
        document.getElementById('openImportDialogBtn')?.addEventListener('click', () => {
             const modal = document.getElementById('importHelperModal');
             modal?.classList.add('hidden');
             this.triggerImportFileDialog();
        });

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

    async handleAddNewPassword() {
        const form = document.getElementById('addPasswordForm');
        const submitBtn = document.getElementById('addPasswordSubmitBtn');
        const feedbackEl = document.getElementById('addPasswordFeedback');

        // Formulardaten auslesen
        const title = document.getElementById('add-title').value.trim();
        const url = document.getElementById('add-url').value.trim();
        const username = document.getElementById('add-username').value.trim();
        const password = document.getElementById('add-password').value;

        // Feedback zurücksetzen
        feedbackEl.textContent = '';
        feedbackEl.className = 'form-feedback';

        // 1. Validierung: Sicherstellen, dass alle Felder ausgefüllt sind
        if (!title || !url || !username || !password) {
            feedbackEl.textContent = 'Bitte füllen Sie alle Felder aus.';
            feedbackEl.classList.add('error');
            return;
        }

        // 2. Button deaktivieren, um doppelte Klicks zu verhindern
        submitBtn.disabled = true;
        submitBtn.textContent = 'Speichern...';

        try {
            // 3. API-Aufruf an das Backend
            const response = await this.makeAPICall('/api/PasswordManager/add_password', 'POST', {
                title,
                url,
                username,
                password,
                folder: 'Default' // Optional: Standardordner setzen
            });

            // 4. Antwort auswerten
            if (response.result?.status === 'ok' || response.status === 'ok') {
                feedbackEl.textContent = 'Erfolgreich gespeichert!';
                feedbackEl.classList.add('success');

                // Nach kurzem Warten Modal schließen und Liste neu laden
                setTimeout(() => {
                    this.closeModal(document.getElementById('addPasswordModal'));
                    form.reset();
                    this.loadPasswords();
                }, 1000);

            } else {
                // Fehler vom Backend anzeigen (z.B. "Eintrag existiert bereits")
                const errorMessage = response.result?.info || response.info || 'Ein unbekannter Fehler ist aufgetreten.';
                feedbackEl.textContent = errorMessage;
                feedbackEl.classList.add('error');
            }
        } catch (error) {
            // Netzwerkfehler oder andere Probleme anzeigen
            console.error('Failed to add new password:', error);
            feedbackEl.textContent = 'Verbindung zum Server fehlgeschlagen.';
            feedbackEl.classList.add('error');
        } finally {
            // 5. Button im Erfolgs- oder Fehlerfall wieder aktivieren
            submitBtn.disabled = false;
            submitBtn.textContent = 'Speichern';
        }
    }

    async playTTS(text, buttonElement) {
            // --- STOPP-LOGIK ---
            // Wenn ein Audio-Objekt existiert und gerade abgespielt wird
            if (this.currentAudio && !this.currentAudio.paused) {
                // Stoppe die Wiedergabe
                this.currentAudio.pause();
                this.currentAudio.currentTime = 0; // Spule zurück zum Anfang

                // Setze den "playing"-Status des alten Buttons zurück
                if (this.currentlyPlayingButton) {
                    this.currentlyPlayingButton.classList.remove('playing');
                }

                // Wenn der Benutzer denselben Button geklickt hat, um zu stoppen, sind wir fertig.
                if (this.currentlyPlayingButton === buttonElement) {
                    this.currentAudio = null;
                    this.currentlyPlayingButton = null;
                    return;
                }
            }

            // --- WIEDERGABE-LOGIK ---
            if (!text) return;

            buttonElement.classList.add('loading');

            try {
                const response = await this.makeAPICall('/api/TTS/speak', 'POST', {
                    text: text,
                    lang: this.voiceLanguage.split('-')[0]
                });

                if (response.result?.data || response.result?.data?.audio_content) {
                    const audioBase64 = response.result.data.audio_content;
                    const audioSrc = `data:audio/mp3;base64,${audioBase64}`;

                    // Erstelle ein neues Audio-Objekt und speichere es im Klassen-Zustand
                    this.currentAudio = new Audio(audioSrc);
                    this.currentlyPlayingButton = buttonElement;

                    buttonElement.classList.remove('loading');
                    buttonElement.classList.add('playing');

                    this.currentAudio.play();

                    // Event-Listener, um den Zustand nach Ende der Wiedergabe aufzuräumen
                    this.currentAudio.onended = () => {
                        buttonElement.classList.remove('playing');
                        this.currentAudio = null;
                        this.currentlyPlayingButton = null;
                    };
                }
            } catch (error) {
                console.error('TTS API error:', error);
                buttonElement.classList.remove('loading');
                // Räume auch im Fehlerfall den Zustand auf
                this.currentAudio = null;
                this.currentlyPlayingButton = null;
            }
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
            const response = await fetch(this.apiBase+'/api/CloudM/Version');
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
            this.lastSelectedInput = window.toolboxContent?.lastFocusedInput || this.lastSelectedInput;
            console.log('lastSelectedInput', this.lastSelectedInput);
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
        console.log('Updating voice transcript:', transcript, this.lastSelectedInput);
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

        // --- PRIORITÄT 1: Versuche, den Text in die aktive Webseite einzufügen ---
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tab && tab.id) {
                // Sende den Text an das content script und warte auf eine Antwort.
                const response = await chrome.tabs.sendMessage(tab.id, {
                    type: 'INSERT_TEXT_INTO_FOCUSED_INPUT',
                    text: transcript
                });

                // Wenn das content script erfolgreich war, ist unsere Arbeit hier erledigt.
                if (response && response.success) {
                    console.log('✅ Voice input successfully handled by the active web page.');
                    return;
                }
            }
        } catch (error) {
            // Dieser Fehler ist normal, wenn die Seite geschützt ist (z.B. chrome://)
            // oder das content script noch nicht geladen wurde. Wir machen einfach weiter.
            console.warn('Could not communicate with content script, proceeding to internal handling.', error.message);
        }

        // --- PRIORITÄT 2: Prüfen, ob ein Feld INNERHALB des Popups fokussiert ist ---
        if (this.lastSelectedInput && this.lastSelectedInput.isConnected) {
            this.lastSelectedInput.value = transcript;
            this.lastSelectedInput.focus();
            this.lastSelectedInput.dispatchEvent(new Event('input', { bubbles: true }));
            console.log('📝 Voice input routed to an input field inside the popup.');
            return;
        }

        // --- FALLBACK: Wenn niemand den Text wollte, als Chat-Nachricht behandeln ---
        console.log('No specific input field found, defaulting to ISAA chat.');
        this.switchTab('isaa');
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.value = transcript;
            this.sendChatMessage();
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
        }else if(tabName === 'isaa'){
            // focus input lement for direct typing
            const inputElement= document.getElementById('chatInput');

            inputElement?.focus();

        } else if (tabName === 'search') {
            const inputElement= document.getElementById('liveSearchInput')
            inputElement?.focus();
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
        const agentModeToggle = document.getElementById('agentModeSwitch')

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
            // Logik basierend auf dem Zustand des Schalters
            if (agentModeToggle.checked) {
                this.hideTypingIndicator();
                await this.runAgentLoop(message);
            } else {
            // Prepare context with page information
             const freshContext = await this.getFreshPageContext();

                const contextData = {
                    mini_task: message,
                    user_task: 'Browser extension chat with page context in lang must response in '+this.voiceLanguage + ` User is on ${JSON.stringify(freshContext)}`,
                    agent_name: 'speed',
                    task_from: 'browser_extension',
                    message_history: this.chatHistory,
                };

            const response = await this.makeAPICall('/api/isaa/mini_task_completion', 'POST', contextData);

            this.hideTypingIndicator();

            const responseText = response.result?.data || response.data || 'I understand. How can I help you further?';
            this.addMessageToHistory('user', message);
            this.addChatMessage('isaa', responseText);

            }

        } catch (error) {
            this.hideTypingIndicator();
            this.addChatMessage('isaa', 'Sorry, I encountered an error. Please try again.');
            console.error('ISAA API error:', error);
        }
    }
    // popup.js (fügen Sie dies als neue Methode in der Klasse hinzu)

    async runAgentLoop(initialTask) {
        let actionHistory = [];
        const maxSteps = 5; // Sicherheitsbremse gegen Endlosschleifen

        this.addChatMessage('isaa', `🤖 Entering agent mode to handle your request: "${initialTask}"`);

        for (let step = 0; step < maxSteps; step++) {
            // 1. LLM bitten, die nächste Aktion zu planen
            const plannedAction = await this.executeStructuredAction(initialTask, actionHistory);

            if (!plannedAction || !plannedAction.action_type) {
                this.addChatMessage('isaa', "I'm not sure how to proceed. Please provide more specific instructions.");
                return;
            }

            // 2. Den "Gedanken" des Agenten dem Benutzer anzeigen
            if (plannedAction.thought) {
                this.addChatMessage('isaa', `🤔 Thought: ${plannedAction.thought}`);
            }

            // 3. Überprüfen, ob die Aufgabe abgeschlossen ist
            if (plannedAction.action_type === 'finish' || !plannedAction.continue) {
                this.addChatMessage('isaa', '✅ Task completed successfully!');
                return;
            }

            // 4. Die geplante Aktion ausführen
            const executionResult = await this.executePageAction(plannedAction);

            // Füge das Ergebnis der Aktion zur Historie für den nächsten Schleifendurchlauf hinzu
            actionHistory.push({
                action: plannedAction,
                result: executionResult.success ? "Success" : `Failed: ${executionResult.error}`
            });

            // Warte einen Moment, damit der Benutzer die Aktion sehen kann
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        this.addChatMessage('isaa', "⚠️ Reached maximum steps. If the task is not complete, please try again with a more specific request.");
    }

    /**
     * Die "Brücke" zum content.js. Ruft den aktuellen Seitenkontext
     * vom aktiven Tab ab.
     * @returns {Promise<object|null>} Das Kontextobjekt oder null bei einem Fehler.
     */
    async getFreshPageContext() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tab && tab.id && tab.url && !tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
                // Sende eine Nachricht und warte auf die Antwort vom content script
                const response = await chrome.tabs.sendMessage(tab.id, { type: 'GET_PAGE_CONTEXT' });

                if (response && response.success) {
                    // Erstelle das Kontextobjekt mit den frischesten Daten
                    return {
                        url: tab.url,
                        title: tab.title,
                        pageIndex: response.pageIndex,
                        summary: response.summary
                    };
                }
            }
        } catch (error) {
            console.warn('Could not fetch fresh page context. The page might be protected or not yet loaded.', error.message);
        }
        return null; // Gebe null zurück, wenn kein Kontext erhalten werden konnte
    }


    async executeStructuredAction(userTask, actionHistory = []) {
        try {
            // Define the NEW action schema with 'thought' and 'continue'
            const actionSchema = {
                'properties': {
                    'thought': {
                        'title': 'Thought',
                        'type': 'string',
                        'description': "Your reasoning and plan. Explain what you are about to do and why. If the plan is multi-step, explain which step this is."
                    },
                    'action_type': {
                        'enum': ['click', 'fill_form', 'navigate', 'scroll', 'extract_data', 'finish'],
                        'title': 'Action Type',
                        'type': 'string',
                        'description': "The type of action to perform. Use 'finish' when the entire task is complete."
                    },
                    'target_selector': {
                        'title': 'Target Selector',
                        'type': 'string',
                        'description': "CSS selector for the target element. Be precise."
                    },
                    'data': {
                        'anyOf': [{ 'additionalProperties': true, 'type': 'object' }, { 'type': 'null' }],
                        'default': null,
                        'title': 'Data'
                    },
                    'continue': {
                        'title': 'Continue Plan',
                        'type': 'boolean',
                        'description': "Set to true if more steps are needed to complete the user's overall goal. Set to false if this is the final step."
                    }
                },
                'required': ['thought', 'action_type', 'continue'],
                'title': 'AgentActionSchema',
                'type': 'object'
            };

            const freshContext = await this.getFreshPageContext();

            const contextInfo = freshContext
                ? `Current page context: "${JSON.stringify(freshContext)}"`
                : 'Could not access the current page context. The page may be protected.';

            const historyInfo = actionHistory.length > 0 ?
                `You have already performed these actions: ${JSON.stringify(actionHistory)}` :
                "This is the first step.";

            // Der Rest der Funktion (API-Aufruf etc.) bleibt unverändert...
            const response = await this.makeAPICall('/api/isaa/format_class', 'POST', {
                format_schema: actionSchema,
                task: `You are a web automation agent. The user's goal is: "${userTask}".
                      ${contextInfo}
                      ${historyInfo}

                      Your task is to decide the single next best action to get closer to the user's goal.

                      Action Guidelines:
                      1.  **Think Step-by-Step**: First, explain your reasoning in the 'thought' field.
                      2.  **Choose an Action**: Select an 'action_type' and its parameters.
                      3.  **Decide to Continue**:
                          - If the user's overall goal requires more actions after this one, set 'continue' to true.
                          - If this single action completes the entire goal, set 'action_type' to 'finish' and 'continue' to false.
                      4.  **Handle Failures**: If a previous action failed, analyze the error in the history and try a different approach.

                      Example for "log me in":
                      - Step 1: { "thought": "I need to fill the username field.", "action_type": "fill_form", "target_selector": "#username", "data": {"value": "user@example.com"}, "continue": true }
                      - Step 2: { "thought": "Now I will fill the password.", "action_type": "fill_form", "target_selector": "#password", "data": {"value": "secret"}, "continue": true }
                      - Step 3: { "thought": "The form is filled, I will now click the login button to complete the task.", "action_type": "click", "target_selector": "button[type='submit']", "continue": false }
                      `,
                agent_name: 'speed',
                auto_context: true
            });

            if (response.result?.data || response.data) {
                const action = response.result?.data || response.data;
                console.log('🎯 Agent planned action:', action);
                return action;
            }
            return null; // Kein Plan vom LLM erhalten
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
            const escapedMessage = this.escapeHtml(message);
            messageEl.innerHTML = `
                <div class="isaa-avatar">🤖</div>
                <div class="message-content">
                    <p>${escapedMessage}</p>
                    <button class="speak-btn" title="Vorlesen" data-text="${escapedMessage}">
                        <svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"></path></svg>
                    </button>
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
        const searchResultsContainer = document.getElementById('searchResults');

        // Leere alte Ergebnisse
        searchResultsContainer.innerHTML = '';

        if (!results || results.length === 0) {
            this.showNoResults();
            return;
        }

        // Erstelle und füge jedes Ergebnis-Element einzeln hinzu
        results.forEach(result => {
            // 1. Erstelle das Hauptelement
            const item = document.createElement('div');
            item.className = 'search-result-item';
            item.dataset.elementId = result.id; // Speichere die ID im dataset

            // 2. Fülle den Inhalt mit innerHTML
            item.innerHTML = `
                <div class="result-title">${this.escapeHtml(result.title)}</div>
                <div class="result-snippet">${this.escapeHtml(result.snippet)}</div>
                <div class="result-actions">
                    <button class="result-action scroll-btn">Scroll to</button>
                    <button class="result-action ask-isaa-btn">Ask ISAA</button>
                </div>
            `;

            // 3. Füge die Event-Listener programmgesteuert hinzu
            const scrollBtn = item.querySelector('.scroll-btn');
            const askIsaBtn = item.querySelector('.ask-isaa-btn');

            // Der entscheidende Teil: Hier wird `this` korrekt gebunden
            scrollBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Verhindert, dass andere Klick-Events ausgelöst werden
                this.scrollToElement(result.id);
            });

            askIsaBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.askISAAAboutSection(result.id);
            });

            // 4. Füge das fertige Element dem Container hinzu
            searchResultsContainer.appendChild(item);
        });
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
        let passwordList = document.getElementById('passwordList');

        // try {
            const response = await this.makeAPICall('/api/PasswordManager/list_passwords', 'POST', {});
            const passwords = response?.result?.data || response?.data || [];

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
                    <div class="password-meta">
                        <span>Username: ${this.escapeHtml(pwd.username)}</span>
                        ${pwd.has_totp ? '<span class="totp-badge">2FA</span>' : ''}
                    </div>
                    <div class="password-actions-container">
                        ${!pwd.hasTotp ? `<button class="password-action-btn add-totp-btn" title="2FA hinzufügen" data-id="${pwd.id}" data-title="${this.escapeHtml(pwd.title)}"><svg fill="#ffffff" width="256px" height="256px" viewBox="0 0 14.00 14.00" role="img" focusable="false" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" stroke="#ffffff" stroke-width="0.00014">

<g id="SVGRepo_bgCarrier" stroke-width="0"/>

<g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"/>

<g id="SVGRepo_iconCarrier">

<path d="m 1,8.9182799 0,-0.5504 0.1446541,-0.1223 c 0.07956,-0.067 0.1559748,-0.1348 0.1698113,-0.1501 0.013837,-0.015 0.4157233,-0.3801 0.8930818,-0.8106 0.4773584,-0.4305 0.9001757,-0.8198 0.9395935,-0.8651 0.1263871,-0.1452 0.1673248,-0.2541 0.1673248,-0.445 0,-0.1593 -0.00749,-0.185 -0.082936,-0.284 -0.1118052,-0.1468 -0.260361,-0.2252 -0.4566427,-0.2409 -0.3466544,-0.028 -0.6163161,0.1738 -0.6885768,0.5148 -0.018152,0.086 -0.027531,0.095 -0.094116,0.09 -0.040675,0 -0.2691142,-0.018 -0.5076424,-0.035 l -0.4336875,-0.03 0.013954,-0.1033 c 0.054338,-0.4024 0.1755991,-0.6633 0.4232304,-0.9107 0.2888481,-0.2885 0.645978,-0.4286 1.1597502,-0.455 0.8324196,-0.043 1.4609999,0.342 1.6363777,1.0015 0.115527,0.4345 0.048223,0.946 -0.1695379,1.2885 -0.1248957,0.1964 -0.2691691,0.342 -0.7397055,0.7465 -0.7484706,0.6433 -1.1422284,0.9894 -1.1422365,1.0041 -4.4e-6,0.01 0.4811277,0.011 1.0691825,0.01 l 1.0691902,-0.01 0,0.4533 0,0.4534 -1.6855346,0 -1.6855346,0 0,-0.5504 z m 3.8490567,0.2116 0,-0.3389 0.3081761,-0.01 0.3081761,-0.01 0.00646,-1.7673 0.00646,-1.7673 -0.3146355,0 -0.314635,0 0,-0.3522 0,-0.3522 1.9123131,0 1.9123132,0 -0.0067,0.7736 -0.0067,0.7736 -0.3270441,0 -0.327044,0 -0.0069,-0.4218 -0.0069,-0.4217 -0.8610252,0.01 -0.8610254,0.01 -0.00708,0.6163 c -0.0039,0.339 -8.513e-4,0.6418 0.00677,0.673 l 0.013852,0.057 0.7731057,0 0.7731059,0 -0.0076,0.3393 -0.0076,0.3393 -0.7722276,0.01 -0.772228,0.01 0,0.7421 0,0.7422 0.3081761,0.01 0.3081761,0.01 0,0.3389 0,0.3388 -1.0188679,0 -1.0188679,0 0,-0.3388 z m 3.0188682,0 0,-0.3389 0.294784,-0.01 0.294784,-0.01 0.2064541,-0.5787 c 0.11355,-0.3182 0.326822,-0.9182 0.4739389,-1.3333 0.1471161,-0.4151 0.2942611,-0.8283 0.3269881,-0.9182 0.1483019,-0.4076 0.2646859,-0.7417 0.2646859,-0.76 0,-0.011 -0.124528,-0.02 -0.27673,-0.02 l -0.276729,0 0,-0.3145 0,-0.3144 1.2704401,0 1.27044,0 0,0.3144 0,0.3145 -0.292,0 c -0.268053,0 -0.29055,0 -0.274332,0.044 0.02421,0.06 0.508058,1.3781 0.70431,1.9183 0.08798,0.2421 0.195184,0.5364 0.238229,0.654 0.04304,0.1177 0.142944,0.3922 0.222,0.6101 l 0.143738,0.3962 0.270537,0 0.270537,0 0,0.3397 0,0.3396 -0.981132,0 -0.981132,0 0,-0.3396 0,-0.3397 0.289308,0 c 0.215114,0 0.289097,-0.01 0.288485,-0.031 -4.53e-4,-0.017 -0.07193,-0.2466 -0.15883,-0.5095 l -0.158007,-0.4779 -0.845257,-0.01 -0.845257,-0.01 -0.165724,0.4972 c -0.09115,0.2735 -0.165724,0.5057 -0.165724,0.5161 0,0.01 0.124528,0.019 0.276729,0.019 l 0.27673,0 0,0.3397 0,0.3396 -0.981132,0 -0.9811321,0 0,-0.3388 z m 3.1824061,-2.1329 c -0.01907,-0.059 -0.09054,-0.2824 -0.158819,-0.4969 -0.06828,-0.2144 -0.189971,-0.5965 -0.270415,-0.849 -0.08044,-0.2525 -0.152629,-0.4588 -0.16041,-0.4584 -0.01208,6e-4 -0.603088,1.77 -0.6260531,1.8743 -0.007,0.032 0.08088,0.037 0.6211141,0.037 l 0.629248,0 -0.03467,-0.1069 z"/>

</g>

</svg></button>` : ''}
                        <button class="password-action-btn delete-btn" title="Löschen" data-id="${pwd.id}">🗑️</button>
                    </div>
                </div>
            `).join('');
            // Event-Delegation für Aktionen in der Passwortliste
            passwordList = document.getElementById('passwordList');
            passwordList?.addEventListener('click', (e) => {
                const target = e.target;
                const passwordItem = target.closest('.password-item');

                if (target.closest('.delete-btn')) {
                    console.log('Delete button clicked', target.closest('.delete-btn'));
                    const passwordId = target.closest('.delete-btn').dataset.id;
                    this.deletePassword(passwordId);
                } else if (target.closest('.add-totp-btn')) {
                    const btn = target.closest('.add-totp-btn');
                    this.openAddTotpModal(btn.dataset.id, btn.dataset.title);
                } else if (passwordItem) {
                    // Bestehende Logik zum automatischen Ausfüllen beim Klick auf das Item
                    this.usePassword(passwordItem.dataset.id);
                }
            });

            // Listener für das Absenden des 2FA-Formulars
            document.getElementById('addTotpForm')?.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleAddTotpSubmit();
            });

            // Add click handlers
            document.querySelectorAll('.password-item').forEach(item => {
                item.addEventListener('click', () => {
                    const passwordId = item.dataset.id;
                    this.usePassword(passwordId);
                });
            });

        // } catch (error) {
        //     console.error('Failed to load passwords:', error);
        //     passwordList.innerHTML = `
        //         <div class="error-message">
        //             <p>Failed to load passwords. Please check your connection.</p>
        //         </div>
        //     `+JSON.stringify(error);
        // }
    }


    /**
     * Zeigt ein benutzerdefiniertes Bestätigungs-Modal an und gibt ein Promise zurück,
     * das mit `true` (bestätigt) oder `false` (abgebrochen) aufgelöst wird.
     * @param {string} message - Die Frage, die dem Benutzer gestellt wird.
     * @param {string} [title="Bestätigung erforderlich"] - Der Titel des Modals.
     * @returns {Promise<boolean>}
     */
    showConfirmation(message, title = "Bestätigung erforderlich") {
        return new Promise(resolve => {
            const modal = document.getElementById('confirmationModal');
            const titleEl = document.getElementById('confirmationTitle');
            const messageEl = document.getElementById('confirmationMessage');
            const confirmBtn = document.getElementById('confirmBtn-confirm');
            const cancelBtn = document.getElementById('confirmBtn-cancel');
            const closeBtns = modal.querySelectorAll('.modal-close');

            // Setze Inhalt und zeige Modal
            titleEl.textContent = title;
            messageEl.textContent = message;
            modal.classList.remove('hidden');

            // Temporäre Handler, die sich selbst entfernen
            const handleConfirm = () => {
                cleanup();
                resolve(true);
            };
            const handleCancel = () => {
                cleanup();
                resolve(false);
            };

            // Event Listener hinzufügen
            confirmBtn.addEventListener('click', handleConfirm, { once: true });
            cancelBtn.addEventListener('click', handleCancel, { once: true });
            closeBtns.forEach(btn => btn.addEventListener('click', handleCancel, { once: true }));

            // Aufräumfunktion, um Listener zu entfernen und das Modal zu schließen
            const cleanup = () => {
                this.closeModal(modal);
                confirmBtn.removeEventListener('click', handleConfirm);
                cancelBtn.removeEventListener('click', handleCancel);
                closeBtns.forEach(btn => btn.removeEventListener('click', handleCancel));
            };
        });
    }

    async deletePassword(passwordId) {
        if (!passwordId) return;

        // Rufe das neue, asynchrone Bestätigungs-Modal auf
        const confirmed = await this.showConfirmation(
            "Sind Sie sicher, dass Sie diesen Passworteintrag endgültig löschen möchten?",
            "Löschen bestätigen"
        );

        // Fahre nur fort, wenn der Benutzer bestätigt hat
        if (confirmed) {
            try {
                // Zeige im UI, dass gelöscht wird (optional, aber gute UX)
                const itemToDelete = document.querySelector(`.password-item[data-id="${passwordId}"]`);
                if(itemToDelete) itemToDelete.style.opacity = '0.5';

                await this.makeAPICall('/api/PasswordManager/delete_password', 'POST', { entry_id: passwordId });

                // Lade die Liste neu, um den gelöschten Eintrag zu entfernen
                this.loadPasswords();
            } catch (error) {
                console.error('Failed to delete password:', error);
                alert('Das Passwort konnte nicht gelöscht werden.'); // Fallback-Fehlermeldung
            }
        }
    }

    openAddTotpModal(passwordId, title) {
        // Modal-Felder mit den Daten des Eintrags füllen
        document.getElementById('totp-entry-id').value = passwordId;
        document.getElementById('totp-modal-title').textContent = title;

        // Feedback und Formular zurücksetzen
        document.getElementById('addTotpFeedback').textContent = '';
        document.getElementById('addTotpForm').reset();

        // Modal anzeigen
        document.getElementById('addTotpModal')?.classList.remove('hidden');
    }

    async handleAddTotpSubmit() {
        const entryId = document.getElementById('totp-entry-id').value;
        const secret = document.getElementById('totp-secret-input').value.trim();
        const submitBtn = document.getElementById('addTotpSubmitBtn');
        const feedbackEl = document.getElementById('addTotpFeedback');

        feedbackEl.textContent = '';
        feedbackEl.className = 'form-feedback';

        if (!secret) {
            feedbackEl.textContent = 'Bitte geben Sie einen geheimen Schlüssel ein.';
            feedbackEl.classList.add('error');
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = 'Prüfen...';

        try {
            const response = await this.makeAPICall('/api/PasswordManager/add_totp_secret', 'POST', {
                entry_id: entryId,
                secret: secret
            });

            if (response.result?.status === 'ok') {
                feedbackEl.textContent = '2FA erfolgreich aktiviert!';
                feedbackEl.classList.add('success');
                setTimeout(() => {
                    this.closeModal(document.getElementById('addTotpModal'));
                    this.loadPasswords(); // Lade die Liste neu, um den 2FA-Button auszublenden
                }, 1000);
            } else {
                feedbackEl.textContent = response.result?.info || 'Ungültiger Schlüssel.';
                feedbackEl.classList.add('error');
            }
        } catch (error) {
            console.error('Failed to add TOTP secret:', error);
            feedbackEl.textContent = 'Ein Fehler ist aufgetreten.';
            feedbackEl.classList.add('error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = '2FA Aktivieren';
        }
    }

    async usePassword(passwordId) {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            // Get password details
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', {
                url: tab.url
            });

            if (response.result.data) {
                // Send to content script for autofill
                await chrome.tabs.sendMessage(tab.id, {
                    type: 'AUTOFILL_PASSWORD',
                    data: response.result.data
                });

                // Show TOTP if available
                if (response.result.data.totp_code) {
                    this.showTOTPModal(response.result.data.totp_code, response.result.data.title, response.result.data.time_remaining);
                }

                // Close popup after successful autofill
                // setTimeout(() => window.close(), 1000); // ENTFERNEN
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

            if (response.result.data) {
                await chrome.tabs.sendMessage(tab.id, {
                    type: 'AUTOFILL_PASSWORD',
                    data: response.result.data
                });

                if (response.result.data.totp_code) {
                    this.showTOTPModal(response.result.data.totp_code, response.result.data.title, response.result.data.time_remaining);
                }

                // setTimeout(() => window.close(), 1000); // ENTFERNEN
            }

        } catch (error) {
            console.error('Autofill error:', error);
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

            if (response.result.data && response.result.data.password) {
                // Copy to clipboard
                await navigator.clipboard.writeText(response.result.data.password);

                // Send to content script for filling
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                await chrome.tabs.sendMessage(tab.id, {
                    type: 'FILL_GENERATED_PASSWORD',
                    password: response.result.data.password
                });
            }

        } catch (error) {
            console.error('Password generation error:', error);
        }
    }

    async importPasswords() {
        //const modal = document.getElementById('importHelperModal');
        //modal?.classList.remove('hidden');

        //// Der eigentliche Datei-Dialog wird nun vom Button im Modal ausgelöst
        //document.getElementById('openImportDialogBtn')?.addEventListener('click', () => {
        //     modal?.classList.add('hidden');
        //     this.triggerImportFileDialog();
        //});
        console.log("Import-Prozess wird über UI-Events gesteuert.");
    }

    triggerImportFileDialog() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.csv,.json';

        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            try {
                const content = await file.text();
                const format = file.name.endsWith('.json') ? 'json' : 'chrome'; // Simple format detection

                await this.makeAPICall('/api/PasswordManager/import_passwords', 'POST', {
                    file_content: content,
                    file_format: format,
                    folder: 'Imported from Browser'
                });

                this.loadPasswords(); // Refresh list

            } catch (error) {
                console.error('Import error:', error);
                // Optional: Zeige eine Fehlermeldung im UI an
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
