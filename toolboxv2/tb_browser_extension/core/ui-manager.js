// ToolBox Browser Extension - UI Manager
// Enhanced user interface management with modern design

class TBUIManager {
    constructor() {
        this.panel = null;
        this.voiceIndicator = null;
        this.searchOverlay = null;
        this.notifications = [];
        this.isVisible = false;
        this.settings = {
            theme: 'dark',
            position: 'top-right',
            opacity: 0.95,
            animations: true,
            compactMode: false,
            autoHide: true,
            autoHideDelay: 5000
        };

        // ISAA Integration
        this.isaaPlugin = null;
        this.isaaEnabled = true;
        this.isaaChatVisible = false;

        this.init();
    }

    async initializeISAAIntegration() {
        // Wait for ISAA plugin to be available
        if (typeof window.TBISAAPlugin !== 'undefined') {
            this.isaaPlugin = window.TBISAAPlugin;
            TBUtils.info('UIManager', 'ISAA Plugin integrated successfully');

            // Add ISAA chat button to the panel
            this.addISAAChatButton();
        } else {
            // Retry after a short delay
            setTimeout(() => {
                this.initializeISAAIntegration();
            }, 1000);
        }
    }

    addISAAChatButton() {
        const actionsContainer = document.querySelector('.tb-actions');
        if (actionsContainer && this.isaaPlugin) {
            const isaaButton = document.createElement('button');
            isaaButton.className = 'tb-action-btn tb-isaa-chat';
            isaaButton.innerHTML = `
                <span class="tb-btn-icon">🤖</span>
                <span class="tb-btn-text">ISAA Chat</span>
            `;
            isaaButton.title = 'Chat with ISAA - AI Web Assistant';

            isaaButton.addEventListener('click', () => {
                this.toggleISAAChat();
            });

            // Insert before the last button (usually settings)
            const lastButton = actionsContainer.lastElementChild;
            actionsContainer.insertBefore(isaaButton, lastButton);
        }
    }

    async init() {
        try {
            await this.loadSettings();
            this.createMainPanel();
            this.createVoiceIndicator();
            this.setupEventListeners();
            await this.initializeISAAIntegration();

            TBUtils.info('UIManager', 'UI Manager initialized');
        } catch (error) {
            TBUtils.handleError('UIManager', error);
        }
    }

    async loadSettings() {
        const stored = await TBUtils.getStorage([
            'ui_theme',
            'ui_position',
            'ui_opacity',
            'ui_animations',
            'ui_compact_mode',
            'ui_auto_hide',
            'ui_auto_hide_delay'
        ]);

        this.settings = {
            ...this.settings,
            theme: stored.ui_theme || this.settings.theme,
            position: stored.ui_position || this.settings.position,
            opacity: stored.ui_opacity || this.settings.opacity,
            animations: stored.ui_animations !== false,
            compactMode: stored.ui_compact_mode || this.settings.compactMode,
            autoHide: stored.ui_auto_hide !== false,
            autoHideDelay: stored.ui_auto_hide_delay || this.settings.autoHideDelay
        };
    }

    createMainPanel() {
        // Create main panel container
        this.panel = document.createElement('div');
        this.panel.id = 'tb-main-panel';
        this.panel.className = `tb-panel tb-theme-${this.settings.theme} tb-position-${this.settings.position}`;

        // Create header
        const header = this.createPanelHeader();
        this.panel.appendChild(header);

        // Create content
        const content = this.createPanelContent();
        this.panel.appendChild(content);

        this.panel.style.opacity = this.settings.opacity;

        if (document.body) {
            document.body.appendChild(this.panel);
        } else {
            // Wait for DOM to be ready
            document.addEventListener('DOMContentLoaded', () => {
                if (document.body) {
                    document.body.appendChild(this.panel);
                }
            });
        }
    }

    createPanelHeader() {
        const header = document.createElement('div');
        header.className = 'tb-panel-header';

        // Minimal logo section
        const logo = document.createElement('div');
        logo.className = 'tb-logo';

        const iconSpan = document.createElement('span');
        iconSpan.className = 'tb-icon';
        iconSpan.textContent = '🧰';

        const titleSpan = document.createElement('span');
        titleSpan.className = 'tb-title';
        titleSpan.textContent = 'ToolBox';

        logo.appendChild(iconSpan);
        logo.appendChild(titleSpan);

        // Minimal controls - only close button
        const controls = document.createElement('div');
        controls.className = 'tb-controls';

        const closeButton = document.createElement('button');
        closeButton.className = 'tb-btn tb-btn-icon';
        closeButton.id = 'tb-close';
        closeButton.title = 'Close';

        const closeIcon = document.createElement('span');
        closeIcon.className = 'tb-icon';
        closeIcon.textContent = '✕';

        closeButton.appendChild(closeIcon);
        controls.appendChild(closeButton);

        header.appendChild(logo);
        header.appendChild(controls);

        return header;
    }

    createPanelContent() {
        const content = document.createElement('div');
        content.className = 'tb-panel-content';

        // Quick actions
        const quickActions = document.createElement('div');
        quickActions.className = 'tb-quick-actions';

        const actionButtons = [
            { id: 'tb-isaa-chat', icon: '🤖', text: 'ISAA Assistant', class: 'tb-btn-primary' },
            { id: 'tb-generate-password', icon: '🔑', text: 'Generate Password', class: 'tb-btn-secondary' },
            { id: 'tb-auto-login', icon: '🔐', text: 'Auto Fill', class: 'tb-btn-secondary' },
            { id: 'tb-password-manager', icon: '🔒', text: 'Passwords', class: 'tb-btn-secondary' }
        ];

        actionButtons.forEach(btn => {
            const button = document.createElement('button');
            button.className = `tb-btn ${btn.class || 'tb-btn-primary'}`;
            button.id = btn.id;

            const icon = document.createElement('span');
            icon.className = 'tb-icon';
            icon.textContent = btn.icon;

            button.appendChild(icon);
            button.appendChild(document.createTextNode(' ' + btn.text));
            quickActions.appendChild(button);
        });

        // Status bar
        const statusBar = document.createElement('div');
        statusBar.className = 'tb-status-bar';

        const statusItems = [
            { label: 'Status:', id: 'tb-connection-status', value: 'Connected' },
            { label: 'Actions:', id: 'tb-actions-count', value: '0' }
        ];

        statusItems.forEach(item => {
            const statusItem = document.createElement('div');
            statusItem.className = 'tb-status-item';

            const label = document.createElement('span');
            label.className = 'tb-status-label';
            label.textContent = item.label;

            const value = document.createElement('span');
            value.className = 'tb-status-value';
            value.id = item.id;
            value.textContent = item.value;

            statusItem.appendChild(label);
            statusItem.appendChild(value);
            statusBar.appendChild(statusItem);
        });

        // Integrated ISAA section
        const isaaSection = document.createElement('div');
        isaaSection.className = 'tb-isaa-section';
        isaaSection.id = 'tb-isaa-section';

        // Input container with audio and send buttons
        const inputContainer = document.createElement('div');
        inputContainer.className = 'tb-isaa-input-container';

        const isaaInput = document.createElement('input');
        isaaInput.type = 'text';
        isaaInput.className = 'tb-isaa-input';
        isaaInput.id = 'tb-isaa-input';
        isaaInput.placeholder = 'Ask ISAA or search...';

        const audioBtn = document.createElement('button');
        audioBtn.className = 'tb-isaa-audio-btn';
        audioBtn.id = 'tb-isaa-audio';
        audioBtn.title = 'Voice Input for Selected Text';
        audioBtn.innerHTML = '<span>🎤</span>';

        const sendBtn = document.createElement('button');
        sendBtn.className = 'tb-isaa-send-btn';
        sendBtn.id = 'tb-isaa-send';
        sendBtn.title = 'Send to ISAA';
        sendBtn.textContent = 'Send';

        inputContainer.appendChild(isaaInput);
        inputContainer.appendChild(audioBtn);
        inputContainer.appendChild(sendBtn);

        // Chat container (initially hidden)
        const chatContainer = document.createElement('div');
        chatContainer.className = 'tb-isaa-chat-container';
        chatContainer.id = 'tb-isaa-chat-container';
        chatContainer.style.display = 'none';

        const messagesDiv = document.createElement('div');
        messagesDiv.className = 'tb-isaa-messages';
        messagesDiv.id = 'tb-isaa-messages';
        messagesDiv.innerHTML = `
            <div class="tb-isaa-welcome">
                <div class="tb-icon">🤖</div>
                <p>ISAA AI Assistant Ready</p>
                <small>Ask me to fill forms, extract data, or navigate pages</small>
            </div>
        `;

        chatContainer.appendChild(messagesDiv);

        // Results container
        const resultsDiv = document.createElement('div');
        resultsDiv.className = 'tb-isaa-results';
        resultsDiv.id = 'tb-isaa-results';
        resultsDiv.innerHTML = `
            <div class="tb-search-placeholder">
                <div class="tb-icon">🔍</div>
                <p>Type to search or chat with ISAA</p>
                <small>Intelligent search, form filling, and page interaction</small>
            </div>
        `;

        isaaSection.appendChild(inputContainer);
        isaaSection.appendChild(chatContainer);
        isaaSection.appendChild(resultsDiv);

        content.appendChild(quickActions);
        content.appendChild(isaaSection);
        content.appendChild(statusBar);

        return content;
    }

    createVoiceIndicator() {
        this.voiceIndicator = document.createElement('div');
        this.voiceIndicator.id = 'tb-voice-indicator';
        this.voiceIndicator.className = 'tb-voice-indicator tb-hidden';

        const content = document.createElement('div');
        content.className = 'tb-voice-content';

        // Voice icon
        const iconDiv = document.createElement('div');
        iconDiv.className = 'tb-voice-icon';
        const pulse = document.createElement('span');
        pulse.className = 'tb-pulse';
        pulse.textContent = '🎤';
        iconDiv.appendChild(pulse);

        // Voice text
        const textDiv = document.createElement('div');
        textDiv.className = 'tb-voice-text';
        const status = document.createElement('div');
        status.className = 'tb-voice-status';
        status.textContent = 'Listening...';
        const transcript = document.createElement('div');
        transcript.className = 'tb-voice-transcript';
        textDiv.appendChild(status);
        textDiv.appendChild(transcript);

        // Voice controls
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'tb-voice-controls';
        const stopBtn = document.createElement('button');
        stopBtn.className = 'tb-btn tb-btn-small';
        stopBtn.id = 'tb-voice-stop';
        stopBtn.textContent = 'Stop';
        controlsDiv.appendChild(stopBtn);

        content.appendChild(iconDiv);
        content.appendChild(textDiv);
        content.appendChild(controlsDiv);
        this.voiceIndicator.appendChild(content);

        if (document.body) {
            document.body.appendChild(this.voiceIndicator);
        }
    }



    setupEventListeners() {
        // Panel controls
        document.getElementById('tb-voice-toggle')?.addEventListener('click', () => {
            this.toggleVoice();
        });

        document.getElementById('tb-search-toggle')?.addEventListener('click', () => {
            this.toggleSearch();
        });

        document.getElementById('tb-close')?.addEventListener('click', () => {
            this.hidePanel();
        });

        // ISAA controls
        document.getElementById('tb-isaa-chat')?.addEventListener('click', () => {
            this.toggleISAAChat();
        });

        // Password management controls
        document.getElementById('tb-generate-password')?.addEventListener('click', () => {
            this.generatePassword();
        });

        document.getElementById('tb-auto-login')?.addEventListener('click', () => {
            this.autoFillPassword();
        });

        document.getElementById('tb-password-manager')?.addEventListener('click', () => {
            this.openPasswordManager();
        });

        // ISAA input and controls
        const isaaInput = document.getElementById('tb-isaa-input');
        const isaaAudio = document.getElementById('tb-isaa-audio');
        const isaaSend = document.getElementById('tb-isaa-send');

        if (isaaInput) {
            isaaInput.addEventListener('input', (e) => {
                this.handleISAAInput(e.target.value);
            });



            isaaInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    this.sendToISAA(e.target.value);
                } else if (e.key === 'Escape') {
                    this.hideISAAChat();
                }
            });
        }

        if (isaaAudio) {
            isaaAudio.addEventListener('click', () => {
                this.handleAudioInput();
            });
        }

        if (isaaSend) {
            isaaSend.addEventListener('click', () => {
                const input = document.getElementById('tb-isaa-input');
                if (input && input.value.trim()) {
                    this.sendToISAA(input.value.trim());
                }
            });
        }

        // Voice controls
        document.getElementById('tb-voice-stop')?.addEventListener('click', () => {
            this.hideVoiceIndicator();
            if (window.tbVoiceEngine) {
                window.tbVoiceEngine.stopListening();
            }
        });

        // Search controls
        const searchInput = document.getElementById('tb-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', TBUtils.debounce((e) => {
                this.performSearch(e.target.value);
            }, 300));

            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.hideSearch();
                } else if (e.key === 'Enter') {
                    this.performSearch(e.target.value, { immediate: true });
                }
            });
        }

        document.getElementById('tb-search-voice')?.addEventListener('click', () => {
            this.startVoiceSearch();
        });

        document.getElementById('tb-search-close')?.addEventListener('click', () => {
            this.hideSearch();
        });

        document.getElementById('tb-search-prev')?.addEventListener('click', () => {
            this.navigateSearchResults(-1);
        });

        document.getElementById('tb-search-next')?.addEventListener('click', () => {
            this.navigateSearchResults(1);
        });

        // Auto-hide functionality
        if (this.settings.autoHide) {
            this.setupAutoHide();
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }

    setupAutoHide() {
        let hideTimeout;

        const resetHideTimer = () => {
            clearTimeout(hideTimeout);
            if (this.isVisible) {
                hideTimeout = setTimeout(() => {
                    this.hidePanel();
                }, this.settings.autoHideDelay);
            }
        };

        this.panel.addEventListener('mouseenter', () => {
            clearTimeout(hideTimeout);
        });

        this.panel.addEventListener('mouseleave', resetHideTimer);

        // Start timer when panel becomes visible
        this.on('panel-shown', resetHideTimer);
    }

    handleKeyboardShortcuts(event) {
        // Alt+T to toggle panel
        if (event.altKey && event.key.toLowerCase() === 't' && !this.isInInputField()) {
            event.preventDefault();
            this.togglePanel();
        }

        // Ctrl+Shift+S for smart search
        if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 's') {
            event.preventDefault();
            this.showSearch();
        }

        // Ctrl+Shift+V for voice
        if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'v') {
            event.preventDefault();
            this.toggleVoice();
        }
    }

    isInInputField() {
        const activeElement = document.activeElement;
        return activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.contentEditable === 'true'
        );
    }

    // Panel management
    showPanel() {
        if (this.isVisible) return;

        this.panel.classList.remove('tb-hidden');
        this.isVisible = true;

        if (this.settings.animations) {
            TBUtils.fadeIn(this.panel);
        }

        this.emit('panel-shown');
        TBUtils.info('UIManager', 'Panel shown');
    }

    hidePanel() {
        if (!this.isVisible) return;

        if (this.settings.animations) {
            TBUtils.fadeOut(this.panel);
        } else {
            this.panel.classList.add('tb-hidden');
        }

        this.isVisible = false;
        this.emit('panel-hidden');
        TBUtils.info('UIManager', 'Panel hidden');
    }

    togglePanel() {
        if (this.isVisible) {
            this.hidePanel();
        } else {
            this.showPanel();
        }
    }

    // Voice indicator management
    showVoiceIndicator(state = 'listening') {
        this.voiceIndicator.classList.remove('tb-hidden');
        this.voiceIndicator.setAttribute('data-state', state);

        const statusElement = this.voiceIndicator.querySelector('.tb-voice-status');
        if (statusElement) {
            const statusText = {
                listening: 'Listening...',
                processing: 'Processing...',
                speaking: 'Speaking...',
                error: 'Error occurred'
            };
            statusElement.textContent = statusText[state] || 'Active';
        }

        if (this.settings.animations) {
            TBUtils.fadeIn(this.voiceIndicator);
        }
    }

    updateVoiceIndicator(text) {
        const transcriptElement = this.voiceIndicator.querySelector('.tb-voice-transcript');
        if (transcriptElement) {
            transcriptElement.textContent = text;
        }
    }

    hideVoiceIndicator() {
        if (this.settings.animations) {
            TBUtils.fadeOut(this.voiceIndicator);
        } else {
            this.voiceIndicator.classList.add('tb-hidden');
        }
    }

    // Search management (integrated in main panel)
    showSearch() {
        // Show main panel if hidden
        if (!this.isVisible) {
            this.showPanel();
        }

        // Show search section
        const searchSection = document.getElementById('tb-search-section');
        if (searchSection) {
            searchSection.classList.remove('tb-hidden');

            if (this.settings.animations) {
                TBUtils.fadeIn(searchSection);
            }
        }

        // Focus search input
        const searchInput = document.getElementById('tb-search-input');
        if (searchInput) {
            setTimeout(() => searchInput.focus(), 100);
        }
    }

    hideSearch() {
        const searchSection = document.getElementById('tb-search-section');
        if (searchSection) {
            if (this.settings.animations) {
                TBUtils.fadeOut(searchSection);
            } else {
                searchSection.classList.add('tb-hidden');
            }
        }

        // Clear search results
        this.clearSearchResults();
    }

    toggleSearch() {
        const searchSection = document.getElementById('tb-search-section');
        if (searchSection && searchSection.classList.contains('tb-hidden')) {
            this.showSearch();
        } else {
            this.hideSearch();
        }
    }

    async performSearch(query, options = {}) {
        if (!query.trim()) {
            this.clearSearchResults();
            return;
        }

        // Show loading state
        this.showSearchLoading();

        try {
            // Combine multiple search approaches
            const searchPromises = [];

            // 1. Basic page search
            if (window.tbSearchEngine) {
                searchPromises.push(window.tbSearchEngine.search(query, options));
            }

            // 2. ISAA-powered local analysis
            searchPromises.push(this.performISAASearch(query, options));

            const allResults = await Promise.allSettled(searchPromises);
            const combinedResults = this.combineSearchResults(allResults, query);

            this.updateSearchResults(combinedResults, query);
        } catch (error) {
            TBUtils.handleError('UIManager', error);
            this.showSearchError('Search failed. Please try again.');
        }
    }

    async performISAASearch(query, options = {}) {
        try {
            // ISAA-2 Local Intelligence Search
            const results = [];

            // 1. Analyze page content with ISAA
            const pageAnalysis = this.analyzePageContent(query);
            if (pageAnalysis.length > 0) {
                results.push(...pageAnalysis);
            }

            // 2. Search local documentation patterns
            const docResults = this.searchLocalDocs(query);
            if (docResults.length > 0) {
                results.push(...docResults);
            }

            // 3. Smart suggestions based on context
            const smartSuggestions = this.generateSmartSuggestions(query);
            if (smartSuggestions.length > 0) {
                results.push(...smartSuggestions);
            }

            return results;
        } catch (error) {
            TBUtils.warn('UIManager', 'ISAA search failed:', error);
            return [];
        }
    }

    analyzePageContent(query) {
        const results = [];
        const queryLower = query.toLowerCase();

        // Find all text nodes that contain the query
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const matches = [];
        let node;

        while (node = walker.nextNode()) {
            const text = node.textContent.toLowerCase();
            if (text.includes(queryLower) && text.trim().length > 20) {
                const element = node.parentElement;
                if (element && this.isVisibleElement(element)) {
                    matches.push({
                        element: element,
                        text: node.textContent.trim(),
                        position: element.getBoundingClientRect()
                    });
                }
            }
        }

        // Sort by position (top to bottom) and take top 3
        matches.sort((a, b) => a.position.top - b.position.top);

        matches.slice(0, 3).forEach((match, index) => {
            const snippet = match.text.substring(0, 150);
            results.push({
                title: `Page Content Match ${index + 1}`,
                snippet: snippet + (match.text.length > 150 ? '...' : ''),
                score: 85 - (index * 10),
                source: 'page',
                type: 'content',
                action: 'Scroll to content',
                element: match.element, // Store reference to actual element
                elementId: `tb-search-target-${Date.now()}-${index}` // Unique ID for targeting
            });

            // Add a temporary ID to the element for scrolling
            match.element.setAttribute('data-tb-search-target', `tb-search-target-${Date.now()}-${index}`);
        });

        return results;
    }

    isVisibleElement(element) {
        // Check if element is visible and has meaningful content
        const style = window.getComputedStyle(element);
        return style.display !== 'none' &&
               style.visibility !== 'hidden' &&
               style.opacity !== '0' &&
               element.offsetWidth > 0 &&
               element.offsetHeight > 0;
    }

    searchLocalDocs(query) {
        const results = [];
        const queryLower = query.toLowerCase();

        // Local documentation patterns
        const docPatterns = {
            'login': {
                title: 'ToolBox Login System',
                snippet: 'Use Auto Fill button to automatically fill login forms. Supports password generation and secure storage.',
                score: 90,
                action: 'Click Auto Fill button',
                type: 'documentation'
            },
            'password': {
                title: 'Password Management',
                snippet: 'Generate secure passwords with the Generate button. Passwords are encrypted and stored locally.',
                score: 95,
                action: 'Click Generate button',
                type: 'documentation'
            },
            'search': {
                title: 'Smart Search',
                snippet: 'ISAA-powered search analyzes page content and provides intelligent suggestions.',
                score: 88,
                action: 'Continue typing for more results',
                type: 'documentation'
            },
            'voice': {
                title: 'Voice Commands',
                snippet: 'Use voice commands to control ToolBox. Say "ToolBox" followed by your command.',
                score: 85,
                action: 'Activate voice commands',
                type: 'documentation'
            },
            'gesture': {
                title: 'Gesture Controls',
                snippet: 'Draw gestures to control ToolBox: Circle to toggle, lines to scroll, zigzag to refresh.',
                score: 82,
                action: 'Show gesture help',
                type: 'help'
            }
        };

        // Find matching patterns
        Object.entries(docPatterns).forEach(([key, doc]) => {
            if (queryLower.includes(key) || key.includes(queryLower)) {
                results.push({
                    ...doc,
                    source: 'docs',
                    type: 'documentation'
                });
            }
        });

        return results;
    }

    generateSmartSuggestions(query) {
        const results = [];
        const queryLower = query.toLowerCase();

        // Context-aware suggestions
        const suggestions = [];

        if (queryLower.includes('log') || queryLower.includes('sign')) {
            suggestions.push({
                title: 'Quick Login Suggestion',
                snippet: 'Detected login context. Use the Auto Fill button to automatically fill login forms.',
                score: 92,
                source: 'isaa',
                type: 'suggestion',
                action: 'Click Auto Fill (🔓)'
            });
        }

        if (queryLower.includes('pass') || queryLower.includes('secure')) {
            suggestions.push({
                title: 'Password Security',
                snippet: 'Generate a secure password with the Generate button. Uses advanced encryption.',
                score: 90,
                source: 'isaa',
                type: 'suggestion',
                action: 'Click Generate (🔑)'
            });
        }

        if (queryLower.includes('help') || queryLower.includes('how')) {
            suggestions.push({
                title: 'ToolBox Help',
                snippet: 'Use gestures, voice commands, or keyboard shortcuts to control ToolBox efficiently.',
                score: 85,
                source: 'isaa',
                type: 'help',
                action: 'Try Alt+T to toggle panel'
            });
        }

        return suggestions;
    }

    combineSearchResults(allResults, query) {
        const combined = [];

        allResults.forEach((result, index) => {
            if (result.status === 'fulfilled' && result.value) {
                const results = Array.isArray(result.value) ? result.value : [result.value];
                results.forEach(item => {
                    // Preserve the source from ISAA results, or set based on index
                    const source = item.source || (index === 0 ? 'page' : 'isaa');
                    combined.push({
                        ...item,
                        source: source,
                        query: query
                    });
                });
            }
        });

        // Sort by relevance score (highest first)
        return combined.sort((a, b) => (b.score || 0) - (a.score || 0));
    }

    showSearchLoading() {
        const resultsContainer = document.getElementById('tb-search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="tb-search-loading">
                    <span class="tb-icon tb-spin">🔍</span>
                    <p>Searching with AI assistance...</p>
                </div>
            `;
        }
    }

    showSearchError(message) {
        const resultsContainer = document.getElementById('tb-search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="tb-search-error">
                    <span class="tb-icon">⚠️</span>
                    <p>${message}</p>
                </div>
            `;
        }
    }

    updateSearchResults(results, query) {
        const resultsContainer = document.getElementById('tb-search-results');
        const statsContainer = document.getElementById('tb-search-stats');

        if (!resultsContainer) return;

        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="tb-search-no-results">
                    <span class="tb-icon">🔍</span>
                    <p>No results found for "${query}"</p>
                </div>
            `;
        } else {
            resultsContainer.innerHTML = results.map((result, index) => {
                const sourceIcon = {
                    'page': '📄',
                    'isaa': '🧠',
                    'docs': '📚'
                }[result.source] || '🔍';

                const sourceLabel = {
                    'page': 'Page Content',
                    'isaa': 'ISAA-2 Intelligence',
                    'docs': 'Documentation'
                }[result.source] || 'Search Result';

                return `
                    <div class="tb-search-result tb-result-${result.source}" data-index="${index}">
                        <div class="tb-result-header">
                            <span class="tb-result-source">${sourceIcon} ${sourceLabel}</span>
                            <span class="tb-result-score">${Math.round(result.score || 0)}%</span>
                        </div>
                        <div class="tb-result-content">
                            <div class="tb-result-title">${result.title || 'Result'}</div>
                            <div class="tb-result-snippet">${result.snippet || result.content || 'No preview available'}</div>
                            ${result.action ? `<div class="tb-result-action">💡 ${result.action}</div>` : ''}
                        </div>
                    </div>
                `;
            }).join('');

            // Add click handlers
            resultsContainer.querySelectorAll('.tb-search-result').forEach((element, index) => {
                element.addEventListener('click', () => {
                    this.selectSearchResult(index, results[index]);
                });
            });
        }

        if (statsContainer) {
            statsContainer.textContent = `${results.length} results found`;
        }
    }

    clearSearchResults() {
        const resultsContainer = document.getElementById('tb-search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="tb-search-placeholder">
                    <span class="tb-icon">🔍</span>
                    <p>Start typing to search this page with AI assistance</p>
                </div>
            `;
        }
    }

    selectSearchResult(index, result = null) {
        if (!result) {
            // Fallback to old behavior
            if (window.tbSearchEngine) {
                window.tbSearchEngine.scrollToResult(index);
            }
            return;
        }

        // Handle different types of search results
        switch (result.type) {
            case 'content':
                this.scrollToContent(result);
                break;
            case 'documentation':
                this.handleDocumentationAction(result);
                break;
            case 'suggestion':
                this.handleSuggestionAction(result);
                break;
            case 'help':
                this.handleHelpAction(result);
                break;
            default:
                // Generic action handling
                if (result.action) {
                    this.executeSearchAction(result);
                }
                break;
        }

        // Visual feedback
        this.highlightSearchResult(index);
        this.showNotification(`Action: ${result.action}`, 'info', 2000);
    }

    scrollToContent(result) {
        if (result.elementId) {
            // Find the element by our custom data attribute
            const targetElement = document.querySelector(`[data-tb-search-target="${result.elementId}"]`);
            if (targetElement) {
                // Smooth scroll to the element
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center',
                    inline: 'nearest'
                });

                // Highlight the element temporarily
                this.highlightElement(targetElement);
                return;
            }
        }

        // Fallback: scroll to element if available
        if (result.element) {
            result.element.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
                inline: 'nearest'
            });
            this.highlightElement(result.element);
        }
    }

    highlightElement(element) {
        // Add temporary highlight
        const originalStyle = element.style.cssText;
        element.style.cssText += `
            background: rgba(108, 142, 232, 0.3) !important;
            border: 2px solid var(--tb-accent-primary) !important;
            border-radius: 4px !important;
            transition: all 0.3s ease !important;
        `;

        // Remove highlight after 3 seconds
        setTimeout(() => {
            element.style.cssText = originalStyle;
        }, 3000);
    }

    handleDocumentationAction(result) {
        // Handle documentation-specific actions
        if (result.action.includes('Auto Fill')) {
            document.getElementById('tb-auto-login')?.click();
        } else if (result.action.includes('Generate')) {
            document.getElementById('tb-generate-password')?.click();
        } else if (result.action.includes('Ctrl+Shift+V')) {
            // Trigger voice command
            this.toggleVoice();
        }
    }

    handleSuggestionAction(result) {
        // Handle suggestion-specific actions
        if (result.action.includes('Auto Fill')) {
            document.getElementById('tb-auto-login')?.click();
        } else if (result.action.includes('Generate')) {
            document.getElementById('tb-generate-password')?.click();
        }
    }

    handleHelpAction(result) {
        // Handle help-specific actions
        if (result.action.includes('Alt+T')) {
            // Show panel toggle demonstration
            this.showNotification('Press Alt+T to toggle ToolBox panel', 'info', 4000);
        }
    }

    executeSearchAction(result) {
        // Generic action execution
        TBUtils.info('UIManager', `Executing search action: ${result.action}`);

        // Parse common actions
        if (result.action.includes('screenshot')) {
            document.getElementById('tb-screenshot')?.click();
        } else if (result.action.includes('circle')) {
            this.showNotification('Try drawing a circle with your mouse while holding right-click', 'info', 4000);
        }
    }

    highlightSearchResult(index) {
        // Remove previous highlights
        document.querySelectorAll('.tb-search-result').forEach(r => r.classList.remove('tb-active'));

        // Highlight selected result
        const results = document.querySelectorAll('.tb-search-result');
        if (results[index]) {
            results[index].classList.add('tb-active');
        }
    }

    navigateSearchResults(direction) {
        // Implementation for navigating through search results
        const results = document.querySelectorAll('.tb-search-result');
        const current = document.querySelector('.tb-search-result.tb-active');
        let newIndex = 0;

        if (current) {
            const currentIndex = parseInt(current.dataset.index);
            newIndex = currentIndex + direction;
        }

        if (newIndex < 0) newIndex = results.length - 1;
        if (newIndex >= results.length) newIndex = 0;

        results.forEach(r => r.classList.remove('tb-active'));
        if (results[newIndex]) {
            results[newIndex].classList.add('tb-active');
            this.selectSearchResult(newIndex);
        }
    }

    // ==================== ISAA INTEGRATION FUNCTIONALITY ====================

    toggleISAAChat() {
        const chatContainer = document.getElementById('tb-isaa-chat-container');
        if (!chatContainer) return;

        if (this.isaaChatVisible) {
            this.hideISAAChat();
        } else {
            this.showISAAChat();
        }
    }

    showISAAChat() {
        const chatContainer = document.getElementById('tb-isaa-chat-container');
        const resultsDiv = document.getElementById('tb-isaa-results');

        if (chatContainer && resultsDiv) {
            chatContainer.style.display = 'block';
            resultsDiv.style.display = 'none';
            this.isaaChatVisible = true;

            // Focus input
            const input = document.getElementById('tb-isaa-input');
            if (input) {
                input.placeholder = 'Chat with ISAA...';
                setTimeout(() => input.focus(), 100);
            }
        }
    }

    hideISAAChat() {
        const chatContainer = document.getElementById('tb-isaa-chat-container');
        const resultsDiv = document.getElementById('tb-isaa-results');

        if (chatContainer && resultsDiv) {
            chatContainer.style.display = 'none';
            resultsDiv.style.display = 'block';
            this.isaaChatVisible = false;

            // Reset input placeholder
            const input = document.getElementById('tb-isaa-input');
            if (input) {
                input.placeholder = 'Ask ISAA or search...';
            }
        }
    }

    handleISAAInput(value) {
        if (!value.trim()) {
            // Show placeholder when empty
            const resultsDiv = document.getElementById('tb-isaa-results');
            if (resultsDiv && !this.isaaChatVisible) {
                resultsDiv.innerHTML = `
                    <div class="tb-search-placeholder">
                        <div class="tb-icon">🔍</div>
                        <p>Type to search or chat with ISAA</p>
                        <small>Intelligent search, form filling, and page interaction</small>
                    </div>
                `;
            }
            return;
        }

        // Show live search results as user types
        if (!this.isaaChatVisible) {
            clearTimeout(this.searchTimeout);
            this.searchTimeout = setTimeout(() => {
                this.performLiveSearch(value);
            }, 300);
        }
    }

    async performLiveSearch(query) {
        const resultsDiv = document.getElementById('tb-isaa-results');
        if (!resultsDiv) return;

        // Show loading state
        resultsDiv.innerHTML = `
            <div class="tb-search-loading">
                <div class="tb-spinner"></div>
                <p>Searching with ISAA...</p>
            </div>
        `;

        try {
            // Try ISAA API first
            const response = await this.callToolBoxISAA(query);

            // Show results
            this.displaySearchResults(query, response);
        } catch (error) {
            // Fallback to local search
            TBUtils.warn('UIManager', 'ISAA search failed, using local search');
            this.performSearch(query);
        }
    }

    displaySearchResults(query, response) {
        const resultsDiv = document.getElementById('tb-isaa-results');
        if (!resultsDiv) return;

        const resultsHTML = `
            <div class="tb-search-results">
                <div class="tb-search-header">
                    <h4>🤖 ISAA Results for "${query}"</h4>
                </div>
                <div class="tb-search-response">
                    <div class="tb-response-content">
                        ${response.content}
                    </div>
                    ${response.actions && response.actions.length > 0 ? `
                        <div class="tb-response-actions">
                            ${response.actions.map(action => `
                                <button class="tb-action-btn" onclick="window.tbUIManager.executeAction('${action}')">
                                    ${action}
                                </button>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
                <div class="tb-search-suggestions">
                    <small>💡 Try: "fill this form", "click login button", "extract all links"</small>
                </div>
            </div>
        `;

        resultsDiv.innerHTML = resultsHTML;
    }

    isSearchIntent(query) {
        const searchKeywords = ['find', 'search', 'get', 'extract', 'show', 'list'];
        const chatKeywords = ['fill', 'click', 'navigate', 'help', 'how', 'what', 'why', 'can you'];

        const queryLower = query.toLowerCase();
        const hasSearchKeywords = searchKeywords.some(keyword => queryLower.includes(keyword));
        const hasChatKeywords = chatKeywords.some(keyword => queryLower.includes(keyword));

        // If it has chat keywords or is a question, treat as chat
        if (hasChatKeywords || queryLower.includes('?')) {
            return false;
        }

        // If it has search keywords or is short, treat as search
        return hasSearchKeywords || query.length < 20;
    }

    async sendToISAA(message) {
        if (!message.trim()) return;

        try {
            // Clear input
            const input = document.getElementById('tb-isaa-input');
            if (input) {
                input.value = '';
            }

            // Add user message to chat
            this.addISAAMessage(message, 'user');

            // Show loading
            this.addISAAMessage('🤔 Thinking...', 'assistant', true);

            // Send to ToolBox Python ISAA system
            const response = await this.callToolBoxISAA(message);

            // Remove loading message
            this.removeLoadingMessage();

            // Add response
            this.addISAAMessage(response.content, 'assistant');

            // Execute actions if any
            if (response.actions && response.actions.length > 0) {
                await this.executeISAAActions(response.actions);
            }

        } catch (error) {
            this.removeLoadingMessage();
            TBUtils.handleError('UIManager', error);
            this.addISAAMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        }
    }

    async callToolBoxISAA(message) {
        // Analyze the message to determine the best approach
        const intent = this.analyzeMessageIntent(message);

        try {
            if (intent.requiresStructuredOutput) {
                // Use format_class for structured data extraction
                return await this.callFormatClass(intent.schema, message);
            } else {
                // Use mini_task_completion for general tasks
                return await this.callMiniTaskCompletion(message, intent);
            }
        } catch (error) {
            // Fallback to local processing
            TBUtils.warn('UIManager', 'ToolBox ISAA unavailable, using local processing');
            return await this.processLocally(message, intent);
        }
    }

    analyzeMessageIntent(message) {
        const messageLower = message.toLowerCase();

        // Form filling intent
        if (messageLower.includes('fill') && (messageLower.includes('form') || messageLower.includes('field'))) {
            return {
                type: 'form-filling',
                requiresStructuredOutput: true,
                schema: 'FormFillRequest',
                priority: 'high'
            };
        }

        // Data extraction intent
        if (messageLower.includes('extract') || messageLower.includes('get') || messageLower.includes('find')) {
            return {
                type: 'data-extraction',
                requiresStructuredOutput: true,
                schema: 'DataExtractionRequest',
                priority: 'medium'
            };
        }

        // Navigation intent
        if (messageLower.includes('click') || messageLower.includes('navigate') || messageLower.includes('go to')) {
            return {
                type: 'navigation',
                requiresStructuredOutput: true,
                schema: 'NavigationRequest',
                priority: 'high'
            };
        }

        // General chat
        return {
            type: 'chat',
            requiresStructuredOutput: false,
            priority: 'low'
        };
    }

    async callMiniTaskCompletion(message, intent) {
        // Prepare context about the current page
        const pageContext = this.getPageContext();

        const payload = {
            mini_task: message,
            user_task: `Web interaction on ${pageContext.url}`,
            mode: intent.type,
            agent_name: 'web-assistant',
            task_from: 'browser_extension',
            context: {
                page_url: pageContext.url,
                page_title: pageContext.title,
                page_content: JSON.stringify(pageContext),
                intent: intent
            }
        };

        TBUtils.info('UIManager', 'Calling mini_task_completion:', payload);

        try {
            const response = await fetch('http://localhost:8080/api/isaa/mini_task_completion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API call failed: ${response.status} - ${errorText}`);
            }

            const result = await response.json();

            TBUtils.info('UIManager', 'mini_task_completion response:', result);

            return {
                content: result.response || result.content || result.result || 'Task completed',
                actions: result.actions || []
            };
        } catch (error) {
            TBUtils.handleError('UIManager', 'mini_task_completion failed:', error);
            throw error;
        }

        try {
            // Call ToolBox Python ISAA via command line
            const command = `tb -c isaa mini_task_completion "${taskDescription.replace(/"/g, '\\"')}" --agent_name="WebAssistant" --use_complex=true`;

            const response = await this.executeToolBoxCommand(command);

            return {
                content: response.output || 'I understand your request. Let me help you with that.',
                actions: this.parseActionsFromResponse(response.output, intent)
            };

        } catch (error) {
            throw new Error(`ToolBox ISAA call failed: ${error.message}`);
        }
    }

    async callFormatClass(schemaName, message) {
        // Define schemas for different request types
        const schemas = {
            'FormFillRequest': {
                type: 'object',
                properties: {
                    action: { type: 'string', enum: ['fill_form'] },
                    fields: {
                        type: 'array',
                        items: {
                            type: 'object',
                            properties: {
                                name: { type: 'string' },
                                value: { type: 'string' },
                                selector: { type: 'string' }
                            }
                        }
                    },
                    formSelector: { type: 'string' }
                }
            },
            'DataExtractionRequest': {
                type: 'object',
                properties: {
                    action: { type: 'string', enum: ['extract_data'] },
                    dataTypes: {
                        type: 'array',
                        items: { type: 'string', enum: ['text', 'links', 'images', 'tables', 'forms'] }
                    },
                    selectors: {
                        type: 'array',
                        items: { type: 'string' }
                    }
                }
            },
            'NavigationRequest': {
                type: 'object',
                properties: {
                    action: { type: 'string', enum: ['click_element', 'navigate'] },
                    target: { type: 'string' },
                    selector: { type: 'string' },
                    url: { type: 'string' }
                }
            }
        };

        const schema = schemas[schemaName];
        if (!schema) {
            throw new Error(`Unknown schema: ${schemaName}`);
        }

        try {
            const schemaJson = JSON.stringify(schema).replace(/"/g, '\\"');
            const taskDescription = `Analyze this web interaction request and provide structured output: ${message}`;

            const command = `tb -c isaa format_class "${schemaJson}" "${taskDescription.replace(/"/g, '\\"')}"`;

            const response = await this.executeToolBoxCommand(command);
            const structuredData = JSON.parse(response.output);

            return {
                content: this.generateResponseFromStructuredData(structuredData),
                actions: this.convertStructuredDataToActions(structuredData)
            };

        } catch (error) {
            throw new Error(`ToolBox format_class call failed: ${error.message}`);
        }
    }

    async executeToolBoxCommand(command) {
        try {
            // Send command to background script which can execute system commands
            const response = await chrome.runtime.sendMessage({
                action: 'execute-toolbox-command',
                command: command
            });

            if (response.error) {
                throw new Error(response.error);
            }

            return response;

        } catch (error) {
            throw new Error(`Command execution failed: ${error.message}`);
        }
    }

    async callFormatClass(schema, message) {
        const pageContext = this.getPageContext();

        const payload = {
            format_schema: this.getSchemaDefinition(schema),
            task: `${message}\n\nContext: User is on ${pageContext.url} - ${pageContext.title}`,
            agent_name: 'web-assistant',
            auto_context: true,
            context: {
                page_url: pageContext.url,
                page_title: pageContext.title,
                page_content: JSON.stringify(pageContext)
            }
        };

        TBUtils.info('UIManager', 'Calling format_class:', payload);

        try {
            const response = await fetch('http://localhost:8080/api/isaa/format_class', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API call failed: ${response.status} - ${errorText}`);
            }

            const result = await response.json();

            TBUtils.info('UIManager', 'format_class response:', result);

            return {
                content: result.response || this.formatStructuredResponse(result),
                actions: result.actions || []
            };
        } catch (error) {
            TBUtils.handleError('UIManager', 'format_class failed:', error);
            throw error;
        }
    }

    getSchemaDefinition(schemaName) {
        const schemas = {
            'FormFillRequest': {
                type: 'object',
                properties: {
                    response: { type: 'string', description: 'Response about form filling' },
                    form_data: { type: 'object', description: 'Data to fill in forms' },
                    actions: { type: 'array', items: { type: 'string' }, description: 'Actions to perform' }
                },
                required: ['response']
            },
            'DataExtractionRequest': {
                type: 'object',
                properties: {
                    response: { type: 'string', description: 'Response about data extraction' },
                    extracted_data: { type: 'object', description: 'Extracted information' },
                    data_types: { type: 'array', items: { type: 'string' }, description: 'Types of data found' }
                },
                required: ['response']
            },
            'NavigationRequest': {
                type: 'object',
                properties: {
                    response: { type: 'string', description: 'Response about navigation' },
                    target_elements: { type: 'array', items: { type: 'string' }, description: 'Elements to click' },
                    actions: { type: 'array', items: { type: 'string' }, description: 'Navigation actions' }
                },
                required: ['response']
            }
        };

        return schemas[schemaName] || schemas['FormFillRequest'];
    }

    formatStructuredResponse(result) {
        if (result.response) {
            return result.response;
        }

        let formatted = '';

        if (result.form_data) {
            formatted += 'Form data to fill:\n';
            for (const [key, value] of Object.entries(result.form_data)) {
                formatted += `- ${key}: ${value}\n`;
            }
        }

        if (result.extracted_data) {
            formatted += 'Extracted data:\n';
            formatted += JSON.stringify(result.extracted_data, null, 2);
        }

        if (result.target_elements) {
            formatted += 'Target elements found:\n';
            result.target_elements.forEach((element, index) => {
                formatted += `${index + 1}. ${element}\n`;
            });
        }

        return formatted || 'Task completed successfully';
    }

    async processLocally(message, intent) {
        // Local fallback processing when API is unavailable
        const pageContext = this.getPageContext();

        const responses = {
            'form-filling': `I can help you fill forms. I found ${pageContext.forms} forms on this page.`,
            'data-extraction': `I can help extract data. I found ${pageContext.links} links and ${pageContext.headings} headings on this page.`,
            'navigation': `I can help you navigate. I found ${pageContext.links} links on this page.`,
            'chat': `I'm here to help! The ISAA service is currently unavailable, but I can still assist with basic tasks.`
        };

        return {
            content: responses[intent.type] || responses['chat'],
            actions: []
        };
    }

    getPageContext() {
        const selectedText = window.getSelection().toString().trim();

        return {
            url: window.location.href,
            title: document.title,
            forms: document.querySelectorAll('form').length,
            links: document.querySelectorAll('a[href]').length,
            selectedText: selectedText,
            hasLogin: this.detectLoginForm(),
            hasSearch: this.detectSearchForm(),
            pageType: this.detectPageType()
        };
    }

    detectLoginForm() {
        const forms = document.querySelectorAll('form');
        for (const form of forms) {
            const hasPassword = form.querySelector('input[type="password"]');
            const hasEmail = form.querySelector('input[type="email"], input[name*="email"], input[name*="username"]');
            if (hasPassword && hasEmail) return true;
        }
        return false;
    }

    detectSearchForm() {
        return document.querySelectorAll('input[type="search"], input[name*="search"], input[placeholder*="search"]').length > 0;
    }

    detectPageType() {
        const url = window.location.href.toLowerCase();
        const title = document.title.toLowerCase();

        if (url.includes('login') || title.includes('login') || title.includes('sign in')) return 'login';
        if (url.includes('register') || url.includes('signup') || title.includes('sign up')) return 'register';
        if (url.includes('contact') || title.includes('contact')) return 'contact';
        if (url.includes('search') || title.includes('search')) return 'search';
        if (url.includes('shop') || url.includes('store') || title.includes('shop')) return 'ecommerce';

        return 'general';
    }

    parseActionsFromResponse(response, intent) {
        const actions = [];

        // Parse common action patterns from ISAA response
        if (intent.type === 'form-filling' && response.includes('fill')) {
            actions.push({
                type: 'fill-form',
                description: 'Fill detected form fields'
            });
        }

        if (intent.type === 'navigation' && response.includes('click')) {
            actions.push({
                type: 'click-element',
                description: 'Click specified element'
            });
        }

        if (intent.type === 'data-extraction' && response.includes('extract')) {
            actions.push({
                type: 'extract-data',
                description: 'Extract requested data'
            });
        }

        return actions;
    }

    generateResponseFromStructuredData(data) {
        switch (data.action) {
            case 'fill_form':
                return `I'll fill the form with ${data.fields?.length || 0} field(s). Ready to proceed?`;
            case 'extract_data':
                return `I'll extract ${data.dataTypes?.join(', ') || 'data'} from this page.`;
            case 'click_element':
                return `I'll click on "${data.target}" for you.`;
            case 'navigate':
                return `I'll navigate to ${data.url || 'the specified location'}.`;
            default:
                return 'I understand your request and will help you with that.';
        }
    }

    convertStructuredDataToActions(data) {
        return [{
            type: data.action,
            data: data,
            description: this.generateResponseFromStructuredData(data)
        }];
    }

    async processLocally(message, intent) {
        // Fallback local processing when ToolBox Python is unavailable
        const pageContext = this.getPageContext();

        switch (intent.type) {
            case 'form-filling':
                return await this.handleLocalFormFilling(message, pageContext);
            case 'data-extraction':
                return await this.handleLocalDataExtraction(message, pageContext);
            case 'navigation':
                return await this.handleLocalNavigation(message, pageContext);
            default:
                return await this.handleLocalChat(message, pageContext);
        }
    }

    async handleLocalFormFilling(message, context) {
        const forms = document.querySelectorAll('form');
        if (forms.length === 0) {
            return {
                content: "I don't see any forms on this page to fill.",
                actions: []
            };
        }

        // Extract data from message
        const extractedData = this.extractDataFromMessage(message);

        return {
            content: `I found ${forms.length} form(s) on this page. I can help fill them with the information you provided.`,
            actions: [{
                type: 'fill-form',
                data: extractedData,
                description: 'Fill form with extracted data'
            }]
        };
    }

    async handleLocalDataExtraction(message, context) {
        const dataTypes = this.identifyDataTypes(message);
        const extractedData = this.extractPageData(dataTypes);

        return {
            content: `I extracted ${Object.keys(extractedData).length} types of data from this page.`,
            actions: [{
                type: 'extract-data',
                data: extractedData,
                description: 'Show extracted data'
            }]
        };
    }

    async handleLocalNavigation(message, context) {
        const targets = this.findNavigationTargets(message);

        if (targets.length === 0) {
            return {
                content: "I couldn't find any clickable elements matching your request.",
                actions: []
            };
        }

        return {
            content: `I found ${targets.length} element(s) that match your request.`,
            actions: targets.map(target => ({
                type: 'click-element',
                target: target.element,
                description: `Click "${target.description}"`
            }))
        };
    }

    async handleLocalChat(message, context) {
        const responses = [
            `I can see you're on "${context.title}". How can I help you interact with this page?`,
            `This page has ${context.forms} form(s) and ${context.links} link(s). What would you like me to do?`,
            `I'm here to help you with web automation. I can fill forms, extract data, or navigate pages.`,
            `Based on this ${context.pageType} page, I can assist with various tasks. What do you need?`
        ];

        return {
            content: responses[Math.floor(Math.random() * responses.length)],
            actions: []
        };
    }

    handleAudioInput() {
        const selectedText = window.getSelection().toString().trim();

        if (!selectedText) {
            this.showNotification('Please select some text first, then click the audio button', 'info', 3000);
            return;
        }

        // Start voice recognition for the selected text
        this.startVoiceRecognitionForText(selectedText);
    }

    startVoiceRecognitionForText(selectedText) {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.showNotification('Speech recognition not supported in this browser', 'error', 3000);
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        const audioBtn = document.getElementById('tb-isaa-audio');
        if (audioBtn) {
            audioBtn.innerHTML = '<span>🔴</span>';
            audioBtn.disabled = true;
        }

        recognition.onstart = () => {
            this.showNotification(`Listening for input about: "${selectedText.substring(0, 50)}..."`, 'info', 2000);
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const combinedMessage = `For the selected text "${selectedText}": ${transcript}`;

            const input = document.getElementById('tb-isaa-input');
            if (input) {
                input.value = combinedMessage;
                input.focus();
            }
        };

        recognition.onerror = (event) => {
            this.showNotification(`Speech recognition error: ${event.error}`, 'error', 3000);
        };

        recognition.onend = () => {
            if (audioBtn) {
                audioBtn.innerHTML = '<span>🎤</span>';
                audioBtn.disabled = false;
            }
        };

        recognition.start();
    }

    addISAAMessage(content, role, isLoading = false) {
        const messagesContainer = document.getElementById('tb-isaa-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `tb-isaa-message tb-isaa-${role}`;

        if (isLoading) {
            messageDiv.classList.add('tb-loading');
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'tb-isaa-message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    removeLoadingMessage() {
        const messagesContainer = document.getElementById('tb-isaa-messages');
        if (!messagesContainer) return;

        const loadingMessage = messagesContainer.querySelector('.tb-loading');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    async executeISAAActions(actions) {
        for (const action of actions) {
            try {
                await this.executeISAAAction(action);
                await this.delay(500); // Small delay between actions
            } catch (error) {
                TBUtils.handleError('UIManager', error);
                this.addISAAMessage(`Failed to execute action: ${error.message}`, 'assistant');
            }
        }
    }

    async executeISAAAction(action) {
        switch (action.type) {
            case 'fill-form':
                return await this.executeFormFill(action.data);
            case 'click-element':
                return await this.executeElementClick(action.target);
            case 'extract-data':
                return await this.executeDataExtraction(action.data);
            default:
                throw new Error(`Unknown action type: ${action.type}`);
        }
    }

    async executeFormFill(data) {
        // Implementation for form filling
        const forms = document.querySelectorAll('form');
        if (forms.length === 0) {
            throw new Error('No forms found on page');
        }

        // Fill the first form with available data
        const form = forms[0];
        const inputs = form.querySelectorAll('input, select, textarea');

        let filledCount = 0;
        inputs.forEach(input => {
            if (input.type === 'hidden' || input.type === 'submit') return;

            const fieldName = (input.name || input.id || '').toLowerCase();
            const fieldLabel = this.getFieldLabel(input).toLowerCase();

            // Match data to fields
            for (const [key, value] of Object.entries(data)) {
                if (fieldName.includes(key.toLowerCase()) || fieldLabel.includes(key.toLowerCase())) {
                    input.value = value;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                    filledCount++;
                    break;
                }
            }
        });

        this.addISAAMessage(`Filled ${filledCount} form field(s) successfully.`, 'assistant');
    }

    async executeElementClick(target) {
        if (target && target.click) {
            // Scroll into view
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Highlight briefly
            const originalStyle = target.style.cssText;
            target.style.cssText += 'outline: 2px solid #6c8ee8; background: rgba(108, 142, 232, 0.1);';

            setTimeout(() => {
                target.click();
                target.style.cssText = originalStyle;
            }, 500);

            this.addISAAMessage(`Clicked on "${target.textContent?.trim() || 'element'}" successfully.`, 'assistant');
        } else {
            throw new Error('Target element not found or not clickable');
        }
    }

    async executeDataExtraction(data) {
        // Display extracted data in a formatted way
        let formattedData = 'Extracted Data:\n\n';

        for (const [type, items] of Object.entries(data)) {
            formattedData += `**${type.toUpperCase()}:**\n`;

            if (Array.isArray(items)) {
                items.slice(0, 5).forEach((item, index) => {
                    if (typeof item === 'object') {
                        formattedData += `${index + 1}. ${JSON.stringify(item)}\n`;
                    } else {
                        formattedData += `${index + 1}. ${item}\n`;
                    }
                });

                if (items.length > 5) {
                    formattedData += `... and ${items.length - 5} more items\n`;
                }
            }

            formattedData += '\n';
        }

        this.addISAAMessage(formattedData, 'assistant');
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    showISAAChat() {
        // Create ISAA chat overlay
        const chatOverlay = document.createElement('div');
        chatOverlay.className = 'tb-isaa-chat-overlay';
        chatOverlay.innerHTML = `
            <div class="tb-isaa-chat-container">
                <div class="tb-isaa-chat-header">
                    <div class="tb-isaa-chat-title">
                        <span class="tb-isaa-icon">🤖</span>
                        <span>ISAA Web Assistant</span>
                    </div>
                    <button class="tb-isaa-chat-close" title="Close Chat">×</button>
                </div>
                <div class="tb-isaa-chat-messages" id="tb-isaa-messages">
                    <div class="tb-isaa-message tb-isaa-assistant">
                        <div class="tb-isaa-message-content">
                            Hello! I'm ISAA, your intelligent web assistant. I can help you:
                            <ul>
                                <li>🔍 Fill out forms automatically</li>
                                <li>🖱️ Navigate and click elements</li>
                                <li>📊 Extract data from pages</li>
                                <li>💬 Answer questions about this page</li>
                            </ul>
                            What would you like me to help you with?
                        </div>
                    </div>
                </div>
                <div class="tb-isaa-chat-input-container">
                    <input type="text"
                           class="tb-isaa-chat-input"
                           id="tb-isaa-input"
                           placeholder="Ask ISAA to help with this page..."
                           autocomplete="off">
                    <button class="tb-isaa-chat-send" id="tb-isaa-send">
                        <span>Send</span>
                    </button>
                </div>
                <div class="tb-isaa-quick-actions">
                    <button class="tb-isaa-quick-btn" data-action="fill-forms">Fill Forms</button>
                    <button class="tb-isaa-quick-btn" data-action="extract-data">Extract Data</button>
                    <button class="tb-isaa-quick-btn" data-action="find-links">Find Links</button>
                </div>
            </div>
        `;

        document.body.appendChild(chatOverlay);
        this.isaaChatVisible = true;

        // Setup event listeners
        this.setupISAAChatListeners(chatOverlay);

        // Focus input
        const input = chatOverlay.querySelector('#tb-isaa-input');
        setTimeout(() => input.focus(), 100);

        // Start ISAA session
        this.startISAASession();
    }

    hideISAAChat() {
        const chatOverlay = document.querySelector('.tb-isaa-chat-overlay');
        if (chatOverlay) {
            chatOverlay.remove();
        }
        this.isaaChatVisible = false;
    }

    setupISAAChatListeners(chatOverlay) {
        const input = chatOverlay.querySelector('#tb-isaa-input');
        const sendBtn = chatOverlay.querySelector('#tb-isaa-send');
        const closeBtn = chatOverlay.querySelector('.tb-isaa-chat-close');
        const quickBtns = chatOverlay.querySelectorAll('.tb-isaa-quick-btn');

        // Close button
        closeBtn.addEventListener('click', () => {
            this.hideISAAChat();
        });

        // Send message
        const sendMessage = async () => {
            const message = input.value.trim();
            if (!message) return;

            input.value = '';
            await this.sendISAAMessage(message);
        };

        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Quick action buttons
        quickBtns.forEach(btn => {
            btn.addEventListener('click', async () => {
                const action = btn.dataset.action;
                await this.handleISAAQuickAction(action);
            });
        });

        // Close on outside click
        chatOverlay.addEventListener('click', (e) => {
            if (e.target === chatOverlay) {
                this.hideISAAChat();
            }
        });
    }

    async startISAASession() {
        if (this.isaaPlugin) {
            try {
                this.isaaSessionId = await this.isaaPlugin.startChatSession();
                TBUtils.info('UIManager', `ISAA session started: ${this.isaaSessionId}`);
            } catch (error) {
                TBUtils.handleError('UIManager', error);
            }
        }
    }

    async sendISAAMessage(message) {
        if (!this.isaaPlugin) return;

        // Add user message to chat
        this.addISAAMessage(message, 'user');

        try {
            // Send to ISAA plugin
            const response = await this.isaaPlugin.sendChatMessage(message, this.isaaSessionId);

            // Add assistant response
            this.addISAAMessage(response.response, 'assistant');

            // Show actions if any
            if (response.actions && response.actions.length > 0) {
                this.showISAAActions(response.actions);
            }

        } catch (error) {
            TBUtils.handleError('UIManager', error);
            this.addISAAMessage('Sorry, I encountered an error processing your request.', 'assistant');
        }
    }

    addISAAMessage(content, role) {
        const messagesContainer = document.querySelector('#tb-isaa-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `tb-isaa-message tb-isaa-${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'tb-isaa-message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showISAAActions(actions) {
        const messagesContainer = document.querySelector('#tb-isaa-messages');
        if (!messagesContainer) return;

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'tb-isaa-actions';

        actions.forEach(action => {
            const actionBtn = document.createElement('button');
            actionBtn.className = 'tb-isaa-action-btn';
            actionBtn.textContent = `Execute: ${action.type}`;
            actionBtn.addEventListener('click', async () => {
                await this.executeISAAAction(action);
                actionBtn.disabled = true;
                actionBtn.textContent = 'Executed ✓';
            });
            actionsDiv.appendChild(actionBtn);
        });

        messagesContainer.appendChild(actionsDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async executeISAAAction(action) {
        if (!this.isaaPlugin) return;

        try {
            await this.isaaPlugin.executeAction(action);
            this.addISAAMessage(`Action executed: ${action.type}`, 'assistant');
        } catch (error) {
            TBUtils.handleError('UIManager', error);
            this.addISAAMessage(`Failed to execute action: ${error.message}`, 'assistant');
        }
    }

    async handleISAAQuickAction(action) {
        const quickMessages = {
            'fill-forms': 'Please help me fill out any forms on this page',
            'extract-data': 'Extract all important data from this page',
            'find-links': 'Find and list all the important links on this page'
        };

        const message = quickMessages[action];
        if (message) {
            await this.sendISAAMessage(message);
        }
    }

    async handleAudioInput() {
        try {
            // Get selected text first
            const selectedText = window.getSelection().toString().trim();

            if (selectedText) {
                // Use selected text as input
                const input = document.getElementById('tb-isaa-input');
                if (input) {
                    input.value = selectedText;
                    input.focus();
                }

                this.showNotification(`Selected text added: "${selectedText.substring(0, 50)}${selectedText.length > 50 ? '...' : ''}"`, 'info', 3000);
            } else {
                // Start voice recognition for input
                if (window.tbVoiceEngine) {
                    this.showVoiceIndicator();

                    const result = await window.tbVoiceEngine.startListening();

                    if (result && result.transcript) {
                        const input = document.getElementById('tb-isaa-input');
                        if (input) {
                            input.value = result.transcript;
                            input.focus();
                        }
                    }

                    this.hideVoiceIndicator();
                } else {
                    this.showNotification('Voice recognition not available', 'warning', 3000);
                }
            }
        } catch (error) {
            TBUtils.handleError('UIManager', error);
            this.hideVoiceIndicator();
        }
    }

    addISAAMessage(content, role, isLoading = false) {
        const messagesContainer = document.getElementById('tb-isaa-messages');
        if (!messagesContainer) return;

        // Remove welcome message if it exists
        const welcome = messagesContainer.querySelector('.tb-isaa-welcome');
        if (welcome) {
            welcome.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `tb-isaa-message tb-isaa-${role}`;

        if (isLoading) {
            messageDiv.classList.add('tb-loading');
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'tb-isaa-message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    removeLoadingMessage() {
        const messagesContainer = document.getElementById('tb-isaa-messages');
        if (!messagesContainer) return;

        const loadingMessage = messagesContainer.querySelector('.tb-loading');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    analyzeMessageIntent(message) {
        const messageLower = message.toLowerCase();

        // Form-related intents
        if (messageLower.includes('fill') || messageLower.includes('form') || messageLower.includes('input')) {
            return {
                type: 'form-filling',
                confidence: 0.9,
                entities: this.extractFormEntities(message)
            };
        }

        // Navigation intents
        if (messageLower.includes('click') || messageLower.includes('navigate') || messageLower.includes('go to')) {
            return {
                type: 'navigation',
                confidence: 0.85,
                entities: this.extractNavigationEntities(message)
            };
        }

        // Scraping intents
        if (messageLower.includes('extract') || messageLower.includes('get') || messageLower.includes('find')) {
            return {
                type: 'scraping',
                confidence: 0.8,
                entities: this.extractScrapingEntities(message)
            };
        }

        // General chat
        return {
            type: 'chat',
            confidence: 0.7,
            entities: {}
        };
    }

    extractFormEntities(message) {
        const entities = {};

        // Extract email addresses
        const emailMatch = message.match(/([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/);
        if (emailMatch) entities.email = emailMatch[1];

        // Extract phone numbers
        const phoneMatch = message.match(/(\+?[\d\s\-\(\)]{10,})/);
        if (phoneMatch) entities.phone = phoneMatch[1];

        // Extract names (simple pattern)
        const nameMatch = message.match(/name[:\s]+([a-zA-Z\s]+)/i);
        if (nameMatch) entities.name = nameMatch[1].trim();

        return entities;
    }

    extractNavigationEntities(message) {
        const entities = {};

        // Extract button/link text to click
        const clickMatch = message.match(/click[:\s]+["']?([^"']+)["']?/i);
        if (clickMatch) entities.target = clickMatch[1].trim();

        return entities;
    }

    extractScrapingEntities(message) {
        const entities = {};

        // Extract what type of data to scrape
        if (message.includes('title')) entities.type = 'titles';
        if (message.includes('link')) entities.type = 'links';
        if (message.includes('image')) entities.type = 'images';
        if (message.includes('text')) entities.type = 'text';
        if (message.includes('table')) entities.type = 'tables';

        return entities;
    }

    async executeISAAActions(actions) {
        for (const action of actions) {
            try {
                await this.executeISAAAction(action);
                await this.delay(500); // Small delay between actions
            } catch (error) {
                TBUtils.handleError('UIManager', error);
            }
        }
    }

    async executeISAAAction(action) {
        switch (action.type) {
            case 'fill-forms':
                await this.fillFormsWithData(action.data);
                break;
            case 'click-element':
                await this.clickElementByDescription(action.target);
                break;
            case 'extract-data':
                await this.extractPageData(action.types);
                break;
            default:
                this.addISAAMessage(`Executed: ${action.description}`, 'assistant');
        }
    }

    async fillFormsWithData(data) {
        const forms = document.querySelectorAll('form');
        let filledCount = 0;

        forms.forEach(form => {
            const inputs = form.querySelectorAll('input, select, textarea');
            inputs.forEach(input => {
                const fieldName = input.name || input.id || '';
                const fieldType = input.type || input.tagName.toLowerCase();

                // Match data to fields
                for (const [key, value] of Object.entries(data)) {
                    if (fieldName.toLowerCase().includes(key.toLowerCase()) ||
                        input.placeholder?.toLowerCase().includes(key.toLowerCase())) {
                        input.value = value;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        input.dispatchEvent(new Event('change', { bubbles: true }));
                        filledCount++;
                        break;
                    }
                }
            });
        });

        this.addISAAMessage(`Filled ${filledCount} form fields`, 'assistant');
    }

    async clickElementByDescription(description) {
        const elements = document.querySelectorAll('a, button, [onclick], [role="button"]');

        for (const element of elements) {
            const text = element.textContent.trim().toLowerCase();
            if (text.includes(description.toLowerCase())) {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                await this.delay(300);
                element.click();
                this.addISAAMessage(`Clicked: ${text}`, 'assistant');
                return;
            }
        }

        this.addISAAMessage(`Could not find element: ${description}`, 'assistant');
    }

    async extractPageData(types) {
        const data = {};

        if (types.includes('titles')) {
            data.titles = Array.from(document.querySelectorAll('h1, h2, h3')).map(h => h.textContent.trim());
        }

        if (types.includes('links')) {
            data.links = Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: a.textContent.trim(),
                url: a.href
            }));
        }

        if (types.includes('images')) {
            data.images = Array.from(document.querySelectorAll('img')).map(img => ({
                alt: img.alt,
                src: img.src
            }));
        }

        this.addISAAMessage(`Extracted data: ${JSON.stringify(data, null, 2)}`, 'assistant');
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Action handlers
    async executeAction(action) {
        try {
            TBUtils.info('UIManager', `Executing action: ${action}`);

            if (typeof chrome !== 'undefined' && chrome.runtime) {
                const response = await chrome.runtime.sendMessage({
                    type: 'TB_UI_ACTION',
                    action,
                    timestamp: Date.now()
                });

                if (response && response.notification) {
                    this.showNotification(response.notification.message, response.notification.type);
                }
            }

            // Update action counter
            this.incrementActionCounter();
        } catch (error) {
            TBUtils.handleError('UIManager', error);
            this.showNotification('Action failed', 'error');
        }
    }

    toggleVoice() {
        if (window.tbVoiceEngine) {
            window.tbVoiceEngine.toggle();
        }
    }

    startVoiceSearch() {
        if (window.tbVoiceEngine) {
            window.tbVoiceEngine.startListening();
            this.showVoiceIndicator('listening');
        }
    }

    openSettings() {
        // Implementation for opening settings
        this.showNotification('Settings panel coming soon!', 'info');
    }

    // Notification system
    showNotification(message, type = 'info', duration = 3000) {
        const notification = TBUtils.createElement('div', {
            className: `tb-notification tb-notification-${type}`,
            innerHTML: `
                <div class="tb-notification-content">
                    <span class="tb-notification-icon">${this.getNotificationIcon(type)}</span>
                    <span class="tb-notification-message">${message}</span>
                    <button class="tb-notification-close">✕</button>
                </div>
            `
        });

        if (document.body) {
            document.body.appendChild(notification);
        }
        this.notifications.push(notification);

        // Auto-remove after duration
        setTimeout(() => {
            this.removeNotification(notification);
        }, duration);

        // Close button handler
        notification.querySelector('.tb-notification-close').addEventListener('click', () => {
            this.removeNotification(notification);
        });

        if (this.settings.animations) {
            TBUtils.fadeIn(notification);
        }
    }

    removeNotification(notification) {
        const index = this.notifications.indexOf(notification);
        if (index > -1) {
            this.notifications.splice(index, 1);
        }

        if (this.settings.animations) {
            TBUtils.fadeOut(notification);
            setTimeout(() => notification.remove(), 300);
        } else {
            notification.remove();
        }
    }

    getNotificationIcon(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        return icons[type] || icons.info;
    }

    // Utility methods
    incrementActionCounter() {
        const counter = document.getElementById('tb-actions-count');
        if (counter) {
            const current = parseInt(counter.textContent) || 0;
            counter.textContent = current + 1;
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('tb-connection-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `tb-status-value tb-status-${status.toLowerCase()}`;
        }
    }

    // Event system
    on(event, handler) {
        if (!this.eventHandlers) {
            this.eventHandlers = new Map();
        }

        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }

        this.eventHandlers.get(event).push(handler);
    }

    emit(event, data = null) {
        if (this.eventHandlers && this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    TBUtils.handleError('UIManager', error);
                }
            });
        }
    }

    // Settings management
    async updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        await TBUtils.setStorage({
            ui_theme: this.settings.theme,
            ui_position: this.settings.position,
            ui_opacity: this.settings.opacity,
            ui_animations: this.settings.animations,
            ui_compact_mode: this.settings.compactMode,
            ui_auto_hide: this.settings.autoHide,
            ui_auto_hide_delay: this.settings.autoHideDelay
        });

        // Apply settings
        this.applySettings();
    }

    applySettings() {
        if (this.panel) {
            this.panel.className = `tb-panel tb-theme-${this.settings.theme} tb-position-${this.settings.position}`;
            this.panel.style.opacity = this.settings.opacity;
        }
    }

    // ==================== PASSWORD MANAGEMENT ====================

    async generatePassword() {
        try {
            const response = await fetch('http://localhost:8080/api/PasswordManager/generate_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    length: 16,
                    include_symbols: true,
                    include_numbers: true,
                    include_uppercase: true,
                    include_lowercase: true,
                    exclude_ambiguous: true
                })
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success && result.data && result.data.password) {
                const password = result.data.password;

                // Find password field and fill it
                const passwordField = document.querySelector('input[type="password"]:not([readonly]):not([disabled])');
                if (passwordField) {
                    passwordField.value = password;
                    passwordField.dispatchEvent(new Event('input', { bubbles: true }));
                    passwordField.dispatchEvent(new Event('change', { bubbles: true }));

                    // Highlight the field briefly
                    this.highlightElement(passwordField);
                }

                // Copy to clipboard
                await navigator.clipboard.writeText(password);
                this.showNotification('🔑 Password generated and copied to clipboard', 'success', 4000);

                // Show password strength
                this.showPasswordStrength(password);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            TBUtils.handleError('UIManager', 'Password generation failed:', error);
            this.showNotification('❌ Password generation failed', 'error', 3000);
        }
    }

    async autoFillPassword() {
        try {
            const response = await fetch('http://localhost:8080/api/PasswordManager/get_password_for_autofill', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    url: window.location.href
                })
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success && result.data && result.data.entry) {
                const entry = result.data.entry;

                // Find form fields
                const usernameField = document.querySelector('input[type="email"], input[type="text"], input[name*="username"], input[name*="email"]');
                const passwordField = document.querySelector('input[type="password"]');

                if (usernameField && passwordField) {
                    usernameField.value = entry.username;
                    passwordField.value = entry.password;

                    // Trigger events
                    usernameField.dispatchEvent(new Event('input', { bubbles: true }));
                    usernameField.dispatchEvent(new Event('change', { bubbles: true }));
                    passwordField.dispatchEvent(new Event('input', { bubbles: true }));
                    passwordField.dispatchEvent(new Event('change', { bubbles: true }));

                    // Highlight fields
                    this.highlightElement(usernameField);
                    this.highlightElement(passwordField);

                    this.showNotification('✅ Password auto-filled', 'success', 3000);

                    // Show 2FA code if available
                    if (result.data.totp_code) {
                        this.showTOTPCode(result.data.totp_code, entry.totp_issuer || entry.title);
                    }
                } else {
                    this.showNotification('❌ No login form found on this page', 'warning', 3000);
                }
            } else {
                this.showNotification('❌ No matching password found', 'warning', 3000);
            }
        } catch (error) {
            TBUtils.handleError('UIManager', 'Auto-fill failed:', error);
            this.showNotification('❌ Auto-fill failed', 'error', 3000);
        }
    }

    async openPasswordManager() {
        try {
            // Create password manager modal
            const modal = this.createPasswordManagerModal();
            document.body.appendChild(modal);

            // Load passwords
            await this.loadPasswordList(modal);
        } catch (error) {
            TBUtils.handleError('UIManager', 'Failed to open password manager:', error);
            this.showNotification('❌ Failed to open password manager', 'error', 3000);
        }
    }

    createPasswordManagerModal() {
        const modal = document.createElement('div');
        modal.className = 'tb-modal-overlay';
        modal.innerHTML = `
            <div class="tb-modal tb-password-manager-modal">
                <div class="tb-modal-header">
                    <h3>🔒 Password Manager</h3>
                    <button class="tb-modal-close">✕</button>
                </div>
                <div class="tb-modal-content">
                    <div class="tb-password-toolbar">
                        <input type="text" id="tb-password-search" placeholder="Search passwords..." class="tb-search-input">
                        <button id="tb-add-password" class="tb-btn tb-btn-primary">Add Password</button>
                        <button id="tb-import-passwords" class="tb-btn tb-btn-secondary">Import</button>
                    </div>
                    <div id="tb-password-list" class="tb-password-list">
                        <div class="tb-loading">Loading passwords...</div>
                    </div>
                </div>
            </div>
        `;

        // Event listeners
        modal.querySelector('.tb-modal-close').addEventListener('click', () => {
            modal.remove();
        });

        modal.querySelector('#tb-password-search').addEventListener('input', (e) => {
            this.filterPasswordList(e.target.value);
        });

        modal.querySelector('#tb-add-password').addEventListener('click', () => {
            this.showAddPasswordDialog();
        });

        modal.querySelector('#tb-import-passwords').addEventListener('click', () => {
            this.showImportDialog();
        });

        return modal;
    }

    async loadPasswordList(modal) {
        try {
            const response = await fetch('http://localhost:8080/api/PasswordManager/list_passwords', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({})
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success && result.data) {
                this.displayPasswordList(result.data.passwords || [], modal);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            TBUtils.handleError('UIManager', 'Failed to load passwords:', error);
            const listContainer = modal.querySelector('#tb-password-list');
            listContainer.innerHTML = '<div class="tb-error">Failed to load passwords</div>';
        }
    }

    displayPasswordList(passwords, modal) {
        const listContainer = modal.querySelector('#tb-password-list');

        if (passwords.length === 0) {
            listContainer.innerHTML = `
                <div class="tb-empty-state">
                    <div class="tb-icon">🔒</div>
                    <p>No passwords saved yet</p>
                    <button class="tb-btn tb-btn-primary" onclick="this.closest('.tb-modal').querySelector('#tb-add-password').click()">
                        Add Your First Password
                    </button>
                </div>
            `;
            return;
        }

        const passwordsHTML = passwords.map(password => `
            <div class="tb-password-item" data-id="${password.id}">
                <div class="tb-password-info">
                    <div class="tb-password-title">${password.title || password.url}</div>
                    <div class="tb-password-username">${password.username}</div>
                    <div class="tb-password-url">${password.url}</div>
                    ${password.totp_secret ? '<div class="tb-password-2fa">🔐 2FA Enabled</div>' : ''}
                </div>
                <div class="tb-password-actions">
                    ${password.totp_secret ? `<button class="tb-btn tb-btn-small" onclick="window.tbUIManager.showTOTP('${password.id}')">2FA</button>` : ''}
                    <button class="tb-btn tb-btn-small" onclick="window.tbUIManager.copyPassword('${password.id}')">Copy</button>
                    <button class="tb-btn tb-btn-small" onclick="window.tbUIManager.fillPassword('${password.id}')">Fill</button>
                    <button class="tb-btn tb-btn-small tb-btn-danger" onclick="window.tbUIManager.deletePassword('${password.id}')">Delete</button>
                </div>
            </div>
        `).join('');

        listContainer.innerHTML = passwordsHTML;
    }

    showPasswordStrength(password) {
        const strength = this.calculatePasswordStrength(password);
        const strengthText = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'][strength];
        const strengthColor = ['#ff4444', '#ff8800', '#ffaa00', '#88cc00', '#44cc44'][strength];

        this.showNotification(`Password strength: ${strengthText}`, 'info', 3000, strengthColor);
    }

    calculatePasswordStrength(password) {
        let score = 0;

        // Length
        if (password.length >= 8) score++;
        if (password.length >= 12) score++;
        if (password.length >= 16) score++;

        // Character types
        if (/[a-z]/.test(password)) score++;
        if (/[A-Z]/.test(password)) score++;
        if (/[0-9]/.test(password)) score++;
        if (/[^A-Za-z0-9]/.test(password)) score++;

        return Math.min(Math.floor(score / 1.5), 4);
    }

    showTOTPCode(code, issuer) {
        // Remove existing TOTP display
        const existing = document.querySelector('.tb-totp-display');
        if (existing) existing.remove();

        const totpDiv = document.createElement('div');
        totpDiv.className = 'tb-totp-display';
        totpDiv.innerHTML = `
            <div class="tb-totp-header">
                <span class="tb-totp-issuer">${issuer || '2FA Code'}</span>
                <button class="tb-totp-close">✕</button>
            </div>
            <div class="tb-totp-code" onclick="navigator.clipboard.writeText('${code}')">${code}</div>
            <div class="tb-totp-timer">
                <div class="tb-totp-progress"></div>
                <span class="tb-totp-countdown">30</span>
            </div>
            <div class="tb-totp-hint">Click code to copy</div>
        `;

        document.body.appendChild(totpDiv);

        // Auto-remove after 30 seconds
        setTimeout(() => {
            if (totpDiv.parentNode) {
                totpDiv.remove();
            }
        }, 30000);

        // Close button
        totpDiv.querySelector('.tb-totp-close').addEventListener('click', () => {
            totpDiv.remove();
        });

        // Start countdown
        this.startTOTPCountdown(totpDiv);
    }

    startTOTPCountdown(totpDiv) {
        const progressBar = totpDiv.querySelector('.tb-totp-progress');
        const countdown = totpDiv.querySelector('.tb-totp-countdown');

        let timeLeft = 30 - (Math.floor(Date.now() / 1000) % 30);

        const updateTimer = () => {
            timeLeft = 30 - (Math.floor(Date.now() / 1000) % 30);
            if (timeLeft === 30) timeLeft = 0;

            countdown.textContent = timeLeft;
            progressBar.style.width = `${(timeLeft / 30) * 100}%`;

            if (timeLeft === 0) {
                totpDiv.remove();
            }
        };

        updateTimer();
        const interval = setInterval(updateTimer, 1000);

        // Clean up interval when element is removed
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.removedNodes.forEach((node) => {
                    if (node === totpDiv) {
                        clearInterval(interval);
                        observer.disconnect();
                    }
                });
            });
        });

        observer.observe(document.body, { childList: true });
    }

    highlightElement(element) {
        const originalStyle = element.style.cssText;
        element.style.cssText += `
            border: 2px solid var(--tb-accent-primary) !important;
            box-shadow: 0 0 8px rgba(108, 142, 232, 0.5) !important;
            transition: all 0.3s ease !important;
        `;

        setTimeout(() => {
            element.style.cssText = originalStyle;
        }, 2000);
    }

    async copyPassword(passwordId) {
        try {
            const response = await fetch('http://localhost:8080/api/PasswordManager/get_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ entry_id: passwordId })
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success && result.data) {
                await navigator.clipboard.writeText(result.data.password);
                this.showNotification('📋 Password copied to clipboard', 'success', 2000);
            } else {
                throw new Error('Password not found');
            }
        } catch (error) {
            TBUtils.handleError('UIManager', 'Failed to copy password:', error);
            this.showNotification('❌ Failed to copy password', 'error', 3000);
        }
    }

    async fillPassword(passwordId) {
        try {
            const response = await fetch('http://localhost:8080/api/PasswordManager/get_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ entry_id: passwordId })
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success && result.data) {
                const entry = result.data;

                // Find form fields
                const usernameField = document.querySelector('input[type="email"], input[type="text"], input[name*="username"], input[name*="email"]');
                const passwordField = document.querySelector('input[type="password"]');

                if (usernameField && passwordField) {
                    usernameField.value = entry.username;
                    passwordField.value = entry.password;

                    // Trigger events
                    usernameField.dispatchEvent(new Event('input', { bubbles: true }));
                    usernameField.dispatchEvent(new Event('change', { bubbles: true }));
                    passwordField.dispatchEvent(new Event('input', { bubbles: true }));
                    passwordField.dispatchEvent(new Event('change', { bubbles: true }));

                    // Highlight fields
                    this.highlightElement(usernameField);
                    this.highlightElement(passwordField);

                    this.showNotification('✅ Password filled', 'success', 3000);

                    // Close modal
                    const modal = document.querySelector('.tb-modal-overlay');
                    if (modal) modal.remove();
                } else {
                    this.showNotification('❌ No login form found', 'warning', 3000);
                }
            } else {
                throw new Error('Password not found');
            }
        } catch (error) {
            TBUtils.handleError('UIManager', 'Failed to fill password:', error);
            this.showNotification('❌ Failed to fill password', 'error', 3000);
        }
    }

    async showTOTP(passwordId) {
        try {
            const response = await fetch('http://localhost:8080/api/PasswordManager/generate_totp_code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ entry_id: passwordId })
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success && result.data) {
                this.showTOTPCode(result.data.code, result.data.issuer || result.data.account);
            } else {
                throw new Error('No TOTP configured');
            }
        } catch (error) {
            TBUtils.handleError('UIManager', 'Failed to generate TOTP:', error);
            this.showNotification('❌ Failed to generate 2FA code', 'error', 3000);
        }
    }

    async deletePassword(passwordId) {
        if (!confirm('Are you sure you want to delete this password?')) {
            return;
        }

        try {
            const response = await fetch('http://localhost:8080/api/PasswordManager/delete_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ entry_id: passwordId })
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success) {
                this.showNotification('🗑️ Password deleted', 'success', 3000);

                // Refresh password list
                const modal = document.querySelector('.tb-password-manager-modal');
                if (modal) {
                    await this.loadPasswordList(modal);
                }
            } else {
                throw new Error('Delete failed');
            }
        } catch (error) {
            TBUtils.handleError('UIManager', 'Failed to delete password:', error);
            this.showNotification('❌ Failed to delete password', 'error', 3000);
        }
    }

    filterPasswordList(searchTerm) {
        const passwordItems = document.querySelectorAll('.tb-password-item');
        const searchLower = searchTerm.toLowerCase();

        passwordItems.forEach(item => {
            const title = item.querySelector('.tb-password-title').textContent.toLowerCase();
            const username = item.querySelector('.tb-password-username').textContent.toLowerCase();
            const url = item.querySelector('.tb-password-url').textContent.toLowerCase();

            const matches = title.includes(searchLower) ||
                          username.includes(searchLower) ||
                          url.includes(searchLower);

            item.style.display = matches ? 'flex' : 'none';
        });
    }

    showAddPasswordDialog() {
        // Implementation for adding new password
        this.showNotification('Add password dialog coming soon!', 'info', 3000);
    }

    showImportDialog() {
        // Implementation for importing passwords
        this.showNotification('Import dialog coming soon!', 'info', 3000);
    }
}

// Export for global use
if (typeof window !== 'undefined') {
    window.TBUIManager = TBUIManager;

    // Initialize UI Manager when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.tbUIManager = new TBUIManager();
        });
    } else {
        window.tbUIManager = new TBUIManager();
    }
}
