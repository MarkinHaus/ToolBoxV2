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

        this.init();
    }

    async init() {
        try {
            await this.loadSettings();
            this.createMainPanel();
            this.createVoiceIndicator();
            this.setupEventListeners();

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

        // Logo section
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

        // Controls section
        const controls = document.createElement('div');
        controls.className = 'tb-controls';

        const buttons = [
            { id: 'tb-voice-toggle', icon: '🎤', title: 'Voice Commands' },
            { id: 'tb-search-toggle', icon: '🔍', title: 'Smart Search' },
            { id: 'tb-settings-toggle', icon: '⚙️', title: 'Settings' },
            { id: 'tb-close', icon: '✕', title: 'Close' }
        ];

        buttons.forEach(btn => {
            const button = document.createElement('button');
            button.className = 'tb-btn tb-btn-icon';
            button.id = btn.id;
            button.title = btn.title;

            const icon = document.createElement('span');
            icon.className = 'tb-icon';
            icon.textContent = btn.icon;

            button.appendChild(icon);
            controls.appendChild(button);
        });

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
            { id: 'tb-auto-login', icon: '🔓', text: 'Auto Fill', class: 'tb-btn-success' },
            { id: 'tb-generate-password', icon: '🔑', text: 'Generate', class: 'tb-btn-warning' },
            { id: 'tb-screenshot', icon: '📸', text: 'Screenshot', class: 'tb-btn-info' }
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

        // Search section (initially hidden)
        const searchSection = document.createElement('div');
        searchSection.className = 'tb-search-section tb-hidden';
        searchSection.id = 'tb-search-section';

        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.className = 'tb-search-input';
        searchInput.id = 'tb-search-input';
        searchInput.placeholder = 'Smart search with AI...';

        const searchResults = document.createElement('div');
        searchResults.className = 'tb-search-results';
        searchResults.id = 'tb-search-results';
        searchResults.innerHTML = '<div class="tb-search-placeholder"><span class="tb-icon">🔍</span><p>Start typing to search...</p></div>';

        searchSection.appendChild(searchInput);
        searchSection.appendChild(searchResults);

        content.appendChild(quickActions);
        content.appendChild(searchSection);
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

        document.getElementById('tb-settings-toggle')?.addEventListener('click', () => {
            this.openSettings();
        });

        document.getElementById('tb-close')?.addEventListener('click', () => {
            this.hidePanel();
        });

        // Quick actions
        document.getElementById('tb-auto-login')?.addEventListener('click', () => {
            this.executeAction('auto-login');
        });

        document.getElementById('tb-generate-password')?.addEventListener('click', () => {
            this.executeAction('generate-password');
        });



        document.getElementById('tb-screenshot')?.addEventListener('click', () => {
            this.executeAction('screenshot');
        });

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
}

// Export for global use
if (typeof window !== 'undefined') {
    window.TBUIManager = TBUIManager;
}
