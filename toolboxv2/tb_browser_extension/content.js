// ToolBox Browser Extension - Enhanced Content Script
// Main content script with integrated voice, search, and gesture capabilities

class TBExtensionContent {
    constructor() {
        this.initialized = false;
        this.components = {};
        this.settings = {
            enableVoice: true,
            enableSearch: true,
            enableGestures: true,
            enableAutoLogin: true,
            debugMode: false
        };

        this.init();
    }

    async init() {
        try {
            TBUtils.info('Content', 'Initializing ToolBox Extension v2.0.0');

            // Load settings
            await this.loadSettings();

            // Initialize core components
            await this.initializeComponents();

            // Setup message handling
            this.setupMessageHandling();

            // Setup page monitoring
            this.setupPageMonitoring();

            // Auto-detect features
            this.detectPageFeatures();

            this.initialized = true;
            TBUtils.info('Content', '🚀 ToolBox Extension Content Ready');

        } catch (error) {
            TBUtils.handleError('Content', error);
        }
    }

    async loadSettings() {
        const stored = await TBUtils.getStorage([
            'enable_voice',
            'enable_search',
            'enable_gestures',
            'enable_auto_login',
            'debug_mode'
        ]);

        this.settings = {
            ...this.settings,
            enableVoice: stored.enable_voice !== false,
            enableSearch: stored.enable_search !== false,
            enableGestures: stored.enable_gestures !== false,
            enableAutoLogin: stored.enable_auto_login !== false,
            debugMode: stored.debug_mode || false
        };

        TBUtils._debug = this.settings.debugMode;
    }

    async initializeComponents() {
        // Initialize UI Manager first
        this.components.uiManager = new TBUIManager();
        window.tbUIManager = this.components.uiManager;

        // Initialize Voice Engine
        if (this.settings.enableVoice) {
            this.components.voiceEngine = new TBVoiceEngine();
            window.tbVoiceEngine = this.components.voiceEngine;
        }

        // Initialize Search Engine
        if (this.settings.enableSearch) {
            this.components.searchEngine = new TBSearchEngine();
            window.tbSearchEngine = this.components.searchEngine;
        }

        // Initialize Gesture Detector
        if (this.settings.enableGestures) {
            this.components.gestureDetector = new TBGestureDetector();
            window.tbGestureDetector = this.components.gestureDetector;
        }

        // Initialize legacy components
        this.components.passwordManager = new PasswordManager();
        this.components.smartSearch = new SmartSearch();

        TBUtils.info('Content', 'All components initialized');
    }

    setupMessageHandling() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sendResponse);
            return true; // Keep message channel open for async responses
        });
    }

    setupPageMonitoring() {
        // Monitor for page changes
        let lastUrl = location.href;
        const observer = new MutationObserver(() => {
            const currentUrl = location.href;
            if (currentUrl !== lastUrl) {
                lastUrl = currentUrl;
                this.onPageChange();
            }
        });

        if (document.documentElement) {
            observer.observe(document.documentElement, {
                subtree: true,
                childList: true
            });
        }

        // Monitor for form changes
        this.monitorForms();
    }

    onPageChange() {
        TBUtils.info('Content', `Page changed to: ${location.href}`);

        // Rebuild search index
        if (this.components.searchEngine) {
            this.components.searchEngine.buildPageIndex();
        }

        // Re-detect page features
        this.detectPageFeatures();
    }

    monitorForms() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const forms = node.querySelectorAll ? node.querySelectorAll('form') : [];
                        forms.forEach(form => this.processForm(form));

                        if (node.tagName === 'FORM') {
                            this.processForm(node);
                        }
                    }
                });
            });
        });

        if (document.body) {
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }
    }

    detectPageFeatures() {
        // Auto-detect login forms
        if (this.settings.enableAutoLogin) {
            this.detectLoginForms();
        }

        // Detect page type and content
        this.analyzePageContent();
    }

    async handleMessage(message, sendResponse) {
        try {
            TBUtils.info('Content', 'Received message:', message);

            switch (message.type) {
                case 'TB_TOGGLE_PANEL':
                    this.togglePanel();
                    sendResponse({ success: true });
                    break;

                case 'TB_VOICE_COMMAND':
                    await this.handleVoiceCommand(message.command, message.data);
                    sendResponse({ success: true });
                    break;

                case 'TB_GESTURE_ACTION':
                    await this.handleGestureAction(message.action, message.data);
                    sendResponse({ success: true });
                    break;

                case 'TB_SMART_SEARCH':
                    await this.performSmartSearch(message.text);
                    sendResponse({ success: true });
                    break;

                case 'TB_AUTO_LOGIN':
                    await this.autoLogin(message.url);
                    sendResponse({ success: true });
                    break;

                case 'TB_INSERT_PASSWORD':
                    this.insertPassword(message.password);
                    sendResponse({ success: true });
                    break;

                case 'TB_AI_ANALYZE':
                    await this.aiAnalyzePage();
                    sendResponse({ success: true });
                    break;

                case 'TB_SCREENSHOT':
                    await this.takeScreenshot();
                    sendResponse({ success: true });
                    break;

                case 'TB_UPDATE_SETTINGS':
                    await this.updateSettings(message.settings);
                    sendResponse({ success: true });
                    break;

                default:
                    TBUtils.warn('Content', `Unknown message type: ${message.type}`);
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            TBUtils.handleError('Content', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    async handleVoiceCommand(command, data) {
        TBUtils.info('Content', `Handling voice command: ${command}`);

        switch (command) {
            case 'toggle-panel':
                this.togglePanel();
                break;

            case 'close-panel':
                if (this.components.uiManager) {
                    this.components.uiManager.hidePanel();
                }
                break;

            case 'search-page':
                if (this.components.uiManager) {
                    this.components.uiManager.showSearch();
                    if (data.text) {
                        await this.performSmartSearch(data.text);
                    }
                }
                break;

            case 'analyze-page':
                await this.aiAnalyzePage();
                break;

            case 'auto-login':
                await this.autoLogin();
                break;

            case 'generate-password':
                await this.generatePassword();
                break;

            case 'screenshot':
                await this.takeScreenshot();
                break;

            case 'scroll':
                this.handleScrollCommand(data.direction);
                break;

            case 'navigate':
                this.handleNavigateCommand(data.direction);
                break;

            case 'refresh':
                location.reload();
                break;

            case 'isaa-query':
                await this.queryISAA(data.text);
                break;

            case 'isaa-help':
                await this.showISAAHelp();
                break;

            case 'smart-search':
                await this.performSmartSearch(data.text);
                break;

            default:
                TBUtils.warn('Content', `Unknown voice command: ${command}`);
        }
    }

    async handleGestureAction(action, data) {
        TBUtils.info('Content', `Handling gesture action: ${action}`);

        switch (action) {
            case 'toggle-panel':
                this.togglePanel();
                break;

            case 'navigate-forward':
                history.forward();
                break;

            case 'navigate-back':
                history.back();
                break;

            case 'scroll-top':
                window.scrollTo({ top: 0, behavior: 'smooth' });
                break;

            case 'scroll-bottom':
                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                break;

            case 'refresh-page':
                location.reload();
                break;

            case 'smart-search':
                if (this.components.uiManager) {
                    this.components.uiManager.showSearch();
                }
                break;

            default:
                TBUtils.warn('Content', `Unknown gesture action: ${action}`);
        }
    }

    togglePanel() {
        if (this.components.uiManager) {
            this.components.uiManager.togglePanel();
        }
    }

    async performSmartSearch(query) {
        if (this.components.searchEngine && query) {
            const results = await this.components.searchEngine.smartSearch(query);
            TBUtils.info('Content', `Smart search found ${results.length} results`);
            return results;
        }
        return [];
    }

    handleScrollCommand(direction) {
        const scrollAmount = window.innerHeight * 0.8;

        switch (direction) {
            case 'up':
                window.scrollBy({ top: -scrollAmount, behavior: 'smooth' });
                break;
            case 'down':
                window.scrollBy({ top: scrollAmount, behavior: 'smooth' });
                break;
        }
    }

    handleNavigateCommand(direction) {
        switch (direction) {
            case 'back':
                history.back();
                break;
            case 'forward':
                history.forward();
                break;
        }
    }

    async queryISAA(text) {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_ISAA_REQUEST',
                function: 'query',
                data: {
                    query: text,
                    context: this.getPageContext()
                }
            });

            if (response && response.answer && this.components.voiceEngine) {
                await this.components.voiceEngine.speak(response.answer);
            }

            return response;
        } catch (error) {
            TBUtils.handleError('Content', error);
        }
    }

    async showISAAHelp() {
        const helpText = `
            ToolBox Voice Commands:
            - "Open ToolBox" or "Close ToolBox" - Toggle panel
            - "Search page [query]" - Smart search
            - "Analyze page" - AI analysis
            - "Auto login" - Automatic login
            - "Generate password" - Create secure password
            - "Take screenshot" - Capture page
            - "Ask ISAA [question]" - Query AI assistant
            - "Scroll up/down" - Navigate page
            - "Go back/forward" - Browser navigation
            - "Refresh page" - Reload current page
        `;

        if (this.components.voiceEngine) {
            await this.components.voiceEngine.speak("Here are the available voice commands");
        }

        if (this.components.uiManager) {
            this.components.uiManager.showNotification(helpText, 'info', 10000);
        }
    }

    getPageContext() {
        return {
            url: window.location.href,
            title: document.title,
            domain: window.location.hostname,
            text: document.body.innerText.substring(0, 2000),
            timestamp: Date.now()
        };
    }

    analyzePageContent() {
        const analysis = {
            hasLoginForm: this.hasLoginForm(),
            hasSearchForm: this.hasSearchForm(),
            hasContactForm: this.hasContactForm(),
            isEcommerce: this.isEcommercePage(),
            isSocialMedia: this.isSocialMediaPage(),
            isNewsArticle: this.isNewsArticle(),
            language: this.detectLanguage(),
            readingTime: this.estimateReadingTime()
        };

        TBUtils.info('Content', 'Page analysis:', analysis);
        return analysis;
    }

    processForm(form) {
        if (!form || form.tagName !== 'FORM') return;

        try {
            // Check if form has password field (login form)
            const passwordField = form.querySelector('input[type="password"]');
            const usernameField = form.querySelector('input[type="email"], input[type="text"], input[name*="user"], input[name*="email"]');

            if (passwordField && usernameField) {
                // This is likely a login form
                this.enhanceLoginForm(form, usernameField, passwordField);
            }

            // Check for other form types
            const submitButton = form.querySelector('input[type="submit"], button[type="submit"], button:not([type])');
            if (submitButton) {
                // Add ToolBox enhancement indicator
                if (!form.classList.contains('tb-enhanced')) {
                    form.classList.add('tb-enhanced');
                    TBUtils.info('Content', 'Enhanced form:', form);
                }
            }
        } catch (error) {
            TBUtils.error('Content', 'Error processing form:', error);
        }
    }

    enhanceLoginForm(form, usernameField, passwordField) {
        try {
            // Add auto-fill button if password manager is available
            if (!form.querySelector('.tb-autofill-btn')) {
                const autofillBtn = document.createElement('button');
                autofillBtn.type = 'button';
                autofillBtn.className = 'tb-autofill-btn';
                autofillBtn.innerHTML = '🔐';
                autofillBtn.title = 'ToolBox Auto-fill';
                autofillBtn.style.cssText = `
                    position: absolute;
                    right: 5px;
                    top: 50%;
                    transform: translateY(-50%);
                    background: var(--tb-accent-primary, #6c8ee8);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    width: 24px;
                    height: 24px;
                    cursor: pointer;
                    font-size: 12px;
                    z-index: 1000;
                    transition: all 0.2s ease;
                `;

                // Position relative to username field
                if (usernameField.parentNode) {
                    usernameField.parentNode.style.position = 'relative';
                    usernameField.parentNode.appendChild(autofillBtn);

                    autofillBtn.addEventListener('click', async (e) => {
                        e.preventDefault();
                        await this.handleAutoFill(usernameField, passwordField);
                    });
                }
            }
        } catch (error) {
            TBUtils.error('Content', 'Error enhancing login form:', error);
        }
    }

    async handleAutoFill(usernameField, passwordField) {
        try {
            // Send message to background script for password manager
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'get_password_for_autofill',
                args: { url: window.location.href }
            });

            if (response && response.success && response.data && response.data.entry) {
                const entry = response.data.entry;
                usernameField.value = entry.username;
                passwordField.value = entry.password;

                // Trigger change events
                usernameField.dispatchEvent(new Event('input', { bubbles: true }));
                passwordField.dispatchEvent(new Event('input', { bubbles: true }));

                TBUtils.info('Content', 'Auto-fill completed');
            } else {
                TBUtils.warn('Content', 'No matching password found');
            }
        } catch (error) {
            TBUtils.error('Content', 'Auto-fill failed:', error);
        }
    }

    hasLoginForm() {
        const loginSelectors = [
            'form[action*="login"]',
            'form[action*="signin"]',
            'form[action*="auth"]',
            'input[type="password"]',
            'input[name*="password"]',
            'input[name*="login"]',
            'input[name*="email"]'
        ];

        return loginSelectors.some(selector => document.querySelector(selector));
    }

    hasSearchForm() {
        return document.querySelector('input[type="search"], input[name*="search"], input[name*="query"]') !== null;
    }

    hasContactForm() {
        const contactSelectors = [
            'form[action*="contact"]',
            'input[name*="email"]',
            'textarea[name*="message"]',
            'input[name*="subject"]'
        ];

        return contactSelectors.some(selector => document.querySelector(selector));
    }

    isEcommercePage() {
        const ecommerceIndicators = [
            '.price',
            '.cart',
            '.add-to-cart',
            '.buy-now',
            '.checkout',
            '[data-price]',
            '.product'
        ];

        return ecommerceIndicators.some(selector => document.querySelector(selector));
    }

    isSocialMediaPage() {
        const socialDomains = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com'];
        return socialDomains.some(domain => window.location.hostname.includes(domain));
    }

    isNewsArticle() {
        return document.querySelector('article, .article, .post, .news-content, [role="article"]') !== null;
    }

    detectLanguage() {
        return document.documentElement.lang || 'en';
    }

    estimateReadingTime() {
        try {
            if (!document.body) return 0;
            const text = document.body.innerText || '';
            const wordsPerMinute = 200;
            const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
            return Math.ceil(wordCount / wordsPerMinute);
        } catch (error) {
            TBUtils.warn('Content', 'Could not estimate reading time:', error);
            return 0;
        }
    }

    async injectTBCore() {
        // Inject ToolBox JavaScript core
        const script = document.createElement('script');
        script.src = chrome.runtime.getURL('tb-core.js');
        script.onload = () => {
            // Initialize TB object
            window.TB = new ToolBoxCore();
            this.dispatchEvent(new CustomEvent('tb:ready'));
        };
        (document.head || document.documentElement).appendChild(script);
    }



    detectLoginForms() {
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            const emailInput = form.querySelector('input[type="email"], input[name*="email"], input[name*="username"]');
            const passwordInput = form.querySelector('input[type="password"]');

            if (emailInput && passwordInput) {
                this.enhanceLoginForm(form, emailInput, passwordInput);
            }
        });
    }

    enhanceLoginForm(form, emailInput, passwordInput) {
        // Add ToolBox login button
        const tbButton = document.createElement('button');
        tbButton.type = 'button';
        tbButton.className = 'tb-auto-login-btn';
        tbButton.innerHTML = '🔐 ToolBox Login';
        tbButton.onclick = () => this.autoLogin(window.location.href);

        form.appendChild(tbButton);

        // Add password generator
        const genButton = document.createElement('button');
        genButton.type = 'button';
        genButton.className = 'tb-gen-password-btn';
        genButton.innerHTML = '🔑';
        genButton.onclick = () => this.generatePassword(passwordInput);

        passwordInput.parentNode.insertBefore(genButton, passwordInput.nextSibling);
    }

    async autoLogin(url) {
        try {
            const credentials = await this.passwordManager.getCredentials(url);
            if (credentials) {
                this.fillCredentials(credentials);
                this.showNotification('✅ Auto-filled credentials');
            } else {
                this.showNotification('❌ No credentials found for this site');
            }
        } catch (error) {
            this.showNotification('❌ Auto-login failed');
        }
    }

    fillCredentials(credentials) {
        const emailInput = document.querySelector('input[type="email"], input[name*="email"], input[name*="username"]');
        const passwordInput = document.querySelector('input[type="password"]');

        if (emailInput) emailInput.value = credentials.email;
        if (passwordInput) passwordInput.value = credentials.password;

        // Trigger change events
        [emailInput, passwordInput].forEach(input => {
            if (input) {
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    }

    async generatePassword(input) {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'generate_password',
                args: { length: 16, include_symbols: true }
            });

            if (response.success) {
                input.value = response.data.password;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                this.showNotification('🔑 Password generated');
            }
        } catch (error) {
            this.showNotification('❌ Password generation failed');
        }
    }

    async aiAnalyze() {
        try {
            const pageContent = {
                title: document.title,
                url: window.location.href,
                text: document.body.innerText.substring(0, 5000),
                forms: document.querySelectorAll('form').length,
                images: document.querySelectorAll('img').length
            };

            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'AIAnalyzer',
                function: 'analyze_page',
                args: pageContent
            });

            if (response.success) {
                this.showAnalysisResults(response.data);
            }
        } catch (error) {
            this.showNotification('❌ AI analysis failed');
        }
    }

    showAnalysisResults(analysis) {
        const modal = document.createElement('div');
        modal.className = 'tb-analysis-modal';
        modal.innerHTML = `
            <div class="tb-modal-content">
                <h3>🤖 AI Page Analysis</h3>
                <div class="tb-analysis-results">
                    <p><strong>Page Type:</strong> ${analysis.page_type}</p>
                    <p><strong>Sentiment:</strong> ${analysis.sentiment}</p>
                    <p><strong>Key Topics:</strong> ${analysis.topics.join(', ')}</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
                        ${analysis.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
                <button onclick="this.parentElement.parentElement.remove()">Close</button>
            </div>
        `;
        document.body.appendChild(modal);
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'tb-notification';
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => notification.remove(), 3000);
    }
}

// Gesture Detection
class GestureDetector {
    constructor() {
        this.isDrawing = false;
        this.path = [];
        this.circleCallback = null;
        this.setupListeners();
    }

    setupListeners() {
        document.addEventListener('mousedown', (e) => {
            if (e.button === 1) { // Middle mouse button
                this.startGesture(e);
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (this.isDrawing) {
                this.addPoint(e);
            }
        });

        document.addEventListener('mouseup', () => {
            if (this.isDrawing) {
                this.endGesture();
            }
        });
    }

    startGesture(e) {
        this.isDrawing = true;
        this.path = [{ x: e.clientX, y: e.clientY }];
        e.preventDefault();
    }

    addPoint(e) {
        this.path.push({ x: e.clientX, y: e.clientY });

        // Check for circle gesture
        if (this.path.length > 10 && this.isCircleGesture()) {
            this.triggerCircleGesture();
            this.endGesture();
        }
    }

    isCircleGesture() {
        if (this.path.length < 20) return false;

        const start = this.path[0];
        const end = this.path[this.path.length - 1];
        const distance = Math.sqrt(
            Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2)
        );

        // Circle if start and end are close
        return distance < 50;
    }

    triggerCircleGesture() {
        if (this.circleCallback) {
            this.circleCallback();
        }
    }

    onCircleGesture(callback) {
        this.circleCallback = callback;
    }

    endGesture() {
        this.isDrawing = false;
        this.path = [];
    }
}

// Voice Commands
class VoiceCommands {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.setupRecognition();
    }

    setupRecognition() {
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onresult = (event) => {
                const command = event.results[0][0].transcript.toLowerCase();
                this.processCommand(command);
            };

            this.recognition.onerror = () => {
                this.isListening = false;
            };

            this.recognition.onend = () => {
                this.isListening = false;
            };
        }
    }

    start() {
        if (this.recognition && !this.isListening) {
            this.isListening = true;
            this.recognition.start();
            this.showVoiceIndicator();
        }
    }

    processCommand(command) {
        const commands = {
            'toolbox panel': () => tbExtension.togglePanel(),
            'auto login': () => tbExtension.autoLogin(window.location.href),
            'generate password': () => this.generatePasswordVoice(),
            'analyze page': () => tbExtension.aiAnalyze(),
            'search': (text) => tbExtension.smartSearch.search(text)
        };

        for (const [trigger, action] of Object.entries(commands)) {
            if (command.includes(trigger)) {
                action(command.replace(trigger, '').trim());
                break;
            }
        }
    }

    showVoiceIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'tb-voice-indicator';
        indicator.innerHTML = '🎤 Listening...';
        document.body.appendChild(indicator);

        setTimeout(() => indicator.remove(), 3000);
    }
}

// Password Manager
class PasswordManager {
    async getCredentials(url) {
        const domain = new URL(url).hostname;
        const stored = await chrome.storage.sync.get(`credentials_${domain}`);
        return stored[`credentials_${domain}`];
    }

    async saveCredentials(url, credentials) {
        const domain = new URL(url).hostname;
        await chrome.storage.sync.set({
            [`credentials_${domain}`]: {
                ...credentials,
                saved_at: Date.now()
            }
        });
    }
}

// Smart Search
class SmartSearch {
    async search(query) {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'SmartSearch',
                function: 'search',
                args: {
                    query,
                    context: {
                        url: window.location.href,
                        title: document.title
                    }
                }
            });

            if (response.success) {
                this.showSearchResults(response.data);
            }
        } catch (error) {
            console.error('Smart search failed:', error);
        }
    }

    showSearchResults(results) {
        const modal = document.createElement('div');
        modal.className = 'tb-search-modal';
        modal.innerHTML = `
            <div class="tb-modal-content">
                <h3>🔍 Smart Search Results</h3>
                <div class="tb-search-results">
                    ${results.map(r => `
                        <div class="tb-search-result">
                            <h4><a href="${r.url}" target="_blank">${r.title}</a></h4>
                            <p>${r.description}</p>
                        </div>
                    `).join('')}
                </div>
                <button onclick="this.parentElement.parentElement.remove()">Close</button>
            </div>
        `;
        document.body.appendChild(modal);
    }
}

// ToolBox Panel
class TBPanel {
    constructor() {
        this.element = this.createElement();
        this.isVisible = false;
        this.plugins = new Map();
        this.loadPlugins();
    }

    createElement() {
        const panel = document.createElement('div');
        panel.className = 'tb-panel';
        panel.innerHTML = `
            <div class="tb-panel-header">
                <h3>🧰 ToolBox</h3>
                <button class="tb-close-btn">×</button>
            </div>
            <div class="tb-panel-content">
                <div class="tb-quick-actions">
                    <button class="tb-btn" data-action="auth">🔐 Auto Login</button>
                    <button class="tb-btn" data-action="password">🔑 Generate Password</button>
                    <button class="tb-btn" data-action="ai">🤖 AI Analyze</button>
                    <button class="tb-btn" data-action="search">🔍 Smart Search</button>
                </div>
                <div class="tb-plugins-container">
                    <h4>Plugins</h4>
                    <div class="tb-plugins-list"></div>
                </div>
                <div class="tb-analytics-toggle">
                    <label>
                        <input type="checkbox" id="tb-analytics">
                        📊 Analytics Dashboard
                    </label>
                </div>
            </div>
        `;

        this.setupEventListeners(panel);
        return panel;
    }

    setupEventListeners(panel) {
        // Close button
        panel.querySelector('.tb-close-btn').onclick = () => this.hide();

        // Quick actions
        panel.querySelectorAll('.tb-btn').forEach(btn => {
            btn.onclick = () => this.handleAction(btn.dataset.action);
        });

        // Analytics toggle
        panel.querySelector('#tb-analytics').onchange = (e) => {
            this.toggleAnalytics(e.target.checked);
        };
    }

    async loadPlugins() {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PluginManager',
                function: 'list_plugins'
            });

            if (response.success) {
                this.renderPlugins(response.data);
            }
        } catch (error) {
            console.error('Failed to load plugins:', error);
        }
    }

    renderPlugins(plugins) {
        const container = this.element.querySelector('.tb-plugins-list');
        container.innerHTML = plugins.map(plugin => `
            <div class="tb-plugin-item">
                <span>${plugin.icon} ${plugin.name}</span>
                <button onclick="tbExtension.panel.activatePlugin('${plugin.id}')">
                    Activate
                </button>
            </div>
        `).join('');
    }

    async activatePlugin(pluginId) {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PluginManager',
                function: 'activate_plugin',
                args: { plugin_id: pluginId }
            });

            if (response.success) {
                this.showNotification(`✅ Plugin ${pluginId} activated`);
            }
        } catch (error) {
            this.showNotification(`❌ Failed to activate plugin`);
        }
    }

    handleAction(action) {
        switch (action) {
            case 'auth':
                tbExtension.autoLogin(window.location.href);
                break;
            case 'password':
                this.generatePasswordAction();
                break;
            case 'ai':
                tbExtension.aiAnalyze();
                break;
            case 'search':
                this.showSearchInput();
                break;
        }
    }

    generatePasswordAction() {
        const passwordInput = document.querySelector('input[type="password"]');
        if (passwordInput) {
            tbExtension.generatePassword(passwordInput);
        } else {
            this.showNotification('❌ No password field found');
        }
    }

    showSearchInput() {
        const input = prompt('Enter search query:');
        if (input) {
            tbExtension.smartSearch.search(input);
        }
    }

    toggleAnalytics(enabled) {
        if (enabled) {
            this.showAnalyticsDashboard();
        } else {
            this.hideAnalyticsDashboard();
        }
    }

    showAnalyticsDashboard() {
        // Create analytics dashboard
        const dashboard = document.createElement('div');
        dashboard.className = 'tb-analytics-dashboard';
        dashboard.innerHTML = `
            <div class="tb-analytics-content">
                <h4>📊 Analytics Dashboard</h4>
                <div class="tb-stats">
                    <div class="tb-stat">
                        <span class="tb-stat-label">Page Views Today</span>
                        <span class="tb-stat-value" id="tb-pageviews">-</span>
                    </div>
                    <div class="tb-stat">
                        <span class="tb-stat-label">Time on Site</span>
                        <span class="tb-stat-value" id="tb-time">-</span>
                    </div>
                    <div class="tb-stat">
                        <span class="tb-stat-label">Actions Performed</span>
                        <span class="tb-stat-value" id="tb-actions">-</span>
                    </div>
                </div>
            </div>
        `;

        this.element.querySelector('.tb-panel-content').appendChild(dashboard);
        this.loadAnalyticsData();
    }

    async loadAnalyticsData() {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'Analytics',
                function: 'get_stats'
            });

            if (response.success) {
                const stats = response.data;
                document.getElementById('tb-pageviews').textContent = stats.pageviews;
                document.getElementById('tb-time').textContent = stats.time_on_site;
                document.getElementById('tb-actions').textContent = stats.actions;
            }
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }

    hideAnalyticsDashboard() {
        const dashboard = this.element.querySelector('.tb-analytics-dashboard');
        if (dashboard) {
            dashboard.remove();
        }
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    show() {
        this.element.style.display = 'block';
        this.isVisible = true;
    }

    hide() {
        this.element.style.display = 'none';
        this.isVisible = false;
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'tb-panel-notification';
        notification.textContent = message;
        this.element.appendChild(notification);

        setTimeout(() => notification.remove(), 3000);
    }
}

// Initialize extension when DOM is ready
function initializeExtension() {
    try {
        const tbExtension = new TBExtensionContent();
        window.tbExtension = tbExtension; // Make it globally accessible
    } catch (error) {
        console.error('Failed to initialize ToolBox extension:', error);
        // Retry after a short delay
        setTimeout(initializeExtension, 1000);
    }
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeExtension);
} else {
    // DOM is already ready
    initializeExtension();
}
