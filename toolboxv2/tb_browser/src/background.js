// ToolBox Pro - Background Service Worker
// Handles API communication, gesture events, and extension coordination

class ToolBoxBackground {
    constructor() {
        this.apiBase = 'http://localhost:8080';
        this.isConnected = false;
        this.activeTab = null;
        this.gestureHistory = [];

        this.init();
    }

    init() {
        console.log('🚀 ToolBox Pro background service worker starting...');

         this.loadBackendSettings().then(() => {
            // Setup event listeners
            this.setupEventListeners();

            // Check API connection
            this.checkConnection();

            // Setup periodic tasks
            this.setupPeriodicTasks();


            console.log('✅ ToolBox Pro background service worker initialized');
        });
    }

    async loadBackendSettings() {
    try {
        const stored = await chrome.storage.sync.get(['toolboxSettings']);
        if (stored.toolboxSettings) {
            const settings = stored.toolboxSettings;

            // Backend-URL setzen
            switch (settings.backend) {
                case 'local':
                    this.apiBase = 'http://localhost:8080';
                    break;
                case 'remote':
                    this.apiBase = 'https://simplecore.app';
                    break;
                case 'custom':
                    this.apiBase = settings.customBackendUrl || 'http://localhost:8080';
                    break;
            }

            // Auth-Daten übernehmen
            this.authData = {
                username: settings.username || null,
                jwt: settings.jwt || null,
                isAuthenticated: settings.isAuthenticated || false
            };

            console.log('Backend configured:', this.apiBase);

            // Auto-login if we have credentials
            if (this.authData.username && !this.authData.isAuthenticated) {
                console.log('Attempting auto-login for:', this.authData.username);
                await this.attemptReAuth();
            }
        }
    } catch (error) {
        console.warn('Failed to load backend settings:', error);
    }
}

    setupEventListeners() {
        // Extension installation
        chrome.runtime.onInstalled.addListener((details) => {
            this.handleInstallation(details);
        });

        // Hinzufügen des Action-Listeners für das Side Panel
        chrome.action.onClicked.addListener((tab) => {
            chrome.sidePanel.open({ windowId: tab.windowId });
        });

        // Tab updates
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdate(tabId, changeInfo, tab);
        });

        // Tab activation
        chrome.tabs.onActivated.addListener((activeInfo) => {
            this.handleTabActivation(activeInfo);
        });

        // Message handling
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });

        // Command handling (keyboard shortcuts)
        chrome.commands.onCommand.addListener((command) => {
            this.handleCommand(command);
        });

        // Web navigation events
        chrome.webNavigation.onCompleted.addListener((details) => {
            this.handleNavigationCompleted(details);
            this.checkForSavedCredentials(details.tabId);
        });

        chrome.notifications.onButtonClicked.addListener((notificationId, buttonIndex) => {
            if (notificationId.startsWith('save-password-')) {
                const credentials = JSON.parse(notificationId.replace('save-password-', ''));
                if (buttonIndex === 0) { // "Speichern"-Button
                    this.makeAPICall('/api/PasswordManager/add_password', 'POST', credentials);
                }
                chrome.notifications.clear(notificationId);
            }
        });

    }

    async checkForSavedCredentials(tabId) {
        const result = await chrome.storage.session.get('potential_credentials_to_save');
        const credentials = result.potential_credentials_to_save;

        if (credentials) {
            // Lösche die temporären Daten
            await chrome.storage.session.remove('potential_credentials_to_save');

            // Prüfe, ob diese Anmeldeinformationen bereits existieren
            const checkResult = await this.makeAPICall('/api/PasswordManager/get_password_by_url_username', 'POST', {
                url: credentials.url,
                username: credentials.username
            });

            // Zeige die Benachrichtigung nur an, wenn die Daten noch nicht existieren
            if (checkResult && checkResult.status === 'error') {
                 chrome.notifications.create(`save-password-${JSON.stringify(credentials)}`, {
                    type: 'basic',
                    iconUrl: 'icons/tb48.png',
                    title: 'Passwort speichern?',
                    message: `Möchten Sie das Passwort für ${credentials.username} auf dieser Seite speichern?`,
                    buttons: [{ title: 'Speichern' }, { title: 'Niemals' }]
                });
            }
        }
    }

    async handleInstallation(details) {
        console.log('📦 Extension installed/updated:', details.reason);

        // Initialize storage
        await this.initializeStorage();

        // Extension installed successfully
        if (details.reason === 'install') {
            console.log('🎉 ToolBox Pro installed successfully!');
        }
    }
    async initializeStorage() {
        const defaultSettings = {
            gestureSettings: {
                enabled: true,
                sensitivity: 1.0,
                minSwipeDistance: 100,
                enableMouse: true,
                enableTouch: true
            },
            voiceSettings: {
                enabled: true,
                language: 'en-US',
                autoSpeak: false,
                wakeWords: ['toolbox', 'isaa']
            },
            passwordSettings: {
                autoFill: true,
                generateLength: 16,
                includeSymbols: true
            }
        };

        const stored = await chrome.storage.sync.get(Object.keys(defaultSettings));

        // Set defaults for missing settings
        const toSet = {};
        for (const [key, value] of Object.entries(defaultSettings)) {
            if (!stored[key]) {
                toSet[key] = value;
            }
        }

        if (Object.keys(toSet).length > 0) {
            await chrome.storage.sync.set(toSet);
        }
    }

    setupPeriodicTasks() {
        // Check API connection every 30 seconds
        setInterval(() => {
            this.checkConnection();
        }, 30000);

        // Clean up gesture history every 5 minutes
        setInterval(() => {
            this.cleanupGestureHistory();
        }, 300000);
    }

    async checkConnection() {
        try {
            // Verwende jetzt das konfigurierte Backend
            const response = await fetch(`${this.apiBase}/api/CloudM/openVersion`);
            this.isConnected = response.ok;

            chrome.action.setBadgeText({
                text: this.isConnected ? '' : '!'
            });

            chrome.action.setBadgeBackgroundColor({
                color: this.isConnected ? '#4CAF50' : '#F44336'
            });

        } catch (error) {
            this.isConnected = false;
            chrome.action.setBadgeText({ text: '!' });
            chrome.action.setBadgeBackgroundColor({ color: '#F44336' });
        }
    }

    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'API_REQUEST':
                    const result = await this.makeAPICall(
                        message.data.endpoint,
                        message.data.method,
                        message.data.body
                    );
                    sendResponse({ success: true, data: result });
                    break;
                case 'MANUAL_SAVE_PASSWORD':
                    this.makeAPICall('/api/PasswordManager/add_password', 'POST', message.credentials)
                        .then(() => sendResponse({ success: true }))
                        .catch(err => sendResponse({ success: false, error: err.message }));
                    break;

                case 'CHECK_SAVED_CREDENTIALS':
                    this.checkSavedCredentials(message.url)
                        .then(hasCredentials => sendResponse({ hasCredentials }))
                        .catch(err => sendResponse({ hasCredentials: false, error: err.message }));
                    break;

                case 'AUTOFILL_FROM_INDICATOR':
                    this.getPasswordForAutofill(message.url)
                        .then(data => sendResponse({ success: true, data }))
                        .catch(err => sendResponse({ success: false, error: err.message }));
                    break;

                case 'POTENTIAL_CREDENTIALS_DETECTED':
                // Speichere die Daten im Session Storage, da der background script die Berechtigung hat
                    chrome.storage.session.set({ 'potential_credentials_to_save': message.credentials })
                        .then(() => sendResponse({ success: true }))
                        .catch(err => sendResponse({ success: false, error: err.message }));
                    break;

                case 'GESTURE_DETECTED':
                    this.handleGesture(message.gesture, sender.tab);
                    sendResponse({ success: true });
                    break;

                case 'OPEN_POPUP':
                     if (sender.tab) {
                        await chrome.sidePanel.open({ windowId: sender.tab.windowId });
                    }
                    sendResponse({ success: true });
                    break;

                case 'GET_TAB_INFO':
                    const tabInfo = await this.getTabInfo(sender.tab.id);
                    sendResponse({ success: true, data: tabInfo });
                    break;
                case 'RELOAD_SETTINGS':
                    await this.loadBackendSettings();
                    sendResponse({ success: true, apiBase: this.apiBase });
                    break;

                case 'GET_AUTH_STATUS':
                    sendResponse({
                        success: true,
                        isAuthenticated: this.authData.isAuthenticated,
                        username: this.authData.username
                    });
                    break;

                case 'START_WEB_LOGIN':
                    this.startWebLogin(message.backend)
                        .then(result => sendResponse(result))
                        .catch(err => sendResponse({ success: false, error: err.message }));
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Background message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    async handleCommand(command) {
        console.log('⌨️ Command received:', command);

        const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });

        switch (command) {
            case 'toggle-toolbox':
                if (activeTab) {
                    await chrome.sidePanel.open({ windowId: activeTab.windowId });
                }
                break;

            case 'voice-command':
                await this.activateVoiceCommand(activeTab);
                break;

            case 'quick-search':
                await this.activateQuickSearch(activeTab);
                break;

            case 'password-autofill':
                await this.triggerPasswordAutofill(activeTab);
                break;
        }
    }

    handleTabUpdate(tabId, changeInfo, tab) {
        if (changeInfo.status === 'complete' && tab.url) {
            this.activeTab = tab;

            // Inject content script if needed
            this.ensureContentScript(tabId);
        }
    }

    handleTabActivation(activeInfo) {
        chrome.tabs.get(activeInfo.tabId, (tab) => {
            this.activeTab = tab;
        });
    }

    handleNavigationCompleted(details) {
        if (details.frameId === 0) { // Main frame only
            // Ensure content script is injected
            this.ensureContentScript(details.tabId);
        }
    }

    async ensureContentScript(tabId) {
        try {
            // Check if content script is already injected
            const response = await chrome.tabs.sendMessage(tabId, { type: 'PING' });
        } catch (error) {
            // Content script not present, inject it
            try {
                await chrome.scripting.executeScript({
                    target: { tabId: tabId },
                    files: ['src/content.js']
                });

                await chrome.scripting.insertCSS({
                    target: { tabId: tabId },
                    files: ['src/content.css']
                });
            } catch (injectionError) {
                console.warn('Failed to inject content script:', injectionError);
            }
        }
    }

    handleGesture(gesture, tab) {
        console.log('🎯 Gesture detected:', gesture, 'on tab:', tab.id);

        // Record gesture
        this.gestureHistory.push({
            gesture,
            tabId: tab.id,
            url: tab.url,
            timestamp: Date.now()
        });

        // Provide feedback
        this.showGestureFeedback(gesture, tab.id);
    }

    async showGestureFeedback(gesture, tabId) {
        const feedbackMessages = {
            'swipe-left': '← Back',
            'swipe-right': '→ Forward',
            'swipe-up': '↑ Scroll Up',
            'swipe-down': '↓ Scroll Down'
        };

        const message = feedbackMessages[gesture] || gesture;

        try {
            await chrome.tabs.sendMessage(tabId, {
                type: 'SHOW_GESTURE_FEEDBACK',
                message: message
            });
        } catch (error) {
            console.warn('Failed to show gesture feedback:', error);
        }
    }

        /* Veraltet: Diese Funktion wird nicht mehr benötigt
    async openPopup(position) {
        // Open extension popup
        chrome.action.openPopup();
    }
    */


    async activateVoiceCommand(tab) {
        try {
            await chrome.tabs.sendMessage(tab.id, {
                type: 'ACTIVATE_VOICE'
            });
        } catch (error) {
            console.warn('Failed to activate voice command:', error);
        }
    }

    async activateQuickSearch(tab) {
        chrome.sidePanel.open({ windowId: tab.windowId });
        // The popup will handle switching to search tab
    }

    async checkSavedCredentials(url) {
        try {
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', {
                url: url
            });

            // Check if we got valid data back
            if (response && response.result && response.result.data) {
                return true;
            } else if (response && response.data) {
                return true;
            }

            return false;
        } catch (error) {
            console.warn('Failed to check saved credentials:', error);
            return false;
        }
    }

    async getPasswordForAutofill(url) {
        try {
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', {
                url: url
            });

            if (response && response.result && response.result.data) {
                return response.result.data;
            } else if (response && response.data) {
                return response.data;
            }

            throw new Error('No password data found');
        } catch (error) {
            console.error('Failed to get password for autofill:', error);
            throw error;
        }
    }

    async triggerPasswordAutofill(tab) {
        try {
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', {
                url: tab.url
            });

            if (response.data) {
                await chrome.tabs.sendMessage(tab.id, {
                    type: 'AUTOFILL_PASSWORD',
                    data: response.data
                });
            } else {
                this.showNotification(
                    'No Password Found',
                    'No saved password found for this website.'
                );
            }
        } catch (error) {
            console.error('Password autofill error:', error);
        }
    }

    async getTabInfo(tabId) {
        const tab = await chrome.tabs.get(tabId);
        return {
            id: tab.id,
            url: tab.url,
            title: tab.title,
            favIconUrl: tab.favIconUrl
        };
    }

    async makeAPICall(endpoint, method = 'POST', data = null, includeAuth = true) {
        // Stelle sicher, dass Settings geladen sind
        if (!this.apiBase || this.apiBase === 'http://localhost:8080') {
            await this.loadBackendSettings();
        }

        const url = `${this.apiBase}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
        };

        // Auth-Header hinzufügen, wenn verfügbar und gewünscht
        if (includeAuth && this.authData.isAuthenticated) {
            if (this.authData.jwt) {
                headers['Authorization'] = `Bearer ${this.authData.jwt}`;
            }
            // Username als Custom Header (falls dein Backend das erwartet)
            if (this.authData.username) {
                headers['X-Username'] = this.authData.username;
            }
        }

        const options = {
            method,
            headers,
            credentials: 'include' // Für Session-Cookies
        };

        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);

            // 401 = Session abgelaufen
            if (response.status === 401 && includeAuth) {
                console.warn('Session expired, attempting re-auth...');
                const reAuthSuccess = await this.attemptReAuth();

                if (reAuthSuccess) {
                    // Retry mit neuem Token
                    return this.makeAPICall(endpoint, method, data, includeAuth);
                } else {
                    throw new Error('Session expired and re-authentication failed');
                }
            }

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API call to ${endpoint} failed:`, error);
            throw error;
        }
    }

    async attemptReAuth() {
    if (!this.authData.username) {
        console.warn('Cannot re-auth without username');
        return false;
    }

    try {
        const payload = {
            Username: this.authData.username
        };

        if (this.authData.jwt) {
            payload.Jwt_claim = this.authData.jwt;
        }

        const response = await this.makeAPICall(
            '/validateSession',
            'POST',
            payload,
            false // Keine Auth-Header beim Re-Auth
        );

        if (response.result?.data_info === 'Valid Session') {
            this.authData.isAuthenticated = true;

            // Settings aktualisieren
            const stored = await chrome.storage.sync.get(['toolboxSettings']);
            if (stored.toolboxSettings) {
                stored.toolboxSettings.isAuthenticated = true;
                await chrome.storage.sync.set({ toolboxSettings: stored.toolboxSettings });
            }

            console.log('Re-authentication successful');
            return true;
        }
    } catch (error) {
        console.error('Re-authentication failed:', error);
    }

    // Bei Fehlschlag: Logout
    this.authData.isAuthenticated = false;
    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    if (stored.toolboxSettings) {
        stored.toolboxSettings.isAuthenticated = false;
        await chrome.storage.sync.set({ toolboxSettings: stored.toolboxSettings });
    }

    return false;
}

    async startWebLogin(backend = null) {
        console.log('🔐 Starting web login flow...');

        // Determine backend URL
        let loginBase = this.apiBase;
        if (backend === 'local') {
            loginBase = 'http://localhost:8080';
        } else if (backend === 'remote') {
            loginBase = 'https://simplecore.app';
        }

        // Generate unique session ID for this login attempt
        const sessionId = this.generateSessionId();

        // Create login URL using open_web_login_web
        const loginUrl = `${loginBase}/api/CloudM/open_web_login_web?session_id=${sessionId}&return_to=browser&log_in_for=Browser`;

        console.log('Opening login page:', loginUrl);

        // Open login page in new tab
        await chrome.tabs.create({ url: loginUrl, active: true });

        // Start polling for authentication completion
        return this.pollForAuthCompletion(loginBase, sessionId);
    }

    generateSessionId() {
        // Generate a random session ID (similar to uuid.uuid4().hex in Python)
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        }).replace(/-/g, '');
    }

    async pollForAuthCompletion(baseUrl, sessionId, timeout = 300000) {
        console.log('⏳ Polling for authentication completion...');

        const startTime = Date.now();
        const pollInterval = 2000; // Poll every 2 seconds

        while (Date.now() - startTime < timeout) {
            try {
                // Check if authentication is complete
                const response = await fetch(`${baseUrl}/api/CloudM/open_check_cli_auth`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ session_id: sessionId })
                });

                if (!response.ok) {
                    console.warn('Auth check failed:', response.status);
                    await this.sleep(pollInterval);
                    continue;
                }

                const result = await response.json();

                if (result.authenticated || (result.result && result.result.authenticated)) {
                    const authData = result.result || result;
                    const jwtToken = authData.jwt_token;
                    const username = authData.username;

                    if (jwtToken && username) {
                        console.log('✅ Authentication successful!');

                        // Store credentials
                        await this.storeAuthCredentials(username, jwtToken, baseUrl);

                        // Show success notification
                        this.showNotification(
                            'Login Successful',
                            `Welcome back, ${username}!`
                        );

                        return { success: true, username, message: 'Login successful' };
                    }
                }

            } catch (error) {
                console.warn('Polling error:', error);
            }

            // Wait before next poll
            await this.sleep(pollInterval);
        }

        // Timeout reached
        console.error('❌ Authentication timeout');
        this.showNotification(
            'Login Timeout',
            'Authentication window expired. Please try again.'
        );

        return { success: false, error: 'Authentication timeout' };
    }

    async storeAuthCredentials(username, jwtToken, baseUrl) {
        // Update internal auth data
        this.authData = {
            username: username,
            jwt: jwtToken,
            isAuthenticated: true
        };

        // Determine backend type from URL
        let backend = 'local';
        if (baseUrl.includes('simplecore.app')) {
            backend = 'remote';
        } else if (baseUrl !== 'http://localhost:8080') {
            backend = 'custom';
        }

        // Update settings in storage
        const stored = await chrome.storage.sync.get(['toolboxSettings']);
        const settings = stored.toolboxSettings || {};

        settings.username = username;
        settings.jwt = jwtToken;
        settings.isAuthenticated = true;
        settings.backend = backend;
        if (backend === 'custom') {
            settings.customBackendUrl = baseUrl;
        }

        await chrome.storage.sync.set({ toolboxSettings: settings });

        // Update apiBase
        this.apiBase = baseUrl;

        console.log('✅ Credentials stored successfully');
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    showNotification(title, message) {
        try {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/tb48.png',
                title: title,
                message: message
            });
        } catch (error) {
            console.log('Notification:', title, '-', message);
        }
    }

    cleanupGestureHistory() {
        const oneHourAgo = Date.now() - (60 * 60 * 1000);
        this.gestureHistory = this.gestureHistory.filter(
            gesture => gesture.timestamp > oneHourAgo
        );
    }
}

// Initialize background service worker
const toolboxBackground = new ToolBoxBackground();
