// ToolBox Pro - Background Service Worker
// Handles API communication, gesture events, and extension coordination
import { AgentView } from './agent_view.js';
import { PromptEngine } from './prompts/prompt_engine.js';

class ToolBoxBackground {
    constructor() {
        this.apiBase = 'http://localhost:8080';
        this.isConnected = false;
        this.activeTab = null;
        this.gestureHistory = [];
        this.useNative = false;
        this.authData = { username: null, jwt: null, isAuthenticated: false };
        this.promptRules = [];
        this.promptLibrary = { prompts: {}, category_map: {} };
        this.promptEngine = new PromptEngine();
        this.agentViews = {}; // tabId → AgentView state

        this.init();
    }

    init() {
        console.log('🚀 ToolBox Pro background service worker starting...');
        setTimeout(async () => {
            await this.promptEngine.init();
        },1)
        this.loadBackendSettings().then(() => {
            this.setupEventListeners();
            this.checkConnection();
            this.setupPeriodicTasks();
            this.loadPromptLibrary();
            console.log('✅ ToolBox Pro background service worker initialized');
        });
    }

    async loadBackendSettings() {
        try {
            const stored = await chrome.storage.sync.get(['toolboxSettings']);
            if (stored.toolboxSettings) {
                const settings = stored.toolboxSettings;

                switch (settings.backend) {
                    case 'local':
                        this.apiBase = 'http://localhost:8080';
                        this.useNative = false;
                        break;
                    case 'tauri':
                        // Tauri worker läuft auf Port 5000 (kein dist, kein web-login)
                        this.apiBase = 'http://localhost:5000';
                        this.useNative = false;
                        break;
                    case 'native':
                        // Kein HTTP-Server -- direkt via Chrome Native Messaging
                        this.apiBase = null;
                        this.useNative = true;
                        break;
                    case 'remote':
                        this.apiBase = 'https://simplecore.app';
                        this.useNative = false;
                        break;
                    case 'custom':
                        this.apiBase = settings.customBackendUrl || 'http://localhost:8080';
                        this.useNative = false;
                        break;
                    default:
                        this.apiBase = 'http://localhost:8080';
                        this.useNative = false;
                }

                this.authData = {
                    username: settings.username || null,
                    jwt: settings.jwt || null,
                    isAuthenticated: settings.isAuthenticated || false
                };

                console.log('Backend configured:', this.useNative ? 'native' : this.apiBase);

                if (this.authData.username && !this.authData.isAuthenticated) {
                    await this.attemptReAuth();
                }
            }
        } catch (error) {
            console.warn('Failed to load backend settings:', error);
        }
    }

    // ─── Native Messaging ────────────────────────────────────────────────────

    async makeNativeCall(action, payload) {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendNativeMessage(
                'com.toolbox.native',
                { action, payload: payload || {} },
                (response) => {
                    if (chrome.runtime.lastError) {
                        reject(new Error(chrome.runtime.lastError.message));
                    } else {
                        resolve(response);
                    }
                }
            );
        });
    }

    // ─── Tauri Worker Auto-Detection ─────────────────────────────────────────

    async detectTauriWorker() {
        // Prüft ob ein Tauri-Worker auf Port 5000 läuft
        try {
            const resp = await fetch('http://localhost:5000/health', { signal: AbortSignal.timeout(800) });
            return resp.ok;
        } catch {
            return false;
        }
    }

    // ─── Session Sync: Native → HTTP ─────────────────────────────────────────
    // Wenn Tauri-Backend: CLI-Session via Native Host lesen, dann als Bearer an den
    // lokalen HTTP-Worker weitergeben (kein OAuth-Redirect nötig).

    async syncNativeSessionToTauri() {
        try {
            const nativeResp = await this.makeNativeCall('validate_session', {});
            if (!nativeResp.authenticated) return false;

            // JWT aus native session holen
            const jwtResp = await this.makeNativeCall('get_session_jwt', {});
            if (!jwtResp.success || !jwtResp.jwt) return false;

            this.authData.jwt = jwtResp.jwt;
            this.authData.username = nativeResp.username;
            this.authData.isAuthenticated = true;

            const stored = await chrome.storage.sync.get(['toolboxSettings']);
            const s = stored.toolboxSettings || {};
            s.jwt = jwtResp.jwt;
            s.username = nativeResp.username;
            s.isAuthenticated = true;
            await chrome.storage.sync.set({ toolboxSettings: s });
            console.log('✅ Native session synced to Tauri backend');
            return true;
        } catch (e) {
            console.warn('Native→Tauri session sync failed:', e.message);
            return false;
        }
    }

    // ─── Setup ───────────────────────────────────────────────────────────────

    setupEventListeners() {
        chrome.runtime.onInstalled.addListener((details) => this.handleInstallation(details));

        chrome.action.onClicked.addListener((tab) => {
            chrome.sidePanel.open({ windowId: tab.windowId });
        });

        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdate(tabId, changeInfo, tab);
        });

        chrome.tabs.onActivated.addListener((activeInfo) => this.handleTabActivation(activeInfo));

        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true;
        });

        chrome.commands.onCommand.addListener((command) => this.handleCommand(command));

        chrome.webNavigation.onCompleted.addListener((details) => {
            this.handleNavigationCompleted(details);
            this.checkForSavedCredentials(details.tabId);
        });

        chrome.notifications.onButtonClicked.addListener((notificationId, buttonIndex) => {
            if (notificationId.startsWith('save-password-')) {
                const credentials = JSON.parse(notificationId.replace('save-password-', ''));
                if (buttonIndex === 0) {
                    this.makeAPICall('/api/PasswordManager/add_password', 'POST', credentials);
                }
                chrome.notifications.clear(notificationId);
            }
        });
    }

    async checkForSavedCredentials(tabId) {
        const result = await chrome.storage.session.get('potential_credentials_to_save');
        const credentials = result.potential_credentials_to_save;
        if (!credentials) return;

        await chrome.storage.session.remove('potential_credentials_to_save');

        const checkResult = await this.makeAPICall(
            '/api/PasswordManager/get_password_by_url_username', 'POST',
            { url: credentials.url, username: credentials.username }
        );

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

    async handleInstallation(details) {
        await this.initializeStorage();
        if (details.reason === 'install') console.log('🎉 ToolBox Pro installed!');
    }

    async initializeStorage() {
        const defaultSettings = {
            gestureSettings: { enabled: true, sensitivity: 1.0, minSwipeDistance: 100, enableMouse: true, enableTouch: true },
            voiceSettings: { enabled: true, language: 'en-US', autoSpeak: false, wakeWords: ['toolbox', 'isaa'] },
            passwordSettings: { autoFill: true, generateLength: 16, includeSymbols: true }
        };
        const stored = await chrome.storage.sync.get(Object.keys(defaultSettings));
        const toSet = {};
        for (const [key, value] of Object.entries(defaultSettings)) {
            if (!stored[key]) toSet[key] = value;
        }
        if (Object.keys(toSet).length > 0) await chrome.storage.sync.set(toSet);
    }

    setupPeriodicTasks() {
        setInterval(() => this.checkConnection(), 30000);
        setInterval(() => this.cleanupGestureHistory(), 300000);
        setInterval(() => this.syncPromptLibrary(), 6 * 60 * 60 * 1000);
    }
    async checkConnection() {
        try {
            if (this.useNative) {
                const resp = await this.makeNativeCall('ping', {});
                this.isConnected = resp && resp.success;
            } else {
                const response = await fetch(`${this.apiBase}/health`, { signal: AbortSignal.timeout(2000) });
                this.isConnected = response.ok;
                // Tauri-Backend ohne Session: einmalig CLI-Session sync versuchen
                if (this.isConnected && !this.authData.isAuthenticated) {
                    const stored = await chrome.storage.sync.get(['toolboxSettings']);
                    if (stored.toolboxSettings?.backend === 'tauri') {
                        await this.syncNativeSessionToTauri();
                    }
                }
            }

            chrome.action.setBadgeText({ text: this.isConnected ? '' : '!' });
            chrome.action.setBadgeBackgroundColor({ color: this.isConnected ? '#4CAF50' : '#F44336' });
        } catch {
            this.isConnected = false;
            chrome.action.setBadgeText({ text: '!' });
            chrome.action.setBadgeBackgroundColor({ color: '#F44336' });
        }
    }

    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'API_REQUEST': {
                    const result = await this.makeAPICall(
                        message.data.endpoint,
                        message.data.method,
                        message.data.body
                    );
                    sendResponse({ success: true, data: result });
                    break;
                }
                case 'NATIVE_CALL': {
                    const result = await this.makeNativeCall(message.action, message.payload);
                    sendResponse({ success: true, data: result });
                    break;
                }
                case 'MANUAL_SAVE_PASSWORD':
                    this.makeAPICall('/api/PasswordManager/add_password', 'POST', message.credentials)
                        .then(() => sendResponse({ success: true }))
                        .catch(err => sendResponse({ success: false, error: err.message }));
                    break;
                case 'CHECK_SAVED_CREDENTIALS':
                    this.checkSavedCredentials(message.url)
                        .then(hasCredentials => sendResponse({ hasCredentials }))
                        .catch(() => sendResponse({ hasCredentials: false }));
                    break;
                case 'AUTOFILL_FROM_INDICATOR':
                    this.getPasswordForAutofill(message.url)
                        .then(data => sendResponse({ success: true, data }))
                        .catch(err => sendResponse({ success: false, error: err.message }));
                    break;
                case 'POTENTIAL_CREDENTIALS_DETECTED':
                    chrome.storage.session.set({ 'potential_credentials_to_save': message.credentials })
                        .then(() => sendResponse({ success: true }))
                        .catch(err => sendResponse({ success: false, error: err.message }));
                    break;
                case 'GESTURE_DETECTED':
                    this.handleGesture(message.gesture, sender.tab);
                    sendResponse({ success: true });
                    break;
                case 'OPEN_POPUP':
                    if (sender.tab) await chrome.sidePanel.open({ windowId: sender.tab.windowId });
                    sendResponse({ success: true });
                    break;
                case 'GET_TAB_INFO': {
                    const tabInfo = await this.getTabInfo(sender.tab.id);
                    sendResponse({ success: true, data: tabInfo });
                    break;
                }
                case 'RELOAD_SETTINGS':
                    await this.loadBackendSettings();
                    sendResponse({ success: true, apiBase: this.apiBase, useNative: this.useNative });
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
                case 'DETECT_TAURI':
                    this.detectTauriWorker()
                        .then(found => sendResponse({ found }));
                    break;

                // ── Prompt Library ──────────────────────────────────────────
                case 'MATCH_SITE': {
                    const rule = this.matchSiteRule(message.url);
                    sendResponse(rule || null);
                    break;
                }
                case 'GET_PROMPTS_FOR_SITE': {
                    const rule = this.matchSiteRule(message.url);
                    if (!rule) { sendResponse(null); break; }
                    const cat = rule.category || '_default';
                    const ids = this.promptLibrary.category_map?.[cat]
                        || this.promptLibrary.category_map?.['_default'] || [];
                    const prompts = ids.map(id => this.promptLibrary.prompts?.[id]).filter(Boolean)
                        .sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
                    const preprompts = (rule.preprompts || [])
                        .map(id => this.promptLibrary.prompts?.[id]).filter(Boolean);
                    sendResponse({ prompts, preprompts, rule });
                    break;
                }
                case 'BUILD_INJECT': {
                    const rule = this.matchSiteRule(message.url);
                    const selected = this.promptLibrary.prompts?.[message.promptId];
                    if (!selected) { sendResponse(null); break; }
                    const preprompts = (rule?.preprompts || [])
                        .map(id => this.promptLibrary.prompts?.[id])
                        .filter(p => p?.auto_prepend);
                    const parts = [...preprompts.map(p => p.content), selected.content];
                    sendResponse(parts.join('\n\n'));
                    break;
                }
                case 'AI_SITE_DETECTED':
                    // Update badge to show AI site detected
                    chrome.action.setBadgeText({ text: '✓', tabId: sender.tab?.id });
                    chrome.action.setBadgeBackgroundColor({ color: '#4a4a8f', tabId: sender.tab?.id });
                    sendResponse({ success: true });
                    break;
                case 'PROMPT_SYNC':
                    this.syncPromptLibrary()
                        .then(ok => sendResponse({ success: ok }));
                    break;

                // ── Agent View (CDP) ────────────────────────────────────────
                case 'AGENT_VIEW': {
                    const tabId = message.tabId || sender.tab?.id;
                    if (!tabId) { sendResponse({ error: 'no tabId' }); break; }
                    this.handleAgentView(tabId, message.action, message)
                        .then(result => sendResponse(result))
                        .catch(err => sendResponse({ error: err.message }));
                    break;
                }
                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Background message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    async handleCommand(command) {
        const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
        switch (command) {
            case 'toggle-toolbox':
                if (activeTab) await chrome.sidePanel.open({ windowId: activeTab.windowId });
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
            case 'prompt-inject':
                if (activeTab) {
                    chrome.tabs.sendMessage(activeTab.id, { type: 'TB_TOGGLE_PANEL' });
                }
                break;
        }
    }

    handleTabUpdate(tabId, changeInfo, tab) {
        if (changeInfo.status === 'complete' && tab.url) {
            this.activeTab = tab;
            this.ensureContentScript(tabId);
        }
    }

    handleTabActivation(activeInfo) {
        chrome.tabs.get(activeInfo.tabId, (tab) => { this.activeTab = tab; });
    }

    handleNavigationCompleted(details) {
        if (details.frameId === 0) this.ensureContentScript(details.tabId);
    }

    async ensureContentScript(tabId) {
        try {
            await chrome.tabs.sendMessage(tabId, { type: 'PING' });
        } catch {
            try {
                await chrome.scripting.executeScript({ target: { tabId }, files: ['src/content.js'] });
                await chrome.scripting.insertCSS({ target: { tabId }, files: ['src/content.css'] });
            } catch (e) {
                console.warn('Failed to inject content script:', e);
            }
        }
    }

    handleGesture(gesture, tab) {
        this.gestureHistory.push({ gesture, tabId: tab.id, url: tab.url, timestamp: Date.now() });
        this.showGestureFeedback(gesture, tab.id);
    }

    async showGestureFeedback(gesture, tabId) {
        const msgs = { 'swipe-left': '← Back', 'swipe-right': '→ Forward', 'swipe-up': '↑ Scroll Up', 'swipe-down': '↓ Scroll Down' };
        try {
            await chrome.tabs.sendMessage(tabId, { type: 'SHOW_GESTURE_FEEDBACK', message: msgs[gesture] || gesture });
        } catch {}
    }

    async activateVoiceCommand(tab) {
        try { await chrome.tabs.sendMessage(tab.id, { type: 'ACTIVATE_VOICE' }); } catch {}
    }

    async activateQuickSearch(tab) {
        chrome.sidePanel.open({ windowId: tab.windowId });
    }

    async checkSavedCredentials(url) {
        try {
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', { url });
            return !!(response?.result?.data || response?.data);
        } catch {
            return false;
        }
    }

    async getPasswordForAutofill(url) {
        const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', { url });
        if (response?.result?.data) return response.result.data;
        if (response?.data) return response.data;
        throw new Error('No password data found');
    }

    async triggerPasswordAutofill(tab) {
        try {
            const response = await this.makeAPICall('/api/PasswordManager/get_password_for_autofill', 'POST', { url: tab.url });
            if (response.data) {
                await chrome.tabs.sendMessage(tab.id, { type: 'AUTOFILL_PASSWORD', data: response.data });
            }
        } catch (e) {
            console.error('Password autofill error:', e);
        }
    }

    async getTabInfo(tabId) {
        const tab = await chrome.tabs.get(tabId);
        return { id: tab.id, url: tab.url, title: tab.title, favIconUrl: tab.favIconUrl };
    }

    // ─── Core API Call (HTTP + Native Fallback) ───────────────────────────────

    async makeAPICall(endpoint, method = 'POST', data = null, includeAuth = true) {
        // Native mode: endpoint → action mapping
        if (this.useNative) {
            const action = endpoint.replace(/^\/api\//, '').replace(/\//g, '_');
            return this.makeNativeCall(action, data);
        }

        if (!this.apiBase) {
            throw new Error('No backend configured');
        }

        const url = `${this.apiBase}${endpoint}`;
        const headers = { 'Content-Type': 'application/json' };

        if (includeAuth && this.authData.isAuthenticated) {
            if (this.authData.jwt) headers['Authorization'] = `Bearer ${this.authData.jwt}`;
            if (this.authData.username) headers['X-Username'] = this.authData.username;
        }

        const options = { method, headers, credentials: 'include' };
        if (data && method !== 'GET') options.body = JSON.stringify(data);

        try {
            const response = await fetch(url, options);

            if (response.status === 401 && includeAuth) {
                const reAuthSuccess = await this.attemptReAuth();
                if (reAuthSuccess) return this.makeAPICall(endpoint, method, data, includeAuth);
                throw new Error('Session expired and re-authentication failed');
            }

            if (!response.ok) throw new Error(`API call failed: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error(`API call to ${endpoint} failed:`, error);
            throw error;
        }
    }

    async attemptReAuth() {
        if (!this.authData.username) return false;

        // Tauri-Backend: Versuche CLI-Session aus Native Host
        try {
            const stored = await chrome.storage.sync.get(['toolboxSettings']);
            if (stored.toolboxSettings?.backend === 'tauri') {
                return await this.syncNativeSessionToTauri();
            }
        } catch {}

        try {
            const payload = { Username: this.authData.username };
            if (this.authData.jwt) payload.Jwt_claim = this.authData.jwt;

            const response = await this.makeAPICall('/validateSession', 'POST', payload, false);

            if (response.result?.data_info === 'Valid Session') {
                this.authData.isAuthenticated = true;
                const stored = await chrome.storage.sync.get(['toolboxSettings']);
                if (stored.toolboxSettings) {
                    stored.toolboxSettings.isAuthenticated = true;
                    await chrome.storage.sync.set({ toolboxSettings: stored.toolboxSettings });
                }
                return true;
            }
        } catch (error) {
            console.error('Re-authentication failed:', error);
        }

        this.authData.isAuthenticated = false;
        const stored = await chrome.storage.sync.get(['toolboxSettings']);
        if (stored.toolboxSettings) {
            stored.toolboxSettings.isAuthenticated = false;
            await chrome.storage.sync.set({ toolboxSettings: stored.toolboxSettings });
        }
        return false;
    }

    // ─── Web Login ────────────────────────────────────────────────────────────

    async startWebLogin(backend = null) {
        // Tauri und Native haben keinen web-basierten Login-Endpunkt (kein /dist).
        // Alternativen: (a) Remote-OAuth, (b) CLI-Session via Native Host auslesen.
        const effectiveBackend = backend || (await chrome.storage.sync.get(['toolboxSettings'])).toolboxSettings?.backend;

        if (effectiveBackend === 'tauri' || effectiveBackend === 'native') {
            // Versuche zuerst native CLI-Session
            try {
                const nativeResp = await this.makeNativeCall('validate_session', {});
                if (nativeResp.authenticated) {
                    // Hole JWT für HTTP-Requests
                    const jwtResp = await this.makeNativeCall('get_session_jwt', {}).catch(() => ({ success: false }));
                    await this.storeAuthCredentials(
                        nativeResp.username,
                        jwtResp.jwt || '',
                        effectiveBackend === 'tauri' ? 'http://localhost:5000' : null
                    );
                    return { success: true, username: nativeResp.username, message: 'CLI session loaded' };
                }
            } catch {}

            // Kein CLI-Session gefunden: Benutzer muss tb login ausführen
            return {
                success: false,
                error: "Kein lokales Login möglich (kein Web-Frontend im Worker). Bitte 'tb login' im Terminal ausführen oder Remote-Backend wählen.",
                requireCLI: true
            };
        }

        // Standard Web-Login (local 8080 oder remote)
        let loginBase = this.apiBase;
        if (backend === 'local') loginBase = 'http://localhost:8080';
        else if (backend === 'remote') loginBase = 'https://simplecore.app';

        const sessionId = this.generateSessionId();
        const loginUrl = `${loginBase}/api/CloudM/open_web_login_web?session_id=${sessionId}&return_to=browser&log_in_for=Browser`;

        await chrome.tabs.create({ url: loginUrl, active: true });
        return this.pollForAuthCompletion(loginBase, sessionId);
    }

    generateSessionId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
            const r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        }).replace(/-/g, '');
    }

    async pollForAuthCompletion(baseUrl, sessionId, timeout = 300000) {
        const startTime = Date.now();
        const pollInterval = 2000;

        while (Date.now() - startTime < timeout) {
            try {
                const response = await fetch(`${baseUrl}/api/CloudM/open_check_cli_auth`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });

                if (!response.ok) { await this.sleep(pollInterval); continue; }

                const result = await response.json();
                if (result.authenticated || result.result?.authenticated) {
                    const authData = result.result || result;
                    const jwtToken = authData.jwt_token;
                    const username = authData.username;

                    if (jwtToken && username) {
                        await this.storeAuthCredentials(username, jwtToken, baseUrl);
                        this.showNotification('Login Successful', `Welcome back, ${username}!`);
                        return { success: true, username, message: 'Login successful' };
                    }
                }
            } catch {}
            await this.sleep(pollInterval);
        }

        this.showNotification('Login Timeout', 'Authentication window expired.');
        return { success: false, error: 'Authentication timeout' };
    }

    async storeAuthCredentials(username, jwtToken, baseUrl) {
        this.authData = { username, jwt: jwtToken, isAuthenticated: true };

        let backend = 'local';
        if (baseUrl?.includes('simplecore.app')) backend = 'remote';
        else if (baseUrl === 'http://localhost:5000') backend = 'tauri';
        else if (baseUrl && baseUrl !== 'http://localhost:8080') backend = 'custom';

        const stored = await chrome.storage.sync.get(['toolboxSettings']);
        const settings = stored.toolboxSettings || {};
        settings.username = username;
        settings.jwt = jwtToken;
        settings.isAuthenticated = true;
        settings.backend = backend;
        if (backend === 'custom') settings.customBackendUrl = baseUrl;
        if (baseUrl) this.apiBase = baseUrl;

        await chrome.storage.sync.set({ toolboxSettings: settings });
    }

    sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    showNotification(title, message) {
        try {
            chrome.notifications.create({ type: 'basic', iconUrl: 'icons/tb48.png', title, message });
        } catch { console.log('Notification:', title, '-', message); }
    }

    cleanupGestureHistory() {
        const oneHourAgo = Date.now() - 3600000;
        this.gestureHistory = this.gestureHistory.filter(g => g.timestamp > oneHourAgo);
    }

    // ── Prompt Library ──────────────────────────────────────────────────────

    async loadPromptLibrary() {
        const stored = await chrome.storage.local.get(['promptLibrary', 'siteRules']);
        if (stored.promptLibrary) this.promptLibrary = stored.promptLibrary;
        if (stored.siteRules) this.promptRules = stored.siteRules.rules || [];
        // Load bundled defaults if nothing stored
        if (!this.promptRules.length) {
            try {
                const r = await fetch(chrome.runtime.getURL('src/prompts/site_rules.json'));
                this.promptRules = (await r.json()).rules || [];
            } catch (e) { console.warn('[Prompts] site_rules load failed:', e); }
        }
        if (!Object.keys(this.promptLibrary.prompts || {}).length) {
            try {
                const r = await fetch(chrome.runtime.getURL('src/prompts/library.json'));
                this.promptLibrary = await r.json();
            } catch (e) { console.warn('[Prompts] library load failed:', e); }
        }
    }

    matchSiteRule(url) {
        if (!url) return null;
        try {
            const u = new URL(url);
            for (const rule of this.promptRules) {
                const m = rule.match;
                if (m.hostname && u.hostname !== m.hostname) continue;
                if (m.hostname_contains && !u.hostname.includes(m.hostname_contains)) continue;
                if (m.path_prefix && !u.pathname.startsWith(m.path_prefix)) continue;
                if (m.port) {
                    const p = u.port || (u.protocol === 'https:' ? '443' : '80');
                    if (p !== String(m.port)) continue;
                }
                if (m.port_range) {
                    const p = parseInt(u.port) || (u.protocol === 'https:' ? 443 : 80);
                    if (p < m.port_range[0] || p > m.port_range[1]) continue;
                }
                return rule;
            }
        } catch (_) {}
        return null;
    }

    async syncPromptLibrary() {
        if (!this.apiBase) return false;
        try {
            const res = await fetch(`${this.apiBase}/api/prompts/library`, {
                headers: this.authData.jwt ? { Authorization: `Bearer ${this.authData.jwt}` } : {}
            });
            if (!res.ok) return false;
            const data = await res.json();
            this.promptLibrary = data;
            await chrome.storage.local.set({ promptLibrary: data, lastPromptSync: Date.now() });
            return true;
        } catch (_) { return false; }
    }

    // ── Agent View (CDP via chrome.debugger) ────────────────────────────────

    async handleAgentView(tabId, action, msg) {
        if (!this.agentViews[tabId]) {
            this.agentViews[tabId] = new AgentView(tabId);
        }
        const agent = this.agentViews[tabId];

        try {
            switch (action) {
                case 'snapshot':
                    return await agent.getStructuredView(msg.maxDepth || 5);
                case 'execute':
                    // Die executeAction-Methode in AgentView ist intelligent genug,
                    // um die verschiedenen Aktionstypen (click, type, etc.) zu verarbeiten.
                    return await agent.executeAction(msg.agentAction);
                case 'detach':
                    await agent.detach();
                    delete this.agentViews[tabId];
                    return { ok: true };
                default:
                    return { error: `Unbekannte Agenten-Aktion: ${action}` };
            }
        } catch (e) {
            // Wenn ein Fehler auftritt (z.B. Tab geschlossen), räumen wir auf
            await agent.detach().catch(() => {}); // Fehler beim Detach ignorieren
            delete this.agentViews[tabId];
            throw e; // Fehler an den Aufrufer weiterleiten
        }
    }
}

const toolboxBackground = new ToolBoxBackground();
