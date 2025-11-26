// tbjs/core/user.js
// ToolBox V2 User Management with Clerk Integration

import TB from '../index.js';

const USER_STATE_KEY = 'tbjs_user_session';
const USER_DATA_TIMESTAMP_KEY = 'tbjs_user_data_timestamp';
const INACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes

const defaultUserState = {
    isAuthenticated: false,
    username: null,
    email: null,
    userId: null,  // Clerk user ID
    userLevel: 1,
    token: null,
    userData: {},
    settings: {},
    modData: {}
};

// Clerk instance (will be initialized)
let clerkInstance = null;
let clerkConfig = null;
let clerkInitPromise = null;

const user = {
    _lastActivityTimestamp: Date.now(),
    _initPromise: null,

    // =================== Initialization ===================

    _initActivityMonitor() {
        const activityEvents = ['mousemove', 'keydown', 'scroll', 'click'];
        const handler = () => {
            this._lastActivityTimestamp = Date.now();
        };
        activityEvents.forEach(event =>
            document.addEventListener(event, handler, { passive: true })
        );
        TB.logger.info('[User] Activity monitor initialized.');
    },

    _isUserActive() {
        return (Date.now() - this._lastActivityTimestamp) < INACTIVITY_TIMEOUT;
    },

    async init(forceServerFetch = false) {
        // Verhindere mehrfache Initialisierung
        if (this._initPromise) {
            return this._initPromise;
        }

        this._initPromise = this._doInit(forceServerFetch);
        return this._initPromise;
    },

    async _doInit(forceServerFetch = false) {
        this._initActivityMonitor();
        TB.logger.info('[User] Initializing with Clerk...');

        // Load state from localStorage
        let initialState = TB.state.get('user');

        if (!initialState || Object.keys(initialState).length === 0) {
            try {
                const storedSession = localStorage.getItem(USER_STATE_KEY);
                if (storedSession) {
                    initialState = JSON.parse(storedSession);
                    TB.logger.debug('[User] Loaded session from localStorage.');
                }
            } catch (e) {
                TB.logger.warn('[User] Could not parse stored session.', e);
                localStorage.removeItem(USER_STATE_KEY);
                initialState = null;
            }
        }

        const mergedState = { ...defaultUserState, ...(initialState || {}) };
        TB.state.set('user', mergedState);

        // Initialize Clerk
        await this._initClerk();

        // Listen for state changes to persist
        TB.events.on('state:changed:user', (newState) => {
            try {
                localStorage.setItem(USER_STATE_KEY, JSON.stringify(newState));
                localStorage.setItem(USER_DATA_TIMESTAMP_KEY, Date.now().toString());
            } catch (e) {
                TB.logger.error('[User] Failed to save user session:', e);
            }
        });

        return true;
    },

    async _initClerk() {
        // Verhindere mehrfache Clerk-Initialisierung
        if (clerkInitPromise) {
            return clerkInitPromise;
        }

        clerkInitPromise = this._doInitClerk();
        return clerkInitPromise;
    },

    async _doInitClerk() {
        try {
            // 1. Get Clerk config from backend FIRST
            TB.logger.debug('[User] Fetching Clerk config...');

            const configResponse = await fetch('/api/CloudM.AuthClerk/get_clerk_config');
            const configResult = await configResponse.json();

            TB.logger.debug('[User] Clerk config result:', configResult);

            // Prüfe auf "none" string
            if (configResult.error !== "none" || !configResult.result?.data?.publishable_key) {
                TB.logger.error('[User] Failed to get Clerk config:', configResult.info?.help_text);
                return false;
            }

            clerkConfig = configResult.result.data;
            const publishableKey = clerkConfig.publishable_key;

            TB.logger.debug('[User] Got publishable key:', publishableKey.substring(0, 20) + '...');

            // 2. Lade Clerk SDK und initialisiere
            await this._loadAndInitClerk(publishableKey);

            if (!clerkInstance) {
                TB.logger.error('[User] Failed to initialize Clerk instance.');
                return false;
            }

            TB.logger.info('[User] Clerk loaded successfully.');

            // Check if already signed in
            if (clerkInstance.user) {
                TB.logger.info('[User] User already signed in:', clerkInstance.user.id);
                await this._onClerkSignIn(clerkInstance.user);
            }

            // Listen for Clerk events
            clerkInstance.addListener((event) => {
                if (event.user) {
                    this._onClerkSignIn(event.user);
                } else if (event.session === null) {
                    this._onClerkSignOut();
                }
            });

            TB.logger.info('[User] Clerk initialized successfully.');
            return true;

        } catch (e) {
            TB.logger.error('[User] Clerk initialization failed:', e);
            return false;
        }
    },

    async _loadAndInitClerk(publishableKey) {
        /**
         * Lade Clerk SDK und initialisiere es mit dem publishableKey
         *
         * WICHTIG: Das Clerk browser SDK initialisiert sich selbst wenn es
         * ein data-clerk-publishable-key Attribut am Script-Tag findet.
         */

        // Prüfe ob Clerk bereits verfügbar ist
        if (window.Clerk) {
            TB.logger.debug('[User] Clerk already in window.');
            return this._initExistingClerk(publishableKey);
        }

        return new Promise((resolve, reject) => {
            TB.logger.debug('[User] Loading Clerk SDK...');

            const script = document.createElement('script');

            // WICHTIG: Setze data-clerk-publishable-key BEVOR das Script lädt
            // Clerk sucht danach und initialisiert sich automatisch
            script.setAttribute('data-clerk-publishable-key', publishableKey);
            script.src = 'https://cdn.jsdelivr.net/npm/@clerk/clerk-js@latest/dist/clerk.browser.js';
            script.async = true;

            script.onload = async () => {
                TB.logger.debug('[User] Clerk script loaded, waiting for initialization...');

                // Warte bis Clerk verfügbar ist
                let attempts = 0;
                const maxAttempts = 50; // 5 Sekunden

                const waitForClerk = async () => {
                    attempts++;

                    if (window.Clerk) {
                        try {
                            await this._initExistingClerk(publishableKey);
                            resolve(true);
                        } catch (e) {
                            reject(e);
                        }
                        return;
                    }

                    if (attempts >= maxAttempts) {
                        reject(new Error('Clerk not available after timeout'));
                        return;
                    }

                    setTimeout(waitForClerk, 100);
                };

                await waitForClerk();
            };

            script.onerror = (e) => {
                TB.logger.error('[User] Failed to load Clerk SDK:', e);
                reject(new Error('Failed to load Clerk SDK'));
            };

            document.head.appendChild(script);
        });
    },

    async _initExistingClerk(publishableKey) {
        /**
         * Initialisiere ein bestehendes Clerk-Objekt
         */
        TB.logger.debug('[User] Initializing existing Clerk...');

        const ClerkObj = window.Clerk;

        // Clerk kann als Konstruktor ODER als bereits initialisierte Instanz vorliegen
        if (typeof ClerkObj === 'function' && ClerkObj.prototype && ClerkObj.prototype.constructor === ClerkObj) {
            // Es ist ein Konstruktor (Klasse)
            TB.logger.debug('[User] Clerk is a constructor, creating instance...');
            try {
                clerkInstance = new ClerkObj(publishableKey);
                await clerkInstance.load();
            } catch (e) {
                TB.logger.error('[User] Failed to create Clerk instance:', e);
                throw e;
            }
        } else if (ClerkObj && typeof ClerkObj.load === 'function') {
            // Es ist bereits eine Instanz mit load-Methode
            TB.logger.debug('[User] Clerk is already an instance.');
            clerkInstance = ClerkObj;

            if (!clerkInstance.loaded) {
                try {
                    await clerkInstance.load();
                } catch (e) {
                    // Clerk könnte bereits geladen sein
                    TB.logger.debug('[User] Clerk.load() threw, might already be loaded:', e.message);
                }
            }
        } else if (ClerkObj && ClerkObj.user !== undefined) {
            // Clerk ist bereits vollständig initialisiert
            TB.logger.debug('[User] Clerk is fully initialized.');
            clerkInstance = ClerkObj;
        } else {
            TB.logger.error('[User] Unknown Clerk state:', typeof ClerkObj);
            throw new Error('Unknown Clerk state');
        }

        TB.logger.debug('[User] Clerk instance ready.');
    },

    _updateUserState(updates, clearExisting = false) {
        const currentState = clearExisting ? defaultUserState : (TB.state.get('user') || defaultUserState);
        const newState = { ...currentState, ...updates };
        TB.state.set('user', newState);
        TB.events.emit('user:stateChanged', newState);
        return newState;
    },

    // =================== Clerk Event Handlers ===================

    async _onClerkSignIn(clerkUser) {
        TB.logger.info('[User] Clerk sign-in detected:', clerkUser.id);

        const email = clerkUser.emailAddresses?.[0]?.emailAddress || '';
        const username = clerkUser.username || email.split('@')[0];

        // Get session token
        let token = null;
        if (clerkInstance?.session) {
            try {
                token = await clerkInstance.session.getToken();
            } catch (e) {
                TB.logger.warn('[User] Failed to get session token:', e);
            }
        }

        // Notify backend of sign-in
        const userData = {
            id: clerkUser.id,
            username: username,
            email_addresses: clerkUser.emailAddresses,
            session_token: token
        };

        try {
            // 1. Backend über Login informieren
            await fetch('/api/CloudM.AuthClerk/on_sign_in', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_data: userData })
            });

            // 2. Server-Session validieren
            await fetch('/validateSession', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_token: token,
                    clerk_user_id: clerkUser.id
                })
            });
        } catch (e) {
            TB.logger.error('[User] Failed to notify backend:', e);
        }

        // Load user data from backend
        let settings = {};
        let userLevel = 1;
        let modData = {};

        try {
            const userDataResponse = await fetch('/api/CloudM.AuthClerk/get_user_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clerk_user_id: clerkUser.id })
            });
            const userDataResult = await userDataResponse.json();

            if (userDataResult.error === "none" && userDataResult.result?.data) {
                const data = userDataResult.result.data;
                settings = data.settings || {};
                userLevel = data.level || 1;
                modData = data.mod_data || {};
            }
        } catch (e) {
            TB.logger.warn('[User] Failed to load user data:', e);
        }

        // Update state
        this._updateUserState({
            isAuthenticated: true,
            username: username,
            email: email,
            userId: clerkUser.id,
            userLevel: userLevel,
            token: token,
            settings: settings,
            modData: modData,
            userData: {
                firstName: clerkUser.firstName,
                lastName: clerkUser.lastName,
                imageUrl: clerkUser.imageUrl
            }
        });

        TB.events.emit('user:signedIn', { username, userId: clerkUser.id });
    },

    async _onClerkSignOut() {
        TB.logger.info('[User] Clerk sign-out detected.');

        const userId = this.getUserId();
        if (userId) {
            try {
                await fetch('/api/CloudM.AuthClerk/on_sign_out', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ clerk_user_id: userId })
                });
            } catch (e) {
                TB.logger.warn('[User] Failed to notify backend of sign-out:', e);
            }
        }

        this._updateUserState({}, true);  // Reset to default
        TB.events.emit('user:signedOut');
    },

    // =================== Authentication Methods ===================

    async signIn() {
        if (!clerkInstance) {
            await this._initClerk();
        }

        if (!clerkInstance) {
            TB.logger.error('[User] Clerk not initialized');
            return { success: false, message: 'Authentication service not available' };
        }

        try {
            await clerkInstance.openSignIn({
                afterSignInUrl: window.location.href,
                afterSignUpUrl: window.location.href
            });
            return { success: true };
        } catch (e) {
            TB.logger.error('[User] Sign-in error:', e);
            return { success: false, message: e.message };
        }
    },

    async signUp() {
        if (!clerkInstance) {
            await this._initClerk();
        }

        if (!clerkInstance) {
            TB.logger.error('[User] Clerk not initialized');
            return { success: false, message: 'Authentication service not available' };
        }

        try {
            await clerkInstance.openSignUp({
                afterSignInUrl: window.location.href,
                afterSignUpUrl: window.location.href
            });
            return { success: true };
        } catch (e) {
            TB.logger.error('[User] Sign-up error:', e);
            return { success: false, message: e.message };
        }
    },

    async signOut() {
        if (!clerkInstance) {
            this._updateUserState({}, true);
            return { success: true };
        }

        try {
            await clerkInstance.signOut();
            return { success: true, message: 'Signed out successfully' };
        } catch (e) {
            TB.logger.error('[User] Sign-out error:', e);
            this._updateUserState({}, true);
            return { success: false, message: e.message };
        }
    },

    async logout(notifyServer = true) {
        return this.signOut();
    },

    // =================== User Data Management ===================

    async updateSettings(settings) {
        const userId = this.getUserId();
        if (!userId) {
            return { success: false, message: 'Not authenticated' };
        }

        try {
            const response = await fetch('/api/CloudM.AuthClerk/update_user_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    clerk_user_id: userId,
                    settings: settings
                })
            });
            const result = await response.json();

            if (result.error === "none") {
                const currentSettings = TB.state.get('user.settings') || {};
                this._updateUserState({
                    settings: { ...currentSettings, ...settings }
                });
                return { success: true };
            }

            return { success: false, message: result.info?.help_text || 'Failed to update settings' };
        } catch (e) {
            TB.logger.error('[User] Update settings error:', e);
            return { success: false, message: e.message };
        }
    },

    async updateModData(modName, data) {
        const userId = this.getUserId();
        if (!userId) {
            return { success: false, message: 'Not authenticated' };
        }

        try {
            const currentModData = TB.state.get('user.modData') || {};
            const updatedModData = {
                ...currentModData,
                [modName]: { ...(currentModData[modName] || {}), ...data }
            };

            const response = await fetch('/api/CloudM.AuthClerk/update_user_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    clerk_user_id: userId,
                    mod_data: updatedModData
                })
            });
            const result = await response.json();

            if (result.error === "none") {
                this._updateUserState({ modData: updatedModData });
                return { success: true };
            }

            return { success: false, message: result.info?.help_text || 'Failed to update mod data' };
        } catch (e) {
            TB.logger.error('[User] Update mod data error:', e);
            return { success: false, message: e.message };
        }
    },

    async fetchUserData() {
        const userId = this.getUserId();
        if (!userId) {
            return { success: false, message: 'Not authenticated' };
        }

        try {
            const response = await fetch('/api/CloudM.AuthClerk/get_user_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clerk_user_id: userId })
            });
            const result = await response.json();

            if (result.error === "none" && result.result?.data) {
                const data = result.result.data;
                this._updateUserState({
                    settings: data.settings || {},
                    userLevel: data.level || 1,
                    modData: data.mod_data || {}
                });
                return { success: true, data: data };
            }

            return { success: false, message: result.info?.help_text || 'Failed to fetch user data' };
        } catch (e) {
            TB.logger.error('[User] Fetch user data error:', e);
            return { success: false, message: e.message };
        }
    },

    // =================== Getters ===================

    isAuthenticated() {
        return TB.state.get('user.isAuthenticated') || false;
    },

    getUsername() {
        return TB.state.get('user.username') || null;
    },

    getEmail() {
        return TB.state.get('user.email') || null;
    },

    getUserId() {
        return TB.state.get('user.userId') || null;
    },

    getUserLevel() {
        return TB.state.get('user.userLevel') || 1;
    },

    getToken() {
        return TB.state.get('user.token') || null;
    },

    getSettings() {
        return TB.state.get('user.settings') || {};
    },

    getSetting(key, defaultValue = null) {
        const settings = this.getSettings();
        return settings[key] !== undefined ? settings[key] : defaultValue;
    },

    getModData(modName) {
        const modData = TB.state.get('user.modData') || {};
        return modData[modName] || {};
    },

    getUserData(key) {
        return TB.state.get(`user.userData.${key}`);
    },

    // =================== Clerk Direct Access ===================

    getClerkInstance() {
        return clerkInstance;
    },

    getClerkUser() {
        return clerkInstance?.user || null;
    },

    getClerkConfig() {
        return clerkConfig;
    },

    async getSessionToken() {
        if (clerkInstance?.session) {
            try {
                return await clerkInstance.session.getToken();
            } catch (e) {
                TB.logger.warn('[User] Failed to get session token:', e);
            }
        }
        return this.getToken();
    },

    isClerkReady() {
        return clerkInstance !== null;
    },

    // =================== UI Mount Points ===================

    async mountSignIn(elementOrSelector, options = {}) {
        if (!clerkInstance) {
            const success = await this._initClerk();
            if (!success) {
                const element = typeof elementOrSelector === 'string'
                    ? document.querySelector(elementOrSelector)
                    : elementOrSelector;

                if (element) {
                    element.innerHTML = `
                        <div style="text-align: center; color: #ef4444; padding: 20px;">
                            <p>❌ Authentication service not available</p>
                            <button onclick="location.reload()" style="margin-top: 16px; padding: 8px 16px; cursor: pointer; background: #6366f1; color: white; border: none; border-radius: 6px;">
                                Retry
                            </button>
                        </div>
                    `;
                }
                return;
            }
        }

        const element = typeof elementOrSelector === 'string'
            ? document.querySelector(elementOrSelector)
            : elementOrSelector;

        if (!element) {
            TB.logger.error('[User] Element not found for sign-in mount');
            return;
        }

        element.innerHTML = '';

        clerkInstance.mountSignIn(element, {
            afterSignInUrl: options.afterSignInUrl || window.location.href,
            signUpUrl: options.signUpUrl || clerkConfig?.sign_up_url || '/web/assets/signup.html',
            appearance: {
                elements: {
                    rootBox: { width: '100%' },
                    card: {
                        background: 'transparent',
                        boxShadow: 'none',
                        border: 'none'
                    },
                    formButtonPrimary: {
                        backgroundColor: '#6366f1'
                    }
                }
            },
            ...options
        });
    },

    async mountSignUp(elementOrSelector, options = {}) {
        if (!clerkInstance) {
            const success = await this._initClerk();
            if (!success) {
                const element = typeof elementOrSelector === 'string'
                    ? document.querySelector(elementOrSelector)
                    : elementOrSelector;

                if (element) {
                    element.innerHTML = `
                        <div style="text-align: center; color: #ef4444; padding: 20px;">
                            <p>❌ Authentication service not available</p>
                            <button onclick="location.reload()" style="margin-top: 16px; padding: 8px 16px; cursor: pointer; background: #6366f1; color: white; border: none; border-radius: 6px;">
                                Retry
                            </button>
                        </div>
                    `;
                }
                return;
            }
        }

        const element = typeof elementOrSelector === 'string'
            ? document.querySelector(elementOrSelector)
            : elementOrSelector;

        if (!element) {
            TB.logger.error('[User] Element not found for sign-up mount');
            return;
        }

        element.innerHTML = '';

        clerkInstance.mountSignUp(element, {
            afterSignUpUrl: options.afterSignUpUrl || window.location.href,
            signInUrl: options.signInUrl || clerkConfig?.sign_in_url || '/web/assets/login.html',
            appearance: {
                elements: {
                    rootBox: { width: '100%' },
                    card: {
                        background: 'transparent',
                        boxShadow: 'none',
                        border: 'none'
                    },
                    formButtonPrimary: {
                        backgroundColor: '#6366f1'
                    }
                }
            },
            ...options
        });
    },

    async mountUserButton(elementOrSelector, options = {}) {
        if (!clerkInstance) {
            await this._initClerk();
        }

        if (!clerkInstance) {
            TB.logger.error('[User] Clerk not initialized');
            return;
        }

        const element = typeof elementOrSelector === 'string'
            ? document.querySelector(elementOrSelector)
            : elementOrSelector;

        if (!element) {
            TB.logger.error('[User] Element not found for user button mount');
            return;
        }

        clerkInstance.mountUserButton(element, {
            afterSignOutUrl: options.afterSignOutUrl || clerkConfig?.sign_in_url || '/web/assets/login.html',
            ...options
        });
    },

    unmountAll() {
        if (clerkInstance) {
            try {
                clerkInstance.unmountSignIn?.();
                clerkInstance.unmountSignUp?.();
                clerkInstance.unmountUserButton?.();
            } catch (e) {
                TB.logger.warn('[User] Error unmounting Clerk components:', e);
            }
        }
    },

    // =================== Auth Data Setter ===================

    setAuthData({ token, userId, username }) {
        this._updateUserState({
            isAuthenticated: true,
            token: token,
            userId: userId,
            username: username
        });
    }
};

export default user;
