// tbjs/core/user.js
// ToolBox V2 User Management with Custom Auth (CloudM.Auth)

import TB from '../index.js';
import env from './env.js';

const USER_STATE_KEY = 'tbjs_user_session';
const USER_DATA_TIMESTAMP_KEY = 'tbjs_user_data_timestamp';
const INACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes
const TOKEN_REFRESH_INTERVAL = 10 * 60 * 1000; // 10 minutes

const defaultUserState = {
    isAuthenticated: false,
    username: null,
    email: null,
    userId: null,
    userLevel: 1,
    token: null,
    refreshToken: null,
    userData: {},
    settings: {},
    modData: {}
};

// Auth state tracking
let authInitialized = false;

const user = {
    _lastActivityTimestamp: Date.now(),
    _initPromise: null,
    _tokenRefreshTimerId: null,
    _signInInProgress: false,
    _lastSignInUserId: null,
    _authCallbackProcessed: false, // NEW: Track if callback was already processed

    // =================== URL Resolution (Web vs Tauri Worker) ===================

    /**
     * Returns the base URL for non-API routes (e.g. /validateSession, /auth/discord/url).
     * Web: '' (relative), Tauri: 'http://localhost:5000' (worker).
     */
    _getBaseUrl() {
        if (env.isTauri()) {
            return env.getWorkerHttpUrl() || 'http://localhost:5000';
        }
        return '';
    },

    /**
     * Returns the base URL for /api/* routes.
     * Web: '' (relative), Tauri: 'http://localhost:5000' (worker).
     */
    _getApiBaseUrl() {
        if (env.isTauri()) {
            return env.getWorkerHttpUrl() || 'http://localhost:5000';
        }
        return '';
    },

    // =================== Backend Session Validation ===================

    /**
     * Validates the current session token with the backend.
     * @returns {Promise<boolean>} True if backend confirms valid session
     */
    async validateBackendSession() {
        const token = this.getToken();
        const userId = this.getUserId();

        if (!token || !userId) {
            TB.logger.debug('[User] Cannot validate session: missing token or userId');
            return false;
        }

        try {
            TB.logger.debug('[User] Validating session with backend...');

            const response = await fetch(`${this._getBaseUrl()}/validateSession`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    session_token: token,
                    user_id: userId
                })
            });

            if (!response.ok) {
                TB.logger.warn(`[User] Backend validation HTTP error: ${response.status}`);
                return false;
            }

            const result = await response.json();

            if (result.error === "none" && result.result?.data?.authenticated === true) {
                TB.logger.debug('[User] Backend confirmed session validity');
                return true;
            }

            TB.logger.warn('[User] Backend rejected session token:', result.error);
            return false;

        } catch (e) {
            TB.logger.error('[User] Backend validation exception:', e);
            return false;
        }
    },

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

    // =========================================================================
    // FIX 1: init() re-checks auth callback on subsequent calls
    // =========================================================================
    // Problem: In Tauri, _doInit() runs on /index.html where URL has no token
    //          params. Router later navigates to /login.html?token=eyJ... via SPA.
    //          login.html calls TB.user.init() but the old code returned the
    //          already-resolved promise without re-checking URL params.
    // Fix:    If init was already called and user is NOT authenticated, re-run
    //         _checkAuthCallback() which now reads current window.location.
    // =========================================================================
    async init(forceServerFetch = false) {
        if (this._initPromise) {
            await this._initPromise;

            // Re-check auth callback if user is still not authenticated
            // This handles the Tauri case where router navigated to a URL
            // with token params AFTER the initial init completed
            if (!this.isAuthenticated() && !this._authCallbackProcessed) {
                await this._checkAuthCallback();
            }

            return true;
        }
        this._initPromise = this._doInit(forceServerFetch);
        return this._initPromise;
    },

    async _doInit(forceServerFetch = false) {
        this._initActivityMonitor();
        TB.logger.info('[User] Initializing with Custom Auth...');

        // STEP 1: Load from localStorage FIRST (synchronous, before async ops)
        let initialState = null;

        // Try localStorage first (most likely to have fresh data from OAuth callback)
        try {
            const storedSession = localStorage.getItem(USER_STATE_KEY);
            if (storedSession) {
                initialState = JSON.parse(storedSession);
                TB.logger.debug('[User] Loaded session from localStorage.');
            }
        } catch (e) {
            TB.logger.warn('[User] Could not parse stored session.', e);
            localStorage.removeItem(USER_STATE_KEY);
        }

        // Fallback to TB.state if localStorage is empty
        if (!initialState || Object.keys(initialState).length === 0) {
            initialState = TB.state.get('user');
        }

        // STEP 2: Set state IMMEDIATELY (synchronous)
        // This ensures token is available for fetch override before any async ops
        const mergedState = { ...defaultUserState, ...(initialState || {}) };
        TB.state.set('user', mergedState);

        if (mergedState.userId) {
            TB.logger.info('[User] Session restored:', {
                isAuthenticated: mergedState.isAuthenticated,
                userId: mergedState.userId,
                username: mergedState.username
            });
        }

        // STEP 3: Register state change listener for future updates (NOT for initial load)
        TB.events.on('state:changed:user', (newState) => {
            try {
                localStorage.setItem(USER_STATE_KEY, JSON.stringify(newState));
                localStorage.setItem(USER_DATA_TIMESTAMP_KEY, Date.now().toString());
            } catch (e) {
                TB.logger.error('[User] Failed to save user session:', e);
            }
        });

        // STEP 4: Async operations (token validation, OAuth callback processing)
        // Check if we have a stored token and validate it
        if (mergedState.token && mergedState.isAuthenticated) {
            const isValid = await this.validateBackendSession();
            if (isValid) {
                TB.logger.info('[User] Restored valid session from storage.');
                this._startTokenRefreshTimer();
                // Don't emit user:signedIn here - it may trigger unwanted redirects
            } else {
                TB.logger.warn('[User] Stored session invalid, clearing.');
                this._updateUserState({}, true);
            }
        }

        // Check URL for auth callback tokens (OAuth redirect)
        await this._checkAuthCallback();

        authInitialized = true;
        return true;
    },

    // =========================================================================
    // FIX 2: _checkAuthCallback() uses /validateSession (AuthHandler) instead of
    //         /api/CloudM.Auth/validate_session (blocked by AccessController at level=0)
    // =========================================================================
    // Problem: After OAuth redirect, the client has a JWT token but NO cookie session.
    //          /api/CloudM.Auth/* goes through ToolBoxHandler → AccessController which
    //          checks the cookie session (level=0 = anonymous) → 401.
    //          /validateSession goes through AuthHandler which validates the JWT
    //          directly, bypassing AccessController.
    // =========================================================================
    /**
     * Check URL for auth callback parameters (after OAuth redirect).
     * Handles ?token=xxx&refresh_token=xxx from OAuth callbacks.
     *
     * CRITICAL: Uses /validateSession (AuthHandler endpoint) which bypasses
     * ToolBoxHandler access control. /api/CloudM.Auth/* would fail because
     * the AccessController requires a valid cookie session, which doesn't
     * exist at this point (cross-origin: Tauri → Worker, or fresh browser).
     */
    async _checkAuthCallback() {
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');
        const refreshToken = urlParams.get('refresh_token');

        if (!token) return;

        // Guard: don't process the same callback twice
        if (this._authCallbackProcessed) return;
        this._authCallbackProcessed = true;

        TB.logger.info('[User] Auth callback detected, processing tokens...');

        // Clean up URL (remove token params) IMMEDIATELY to prevent re-processing
        const cleanUrl = new URL(window.location.href);
        ['token', 'refresh_token', 'user_id', 'username'].forEach(p => cleanUrl.searchParams.delete(p));
        window.history.replaceState({}, '', cleanUrl.toString());

        try {
            // ================================================================
            // Use /validateSession (AuthHandler) — NOT /api/CloudM.Auth/validate_session
            // AuthHandler endpoints bypass AccessController, which is critical because
            // at this point we have NO cookie session (cross-origin from bridge page)
            // ================================================================
            const baseUrl = this._getBaseUrl();
            TB.logger.debug('[User] Validating OAuth token via /validateSession...');

            const response = await fetch(`${baseUrl}/validateSession`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    session_token: token,
                    user_id: urlParams.get('user_id') || ''
                })
            });

            if (!response.ok) {
                throw new Error(`Validation HTTP ${response.status}`);
            }

            const result = await response.json();

            if (result.error !== "none" || !result.result?.data?.authenticated) {
                throw new Error(result.error || 'Token rejected by server');
            }

            const userData = result.result.data;
            TB.logger.debug('[User] Token validated, user:', userData.user_id);

            // Fetch full user data via /api_user_data (also AuthHandler, no AccessController)
            let settings = {};
            let userLevel = userData.level || 1;
            let modData = {};

            try {
                const udResp = await fetch(`${baseUrl}/api_user_data`, {
                    method: 'GET',
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (udResp.ok) {
                    const udResult = await udResp.json();
                    if (udResult.error === "none" && udResult.result?.data) {
                        const data = udResult.result.data;
                        settings = data.settings || {};
                        userLevel = data.level || userLevel;
                        modData = data.mod_data || {};
                    }
                }
            } catch (e) {
                TB.logger.debug('[User] User data fetch optional, continuing:', e.message);
            }

            // Create new state object
            const newState = {
                isAuthenticated: true,
                username: userData.username || userData.user_name || urlParams.get('username') || '',
                email: userData.email || '',
                userId: userData.user_id,
                userLevel: userLevel,
                token: token,
                refreshToken: refreshToken || '',
                settings: settings,
                modData: modData,
                userData: {
                    provider: userData.provider || '',
                    imageUrl: userData.image_url || ''
                }
            };

            // Update state via _updateUserState (uses TB.state.set)
            this._updateUserState(newState);

            // CRITICAL: Explicitly write to localStorage IMMEDIATELY (synchronous)
            // This ensures tokens are available before the user:signedIn event triggers navigation
            localStorage.setItem(USER_STATE_KEY, JSON.stringify(TB.state.get('user')));
            localStorage.setItem(USER_DATA_TIMESTAMP_KEY, Date.now().toString());

            TB.logger.info('[User] Auth callback processed successfully, userId:', newState.userId);
            TB.events.emit('user:signedIn', {
                username: newState.username,
                userId: newState.userId
            });

            this._startTokenRefreshTimer();
            this._handlePostAuthRedirect();

        } catch (e) {
            TB.logger.error('[User] Auth callback FAILED:', e);
            this._authCallbackProcessed = false; // Allow retry
            await this._handleAuthFailure(e.message);
        }
    },

    // =========================================================================
    // FIX 3: Public method for login.html to process auth tokens directly
    // =========================================================================
    /**
     * Process auth callback from a given URL string.
     * Use this when the URL has token params but _checkAuthCallback couldn't
     * run at the right time (e.g., SPA navigation changed the URL after init).
     *
     * @param {string} [url] - URL to extract tokens from. Defaults to window.location.href.
     * @returns {Promise<boolean>} True if tokens were found and processed.
     */
    async processAuthCallback(url) {
        const targetUrl = url || window.location.href;
        const urlParams = new URLSearchParams(new URL(targetUrl, window.location.origin).search);
        const token = urlParams.get('token');

        if (!token) {
            TB.logger.debug('[User] processAuthCallback: no token found in URL');
            return false;
        }

        if (this.isAuthenticated()) {
            TB.logger.debug('[User] processAuthCallback: already authenticated');
            return true;
        }

        // Reset the guard so _checkAuthCallback can run
        this._authCallbackProcessed = false;

        // Temporarily update window.location.search if needed (for _checkAuthCallback)
        // Actually, we can just call _checkAuthCallback since window.location should have the params
        await this._checkAuthCallback();
        return this.isAuthenticated();
    },

    _updateUserState(updates, clearExisting = false) {
        const currentState = clearExisting ? defaultUserState : (TB.state.get('user') || defaultUserState);
        const newState = { ...currentState, ...updates };
        TB.state.set('user', newState);
        // Emit both events: the state system event (triggers localStorage persist)
        // and the user-specific event for UI listeners
        TB.events.emit('state:changed:user', newState);
        TB.events.emit('user:stateChanged', newState);
        return newState;
    },

    // =================== OAuth & Passkey Methods ===================

    /**
     * Login with Discord OAuth.
     * Fetches the OAuth URL from the backend and redirects the browser.
     */
    async loginWithDiscord() {
        TB.logger.info('[User] Starting Discord OAuth login...');
        try {
            // Fix: In Tauri, window.location.origin may be empty. Use a proper origin.
            let redirectAfter;
            if (env.isTauri()) {
                // In Tauri, use the Tauri dev URL or configured base URL
                // The worker will redirect back to this origin after OAuth
                redirectAfter = encodeURIComponent(window.location.origin || 'http://localhost:8080');
                TB.logger.debug(`[User] Tauri mode, using redirect_after: ${redirectAfter}`);
            } else {
                redirectAfter = encodeURIComponent(window.location.origin);
            }
            const response = await fetch(`${this._getBaseUrl()}/auth/discord/url?redirect_after=${redirectAfter}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const json = await response.json();
            const authUrl = json.result?.auth_url || json.auth_url;
            if (!authUrl) throw new Error('No auth URL returned from server');
            TB.logger.debug(`[User] Redirecting to Discord OAuth URL`);
            window.location.href = authUrl;
        } catch (e) {
            TB.logger.error('[User] Discord login failed:', e);
            TB.events.emit('user:authError', { message: e.message });
            throw e;
        }
    },

    /**
     * Login with Google OAuth.
     * Fetches the OAuth URL from the backend and redirects the browser.
     */
    async loginWithGoogle() {
        TB.logger.info('[User] Starting Google OAuth login...');
        try {
            // Fix: In Tauri, window.location.origin may be empty. Use a proper origin.
            let redirectAfter;
            if (env.isTauri()) {
                // In Tauri, use the Tauri dev URL or configured base URL
                redirectAfter = encodeURIComponent(window.location.origin || 'http://localhost:8080');
                TB.logger.debug(`[User] Tauri mode, using redirect_after: ${redirectAfter}`);
            } else {
                redirectAfter = encodeURIComponent(window.location.origin);
            }
            const response = await fetch(`${this._getBaseUrl()}/auth/google/url?redirect_after=${redirectAfter}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const json = await response.json();
            const authUrl = json.result?.auth_url || json.auth_url;
            if (!authUrl) throw new Error('No auth URL returned from server');
            TB.logger.debug(`[User] Redirecting to Google OAuth URL`);
            window.location.href = authUrl;
        } catch (e) {
            TB.logger.error('[User] Google login failed:', e);
            TB.events.emit('user:authError', { message: e.message });
            throw e;
        }
    },

    /**
     * Login with WebAuthn Passkey.
     * @param {string|null} username - Optional username hint
     */
    async loginWithPasskey(username = null) {
        TB.logger.info('[User] Starting Passkey login...');

        if (typeof window.PublicKeyCredential === 'undefined') {
            throw new Error('Passkeys are not supported on this device');
        }

        try {
            // Step 1: Get challenge from server
            const startResponse = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/passkey_login_start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username })
            });
            if (!startResponse.ok) throw new Error(`HTTP ${startResponse.status}`);

            const startJson = await startResponse.json();
            const options = startJson.result || startJson;

            options.challenge = this._base64ToArrayBuffer(options.challenge);
            if (options.allowCredentials) {
                options.allowCredentials = options.allowCredentials.map(cred => ({
                    ...cred,
                    id: this._base64ToArrayBuffer(cred.id)
                }));
            }

            // Step 2: Ask authenticator
            const credential = await navigator.credentials.get({ publicKey: options });

            // Step 3: Send response to server
            const finishResponse = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/passkey_login_finish`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    credential: this._credentialToJSON(credential),
                    username
                })
            });
            if (!finishResponse.ok) throw new Error(`HTTP ${finishResponse.status}`);

            const finishJson = await finishResponse.json();
            const data = finishJson.result || finishJson;

            if (data.access_token || data.token) {
                this.setAuthData({
                    token: data.access_token || data.token,
                    refreshToken: data.refresh_token,
                    userId: data.user_id || data.user?.user_id,
                    username: data.username || data.user?.username || username,
                    email: data.email || null
                });
            }

            return data;
        } catch (e) {
            TB.logger.error('[User] Passkey login failed:', e);
            TB.events.emit('user:authError', { message: e.message });
            throw e;
        }
    },

    /**
     * Register a new Passkey for the current user.
     * @param {string} username
     * @param {string} displayName
     */
    async registerPasskey(username, displayName) {
        TB.logger.info('[User] Starting Passkey registration...');

        if (typeof window.PublicKeyCredential === 'undefined') {
            throw new Error('Passkeys are not supported on this device');
        }

        const token = this.getToken();

        try {
            const startResponse = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/passkey_register_start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                body: JSON.stringify({ username, displayName })
            });
            if (!startResponse.ok) throw new Error(`HTTP ${startResponse.status}`);

            const startJson = await startResponse.json();
            const options = startJson.result || startJson;

            options.challenge = this._base64ToArrayBuffer(options.challenge);
            options.user.id = this._base64ToArrayBuffer(options.user.id);
            if (options.excludeCredentials) {
                options.excludeCredentials = options.excludeCredentials.map(cred => ({
                    ...cred,
                    id: this._base64ToArrayBuffer(cred.id)
                }));
            }

            const credential = await navigator.credentials.create({ publicKey: options });

            const finishResponse = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/passkey_register_finish`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                body: JSON.stringify({
                    credential: this._credentialToJSON(credential),
                    username,
                    displayName
                })
            });
            if (!finishResponse.ok) throw new Error(`HTTP ${finishResponse.status}`);

            const finishJson = await finishResponse.json();
            const data = finishJson.result || finishJson;

            // If registration returns tokens, set auth state
            if (data.access_token || data.token) {
                this.setAuthData({
                    token: data.access_token || data.token,
                    refreshToken: data.refresh_token,
                    userId: data.user_id || data.user?.user_id,
                    username: username,
                    email: data.email || null
                });
            }

            return data;
        } catch (e) {
            TB.logger.error('[User] Passkey registration failed:', e);
            TB.events.emit('user:authError', { message: e.message });
            throw e;
        }
    },

    // =================== WebAuthn Helpers ===================

    /**
     * Convert base64url string (from py_webauthn) to ArrayBuffer.
     * base64url uses: - instead of +, _ instead of /, no = padding.
     */
    _base64ToArrayBuffer(base64url) {
        // Convert base64url to standard base64
        let base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
        // Add padding
        const pad = (4 - (base64.length % 4)) % 4;
        base64 += '='.repeat(pad);
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    },

    /**
     * Convert ArrayBuffer to base64url string (for py_webauthn).
     */
    _arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        // Standard base64
        let base64 = btoa(binary);
        // Convert to base64url
        return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    },

    /**
     * Convert PublicKeyCredential to JSON-serializable object.
     */
    _credentialToJSON(credential) {
        return {
            id: credential.id,
            rawId: this._arrayBufferToBase64(credential.rawId),
            type: credential.type,
            response: {
                clientDataJSON: this._arrayBufferToBase64(credential.response.clientDataJSON),
                authenticatorData: credential.response.authenticatorData
                    ? this._arrayBufferToBase64(credential.response.authenticatorData) : undefined,
                signature: credential.response.signature
                    ? this._arrayBufferToBase64(credential.response.signature) : undefined,
                attestationObject: credential.response.attestationObject
                    ? this._arrayBufferToBase64(credential.response.attestationObject) : undefined,
                userHandle: credential.response.userHandle
                    ? this._arrayBufferToBase64(credential.response.userHandle) : null
            }
        };
    },

    // =================== Authentication Methods ===================

    /**
     * Redirect to sign-in page.
     * Custom Auth uses its own login page.
     */
    async signIn(options = {}) {
        const currentUrl = window.location.href;
        const loginUrl = options.loginUrl || '/web/scripts/login.html';
        const next = options.next || currentUrl;

        window.location.href = `${loginUrl}?next=${encodeURIComponent(next)}`;
        return { success: true };
    },

    /**
     * Redirect to sign-up page.
     */
    async signUp(options = {}) {
        const currentUrl = window.location.href;
        const signUpUrl = options.signUpUrl || '/web/scripts/login.html?mode=signup';
        const next = options.next || currentUrl;

        window.location.href = `${signUpUrl}&next=${encodeURIComponent(next)}`;
        return { success: true };
    },

    /**
     * Sign out the user.
     */
    async signOut() {
        TB.logger.info('[User] Signing out...');

        this._stopTokenRefreshTimer();
        this._authCallbackProcessed = false; // Reset for next login

        const userId = this.getUserId();
        const token = this.getToken();

        // Notify backend
        if (userId && token) {
            try {
                await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/logout`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({ token: token })
                });
            } catch (e) {
                TB.logger.warn('[User] Failed to notify backend of sign-out:', e);
            }
        }

        // Clear local state
        this._updateUserState({}, true);
        TB.events.emit('user:signedOut');

        TB.logger.info('[User] Signed out successfully');
        return { success: true, message: 'Signed out successfully' };
    },

    async logout(notifyServer = true) {
        return this.signOut();
    },

    // =================== Auth Failure & Redirect ===================

    async _handleAuthFailure(message) {
        TB.logger.error('[User] Handling auth failure:', message);

        // Reset local state
        this._updateUserState({}, true);
        this._stopTokenRefreshTimer();

        // Notify user
        if (window.TB?.ui?.Toast?.showError) {
            window.TB.ui.Toast.showError(`Login failed: ${message}`);
        }

        TB.events.emit('user:authError', { message });

        // Redirect to login if needed
        const currentPath = window.location.pathname;
        if (!currentPath.includes('/login.html') && !currentPath.includes('/signup.html')) {
            setTimeout(() => {
                window.location.href = '/web/scripts/login.html';
            }, 2000);
        }
    },

    _handlePostAuthRedirect() {
        const currentPath = window.location.pathname;
        const isAuthPage = currentPath.includes('/login.html') || currentPath.includes('/signup.html');

        if (isAuthPage) {
            TB.logger.info('[User] On auth page after sign-in, redirecting...');

            let redirectUrl = '/web/mainContent.html';

            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.has('next')) {
                redirectUrl = urlParams.get('next');
            }

            TB.logger.info('[User] Redirecting to:', redirectUrl);

            setTimeout(() => {
                if (window.TB?.router?.navigateTo) {
                    window.TB.router.navigateTo(redirectUrl);
                } else {
                    window.location.href = redirectUrl;
                }
            }, 500);
        }
    },

    // =================== Token Management ===================

    /**
     * Get current session token. Returns stored JWT token.
     */
    async getSessionToken() {
        return this.getToken();
    },

    /**
     * Refresh the JWT access token using the refresh token.
     */
    async _refreshToken() {
        const refreshToken = TB.state.get('user.refreshToken');
        const currentToken = this.getToken();
        if (!refreshToken && !currentToken) {
            throw new Error('No active session');
        }

        // Skip refresh if token was updated very recently (< 2 min)
        const lastUpdate = parseInt(localStorage.getItem(USER_DATA_TIMESTAMP_KEY) || '0');
        if (Date.now() - lastUpdate < 120000) {
            TB.logger.debug('[User] Token recently updated, skipping refresh');
            return;
        }

        try {
            const response = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/refresh_token`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: refreshToken, token: currentToken })
            });

            if (!response.ok) throw new Error(`Refresh failed: ${response.status}`);
            const result = await response.json();

            if (result.error !== "none" || !result.result?.data?.access_token) {
                throw new Error('Token refresh rejected');
            }

            const data = result.result.data;
            this._updateUserState({
                token: data.access_token,
                refreshToken: data.refresh_token || refreshToken
            });
            TB.logger.info('[User] Session token refreshed');
        } catch (e) {
            TB.logger.error('[User] Token refresh failed:', e);
            await this.signOut();
            throw e;
        }
    },

    /**
     * Start periodic token refresh timer.
     */
    _startTokenRefreshTimer() {
        this._stopTokenRefreshTimer();

        this._tokenRefreshTimerId = setInterval(async () => {
            try {
                if (!this._isUserActive() || !this.getToken()) return;
                // Proactive refresh — single call, no validate-then-refresh double-call
                await this._refreshToken();
            } catch (e) {
                TB.logger.warn('[User] Token auto-refresh failed:', e);
            }
        }, TOKEN_REFRESH_INTERVAL);

        TB.logger.debug('[User] Token refresh timer started');
    },

    /**
     * Stop the token refresh timer.
     */
    _stopTokenRefreshTimer() {
        if (this._tokenRefreshTimerId) {
            clearInterval(this._tokenRefreshTimerId);
            this._tokenRefreshTimerId = null;
            TB.logger.debug('[User] Token refresh timer stopped');
        }
    },

    // =================== User Data Management ===================

    async updateSettings(settings) {
        const userId = this.getUserId();
        const token = this.getToken();
        if (!userId || !token) {
            return { success: false, message: 'Not authenticated' };
        }

        try {
            const response = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/update_user_data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    user_id: userId,
                    settings: settings
                })
            });
            const result = await response.json();

            if (result.error === "none") {
                const currentSettings = TB.state.get('user.settings') || {};
                this._updateUserState({ settings: { ...currentSettings, ...settings } });
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
        const token = this.getToken();
        if (!userId || !token) {
            return { success: false, message: 'Not authenticated' };
        }

        try {
            const currentModData = TB.state.get('user.modData') || {};
            const updatedModData = {
                ...currentModData,
                [modName]: { ...(currentModData[modName] || {}), ...data }
            };

            const response = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/update_user_data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    user_id: userId,
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
        const token = this.getToken();
        if (!userId || !token) {
            return { success: false, message: 'Not authenticated' };
        }

        try {
            const response = await fetch(`${this._getApiBaseUrl()}/api/CloudM.Auth/get_user_data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ user_id: userId })
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

    /**
     * Checks if the user is authenticated.
     * @param {boolean} [checkBackend=false] - If true, performs async backend check.
     * @returns {boolean|Promise<boolean>}
     */
    isAuthenticated(checkBackend = false) {
        const localState = TB.state.get('user.isAuthenticated') || false;

        if (!checkBackend || !localState) {
            return localState;
        }

        return (async () => {
            const isValid = await this.validateBackendSession();

            if (!isValid && localState) {
                TB.logger.warn('[User] Session invalid on backend. Logging out.');
                await this._handleAuthFailure('Session expired or revoked by server');
                return false;
            }

            return isValid;
        })();
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

    getRefreshToken() {
        return TB.state.get('user.refreshToken') || null;
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

    /**
     * Check if auth system is initialized and ready.
     */
    isAuthReady() {
        return authInitialized;
    },

    // =================== UI Mount Points (Custom Auth) ===================

    /**
     * Mount sign-in form or redirect to login page.
     * For Custom Auth, we redirect to the login page since there's no embeddable SDK.
     */
    async mountSignIn(elementOrSelector, options = {}) {
        const element = typeof elementOrSelector === 'string'
            ? document.querySelector(elementOrSelector)
            : elementOrSelector;

        if (!element) {
            TB.logger.error('[User] Element not found for sign-in mount');
            return;
        }

        // Custom Auth: render a simple redirect/iframe or load login form inline
        const urlParams = new URLSearchParams(window.location.search);
        const redirectUrl = urlParams.get('next') || options.afterSignInUrl || '/web/mainContent.html';
        const loginPageUrl = options.loginUrl || '/web/scripts/login.html';

        // If we're already on the login page, the login.html handles everything
        if (window.location.pathname.includes('/login.html')) {
            TB.logger.debug('[User] Already on login page, login.html handles auth UI');
            return;
        }

        // Otherwise redirect to login page
        window.location.href = `${loginPageUrl}?next=${encodeURIComponent(redirectUrl)}`;
    },

    /**
     * Mount sign-up form or redirect to signup page.
     */
    async mountSignUp(elementOrSelector, options = {}) {
        const element = typeof elementOrSelector === 'string'
            ? document.querySelector(elementOrSelector)
            : elementOrSelector;

        if (!element) {
            TB.logger.error('[User] Element not found for sign-up mount');
            return;
        }

        const urlParams = new URLSearchParams(window.location.search);
        const redirectUrl = options.afterSignUpUrl || urlParams.get('next') || '/web/mainContent.html';
        const loginPageUrl = options.loginUrl || '/web/scripts/login.html';

        window.location.href = `${loginPageUrl}?mode=signup&next=${encodeURIComponent(redirectUrl)}`;
    },

    /**
     * Mount user button (profile/logout dropdown).
     * For Custom Auth, this creates a simple user menu.
     */
    async mountUserButton(elementOrSelector, options = {}) {
        const element = typeof elementOrSelector === 'string'
            ? document.querySelector(elementOrSelector)
            : elementOrSelector;

        if (!element) {
            TB.logger.error('[User] Element not found for user button mount');
            return;
        }

        const username = this.getUsername() || 'User';
        const imageUrl = this.getUserData('imageUrl');

        element.innerHTML = `
            <div class="tb-user-button" style="position: relative; display: inline-block;">
                <button class="tb-user-avatar" style="
                    display: flex; align-items: center; gap: 8px;
                    background: none; border: 1px solid var(--color-border, #d1d5db);
                    border-radius: 9999px; padding: 4px 12px 4px 4px; cursor: pointer;
                    color: var(--color-text, inherit); font-size: 14px;
                ">
                    ${imageUrl
                        ? `<img src="${imageUrl}" alt="${username}" style="width: 28px; height: 28px; border-radius: 50%;" />`
                        : `<span style="width: 28px; height: 28px; border-radius: 50%; background: var(--color-primary, #6366f1); color: white; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 12px;">${username.charAt(0).toUpperCase()}</span>`
                    }
                    <span>${username}</span>
                </button>
                <div class="tb-user-dropdown" style="
                    display: none; position: absolute; top: 100%; right: 0; margin-top: 4px;
                    background: var(--color-bg, white); border: 1px solid var(--color-border, #d1d5db);
                    border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    min-width: 160px; z-index: 1000; overflow: hidden;
                ">
                    <a href="/web/mainContent.html" style="display: block; padding: 10px 16px; text-decoration: none; color: var(--color-text, inherit);">Dashboard</a>
                    <hr style="margin: 0; border: none; border-top: 1px solid var(--color-border, #e5e7eb);" />
                    <button class="tb-user-signout" style="
                        display: block; width: 100%; text-align: left; padding: 10px 16px;
                        background: none; border: none; cursor: pointer;
                        color: var(--color-error, #ef4444); font-size: 14px;
                    ">Sign Out</button>
                </div>
            </div>
        `;

        // Toggle dropdown
        const avatarBtn = element.querySelector('.tb-user-avatar');
        const dropdown = element.querySelector('.tb-user-dropdown');
        avatarBtn?.addEventListener('click', () => {
            dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!element.contains(e.target)) {
                dropdown.style.display = 'none';
            }
        });

        // Sign out button
        const signOutBtn = element.querySelector('.tb-user-signout');
        signOutBtn?.addEventListener('click', async () => {
            await this.signOut();
            const afterSignOutUrl = options.afterSignOutUrl || '/web/scripts/login.html';
            window.location.href = afterSignOutUrl;
        });
    },

    unmountAll() {
        // No SDK components to unmount with Custom Auth
        TB.logger.debug('[User] unmountAll called (no-op for Custom Auth)');
    },

    // =================== Auth Data Setter ===================

    setAuthData({ token, refreshToken, userId, username, email }) {
        this._updateUserState({
            isAuthenticated: true,
            token: token,
            refreshToken: refreshToken || null,
            userId: userId,
            username: username,
            email: email || null
        });

        this._startTokenRefreshTimer();
        TB.events.emit('user:signedIn', { username, userId });
    }
};

export default user;
