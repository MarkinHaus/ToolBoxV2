// tbjs/core/user.js
// ToolBox V2 User Management with Custom Auth (CloudM.Auth)

import TB from '../index.js';

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

            const response = await fetch('/validateSession', {
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

    async init(forceServerFetch = false) {
        if (this._initPromise) {
            return this._initPromise;
        }
        this._initPromise = this._doInit(forceServerFetch);
        return this._initPromise;
    },

    async _doInit(forceServerFetch = false) {
        this._initActivityMonitor();
        TB.logger.info('[User] Initializing with Custom Auth...');

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

        // Check if we have a stored token and validate it
        if (mergedState.token && mergedState.isAuthenticated) {
            const isValid = await this.validateBackendSession();
            if (isValid) {
                TB.logger.info('[User] Restored valid session from storage.');
                authInitialized = true;
                this._startTokenRefreshTimer();
                TB.events.emit('user:signedIn', {
                    username: mergedState.username,
                    userId: mergedState.userId
                });
            } else {
                TB.logger.warn('[User] Stored session invalid, clearing.');
                this._updateUserState({}, true);
            }
        }

        // Check URL for auth callback tokens (OAuth redirect)
        await this._checkAuthCallback();

        // Listen for state changes to persist
        TB.events.on('state:changed:user', (newState) => {
            try {
                localStorage.setItem(USER_STATE_KEY, JSON.stringify(newState));
                localStorage.setItem(USER_DATA_TIMESTAMP_KEY, Date.now().toString());
            } catch (e) {
                TB.logger.error('[User] Failed to save user session:', e);
            }
        });

        authInitialized = true;
        return true;
    },

    /**
     * Check URL for auth callback parameters (after OAuth redirect).
     * Handles ?token=xxx&refresh_token=xxx from OAuth callbacks.
     */
    async _checkAuthCallback() {
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');
        const refreshToken = urlParams.get('refresh_token');

        if (!token) return;

        TB.logger.info('[User] Auth callback detected, processing tokens...');

        try {
            // Clean up URL (remove token params)
            const cleanUrl = new URL(window.location.href);
            cleanUrl.searchParams.delete('token');
            cleanUrl.searchParams.delete('refresh_token');
            window.history.replaceState({}, '', cleanUrl.toString());

            // Validate the token
            const response = await fetch('/api/CloudM.Auth/validate_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ token: token })
            });

            if (!response.ok) {
                throw new Error(`Validation failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.error !== "none" || !result.result?.data?.authenticated) {
                throw new Error('Token validation rejected');
            }

            const userData = result.result.data;

            // Fetch full user data
            const userDataResponse = await fetch('/api/CloudM.Auth/get_user_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ user_id: userData.user_id })
            });

            let settings = {};
            let userLevel = 1;
            let modData = {};

            if (userDataResponse.ok) {
                const userDataResult = await userDataResponse.json();
                if (userDataResult.error === "none" && userDataResult.result?.data) {
                    const data = userDataResult.result.data;
                    settings = data.settings || {};
                    userLevel = data.level || 1;
                    modData = data.mod_data || {};
                }
            }

            // Update state
            this._updateUserState({
                isAuthenticated: true,
                username: userData.username || userData.user_name || '',
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
            });

            TB.logger.info('[User] Auth callback processed successfully');
            TB.events.emit('user:signedIn', {
                username: userData.username,
                userId: userData.user_id
            });

            this._startTokenRefreshTimer();
            this._handlePostAuthRedirect();

        } catch (e) {
            TB.logger.error('[User] Auth callback failed:', e);
            await this._handleAuthFailure(e.message);
        }
    },

    _updateUserState(updates, clearExisting = false) {
        const currentState = clearExisting ? defaultUserState : (TB.state.get('user') || defaultUserState);
        const newState = { ...currentState, ...updates };
        TB.state.set('user', newState);
        TB.events.emit('user:stateChanged', newState);
        return newState;
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

        const userId = this.getUserId();
        const token = this.getToken();

        // Notify backend
        if (userId && token) {
            try {
                await fetch('/api/CloudM.Auth/logout', {
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
        TB.logger.debug('[User] Refreshing session token...');

        const refreshToken = TB.state.get('user.refreshToken');
        const currentToken = this.getToken();

        if (!refreshToken && !currentToken) {
            TB.logger.warn('[User] No tokens available for refresh');
            throw new Error('No active session');
        }

        try {
            const response = await fetch('/api/CloudM.Auth/refresh_token', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    refresh_token: refreshToken,
                    token: currentToken
                })
            });

            if (!response.ok) {
                throw new Error(`Refresh failed: ${response.status}`);
            }

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
                if (this._isUserActive() && this.getToken()) {
                    // Validate first, refresh if needed
                    const isValid = await this.validateBackendSession();
                    if (!isValid) {
                        await this._refreshToken();
                    }
                }
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
            const response = await fetch('/api/CloudM.Auth/update_user_data', {
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

            const response = await fetch('/api/CloudM.Auth/update_user_data', {
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
            const response = await fetch('/api/CloudM.Auth/get_user_data', {
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
