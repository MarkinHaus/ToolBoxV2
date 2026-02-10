/**
 * CustomAuth - Custom Authentication Module for ToolBoxV2
 * Replaces Clerk SDK with provider-agnostic auth system
 *
 * Supports: Discord OAuth, Google OAuth, Passkeys (WebAuthn)
 * Integrates with TBJS (TB.api, TB.events) when available
 *
 * @module CustomAuth
 * @version 2.0.0
 */

class CustomAuth {
    constructor() {
        // Provider configuration
        this.providers = {
            discord: {
                enabled: true,
                loginUrl: '/auth/discord/url',
                color: '#5865F2'
            },
            google: {
                enabled: true,
                loginUrl: '/auth/google/url',
                color: '#4285F4'
            },
            passkey: {
                enabled: typeof window.PublicKeyCredential !== 'undefined',
                supported: this._checkPasskeySupport()
            }
        };

        // Token storage
        this.token = localStorage.getItem('tb_auth_token');
        this.refreshToken = localStorage.getItem('tb_auth_refresh_token');
        this.user = this._loadUser();

        // Event listeners
        this.listeners = {
            'user:signedIn': [],
            'user:signedOut': [],
            'user:authError': [],
            'user:stateChanged': []
        };

        // Auto-refresh timer
        this.refreshTimer = null;
        this._startAutoRefresh();

        // OAuth callback handler
        this._handleOAuthCallback();
    }

    /**
     * Check if WebAuthn Passkeys are supported
     */
    _checkPasskeySupport() {
        if (typeof window.PublicKeyCredential === 'undefined') {
            return false;
        }
        return window.PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable?.() ?? true;
    }

    /**
     * Load user from localStorage
     */
    _loadUser() {
        try {
            const userJson = localStorage.getItem('tb_auth_user');
            return userJson ? JSON.parse(userJson) : null;
        } catch (e) {
            console.error('Failed to load user:', e);
            return null;
        }
    }

    /**
     * Save user to localStorage
     */
    _saveUser() {
        try {
            localStorage.setItem('tb_auth_user', JSON.stringify(this.user));
        } catch (e) {
            console.error('Failed to save user:', e);
        }
    }

    /**
     * Emit event to all registered listeners + TB.events if available
     */
    _emit(eventName, data = {}) {
        if (this.listeners[eventName]) {
            this.listeners[eventName].forEach(callback => {
                try {
                    callback(data);
                } catch (e) {
                    console.error(`Error in event listener for ${eventName}:`, e);
                }
            });
        }
        // Integrate with TBJS event system if available
        if (typeof TB !== 'undefined' && TB.events && TB.events.emit) {
            try {
                TB.events.emit(eventName, data);
            } catch (e) { /* TBJS not ready */ }
        }
    }

    /**
     * Handle OAuth callback from Discord/Google
     * Checks URL parameters for auth token
     */
    _handleOAuthCallback() {
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');
        const refreshToken = urlParams.get('refresh_token');

        if (token) {
            this.token = token;
            this.refreshToken = refreshToken || this.refreshToken;
            this.save();

            // Fetch user data
            this._fetchUserData().then(() => {
                this._emit('user:signedIn', { user: this.user });
                this._emit('user:stateChanged', { user: this.user });

                // Clean URL
                window.history.replaceState({}, document.title, window.location.pathname);
            }).catch(error => {
                this._emit('user:authError', { error });
            });
        }
    }

    /**
     * Fetch user data from backend
     */
    async _fetchUserData() {
        try {
            const response = await fetch('/api/CloudM.Auth/get_user_data', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const json = await response.json();
            // API returns {error: 0, result: {...}, info: {...}}
            const data = json.result || json;
            this.user = data;
            this._saveUser();
        } catch (error) {
            console.error('Failed to fetch user data:', error);
            throw error;
        }
    }

    /**
     * Start auto-refresh timer (every 14 minutes)
     */
    _startAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }

        this.refreshTimer = setInterval(() => {
            if (this.token && this.refreshToken) {
                this.refreshAccessToken().catch(error => {
                    console.error('Auto-refresh failed:', error);
                    this._emit('user:authError', { error });
                });
            }
        }, 14 * 60 * 1000);
    }

    // ==================== OAUTH METHODS ====================

    /**
     * Login with Discord OAuth
     */
    async loginWithDiscord() {
        if (!this.providers.discord.enabled) {
            throw new Error('Discord OAuth is not enabled');
        }

        try {
            const response = await fetch(this.providers.discord.loginUrl);
            const json = await response.json();
            const authUrl = json.result?.auth_url || json.auth_url;
            if (!authUrl) throw new Error('No auth URL returned');
            window.location.href = authUrl;
        } catch (error) {
            console.error('Failed to get Discord auth URL:', error);
            throw error;
        }
    }

    /**
     * Login with Google OAuth
     */
    async loginWithGoogle() {
        if (!this.providers.google.enabled) {
            throw new Error('Google OAuth is not enabled');
        }

        try {
            const response = await fetch(this.providers.google.loginUrl);
            const json = await response.json();
            const authUrl = json.result?.auth_url || json.auth_url;
            if (!authUrl) throw new Error('No auth URL returned');
            window.location.href = authUrl;
        } catch (error) {
            console.error('Failed to get Google auth URL:', error);
            throw error;
        }
    }

    // ==================== PASSKEY METHODS ====================

    /**
     * Login with WebAuthn Passkey
     */
    async loginWithPasskey(username = null) {
        if (!this.providers.passkey.enabled || !this.providers.passkey.supported) {
            throw new Error('Passkeys are not supported on this device');
        }

        try {
            const response = await fetch('/api/CloudM.Auth/passkey_login_start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const json = await response.json();
            const options = json.result || json;

            // Convert base64 to ArrayBuffer
            options.challenge = this._base64ToArrayBuffer(options.challenge);
            if (options.allowCredentials) {
                options.allowCredentials = options.allowCredentials.map(cred => ({
                    ...cred,
                    id: this._base64ToArrayBuffer(cred.id)
                }));
            }

            const credential = await navigator.credentials.get({
                publicKey: options
            });

            const finishResponse = await fetch('/api/CloudM.Auth/passkey_login_finish', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    credential: this._credentialToJSON(credential),
                    username
                })
            });

            if (!finishResponse.ok) {
                throw new Error(`HTTP ${finishResponse.status}`);
            }

            const finishJson = await finishResponse.json();
            const data = finishJson.result || finishJson;
            this.token = data.access_token || data.token;
            this.refreshToken = data.refresh_token;
            this.user = data.user || data;
            this.save();

            this._emit('user:signedIn', { user: this.user });
            this._emit('user:stateChanged', { user: this.user });

            return data;
        } catch (error) {
            console.error('Passkey login failed:', error);
            this._emit('user:authError', { error });
            throw error;
        }
    }

    /**
     * Register a new Passkey for the current user
     */
    async registerPasskey(username, displayName) {
        if (!this.providers.passkey.enabled || !this.providers.passkey.supported) {
            throw new Error('Passkeys are not supported on this device');
        }

        try {
            const response = await fetch('/api/CloudM.Auth/passkey_register_start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({ username, displayName })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const json = await response.json();
            const options = json.result || json;

            options.challenge = this._base64ToArrayBuffer(options.challenge);
            options.user.id = this._base64ToArrayBuffer(options.user.id);

            if (options.excludeCredentials) {
                options.excludeCredentials = options.excludeCredentials.map(cred => ({
                    ...cred,
                    id: this._base64ToArrayBuffer(cred.id)
                }));
            }

            const credential = await navigator.credentials.create({
                publicKey: options
            });

            const finishResponse = await fetch('/api/CloudM.Auth/passkey_register_finish', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({
                    credential: this._credentialToJSON(credential),
                    username,
                    displayName
                })
            });

            if (!finishResponse.ok) {
                throw new Error(`HTTP ${finishResponse.status}`);
            }

            const finishJson = await finishResponse.json();
            return finishJson.result || finishJson;
        } catch (error) {
            console.error('Passkey registration failed:', error);
            this._emit('user:authError', { error });
            throw error;
        }
    }

    /**
     * Convert WebAuthn credential to JSON (base64)
     */
    _credentialToJSON(credential) {
        return {
            id: credential.id,
            rawId: this._arrayBufferToBase64(credential.rawId),
            type: credential.type,
            response: {
                clientDataJSON: this._arrayBufferToBase64(credential.response.clientDataJSON),
                attestationObject: credential.response.attestationObject ?
                    this._arrayBufferToBase64(credential.response.attestationObject) : null,
                authenticatorData: credential.response.authenticatorData ?
                    this._arrayBufferToBase64(credential.response.authenticatorData) : null,
                signature: credential.response.signature ?
                    this._arrayBufferToBase64(credential.response.signature) : null,
                userHandle: credential.response.userHandle ?
                    this._arrayBufferToBase64(credential.response.userHandle) : null
            }
        };
    }

    _base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    _arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    // ==================== TOKEN METHODS ====================

    async getAccessToken() {
        return this.token;
    }

    async getRefreshToken() {
        return this.refreshToken;
    }

    async refreshAccessToken() {
        if (!this.refreshToken) {
            throw new Error('No refresh token available');
        }

        try {
            const response = await fetch('/api/CloudM.Auth/refresh_token', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: this.refreshToken })
            });

            if (!response.ok) {
                if (response.status === 401) {
                    this.logout();
                    throw new Error('Session expired');
                }
                throw new Error(`HTTP ${response.status}`);
            }

            const json = await response.json();
            const data = json.result || json;
            this.token = data.access_token || data.token;
            this.refreshToken = data.refresh_token || this.refreshToken;
            this.save();

            return this.token;
        } catch (error) {
            console.error('Token refresh failed:', error);
            this._emit('user:authError', { error });
            throw error;
        }
    }

    // ==================== USER METHODS ====================

    getUser() {
        return this.user;
    }

    isAuthenticated() {
        return !!this.token && !!this.user;
    }

    async updateUserData(updates) {
        try {
            const response = await fetch('/api/CloudM.Auth/update_user_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify(updates)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const json = await response.json();
            this.user = { ...this.user, ...updates };
            this._saveUser();
            this._emit('user:stateChanged', { user: this.user });

            return json.result || json;
        } catch (error) {
            console.error('Failed to update user data:', error);
            throw error;
        }
    }

    // ==================== LOGOUT ====================

    async logout() {
        try {
            if (this.token) {
                await fetch('/api/CloudM.Auth/logout', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.token}`
                    }
                });
            }
        } catch (error) {
            console.error('Logout request failed:', error);
        } finally {
            this.clear();
            this._emit('user:signedOut');
        }
    }

    clear() {
        localStorage.removeItem('tb_auth_token');
        localStorage.removeItem('tb_auth_refresh_token');
        localStorage.removeItem('tb_auth_user');
        this.token = null;
        this.refreshToken = null;
        this.user = null;

        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    save() {
        if (this.token) localStorage.setItem('tb_auth_token', this.token);
        if (this.refreshToken) localStorage.setItem('tb_auth_refresh_token', this.refreshToken);
        this._saveUser();
    }

    // ==================== EVENT SYSTEM ====================

    on(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event].push(callback);
        }
    }

    off(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        }
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CustomAuth;
} else {
    window.CustomAuth = CustomAuth;
}
