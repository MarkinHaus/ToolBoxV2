// tbjs/core/userV2.js
// User Authentication & Session Management V2
// Version: 2.0.0 - WebAuthn Only, Auto-Refresh Tokens

import TB from '../index.js';
import * as crypto from './cryptoV2.js';

const INACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes
const USER_STATE_KEY = 'tbjs_user_session_v2';
const TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // Refresh 5 min before expiry

const defaultUserState = {
    isAuthenticated: false,
    username: null,
    userLevel: null,
    accessToken: null,
    refreshToken: null,
    userData: {},
    tokenExpiry: null,
};

class UserV2 {
    constructor() {
        this._state = { ...defaultUserState };
        this._lastActivity = Date.now();
        this._refreshTimer = null;
        this._activityListeners = [];

        // Load persisted state
        this._loadState();

        // Setup activity tracking
        this._setupActivityTracking();

        // Setup auto-refresh
        if (this._state.isAuthenticated) {
            this._scheduleTokenRefresh();
        }
    }

    // =================== State Management ===================

    _loadState() {
        try {
            const stored = localStorage.getItem(USER_STATE_KEY);
            if (stored) {
                const parsed = JSON.parse(stored);
                this._state = { ...defaultUserState, ...parsed };

                // Check if token is expired
                if (this._state.tokenExpiry && Date.now() > this._state.tokenExpiry) {
                    TB.logger?.warn('[UserV2] Stored token expired, clearing session');
                    this._clearState();
                }
            }
        } catch (error) {
            TB.logger?.error('[UserV2] Error loading state:', error);
            this._clearState();
        }
    }

    _saveState() {
        try {
            localStorage.setItem(USER_STATE_KEY, JSON.stringify(this._state));
        } catch (error) {
            TB.logger?.error('[UserV2] Error saving state:', error);
        }
    }

    _clearState() {
        this._state = { ...defaultUserState };
        localStorage.removeItem(USER_STATE_KEY);
        if (this._refreshTimer) {
            clearTimeout(this._refreshTimer);
            this._refreshTimer = null;
        }
    }

    _updateState(updates) {
        this._state = { ...this._state, ...updates };
        this._saveState();
        TB.state?.set('user', this._state);
    }

    // =================== Activity Tracking ===================

    _setupActivityTracking() {
        const events = ['mousedown', 'keydown', 'scroll', 'touchstart'];
        const updateActivity = () => {
            this._lastActivity = Date.now();
        };

        events.forEach(event => {
            window.addEventListener(event, updateActivity, { passive: true });
        });
    }

    _isUserActive() {
        return (Date.now() - this._lastActivity) < INACTIVITY_TIMEOUT;
    }

    // =================== Token Management ===================

    _parseJWT(token) {
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            }).join(''));
            return JSON.parse(jsonPayload);
        } catch (error) {
            TB.logger?.error('[UserV2] Error parsing JWT:', error);
            return null;
        }
    }

    _scheduleTokenRefresh() {
        if (this._refreshTimer) {
            clearTimeout(this._refreshTimer);
        }

        if (!this._state.accessToken || !this._state.tokenExpiry) {
            return;
        }

        const timeUntilExpiry = this._state.tokenExpiry - Date.now();
        const refreshTime = Math.max(0, timeUntilExpiry - TOKEN_REFRESH_THRESHOLD);

        TB.logger?.debug(`[UserV2] Scheduling token refresh in ${Math.round(refreshTime / 1000)}s`);

        this._refreshTimer = setTimeout(async () => {
            await this._refreshToken();
        }, refreshTime);
    }

    async _refreshToken() {
        if (!this._state.refreshToken) {
            TB.logger?.warn('[UserV2] No refresh token available');
            await this.logout(false);
            return;
        }

        if (!this._isUserActive()) {
            TB.logger?.warn('[UserV2] User inactive, logging out instead of refreshing');
            await this.logout(false);
            return;
        }

        try {
            TB.logger?.info('[UserV2] Refreshing access token');

            const result = await TB.api.request(
                'CloudM.AuthManagerV2',
                'refresh_token',
                { refresh_token: this._state.refreshToken },
                'POST'
            );

            if (result.success && result.data) {
                const { access_token, refresh_token } = result.data;
                const payload = this._parseJWT(access_token);

                this._updateState({
                    accessToken: access_token,
                    refreshToken: refresh_token || this._state.refreshToken,
                    tokenExpiry: payload.exp * 1000
                });

                this._scheduleTokenRefresh();
                TB.logger?.info('[UserV2] Token refreshed successfully');
            } else {
                TB.logger?.error('[UserV2] Token refresh failed:', result.message);
                await this.logout(false);
            }
        } catch (error) {
            TB.logger?.error('[UserV2] Token refresh error:', error);
            await this.logout(false);
        }
    }

    // =================== Registration ===================

    async signup(username, email, inviteCode = null, deviceLabel = 'My Device') {
        TB.logger?.info(`[UserV2] Starting signup for ${username}`);

        try {
            // Check WebAuthn availability
            if (!crypto.isWebAuthnAvailable()) {
                throw new Error('WebAuthn not supported in this browser');
            }

            // Step 1: Start registration
            const startResult = await TB.api.request(
                'CloudM.AuthManagerV2',
                'register_start',
                {
                    username,
                    email,
                    invite_code: inviteCode,
                    device_label: deviceLabel
                },
                'POST'
            );

            if (!startResult.success) {
                throw new Error(startResult.message || 'Registration start failed');
            }

            const { options, session_id } = startResult.data;

            // Step 2: Create WebAuthn credential
            TB.logger?.info('[UserV2] Creating WebAuthn credential');
            const credential = await crypto.registerWebAuthnCredential(options);

            // Step 3: Finish registration
            const finishResult = await TB.api.request(
                'CloudM.AuthManagerV2',
                'register_finish',
                {
                    username,
                    email,
                    session_id,
                    credential,
                    device_label: deviceLabel
                },
                'POST'
            );

            if (!finishResult.success) {
                throw new Error(finishResult.message || 'Registration finish failed');
            }

            // Update state with tokens
            const { access_token, refresh_token, user } = finishResult.data;
            const payload = this._parseJWT(access_token);

            this._updateState({
                isAuthenticated: true,
                username: user.username,
                userLevel: user.level,
                accessToken: access_token,
                refreshToken: refresh_token,
                userData: user,
                tokenExpiry: payload.exp * 1000
            });

            this._scheduleTokenRefresh();

            TB.logger?.info('[UserV2] Signup successful');
            return { success: true, user };

        } catch (error) {
            TB.logger?.error('[UserV2] Signup error:', error);
            return { success: false, message: error.message };
        }
    }

    // =================== Login ===================

    async login(username, deviceLabel = null) {
        TB.logger?.info(`[UserV2] Starting login for ${username}`);

        try {
            // Check WebAuthn availability
            if (!crypto.isWebAuthnAvailable()) {
                throw new Error('WebAuthn not supported in this browser');
            }

            // Step 1: Start authentication
            const startResult = await TB.api.request(
                'CloudM.AuthManagerV2',
                'login_start',
                { username },
                'POST'
            );

            if (!startResult.success) {
                throw new Error(startResult.message || 'Login start failed');
            }

            const { options, session_id } = startResult.data;

            // Step 2: Get WebAuthn assertion
            TB.logger?.info('[UserV2] Getting WebAuthn assertion');
            const credential = await crypto.authenticateWebAuthn(options);

            // Step 3: Finish authentication
            const finishResult = await TB.api.request(
                'CloudM.AuthManagerV2',
                'login_finish',
                {
                    username,
                    session_id,
                    credential
                },
                'POST'
            );

            if (!finishResult.success) {
                throw new Error(finishResult.message || 'Login finish failed');
            }

            // Update state with tokens
            const { access_token, refresh_token, user } = finishResult.data;
            const payload = this._parseJWT(access_token);

            this._updateState({
                isAuthenticated: true,
                username: user.username,
                userLevel: user.level,
                accessToken: access_token,
                refreshToken: refresh_token,
                userData: user,
                tokenExpiry: payload.exp * 1000
            });

            this._scheduleTokenRefresh();

            TB.logger?.info('[UserV2] Login successful');
            return { success: true, user };

        } catch (error) {
            TB.logger?.error('[UserV2] Login error:', error);
            return { success: false, message: error.message };
        }
    }

    // =================== Logout ===================

    async logout(redirect = true) {
        TB.logger?.info('[UserV2] Logging out');

        this._clearState();

        if (redirect && TB.router) {
            TB.router.navigate('/login');
        }

        return { success: true };
    }

    // =================== Getters ===================

    isAuthenticated() {
        return this._state.isAuthenticated && this._state.accessToken && !this._isTokenExpired();
    }

    _isTokenExpired() {
        if (!this._state.tokenExpiry) return true;
        return Date.now() > this._state.tokenExpiry;
    }

    getUsername() {
        return this._state.username;
    }

    getUserLevel() {
        return this._state.userLevel;
    }

    getAccessToken() {
        return this._state.accessToken;
    }

    getUserData() {
        return this._state.userData;
    }

    getState() {
        return { ...this._state };
    }
}

// Export singleton instance
export default new UserV2();

