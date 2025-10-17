// tbjs/core/user.js
import TB from '../index.js';
import * as crypto from './crypto.js';


const INACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes
const USER_STATE_KEY = 'tbjs_user_session';
const USER_DATA_TIMESTAMP_KEY = 'tbjs_user_data_timestamp';

const defaultUserState = {
    isAuthenticated: false,
    username: null,
    userLevel: null,
    token: null,
    isDeviceRegisteredWithKey: false,
    userData: {},
};

const user = {
    _isRefreshing: false,
    _refreshPromise: null,
    _lastActivityTimestamp: Date.now(),

    _initActivityMonitor() {
        const activityEvents = ['mousemove', 'keydown', 'scroll', 'click'];
        const handler = () => {
            this._lastActivityTimestamp = Date.now();
        };
        activityEvents.forEach(event => document.addEventListener(event, handler, { passive: true }));
        TB.logger.info('[User] Activity monitor initialized.');
    },

    _isUserActive() {
        return (Date.now() - this._lastActivityTimestamp) < INACTIVITY_TIMEOUT;
    },

    async _refreshToken() {
        // Safety Guard: If a refresh is already in progress, wait for it to complete.
        if (this._isRefreshing) {
            TB.logger.debug('[User] Waiting for existing session refresh to complete...');
            return this._refreshPromise;
        }

        // Safety Guard: If user is inactive, log them out instead of refreshing.
        if (!this._isUserActive()) {
            TB.logger.warn('[User] User inactive. Logging out instead of refreshing session.');
            await this.logout(false);
            throw new Error("User inactive, session expired.");
        }

        this._isRefreshing = true;
        this._refreshPromise = new Promise(async (resolve, reject) => {
            try {
                TB.logger.info('[User] Session expired. Attempting silent refresh...');
                const username = this.getUsername();
                if (!username) {
                    throw new Error("Cannot refresh session without a username.");
                }

                const privateKeyBase64 = await crypto.retrievePrivateKey(username);
                if (!privateKeyBase64) {
                    throw new Error(`No private key found for ${username}. Cannot refresh session.`);
                }

                // This logic is identical to loginWithDeviceKey, but silent.
                const challengeResult = await TB.api.request('/CloudM.AuthManager/get_to_sing_data', `username=${username}&personal_key=false`, {}, 'GET');
                if (challengeResult.error !== TB.ToolBoxError.none || !challengeResult.get()?.challenge) {
                    throw new Error(challengeResult.info.help_text || "Failed to get refresh challenge.");
                }
                const challenge = challengeResult.get().challenge;

                const signature = await crypto.signMessage(privateKeyBase64, challenge);
                const validationPayload = { username, signature };
                const validationResult = await TB.api.request('CloudM.AuthManager', 'validate_device', validationPayload, 'POST');

                if (validationResult.error === TB.ToolBoxError.none && validationResult.get()) {
                    const responseData = validationResult.get();
                    let token = responseData.key;
                    if (responseData.toPrivat && token) {
                        token = await crypto.decryptAsymmetric(token, privateKeyBase64, true);
                    }

                    this._updateUserState({ token: token }); // Only update the token
                    TB.logger.info('[User] Session silently refreshed.');
                    resolve(token);
                } else {
                    throw new Error(validationResult.info.help_text || "Silent refresh validation failed.");
                }
            } catch (error) {
                TB.logger.error('[User] Silent session refresh failed. Logging out.', error);
                await this.logout(false);
                reject(error);
            } finally {
                this._isRefreshing = false;
                this._refreshPromise = null;
            }
        });
        return this._refreshPromise;
    },

    async init(forceServerFetch = false) {
        this._initActivityMonitor();
        TB.logger.info('[User] Initializing...');
        let initialState = TB.state.get('user');

        if (!initialState || Object.keys(initialState).length === 0) {
            try {
                const storedSession = localStorage.getItem(USER_STATE_KEY);
                if (storedSession) {
                    initialState = JSON.parse(storedSession);
                    TB.logger.debug('[User] Loaded session from localStorage.');
                }
            } catch (e) {
                TB.logger.warn('[User] Could not parse stored session from localStorage.', e);
                localStorage.removeItem(USER_STATE_KEY);
                initialState = null;
            }
        }

        const mergedState = { ...defaultUserState, ...(initialState || {}) };
        TB.state.set('user', mergedState);

        if (mergedState.isAuthenticated && mergedState.token) {
            TB.logger.info(`[User] Found existing session for ${mergedState.username}. Validating...`);
            try {
                const result = await TB.api.AuthHttpPostData(mergedState.username);

                if (result.error === TB.ToolBoxError.none) {
                    TB.logger.info(`[User] Session for ${mergedState.username} is valid.`);
                    // Existing data sync logic...
                } else {
                    // This is where the session has likely expired.
                    TB.logger.warn(`[User] Initial session validation failed. Attempting refresh...`, result.info.help_text);
                    await this._refreshToken();
                }
            } catch (error) {
                // This catch is for network errors or if _refreshToken itself fails
                TB.logger.error('[User] Error validating or refreshing session during init.', error);
                // The logout is handled inside _refreshToken, so no need to call it again here.
            }
        } else {
            TB.logger.info('[User] No active session found or token missing.');
        }

        TB.events.on('state:changed:user', (newState) => {
            try {
                localStorage.setItem(USER_STATE_KEY, JSON.stringify(newState));
                localStorage.setItem(USER_DATA_TIMESTAMP_KEY, Date.now().toString());
            } catch (e) {
                TB.logger.error('[User] Failed to save user session to localStorage:', e);
            }
        });
    },

    _updateUserState(updates, clearExisting = false) {
        const currentState = clearExisting ? defaultUserState : (TB.state.get('user') || defaultUserState);
        const preservedFields = clearExisting ? {} : {
            isAuthenticated: currentState.isAuthenticated,
            username: currentState.username,
            token: currentState.token
        };
        const newState = { ...preservedFields, ...currentState, ...updates };

        TB.state.set('user', newState);
        TB.events.emit('user:stateChanged', newState);
        return newState;
    },

    // --- Authentication Methods ---

    async signup(username, email, initiationKey, registerAsPersona = false) {
        TB.logger.info(`[User] Attempting signup for ${username}`);
        try {
            const keys = await crypto.generateAsymmetricKeys();
            await crypto.storePrivateKey(keys.privateKey_base64, username);
            await crypto.storePublicKey(keys.publicKey_base64, username);
            // This is a placeholder for your actual signup logic
            // It likely involves one or two API calls.
            const payload = {
                name: username,
                email: email,
                pub_key: keys.publicKey,//initiation_key: initiationKey,
                invitation: initiationKey, // registerAsPersona
                web_data: true,
                as_base64: false
            };
            const result = await TB.api.request('CloudM.AuthManager', 'create_user', payload, 'POST'); // Adjust endpoint

            if (result.error === TB.ToolBoxError.none && result.get()) {
                // 2. Perform WebAuthn registration

                const { challenge, userId ,...rest } = result.get();
                // The 'sing' parameter's purpose from original cryp.js unclear, may need token or specific server data.
                // For adding a credential to an existing user, 'sing' might be the current session token.
                const registrationPayload = await crypto.registerWebAuthnCredential({ challenge, userId, username },
                await crypto.signMessage(keys.privateKey_base64, challenge));
                // 3. Send new WebAuthn credential to server
                const _result = await TB.api.request('CloudM.AuthManager', 'register_user_personal_key', registrationPayload, 'POST');
                if (_result.error === TB.ToolBoxError.none) {
                    TB.logger.info(`[User] WebAuthn (Persona) registration successful for ${username}`);
                    return await this.loginWithDeviceKey(username, keys.privateKey_base64);
                } else {
                    return { success: false, message: _result.info.help_text || "Failed to register WebAuthn credential." };
                }
            } else {
                TB.logger.warn(`[User] Signup failed for ${username}: ${result.info.help_text}`);
                return { success: false, message: result.info.help_text || "Signup failed." };
            }
        } catch (error) {
            TB.logger.error(`[User] Signup error for ${username}:`, error);
            return { success: false, message: error.message || "An unexpected error occurred during signup." };
        }
    },

    async loginWithDeviceKey(username, privateKeyBase64=false) {
        TB.logger.info(`[User] Attempting device key login for ${username}`);
        try {
            if (!privateKeyBase64){
                privateKeyBase64 = await crypto.retrievePrivateKey(username);
            }
            this._updateUserState({
                        isAuthenticated: false,
                        username: username,
                        userLevel: 0,
                        token: null,
                        isDeviceRegisteredWithKey: false,
                        userData: {}
                    }, true);
            if (!privateKeyBase64) {
                return { success: false, message: `No device key found for ${username}. Consider registering this device or using another login method.` };
            }

            // 1. Get challenge from server
            const challengeResult = await TB.api.request('/CloudM.AuthManager/get_to_sing_data', 'username='+username+'&personal_key=false', {}, 'GET');
            if (challengeResult.error !== TB.ToolBoxError.none || !challengeResult.get()?.challenge) {
                return { success: false, message: challengeResult.info.help_text || "Failed to get login challenge." };
            }
            const challenge = challengeResult.get().challenge;

            // 2. Sign challenge
            const signature = await crypto.signMessage(privateKeyBase64, challenge);

            // 3. Validate signature with server
            const validationPayload = { username, signature };
            const validationResult = await TB.api.request('CloudM.AuthManager', 'validate_device', validationPayload, 'POST');

            if (validationResult.error === TB.ToolBoxError.none && validationResult.get()) {
                const responseData = validationResult.get();
                let token = responseData.key;
                if (responseData.toPrivat && token) { // Assuming 'toPrivat' means the key is encrypted
                    token = await crypto.decryptAsymmetric(token, privateKeyBase64, true);
                }
                this._updateUserState({
                    isAuthenticated: false,
                    username: responseData.username || username,
                    userLevel: responseData.userLevel || 0,
                    token: token,
                    isDeviceRegisteredWithKey: false,
                    userData: responseData.userData || {}
                }, true);
                const validationResult_ = await window.TB.api.AuthHttpPostData(username);
                if (validationResult_.error === TB.ToolBoxError.none) {
                    this._updateUserState({
                        isAuthenticated: true,
                        username: responseData.username || username,
                        userLevel: responseData.userLevel || 0,
                        token: token,
                        isDeviceRegisteredWithKey: true,
                        userData: responseData.userData || {}
                    }, true);
                    TB.logger.info(`[User] Device key login successful for ${username}`);
                    return {success: true, message: "Login successful.", data: responseData};
                } else {
                    localStorage.removeItem("tb_pk_"+username)
                return { success: false, message: validationResult.info.help_text || "Device key login failed." };
            }
            } else {
                localStorage.removeItem("tb_pk_"+username)
                return { success: false, message: validationResult.info.help_text || "Device key login failed." };
            }
        } catch (error) {
            localStorage.removeItem("tb_pk_"+username)
            TB.logger.error(`[User] Device key login error for ${username}:`, error);
            return { success: false, message: error.message || "An unexpected error occurred." };
        }
    },

    async loginWithWebAuthn(username) {
        TB.logger.info(`[User] Attempting WebAuthn login for ${username}`);
        try {
            // 1. Get challenge and credential ID (rawId) from server
            // The original 'get_to_sing_data' with personal_key=true needs to return 'rowId' (rawId) and 'challenge'
            const challengeResult = await TB.api.request('/CloudM.AuthManager/get_to_sing_data', 'username='+username+'&personal_key=true', {}, 'GET');
            if (challengeResult.error !== TB.ToolBoxError.none || !challengeResult.get()?.challenge) {
                return { success: false, message: challengeResult.info.help_text || "Failed to get WebAuthn login challenge." };
            }
            const { rowId, challenge } = challengeResult.get();

            // 2. Perform WebAuthn assertion
            const assertionPayload = await crypto.authorizeWebAuthnCredential(rowId, challenge, username);

            // 3. Send assertion to server for validation
            const validationResult = await TB.api.request('CloudM.AuthManager', 'validate_persona', assertionPayload, 'POST');
            if (validationResult.error === TB.ToolBoxError.none && validationResult.get()) {
                const responseData = validationResult.get();
                await window.TB.router.navigateTo(responseData);
                return { success: true, message: "WebAuthn login successful.", data: responseData };
            } else {
                return { success: false, message: validationResult.info.help_text || "WebAuthn login failed." };
            }
        } catch (error) {
            TB.logger.error(`[User] WebAuthn login error for ${username}:`, error);
            if (error.name === 'NotAllowedError') {
                 return { success: false, message: "WebAuthn operation was cancelled or not allowed." };
            }
            return { success: false, message: error.message || "An unexpected error occurred during WebAuthn login." };
        }
    },

    async requestMagicLink(username) {
        TB.logger.info(`[User] Requesting magic link for ${username}`);
        try {
            const result = await TB.api.request('/CloudM.AuthManager/get_magic_link_email', 'username='+username, {}, 'GET');
            if (result.error === TB.ToolBoxError.none) {
                TB.logger.info(`[User] Magic link request successful for ${username}.`);
                return { success: true, message: result.info.help_text || "Magic link email sent." };
            } else {
                return { success: false, message: result.info.help_text || "Failed to send magic link." };
            }
        } catch (error) {
            TB.logger.error(`[User] Magic link request error for ${username}:`, error);
            return { success: false, message: error.message || "An unexpected error occurred." };
        }
    },

    // For m_link.js functionality: this is device registration via magic link invitation
    async registerDeviceWithInvitation(username, invitationKey) {
        TB.logger.info(`[User] Registering device for ${username} with invitation key.`);
        try {
            const keys = await crypto.generateAsymmetricKeys();
            await crypto.storePrivateKey(keys.privateKey_base64, username);
            await crypto.storePublicKey(keys.publicKey_base64, username);
            const payload = {
                name: username,
                pub_key: keys.publicKey, // PEM format from generateAsymmetricKeys
                invitation: invitationKey,
                web_data: true,
                as_base64: false // Assuming server expects PEM
            };
            const result = await TB.api.request('CloudM.AuthManager', 'add_user_device', payload, 'POST');

            if (result.error === TB.ToolBoxError.none && result.get()) {
                const responseData = result.get();
                // The response dSync was decrypted in original code, assuming it's part of device setup
                // If it's a session key or similar, handle it.
                if (responseData.dSync) {
                    const deviceSpecificData = await crypto.decryptAsymmetric(responseData.dSync, keys.privateKey_base64, true);
                    TB.logger.info("[User] Decrypted dSync data:", deviceSpecificData);
                    // Store or use deviceSpecificData as needed, e.g., in TB.state.set('user.deviceSyncKey', deviceSpecificData);
                }

                TB.logger.info(`[User] Device registration successful for ${username}. Now attempting login.`);
                // After successful device registration, typically log in the user with this new device.
                return this.loginWithDeviceKey(username, keys.privateKey_base64);
            } else {
                return { success: false, message: result.info.help_text || "Device registration via invitation failed." };
            }
        } catch (error) {
            TB.logger.error(`[User] Error registering device with invitation for ${username}:`, error);
            return { success: false, message: error.message || "An unexpected error occurred." };
        }
    },

    async registerWebAuthnForCurrentUser(username, invitationKey) {
        TB.logger.info(`[User] Attempting to register WebAuthn (Persona) for current user ${username}`);
        if (!this.isAuthenticated() || this.getUsername() !== username) {
            return { success: false, message: "User must be logged in to register a WebAuthn credential." };
        }
        try {
            // 1. Get WebAuthn registration challenge from server
            // This endpoint needs to be specific for adding a new WebAuthn credential to an *existing, authenticated* user.
            // The old 'create_user_with_init_key_personal' might not be it.
            // Let's assume an endpoint 'getWebAuthnRegistrationChallengeForUser'
            let publicKeyBase64 = await crypto.retrievePublicKey(username)
            let privateKeyBase64 = await crypto.retrievePrivateKey(username);
            if (!privateKeyBase64){
                const keys = await crypto.generateAsymmetricKeys();
                await crypto.storePrivateKey(keys.privateKey_base64, username);
                await crypto.storePublicKey(keys.publicKey_base64, username);
                privateKeyBase64 = keys.privateKey_base64;
                publicKeyBase64 = keys.publicKey_base64;
            }
            const payload = {
                name: username,
                pub_key: publicKeyBase64, // PEM format from generateAsymmetricKeys
                invitation: invitationKey,
                web_data: true,
                as_base64: false // Assuming server expects PEM
            };
            const challengeRes = await TB.api.request('CloudM.AuthManager', 'add_user_device', payload, 'POST');

            if (challengeRes.error !== TB.ToolBoxError.none || !challengeRes.get()?.challenge) {
                return { success: false, message: challengeRes.info.help_text || "Failed to get WebAuthn registration challenge."};
            }
            const { challenge, userId, ...rest } = challengeRes.get(); // Server provides challenge and its internal userId for WebAuthn

            const registrationPayload = await crypto.registerWebAuthnCredential({ challenge, userId, username },
                await crypto.signMessage(privateKeyBase64, challenge));

            // 3. Send new WebAuthn credential to server
            const result = await TB.api.request('CloudM.AuthManager', 'register_user_personal_key', registrationPayload, 'POST');
            if (result.error === TB.ToolBoxError.none) {
                TB.logger.info(`[User] WebAuthn (Persona) registration successful for ${username}`);
                return { success: true, message: result.info.help_text || "WebAuthn credential registered." };
            } else {
                return { success: false, message: result.info.help_text || "Failed to register WebAuthn credential." };
            }

        } catch (error) {
            TB.logger.error(`[User] WebAuthn registration error for ${username}:`, error);
            if (error.name === 'NotAllowedError') {
                 return { success: false, message: "WebAuthn operation was cancelled or not allowed." };
            }
            return { success: false, message: error.message || "An unexpected error occurred." };
        }
    },

    async checkSessionValidity() {
        if (!this.isAuthenticated() || !this.getToken()) return false;
        try {
            const result = await TB.api.request('/IsValidSession', null, { token: this.getToken() }, 'POST', 'never', true);
            return result.error === TB.ToolBoxError.none;
        } catch (e) {
            TB.logger.error("[User] Error checking session validity", e);
            return false;
        }
    },

    async logout(notifyServer = true) {
        const currentUser = TB.state.get('user');
        TB.logger.info(`[User] Logging out ${currentUser.username || 0}. Notify server: ${notifyServer}`);
        if (notifyServer && currentUser.isAuthenticated && currentUser.token) {
            try {
                // Use the new TB.api.logoutServer()
                const logoutResult = await TB.api.logoutServer();
                if (logoutResult.error !== TB.ToolBoxError.none) {
                     TB.logger.warn('[User] Server logout returned an error/warning:', logoutResult.info.help_text);
                } else {
                     TB.logger.info('[User] Server logout acknowledged.');
                }
            } catch (error) {
                TB.logger.warn('[User] Error notifying server of logout:', error);
            }
        }
        this._updateUserState({}, true); // Reset to defaultUserState
        localStorage.removeItem(USER_STATE_KEY);
        localStorage.removeItem(USER_DATA_TIMESTAMP_KEY);
        TB.events.emit('user:loggedOut');
    },

    // --- Getters for user state ---
    isAuthenticated: () => TB.state.get('user.isAuthenticated') || false,
    getUsername: () => TB.state.get('user.username') || null,
    getUserLevel: () => TB.state.get('user.userLevel') || null,
    getToken: () => TB.state.get('user.token') || null,
    isDeviceRegisteredWithKey: () => TB.state.get('user.isDeviceRegisteredWithKey') || false,

    // --- User-specific data management ---
    getUserData(key) {
        return TB.state.get(`user.userData.${key}`);
    },
    setUserData(keyOrObject, value, syncToServer = false) {
        let updates;
        if (typeof keyOrObject === 'string') {
            updates = { [keyOrObject]: value };
            TB.state.set(`user.userData.${keyOrObject}`, value);
        } else if (typeof keyOrObject === 'object') {
            updates = keyOrObject;
            for (const k in updates) {
                TB.state.set(`user.userData.${k}`, updates[k]);
            }
        } else {
            TB.logger.warn('[User] Invalid setUserData arguments.');
            return;
        }

        if (syncToServer && this.isAuthenticated()) {
            // Debounce this in a real app
            this.syncUserData(updates).catch(err => TB.logger.error("[User] Auto-sync after setUserData failed:", err));
        } else {
            // If not syncing to server, still update local timestamp
            localStorage.setItem(USER_DATA_TIMESTAMP_KEY, Date.now().toString());
        }
    },

    async syncUserData(updatedFields = null) { // updatedFields can be an object of specific fields changed
        if (!this.isAuthenticated()) return { success: false, message: "Not authenticated." };
        const userDataToSync = updatedFields || TB.state.get('user.userData');
        const localModTime = Date.now();

        try {
            TB.logger.debug('[User] Syncing user data to server:', userDataToSync);
            // The server endpoint should ideally handle partial updates and timestamp checking.
            // Send localModTime so server can decide if it has newer data.
            const result = await TB.api.request('UserManager', 'updateUserData', { data: userDataToSync, lastModified: localModTime }, 'POST');
            if (result.error === TB.ToolBoxError.none) {
                TB.logger.info('[User] User data sync acknowledged by server.');
                // Server might return its own 'lastModified' timestamp if it made further changes
                const serverResponse = result.get();
                if (serverResponse && serverResponse.lastModified) {
                    localStorage.setItem(USER_DATA_TIMESTAMP_KEY, serverResponse.lastModified.toString());
                    if (serverResponse.userData) { // If server sends back merged/updated data
                        this._updateUserState({ userData: serverResponse.userData }, false);
                    }
                } else {
                    localStorage.setItem(USER_DATA_TIMESTAMP_KEY, localModTime.toString());
                }
                return { success: true };
            }
            return { success: false, message: result.info.help_text || "Failed to sync user data." };
        } catch (error) {
            TB.logger.error('[User] Error syncing user data:', error);
            return { success: false, message: error.message };
        }
    },
    async fetchUserData() {
        if (!this.isAuthenticated()) return { success: false, message: "Not authenticated." };
        try {
            TB.logger.debug('[User] Fetching user data from server.');
            const result = await TB.api.request('UserManager', 'getUserData', {}, 'GET'); // Adjust endpoint
            if (result.error === TB.ToolBoxError.none && result.get()) {
                this._updateUserState({ userData: result.get() });
                TB.logger.info('[User] User data fetched successfully.');
                return { success: true, data: result.get() };
            }
            return { success: false, message: result.info.help_text || "Failed to fetch user data." };
        } catch (error) {
            TB.logger.error('[User] Error fetching user data:', error);
            return { success: false, message: error.message };
        }
    }
};

export default user;
