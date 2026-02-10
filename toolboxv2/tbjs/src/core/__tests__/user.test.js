// tbjs/core/__tests__/user.test.js
// Comprehensive tests for User module with Custom Auth (CloudM.Auth)

// Mock dependencies before importing user module
jest.mock('../config.js', () => ({
    get: jest.fn(),
}));
jest.mock('../logger.js', () => ({
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    log: jest.fn(),
}));
jest.mock('../events.js', () => ({
    emit: jest.fn(),
    on: jest.fn(),
}));

// Create a shared state store that persists across module resets
const stateStore = { user: null };

// Helper to get nested property from state using dot notation
const getNestedState = (key) => {
    if (!key) return null;
    const parts = key.split('.');
    let value = stateStore;
    for (const part of parts) {
        if (value === null || value === undefined) return null;
        value = value[part];
    }
    return value;
};

// Mock TB
jest.mock('../../index.js', () => ({
    state: {
        get: jest.fn((key) => getNestedState(key)),
        set: jest.fn((key, value) => {
            if (key === 'user') stateStore.user = value;
        }),
    },
    events: {
        emit: jest.fn(),
        on: jest.fn(),
    },
    logger: {
        debug: jest.fn(),
        info: jest.fn(),
        warn: jest.fn(),
        error: jest.fn(),
    },
    ui: {
        Toast: {
            showError: jest.fn(),
            showSuccess: jest.fn(),
        }
    }
}));

import TB from '../../index.js';
import config from '../config.js';

// Mock fetch globally
global.fetch = jest.fn();

// Mock localStorage
const localStorageMock = {
    store: {},
    getItem: jest.fn((key) => localStorageMock.store[key] || null),
    setItem: jest.fn((key, value) => { localStorageMock.store[key] = value; }),
    removeItem: jest.fn((key) => { delete localStorageMock.store[key]; }),
    clear: jest.fn(() => { localStorageMock.store = {}; }),
};
Object.defineProperty(global, 'localStorage', { value: localStorageMock });

// Mock window.location
const mockLocation = {
    href: 'http://localhost:3000/web/mainContent.html',
    pathname: '/web/mainContent.html',
    search: '',
    hash: '',
    origin: 'http://localhost:3000'
};

// Mock window.history.replaceState
const mockReplaceState = jest.fn();

global.window = {
    ...global.window,
    location: mockLocation,
    history: { replaceState: mockReplaceState },
    TB: {
        ui: { Toast: { showError: jest.fn() } },
        router: { navigateTo: jest.fn() },
    }
};

// Mock document for activity monitor
global.document = {
    ...global.document,
    addEventListener: jest.fn(),
    querySelector: jest.fn(),
    contains: jest.fn(),
};

// Import user module once
import user from '../user.js';

describe('User Module (Custom Auth)', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        jest.useFakeTimers();
        localStorageMock.clear();
        stateStore.user = null;

        // Reset fetch mock
        global.fetch.mockReset();

        // Reset window.location
        mockLocation.href = 'http://localhost:3000/web/mainContent.html';
        mockLocation.pathname = '/web/mainContent.html';
        mockLocation.search = '';

        // Mock config
        config.get.mockImplementation((key) => {
            if (key === 'baseApiUrl') return 'http://localhost:5000/api';
            return undefined;
        });

        // Clear timers
        user._stopTokenRefreshTimer();
        user._initPromise = null;
    });

    afterEach(() => {
        jest.useRealTimers();
    });

    // =================== init ===================

    describe('init', () => {
        it('should initialize with default state when no stored session', async () => {
            localStorageMock.getItem.mockReturnValue(null);

            await user.init();

            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: false,
                username: null,
                token: null,
            }));
        });

        it('should restore and validate session from localStorage', async () => {
            const storedSession = {
                isAuthenticated: true,
                username: 'testuser',
                userId: 'user_123',
                token: 'stored_jwt_token',
                refreshToken: 'stored_refresh_token',
            };
            localStorageMock.getItem.mockImplementation((key) => {
                if (key === 'tbjs_user_session') return JSON.stringify(storedSession);
                return null;
            });

            // Mock validateBackendSession success
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'none',
                    result: { data: { authenticated: true } }
                })
            });

            stateStore.user = storedSession;

            await user.init();

            expect(TB.events.emit).toHaveBeenCalledWith('user:signedIn', expect.objectContaining({
                username: 'testuser',
                userId: 'user_123',
            }));
        });

        it('should clear state if stored session is invalid on backend', async () => {
            const storedSession = {
                isAuthenticated: true,
                username: 'testuser',
                userId: 'user_123',
                token: 'expired_jwt_token',
                refreshToken: 'expired_refresh_token',
            };
            localStorageMock.getItem.mockImplementation((key) => {
                if (key === 'tbjs_user_session') return JSON.stringify(storedSession);
                return null;
            });

            stateStore.user = storedSession;

            // Mock validateBackendSession failure
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'session_invalid',
                    result: { data: { authenticated: false } }
                })
            });

            await user.init();

            // State should have been cleared via _updateUserState({}, true)
            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: false,
            }));
        });

        it('should return existing promise if init called multiple times', async () => {
            localStorageMock.getItem.mockReturnValue(null);

            const promise1 = user.init();
            const promise2 = user.init();

            expect(promise1).toBe(promise2);
            await promise1;
        });

        it('should set up state persistence listener', async () => {
            localStorageMock.getItem.mockReturnValue(null);

            await user.init();

            expect(TB.events.on).toHaveBeenCalledWith('state:changed:user', expect.any(Function));
        });
    });

    // =================== validateBackendSession ===================

    describe('validateBackendSession', () => {
        it('should return false when no token is available', async () => {
            stateStore.user = { token: null, userId: null };
            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });

        it('should return false when no userId is available', async () => {
            stateStore.user = { token: 'some_token', userId: null };
            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });

        it('should return true when backend confirms valid session', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_123',
                token: 'valid_jwt_token',
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'none',
                    result: { data: { authenticated: true } }
                })
            });

            const result = await user.validateBackendSession();
            expect(result).toBe(true);
            expect(global.fetch).toHaveBeenCalledWith('/validateSession', expect.objectContaining({
                method: 'POST',
                headers: expect.objectContaining({
                    'Authorization': 'Bearer valid_jwt_token'
                })
            }));
        });

        it('should return false when backend rejects session', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_123',
                token: 'invalid_jwt_token',
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'session_invalid',
                    result: { data: { authenticated: false } }
                })
            });

            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });

        it('should return false on HTTP error', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_123',
                token: 'some_token',
            };

            global.fetch.mockResolvedValueOnce({
                ok: false,
                status: 500,
            });

            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });

        it('should return false on network error', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_123',
                token: 'some_token',
            };

            global.fetch.mockRejectedValueOnce(new Error('Network error'));

            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });
    });

    // =================== signIn ===================

    describe('signIn', () => {
        it('should redirect to default login URL', async () => {
            mockLocation.href = 'http://localhost:3000/some-page';

            const result = await user.signIn();

            expect(mockLocation.href).toContain('/web/scripts/login.html');
            expect(result).toEqual({ success: true });
        });

        it('should redirect to custom login URL with next param', async () => {
            mockLocation.href = 'http://localhost:3000/dashboard';

            await user.signIn({ loginUrl: '/custom-login', next: '/dashboard' });

            expect(mockLocation.href).toBe('/custom-login?next=%2Fdashboard');
        });
    });

    // =================== signOut ===================

    describe('signOut', () => {
        it('should clear user state and notify backend', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_123',
                username: 'testuser',
                token: 'jwt_token_123',
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ error: 'none' })
            });

            const result = await user.signOut();

            expect(result).toEqual({ success: true, message: 'Signed out successfully' });
            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: false,
            }));
            expect(TB.events.emit).toHaveBeenCalledWith('user:signedOut');
        });

        it('should clear state even if backend logout fails', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_123',
                token: 'jwt_token_123',
            };

            global.fetch.mockRejectedValueOnce(new Error('Network error'));

            const result = await user.signOut();

            expect(result).toEqual({ success: true, message: 'Signed out successfully' });
            expect(TB.state.set).toHaveBeenCalled();
            expect(TB.events.emit).toHaveBeenCalledWith('user:signedOut');
        });

        it('should skip backend call when no token available', async () => {
            stateStore.user = {
                isAuthenticated: false,
                userId: null,
                token: null,
            };

            await user.signOut();

            expect(global.fetch).not.toHaveBeenCalled();
            expect(TB.events.emit).toHaveBeenCalledWith('user:signedOut');
        });

        it('should stop token refresh timer on signOut', async () => {
            stateStore.user = { isAuthenticated: false, token: null, userId: null };

            // Spy on _stopTokenRefreshTimer
            const spy = jest.spyOn(user, '_stopTokenRefreshTimer');

            await user.signOut();

            expect(spy).toHaveBeenCalled();
            spy.mockRestore();
        });
    });

    // =================== Getters ===================

    describe('isAuthenticated', () => {
        it('should return false when user state is not set', () => {
            stateStore.user = null;
            expect(user.isAuthenticated()).toBe(false);
        });

        it('should return false when isAuthenticated is false in state', () => {
            stateStore.user = { isAuthenticated: false };
            expect(user.isAuthenticated()).toBe(false);
        });

        it('should return true when isAuthenticated is true in state', () => {
            stateStore.user = { isAuthenticated: true };
            expect(user.isAuthenticated()).toBe(true);
        });

        it('should perform backend check when checkBackend=true and locally authenticated', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_123',
                token: 'valid_token',
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'none',
                    result: { data: { authenticated: true } }
                })
            });

            const result = await user.isAuthenticated(true);
            expect(result).toBe(true);
        });
    });

    describe('getUserId', () => {
        it('should return null when user state is not set', () => {
            stateStore.user = null;
            expect(user.getUserId()).toBeNull();
        });

        it('should return userId from state', () => {
            stateStore.user = { userId: 'user_123' };
            expect(user.getUserId()).toBe('user_123');
        });
    });

    describe('getUsername', () => {
        it('should return null when user state is not set', () => {
            stateStore.user = null;
            expect(user.getUsername()).toBeNull();
        });

        it('should return username from state', () => {
            stateStore.user = { username: 'testuser' };
            expect(user.getUsername()).toBe('testuser');
        });
    });

    describe('getEmail', () => {
        it('should return null when user state is not set', () => {
            stateStore.user = null;
            expect(user.getEmail()).toBeNull();
        });

        it('should return email from state', () => {
            stateStore.user = { email: 'test@example.com' };
            expect(user.getEmail()).toBe('test@example.com');
        });
    });

    describe('getToken', () => {
        it('should return null when no token is set', () => {
            stateStore.user = null;
            expect(user.getToken()).toBeNull();
        });

        it('should return token from state', () => {
            stateStore.user = { token: 'jwt_token_abc' };
            expect(user.getToken()).toBe('jwt_token_abc');
        });
    });

    describe('getRefreshToken', () => {
        it('should return null when no refresh token is set', () => {
            stateStore.user = null;
            expect(user.getRefreshToken()).toBeNull();
        });

        it('should return refreshToken from state', () => {
            stateStore.user = { refreshToken: 'refresh_token_xyz' };
            expect(user.getRefreshToken()).toBe('refresh_token_xyz');
        });
    });

    describe('getUserLevel', () => {
        it('should return 1 as default level', () => {
            stateStore.user = null;
            expect(user.getUserLevel()).toBe(1);
        });

        it('should return userLevel from state', () => {
            stateStore.user = { userLevel: 3 };
            expect(user.getUserLevel()).toBe(3);
        });
    });

    describe('getSettings / getSetting', () => {
        it('should return empty object when no settings', () => {
            stateStore.user = null;
            expect(user.getSettings()).toEqual({});
        });

        it('should return settings from state', () => {
            stateStore.user = { settings: { theme: 'dark', lang: 'en' } };
            expect(user.getSettings()).toEqual({ theme: 'dark', lang: 'en' });
        });

        it('should return specific setting value', () => {
            stateStore.user = { settings: { theme: 'dark' } };
            expect(user.getSetting('theme')).toBe('dark');
        });

        it('should return default value for missing setting', () => {
            stateStore.user = { settings: {} };
            expect(user.getSetting('missing', 'fallback')).toBe('fallback');
        });
    });

    describe('getSessionToken', () => {
        it('should return token from state', async () => {
            stateStore.user = { token: 'jwt_session_token' };
            const token = await user.getSessionToken();
            expect(token).toBe('jwt_session_token');
        });

        it('should return null when no token available', async () => {
            stateStore.user = null;
            const token = await user.getSessionToken();
            expect(token).toBeNull();
        });
    });

    // =================== updateSettings ===================

    describe('updateSettings', () => {
        it('should return failure when not authenticated', async () => {
            stateStore.user = { userId: null, token: null };
            const result = await user.updateSettings({ theme: 'dark' });
            expect(result).toEqual({ success: false, message: 'Not authenticated' });
        });

        it('should send settings to backend and update local state', async () => {
            stateStore.user = {
                userId: 'user_123',
                token: 'jwt_token',
                settings: { lang: 'en' },
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ error: 'none' })
            });

            const result = await user.updateSettings({ theme: 'dark' });

            expect(result).toEqual({ success: true });
            expect(global.fetch).toHaveBeenCalledWith('/api/CloudM.Auth/update_user_data', expect.objectContaining({
                method: 'POST',
                headers: expect.objectContaining({
                    'Authorization': 'Bearer jwt_token',
                }),
            }));
            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                settings: expect.objectContaining({ lang: 'en', theme: 'dark' })
            }));
        });

        it('should return failure when backend rejects update', async () => {
            stateStore.user = { userId: 'user_123', token: 'jwt_token', settings: {} };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'permission_denied',
                    info: { help_text: 'No permission' }
                })
            });

            const result = await user.updateSettings({ theme: 'dark' });
            expect(result).toEqual({ success: false, message: 'No permission' });
        });

        it('should return failure on network error', async () => {
            stateStore.user = { userId: 'user_123', token: 'jwt_token', settings: {} };

            global.fetch.mockRejectedValueOnce(new Error('Connection refused'));

            const result = await user.updateSettings({ theme: 'dark' });
            expect(result).toEqual({ success: false, message: 'Connection refused' });
        });
    });

    // =================== fetchUserData ===================

    describe('fetchUserData', () => {
        it('should return failure when not authenticated', async () => {
            stateStore.user = { userId: null, token: null };
            const result = await user.fetchUserData();
            expect(result).toEqual({ success: false, message: 'Not authenticated' });
        });

        it('should fetch and update user data from backend', async () => {
            stateStore.user = {
                userId: 'user_123',
                token: 'jwt_token',
                settings: {},
                userLevel: 1,
                modData: {},
            };

            const backendData = {
                settings: { theme: 'dark', notifications: true },
                level: 5,
                mod_data: { modA: { key: 'value' } },
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'none',
                    result: { data: backendData }
                })
            });

            const result = await user.fetchUserData();

            expect(result).toEqual({ success: true, data: backendData });
            expect(global.fetch).toHaveBeenCalledWith('/api/CloudM.Auth/get_user_data', expect.objectContaining({
                method: 'POST',
            }));
            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                settings: { theme: 'dark', notifications: true },
                userLevel: 5,
                modData: { modA: { key: 'value' } },
            }));
        });

        it('should return failure on backend error', async () => {
            stateStore.user = { userId: 'user_123', token: 'jwt_token' };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'not_found',
                    info: { help_text: 'User not found' },
                })
            });

            const result = await user.fetchUserData();
            expect(result).toEqual({ success: false, message: 'User not found' });
        });

        it('should return failure on network error', async () => {
            stateStore.user = { userId: 'user_123', token: 'jwt_token' };

            global.fetch.mockRejectedValueOnce(new Error('Timeout'));

            const result = await user.fetchUserData();
            expect(result).toEqual({ success: false, message: 'Timeout' });
        });
    });

    // =================== _checkAuthCallback ===================

    describe('_checkAuthCallback', () => {
        it('should do nothing when no token in URL', async () => {
            mockLocation.search = '';
            await user._checkAuthCallback();
            expect(global.fetch).not.toHaveBeenCalled();
        });

        it('should process auth callback tokens from URL', async () => {
            mockLocation.search = '?token=oauth_jwt_token&refresh_token=oauth_refresh_token';
            mockLocation.href = 'http://localhost:3000/web/mainContent.html?token=oauth_jwt_token&refresh_token=oauth_refresh_token';

            // Mock validate_session call
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'none',
                    result: {
                        data: {
                            authenticated: true,
                            user_id: 'oauth_user_1',
                            username: 'oauthuser',
                            email: 'oauth@example.com',
                            provider: 'github',
                        }
                    }
                })
            });

            // Mock get_user_data call
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'none',
                    result: {
                        data: {
                            settings: { theme: 'light' },
                            level: 2,
                            mod_data: {},
                        }
                    }
                })
            });

            await user._checkAuthCallback();

            expect(global.fetch).toHaveBeenCalledTimes(2);
            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: true,
                username: 'oauthuser',
                userId: 'oauth_user_1',
                token: 'oauth_jwt_token',
                refreshToken: 'oauth_refresh_token',
            }));
            expect(TB.events.emit).toHaveBeenCalledWith('user:signedIn', expect.objectContaining({
                username: 'oauthuser',
                userId: 'oauth_user_1',
            }));
        });

        it('should handle auth callback failure gracefully', async () => {
            mockLocation.search = '?token=bad_token';
            mockLocation.href = 'http://localhost:3000/callback?token=bad_token';

            global.fetch.mockResolvedValueOnce({
                ok: false,
                status: 401,
            });

            await user._checkAuthCallback();

            // Should call _handleAuthFailure which clears state
            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: false,
            }));
        });
    });

    // =================== _refreshToken ===================

    describe('_refreshToken', () => {
        it('should throw when no tokens available', async () => {
            stateStore.user = { token: null, refreshToken: null };

            await expect(user._refreshToken()).rejects.toThrow('No active session');
        });

        it('should refresh token via backend and update state', async () => {
            stateStore.user = {
                token: 'old_jwt',
                refreshToken: 'old_refresh',
                isAuthenticated: true,
                userId: 'user_123',
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    error: 'none',
                    result: {
                        data: {
                            access_token: 'new_jwt',
                            refresh_token: 'new_refresh',
                        }
                    }
                })
            });

            await user._refreshToken();

            expect(global.fetch).toHaveBeenCalledWith('/api/CloudM.Auth/refresh_token', expect.objectContaining({
                method: 'POST',
            }));
            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                token: 'new_jwt',
                refreshToken: 'new_refresh',
            }));
        });

        it('should sign out and throw when refresh fails', async () => {
            stateStore.user = {
                token: 'old_jwt',
                refreshToken: 'old_refresh',
                isAuthenticated: true,
                userId: 'user_123',
            };

            global.fetch.mockResolvedValueOnce({
                ok: false,
                status: 401,
            });

            // signOut backend call (called during failure handling)
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ error: 'none' }),
            });

            await expect(user._refreshToken()).rejects.toThrow();
            expect(TB.events.emit).toHaveBeenCalledWith('user:signedOut');
        });
    });

    // =================== _updateUserState ===================

    describe('_updateUserState', () => {
        it('should update user state with new values', () => {
            stateStore.user = { isAuthenticated: false, username: null };

            user._updateUserState({ isAuthenticated: true, username: 'newuser' });

            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: true,
                username: 'newuser',
            }));
            expect(TB.events.emit).toHaveBeenCalledWith('user:stateChanged', expect.any(Object));
        });

        it('should reset to defaults when clearExisting is true', () => {
            stateStore.user = {
                isAuthenticated: true,
                username: 'olduser',
                token: 'old_token',
            };

            user._updateUserState({}, true);

            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: false,
                username: null,
                token: null,
            }));
        });
    });

    // =================== setAuthData ===================

    describe('setAuthData', () => {
        it('should set auth data and emit signedIn event', () => {
            stateStore.user = { isAuthenticated: false };

            user.setAuthData({
                token: 'new_jwt',
                refreshToken: 'new_refresh',
                userId: 'user_456',
                username: 'newuser',
                email: 'new@example.com',
            });

            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: true,
                token: 'new_jwt',
                refreshToken: 'new_refresh',
                userId: 'user_456',
                username: 'newuser',
                email: 'new@example.com',
            }));
            expect(TB.events.emit).toHaveBeenCalledWith('user:signedIn', {
                username: 'newuser',
                userId: 'user_456',
            });
        });
    });

    // =================== isAuthReady ===================

    describe('isAuthReady', () => {
        it('should return the auth initialized state', () => {
            // isAuthReady is based on internal authInitialized flag
            // After a successful init it should return true
            const result = user.isAuthReady();
            expect(typeof result).toBe('boolean');
        });
    });

    // =================== Token refresh timer ===================

    describe('token refresh timer', () => {
        it('should stop timer when _stopTokenRefreshTimer is called', () => {
            user._tokenRefreshTimerId = setInterval(() => {}, 1000);
            user._stopTokenRefreshTimer();
            expect(user._tokenRefreshTimerId).toBeNull();
        });
    });
});
