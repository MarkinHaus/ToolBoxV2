// tbjs/core/__tests__/user.test.js
// Comprehensive tests for User module with Clerk integration

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

// Mock Clerk instance
const mockClerkUser = {
    id: 'user_test123',
    username: 'testuser',
    emailAddresses: [{ emailAddress: 'test@test.com' }],
    firstName: 'Test',
    lastName: 'User',
    imageUrl: 'https://example.com/avatar.jpg'
};

const mockClerkSession = {
    getToken: jest.fn().mockResolvedValue('mock_jwt_token_12345'),
};

const mockClerkInstance = {
    user: null,
    session: null,
    loaded: true,
    load: jest.fn().mockResolvedValue(undefined),
    addListener: jest.fn(),
    signOut: jest.fn().mockResolvedValue(undefined),
    openSignIn: jest.fn().mockResolvedValue(undefined),
    openSignUp: jest.fn().mockResolvedValue(undefined),
    mountSignIn: jest.fn(),
    mountSignUp: jest.fn(),
};

// Mock window.Clerk
global.window = {
    ...global.window,
    Clerk: mockClerkInstance,
    location: {
        href: 'http://localhost:3000/web/assets/login.html',
        pathname: '/web/assets/login.html',
        search: '',
        hash: '',
        origin: 'http://localhost:3000'
    }
};

// Import user module once
import user from '../user.js';

describe('User Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorageMock.clear();
        stateStore.user = null;

        // Reset Clerk mock state
        mockClerkInstance.user = null;
        mockClerkInstance.session = null;

        // Reset fetch mock
        global.fetch.mockReset();

        // Mock config
        config.get.mockImplementation((key) => {
            if (key === 'baseApiUrl') return 'http://localhost:5000/api';
            return undefined;
        });
    });

    describe('validateBackendSession', () => {
        it('should return false when no token is available', async () => {
            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });

        it('should return false when no userId is available', async () => {
            mockClerkInstance.session = mockClerkSession;
            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });

        it('should return true when backend confirms valid session', async () => {
            // Setup authenticated state
            mockClerkInstance.user = mockClerkUser;
            mockClerkInstance.session = mockClerkSession;
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_test123',
                token: 'mock_jwt_token_12345'
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
                    'Authorization': expect.stringContaining('Bearer')
                })
            }));
        });

        it('should return false when backend rejects session', async () => {
            mockClerkInstance.user = mockClerkUser;
            mockClerkInstance.session = mockClerkSession;
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_test123',
                token: 'mock_jwt_token_12345'
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

        it('should return false on network error', async () => {
            mockClerkInstance.user = mockClerkUser;
            mockClerkInstance.session = mockClerkSession;
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_test123',
                token: 'mock_jwt_token_12345'
            };

            global.fetch.mockRejectedValueOnce(new Error('Network error'));

            const result = await user.validateBackendSession();
            expect(result).toBe(false);
        });
    });

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
    });

    describe('getUserId', () => {
        it('should return null when user state is not set', () => {
            stateStore.user = null;
            expect(user.getUserId()).toBeNull();
        });

        it('should return userId from state', () => {
            stateStore.user = { userId: 'user_test123' };
            expect(user.getUserId()).toBe('user_test123');
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

    describe('getSessionToken', () => {
        it('should return token from state when available', async () => {
            stateStore.user = { token: 'state_token_123' };

            const token = await user.getSessionToken();
            expect(token).toBe('state_token_123');
        });

        it('should return null when no token available', async () => {
            stateStore.user = null;

            const token = await user.getSessionToken();
            expect(token).toBeNull();
        });
    });

    describe('signOut', () => {
        it('should clear user state on signOut', async () => {
            stateStore.user = {
                isAuthenticated: true,
                userId: 'user_test123'
            };

            // Mock the backend logout call
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ error: 'none' })
            });

            await user.signOut();

            // State should be cleared
            expect(TB.state.set).toHaveBeenCalled();
        });
    });

    describe('_updateUserState', () => {
        it('should update user state with new values', () => {
            stateStore.user = { isAuthenticated: false, username: null };

            user._updateUserState({ isAuthenticated: true, username: 'newuser' });

            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: true,
                username: 'newuser'
            }));
        });

        it('should clear state when clearExisting is true', () => {
            stateStore.user = {
                isAuthenticated: true,
                username: 'olduser',
                token: 'old_token'
            };

            user._updateUserState({}, true);

            expect(TB.state.set).toHaveBeenCalledWith('user', expect.objectContaining({
                isAuthenticated: false
            }));
        });
    });
});

