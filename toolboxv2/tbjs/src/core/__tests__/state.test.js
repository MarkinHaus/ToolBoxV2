// tbjs/src/core/__tests__/state.test.js
// Tests für das State-Management Modul

import AppState from '../state.js';
import events from '../events.js';
import logger from '../logger.js';

// Mock dependencies
jest.mock('../events.js', () => ({
    emit: jest.fn(),
}));

jest.mock('../logger.js', () => ({
    log: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
}));

describe('AppState', () => {
    beforeEach(() => {
        // Reset state before each test
        AppState._currentState = {};
        AppState._persistentKeys = new Set();
        localStorage.clear();
        jest.clearAllMocks();
    });

    describe('init', () => {
        it('should initialize with empty state', () => {
            AppState.init();
            expect(AppState.get()).toEqual({});
        });

        it('should initialize with provided initial state', () => {
            AppState.init({ user: 'test', count: 42 });
            expect(AppState.get('user')).toBe('test');
            expect(AppState.get('count')).toBe(42);
        });

        it('should load persisted state from localStorage', () => {
            localStorage.setItem('tbjs_app_state_savedKey', JSON.stringify({ nested: 'value' }));
            AppState.init();
            expect(AppState.get('savedKey')).toEqual({ nested: 'value' });
        });

        it('should merge persisted state with initial state (initial takes precedence)', () => {
            localStorage.setItem('tbjs_app_state_config', JSON.stringify({ a: 1, b: 2 }));
            AppState.init({ config: { b: 99, c: 3 } });
            // Initial state should take precedence for conflicts
            expect(AppState.get('config.b')).toBe(99);
            expect(AppState.get('config.c')).toBe(3);
        });
    });

    describe('get', () => {
        beforeEach(() => {
            AppState.init({ user: { name: 'Alice', settings: { theme: 'dark' } }, count: 10 });
        });

        it('should return entire state when no key provided', () => {
            const state = AppState.get();
            expect(state).toHaveProperty('user');
            expect(state).toHaveProperty('count');
        });

        it('should return value for simple key', () => {
            expect(AppState.get('count')).toBe(10);
        });

        it('should return value for nested key using dot notation', () => {
            expect(AppState.get('user.name')).toBe('Alice');
            expect(AppState.get('user.settings.theme')).toBe('dark');
        });

        it('should return undefined for non-existent key', () => {
            expect(AppState.get('nonExistent')).toBeUndefined();
            expect(AppState.get('user.nonExistent')).toBeUndefined();
        });
    });

    describe('set', () => {
        beforeEach(() => {
            AppState.init({});
        });

        it('should set a simple value', () => {
            AppState.set('name', 'Bob');
            expect(AppState.get('name')).toBe('Bob');
        });

        it('should set a nested value using dot notation', () => {
            AppState.set('user.profile.email', 'bob@example.com');
            expect(AppState.get('user.profile.email')).toBe('bob@example.com');
        });

        it('should emit state:changed event', () => {
            AppState.set('key', 'value');
            expect(events.emit).toHaveBeenCalledWith('state:changed', expect.objectContaining({
                key: 'key',
                value: 'value',
            }));
        });

        it('should emit specific state:changed:key event', () => {
            AppState.set('user.name', 'Charlie');
            expect(events.emit).toHaveBeenCalledWith('state:changed:user:name', 'Charlie');
        });

        it('should persist to localStorage by default', () => {
            AppState.set('persistedKey', { data: 123 });
            const stored = JSON.parse(localStorage.getItem('tbjs_app_state_persistedKey'));
            expect(stored).toEqual({ data: 123 });
        });

        it('should not persist when persist option is false', () => {
            AppState.set('tempKey', 'tempValue', { persist: false });
            expect(localStorage.getItem('tbjs_app_state_tempKey')).toBeNull();
        });
    });

    describe('delete', () => {
        beforeEach(() => {
            AppState.init({ toDelete: 'value', nested: { child: 'data' } });
        });

        it('should delete a simple key', () => {
            AppState.delete('toDelete');
            expect(AppState.get('toDelete')).toBeUndefined();
        });

        it('should delete a nested key', () => {
            AppState.delete('nested.child');
            expect(AppState.get('nested.child')).toBeUndefined();
            expect(AppState.get('nested')).toEqual({});
        });

        it('should emit state:changed event on delete', () => {
            AppState.delete('toDelete');
            expect(events.emit).toHaveBeenCalledWith('state:changed', expect.objectContaining({
                key: 'toDelete',
                value: undefined,
            }));
        });

        it('should remove from localStorage when deleting persisted key', () => {
            AppState.set('persistedToDelete', 'value');
            expect(localStorage.getItem('tbjs_app_state_persistedToDelete')).not.toBeNull();
            AppState.delete('persistedToDelete');
            expect(localStorage.getItem('tbjs_app_state_persistedToDelete')).toBeNull();
        });

        it('should handle deleting non-existent key gracefully', () => {
            expect(() => AppState.delete('nonExistent')).not.toThrow();
            expect(() => AppState.delete('nested.nonExistent.deep')).not.toThrow();
        });
    });

    describe('Legacy API (initVar, getVar, setVar, delVar)', () => {
        beforeEach(() => {
            AppState.init({});
        });

        it('initVar should set value only if not already defined', () => {
            AppState.initVar('newVar', 'initial');
            expect(AppState.getVar('newVar')).toBe('initial');

            AppState.initVar('newVar', 'changed');
            expect(AppState.getVar('newVar')).toBe('initial'); // Should not change
        });

        it('setVar should update existing value', () => {
            AppState.setVar('myVar', 'first');
            expect(AppState.getVar('myVar')).toBe('first');

            AppState.setVar('myVar', 'second');
            expect(AppState.getVar('myVar')).toBe('second');
        });

        it('delVar should remove variable', () => {
            AppState.setVar('toRemove', 'value');
            expect(AppState.getVar('toRemove')).toBe('value');

            AppState.delVar('toRemove');
            expect(AppState.getVar('toRemove')).toBeUndefined();
        });

        it('getVar should return undefined for non-existent variable', () => {
            expect(AppState.getVar('doesNotExist')).toBeUndefined();
        });
    });

    describe('Persistence', () => {
        it('should handle localStorage errors gracefully', () => {
            // Simulate localStorage quota exceeded
            const originalSetItem = localStorage.setItem;
            localStorage.setItem = jest.fn(() => {
                throw new Error('QuotaExceededError');
            });

            expect(() => AppState.set('errorKey', 'value')).not.toThrow();
            expect(logger.error).toHaveBeenCalled();

            localStorage.setItem = originalSetItem;
        });

        it('should handle malformed JSON in localStorage', () => {
            localStorage.setItem('tbjs_app_state_malformed', 'not valid json{');
            expect(() => AppState.init()).not.toThrow();
            expect(logger.error).toHaveBeenCalled();
        });
    });
});

