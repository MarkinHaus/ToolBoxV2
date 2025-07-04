// tbjs/core/state.js
// Manages global application state.
// Original: TBv, TBc, and parts of TBf related to var management from original index.js, also initState, persistState.

import events from './events.js'; // Use TB.events after full TB init
import logger from './logger.js'; // Use TB.logger

const PERSISTENCE_KEY_PREFIX = 'tbjs_app_state_';

const AppState = {
    _hooks: new Set(),
    _currentState: {},
    _persistentKeys: new Set(), // Keys to automatically persist to localStorage

    init: (initialState = {}) => {
        AppState._currentState = { ...initialState };
        AppState._loadPersistedState();
        logger.log('[State] Initialized with:', AppState._currentState);
    },

    get: (key) => {
        if (key === undefined) {
            return { ...AppState._currentState };
        }
        const keys = key.split('.');
        let value = AppState._currentState;
        for (const k of keys) {
            if (value && typeof value === 'object' && k in value) {
                value = value[k];
            } else {
                // logger.warn(`[State] Key "${key}" not found.`);
                return undefined;
            }
        }
        return value;
    },

    set: (key, value, options = { persist: true }) => {
        const keys = key.split('.');
        let current = AppState._currentState;
        keys.forEach((k, index) => {
            if (index === keys.length - 1) {
                current[k] = value;
            } else {
                if (!current[k] || typeof current[k] !== 'object') {
                    current[k] = {};
                }
                current = current[k];
            }
        });

        if (options.persist) {
            AppState._persistentKeys.add(keys[0]); // Persist the top-level key
            AppState._persistKey(keys[0]);
        }
        logger.log(`[State] Set "${key}":`, value);
        events.emit('state:changed', { key, value, fullState: AppState._currentState });
        events.emit(`state:changed:${key.replace(/\./g, ':')}`, value); // More specific event
    },

    delete: (key, options = { persist: true }) => {
        const keys = key.split('.');
        let current = AppState._currentState;
        let deleted = false;
        for (let i = 0; i < keys.length; i++) {
            const k = keys[i];
            if (i === keys.length - 1 && current && typeof current === 'object' && k in current) {
                delete current[k];
                deleted = true;
                break;
            }
            if (!current || typeof current !== 'object' || !(k in current)) {
                // logger.warn(`[State] Cannot delete: Key part "${k}" not found in path "${key}".`);
                return;
            }
            current = current[k];
        }

        if (deleted) {
            if (options.persist || AppState._persistentKeys.has(keys[0])) {
                AppState._persistKey(keys[0]);
            }
            logger.log(`[State] Deleted "${key}".`);
            events.emit('state:changed', { key, value: undefined, fullState: AppState._currentState });
            events.emit(`state:changed:${key.replace(/\./g, ':')}`, undefined);
        }
    },

    // For managing persisted state (replaces original TBc logic)
    _persistKey: (key) => {
        if (AppState._currentState.hasOwnProperty(key)) {
            try {
                localStorage.setItem(
                    `${PERSISTENCE_KEY_PREFIX}${key}`,
                    JSON.stringify(AppState._currentState[key])
                );
                // logger.debug(`[State] Persisted key "${key}".`);
            } catch (error) {
                logger.error(`[State] Error persisting key "${key}":`, error);
            }
        } else { // Key was deleted
             localStorage.removeItem(`${PERSISTENCE_KEY_PREFIX}${key}`);
        }
    },

    _loadPersistedState: () => {
        for (let i = 0; i < localStorage.length; i++) {
            const storageKey = localStorage.key(i);
            if (storageKey.startsWith(PERSISTENCE_KEY_PREFIX)) {
                const appKey = storageKey.substring(PERSISTENCE_KEY_PREFIX.length);
                try {
                    const value = JSON.parse(localStorage.getItem(storageKey));
                    if (AppState._currentState[appKey] === undefined || typeof AppState._currentState[appKey] !== typeof value) {
                         AppState._currentState[appKey] = value;
                    } else if (typeof AppState._currentState[appKey] === 'object' && AppState._currentState[appKey] !== null) {
                        AppState._currentState[appKey] = {...value, ...AppState._currentState[appKey]}; // Merge, initialState takes precedence for conflicts
                    }
                    AppState._persistentKeys.add(appKey);
                    // logger.debug(`[State] Loaded persisted key "${appKey}".`);
                } catch (error) {
                    logger.error(`[State] Error loading persisted key "${appKey}":`, error);
                }
            }
        }
    },

    // Direct access for initVar, delVar, getVar, setVar from original TBf
    // These should ideally be phased out in favor of structured state access.
    initVar: (v_name, v_value) => { if (AppState.get(v_name) === undefined) AppState.set(v_name, v_value, { persist: true }); },
    delVar: (v_name) => AppState.delete(v_name, { persist: true }),
    getVar: (v_name) => AppState.get(v_name),
    setVar: (v_name, v_value) => AppState.set(v_name, v_value, { persist: true }),
};

export default AppState;
