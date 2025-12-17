// tbjs/src/core/config.js
import logger from './logger.js'; // Ensure logger is imported if used here

let _config = {};


const Config = {
    init: (initialUserConfig = {}) => {
        const defaultConfig = {
            appRootId: 'app-root',
            baseApiUrl: '/api',
            baseFileUrl: window.location.origin,
            initialState: {},
            themeSettings: {
                defaultPreference: 'system',
                background: {
                    type: 'color',
                    light: { color: '#FFFFFF', image: '' },
                    dark: { color: '#121212', image: '' },
                    placeholder: { image: '', displayUntil3DReady: true }
                }
            },
            routes: [],
            logLevel: 'info',
            // isProduction: process.env.NODE_ENV === 'production', // Better set by user config
            serviceWorker: {
                enabled: false,
                url: '/sw.js',
                scope: '/'
            }
        };

        // Simple deep merge for specific known nested objects
        _config = {
            ...defaultConfig,
            ...initialUserConfig,
            themeSettings: {
                ...(defaultConfig.themeSettings || {}),
                ...(initialUserConfig.themeSettings || {}),
                background: {
                    ...(defaultConfig.themeSettings?.background || {}),
                    ...(initialUserConfig.themeSettings?.background || {}),
                    light: {
                        ...(defaultConfig.themeSettings?.background?.light || {}),
                        ...(initialUserConfig.themeSettings?.background?.light || {}),
                    },
                    dark: {
                        ...(defaultConfig.themeSettings?.background?.dark || {}),
                        ...(initialUserConfig?.themeSettings?.background?.dark || {}),
                    },
                    placeholder: {
                        ...(defaultConfig.themeSettings?.background?.placeholder || {}),
                        ...(initialUserConfig.themeSettings?.background?.placeholder || {}),
                    }
                }
            },
            serviceWorker: {
                ...(defaultConfig.serviceWorker || {}),
                ...(initialUserConfig.serviceWorker || {})
            }
        };

        if (typeof _config.isProduction === 'undefined') {
            // Infer if not explicitly set, e.g. based on hostname or a global var
             _config.isProduction = !(window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
        }

        // Detect Tauri environment and set API URLs
        const isTauri = window.location.hostname === 'tauri.localhost' ||
                        window.location.protocol === 'tauri:' ||
                        (typeof window.__TAURI__ !== 'undefined');

        // Remote API fallback (for mobile or when worker unavailable)
        const remoteApiUrl = initialUserConfig.remoteApiUrl || 'https://simplecore.app/api';
        const remoteWsUrl = initialUserConfig.remoteWsUrl || 'wss://simplecore.app';

        if (isTauri && !initialUserConfig.baseApiUrl) {
            const workerHttpPort = initialUserConfig.workerHttpPort || 5000;
            const workerWsPort = initialUserConfig.workerWsPort || 5001;
            const localApiUrl = `http://localhost:${workerHttpPort}/api`;
            const localWsUrl = `ws://localhost:${workerWsPort}`;

            // Detect platform from user agent
            const ua = navigator.userAgent.toLowerCase();
            const isIOS = /iphone|ipad|ipod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
            const isAndroid = /android/.test(ua);
            const isMobile = isIOS || isAndroid;

            _config.isTauri = true;
            _config.isMobile = isMobile;
            _config.isIOS = isIOS;
            _config.isAndroid = isAndroid;

            if (isIOS) {
                // iOS: Always use remote API (no local worker possible)
                _config.baseApiUrl = remoteApiUrl;
                _config.baseWsUrl = remoteWsUrl;
                _config.useRemoteApi = true;
                console.log('[Config] Tauri iOS detected, using remote API:', remoteApiUrl);
            } else if (isAndroid) {
                // Android: Check if worker is available, fallback to remote
                _config.baseApiUrl = localApiUrl;
                _config.baseWsUrl = localWsUrl;
                _config.useRemoteApi = false;
                _config._remoteApiUrl = remoteApiUrl;
                _config._remoteWsUrl = remoteWsUrl;

                // Async worker check - will update config if worker unavailable
                Config._checkWorkerAndFallback(localApiUrl, remoteApiUrl, remoteWsUrl);
                console.log('[Config] Tauri Android detected, checking worker availability...');
            } else {
                // Desktop: Use local worker
                _config.baseApiUrl = localApiUrl;
                _config.baseWsUrl = localWsUrl;
                _config.useRemoteApi = false;
                console.log('[Config] Tauri Desktop detected, using local worker:', localApiUrl);
            }
        } else {
            _config.isTauri = isTauri;
        }

        // Ensure baseFileUrl ends with a slash if it's not just the origin and contains a path
        if (_config.baseFileUrl && new URL(_config.baseFileUrl).pathname !== '/' && !_config.baseFileUrl.endsWith('/')) {
            _config.baseFileUrl += '/';
        }
         // Ensure baseApiUrl is absolute (skip if already set for Tauri)
        if (_config.baseApiUrl && !_config.baseApiUrl.startsWith('http') && !_config.baseApiUrl.startsWith('/')) {
            _config.baseApiUrl = '/' + _config.baseApiUrl;
        }
        // If baseApiUrl is relative, make it absolute from baseFileUrl or origin
        if (_config.baseApiUrl && _config.baseApiUrl.startsWith('/')) {
             _config.baseApiUrl = new URL(_config.baseApiUrl, window.location.origin).href;
        }

        // After logger is initialized in TB.init, this can be moved there or use a preliminary log.
        // For now, console.log is fine for config init.
        console.log('[Config] Initialized Config:', JSON.parse(JSON.stringify(_config))); // Use JSON stringify/parse for clean log
    },

    get: (key) => {
        if (!key) return undefined;

        if (key.includes('.')) {
            const keys = key.split('.');
            let current = _config;
            for (let i = 0; i < keys.length; i++) {
                if (current && typeof current === 'object' && keys[i] in current) {
                    current = current[keys[i]];
                } else {
                    // logger.warn(`[Config] Key "${key}" not found (path: ${keys.slice(0, i+1).join('.')})`);
                    // console.warn(`[Config] GET: Key "${key}" not found (path: ${keys.slice(0, i+1).join('.')}). Available config:`, _config);
                    return undefined;
                }
            }
            return current;
        }

        const value = _config[key];
        // if (value === undefined) {
            // logger.warn(`[Config] Key "${key}" not found.`);
            // console.warn(`[Config] GET: Key "${key}" not found. Available config:`, _config);
        // }
        return value;
    },

    getAll: () => {
        return { ..._config };
    },

    set: (key, value) => {
        if (key.includes('.')) {
            const keys = key.split('.');
            let current = _config;
            for (let i = 0; i < keys.length - 1; i++) {
                if (!current[keys[i]] || typeof current[keys[i]] !== 'object') {
                    current[keys[i]] = {};
                }
                current = current[keys[i]];
            }
            current[keys[keys.length - 1]] = value;
        } else {
            _config[key] = value;
        }
        logger.debug(`[Config] Set: ${key} =`, value);
    },

    /**
     * Check if local worker is available (Android), fallback to remote API if not.
     * @param {string} localApiUrl - Local worker API URL
     * @param {string} remoteApiUrl - Remote API URL fallback
     * @param {string} remoteWsUrl - Remote WebSocket URL fallback
     */
    _checkWorkerAndFallback: async (localApiUrl, remoteApiUrl, remoteWsUrl) => {
        try {
            // Try to reach the local worker with a health check
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout

            const response = await fetch(`${localApiUrl.replace('/api', '')}/health`, {
                method: 'GET',
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (response.ok) {
                console.log('[Config] Local worker is available');
                _config.useRemoteApi = false;
                return;
            }
        } catch (error) {
            console.log('[Config] Local worker not available, falling back to remote API:', error.message);
        }

        // Fallback to remote API
        _config.baseApiUrl = remoteApiUrl;
        _config.baseWsUrl = remoteWsUrl;
        _config.useRemoteApi = true;
        console.log('[Config] Using remote API:', remoteApiUrl);
    }
};

export default Config;
