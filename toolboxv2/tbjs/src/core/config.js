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


        // Ensure baseFileUrl ends with a slash if it's not just the origin and contains a path
        if (_config.baseFileUrl && new URL(_config.baseFileUrl).pathname !== '/' && !_config.baseFileUrl.endsWith('/')) {
            _config.baseFileUrl += '/';
        }
         // Ensure baseApiUrl is absolute
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
    }
};

export default Config;
