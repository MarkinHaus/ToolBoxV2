// tbjs/core/config.js
// Manages framework configuration.
// Original: Parts of rpIdUrl_f, rpIdUrl_fs, and implicit configurations in original index.js

const defaultConfig = {
    appRootId: 'app-root',
    baseApiUrl: '/api',
    baseFileUrl: '', // For fetching HTML views, defaults to origin
    logLevel: 'debug',
    isProduction: process.env.NODE_ENV === 'production',
    defaultTheme: 'system', // 'light', 'dark', or 'system'
};

let currentConfig = { ...defaultConfig };

const ConfigManager = {
    init: (userConfig = {}) => {
        currentConfig = { ...defaultConfig, ...userConfig };

        if (!currentConfig.baseFileUrl) {
            currentConfig.baseFileUrl = window.location.origin;
        }
        // Ensure baseApiUrl and baseFileUrl are absolute or correctly prefixed if relative
        if (currentConfig.baseApiUrl.startsWith('/') && !currentConfig.baseApiUrl.startsWith(window.location.origin)) {
            currentConfig.baseApiUrl = `${window.location.origin}${currentConfig.baseApiUrl}`;
        }
         if (currentConfig.baseFileUrl.startsWith('/') && !currentConfig.baseFileUrl.startsWith(window.location.origin)) {
            currentConfig.baseFileUrl = `${window.location.origin}${currentConfig.baseFileUrl}`;
        }


        console.log("Initialized Config:", currentConfig); // Use TB.logger once available
    },

    get: (key) => {
        if (key in currentConfig) {
            return currentConfig[key];
        }
        console.warn(`[Config] Key "${key}" not found.`); // Use TB.logger
        return undefined;
    },

    set: (key, value) => {
        currentConfig[key] = value;
        // Potentially emit an event: TB.events.emit('config:updated', { key, value });
    },

    getAll: () => {
        return { ...currentConfig };
    }
};

export default ConfigManager;
