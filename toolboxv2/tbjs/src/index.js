// tbjs/index.js
// Original: This file orchestrates the initialization of the entire tbjs framework.
// It replaces the top-level IIFE and global setup in the original index.js.

// Importiere den CSS-Einstiegspunkt, damit Webpack ihn verarbeitet
import './styles/tbjs-main.css';

import * as coreModules from './core/index.js';
import * as uiModules from './ui/index.js';

const TB = {
    ...coreModules, // Includes config, state, router, api, env, events, logger, crypto, utils, sse
    ui: uiModules,  // Includes theme, htmxIntegration, and all components

    VERSION: '0.1.0-alpha', // Framework version

    /**
     * Initializes the tbjs framework.
     * @param {object} config - Configuration options for the framework.
     * @param {string} [config.appRootId='app-root'] - The ID of the main DOM element for the app.
     * @param {string} config.baseApiUrl - Base URL for API calls (e.g., '/api').
     * @param {string} [config.baseFileUrl] - Base URL for fetching static HTML files for routing. (Derived from window.location if not set)
     * @param {object} [config.initialState={}] - Initial state for the application.
     * @param {object} [config.themeSettings={}] - Settings for UI theme (e.g., defaultMode: 'dark').
     * @param {Array<object>} [config.routes=[]] - Predefined routes for the router.
     */
    init: function(config = {}) { // Use 'function' to ensure 'this' refers to TB object if needed later

        this.config.init(config); // Must be first to set up base URLs etc.
                                     // Original: rpIdUrl_f, rpIdUrl_fs from original index.js
        if(!document.getElementById(this.config.get('appRootId'))){
             window.location.href = '/'
        }
        this.logger.init({ logLevel: config.logLevel || (this.config.get('isProduction') ? 'warn' : 'debug') });
        this.logger.log(`tbjs v${this.VERSION} initializing...`);

        this.env.detect(); // Detects Web/Tauri environment
                           // Original: Logic checking window.__TAURI__ in original autoDetection.js and index.js

        this.state.init(config.initialState); // Initializes application state
                                          // Original: initState and TBc/TBv logic from original index.js

        this.ui.theme.init(config.themeSettings); // Sets up dark/light mode
                                                // Original: initDome dark mode part, loadDarkModeState from original index.js & scripts.js


        this.router.init(
            document.getElementById(this.config.get('appRootId')),
            config.routes || []
        ); // Initializes the SPA router
           // Original: router, renderer, updateDome, linkEffect from original index.js

        this.ui.htmxIntegration.init(); // Sets up HTMX listeners
                                       // Original: handleHtmxAfterRequest from original index.js

        if (this.env.isWeb()) {
            this.logger.log('Running in Web environment.');
            // Example: this.sw.register(); // If you add a sw.js module to core
            // Original: registerServiceWorker/unRegisterServiceWorker from original index.js

            // Auto-disconnect SSE on window unload
            window.addEventListener('beforeunload', () => {
                this.sse.disconnectAll();
            });
        } else if (this.env.isTauri()) {
            this.logger.log('Running in Tauri environment.');
            // Tauri specific initializations can go here or be triggered by events
        }

        this.events.emit('tbjs:initialized', this);
        this.logger.log('tbjs initialization complete.');
        return this; // Return the fully initialized TB object
    }
};

// Freeze parts of TB to prevent accidental modification if desired
// Object.freeze(TB.core);
// Object.freeze(TB.ui);

export default TB;
