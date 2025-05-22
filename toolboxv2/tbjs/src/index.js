// tbjs/src/index.js
// Original: This file orchestrates the initialization of the entire tbjs framework.
// It replaces the top-level IIFE and global setup in the original index.js.

// Importiere den CSS-Einstiegspunkt, damit Webpack ihn verarbeitet
import './styles/tbjs-main.css';

import * as coreModules from './core/index.js';
import * as uiModules from './ui/index.js';

const TB = {
    ...coreModules, // Includes config, state, router, api, env, events, logger, crypto, utils, sse, sw
    ui: uiModules,  // Includes theme, htmxIntegration, and all components

    VERSION: '0.1.0-alpha', // Framework version

    /**
     * Initializes the tbjs framework.
     * @param {object} userAppConfig - Configuration options for the framework.
     * @param {string} [userAppConfig.appRootId='app-root'] - The ID of the main DOM element for the app.
     * @param {string} userAppConfig.baseApiUrl - Base URL for API calls (e.g., '/api').
     * @param {string} [userAppConfig.baseFileUrl] - Base URL for fetching static HTML files for routing. (Derived from window.location if not set)
     * @param {object} [userAppConfig.initialState={}] - Initial state for the application.
     * @param {object} [userAppConfig.themeSettings={}] - Settings for UI theme (e.g., defaultMode: 'dark').
     * @param {Array<object>} [userAppConfig.routes=[]] - Predefined routes for the router.
     * @param {object} [userAppConfig.serviceWorker] - Service Worker configuration.
     * @param {boolean} [userAppConfig.serviceWorker.enabled=false] - Whether to register the service worker.
     * @param {string} [userAppConfig.serviceWorker.url='/sw.js'] - Path to the service worker script.
     * @param {string} [userAppConfig.serviceWorker.scope='/'] - Scope for the service worker.
     */
    init: function(userAppConfig = {}) { // Renamed from config to avoid clash
    const intendedPathKey = 'tbjs_intended_path';

    // Pass userAppConfig directly. config.js now handles defaults and merging.
    this.config.init(userAppConfig);

    // The logger init needs to happen AFTER config.init so it can use configured logLevel and isProduction
    this.logger.init({
        logLevel: this.config.get('logLevel') || (this.config.get('isProduction') ? 'warn' : 'debug')
    });
    this.logger.log(`tbjs v${this.VERSION} initializing... (Production: ${this.config.get('isProduction')})`);


    if(!document.getElementById(this.config.get('appRootId'))){
         sessionStorage.setItem(intendedPathKey, window.location.pathname + window.location.search + window.location.hash);
         window.location.href = '/index.html'; // or a dedicated loader page
         return; // Stop execution if root element not found
    }
    this.env.detect();
    this.state.init(this.config.get('initialState')); // Get from config
    this.ui.theme.init(this.config.get('themeSettings')); // Get from config

    setTimeout(async () => {
        await this.user.init();
    }, 1);

    // Router.init will navigate to the current window.location.pathname + search + hash
    this.router.init(
        document.getElementById(this.config.get('appRootId')),
        this.config.get('routes') || []
    );

    const intendedPath = sessionStorage.getItem(intendedPathKey);
    if (intendedPath) {
        this.logger.log(`[TB.init] Found intended path: ${intendedPath}. Navigating...`);
        sessionStorage.removeItem(intendedPathKey);
        this.router.navigateTo(intendedPath, true, false); // replace, not initialLoad
    } else {
        const currentRouterPath = this.router.getCurrentPath();
        const defaultAppPath = "/index.html"; // Your default content page
        const rootPaths = ['/', '/index.html'];

        // Add baseFileUrl variations to rootPaths if baseFileUrl is set and not just "/"
        const baseFileUrl = this.config.get('baseFileUrl');
        if (baseFileUrl) {
            const baseFileUrlObject = new URL(baseFileUrl);
            if (baseFileUrlObject.pathname && baseFileUrlObject.pathname !== '/') {
                rootPaths.push(baseFileUrlObject.pathname); // e.g., /app/
                rootPaths.push(baseFileUrlObject.pathname.replace(/\/$/, '') + '/index.html'); // e.g., /app/index.html
            }
        }

        // Normalize currentRouterPath for comparison against root paths
        // The router.getCurrentPath() should already be normalized relative to baseFileUrl
        if (rootPaths.some(rp => currentRouterPath === rp || currentRouterPath.startsWith(rp + '?') || currentRouterPath.startsWith(rp + '#')) && currentRouterPath !== defaultAppPath) {
             this.logger.log(`[TB.init] Initial path ${currentRouterPath} is a root equivalent. Navigating to default: ${defaultAppPath}`);
             this.router.navigateTo(defaultAppPath, true, false); // replace, not initialLoad
        }
    }

    this.ui.htmxIntegration.init();

    if (this.env.isWeb()) {
        this.logger.log('[Running] in Web environment.');
        const swConfigEnabled = this.config.get('serviceWorker.enabled');
        if (swConfigEnabled) { // Check resolved config
            this.sw.register().catch(e => this.logger.error('[TB.init] SW registration promise rejected:', e));
        } else {
            this.logger.info('[TB.init] Service Worker is disabled by configuration.');
            // Optionally unregister if you want to ensure no SW runs when disabled
            // this.sw.unregister().catch(e => this.logger.error('[TB.init] SW unregistration promise rejected:', e));
        }

        window.addEventListener('beforeunload', () => {
            this.sse.disconnectAll();
        });
    } else if (this.env.isTauri()) {
        this.logger.log('[Running] in Tauri environment.');
    }

    this.events.emit('tbjs:initialized', this);
    this.logger.log('tbjs initialization complete.');
    return this;
}
};

export default TB;
