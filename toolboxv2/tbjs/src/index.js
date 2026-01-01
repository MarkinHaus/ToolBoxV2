// tbjs/src/index.js
// Original: This file orchestrates the initialization of the entire tbjs framework.
// It replaces the top-level IIFE and global setup in the original index.js.

// Importiere den CSS-Einstiegspunkt, damit Webpack ihn verarbeitet
import './styles/tbjs-main.css';

import * as coreModules from './core/index.js';
import * as uiModules from './ui/index.js';

// Queue for functions to be run once initialization is complete.
const onceAfterInitQueue = [];
// A Set to track all functions ever passed to `TB.once` to ensure they truly only run once.
const onceRunFunctions = new Set();

const TB = {
    ...coreModules, // Includes config, state, router, api, env, events, logger, crypto, utils, sse, sw
    ui: uiModules,  // Includes theme, htmxIntegration, and all components
    VERSION: '0.1.0-alpha', // Framework version
    isInit: false,

     /**
     * Queues a function to run exactly once, after TB.init() is complete.
     * If the same function reference is passed multiple times, it will still only execute once.
     * If initialization is already done, it runs the function immediately (if it hasn't run before).
     * @param {Function} fn The function to run.
     */
    once: function(fn) {
        // If this function has already been queued or run, do nothing.
        if (onceRunFunctions.has(fn)) {
            return;
        }

        // Add the function to the tracking set to prevent future executions.
        onceRunFunctions.add(fn);

        this.onLoaded(fn);


    },
    onLoaded: function(fn) {
         if (this.isInit) {
            try {
                // If already initialized, run the function immediately.
                fn();
            } catch (e) {
                // The logger should be available at this point.
                this.logger.error('[TB.once] Error running function immediately:', e);
            }
        } else {
            // Not initialized yet, so add the function to the queue.
            onceAfterInitQueue.push(fn);
        }
    },

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
    init: async function(userAppConfig = {}) { // Renamed from config to avoid clash
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

    await this.user.init();

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

    // This test call will now be correctly queued by the new TB.once() method.
    this.once(() => { console.log("Once Online"); });

    // Initialize platform-specific UI (Desktop status bar, Mobile bottom nav, etc.)
    if (this.env.isTauri()) {
        this.platformComponents = this.ui.initPlatformUI({
            onCapture: async ({ text, tags }) => {
                try {
                    const response = await fetch(`${this.config.get('baseApiUrl')}/vault/capture`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, tags })
                    });
                    if (!response.ok) throw new Error('Capture failed');
                    this.logger.log('[Platform] Quick capture saved');
                } catch (error) {
                    this.logger.error('[Platform] Capture error:', error);
                }
            },
            onNavigate: (route) => {
                this.router.navigateTo(route);
            }
        });
        this.logger.log('[Platform] Platform UI initialized');
    }

    // Set the initialization flag to true.
    this.isInit = true;

    // Run all functions that were queued by TB.once() before init was complete.
    for (const queuedFn of onceAfterInitQueue) {
        try {
            queuedFn();
        } catch (e) {
            this.logger.error('[TB.init] Error running queued function from once():', e);
        }
    }
    // Clear the queue now that it has been processed.
    onceAfterInitQueue.length = 0;
    this.events.emit('tbjs:initialized', this);
    this.logger.log('tbjs initialization complete.');
    return this;
}
};

export default TB;
