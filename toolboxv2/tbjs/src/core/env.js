// tbjs/core/env.js
import logger from './logger.js';

const Environment = {
    _isTauri: false,
    _isWeb: true,
    _isMobile: false,
    _isDesktop: false,
    _platformInfo: {},

    // NEU: Tauri Worker URLs
    _workerUrls: {
        http: null,
        ws: null,
        sse: null
    },

    detect: () => {
        if (typeof window === 'undefined' || typeof navigator === 'undefined') {
            logger.warn('[Env] No window or navigator object found.');
            return;
        }

        const userAgent = navigator.userAgent || '';
        const platform = navigator.platform || '';

        // Tauri detection
        const isTauri = !!window.__TAURI__;
        Environment._isTauri = isTauri;

        // Mobile detection
        const ios = /iPad|iPhone|iPod/.test(userAgent) || (navigator.maxTouchPoints > 2 && /MacIntel/.test(platform));
        const android = /Android/.test(userAgent);
        const mobileUA = /webOS|BlackBerry|Opera Mini|Opera Mobi|IEMobile/i.test(userAgent);
        const isMobile = ios || android || mobileUA;

        // Desktop detection
        const isDesktop = !isMobile;

        // Platform
        const isMac = ios || /Mac/.test(platform);
        const isWindows = /Win/.test(platform);
        const isLinux = /Linux/.test(platform);
        const isChromeOS = /\bCrOS\b/.test(userAgent);

        // Store values
        Environment._isWeb = !isTauri;
        Environment._isMobile = isMobile;
        Environment._isDesktop = isDesktop;
        Environment._platformInfo = {
            userAgent,
            platform,
            isTauri,
            isMobile,
            isDesktop,
            isMac,
            isWindows,
            isLinux,
            isChromeOS
        };

        logger.log(`[Env] Detection Complete: Tauri=${isTauri}, Mobile=${isMobile}, Desktop=${isDesktop}`);
        logger.debug('[Env] Platform Info:', Environment._platformInfo);

        // NEU: Auto-init Tauri Worker URLs
        if (isTauri) {
            Environment._initTauriWorkerUrls();
        }
    },

    // NEU: Tauri Worker URL Initialization
    _initTauriWorkerUrls: async () => {
        if (!Environment._isTauri) return;

        try {
            const { invoke } = window.__TAURI__.core;
            const status = await invoke('get_worker_status');

            if (status && status.running) {
                Environment._workerUrls.http = status.http_url;  // "http://localhost:5000"
                Environment._workerUrls.ws = status.ws_url;      // "ws://localhost:5001"
                Environment._workerUrls.sse = status.http_url;   // SSE geht über HTTP

                logger.log('[Env] Tauri Worker URLs initialized:', Environment._workerUrls);
            } else {
                // Fallback zu Default-Ports
                Environment._workerUrls.http = 'http://localhost:5000';
                Environment._workerUrls.ws = 'ws://localhost:5001';
                Environment._workerUrls.sse = 'http://localhost:5000';
                logger.warn('[Env] Worker not running, using default URLs');
            }
        } catch (error) {
            // Fallback wenn Tauri Command nicht verfügbar
            Environment._workerUrls.http = 'http://localhost:5000';
            Environment._workerUrls.ws = 'ws://localhost:5001';
            Environment._workerUrls.sse = 'http://localhost:5000';
            logger.warn('[Env] Could not get worker status, using defaults:', error);
        }
    },

    isTauri: () => Environment._isTauri,
    isWeb: () => Environment._isWeb,
    isMobile: () => Environment._isMobile,
    isDesktop: () => Environment._isDesktop,
    getPlatformInfo: () => Environment._platformInfo,

    // NEU: Worker URL Getters
    getWorkerHttpUrl: () => Environment._workerUrls.http,
    getWorkerWsUrl: () => Environment._workerUrls.ws,
    getWorkerSseUrl: () => Environment._workerUrls.sse,

    // NEU: Manuelles Setzen (falls nötig)
    setWorkerUrls: (urls) => {
        if (urls.http) Environment._workerUrls.http = urls.http;
        if (urls.ws) Environment._workerUrls.ws = urls.ws;
        if (urls.sse) Environment._workerUrls.sse = urls.sse;
        logger.debug('[Env] Worker URLs manually set:', Environment._workerUrls);
    },

    tauri: {
        // Optional helpers for tauri-specific code
    }
};

export default Environment;
