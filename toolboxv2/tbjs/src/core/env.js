import logger from './logger.js';

const Environment = {
    _isTauri: false,
    _isWeb: true,
    _isMobile: false,
    _isDesktop: false,
    _platformInfo: {},

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
    },

    isTauri: () => Environment._isTauri,
    isWeb: () => Environment._isWeb,
    isMobile: () => Environment._isMobile,
    isDesktop: () => Environment._isDesktop,
    getPlatformInfo: () => Environment._platformInfo,

    tauri: {
        // Optional helpers for tauri-specific code
    }
};

export default Environment;
