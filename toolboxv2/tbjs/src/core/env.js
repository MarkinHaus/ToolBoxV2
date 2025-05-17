// tbjs/core/env.js
// Detects and provides information about the current runtime environment.
// Original: Logic checking window.__TAURI__ in original autoDetection.js and index.js.

import logger from './logger.js';

const Environment = {
    _isTauri: undefined,
    _isMobile: undefined, // Specifically for Tauri mobile if detectable

    detect: () => {
        Environment._isTauri = typeof window !== 'undefined' && !!window.__TAURI__;
        if (Environment._isTauri) {
            // Try to detect if it's Tauri mobile, this might need refinement
            // based on how Tauri mobile exposes itself (e.g., specific __TAURI__ properties or navigator.userAgent checks)
            // For now, a placeholder:
            // Environment._isMobile = window.__TAURI__ && (window.__TAURI__.platform === 'android' || window.__TAURI__.platform === 'ios');
            // logger.log(`[Env] Tauri detected. Mobile: ${Environment._isMobile}`);
            logger.log(`[Env] Tauri detected.`);

            // Original autoDetection.js logic for spawning commands might be exposed here or in a tauri specific submodule
            // Example: Environment.tauriCommands.run('get-version').then(...)
        } else {
            // logger.log('[Env] Web environment detected.');
        }
    },

    isTauri: () => Environment._isTauri,
    isWeb: () => !Environment._isTauri,
    isMobile: () => Environment._isTauri && Environment._isMobile, // Only true if Tauri and mobile

    // Placeholder for Tauri specific APIs that might be abstracted
    tauri: {
        // Example:
        // async invoke(command, args) {
        //     if (!Environment.isTauri()) {
        //         logger.warn('[Env] Tauri invoke called in non-Tauri environment.');
        //         return Promise.reject('Not in Tauri environment');
        //     }
        //     return window.__TAURI__.invoke(command, args);
        // },
        // async showWindow(label) { ... }
        // This overlaps with TB.api.request which already handles invoke.
        // This tauri submodule could be for non-API tauri interactions like window management.
    }
};

export default Environment;
