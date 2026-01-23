/**
 * Desktop Platform Detection & Integration
 * =========================================
 * 
 * Detects if running in Tauri (desktop/mobile) and provides
 * platform-specific functionality.
 * 
 * UPDATED: Added HUD Mode support and async platform detection
 */

// Check if running in Tauri
export const isTauri = () => {
    return typeof window !== 'undefined' && 
           window.__TAURI__ !== undefined;
};

// Check if mobile (Tauri mobile) - synchronous fallback
export const isMobile = () => {
    if (!isTauri()) {
        // Fallback to user agent detection for web
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
    // Check for mobile user agent or Tauri mobile
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

// Async check if mobile (more accurate for Tauri)
export const isMobileAsync = async () => {
    if (!isTauri()) {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
    
    try {
        const { invoke } = window.__TAURI__.core;
        return await invoke('is_mobile');
    } catch (e) {
        console.warn('[Platform] Could not detect platform:', e);
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
};

// Check if desktop
export const isDesktop = () => {
    return isTauri() && !isMobile();
};

// Async check if desktop
export const isDesktopAsync = async () => {
    return !(await isMobileAsync());
};

// Check if web browser
export const isWeb = () => {
    return !isTauri();
};

// Get platform name
export const getPlatform = () => {
    if (isDesktop()) return 'desktop';
    if (isMobile()) return 'mobile';
    return 'web';
};

// Async get platform name
export const getPlatformAsync = async () => {
    if (!isTauri()) return 'web';
    const mobile = await isMobileAsync();
    return mobile ? 'mobile' : 'desktop';
};

/**
 * Get API URLs from Tauri
 * Returns { api_url, ws_url, is_remote }
 */
export const getApiUrls = async () => {
    if (!isTauri()) {
        // Default for web development
        return {
            api_url: 'http://localhost:5000/api',
            ws_url: 'ws://localhost:5001',
            is_remote: false
        };
    }
    
    try {
        const { invoke } = window.__TAURI__.core;
        return await invoke('get_api_urls');
    } catch (e) {
        console.error('[Platform] Could not get API URLs:', e);
        return {
            api_url: 'http://localhost:5000/api',
            ws_url: 'ws://localhost:5001',
            is_remote: false
        };
    }
};

/**
 * Switch mode (desktop only) - app or hud
 */
export const switchMode = async (mode) => {
    if (!isTauri()) {
        console.warn('[Platform] switchMode not available in web');
        return;
    }
    
    try {
        const { invoke } = window.__TAURI__.core;
        await invoke('switch_mode', { mode });
    } catch (e) {
        console.error('[Platform] Switch mode failed:', e);
    }
};

/**
 * Get current mode (desktop only)
 */
export const getCurrentMode = async () => {
    if (!isTauri()) return 'app';
    
    try {
        const { invoke } = window.__TAURI__.core;
        return await invoke('get_current_mode');
    } catch (e) {
        return 'app';
    }
};

/**
 * Tauri API wrapper with fallbacks
 */
export class TauriAPI {
    constructor() {
        this.available = isTauri();
    }

    /**
     * Invoke a Tauri command
     */
    async invoke(command, args = {}) {
        if (!this.available) {
            console.warn(`Tauri not available, cannot invoke: ${command}`);
            return null;
        }

        try {
            const { invoke } = window.__TAURI__.core;
            return await invoke(command, args);
        } catch (error) {
            console.error(`Tauri invoke error (${command}):`, error);
            throw error;
        }
    }

    /**
     * Start the Python worker
     */
    async startWorker() {
        return this.invoke('start_worker');
    }

    /**
     * Stop the Python worker
     */
    async stopWorker() {
        return this.invoke('stop_worker');
    }

    /**
     * Get worker status
     */
    async getWorkerStatus() {
        return this.invoke('get_worker_status');
    }

    /**
     * Set API endpoint
     */
    async setApiEndpoint(endpoint) {
        return this.invoke('set_api_endpoint', { endpoint });
    }

    /**
     * Get data paths
     */
    async getDataPaths() {
        return this.invoke('get_data_paths');
    }

    /**
     * Check worker health
     */
    async checkWorkerHealth() {
        return this.invoke('check_worker_health');
    }

    /**
     * Get API URLs (for WebSocket connection)
     */
    async getApiUrls() {
        return getApiUrls();
    }

    /**
     * Switch mode (app/hud) - Desktop only
     */
    async switchMode(mode) {
        return switchMode(mode);
    }

    /**
     * Get current mode - Desktop only
     */
    async getCurrentMode() {
        return getCurrentMode();
    }

    /**
     * Show native notification
     */
    async notify(title, body, icon = null) {
        if (!this.available) {
            // Fallback to browser notification
            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification(title, { body, icon });
            }
            return;
        }

        try {
            const { sendNotification, isPermissionGranted, requestPermission } = 
                await import('@tauri-apps/plugin-notification');
            
            let permitted = await isPermissionGranted();
            if (!permitted) {
                const permission = await requestPermission();
                permitted = permission === 'granted';
            }

            if (permitted) {
                await sendNotification({ title, body });
            }
        } catch (error) {
            console.error('Notification error:', error);
        }
    }

    /**
     * Open URL in default browser
     */
    async openUrl(url) {
        if (!this.available) {
            window.open(url, '_blank');
            return;
        }

        try {
            const { open } = await import('@tauri-apps/plugin-shell');
            await open(url);
        } catch (error) {
            console.error('Open URL error:', error);
            window.open(url, '_blank');
        }
    }

    /**
     * Save settings
     */
    async saveSettings(settings) {
        return this.invoke('save_settings', { settings });
    }

    /**
     * Load settings
     */
    async loadSettings() {
        return this.invoke('load_settings');
    }
}

// Singleton instance
export const tauriAPI = new TauriAPI();

/**
 * Platform-specific styles
 */
export const platformStyles = {
    desktop: {
        // Desktop gets system tray, hotkeys, HUD mode
        hasTray: true,
        hasHotkeys: true,
        hasNativeMenus: true,
        hasHudMode: true,
        windowControls: 'native'
    },
    mobile: {
        // Mobile gets touch-friendly UI, no HUD
        hasTray: false,
        hasHotkeys: false,
        hasNativeMenus: false,
        hasHudMode: false,
        hasTabBar: true,
        windowControls: 'none'
    },
    web: {
        // Web is standard browser, no HUD
        hasTray: false,
        hasHotkeys: true, // keyboard shortcuts still work
        hasNativeMenus: false,
        hasHudMode: false,
        windowControls: 'none'
    }
};

/**
 * Get current platform styles
 */
export const getCurrentPlatformStyles = () => {
    return platformStyles[getPlatform()];
};

/**
 * Get current platform styles (async)
 */
export const getCurrentPlatformStylesAsync = async () => {
    const platform = await getPlatformAsync();
    return platformStyles[platform];
};

/**
 * Platform Detection Helper Object
 * Provides a unified interface for all platform checks
 */
export const Platform = {
    isTauri,
    isMobile,
    isMobileAsync,
    isDesktop,
    isDesktopAsync,
    isWeb,
    getPlatform,
    getPlatformAsync,
    getApiUrls,
    switchMode,
    getCurrentMode,
    tauriAPI,
    platformStyles,
    getCurrentPlatformStyles,
    getCurrentPlatformStylesAsync
};

export default {
    isTauri,
    isMobile,
    isMobileAsync,
    isDesktop,
    isDesktopAsync,
    isWeb,
    getPlatform,
    getPlatformAsync,
    getApiUrls,
    switchMode,
    getCurrentMode,
    tauriAPI,
    platformStyles,
    getCurrentPlatformStyles,
    getCurrentPlatformStylesAsync,
    Platform
};
