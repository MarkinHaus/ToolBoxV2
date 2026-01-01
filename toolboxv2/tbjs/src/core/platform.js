/**
 * Desktop Platform Detection & Integration
 * =========================================
 * 
 * Detects if running in Tauri (desktop/mobile) and provides
 * platform-specific functionality.
 */

// Check if running in Tauri
export const isTauri = () => {
    return typeof window !== 'undefined' && 
           window.__TAURI__ !== undefined;
};

// Check if mobile (Tauri mobile)
export const isMobile = () => {
    if (!isTauri()) return false;
    // Check for mobile user agent or Tauri mobile
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

// Check if desktop
export const isDesktop = () => {
    return isTauri() && !isMobile();
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
            const { invoke } = await import('@tauri-apps/api/core');
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
}

// Singleton instance
export const tauriAPI = new TauriAPI();

/**
 * Platform-specific styles
 */
export const platformStyles = {
    desktop: {
        // Desktop gets system tray, hotkeys, etc.
        hasTray: true,
        hasHotkeys: true,
        hasNativeMenus: true,
        windowControls: 'native'
    },
    mobile: {
        // Mobile gets touch-friendly UI
        hasTray: false,
        hasHotkeys: false,
        hasNativeMenus: false,
        windowControls: 'none'
    },
    web: {
        // Web is standard browser
        hasTray: false,
        hasHotkeys: true, // keyboard shortcuts still work
        hasNativeMenus: false,
        windowControls: 'none'
    }
};

/**
 * Get current platform styles
 */
export const getCurrentPlatformStyles = () => {
    return platformStyles[getPlatform()];
};

export default {
    isTauri,
    isMobile,
    isDesktop,
    isWeb,
    getPlatform,
    tauriAPI,
    platformStyles,
    getCurrentPlatformStyles
};
