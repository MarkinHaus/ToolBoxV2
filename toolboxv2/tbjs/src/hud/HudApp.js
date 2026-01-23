/**
 * HUD Application Controller
 * ==========================
 * 
 * Main controller for HUD mode on Desktop
 * Coordinates WebSocket, Widgets, and Settings
 * 
 * Note: For standalone use, the embedded script in hud.html provides
 * similar functionality. This module is for programmatic integration.
 * 
 * Usage:
 * ```javascript
 * import { HudApp } from './hud/HudApp.js';
 * 
 * const app = new HudApp({
 *     widgetContainer: document.getElementById('widget-area'),
 *     onStatusChange: (status) => updateStatusUI(status)
 * });
 * 
 * await app.init();
 * ```
 */

import { HudWebSocket } from './HudWebSocket.js';
import { WidgetArea } from './WidgetArea.js';
import { getApiUrls, isTauri, switchMode } from '../core/platform.js';

export class HudApp {
    /**
     * Create HUD Application
     * @param {Object} options - Configuration options
     * @param {HTMLElement} options.widgetContainer - Container for widgets
     * @param {Function} options.onStatusChange - Callback for status updates
     * @param {Function} options.onConnect - Callback when connected
     * @param {Function} options.onDisconnect - Callback when disconnected
     */
    constructor(options = {}) {
        this.options = options;
        this.ws = null;
        this.widgetArea = null;
        this.connected = false;
        this.startTime = Date.now();
        
        // Callbacks
        this.onStatusChange = options.onStatusChange || (() => {});
        this.onConnect = options.onConnect || (() => {});
        this.onDisconnect = options.onDisconnect || (() => {});
    }
    
    /**
     * Initialize the HUD application
     */
    async init() {
        console.log('[HudApp] Initializing...');
        
        // Setup widget area
        if (this.options.widgetContainer) {
            this.widgetArea = new WidgetArea({
                container: this.options.widgetContainer,
                onWidgetClick: (id) => this.handleWidgetClick(id)
            });
            this.widgetArea.showLoading();
        }
        
        // Setup WebSocket
        this.ws = new HudWebSocket({
            onConnect: () => this.handleConnect(),
            onDisconnect: () => this.handleDisconnect(),
            onMessage: (data) => this.handleMessage(data),
            onError: (err) => this.handleError(err)
        });
        
        // Connect
        await this.ws.connect();
    }
    
    /**
     * Handle WebSocket connect
     */
    handleConnect() {
        console.log('[HudApp] Connected');
        this.connected = true;
        this.startTime = Date.now();
        
        // Request initial data
        this.ws.requestStatus();
        this.ws.requestWidgets();
        
        // Notify
        this.onConnect();
    }
    
    /**
     * Handle WebSocket disconnect
     */
    handleDisconnect() {
        console.log('[HudApp] Disconnected');
        this.connected = false;
        
        if (this.widgetArea) {
            this.widgetArea.showError('Connection lost. Click refresh to retry.');
        }
        
        this.onDisconnect();
    }
    
    /**
     * Handle incoming WebSocket message
     * @param {Object} data - Parsed message data
     */
    handleMessage(data) {
        const msgType = data.type || data.event;
        
        switch (msgType) {
            case 'status':
            case 'system_status':
            case 'worker_status':
                this.handleStatus(data.data || data);
                break;
                
            case 'widgets':
            case 'widget_list':
                this.handleWidgets(data.widgets || data.data || []);
                break;
                
            case 'widget_update':
                this.handleWidgetUpdate(data);
                break;
                
            case 'pong':
                // Heartbeat response - ignore
                break;
                
            default:
                // Handle generic messages that might be status updates
                if (data.running !== undefined || data.worker_running !== undefined) {
                    this.handleStatus(data);
                }
        }
    }
    
    /**
     * Handle status update
     * @param {Object} data - Status data
     */
    handleStatus(data) {
        // Update widget area with system status
        if (this.widgetArea) {
            this.widgetArea.updateSystemWidget({
                ...data,
                uptime_seconds: data.uptime_seconds || 
                    Math.floor((Date.now() - this.startTime) / 1000)
            });
        }
        
        // Notify listeners
        this.onStatusChange({
            connected: true,
            isRemote: data.is_remote || data.mode === 'remote',
            workerRunning: data.worker_running ?? data.running ?? true,
            modCount: data.active_mods || data.mod_count || 0,
            uptime: data.uptime_seconds || Math.floor((Date.now() - this.startTime) / 1000)
        });
    }
    
    /**
     * Handle widgets list
     * @param {Array} widgets - Array of widgets
     */
    handleWidgets(widgets) {
        if (this.widgetArea) {
            this.widgetArea.setWidgets(widgets);
        }
    }
    
    /**
     * Handle widget update
     * @param {Object} data - Widget update data
     */
    handleWidgetUpdate(data) {
        if (this.widgetArea && data.widget_id) {
            this.widgetArea.updateWidget(data.widget_id, data);
        }
    }
    
    /**
     * Handle WebSocket error
     * @param {Error} err - Error object
     */
    handleError(err) {
        console.error('[HudApp] WebSocket error:', err);
        
        if (this.widgetArea) {
            this.widgetArea.showError('Connection error. Click refresh to retry.');
        }
    }
    
    /**
     * Handle widget click
     * @param {string} widgetId - Widget ID that was clicked
     */
    handleWidgetClick(widgetId) {
        console.log('[HudApp] Widget clicked:', widgetId);
        // Can be extended to handle widget actions
    }
    
    /**
     * Refresh connection
     */
    refresh() {
        if (this.ws) {
            this.ws.resetReconnects();
            this.ws.disconnect();
        }
        
        if (this.widgetArea) {
            this.widgetArea.showLoading();
        }
        
        // Reconnect
        if (this.ws) {
            this.ws.connect();
        }
    }
    
    /**
     * Switch to App mode
     */
    async switchToApp() {
        try {
            await switchMode('app');
        } catch (e) {
            console.error('[HudApp] Failed to switch mode:', e);
        }
    }
    
    /**
     * Send a message to the server
     * @param {Object} data - Data to send
     */
    send(data) {
        if (this.ws) {
            return this.ws.send(data);
        }
        return false;
    }
    
    /**
     * Check if connected
     * @returns {boolean} True if connected
     */
    isConnected() {
        return this.connected;
    }
    
    /**
     * Destroy the application
     */
    destroy() {
        if (this.ws) {
            this.ws.disconnect();
            this.ws = null;
        }
        
        if (this.widgetArea) {
            this.widgetArea.clear();
            this.widgetArea = null;
        }
        
        this.connected = false;
    }
}

export default HudApp;
