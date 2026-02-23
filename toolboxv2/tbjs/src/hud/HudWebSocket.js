/**
 * HUD WebSocket Connection
 * ========================
 * 
 * Connects to EXISTING ToolBoxV2 WebSocket API
 * This is a lightweight wrapper specifically for HUD use
 * 
 * Usage:
 * ```javascript
 * import { HudWebSocket } from './hud/HudWebSocket.js';
 * 
 * const ws = new HudWebSocket({
 *     onConnect: () => console.log('Connected'),
 *     onDisconnect: () => console.log('Disconnected'),
 *     onMessage: (data) => console.log('Message:', data),
 *     onError: (err) => console.error('Error:', err)
 * });
 * 
 * await ws.connect();
 * ws.send({ type: 'get_status' });
 * ```
 */

import { getApiUrls, isTauri } from '../core/platform.js';

export class HudWebSocket {
    /**
     * Create a HUD WebSocket connection
     * @param {Object} callbacks - Event callbacks
     * @param {Function} callbacks.onConnect - Called when connected
     * @param {Function} callbacks.onDisconnect - Called when disconnected
     * @param {Function} callbacks.onMessage - Called with parsed message data
     * @param {Function} callbacks.onError - Called on error
     */
    constructor(callbacks = {}) {
        this.ws = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnects = 5;
        this.reconnectDelay = 2000;
        this.heartbeatInterval = null;
        this.heartbeatRate = 30000; // 30 seconds
        
        this.callbacks = {
            onConnect: callbacks.onConnect || (() => {}),
            onDisconnect: callbacks.onDisconnect || (() => {}),
            onMessage: callbacks.onMessage || (() => {}),
            onError: callbacks.onError || (() => {})
        };
    }
    
    /**
     * Connect to WebSocket server
     * Uses existing ToolBoxV2 API URL
     * CRITICAL: Must include authentication token to avoid session invalidation
     */
    async connect() {
        try {
            // Get authentication token BEFORE connecting
            // Try multiple sources to ensure we have the token
            let token = null;
            if (typeof TB !== 'undefined') {
                if (TB.state?.get) {
                    token = TB.state.get('user.token');
                }
                if (!token && TB.user?.getToken) {
                    token = TB.user.getToken();
                }
            }

            // If still no token, check localStorage directly
            if (!token) {
                try {
                    const storedSession = localStorage.getItem('tbjs_user_session');
                    if (storedSession) {
                        const session = JSON.parse(storedSession);
                        token = session.token;
                    }
                } catch (e) {
                    console.warn('[HudWS] Failed to read token from localStorage:', e);
                }
            }

            if (!token) {
                const errorMsg = '[HudWS] Cannot connect: No authentication token available. User must be logged in first.';
                console.error(errorMsg);
                this.callbacks.onError(new Error(errorMsg));
                return;
            }

            const urls = await getApiUrls();
            // Add session_token as query parameter to authenticate the WebSocket connection
            const wsUrl = `${urls.ws_url}?session_token=${encodeURIComponent(token)}`;

            console.log('[HudWS] Connecting to:', wsUrl.replace(/session_token=[^&]+/, 'session_token=***'));
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('[HudWS] Connected');
                this.connected = true;
                this.reconnectAttempts = 0;
                this.startHeartbeat();
                this.callbacks.onConnect();
            };
            
            this.ws.onclose = (e) => {
                console.log('[HudWS] Closed:', e.code, e.reason);
                this.connected = false;
                this.stopHeartbeat();
                this.callbacks.onDisconnect();
                
                // Only reconnect if not manually closed
                if (e.code !== 1000) {
                    this.scheduleReconnect();
                }
            };
            
            this.ws.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    this.callbacks.onMessage(data);
                } catch (err) {
                    console.error('[HudWS] Parse error:', err);
                    // Still pass raw data
                    this.callbacks.onMessage({ raw: e.data, parseError: true });
                }
            };
            
            this.ws.onerror = (e) => {
                console.error('[HudWS] Error:', e);
                this.callbacks.onError(e);
            };
            
        } catch (e) {
            console.error('[HudWS] Connection failed:', e);
            this.callbacks.onError(e);
            this.scheduleReconnect();
        }
    }
    
    /**
     * Send data to the server
     * @param {Object} data - Data to send (will be JSON stringified)
     * @returns {boolean} True if sent successfully
     */
    send(data) {
        if (!this.connected || !this.ws) {
            console.warn('[HudWS] Cannot send - not connected');
            return false;
        }
        
        try {
            this.ws.send(JSON.stringify(data));
            return true;
        } catch (e) {
            console.error('[HudWS] Send error:', e);
            return false;
        }
    }
    
    /**
     * Request status from server
     */
    requestStatus() {
        return this.send({ type: 'get_status' });
    }
    
    /**
     * Request widgets from server
     */
    requestWidgets() {
        return this.send({ type: 'get_widgets' });
    }
    
    /**
     * Start heartbeat to keep connection alive
     */
    startHeartbeat() {
        this.stopHeartbeat();
        this.heartbeatInterval = setInterval(() => {
            this.send({ type: 'ping' });
        }, this.heartbeatRate);
    }
    
    /**
     * Stop heartbeat
     */
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    /**
     * Schedule a reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnects) {
            console.log('[HudWS] Max reconnects reached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * this.reconnectAttempts, 
            10000
        );
        
        console.log(`[HudWS] Reconnecting in ${delay}ms (${this.reconnectAttempts}/${this.maxReconnects})`);
        
        setTimeout(() => this.connect(), delay);
    }
    
    /**
     * Reset reconnection attempts (call after manual refresh)
     */
    resetReconnects() {
        this.reconnectAttempts = 0;
    }
    
    /**
     * Disconnect from server
     */
    disconnect() {
        this.stopHeartbeat();
        this.maxReconnects = 0; // Prevent auto-reconnect
        
        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }
        
        this.connected = false;
    }
    
    /**
     * Check if currently connected
     * @returns {boolean} True if connected
     */
    isConnected() {
        return this.connected;
    }
    
    /**
     * Get the WebSocket ready state
     * @returns {number} WebSocket.CONNECTING (0), OPEN (1), CLOSING (2), CLOSED (3)
     */
    getReadyState() {
        return this.ws ? this.ws.readyState : WebSocket.CLOSED;
    }
}

export default HudWebSocket;
