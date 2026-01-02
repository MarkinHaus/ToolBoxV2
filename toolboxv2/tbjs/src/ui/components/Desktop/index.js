/**
 * Desktop/Mobile Platform Components
 * ===================================
 *
 * Platform-specific UI components for Tauri desktop and mobile apps.
 */

import { isTauri, isDesktop, isMobile, tauriAPI } from '../../../core/platform.js';
import './Desktop.css';

/**
 * Quick Capture Popup - Floating capture window (Ctrl+Shift+C)
 */
export class QuickCapturePopup {
    constructor(options = {}) {
        this.onCapture = options.onCapture || (() => {});
        this.element = null;
        this.isVisible = false;
    }

    create() {
        if (!isDesktop()) return;

        this.element = document.createElement('div');
        this.element.className = 'tb-quick-capture';
        this.element.innerHTML = `
            <div class="tb-quick-capture-header">
                <span>⚡ Quick Capture</span>
                <button class="tb-quick-capture-close">×</button>
            </div>
            <div class="tb-quick-capture-body">
                <textarea placeholder="Capture your thought... #tags" rows="3"></textarea>
                <div class="tb-quick-capture-actions">
                    <span class="tb-quick-capture-hint">Ctrl+Enter to save</span>
                    <button class="tb-quick-capture-save">Save</button>
                </div>
            </div>
        `;

        document.body.appendChild(this.element);
        this._bindEvents();
    }

    _bindEvents() {
        const closeBtn = this.element.querySelector('.tb-quick-capture-close');
        const saveBtn = this.element.querySelector('.tb-quick-capture-save');
        const textarea = this.element.querySelector('textarea');

        closeBtn.addEventListener('click', () => this.hide());
        saveBtn.addEventListener('click', () => this._save());

        textarea.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this._save();
            }
            if (e.key === 'Escape') {
                this.hide();
            }
        });

        // Global hotkey
        this._keyHandler = (e) => {
            if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'c') {
                e.preventDefault();
                this.toggle();
            }
        };
        document.addEventListener('keydown', this._keyHandler);

        // Listen for custom event (for programmatic triggering)
        window.addEventListener('tb:quickCapture', () => this.show());
    }

    async _save() {
        const textarea = this.element.querySelector('textarea');
        const text = textarea.value.trim();
        if (!text) return;

        // Extract tags
        const tagRegex = /#(\w+)/g;
        const tags = [];
        let match;
        while ((match = tagRegex.exec(text)) !== null) {
            tags.push(match[1]);
        }

        try {
            await this.onCapture({ text, tags });
            textarea.value = '';
            this.hide();
            tauriAPI.notify('Captured!', text.substring(0, 50));
        } catch (error) {
            console.error('Capture failed:', error);
        }
    }

    show() {
        if (!this.element) return;
        this.element.classList.add('visible');
        this.isVisible = true;
        this.element.querySelector('textarea').focus();
    }

    hide() {
        if (!this.element) return;
        this.element.classList.remove('visible');
        this.isVisible = false;
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    destroy() {
        if (this.element) {
            this.element.remove();
            this.element = null;
        }
    }
}

/**
 * Desktop Status Bar - Fixed bar at bottom of screen
 * Maintains persistent WebSocket connection for notifications
 */
export class DesktopStatusBar {
    constructor(options = {}) {
        this.items = options.items || [];
        this.element = null;
        this._updateInterval = null;
        this._updateIntervalMs = options.updateInterval || 10000; // 10 seconds default
        this._ws = null;
        this._wsReconnectTimeout = null;
        this._wsReconnectDelay = 3000;
        this._maxReconnectDelay = 30000;
        this._onNotification = options.onNotification || null;
    }

    create() {
        if (!isDesktop()) return;

        this.element = document.createElement('div');
        this.element.className = 'tb-status-bar';
        this._render();
        document.body.appendChild(this.element);
        document.body.classList.add('platform-desktop');

        // Check HTTP worker status
        this._checkWorkerStatus();

        // Start persistent WebSocket connection
        this._connectWebSocket();

        // Periodic HTTP health check only
        this._updateInterval = setInterval(() => {
            this._checkWorkerStatus();
        }, this._updateIntervalMs);
    }

    _render() {
        this.element.innerHTML = `
            <div class="tb-status-bar-left">
                <span class="tb-status-item" data-id="worker" title="HTTP Worker Status">
                    <span class="tb-status-dot"></span>
                    <span class="tb-status-label">Worker</span>
                </span>
                <span class="tb-status-item" data-id="ws" title="WebSocket Status">
                    <span class="tb-status-dot"></span>
                    <span class="tb-status-label">WS</span>
                </span>
            </div>
            <div class="tb-status-bar-center">
                <span class="tb-status-info" data-id="endpoint"></span>
            </div>
            <div class="tb-status-bar-right">
                <span class="tb-status-item" data-id="hotkey">Ctrl+Shift+C: Capture</span>
            </div>
        `;
    }

    async _checkWorkerStatus() {
        try {
            const response = await fetch(`${this._getApiBase()}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(3000)
            });

            if (response.ok) {
                const status = await response.json();
                this.updateItem('worker', status?.status === 'healthy' ? 'online' : 'warning');

                if (status?.worker_id) {
                    this._updateEndpointInfo(status.worker_id);
                }
            } else {
                this.updateItem('worker', 'warning');
            }
        } catch {
            this.updateItem('worker', 'offline');
            this._updateEndpointInfo('disconnected');
        }
    }

    _connectWebSocket() {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            return; // Already connected
        }

        const wsUrl = this._getWsUrl();
        console.log('[StatusBar] Connecting to WebSocket:', wsUrl);

        try {
            this._ws = new WebSocket(wsUrl);

            this._ws.onopen = () => {
                console.log('[StatusBar] WebSocket connected');
                this.updateItem('ws', 'online');
                this._wsReconnectDelay = 3000; // Reset delay on success
            };

            this._ws.onmessage = async (event) => {
                await this._handleWsMessage(event.data);
            };

            this._ws.onclose = (event) => {
                console.log('[StatusBar] WebSocket closed:', event.code, event.reason);
                this.updateItem('ws', 'offline');
                this._scheduleReconnect();
            };

            this._ws.onerror = (error) => {
                console.error('[StatusBar] WebSocket error:', error);
                this.updateItem('ws', 'offline');
            };
        } catch (error) {
            console.error('[StatusBar] WebSocket connection failed:', error);
            this.updateItem('ws', 'offline');
            this._scheduleReconnect();
        }
    }

    _scheduleReconnect() {
        if (this._wsReconnectTimeout) {
            clearTimeout(this._wsReconnectTimeout);
        }

        console.log(`[StatusBar] Reconnecting in ${this._wsReconnectDelay / 1000}s...`);

        this._wsReconnectTimeout = setTimeout(() => {
            this._connectWebSocket();
        }, this._wsReconnectDelay);

        // Exponential backoff
        this._wsReconnectDelay = Math.min(this._wsReconnectDelay * 1.5, this._maxReconnectDelay);
    }

    async _handleWsMessage(data) {
        try {
            const message = JSON.parse(data);
            console.log('[StatusBar] WS message received:', message);

            // Handle notification events
            if (message.type === 'notification' || message.event === 'notification') {
                await this._showNotification(message.data || message);
            }

            // Handle status updates
            if (message.type === 'status' || message.event === 'status') {
                this._handleStatusUpdate(message.data || message);
            }

            // Call custom handler if provided
            if (this._onNotification) {
                this._onNotification(message);
            }
        } catch (error) {
            console.warn('[StatusBar] Failed to parse WS message:', error);
        }
    }

    async _showNotification(notification) {
        const title = notification.title || 'ToolBox';
        const body = notification.content || notification.message || notification.body || '';
        const icon = notification.icon || null;

        console.log('[StatusBar] Showing notification:', title, body);

        // Use Tauri native notifications
        await tauriAPI.notify(title, body, icon);
    }

    _handleStatusUpdate(status) {
        if (status.worker !== undefined) {
            this.updateItem('worker', status.worker);
        }
        if (status.ws !== undefined) {
            this.updateItem('ws', status.ws);
        }
        if (status.endpoint) {
            this._updateEndpointInfo(status.endpoint);
        }
    }

    _getApiBase() {
        return 'http://localhost:5000';
    }

    _getWsUrl() {
        const wsId = localStorage.getItem('WsID') || `status-${Date.now()}`;
        return `ws://localhost:5001/ws/${wsId}`;
    }

    _updateEndpointInfo(url) {
        const infoEl = this.element?.querySelector('[data-id="endpoint"]');
        if (infoEl) {
            infoEl.textContent = url;
            infoEl.title = `Connected to: ${url}`;
        }
    }

    _updateLabel(id, label) {
        const item = this.element?.querySelector(`[data-id="${id}"]`);
        if (!item) return;
        const labelEl = item.querySelector('.tb-status-label');
        if (labelEl) {
            labelEl.textContent = label;
        }
    }

    updateItem(id, status) {
        const item = this.element?.querySelector(`[data-id="${id}"]`);
        if (!item) return;
        const dot = item.querySelector('.tb-status-dot');
        if (dot) {
            dot.className = `tb-status-dot ${status}`;
        }
    }

    /**
     * Force refresh status and reconnect WebSocket
     */
    async refresh() {
        await this._checkWorkerStatus();
        this._connectWebSocket();
    }

    /**
     * Send a message through the WebSocket connection
     */
    sendMessage(message) {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            const payload = typeof message === 'string' ? message : JSON.stringify(message);
            this._ws.send(payload);
            return true;
        }
        console.warn('[StatusBar] Cannot send message: WebSocket not connected');
        return false;
    }

    /**
     * Check if WebSocket is connected
     */
    isWsConnected() {
        return this._ws && this._ws.readyState === WebSocket.OPEN;
    }

    destroy() {
        // Clear intervals
        if (this._updateInterval) {
            clearInterval(this._updateInterval);
            this._updateInterval = null;
        }

        // Clear reconnect timeout
        if (this._wsReconnectTimeout) {
            clearTimeout(this._wsReconnectTimeout);
            this._wsReconnectTimeout = null;
        }

        // Close WebSocket
        if (this._ws) {
            this._ws.onclose = null; // Prevent reconnect on intentional close
            this._ws.close();
            this._ws = null;
        }

        // Remove element
        if (this.element) {
            this.element.remove();
            document.body.classList.remove('platform-desktop');
        }
    }
}

/**
 * Mobile Bottom Navigation
 */
export class MobileBottomNav {
    constructor(options = {}) {
        this.items = options.items || [];
        this.onNavigate = options.onNavigate || (() => {});
        this.element = null;
        this.activeRoute = '/';
    }

    create() {
        if (!isMobile()) return;

        this.element = document.createElement('nav');
        this.element.className = 'tb-bottom-nav';
        this._render();
        document.body.appendChild(this.element);
        document.body.classList.add('platform-mobile');

        // Listen for route changes
        window.addEventListener('hashchange', () => this._updateActive());
        this._updateActive();
    }

    _render() {
        this.element.innerHTML = this.items.map(item => `
            <button class="tb-bottom-nav-item" data-route="${item.route}">
                <span class="tb-bottom-nav-icon">${item.icon}</span>
                <span class="tb-bottom-nav-label">${item.label}</span>
            </button>
        `).join('');

        this.element.querySelectorAll('.tb-bottom-nav-item').forEach(btn => {
            btn.addEventListener('click', () => {
                const route = btn.dataset.route;
                this.onNavigate(route);
                this.setActive(route);
            });
        });
    }

    _updateActive() {
        const hash = window.location.hash.replace('#', '') || '/';
        this.setActive(hash);
    }

    setActive(route) {
        this.activeRoute = route;
        this.element?.querySelectorAll('.tb-bottom-nav-item').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.route === route);
        });
    }

    destroy() {
        if (this.element) {
            this.element.remove();
            document.body.classList.remove('platform-mobile');
        }
    }
}

/**
 * System Tray Integration (Desktop only)
 */
export class SystemTray {
    constructor(options = {}) {
        this.tooltip = options.tooltip || 'ToolBox';
        this.onAction = options.onAction || (() => {});
    }

    async create() {
        if (!isDesktop()) return;
        // Tray is handled by Tauri backend
        // This class provides JS interface for updates
    }

    async updateStatus(status) {
        try {
            await tauriAPI.invoke('update_tray_status', { status });
        } catch (error) {
            console.warn('Tray update failed:', error);
        }
    }

    async showNotification(title, body) {
        await tauriAPI.notify(title, body);
    }
}

/**
 * Initialize platform-specific UI
 */
export function initPlatformUI(options = {}) {
    const components = {};

    if (isDesktop()) {
        // Desktop: Status bar + Quick capture
        components.statusBar = new DesktopStatusBar();
        components.statusBar.create();

        components.quickCapture = new QuickCapturePopup({
            onCapture: options.onCapture || (async ({ text, tags }) => {
                const response = await fetch('/api/vault/capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, tags })
                });
                if (!response.ok) throw new Error('Capture failed');
            })
        });
        components.quickCapture.create();

        components.tray = new SystemTray();
        components.tray.create();
    }

    if (isMobile()) {
        // Mobile: Bottom navigation
        components.bottomNav = new MobileBottomNav({
            items: options.navItems || [
                { icon: '🏠', label: 'Home', route: '/' },
                { icon: '📝', label: 'Notes', route: '/vault' },
                { icon: '⚡', label: 'Capture', route: '/capture' },
                { icon: '🤖', label: 'Bots', route: '/bots' },
                { icon: '⚙️', label: 'Settings', route: '/settings' }
            ],
            onNavigate: options.onNavigate || ((route) => {
                window.location.hash = route;
            })
        });
        components.bottomNav.create();
    }

    return components;
}

export default {
    QuickCapturePopup,
    DesktopStatusBar,
    MobileBottomNav,
    SystemTray,
    initPlatformUI
};

