// tbjs/core/notifications.js
// ============================================================================
// Cross-Platform Notification System for ToolBoxV2
// ============================================================================
//
// Handles incoming notifications from the backend (via WebSocket) and
// displays them appropriately depending on the runtime environment:
//
//   - Web Browser (foreground): Glassmorphism HTML toast
//   - Web Browser (background tab): Browser Notification API
//   - Tauri Desktop App: Native OS notification via Tauri plugin
//
// Backend sends notifications as WebSocket messages with this shape:
//   { "type": "notification", "data": { "title", "content", "level", "icon" } }
//
// Usage:
//   import { NotificationManager } from '../core/notifications.js';
//
//   // Auto-init (listens on TB.events 'ws:message')
//   const manager = new NotificationManager();
//   manager.init();
//
//   // Manual trigger
//   manager.show({ title: 'Hello', content: 'World', level: 'success' });
//
// Integration points:
//   - WsManager (core/ws.js): emits 'ws:message' → manager picks up type=notification
//   - HudWebSocket (hud/HudWebSocket.js): onMessage callback → manager.handleWsMessage()
//   - Can also be used standalone for client-side notifications
//
// File location: toolboxv2/tbjs/src/core/notifications.js

import logger from './logger.js';
import env from './env.js';

// ============================================================================
// Constants
// ============================================================================

const TOAST_TIMEOUT_MS = 5000;
const MAX_VISIBLE_TOASTS = 5;
const CONTAINER_ID = 'tb-notification-container';
const STYLE_ID = 'tb-notification-styles';

const LEVEL_CONFIG = {
    info:    { icon: 'ℹ️', borderColor: '#6366f1' },
    success: { icon: '✅', borderColor: '#27ae60' },
    warning: { icon: '⚠️', borderColor: '#f39c12' },
    error:   { icon: '❌', borderColor: '#e74c3c' },
};

// ============================================================================
// NotificationManager
// ============================================================================

export class NotificationManager {
    constructor() {
        this._container = null;
        this._activeToasts = [];
        this._initialized = false;
        this._browserPermissionAsked = false;
    }

    // ----------------------------------------------------------------
    // Lifecycle
    // ----------------------------------------------------------------

    /**
     * Initialize: inject styles, create container, bind WS events.
     * Safe to call multiple times (idempotent).
     */
    init() {
        if (this._initialized) return;
        this._initialized = true;

        this._injectStyles();
        this._ensureContainer();
        this._bindWsEvents();
        this._requestBrowserPermission();

        logger.log('[Notify] NotificationManager initialized');
    }

    /**
     * Tear down: remove container, styles, unbind events.
     */
    destroy() {
        if (this._container) {
            this._container.remove();
            this._container = null;
        }
        const style = document.getElementById(STYLE_ID);
        if (style) style.remove();

        this._activeToasts = [];
        this._initialized = false;
    }

    // ----------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------

    /**
     * Handle a raw WebSocket message object.
     * Call this from HudWebSocket.onMessage or any WS handler.
     *
     * @param {Object} data - Parsed WS message
     * @returns {boolean} true if it was a notification and was handled
     */
    handleWsMessage(data) {
        if (!data || data.type !== 'notification') return false;

        const payload = data.data || data;
        this.show({
            title:   payload.title   || 'Notification',
            content: payload.content || payload.message || '',
            level:   payload.level   || 'info',
            icon:    payload.icon    || null,
        });
        return true;
    }

    /**
     * Show a notification (main entry point).
     *
     * @param {Object} opts
     * @param {string} opts.title
     * @param {string} opts.content
     * @param {string} [opts.level='info']  - info | success | warning | error
     * @param {string} [opts.icon]          - optional icon URL (Tauri native)
     * @param {number} [opts.timeout]       - auto-dismiss ms (0 = sticky)
     */
    show(opts = {}) {
        const { title, content, level = 'info', icon = null, timeout = TOAST_TIMEOUT_MS } = opts;

        if (env.isTauri()) {
            this._showTauriNative(title, content, icon, level);
        } else {
            this._showWebToast(title, content, level, timeout);

            // Additionally: Browser Notification API when tab is hidden
            if (document.visibilityState === 'hidden') {
                this._showBrowserNotification(title, content, icon);
            }
        }
    }

    // ----------------------------------------------------------------
    // Tauri Native
    // ----------------------------------------------------------------

    /** @private */
    async _showTauriNative(title, body, icon, level) {
        try {
            const tauriNotify = window.__TAURI__?.notification;

            if (tauriNotify && tauriNotify.sendNotification) {
                let granted = await tauriNotify.isPermissionGranted();
                if (!granted) {
                    const result = await tauriNotify.requestPermission();
                    granted = result === 'granted';
                }

                if (granted) {
                    tauriNotify.sendNotification({
                        title,
                        body,
                        ...(icon ? { icon } : {}),
                    });
                    logger.debug('[Notify] Tauri native notification sent');
                    return;
                }
            }

            // Tauri v2 plugin API (@tauri-apps/plugin-notification)
            // Falls der obige Weg nicht funktioniert
            try {
                const plugin = await import('@tauri-apps/plugin-notification');
                let perm = await plugin.isPermissionGranted();
                if (!perm) {
                    perm = (await plugin.requestPermission()) === 'granted';
                }
                if (perm) {
                    await plugin.sendNotification({ title, body });
                    logger.debug('[Notify] Tauri v2 plugin notification sent');
                    return;
                }
            } catch (_) {
                // Plugin nicht verfügbar
            }

            // Fallback: Web-Toast auch in Tauri
            logger.warn('[Notify] Tauri notification API not available, fallback to web toast');
            this._showWebToast(title, body, level, TOAST_TIMEOUT_MS);

        } catch (e) {
            logger.error('[Notify] Tauri notification error:', e);
            this._showWebToast(title, body, level, TOAST_TIMEOUT_MS);
        }
    }

    // ----------------------------------------------------------------
    // Browser Notification API (background tab)
    // ----------------------------------------------------------------

    /** @private */
    _requestBrowserPermission() {
        if (env.isTauri()) return;
        if (this._browserPermissionAsked) return;
        if (!('Notification' in window)) return;

        this._browserPermissionAsked = true;

        if (Notification.permission !== 'granted' && Notification.permission !== 'denied') {
            // Don't ask immediately - wait for user interaction
            const askOnce = () => {
                Notification.requestPermission();
                document.removeEventListener('click', askOnce);
            };
            document.addEventListener('click', askOnce, { once: true });
        }
    }

    /** @private */
    _showBrowserNotification(title, body, icon) {
        if (!('Notification' in window)) return;
        if (Notification.permission !== 'granted') return;

        try {
            new Notification(title, {
                body,
                ...(icon ? { icon } : {}),
                silent: false,
            });
        } catch (e) {
            logger.debug('[Notify] Browser notification failed:', e);
        }
    }

    // ----------------------------------------------------------------
    // Web Toast (HTML/CSS)
    // ----------------------------------------------------------------

    /** @private */
    _showWebToast(title, message, level = 'info', timeout = TOAST_TIMEOUT_MS) {
        this._ensureContainer();

        // Limit visible toasts
        while (this._activeToasts.length >= MAX_VISIBLE_TOASTS) {
            this._removeToast(this._activeToasts[0]);
        }

        const cfg = LEVEL_CONFIG[level] || LEVEL_CONFIG.info;

        const toast = document.createElement('div');
        toast.className = `tb-toast tb-toast--${level}`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'polite');

        toast.innerHTML = `
            <div class="tb-toast__icon">${cfg.icon}</div>
            <div class="tb-toast__body">
                <div class="tb-toast__title">${this._esc(title)}</div>
                ${message ? `<div class="tb-toast__msg">${this._esc(message)}</div>` : ''}
            </div>
            <button class="tb-toast__close" aria-label="Close">×</button>
        `;

        // Close button
        toast.querySelector('.tb-toast__close').addEventListener('click', () => {
            this._removeToast(toast);
        });

        // Auto-dismiss
        let timerId = null;
        if (timeout > 0) {
            timerId = setTimeout(() => this._removeToast(toast), timeout);
        }
        toast._timerId = timerId;

        this._container.appendChild(toast);
        this._activeToasts.push(toast);

        // Trigger enter animation
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                toast.classList.add('tb-toast--visible');
            });
        });
    }

    /** @private */
    _removeToast(toast) {
        if (!toast || !toast.parentElement) return;

        if (toast._timerId) clearTimeout(toast._timerId);

        toast.classList.remove('tb-toast--visible');

        const onEnd = () => {
            if (toast.parentElement) toast.remove();
            const idx = this._activeToasts.indexOf(toast);
            if (idx !== -1) this._activeToasts.splice(idx, 1);
        };

        toast.addEventListener('transitionend', onEnd, { once: true });
        // Safety fallback if transitionend never fires
        setTimeout(onEnd, 400);
    }

    // ----------------------------------------------------------------
    // DOM Setup
    // ----------------------------------------------------------------

    /** @private */
    _ensureContainer() {
        if (this._container && document.body.contains(this._container)) return;

        let existing = document.getElementById(CONTAINER_ID);
        if (existing) {
            this._container = existing;
            return;
        }

        this._container = document.createElement('div');
        this._container.id = CONTAINER_ID;
        document.body.appendChild(this._container);
    }

    /** @private */
    _injectStyles() {
        if (document.getElementById(STYLE_ID)) return;

        const style = document.createElement('style');
        style.id = STYLE_ID;
        style.textContent = `
/* ── Toast Container ─────────────────────────────────────────── */
#${CONTAINER_ID} {
    position: fixed;
    top: 16px;
    right: 16px;
    z-index: 99999;
    display: flex;
    flex-direction: column;
    gap: 8px;
    pointer-events: none;
    max-height: 100vh;
    overflow: visible;
}

/* ── Single Toast ────────────────────────────────────────────── */
.tb-toast {
    pointer-events: auto;
    display: flex;
    align-items: flex-start;
    width: 340px;
    max-width: calc(100vw - 32px);
    padding: 12px 14px;
    border-radius: 10px;

    /* Glassmorphism */
    background: rgba(22, 22, 38, 0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid var(--tb-toast-accent, #6366f1);

    color: #e8e8f0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35),
                0 0 0 1px rgba(255, 255, 255, 0.04) inset;

    /* Enter/exit animation */
    transform: translateX(110%);
    opacity: 0;
    transition: transform 0.3s cubic-bezier(0.22, 1, 0.36, 1),
                opacity 0.25s ease;
}

.tb-toast--visible {
    transform: translateX(0);
    opacity: 1;
}

/* ── Level Accents ───────────────────────────────────────────── */
.tb-toast--info    { --tb-toast-accent: #6366f1; }
.tb-toast--success { --tb-toast-accent: #27ae60; }
.tb-toast--warning { --tb-toast-accent: #f39c12; }
.tb-toast--error   { --tb-toast-accent: #e74c3c; }

/* ── Icon ────────────────────────────────────────────────────── */
.tb-toast__icon {
    flex-shrink: 0;
    font-size: 18px;
    line-height: 1;
    margin-right: 10px;
    margin-top: 1px;
}

/* ── Body ────────────────────────────────────────────────────── */
.tb-toast__body {
    flex: 1;
    min-width: 0;
}

.tb-toast__title {
    font-weight: 600;
    font-size: 13px;
    line-height: 1.3;
    color: #fff;
    word-break: break-word;
}

.tb-toast__msg {
    font-size: 12px;
    line-height: 1.4;
    color: rgba(255, 255, 255, 0.6);
    margin-top: 2px;
    word-break: break-word;
}

/* ── Close Button ────────────────────────────────────────────── */
.tb-toast__close {
    flex-shrink: 0;
    background: none;
    border: none;
    color: rgba(255, 255, 255, 0.3);
    font-size: 18px;
    line-height: 1;
    cursor: pointer;
    padding: 0 0 0 8px;
    transition: color 0.15s ease;
}
.tb-toast__close:hover {
    color: #fff;
}

/* ── Mobile responsive ───────────────────────────────────────── */
@media (max-width: 480px) {
    #${CONTAINER_ID} {
        top: auto;
        bottom: 72px; /* above MobileNav */
        right: 8px;
        left: 8px;
    }
    .tb-toast {
        width: 100%;
    }
}
`;
        document.head.appendChild(style);
    }

    // ----------------------------------------------------------------
    // WS Event Binding
    // ----------------------------------------------------------------

    /** @private */
    _bindWsEvents() {
        // If TB global is available, listen on its event bus
        // This catches messages from both WsManager (core/ws.js) and any other WS source
        try {
            if (typeof TB !== 'undefined' && TB.events) {
                TB.events.on('ws:message', (evt) => {
                    if (evt && evt.data) {
                        this.handleWsMessage(evt.data);
                    }
                });
                logger.debug('[Notify] Bound to TB.events ws:message');
            }
        } catch (_) {
            // TB not available yet — caller should use handleWsMessage() manually
        }
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    /** @private */
    _esc(text) {
        if (!text) return '';
        const d = document.createElement('div');
        d.textContent = text;
        return d.innerHTML;
    }
}

// ============================================================================
// Singleton instance (optional convenience)
// ============================================================================

let _instance = null;

/**
 * Get or create the global NotificationManager singleton.
 * @returns {NotificationManager}
 */
export function getNotificationManager() {
    if (!_instance) {
        _instance = new NotificationManager();
    }
    return _instance;
}

export default NotificationManager;
