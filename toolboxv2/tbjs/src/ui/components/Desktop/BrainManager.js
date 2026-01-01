/**
 * Brain Manager Component
 * =======================
 *
 * Control panel for the Brain System (Discord, Telegram, Vault, Sync).
 * Provides status monitoring, quick actions, and settings.
 */

import { tauriAPI, isDesktop } from '../../../core/platform.js';

/**
 * Brain Manager - Control panel for brain services
 */
export class BrainManager {
    constructor(options = {}) {
        this.container = options.container || null;
        this.apiBase = options.apiBase || '/api';
        this.wsUrl = options.wsUrl || null;
        this.ws = null;
        this.status = {
            discord: { connected: false, users: 0 },
            telegram: { connected: false, users: 0 },
            vault: { connected: false, notes: 0 },
            sync: { active: false, lastSync: null }
        };
        this.activities = [];
        this.onStatusChange = options.onStatusChange || (() => {});
    }

    /**
     * Create and mount the component
     */
    create(container = null) {
        this.container = container || this.container || document.createElement('div');
        this.container.className = 'tb-brain-manager';
        this._render();
        this._bindEvents();
        this._connectWebSocket();
        this._fetchInitialStatus();
        return this.container;
    }

    _render() {
        this.container.innerHTML = `
            <div class="tb-brain-header">
                <h2>🧠 Brain Control</h2>
                <button class="tb-brain-settings-btn" title="Settings">⚙️</button>
            </div>

            <div class="tb-brain-status-grid">
                <div class="tb-brain-card" data-service="discord">
                    <div class="tb-brain-card-icon">💬</div>
                    <div class="tb-brain-card-info">
                        <span class="tb-brain-card-title">Discord</span>
                        <span class="tb-brain-card-status">Checking...</span>
                    </div>
                    <div class="tb-brain-card-dot"></div>
                </div>

                <div class="tb-brain-card" data-service="telegram">
                    <div class="tb-brain-card-icon">📱</div>
                    <div class="tb-brain-card-info">
                        <span class="tb-brain-card-title">Telegram</span>
                        <span class="tb-brain-card-status">Checking...</span>
                    </div>
                    <div class="tb-brain-card-dot"></div>
                </div>

                <div class="tb-brain-card" data-service="vault">
                    <div class="tb-brain-card-icon">📚</div>
                    <div class="tb-brain-card-info">
                        <span class="tb-brain-card-title">Vault</span>
                        <span class="tb-brain-card-status">Checking...</span>
                    </div>
                    <div class="tb-brain-card-dot"></div>
                </div>

                <div class="tb-brain-card" data-service="sync">
                    <div class="tb-brain-card-icon">🔄</div>
                    <div class="tb-brain-card-info">
                        <span class="tb-brain-card-title">Sync</span>
                        <span class="tb-brain-card-status">Checking...</span>
                    </div>
                    <div class="tb-brain-card-dot"></div>
                </div>
            </div>

            <div class="tb-brain-actions">
                <h3>Quick Actions</h3>
                <div class="tb-brain-action-grid">
                    <button class="tb-brain-action" data-action="capture">
                        <span>⚡</span> Capture
                    </button>
                    <button class="tb-brain-action" data-action="note">
                        <span>📝</span> New Note
                    </button>
                    <button class="tb-brain-action" data-action="daily">
                        <span>📅</span> Daily
                    </button>
                    <button class="tb-brain-action" data-action="search">
                        <span>🔍</span> Search
                    </button>
                </div>
            </div>

            <div class="tb-brain-activity">
                <h3>Recent Activity</h3>
                <div class="tb-brain-activity-list">
                    <div class="tb-brain-activity-empty">No recent activity</div>
                </div>
            </div>
        `;
    }

    _bindEvents() {
        // Settings button
        this.container.querySelector('.tb-brain-settings-btn')?.addEventListener('click', () => {
            this._showSettings();
        });

        // Quick actions
        this.container.querySelectorAll('.tb-brain-action').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                this._handleAction(action);
            });
        });

        // Service cards (click to toggle/view details)
        this.container.querySelectorAll('.tb-brain-card').forEach(card => {
            card.addEventListener('click', () => {
                const service = card.dataset.service;
                this._showServiceDetails(service);
            });
        });
    }

    async _fetchInitialStatus() {
        try {
            const response = await fetch(`${this.apiBase}/brain/status`);
            if (response.ok) {
                const data = await response.json();
                this._updateStatus(data);
            }
        } catch (error) {
            console.warn('Failed to fetch brain status:', error);
            this._setAllOffline();
        }
    }

    _connectWebSocket() {
        if (!this.wsUrl) return;

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                console.log('[BrainManager] WebSocket connected');
                this.ws.send(JSON.stringify({ type: 'subscribe', channel: 'brain_status' }));
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'status_update') {
                        this._updateStatus(data.status);
                    } else if (data.type === 'activity') {
                        this._addActivity(data.activity);
                    }
                } catch (e) {
                    console.warn('[BrainManager] Invalid WS message:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('[BrainManager] WebSocket closed, reconnecting...');
                setTimeout(() => this._connectWebSocket(), 5000);
            };

            this.ws.onerror = (error) => {
                console.error('[BrainManager] WebSocket error:', error);
            };
        } catch (error) {
            console.error('[BrainManager] Failed to connect WebSocket:', error);
        }
    }

    _updateStatus(data) {
        if (data.discord) this.status.discord = data.discord;
        if (data.telegram) this.status.telegram = data.telegram;
        if (data.vault) this.status.vault = data.vault;
        if (data.sync) this.status.sync = data.sync;

        this._renderStatus();
        this.onStatusChange(this.status);
    }

    _renderStatus() {
        const services = ['discord', 'telegram', 'vault', 'sync'];

        services.forEach(service => {
            const card = this.container.querySelector(`[data-service="${service}"]`);
            if (!card) return;

            const status = this.status[service];
            const dot = card.querySelector('.tb-brain-card-dot');
            const statusText = card.querySelector('.tb-brain-card-status');

            if (status.connected || status.active) {
                dot.className = 'tb-brain-card-dot online';
                if (service === 'discord' || service === 'telegram') {
                    statusText.textContent = `${status.users || 0} users`;
                } else if (service === 'vault') {
                    statusText.textContent = `${status.notes || 0} notes`;
                } else if (service === 'sync') {
                    statusText.textContent = status.lastSync ? `Last: ${this._formatTime(status.lastSync)}` : 'Active';
                }
            } else {
                dot.className = 'tb-brain-card-dot offline';
                statusText.textContent = 'Offline';
            }
        });
    }

    _setAllOffline() {
        ['discord', 'telegram', 'vault', 'sync'].forEach(service => {
            this.status[service] = { connected: false, active: false };
        });
        this._renderStatus();
    }

    _addActivity(activity) {
        this.activities.unshift(activity);
        if (this.activities.length > 10) this.activities.pop();
        this._renderActivities();
    }

    _renderActivities() {
        const list = this.container.querySelector('.tb-brain-activity-list');
        if (!list) return;

        if (this.activities.length === 0) {
            list.innerHTML = '<div class="tb-brain-activity-empty">No recent activity</div>';
            return;
        }

        list.innerHTML = this.activities.map(a => `
            <div class="tb-brain-activity-item">
                <span class="tb-brain-activity-icon">${a.icon || '📌'}</span>
                <span class="tb-brain-activity-text">${a.text}</span>
                <span class="tb-brain-activity-time">${this._formatTime(a.time)}</span>
            </div>
        `).join('');
    }

    _formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleDateString();
    }

    async _handleAction(action) {
        switch (action) {
            case 'capture':
                // Trigger quick capture popup
                window.dispatchEvent(new CustomEvent('tb:quickCapture'));
                break;
            case 'note':
                window.location.hash = '/vault/new';
                break;
            case 'daily':
                window.location.hash = '/vault/daily';
                break;
            case 'search':
                window.location.hash = '/vault/search';
                break;
        }
    }

    _showServiceDetails(service) {
        console.log(`Show details for ${service}`);
        // Could open a modal with more details
    }

    _showSettings() {
        console.log('Show brain settings');
        // Could open settings modal
    }

    destroy() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

export default BrainManager;

