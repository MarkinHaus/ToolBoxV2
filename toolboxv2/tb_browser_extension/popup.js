// ToolBox Extension Popup - Professional Minimalist Design
class TBPopup {
    constructor() {
        this.settings = new Map();
        this.stats = {
            passwords: 0,
            sites: 0,
            timeSaved: '0h'
        };
        this.init();
    }

    async init() {
        try {
            await this.loadSettings();
            await this.loadStats();
            this.setupEventListeners();
            this.checkConnection();
            this.updateUI();
            console.log('🚀 ToolBox Popup initialized');
        } catch (error) {
            console.error('❌ Popup initialization failed:', error);
        }
    }

    async loadSettings() {
        const stored = await chrome.storage.sync.get([
            'autofill_enabled',
            'voice_enabled',
            'notifications_enabled'
        ]);

        this.settings.set('autofill', stored.autofill_enabled ?? true);
        this.settings.set('voice', stored.voice_enabled ?? true);
        this.settings.set('notifications', stored.notifications_enabled ?? false);
    }

    async loadStats() {
        try {
            // Load password count
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'list_passwords',
                args: { limit: 1000 }
            });

            if (response && response.success) {
                this.stats.passwords = response.data?.passwords?.length || 0;
            }

            // Load other stats from storage
            const stored = await chrome.storage.local.get([
                'sites_enhanced',
                'time_saved_minutes'
            ]);

            this.stats.sites = stored.sites_enhanced || 0;
            const minutes = stored.time_saved_minutes || 0;
            this.stats.timeSaved = minutes > 60 ?
                `${Math.floor(minutes / 60)}h` :
                `${minutes}m`;

        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    setupEventListeners() {
        console.log('🔧 Setting up popup event listeners');

        // Password Manager Actions
        this.addClickListener('auto-fill-btn', () => this.autoFill());
        this.addClickListener('generate-pwd-btn', () => this.generatePassword());
        this.addClickListener('open-manager-btn', () => this.openPasswordManager());

        // Quick Actions
        this.addClickListener('toggle-panel', () => this.togglePanel());
        this.addClickListener('voice-command', () => this.activateVoice());
        this.addClickListener('ai-analyze', () => this.aiAnalyze());
        this.addClickListener('smart-search', () => this.smartSearch());

        // Settings Toggles
        this.addClickListener('autofill-toggle', () => this.toggleSetting('autofill'));
        this.addClickListener('voice-toggle', () => this.toggleSetting('voice'));
        this.addClickListener('notifications-toggle', () => this.toggleSetting('notifications'));

        // Footer Links
        this.addClickListener('open-settings', () => this.openSettings());
        this.addClickListener('open-help', () => this.openHelp());
        this.addClickListener('open-about', () => this.openAbout());

        console.log('✅ Event listeners setup complete');
    }

    addClickListener(elementId, handler) {
        const element = document.getElementById(elementId);
        if (element) {
            element.addEventListener('click', (e) => {
                e.preventDefault();
                console.log(`🖱️ Button clicked: ${elementId}`);
                try {
                    handler();
                } catch (error) {
                    console.error(`❌ Error handling click for ${elementId}:`, error);
                    this.showNotification(`Error: ${error.message}`);
                }
            });
            console.log(`✅ Added listener for: ${elementId}`);
        } else {
            console.warn(`⚠️ Element not found: ${elementId}`);
        }
    }

    async checkConnection() {
        try {
            const response = await fetch('http://localhost:8080/api/health');
            const statusEl = document.getElementById('connection-status');
            const statusText = document.getElementById('status-text');

            if (response.ok) {
                statusEl.className = 'status connected';
                statusText.textContent = 'Connected';
            } else {
                throw new Error('Server not responding');
            }
        } catch (error) {
            const statusEl = document.getElementById('connection-status');
            const statusText = document.getElementById('status-text');
            statusEl.className = 'status disconnected';
            statusText.textContent = 'Disconnected';
        }
    }

    updateUI() {
        // Update stats
        document.getElementById('passwords-count').textContent = this.stats.passwords;
        document.getElementById('sites-enhanced').textContent = this.stats.sites;
        document.getElementById('time-saved').textContent = this.stats.timeSaved;

        // Update toggle states
        this.updateToggle('autofill-toggle', this.settings.get('autofill'));
        this.updateToggle('voice-toggle', this.settings.get('voice'));
        this.updateToggle('notifications-toggle', this.settings.get('notifications'));
    }

    updateToggle(id, active) {
        const toggle = document.getElementById(id);
        if (toggle) {
            toggle.classList.toggle('active', active);
        }
    }

    // Password Manager Actions
    async autoFill() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, {
                type: 'TB_PASSWORD_AUTOFILL'
            });
            this.showNotification('🔐 Auto-fill activated');
        } catch (error) {
            this.showNotification('❌ Auto-fill failed');
        }
    }

    async generatePassword() {
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'TB_API_REQUEST',
                module: 'PasswordManager',
                function: 'generate_password',
                args: { length: 16, include_symbols: true }
            });

            if (response && response.success) {
                await navigator.clipboard.writeText(response.data.password);
                this.showNotification('🔑 Password generated and copied');
            }
        } catch (error) {
            this.showNotification('❌ Password generation failed');
        }
    }

    async openPasswordManager() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, {
                type: 'TB_PASSWORD_MANAGER',
                action: 'show_password_list'
            });
            window.close();
        } catch (error) {
            this.showNotification('❌ Failed to open password manager');
        }
    }

    // Quick Actions
    async togglePanel() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, { type: 'TB_TOGGLE_PANEL' });
            window.close();
        } catch (error) {
            this.showNotification('❌ Failed to toggle panel');
        }
    }

    async activateVoice() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, { type: 'TB_VOICE_COMMAND' });
            this.showNotification('🎤 Voice command activated');
        } catch (error) {
            this.showNotification('❌ Voice command failed');
        }
    }

    async aiAnalyze() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, { type: 'TB_AI_ANALYZE' });
            this.showNotification('🤖 AI analysis started');
        } catch (error) {
            this.showNotification('❌ AI analysis failed');
        }
    }

    async smartSearch() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.tabs.sendMessage(tab.id, { type: 'TB_SMART_SEARCH' });
            this.showNotification('🔍 Smart search activated');
        } catch (error) {
            this.showNotification('❌ Smart search failed');
        }
    }

    // Settings
    async toggleSetting(setting) {
        const currentValue = this.settings.get(setting);
        const newValue = !currentValue;
        this.settings.set(setting, newValue);

        // Save to storage
        const storageKey = `${setting}_enabled`;
        await chrome.storage.sync.set({ [storageKey]: newValue });

        // Update UI
        this.updateToggle(`${setting}-toggle`, newValue);

        // Show feedback
        const status = newValue ? 'enabled' : 'disabled';
        this.showNotification(`${setting} ${status}`);
    }

    // Navigation
    openSettings() {
        chrome.runtime.openOptionsPage();
        window.close();
    }

    openHelp() {
        chrome.tabs.create({ url: 'https://toolbox.simplecore.app/help' });
        window.close();
    }

    openAbout() {
        chrome.tabs.create({ url: 'https://toolbox.simplecore.app/about' });
        window.close();
    }

    // Utility
    showNotification(message) {
        // Create temporary notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--tb-accent-primary);
            color: var(--tb-bg-primary);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            z-index: 1000;
            animation: slideDown 0.3s ease-out;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 2000);
    }
}

// Initialize popup when DOM is ready
function initializePopup() {
    if (!window.tbPopup) {
        console.log('🚀 Initializing ToolBox Popup');
        window.tbPopup = new TBPopup();
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePopup);
} else {
    initializePopup();
}

// Add slide down animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateX(-50%) translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
    }
`;
document.head.appendChild(style);
