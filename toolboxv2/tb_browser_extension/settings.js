// ToolBox Extension Settings - Professional Implementation
class TBSettings {
    constructor() {
        this.settings = {
            // Password Manager
            passwordManagerEnabled: true,
            autofillEnabled: false,
            serverUrl: 'http://localhost:8080',
            
            // Voice Commands
            voiceCommandsEnabled: true,
            voiceLanguage: 'en-US',
            wakeWord: 'ToolBox',
            continuousListening: false,
            
            // UI Settings
            gestureFeedbackEnabled: true,
            panelPosition: 'top-right',
            
            // Advanced
            debugMode: false,
            apiTimeout: 10
        };
        
        this.voiceRecognition = null;
        this.currentVoiceInput = null;
        
        this.init();
    }

    async init() {
        try {
            await this.loadSettings();
            this.setupEventListeners();
            this.updateUI();
            this.initVoiceInput();
            console.log('🚀 ToolBox Settings initialized');
        } catch (error) {
            console.error('❌ Settings initialization failed:', error);
        }
    }

    async loadSettings() {
        try {
            const stored = await chrome.storage.sync.get(Object.keys(this.settings));
            this.settings = { ...this.settings, ...stored };
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }

    async saveSettings() {
        try {
            await chrome.storage.sync.set(this.settings);
            this.showStatus('Settings saved successfully!', 'success');
            
            // Notify content scripts of settings change
            const tabs = await chrome.tabs.query({});
            tabs.forEach(tab => {
                chrome.tabs.sendMessage(tab.id, {
                    type: 'TB_SETTINGS_UPDATED',
                    settings: this.settings
                }).catch(() => {}); // Ignore errors for tabs without content script
            });
            
        } catch (error) {
            console.error('Failed to save settings:', error);
            this.showStatus('Failed to save settings!', 'error');
        }
    }

    setupEventListeners() {
        // Toggle switches
        this.setupToggle('password-manager-toggle', 'passwordManagerEnabled');
        this.setupToggle('autofill-toggle', 'autofillEnabled');
        this.setupToggle('voice-commands-toggle', 'voiceCommandsEnabled');
        this.setupToggle('continuous-listening-toggle', 'continuousListening');
        this.setupToggle('gesture-feedback-toggle', 'gestureFeedbackEnabled');
        this.setupToggle('debug-mode-toggle', 'debugMode');

        // Input fields
        this.setupInput('server-url', 'serverUrl');
        this.setupInput('wake-word', 'wakeWord');
        this.setupInput('api-timeout', 'apiTimeout', 'number');

        // Select fields
        this.setupSelect('voice-language', 'voiceLanguage');
        this.setupSelect('panel-position', 'panelPosition');

        // Voice input buttons
        this.setupVoiceInput('server-url-voice', 'server-url');
        this.setupVoiceInput('wake-word-voice', 'wake-word');

        // Action buttons
        document.getElementById('save-btn').addEventListener('click', () => this.saveSettings());
        document.getElementById('reset-btn').addEventListener('click', () => this.resetSettings());
    }

    setupToggle(elementId, settingKey) {
        const toggle = document.getElementById(elementId);
        if (!toggle) return;

        toggle.addEventListener('click', () => {
            const isActive = toggle.classList.contains('active');
            toggle.classList.toggle('active', !isActive);
            this.settings[settingKey] = !isActive;
        });
    }

    setupInput(elementId, settingKey, type = 'text') {
        const input = document.getElementById(elementId);
        if (!input) return;

        input.addEventListener('input', (e) => {
            let value = e.target.value;
            if (type === 'number') {
                value = parseInt(value) || 0;
            }
            this.settings[settingKey] = value;
        });
    }

    setupSelect(elementId, settingKey) {
        const select = document.getElementById(elementId);
        if (!select) return;

        select.addEventListener('change', (e) => {
            this.settings[settingKey] = e.target.value;
        });
    }

    setupVoiceInput(buttonId, inputId) {
        const button = document.getElementById(buttonId);
        const input = document.getElementById(inputId);
        
        if (!button || !input) return;

        button.addEventListener('click', () => {
            this.startVoiceInput(input, button);
        });
    }

    initVoiceInput() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.voiceRecognition = new SpeechRecognition();
            this.voiceRecognition.continuous = false;
            this.voiceRecognition.interimResults = false;
            this.voiceRecognition.lang = this.settings.voiceLanguage;

            this.voiceRecognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                if (this.currentVoiceInput) {
                    this.currentVoiceInput.value = transcript;
                    this.currentVoiceInput.dispatchEvent(new Event('input'));
                }
                this.stopVoiceInput();
            };

            this.voiceRecognition.onerror = (event) => {
                console.error('Voice recognition error:', event.error);
                this.stopVoiceInput();
                this.showStatus('Voice input failed: ' + event.error, 'error');
            };

            this.voiceRecognition.onend = () => {
                this.stopVoiceInput();
            };
        }
    }

    startVoiceInput(input, button) {
        if (!this.voiceRecognition) {
            this.showStatus('Voice recognition not supported', 'error');
            return;
        }

        this.currentVoiceInput = input;
        button.textContent = '🔴';
        button.style.background = '#ef4444';
        
        try {
            this.voiceRecognition.lang = this.settings.voiceLanguage;
            this.voiceRecognition.start();
            this.showStatus('Listening... Speak now', 'success');
        } catch (error) {
            console.error('Failed to start voice recognition:', error);
            this.stopVoiceInput();
        }
    }

    stopVoiceInput() {
        if (this.voiceRecognition) {
            this.voiceRecognition.stop();
        }
        
        // Reset all voice buttons
        document.querySelectorAll('.voice-input-btn').forEach(btn => {
            btn.textContent = '🎤';
            btn.style.background = 'var(--tb-accent-primary)';
        });
        
        this.currentVoiceInput = null;
    }

    updateUI() {
        // Update toggles
        this.updateToggle('password-manager-toggle', this.settings.passwordManagerEnabled);
        this.updateToggle('autofill-toggle', this.settings.autofillEnabled);
        this.updateToggle('voice-commands-toggle', this.settings.voiceCommandsEnabled);
        this.updateToggle('continuous-listening-toggle', this.settings.continuousListening);
        this.updateToggle('gesture-feedback-toggle', this.settings.gestureFeedbackEnabled);
        this.updateToggle('debug-mode-toggle', this.settings.debugMode);

        // Update inputs
        this.updateInput('server-url', this.settings.serverUrl);
        this.updateInput('wake-word', this.settings.wakeWord);
        this.updateInput('api-timeout', this.settings.apiTimeout);

        // Update selects
        this.updateSelect('voice-language', this.settings.voiceLanguage);
        this.updateSelect('panel-position', this.settings.panelPosition);
    }

    updateToggle(elementId, active) {
        const toggle = document.getElementById(elementId);
        if (toggle) {
            toggle.classList.toggle('active', active);
        }
    }

    updateInput(elementId, value) {
        const input = document.getElementById(elementId);
        if (input) {
            input.value = value;
        }
    }

    updateSelect(elementId, value) {
        const select = document.getElementById(elementId);
        if (select) {
            select.value = value;
        }
    }

    async resetSettings() {
        if (confirm('Are you sure you want to reset all settings to defaults?')) {
            try {
                await chrome.storage.sync.clear();
                this.settings = {
                    passwordManagerEnabled: true,
                    autofillEnabled: false,
                    serverUrl: 'http://localhost:8080',
                    voiceCommandsEnabled: true,
                    voiceLanguage: 'en-US',
                    wakeWord: 'ToolBox',
                    continuousListening: false,
                    gestureFeedbackEnabled: true,
                    panelPosition: 'top-right',
                    debugMode: false,
                    apiTimeout: 10
                };
                this.updateUI();
                this.showStatus('Settings reset to defaults', 'success');
            } catch (error) {
                console.error('Failed to reset settings:', error);
                this.showStatus('Failed to reset settings!', 'error');
            }
        }
    }

    showStatus(message, type = 'success') {
        const container = document.getElementById('status-messages');
        if (!container) return;

        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message status-${type}`;
        statusDiv.textContent = message;
        
        container.appendChild(statusDiv);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (statusDiv.parentNode) {
                statusDiv.remove();
            }
        }, 3000);
    }

    // Test connection to ToolBox server
    async testConnection() {
        try {
            const response = await fetch(`${this.settings.serverUrl}/api/health`, {
                method: 'GET',
                timeout: this.settings.apiTimeout * 1000
            });
            
            if (response.ok) {
                this.showStatus('✅ Connected to ToolBox server', 'success');
            } else {
                this.showStatus('❌ Server responded with error', 'error');
            }
        } catch (error) {
            this.showStatus('❌ Cannot connect to ToolBox server', 'error');
        }
    }
}

// Initialize settings when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TBSettings();
});

// Add connection test button functionality
document.addEventListener('DOMContentLoaded', () => {
    // Add test connection button to server URL setting
    const serverUrlSetting = document.querySelector('#server-url').closest('.setting-item');
    if (serverUrlSetting) {
        const testBtn = document.createElement('button');
        testBtn.className = 'btn btn-secondary';
        testBtn.textContent = 'Test';
        testBtn.style.marginLeft = '8px';
        testBtn.onclick = () => window.tbSettings?.testConnection();
        
        const control = serverUrlSetting.querySelector('.setting-control');
        control.appendChild(testBtn);
    }
});
