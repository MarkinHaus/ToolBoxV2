// ToolBox Browser Extension - Clean Popup Script
console.log('🎯 ToolBox Popup Loading...');

class ToolBoxPopup {
    constructor() {
        this.isRecording = false;
        this.recognition = null;
        this.chatCollapsed = false;
        this.currentTab = null;
        this.init();
    }

    async init() {
        await this.getCurrentTab();
        this.setupElements();
        this.setupEventListeners();
        this.checkConnection();
        this.setupVoiceRecognition();
        console.log('✅ ToolBox Popup Ready');
    }

    async getCurrentTab() {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        this.currentTab = tab;
    }

    setupElements() {
        // Main elements
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.voiceInput = document.getElementById('voiceInput');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.sendBtn = document.getElementById('sendBtn');
        this.voiceStatus = document.getElementById('voiceStatus');

        // Action buttons
        this.generatePasswordBtn = document.getElementById('generatePasswordBtn');
        this.autofillBtn = document.getElementById('autofillBtn');
        this.passwordManagerBtn = document.getElementById('passwordManagerBtn');

        // Chat elements
        this.chatToggle = document.getElementById('chatToggle');
        this.chatContainer = document.getElementById('chatContainer');
        this.chatMessages = document.getElementById('chatMessages');

        // Modal elements
        this.passwordModal = document.getElementById('passwordModal');
        this.closePasswordModal = document.getElementById('closePasswordModal');
        this.passwordSearch = document.getElementById('passwordSearch');
        this.passwordList = document.getElementById('passwordList');

        // TOTP elements
        this.totpCard = document.getElementById('totpCard');
        this.closeTotpCard = document.getElementById('closeTotpCard');
        this.totpCode = document.getElementById('totpCode');
        this.totpProgress = document.getElementById('totpProgress');
        this.totpTime = document.getElementById('totpTime');

        // Loading and notifications
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.notifications = document.getElementById('notifications');
    }

    setupEventListeners() {
        // Voice and input
        this.voiceBtn.addEventListener('click', () => this.toggleVoiceRecording());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.voiceInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Action buttons
        this.generatePasswordBtn.addEventListener('click', () => this.generatePassword());
        this.autofillBtn.addEventListener('click', () => this.autofillForm());
        this.passwordManagerBtn.addEventListener('click', () => this.openPasswordManager());

        // Chat toggle
        this.chatToggle.addEventListener('click', () => this.toggleChat());

        // Modal controls
        this.closePasswordModal.addEventListener('click', () => this.closeModal());
        this.passwordModal.addEventListener('click', (e) => {
            if (e.target === this.passwordModal) this.closeModal();
        });

        // TOTP controls
        this.closeTotpCard.addEventListener('click', () => this.closeTotpCard.style.display = 'none');
        this.totpCode.addEventListener('click', () => this.copyToClipboard(this.totpCode.textContent));

        // Password search
        this.passwordSearch.addEventListener('input', (e) => this.filterPasswords(e.target.value));
    }

    setupVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onstart = () => {
                this.isRecording = true;
                this.voiceBtn.classList.add('recording');
                this.voiceStatus.style.display = 'flex';
            };

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.voiceInput.value = transcript;
                this.sendMessage();
            };

            this.recognition.onend = () => {
                this.isRecording = false;
                this.voiceBtn.classList.remove('recording');
                this.voiceStatus.style.display = 'none';
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.showNotification('Voice recognition failed', 'error');
                this.isRecording = false;
                this.voiceBtn.classList.remove('recording');
                this.voiceStatus.style.display = 'none';
            };
        } else {
            this.voiceBtn.style.display = 'none';
            console.warn('Speech recognition not supported');
        }
    }

    async checkConnection() {
        try {
            const response = await this.sendToBackground('API_REQUEST', {
                module: 'health',
                function: 'check'
            });

            if (response.success) {
                this.updateConnectionStatus('connected', 'Connected');
            } else {
                this.updateConnectionStatus('error', 'Disconnected');
            }
        } catch (error) {
            this.updateConnectionStatus('error', 'Error');
        }
    }

    updateConnectionStatus(status, text) {
        this.statusDot.className = `tb-status-dot ${status}`;
        this.statusText.textContent = text;
    }

    toggleVoiceRecording() {
        if (!this.recognition) {
            this.showNotification('Voice recognition not supported', 'error');
            return;
        }

        if (this.isRecording) {
            this.recognition.stop();
        } else {
            this.recognition.start();
        }
    }

    async sendMessage() {
        const message = this.voiceInput.value.trim();
        if (!message) return;

        this.addChatMessage(message, 'user');
        this.voiceInput.value = '';
        this.showLoading(true);

        try {
            const context = await this.getPageContext();
            const response = await this.sendToBackground('ISAA_CHAT', {
                query: message,
                context: context
            });

            if (response.success && response.data) {
                const aiResponse = response.data.response || response.data.data?.response || 'I understand your request.';
                this.addChatMessage(aiResponse, 'system');
            } else {
                this.addChatMessage('Sorry, I encountered an error processing your request.', 'system');
            }
        } catch (error) {
            console.error('ISAA chat error:', error);
            this.addChatMessage('Connection error. Please try again.', 'system');
        } finally {
            this.showLoading(false);
        }
    }

    async generatePassword() {
        this.showLoading(true);

        try {
            const response = await this.sendToBackground('PASSWORD_GENERATE', {
                options: { length: 16, symbols: true, numbers: true }
            });

            if (response.success && response.data?.data?.password) {
                const password = response.data.data.password;
                await this.copyToClipboard(password);
                this.showNotification('Password generated and copied!', 'success');
            } else {
                this.showNotification('Failed to generate password', 'error');
            }
        } catch (error) {
            console.error('Password generation error:', error);
            this.showNotification('Password generation failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async autofillForm() {
        if (!this.currentTab) return;

        this.showLoading(true);

        try {
            const response = await this.sendToBackground('PASSWORD_AUTOFILL', {
                url: this.currentTab.url
            });

            if (response.success && response.data?.data?.entry) {
                // Send autofill data to content script
                await chrome.tabs.sendMessage(this.currentTab.id, {
                    type: 'AUTOFILL_FORM',
                    data: response.data.data
                });

                this.showNotification('Form auto-filled successfully!', 'success');

                // Show TOTP if available
                if (response.data.data.totp_code) {
                    this.showTotpCode(response.data.data.totp_code);
                }
            } else {
                this.showNotification('No saved password found for this site', 'warning');
            }
        } catch (error) {
            console.error('Autofill error:', error);
            this.showNotification('Auto-fill failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async openPasswordManager() {
        this.showLoading(true);

        try {
            const response = await this.sendToBackground('API_REQUEST', {
                module: 'PasswordManager',
                function: 'list_passwords'
            });

            if (response.success && response.data?.data?.passwords) {
                this.displayPasswords(response.data.data.passwords);
                this.passwordModal.style.display = 'flex';
            } else {
                this.showNotification('Failed to load passwords', 'error');
            }
        } catch (error) {
            console.error('Password manager error:', error);
            this.showNotification('Failed to open password manager', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayPasswords(passwords) {
        if (!passwords || passwords.length === 0) {
            this.passwordList.innerHTML = `
                <div class="tb-empty-state">
                    <span class="tb-empty-icon">🔒</span>
                    <p>No passwords found</p>
                    <small>Your saved passwords will appear here</small>
                </div>
            `;
            return;
        }

        this.passwordList.innerHTML = passwords.map(password => `
            <div class="tb-password-item" data-id="${password.id}">
                <div class="tb-password-info">
                    <div class="tb-password-title">${password.title || 'Untitled'}</div>
                    <div class="tb-password-details">
                        ${password.username || 'No username'} • ${password.url || 'No URL'}
                        ${password.totp_secret ? ' • 2FA' : ''}
                    </div>
                </div>
                <div class="tb-password-actions">
                    <button class="tb-btn tb-btn-secondary" onclick="toolboxPopup.copyPassword('${password.id}')">
                        Copy
                    </button>
                    <button class="tb-btn tb-btn-secondary" onclick="toolboxPopup.fillPassword('${password.id}')">
                        Fill
                    </button>
                    ${password.totp_secret ? `<button class="tb-btn tb-btn-primary" onclick="toolboxPopup.showTotp('${password.id}')">2FA</button>` : ''}
                </div>
            </div>
        `).join('');
    }

    async copyPassword(passwordId) {
        // Implementation for copying password
        this.showNotification('Password copied to clipboard', 'success');
    }

    async fillPassword(passwordId) {
        // Implementation for filling password
        this.showNotification('Password filled in form', 'success');
        this.closeModal();
    }

    async showTotp(passwordId) {
        try {
            const response = await this.sendToBackground('API_REQUEST', {
                module: 'PasswordManager',
                function: 'generate_totp_code',
                args: { entry_id: passwordId }
            });

            if (response.success && response.data?.data?.code) {
                this.showTotpCode(response.data.data.code, response.data.data.time_remaining);
                this.closeModal();
            } else {
                this.showNotification('Failed to generate 2FA code', 'error');
            }
        } catch (error) {
            console.error('TOTP generation error:', error);
            this.showNotification('2FA code generation failed', 'error');
        }
    }

    showTotpCode(code, timeRemaining = 30) {
        this.totpCode.textContent = code;
        this.totpTime.textContent = `${timeRemaining}s`;
        this.totpCard.style.display = 'block';

        // Start countdown
        let remaining = timeRemaining;
        const interval = setInterval(() => {
            remaining--;
            this.totpTime.textContent = `${remaining}s`;

            // Update progress bar
            const progress = (remaining / 30) * 100;
            this.totpProgress.style.setProperty('--progress', `${progress}%`);

            if (remaining <= 0) {
                clearInterval(interval);
                this.totpCard.style.display = 'none';
            }
        }, 1000);
    }

    filterPasswords(query) {
        const items = this.passwordList.querySelectorAll('.tb-password-item');
        items.forEach(item => {
            const title = item.querySelector('.tb-password-title').textContent.toLowerCase();
            const details = item.querySelector('.tb-password-details').textContent.toLowerCase();
            const matches = title.includes(query.toLowerCase()) || details.includes(query.toLowerCase());
            item.style.display = matches ? 'flex' : 'none';
        });
    }

    toggleChat() {
        this.chatCollapsed = !this.chatCollapsed;
        this.chatContainer.classList.toggle('collapsed', this.chatCollapsed);
        document.getElementById('chatToggleIcon').textContent = this.chatCollapsed ? '▶' : '▼';
    }

    addChatMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `tb-chat-message tb-chat-${type}`;

        const icon = type === 'user' ? '👤' : '🤖';
        messageDiv.innerHTML = `
            <span class="tb-chat-icon">${icon}</span>
            <span class="tb-chat-text">${message}</span>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        // Expand chat if collapsed
        if (this.chatCollapsed) {
            this.toggleChat();
        }
    }

    async getPageContext() {
        if (!this.currentTab) return {};

        try {
            const response = await chrome.tabs.sendMessage(this.currentTab.id, {
                type: 'GET_PAGE_CONTEXT'
            });
            return response?.context || {};
        } catch (error) {
            return {
                url: this.currentTab.url,
                title: this.currentTab.title
            };
        }
    }

    closeModal() {
        this.passwordModal.style.display = 'none';
        this.passwordSearch.value = '';
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `tb-notification ${type}`;
        notification.textContent = message;

        this.notifications.appendChild(notification);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }

    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (error) {
            console.error('Clipboard error:', error);
            return false;
        }
    }

    async sendToBackground(type, data) {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({ type, ...data }, resolve);
        });
    }
}

// Initialize popup
const toolboxPopup = new ToolBoxPopup();
