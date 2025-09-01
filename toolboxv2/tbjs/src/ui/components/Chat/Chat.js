// tbjs/ui/components/Chat/Chat.js

import TB from '../../../index.js';

const ChatWidget = {
    element: null,
    elements: {},
    roomName: 'public_room',
    moduleName: 'ChatModule',
    username: 'Me',

    init(containerElement) {
        if (!containerElement) {
            TB.logger.error('[ChatWidget] Initialization failed: No container element provided.');
            return;
        }
        this.element = containerElement;
        this.username = TB.state.get('user.username') || 'Guest';

        this.render();
        this.setupEventListeners();
    },

    render() {
        const html = `
            <div class="tb-chat-widget">
                <div class="tb-chat-header">
                    <h3>Chat Room: ${this.roomName}</h3>
                    <div class="tb-chat-status">
                        <div class="tb-chat-status-dot" title="Disconnected"></div>
                        <span class="tb-chat-status-text">Disconnected</span>
                    </div>
                </div>
                <div class="tb-chat-messages">
                    <!-- Nachrichten werden hier angehängt -->
                </div>
                <div class="tb-chat-input-area">
                    <input type="text" class="tb-chat-input" placeholder="Type a message...">
                    <button class="tb-chat-send-btn button">Send</button>
                </div>
            </div>
        `;
        this.element.innerHTML = html;

        // DOM-Elemente cachen
        this.elements.statusDot = this.element.querySelector('.tb-chat-status-dot');
        this.elements.statusText = this.element.querySelector('.tb-chat-status-text');
        this.elements.messagesContainer = this.element.querySelector('.tb-chat-messages');
        this.elements.input = this.element.querySelector('.tb-chat-input');
        this.elements.sendBtn = this.element.querySelector('.tb-chat-send-btn');
    },

    setupEventListeners() {
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        this.elements.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auf globale tbjs-Events für WebSocket-Statusänderungen lauschen
        TB.events.on('ws:open', () => this.updateStatus('Connected', 'connected'));
        TB.events.on('ws:close', () => this.updateStatus('Disconnected', 'disconnected'));
        TB.events.on('ws:error', () => this.updateStatus('Connection Error', 'error'));

        // Auf benutzerdefinierte Chat-Events lauschen
        TB.events.on('ws:event:welcome', (e) => this.addMessage('System', e.data, 'system'));
        TB.events.on('ws:event:user_joined', (e) => this.addMessage('System', e.data, 'system'));
        TB.events.on('ws:event:new_message', (e) => this.addMessage(e.data.user, e.data.text, e.data.user === this.username ? 'user' : 'other'));
        TB.events.on('ws:event:user_left', (e) => this.addMessage('System', e.data, 'system'));
    },

    connect() {
        this.updateStatus('Connecting...', 'connecting');
        const wsPath = `/ws/${this.moduleName}/${this.roomName}`;
        TB.ws.connect(wsPath);
    },

    sendMessage() {
        const messageText = this.elements.input.value.trim();
        if (!messageText) return;

        const payload = {
            // Der Python-Handler erwartet ein `payload`-Argument, das das geparste JSON ist.
            data: { message: messageText }
        };

        if (TB.ws.send(payload)) {
            this.elements.input.value = '';
            this.elements.input.focus();
        } else {
            TB.ui.Toast.showError("Failed to send message. Are you connected?");
        }
    },

    addMessage(user, text, type) {
        const msgEl = document.createElement('div');
        msgEl.classList.add('tb-chat-message');

        let content = '';
        if (type === 'user' || type === 'other') {
            msgEl.classList.add(type === 'user' ? 'user-message' : 'other-message');
            content = `<span class="msg-user">${user}</span><div class="msg-text">${text}</div>`;
        } else { // system
            msgEl.classList.add('system-message');
            content = `<span class="msg-text">${text}</span>`;
        }

        msgEl.innerHTML = content;
        this.elements.messagesContainer.appendChild(msgEl);

        // Nach unten scrollen
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    },

    updateStatus(text, statusClass) {
        this.elements.statusText.textContent = text;
        this.elements.statusDot.className = 'tb-chat-status-dot'; // Klassen zurücksetzen
        if (statusClass) {
            this.elements.statusDot.classList.add(statusClass);
            this.elements.statusDot.title = text;
        }
    },
};

export default ChatWidget;
