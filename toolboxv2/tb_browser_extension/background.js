// ToolBox Browser Extension - Clean Background Service Worker
console.log('🚀 ToolBox Extension Loading...');

class ToolBoxAPI {
    constructor() {
        this.baseURL = null;
        this.connected = false;
        this.init();
    }

    async init() {
        await this.connect();
        this.setupMessageHandlers();
        console.log('✅ ToolBox Extension Ready');
    }

    async connect() {
        const servers = [
            'http://localhost:8080',
            'https://simplecore.app'
        ];

        for (const server of servers) {
            try {
                const response = await fetch(`${server}/api/health`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (response.ok) {
                    this.baseURL = server;
                    this.connected = true;
                    console.log(`✅ Connected to ToolBox: ${server}`);
                    return;
                }
            } catch (error) {
                console.log(`❌ Failed to connect to ${server}`);
            }
        }
        
        console.warn('⚠️ No ToolBox server available');
    }

    async request(module, functionName, args = {}) {
        if (!this.connected) {
            throw new Error('ToolBox server not connected');
        }

        try {
            const response = await fetch(`${this.baseURL}/api/${module}/${functionName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(args)
            });

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    setupMessageHandlers() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async response
        });
    }

    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'API_REQUEST':
                    const result = await this.request(message.module, message.function, message.args);
                    sendResponse({ success: true, data: result });
                    break;

                case 'VOICE_SEARCH':
                    const searchResult = await this.handleVoiceSearch(message.query);
                    sendResponse({ success: true, data: searchResult });
                    break;

                case 'PASSWORD_GENERATE':
                    const password = await this.generatePassword(message.options);
                    sendResponse({ success: true, data: password });
                    break;

                case 'PASSWORD_AUTOFILL':
                    const autofill = await this.getAutofillData(message.url);
                    sendResponse({ success: true, data: autofill });
                    break;

                case 'ISAA_CHAT':
                    const chatResponse = await this.handleISAAChat(message.query, message.context);
                    sendResponse({ success: true, data: chatResponse });
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Message handler error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    async handleVoiceSearch(query) {
        return await this.request('isaa', 'mini_task_completion', {
            mini_task: query,
            user_task: 'Voice search query',
            mode: 'chat',
            agent_name: 'web-assistant'
        });
    }

    async generatePassword(options = {}) {
        return await this.request('PasswordManager', 'generate_password', {
            length: options.length || 16,
            include_symbols: options.symbols !== false,
            include_numbers: options.numbers !== false,
            include_uppercase: options.uppercase !== false,
            include_lowercase: options.lowercase !== false,
            exclude_ambiguous: options.excludeAmbiguous !== false
        });
    }

    async getAutofillData(url) {
        return await this.request('PasswordManager', 'get_password_for_autofill', { url });
    }

    async handleISAAChat(query, context = {}) {
        return await this.request('isaa', 'mini_task_completion', {
            mini_task: query,
            user_task: 'Web interaction assistance',
            mode: 'chat',
            agent_name: 'web-assistant',
            context: context
        });
    }
}

// Initialize ToolBox API
const toolboxAPI = new ToolBoxAPI();

// Handle keyboard shortcuts
chrome.commands.onCommand.addListener(async (command) => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    switch (command) {
        case 'toggle-toolbox':
            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: () => {
                    // Toggle ToolBox panel (will be implemented in content script)
                    window.postMessage({ type: 'TOOLBOX_TOGGLE' }, '*');
                }
            });
            break;

        case 'voice-command':
            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: () => {
                    // Activate voice command (will be implemented in content script)
                    window.postMessage({ type: 'TOOLBOX_VOICE' }, '*');
                }
            });
            break;
    }
});

// Handle extension icon click
chrome.action.onClicked.addListener(async (tab) => {
    // Open popup (default behavior)
    console.log('Extension icon clicked');
});
