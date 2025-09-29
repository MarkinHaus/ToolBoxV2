// ToolBox Browser Extension - Clean Content Script
console.log('🔧 ToolBox Content Script Loading...');

class ToolBoxContent {
    constructor() {
        this.init();
    }

    init() {
        this.setupMessageListeners();
        this.setupFormDetection();
        console.log('✅ ToolBox Content Script Ready');
    }

    setupMessageListeners() {
        // Listen for messages from popup and background
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true;
        });

        // Listen for keyboard shortcuts
        window.addEventListener('message', (event) => {
            if (event.source !== window) return;
            
            switch (event.data.type) {
                case 'TOOLBOX_TOGGLE':
                    this.togglePanel();
                    break;
                case 'TOOLBOX_VOICE':
                    this.activateVoice();
                    break;
            }
        });
    }

    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'AUTOFILL_FORM':
                    const filled = await this.autofillForm(message.data);
                    sendResponse({ success: true, filled });
                    break;

                case 'GET_PAGE_CONTEXT':
                    const context = this.getPageContext();
                    sendResponse({ success: true, context });
                    break;

                case 'FILL_FIELD':
                    this.fillField(message.selector, message.value);
                    sendResponse({ success: true });
                    break;

                case 'CLICK_ELEMENT':
                    this.clickElement(message.selector);
                    sendResponse({ success: true });
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Content script error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    setupFormDetection() {
        // Detect login forms and password fields
        this.detectForms();
        
        // Watch for dynamic form changes
        const observer = new MutationObserver(() => {
            this.detectForms();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    detectForms() {
        const forms = document.querySelectorAll('form');
        const passwordFields = document.querySelectorAll('input[type="password"]');
        const emailFields = document.querySelectorAll('input[type="email"], input[name*="email"], input[id*="email"]');
        
        if (forms.length > 0 || passwordFields.length > 0) {
            // Store form information for autofill
            this.formData = {
                forms: Array.from(forms).map(form => ({
                    action: form.action,
                    method: form.method,
                    fields: this.getFormFields(form)
                })),
                passwordFields: Array.from(passwordFields).map(field => ({
                    id: field.id,
                    name: field.name,
                    selector: this.getSelector(field)
                })),
                emailFields: Array.from(emailFields).map(field => ({
                    id: field.id,
                    name: field.name,
                    selector: this.getSelector(field)
                }))
            };
        }
    }

    getFormFields(form) {
        const inputs = form.querySelectorAll('input, select, textarea');
        return Array.from(inputs).map(input => ({
            type: input.type,
            name: input.name,
            id: input.id,
            placeholder: input.placeholder,
            selector: this.getSelector(input)
        }));
    }

    getSelector(element) {
        if (element.id) return `#${element.id}`;
        if (element.name) return `[name="${element.name}"]`;
        if (element.className) return `.${element.className.split(' ')[0]}`;
        return element.tagName.toLowerCase();
    }

    async autofillForm(data) {
        if (!data || !data.entry) return false;

        const { username, password } = data.entry;
        let filled = false;

        // Fill username/email field
        if (username) {
            const usernameField = this.findUsernameField();
            if (usernameField) {
                this.fillField(usernameField, username);
                filled = true;
            }
        }

        // Fill password field
        if (password) {
            const passwordField = document.querySelector('input[type="password"]');
            if (passwordField) {
                this.fillField(passwordField, password);
                filled = true;
            }
        }

        return filled;
    }

    findUsernameField() {
        // Try different selectors for username fields
        const selectors = [
            'input[type="email"]',
            'input[name*="email"]',
            'input[id*="email"]',
            'input[name*="username"]',
            'input[id*="username"]',
            'input[name*="user"]',
            'input[id*="user"]'
        ];

        for (const selector of selectors) {
            const field = document.querySelector(selector);
            if (field) return field;
        }

        return null;
    }

    fillField(element, value) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (element) {
            element.value = value;
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }

    clickElement(selector) {
        const element = document.querySelector(selector);
        if (element) {
            element.click();
        }
    }

    getPageContext() {
        return {
            url: window.location.href,
            title: document.title,
            domain: window.location.hostname,
            hasPasswordFields: document.querySelectorAll('input[type="password"]').length > 0,
            hasEmailFields: document.querySelectorAll('input[type="email"]').length > 0,
            formCount: document.querySelectorAll('form').length,
            selectedText: window.getSelection().toString().trim()
        };
    }

    togglePanel() {
        // This would toggle a floating panel (to be implemented)
        console.log('Toggle ToolBox panel');
    }

    activateVoice() {
        // This would activate voice recognition (to be implemented)
        console.log('Activate voice command');
    }
}

// Initialize content script
const toolboxContent = new ToolBoxContent();
