// ToolBox Pro - Content Script
// Handles page interaction, gesture detection, search, and autofill

class ToolBoxContent {
    constructor() {
        this.pageIndex = new Map();
        this.highlightedElements = [];
        this.isInitialized = false;
        this.lastFocusedInput = null;
        this.init();
    }

    async init() {
        if (this.isInitialized) return;

        console.log('🚀 ToolBox Pro content script initializing...');

        try {

            // Setup message listeners
            this.setupMessageListeners();

            // Initialize page indexing
            this.initializePageIndexing();

            this.monitorLoginFormSubmissions();

            // Setup form detection
            this.setupFormDetection();

            this.trackLastFocusedInput();

            this.isInitialized = true;
            console.log('✅ ToolBox Pro content script initialized');

        } catch (error) {
            console.error('❌ ToolBox Pro content script initialization failed:', error);
        }
    }

    trackLastFocusedInput() {
        document.addEventListener('focusin', (e) => {
            if (e.target.matches('input[type="text"], input[type="search"], textarea, input[type="email"], input[type="password"]')) {
                this.lastFocusedInput = e.target;
                console.log('🎯 ToolBox focused on input:', this.lastFocusedInput);
            }
        });

        document.addEventListener('mousedown', (e) => {
            // Check if click is NOT on the currently focused input
            if (this.lastFocusedInput && !this.lastFocusedInput.contains(e.target)) {
                console.log('🎯 Click outside input - resetting');
                this.lastFocusedInput = null;
            }
        });

    }

    monitorLoginFormSubmissions() {
        document.addEventListener('submit', (e) => {
            const form = e.target.closest('form[data-tb-form-type="login"]');
            if (!form) return;

            const usernameField = form.querySelector('input[type="email"], input[name*="email"], input[name*="username"], input[name*="login"], input[type="text"]');
            const passwordField = form.querySelector('input[type="password"]');

            if (usernameField && passwordField && passwordField.value) {
                const credentials = {
                    url: window.location.href,
                    username: usernameField.value,
                    password: passwordField.value
                };

                // Sende eine Nachricht an den background script, um die Daten zu speichern
                chrome.runtime.sendMessage({ type: 'POTENTIAL_CREDENTIALS_DETECTED', credentials });
            }
        }, true);
    }

    showGestureFeedback(message) {
        // Remove existing feedback
        const existing = document.querySelector('.tb-gesture-feedback');
        if (existing) {
            existing.remove();
        }

        // Create feedback element
        const feedback = document.createElement('div');
        feedback.className = 'tb-gesture-feedback';
        feedback.textContent = message;
        document.body.appendChild(feedback);

        // Remove after animation
        setTimeout(() => {
            if (feedback.parentNode) {
                feedback.remove();
            }
        }, 1500);
    }

    setupMessageListeners() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });
    }

    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'SEARCH_PAGE':
                    const results = await this.searchPage(message.query);
                    sendResponse({ success: true, results });
                    break;

                case 'INDEX_PAGE':
                    await this.indexPage();
                    sendResponse({ success: true });
                    break;

                case 'SCROLL_TO_ELEMENT':
                    this.scrollToElement(message.elementId);
                    sendResponse({ success: true });
                    break;

                case 'GET_ELEMENT_CONTENT':
                    const content = this.getElementContent(message.elementId);
                    sendResponse({ success: true, content });
                    break;

                case 'AUTOFILL_PASSWORD':
                    await this.autofillPassword(message.data);
                    sendResponse({ success: true });
                    break;

                case 'FILL_GENERATED_PASSWORD':
                    this.fillGeneratedPassword(message.password);
                    sendResponse({ success: true });
                    break;

                case 'GET_PAGE_CONTEXT':
                    const context = await this.getPageContext();
                    sendResponse({ success: true, ...context });
                    break;

                case 'EXECUTE_ACTION':
                    const actionResult = await this.executeAction(message.action);
                    sendResponse(actionResult);
                    break;

                case 'SHOW_GESTURE_FEEDBACK':
                    this.showGestureFeedback(message.message);
                    sendResponse({ success: true });
                    break;
                case 'INSERT_TEXT_INTO_FOCUSED_INPUT':
                    if (this.lastFocusedInput) {
                        this.fillField(this.lastFocusedInput, message.text);
                        sendResponse({ success: true, message: `Text inserted into ${this.lastFocusedInput.tagName}` });
                    } else {
                        sendResponse({ success: false, error: 'No active input field found on the page.' });
                    }
                    break;

                case 'PING':
                    sendResponse({ success: true, status: 'alive' });
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    // Page Indexing Methods
    initializePageIndexing() {
        // Auto-index page on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.indexPage());
        } else {
            this.indexPage();
        }
    }

    async indexPage() {
        console.log('📚 Indexing page content...');

        this.pageIndex.clear();

        // Index different types of content
        this.indexHeadings();
        this.indexParagraphs();
        this.indexLinks();
        this.indexImages();
        this.indexForms();
        this.indexLists();

        console.log(`✅ Page indexed: ${this.pageIndex.size} elements`);
    }

    indexHeadings() {
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        headings.forEach((heading, index) => {
            const id = `heading-${index}`;
            heading.setAttribute('data-tb-id', id);

            this.pageIndex.set(id, {
                type: 'heading',
                element: heading,
                title: heading.textContent.trim(),
                snippet: heading.textContent.trim(),
                level: parseInt(heading.tagName.charAt(1))
            });
        });
    }

    indexParagraphs() {
        const paragraphs = document.querySelectorAll('p');
        paragraphs.forEach((p, index) => {
            const text = p.textContent.trim();
            if (text.length > 20) { // Only index substantial paragraphs
                const id = `paragraph-${index}`;
                p.setAttribute('data-tb-id', id);

                this.pageIndex.set(id, {
                    type: 'paragraph',
                    element: p,
                    title: text.substring(0, 50) + '...',
                    snippet: text.substring(0, 150) + (text.length > 150 ? '...' : ''),
                    fullText: text
                });
            }
        });
    }

    indexLinks() {
        const links = document.querySelectorAll('a[href]');
        links.forEach((link, index) => {
            const text = link.textContent.trim();
            if (text) {
                const id = `link-${index}`;
                link.setAttribute('data-tb-id', id);

                this.pageIndex.set(id, {
                    type: 'link',
                    element: link,
                    title: text,
                    snippet: `Link: ${text} → ${link.href}`,
                    url: link.href
                });
            }
        });
    }

    indexImages() {
        const images = document.querySelectorAll('img[alt], img[title]');
        images.forEach((img, index) => {
            const alt = img.alt || img.title || '';
            if (alt) {
                const id = `image-${index}`;
                img.setAttribute('data-tb-id', id);

                this.pageIndex.set(id, {
                    type: 'image',
                    element: img,
                    title: alt,
                    snippet: `Image: ${alt}`,
                    src: img.src
                });
            }
        });
    }

    indexForms() {
        const forms = document.querySelectorAll('form');
        forms.forEach((form, index) => {
            const inputs = form.querySelectorAll('input, textarea, select');
            if (inputs.length > 0) {
                const id = `form-${index}`;
                form.setAttribute('data-tb-id', id);

                const inputTypes = Array.from(inputs).map(input =>
                    input.type || input.tagName.toLowerCase()
                ).join(', ');

                this.pageIndex.set(id, {
                    type: 'form',
                    element: form,
                    title: `Form with ${inputs.length} fields`,
                    snippet: `Form fields: ${inputTypes}`,
                    inputCount: inputs.length
                });
            }
        });
    }

    indexLists() {
        const lists = document.querySelectorAll('ul, ol');
        lists.forEach((list, index) => {
            const items = list.querySelectorAll('li');
            if (items.length > 0) {
                const id = `list-${index}`;
                list.setAttribute('data-tb-id', id);

                const firstItems = Array.from(items).slice(0, 3)
                    .map(item => item.textContent.trim()).join(', ');

                this.pageIndex.set(id, {
                    type: 'list',
                    element: list,
                    title: `List with ${items.length} items`,
                    snippet: `Items: ${firstItems}${items.length > 3 ? '...' : ''}`,
                    itemCount: items.length
                });
            }
        });
    }

    // Search Methods
    async searchPage(query) {
        const results = [];
        const searchTerms = query.toLowerCase().split(' ').filter(term => term.length > 2);

        if (searchTerms.length === 0) return results;

        // Clear previous highlights
        this.clearHighlights();

        // Search through indexed content
        for (const [id, item] of this.pageIndex) {
            const score = this.calculateSearchScore(item, searchTerms);
            if (score > 0) {
                results.push({
                    id,
                    title: item.title,
                    snippet: this.highlightSearchTerms(item.snippet, searchTerms),
                    type: item.type,
                    score
                });

                // Highlight element on page
                this.highlightElement(item.element, searchTerms);
            }
        }

        // Sort by relevance score
        results.sort((a, b) => b.score - a.score);

        return results.slice(0, 20); // Return top 20 results
    }

    calculateSearchScore(item, searchTerms) {
        let score = 0;
        const text = (item.title + ' ' + item.snippet + ' ' + (item.fullText || '')).toLowerCase();

        searchTerms.forEach(term => {
            const termCount = (text.match(new RegExp(term, 'g')) || []).length;
            score += termCount;

            // Boost score for title matches
            if (item.title.toLowerCase().includes(term)) {
                score += 2;
            }

            // Boost score for exact matches
            if (text.includes(term)) {
                score += 1;
            }
        });

        // Type-based scoring
        switch (item.type) {
            case 'heading':
                score *= 1.5;
                break;
            case 'link':
                score *= 1.2;
                break;
            case 'form':
                score *= 1.3;
                break;
        }

        return score;
    }

    highlightSearchTerms(text, searchTerms) {
        let highlightedText = text;
        searchTerms.forEach(term => {
            const regex = new RegExp(`(${term})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
        });
        return highlightedText;
    }

    highlightElement(element, searchTerms) {
        if (!element) return;

        // Add highlight class
        element.classList.add('tb-search-highlight');
        this.highlightedElements.push(element);

        // Create highlight overlay
        const rect = element.getBoundingClientRect();
        const overlay = document.createElement('div');
        overlay.className = 'tb-highlight-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: ${rect.top}px;
            left: ${rect.left}px;
            width: ${rect.width}px;
            height: ${rect.height}px;
            background: rgba(46, 134, 171, 0.2);
            border: 2px solid #2E86AB;
            border-radius: 4px;
            pointer-events: none;
            z-index: 10000;
            animation: tbHighlightPulse 2s ease-in-out;
        `;

        document.body.appendChild(overlay);
        this.highlightedElements.push(overlay);

        // Remove overlay after animation
        setTimeout(() => {
            if (overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        }, 2000);
    }

    clearHighlights() {
        this.highlightedElements.forEach(element => {
            if (element.classList) {
                element.classList.remove('tb-search-highlight');
            } else if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
        });
        this.highlightedElements = [];
    }

    scrollToElement(elementId) {
        const item = this.pageIndex.get(elementId);
        if (item && item.element) {
            item.element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });

            // Flash highlight
            this.highlightElement(item.element, []);
        } else {
            console.warn('Element not found for scrolling:', elementId);
        }
    }

    getElementContent(elementId) {
        const item = this.pageIndex.get(elementId);
        if (item) {
            return item.fullText || item.snippet || item.title;
        }
        return '';
    }

    async getPageContext() {
        // Generate page summary
        const summary = this.generatePageSummary();

        // Convert pageIndex to serializable format
        const pageIndexData = {};
        for (const [id, item] of this.pageIndex) {
            pageIndexData[id] = {
                type: item.type,
                title: item.title,
                snippet: item.snippet,
                // Don't include the actual DOM element
            };
        }

        return {
            pageIndex: pageIndexData,
            summary: summary,
            url: window.location.href,
            title: document.title
        };
    }

    generatePageSummary() {
        const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
            .slice(0, 5)
            .map(h => h.textContent.trim())
            .filter(text => text.length > 0);

        const mainContent = Array.from(document.querySelectorAll('p'))
            .slice(0, 3)
            .map(p => p.textContent.trim())
            .filter(text => text.length > 50)
            .map(text => text.substring(0, 100) + '...');

        return {
            headings: headings,
            mainContent: mainContent,
            elementCount: this.pageIndex.size,
            hasForm: document.querySelector('form') !== null,
            hasImages: document.querySelector('img') !== null
        };
    }

    async executeAction(action) {
        try {
            console.log('🎯 Executing action:', action);

            switch (action.action_type) {
                case 'click':
                    return await this.executeClickAction(action);
                case 'fill_form':
                    return await this.executeFillFormAction(action);
                case 'navigate':
                    return await this.executeNavigateAction(action);
                case 'scroll':
                    return await this.executeScrollAction(action);
                case 'extract_data':
                    return await this.executeExtractDataAction(action);
                default:
                    return { success: false, error: `Unknown action type: ${action.action_type}` };
            }
        } catch (error) {
            console.error('Action execution error:', error);
            return { success: false, error: error.message };
        }
    }

    async executeClickAction(action) {
        const element = this.findElementBySelector(action.target_selector);
        if (!element) {
            return { success: false, error: `Element not found: ${action.target_selector}. Try being more specific or check if the element exists.` };
        }

        // Highlight element briefly
        this.highlightElement(element, [], 2000);

        // Simulate click with proper event handling
        try {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            await new Promise(resolve => setTimeout(resolve, 500)); // Wait for scroll

            element.click();

            // Also trigger mouse events for better compatibility
            element.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
            element.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));

            return { success: true, message: `Successfully clicked: ${element.tagName.toLowerCase()}${element.id ? '#' + element.id : ''}${element.className ? '.' + element.className.split(' ')[0] : ''}` };
        } catch (error) {
            return { success: false, error: `Failed to click element: ${error.message}` };
        }
    }

    async executeFillFormAction(action) {
        const element = this.findElementBySelector(action.target_selector);
        if (!element) {
            return { success: false, error: `Form element not found: ${action.target_selector}` };
        }

        if (action.data && action.data.value) {
            element.value = action.data.value;
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
        }

        return { success: true, message: `Filled form element: ${action.target_selector}` };
    }

    async executeNavigateAction(action) {
        // Check for URL in data.url or target_selector
        let url = null;

        if (action.data && action.data.url) {
            url = action.data.url;
        } else if (action.target_selector) {
            // Handle relative URLs and absolute URLs
            url = action.target_selector;

            // If it's a relative path, make it absolute
            if (url.startsWith('/')) {
                const currentOrigin = window.location.origin;
                url = currentOrigin + url;
            } else if (!url.startsWith('http://') && !url.startsWith('https://')) {
                // If it doesn't start with protocol, assume it's relative to current page
                const currentBase = window.location.href.split('/').slice(0, -1).join('/');
                url = currentBase + '/' + url;
            }
        }

        if (url) {
            console.log('🧭 Navigating to:', url);
            window.location.href = url;
            return { success: true, message: `Navigating to: ${url}` };
        }

        return { success: false, error: 'No URL provided for navigation. Expected URL in data.url or target_selector.' };
    }

    async executeScrollAction(action) {
        const element = action.target_selector ?
            this.findElementBySelector(action.target_selector) : null;

        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else if (action.data && action.data.direction) {
            const scrollAmount = action.data.amount || 300;
            if (action.data.direction === 'up') {
                window.scrollBy(0, -scrollAmount);
            } else if (action.data.direction === 'down') {
                window.scrollBy(0, scrollAmount);
            }
        }

        return { success: true, message: 'Scroll action completed' };
    }

    async executeExtractDataAction(action) {
        const element = this.findElementBySelector(action.target_selector);
        if (!element) {
            return { success: false, error: `Element not found: ${action.target_selector}` };
        }

        const extractedData = {
            text: element.textContent.trim(),
            html: element.innerHTML,
            attributes: {}
        };

        // Extract common attributes
        ['href', 'src', 'alt', 'title', 'value'].forEach(attr => {
            if (element.hasAttribute(attr)) {
                extractedData.attributes[attr] = element.getAttribute(attr);
            }
        });

        return { success: true, data: extractedData };
    }

    findElementBySelector(selector) {
        try {
            console.log('🔍 Finding element with selector:', selector);

            // Try direct CSS selector first
            let element = document.querySelector(selector);
            if (element) {
                console.log('✅ Found element via CSS selector:', element);
                return element;
            }

            // Try common button/link selectors if selector looks like text
            if (!selector.includes('[') && !selector.includes('#') && !selector.includes('.')) {
                const commonSelectors = [
                    `a[href*="${selector}"]`,
                    `a[href="${selector}"]`,
                    `button:contains("${selector}")`,
                    `a:contains("${selector}")`,
                    `[data-testid*="${selector}"]`,
                    `[aria-label*="${selector}"]`,
                    `[title*="${selector}"]`
                ];

                for (const sel of commonSelectors) {
                    try {
                        element = document.querySelector(sel);
                        if (element) {
                            console.log('✅ Found element via common selector:', sel, element);
                            return element;
                        }
                    } catch (e) {
                        // Ignore selector errors for :contains() etc.
                    }
                }
            }

            // Try finding by text content (for links and buttons)
            const clickableElements = document.querySelectorAll('a, button, [role="button"], [onclick]');
            for (const el of clickableElements) {
                if (el.textContent.trim().toLowerCase().includes(selector.toLowerCase())) {
                    console.log('✅ Found element via text content:', el);
                    return el;
                }
            }

            // Try finding by href attribute for navigation
            if (selector.startsWith('/') || selector.includes('http')) {
                const links = document.querySelectorAll('a[href]');
                for (const link of links) {
                    if (link.href.includes(selector) || link.getAttribute('href') === selector) {
                        console.log('✅ Found link via href:', link);
                        return link;
                    }
                }
            }

            // Try finding by indexed elements
            for (const [id, item] of this.pageIndex) {
                if (item.title.toLowerCase().includes(selector.toLowerCase()) ||
                    item.snippet.toLowerCase().includes(selector.toLowerCase())) {
                    console.log('✅ Found element via page index:', item.element);
                    return item.element;
                }
            }

            console.log('❌ Element not found for selector:', selector);
            return null;
        } catch (error) {
            console.error('Element selection error:', error);
            return null;
        }
    }

    // Form Detection and Autofill Methods
    setupFormDetection() {
        // Monitor for new forms added dynamically
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const forms = node.querySelectorAll ? node.querySelectorAll('form') : [];
                        if (node.tagName === 'FORM') {
                            this.processForm(node);
                        }
                        forms.forEach(form => this.processForm(form));
                    }
                });
            });
        });

        observer.observe(document.body, { childList: true, subtree: true });

        // Process existing forms
        document.querySelectorAll('form').forEach(form => this.processForm(form));
    }

    processForm(form) {
        // Add ToolBox form detection attributes
        form.setAttribute('data-tb-form', 'detected');

        // Detect login forms
        const passwordFields = form.querySelectorAll('input[type="password"]');
        const emailFields = form.querySelectorAll('input[type="email"], input[name*="email"], input[name*="username"], input[name*="login"]');

        if (passwordFields.length > 0 && emailFields.length > 0) {
            form.setAttribute('data-tb-form-type', 'login');
            this.setupFormAutofill(form);
        }
    }

    setupFormAutofill(form) {
        // Wrapper für unsere Indikatoren erstellen
        const wrapper = document.createElement('div');
        wrapper.className = 'tb-form-indicator-wrapper';
        wrapper.style.cssText = `position: absolute; top: -12px; right: -12px; z-index: 1000; display: flex; gap: 4px;`;

        // Bestehender Indikator
        const indicator = document.createElement('div');
        indicator.className = 'tb-form-indicator';
        indicator.title = 'ToolBox Pro hat dieses Formular erkannt';
        indicator.innerHTML = '🔐';

        // NEUER "Speichern"-Button
        const saveBtn = document.createElement('button');
        saveBtn.className = 'tb-form-save-btn';
        saveBtn.title = 'Anmeldeinformationen manuell speichern';
        saveBtn.innerHTML = '💾'; // Save icon
        saveBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.handleManualSave(form);
        };

        wrapper.appendChild(indicator);
        wrapper.appendChild(saveBtn);

        if (getComputedStyle(form).position === 'static') {
            form.style.position = 'relative';
        }
        form.appendChild(wrapper);
    }

    handleManualSave(form) {
        const usernameField = form.querySelector('input[type="email"], input[name*="email"], input[name*="username"], input[name*="login"], input[type="text"]');
        const passwordField = form.querySelector('input[type="password"]');

        if (usernameField && passwordField && usernameField.value && passwordField.value) {
            const credentials = {
                url: window.location.href,
                username: usernameField.value,
                password: passwordField.value,
                title: new URL(window.location.href).hostname // Simple title
            };

            chrome.runtime.sendMessage({ type: 'MANUAL_SAVE_PASSWORD', credentials });
            this.showGestureFeedback('Gespeichert!');
        } else {
            this.showGestureFeedback('Bitte füllen Sie Benutzername und Passwort aus.');
        }
    }

    async autofillPassword(data) {
        console.log('🔐 Autofilling password...');

        try {
            // Find login form
            const forms = document.querySelectorAll('form[data-tb-form-type="login"]');
            let targetForm = forms[0];

            if (!targetForm) {
                // Fallback: find any form with password field
                targetForm = document.querySelector('form:has(input[type="password"])');
            }

            if (!targetForm) {
                console.warn('No suitable form found for autofill');
                return;
            }

            // Fill username/email
            const usernameField = targetForm.querySelector(
                'input[type="email"], input[name*="email"], input[name*="username"], input[name*="login"], input[type="text"]'
            );

            if (usernameField && data.username) {
                this.fillField(usernameField, data.username);
            }

            // Fill password
            const passwordField = targetForm.querySelector('input[type="password"]');
            if (passwordField && data.password) {
                this.fillField(passwordField, data.password);
            }

            // Show success indicator
            this.showAutofillSuccess(targetForm);

        } catch (error) {
            console.error('Autofill error:', error);
        }
    }

    fillField(field, value) {
        // Clear field first
        field.value = '';

        // Simulate typing for better compatibility
        field.focus();

        // Trigger input events
        field.dispatchEvent(new Event('focus', { bubbles: true }));

        // Set value
        field.value = value;

        // Trigger change events
        field.dispatchEvent(new Event('input', { bubbles: true }));
        field.dispatchEvent(new Event('change', { bubbles: true }));
        field.dispatchEvent(new Event('blur', { bubbles: true }));
    }

    fillGeneratedPassword(password) {
        // Find password fields
        const passwordFields = document.querySelectorAll('input[type="password"]');

        passwordFields.forEach(field => {
            this.fillField(field, password);
        });

        if (passwordFields.length > 0) {
            this.showAutofillSuccess(passwordFields[0].closest('form'));
        }
    }

    showAutofillSuccess(form) {
        if (!form) return;

        // Create success indicator
        const success = document.createElement('div');
        success.className = 'tb-autofill-success';
        success.innerHTML = '✅ Autofilled by ToolBox';
        success.style.cssText = `
            position: absolute;
            top: -30px;
            left: 50%;
            transform: translateX(-50%);
            background: #4CAF50;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            z-index: 10000;
            animation: tbAutofillSuccess 3s ease-in-out forwards;
        `;

        form.style.position = 'relative';
        form.appendChild(success);

        // Remove after animation
        setTimeout(() => {
            if (success.parentNode) {
                success.parentNode.removeChild(success);
            }
        }, 3000);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes tbHighlightPulse {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.02); }
    }

    @keyframes tbAutofillSuccess {
        0% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
        20% { opacity: 1; transform: translateX(-50%) translateY(0); }
        80% { opacity: 1; transform: translateX(-50%) translateY(0); }
        100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
    }

    .tb-search-highlight {
        outline: 2px solid #2E86AB !important;
        outline-offset: 2px !important;
    }
`;
document.head.appendChild(style);

// Initialize content script
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.toolboxContent = new ToolBoxContent();
    });
} else {
    window.toolboxContent = new ToolBoxContent();
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolBoxContent;
}
