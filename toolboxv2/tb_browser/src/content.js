// ToolBox Pro - Content Script
// Handles page interaction, gesture detection, search, and autofill

class ToolBoxContent {
    constructor() {
        this.gestureDetector = null;
        this.pageIndex = new Map();
        this.highlightedElements = [];
        this.isInitialized = false;

        this.init();
    }

    async init() {
        if (this.isInitialized) return;

        console.log('🚀 ToolBox Pro content script initializing...');

        try {
            // Load gesture detector
            await this.loadGestureDetector();

            // Setup message listeners
            this.setupMessageListeners();

            // Initialize page indexing
            this.initializePageIndexing();

            // Setup form detection
            this.setupFormDetection();

            this.isInitialized = true;
            console.log('✅ ToolBox Pro content script initialized');

        } catch (error) {
            console.error('❌ ToolBox Pro content script initialization failed:', error);
        }
    }

    async loadGestureDetector() {
        try {
            // Initialize gesture detection directly in content script
            this.initializeGestureDetection();
            console.log('✅ Gesture detector loaded');
        } catch (error) {
            console.error('Failed to load gesture detector:', error);
        }
    }

    initializeGestureDetection() {
        this.gestureState = {
            isTracking: false,
            startX: 0,
            startY: 0,
            currentX: 0,
            currentY: 0,
            startTime: 0,
            lastClickTime: 0
        };

        // Mouse events
        document.addEventListener('mousedown', (e) => this.handleGestureStart(e.clientX, e.clientY, e));
        document.addEventListener('mousemove', (e) => this.handleGestureMove(e.clientX, e.clientY, e));
        document.addEventListener('mouseup', (e) => this.handleGestureEnd(e.clientX, e.clientY, e));

        // Touch events
        document.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                const touch = e.touches[0];
                this.handleGestureStart(touch.clientX, touch.clientY, e);
            }
        }, { passive: true });

        document.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1) {
                const touch = e.touches[0];
                this.handleGestureMove(touch.clientX, touch.clientY, e);
            }
        }, { passive: true });

        document.addEventListener('touchend', (e) => {
            if (e.changedTouches.length === 1) {
                const touch = e.changedTouches[0];
                this.handleGestureEnd(touch.clientX, touch.clientY, e);
            }
        }, { passive: true });

        // Double-click detection
        document.addEventListener('dblclick', (e) => this.handleDoubleClick(e));
    }

    handleGestureStart(x, y, event) {
        this.gestureState.isTracking = true;
        this.gestureState.startX = x;
        this.gestureState.startY = y;
        this.gestureState.currentX = x;
        this.gestureState.currentY = y;
        this.gestureState.startTime = Date.now();
    }

    handleGestureMove(x, y, event) {
        if (!this.gestureState.isTracking) return;

        this.gestureState.currentX = x;
        this.gestureState.currentY = y;
    }

    handleGestureEnd(x, y, event) {
        if (!this.gestureState.isTracking) return;

        this.gestureState.isTracking = false;

        const deltaX = x - this.gestureState.startX;
        const deltaY = y - this.gestureState.startY;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        const duration = Date.now() - this.gestureState.startTime;

        // Minimum distance and maximum duration for gesture recognition
        if (distance > 100 && duration < 1000) {
            const gesture = this.recognizeGesture(deltaX, deltaY, distance);
            if (gesture) {
                this.executeGesture(gesture);
            }
        }
    }

    recognizeGesture(deltaX, deltaY, distance) {
        const absX = Math.abs(deltaX);
        const absY = Math.abs(deltaY);

        // Determine primary direction
        if (absX > absY) {
            // Horizontal gesture
            if (deltaX > 0) {
                return 'swipe-right';
            } else {
                return 'swipe-left';
            }
        } else {
            // Vertical gesture
            if (deltaY > 0) {
                return 'swipe-down';
            } else {
                return 'swipe-up';
            }
        }
    }

    executeGesture(gesture) {
        console.log('🎯 Executing gesture:', gesture);

        switch (gesture) {
            case 'swipe-left':
                window.history.back();
                this.showGestureFeedback('← Back');
                break;
            case 'swipe-right':
                window.history.forward();
                this.showGestureFeedback('→ Forward');
                break;
            case 'swipe-up':
                this.scrollUp();
                this.showGestureFeedback('↑ Scroll Up');
                break;
            case 'swipe-down':
                this.scrollDown();
                this.showGestureFeedback('↓ Scroll Down');
                break;
        }

        // Send to background script
        chrome.runtime.sendMessage({
            type: 'GESTURE_DETECTED',
            gesture: gesture,
            timestamp: Date.now()
        });
    }

    scrollUp() {
        const scrollAmount = Math.min(window.innerHeight * 0.8, 500);
        window.scrollBy({
            top: -scrollAmount,
            behavior: 'smooth'
        });
    }

    scrollDown() {
        const scrollAmount = Math.min(window.innerHeight * 0.8, 500);
        window.scrollBy({
            top: scrollAmount,
            behavior: 'smooth'
        });
    }

    handleDoubleClick(event) {
        console.log('🎯 Double-click detected, opening ToolBox popup');

        // Send message to background script to open popup
        chrome.runtime.sendMessage({
            type: 'OPEN_POPUP',
            position: {
                x: event.clientX,
                y: event.clientY
            }
        });

        this.showGestureFeedback('ToolBox Opening...');
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
        }
    }

    getElementContent(elementId) {
        const item = this.pageIndex.get(elementId);
        if (item) {
            return item.fullText || item.snippet || item.title;
        }
        return '';
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
        // Add subtle indicator for ToolBox-enabled forms
        const indicator = document.createElement('div');
        indicator.className = 'tb-form-indicator';
        indicator.innerHTML = '🔐';
        indicator.style.cssText = `
            position: absolute;
            top: -10px;
            right: -10px;
            width: 20px;
            height: 20px;
            background: #2E86AB;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            z-index: 1000;
            opacity: 0.7;
            transition: opacity 0.3s ease;
        `;

        // Position relative to form
        if (form.style.position !== 'relative') {
            form.style.position = 'relative';
        }

        form.appendChild(indicator);

        // Show on hover
        form.addEventListener('mouseenter', () => {
            indicator.style.opacity = '1';
        });

        form.addEventListener('mouseleave', () => {
            indicator.style.opacity = '0.7';
        });
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
