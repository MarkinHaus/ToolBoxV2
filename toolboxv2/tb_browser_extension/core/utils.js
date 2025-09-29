// ToolBox Browser Extension - Core Utilities
// Enhanced utility functions for the ToolBox browser extension

class TBUtils {
    static version = "2.0.0";
    static _debug = true;

    // Logging utilities
    static log(level, component, message, data = null) {
        if (!this._debug && level === 'debug') return;

        const timestamp = new Date().toISOString();
        const prefix = `[TB-${level.toUpperCase()}] ${timestamp} [${component}]`;

        if (data) {
            console[level](prefix, message, data);
        } else {
            console[level](prefix, message);
        }
    }

    static info(component, message, data = null) {
        this.log('info', component, message, data);
    }

    static warn(component, message, data = null) {
        this.log('warn', component, message, data);
    }

    static error(component, message, data = null) {
        this.log('error', component, message, data);
    }

    static debug(component, message, data = null) {
        this.log('debug', component, message, data);
    }



    // DOM utilities
    static createElement(tag, attributes = {}, children = []) {
        const element = document.createElement(tag);

        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'innerHTML') {
                // Use safer innerHTML setting with error handling
                try {
                    element.innerHTML = value;
                } catch (error) {
                    console.warn('Failed to set innerHTML, using textContent instead:', error);
                    element.textContent = value.replace(/<[^>]*>/g, ''); // Strip HTML tags
                }
            } else if (key.startsWith('data-')) {
                element.setAttribute(key, value);
            } else {
                element[key] = value;
            }
        });

        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else {
                element.appendChild(child);
            }
        });

        return element;
    }

    // Alternative safer method for creating HTML elements
    static createHTMLElement(htmlString) {
        try {
            const template = document.createElement('template');
            template.innerHTML = htmlString.trim();
            return template.content.firstChild;
        } catch (error) {
            console.warn('Failed to create HTML element from string:', error);
            const div = document.createElement('div');
            div.textContent = htmlString.replace(/<[^>]*>/g, '');
            return div;
        }
    }

    static waitForElement(selector, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const element = document.querySelector(selector);
            if (element) {
                resolve(element);
                return;
            }

            const observer = new MutationObserver((mutations, obs) => {
                const element = document.querySelector(selector);
                if (element) {
                    obs.disconnect();
                    resolve(element);
                }
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });

            setTimeout(() => {
                observer.disconnect();
                reject(new Error(`Element ${selector} not found within ${timeout}ms`));
            }, timeout);
        });
    }

    // Animation utilities
    static fadeIn(element, duration = 300) {
        element.style.opacity = '0';
        element.style.display = 'block';

        const start = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);

            element.style.opacity = progress;

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    static fadeOut(element, duration = 300) {
        const start = performance.now();
        const startOpacity = parseFloat(element.style.opacity) || 1;

        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);

            element.style.opacity = startOpacity * (1 - progress);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                element.style.display = 'none';
            }
        };

        requestAnimationFrame(animate);
    }

    // Storage utilities
    static async getStorage(keys) {
        return new Promise((resolve) => {
            chrome.storage.sync.get(keys, resolve);
        });
    }

    static async setStorage(data) {
        return new Promise((resolve) => {
            chrome.storage.sync.set(data, resolve);
        });
    }

    static async getLocalStorage(keys) {
        return new Promise((resolve) => {
            chrome.storage.local.get(keys, resolve);
        });
    }

    static async setLocalStorage(data) {
        return new Promise((resolve) => {
            chrome.storage.local.set(data, resolve);
        });
    }

    // Event utilities
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    static throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // URL and domain utilities
    static getCurrentDomain() {
        return window.location.hostname;
    }

    static isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    // Text processing utilities
    static sanitizeText(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    static extractTextFromElement(element) {
        return element.innerText || element.textContent || '';
    }

    // Performance utilities
    static measurePerformance(name, func) {
        const start = performance.now();
        const result = func();
        const end = performance.now();

        this.debug('Performance', `${name} took ${(end - start).toFixed(2)}ms`);

        return result;
    }

    static async measureAsyncPerformance(name, func) {
        const start = performance.now();
        const result = await func();
        const end = performance.now();

        this.debug('Performance', `${name} took ${(end - start).toFixed(2)}ms`);

        return result;
    }

    // Error handling utilities
    static handleError(component, error, context = {}) {
        this.error(component, `Error: ${error.message}`, {
            error: error.stack,
            context
        });

        // Send error to background script for logging
        if (typeof chrome !== 'undefined' && chrome.runtime) {
            chrome.runtime.sendMessage({
                type: 'TB_ERROR',
                component,
                error: error.message,
                stack: error.stack,
                context,
                timestamp: Date.now()
            }).catch(() => {
                // Ignore errors when sending error reports
            });
        }
    }

    // Feature detection
    static hasFeature(feature) {
        const features = {
            speechRecognition: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
            speechSynthesis: 'speechSynthesis' in window,
            webRTC: 'RTCPeerConnection' in window,
            webGL: !!document.createElement('canvas').getContext('webgl'),
            localStorage: 'localStorage' in window,
            sessionStorage: 'sessionStorage' in window,
            indexedDB: 'indexedDB' in window,
            serviceWorker: 'serviceWorker' in navigator,
            notifications: 'Notification' in window,
            geolocation: 'geolocation' in navigator
        };

        return features[feature] || false;
    }

    // Initialization
    static init() {
        this.info('Utils', `ToolBox Utils v${this.version} initialized`);

        // Set up global error handler
        window.addEventListener('error', (event) => {
            this.handleError('Global', event.error, {
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno
            });
        });

        // Set up unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            this.handleError('Promise', event.reason, {
                promise: event.promise
            });
        });
    }
}

// Initialize utilities when loaded
if (typeof window !== 'undefined') {
    TBUtils.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TBUtils;
} else if (typeof window !== 'undefined') {
    window.TBUtils = TBUtils;
}
