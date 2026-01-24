/**
 * Mobile Navigation - Bottom Tab Bar
 * ===================================
 *
 * Only loaded on mobile platforms (Android/iOS)
 * Provides bottom tab bar navigation for mobile app
 *
 * Usage:
 * ```javascript
 * import { MobileNav } from './mobile/MobileNav.js';
 *
 * const nav = new MobileNav({
 *     tabs: [
 *         { id: 'home', icon: '🏠', label: 'Home' },
 *         { id: 'settings', icon: '⚙️', label: 'Settings' }
 *     ],
 *     activeTab: 'home',
 *     onTabChange: (tabId) => console.log('Tab changed:', tabId)
 * });
 *
 * nav.mount(); // Adds to document.body
 * ```
 */

export class MobileNav {
    /**
     * Create a mobile navigation bar
     * @param {Object} options - Configuration options
     * @param {Array} options.tabs - Array of tab objects { id, icon, label }
     * @param {string} options.activeTab - ID of initially active tab
     * @param {Function} options.onTabChange - Callback when tab changes
     */
    constructor(options = {}) {
        this.tabs = options.tabs || [
            { id: 'hud', icon: '🎯', label: 'Hud' },
            { id: 'app', icon: '🏠', label: 'App' },
            { id: 'settings', icon: '⚙️', label: 'Settings' }
        ];
        this.activeTab = options.activeTab || this.tabs[0]?.id || 'hud';
        this.onTabChange = options.onTabChange || (() => {});
        this.element = null;
        this.styleElement = null;
    }

    /**
     * Render the navigation bar
     * @returns {HTMLElement} The navigation element
     */
    render() {
        // Create container
        this.element = document.createElement('nav');
        this.element.className = 'mobile-nav';
        this.element.setAttribute('role', 'navigation');
        this.element.setAttribute('aria-label', 'Main navigation');

        // Add styles if not already added
        this.injectStyles();

        // Render tabs
        this.element.innerHTML = this.tabs.map(tab => `
            <button
                class="mobile-nav-tab ${tab.id === this.activeTab ? 'active' : ''}"
                data-tab="${this.escapeAttr(tab.id)}"
                role="tab"
                aria-selected="${tab.id === this.activeTab}"
                aria-label="${this.escapeAttr(tab.label)}"
            >
                <span class="mobile-nav-icon" aria-hidden="true">${tab.icon}</span>
                <span class="mobile-nav-label">${this.escapeHtml(tab.label)}</span>
            </button>
        `).join('');

        // Bind events
        this.bindEvents();

        return this.element;
    }

    /**
     * Inject styles into document head
     */
    injectStyles() {
        if (document.getElementById('mobile-nav-styles')) {
            return; // Already injected
        }

        this.styleElement = document.createElement('style');
        this.styleElement.id = 'mobile-nav-styles';
        this.styleElement.textContent = `
            .mobile-nav {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: var(--bg-secondary, #1a1a2e);
                border-top: 1px solid var(--border-color, rgba(255, 255, 255, 0.1));
                display: flex;
                justify-content: space-around;
                padding: 8px 0;
                padding-bottom: max(8px, env(safe-area-inset-bottom));
                z-index: 1000;
                box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.2);
            }

            .mobile-nav-tab {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 8px 16px;
                min-width: 60px;
                background: none;
                border: none;
                color: var(--text-muted, rgba(255, 255, 255, 0.5));
                cursor: pointer;
                transition: all 0.2s ease;
                -webkit-tap-highlight-color: transparent;
            }

            .mobile-nav-tab:hover {
                color: var(--text-color, #e8e8f0);
            }

            .mobile-nav-tab.active {
                color: var(--accent-color, #6366f1);
            }

            .mobile-nav-tab:focus {
                outline: none;
            }

            .mobile-nav-tab:focus-visible {
                outline: 2px solid var(--accent-color, #6366f1);
                outline-offset: 2px;
                border-radius: 4px;
            }

            .mobile-nav-icon {
                font-size: 20px;
                margin-bottom: 4px;
                line-height: 1;
            }

            .mobile-nav-label {
                font-size: 10px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }

            /* Active indicator animation */
            .mobile-nav-tab.active .mobile-nav-icon {
                transform: scale(1.1);
                transition: transform 0.2s ease;
            }

            /* Adjust main content for nav */
            body.has-mobile-nav {
                padding-bottom: calc(70px + env(safe-area-inset-bottom));
            }

            /* Dark mode support */
            @media (prefers-color-scheme: dark) {
                .mobile-nav {
                    background: #0f0f1a;
                    border-top-color: rgba(255, 255, 255, 0.05);
                }
            }

            /* Landscape orientation - more compact */
            @media (orientation: landscape) and (max-height: 500px) {
                .mobile-nav {
                    padding: 4px 0;
                    padding-bottom: max(4px, env(safe-area-inset-bottom));
                }

                .mobile-nav-tab {
                    padding: 4px 12px;
                }

                .mobile-nav-icon {
                    font-size: 18px;
                    margin-bottom: 2px;
                }

                .mobile-nav-label {
                    font-size: 9px;
                }

                body.has-mobile-nav {
                    padding-bottom: calc(50px + env(safe-area-inset-bottom));
                }
            }
        `;

        document.head.appendChild(this.styleElement);
    }

    /**
     * Bind click events to tabs
     */
    bindEvents() {
        if (!this.element) return;

        this.element.querySelectorAll('.mobile-nav-tab').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabId = e.currentTarget.dataset.tab;
                this.setActiveTab(tabId);
            });

            // Touch feedback
            btn.addEventListener('touchstart', () => {
                btn.style.opacity = '0.7';
            }, { passive: true });

            btn.addEventListener('touchend', () => {
                btn.style.opacity = '1';
            }, { passive: true });
        });
    }

    /**
     * Set the active tab
     * @param {string} tabId - ID of the tab to activate
     */
    setActiveTab(tabId) {
        if (tabId === this.activeTab) return;

        const previousTab = this.activeTab;
        this.activeTab = tabId;

        // Update UI
        if (this.element) {
            this.element.querySelectorAll('.mobile-nav-tab').forEach(btn => {
                const isActive = btn.dataset.tab === tabId;
                btn.classList.toggle('active', isActive);
                btn.setAttribute('aria-selected', isActive.toString());
            });
        }

        // Callback
        this.onTabChange(tabId, previousTab);
    }

    /**
     * Get the currently active tab ID
     * @returns {string} Active tab ID
     */
    getActiveTab() {
        return this.activeTab;
    }

    /**
     * Add a new tab
     * @param {Object} tab - Tab object { id, icon, label }
     * @param {number} position - Optional position to insert at
     */
    addTab(tab, position = -1) {
        if (position >= 0 && position < this.tabs.length) {
            this.tabs.splice(position, 0, tab);
        } else {
            this.tabs.push(tab);
        }

        // Re-render if mounted
        if (this.element && this.element.parentNode) {
            const parent = this.element.parentNode;
            this.unmount();
            this.mount(parent);
        }
    }

    /**
     * Remove a tab by ID
     * @param {string} tabId - ID of tab to remove
     */
    removeTab(tabId) {
        const index = this.tabs.findIndex(t => t.id === tabId);
        if (index >= 0) {
            this.tabs.splice(index, 1);

            // If removed tab was active, switch to first tab
            if (this.activeTab === tabId && this.tabs.length > 0) {
                this.setActiveTab(this.tabs[0].id);
            }

            // Re-render if mounted
            if (this.element && this.element.parentNode) {
                const parent = this.element.parentNode;
                this.unmount();
                this.mount(parent);
            }
        }
    }

    /**
     * Mount the navigation bar to a container
     * @param {HTMLElement} container - Container element (default: document.body)
     */
    mount(container = document.body) {
        const el = this.render();
        container.appendChild(el);
        document.body.classList.add('has-mobile-nav');
    }

    /**
     * Unmount the navigation bar
     */
    unmount() {
        if (this.element) {
            this.element.remove();
            this.element = null;
        }
        document.body.classList.remove('has-mobile-nav');
    }

    /**
     * Show the navigation bar
     */
    show() {
        if (this.element) {
            this.element.style.display = 'flex';
            document.body.classList.add('has-mobile-nav');
        }
    }

    /**
     * Hide the navigation bar
     */
    hide() {
        if (this.element) {
            this.element.style.display = 'none';
            document.body.classList.remove('has-mobile-nav');
        }
    }

    /**
     * Escape HTML special characters
     * @param {string} str - String to escape
     * @returns {string} Escaped string
     */
    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Escape attribute value
     * @param {string} str - String to escape
     * @returns {string} Escaped string
     */
    escapeAttr(str) {
        return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
}

export default MobileNav;
