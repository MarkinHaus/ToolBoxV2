// tbjs/ui/components/NavMenu/NavMenu.js
// NO CHANGES to this file. It's provided as a correct reference.
import TB from '../../../index.js';

const DEFAULT_NAVMENU_OPTIONS = {
    triggerSelector: '#links', // Selector for the menu toggle button
    menuContentHtml: `
           <ul class="nav-list" style="list-style: none; padding: 0; margin: 0;">
            <li>
                <a href="/web/core0/index.html" class="nav-item">
                    <span class="material-symbols-outlined nav-icon">home</span>
                    <span class="nav-text">Home</span>
                </a>
            </li>
            <li>
                <a href="/web/assets/login.html" class="nav-item">
                    <span class="material-symbols-outlined nav-icon">login</span>
                    <span class="nav-text">Login</span>
                </a>
            </li>
            <li>
                <a href="/web/mainContent.html" class="nav-item">
                    <span class="material-symbols-outlined nav-icon">apps</span>
                    <span class="nav-text">Apps</span>
                </a>
            </li>
            <li>
                <a href="/api/CloudM.UI.widget/get_widget" class="nav-item">
                    <span class="material-symbols-outlined nav-icon">settings</span>
                    <span class="nav-text">Config</span>
                </a>
            </li>
            <li>
                <a href="/web/assets/terms.html" class="nav-item">
                    <span class="material-symbols-outlined nav-icon">description</span>
                    <span class="nav-text">Terms & Conditions</span>
                </a>
            </li>
        </ul>
    `,
    menuId: 'tb-nav-menu-modal',
    openIconClass: 'menu', // Material Symbols class
    closeIconClass: 'close',
    customClasses: {
        overlay: 'fixed inset-0 bg-black bg-opacity-30 z-[1040] opacity-0 transition-opacity duration-300 rounded-lg shadow-md border border-gray-200 max-w-sm mx-auto', // Backdrop
        menuContainer: 'fixed top-0 left-0 h-full w-64 sm:w-72 bg-background-color shadow-xl z-[1041] transform -translate-x-full transition-transform duration-300 ease-in-out', // Slide-in menu
        // Or for a centered modal style:
        // menuContainer: 'fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background-color shadow-xl z-[1041] rounded-lg p-6 opacity-0 scale-95 transition-all duration-300',
        iconContainer: '', // For the icon in the trigger
    }
};

class NavMenu {
    constructor(options = {}) {
        this.options = { ...DEFAULT_NAVMENU_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_NAVMENU_OPTIONS.customClasses, ...options.customClasses };

        this.triggerElement = document.querySelector(this.options.triggerSelector);
        if (!this.triggerElement) {
            TB.logger.warn(`[NavMenu] Trigger element "${this.options.triggerSelector}" not found.`);
            return;
        }

        this.menuOverlayElement = null;
        this.menuContainerElement = null;
        this.isOpen = false;

        this._iconElement = this.triggerElement.querySelector('.material-symbols-outlined') || this.triggerElement.querySelector('span'); // Find icon span
        if (!this._iconElement && this.triggerElement.children.length === 0) { // If trigger is empty, create icon span
            this._iconElement = document.createElement('span');
            this._iconElement.className = `material-symbols-outlined ${this.options.customClasses.iconContainer}`;
            this.triggerElement.appendChild(this._iconElement);
        }


        this._boundToggleMenu = this.toggleMenu.bind(this);
        this._boundHandleDocumentClick = this._handleDocumentClick.bind(this);
        this._boundHandleLinkClick = this._handleLinkClick.bind(this);

        this.triggerElement.addEventListener('click', this._boundToggleMenu);
        this._updateIcon(); // Set initial icon

        TB.logger.log('[NavMenu] Initialized.');
    }

    _createMenuDom() {
        // Overlay for backdrop (optional, good for slide-in menus)
        this.menuOverlayElement = document.createElement('div');
        this.menuOverlayElement.id = `${this.options.menuId}-overlay`;
        this.menuOverlayElement.className = this.options.customClasses.overlay;
        this.menuOverlayElement.addEventListener('click', () => this.closeMenu()); // Close on overlay click

        // Menu container
        this.menuContainerElement = document.createElement('div');
        this.menuContainerElement.id = this.options.menuId;
        this.menuContainerElement.className = this.options.customClasses.menuContainer;
        this.menuContainerElement.innerHTML = this.options.menuContentHtml;

        // Add link click listener to menu container
        this.menuContainerElement.addEventListener('click', this._boundHandleLinkClick);


        document.body.appendChild(this.menuOverlayElement);
        document.getElementById("Nav-Main").appendChild(this.menuContainerElement); // Assuming Nav-Main exists

        TB.ui.processDynamicContent(this.menuContainerElement); // Process links for router etc.
    }

    _handleLinkClick(event) {
        // If a link inside the menu is clicked, close the menu after navigation (if router handles it)
        const link = event.target.closest('a');
        if (link && link.href) {
             // Check if it's an internal link handled by TB.router
            const targetUrl = new URL(link.href, window.location.origin);
            if (targetUrl.origin === window.location.origin && !link.hasAttribute('data-external-link') && !link.hasAttribute('download')) {
                 this.closeMenu();
                 // TB.router.navigateTo will be called by the global link handler or if this one does it.
            }
        }
    }

    _updateIcon() {
        if (this._iconElement) {
            this._iconElement.textContent = this.isOpen ? this.options.closeIconClass : this.options.openIconClass;
            this._iconElement.style.transform = this.isOpen ? 'rotate(360deg)' : 'rotate(0deg)';
            this._iconElement.style.transition = 'transform 0.5s ease';
        }
        if (this.triggerElement) {
            this.triggerElement.setAttribute('aria-expanded', this.isOpen.toString());
        }
    }

    _handleDocumentClick(event) {
        // Close if click is outside the menu and not on the trigger
        if (this.isOpen &&
            this.menuContainerElement && !this.menuContainerElement.contains(event.target) &&
            this.triggerElement && !this.triggerElement.contains(event.target)) {
            this.closeMenu();
        }
    }

    toggleMenu() {
        this.isOpen ? this.closeMenu() : this.openMenu();
    }

    openMenu() {
        if (this.isOpen) return;
        if (!this.menuContainerElement) this._createMenuDom();

        // Ensure DOM elements are actually there before trying to style them
        if (!this.menuOverlayElement || !this.menuContainerElement) {
            TB.logger.error("[NavMenu] Menu DOM elements not created properly.");
            return;
        }
        this.menuContainerElement.style.display = 'block';


        // Force reflow for transitions
        void this.menuOverlayElement.offsetWidth;
        void this.menuContainerElement.offsetWidth;

        this.menuOverlayElement.style.opacity = '1';
        this.menuOverlayElement.style.pointerEvents = 'auto';

        this.menuContainerElement.style.transform = 'translateX(0)';

        this.isOpen = true;
        this._updateIcon();
        document.addEventListener('click', this._boundHandleDocumentClick, true); // Use capture phase
        TB.logger.log('[NavMenu] Opened.');
        TB.events.emit('navMenu:opened', this);
    }

    closeMenu() {
        if (!this.isOpen || !this.menuContainerElement) return;
        this.menuContainerElement.style.display = 'none';
        this.menuOverlayElement.style.opacity = '0';
        this.menuOverlayElement.style.pointerEvents = 'none';

        this.menuContainerElement.style.transform = 'translateX(-100%)';

        // Set display to none AFTER transition
        // Or rely on transform to hide it, and set display none in the setTimeout
        // For this setup, transform hides it. Display none can be in timeout.

        this.isOpen = false;
        this._updateIcon();
        document.removeEventListener('click', this._boundHandleDocumentClick, true);

        if (false) {
            setTimeout(() => {
                if (!this.isOpen && this.menuContainerElement) { // Check menuContainerElement still exists
                    this.menuContainerElement.style.display = 'none'; // Hide it fully
                    // Optionally remove DOM if not frequently used
                    if (this.menuContainerElement.parentNode) {
                        this.menuContainerElement.parentNode.removeChild(this.menuContainerElement);
                    }
                    if (this.menuOverlayElement && this.menuOverlayElement.parentNode) {
                         this.menuOverlayElement.parentNode.removeChild(this.menuOverlayElement);
                    }
                    this.menuContainerElement = null; this.menuOverlayElement = null;
                }
            }, 300); // Match transition duration
        }
        TB.logger.log('[NavMenu] Closed.');
        TB.events.emit('navMenu:closed', this);
    }

    destroy() {
        if (this.triggerElement) { // Check if triggerElement was found
            this.triggerElement.removeEventListener('click', this._boundToggleMenu);
        }
        this.closeMenu(); // Ensure it's closed and listeners removed

        if (this.menuContainerElement) { // Check if it exists before trying to remove listener
            this.menuContainerElement.removeEventListener('click', this._boundHandleLinkClick);
            if (this.menuContainerElement.parentNode) { // Check parentNode before removing
                this.menuContainerElement.parentNode.removeChild(this.menuContainerElement);
            }
        }
        if (this.menuOverlayElement && this.menuOverlayElement.parentNode) { // Check parentNode
            this.menuOverlayElement.parentNode.removeChild(this.menuOverlayElement);
        }
        TB.logger.log('[NavMenu] Destroyed.');
    }

    static init(options) {
        return new NavMenu(options);
    }
}

export default NavMenu;
