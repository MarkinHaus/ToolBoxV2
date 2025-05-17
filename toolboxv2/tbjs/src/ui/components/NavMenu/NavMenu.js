// tbjs/ui/components/NavMenu/NavMenu.js
import TB from '../../../index.js';

const DEFAULT_NAVMENU_OPTIONS = {
    triggerSelector: '#links', // Selector for the menu toggle button
    menuContentHtml: `
        <ul class="space-y-2 p-4">
            <li><a href="/" class="block px-3 py-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">Home</a></li>
            <li><a href="/web/mainContent.html" class="block px-3 py-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">Apps</a></li>
            <hr class="my-2 border-gray-300 dark:border-gray-600"/>
            <li><a href="/web/assets/login.html" class="block px-3 py-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">Login</a></li>
            {/* Add more default items or make this configurable */}
        </ul>
    `,
    menuId: 'tb-nav-menu-modal',
    openIconClass: 'menu', // Material Symbols class
    closeIconClass: 'close',
    customClasses: {
        overlay: 'fixed inset-0 bg-black bg-opacity-30 z-[1040] opacity-0 transition-opacity duration-300', // Backdrop
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
        document.body.appendChild(this.menuContainerElement);

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
            // /* Tailwind for icon rotation or animation */
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

        // Force reflow for transitions
        void this.menuOverlayElement.offsetWidth;
        void this.menuContainerElement.offsetWidth;

        this.menuOverlayElement.style.opacity = '1';
        this.menuOverlayElement.style.pointerEvents = 'auto';

        // /* Tailwind: transform translate-x-0 for slide-in */
        this.menuContainerElement.style.transform = 'translateX(0)';
        // /* Or for modal style: */
        // this.menuContainerElement.style.opacity = '1';
        // this.menuContainerElement.style.transform = 'translate(-50%, -50%) scale(1)';


        this.isOpen = true;
        this._updateIcon();
        document.addEventListener('click', this._boundHandleDocumentClick, true); // Use capture phase
        TB.logger.log('[NavMenu] Opened.');
        TB.events.emit('navMenu:opened', this);
    }

    closeMenu() {
        if (!this.isOpen || !this.menuContainerElement) return;

        this.menuOverlayElement.style.opacity = '0';
        this.menuOverlayElement.style.pointerEvents = 'none';

        // /* Tailwind: transform -translate-x-full for slide-in */
        this.menuContainerElement.style.transform = 'translateX(-100%)';
        // /* Or for modal style: */
        // this.menuContainerElement.style.opacity = '0';
        // this.menuContainerElement.style.transform = 'translate(-50%, -50%) scale(0.95)';

        this.isOpen = false;
        this._updateIcon();
        document.removeEventListener('click', this._boundHandleDocumentClick, true);

        // Optionally remove DOM after transition to save resources if not frequently used
        // setTimeout(() => {
        //     if (!this.isOpen && this.menuContainerElement && this.menuContainerElement.parentNode) {
        //         this.menuContainerElement.parentNode.removeChild(this.menuContainerElement);
        //         this.menuOverlayElement.parentNode.removeChild(this.menuOverlayElement);
        //         this.menuContainerElement = null; this.menuOverlayElement = null;
        //     }
        // }, 300); // Match transition duration

        TB.logger.log('[NavMenu] Closed.');
        TB.events.emit('navMenu:closed', this);
    }

    destroy() {
        this.closeMenu(); // Ensure it's closed and listeners removed
        this.triggerElement.removeEventListener('click', this._boundToggleMenu);
        if (this.menuContainerElement) this.menuContainerElement.removeEventListener('click', this._boundHandleLinkClick);

        if (this.menuContainerElement && this.menuContainerElement.parentNode) {
            this.menuContainerElement.parentNode.removeChild(this.menuContainerElement);
        }
        if (this.menuOverlayElement && this.menuOverlayElement.parentNode) {
            this.menuOverlayElement.parentNode.removeChild(this.menuOverlayElement);
        }
        TB.logger.log('[NavMenu] Destroyed.');
    }

    // Static initializer
    static init(options) {
        return new NavMenu(options);
    }
}

export default NavMenu;
