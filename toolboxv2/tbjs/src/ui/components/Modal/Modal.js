// tbjs/ui/components/Modal/Modal.js
import TB from '../../../index.js';

const DEFAULT_MODAL_OPTIONS = {
    content: '',
    title: '', // Optional title for the modal
    closeOnOutsideClick: true,
    closeOnEsc: true,
    buttons: [], // Array of {text: string, action: function, className: string (for styling)}
    onOpen: null,
    onClose: null, // Called when modal is fully closed
    beforeClose: null, // Called before starting to close, can return false to prevent closing
    maxWidth: 'max-w-lg', // Tailwind class for max width
    modalId: null, // Custom ID for the modal element
    customClasses: {
        overlay: '',
        modalContainer: '',
        header: '',
        title: '',
        body: '',
        footer: '',
        closeButton: '',
    }
};

class Modal {
    constructor(options = {}) {
        this.options = { ...DEFAULT_MODAL_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_MODAL_OPTIONS.customClasses, ...options.customClasses };
        this.id = this.options.modalId || TB.utils.uniqueId('tb-modal-');
        this.isOpen = false;
        this._overlayElement = null;
        this._modalElement = null;
        this._boundHandleKeydown = this._handleKeydown.bind(this);
        this._boundHandleOverlayClick = this._handleOverlayClick.bind(this);
    }

    _createDom() {
        // Overlay
        this._overlayElement = document.createElement('div');
        this._overlayElement.id = `${this.id}-overlay`;
        // /* Tailwind: fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 opacity-0 transition-opacity duration-300 ease-in-out */
        this._overlayElement.className = `fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1050] opacity-0 transition-opacity duration-300 ease-in-out ${this.options.customClasses.overlay}`;
        this._overlayElement.addEventListener('click', this._boundHandleOverlayClick);

        // Modal container
        this._modalElement = document.createElement('div');
        this._modalElement.id = this.id;
        this._modalElement.setAttribute('role', 'dialog');
        this._modalElement.setAttribute('aria-modal', 'true');
        this._modalElement.setAttribute('aria-labelledby', `${this.id}-title`);
        // /* Tailwind: bg-white dark:bg-gray-800 rounded-lg shadow-xl transform transition-all duration-300 ease-in-out scale-95 opacity-0 */
        this._modalElement.className = `bg-background-color text-text-color rounded-lg shadow-xl transform transition-all duration-300 ease-in-out scale-95 opacity-0 ${this.options.maxWidth} w-full p-6 ${this.options.customClasses.modalContainer}`; // p-6 for padding

        // Modal Header (optional)
        if (this.options.title || this.options.closeButton !== false) {
            const header = document.createElement('div');
            // /* Tailwind: flex justify-between items-center pb-3 border-b border-gray-200 dark:border-gray-700 */
            header.className = `flex justify-between items-center pb-3 border-b border-border-color ${this.options.customClasses.header}`;

            if (this.options.title) {
                const titleEl = document.createElement('h3');
                titleEl.id = `${this.id}-title`;
                // /* Tailwind: text-lg font-medium text-gray-900 dark:text-white */
                titleEl.className = `text-lg font-medium ${this.options.customClasses.title}`;
                titleEl.textContent = this.options.title;
                header.appendChild(titleEl);
            }

            if (this.options.closeButton !== false) {
                const closeBtn = document.createElement('button');
                // /* Tailwind: text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition */
                closeBtn.className = `material-symbols-outlined p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${this.options.customClasses.closeButton}`;
                closeBtn.innerHTML = 'close'; // Material icon or SVG
                closeBtn.setAttribute('aria-label', 'Close modal');
                closeBtn.onclick = () => this.close();
                header.appendChild(closeBtn);
            }
            this._modalElement.appendChild(header);
        }

        // Modal Body
        const body = document.createElement('div');
        // /* Tailwind: mt-4 */
        body.className = `mt-4 ${this.options.customClasses.body}`;
        if (typeof this.options.content === 'string') {
            body.innerHTML = this.options.content;
        } else if (this.options.content instanceof HTMLElement) {
            body.appendChild(this.options.content);
        }
        this._modalElement.appendChild(body);
        TB.ui.processDynamicContent(body); // Process content for HTMX etc.

        // Modal Footer (optional, for buttons)
        if (this.options.buttons.length > 0) {
            const footer = document.createElement('div');
            // /* Tailwind: mt-6 flex justify-end space-x-3 */
            footer.className = `mt-6 flex flex-wrap justify-end gap-3 ${this.options.customClasses.footer}`;
            this.options.buttons.forEach(btnConfig => {
                const button = TB.ui.Button.create(btnConfig.text, btnConfig.action || (() => this.close()), {
                    variant: btnConfig.variant || 'primary', // 'primary', 'secondary', 'danger' etc.
                    customClasses: btnConfig.className,
                    size: btnConfig.size
                });
                footer.appendChild(button);
            });
            this._modalElement.appendChild(footer);
        }

        this._overlayElement.appendChild(this._modalElement);
        document.body.appendChild(this._overlayElement);
    }

    _handleKeydown(event) {
        if (event.key === 'Escape' && this.options.closeOnEsc) {
            this.close();
        }
    }

    _handleOverlayClick(event) {
        if (event.target === this._overlayElement && this.options.closeOnOutsideClick) {
            this.close();
        }
    }

    show() {
        if (this.isOpen) return;
        this._createDom();

        // Force reflow for transition
        void this._overlayElement.offsetWidth;
        void this._modalElement.offsetWidth;

        this._overlayElement.style.opacity = '1';
        this._modalElement.style.opacity = '1';
        this._modalElement.style.transform = 'scale(1)'; // Or 'translateY(0)' if sliding in

        document.addEventListener('keydown', this._boundHandleKeydown);
        this.isOpen = true;
        TB.logger.log(`[Modal] Shown: #${this.id}`);
        TB.events.emit('modal:shown', this);
        if (this.options.onOpen) this.options.onOpen(this);
        // Focus management: focus first focusable element in modal
        const firstFocusable = this._modalElement.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (firstFocusable) firstFocusable.focus();
    }

    close(triggerEvent = true) {
        if (!this.isOpen) return;

        if (triggerEvent && this.options.beforeClose) {
            if (this.options.beforeClose(this) === false) {
                TB.logger.log(`[Modal] Close prevented by beforeClose: #${this.id}`);
                return; // Prevent closing
            }
        }

        this._overlayElement.style.opacity = '0';
        this._modalElement.style.opacity = '0';
        this._modalElement.style.transform = 'scale(0.95)'; // Or 'translateY(-20px)'

        document.removeEventListener('keydown', this._boundHandleKeydown);

        setTimeout(() => {
            if (this._overlayElement && this._overlayElement.parentNode) {
                this._overlayElement.parentNode.removeChild(this._overlayElement);
            }
            this._overlayElement = null;
            this._modalElement = null;
            this.isOpen = false;
            if (triggerEvent) {
                 TB.logger.log(`[Modal] Closed: #${this.id}`);
                TB.events.emit('modal:closed', this);
                if (this.options.onClose) this.options.onClose(this);
            }
        }, 300); // Match transition duration
    }

    // Static method for convenience
    static show(options) {
        const modalInstance = new Modal(options);
        modalInstance.show();
        return modalInstance;
    }

    // TODO: Add static initAll(selector) for declarative modals in HTML
}

export default Modal;
