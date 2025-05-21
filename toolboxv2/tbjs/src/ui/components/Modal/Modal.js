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
        modalContainer: '', // This will contain the milk glass styles
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
        // Deep merge customClasses
        this.options.customClasses = {
            ...DEFAULT_MODAL_OPTIONS.customClasses,
            ...(options.customClasses || {}),
        };
        this.id = this.options.modalId || TB.utils.uniqueId('tb-modal-');
        this.isOpen = false;
        this._overlayElement = null;
        this._modalElement = null;
        this._boundHandleKeydown = this._handleKeydown.bind(this);
        this._boundHandleOverlayClick = this._handleOverlayClick.bind(this);
    }

    _createDom() {
        // --- Overlay ---
        this._overlayElement = document.createElement('div');
        this._overlayElement.id = `${this.id}-overlay`;
        // **MODIFIED OVERLAY CLASSES:**
        // - `fixed inset-0`: Full screen coverage.
        // - `bg-black/30 dark:bg-black/60`: Semi-transparent backdrop.
        // - `backdrop-blur-sm`: Slight blur on content BEHIND the overlay.
        // - `flex items-center justify-center`: CRITICAL for centering the modal.
        // - `p-4`: Padding so modal doesn't touch screen edges on small screens.
        // - `z-[1050]`: High z-index.
        // - `opacity-0 ...`: For animations.
        this._overlayElement.className = `
            fixed inset-0
            bg-black/30 dark:bg-black/60 backdrop-blur-sm
            flex items-center justify-center
            p-4
            z-[1050]
            opacity-0 transition-opacity duration-300 ease-in-out
            ${this.options.customClasses.overlay}
        `.trim().replace(/\s+/g, ' ');
        this._overlayElement.addEventListener('click', this._boundHandleOverlayClick);

        // --- Modal container ---
        this._modalElement = document.createElement('div');
        this._modalElement.id = this.id;
        this._modalElement.setAttribute('role', 'dialog');
        this._modalElement.setAttribute('aria-modal', 'true');
        // this._modalElement.setAttribute('aria-labelledby', `${this.id}-title`);

        // **MODIFIED MODAL CONTAINER CLASSES for Milk Glass Effect & Rounded Borders:**
        // - `bg-white/60 dark:bg-neutral-800/60`: Semi-transparent background (adjust opacity as needed, e.g., /60, /70).
        // - `backdrop-blur-lg dark:backdrop-blur-xl`: Stronger blur for the modal itself, blurring the overlay.
        // - `border border-white/30 dark:border-neutral-700/30`: Subtle border.
        // - `rounded-2xl`: More pronounced rounded corners.
        // - `text-neutral-900 dark:text-neutral-100`: Base text colors for readability.
        // - `shadow-xl`: Kept.
        // - `transform transition-all ...`: Kept for animations.
        // - `w-full`: Takes available width within the flex-centered container.
        // - `${this.options.maxWidth}`: Constrains the width.
        // - `p-0`: Padding will be handled by header/body/footer or the content itself.
        //            OR, if you want consistent padding, use `p-6` here and remove from content.
        // - `max-h-[90vh] overflow-y-auto`: Prevent modal from exceeding viewport height and enable scroll.
        this._modalElement.className = `
            bg-white/60 dark:bg-neutral-800/70
            backdrop-blur-lg dark:backdrop-blur-xl
            border border-white/40 dark:border-neutral-600/50
            text-neutral-900 dark:text-neutral-100
            rounded-2xl
            shadow-xl
            transform transition-all duration-300 ease-in-out scale-95 opacity-0
            w-full ${this.options.maxWidth}
            p-0 /* Let internal elements or customClasses.body define padding */
            max-h-[90vh] overflow-y-auto /* Scroll for long content */
            flex flex-col /* To make header/body/footer behave well */
            ${this.options.customClasses.modalContainer}
        `.trim().replace(/\s+/g, ' ');


        // Modal Header (optional)
        if (this.options.title || this.options.closeButton !== false) {
            const header = document.createElement('div');
            // **MODIFIED HEADER CLASSES:** Added padding. `flex-shrink-0`
            header.className = `flex justify-between items-center p-4 md:p-6 border-b border-neutral-300/50 dark:border-neutral-700/50 flex-shrink-0 ${this.options.customClasses.header}`;

            if (this.options.title) {
                const titleEl = document.createElement('h3');
                // titleEl.id = `${this.id}-title`;
                titleEl.className = `text-lg font-semibold ${this.options.customClasses.title}`;
                titleEl.textContent = this.options.title;
                header.appendChild(titleEl);
            }

            if (this.options.closeButton !== false) {
                const closeBtn = document.createElement('button');
                // Consider more contrast for close button if needed
                closeBtn.className = `material-symbols-outlined p-1 rounded text-neutral-600 dark:text-neutral-400 hover:bg-neutral-500/20 dark:hover:bg-neutral-300/20 transition ${this.options.customClasses.closeButton}`;
                closeBtn.innerHTML = 'close';
                closeBtn.style.scale = 1.5;
                closeBtn.setAttribute('aria-label', 'Close modal');
                closeBtn.onclick = () => this.close();
                header.appendChild(closeBtn);
            }
            this._modalElement.appendChild(header);
        }

        // Modal Body
        const body = document.createElement('div');
        // **MODIFIED BODY CLASSES:** Added padding, `overflow-y-auto` if header/footer are fixed. `flex-grow`
        body.className = `p-4 md:p-6 flex-grow overflow-y-auto ${this.options.customClasses.body}`;
        if (typeof this.options.content === 'string') {
            body.innerHTML = this.options.content;
        } else if (this.options.content instanceof HTMLElement) {
            body.appendChild(this.options.content);
        }
        this._modalElement.appendChild(body);
        // Removed TB.ui.processDynamicContent(body); - ensure this is called if needed after content insertion by the caller or here.

        // Modal Footer (optional, for buttons)
        if (this.options.buttons.length > 0) {
            const footer = document.createElement('div');
            // **MODIFIED FOOTER CLASSES:** Added padding. `flex-shrink-0`
            footer.className = `mt-auto p-4 md:p-6 flex flex-wrap justify-end gap-3 border-t border-neutral-300/50 dark:border-neutral-700/50 flex-shrink-0 ${this.options.customClasses.footer}`;
            this.options.buttons.forEach(btnConfig => {
                const button = TB.ui.Button.create(btnConfig.text, (() => btnConfig.action(this)) || (() => this.close()), {
                    variant: btnConfig.variant || 'primary',
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
        // Ensure click is directly on overlay, not on modal content bubbling up
        if (event.target === this._overlayElement && this.options.closeOnOutsideClick) {
            this.close();
        }
    }

    show() {
        if (this.isOpen) return;
        if (!this._overlayElement) { // Create DOM only if it doesn't exist
            this._createDom();
        } else {
            // If reusing, ensure it's appended again if somehow removed without proper close
            if (!this._overlayElement.parentNode) {
                document.body.appendChild(this._overlayElement);
            }
        }


        // Ensure transitions apply by forcing reflow AFTER elements are in DOM
        // and BEFORE changing opacity/transform
        requestAnimationFrame(() => {
            requestAnimationFrame(() => { // Double requestAnimationFrame for some browsers
                this._overlayElement.style.opacity = '1';
                this._modalElement.style.opacity = '1';
                this._modalElement.style.transform = 'scale(1)';
            });
        });


        document.addEventListener('keydown', this._boundHandleKeydown);
        this.isOpen = true;
        TB.logger.log(`[Modal] Shown: #${this.id}`);
        TB.events.emit('modal:shown', this);
        if (this.options.onOpen) this.options.onOpen(this);

        const firstFocusable = this._modalElement.querySelector(
            'button, [href], input:not([type="hidden"]), select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (firstFocusable) firstFocusable.focus();
    }

    close(triggerEvent = true) {
        if (!this.isOpen || !this._overlayElement) return; // Check _overlayElement existence

        if (triggerEvent && this.options.beforeClose) {
            if (this.options.beforeClose(this) === false) {
                TB.logger.log(`[Modal] Close prevented by beforeClose: #${this.id}`);
                return;
            }
        }

        this._overlayElement.style.opacity = '0';
        this._modalElement.style.opacity = '0';
        this._modalElement.style.transform = 'scale(0.95)';

        document.removeEventListener('keydown', this._boundHandleKeydown);

        // Use the transitionend event for more robust cleanup
        const onTransitionEnd = () => {
            this._modalElement.removeEventListener('transitionend', onTransitionEnd);
            if (this._overlayElement && this._overlayElement.parentNode) {
                this._overlayElement.parentNode.removeChild(this._overlayElement);
            }
            // Nullify elements only after removal, not if we plan to reuse the DOM structure later.
            // For full cleanup if not reusing:
            // this._overlayElement = null;
            // this._modalElement = null;
            this.isOpen = false; // Set isOpen to false here
            if (triggerEvent) {
                 TB.logger.log(`[Modal] Closed: #${this.id}`);
                TB.events.emit('modal:closed', this);
                if (this.options.onClose) this.options.onClose(this);
            }
        };

        // Fallback if transitionend doesn't fire (e.g., element removed before transition finishes)
        const fallbackTimeout = setTimeout(onTransitionEnd, 350); // Slightly longer than transition

        this._modalElement.addEventListener('transitionend', () => {
            clearTimeout(fallbackTimeout);
            onTransitionEnd();
        });

        // If there's no transition (e.g. duration 0), call cleanup immediately
        if (getComputedStyle(this._modalElement).transitionDuration === '0s') {
            clearTimeout(fallbackTimeout);
            onTransitionEnd();
        } else {
            this.isOpen = false; // Set immediately for logic, actual removal on transition end.
        }
    }

    // Static method for convenience
    static show(options) {
        const modalInstance = new Modal(options);
        modalInstance.show();
        return modalInstance;
    }
}

export default Modal;
