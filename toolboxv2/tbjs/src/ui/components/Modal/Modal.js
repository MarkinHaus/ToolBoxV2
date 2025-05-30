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
            header.className = `flex justify-between items-center p-4 md:p-6 border-b border-neutral-300/50 dark:border-neutral-700/50 flex-shrink-0 ${this.options.customClasses.header}`;

            if (this.options.title) {
                const titleEl = document.createElement('h3');
                // titleEl.id = `${this.id}-title`;
                titleEl.className = `text-lg font-semibold ${this.options.customClasses.title}`;
                titleEl.textContent = this.options.title;
                header.appendChild(titleEl);
            }

            if (this.options.closeButton !== false) { // Check if closeButton is explicitly false
                const closeBtn = document.createElement('button');
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
        body.className = `p-4 md:p-6 flex-grow overflow-y-auto ${this.options.customClasses.body}`;
        if (typeof this.options.content === 'string') {
            body.innerHTML = this.options.content;
        } else if (this.options.content instanceof HTMLElement) {
            body.appendChild(this.options.content);
        }
        this._modalElement.appendChild(body);

        // Modal Footer (optional, for buttons)
        if (this.options.buttons && this.options.buttons.length > 0) {
            const footer = document.createElement('div');
            footer.className = `mt-auto p-4 md:p-6 flex flex-wrap justify-end gap-3 border-t border-neutral-300/50 dark:border-neutral-700/50 flex-shrink-0 ${this.options.customClasses.footer}`;
            this.options.buttons.forEach(btnConfig => {
                const button = TB.ui.Button.create(btnConfig.text, (() => btnConfig.action(this)) || (() => this.close()), {
                    variant: btnConfig.variant || 'primary',
                    customClasses: btnConfig.className, // Pass as customClasses
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
        if (!this._overlayElement) {
            this._createDom();
        } else {
            if (!this._overlayElement.parentNode) {
                document.body.appendChild(this._overlayElement);
            }
        }

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
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
        if (!this.isOpen || !this._overlayElement) return;

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

        const onTransitionEnd = () => {
            this._modalElement.removeEventListener('transitionend', onTransitionEnd);
            if (this._overlayElement && this._overlayElement.parentNode) {
                this._overlayElement.parentNode.removeChild(this._overlayElement);
            }
            this.isOpen = false;
            if (triggerEvent) {
                 TB.logger.log(`[Modal] Closed: #${this.id}`);
                TB.events.emit('modal:closed', this);
                if (this.options.onClose) this.options.onClose(this);
            }
        };

        const fallbackTimeout = setTimeout(onTransitionEnd, 350);

        this._modalElement.addEventListener('transitionend', () => {
            clearTimeout(fallbackTimeout);
            onTransitionEnd();
        });

        if (getComputedStyle(this._modalElement).transitionDuration === '0s') {
            clearTimeout(fallbackTimeout);
            onTransitionEnd();
        } else {
             // this.isOpen = false; // Set isOpen to false in onTransitionEnd to ensure it's false only after full close.
                                  // However, for logical operations, it might be better to set it earlier.
                                  // Current behavior: set to false in onTransitionEnd. This is generally fine.
        }
    }

    // Static method for convenience
    static show(options) {
        const modalInstance = new Modal(options);
        modalInstance.show();
        return modalInstance;
    }

    /**
     * Shows a confirmation modal and returns a Promise that resolves to true or false.
     * @param {object} options - Configuration for the confirmation modal.
     * @param {string} options.title - The title of the confirmation modal.
     * @param {string|HTMLElement} options.content - The content/message of the confirmation modal.
     * @param {string} [options.confirmButtonText='OK'] - Text for the confirm button.
     * @param {string} [options.cancelButtonText='Cancel'] - Text for the cancel button.
     * @param {string} [options.confirmButtonVariant='primary'] - Variant for the confirm button (e.g., 'primary', 'danger').
     * @param {string} [options.cancelButtonVariant='secondary'] - Variant for the cancel button (e.g., 'secondary', 'outline').
     * @param {string} [options.confirmButtonClass=''] - Additional CSS classes for the confirm button.
     * @param {string} [options.cancelButtonClass=''] - Additional CSS classes for the cancel button.
     * @param {boolean} [options.hideCancelButton=false] - If true, the cancel button will not be shown.
     * @param {boolean} [options.resolveOnClose=false] - Value the promise resolves to if closed by ESC, 'X' button, or outside click.
     * @param {object} [options.extraModalOptions={}] - Pass any other valid Modal options (e.g., maxWidth, customClasses, closeButton: false).
     * @returns {Promise<boolean>} A promise that resolves to true if confirmed, false otherwise.
     */
    static async confirm({
        title,
        content,
        confirmButtonText = 'OK',
        cancelButtonText = 'Cancel',
        confirmButtonVariant = 'primary',
        cancelButtonVariant = 'secondary',
        confirmButtonClass = '',
        cancelButtonClass = '',
        hideCancelButton = false,
        resolveOnClose = false,
        ...extraModalOptions // Collects any other options passed to confirm
    } = {}) {
        if (title === undefined || content === undefined) {
            TB.logger.error('[Modal.confirm] Options "title" and "content" are required.');
            return Promise.resolve(false); // Or throw new Error('Modal.confirm requires title and content');
        }

        return new Promise((resolve) => {
            const buttons = [];

            if (!hideCancelButton) {
                buttons.push({
                    text: cancelButtonText,
                    action: (modal) => {
                        modal.close(false); // Pass false to prevent this modal's onClose from triggering
                        resolve(false);
                    },
                    variant: cancelButtonVariant,
                    className: cancelButtonClass,
                });
            }

            buttons.push({
                text: confirmButtonText,
                action: (modal) => {
                    modal.close(false); // Pass false to prevent this modal's onClose from triggering
                    resolve(true);
                },
                variant: confirmButtonVariant,
                className: confirmButtonClass,
            });

            // Define default modal settings specific to confirm dialogs
            const confirmDefaults = {
                maxWidth: 'max-w-md', // Confirm dialogs are often a bit smaller
                closeOnEsc: true,     // Usually, Esc should cancel
                closeOnOutsideClick: true, // Usually, clicking outside should cancel
            };

            // Merge options: confirmDefaults < extraModalOptions < coreConfirmLogic
            const modalSettings = {
                ...confirmDefaults,
                ...extraModalOptions, // User-supplied modal options can override confirmDefaults
                // Core properties for the confirm modal, these override anything from extraModalOptions
                title: title,
                content: content,
                buttons: buttons,
                onClose: () => {
                    // This onClose is specifically for the promise resolution.
                    // It's called if the modal is closed by ESC, 'X' button, or programmatically
                    // via modal.close() or modal.close(true), because our explicit buttons
                    // call modal.close(false) which bypasses this modal's onClose callback.
                    resolve(resolveOnClose);
                },
            };

            const modalInstance = new Modal(modalSettings);
            modalInstance.show();
        });
    }

/**
 * Shows a prompt modal with an input field and returns a Promise that resolves to the input string or null.
 * @param {object} options - Configuration for the prompt modal.
 * @param {string} options.title - The title of the prompt modal.
 * @param {string} [options.placeholder=''] - Placeholder text for the input field.
 * @param {string} [options.defaultValue=''] - Default value in the input field.
 * @param {string} [options.type='text'] - Type of the input (e.g., 'text', 'password', 'email').
 * @param {string} [options.confirmButtonText='OK'] - Text for the confirm button.
 * @param {string} [options.cancelButtonText='Cancel'] - Text for the cancel button.
 * @param {string} [options.confirmButtonVariant='primary'] - Variant for the confirm button.
 * @param {string} [options.cancelButtonVariant='secondary'] - Variant for the cancel button.
 * @param {string} [options.confirmButtonClass=''] - Additional CSS classes for the confirm button.
 * @param {string} [options.cancelButtonClass=''] - Additional CSS classes for the cancel button.
 * @param {boolean} [options.resolveOnClose=false] - If true, resolve with current input value on ESC or outside click. Otherwise, resolve null.
 * @param {object} [options.extraModalOptions={}] - Additional modal options.
 * @returns {Promise<string|null>} A promise that resolves to the entered input or null if cancelled.
 */
static async prompt({
    title,
    placeholder = '',
    defaultValue = '',
    type = 'text',
    confirmButtonText = 'OK',
    cancelButtonText = 'Cancel',
    confirmButtonVariant = 'primary',
    cancelButtonVariant = 'secondary',
    confirmButtonClass = '',
    cancelButtonClass = '',
    resolveOnClose = false,
    ...extraModalOptions
} = {}) {
    if (!title) {
        TB.logger.error('[Modal.prompt] Option "title" is required.');
        return Promise.resolve(null);
    }

    return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = type;
        input.value = defaultValue;
        input.placeholder = placeholder;
        input.className = 'w-full border p-2 rounded focus:outline-none focus:ring focus:border-blue-300';

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                confirmAction();
            }
        });

        const confirmAction = () => {
            modal.close(false);
            resolve(input.value);
        };

        const cancelAction = () => {
            modal.close(false);
            resolve(null);
        };

        const modal = new Modal({
            title,
            content: input,
            buttons: [
                {
                    text: cancelButtonText,
                    action: cancelAction,
                    variant: cancelButtonVariant,
                    className: cancelButtonClass,
                },
                {
                    text: confirmButtonText,
                    action: confirmAction,
                    variant: confirmButtonVariant,
                    className: confirmButtonClass,
                },
            ],
            maxWidth: 'max-w-md',
            closeOnEsc: true,
            closeOnOutsideClick: true,
            onClose: () => resolve(resolveOnClose ? input.value : null),
            ...extraModalOptions,
        });

        modal.show();

        // Focus the input after the modal opens
        setTimeout(() => input.focus(), 50);
    });
}

}

export default Modal;
