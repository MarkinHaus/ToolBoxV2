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
    static _instancesById = {}; // Static cache for getById

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

        this._elementIdForCache = null; // Used by getById to track the original element ID
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
        // Set aria-labelledby if a title will be present
        if (this.options.title) {
            this._modalElement.setAttribute('aria-labelledby', `${this.id}-title`);
        }


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
            max-h-[90vh] /* Content body will handle its own scroll */
            flex flex-col /* To make header/body/footer behave well */
            ${this.options.customClasses.modalContainer}
        `.trim().replace(/\s+/g, ' ');


        // Modal Header (optional)
        if (this.options.title || this.options.closeButton !== false) {
            const header = document.createElement('div');
            header.className = `flex justify-between items-center p-4 md:p-6 border-b border-neutral-300/50 dark:border-neutral-700/50 flex-shrink-0 ${this.options.customClasses.header}`.trim().replace(/\s+/g, ' ');

            if (this.options.title) {
                const titleEl = document.createElement('h3');
                titleEl.id = `${this.id}-title`; // For aria-labelledby
                titleEl.className = `text-lg font-semibold ${this.options.customClasses.title}`.trim().replace(/\s+/g, ' ');
                titleEl.textContent = this.options.title;
                header.appendChild(titleEl);
            }

            if (this.options.closeButton !== false) {
                const closeBtn = document.createElement('button');
                closeBtn.type = 'button'; // Good practice
                closeBtn.className = `material-symbols-outlined p-1 rounded text-neutral-600 dark:text-neutral-400 hover:bg-neutral-500/20 dark:hover:bg-neutral-300/20 transition ${this.options.customClasses.closeButton}`.trim().replace(/\s+/g, ' ');
                closeBtn.innerHTML = 'close';
                // closeBtn.style.scale = 1.5; // Prefer Tailwind/CSS for styling
                closeBtn.setAttribute('aria-label', 'Close modal');
                closeBtn.onclick = () => this.close();
                header.appendChild(closeBtn);
            }
            this._modalElement.appendChild(header);
        }

        // Modal Body
        const body = document.createElement('div');
        // Apply scroll to body, not main container, for fixed header/footer effect
        body.className = `p-4 md:p-6 flex-grow overflow-y-auto ${this.options.customClasses.body}`.trim().replace(/\s+/g, ' ');

        if (typeof this.options.content === 'string') {
            body.innerHTML = this.options.content;
        } else if (this.options.content instanceof HTMLElement) {
            const contentElement = this.options.content;

            // --- FIX: Ensure content element is visible ---
            // If the provided content element had an inline style of 'display: none',
            // clear it so the element becomes visible when appended to the modal.
            if (contentElement.style.display === 'none') {
                contentElement.style.display = ''; // This will revert to its default display (e.g., 'block') or CSS-defined display.
            }
            // --- END FIX ---

            body.appendChild(contentElement);
        }
        this._modalElement.appendChild(body);

        // Modal Footer (optional, for buttons)
        if (this.options.buttons && this.options.buttons.length > 0) {
            const footer = document.createElement('div');
            footer.className = `mt-auto p-4 md:p-6 flex flex-wrap justify-end gap-3 border-t border-neutral-300/50 dark:border-neutral-700/50 flex-shrink-0 ${this.options.customClasses.footer}`.trim().replace(/\s+/g, ' ');
            this.options.buttons.forEach(btnConfig => {
                const actionFn = typeof btnConfig.action === 'function' ? () => btnConfig.action(this) : () => this.close();
                const button = TB.ui.Button.create(btnConfig.text, actionFn, {
                    variant: btnConfig.variant || 'primary',
                    customClasses: btnConfig.className || '',
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

        document.body.classList.add('tb-modal-open'); // Optional: for global styling e.g. no body scroll

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
        if (typeof this.options.onOpen === 'function') this.options.onOpen(this);

        const firstFocusable = this._modalElement.querySelector(
            'button, [href], input:not([type="hidden"]), select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (firstFocusable) firstFocusable.focus();
    }

    close(triggerEvent = true) {
        if (!this.isOpen || !this._overlayElement) return;

        if (triggerEvent && typeof this.options.beforeClose === 'function') {
            if (this.options.beforeClose(this) === false) {
                TB.logger.log(`[Modal] Close prevented by beforeClose: #${this.id}`);
                return;
            }
        }

        this._overlayElement.style.opacity = '0';
        this._modalElement.style.opacity = '0';
        this._modalElement.style.transform = 'scale(0.95)';

        document.removeEventListener('keydown', this._boundHandleKeydown);
        document.body.classList.remove('tb-modal-open');


        let transitionEnded = false;
        const onTransitionEndHandler = () => {
            if (transitionEnded) return;
            transitionEnded = true;

            this._modalElement.removeEventListener('transitionend', onTransitionEndHandler);
            clearTimeout(fallbackTimeout);

            if (this._overlayElement && this._overlayElement.parentNode) {
                this._overlayElement.parentNode.removeChild(this._overlayElement);
            }
            this.isOpen = false;
            if (triggerEvent) {
                 TB.logger.log(`[Modal] Closed: #${this.id}`);
                TB.events.emit('modal:closed', this);
                if (typeof this.options.onClose === 'function') this.options.onClose(this);
            }
        };

        const modalStyle = getComputedStyle(this._modalElement);
        const transitionDuration = parseFloat(modalStyle.transitionDuration) * 1000;
        const fallbackTimeout = setTimeout(onTransitionEndHandler, transitionDuration > 0 ? transitionDuration + 50 : 50); // 50ms buffer

        if (transitionDuration > 0) {
            this._modalElement.addEventListener('transitionend', onTransitionEndHandler);
        } else {
            // No transition, execute immediately
            clearTimeout(fallbackTimeout); // Should already be cleared if transitionDuration is 0, but for safety
            onTransitionEndHandler();
        }
    }

    destroy() {
        this.close(false);

        if (this._overlayElement && this._overlayElement.parentNode) {
            this._overlayElement.parentNode.removeChild(this._overlayElement);
        }
        this._overlayElement = null;
        this._modalElement = null;

        this.isOpen = false;

        if (this._elementIdForCache && Modal._instancesById[this._elementIdForCache] === this) {
            delete Modal._instancesById[this._elementIdForCache];
            TB.logger.debug(`[Modal.destroy] Removed instance for element ID "${this._elementIdForCache}" from cache.`);
        }
        this._elementIdForCache = null;

        TB.logger.log(`[Modal] Destroyed: #${this.id}`);
        TB.events.emit('modal:destroyed', this);
    }

    // Static method for convenience
    static show(options) {
        const modalInstance = new Modal(options);
        modalInstance.show();
        return modalInstance;
    }

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
        ...extraModalOptions
    } = {}) {
        if (title === undefined || content === undefined) {
            TB.logger.error('[Modal.confirm] Options "title" and "content" are required.');
            return Promise.resolve(false);
        }

        return new Promise((resolve) => {
            const buttons = [];

            if (!hideCancelButton) {
                buttons.push({
                    text: cancelButtonText,
                    action: (modal) => {
                        modal.close(false);
                        resolve(false);
                    },
                    variant: cancelButtonVariant,
                    className: cancelButtonClass,
                });
            }

            buttons.push({
                text: confirmButtonText,
                action: (modal) => {
                    modal.close(false);
                    resolve(true);
                },
                variant: confirmButtonVariant,
                className: confirmButtonClass,
            });

            const confirmDefaults = {
                maxWidth: 'max-w-md',
                closeOnEsc: true,
                closeOnOutsideClick: true,
            };

            const modalSettings = {
                ...confirmDefaults,
                ...extraModalOptions,
                title: title,
                content: content,
                buttons: buttons,
                onClose: () => {
                    resolve(resolveOnClose);
                },
            };

            const modalInstance = new Modal(modalSettings);
            modalInstance.show();
        });
    }

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
            const inputId = TB.utils.uniqueId('tb-prompt-input-');
            const inputContainer = document.createElement('div');

            const label = document.createElement('label');
            label.htmlFor = inputId;
            label.textContent = title;
            label.className = 'sr-only';
            inputContainer.appendChild(label);

            const input = document.createElement('input');
            input.id = inputId;
            input.type = type;
            input.value = defaultValue;
            input.placeholder = placeholder;
            input.className = 'w-full border p-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-neutral-700 dark:border-neutral-600 dark:text-white';
            inputContainer.appendChild(input);

            let modalInstance;

            const confirmAction = () => {
                if (modalInstance) modalInstance.close(false);
                resolve(input.value);
            };

            const cancelAction = () => {
                if (modalInstance) modalInstance.close(false);
                resolve(null);
            };

            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    confirmAction();
                }
            });

            const promptDefaults = {
                maxWidth: 'max-w-md',
                closeOnEsc: true,
                closeOnOutsideClick: true,
            };

            modalInstance = new Modal({
                ...promptDefaults,
                ...extraModalOptions,
                title: title,
                content: inputContainer,
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
                onClose: () => {
                    resolve(resolveOnClose ? input.value : null);
                },
            });

            const originalOnOpen = modalInstance.options.onOpen;
            modalInstance.options.onOpen = (modal) => {
                input.focus();
                if (defaultValue) input.select();
                if (typeof originalOnOpen === 'function') originalOnOpen(modal);
            };

            modalInstance.show();

            // Additional focus logic in case onOpen doesn't cover all scenarios or fires too early/late
            if (modalInstance.isOpen && (!document.activeElement || !inputContainer.contains(document.activeElement))) {
                input.focus();
                if (defaultValue) input.select();
            } else if (!modalInstance.isOpen) { // Fallback if show() didn't immediately set isOpen and call onOpen
                 setTimeout(() => {
                    if (modalInstance.isOpen && (!document.activeElement || !inputContainer.contains(document.activeElement))) {
                       input.focus();
                       if (defaultValue) input.select();
                    }
                 }, 50); // Small delay for safety
            }
        });
    }

    static getById(elementId, options = {}) {
        if (typeof elementId !== 'string' || !elementId.trim()) {
            TB.logger.error('[Modal.getById] elementId must be a non-empty string.');
            return null;
        }

        if (Modal._instancesById[elementId]) {
            const existingModal = Modal._instancesById[elementId];
            TB.logger.debug(`[Modal.getById] Returning existing instance for element ID "${elementId}" (Modal ID: #${existingModal.id}).`);
            // Note: Options are not re-applied to existing instances by default.
            // If re-configuration is needed, existingModal.options could be updated,
            // but that might require re-rendering parts or all of the modal.
            return existingModal;
        }

        const contentElement = document.getElementById(elementId);

        if (!contentElement) {
            TB.logger.warn(`[Modal.getById] Element with ID "${elementId}" not found.`);
            return null;
        }

        const modalOptions = {
            ...options,
            content: contentElement,
        };

        // Infer title from content element if not provided in options
        if (!modalOptions.title) {
            if (contentElement.hasAttribute('aria-label')) {
                modalOptions.title = contentElement.getAttribute('aria-label');
            } else if (contentElement.hasAttribute('title') && !modalOptions.title) { // Check again if already set by aria-label
                modalOptions.title = contentElement.getAttribute('title');
            }
        }

        const newModal = new Modal(modalOptions);
        newModal._elementIdForCache = elementId;

        Modal._instancesById[elementId] = newModal;
        TB.logger.log(`[Modal.getById] Created new modal instance (ID: #${newModal.id}) for element ID "${elementId}".`);

        return newModal;
    }
}

export default Modal;
