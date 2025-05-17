// tbjs/ui/components/Toast/Toast.js
import TB from '../../../index.js';

const DEFAULT_TOAST_OPTIONS = {
    message: '',
    type: 'info', // 'info', 'success', 'warning', 'error'
    duration: 5000, // milliseconds, 0 for sticky
    position: 'top-right', // 'top-right', 'top-center', 'top-left', 'bottom-right', ...
    title: '', // Optional title
    actions: [], // [{ text: 'Undo', action: () => {} }]
    customClasses: {
        container: '', // For the individual toast
        title: '',
        message: '',
        actionButton: '',
    },
    icon: true, // Show default icon based on type
    closable: true,
};

let toastContainer = null; // Single container for all toasts at a specific position
const activeToasts = new Map(); // To manage active toasts

class Toast {
    constructor(options = {}) {
        this.options = { ...DEFAULT_TOAST_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_TOAST_OPTIONS.customClasses, ...options.customClasses };
        this.id = TB.utils.uniqueId('tb-toast-');
        this._toastElement = null;
        this._timeoutId = null;
    }

    _getToastContainer() {
        const positionClass = this._getPositionClass();
        const containerId = `tb-toast-container-${this.options.position.replace('-', '')}`;
        if (!toastContainer || toastContainer.id !== containerId) {
             // Remove old container if position changed (simple approach)
            if(toastContainer && toastContainer.parentNode) toastContainer.parentNode.removeChild(toastContainer);

            toastContainer = document.getElementById(containerId);
            if (!toastContainer) {
                toastContainer = document.createElement('div');
                toastContainer.id = containerId;
                // /* Tailwind: fixed z-[1100] flex flex-col space-y-2 */
                // /* Add position classes based on this.options.position */
                toastContainer.className = `fixed z-[1100] flex flex-col gap-2 p-4 ${positionClass} ${this.options.customClasses.toastHost || ''}`;
                document.body.appendChild(toastContainer);
            }
        }
        return toastContainer;
    }

    _getPositionClass() {
        switch (this.options.position) {
            case 'top-right': return 'top-0 right-0';
            case 'top-center': return 'top-0 left-1/2 -translate-x-1/2';
            case 'top-left': return 'top-0 left-0';
            case 'bottom-right': return 'bottom-0 right-0';
            case 'bottom-center': return 'bottom-0 left-1/2 -translate-x-1/2';
            case 'bottom-left': return 'bottom-0 left-0';
            default: return 'top-0 right-0';
        }
    }

    _getTypeClassesAndIcon() {
        // /* Tailwind: Define base, then type-specific styles */
        // Base: bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-100 shadow-lg rounded-md
        const baseClasses = 'bg-background-color text-text-color shadow-lg rounded-md p-4 flex items-start gap-3';
        let typeClasses = '';
        let iconHtml = '';
        // Icons using Material Symbols Outlined
        switch (this.options.type) {
            case 'success':
                // /* Tailwind: border-l-4 border-green-500 */
                typeClasses = 'border-l-4 border-green-500';
                iconHtml = this.options.icon ? '<span class="material-symbols-outlined text-green-500">check_circle</span>' : '';
                break;
            case 'warning':
                typeClasses = 'border-l-4 border-yellow-500';
                iconHtml = this.options.icon ? '<span class="material-symbols-outlined text-yellow-500">warning</span>' : '';
                break;
            case 'error':
                typeClasses = 'border-l-4 border-red-500';
                iconHtml = this.options.icon ? '<span class="material-symbols-outlined text-red-500">error</span>' : '';
                break;
            case 'info':
            default:
                typeClasses = 'border-l-4 border-blue-500';
                iconHtml = this.options.icon ? '<span class="material-symbols-outlined text-blue-500">info</span>' : '';
                break;
        }
        return { wrapper: `${baseClasses} ${typeClasses}`, icon: iconHtml };
    }

    _createDom() {
        const container = this._getToastContainer();
        this._toastElement = document.createElement('div');
        this._toastElement.id = this.id;
        activeToasts.set(this.id, this);

        const { wrapper: wrapperClasses, icon: iconHtml } = this._getTypeClassesAndIcon();
        // /* Tailwind: min-w-[250px] max-w-md opacity-0 translate-y-2 transition-all duration-300 ease-out */
        this._toastElement.className = `min-w-[250px] max-w-xs opacity-0 transform transition-all duration-300 ease-out ${this.options.position.includes('bottom-') ? 'translate-y-2' : '-translate-y-2'} ${wrapperClasses} ${this.options.customClasses.container}`;

        let contentHtml = iconHtml ? `<div class="flex-shrink-0">${iconHtml}</div>` : '';
        contentHtml += '<div class="flex-grow">';

        if (this.options.title) {
            // /* Tailwind: font-semibold */
            contentHtml += `<h4 class="font-semibold ${this.options.customClasses.title}">${this.options.title}</h4>`;
        }
        // /* Tailwind: text-sm */
        contentHtml += `<p class="text-sm ${this.options.customClasses.message}">${this.options.message}</p>`;

        if (this.options.actions.length > 0) {
            // /* Tailwind: mt-2 space-x-2 */
            contentHtml += '<div class="mt-2 flex gap-2">';
            this.options.actions.forEach(action => {
                // /* Tailwind: text-xs font-medium text-blue-600 hover:text-blue-500 */
                contentHtml += `<button class="text-xs font-medium text-accent-color hover:underline ${this.options.customClasses.actionButton}" data-action-idx="${this.options.actions.indexOf(action)}">${action.text}</button>`;
            });
            contentHtml += '</div>';
        }
        contentHtml += '</div>'; // flex-grow

        if (this.options.closable) {
            // /* Tailwind: ml-auto flex-shrink-0 */
            contentHtml += `<div class="ml-auto flex-shrink-0"><button class="material-symbols-outlined text-gray-400 hover:text-gray-600 text-base p-1 -m-1 rounded" data-close-btn="true">close</button></div>`;
        }

        this._toastElement.innerHTML = contentHtml;

        // Add event listeners for actions and close
        this._toastElement.querySelectorAll('[data-action-idx]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.dataset.actionIdx);
                this.options.actions[idx].action();
                this.hide(); // Typically hide after action
            });
        });
        const closeBtn = this._toastElement.querySelector('[data-close-btn]');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }

        if (this.options.position.startsWith('top-')) {
            container.appendChild(this._toastElement); // Add to bottom for top positions
        } else {
            container.prepend(this._toastElement); // Add to top for bottom positions
        }
    }

    show() {
        this._createDom();

        // Force reflow for transition
        void this._toastElement.offsetWidth;

        this._toastElement.style.opacity = '1';
        this._toastElement.style.transform = 'translateY(0)';

        if (this.options.duration > 0) {
            this._timeoutId = setTimeout(() => this.hide(), this.options.duration);
        }

        TB.logger.log(`[Toast] Shown: ${this.options.message}`);
        TB.events.emit('toast:shown', this);
    }

    hide() {
        if (!this._toastElement) return;

        if (this._timeoutId) clearTimeout(this._timeoutId);

        this._toastElement.style.opacity = '0';
        this._toastElement.style.transform = this.options.position.includes('bottom-') ? 'translateY(0.5rem)' : 'translateY(-0.5rem)';
        // Add other exit animations if needed (e.g., slide out)

        setTimeout(() => {
            if (this._toastElement && this._toastElement.parentNode) {
                this._toastElement.parentNode.removeChild(this._toastElement);
            }
            activeToasts.delete(this.id);
            this._toastElement = null;

            // If container is empty, remove it
            if (toastContainer && toastContainer.children.length === 0) {
                toastContainer.parentNode.removeChild(toastContainer);
                toastContainer = null;
            }

            TB.logger.log(`[Toast] Hidden: ${this.options.message}`);
            TB.events.emit('toast:hidden', this);
        }, 300); // Match transition duration
    }

    // Static convenience methods
    static showInfo(message, options = {}) { new Toast({ ...options, message, type: 'info' }).show(); }
    static showSuccess(message, options = {}) { new Toast({ ...options, message, type: 'success' }).show(); }
    static showWarning(message, options = {}) { new Toast({ ...options, message, type: 'warning' }).show(); }
    static showError(message, options = {}) { new Toast({ ...options, message, type: 'error' }).show(); }
    static hideAll() { activeToasts.forEach(toast => toast.hide()); }
}

export default Toast;
