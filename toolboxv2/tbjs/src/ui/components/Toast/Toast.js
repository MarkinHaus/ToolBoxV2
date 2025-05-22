// tbjs/ui/components/Toast/Toast.js
import TB from '../../../index.js';
let activeToasts = new Map();
const DEFAULT_TOAST_OPTIONS = {
    message: '',
    type: 'info', // 'info', 'success', 'warning', 'error'
    duration: 15000, // milliseconds, 0 for sticky
    position: 'top-right', // 'top-right', 'top-center', 'top-left', 'bottom-right', ...
    title: '', // Optional title, will be used as the "from" label
    actions: [], // [{ text: 'Undo', action: () => {} }]
    customClasses: {
        container: '', // For the individual toast
        title: '',
        message: '',
        actionButton: '',
        progressBar: '', // Custom class for the progress bar itself
        toastHost: '', // For the main container holding all toasts
    },
    icon: true, // Show default icon based on type (less prominent in balloon style)
    closable: true,
    showDotOnHide: true, // New option to control dot behavior
    dotDuration: 10000, // How long the dot stays visible (ms)
};

let toastContainers = {}; // Store containers by position

class Toast {
    constructor(options = {}) {
        this.options = { ...DEFAULT_TOAST_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_TOAST_OPTIONS.customClasses, ...options.customClasses };
        this.id = TB.utils.uniqueId('tb-toast-');
        this._toastElement = null;
        this._timeoutId = null;
        this._progressIntervalId = null;
        this._progressBarElement = null;
        this._dotElement = null; // To manage the dot
        this._dotTimeoutId = null; // For auto-removing the dot
        this._isPermanentlyHidden = false; // Flag to prevent dot re-creation
    }

    _getToastContainer() {
        const positionClass = this._getPositionClass();
        const containerId = `tb-toast-container-${this.options.position.replace(/-/g, '')}`;

        if (!toastContainers[this.options.position]) {
            let container = document.getElementById(containerId);
            if (!container) {
                container = document.createElement('div');
                container.id = containerId;
                // container.classList.add("tb-toast-container-mini-button"); // This was being overwritten
                // Tailwind: fixed z-[1100] flex flex-col items-end p-4 gap-3 (for top-right)
                // Adjust alignment based on position
                let alignmentClasses = 'items-end'; // Default for right positions
                if (this.options.position.includes('left')) alignmentClasses = 'items-start';
                if (this.options.position.includes('center')) alignmentClasses = 'items-center';

                container.className = `tb-fixed tb-z-[1100] tb-flex tb-flex-col tb-p-4 tb-gap-3 ${positionClass} ${alignmentClasses} ${this.options.customClasses.toastHost || ''}`;
                document.body.appendChild(container);
            }
            toastContainers[this.options.position] = container;
        }
        return toastContainers[this.options.position];
    }

    _getPositionClass() {
        // These classes position the container itself
        switch (this.options.position) {
            case 'top-right': return 'tb-top-0 tb-right-0';
            case 'top-center': return 'tb-top-0 tb-left-1/2 tb--translate-x-1/2';
            case 'top-left': return 'tb-top-0 tb-left-0';
            case 'bottom-right': return 'tb-bottom-0 tb-right-0';
            case 'bottom-center': return 'tb-bottom-0 tb-left-1/2 tb--translate-x-1/2';
            case 'bottom-left': return 'tb-bottom-0 tb-left-0';
            default: return 'tb-top-0 tb-right-0';
        }
    }

    _getAppearanceStyles() {
        let baseWrapperClasses = `tb-relative tb-flex tb-flex-col tb-items-center tb-bg-background-color/60 dark:tb-bg-background-color/60 tb-backdrop-blur-md tb-rounded-lg tb-p-5 tb-shadow-xl tb-min-w-[250px] tb-max-w-xs md:tb-max-w-sm`;
        baseWrapperClasses += ' tb-speech-balloon-toast';


        let borderColorClass = 'tb-border-gray-400 dark:tb-border-gray-600';
        let accentColorClass = 'tb-text-blue-500 dark:tb-text-blue-400';
        let fromLabel = this.options.title || 'Notification';
        let dotColorClass = 'tb-toast-dot-info';


        switch (this.options.type) {
            case 'success':
                borderColorClass = 'tb-border-green-500 dark:tb-border-green-400';
                accentColorClass = 'tb-text-green-500 dark:tb-text-green-400';
                dotColorClass = 'tb-toast-dot-success';
                if (!this.options.title) fromLabel = 'Success';
                break;
            case 'warning':
                borderColorClass = 'tb-border-yellow-500 dark:tb-border-yellow-400';
                accentColorClass = 'tb-text-yellow-500 dark:tb-text-yellow-400';
                dotColorClass = 'tb-toast-dot-warning';
                if (!this.options.title) fromLabel = 'Warning';
                break;
            case 'error':
                borderColorClass = 'tb-border-red-500 dark:tb-border-red-400';
                accentColorClass = 'tb-text-red-500 dark:tb-text-red-400';
                dotColorClass = 'tb-toast-dot-error';
                if (!this.options.title) fromLabel = 'Error';
                break;
            case 'info':
            default:
                borderColorClass = 'tb-border-blue-500 dark:tb-border-blue-400'; // Default to blue for info if not gray
                accentColorClass = 'tb-text-blue-500 dark:tb-text-blue-400';
                dotColorClass = 'tb-toast-dot-info';
                if (!this.options.title) fromLabel = 'Info';
                break;
        }
        baseWrapperClasses += ` ${borderColorClass.split(' ')[0]}`; // Add only the light mode border for now or handle dark mode separately

        return {
            wrapper: baseWrapperClasses,
            accentColorClass: accentColorClass,
            fromLabel: fromLabel,
            borderColorClass: borderColorClass.split(' ')[0], // Use light mode for tail color logic
            dotColorClass: dotColorClass,
        };
    }

    _createDom() {
        const container = this._getToastContainer();
        this._toastElement = document.createElement('div');
        this._toastElement.id = this.id;


        const { wrapper: wrapperClasses, accentColorClass, fromLabel, borderColorClass, dotColorClass } = this._getAppearanceStyles();

        const initialTransform = this.options.position.includes('bottom-') ? 'tb-translate-y-2' : 'tb--translate-y-2';
        this._toastElement.className = `tb-opacity-0 ${initialTransform} tb-transition-all tb-duration-300 tb-ease-out ${wrapperClasses} ${this.options.customClasses.container}`
        let contentHtml = '';

        let bgColor = '';
        // Determine background color for the "from" label based on type
        switch (this.options.type) {
            case 'success': bgColor = TB.ui.theme?.tailwindResolvedColors?.['green-500'] || '#22c55e'; break;
            case 'warning': bgColor = TB.ui.theme?.tailwindResolvedColors?.['yellow-500'] || '#eab308'; break;
            case 'error': bgColor = TB.ui.theme?.tailwindResolvedColors?.['red-500'] || '#ef4444'; break;
            case 'info':
            default: bgColor = TB.ui.theme?.tailwindResolvedColors?.['blue-500'] || '#3b82f6'; break;
        }


        contentHtml += `
          <div
            class="tb-speech-balloon-from tb-absolute tb--top-2.5 tb-left-1/2 tb--translate-x-1/2 tb-px-2 tb-py-0.5 tb-text-xs tb-text-white tb-font-semibold tb-rounded tb-shadow ${this.options.customClasses.title}"
            style="background-color: ${bgColor};"
          >
            ${fromLabel}
          </div>
        `;
        if (this.options.closable) {
            contentHtml += `<button class="tb-speech-balloon-close-button tb-absolute tb-top-1 tb-right-1 tb-p-1 tb-text-text-color/70 hover:tb-text-text-color tb-rounded-full" data-close-btn="true">
                                <span class="material-symbols-outlined tb-text-base">close</span>
                           </button>`;
        }

        contentHtml += `<div class="tb-speech-balloon-content tb-w-full tb-text-sm tb-mt-2 ${this.options.customClasses.message}">`;
        if (this.options.icon) {
            // Icon HTML was commented out, uncommenting if it's intended. For tests, it doesn't matter now.
            let iconName = 'info';
            if (this.options.type === 'success') iconName = 'check_circle';
            if (this.options.type === 'warning') iconName = 'warning';
            if (this.options.type === 'error') iconName = 'error';
            contentHtml += `<span class="material-symbols-outlined tb-mr-2 tb-align-middle ${accentColorClass}">${iconName}</span>`;
        }
        contentHtml += this.options.message;
        contentHtml += `</div>`;

        if (this.options.actions.length > 0) {
            contentHtml += '<div class="tb-speech-balloon-options tb-mt-3 tb-flex tb-flex-wrap tb-gap-2 tb-justify-center tb-w-full">';
            this.options.actions.forEach((action, idx) => {
                contentHtml += `<button class="tb-px-3 tb-py-1 tb-text-xs tb-font-medium tb-rounded tb-border tb-border-current hover:tb-bg-text-color/10 ${accentColorClass} ${this.options.customClasses.actionButton}" data-action-idx="${idx}">${action.text}</button>`;
            });
            contentHtml += '</div>';
        }

        if (this.options.duration > 0) {
            contentHtml += `<div class="tb-speech-balloon-progress-bar-container tb-w-full tb-h-1.5 tb-bg-text-color/10 tb-mt-4 tb-rounded-full tb-overflow-hidden">
                                <div class="tb-speech-balloon-progress-bar tb-h-full tb-rounded-full ${accentColorClass.replace('tb-text-', 'tb-bg-')} ${this.options.customClasses.progressBar}" style="width: 100%;"></div>
                           </div>`;
        }
        contentHtml += `<div class="tb-speech-balloon-tail"></div>`;


        this._toastElement.innerHTML = contentHtml;
        this._progressBarElement = this._toastElement.querySelector('.tb-speech-balloon-progress-bar');

        this._toastElement.querySelectorAll('[data-action-idx]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.dataset.actionIdx);
                this.options.actions[idx].action();
                this.hide(false); // Typically actions mean the toast is done, no dot.
            });
        });

        const tail = this._toastElement.querySelector('.tb-speech-balloon-tail');
        if (tail) {
            const colorName = borderColorClass.split('-').slice(2).join('-'); // e.g., green-500
            const resolvedColor = TB.ui.theme?.tailwindResolvedColors?.[colorName] || 'currentColor';

            if (this.options.position.startsWith('top-')) {
                this._toastElement.classList.add('tb-toast-position-top');
                tail.style.borderBottomColor = resolvedColor;
            } else {
                this._toastElement.classList.add('tb-toast-position-bottom');
                tail.style.borderTopColor = resolvedColor;
            }
        }

        const closeBtn = this._toastElement.querySelector('[data-close-btn]');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide(this.options.showDotOnHide)); // Respect showDotOnHide for manual close
        }

        const toastContainerElement = this._getToastContainer();
        if (this.options.position.startsWith('top-')) {
            toastContainerElement.appendChild(this._toastElement);
        } else {
            toastContainerElement.prepend(this._toastElement);
        }
        activeToasts.set(this.id, this); // Add to map after it's part of DOM creation logic
    }

    _startProgressBar() {
        if (!this._progressBarElement || this.options.duration <= 0) return;
        const startTime = Date.now();
        const duration = this.options.duration;
        this._progressIntervalId = setInterval(() => {
            const elapsedTime = Date.now() - startTime;
            const progress = Math.max(0, 100 - (elapsedTime / duration) * 100);
            this._progressBarElement.style.width = `${progress}%`;
            if (progress === 0) {
                clearInterval(this._progressIntervalId);
                this._progressIntervalId = null;
            }
        }, 50);
    }

    _stopProgressBar() {
        if (this._progressIntervalId) {
            clearInterval(this._progressIntervalId);
            this._progressIntervalId = null;
        }
    }

    show() {
        if (this._isPermanentlyHidden) return;

        if (this._dotElement && this._dotElement.parentNode) {
            this._dotElement.parentNode.removeChild(this._dotElement);
            this._dotElement = null;
            if (this._dotTimeoutId) clearTimeout(this._dotTimeoutId);
        }

        if (!this._toastElement) {
             this._createDom();
        }
        this._toastElement.classList.remove('tb-hidden');
        this._toastElement.classList.remove('tb-toast-hiding');

        void this._toastElement.offsetWidth;

        this._toastElement.style.opacity = '1';
        this._toastElement.style.transform = 'translateY(0) translateX(0)'; // Ensure both transforms are reset

        if (this.options.duration > 0) {
            this._startProgressBar();
            this._timeoutId = setTimeout(() => this.hide(this.options.showDotOnHide), this.options.duration);
        }

        TB.logger.log(`[Toast] Shown: ${this.options.message.substring(0, 30)}...`);
        TB.events.emit('toast:shown', this);
    }

    hide(showDot = this.options.showDotOnHide) {
        if (!this._toastElement || this._toastElement.classList.contains('tb-toast-hiding')) return;
        this._toastElement.classList.add('tb-toast-hiding');

        if (this._timeoutId) clearTimeout(this._timeoutId);
        this._stopProgressBar();

        this._toastElement.style.opacity = '0';
        const exitTransform = this.options.position.includes('bottom-') ? 'translateY(0.5rem)' : 'translateY(-0.5rem)';
        this._toastElement.style.transform = exitTransform;

        setTimeout(() => {
            if (this._toastElement) {
                this._toastElement.classList.add('tb-hidden');
                this._toastElement.classList.remove('tb-toast-hiding');
            }

            if (showDot && !this._isPermanentlyHidden) {
                this._createDot();
            } else {
                this._removeToastCompletely();
            }
        }, 300);
    }

    _createDot() {
        if (this._dotElement || !this._toastElement?.parentNode) return;

        const { dotColorClass } = this._getAppearanceStyles();
        this._dotElement = document.createElement('div');
        this._dotElement.className = `tb-toast-dot ${dotColorClass}`; // This is a general class, specific color comes from dotColorClass
        this._dotElement.title = `Re-open: ${this.options.title || this.options.message.substring(0,20) + '...'}`;

        this._dotElement.addEventListener('click', () => {
            if (this._dotTimeoutId) clearTimeout(this._dotTimeoutId);
            if (this._dotElement && this._dotElement.parentNode) {
                 this._dotElement.parentNode.removeChild(this._dotElement);
            }
            this._dotElement = null;
            this._isPermanentlyHidden = false;
            this.show();
        });

        const toastContainerElement = this._getToastContainer();
        if (this.options.position.startsWith('top-')) {
            toastContainerElement.appendChild(this._dotElement);
        } else {
            toastContainerElement.prepend(this._dotElement);
        }

        this._dotTimeoutId = setTimeout(() => {
            if (this._dotElement && this._dotElement.parentNode) {
                this._dotElement.parentNode.removeChild(this._dotElement);
            }
            this._dotElement = null;

            if (!this._toastElement || this._toastElement.classList.contains('tb-hidden')) { // Check if it wasn't re-shown
                this._isPermanentlyHidden = true;
                this._removeToastCompletely();
            }
        }, this.options.dotDuration);
    }

    _removeToastCompletely() {
        if (this._toastElement && this._toastElement.parentNode) {
            this._toastElement.parentNode.removeChild(this._toastElement);
        }
        activeToasts.delete(this.id);
        this._toastElement = null;

        const container = toastContainers[this.options.position];
        if (container && container.children.length === 0) {
            if (container.parentNode) container.parentNode.removeChild(container);
            delete toastContainers[this.options.position];
        }
        TB.logger.log(`[Toast] Permanently removed: ${this.options.message.substring(0, 30)}...`);
        TB.events.emit('toast:hidden', this);
    }

    static showInfo(message, options = {}) { new Toast({ ...options, message, type: 'info' }).show(); }
    static showSuccess(message, options = {}) { new Toast({ ...options, message, type: 'success' }).show(); }
    static showWarning(message, options = {}) { new Toast({ ...options, message, type: 'warning' }).show(); }
    static showError(message, options = {}) { new Toast({ ...options, message, type: 'error' }).show(); }

    static hideAll() {
        activeToasts.forEach(toast => {
            toast._isPermanentlyHidden = true;
            toast.hide(false);
        });
    }

    // For testing purposes to reset module-level state
    static __testonly__resetModuleState() {
        activeToasts = new Map();
        toastContainers = {};
    }
    // For testing purposes to inspect module-level state
    static __testonly__getActiveToasts() {
        return activeToasts;
    }
    static __testonly__getToastContainers() { // <--- ADD THIS
        return toastContainers;
    }
}

export default Toast;
