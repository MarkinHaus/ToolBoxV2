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
        // Note: toastHost for the main container holding all toasts is still available
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
        let baseWrapperClasses = `tb-relative tb-flex tb-flex-col tb-items-center tb-bg-background-color/60 dark:tb-bg-background-color/60 tb-backdrop-blur-md tb-rounded-lg tb-p-5 tb-shadow-xl tb-min-w-[250px] tb-max-w-xs md:tb-max-w-sm`; // Removed tb-border, will add specific below
        baseWrapperClasses += ' tb-speech-balloon-toast';


        let borderColorClass = 'tb-border-gray-400 dark:tb-border-gray-600';
        let accentColorClass = 'tb-text-blue-500 dark:tb-text-blue-400'; // For icon and progress bar
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
                accentColorClass = 'tb-text-blue-500 dark:tb-text-blue-400'; // Ensure this is set for info
                dotColorClass = 'tb-toast-dot-info';
                if (!this.options.title) fromLabel = 'Info';
                break;
        }
        baseWrapperClasses += ` ${borderColorClass}`; // Add border directly

        return {
            wrapper: baseWrapperClasses,
            accentColorClass: accentColorClass,
            fromLabel: fromLabel,
            borderColorClass: borderColorClass, // Pass this for the tail
            dotColorClass: dotColorClass,     // Pass this for the dot
        };
    }

    _createDom() {
        const container = this._getToastContainer();
        this._toastElement = document.createElement('div');
        this._toastElement.id = this.id;
        activeToasts.set(this.id, this);

        const { wrapper: wrapperClasses, accentColorClass, fromLabel, borderColorClass, dotColorClass } = this._getAppearanceStyles();

        // Tailwind: opacity-0 translate-y-2 transition-all duration-300 ease-out
        // Adjust translate based on position for entry animation
        const initialTransform = this.options.position.includes('bottom-') ? 'tb-translate-y-2' : 'tb--translate-y-2';
        this._toastElement.className = `tb-opacity-0 ${initialTransform} tb-transition-all tb-duration-300 tb-ease-out ${wrapperClasses} ${this.options.customClasses.container}`
        let contentHtml = '';

        // "From" label (title) - styled like speech_balloon-from
        let bgColor = '';

        switch (this.options.type) {
          case 'success':
            bgColor = '#22c55e';
            break;
          case 'warning':
            bgColor = '#eab308';
            break;
          case 'error':
            bgColor = '#ef4444';
            break;
          case 'info':
            bgColor = '#3b82f6';
            break;
          default:
            bgColor = '#e5e7eb'; // fallback (e.g., gray-200)
        }

        contentHtml += `
          <div
            class="tb-speech-balloon-from tb-absolute tb--top-2.5 tb-left-1/2 tb--translate-x-1/2 tb-px-2 tb-py-0.5 tb-text-xs tb-font-semibold tb-rounded tb-shadow ${this.options.customClasses.title}"
            style="background-color: ${bgColor};"
          >
            ${fromLabel}
          </div>
        `;
        // Close button
        if (this.options.closable) {
            contentHtml += `<button class="tb-speech-balloon-close-button tb-absolute tb-top-1 tb-right-1 tb-p-1 tb-text-text-color/70 hover:tb-text-text-color tb-rounded-full" data-close-btn="true">
                                <span class="material-symbols-outlined tb-text-base">close</span>
                           </button>`;
        }

        // Main content (icon + message)
        contentHtml += `</div><div class="tb-speech-balloon-content tb-w-full tb-text-sm tb-mt-2 ${this.options.customClasses.message}">`; // Added mt-2 for spacing from "from" label
        // Icon (optional and less prominent)
        if (this.options.icon) {
            let iconName = 'info';
            if (this.options.type === 'success') iconName = 'check_circle';
            if (this.options.type === 'warning') iconName = 'warning';
            if (this.options.type === 'error') iconName = 'error';
            // contentHtml += `<span class="material-symbols-outlined tb-mr-2 tb-align-middle ${accentColorClass}">${iconName}</span>`;
        }
        contentHtml += this.options.message; // Message directly
        contentHtml += `</div>`;

        // Actions
        if (this.options.actions.length > 0) {
            contentHtml += '<div class="tb-speech-balloon-options tb-mt-3 tb-flex tb-flex-wrap tb-gap-2 tb-justify-center tb-w-full">';
            this.options.actions.forEach((action, idx) => {
                contentHtml += `<button class="tb-px-3 tb-py-1 tb-text-xs tb-font-medium tb-rounded tb-border tb-border-current hover:tb-bg-text-color/10 ${accentColorClass} ${this.options.customClasses.actionButton}" data-action-idx="${idx}">${action.text}</button>`;
            });
            contentHtml += '</div>';
        }

        // Progress bar container & bar
        if (this.options.duration > 0) {
            contentHtml += `<div class="tb-speech-balloon-progress-bar-container tb-w-full tb-h-1.5 tb-bg-text-color/10 tb-mt-4 tb-rounded-full tb-overflow-hidden">
                                <div class="tb-speech-balloon-progress-bar tb-h-full tb-rounded-full ${accentColorClass.replace('tb-text-', 'tb-bg-')} ${this.options.customClasses.progressBar}" style="width: 100%;"></div>
                           </div>`;
        }
        // Add balloon tail/arrow
        contentHtml += `<div class="tb-speech-balloon-tail"></div>`;


        this._toastElement.innerHTML = contentHtml;
        this._progressBarElement = this._toastElement.querySelector('.tb-speech-balloon-progress-bar');


        // Add event listeners for actions and close
        this._toastElement.querySelectorAll('[data-action-idx]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.dataset.actionIdx);
                this.options.actions[idx].action();
                this.hide();
            });
        });

        const tail = this._toastElement.querySelector('.tb-speech-balloon-tail');
        if (tail) {
            if (this.options.position.startsWith('top-')) {
                this._toastElement.classList.add('tb-toast-position-top');
                // Extract the color from borderColorClass (e.g., "tb-border-green-500" -> "green-500")
                // This is a bit hacky, ideally CSS variables would be better.
                const colorName = borderColorClass.split('-').slice(2).join('-'); // e.g., green-500 or gray-400
                tail.style.borderBottomColor = `var(--tb-color-${colorName}, ${TB.ui.theme?.tailwindResolvedColors?.[colorName] || 'currentColor'})`;
            } else {
                this._toastElement.classList.add('tb-toast-position-bottom');
                const colorName = borderColorClass.split('-').slice(2).join('-');
                tail.style.borderTopColor = `var(--tb-color-${colorName}, ${TB.ui.theme?.tailwindResolvedColors?.[colorName] || 'currentColor'})`;
            }
        }

        const closeBtn = this._toastElement.querySelector('[data-close-btn]');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }

        const toastContainerElement = this._getToastContainer();
        if (this.options.position.startsWith('top-')) {
            toastContainerElement.appendChild(this._toastElement);
        } else {
            toastContainerElement.prepend(this._toastElement);
        }
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
        }, 50); // Update interval for smoother animation
    }

    _stopProgressBar() {
        if (this._progressIntervalId) {
            clearInterval(this._progressIntervalId);
            this._progressIntervalId = null;
        }
    }


 show() {
        if (this._isPermanentlyHidden) return; // Don't reshow if dot was removed due to its own timeout

        // If dot exists, remove it before showing toast again
        if (this._dotElement && this._dotElement.parentNode) {
            this._dotElement.parentNode.removeChild(this._dotElement);
            this._dotElement = null;
            if (this._dotTimeoutId) clearTimeout(this._dotTimeoutId);
        }

        if (!this._toastElement) { // Only create if it doesn't exist (e.g., first show or re-show after dot click)
             this._createDom();
        }
        this._toastElement.classList.remove('tb-hidden'); // Ensure visible if re-showing
        this._toastElement.classList.remove('tb-toast-hiding');


        void this._toastElement.offsetWidth; // Force reflow

        this._toastElement.style.opacity = '1';
        this._toastElement.style.transform = 'translateY(0) translateX(0)';

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
            if (this._toastElement) { // Check if it wasn't removed by another call
                this._toastElement.classList.add('tb-hidden'); // Hide it instead of removing if dot is shown
                this._toastElement.classList.remove('tb-toast-hiding');
            }

            if (showDot && !this._isPermanentlyHidden) {
                this._createDot();
            } else {
                this._removeToastCompletely();
            }
            // TB.logger.log(`[Toast] Hidden (transitioning or dot): ${this.options.message.substring(0, 30)}...`);
            // TB.events.emit('toast:hidden', this); // Emit only when fully gone or dot is shown
        }, 300); // Match transition duration
    }

    _createDot() {
        if (this._dotElement || !this._toastElement?.parentNode) return; // Already has a dot or toast is gone

        const { dotColorClass } = this._getAppearanceStyles();
        this._dotElement = document.createElement('div');
        this._dotElement.className = `tb-toast-dot ${dotColorClass}`;
        this._dotElement.title = `Re-open: ${this.options.title || this.options.message.substring(0,20) + '...'}`;


        this._dotElement.addEventListener('click', () => {
            if (this._dotTimeoutId) clearTimeout(this._dotTimeoutId);
            if (this._dotElement && this._dotElement.parentNode) {
                 this._dotElement.parentNode.removeChild(this._dotElement);
            }
            this._dotElement = null;
            this.show(); // Re-show the toast
        });

        // Insert dot where the toast was, or in the container
        const toastContainerElement = this._getToastContainer();
        if (this.options.position.startsWith('top-')) {
            toastContainerElement.appendChild(this._dotElement);
        } else {
            toastContainerElement.prepend(this._dotElement);
        }

        // Auto-remove dot after a while
        this._dotTimeoutId = setTimeout(() => {
            if (this._dotElement && this._dotElement.parentNode) {
                this._dotElement.parentNode.removeChild(this._dotElement);
            }
            this._dotElement = null;
            this._isPermanentlyHidden = true; // Mark that the dot expired
            this._removeToastCompletely(); // Now fully remove the toast
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
        TB.events.emit('toast:hidden', this); // Emit here for permanent removal
    }

    static showInfo(message, options = {}) { new Toast({ ...options, message, type: 'info' }).show(); }
    static showSuccess(message, options = {}) { new Toast({ ...options, message, type: 'success' }).show(); }
    static showWarning(message, options = {}) { new Toast({ ...options, message, type: 'warning' }).show(); }
    static showError(message, options = {}) { new Toast({ ...options, message, type: 'error' }).show(); }
    static hideAll() {
        activeToasts.forEach(toast => {
            toast._isPermanentlyHidden = true; // Ensure dots don't get created for hideAll
            toast.hide(false); // Hide without creating a dot
        });
    }
}

export default Toast;
