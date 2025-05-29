// tbjs/ui/components/Toast/Toast.js
import TB from '../../../index.js';
let activeToasts = new Map();

const DEFAULT_TOAST_OPTIONS = {
    message: '',
    type: 'info', // 'info', 'success', 'warning', 'error'
    duration: 4500, // milliseconds, 0 for sticky
    position: 'top-right',//['top-right', 'top-center', 'top-left', 'bottom-right', 'bottom-center', 'bottom-left'
    title: '', // Optional title
    actions: [], // [{ text: 'Undo', action: () => {} }]
    customClasses: {
        container: '',
        title: '',
        message: '',
        actionButton: '',
        progressBar: '',
        toastHost: '',
    },
    icon: true,
    closable: true,
    showDotOnHide: false, // Error its not clickable z index ??
    dotDuration: 1300,
};

let toastContainers = {};

class Toast {
    constructor(options = {}) {
        this.options = { ...DEFAULT_TOAST_OPTIONS, ...options };

        const validPositions = ['top-right', 'top-center', 'top-left', 'bottom-right', 'bottom-center', 'bottom-left'];
        if (!validPositions.includes(this.options.position)) {
            TB.logger.warn(`[Toast] Invalid position "${this.options.position}". Defaulting to "top-right".`);
            this.options.position = 'top-right';
        }

        this.options.customClasses = { ...DEFAULT_TOAST_OPTIONS.customClasses, ...options.customClasses };
        this.id = TB.utils.uniqueId('tb-toast-');
        this._toastElement = null;
        this._timeoutId = null;
        this._progressIntervalId = null;
        this._progressBarElement = null;
        this._dotElement = null;
        this._dotTimeoutId = null;
        this._isPermanentlyHidden = false;
    }

    _getToastContainer() {
        const containerId = `tb-toast-container-${this.options.position.replace(/-/g, '')}`;

        if (!toastContainers[this.options.position]) {
            let container = document.getElementById(containerId);
            if (!container) {
                container = document.createElement('div');
                container.id = containerId;

                const positionClasses = this._getContainerPositionClasses();
                container.className = `tb-toast-container ${positionClasses} ${this.options.customClasses.toastHost || ''}`;
                document.body.appendChild(container);
            }
            toastContainers[this.options.position] = container;
        }
        return toastContainers[this.options.position];
    }

    _getContainerPositionClasses() {
        const baseClasses = 'tb-fixed tb-z-[9999] tb-flex tb-flex-col tb-pointer-events-none';
        const spacing = 'tb-p-4 tb-gap-2';

        switch (this.options.position) {
            case 'top-right':
                return `${baseClasses} ${spacing} tb-top-0 tb-right-0 tb-items-end`;
            case 'top-center':
                return `${baseClasses} ${spacing} tb-top-0 tb-left-1/2 tb--translate-x-1/2 tb-items-center`;
            case 'top-left':
                return `${baseClasses} ${spacing} tb-top-0 tb-left-0 tb-items-start`;
            case 'bottom-right':
                return `${baseClasses} ${spacing} tb-bottom-0 tb-right-0 tb-items-end`;
            case 'bottom-center':
                return `${baseClasses} ${spacing} tb-bottom-0 tb-left-1/2 tb--translate-x-1/2 tb-items-center`;
            case 'bottom-left':
                return `${baseClasses} ${spacing} tb-bottom-0 tb-left-0 tb-items-start`;
            default:
                return `${baseClasses} ${spacing} tb-top-0 tb-right-0 tb-items-end`;
        }
    }

    _getTypeConfig() {
        const configs = {
            success: {
                color: '#10b981',
                icon: 'check_circle',
                label: 'Success'
            },
            warning: {
                color: '#f59e0b',
                icon: 'warning',
                label: 'Warning'
            },
            error: {
                color: '#ef4444',
                icon: 'error',
                label: 'Error'
            },
            info: {
                color: '#3b82f6',
                icon: 'info',
                label: 'Info'
            }
        };
        return configs[this.options.type] || configs.info;
    }

    _createDom() {
        const container = this._getToastContainer();
        this._toastElement = document.createElement('div');
        this._toastElement.id = this.id;

        const typeConfig = this._getTypeConfig();
        const fromLabel = this.options.title || typeConfig.label;

        // Modern, slim toast with plastic aesthetic
        this._toastElement.className = `tb-toast ${this.options.customClasses.container}`;
        this._toastElement.style.cssText = `
            --toast-color: ${typeConfig.color};
            pointer-events: auto;
        `;

        let contentHtml = `
            <div class="tb-toast-header">
                ${this.options.icon ? `<span class="tb-toast-icon material-symbols-outlined">${typeConfig.icon}</span>` : ''}
                <span class="tb-toast-title">${TB.utils.escapeHtml(fromLabel)}</span>
                ${this.options.closable ? `
                    <button class="tb-toast-close" data-close-btn="true" aria-label="Close">
                        <span class="material-symbols-outlined">close</span>
                    </button>
                ` : ''}
            </div>
        `;

        if (this.options.message) {
            contentHtml += `
                <div class="tb-toast-message ${this.options.customClasses.message}">
                    ${TB.utils.escapeHtml(this.options.message)}
                </div>
            `;
        }

        if (this.options.actions.length > 0) {
            contentHtml += '<div class="tb-toast-actions">';
            this.options.actions.forEach((action, idx) => {
                contentHtml += `
                    <button class="tb-toast-action-btn ${this.options.customClasses.actionButton}" data-action-idx="${idx}">
                        ${TB.utils.escapeHtml(action.text)}
                    </button>
                `;
            });
            contentHtml += '</div>';
        }

        if (this.options.duration > 0) {
            contentHtml += `
                <div class="tb-toast-progress ${this.options.customClasses.progressBar}">
                    <div class="tb-toast-progress-bar"></div>
                </div>
            `;
        }

        this._toastElement.innerHTML = contentHtml;
        this._progressBarElement = this._toastElement.querySelector('.tb-toast-progress-bar');

        // Event listeners
        this._toastElement.querySelectorAll('[data-action-idx]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.dataset.actionIdx);
                this.options.actions[idx].action();
                this.hide(false);
            });
        });

        const closeBtn = this._toastElement.querySelector('[data-close-btn]');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide(this.options.showDotOnHide));
        }

        // Insert toast
        if (this.options.position.startsWith('bottom-')) {
            container.prepend(this._toastElement);
        } else {
            container.appendChild(this._toastElement);
        }

        activeToasts.set(this.id, this);
    }

    _startProgressBar() {
        if (!this._progressBarElement || this.options.duration <= 0) return;

        const startTime = Date.now();
        const duration = this.options.duration;

        this._progressIntervalId = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progress = Math.max(0, 100 - (elapsed / duration) * 100);

            this._progressBarElement.style.transform = `scaleX(${progress / 100})`;

            if (progress === 0) {
                clearInterval(this._progressIntervalId);
                this._progressIntervalId = null;
            }
        }, 16); // ~60fps
    }

    _stopProgressBar() {
        if (this._progressIntervalId) {
            clearInterval(this._progressIntervalId);
            this._progressIntervalId = null;
        }
    }

    show() {
        if (this._isPermanentlyHidden) return;

        if (this._dotElement?.parentNode) {
            this._dotElement.parentNode.removeChild(this._dotElement);
            this._dotElement = null;
            if (this._dotTimeoutId) clearTimeout(this._dotTimeoutId);
        }

        if (!this._toastElement) {
            this._createDom();
        }

        // Show with animation
        this._toastElement.classList.remove('tb-toast-hidden', 'tb-toast-hiding');
        this._toastElement.classList.add('tb-toast-showing');

        // Start auto-hide timer and progress bar
        if (this.options.duration > 0) {
            this._startProgressBar();
            this._timeoutId = setTimeout(() => this.hide(this.options.showDotOnHide), this.options.duration);
        }

        TB.logger.log(`[Toast] Shown: ${this.options.message.substring(0, 30)}... (ID: ${this.id})`);
        TB.events.emit('toast:shown', this);
    }

    hide(showDot = this.options.showDotOnHide) {
        if (!this._toastElement || this._toastElement.classList.contains('tb-toast-hiding')) return;

        this._toastElement.classList.add('tb-toast-hiding');
        this._toastElement.classList.remove('tb-toast-showing');

        if (this._timeoutId) clearTimeout(this._timeoutId);
        this._stopProgressBar();

        setTimeout(() => {
            if (this._toastElement) {
                this._toastElement.classList.add('tb-toast-hidden');
                this._toastElement.classList.remove('tb-toast-hiding');
            }

            if (showDot && !this._isPermanentlyHidden) {
                this._createDot();
            } else {
                this._removeToastCompletely();
            }
        }, 250); // Match CSS transition duration
    }

    _createDot() {
        if (this._dotElement || !this._toastElement?.parentNode || this._isPermanentlyHidden) return;

        const typeConfig = this._getTypeConfig();
        this._dotElement = document.createElement('div');
        this._dotElement.className = 'tb-toast-dot';
        this._dotElement.style.backgroundColor = typeConfig.color;

        const tooltip = `Re-open: ${this.options.title || this.options.message.substring(0, 20) + '...'}`;
        this._dotElement.title = TB.utils.escapeHtml(tooltip);
        this._dotElement.setAttribute('aria-label', TB.utils.escapeHtml(tooltip));

        this._dotElement.addEventListener('click', () => {
            if (this._dotTimeoutId) clearTimeout(this._dotTimeoutId);
            if (this._dotElement?.parentNode) {
                this._dotElement.parentNode.removeChild(this._dotElement);
            }
            this._dotElement = null;
            this.show();
        });

        const container = this._getToastContainer();
        if (this.options.position.startsWith('bottom-')) {
            container.prepend(this._dotElement);
        } else {
            container.appendChild(this._dotElement);
        }

        // Auto-hide dot after duration
        this._dotTimeoutId = setTimeout(() => {
            if (this._dotElement?.parentNode) {
                this._dotElement.classList.add('tb-toast-dot-hiding');
                setTimeout(() => {
                    if (this._dotElement?.parentNode) {
                        this._dotElement.parentNode.removeChild(this._dotElement);
                    }
                    this._dotElement = null;
                    if (!activeToasts.has(this.id)) {
                        this._isPermanentlyHidden = true;
                        this._removeToastCompletely();
                    }
                }, 200);
            }
        }, this.options.dotDuration);
    }

    _removeToastCompletely() {
        if (this._toastElement?.parentNode) {
            this._toastElement.parentNode.removeChild(this._toastElement);
        }
        activeToasts.delete(this.id);
        this._toastElement = null;

        const container = toastContainers[this.options.position];
        if (container && container.children.length === 0) {
            if (container.parentNode) container.parentNode.removeChild(container);
            delete toastContainers[this.options.position];
        }

        TB.logger.log(`[Toast] Removed: ${this.options.message.substring(0, 30)}... (ID: ${this.id})`);
        TB.events.emit('toast:hidden', this);
    }

    // Static convenience methods
    static showInfo(message, options = {}) {
        return new Toast({ ...options, message, type: 'info' }).show() //(t => t.show());
    }

    static showSuccess(message, options = {}) {
        return new Toast({ ...options, message, type: 'success' }).show() //(t => t.show());
    }

    static showWarning(message, options = {}) {
        return new Toast({ ...options, message, type: 'warning' }).show() //(t => t.show());
    }

    static showError(message, options = {}) {
        return new Toast({ ...options, message, type: 'error' }).show() //(t => t.show());
    }

    static hideAll(immediately = false) {
        activeToasts.forEach(toast => {
            toast._isPermanentlyHidden = true;
            if (immediately) {
                toast._removeToastCompletely();
            } else {
                toast.hide(false);
            }
        });

        if (immediately) {
            activeToasts.clear();
            Object.values(toastContainers).forEach(container => {
                if (container?.parentNode) {
                    container.parentNode.removeChild(container);
                }
            });
            toastContainers = {};
        }
    }

    // Test methods
    static __testonly__resetModuleState() {
        Toast.hideAll(true);
    }

    static __testonly__getActiveToasts() {
        return activeToasts;
    }

    static __testonly__getToastContainers() {
        return toastContainers;
    }
}

export default Toast;
