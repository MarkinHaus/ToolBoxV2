// tbjs/ui/components/Loader/Loader.js
import TB from '../../../index.js';

const LOADER_ID = 'tbjs-page-loader';
const DEFAULT_LOADER_OPTIONS = {
    text: 'Loading...', // Default text below spinner
    fullscreen: true, // Covers the whole page
    customSpinnerHtml: null, // Allow passing custom spinner HTML
    customClasses: {
        overlay: '',
        spinnerContainer: '',
        text: '',
    },
    playAnimation: true, // Whether to play the graphics animation
    animationSequence: "Y2-42", // Default animation sequence
    hideMainContent: true // Whether to hide MainContent div during loading
};

// Store styles to inject once
let stylesInjected = false;
const LOADER_STYLES = `
#${LOADER_ID} {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 2000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.4);
    color: var(--text-color, #ffffff);
    opacity: 0;
    transition: opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

#${LOADER_ID}.tb-bg-animation {
    background: linear-gradient(45deg,
        rgba(0, 0, 0, 0.3) 0%,
        rgba(var(--theme-primary-rgb, 59, 130, 246), 0.2) 25%,
        rgba(0, 0, 0, 0.3) 50%,
        rgba(var(--theme-primary-rgb, 59, 130, 246), 0.2) 75%,
        rgba(0, 0, 0, 0.3) 100%);
    background-size: 400% 400%;
    animation: tbjs-gradient-shift 8s ease-in-out infinite;
}

#${LOADER_ID} .tb-loader-spinner-container {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 2rem;
}

#${LOADER_ID} .tb-loader-spinner-default {
    width: 60px;
    height: 60px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top: 3px solid var(--theme-primary, #3b82f6);
    border-radius: 50%;
    animation: tbjs-spin 1s linear infinite;
    position: relative;
}

#${LOADER_ID} .tb-loader-spinner-default::before {
    content: '';
    position: absolute;
    top: -6px;
    left: -6px;
    right: -6px;
    bottom: -6px;
    border: 2px solid rgba(255, 255, 255, 0.05);
    border-top: 2px solid rgba(var(--theme-primary-rgb, 59, 130, 246), 0.3);
    border-radius: 50%;
    animation: tbjs-spin-reverse 2s linear infinite;
}

#${LOADER_ID} .tb-loader-spinner-default::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 6px;
    height: 6px;
    background: var(--theme-primary, #3b82f6);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: tbjs-pulse-center 1.5s ease-in-out infinite;
}

#${LOADER_ID} .tb-loader-text {
    font-size: 1.125rem;
    font-weight: 500;
    color: var(--text-color, #ffffff);
    text-align: center;
    margin-top: 1rem;
    opacity: 0.9;
    animation: tbjs-text-fade 2s ease-in-out infinite;
    letter-spacing: 0.5px;
}

#${LOADER_ID} .tb-loader-progress {
    width: 200px;
    height: 2px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 1px;
    margin-top: 1.5rem;
    overflow: hidden;
    position: relative;
}

#${LOADER_ID} .tb-loader-progress::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg,
        transparent,
        var(--theme-primary, #3b82f6),
        transparent);
    animation: tbjs-progress-slide 2s ease-in-out infinite;
}

@keyframes tbjs-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes tbjs-spin-reverse {
    0% { transform: rotate(360deg); }
    100% { transform: rotate(0deg); }
}

@keyframes tbjs-pulse-center {
    0%, 100% { opacity: 0.8; transform: translate(-50%, -50%) scale(1); }
    50% { opacity: 1; transform: translate(-50%, -50%) scale(1.2); }
}

@keyframes tbjs-text-fade {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

@keyframes tbjs-progress-slide {
    0% { left: -100%; }
    100% { left: 100%; }
}

@keyframes tbjs-gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    25% { background-position: 100% 50%; }
    50% { background-position: 50% 100%; }
    75% { background-position: 50% 0%; }
}

/* MainContent grayed out and transparent during loading */
#MainContent.tb-loading-hidden {
    opacity: 0.1 !important;
    filter: grayscale(100%) blur(1px) !important;
    pointer-events: none !important;
    transition: opacity 0.4s ease, filter 0.4s ease !important;
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
    #${LOADER_ID} {
        background: rgba(0, 0, 0, 0.5);
    }
}

/* Light mode adjustments */
@media (prefers-color-scheme: light) {
    #${LOADER_ID} {
        background: rgba(255, 255, 255, 0.4);
    }

    #${LOADER_ID} .tb-loader-progress {
        background: rgba(0, 0, 0, 0.1);
    }

    #${LOADER_ID} .tb-loader-spinner-default {
        border-color: rgba(0, 0, 0, 0.1);
    }

    #${LOADER_ID} .tb-loader-spinner-default::before {
        border-color: rgba(0, 0, 0, 0.05);
        border-top-color: rgba(var(--theme-primary-rgb, 59, 130, 246), 0.3);
    }
}
`;

class Loader {
    constructor(options = {}) {
        this.options = { ...DEFAULT_LOADER_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_LOADER_OPTIONS.customClasses, ...options.customClasses };
        this._loaderElement = null;
        this._animationPlaying = false;
        this._ensureStyles();
    }

    _ensureStyles() {
        if (!stylesInjected) {
            const styleSheet = document.createElement("style");
            styleSheet.type = "text/css";
            styleSheet.innerText = LOADER_STYLES;
            document.head.appendChild(styleSheet);
            stylesInjected = true;
        }
    }

    _hideMainContent() {
        if (this.options.hideMainContent) {
            const mainContent = document.getElementById('MainContent');
            if (mainContent) {
                mainContent.classList.add('tb-loading-hidden');
                TB.logger.log('[Loader] MainContent grayed out and made transparent during loading.');
            }
        }
    }

    _showMainContent() {
        if (this.options.hideMainContent) {
            const mainContent = document.getElementById('MainContent');
            if (mainContent) {
                mainContent.classList.remove('tb-loading-hidden');
                TB.logger.log('[Loader] MainContent restored to normal state after loading.');
            }
        }
    }

    _startGraphicsAnimation() {
        if (this.options.playAnimation && window.TB && window.TB.graphics && window.TB.graphics.playAnimationSequence) {
            try {
                // Parse animation sequence - format: "R2:Y3:P1:Z2-5"
                // R = Rotation X, Y = Rotation Y, P = Rotation Z (Pan), Z = Zoom
                // Number after letter = speed, after colon = time, after dash = repeat
                const sequence = this.options.animationSequence;

                // Start the animation sequence
                window.TB.graphics.playAnimationSequence(sequence);
                this._animationPlaying = true;

                TB.logger.log(`[Loader] Graphics animation started: ${sequence}`);
                TB.events.emit('loader:animation:started', { sequence });
            } catch (error) {
                TB.logger.warn('[Loader] Failed to start graphics animation:', error);
            }
        }
    }

    _stopGraphicsAnimation() {
        if (this._animationPlaying && window.TB && window.TB.graphics && window.TB.graphics.stopAnimationSequence) {
            try {
                window.TB.graphics.stopAnimationSequence();
                this._animationPlaying = false;
                TB.logger.log('[Loader] Graphics animation stopped.');
                TB.events.emit('loader:animation:stopped');
            } catch (error) {
                TB.logger.warn('[Loader] Failed to stop graphics animation:', error);
            }
        }
    }

    _createDom() {
        this._loaderElement = document.createElement('div');
        this._loaderElement.id = LOADER_ID;

        // Apply classes including tb-bg-animation for animated background
        const baseClasses = `tb-bg-animation ${this.options.customClasses.overlay}`;
        this._loaderElement.className = baseClasses.trim();

        // Spinner container
        const spinnerContainer = document.createElement('div');
        spinnerContainer.className = `tb-loader-spinner-container ${this.options.customClasses.spinnerContainer}`;

        if (this.options.customSpinnerHtml) {
            spinnerContainer.innerHTML = this.options.customSpinnerHtml;
        } else {
            // Modern default spinner
            const spinner = document.createElement('div');
            spinner.className = 'tb-loader-spinner-default';
            spinnerContainer.appendChild(spinner);
        }
        this._loaderElement.appendChild(spinnerContainer);

        // Loading text
        if (this.options.text) {
            const textElement = document.createElement('p');
            textElement.className = `tb-loader-text ${this.options.customClasses.text}`;
            textElement.textContent = this.options.text;
            this._loaderElement.appendChild(textElement);
        }

        // Progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'tb-loader-progress';
        this._loaderElement.appendChild(progressBar);

        document.body.appendChild(this._loaderElement);
    }

    show() {
        // Prevent multiple loaders of the same type if fullscreen
        if (this.options.fullscreen && document.getElementById(LOADER_ID)) {
            TB.logger.warn('[Loader] Fullscreen loader already visible.');
            return document.getElementById(LOADER_ID);
        }

        // Hide main content
        this._hideMainContent();

        // Create DOM
        this._createDom();

        // Start graphics animation
        this._startGraphicsAnimation();

        // Force reflow for transition
        void this._loaderElement.offsetWidth;
        this._loaderElement.style.opacity = '1';

        TB.logger.log('[Loader] Modern loader shown with animations.');
        TB.events.emit('loader:shown', this);
        return this._loaderElement;
    }

    hide() {
        const loader = this._loaderElement || document.getElementById(LOADER_ID);
        if (loader) {
            // Stop graphics animation
            this._stopGraphicsAnimation();

            loader.style.opacity = '0';
            setTimeout(() => {
                if (loader.parentNode) {
                    loader.parentNode.removeChild(loader);
                }
                this._loaderElement = null;

                // Show main content
                this._showMainContent();

                TB.logger.log('[Loader] Modern loader hidden.');
                TB.events.emit('loader:hidden', this);
            }, 400); // Match transition duration
        }
    }

    // Update animation sequence dynamically
    updateAnimation(newSequence) {
        this.options.animationSequence = newSequence;
        if (this._animationPlaying) {
            this._stopGraphicsAnimation();
            this._startGraphicsAnimation();
        }
    }

    // Static convenience methods for a global/singleton page loader
    static show(textOrOptions) {
        const options = typeof textOrOptions === 'string' ? { text: textOrOptions } : textOrOptions;
        const loaderInstance = new Loader({ ...options, fullscreen: true });
        return loaderInstance.show();
    }

    static hide(loaderElement) {
        const targetId = loaderElement ? loaderElement.id : LOADER_ID;
        const loaderToHide = document.getElementById(targetId);
        if (loaderToHide) {
            loaderToHide.style.opacity = '0';
            setTimeout(() => {
                if (loaderToHide.parentNode) {
                    loaderToHide.parentNode.removeChild(loaderToHide);
                }

                // Show main content when hiding static loader
                const mainContent = document.getElementById('MainContent');
                if (mainContent) {
                    mainContent.classList.remove('tb-loading-hidden');
                }

                TB.logger.log('[Loader] Static loader hidden.');
            }, 400);
        }
    }

    // Static method to update animation for current loader
    static updateAnimation(newSequence) {
        const currentLoader = document.getElementById(LOADER_ID);
        if (currentLoader && window.TB && window.TB.graphics) {
            try {
                window.TB.graphics.playAnimationSequence(newSequence);
                TB.logger.log(`[Loader] Animation updated to: ${newSequence}`);
            } catch (error) {
                TB.logger.warn('[Loader] Failed to update animation:', error);
            }
        }
    }
}

export default Loader;
