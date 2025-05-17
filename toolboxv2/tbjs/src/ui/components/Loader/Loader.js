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
    }
};

// Store styles to inject once
let stylesInjected = false;
const LOADER_STYLES = `
#${LOADER_ID} { /* Tailwind equivalent: fixed inset-0 z-[2000] flex flex-col items-center justify-center bg-black bg-opacity-75 */ }
#${LOADER_ID} .tb-loader-spinner-default {
    /* Tailwind equivalent: border-4 border-gray-300 border-t-primary-500 rounded-full w-12 h-12 animate-spin */
    width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.3);
    border-top-color: #fff; border-radius: 50%; animation: tbjs_spin 1s linear infinite;
}
#${LOADER_ID} .tb-loader-text { /* Tailwind: mt-4 text-white text-lg */ }
@keyframes tbjs_spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
`;


class Loader {
    constructor(options = {}) {
        this.options = { ...DEFAULT_LOADER_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_LOADER_OPTIONS.customClasses, ...options.customClasses };
        this._loaderElement = null;
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

    _createDom() {
        this._loaderElement = document.createElement('div');
        this._loaderElement.id = LOADER_ID;
        // Apply Tailwind classes or use defaults from injected styles
        this._loaderElement.className = `fixed inset-0 z-[2000] flex flex-col items-center justify-center bg-black bg-opacity-75 text-white opacity-0 transition-opacity duration-300 ${this.options.customClasses.overlay}`;

        const spinnerContainer = document.createElement('div');
        spinnerContainer.className = this.options.customClasses.spinnerContainer;

        if (this.options.customSpinnerHtml) {
            spinnerContainer.innerHTML = this.options.customSpinnerHtml;
        } else {
            // Default spinner (can be replaced with Tailwind styled div)
            const spinner = document.createElement('div');
            spinner.className = 'tb-loader-spinner-default'; // Styled by injected CSS
            spinnerContainer.appendChild(spinner);
        }
        this._loaderElement.appendChild(spinnerContainer);

        if (this.options.text) {
            const textElement = document.createElement('p');
            textElement.className = `tb-loader-text mt-4 text-lg ${this.options.customClasses.text}`;
            textElement.textContent = this.options.text;
            this._loaderElement.appendChild(textElement);
        }
        document.body.appendChild(this._loaderElement);
    }

    show() {
        // Prevent multiple loaders of the same type if fullscreen
        if (this.options.fullscreen && document.getElementById(LOADER_ID)) {
            TB.logger.warn('[Loader] Fullscreen loader already visible.');
            return document.getElementById(LOADER_ID); // Return existing
        }
        this._createDom();

        // Force reflow for transition
        void this._loaderElement.offsetWidth;
        this._loaderElement.style.opacity = '1';

        TB.logger.log('[Loader] Shown.');
        TB.events.emit('loader:shown', this);
        return this._loaderElement; // Return the DOM element
    }

    hide() {
        const loader = this._loaderElement || document.getElementById(LOADER_ID);
        if (loader) {
            loader.style.opacity = '0';
            setTimeout(() => {
                if (loader.parentNode) {
                    loader.parentNode.removeChild(loader);
                }
                this._loaderElement = null;
                TB.logger.log('[Loader] Hidden.');
                TB.events.emit('loader:hidden', this);
            }, 300); // Match transition duration
        }
    }

    // Static convenience methods for a global/singleton page loader
    static show(textOrOptions) {
        const options = typeof textOrOptions === 'string' ? { text: textOrOptions } : textOrOptions;
        const loaderInstance = new Loader({ ...options, fullscreen: true }); // Ensure it's fullscreen
        return loaderInstance.show(); // Return DOM element
    }

    static hide(loaderElement) { // Accept element to hide specific instance
        const targetId = loaderElement ? loaderElement.id : LOADER_ID;
        const loaderToHide = document.getElementById(targetId);
        if (loaderToHide) {
            loaderToHide.style.opacity = '0';
            setTimeout(() => {
                if (loaderToHide.parentNode) {
                    loaderToHide.parentNode.removeChild(loaderToHide);
                }
                 TB.logger.log('[Loader] Static hide called.');
            }, 300);
        }
    }
}

export default Loader;
