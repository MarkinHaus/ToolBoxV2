// tbjs/src/ui/components/darkModeToggle.js
import TB from '../../../index.js';

const DEFAULT_OPTIONS = {
    containerSelector: '#darkModeToggleContainer', // The main clickable element / wrapper
    iconSelector: '.tb-toggle-icon',          // The span containing the material icon
    checkboxSelector: '#darkModeSwitch',       // The actual <input type="checkbox">
    lightModeIconClass: 'light_mode',
    darkModeIconClass: 'dark_mode',
    rotationActiveDeg: '360deg', // For dark mode (active state)
    rotationInactiveDeg: '0deg', // For light mode (inactive state)
    rotationTransition: 'transform 0.5s ease'
};

class DarkModeToggle {
    constructor(options = {}) {
        this.options = { ...DEFAULT_OPTIONS, ...options };

        this.containerElement = document.querySelector(this.options.containerSelector);
        if (!this.containerElement) {
            TB.logger.warn(`[DarkModeToggle] Container "${this.options.containerSelector}" not found.`);
            return;
        }

        this.iconElement = this.containerElement.querySelector(this.options.iconSelector);
        if (!this.iconElement) {
            if (this.containerElement.matches(this.options.iconSelector)) {
                this.iconElement = this.containerElement;
            } else {
                TB.logger.warn(`[DarkModeToggle] Icon element "${this.options.iconSelector}" not found in "${this.options.containerSelector}".`);
            }
        }

        this.checkboxElement = document.querySelector(this.options.checkboxSelector);

        // Bound event handlers
        this._boundHandleInteraction = this._handleInteraction.bind(this);
        this._boundUpdateVisualsFromEvent = (eventData) => this.updateVisuals(eventData.mode);
        this._boundContainerClickListener = null; // For conditional listener

        this._attachEventListeners();
        this._initializeVisuals();

        TB.logger.log(`[DarkModeToggle] Initialized for container: "${this.options.containerSelector}"`);
    }

    _attachEventListeners() {
        if (this.checkboxElement) {
            this.checkboxElement.addEventListener('change', this._boundHandleInteraction);

            const isContainerLabelForCheckbox =
                this.containerElement.tagName === 'LABEL' &&
                this.containerElement.getAttribute('for') === this.checkboxElement.id;

            if (!isContainerLabelForCheckbox) {
                this._boundContainerClickListener = (event) => {
                    if (event.target === this.checkboxElement || event.target.closest(`label[for="${this.checkboxElement.id}"]`)) {
                        return;
                    }
                    this.checkboxElement.checked = !this.checkboxElement.checked;
                    this.checkboxElement.dispatchEvent(new Event('change', { bubbles: true }));
                };
                this.containerElement.addEventListener('click', this._boundContainerClickListener);
            }
        } else {
            this.containerElement.addEventListener('click', this._boundHandleInteraction);
        }

        TB.events.on('theme:changed', this._boundUpdateVisualsFromEvent);
    }

    _handleInteraction(event) {
        if (!TB.ui || !TB.ui.theme) {
            TB.logger.error("[DarkModeToggle] ThemeManager (TB.ui.theme) not available.");
            return;
        }

        if (this.checkboxElement && event.type === 'change' && event.target === this.checkboxElement) {
            const newPreference = this.checkboxElement.checked ? 'dark' : 'light';
            TB.ui.theme.setPreference(newPreference);
        } else if (!this.checkboxElement && event.type === 'click') {
            TB.ui.theme.togglePreference();
        }
    }

    _initializeVisuals() {
        if (TB.ui && TB.ui.theme && typeof TB.ui.theme.getCurrentMode === 'function') {
            this.updateVisuals(TB.ui.theme.getCurrentMode());
        } else {
            TB.logger.warn('[DarkModeToggle] TB.ui.theme not available for initial state. Visuals defaulting to light.');
            this.updateVisuals('light');
        }
    }

    updateVisuals(themeMode) {
        if (this.iconElement) {
            this.iconElement.textContent = themeMode === 'dark' ? this.options.darkModeIconClass : this.options.lightModeIconClass;
            this.iconElement.style.transition = this.options.rotationTransition;
            // Active state (dark) is rotated, inactive (light) is not.
            this.iconElement.style.transform = themeMode === 'dark' ? `rotate(${this.options.rotationActiveDeg})` : `rotate(${this.options.rotationInactiveDeg})`;
        }

        if (this.checkboxElement) {
            this.checkboxElement.checked = themeMode === 'dark';
        }
    }

    destroy() {
        if (this.checkboxElement) {
            this.checkboxElement.removeEventListener('change', this._boundHandleInteraction);
        }

        if (this._boundContainerClickListener && this.containerElement) {
            this.containerElement.removeEventListener('click', this._boundContainerClickListener);
        } else if (!this.checkboxElement && this.containerElement) { // Only if it was the primary interaction
             this.containerElement.removeEventListener('click', this._boundHandleInteraction);
        }

        TB.events.off('theme:changed', this._boundUpdateVisualsFromEvent);
        TB.logger.log(`[DarkModeToggle] Destroyed for container: "${this.options.containerSelector}"`);
    }

    static init(optionsOrSelector) {
        let options = {};
        if (typeof optionsOrSelector === 'string') {
            options.containerSelector = optionsOrSelector;
        } else if (optionsOrSelector) {
            options = { ...optionsOrSelector };
        }
        // Ensure TB.ui.theme is available before proceeding, or handle gracefully.
        if (TB.ui && TB.ui.theme) {
            return new DarkModeToggle(options);
        } else {
            TB.logger.error("[DarkModeToggle.init] TB.ui.theme is not initialized. Cannot create DarkModeToggle instance.");
            return null;
        }
    }
}

export default DarkModeToggle;
