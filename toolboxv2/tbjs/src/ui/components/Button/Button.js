// tbjs/ui/components/Button/Button.js
import TB from '../../../index.js';

const DEFAULT_BUTTON_OPTIONS = {
    text: 'Button',
    action: null, // Click handler
    variant: 'primary', // 'primary', 'secondary', 'danger', 'outline', 'ghost', 'link'
    size: 'md', // 'sm', 'md', 'lg'
    type: 'button', // 'button', 'submit', 'reset'
    disabled: false,
    isLoading: false,
    iconLeft: null, // HTML string for left icon (e.g., Material Symbol span)
    iconRight: null, // HTML string for right icon
    customClasses: '', // Additional custom classes
    attributes: {} // Custom attributes { 'data-id': '123' }
};

class Button {
    constructor(options = {}) {
        this.options = { ...DEFAULT_BUTTON_OPTIONS, ...options };
        this.element = this._createDom();
        if (this.options.action) {
            this.element.addEventListener('click', (e) => {
                if (this.options.isLoading || this.options.disabled) return;
                this.options.action(e, this);
            });
        }
    }

    _getBaseClasses() {
        // /* Tailwind: inline-flex items-center justify-center rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150 */
        return 'inline-flex items-center justify-center rounded-md font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150';
    }

    _getVariantClasses() {
        // /* Tailwind: examples */
        switch (this.options.variant) {
            case 'primary': return 'bg-primary-600 text-white hover:bg-primary-700 focus-visible:ring-primary-500';
            case 'secondary': return 'bg-gray-200 text-gray-800 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 focus-visible:ring-gray-500';
            case 'danger': return 'bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-500';
            case 'outline': return 'border border-primary-600 text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 focus-visible:ring-primary-500';
            case 'ghost': return 'text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 focus-visible:ring-primary-500';
            case 'link': return 'text-primary-600 hover:underline focus-visible:ring-primary-500 p-0'; // Links might not need padding
            default: return 'bg-gray-500 text-white hover:bg-gray-600 focus-visible:ring-gray-400';
        }
    }

    _getSizeClasses() {
        // /* Tailwind: examples */
        if (this.options.variant === 'link') return ''; // Links usually don't have padding from button sizes
        switch (this.options.size) {
            case 'sm': return 'px-2.5 py-1.5 text-xs';
            case 'md': return 'px-4 py-2 text-sm';
            case 'lg': return 'px-6 py-3 text-base';
            default: return 'px-4 py-2 text-sm';
        }
    }

    _createDom() {
        const btn = document.createElement('button');
        btn.type = this.options.type;
        btn.className = `${this._getBaseClasses()} ${this._getVariantClasses()} ${this._getSizeClasses()} ${this.options.customClasses}`;

        if (this.options.disabled) btn.disabled = true;
        if (this.options.isLoading) this.setLoading(true, false); // Don't update DOM yet

        for (const attr in this.options.attributes) {
            btn.setAttribute(attr, this.options.attributes[attr]);
        }

        // Content
        let contentHtml = '';
        if (this.options.iconLeft) contentHtml += `<span class="mr-2 -ml-0.5 h-5 w-5">${this.options.iconLeft}</span>`; // Adjust spacing as needed
        contentHtml += `<span>${this.options.text}</span>`;
        if (this.options.iconRight) contentHtml += `<span class="ml-2 -mr-0.5 h-5 w-5">${this.options.iconRight}</span>`;
        btn.innerHTML = contentHtml;

        return btn;
    }

    setText(text) {
        this.options.text = text;
        const textSpan = this.element.querySelector('span:not([class*="material-symbols"])'); // Find main text span
        if (textSpan) textSpan.textContent = text;
    }

    setLoading(isLoading, updateDom = true) {
        this.options.isLoading = isLoading;
        if (updateDom) {
            this.element.disabled = isLoading || this.options.disabled;
            // /* Tailwind: Add spinner, hide text, or change text */
            if (isLoading) {
                this.element.classList.add('opacity-75', 'cursor-wait'); // Example
                // You might want to replace text with a spinner SVG or animation
                this._originalText = this.element.querySelector('span:not([class*="material-symbols"])').textContent;
                this.setText('Loading...'); // Or use an icon
            } else {
                this.element.classList.remove('opacity-75', 'cursor-wait');
                if(this._originalText) this.setText(this._originalText);
            }
        }
    }

    setDisabled(isDisabled, updateDom = true) {
        this.options.disabled = isDisabled;
        if (updateDom) {
            this.element.disabled = isDisabled || this.options.isLoading;
        }
    }

    // Static method for convenience, returns the DOM element
    static create(text, action, options = {}) {
        const btnInstance = new Button({ ...options, text, action });
        return btnInstance.element;
    }

    // TODO: static initAll(selector) to upgrade existing buttons
}

export default Button;
