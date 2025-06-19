// tbjs/ui/components/Autocomplete/Autocomplete.js
import TB from '../../../index.js';

const DEFAULT_AUTOCOMPLETE_OPTIONS = {
    source: [], // Array of strings or function that returns/resolves to an array
    minLength: 1,
    onSelect: null, // (value, inputElement) => {}
    customClasses: {
        list: 'autocomplete-items border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 z-10 max-h-60 overflow-y-auto',
        item: 'p-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer',
        activeItem: 'bg-primary-500 text-white', // Or just primary-100 for highlight
        highlight: 'font-bold'
    }
};

class AutocompleteWidget {
    constructor(inputElement, options = {}) {
        if (!inputElement) {
            TB.logger.error('[Autocomplete] Input element not provided.');
            return;
        }
        this.inputElement = inputElement;
        this.options = { ...DEFAULT_AUTOCOMPLETE_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_AUTOCOMPLETE_OPTIONS.customClasses, ...options.customClasses };
        this.currentFocus = -1;
        this.listElement = null;

        this._boundHandleInput = this._handleInput.bind(this);
        this._boundHandleKeyDown = this._handleKeyDown.bind(this);
        this._boundHandleDocumentClick = this._handleDocumentClick.bind(this);

        this.inputElement.addEventListener('input', this._boundHandleInput);
        this.inputElement.addEventListener('keydown', this._boundHandleKeyDown);
        document.addEventListener('click', this._boundHandleDocumentClick);

        // Add ARIA attributes
        this.inputElement.setAttribute('autocomplete', 'off');
        this.inputElement.setAttribute('role', 'combobox');
        this.inputElement.setAttribute('aria-autocomplete', 'list');
        this.inputElement.setAttribute('aria-expanded', 'false');
        this.inputElement.setAttribute('aria-haspopup', 'listbox');

        TB.logger.log('[Autocomplete] Initialized for input:', inputElement);
    }

    async _getSourceData() {
        if (typeof this.options.source === 'function') {
            try {
                const data = await this.options.source(this.inputElement.value);
                return Array.isArray(data) ? data : [];
            } catch (e) {
                TB.logger.error('[Autocomplete] Error fetching source data:', e);
                return [];
            }
        }
        return Array.isArray(this.options.source) ? this.options.source : [];
    }

    async _handleInput() {
        const val = this.inputElement.value;
        this._closeAllLists();
        if (!val || val.length < this.options.minLength) {
            this.inputElement.setAttribute('aria-expanded', 'false');
            return false;
        }

        this.currentFocus = -1;
        this.listElement = document.createElement('div');
        this.listElement.setAttribute('id', `${this.inputElement.id || TB.utils.uniqueId('ac-list-')}`);
        this.listElement.setAttribute('role', 'listbox');
        // /* Tailwind: absolute left-0 right-0 mt-1 */ (position relative to input's parent)
        this.listElement.className = `absolute left-0 right-0 mt-1 ${this.options.customClasses.list}`;
        this.inputElement.parentNode.style.position = 'relative'; // Ensure parent is positioned
        this.inputElement.parentNode.appendChild(this.listElement);

        const data = await this._getSourceData();
        data.forEach(item => {
            if (item.substr(0, val.length).toUpperCase() === val.toUpperCase()) {
                const itemElement = document.createElement('div');
                itemElement.setAttribute('role', 'option');
                itemElement.className = this.options.customClasses.item;
                // /* Tailwind: for highlight */
                itemElement.innerHTML = `<span class="${this.options.customClasses.highlight}">${item.substr(0, val.length)}</span>${item.substr(val.length)}`;
                itemElement.dataset.value = item;

                itemElement.addEventListener('click', () => {
                    this.inputElement.value = item;
                    this._closeAllLists();
                    this.inputElement.setAttribute('aria-expanded', 'false');
                    if (this.options.onSelect) this.options.onSelect(item, this.inputElement);
                    TB.events.emit('autocomplete:selected', { value: item, input: this.inputElement });
                });
                this.listElement.appendChild(itemElement);
            }
        });

        if (this.listElement.children.length > 0) {
            this.inputElement.setAttribute('aria-expanded', 'true');
            this.inputElement.setAttribute('aria-controls', this.listElement.id);
        } else {
            this._closeAllLists(); // No matches
            this.inputElement.setAttribute('aria-expanded', 'false');
        }
    }

    _handleKeyDown(e) {
        if (!this.listElement || this.listElement.children.length === 0) return;
        const items = Array.from(this.listElement.children);

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            this.currentFocus++;
            this._addActive(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            this.currentFocus--;
            this._addActive(items);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (this.currentFocus > -1 && items[this.currentFocus]) {
                items[this.currentFocus].click();
            } else if (items.length > 0 && this.inputElement.value.length >= this.options.minLength) {
                // Optional: Select first item if Enter is pressed and no specific focus
                // items[0].click();
            }
        } else if (e.key === 'Escape') {
            this._closeAllLists();
            this.inputElement.setAttribute('aria-expanded', 'false');
        }
    }

    _addActive(items) {
        if (!items || items.length === 0) return false;
        this._removeActive(items);
        if (this.currentFocus >= items.length) this.currentFocus = 0;
        if (this.currentFocus < 0) this.currentFocus = items.length - 1;

        items[this.currentFocus].classList.add(...this.options.customClasses.activeItem.split(' '));
        this.inputElement.setAttribute('aria-activedescendant', items[this.currentFocus].id || (items[this.currentFocus].id = TB.utils.uniqueId('ac-item-')));
        // Scroll into view if needed
        items[this.currentFocus].scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }

    _removeActive(items) {
        items.forEach(item => item.classList.remove(...this.options.customClasses.activeItem.split(' ')));
    }

    _closeAllLists() {
        if (this.listElement && this.listElement.parentNode) {
            this.listElement.parentNode.removeChild(this.listElement);
        }
        this.listElement = null;
    }

    _handleDocumentClick(e) {
        if (e.target !== this.inputElement && (!this.listElement || !this.listElement.contains(e.target))) {
            this._closeAllLists();
            this.inputElement.setAttribute('aria-expanded', 'false');
        }
    }

    destroy() {
        this._closeAllLists();
        this.inputElement.removeEventListener('input', this._boundHandleInput);
        this.inputElement.removeEventListener('keydown', this._boundHandleKeyDown);
        document.removeEventListener('click', this._boundHandleDocumentClick);
        // Remove ARIA attributes
        ['autocomplete', 'role', 'aria-autocomplete', 'aria-expanded', 'aria-haspopup', 'aria-controls', 'aria-activedescendant'].forEach(attr => this.inputElement.removeAttribute(attr));
        TB.logger.log('[Autocomplete] Destroyed for input:', this.inputElement);
    }

    // Static initializer
    static initAll(selector = 'input[data-tb-autocomplete]') {
        document.querySelectorAll(selector).forEach(el => {
            const sourceAttr = el.dataset.tbAutocompleteSource;
            let source = [];
            if (sourceAttr) {
                try { source = JSON.parse(sourceAttr); }
                catch (e) {
                    // Check if it's a global function name
                    if (typeof window[sourceAttr] === 'function') source = window[sourceAttr];
                    else TB.logger.warn(`[Autocomplete] Invalid JSON or function name for source on element:`, el);
                }
            }
            new AutocompleteWidget(el, { source });
        });
    }
}

export default AutocompleteWidget;
