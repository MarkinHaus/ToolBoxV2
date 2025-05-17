// tbjs/ui/components/CookieBanner/CookieBanner.js
import TB from '../../../index.js';

const STORAGE_KEY = 'tbjs_cookie_consent';
const DEFAULT_COOKIE_BANNER_OPTIONS = {
    title: 'We value your privacy.',
    message: 'This website uses cookies to enhance user experience. By continuing to browse this site, you agree to our use of cookies.',
    termsLink: '/web/assets/terms.html', // Default terms link
    termsLinkText: 'Terms and Conditions',
    acceptMinimalText: 'Accept & Continue',
    showAdvancedOptions: true,
    advancedOptionsText: 'Customize Settings',
    onConsent: null, // Callback: (consentSettings) => {}
    customClasses: {
        banner: '',
        modal: '',
        // ... other elements
    }
};

class CookieBanner {
    constructor(options = {}) {
        this.options = { ...DEFAULT_COOKIE_BANNER_OPTIONS, ...options };
        this.options.customClasses = { ...DEFAULT_COOKIE_BANNER_OPTIONS.customClasses, ...options.customClasses };

        this.bannerElement = null;
        this.modalElement = null;
        this._init();
    }

    _init() {
        const savedSettings = localStorage.getItem(STORAGE_KEY);
        if (savedSettings) {
            const consent = JSON.parse(savedSettings);
            this._triggerConsent(consent);
            return; // Don't show banner if already consented
        }
        this._createBanner();
        if (this.options.showAdvancedOptions) {
            this._createModal();
        }
    }

    _createBanner() {
        this.bannerElement = document.createElement('div');
        this.bannerElement.id = 'tbjs-cookie-banner';
        // /* Tailwind: fixed bottom-0 left-0 right-0 bg-gray-100 dark:bg-gray-800 p-4 shadow-md z-[1000] text-sm text-gray-700 dark:text-gray-200 */
        this.bannerElement.className = `fixed bottom-0 inset-x-0 bg-gray-100 dark:bg-gray-900 p-4 border-t border-gray-300 dark:border-gray-700 shadow-lg z-[1000] text-sm text-text-color ${this.options.customClasses.banner}`;
        // Original structure: close button, title, message, terms, accept, options
        this.bannerElement.innerHTML = `
            <div class="max-w-screen-lg mx-auto flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
                <div class="flex-grow">
                    <button id="tb-cookie-banner-close" title="Accept Recommended" class="absolute top-2 right-2 material-symbols-outlined text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 text-xl p-1 -m-1">×</button>
                    <h3 class="font-semibold text-base">${this.options.title}</h3>
                    <p class="mt-1">${this.options.message}
                        ${this.options.termsLink ? `<a href="${this.options.termsLink}" class="underline hover:text-primary-500 ml-1" target="_blank">${this.options.termsLinkText}</a>` : ''}
                    </p>
                </div>
                <div class="flex-shrink-0 flex flex-wrap gap-2 items-center">
                    ${this.options.showAdvancedOptions ? `<button id="tb-cookie-banner-show-complex" class="text-xs underline hover:text-primary-500">${this.options.advancedOptionsText}</button>` : ''}
                    <button id="tb-cookie-banner-accept-minimal" class="px-4 py-2 rounded-md bg-primary-600 text-white text-xs font-medium hover:bg-primary-700">${this.options.acceptMinimalText}</button>
                </div>
            </div>
        `;
        document.body.appendChild(this.bannerElement);
        this._addBannerListeners();
    }

    _addBannerListeners() {
        this.bannerElement.querySelector('#tb-cookie-banner-accept-minimal').addEventListener('click', () => {
            this._saveSettings({ analytics: true, preferences: true, essential: true, source: 'accept_minimal' }); // Example consent object
            this._hide();
        });
        this.bannerElement.querySelector('#tb-cookie-banner-close').addEventListener('click', () => {
            this._saveSettings({ analytics: true, preferences: true, essential: true, source: 'close_banner_implicit_accept' });
            this._hide();
        });

        if (this.options.showAdvancedOptions) {
            const showComplexBtn = this.bannerElement.querySelector('#tb-cookie-banner-show-complex');
            if (showComplexBtn) showComplexBtn.addEventListener('click', () => this._showModal());
        }
    }

    _createModal() {
        // This modal reuses TB.ui.Modal if available and preferred, or creates its own simple one.
        // For simplicity, let's make a dedicated one based on original structure.
        this.modalElement = document.createElement('div');
        this.modalElement.id = 'tbjs-cookie-modal';
        // /* Tailwind: fixed inset-0 bg-black bg-opacity-50 z-[1050] hidden items-center justify-center */
        this.modalElement.className = `fixed inset-0 bg-black bg-opacity-50 z-[1050] hidden items-center justify-center p-4 ${this.options.customClasses.modal}`;
        // Original: modal with steps
        // This will be a simplified version for now, can be expanded with steps
        this.modalElement.innerHTML = `
            <div class="bg-background-color p-6 rounded-lg shadow-xl max-w-md w-full">
                <h3 class="text-lg font-semibold mb-4">Cookie Settings</h3>
                <p class="text-sm mb-4">Customize your cookie preferences below. Some cookies are essential for the website to function.</p>
                <div class="space-y-3">
                    <div><label class="flex items-center"><input type="checkbox" id="tb-cookie-essential" checked disabled class="form-checkbox h-4 w-4 text-primary-600 rounded"> <span class="ml-2 text-sm">Essential Cookies</span></label></div>
                    <div><label class="flex items-center"><input type="checkbox" id="tb-cookie-preferences" checked class="form-checkbox h-4 w-4 text-primary-600 rounded"> <span class="ml-2 text-sm">Preferences & Customization</span></label></div>
                    <div><label class="flex items-center"><input type="checkbox" id="tb-cookie-analytics" checked class="form-checkbox h-4 w-4 text-primary-600 rounded"> <span class="ml-2 text-sm">Analytics & Performance</span></label></div>
                </div>
                <div class="mt-6 flex justify-end gap-3">
                    <button id="tb-cookie-modal-save" class="px-4 py-2 rounded-md bg-primary-600 text-white text-sm font-medium hover:bg-primary-700">Save Preferences</button>
                    <button id="tb-cookie-modal-accept-all" class="px-4 py-2 rounded-md bg-gray-200 text-gray-800 text-sm font-medium hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600">Accept All</button>
                </div>
            </div>
        `;
        document.body.appendChild(this.modalElement);
        this._addModalListeners();
    }

    _addModalListeners() {
        this.modalElement.addEventListener('click', (e) => { // Close on overlay click
            if (e.target === this.modalElement) this._hideModal();
        });
        this.modalElement.querySelector('#tb-cookie-modal-save').addEventListener('click', () => {
            const consent = {
                essential: true, // Always true
                preferences: this.modalElement.querySelector('#tb-cookie-preferences').checked,
                analytics: this.modalElement.querySelector('#tb-cookie-analytics').checked,
                source: 'modal_save'
            };
            this._saveSettings(consent);
            this._hideModal();
            this._hide();
        });
        this.modalElement.querySelector('#tb-cookie-modal-accept-all').addEventListener('click', () => {
            this._saveSettings({ essential: true, preferences: true, analytics: true, source: 'modal_accept_all' });
            this._hideModal();
            this._hide();
        });
    }

    _saveSettings(consentSettings) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(consentSettings));
        TB.logger.log('[CookieBanner] Consent saved:', consentSettings);
        this._triggerConsent(consentSettings);
    }

    _triggerConsent(consentSettings){
        TB.events.emit('cookieConsent:updated', consentSettings);
        if (this.options.onConsent) {
            this.options.onConsent(consentSettings);
        }
        // Example: Initialize PostHog or other analytics based on consent
        // if (consentSettings.analytics && typeof posthog !== 'undefined') {
        //     posthog.init('YOUR_API_KEY', { api_host: 'YOUR_HOST', person_profiles: 'identified_only' });
        // }
    }

    _hide() {
        if (this.bannerElement) {
            // /* Tailwind: opacity-0 -translate-y-full or similar for exit animation */
            this.bannerElement.style.opacity = '0';
            this.bannerElement.style.transform = 'translateY(100%)';
            setTimeout(() => {
                if (this.bannerElement && this.bannerElement.parentNode) {
                    this.bannerElement.parentNode.removeChild(this.bannerElement);
                }
                this.bannerElement = null;
            }, 300); // Animation duration
        }
    }

    _showModal() {
        if (this.modalElement) {
            this.modalElement.classList.remove('hidden');
            this.modalElement.classList.add('flex'); // If using flex for centering
            // Add entry animation class if needed
        }
    }

    _hideModal() {
        if (this.modalElement) {
            this.modalElement.classList.add('hidden');
            this.modalElement.classList.remove('flex');
            // Add exit animation class if needed
        }
    }

    static show(options) {
        return new CookieBanner(options);
    }

    // Method to get current consent status
    static getConsent() {
        const saved = localStorage.getItem(STORAGE_KEY);
        return saved ? JSON.parse(saved) : null;
    }
}

export default CookieBanner;
