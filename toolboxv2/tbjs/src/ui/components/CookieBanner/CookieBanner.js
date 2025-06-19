// tbjs/ui/components/CookieBanner/CookieBanner.js
import TB from '../../../index.js'; // Assuming TB.ui.Modal is available via this import

const STORAGE_KEY = 'tbjs_cookie_consent';
const DEFAULT_COOKIE_BANNER_OPTIONS = {
    title: 'We value your privacy.',
    message: 'This website uses cookies to enhance user experience. By continuing to browse this site, you agree to our use of cookies.',
    termsLink: '/web/assets/terms.html', // Default terms link
    termsLinkText: 'Terms and Conditions',
    acceptMinimalText: 'Accept & Continue',
    showAdvancedOptions: true,
    advancedOptionsText: 'Customize Settings',
    onConsent: onConsentInitPostHog, // Callback: (consentSettings) => {}
    customClasses: {
        banner: '', // For the banner itself
        modalContainer: '', // To be passed to TB.ui.Modal's customClasses.modalContainer
        // Add other custom classes for banner internal elements if needed
    },
    // Default consent state for checkboxes if no saved settings
    defaultPreferences: true,
    defaultAnalytics: true,
};

function onConsentInitPostHog(consentSettings) {
    // get mode ["none", "identified_only", "always"]
    let _mode = ""
    if (consentSettings.analytics) {
        if(consentSettings.preferences){
            _mode = "always"
        } else {
            _mode = "identified_only"
        }
    } else {
        return;
    }
    function initPosthog(mode) {
    !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey getNextSurveyStep identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug getPageViewId".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
    posthog.init('phc_zsEwhB79hF41y7DjaAkGSExrJNuPffyOUKlU1bM0r3V', {api_host: 'https://eu.i.posthog.com', person_profiles: mode});
  }
    initPosthog(_mode)
}

class CookieBanner {
    constructor(options = {}) {
        this.options = {
            ...DEFAULT_COOKIE_BANNER_OPTIONS,
            ...options,
            customClasses: { // Deep merge customClasses
                ...DEFAULT_COOKIE_BANNER_OPTIONS.customClasses,
                ...(options.customClasses || {}),
            }
        };

        this.bannerElement = null;
        this.settingsModalInstance = null; // To hold the TB.ui.Modal instance
        this._init();
    }

    _init() {
        const savedSettings = this._getSavedSettings();
        if (savedSettings) {
            this._triggerConsent(savedSettings);
            // Optionally, you might still want to provide a way to change settings later,
            // e.g., via a link in the footer, not covered by this banner logic.
            return; // Don't show banner if already consented
        }
        this._createBanner();
    }

    _getSavedSettings() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                return JSON.parse(saved);
            } catch (e) {
                TB.logger.error('[CookieBanner] Error parsing saved settings from localStorage:', e);
                localStorage.removeItem(STORAGE_KEY); // Clear corrupted data
            }
        }
        return null;
    }

    _getInitialPreferenceState(type, defaultValue) {
        const savedSettings = this._getSavedSettings();
        if (savedSettings && savedSettings[type] !== undefined) {
            return savedSettings[type];
        }
        return defaultValue;
    }

    _createBanner() {
        if (this.bannerElement) return; // Already created

        this.bannerElement = document.createElement('div');
        this.bannerElement.id = 'tbjs-cookie-banner';
        // this.bannerElement.className = `fixed bottom-0 inset-x-0 bg-gray-100 dark:bg-gray-900 p-4 border-t border-gray-300 dark:border-gray-700 shadow-lg z-[1000] text-sm text-text-color transition-all duration-300 ease-in-out transform translate-y-full opacity-0 ${this.options.customClasses.banner}`;
        this.bannerElement.innerHTML = `
            <div class="max-w-screen-lg mx-auto flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
                <div class="flex-grow">
                    <span id="tb-cookie-banner-close" title="Accept Recommended & Close" class="absolute top-2 right-2 material-symbols-outlined text-gray-500 hover:text-gray-700 dark:text-gray-300 dark:hover:text-gray-400 text-xl p-1 -m-1 cursor-pointer">close</span>
                    <h3 class="font-semibold text-base">${this.options.title}</h3>
                    <p class="mt-1 text-xs sm:text-sm">
                        ${this.options.message}
                        ${this.options.termsLink ? `<a href="${this.options.termsLink}" class="underline hover:text-primary-500 ml-1" target="_blank" rel="noopener noreferrer">${this.options.termsLinkText}</a>` : ''}
                    </p>
                </div>
                <div class="flex-shrink-0 flex flex-wrap gap-2 items-center">
                    ${this.options.showAdvancedOptions ? `<button id="tb-cookie-banner-show-settings" class="text-xs underline hover:text-primary-500">${this.options.advancedOptionsText}</button>` : ''}
                    <button id="tb-cookie-banner-accept-minimal" class="px-3 py-1.5 sm:px-4 sm:py-2 rounded-md bg-primary-600 text-white text-xs font-medium hover:bg-primary-700">${this.options.acceptMinimalText}</button>
                </div>
            </div>
        `;
        document.body.appendChild(this.bannerElement);

        // Trigger enter animation
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                this.bannerElement.style.transform = 'translateY(0)';
                this.bannerElement.style.opacity = '1';
            });
        });

        this._addBannerListeners();
        TB.events.emit('cookieBanner:shown');
        TB.logger.log('[CookieBanner] Banner shown.');
    }

    _addBannerListeners() {
        const acceptMinimalBtn = this.bannerElement.querySelector('#tb-cookie-banner-accept-minimal');
        if (acceptMinimalBtn) {
            acceptMinimalBtn.addEventListener('click', () => {
                this._saveSettingsAndHide({
                    analytics: true, preferences: true, essential: true, source: 'accept_minimal'
                });
            });
        }

        const closeBtn = this.bannerElement.querySelector('#tb-cookie-banner-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                // Implicitly accept recommended settings when "X" is clicked on banner
                this._saveSettingsAndHide({
                    analytics: true, preferences: true, essential: true, source: 'close_banner_implicit_accept'
                });
            });
        }

        if (this.options.showAdvancedOptions) {
            const showSettingsBtn = this.bannerElement.querySelector('#tb-cookie-banner-show-settings');
            if (showSettingsBtn) {
                showSettingsBtn.addEventListener('click', () => this._showSettingsModal());
            }
        }
    }

    _showSettingsModal() {
        if (this.settingsModalInstance && this.settingsModalInstance.isOpen) {
            TB.logger.log('[CookieBanner] Settings modal is already open.');
            // Potentially focus the modal if Modal.js supports it: this.settingsModalInstance.focus();
            return;
        }

        // Determine initial checkbox states
        const initialPreferences = this._getInitialPreferenceState('preferences', this.options.defaultPreferences);
        const initialAnalytics = this._getInitialPreferenceState('analytics', this.options.defaultAnalytics);

        const modalContent = `
            <div class="space-y-3">
                <div>
                    <label class="flex items-center">
                        <input type="checkbox" id="tb-cookie-essential" checked disabled
                               class="form-checkbox h-4 w-4 text-primary-600 rounded focus:ring-primary-500 dark:focus:ring-offset-neutral-800/20">
                        <span class="ml-2 text-sm">Essential Cookies</span>
                    </label>
                   </div>
                <div>
                    <label class="flex items-center">
                        <input type="checkbox" id="tb-cookie-preferences" ${initialPreferences ? 'checked' : ''}
                               class="form-checkbox h-4 w-4 text-primary-600 rounded focus:ring-primary-500 dark:focus:ring-offset-neutral-800/20">
                        <span class="ml-2 text-sm">Preferences & Customization</span>
                    </label>
                     </div>
                <div>
                    <label class="flex items-center">
                        <input type="checkbox" id="tb-cookie-analytics" ${initialAnalytics ? 'checked' : ''}
                               class="form-checkbox h-4 w-4 text-primary-600 rounded focus:ring-primary-500 dark:focus:ring-offset-neutral-800/20">
                        <span class="ml-2 text-sm">Analytics & Performance</span>
                    </label>
                    <p class="text-xs text-neutral-600 dark:text-neutral-400 ml-6">Help us improve the website by collecting anonymous usage data.</p>
                </div>
            </div>
        `;

        if (!TB.ui || !TB.ui.Modal) {
            TB.logger.error('[CookieBanner] TB.ui.Modal component is not available.');
            alert('Error: Cookie settings cannot be displayed at the moment.'); // Basic fallback
            return;
        }

        this.settingsModalInstance = TB.ui.Modal.show({
            title: 'Customize Cookie Settings',
            content: modalContent,
            maxWidth: 'max-w-md',
            closeOnOutsideClick: true, // Default for Modal.js, but can be explicit
            closeOnEsc: true,        // Default for Modal.js
            customClasses: {
                modalContainer: this.options.customClasses.modalContainer || '', // Pass through custom class
                // You could add other custom classes for header, body, footer of the modal if needed
            },
            buttons: [
                {
                    text: 'Save Preferences',
                    action: (modal) => { // Modal instance is passed as 'this' or first arg by convention
                        const preferencesChecked = modal._modalElement.querySelector('#tb-cookie-preferences').checked;
                        const analyticsChecked = modal._modalElement.querySelector('#tb-cookie-analytics').checked;

                        this._saveSettingsAndHide({
                            essential: true,
                            preferences: preferencesChecked,
                            analytics: analyticsChecked,
                            source: 'modal_save'
                        });
                        modal.close(); // Close the modal itself
                    },
                    variant: 'primary',
                    // className: 'bg-primary-600 hover:bg-primary-700...' // Or rely on variant styling
                },
                {
                    text: 'Accept All',
                    action: (modal) => {
                        this._saveSettingsAndHide({
                            essential: true, preferences: true, analytics: true, source: 'modal_accept_all'
                        });
                        modal.close();
                    },
                    variant: 'secondary',
                }
            ],
            onClose: () => {
                TB.logger.log('[CookieBanner] Settings modal closed.');
                this.settingsModalInstance = null; // Allow modal to be re-created with fresh state if opened again
            }
        });
        TB.logger.log('[CookieBanner] Settings modal shown.');
    }

    _saveSettingsAndHide(consentSettings) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(consentSettings));
        TB.logger.log('[CookieBanner] Consent saved:', consentSettings);
        this._triggerConsent(consentSettings);
        this._hideBanner();
    }

    _triggerConsent(consentSettings) {
        TB.events.emit('cookieConsent:updated', consentSettings);
        if (this.options.onConsent) {
            try {
                this.options.onConsent(consentSettings);
            } catch (e) {
                TB.logger.error('[CookieBanner] Error in onConsent callback:', e);
            }
        }
    }

    _hideBanner() {
        if (this.bannerElement) {
            this.bannerElement.style.transform = 'translateY(100%)';
            this.bannerElement.style.opacity = '0';
            this.bannerElement.addEventListener('transitionend', () => {
                if (this.bannerElement && this.bannerElement.parentNode) {
                    this.bannerElement.parentNode.removeChild(this.bannerElement);
                }
                this.bannerElement = null;
                TB.events.emit('cookieBanner:hidden');
                TB.logger.log('[CookieBanner] Banner hidden and removed.');
            }, { once: true }); // Ensure listener is removed after firing

            // Fallback in case transitionend doesn't fire
            setTimeout(() => {
                if (this.bannerElement && this.bannerElement.parentNode) {
                    this.bannerElement.parentNode.removeChild(this.bannerElement);
                    this.bannerElement = null;
                    TB.events.emit('cookieBanner:hidden');
                    TB.logger.log('[CookieBanner] Banner hidden and removed (fallback).');
                }
            }, 500); // Duration slightly longer than CSS transition
        }
    }

    // Static method for convenience to initialize
    static show(options) {
        return new CookieBanner(options);
    }

    // Static method to get current consent status
    static getConsent() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                return JSON.parse(saved);
            } catch (e) {
                // Don't log error here as it's a static getter, could be noisy
                // Potentially return a specific error object or null
            }
        }
        return null;
    }

    // Static method to clear consent (for testing or user action)
    static clearConsent() {
        localStorage.removeItem(STORAGE_KEY);
        TB.logger.log('[CookieBanner] Consent cleared from localStorage.');
        TB.events.emit('cookieConsent:cleared');
    }
}

export default CookieBanner;
