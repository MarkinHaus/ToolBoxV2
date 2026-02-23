// tbjs/ui/components/CookieBanner/CookieBanner.js
import TB from '../../../index.js';
import { setLogConsent, getLogConsent } from '../../../core/logger.js';

const STORAGE_KEY = 'tbjs_cookie_consent';

/**
 * Map checkbox state → logger consent level (stored in cookie `tb_log_consent`).
 *
 *   analytics + preferences  → 'all'      (DEBUG, INFO, WARN, ERROR, AUDIT)
 *   analytics only           → 'essential' (WARN, ERROR, AUDIT)
 *   preferences only         → 'errors'   (ERROR, AUDIT)
 *   neither                  → 'none'     (nothing sent)
 */
function consentToLogLevel(settings) {
    if (settings.analytics && settings.preferences) return 'all';
    if (settings.analytics) return 'essential';
    if (settings.preferences) return 'errors';
    return 'none';
}

/**
 * Reverse: logger consent level → checkbox state (for modal pre-fill).
 */
function logLevelToConsent(level) {
    switch (level) {
        case 'all':       return { analytics: true,  preferences: true };
        case 'essential': return { analytics: true,  preferences: false };
        case 'errors':    return { analytics: false, preferences: true };
        default:          return { analytics: false, preferences: false };
    }
}

function onConsentDefault(consentSettings) {
    const level = consentToLogLevel(consentSettings);
    setLogConsent(level);
    TB.logger.info('[CookieBanner] Log consent set to:', level);
}

const DEFAULT_COOKIE_BANNER_OPTIONS = {
    title: 'Wir respektieren deine Privatsphäre.',
    message: 'Diese Seite verwendet Cookies und sendet anonymisierte Nutzungsdaten zur Verbesserung. Du entscheidest, was gesendet wird.',
    termsLink: '/web/assets/terms.html',
    termsLinkText: 'Nutzungsbedingungen',
    acceptMinimalText: 'Akzeptieren',
    showAdvancedOptions: true,
    advancedOptionsText: 'Einstellungen',
    onConsent: onConsentDefault,
    customClasses: {
        banner: '',
        modalContainer: '',
    },
    defaultPreferences: true,
    defaultAnalytics: true,
};


class CookieBanner {
    constructor(options = {}) {
        this.options = {
            ...DEFAULT_COOKIE_BANNER_OPTIONS,
            ...options,
            customClasses: {
                ...DEFAULT_COOKIE_BANNER_OPTIONS.customClasses,
                ...(options.customClasses || {}),
            }
        };

        this.bannerElement = null;
        this.settingsModalInstance = null;
        this._init();
    }

    _init() {
        const savedSettings = this._getSavedSettings();
        if (savedSettings) {
            this._triggerConsent(savedSettings);
            return;
        }
        this._createBanner();
    }

    _getSavedSettings() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                return JSON.parse(saved);
            } catch (e) {
                TB.logger.error('[CookieBanner] Error parsing saved settings:', e);
                localStorage.removeItem(STORAGE_KEY);
            }
        }
        return null;
    }

    _getInitialPreferenceState(type, defaultValue) {
        // First try localStorage (previous explicit choice)
        const savedSettings = this._getSavedSettings();
        if (savedSettings && savedSettings[type] !== undefined) {
            return savedSettings[type];
        }
        // Then try to infer from existing log consent cookie
        const logLevel = getLogConsent();
        if (logLevel !== 'none') {
            const inferred = logLevelToConsent(logLevel);
            if (inferred[type] !== undefined) return inferred[type];
        }
        return defaultValue;
    }

    _createBanner() {
        if (this.bannerElement) return;

        this.bannerElement = document.createElement('div');
        this.bannerElement.id = 'tbjs-cookie-banner';
        this.bannerElement.innerHTML = `
            <div>
                <div class="flex-grow">
                    <span id="tb-cookie-banner-close" title="Schließen" class="material-symbols-outlined" style="cursor:pointer">close</span>
                    <h3>${this.options.title}</h3>
                    <p>
                        ${this.options.message}
                        ${this.options.termsLink ? `<a href="${this.options.termsLink}" style="text-decoration:underline;margin-left:4px" target="_blank" rel="noopener noreferrer">${this.options.termsLinkText}</a>` : ''}
                    </p>
                </div>
                <div style="flex-shrink:0;display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center">
                    ${this.options.showAdvancedOptions ? `<button id="tb-cookie-banner-show-settings">${this.options.advancedOptionsText}</button>` : ''}
                    <button id="tb-cookie-banner-accept-minimal">${this.options.acceptMinimalText}</button>
                </div>
            </div>
        `;
        document.body.appendChild(this.bannerElement);

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
        const acceptBtn = this.bannerElement.querySelector('#tb-cookie-banner-accept-minimal');
        if (acceptBtn) {
            acceptBtn.addEventListener('click', () => {
                this._saveSettingsAndHide({
                    analytics: true, preferences: true, essential: true, source: 'accept_minimal'
                });
            });
        }

        const closeBtn = this.bannerElement.querySelector('#tb-cookie-banner-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                // Close = nur essentials (WARN + ERROR + AUDIT)
                this._saveSettingsAndHide({
                    analytics: true, preferences: false, essential: true, source: 'close_banner'
                });
            });
        }

        if (this.options.showAdvancedOptions) {
            const settingsBtn = this.bannerElement.querySelector('#tb-cookie-banner-show-settings');
            if (settingsBtn) {
                settingsBtn.addEventListener('click', () => this._showSettingsModal());
            }
        }
    }

    _showSettingsModal() {
        if (this.settingsModalInstance && this.settingsModalInstance.isOpen) return;

        const initialPreferences = this._getInitialPreferenceState('preferences', this.options.defaultPreferences);
        const initialAnalytics = this._getInitialPreferenceState('analytics', this.options.defaultAnalytics);

        const modalContent = `
            <div style="display:flex;flex-direction:column;gap:0.75rem">
                <div>
                    <label style="display:flex;align-items:center;gap:0.5rem">
                        <input type="checkbox" id="tb-cookie-essential" checked disabled>
                        <span>Essenzielle Cookies</span>
                    </label>
                    <p style="font-size:0.75rem;opacity:0.7;margin-left:1.5rem">Immer aktiv — für grundlegende Funktionen nötig.</p>
                </div>
                <div>
                    <label style="display:flex;align-items:center;gap:0.5rem">
                        <input type="checkbox" id="tb-cookie-preferences" ${initialPreferences ? 'checked' : ''}>
                        <span>Fehler-Logging</span>
                    </label>
                    <p style="font-size:0.75rem;opacity:0.7;margin-left:1.5rem">Fehlerberichte werden an den Server gesendet, um Probleme zu beheben.</p>
                </div>
                <div>
                    <label style="display:flex;align-items:center;gap:0.5rem">
                        <input type="checkbox" id="tb-cookie-analytics" ${initialAnalytics ? 'checked' : ''}>
                        <span>Nutzungsanalyse</span>
                    </label>
                    <p style="font-size:0.75rem;opacity:0.7;margin-left:1.5rem">Anonymisierte Nutzungsdaten (Warnungen, Audit-Events) helfen uns, die Seite zu verbessern.</p>
                </div>
                <div style="margin-top:0.5rem;padding:0.5rem;border-radius:0.375rem;background:rgba(128,128,128,0.1);font-size:0.7rem;opacity:0.8">
                    <strong>Was wird gesendet?</strong><br>
                    Alle + Analyse → alle Logs &amp; Audit-Events<br>
                    Nur Analyse → Warnungen, Fehler &amp; Audit<br>
                    Nur Fehler → nur Fehlermeldungen &amp; Audit<br>
                    Keines → nichts wird gesendet
                </div>
            </div>
        `;

        if (!TB.ui || !TB.ui.Modal) {
            TB.logger.error('[CookieBanner] TB.ui.Modal not available.');
            return;
        }

        this.settingsModalInstance = TB.ui.Modal.show({
            title: 'Datenschutz-Einstellungen',
            content: modalContent,
            maxWidth: 'max-w-md',
            closeOnOutsideClick: true,
            closeOnEsc: true,
            customClasses: {
                modalContainer: this.options.customClasses.modalContainer || '',
            },
            buttons: [
                {
                    text: 'Speichern',
                    action: (modal) => {
                        const preferencesChecked = modal._modalElement.querySelector('#tb-cookie-preferences').checked;
                        const analyticsChecked = modal._modalElement.querySelector('#tb-cookie-analytics').checked;
                        this._saveSettingsAndHide({
                            essential: true,
                            preferences: preferencesChecked,
                            analytics: analyticsChecked,
                            source: 'modal_save',
                        });
                        modal.close();
                    },
                    variant: 'primary',
                },
                {
                    text: 'Alles akzeptieren',
                    action: (modal) => {
                        this._saveSettingsAndHide({
                            essential: true, preferences: true, analytics: true, source: 'modal_accept_all',
                        });
                        modal.close();
                    },
                    variant: 'secondary',
                },
                {
                    text: 'Alles ablehnen',
                    action: (modal) => {
                        this._saveSettingsAndHide({
                            essential: true, preferences: false, analytics: false, source: 'modal_reject_all',
                        });
                        modal.close();
                    },
                    variant: 'ghost',
                },
            ],
            onClose: () => {
                this.settingsModalInstance = null;
            },
        });
    }

    _saveSettingsAndHide(consentSettings) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(consentSettings));
        this._triggerConsent(consentSettings);
        this._hideBanner();
    }

    _triggerConsent(consentSettings) {
        TB.events.emit('cookieConsent:updated', consentSettings);
        if (this.options.onConsent) {
            try {
                this.options.onConsent(consentSettings);
            } catch (e) {
                TB.logger.error('[CookieBanner] onConsent callback error:', e);
            }
        }
    }

    _hideBanner() {
        if (!this.bannerElement) return;

        this.bannerElement.style.transform = 'translateY(100%)';
        this.bannerElement.style.opacity = '0';

        const cleanup = () => {
            if (this.bannerElement && this.bannerElement.parentNode) {
                this.bannerElement.parentNode.removeChild(this.bannerElement);
            }
            this.bannerElement = null;
            TB.events.emit('cookieBanner:hidden');
        };

        this.bannerElement.addEventListener('transitionend', cleanup, { once: true });
        setTimeout(() => { if (this.bannerElement) cleanup(); }, 500);
    }

    /** Show banner (convenience static). */
    static show(options) {
        return new CookieBanner(options);
    }

    /** Get current consent from localStorage. */
    static getConsent() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try { return JSON.parse(saved); } catch { /* corrupt */ }
        }
        return null;
    }

    /** Get current log consent level from cookie. */
    static getLogLevel() {
        return getLogConsent();
    }

    /** Update consent programmatically (e.g. from a settings page). */
    static updateConsent(settings) {
        const merged = { essential: true, source: 'programmatic', ...settings };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
        const level = consentToLogLevel(merged);
        setLogConsent(level);
        TB.events.emit('cookieConsent:updated', merged);
        return level;
    }

    /** Clear all consent — banner will show again on next page load. */
    static clearConsent() {
        localStorage.removeItem(STORAGE_KEY);
        setLogConsent('none');
        TB.events.emit('cookieConsent:cleared');
    }
}

export default CookieBanner;
