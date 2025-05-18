// tbjs/src/ui/components/darkModeToggle.js
import TB from '../../../index.js';

const DEFAULT_SELECTOR = '#darkModeToggleContainer';
const ICON_SELECTOR = '.tb-toggle-icon'; // In deinem HTML z.B. <span class="tb-toggle-icon material-symbols-outlined"></span>
const CHECKBOX_SELECTOR = '#darkModeSwitch'; // Versteckte Checkbox

export function init(selector = DEFAULT_SELECTOR) {
    const toggleContainer = document.querySelector(selector);
    if (!toggleContainer) {
        TB.logger.warn(`[DarkModeToggle] Container "${selector}" not found.`);
        return;
    }

    const toggleIconElement = toggleContainer.querySelector(ICON_SELECTOR);
    const checkboxElement = document.querySelector(CHECKBOX_SELECTOR); // Globale Checkbox

    function updateToggleVisuals(themeMode) {
        if (toggleIconElement) {
            toggleIconElement.textContent = themeMode === 'dark' ? 'dark_mode' : 'light_mode';
            // Rotationslogik: Setze Klasse basierend auf dem aktiven Modus
            toggleIconElement.classList.toggle('tb-rotated', themeMode === 'dark');
        }
        if (checkboxElement) {
            checkboxElement.checked = themeMode === 'dark';
        }
    }

    // Initialen Zustand setzen
    updateToggleVisuals(TB.ui.theme.getCurrentMode());

    // Auf Theme-Änderungen hören, um UI zu aktualisieren (falls von woanders geändert)
    TB.events.on('theme:changed', (eventData) => { // eventData ist { mode: 'dark' }
        updateToggleVisuals(eventData.mode);
    });

    // Event Listener für den Klick auf den Container
    toggleContainer.addEventListener('click', () => {
        TB.ui.theme.togglePreference(); // ThemeManager kümmert sich um den Rest
        // Die UI wird durch das 'theme:changed' Event aktualisiert
    });

    // Optional: Auf Änderungen an der (versteckten) Checkbox hören
    if (checkboxElement) {
        checkboxElement.addEventListener('change', (event) => {
            const newPreference = event.target.checked ? 'dark' : 'light';
            TB.ui.theme.setPreference(newPreference);
        });
    }

    TB.logger.log(`[DarkModeToggle] Initialized for element: "${selector}"`);
}
