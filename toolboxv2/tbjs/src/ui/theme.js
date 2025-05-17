// tbjs/ui/theme.js
// Manages UI themes, primarily dark/light mode.
// Original: initDome dark mode part, toggleDarkMode, loadDarkModeState from original index.js & scripts.js

import TB from '../index.js'; // Full TB object for config, events, logger

const THEME_STORAGE_KEY = 'tbjs_theme_preference';

const ThemeManager = {
    _currentMode: 'light', // 'light', 'dark', or 'system' determined effective mode
    _preference: 'system', // 'light', 'dark', or 'system'

    init: (settings = {}) => {
        ThemeManager._preference = localStorage.getItem(THEME_STORAGE_KEY) || settings.defaultMode || TB.config.get('defaultTheme') || 'system';
        ThemeManager.applyTheme();

        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', ThemeManager.applyTheme);
        TB.events.on('theme:toggle', ThemeManager.toggleMode); // For programatic toggle
        TB.events.on('theme:setPreference', ThemeManager.setPreference);

        TB.logger.log(`[Theme] Initialized with preference: ${ThemeManager._preference}`);
    },

    applyTheme: () => {
        let newMode;
        if (ThemeManager._preference === 'system') {
            newMode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        } else {
            newMode = ThemeManager._preference;
        }

        if (newMode !== ThemeManager._currentMode) {
            ThemeManager._currentMode = newMode;
            document.documentElement.classList.remove('light', 'dark'); // Using html tag for global theme
            document.documentElement.classList.add(ThemeManager._currentMode);
            document.documentElement.setAttribute('data-theme', ThemeManager._currentMode); // For compatibility with old system

            TB.logger.log(`[Theme] Applied theme: ${ThemeManager._currentMode}`);
            TB.events.emit('theme:changed', ThemeManager._currentMode);
        }
    },

    toggleMode: () => {
        // Toggles between light and dark, and sets preference accordingly.
        // If current preference is 'system', it picks the opposite of current effective mode.
        let newPreference;
        if (ThemeManager._currentMode === 'dark') {
            newPreference = 'light';
        } else {
            newPreference = 'dark';
        }
        ThemeManager.setPreference(newPreference);
    },

    setPreference: (preference) => { // 'light', 'dark', or 'system'
        if (['light', 'dark', 'system'].includes(preference)) {
            ThemeManager._preference = preference;
            localStorage.setItem(THEME_STORAGE_KEY, preference);
            TB.logger.log(`[Theme] Preference set to: ${preference}`);
            ThemeManager.applyTheme();
        } else {
            TB.logger.warn(`[Theme] Invalid theme preference: ${preference}`);
        }
    },

    getCurrentMode: () => ThemeManager._currentMode,
    getPreference: () => ThemeManager._preference,
};

export default ThemeManager;
