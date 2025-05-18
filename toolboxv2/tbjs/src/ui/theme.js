// tbjs/src/ui/theme.js
import TB from '../index.js';

const THEME_STORAGE_KEY = 'tbjs_theme_preference';
const BACKGROUND_CONTAINER_ID = 'appBackgroundContainer'; // ID für einen dedizierten Hintergrundcontainer

// Standard-Hintergrund-Konfiguration
const DEFAULT_BACKGROUND_CONFIG = {
    type: 'color', // '3d', 'image', 'color', 'none'
    light: {
        color: '#cccccc', // Standard helle Hintergrundfarbe
        image: null,      // Pfad zum hellen Hintergrundbild
        // 3D wird vom graphics Modul gehandhabt
    },
    dark: {
        color: '#1a1a1a', // Standard dunkle Hintergrundfarbe
        image: null,      // Pfad zum dunklen Hintergrundbild
    },
    placeholder: {        // Optionales Placeholder-Bild während 3D lädt
        image: null,      // Pfad zum Placeholder-Bild
        displayUntil3DReady: true, // true: Placeholder bis 3D bereit, false: Placeholder bleibt
    }
};

const ThemeManager = {
    _currentMode: 'light',
    _preference: 'system',
    _config: {
        background: { ...DEFAULT_BACKGROUND_CONFIG }
    },
    _backgroundContainer: null,
    _is3DGraphicsReady: false, // Status, ob das 3D-Modul bereit ist

    init: (settings = {}) => {
        // Theme-Präferenz laden
        ThemeManager._preference = localStorage.getItem(THEME_STORAGE_KEY) || settings.defaultThemePreference || TB.config.get('theme.defaultPreference') || 'system';

        // Hintergrundkonfiguration mergen
        const userBgConfig = settings.background || TB.config.get('theme.background') || {};
        ThemeManager._config.background = {
            type: userBgConfig.type || DEFAULT_BACKGROUND_CONFIG.type,
            light: { ...DEFAULT_BACKGROUND_CONFIG.light, ...userBgConfig.light },
            dark: { ...DEFAULT_BACKGROUND_CONFIG.dark, ...userBgConfig.dark },
            placeholder: { ...DEFAULT_BACKGROUND_CONFIG.placeholder, ...userBgConfig.placeholder }
        };

        // Hintergrundcontainer finden oder erstellen
        ThemeManager._backgroundContainer = document.getElementById(BACKGROUND_CONTAINER_ID);
        if (!ThemeManager._backgroundContainer) {
            ThemeManager._backgroundContainer = document.createElement('div');
            ThemeManager._backgroundContainer.id = BACKGROUND_CONTAINER_ID;
            ThemeManager._backgroundContainer.style.position = 'fixed';
            ThemeManager._backgroundContainer.style.zIndex = '-1'; // Hinter allem anderen
            ThemeManager._backgroundContainer.style.top = '0';
            ThemeManager._backgroundContainer.style.left = '0';
            ThemeManager._backgroundContainer.style.width = '100%';
            ThemeManager._backgroundContainer.style.height = '100%';
            document.body.prepend(ThemeManager._backgroundContainer); // Am Anfang des Body einfügen
        }

        // Theme anwenden
        ThemeManager._applyEffectiveTheme();

        // Listener
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', ThemeManager._applyEffectiveTheme);
        TB.events.on('theme:toggle', ThemeManager.togglePreference);
        TB.events.on('theme:setPreference', ThemeManager.setPreference);
        TB.events.on('graphics:initialized', () => {
            ThemeManager._is3DGraphicsReady = true;
            ThemeManager._applyBackground(); // Hintergrund neu anwenden, falls 3D jetzt bereit ist
        });
         TB.events.on('graphics:disposed', () => {
            ThemeManager._is3DGraphicsReady = false;
            ThemeManager._applyBackground(); // Hintergrund neu anwenden, falls 3D nicht mehr bereit ist
        });


        TB.logger.log(`[Theme] Initialized. Preference: ${ThemeManager._preference}. Background type: ${ThemeManager._config.background.type}`);
    },

    _applyEffectiveTheme: () => {
        let newMode;
        if (ThemeManager._preference === 'system') {
            newMode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        } else {
            newMode = ThemeManager._preference;
        }

        if (newMode !== ThemeManager._currentMode) {
            ThemeManager._currentMode = newMode;
            document.documentElement.classList.remove('light', 'dark');
            document.documentElement.classList.add(ThemeManager._currentMode);
            document.documentElement.setAttribute('data-theme', ThemeManager._currentMode);

            TB.logger.log(`[Theme] Effective mode changed to: ${ThemeManager._currentMode}`);
            TB.events.emit('theme:changed', { mode: ThemeManager._currentMode }); // Event mit Mode-Objekt
        }
        // Hintergrund immer anwenden, da er sich auch ohne Mode-Wechsel ändern könnte (z.B. 3D wird bereit)
        ThemeManager._applyBackground();
    },

    _applyBackground: () => {
        if (!ThemeManager._backgroundContainer) return;

        const bgConfig = ThemeManager._config.background;
        const currentThemeConfig = ThemeManager._currentMode === 'dark' ? bgConfig.dark : bgConfig.light;
        const threeDCanvasContainer = document.getElementById('threeDScene'); // Standard-ID aus deinem Code

        // Zuerst alle Hintergründe "aufräumen"
        ThemeManager._backgroundContainer.style.backgroundImage = '';
        ThemeManager._backgroundContainer.style.backgroundColor = 'transparent';
        if (threeDCanvasContainer) threeDCanvasContainer.style.display = 'none';
        if (TB.graphics && typeof TB.graphics.pause === 'function') TB.graphics.pause(); // Optional: Pausiere 3D-Rendering


        let appliedBgType = 'none';

        // Placeholder-Logik
        const placeholderImage = bgConfig.placeholder.image;
        let showPlaceholder = false;

        if (placeholderImage && bgConfig.type === '3d') {
            if (bgConfig.placeholder.displayUntil3DReady && !ThemeManager._is3DGraphicsReady) {
                showPlaceholder = true;
            } else if (!bgConfig.placeholder.displayUntil3DReady) {
                // Wenn Placeholder immer angezeigt werden soll (anstelle von 3D oder wenn 3D fehlschlägt)
                // Dies überschreibt effektiv den 3D-Typ, wenn 'displayUntil3DReady' false ist.
                // Diese Logik könnte man anpassen, je nach gewünschtem Verhalten.
                // Für den Fall "immer Placeholder statt 3D" wäre bgConfig.type nicht '3d'
            }
        }


        if (showPlaceholder) {
            ThemeManager._backgroundContainer.style.backgroundImage = `url('${placeholderImage}')`;
            ThemeManager._backgroundContainer.style.backgroundSize = 'cover';
            ThemeManager._backgroundContainer.style.backgroundPosition = 'center center';
            appliedBgType = `placeholder (${placeholderImage})`;
        } else {
            // Reguläre Hintergrundanwendung
            switch (bgConfig.type) {
                case '3d':
                    if (ThemeManager._is3DGraphicsReady && threeDCanvasContainer && TB.graphics) {
                        threeDCanvasContainer.style.display = 'block';
                         if (typeof TB.graphics.resume === 'function') TB.graphics.resume(); // Optional: Fortsetzen
                        TB.graphics.updateTheme(ThemeManager._currentMode); // 3D-Szene Theme anpassen
                        appliedBgType = '3d';
                    } else {
                        // Fallback, wenn 3D nicht bereit ist oder nicht konfiguriert
                        ThemeManager._backgroundContainer.style.backgroundColor = currentThemeConfig.color;
                        appliedBgType = `color (3D fallback: ${currentThemeConfig.color})`;
                    }
                    break;
                case 'image':
                    if (currentThemeConfig.image) {
                        ThemeManager._backgroundContainer.style.backgroundImage = `url('${currentThemeConfig.image}')`;
                        ThemeManager._backgroundContainer.style.backgroundSize = 'cover';
                        ThemeManager._backgroundContainer.style.backgroundPosition = 'center center';
                        appliedBgType = `image (${currentThemeConfig.image})`;
                    } else {
                        // Fallback, wenn Bild nicht vorhanden
                        ThemeManager._backgroundContainer.style.backgroundColor = currentThemeConfig.color;
                        appliedBgType = `color (image fallback: ${currentThemeConfig.color})`;
                    }
                    break;
                case 'color':
                    ThemeManager._backgroundContainer.style.backgroundColor = currentThemeConfig.color;
                    appliedBgType = `color (${currentThemeConfig.color})`;
                    break;
                case 'none':
                default:
                    // Nichts tun, Container ist bereits transparent
                    appliedBgType = 'none';
                    break;
            }
        }
        TB.logger.log(`[Theme] Background applied: ${appliedBgType}`);
    },

    togglePreference: () => {
        let newPreference;
        if (ThemeManager._currentMode === 'dark') { // Basierend auf dem *effektiven* Modus togglen
            newPreference = 'light';
        } else {
            newPreference = 'dark';
        }
        ThemeManager.setPreference(newPreference);
    },

    setPreference: (preference) => {
        if (['light', 'dark', 'system'].includes(preference)) {
            ThemeManager._preference = preference;
            localStorage.setItem(THEME_STORAGE_KEY, preference);
            TB.logger.log(`[Theme] Preference set to: ${preference}`);
            ThemeManager._applyEffectiveTheme(); // Ruft intern auch _applyBackground auf
        } else {
            TB.logger.warn(`[Theme] Invalid theme preference: ${preference}`);
        }
    },

    // Öffentliche Getter
    getCurrentMode: () => ThemeManager._currentMode,
    getPreference: () => ThemeManager._preference,
    getBackgroundConfig: () => ({ ...ThemeManager._config.background }), // Kopie zurückgeben
};

export default ThemeManager;
