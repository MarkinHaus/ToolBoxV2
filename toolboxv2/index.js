// toolboxv2/index.js

// Importiere die tbjs Bibliothek und ihr CSS
import TB from 'tbjs/src/index.js';
import 'tbjs/dist/tbjs.css';
import {NavMenu} from "tbjs/src/ui/index.js"; // Stellt sicher, dass das CSS von tbjs geladen wird

// Optional: Importiere das Haupt-CSS deiner Anwendung, wenn es separat existiert
// import './styles/main.css';

// Globale Abhängigkeiten (htmx, Three.js):
// Stelle sicher, dass htmx und THREE global verfügbar sind, BEVOR dieser Code ausgeführt wird.
// Das erreichst du am besten, indem du sie über <script defer src="..."></script>
// in deinen HTML-Templates (die von HtmlWebpackPlugin verarbeitet werden) einbindest.
// Die 'defer'-Attribute sorgen für die richtige Ausführungsreihenfolge.

// Definiere eine Funktion für die Initialisierung, um den globalen Scope sauber zu halten
function initializeApp() {
    const isProduction = false;
    console.log('isProduction: ', isProduction);
    const tbjsConfig = {
        appRootId: 'MainContent', // ID des Haupt-Containers für tbjs
        baseApiUrl: '/api',       // Für API-Aufrufe

        // baseFileUrl: '/web/core0', // Optional: Wenn tbjs HTML-Partials von einem bestimmten Pfad laden soll
                                    // Ansonsten leitet es sich oft von window.location ab oder ist nicht nötig.

        initialState: {
            // Hier den Anwendungs-spezifischen initialen Zustand definieren
            // z.B. userInfo: null, preferences: {}
        },
        themeSettings: {

            defaultPreference: 'system', // 'light', 'dark', oder 'system'
            background: {
                type: '3d', // '3d', 'image', 'color', 'none'
                light: {
                    color: '#E0E0E0',
                    image: '/assets/backgrounds/light-bg.jpg', // Beispielpfad
                },
                dark: {
                    color: '#212121',
                    image: '/assets/backgrounds/dark-bg.jpg',   // Beispielpfad
                },
                placeholder: {
                    image: '/assets/backgrounds/placeholder-loading.gif', // Beispiel
                    displayUntil3DReady: true // true: Placeholder bis 3D bereit, false: Placeholder wird durch konfig. Typ ersetzt
                }
            }
        },
        routes: [
            // Definiere hier die Haupt-Routen deiner Anwendung, die tbjs verwalten soll
            // Beispiel:
            // { path: '/', view: 'views/home.html', controller: async () => (await import('./controllers/homeController.js')).default },
            // { path: '/login', view: 'views/login.html', controller: async () => (await import('./controllers/loginController.js')).default },
            // { path: '/dashboard', view: 'views/dashboard.html', requiresAuth: true, controller: async () => (await import('./controllers/dashboardController.js')).default },
            // Die Struktur von 'routes' hängt davon ab, wie dein TB.router.init sie erwartet.
        ],
        logLevel: isProduction ? 'warn' : 'debug', // Im Produktivbetrieb weniger gesprächig
        // Füge hier weitere Konfigurationen hinzu, die TB.init erwartet oder deine App benötigt
    };

    // Initialisiere tbjs
    // Es ist gut, die TB-Instanz zu speichern, falls sie später direkt benötigt wird,
    // obwohl die meisten Interaktionen über ihre Module erfolgen sollten.

        // Event-Listener für erfolgreiche Initialisierung (optional, aber nützlich für Debugging oder nachgelagerte Schritte)
    TB.events.on('tbjs:initialized', (initializedTB) => {
        if (!isProduction) {
            initializedTB.logger.log('Event tbjs:initialized empfangen. Framework ist bereit.');
        }
        // Hier könnten weitere Initialisierungsschritte der Hauptanwendung erfolgen,
        // die darauf warten, dass TB vollständig bereit ist.
          // Initialisiere das Navigationsmenü, nachdem TB bereit ist  // <--- HINZUGEFÜGT
        // Die Standardoptionen in NavMenu.js verwenden '#links' als Trigger, was zu deinem HTML passt.
        const mainNavMenu = NavMenu.init({
            // Du kannst hier Optionen überschreiben, falls nötig:
            // menuContentHtml: `
            //     <ul>
            //         <li><a href="/custom-page" class="block px-3 py-2">Custom Page</a></li>
            //         <li><a hx-get="/htmx-content" hx-target="#MainContent" class="block px-3 py-2">HTMX Link</a></li>
            //     </ul>
            // `,
            // openIconClass: 'menu_open', // Falls du ein anderes Icon möchtest
            // customClasses: { // Um die Tailwind-ähnlichen Klassen zu überschreiben, falls dein CSS andere Namen verwendet
            //    overlay: 'my-custom-overlay-class',
            //    menuContainer: 'my-custom-menu-container-class'
            // }
        });

        // Annahme: TB.core.threeSetup ist ein Modul/Funktion, die Three.js initialisiert
        // und ein Objekt mit { renderer, scene, ambientLight, pointLights } zurückgibt oder speichert
         if (initializedTB.ui.darkModeToggle) {
            initializedTB.ui.darkModeToggle.init('#darkModeToggleContainer');
        }
        // Optional: Speichere die Instanz, wenn du später darauf zugreifen musst
        // initializedTB.mainNavMenu = mainNavMenu;
        initializedTB.logger.info('[App] NavMenu initialized.'); // <--- HINZUGEFÜGT
        loadPlatformSpecificFeatures(initializedTB);
        console.log("NavMenu initialized");
    });

    window.AppTB = TB.init(tbjsConfig); // window.AppTB ist optional, TB selbst ist ja schon importiert

    if (!isProduction) {
        AppTB.logger.log('TB.js wurde in der Hauptanwendung initialisiert (Development Mode).');
    }

    if (TB.ui.theme.getBackgroundConfig().type === '3d') {
        TB.graphics.init('#threeDScene', { /* graphics options */ });
    }

    // Fehlerbehandlung für die Initialisierung (optional, aber gut für Robustheit)
    // Dies hängt davon ab, ob TB.init Fehler wirft oder ein Fehler-Event auslöst
    // try {
    //     TB.init(tbjsConfig);
    // } catch (error) {
    //     console.error("Fehler bei der Initialisierung von TB.js:", error);
    //     // Zeige eine Fehlermeldung im UI an, statt die App abstürzen zu lassen
    //     document.body.innerHTML = '<p>Fehler beim Laden der Anwendung. Bitte versuchen Sie es später erneut.</p>';
    // }
}

function loadPlatformSpecificFeatures(currentTB) {
    const threeDSceneElement = document.getElementById('threeDScene');

    if (currentTB.env.isTauri()) {
        currentTB.logger.info('Tauri-Umgebung erkannt. Lade Desktop-spezifische Features...');
        // Beispiel: 3D-Hintergrund immer für Desktop-Tauri
        if (threeDSceneElement && typeof currentTB.ui.initializeGlobalBackground === 'function') {
            currentTB.ui.initializeGlobalBackground(threeDSceneElement);
        } else {
            currentTB.logger.warn('initializeGlobalBackground Funktion nicht gefunden oder threeDSceneElement fehlt.');
        }
        // Weitere Tauri-spezifische Initialisierungen
        // z.B. Einrichten von Event-Listenern für Tauri-Events
        // async function setupTauriListeners() {
        //   const { listen } = await import('@tauri-apps/api/event');
        //   await listen('my-custom-event', (event) => {
        //     currentTB.logger.log('Tauri event received:', event.payload);
        //   });
        // }
        // setupTauriListeners();

    } else if (currentTB.env.isWeb()) {
        currentTB.logger.info('Web-Umgebung erkannt.');
        if (currentTB.env.isMobile()) {
            currentTB.logger.info('Mobile Web-Client. Lade optimierte Features...');
            // Für Mobile evtl. keinen oder einen einfacheren Hintergrund
            if (threeDSceneElement) {
                threeDSceneElement.style.display = 'none'; // Beispiel: 3D-Szene ausblenden
            }
            // Weitere Mobile-spezifische Anpassungen
        } else {
            currentTB.logger.info('Desktop Web-Client. Lade volle Features...');
            // Für Desktop-Web den 3D-Hintergrund (wenn Performance es zulässt - evtl. hier eine zusätzliche Prüfung)
            if (threeDSceneElement && typeof currentTB.ui.initializeGlobalBackground === 'function') {
                // Optionale Performance-Prüfung für Web
                // const canRun3D = !navigator.connection?.saveData && window.matchMedia('(min-width: 768px)').matches;
                // if (canRun3D) {
                //    currentTB.ui.initializeGlobalBackground(threeDSceneElement);
                // } else {
                //    currentTB.logger.info('3D background skipped due to performance considerations or small screen.');
                //    if (threeDSceneElement) threeDSceneElement.style.display = 'none';
                // }
                currentTB.ui.initializeGlobalBackground(threeDSceneElement); // Vorerst immer laden
            } else {
                currentTB.logger.warn('initializeGlobalBackground Funktion nicht gefunden oder threeDSceneElement fehlt.');
            }
        }
        // Allgemeine Web-spezifische Initialisierungen (z.B. Service Worker, falls genutzt)
        // if ('serviceWorker' in navigator && isProduction) {
        //   window.addEventListener('load', () => {
        //     navigator.serviceWorker.register('/service-worker.js')
        //       .then(registration => currentTB.logger.info('ServiceWorker registration successful: ', registration.scope))
        //       .catch(err => currentTB.logger.error('ServiceWorker registration failed: ', err));
        //   });
        // }
    }
}

// Stelle sicher, dass der DOM geladen ist, bevor die App initialisiert wird
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOMContentLoaded wurde bereits ausgelöst
    initializeApp();
}
