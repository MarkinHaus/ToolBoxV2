// toolboxv2/index.js

// Importiere die tbjs Bibliothek und ihr CSS
import TB from 'tbjs';
import 'tbjs/dist/tbjs.css'; // Stellt sicher, dass das CSS von tbjs geladen wird

// Importiere das Haupt-CSS deiner Anwendung (falls vorhanden und von Webpack verarbeitet)
// import './styles/main.css'; // Beispielpfad

// Globale Bereitstellung von Three.js und htmx, falls nicht schon durch <script> Tags geschehen
// Da tbjs 'three' als 'THREE' und 'htmx.org' als 'htmx' externalisiert,
// müssen diese global verfügbar sein, BEVOR TB.init() versucht, darauf zuzugreifen.
// Deine toolboxv2/index.html lädt htmx bereits per <script>.
// Für Three.js: Entweder auch per <script> in den HTML-Templates laden,
// oder wenn du es über Webpack in der Haupt-App importierst, global machen:
// import * as THREE_MODULE from 'three'; // oder from deinem Alias 'Three'
// window.THREE = THREE_MODULE;
// Besser ist es, wenn die HTML-Templates Three.js laden, wenn tbjs es so erwartet.

// Warte, bis der DOM vollständig geladen ist
document.addEventListener('DOMContentLoaded', () => {
    // Hol dir die Konfiguration für TB.init
    // Diese Werte müssen ggf. dynamisch gesetzt oder aus einer globalen Konfig gelesen werden
    const tbjsConfig = {
        appRootId: 'MainContent', // Die ID des Elements, das tbjs verwalten soll.
                                  // Deine toolboxv2/index.html hat <main id="MainContent">.
                                  // Stelle sicher, dass alle deine HTML-Dateien,
                                  // die von HtmlWebpackPlugin generiert werden und tbjs nutzen sollen,
                                  // ein solches Element haben.
        baseApiUrl: '/api',       // Entsprechend deiner Proxy-Config im devServer
        baseWsUrl: (window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host + "/talk", // Beispiel für WS
        // baseFileUrl: '/web/core0', // Basis für das Laden von HTML-Partials durch den tbjs-Router
        initialState: {
            // Dein initialer App-State
        },
        themeSettings: {
            defaultMode: 'light', // oder 'dark'
            // Weitere Theme-Einstellungen
        },
        routes: [
            // Definiere hier Routen für den tbjs-Router, falls er welche benötigt
            // z.B. { path: '/', component: 'home-component' },
            //      { path: '/login', component: 'login-component' }
        ],
        logLevel: process.env.NODE_ENV === 'production' ? 'warn' : 'debug'
    };

    // Initialisiere tbjs
    TB.init(tbjsConfig);

    TB.logger.log('TB.js wurde in der Hauptanwendung initialisiert.');

    // Deine restliche Anwendungslogik der Haupt-App kann hier starten
    // oder auf tbjs:initialized Event hören.
    // TB.events.on('tbjs:initialized', () => {
    //    console.log('TB.js ist jetzt voll einsatzbereit!');
    //    // Weitere Initialisierungsschritte der Haupt-App
    // });

    // Plattformspezifische Logik kann hier oder in TB.js Modulen erfolgen
    if (TB.env.isTauri()) {
        TB.logger.log('Läuft in Tauri-Umgebung. Desktop-spezifische Anpassungen laden...');
        // Lade Desktop-spezifische Komponenten/Assets
        // z.B. initialisiere den 3D Hintergrund
        // if (typeof TB.ui.initializeGlobalBackground === 'function') {
        //     TB.ui.initializeGlobalBackground(document.getElementById('threeDScene'));
        // }
    } else if (TB.env.isMobile()) {
        TB.logger.log('Läuft auf Mobile-Web. Mobile-spezifische Anpassungen laden...');
        // Lade Mobile-spezifische Komponenten/Assets
        // Evtl. 3D Hintergrund deaktivieren oder reduzierte Version
    } else {
        TB.logger.log('Läuft im Standard-Webbrowser. Web-spezifische Anpassungen laden...');
        // Lade Web-spezifische Komponenten/Assets
        // z.B. initialisiere den 3D Hintergrund
        // if (typeof TB.ui.initializeGlobalBackground === 'function') {
        //     TB.ui.initializeGlobalBackground(document.getElementById('threeDScene'));
        // }
    }
});
