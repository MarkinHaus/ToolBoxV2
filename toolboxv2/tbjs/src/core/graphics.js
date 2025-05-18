// tbjs/src/core/graphics.js
import * as THREE from 'three'; // Importiere THREE.js direkt
                                // Stelle sicher, dass 'three' in tbjs/package.json (peer)Dependencies
                                // und in tbjs/webpack.config.js als 'external' deklariert ist.
import TB from '../index.js'; // Zugriff auf TB-Core

// Private Modulvariablen
let renderer, scene, camera, mainGroup;
let ambientLightInstance, pointLightInstances = [];
let isInitialized = false;
let isPaused = false;
let animationFrameId;

// Konfigurierbare Parameter (könnten über TB.config oder init-Optionen kommen)
const INITIAL_CAMERA_Z = 10;
const INITIAL_CAMERA_Y = 3.2;
const SIERPINSKI_DEPTH = 5;
const SIERPINSKI_SIZE = 12;
const CAMERA_ZOOM_MIN = -2;
const CAMERA_ZOOM_MAX = 12;
const ZOOM_STEP_FACTOR = 0.08; // Ersetzt 'sk' und 'sk2'

// Animationsparameter
let animParams = {
    factor: 12, // animantionFactorIdeal
    factorClick: 8,
    x: 0.002,
    y: 0.002,
    z: 0.002,
    isMouseDown: false,
    startX: 0,
    startY: 0,
};

// Material
const milkGlassMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0.5,
    roughness: 0.5,
    transparent: true,
    opacity: 0.6,
    clearcoat: 0.5,
    clearcoatRoughness: 0.8,
    reflectivity: 0.4,
});

/**
 * Initialisiert die 3D-Grafikszene.
 * @param {string} canvasContainerSelector - CSS-Selektor des DOM-Elements, das den Canvas enthalten soll.
 * @param {object} [options={}] - Zusätzliche Konfigurationsoptionen.
 */
export function init(canvasContainerSelector, options = {}) {
    if (isInitialized) {
        TB.logger.warn('Graphics: Bereits initialisiert.');
        return getContext();
    }

    const container = document.querySelector(canvasContainerSelector);
    if (!container) {
        TB.logger.error(`Graphics: Container "${canvasContainerSelector}" nicht gefunden.`);
        return null;
    }

    // 1. Renderer erstellen
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); // alpha: true für transparenten Hintergrund
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio); // Für schärfere Darstellung auf HiDPI-Displays
    container.innerHTML = ''; // Vorherigen Inhalt leeren
    container.appendChild(renderer.domElement);

    // 2. Szene erstellen
    scene = new THREE.Scene();

    // 3. Kamera erstellen
    camera = new THREE.PerspectiveCamera(90, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, options.cameraY || INITIAL_CAMERA_Y, options.cameraZ || INITIAL_CAMERA_Z);
    scene.add(camera); // Kamera zur Szene hinzufügen, falls sie sich bewegt oder fokussiert

    // 4. Hauptgeometrie (Sierpinski) erstellen und hinzufügen
    mainGroup = createSierpinski(SIERPINSKI_DEPTH, SIERPINSKI_SIZE);
    scene.add(mainGroup);

    // 5. Lichter erstellen (initial für den aktuellen Theme-Modus)
    // Die eigentliche Farbsetzung der Lichter wird vom darkModeToggle-Modul gesteuert
    // Hier nur Platzhalter oder Standardlichter
    ambientLightInstance = new THREE.AmbientLight(0xffffff, 0.5); // Temporäre Farbe/Intensität
    scene.add(ambientLightInstance);

    const pl1 = new THREE.PointLight(0xffffff, 0.8, 100); // Temporäre Farbe/Intensität
    pl1.position.set(5, 5, 5);
    scene.add(pl1);
    pointLightInstances.push(pl1);

    const pl2 = new THREE.PointLight(0xffffff, 0.8, 100); // Temporäre Farbe/Intensität
    pl2.position.set(-5, -5, -5);
    scene.add(pl2);
    pointLightInstances.push(pl2);

    // 6. Event-Listener für Interaktionen und Fenstergröße
    _addEventListeners();

    isInitialized = true;
    TB.logger.log('Graphics: Initialisiert.');

    // 7. Animationsloop starten
    _animate();

    // 8. Kontext für darkModeToggle bereitstellen und Theme anwenden
    const currentTheme = TB.ui.theme ? TB.ui.theme.getCurrentMode() : 'light'; // Annahme: TB.ui.theme existiert
    updateTheme(currentTheme); // Wende das initiale Theme an

    // Service Worker Deinstallation (wenn wirklich gewünscht)
    if (options.uninstallServiceWorkers) {
        _uninstallServiceWorkers();
    }

    // Loader ausblenden (sollte idealerweise über TB.events gesteuert werden)
    setTimeout(() => {
        const loader = document.querySelector('.loaderCenter'); // Sei spezifischer mit dem Selektor
        if (loader) loader.style.display = 'none';
    }, options.loaderHideDelay || 150);


    TB.events.emit('graphics:initialized', getContext());
    return getContext();
}

/**
 * Gibt den aktuellen Grafikkontext zurück.
 */
export function getContext() {
    if (!isInitialized) return null;
    return {
        renderer,
        scene,
        camera,
        ambientLightInstance,
        pointLightInstances,
    };
}

/**
 * Passt die 3D-Szene an das gegebene Theme an (hell/dunkel).
 * Diese Funktion wird vom darkModeToggle-Modul aufgerufen.
 * @param {string} theme - 'light' oder 'dark'
 */
export function updateTheme(theme) {
    if (!isInitialized) return;

    let clearColor, ambientColorHex, pointColorHex;

    if (theme === 'dark') {
        clearColor = 0x000000;
        ambientColorHex = 0x181823;
        pointColorHex = 0x404060; // Gedämpfter für dunklen Modus
    } else { // light
        clearColor = 0xcccccc;
        ambientColorHex = 0x7070B0; // Heller, leicht bläulich
        pointColorHex = 0xffffff;
    }

    renderer.setClearColor(clearColor);

    if (ambientLightInstance) {
        ambientLightInstance.color.setHex(ambientColorHex);
    }

    pointLightInstances.forEach(pl => {
        pl.color.setHex(pointColorHex);
    });

    TB.logger.log(`Graphics: Theme auf "${theme}" aktualisiert.`);
}


/**
 * Interne Animationsfunktion.
 */
function _animate() {
    if (isPaused) return;
    animationFrameId = requestAnimationFrame(_animate);

    if (mainGroup) {
        mainGroup.rotation.x += animParams.x / animParams.factor;
        mainGroup.rotation.y += animParams.y / animParams.factor;
        mainGroup.rotation.z += animParams.z / animParams.factor;
    }

    renderer.render(scene, camera);
}

/**
 * Erstellt die Sierpinski-Tetraeder-Geometrie.
 */
function createSierpinski(depth, size) {
    const rootGroup = new THREE.Group();
    let collectedSubGroups = []; // Für die alte 'groops' Logik, falls noch benötigt

    function subdivide(currentDepth, currentSize, parentGroup) {
        if (currentDepth === 0) {
            const tetra = new THREE.TetrahedronGeometry(currentSize * 0.6);
            const mesh = new THREE.Mesh(tetra, milkGlassMaterial);
            parentGroup.add(mesh);
            // mashs.push(mesh); // Globale 'mashs' vermeiden, wenn möglich
            return;
        }

        const newSize = currentSize / 2;
        const newDepth = currentDepth - 1;
        const offsetFactor = Math.sqrt(3) / 4; // Anpassung für Tetraeder-Layout

        // Positionen der 4 Kind-Tetraeder
        const positions = [
            new THREE.Vector3(-newSize * offsetFactor, -newSize * offsetFactor / Math.sqrt(3), -newSize * offsetFactor / Math.sqrt(6)),
            new THREE.Vector3( newSize * offsetFactor, -newSize * offsetFactor / Math.sqrt(3), -newSize * offsetFactor / Math.sqrt(6)),
            new THREE.Vector3( 0,                      2*newSize * offsetFactor / Math.sqrt(3), -newSize * offsetFactor / Math.sqrt(6)),
            new THREE.Vector3( 0,                      0,                                       3*newSize * offsetFactor / Math.sqrt(6))
        ];


        positions.forEach(pos => {
            const subGroup = new THREE.Group();
            subGroup.position.copy(pos);
            subdivide(newDepth, newSize, subGroup);
            parentGroup.add(subGroup);
            if (currentDepth > 1) { // Wie deine alte Bedingung, falls für spezifische Animationen
                collectedSubGroups.push(subGroup);
            }
        });
    }

    subdivide(depth, size, rootGroup);
    // Hier könntest du collectedSubGroups an eine andere Stelle im Modul übergeben,
    // falls die Subgruppen separat animiert werden sollen.
    // Aktuell rotiert nur `mainGroup`.
    return rootGroup;
}

/**
 * Setzt die Animationsparameter für die Rotation.
 * @param {number} x - Rotationsgeschwindigkeit um X.
 * @param {number} y - Rotationsgeschwindigkeit um Y.
 * @param {number} z - Rotationsgeschwindigkeit um Z.
 * @param {number} [factor] - Animationsfaktor.
 */
export function setAnimationSpeed(x, y, z, factor) {
    animParams.x = x;
    animParams.y = y;
    animParams.z = z;
    if (factor !== undefined) {
        animParams.factor = factor;
    }
}

/**
 * Passt den Kamera-Zoom an.
 * @param {number} deltaZoom - Änderung des Zoomlevels.
 */
export function adjustCameraZoom(deltaZoom) {
    if (!isInitialized) return;
    camera.position.z += deltaZoom * ZOOM_STEP_FACTOR;
    // Zoom-Grenzen einhalten
    camera.position.z = Math.max(CAMERA_ZOOM_MIN, Math.min(CAMERA_ZOOM_MAX, camera.position.z));
    camera.updateProjectionMatrix();
}


function _handleResize() {
    if (!isInitialized) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    TB.logger.log('Graphics: Fenstergröße angepasst.');
}

function _handleMouseDown(event) {
    animParams.isMouseDown = true;
    animParams.factor = animParams.factorClick;
    const { clientX, clientY } = (event.touches ? event.touches[0] : event);
    animParams.startX = clientX;
    animParams.startY = clientY;

    const normX = (clientX / window.innerWidth) * 2 - 1;
    const normY = -((clientY / window.innerHeight) * 2 - 1);
    animParams.x = normY * 0.02; // Initial rotation
    animParams.y = normX * 0.02;
    animParams.z = (normX + normY) * 0.01;
    event.preventDefault(); // Verhindert Scrollen bei Touch auf Canvas
}

function _handleMouseMove(event) {
    if (!animParams.isMouseDown || !isInitialized) return;
    const { clientX, clientY } = (event.touches ? event.touches[0] : event);
    const deltaX = clientX - animParams.startX;
    const deltaY = clientY - animParams.startY;

    animParams.x = (deltaY / window.innerHeight) * Math.PI / 20; // Skalierung angepasst
    animParams.y = (deltaX / window.innerWidth) * Math.PI / 20;
    animParams.z = ((deltaX + deltaY) / (window.innerWidth + window.innerHeight)) * Math.PI / 20;
}

function _handleMouseUp() {
    if (!isInitialized) return;
    animParams.isMouseDown = false;
    animParams.factor = 12; // animantionFactorIdeal
    animParams.x = 0.002; // Reset to default spin
    animParams.y = 0.002;
    animParams.z = 0.002;
}

function _handleWheel(event) {
    if (!isInitialized) return;
    const delta = event.deltaY > 0 ? -1 : 1; // Umgekehrte Richtung für Zoom-Effekt
    adjustCameraZoom(delta);
    event.preventDefault();
}

function _addEventListeners() {
    window.addEventListener('resize', _handleResize);

    const canvasElement = renderer.domElement;
    canvasElement.addEventListener('mousedown', _handleMouseDown);
    canvasElement.addEventListener('touchstart', _handleMouseDown, { passive: false });

    // Mousemove und mouseup auf document, um Bewegung außerhalb des Canvas zu erfassen
    document.addEventListener('mousemove', _handleMouseMove);
    document.addEventListener('touchmove', _handleMouseMove, { passive: false });

    document.addEventListener('mouseup', _handleMouseUp);
    document.addEventListener('touchend', _handleMouseUp);

    canvasElement.addEventListener('wheel', _handleWheel, { passive: false });

    // Alte Slider-Logik (Beispiel, wie man sie integrieren könnte, wenn sie noch gebraucht wird)
    // Diese müssten Events auslösen, auf die dieses Modul hört, oder direkt Funktionen aufrufen
    // document.getElementById('slideX')?.addEventListener('change', (e) => updateRotation('x', e.target.value));
}

function _removeEventListeners() {
    window.removeEventListener('resize', _handleResize);
    if (renderer && renderer.domElement) {
        const canvasElement = renderer.domElement;
        canvasElement.removeEventListener('mousedown', _handleMouseDown);
        canvasElement.removeEventListener('touchstart', _handleMouseDown);
        canvasElement.removeEventListener('wheel', _handleWheel);
    }
    document.removeEventListener('mousemove', _handleMouseMove);
    document.removeEventListener('touchmove', _handleMouseMove);
    document.removeEventListener('mouseup', _handleMouseUp);
    document.removeEventListener('touchend', _handleMouseUp);
}

function _uninstallServiceWorkers() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(registrations => {
            registrations.forEach(registration => {
                registration.unregister().then(success => {
                    if (success) {
                        TB.logger.log(`ServiceWorker ${registration.scope} erfolgreich entfernt.`);
                    } else {
                        TB.logger.warn(`Fehler beim Entfernen von ServiceWorker ${registration.scope}.`);
                    }
                });
            });
        }).catch(err => TB.logger.error('Fehler beim Abrufen der ServiceWorker-Registrierungen:', err));
    }
}


/**
 * Räumt die Grafikressourcen auf und entfernt Event-Listener.
 */
export function dispose() {
    if (!isInitialized) return;

    _removeEventListeners();
    cancelAnimationFrame(animationFrameId);

    if (scene) {
        // Alle Objekte aus der Szene entfernen und disposen
        scene.traverse(object => {
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        scene.clear(); // Entfernt alle Objekte
    }

    if (renderer) {
        renderer.dispose();
        renderer.domElement.remove(); // DOM-Element entfernen
    }

    // Variablen zurücksetzen
    renderer = null;
    scene = null;
    camera = null;
    mainGroup = null;
    ambientLightInstance = null;
    pointLightInstances = [];
    isInitialized = false;
    TB.events.emit('graphics:disposed'); // Neues Event
    TB.logger.log('Graphics: Ressourcen freigegeben und Modul deinitialisiert.');
}

// Optionale Pause/Resume-Funktionen
export function pause() {
    if (isInitialized && !isPaused) {
        cancelAnimationFrame(animationFrameId);
        isPaused = true;
        TB.logger.log('Graphics: Rendering paused.');
    }
}
export function resume() {
    if (isInitialized && isPaused) {
        isPaused = false;
        _animate(); // Animationsloop neu starten
        TB.logger.log('Graphics: Rendering resumed.');
    }
}
