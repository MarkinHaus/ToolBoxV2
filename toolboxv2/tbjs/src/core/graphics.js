// tbjs/src/core/graphics.js
import * as THREE from 'three';
import TB from '../index.js';

let renderer, scene, camera, mainGroup;
let ambientLightInstance, pointLightInstances = [];
let isInitialized = false;
let isPaused = false;
let animationFrameId;
let themeChangeHandler = null;

let subGroupsToAnimate = []; // To store sub-groups for independent animation

// Constants from old script & merged
const INITIAL_CAMERA_Z = 10;
const INITIAL_CAMERA_Y = 3.2;
let currentSierpinskiDepth = 5;
const SIERPINSKI_SIZE = 12;
const CAMERA_ZOOM_MIN = -2;
const CAMERA_ZOOM_MAX = 12;
const ZOOM_STEP_FACTOR = 0.08;

// Animation parameters
let animParams = {
    factor: 21,         // Default: animantionFactorIdeal
    factorIdeal: 21,
    factorClick: 12,     // animantionFactorKlick
    x: 0.002,           // Default animationX (base speed for rotation)
    y: 0.002,           // Default animationY
    z: 0.002,           // Default animationZ
    isMouseDown: false,
    startX: 0,
    startY: 0,
    // Store the interactive rotation speeds separately
    interactiveX: 0,
    interactiveY: 0,
    interactiveZ: 0,
};

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

export function init(canvasContainerSelector, options = {}) {
    if (isInitialized) {
        TB.logger.warn('[Graphics] Already initialized.');
        return getContext();
    }

    const container = document.querySelector(canvasContainerSelector);
    if (!container) {
        TB.logger.error(`[Graphics] Container "${canvasContainerSelector}" not found.`);
        return null;
    }

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(90, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, options.cameraY || INITIAL_CAMERA_Y, options.cameraZ || INITIAL_CAMERA_Z);

    currentSierpinskiDepth = options.sierpinskiDepth || currentSierpinskiDepth;
    _buildSierpinski(); // Builds mainGroup and populates subGroupsToAnimate

    ambientLightInstance = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLightInstance);

    const pl1 = new THREE.PointLight(0xffffff, 1, 100);
    pl1.position.set(2, 2, 2);
    scene.add(pl1);
    pointLightInstances.push(pl1);

    const pl2 = new THREE.PointLight(0xffffff, 1, 100);
    pl2.position.set(2, -2, 2);
    scene.add(pl2);
    pointLightInstances.push(pl2);

    _addEventListeners();
    isInitialized = true;
    TB.logger.log('[Graphics] Initialized.');
    _animate();

    const currentTheme = (TB.ui && TB.ui.theme && typeof TB.ui.theme.getCurrentMode === 'function')
        ? TB.ui.theme.getCurrentMode()
        : 'light';
    updateTheme(currentTheme);

    themeChangeHandler = (eventData) => { if (isInitialized) updateTheme(eventData.mode); };
    TB.events.on('theme:changed', themeChangeHandler);

    if (options.uninstallServiceWorkers !== false) _uninstallServiceWorkers();
    setTimeout(() => {
        const loader = document.querySelector('.loaderCenter');
        if (loader) loader.style.display = 'none';
    }, options.loaderHideDelay || 122);

    TB.events.emit('graphics:initialized', getContext());
    return getContext();
}

function _buildSierpinski() {
    if (mainGroup) {
        scene.remove(mainGroup);
        // No need to traverse and dispose geometry here if _disposeMainGroup handles it
        _disposeMainGroup(); // Clear old group and its contents
    }
    subGroupsToAnimate = []; // Reset sub-groups
    mainGroup = createSierpinski(currentSierpinskiDepth, SIERPINSKI_SIZE, milkGlassMaterial);
    scene.add(mainGroup);
}

function _disposeMainGroup() {
    if (mainGroup) {
        mainGroup.traverse(object => {
            if (object.geometry) object.geometry.dispose();
            // Materials are more complex; the shared milkGlassMaterial is disposed in main dispose()
        });
        mainGroup = null;
    }
    subGroupsToAnimate = [];
}


export function getContext() {
    if (!isInitialized) return null;
    return { renderer, scene, camera, ambientLightInstance, pointLightInstances, mainGroup };
}

export function updateTheme(theme) {
    if (!isInitialized) return;
    let clearColorHex, ambientColorHex, pointColorHex;
    if (theme === 'dark') {
        clearColorHex = 0x000000; ambientColorHex = 0x181823; pointColorHex = 0x404060;
    } else {
        clearColorHex = 0xcccccc; ambientColorHex = 0x537FE7; pointColorHex = 0xffffff;
    }
    renderer.setClearColor(clearColorHex, 1);
    if (ambientLightInstance) ambientLightInstance.color.setHex(ambientColorHex);
    pointLightInstances.forEach(pl => pl.color.setHex(pointColorHex));
    TB.logger.log(`[Graphics] Theme updated to "${theme}". Clear: #${clearColorHex.toString(16)}, Amb: #${ambientColorHex.toString(16)}, Point: #${pointColorHex.toString(16)}`);
}

function _animate() {
    if (isPaused || !isInitialized) return;
    animationFrameId = requestAnimationFrame(_animate);

    let currentAnimX = animParams.x;
    let currentAnimY = animParams.y;
    let currentAnimZ = animParams.z;

    if (animParams.isMouseDown) {
        // During interaction, use the values derived from mouse/touch movement
        currentAnimX = animParams.interactiveX;
        currentAnimY = animParams.interactiveY;
        currentAnimZ = animParams.interactiveZ;
    }

    if (mainGroup) {
        mainGroup.rotation.x += currentAnimX / animParams.factor;
        mainGroup.rotation.y += currentAnimY / animParams.factor;
        mainGroup.rotation.z += currentAnimZ / animParams.factor;
    }

    // Animate sub-groups if any (replicating old script's groops animation)
    for (let i = 0; i < subGroupsToAnimate.length; i++) {
        subGroupsToAnimate[i].rotation.x += currentAnimX / animParams.factor;
        subGroupsToAnimate[i].rotation.y += currentAnimY / animParams.factor;
        subGroupsToAnimate[i].rotation.z += currentAnimZ / animParams.factor;
    }

    renderer.render(scene, camera);
}

function createSierpinski(depth, size, material) {
    const rootGroup = new THREE.Group();

    function subdivide(currentDepth, currentSize, parentGroup, isRootCall = false) {
        if (currentDepth === 0) {
            const tetra = new THREE.TetrahedronGeometry(currentSize * 0.6);
            const mesh = new THREE.Mesh(tetra, material);
            parentGroup.add(mesh);
            return;
        }

        const newSize = currentSize / 2;
        const newDepth = currentDepth - 1;
        const positions = [
            new THREE.Vector3(-newSize * Math.sqrt(3) / 4, 0, -newSize / 4),
            new THREE.Vector3( newSize * Math.sqrt(3) / 4, 0, -newSize / 4),
            new THREE.Vector3(0,                           0,  newSize / 2),
            new THREE.Vector3(0, newSize * Math.sqrt(2/3),  0)
        ];

        positions.forEach(pos => {
            const subGroup = new THREE.Group();
            subGroup.position.copy(pos);
            subdivide(newDepth, newSize, subGroup); // Recursive call
            parentGroup.add(subGroup);

            // Collect sub-groups for animation, similar to old script's `if (depth !== 1)`
            // This collects all groups that are not the final tetrahedrons themselves.
            if (newDepth > 0) { // If this subGroup will itself contain more subdivisions
                subGroupsToAnimate.push(subGroup);
            }
        });
    }

    subdivide(depth, size, rootGroup, true);
    return rootGroup;
}

export function setSierpinskiDepth(newDepth) {
    if (!isInitialized || newDepth === currentSierpinskiDepth || newDepth < 0) return;
    TB.logger.log(`[Graphics] Changing Sierpinski depth from ${currentSierpinskiDepth} to ${newDepth}`);
    currentSierpinskiDepth = newDepth;
    _buildSierpinski(); // Rebuilds mainGroup and populates subGroupsToAnimate
}

export function setAnimationSpeed(x, y, z, factor) {
    animParams.x = x; animParams.y = y; animParams.z = z;
    if (factor !== undefined) animParams.factor = factor;
    // If setting base speed, also reset interactive if not interacting
    if (!animParams.isMouseDown) {
        animParams.interactiveX = 0; animParams.interactiveY = 0; animParams.interactiveZ = 0;
    }
}

export function adjustCameraZoom(deltaZoomAmount) {
    if (!isInitialized) return;
    camera.position.z += deltaZoomAmount;
    if (camera.position.z <= CAMERA_ZOOM_MIN) camera.position.z = CAMERA_ZOOM_MAX;
    else if (camera.position.z > CAMERA_ZOOM_MAX) camera.position.z = CAMERA_ZOOM_MIN;
    camera.updateProjectionMatrix();
}

export function setCameraZoom(absoluteZoomValue) {
    if (!isInitialized) return;
    camera.position.z = absoluteZoomValue;
    if (camera.position.z <= CAMERA_ZOOM_MIN) camera.position.z = CAMERA_ZOOM_MAX;
    else if (camera.position.z > CAMERA_ZOOM_MAX) camera.position.z = CAMERA_ZOOM_MIN;
    camera.updateProjectionMatrix();
}

function _handleResize() {
    if (!isInitialized) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    TB.logger.log('[Graphics] Window resized.');
}

function _handleMouseDown(event) {
    if (!isInitialized) return;
    animParams.isMouseDown = true;
    animParams.factor = animParams.factorClick; // Use faster factor for interaction
    const { clientX, clientY } = (event.touches ? event.touches[0] : event);
    animParams.startX = clientX;
    animParams.startY = clientY;

    const normX = (clientX / window.innerWidth) * 2 - 1;
    const normY = -((clientY / window.innerHeight) * 2 - 1);

    // Set interactive rotation speeds based on initial click
    animParams.interactiveX = normY * 0.02;
    animParams.interactiveY = normX * 0.02;
    animParams.interactiveZ = (normX + normY) * 0.01;

    // if (event.cancelable && event.type.includes('touch')) event.preventDefault();
}

function _handleMouseMove(event) {
    if (!animParams.isMouseDown || !isInitialized) return;
    const { clientX, clientY } = (event.touches ? event.touches[0] : event);
    const deltaX = clientX - animParams.startX;
    const deltaY = clientY - animParams.startY;

    // Update interactive rotation speeds based on mouse/touch delta
    animParams.interactiveX = (deltaY / window.innerHeight) * (Math.PI / 25);
    animParams.interactiveY = (deltaX / window.innerWidth) * (Math.PI / 25);
    animParams.interactiveZ = ((deltaX + deltaY) / (window.innerWidth + window.innerHeight)) * (Math.PI / 25);

    TB.logger.debug(`[Graphics] Interactive anim: X=${animParams.interactiveX.toFixed(3)}, Y=${animParams.interactiveY.toFixed(3)}, Z=${animParams.interactiveZ.toFixed(3)}`);

    // if (event.cancelable && event.type.includes('touch')) event.preventDefault();
}

function _handleMouseUp() {
    if (!isInitialized || !animParams.isMouseDown) return;
    animParams.isMouseDown = false;
    animParams.factor = animParams.factorIdeal; // Reset to ideal factor
    // Interactive params will be ignored by _animate loop when isMouseDown is false
    // The base animParams.x,y,z will take over for the gentle spin.
}

function _handleWheel(event) {
    if (!isInitialized) return;
    const zoomAmount = (event.deltaY > 0 ? -1 : 1) * ZOOM_STEP_FACTOR;
    adjustCameraZoom(zoomAmount);
    // if (event.cancelable) event.preventDefault();
}

function _addEventListeners() {
    window.addEventListener('resize', _handleResize);
    document.body.addEventListener('mousedown', _handleMouseDown);
    document.body.addEventListener('wheel', _handleWheel, { passive: true });
    document.body.addEventListener('mousemove', _handleMouseMove);
    document.body.addEventListener('mouseup', _handleMouseUp);
    document.body.addEventListener('touchstart', _handleMouseDown, { passive: true });
    document.body.addEventListener('touchmove', _handleMouseMove, { passive: true });
    document.body.addEventListener('touchend', _handleMouseUp);
}

function _removeEventListeners() {
    window.removeEventListener('resize', _handleResize);

    document.body.removeEventListener('mousedown', _handleMouseDown);
    document.body.removeEventListener('wheel', _handleWheel);
    document.body.removeEventListener('touchstart', _handleMouseDown);

    document.body.removeEventListener('mousemove', _handleMouseMove);
    document.body.removeEventListener('mouseup', _handleMouseUp);
    document.body.removeEventListener('touchmove', _handleMouseMove);
    document.body.removeEventListener('touchend', _handleMouseUp);
}

function _uninstallServiceWorkers() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(registrations => {
            registrations.forEach(registration => {
                registration.unregister().then(success => {
                    TB.logger.log(`[Graphics] ServiceWorker ${registration.scope} ${success ? 'unregistered' : 'unregistration failed'}.`);
                });
            });
        }).catch(err => TB.logger.error('[Graphics] Error getting ServiceWorker registrations:', err));
    }
}

export function dispose() {
    if (!isInitialized) return;
    if (themeChangeHandler) { TB.events.off('theme:changed', themeChangeHandler); themeChangeHandler = null; }
    _removeEventListeners();
    cancelAnimationFrame(animationFrameId); animationFrameId = null;

    _disposeMainGroup(); // Dispose current main group and clear subGroupsToAnimate
    if (scene) scene.clear();
    milkGlassMaterial.dispose();

    if (renderer) {
        renderer.dispose();
        if (renderer.domElement.parentElement) renderer.domElement.parentElement.removeChild(renderer.domElement);
    }
    renderer = null; scene = null; camera = null;
    ambientLightInstance = null; pointLightInstances = [];
    isInitialized = false; isPaused = false;
    TB.events.emit('graphics:disposed');
    TB.logger.log('[Graphics] Disposed.');
}

export function pause() { if (isInitialized && !isPaused) { isPaused = true; TB.logger.log('[Graphics] Rendering paused.'); } }
export function resume() { if (isInitialized && isPaused) { isPaused = false; _animate(); TB.logger.log('[Graphics] Rendering resumed.'); } }
