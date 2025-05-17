// tbjs/ui/components/ThreeDeeBackground/ThreeDeeBackground.js
import TB from '../../../index.js';
// Dynamically import Three.js only if this component is used to keep initial bundle small
// Or assume Three.js is globally available or managed by Webpack aliases.
// For this example, let's assume dynamic import or it's already loaded.

// Placeholder for Three.js library
let THREE;

const DEFAULT_3D_OPTIONS = {
    targetElementId: 'threeDScene', // Where to mount the canvas
    enableInteractions: true,
    initialZoom: 10,
    backgroundColorDark: 0x000000,
    backgroundColorLight: 0xcccccc,
    ambientLightColorDark: 0x181823,
    ambientLightColorLight: 0x537FE7,
    // ... other three.js specific params ...
};

class ThreeDeeBackground {
    constructor(options = {}) {
        this.options = { ...DEFAULT_3D_OPTIONS, ...options };
        this.targetElement = document.getElementById(this.options.targetElementId);

        if (!this.targetElement) {
            TB.logger.error(`[3DBackground] Target element #${this.options.targetElementId} not found.`);
            return;
        }

        this.renderer = null;
        this.scene = null;
        this.camera = null;
        this.mainObject = null; // The Sierpinski triangle or main geometry
        this.animationFrameId = null;

        this.mouseX = 0;
        this.mouseY = 0;
        this.animationParams = { x: 0.002, y: 0.002, z: 0.002, factor: 12 };
        this.isMouseDown = false;

        this._boundAnimate = this._animate.bind(this);
        this._boundOnResize = this._onWindowResize.bind(this);
        this._boundHandleThemeChange = this._handleThemeChange.bind(this);
        // Mouse/Touch interaction bindings
        this._boundStartInteract = this._startBgInteract.bind(this);
        this._boundMoveInteract = this._handleBgMove.bind(this);
        this._boundEndInteract = this._endBgInteract.bind(this);


        this._init();
    }

    async _init() {
        if (!THREE) {
            try {
                // Using 'three' as per your node_modules structure in original index.js
                const threeModule = await import('three'); // Or '/web/node_modules/three/src/Three.js' if direct path
                THREE = threeModule;
            } catch (e) {
                TB.logger.error('[3DBackground] Failed to load Three.js:', e);
                return;
            }
        }
        if (!THREE) { // Still not loaded
            TB.logger.error('[3DBackground] Three.js library not available.');
            return;
        }


        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); // alpha:true for transparent bg if needed
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(90, this.targetElement.clientWidth / this.targetElement.clientHeight, 0.1, 1000);

        this.renderer.setSize(this.targetElement.clientWidth, this.targetElement.clientHeight);
        this.targetElement.appendChild(this.renderer.domElement);

        // Milk glass material (original: milkGlassMaterial)
        this.milkGlassMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xffffff, metalness: 0.5, roughness: 0.5, transparent: true,
            clearcoat: 0.5, clearcoatRoughness: 0.8, reflectivity: 0.4, opacity: 0.6
        });

        // Main object (Sierpinski triangle, original: triangleGeometry)
        this.mainObject = this._createSierpinskiTriangle(5, 12);
        this.scene.add(this.mainObject);

        this.camera.position.z = this.options.initialZoom;
        this.camera.position.y = 3.2; // from original

        // Lighting (original: pointLightSto, ambientLightSto)
        this.pointLights = [];
        this.ambientLight = null;
        this._setupLighting();

        TB.events.on('theme:changed', this._boundHandleThemeChange);
        this._handleThemeChange(TB.ui.theme.getCurrentMode()); // Set initial theme

        window.addEventListener('resize', this._boundOnResize);
        if (this.options.enableInteractions) {
            this._addInteractionListeners();
        }

        this._animate();
        TB.logger.log('[3DBackground] Initialized.');
    }

    _setupLighting() {
        // Clear existing lights before re-adding
        this.pointLights.forEach(light => this.scene.remove(light));
        this.pointLights = [];
        if (this.ambientLight) this.scene.remove(this.ambientLight);

        const isDark = TB.ui.theme.getCurrentMode() === 'dark';
        const ambientColor = isDark ? this.options.ambientLightColorDark : this.options.ambientLightColorLight;
        const pointColor1 = ambientColor; // Or a different color
        const pointColor2 = 0xffffff; // General white highlight

        this.ambientLight = new THREE.AmbientLight(ambientColor);
        this.scene.add(this.ambientLight);

        const pLight1 = new THREE.PointLight(pointColor1, 1, 100);
        pLight1.position.set(2, -2, 2);
        this.scene.add(pLight1);
        this.pointLights.push(pLight1);

        const pLight2 = new THREE.PointLight(pointColor2, 1, 100);
        pLight2.position.set(2, 2, 2);
        this.scene.add(pLight2);
        this.pointLights.push(pLight2);
    }

    _handleThemeChange(newMode) {
        const isDark = newMode === 'dark';
        this.renderer.setClearColor(isDark ? this.options.backgroundColorDark : this.options.backgroundColorLight, 1);
        this._setupLighting(); // Re-setup lights with new theme colors
        TB.logger.debug(`[3DBackground] Theme changed to ${newMode}`);
    }

    _createSierpinskiTriangle(depth, size) {
        // Original: createSierpinskiTriangle function
        const group = new THREE.Group();
        // Keep track of sub-groups if needed for individual animation (original: groops global)
        this._subGroups = [];

        const createRecursive = (currentDepth, currentSize, parentGroup) => {
            if (currentDepth === 0) {
                const triangle = new THREE.TetrahedronGeometry(currentSize * 0.6);
                const mesh = new THREE.Mesh(triangle, this.milkGlassMaterial);
                parentGroup.add(mesh);
            } else {
                const newSize = currentSize / 2;
                const newDepth = currentDepth - 1;
                const offset = newSize * Math.sqrt(3) / 2;

                const positions = [
                    { x: -offset / 2, y: 0, z: -newSize / 4 }, // Bottom-left
                    { x: offset / 2, y: 0, z: -newSize / 4 },  // Bottom-right
                    { x: 0, y: 0, z: newSize / 2 },           // Bottom-back
                    { x: 0, y: newSize * Math.sqrt(2 / 3), z: 0 } // Top
                ];

                positions.forEach(pos => {
                    const subGroup = new THREE.Group();
                    subGroup.position.set(pos.x, pos.y, pos.z);
                    createRecursive(newDepth, newSize, subGroup);
                    parentGroup.add(subGroup);
                    if (currentDepth !== 1) this._subGroups.push(subGroup);
                });
            }
        };
        createRecursive(depth, size, group);
        return group;
    }

    _animate() {
        this.animationFrameId = requestAnimationFrame(this._boundAnimate);

        if (this.mainObject) {
            const factor = this.animationParams.factor;
            this.mainObject.rotation.x += this.animationParams.x / factor;
            this.mainObject.rotation.y += this.animationParams.y / factor;
            this.mainObject.rotation.z += this.animationParams.z / factor;

            // Animate sub-groups if any (original: groops)
            // this._subGroups.forEach(g => { /* similar rotation */ });
        }
        this.renderer.render(this.scene, this.camera);
    }

    _onWindowResize() {
        if (!this.targetElement) return;
        this.camera.aspect = this.targetElement.clientWidth / this.targetElement.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.targetElement.clientWidth, this.targetElement.clientHeight);
    }

    // Interaction methods (original: StartBgInteract, handleBgMove, EndBgInteract)
    _addInteractionListeners() {
        this.targetElement.addEventListener('mousedown', this._boundStartInteract);
        this.targetElement.addEventListener('mousemove', this._boundMoveInteract);
        this.targetElement.addEventListener('mouseup', this._boundEndInteract);
        this.targetElement.addEventListener('mouseleave', this._boundEndInteract); // Important for mouseup outside

        this.targetElement.addEventListener('touchstart', this._boundStartInteract, { passive: true });
        this.targetElement.addEventListener('touchmove', this._boundMoveInteract, { passive: true });
        this.targetElement.addEventListener('touchend', this._boundEndInteract);

        // Wheel for zoom (original: window.onwheel)
        this.targetElement.addEventListener('wheel', (e) => {
            e.preventDefault(); // Prevent page scroll if targetElement itself is scrollable
            const zoomSpeedFactor = 0.2; // Adjust as needed
            this.camera.position.z += e.deltaY > 0 ? zoomSpeedFactor : -zoomSpeedFactor;
            // Clamp zoom:
            this.camera.position.z = Math.max(2, Math.min(20, this.camera.position.z));
            this.camera.updateProjectionMatrix();
        }, { passive: false });
    }

    _startBgInteract(event) {
        this.isMouseDown = true;
        this.animationParams.factor = 8; // Klick factor
        const M_PI_25 = Math.PI / 25;

        let clientX, clientY;
        if (event.type.includes('mouse')) {
            clientX = event.clientX;
            clientY = event.clientY;
        } else if (event.type.includes('touch') && event.touches[0]) {
            clientX = event.touches[0].clientX;
            clientY = event.touches[0].clientY;
        } else return;

        this._startX = clientX; // Store initial for delta calculation
        this._startY = clientY;

        const rect = this.targetElement.getBoundingClientRect();
        const normalizedX = ((clientX - rect.left) / rect.width) * 2 - 1;
        const normalizedY = -(((clientY - rect.top) / rect.height) * 2 - 1);

        this.animationParams.x = normalizedY * 0.02;
        this.animationParams.y = normalizedX * 0.02;
        this.animationParams.z = (normalizedX + normalizedY) * 0.01;
    }

    _handleBgMove(event) {
        if (!this.isMouseDown) return;
        let clientX, clientY;
        if (event.type.includes('mouse')) {
            clientX = event.clientX;
            clientY = event.clientY;
        } else if (event.type.includes('touch') && event.touches[0]) {
            clientX = event.touches[0].clientX;
            clientY = event.touches[0].clientY;
        } else return;

        const M_PI_25 = Math.PI / 25; // Scale factor
        const deltaX = clientX - this._startX;
        const deltaY = clientY - this._startY;

        // Update based on delta for continuous control
        this.animationParams.x = (deltaY / this.targetElement.clientHeight) * M_PI_25;
        this.animationParams.y = (deltaX / this.targetElement.clientWidth) * M_PI_25;
        this.animationParams.z = ((deltaX + deltaY) / (this.targetElement.clientWidth + this.targetElement.clientHeight)) * M_PI_25;

        // For next delta calculation, update start (or keep original for absolute from start)
        // this._startX = clientX;
        // this._startY = clientY;
    }

    _endBgInteract() {
        if (!this.isMouseDown) return;
        this.isMouseDown = false;
        this.animationParams.factor = 12; // Ideal factor
        this.animationParams.x = 0.002;
        this.animationParams.y = 0.002;
        this.animationParams.z = 0.002;
    }

    // Public control methods (original: Set_animation_xyz, Set_zoom, EndBgInteract)
    setAnimation(x, y, z, speedFactor = 8) {
        this.animationParams = { x, y, z, factor: speedFactor };
    }

    setZoom(deltaZoom) {
        this.camera.position.z += deltaZoom;
        this.camera.position.z = Math.max(2, Math.min(20, this.camera.position.z)); // Clamp
        this.camera.updateProjectionMatrix();
    }

    resetInteraction() {
        this._endBgInteract(); // Resets animation params to default idle
    }

    destroy() {
        cancelAnimationFrame(this.animationFrameId);
        window.removeEventListener('resize', this._boundOnResize);
        TB.events.off('theme:changed', this._boundHandleThemeChange);

        if (this.options.enableInteractions) {
            this.targetElement.removeEventListener('mousedown', this._boundStartInteract);
            // ... remove all interaction listeners
        }

        if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
        if (this.renderer) this.renderer.dispose();
        if (this.scene) { /* dispose geometries/materials if needed */ }

        this.targetElement = null;
        TB.logger.log('[3DBackground] Destroyed.');
    }
}

export default ThreeDeeBackground;
