// tbjs/ui/components/ThreeDeeBackground/ThreeDeeBackground.js
import TB from '../../../index.js';
// Placeholder for Three.js library
let THREE;

const DEFAULT_3D_OPTIONS = {
    targetElementId: 'threeDScene',
    enableInteractions: true,
    initialZoom: 10,
    backgroundColorDark: 0x000000,
    backgroundColorLight: 0xcccccc,
    ambientLightColorDark: 0x181823, // Example: dark blue/grey
    ambientLightColorLight: 0x537FE7, // Example: light blue
    pointLightColorDark: 0x333344,    // Dim point light for dark mode
    pointLightColorLight: 0xffffff,   // Bright point light for light mode
    idleAnimation: { x: 0.002, y: 0.002, z: 0.002, factor: 12 },
    interactionFactor: 8, // Speed factor during interaction
    animationBaseSpeed: 0.02, // Corresponds to 'f' in original animator
    animationSpeedFactor: 12, // Corresponds to 's' in original animator
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
        this.mainObject = null;
        this.animationFrameId = null;

        // Animation state
        this.currentRotationSpeed = { ...this.options.idleAnimation }; // Initially idle
        this.isUserInteracting = false;
        this.programmedAnimationTimeoutId = null;
        this.currentAnimationSequenceQueue = []; // For chained animations

        this._boundAnimate = this._animate.bind(this);
        this._boundOnResize = this._onWindowResize.bind(this);
        this._boundHandleThemeChange = this._handleThemeChange.bind(this);
        this._boundStartInteract = this._startBgInteract.bind(this);
        this._boundMoveInteract = this._handleBgMove.bind(this);
        this._boundEndInteract = this._endBgInteract.bind(this);

        this._init();
    }

    async _init() {
        if (!THREE) {
            try {
                const threeModule = await import('three');
                THREE = threeModule;
            } catch (e) {
                TB.logger.error('[3DBackground] Failed to load Three.js:', e);
                return;
            }
        }
        if (!THREE) {
            TB.logger.error('[3DBackground] Three.js library not available.');
            return;
        }

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(90, this.targetElement.clientWidth / this.targetElement.clientHeight, 0.1, 1000);

        this.renderer.setSize(this.targetElement.clientWidth, this.targetElement.clientHeight);
        this.targetElement.appendChild(this.renderer.domElement);

        this.milkGlassMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xffffff, metalness: 0.5, roughness: 0.5, transparent: true,
            clearcoat: 0.5, clearcoatRoughness: 0.8, reflectivity: 0.4, opacity: 0.6
        });

        this.mainObject = this._createSierpinskiTriangle(5, 12);
        this.scene.add(this.mainObject);

        this.camera.position.z = this.options.initialZoom;
        this.camera.position.y = 3.2;

        this.pointLights = [];
        this.ambientLight = null;
        this._setupLighting();

        TB.events.on('theme:changed', this._boundHandleThemeChange);
        this._handleThemeChange({ mode: TB.ui.theme.getCurrentMode() }); // Match expected event data structure

        window.addEventListener('resize', this._boundOnResize);
        if (this.options.enableInteractions) {
            this._addInteractionListeners();
        }

        this._animate();
        TB.logger.log('[3DBackground] Initialized.');
    }

    _setupLighting() {
        this.pointLights.forEach(light => this.scene.remove(light));
        this.pointLights = [];
        if (this.ambientLight) this.scene.remove(this.ambientLight);

        const isDark = TB.ui.theme.getCurrentMode() === 'dark';
        const ambientColor = isDark ? this.options.ambientLightColorDark : this.options.ambientLightColorLight;
        const pointColor = isDark ? this.options.pointLightColorDark : this.options.pointLightColorLight;

        this.ambientLight = new THREE.AmbientLight(ambientColor);
        this.scene.add(this.ambientLight);

        const pLight1 = new THREE.PointLight(pointColor, 1, 100);
        pLight1.position.set(5, -5, 5); // Wider spread
        this.scene.add(pLight1);
        this.pointLights.push(pLight1);

        const pLight2 = new THREE.PointLight(pointColor, 0.8, 100);
        pLight2.position.set(-5, 5, 3); // Different position
        this.scene.add(pLight2);
        this.pointLights.push(pLight2);
    }

    _handleThemeChange(eventData) {
        const newMode = eventData.mode; // Assuming event sends { mode: 'dark'/'light' }
        const isDark = newMode === 'dark';
        this.renderer.setClearColor(isDark ? this.options.backgroundColorDark : this.options.backgroundColorLight, 1);
        this._setupLighting();
        TB.logger.debug(`[3DBackground] Theme changed to ${newMode}`);
    }

    _createSierpinskiTriangle(depth, size) {
        const group = new THREE.Group();
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
                    { x: -offset / 2, y: 0, z: -newSize / 4 },
                    { x: offset / 2, y: 0, z: -newSize / 4 },
                    { x: 0, y: 0, z: newSize / 2 },
                    { x: 0, y: newSize * Math.sqrt(2 / 3), z: 0 }
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
            // currentRotationSpeed is updated by interactions or programmed animations
            this.mainObject.rotation.x += this.currentRotationSpeed.x / this.currentRotationSpeed.factor;
            this.mainObject.rotation.y += this.currentRotationSpeed.y / this.currentRotationSpeed.factor;
            this.mainObject.rotation.z += this.currentRotationSpeed.z / this.currentRotationSpeed.factor;
        }
        this.renderer.render(this.scene, this.camera);
    }

    _onWindowResize() {
        if (!this.targetElement) return;
        this.camera.aspect = this.targetElement.clientWidth / this.targetElement.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.targetElement.clientWidth, this.targetElement.clientHeight);
    }

    _addInteractionListeners() {
        this.targetElement.addEventListener('mousedown', this._boundStartInteract);
        this.targetElement.addEventListener('mousemove', this._boundMoveInteract);
        this.targetElement.addEventListener('mouseup', this._boundEndInteract);
        this.targetElement.addEventListener('mouseleave', this._boundEndInteract);
        this.targetElement.addEventListener('touchstart', this._boundStartInteract, { passive: false }); // passive: false if preventDefault is used
        this.targetElement.addEventListener('touchmove', this._boundMoveInteract, { passive: false });
        this.targetElement.addEventListener('touchend', this._boundEndInteract);
        this.targetElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeedFactor = 0.2;
            this.setZoom(e.deltaY > 0 ? zoomSpeedFactor : -zoomSpeedFactor);
        }, { passive: false });
    }

    _startBgInteract(event) {
        if (event.type.includes('touch')) event.preventDefault(); // Prevent scrolling on touch
        this.isUserInteracting = true;

        // Stop any ongoing programmed animation sequence from progressing further
        if (this.programmedAnimationTimeoutId) {
            clearTimeout(this.programmedAnimationTimeoutId);
            this.programmedAnimationTimeoutId = null;
        }
        // this.currentAnimationSequenceQueue = []; // Optionally clear queue on new interaction

        let clientX, clientY;
        if (event.type.includes('mouse')) {
            clientX = event.clientX;
            clientY = event.clientY;
        } else if (event.type.includes('touch') && event.touches[0]) {
            clientX = event.touches[0].clientX;
            clientY = event.touches[0].clientY;
        } else return;

        this._startX = clientX;
        this._startY = clientY;

        const rect = this.targetElement.getBoundingClientRect();
        const normalizedX = ((clientX - rect.left) / rect.width) * 2 - 1;
        const normalizedY = -(((clientY - rect.top) / rect.height) * 2 - 1);

        // Set current rotation speed based on initial click/touch position
        this.currentRotationSpeed = {
            x: normalizedY * 0.02, // Base sensitivity
            y: normalizedX * 0.02,
            z: (normalizedX + normalizedY) * 0.01,
            factor: this.options.interactionFactor
        };
    }

    _handleBgMove(event) {
        if (!this.isUserInteracting) return;
        if (event.type.includes('touch')) event.preventDefault();

        let clientX, clientY;
        if (event.type.includes('mouse')) {
            clientX = event.clientX;
            clientY = event.clientY;
        } else if (event.type.includes('touch') && event.touches[0]) {
            clientX = event.touches[0].clientX;
            clientY = event.touches[0].clientY;
        } else return;

        const M_PI_25 = Math.PI / 25; // Sensitivity factor
        const deltaX = clientX - this._startX;
        const deltaY = clientY - this._startY;

        // Update current rotation speed based on movement delta
        this.currentRotationSpeed = {
            x: (deltaY / this.targetElement.clientHeight) * M_PI_25,
            y: (deltaX / this.targetElement.clientWidth) * M_PI_25,
            z: ((deltaX + deltaY) / (this.targetElement.clientWidth + this.targetElement.clientHeight)) * M_PI_25,
            factor: this.options.interactionFactor // Keep interaction speed factor
        };
        // No need to update _startX/_startY here if we want deltas from initial touch point
    }

    _endBgInteract() {
        if (!this.isUserInteracting) return;
        this.isUserInteracting = false;
        // Revert to idle animation or resume programmed animation
        // For now, just revert to idle. Resuming sequence is more complex.
        if (this.currentAnimationSequenceQueue.length > 0) {
            // If a sequence was interrupted, try to play the next step
            // This is a simple resume; more robust would require storing the exact state
            const nextAnimation = this.currentAnimationSequenceQueue.shift();
            if (nextAnimation) {
                this.playAnimationSequence(nextAnimation.sequence, nextAnimation.onComplete, nextAnimation.baseSpeed, nextAnimation.speedFactor);
            } else {
                 this.currentRotationSpeed = { ...this.options.idleAnimation };
            }
        } else {
            this.currentRotationSpeed = { ...this.options.idleAnimation };
        }
    }

    _parseAnimationInput(input) {
        const regex = /^([RPYZ])(\d+)([+-])(\d+)(\d+)$/;
        const match = input.match(regex);
        if (match) {
            return {
                animationType: match[1],
                repeat: parseInt(match[2]),
                direction: match[3] === "+" ? 1 : -1,
                speed: parseInt(match[4]),
                duration: parseInt(match[5]) *10 + parseInt(match[2]) * 1000 // 'complex' from original
            };
        }
        return null;
    }
/**
 * ThreeDeeBackground Class
 *
 * Manages a 3D background scene using Three.js.
 * Supports theme changes, user interactions, and programmed animation sequences.
 *
 * Animation Sequence String Format:
 * Each step in a sequence is a 5-part string, e.g., "T R D S C"
 * Multiple steps can be chained with a colon ":", e.g., "TRDSC:TRDSC"
 *
 * Parts:
 * 1. T = Type (Character):
 *    - 'R': Rotate around X-axis (Roll)
 *    - 'P': Rotate around Z-axis (Pan/Yaw, object's Z)
 *    - 'Y': Rotate around Y-axis (Yaw/Pitch, object's Y)
 *    - 'Z': Zoom action
 *
 * 2. R = Repeat Count (Integer, e.g., 0, 1, 5):
 *    - Primarily for 'Z' (Zoom) type: number of discrete zoom steps.
 *    - For 'R', 'P', 'Y': Contributes to the step's total duration calculation.
 *
 * 3. D = Direction (Character: '+' or '-'):
 *    - '+': Positive direction (e.g., counter-clockwise rotation, zoom "in" or further).
 *    - '-': Negative direction.
 *
 * 4. S = Speed Multiplier (Integer, e.g., 0, 1, 3):
 *    - Multiplies a base animation speed.
 *    - For rotations: affects rotation rate.
 *    - For zoom: affects zoom amount per step (calculated as direction * speed / 10).
 *
 * 5. C = Complexity/Duration Multiplier (Integer, e.g., 0, 1, 2):
 *    - Affects the duration this animation step remains active.
 *    - Duration (ms) = (C_Value *10+ 1 + R_Value * 1000).
 *
 * Example: "R1+32"
 *   - Rotate (R) around X-axis.
 *   - Repeat '1' (for duration calculation).
 *   - Direction '+' (positive).
 *   - Speed multiplier '3'.
 *   - Complexity '2'.
 *   - Result: Rotate around X at 3x speed for (2+1*1000) = 1020ms.
 *
 * Example: "Z2-11:R0+10"
 *   - Step 1: Zoom (Z), 2 steps, negative direction (out), speed 1 (0.1 units/step). Duration (1*10+2*1000)=2010ms for this phase.
 *   - Step 2 (after Step 1 completes): Rotate (R) X-axis, repeat 0, positive, speed 1. Duration (0+1+0*1000)=1ms.
 */
    playAnimationSequence(sequenceString, onSequenceComplete = null, baseSpeedOverride = null, speedFactorOverride = null) {
        TB.logger.log('[3DBackground] Playing animation sequence:', sequenceString);

        // Clear any existing programmed animation timeout
        if (this.programmedAnimationTimeoutId) {
            clearTimeout(this.programmedAnimationTimeoutId);
            this.programmedAnimationTimeoutId = null;
        }

        const animationSteps = sequenceString.split(':');
        this.currentAnimationSequenceQueue = animationSteps.map((step, index) => ({
            sequence: step,
            onComplete: index === animationSteps.length - 1 ? onSequenceComplete : null,
            baseSpeed: baseSpeedOverride,
            speedFactor: speedFactorOverride
        }));

        this._executeNextAnimationStep();
    }

    _executeNextAnimationStep() {
        if (this.isUserInteracting || this.currentAnimationSequenceQueue.length === 0) {
            if (this.currentAnimationSequenceQueue.length === 0 && !this.isUserInteracting) {
                 // Sequence finished and no user interaction, revert to idle
                 this.currentRotationSpeed = { ...this.options.idleAnimation };
                 TB.logger.log('[3DBackground] Animation sequence complete. Reverting to idle.');
            }
            return; // Don't proceed if user is interacting or queue is empty
        }

        const nextStepInfo = this.currentAnimationSequenceQueue.shift();
        const animationParams = this._parseAnimationInput(nextStepInfo.sequence);

        if (!animationParams) {
            TB.logger.warn('[3DBackground] Invalid animation step string:', nextStepInfo.sequence);
            if (this.currentAnimationSequenceQueue.length > 0) {
                this._executeNextAnimationStep(); // Try next step
            } else {
                this.currentRotationSpeed = { ...this.options.idleAnimation };
                if (nextStepInfo.onComplete && typeof nextStepInfo.onComplete === 'function') {
                    nextStepInfo.onComplete();
                }
            }
            return;
        }

        const f = nextStepInfo.baseSpeed !== null ? nextStepInfo.baseSpeed : this.options.animationBaseSpeed;
        const s = nextStepInfo.speedFactor !== null ? nextStepInfo.speedFactor : this.options.animationSpeedFactor;

        const { animationType, repeat, direction, speed, duration } = animationParams;

        switch (animationType) {
            case "Y":
                this.currentRotationSpeed = { x: 0, y: f * direction * speed, z: 0, factor: s * speed };
                break;
            case "R":
                this.currentRotationSpeed = { x: f * direction * speed, y: 0, z: 0, factor: s * speed };
                break;
            case "P":
                this.currentRotationSpeed = { x: 0, y: 0, z: f * direction * speed, factor: s * speed };
                break;
            case "Z":
                // Zoom is discrete, not continuous rotation speed
                this.currentRotationSpeed = { ...this.options.idleAnimation }; // Keep idle rotation during zoom sequence
                this._performRepeatZoom(direction, speed, repeat, s * 3); // s*3 from original `smo`
                break;
            default:
                TB.logger.warn("[3DBackground] Invalid animation type in sequence:", animationType);
                this.currentRotationSpeed = { ...this.options.idleAnimation }; // Revert to idle
        }

        TB.logger.debug(`[3DBackground] Step: ${nextStepInfo.sequence}, Duration: ${duration}ms, New Speed:`, this.currentRotationSpeed);

        this.programmedAnimationTimeoutId = setTimeout(() => {
            this.programmedAnimationTimeoutId = null;
            if (this.isUserInteracting) {
                // If user started interacting during timeout, store remaining sequence
                 if (this.currentAnimationSequenceQueue.length > 0) {
                    TB.logger.log('[3DBackground] User interaction interrupted sequence. Queueing remaining steps.');
                }
                return; // Don't auto-proceed if user is now interacting
            }

            if (this.currentAnimationSequenceQueue.length > 0) {
                this._executeNextAnimationStep();
            } else {
                // Last step of the entire sequence has finished
                this.currentRotationSpeed = { ...this.options.idleAnimation }; // Revert to idle
                TB.logger.log('[3DBackground] Animation sequence fully completed.');
                if (nextStepInfo.onComplete && typeof nextStepInfo.onComplete === 'function') {
                    nextStepInfo.onComplete();
                }
            }
        }, duration);
    }

    _performRepeatZoom(direction, speed, repeatCount, smoothness) {
        if (repeatCount <= 0 || this.isUserInteracting) { // Stop if user interacts
            if (this.isUserInteracting) TB.logger.debug("[3DBackground] Zoom sequence interrupted by user.");
            return;
        }
        this.setZoom(direction * speed / 10); // Original Set_zoom scaling

        setTimeout(() => {
            if (!this.isUserInteracting) { // Check again before next step
                this._performRepeatZoom(direction, speed, repeatCount - 1, smoothness);
            }
        }, 1000 / smoothness);
    }

    // Public control methods
    setManualAnimationSpeed(x, y, z, speedFactor = this.options.idleAnimation.factor) {
        // This method allows direct control, bypassing programmed sequences
        if (this.programmedAnimationTimeoutId) {
            clearTimeout(this.programmedAnimationTimeoutId);
            this.programmedAnimationTimeoutId = null;
        }
        this.currentAnimationSequenceQueue = [];
        this.currentRotationSpeed = { x, y, z, factor: speedFactor };
        this.isUserInteracting = false; // Assume manual override means no longer "user driven" in the click/drag sense
    }

    setZoom(deltaZoom) {
        this.camera.position.z += deltaZoom;
        this.camera.position.z = Math.max(2, Math.min(30, this.camera.position.z)); // Increased max zoom
        this.camera.updateProjectionMatrix();
    }

    resetToIdleAnimation() {
        if (this.programmedAnimationTimeoutId) {
            clearTimeout(this.programmedAnimationTimeoutId);
            this.programmedAnimationTimeoutId = null;
        }
        this.currentAnimationSequenceQueue = [];
        this.currentRotationSpeed = { ...this.options.idleAnimation };
        this.isUserInteracting = false;
        TB.logger.log('[3DBackground] Reset to idle animation.');
    }

    destroy() {
        cancelAnimationFrame(this.animationFrameId);
        if (this.programmedAnimationTimeoutId) {
            clearTimeout(this.programmedAnimationTimeoutId);
        }
        window.removeEventListener('resize', this._boundOnResize);
        TB.events.off('theme:changed', this._boundHandleThemeChange);

        if (this.options.enableInteractions) {
            this.targetElement.removeEventListener('mousedown', this._boundStartInteract);
            this.targetElement.removeEventListener('mousemove', this._boundMoveInteract);
            this.targetElement.removeEventListener('mouseup', this._boundEndInteract);
            this.targetElement.removeEventListener('mouseleave', this._boundEndInteract);
            this.targetElement.removeEventListener('touchstart', this._boundStartInteract);
            this.targetElement.removeEventListener('touchmove', this._boundMoveInteract);
            this.targetElement.removeEventListener('touchend', this._boundEndInteract);
            // Wheel listener is anonymous, can't remove directly without storing it.
            // A common pattern is to wrap it if removal is critical:
            // this._boundWheelHandler = (e) => { ... };
            // this.targetElement.addEventListener('wheel', this._boundWheelHandler, ...);
            // this.targetElement.removeEventListener('wheel', this._boundWheelHandler);
        }

        if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
        if (this.renderer) this.renderer.dispose();
        if (this.scene) {
            // Basic cleanup
            this.scene.traverse(object => {
                if (object.geometry) object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }
        THREE = null; // Allow Three.js to be garbage collected if no other references
        this.targetElement = null;
        TB.logger.log('[3DBackground] Destroyed.');
    }
}

export default ThreeDeeBackground;
