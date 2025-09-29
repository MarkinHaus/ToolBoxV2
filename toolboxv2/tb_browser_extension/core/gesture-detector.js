// ToolBox Browser Extension - Enhanced Gesture Detector
// Advanced gesture recognition for mouse and touch interactions

class TBGestureDetector {
    constructor() {
        this.isTracking = false;
        this.gesturePoints = [];
        this.startTime = 0;
        this.lastPoint = null;
        this.gestures = new Map();
        this.settings = {
            minPoints: 15,
            maxTime: 5000,
            circleThreshold: 0.85,
            circleMinTime: 1500, // Circle needs more time to be distinguishable
            lineThreshold: 0.9,
            minDistance: 120,
            maxDistance: 600,
            sensitivity: 1.0,
            enableMouse: true,
            enableTouch: true,
            enableKeyboard: true
        };

        this.init();
    }

    async init() {
        try {
            await this.loadSettings();
            this.setupGesturePatterns();
            this.setupEventListeners();

            TBUtils.info('GestureDetector', 'Gesture detector initialized');
        } catch (error) {
            TBUtils.handleError('GestureDetector', error);
        }
    }

    async loadSettings() {
        const stored = await TBUtils.getStorage([
            'gesture_min_points',
            'gesture_max_time',
            'gesture_circle_threshold',
            'gesture_line_threshold',
            'gesture_min_distance',
            'gesture_max_distance',
            'gesture_sensitivity',
            'gesture_enable_mouse',
            'gesture_enable_touch',
            'gesture_enable_keyboard'
        ]);

        this.settings = {
            ...this.settings,
            minPoints: stored.gesture_min_points || this.settings.minPoints,
            maxTime: stored.gesture_max_time || this.settings.maxTime,
            circleThreshold: stored.gesture_circle_threshold || this.settings.circleThreshold,
            lineThreshold: stored.gesture_line_threshold || this.settings.lineThreshold,
            minDistance: stored.gesture_min_distance || this.settings.minDistance,
            maxDistance: stored.gesture_max_distance || this.settings.maxDistance,
            sensitivity: stored.gesture_sensitivity || this.settings.sensitivity,
            enableMouse: stored.gesture_enable_mouse !== false,
            enableTouch: stored.gesture_enable_touch !== false,
            enableKeyboard: stored.gesture_enable_keyboard !== false
        };
    }

    setupGesturePatterns() {
        // Define gesture patterns and their handlers
        this.gestures.set('circle', {
            pattern: 'circle',
            handler: () => this.executeGesture('refresh-page'),
            description: 'Draw a zigzag to refresh page'
        });

        this.gestures.set('line-right', {
            pattern: 'line-right',
            handler: () => this.executeGesture('go-forward'),
            description: 'Draw a line right to go forward'
        });

        this.gestures.set('line-left', {
            pattern: 'line-left',
            handler: () => this.executeGesture('go-back'),
            description: 'Draw a line left to go back'
        });

        this.gestures.set('line-up', {
            pattern: 'line-up',
            handler: () => this.executeGesture('scroll-top'),
            description: 'Draw a line up to scroll to top'
        });

        this.gestures.set('line-down', {
            pattern: 'line-down',
            handler: () => this.executeGesture('scroll-bottom'),
            description: 'Draw a line down to scroll to bottom'
        });

        this.gestures.set('zigzag', {
            pattern: 'zigzag',
            handler: () => this.executeGesture('toggle-panel'),
            description: 'Draw a circle to toggle ToolBox panel'
        });

        this.gestures.set('double-tap', {
            pattern: 'double-tap',
            handler: () => this.executeGesture('smart-search'),
            description: 'Double tap to open smart search'
        });
    }

    setupEventListeners() {
        if (this.settings.enableMouse) {
            this.setupMouseListeners();
        }

        if (this.settings.enableTouch) {
            this.setupTouchListeners();
        }

        if (this.settings.enableKeyboard) {
            this.setupKeyboardListeners();
        }
    }

    setupMouseListeners() {
        let rightMouseDown = false;
        let lastClickTime = 0;
        let clickCount = 0;

        document.addEventListener('mousedown', (event) => {
            if (event.button === 2) { // Right mouse button
                rightMouseDown = true;
                this.startGesture(event.clientX, event.clientY);
                event.preventDefault();
            } else if (event.button === 0) { // Left mouse button
                const currentTime = Date.now();
                if (currentTime - lastClickTime < 300) {
                    clickCount++;
                    if (clickCount === 2) {
                        this.handleDoubleClick(event);
                        clickCount = 0;
                    }
                } else {
                    clickCount = 1;
                }
                lastClickTime = currentTime;
            }
        });

        document.addEventListener('mousemove', (event) => {
            if (rightMouseDown && this.isTracking) {
                this.addGesturePoint(event.clientX, event.clientY);
                event.preventDefault();
            }
        });

        document.addEventListener('mouseup', (event) => {
            if (event.button === 2 && rightMouseDown) {
                rightMouseDown = false;
                this.endGesture();
                event.preventDefault();
            }
        });

        // Prevent context menu when gesture is detected
        document.addEventListener('contextmenu', (event) => {
            if (this.gesturePoints.length > 0) {
                event.preventDefault();
            }
        });
    }

    setupTouchListeners() {
        let touchStartTime = 0;
        let lastTouchEnd = 0;

        document.addEventListener('touchstart', (event) => {
            const touch = event.touches[0];
            touchStartTime = Date.now();

            if (event.touches.length === 1) {
                this.startGesture(touch.clientX, touch.clientY);
            }
        });

        document.addEventListener('touchmove', (event) => {
            if (event.touches.length === 1 && this.isTracking) {
                const touch = event.touches[0];
                this.addGesturePoint(touch.clientX, touch.clientY);
                event.preventDefault();
            }
        });

        document.addEventListener('touchend', (event) => {
            const currentTime = Date.now();

            if (this.isTracking) {
                this.endGesture();
            }

            // Handle double tap
            if (currentTime - touchStartTime < 200) { // Quick tap
                if (currentTime - lastTouchEnd < 300) { // Double tap
                    this.handleDoubleClick(event);
                }
                lastTouchEnd = currentTime;
            }
        });
    }

    setupKeyboardListeners() {
        let keySequence = [];
        let lastKeyTime = 0;

        document.addEventListener('keydown', (event) => {
            const currentTime = Date.now();

            // Reset sequence if too much time has passed
            if (currentTime - lastKeyTime > 1000) {
                keySequence = [];
            }

            keySequence.push(event.key.toLowerCase());
            lastKeyTime = currentTime;

            // Check for key gesture patterns
            this.checkKeyGestures(keySequence);

            // Keep only last 10 keys
            if (keySequence.length > 10) {
                keySequence = keySequence.slice(-10);
            }
        });
    }

    checkKeyGestures(sequence) {
        const sequenceStr = sequence.join('');

        // T key for toggle (when not in input field)
        if (sequence[sequence.length - 1] === 't' && !this.isInInputField()) {
            this.executeGesture('toggle-panel');
        }

        // Konami code style sequences
        if (sequenceStr.includes('toolbox')) {
            this.executeGesture('toggle-panel');
        }

        if (sequenceStr.includes('search')) {
            this.executeGesture('smart-search');
        }

        if (sequenceStr.includes('voice')) {
            this.executeGesture('voice-command');
        }
    }

    isInInputField() {
        const activeElement = document.activeElement;
        return activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.contentEditable === 'true'
        );
    }

    startGesture(x, y) {
        this.isTracking = true;
        this.gesturePoints = [{ x, y, time: Date.now() }];
        this.startTime = Date.now();
        this.lastPoint = { x, y };

        TBUtils.debug('GestureDetector', 'Started gesture tracking');
    }

    addGesturePoint(x, y) {
        if (!this.isTracking) return;

        const currentTime = Date.now();
        const distance = this.calculateDistance(this.lastPoint, { x, y });

        // Only add point if it's far enough from the last point
        if (distance > 5) {
            this.gesturePoints.push({ x, y, time: currentTime });
            this.lastPoint = { x, y };

            // Limit number of points to prevent memory issues
            if (this.gesturePoints.length > 200) {
                this.gesturePoints = this.gesturePoints.slice(-100);
            }
        }

        // Check if gesture is taking too long
        if (currentTime - this.startTime > this.settings.maxTime) {
            this.cancelGesture();
        }
    }

    endGesture() {
        if (!this.isTracking) return;

        this.isTracking = false;

        if (this.gesturePoints.length >= this.settings.minPoints) {
            const recognizedGesture = this.recognizeGesture();
            if (recognizedGesture) {
                TBUtils.info('GestureDetector', `Recognized gesture: ${recognizedGesture}`);
                this.executeRecognizedGesture(recognizedGesture);
            }
        }

        this.resetGesture();
    }

    cancelGesture() {
        this.isTracking = false;
        this.resetGesture();
        TBUtils.debug('GestureDetector', 'Gesture cancelled');
    }

    resetGesture() {
        this.gesturePoints = [];
        this.startTime = 0;
        this.lastPoint = null;
    }

    recognizeGesture() {
        if (this.gesturePoints.length < this.settings.minPoints) {
            return null;
        }

        // Calculate gesture properties
        const totalDistance = this.calculateTotalDistance();
        const boundingBox = this.calculateBoundingBox();
        const directionalChanges = this.calculateDirectionalChanges();

        // Check for circle
        if (this.isCircle()) {
            return 'circle';
        }

        // Check for lines
        const lineDirection = this.getLineDirection();
        if (lineDirection) {
            return `line-${lineDirection}`;
        }

        // Check for zigzag
        if (this.isZigzag()) {
            return 'zigzag';
        }

        return null;
    }

    isCircle() {
        if (this.gesturePoints.length < this.settings.minPoints) return false;

        // Check if gesture took enough time (circles need more time)
        const gestureTime = Date.now() - this.gestureStartTime;
        if (gestureTime < this.settings.circleMinTime) return false;

        const center = this.calculateCenter();
        const avgRadius = this.calculateAverageRadius(center);
        const radiusVariance = this.calculateRadiusVariance(center, avgRadius);

        // Check if the gesture forms a circle (more strict requirements)
        const circleScore = 1 - (radiusVariance / avgRadius);

        return circleScore >= this.settings.circleThreshold &&
               avgRadius >= this.settings.minDistance / 3 &&
               avgRadius <= this.settings.maxDistance / 2 &&
               this.gesturePoints.length >= 20; // More points required for circle
    }

    getLineDirection() {
        const start = this.gesturePoints[0];
        const end = this.gesturePoints[this.gesturePoints.length - 1];

        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < this.settings.minDistance) return null;

        const angle = Math.atan2(dy, dx) * 180 / Math.PI;
        const absAngle = Math.abs(angle);

        // Check if it's mostly straight
        const straightness = this.calculateStraightness();
        if (straightness < this.settings.lineThreshold) return null;

        // Determine direction
        if (absAngle < 45 || absAngle > 135) {
            return dx > 0 ? 'right' : 'left';
        } else {
            return dy > 0 ? 'down' : 'up';
        }
    }

    isZigzag() {
        const directionalChanges = this.calculateDirectionalChanges();

        // Check if overall motion is downward
        const start = this.gesturePoints[0];
        const end = this.gesturePoints[this.gesturePoints.length - 1];
        const overallDy = end.y - start.y;

        // Zigzag only works for downward motion
        return directionalChanges >= 4 &&
               directionalChanges <= 10 &&
               overallDy > 50; // Must move down at least 50px
    }

    calculateDistance(p1, p2) {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    calculateTotalDistance() {
        let total = 0;
        for (let i = 1; i < this.gesturePoints.length; i++) {
            total += this.calculateDistance(this.gesturePoints[i - 1], this.gesturePoints[i]);
        }
        return total;
    }

    calculateBoundingBox() {
        const xs = this.gesturePoints.map(p => p.x);
        const ys = this.gesturePoints.map(p => p.y);

        return {
            minX: Math.min(...xs),
            maxX: Math.max(...xs),
            minY: Math.min(...ys),
            maxY: Math.max(...ys)
        };
    }

    calculateCenter() {
        const sumX = this.gesturePoints.reduce((sum, p) => sum + p.x, 0);
        const sumY = this.gesturePoints.reduce((sum, p) => sum + p.y, 0);

        return {
            x: sumX / this.gesturePoints.length,
            y: sumY / this.gesturePoints.length
        };
    }

    calculateAverageRadius(center) {
        const distances = this.gesturePoints.map(p => this.calculateDistance(center, p));
        return distances.reduce((sum, d) => sum + d, 0) / distances.length;
    }

    calculateRadiusVariance(center, avgRadius) {
        const distances = this.gesturePoints.map(p => this.calculateDistance(center, p));
        const variance = distances.reduce((sum, d) => sum + Math.pow(d - avgRadius, 2), 0) / distances.length;
        return Math.sqrt(variance);
    }

    calculateStraightness() {
        if (this.gesturePoints.length < 3) return 1;

        const start = this.gesturePoints[0];
        const end = this.gesturePoints[this.gesturePoints.length - 1];
        const directDistance = this.calculateDistance(start, end);
        const totalDistance = this.calculateTotalDistance();

        return directDistance / totalDistance;
    }

    calculateDirectionalChanges() {
        if (this.gesturePoints.length < 3) return 0;

        let changes = 0;
        let lastDirection = null;

        for (let i = 1; i < this.gesturePoints.length; i++) {
            const prev = this.gesturePoints[i - 1];
            const curr = this.gesturePoints[i];

            const dx = curr.x - prev.x;
            const dy = curr.y - prev.y;

            let direction;
            if (Math.abs(dx) > Math.abs(dy)) {
                direction = dx > 0 ? 'right' : 'left';
            } else {
                direction = dy > 0 ? 'down' : 'up';
            }

            if (lastDirection && direction !== lastDirection) {
                changes++;
            }
            lastDirection = direction;
        }

        return changes;
    }

    executeRecognizedGesture(gestureType) {
        const gesture = this.gestures.get(gestureType);
        if (gesture && gesture.handler) {
            gesture.handler();
        }
    }

    handleDoubleClick(event) {
        this.executeGesture('smart-search', {
            x: event.clientX || (event.touches && event.touches[0].clientX),
            y: event.clientY || (event.touches && event.touches[0].clientY)
        });
    }

    async executeGesture(action, data = {}) {
        try {
            TBUtils.info('GestureDetector', `Executing gesture action: ${action}`);

            // Execute gesture actions directly
            switch (action) {
                case 'scroll-top':
                    // Full scroll to top (100%)
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    break;

                case 'scroll-bottom':
                    // Full scroll to bottom (100%)
                    window.scrollTo({
                        top: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
                        behavior: 'smooth'
                    });
                    break;

                case 'scroll-up':
                    // Scroll up by viewport height (full page)
                    window.scrollBy({ top: -window.innerHeight, behavior: 'smooth' });
                    break;

                case 'scroll-down':
                    // Scroll down by viewport height (full page)
                    window.scrollBy({ top: window.innerHeight, behavior: 'smooth' });
                    break;

                case 'scroll-left':
                    // Scroll left by viewport width (full page)
                    window.scrollBy({ left: -window.innerWidth, behavior: 'smooth' });
                    break;

                case 'scroll-right':
                    // Scroll right by viewport width (full page)
                    window.scrollBy({ left: window.innerWidth, behavior: 'smooth' });
                    break;

                case 'go-back':
                    window.history.back();
                    break;

                case 'go-forward':
                    window.history.forward();
                    break;

                case 'refresh-page':
                    window.location.reload();
                    break;

                case 'toggle-panel':
                    // Send message to content script to toggle panel
                    if (window.tbExtension) {
                        window.tbExtension.togglePanel();
                    }
                    break;

                case 'voice-command':
                    // Activate voice commands
                    if (window.tbExtension && window.tbExtension.components.voiceEngine) {
                        window.tbExtension.components.voiceEngine.startListening();
                    }
                    break;

                case 'smart-search':
                    // Show search in main panel
                    if (window.tbExtension && window.tbExtension.components.uiManager) {
                        window.tbExtension.components.uiManager.showSearch();
                    }
                    break;

                case 'close-tab':
                    // Send message to background to close tab
                    if (typeof chrome !== 'undefined' && chrome.runtime) {
                        chrome.runtime.sendMessage({ type: 'TB_CLOSE_TAB' });
                    }
                    break;

                case 'new-tab':
                    // Send message to background to open new tab
                    if (typeof chrome !== 'undefined' && chrome.runtime) {
                        chrome.runtime.sendMessage({ type: 'TB_NEW_TAB' });
                    }
                    break;

                default:
                    TBUtils.warn('GestureDetector', `Unknown gesture action: ${action}`);
                    break;
            }

            // Show visual feedback
            this.showGestureFeedback(action);

        } catch (error) {
            TBUtils.handleError('GestureDetector', error);
        }
    }

    showGestureFeedback(action) {
        // Create visual feedback for gesture execution
        const feedback = document.createElement('div');
        feedback.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--tb-accent-primary, #6c8ee8);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            z-index: 999999;
            animation: slideInFade 0.3s ease-out;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;

        const actionNames = {
            'scroll-top': '⬆️ Scroll to Top (100%)',
            'scroll-bottom': '⬇️ Scroll to Bottom (100%)',
            'scroll-up': '↑ Scroll Up (Full Page)',
            'scroll-down': '↓ Scroll Down (Full Page)',
            'scroll-left': '← Scroll Left (Full Page)',
            'scroll-right': '→ Scroll Right (Full Page)',
            'go-back': '← Go Back',
            'go-forward': '→ Go Forward',
            'refresh-page': '🔄 Refresh Page',
            'toggle-panel': '📱 Toggle Panel',
            'voice-command': '🎤 Voice Command',
            'smart-search': '🔍 Smart Search',
            'close-tab': '✕ Close Tab',
            'new-tab': '+ New Tab'
        };

        feedback.textContent = actionNames[action] || `✓ ${action}`;

        if (document.body) {
            document.body.appendChild(feedback);

            // Remove after 2 seconds
            setTimeout(() => {
                if (feedback.parentNode) {
                    feedback.remove();
                }
            }, 2000);
        }
    }

    // Public API
    async updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        await TBUtils.setStorage({
            gesture_min_points: this.settings.minPoints,
            gesture_max_time: this.settings.maxTime,
            gesture_circle_threshold: this.settings.circleThreshold,
            gesture_line_threshold: this.settings.lineThreshold,
            gesture_min_distance: this.settings.minDistance,
            gesture_max_distance: this.settings.maxDistance,
            gesture_sensitivity: this.settings.sensitivity,
            gesture_enable_mouse: this.settings.enableMouse,
            gesture_enable_touch: this.settings.enableTouch,
            gesture_enable_keyboard: this.settings.enableKeyboard
        });
    }

    getAvailableGestures() {
        return Array.from(this.gestures.values()).map(g => ({
            pattern: g.pattern,
            description: g.description
        }));
    }
}

// Export for global use
if (typeof window !== 'undefined') {
    window.TBGestureDetector = TBGestureDetector;
}
