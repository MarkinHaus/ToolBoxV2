// ToolBox Pro - Advanced Gesture Detection System
// Handles swipe gestures, double-click activation, and dynamic scrolling

class TBGestureDetector {
    constructor() {
        this.isTracking = false;
        this.gesturePoints = [];
        this.startTime = 0;
        this.lastPoint = null;
        this.isEnabled = true;

        // Gesture settings
        this.settings = {
            minSwipeDistance: 100,
            maxSwipeTime: 800,
            minSwipeVelocity: 0.3,
            scrollSensitivity: 2.0,
            doubleClickDelay: 300,
            enableMouse: true,
            enableTouch: true,
            swipeThreshold: 0.7 // Minimum directional consistency
        };

        // State tracking
        this.lastClickTime = 0;
        this.clickCount = 0;
        this.lastScrollTime = 0;
        this.scrollAccumulator = 0;

        this.init();
    }

    async init() {

        try {
            await this.loadSettings();
            this.setupEventListeners();
            console.log('🎯 Gesture detector initialized');
            // play sound
        } catch (error) {
            console.error('Gesture detector initialization failed:', error);
        }
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get('gestureSettings');
            if (result.gestureSettings) {
                this.settings = { ...this.settings, ...result.gestureSettings };
            }
        } catch (error) {
            console.warn('Failed to load gesture settings, using defaults');
        }
    }

    setupEventListeners() {
        // Touch events for mobile/tablet
        if (this.settings.enableTouch) {
            document.addEventListener('touchstart', (e) => {
                // Only handle if exactly 2 fingers are touching
                if (e.touches.length === 2) {
                    this.handleTouchStart(e);
                }
            }, { passive: false });

            document.addEventListener('touchmove', (e) => {
                // Only handle if exactly 2 fingers are touching
                if (e.touches.length === 2) {
                    this.handleTouchMove(e);
                }
            }, { passive: false });

            document.addEventListener('touchend', (e) => {
                // Handle touch end if we were tracking a 2-finger gesture
                // Check if we had 2 fingers before (touches + changedTouches)
                if (e.touches.length + e.changedTouches.length >= 2) {
                    this.handleTouchEnd(e);
                }
            }, { passive: false });
        }

        // Mouse events for desktop - RIGHT BUTTON ONLY
        if (this.settings.enableMouse) {
            document.addEventListener('mousedown', (e) => {
                // Only handle right mouse button (button === 2)
                if (e.button === 2) {
                    //e.preventDefault(); // Prevent context menu
                    this.handleMouseDown(e);
                }
            });

            document.addEventListener('mousemove', (e) => {
                // Only handle move if right button is held down
                this.handleMouseMove(e);
            });

            document.addEventListener('mouseup', (e) => {
                // Only handle right mouse button release
                if (e.button === 2) {
                    this.handleMouseUp(e);
                }
            });

            // Remove click handler or filter for right button if needed
            document.addEventListener('click', (e) => {
                // Note: right-click typically fires 'contextmenu' event, not 'click'
                // You may want to use 'contextmenu' event instead
                this.handleClick(e);
            });

            // Optional: Prevent context menu on right-click
            document.addEventListener('contextmenu', (e) => {
                e.preventDefault();
            });
        }

        // Wheel events for scroll gestures
        document.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }

    // Touch Event Handlers
    handleTouchStart(e) {
        if (!this.isEnabled || e.touches.length > 1) return;

        const touch = e.touches[0];
        this.startGesture(touch.clientX, touch.clientY);

        // Prevent default for gesture areas
        if (this.isGestureArea(touch.clientX, touch.clientY)) {
            e.preventDefault();
        }
    }

    handleTouchMove(e) {
        if (!this.isTracking || e.touches.length > 1) return;

        const touch = e.touches[0];
        this.updateGesture(touch.clientX, touch.clientY);

        // Prevent scrolling during gesture
        if (this.isTracking) {
            e.preventDefault();
        }
    }

    handleTouchEnd(e) {
        if (!this.isTracking) return;

        this.endGesture();
    }

    // Mouse Event Handlers
    handleMouseDown(e) {
        this.startGesture(e.clientX, e.clientY);
    }

    handleMouseMove(e) {
        if (!this.isTracking) return;

        this.updateGesture(e.clientX, e.clientY);
    }

    handleMouseUp(e) {
        if (!this.isTracking) return;

        this.endGesture();
    }

    handleClick(e) {
        const currentTime = Date.now();
        console.log('click', e.button, currentTime - this.lastClickTime < this.settings.doubleClickDelay, this.clickCount);
        // Double-click detection
        if (currentTime - this.lastClickTime < this.settings.doubleClickDelay) {
            this.clickCount++;
            if (this.clickCount === 3) {
                this.handleTripleClick(e);
                this.clickCount = 0;
            }
        } else {
            this.clickCount = 1;
        }

        this.lastClickTime = currentTime;
    }

    handleWheel(e) {
        if (!this.isEnabled) return;

        const currentTime = Date.now();
        const deltaY = e.deltaY;

        // Accumulate scroll for dynamic scrolling
        this.scrollAccumulator += Math.abs(deltaY);

        // Dynamic scroll based on gesture length
        if (currentTime - this.lastScrollTime > 100) {
            const scrollMultiplier = Math.min(this.scrollAccumulator / 100, 5) * this.settings.scrollSensitivity;

            if (Math.abs(deltaY) > 10) {
                this.performDynamicScroll(deltaY > 0 ? 'down' : 'up', scrollMultiplier);
            }

            this.lastScrollTime = currentTime;
            this.scrollAccumulator = 0;
        }
    }

    handleKeyboard(e) {
        // Alt + Arrow keys for navigation
        if (e.altKey) {
            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.executeGesture('swipe-left');
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.executeGesture('swipe-right');
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.executeGesture('swipe-up');
                    break;
            }
        }
    }

    // Gesture Processing
    startGesture(x, y) {
        this.isTracking = true;
        this.gesturePoints = [{ x, y, time: Date.now() }];
        this.startTime = Date.now();
        this.lastPoint = { x, y };
    }

    updateGesture(x, y) {
        if (!this.isTracking) return;

        const currentTime = Date.now();
        this.gesturePoints.push({ x, y, time: currentTime });
        this.lastPoint = { x, y };

        // Limit points to prevent memory issues
        if (this.gesturePoints.length > 50) {
            this.gesturePoints = this.gesturePoints.slice(-25);
        }
    }

    endGesture() {
        if (!this.isTracking) return;

        this.isTracking = false;

        const gesture = this.recognizeGesture();
        if (gesture) {
            this.executeGesture(gesture);
        }

        this.resetGesture();
    }

    recognizeGesture() {
        if (this.gesturePoints.length < 3) return null;

        const startPoint = this.gesturePoints[0];
        const endPoint = this.gesturePoints[this.gesturePoints.length - 1];
        const duration = endPoint.time - startPoint.time;

        // Check if gesture is too slow
        if (duration > this.settings.maxSwipeTime) return null;

        const deltaX = endPoint.x - startPoint.x;
        const deltaY = endPoint.y - startPoint.y;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

        // Check minimum distance
        if (distance < this.settings.minSwipeDistance) return null;

        // Calculate velocity
        const velocity = distance / duration;
        if (velocity < this.settings.minSwipeVelocity) return null;

        // Determine direction
        const angle = Math.atan2(deltaY, deltaX) * (180 / Math.PI);
        const absAngle = Math.abs(angle);

        // Check directional consistency
        const consistency = this.calculateDirectionalConsistency();
        if (consistency < this.settings.swipeThreshold) return null;

        // Classify gesture
        if (absAngle <= 45 || absAngle >= 135) {
            // Horizontal swipe
            return deltaX > 0 ? 'swipe-right' : 'swipe-left';
        } else {
            // Vertical swipe
            return deltaY > 0 ? 'swipe-down' : 'swipe-up';
        }
    }

    calculateDirectionalConsistency() {
        if (this.gesturePoints.length < 3) return 0;

        let consistentMoves = 0;
        let totalMoves = 0;

        for (let i = 1; i < this.gesturePoints.length - 1; i++) {
            const prev = this.gesturePoints[i - 1];
            const curr = this.gesturePoints[i];
            const next = this.gesturePoints[i + 1];

            const dir1 = Math.atan2(curr.y - prev.y, curr.x - prev.x);
            const dir2 = Math.atan2(next.y - curr.y, next.x - curr.x);

            const angleDiff = Math.abs(dir1 - dir2);
            if (angleDiff < Math.PI / 4) { // Within 45 degrees
                consistentMoves++;
            }
            totalMoves++;
        }

        return totalMoves > 0 ? consistentMoves / totalMoves : 0;
    }

    executeGesture(gesture) {
        console.log(`🎯 Executing gesture: ${gesture}`);

        switch (gesture) {
            case 'swipe-left':
                this.navigateBack();
                break;
            case 'swipe-right':
                this.navigateForward();
                break;
            case 'swipe-up':
                this.scrollUp();
                break;
            case 'swipe-down':
                this.scrollDown();
                break;
        }

        // Send to background script
        chrome.runtime.sendMessage({
            type: 'GESTURE_DETECTED',
            gesture: gesture,
            timestamp: Date.now()
        });
    }

    handleTripleClick(e) {
        console.log('🎯 Double-click detected, opening ToolBox popup');

        // Open extension popup
        chrome.runtime.sendMessage({
            type: 'OPEN_POPUP',
            position: { x: e.clientX, y: e.clientY }
        });
    }

    // Navigation Actions
    navigateBack() {
        window.history.back();
    }

    navigateForward() {
        window.history.forward();
    }

    scrollUp() {
        const scrollAmount = this.calculateDynamicScrollAmount('up');
        window.scrollBy({ top: -scrollAmount, behavior: 'smooth' });
    }

    scrollDown() {
        const scrollAmount = this.calculateDynamicScrollAmount('down');
        window.scrollBy({ top: scrollAmount, behavior: 'smooth' });
    }

    performDynamicScroll(direction, multiplier) {
        const baseAmount = 100;
        const scrollAmount = baseAmount * multiplier;

        window.scrollBy({
            top: direction === 'up' ? -scrollAmount : scrollAmount,
            behavior: 'smooth'
        });
    }

    calculateDynamicScrollAmount(direction) {
        const viewportHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const currentScroll = window.pageYOffset;

        // Calculate percentage of page scrolled
        const scrollPercentage = currentScroll / (documentHeight - viewportHeight);

        // Dynamic scroll amount based on gesture length and page position
        const baseAmount = viewportHeight * 0.3; // 30% of viewport
        const dynamicMultiplier = 1 + (this.gesturePoints.length / 20); // More points = longer gesture

        return Math.min(baseAmount * dynamicMultiplier, viewportHeight * 0.8);
    }

    isGestureArea(x, y) {
        // Define areas where gestures are allowed
        const margin = 50;
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;

        // Allow gestures on edges and center areas
        return (
            x < margin || // Left edge
            x > windowWidth - margin || // Right edge
            y < margin || // Top edge
            y > windowHeight - margin || // Bottom edge
            (x > windowWidth * 0.3 && x < windowWidth * 0.7) // Center area
        );
    }

    resetGesture() {
        this.gesturePoints = [];
        this.startTime = 0;
        this.lastPoint = null;
    }

    // Public API
    enable() {
        this.isEnabled = true;
    }

    disable() {
        this.isEnabled = false;
        this.resetGesture();
    }

    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        chrome.storage.sync.set({ gestureSettings: this.settings });
    }
}

// Initialize gesture detector
if (typeof window !== 'undefined') {
    window.tbGestureDetector = new TBGestureDetector();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TBGestureDetector;
}
