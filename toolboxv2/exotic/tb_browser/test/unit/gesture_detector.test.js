'use strict';
/**
 * Unit Tests — TBGestureDetector
 */

const chrome = require('../mocks/chrome');

// ─── Inline TBGestureDetector (DOM-independent core logic) ───────────────────
class TBGestureDetector {
  constructor() {
    this.isTracking = false;
    this.gesturePoints = [];
    this.startTime = 0;
    this.lastPoint = null;
    this.isEnabled = true;
    this.mouseMoved = false;
    this.mouseMovedCounter = 0;
    this.settings = {
      minSwipeDistance: 100,
      maxSwipeTime: 800,
      minSwipeVelocity: 0.3,
      scrollSensitivity: 2.0,
      doubleClickDelay: 300,
      enableMouse: true,
      enableTouch: true,
      swipeThreshold: 0.7,
    };
    this.lastClickTime = 0;
    this.clickCount = 0;
    this._executed = [];
  }

  _safeSend(msg) {
    try {
      if (!chrome?.runtime?.id) return;
      chrome.runtime.sendMessage(msg);
    } catch {}
  }

  startGesture(x, y) {
    this.isTracking = true;
    this.gesturePoints = [{ x, y, time: Date.now() }];
    this.startTime = Date.now();
    this.lastPoint = { x, y };
  }

  updateGesture(x, y) {
    if (!this.isTracking) return;
    this.gesturePoints.push({ x, y, time: Date.now() });
    this.lastPoint = { x, y };
    if (this.gesturePoints.length > 50) this.gesturePoints = this.gesturePoints.slice(-25);
  }

  endGesture() {
    if (!this.isTracking) return;
    this.isTracking = false;
    const gesture = this.recognizeGesture();
    if (gesture) this.executeGesture(gesture);
    this.resetGesture();
  }

  recognizeGesture() {
    if (this.gesturePoints.length < 3) return null;
    const start = this.gesturePoints[0];
    const end = this.gesturePoints[this.gesturePoints.length - 1];
    const duration = end.time - start.time;
    if (duration > this.settings.maxSwipeTime) return null;
    const dx = end.x - start.x, dy = end.y - start.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < this.settings.minSwipeDistance) return null;
    const velocity = distance / duration;
    if (velocity < this.settings.minSwipeVelocity) return null;
    const consistency = this.calculateDirectionalConsistency();
    if (consistency < this.settings.swipeThreshold) return null;
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    const absAngle = Math.abs(angle);
    if (absAngle <= 45 || absAngle >= 135) return dx > 0 ? 'swipe-right' : 'swipe-left';
    return dy > 0 ? 'swipe-down' : 'swipe-up';
  }

  calculateDirectionalConsistency() {
    if (this.gesturePoints.length < 3) return 0;
    let consistent = 0, total = 0;
    for (let i = 1; i < this.gesturePoints.length - 1; i++) {
      const prev = this.gesturePoints[i - 1];
      const curr = this.gesturePoints[i];
      const next = this.gesturePoints[i + 1];
      const d1 = Math.atan2(curr.y - prev.y, curr.x - prev.x);
      const d2 = Math.atan2(next.y - curr.y, next.x - curr.x);
      if (Math.abs(d1 - d2) < Math.PI / 4) consistent++;
      total++;
    }
    return total > 0 ? consistent / total : 0;
  }

  executeGesture(gesture) {
    this._executed.push(gesture);
    this._safeSend({ type: 'GESTURE_DETECTED', gesture, timestamp: Date.now() });
  }

  handleTripleClick(e) {
    this._safeSend({ type: 'OPEN_POPUP', position: { x: e?.clientX ?? 0, y: e?.clientY ?? 0 } });
  }

  handleClick(e) {
    const now = Date.now();
    if (now - this.lastClickTime < this.settings.doubleClickDelay) {
      this.clickCount++;
      if (this.clickCount === 3) { this.handleTripleClick(e); this.clickCount = 0; }
    } else {
      this.clickCount = 1;
    }
    this.lastClickTime = now;
  }

  resetGesture() {
    this.gesturePoints = [];
    this.startTime = 0;
    this.lastPoint = null;
  }

  enable()  { this.isEnabled = true; }
  disable() { this.isEnabled = false; this.resetGesture(); }
}

// ─── Helper: build gesture points along a path ────────────────────────────────
function makePoints(dx, dy, count = 10, durationMs = 200) {
  const now = Date.now();
  return Array.from({ length: count }, (_, i) => ({
    x: (dx / count) * i,
    y: (dy / count) * i,
    time: now + (durationMs / count) * i,
  }));
}

// ─── Tests ───────────────────────────────────────────────────────────────────
beforeEach(() => chrome._reset());

describe('startGesture / updateGesture / resetGesture', () => {
  test('startGesture sets tracking and initial point', () => {
    const g = new TBGestureDetector();
    g.startGesture(10, 20);
    expect(g.isTracking).toBe(true);
    expect(g.gesturePoints).toHaveLength(1);
    expect(g.gesturePoints[0]).toMatchObject({ x: 10, y: 20 });
  });

  test('updateGesture appends point only when tracking', () => {
    const g = new TBGestureDetector();
    g.updateGesture(5, 5);
    expect(g.gesturePoints).toHaveLength(0); // not tracking

    g.startGesture(0, 0);
    g.updateGesture(10, 0);
    expect(g.gesturePoints).toHaveLength(2);
  });

  test('gesturePoints capped: never exceeds 50 (slice at 50 → 25)', () => {
    const g = new TBGestureDetector();
    g.startGesture(0, 0);
    for (let i = 0; i < 200; i++) g.updateGesture(i, 0);
    // Rule: if length > 50 → slice to last 25. So max before next slice is 50.
    expect(g.gesturePoints.length).toBeLessThanOrEqual(50);
  });

  test('resetGesture clears state', () => {
    const g = new TBGestureDetector();
    g.startGesture(0, 0);
    g.updateGesture(100, 0);
    g.resetGesture();
    expect(g.gesturePoints).toHaveLength(0);
    expect(g.lastPoint).toBeNull();
  });
});

describe('recognizeGesture', () => {
  test('returns null for < 3 points', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(200, 0, 2);
    expect(g.recognizeGesture()).toBeNull();
  });

  test('detects swipe-right', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(200, 0, 15, 300);
    expect(g.recognizeGesture()).toBe('swipe-right');
  });

  test('detects swipe-left', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(-200, 0, 15, 300);
    expect(g.recognizeGesture()).toBe('swipe-left');
  });

  test('detects swipe-down', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(0, 200, 15, 300);
    expect(g.recognizeGesture()).toBe('swipe-down');
  });

  test('detects swipe-up', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(0, -200, 15, 300);
    expect(g.recognizeGesture()).toBe('swipe-up');
  });

  test('returns null for too slow gesture', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(200, 0, 10, 900); // > maxSwipeTime
    expect(g.recognizeGesture()).toBeNull();
  });

  test('returns null for too short distance', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(10, 0, 10, 100); // < minSwipeDistance
    expect(g.recognizeGesture()).toBeNull();
  });
});

describe('endGesture', () => {
  test('executes recognized gesture', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(200, 0, 15, 300);
    g.isTracking = true;
    g.endGesture();
    expect(g._executed).toContain('swipe-right');
  });

  test('does nothing if not tracking', () => {
    const g = new TBGestureDetector();
    g.endGesture();
    expect(g._executed).toHaveLength(0);
  });

  test('sends GESTURE_DETECTED to background', () => {
    const g = new TBGestureDetector();
    g.gesturePoints = makePoints(200, 0, 15, 300);
    g.isTracking = true;
    g.endGesture();
    expect(chrome.runtime.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'GESTURE_DETECTED', gesture: 'swipe-right' })
    );
  });
});

describe('triple click → OPEN_POPUP', () => {
  test('3 rapid clicks fire OPEN_POPUP', () => {
    const g = new TBGestureDetector();
    const now = Date.now();
    jest.spyOn(Date, 'now').mockReturnValueOnce(now)
      .mockReturnValueOnce(now + 50)
      .mockReturnValueOnce(now + 100);

    g.handleClick({ clientX: 100, clientY: 200 });
    g.handleClick({ clientX: 100, clientY: 200 });
    g.handleClick({ clientX: 100, clientY: 200 });

    expect(chrome.runtime.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'OPEN_POPUP' })
    );
    jest.restoreAllMocks();
  });

  test('slow clicks do not fire OPEN_POPUP', () => {
    const g = new TBGestureDetector();
    g.handleClick({ clientX: 0, clientY: 0 });
    // Wait longer than doubleClickDelay
    g.lastClickTime = Date.now() - 500;
    g.handleClick({ clientX: 0, clientY: 0 });
    g.lastClickTime = Date.now() - 500;
    g.handleClick({ clientX: 0, clientY: 0 });
    expect(chrome.runtime.sendMessage).not.toHaveBeenCalled();
  });
});

describe('_safeSend', () => {
  test('does not throw when context invalidated', () => {
    const g = new TBGestureDetector();
    chrome.runtime.id = null;
    expect(() => g._safeSend({ type: 'TEST' })).not.toThrow();
  });
});

describe('enable / disable', () => {
  test('disable clears gesture state', () => {
    const g = new TBGestureDetector();
    g.startGesture(0, 0);
    g.updateGesture(50, 0);
    g.disable();
    expect(g.isEnabled).toBe(false);
    expect(g.gesturePoints).toHaveLength(0);
  });

  test('enable sets isEnabled true', () => {
    const g = new TBGestureDetector();
    g.disable();
    g.enable();
    expect(g.isEnabled).toBe(true);
  });
});
