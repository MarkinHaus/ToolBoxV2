/**
 * BrainManager Component Tests
 * ============================
 */

import { BrainManager } from '../BrainManager.js';

// Mock the platform module
jest.mock('../../../../core/platform.js', () => ({
    tauriAPI: {
        invoke: jest.fn()
    },
    isDesktop: jest.fn(() => true),
    isMobile: jest.fn(() => false),
    isTauri: jest.fn(() => true)
}));

// Mock fetch
global.fetch = jest.fn();

// Mock WebSocket
class MockWebSocket {
    constructor(url) {
        this.url = url;
        this.onopen = null;
        this.onmessage = null;
        this.onclose = null;
        this.onerror = null;
        this.readyState = 1;
        setTimeout(() => this.onopen?.(), 0);
    }
    send(data) { this.lastSent = data; }
    close() { this.onclose?.(); }
}
global.WebSocket = MockWebSocket;

describe('BrainManager', () => {
    let manager;
    let container;

    beforeEach(() => {
        document.body.innerHTML = '';
        container = document.createElement('div');
        document.body.appendChild(container);

        global.fetch.mockReset();
        global.fetch.mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({
                discord: { connected: true, users: 5 },
                telegram: { connected: false, users: 0 },
                vault: { connected: true, notes: 42 },
                sync: { active: true, lastSync: Date.now() }
            })
        });

        manager = new BrainManager({
            container,
            apiBase: '/api',
            onStatusChange: jest.fn()
        });
    });

    afterEach(() => {
        manager.destroy();
    });

    it('should create brain manager UI on create()', () => {
        manager.create(container);
        expect(container.querySelector('.tb-brain-header')).toBeTruthy();
        expect(container.querySelector('.tb-brain-status-grid')).toBeTruthy();
    });

    it('should render all service cards', () => {
        manager.create(container);
        expect(container.querySelector('[data-service="discord"]')).toBeTruthy();
        expect(container.querySelector('[data-service="telegram"]')).toBeTruthy();
        expect(container.querySelector('[data-service="vault"]')).toBeTruthy();
        expect(container.querySelector('[data-service="sync"]')).toBeTruthy();
    });

    it('should render quick action buttons', () => {
        manager.create(container);
        expect(container.querySelector('[data-action="capture"]')).toBeTruthy();
        expect(container.querySelector('[data-action="note"]')).toBeTruthy();
        expect(container.querySelector('[data-action="daily"]')).toBeTruthy();
        expect(container.querySelector('[data-action="search"]')).toBeTruthy();
    });

    it('should fetch initial status on create()', async () => {
        manager.create(container);
        await new Promise(resolve => setTimeout(resolve, 10));
        expect(global.fetch).toHaveBeenCalledWith('/api/brain/status');
    });

    it('should update status display after fetch', async () => {
        manager.create(container);
        await new Promise(resolve => setTimeout(resolve, 50));

        const discordCard = container.querySelector('[data-service="discord"]');
        const dot = discordCard.querySelector('.tb-brain-card-dot');
        expect(dot.classList.contains('online')).toBe(true);
    });

    it('should set all services offline on fetch error', async () => {
        global.fetch.mockRejectedValue(new Error('Network error'));
        manager.create(container);
        await new Promise(resolve => setTimeout(resolve, 50));

        const discordCard = container.querySelector('[data-service="discord"]');
        const dot = discordCard.querySelector('.tb-brain-card-dot');
        expect(dot.classList.contains('offline')).toBe(true);
    });

    it('should dispatch tb:quickCapture event on capture action', () => {
        manager.create(container);
        const dispatchSpy = jest.spyOn(window, 'dispatchEvent');

        const captureBtn = container.querySelector('[data-action="capture"]');
        captureBtn.click();

        expect(dispatchSpy).toHaveBeenCalledWith(
            expect.objectContaining({ type: 'tb:quickCapture' })
        );
        dispatchSpy.mockRestore();
    });

    it('should format time correctly', () => {
        manager.create(container);

        const now = Date.now();
        expect(manager._formatTime(now)).toBe('Just now');
        expect(manager._formatTime(now - 120000)).toBe('2m ago');
        expect(manager._formatTime(now - 7200000)).toBe('2h ago');
    });

    it('should add activity to list', () => {
        manager.create(container);
        manager._addActivity({ icon: '📝', text: 'Test activity', time: Date.now() });

        const activityList = container.querySelector('.tb-brain-activity-list');
        expect(activityList.querySelector('.tb-brain-activity-item')).toBeTruthy();
    });

    it('should limit activities to 10 items', () => {
        manager.create(container);
        for (let i = 0; i < 15; i++) {
            manager._addActivity({ text: `Activity ${i}`, time: Date.now() });
        }
        expect(manager.activities.length).toBe(10);
    });

    it('should clean up on destroy()', () => {
        manager.create(container);
        manager.destroy();
        expect(container.innerHTML).toBe('');
    });
});

