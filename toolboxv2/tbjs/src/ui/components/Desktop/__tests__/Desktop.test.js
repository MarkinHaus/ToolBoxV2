/**
 * Desktop Components Tests
 * ========================
 *
 * Tests for QuickCapturePopup, DesktopStatusBar, MobileBottomNav, SystemTray
 */

import {
    QuickCapturePopup,
    DesktopStatusBar,
    MobileBottomNav,
    SystemTray,
    initPlatformUI
} from '../index.js';

// Mock the platform module
jest.mock('../../../../core/platform.js', () => ({
    tauriAPI: {
        invoke: jest.fn(),
        getWorkerStatus: jest.fn().mockResolvedValue({ running: true }),
        notify: jest.fn()
    },
    isDesktop: jest.fn(() => true),
    isMobile: jest.fn(() => false),
    isTauri: jest.fn(() => true)
}));

import { tauriAPI, isDesktop, isMobile } from '../../../../core/platform.js';

describe('QuickCapturePopup', () => {
    let popup;

    beforeEach(() => {
        document.body.innerHTML = '';
        popup = new QuickCapturePopup({
            onCapture: jest.fn().mockResolvedValue(undefined)
        });
    });

    afterEach(() => {
        popup.destroy();
    });

    it('should create popup element on create()', () => {
        popup.create();
        expect(document.querySelector('.tb-quick-capture')).toBeTruthy();
    });

    it('should be hidden by default', () => {
        popup.create();
        expect(popup.element.classList.contains('visible')).toBe(false);
    });

    it('should show popup on show()', () => {
        popup.create();
        popup.show();
        expect(popup.element.classList.contains('visible')).toBe(true);
    });

    it('should hide popup on hide()', () => {
        popup.create();
        popup.show();
        popup.hide();
        expect(popup.element.classList.contains('visible')).toBe(false);
    });

    it('should toggle visibility on toggle()', () => {
        popup.create();
        popup.toggle();
        expect(popup.element.classList.contains('visible')).toBe(true);
        popup.toggle();
        expect(popup.element.classList.contains('visible')).toBe(false);
    });

    it('should respond to tb:quickCapture event', () => {
        popup.create();
        window.dispatchEvent(new CustomEvent('tb:quickCapture'));
        expect(popup.element.classList.contains('visible')).toBe(true);
    });

    it('should have keyboard handler registered', () => {
        popup.create();
        // Verify the handler is set up (we can't easily test keyboard events in jsdom)
        expect(popup._keyHandler).toBeDefined();
    });

    it('should call onCapture when saving', async () => {
        const onCapture = jest.fn().mockResolvedValue(undefined);
        popup = new QuickCapturePopup({ onCapture });
        popup.create();
        popup.show();

        const textarea = popup.element.querySelector('textarea');
        textarea.value = 'Test note #tag1 #tag2';

        const saveBtn = popup.element.querySelector('.tb-quick-capture-save');
        saveBtn.click();

        // Wait for async operation
        await new Promise(resolve => setTimeout(resolve, 10));

        expect(onCapture).toHaveBeenCalledWith({
            text: 'Test note #tag1 #tag2',
            tags: ['tag1', 'tag2']
        });
    });

    it('should extract tags from text when saving', async () => {
        const onCapture = jest.fn().mockResolvedValue(undefined);
        popup = new QuickCapturePopup({ onCapture });
        popup.create();
        popup.show();

        const textarea = popup.element.querySelector('textarea');
        textarea.value = 'Hello #world #test123 no-tag';

        const saveBtn = popup.element.querySelector('.tb-quick-capture-save');
        saveBtn.click();

        await new Promise(resolve => setTimeout(resolve, 10));

        expect(onCapture).toHaveBeenCalledWith({
            text: 'Hello #world #test123 no-tag',
            tags: ['world', 'test123']
        });
    });

    it('should remove element on destroy()', () => {
        popup.create();
        popup.destroy();
        expect(document.querySelector('.tb-quick-capture')).toBeNull();
    });
});

describe('DesktopStatusBar', () => {
    let statusBar;

    beforeEach(() => {
        document.body.innerHTML = '';
        isDesktop.mockReturnValue(true);
        statusBar = new DesktopStatusBar();
    });

    afterEach(() => {
        statusBar.destroy();
    });

    it('should create status bar element on create()', () => {
        statusBar.create();
        expect(document.querySelector('.tb-status-bar')).toBeTruthy();
    });

    it('should add platform-desktop class to body', () => {
        statusBar.create();
        expect(document.body.classList.contains('platform-desktop')).toBe(true);
    });

    it('should display worker status item', () => {
        statusBar.create();
        expect(statusBar.element.querySelector('[data-id="worker"]')).toBeTruthy();
    });

    it('should update status dot class on updateItem()', () => {
        statusBar.create();
        statusBar.updateItem('worker', 'online');
        const dot = statusBar.element.querySelector('[data-id="worker"] .tb-status-dot');
        expect(dot.classList.contains('online')).toBe(true);
    });

    it('should not create on mobile', () => {
        isDesktop.mockReturnValue(false);
        statusBar = new DesktopStatusBar();
        statusBar.create();
        expect(document.querySelector('.tb-status-bar')).toBeNull();
    });

    it('should remove element on destroy()', () => {
        statusBar.create();
        statusBar.destroy();
        expect(document.querySelector('.tb-status-bar')).toBeNull();
        expect(document.body.classList.contains('platform-desktop')).toBe(false);
    });
});

describe('MobileBottomNav', () => {
    let nav;

    beforeEach(() => {
        document.body.innerHTML = '';
        isMobile.mockReturnValue(true);
        nav = new MobileBottomNav({
            items: [
                { icon: '🏠', label: 'Home', route: '/' },
                { icon: '📝', label: 'Notes', route: '/notes' }
            ],
            onNavigate: jest.fn()
        });
    });

    afterEach(() => {
        nav.destroy();
    });

    it('should create bottom nav element on create()', () => {
        nav.create();
        expect(document.querySelector('.tb-bottom-nav')).toBeTruthy();
    });

    it('should add platform-mobile class to body', () => {
        nav.create();
        expect(document.body.classList.contains('platform-mobile')).toBe(true);
    });

    it('should render navigation items', () => {
        nav.create();
        const items = nav.element.querySelectorAll('.tb-bottom-nav-item');
        expect(items.length).toBe(2);
    });

    it('should call onNavigate when item is clicked', () => {
        const onNavigate = jest.fn();
        nav = new MobileBottomNav({
            items: [{ icon: '🏠', label: 'Home', route: '/' }],
            onNavigate
        });
        nav.create();

        const item = nav.element.querySelector('.tb-bottom-nav-item');
        item.click();

        expect(onNavigate).toHaveBeenCalledWith('/');
    });

    it('should set active class on setActive()', () => {
        nav.create();
        nav.setActive('/notes');
        const activeItem = nav.element.querySelector('[data-route="/notes"]');
        expect(activeItem.classList.contains('active')).toBe(true);
    });

    it('should not create on desktop', () => {
        isMobile.mockReturnValue(false);
        nav = new MobileBottomNav({ items: [] });
        nav.create();
        expect(document.querySelector('.tb-bottom-nav')).toBeNull();
    });

    it('should remove element on destroy()', () => {
        nav.create();
        nav.destroy();
        expect(document.querySelector('.tb-bottom-nav')).toBeNull();
        expect(document.body.classList.contains('platform-mobile')).toBe(false);
    });
});

describe('SystemTray', () => {
    let tray;

    beforeEach(() => {
        isDesktop.mockReturnValue(true);
        tray = new SystemTray({ tooltip: 'Test App' });
    });

    it('should call tauriAPI.invoke on updateStatus()', async () => {
        await tray.updateStatus('online');
        expect(tauriAPI.invoke).toHaveBeenCalledWith('update_tray_status', { status: 'online' });
    });

    it('should call tauriAPI.notify on showNotification()', async () => {
        await tray.showNotification('Title', 'Body');
        expect(tauriAPI.notify).toHaveBeenCalledWith('Title', 'Body');
    });
});

describe('initPlatformUI', () => {
    beforeEach(() => {
        document.body.innerHTML = '';
    });

    it('should create desktop components when isDesktop is true', () => {
        isDesktop.mockReturnValue(true);
        isMobile.mockReturnValue(false);

        const components = initPlatformUI();

        expect(components.statusBar).toBeDefined();
        expect(components.quickCapture).toBeDefined();
        expect(components.tray).toBeDefined();
        expect(document.querySelector('.tb-status-bar')).toBeTruthy();
    });

    it('should create mobile components when isMobile is true', () => {
        isDesktop.mockReturnValue(false);
        isMobile.mockReturnValue(true);

        const components = initPlatformUI();

        expect(components.bottomNav).toBeDefined();
        expect(document.querySelector('.tb-bottom-nav')).toBeTruthy();
    });
});

