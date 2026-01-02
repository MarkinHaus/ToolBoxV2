// tbjs/src/core/__tests__/router.test.js
// Tests für das Client-Side Routing Modul

import config from '../config.js';
import api from '../api.js';
import events from '../events.js';
import logger from '../logger.js';

// Mock dependencies
jest.mock('../config.js', () => ({
    get: jest.fn((key) => {
        if (key === 'baseFileUrl') return 'http://localhost';
        if (key === 'baseUrl') return '/';
        if (key === 'isProduction') return false;
        return undefined;
    }),
}));

jest.mock('../api.js', () => ({
    fetchHtml: jest.fn(),
}));

jest.mock('../events.js', () => ({
    emit: jest.fn(),
}));

jest.mock('../logger.js', () => ({
    log: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
}));

// Mock UI components - use mockLoader variable prefix to allow document access
const mockLoaderElement = { remove: jest.fn() };
jest.mock('../../ui/index.js', () => ({
    Loader: {
        show: jest.fn(() => mockLoaderElement),
        hide: jest.fn(),
    },
    processDynamicContent: jest.fn(),
}));

import Router from '../router.js';

describe('Router', () => {
    let rootElement;

    beforeEach(() => {
        // Create root element
        rootElement = document.createElement('div');
        rootElement.id = 'app';
        document.body.appendChild(rootElement);

        // Reset mocks
        jest.clearAllMocks();
        api.fetchHtml.mockResolvedValue('<div>Test Content</div>');

        // Reset history
        window.history.replaceState({}, '', '/');
    });

    afterEach(() => {
        // Cleanup
        if (rootElement && rootElement.parentNode) {
            rootElement.parentNode.removeChild(rootElement);
        }
        // Remove event listeners
        window.removeEventListener('popstate', Router.handlePopState);
        document.removeEventListener('click', Router.handleLinkClick);
    });

    describe('init', () => {
        it('should initialize with root element', () => {
            Router.init(rootElement);

            expect(logger.log).toHaveBeenCalledWith(
                expect.stringContaining('[Router] Initialized'),
                expect.anything()
            );
        });

        it('should log error if no root element provided', () => {
            Router.init(null);

            expect(logger.error).toHaveBeenCalledWith('[Router] Root element not provided.');
        });

        it('should navigate to initial URL on init', () => {
            Router.init(rootElement);

            expect(events.emit).toHaveBeenCalledWith(
                'router:beforeNavigation',
                expect.objectContaining({ to: expect.any(String) })
            );
        });
    });

    describe('navigateTo', () => {
        beforeEach(() => {
            Router.init(rootElement);
            jest.clearAllMocks();
        });

        it('should fetch HTML content for path', async () => {
            api.fetchHtml.mockResolvedValue('<div>Page Content</div>');

            await Router.navigateTo('/test-page');

            expect(api.fetchHtml).toHaveBeenCalledWith('/test-page');
        });

        it('should render fetched content in root element', async () => {
            api.fetchHtml.mockResolvedValue('<div id="content">Hello World</div>');

            await Router.navigateTo('/hello');

            expect(rootElement.innerHTML).toContain('Hello World');
        });

        it('should emit router:beforeNavigation event', async () => {
            await Router.navigateTo('/new-path');

            expect(events.emit).toHaveBeenCalledWith('router:beforeNavigation', expect.objectContaining({
                to: '/new-path'
            }));
        });

        it('should emit router:navigationSuccess event on success', async () => {
            api.fetchHtml.mockResolvedValue('<div>Success</div>');

            await Router.navigateTo('/success');

            expect(events.emit).toHaveBeenCalledWith('router:navigationSuccess', expect.objectContaining({
                path: '/success'
            }));
        });

        it('should not navigate if already at same path', async () => {
            await Router.navigateTo('/same-path');
            jest.clearAllMocks();

            await Router.navigateTo('/same-path');

            expect(api.fetchHtml).not.toHaveBeenCalled();
            expect(logger.log).toHaveBeenCalledWith(expect.stringContaining('Already at path'));
        });

        it('should push state to history by default', async () => {
            const pushStateSpy = jest.spyOn(window.history, 'pushState');

            await Router.navigateTo('/push-test');

            expect(pushStateSpy).toHaveBeenCalledWith(
                expect.objectContaining({ path: '/push-test' }),
                '',
                '/push-test'
            );
            pushStateSpy.mockRestore();
        });

        it('should replace state when replace=true', async () => {
            const replaceStateSpy = jest.spyOn(window.history, 'replaceState');

            await Router.navigateTo('/replace-test', true);

            expect(replaceStateSpy).toHaveBeenCalled();
            replaceStateSpy.mockRestore();
        });
    });

    describe('error handling', () => {
        beforeEach(() => {
            Router.init(rootElement);
            jest.clearAllMocks();
        });

        it('should navigate to 404 page on HTTP 404', async () => {
            api.fetchHtml.mockResolvedValue('HTTP error! status: 404');

            await Router.navigateTo('/not-found');

            expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining('404'));
        });

        it('should navigate to 401 page on HTTP 401', async () => {
            api.fetchHtml.mockResolvedValue('HTTP error! status: 401');

            await Router.navigateTo('/unauthorized');

            expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining('401'));
        });

        it('should emit router:navigationError on fetch error', async () => {
            api.fetchHtml.mockRejectedValue(new Error('Network error'));

            await Router.navigateTo('/error-page');

            expect(events.emit).toHaveBeenCalledWith('router:navigationError', expect.objectContaining({
                path: '/error-page',
                error: expect.any(Error)
            }));
        });

        it('should display error message in root element on error', async () => {
            api.fetchHtml.mockRejectedValue(new Error('Test error'));

            await Router.navigateTo('/error');

            expect(rootElement.innerHTML).toContain('Error loading page');
        });
    });

    describe('getCurrentPath', () => {
        beforeEach(() => {
            Router.init(rootElement);
        });

        it('should return current path after navigation', async () => {
            await Router.navigateTo('/current-path-test');

            expect(Router.getCurrentPath()).toBe('/current-path-test');
        });
    });

    describe('handleLinkClick', () => {
        beforeEach(() => {
            Router.init(rootElement);
            jest.clearAllMocks();
        });

        it('should intercept internal link clicks', async () => {
            const link = document.createElement('a');
            link.href = 'http://localhost/internal-page';
            rootElement.appendChild(link);

            const event = new MouseEvent('click', { bubbles: true });
            Object.defineProperty(event, 'target', { value: link });

            // Simulate click
            link.dispatchEvent(event);

            // Should have attempted navigation
            await new Promise(r => setTimeout(r, 50));
            expect(api.fetchHtml).toHaveBeenCalled();
        });

        it('should not intercept external links', () => {
            const link = document.createElement('a');
            link.href = 'https://external-site.com/page';
            rootElement.appendChild(link);

            const event = new MouseEvent('click', { bubbles: true });
            link.dispatchEvent(event);

            expect(logger.log).toHaveBeenCalledWith(expect.stringContaining('External link'));
        });

        it('should not intercept links with data-external-link attribute', () => {
            const link = document.createElement('a');
            link.href = 'http://localhost/page';
            link.setAttribute('data-external-link', '');
            rootElement.appendChild(link);

            const event = new MouseEvent('click', { bubbles: true });
            link.dispatchEvent(event);

            // Should not navigate via router
            expect(api.fetchHtml).not.toHaveBeenCalled();
        });

        it('should not intercept download links', () => {
            const link = document.createElement('a');
            link.href = 'http://localhost/file.pdf';
            link.setAttribute('download', '');
            rootElement.appendChild(link);

            const event = new MouseEvent('click', { bubbles: true });
            link.dispatchEvent(event);

            expect(api.fetchHtml).not.toHaveBeenCalled();
        });
    });

    describe('clearCache', () => {
        beforeEach(() => {
            Router.init(rootElement);
            sessionStorage.setItem('tb_router_cache_/page1', '<div>Page 1</div>');
            sessionStorage.setItem('tb_router_cache_/page2', '<div>Page 2</div>');
        });

        it('should clear specific path cache', () => {
            Router.clearCache('/page1');

            expect(sessionStorage.getItem('tb_router_cache_/page1')).toBeNull();
            expect(sessionStorage.getItem('tb_router_cache_/page2')).not.toBeNull();
        });

        it('should clear all caches when no path provided', () => {
            Router.clearCache();

            expect(sessionStorage.getItem('tb_router_cache_/page1')).toBeNull();
            expect(sessionStorage.getItem('tb_router_cache_/page2')).toBeNull();
        });
    });

    describe('handlePopState', () => {
        beforeEach(() => {
            Router.init(rootElement);
            jest.clearAllMocks();
        });

        it('should navigate to path from popstate event', async () => {
            const event = new PopStateEvent('popstate', {
                state: { path: '/back-page' }
            });

            Router.handlePopState(event);

            await new Promise(r => setTimeout(r, 50));
            expect(api.fetchHtml).toHaveBeenCalledWith('/back-page');
        });

        it('should handle popstate with null state gracefully', async () => {
            // This tests that handlePopState doesn't throw when state is null
            const event = new PopStateEvent('popstate', { state: null });

            expect(() => Router.handlePopState(event)).not.toThrow();
        });
    });
});

