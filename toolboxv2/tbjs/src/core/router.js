// tbjs/core/router.js
// Handles SPA routing, view fetching, and rendering.
// Original: router, renderer, updateDome, linkEffect, loadHtmlFile, dashboard_init, DOME global var from original index.js

import config from './config.js';
import api from './api.js';
import events from './events.js';
import logger from './logger.js';
import {Loader,processDynamicContent} from '../ui/index.js'; // For loader and processing dynamic content

let appRootElement = null;
let currentPath = '';
let scriptCache = new Set(); // Replaces scriptSto

const Router = {
    init: (rootElement, predefinedRoutes = []) => {
        if (!rootElement) {
            logger.error('[Router] Root element not provided or not found.');
            return;
        }
        appRootElement = rootElement;
        // TODO: Process predefinedRoutes if needed (e.g., for static content or components)

        window.addEventListener('popstate', Router.handlePopState);
        document.addEventListener('click', Router.handleLinkClick); // Delegated event listener

        // Initial route handling
        const initialUrl = window.location.pathname + window.location.search + window.location.hash;
        Router.navigateTo(initialUrl, true, true); // true for replaceState, true for isInitialLoad
        logger.log('[Router] Initialized.');
    },

    navigateTo: async (path, replace = false, isInitialLoad = false) => {
        if (path === currentPath && !isInitialLoad) {
            logger.log(`[Router] Already at path: ${path}`);
            return;
        }

        const fullUrl = new URL(path, config.get('baseFileUrl')); // Ensure it's absolute for fetching
        let cleanPath = fullUrl.pathname + fullUrl.search + fullUrl.hash;

        // Normalize: if baseFileUrl is origin, path might be like /web/core0/index.html
        // If path starts with baseFileUrl, strip it for history state
        if (cleanPath.startsWith(config.get('baseFileUrl'))) {
           cleanPath = cleanPath.substring(config.get('baseFileUrl').length);
        }
        if (!cleanPath.startsWith('/')) cleanPath = '/' + cleanPath;


        // Default route (original: if url === "/" etc. in old router)
        if (cleanPath === "/" || cleanPath === "" || cleanPath === "/web" || cleanPath === "/web/") {
            cleanPath = "/web/core0/index.html"; // Adjust to your new default
        }

        logger.log(`[Router] Navigating to: ${cleanPath}`);
        events.emit('router:beforeNavigation', { from: currentPath, to: cleanPath });

        // Show loader (original: createLoadingView, createLoadingManager)
        const loadingView = Loader ? Loader.show() : null;

        try {
            // Fetch content (original: loadHtmlFile, fetchFromLocal, fetchFromBackend)
            const htmlContent = await api.fetchHtml(cleanPath); // api.fetchHtml handles fallbacks

            if (htmlContent.startsWith("HTTP error! status: 404")) {
                logger.warn(`[Router] 404 for path: ${cleanPath}. Redirecting to 404 page.`);
                await Router.navigateTo('/web/assets/404.html'); // Adjust path
                return;
            }
            if (htmlContent.startsWith("HTTP error! status: 401")) {
                 logger.warn(`[Router] 401 for path: ${cleanPath}. Redirecting to 401 page.`);
                await Router.navigateTo('/web/assets/401.html'); // Adjust path
                return;
            }

            // Render content (original: renderer function)
            appRootElement.innerHTML = htmlContent;
            processDynamicContent(appRootElement, { addScripts: true, scriptCache }); // Replaces updateDome

            if (!replace) {
                history.pushState({ path: cleanPath }, '', cleanPath);
            } else {
                history.replaceState({ path: cleanPath }, '', cleanPath);
            }
            currentPath = cleanPath;

            appRootElement.scrollTop = 0; // Scroll to top
            events.emit('router:navigationSuccess', { path: cleanPath, contentSource: 'fetched' });

        } catch (error) {
            logger.error(`[Router] Error navigating to ${cleanPath}:`, error);
            events.emit('router:navigationError', { path: cleanPath, error });
            // Optionally, render an error view
            // appRootElement.innerHTML = `<p>Error loading page. Please try again.</p>`;
        } finally {
            if (loadingView && Loader) Loader.hide(loadingView);
        }
    },

    handlePopState: (event) => {
        const path = event.state ? event.state.path : (window.location.pathname + window.location.search);
        if (path) {
            logger.log(`[Router] Popstate event for path: ${path}`);
            // Don't push to history, just render
            Router.navigateTo(path, true); // true for replace, effectively just rendering
        }
    },

    handleLinkClick: (event) => {
        const link = event.target.closest('a');
        if (link && link.href && !link.hasAttribute('data-external-link') && !link.hasAttribute('download')) {
            const targetUrl = new URL(link.href);
            const currentOrigin = new URL(window.location.href).origin;

            // Only handle internal links
            if (targetUrl.origin === currentOrigin) {
                event.preventDefault();

                Router.navigateTo(link.pathname + link.search + link.hash).then(r => logger.log("[Don Routing]"));
            } else {
                logger.log(`[Router] External link clicked: ${link.href}`);
                // Let browser handle external link
            }
        }
    },

    getCurrentPath: () => currentPath,
};

export default Router;
