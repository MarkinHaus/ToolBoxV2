// tbjs/core/router.js
// Handles SPA routing, view fetching, and rendering.
// Original: router, renderer, updateDome, linkEffect, loadHtmlFile, dashboard_init, DOME global var from original index.js

import config from './config.js';
import api from './api.js';
import events from './events.js';
import logger from './logger.js';
import {Loader, processDynamicContent as uiProcessDynamicContent} from '../ui/index.js'; // For loader and processing dynamic content

let appRootElement = null;
let currentPath = '';
let scriptCache = new Set(); // Stores src of loaded scripts to prevent re-fetching/re-executing
let activeViewScripts = new Set(); // Stores script elements dynamically added for the current view
let globalScriptCache = new Set(); // Nur für MAIN_BUNDLE und global="true" Scripts
let activeViewStyles = new Set(); // Track page-specific stylesheets

const ROUTER_CACHE_PREFIX = 'tb_router_cache_';
const USE_SESSION_CACHE = !config.get('isProduction'); // auto tur in production

const MAIN_BUNDLE_FILENAMES = ['main.js', 'bundle.js', 'app.js', "vendor", 'main-'];

const isGlobalScript = (scriptElement, scriptSrc) => {
    // Explizit als global markiert
    if (scriptElement.getAttribute('global') === 'true') {
        return true;
    }

    // Main bundle check
    if (scriptSrc) {
        try {
            const scriptUrl = new URL(scriptSrc, window.location.origin);
            const scriptFileName = scriptUrl.pathname.substring(scriptUrl.pathname.lastIndexOf('/') + 1);
            return MAIN_BUNDLE_FILENAMES.some(bn => scriptFileName.startsWith(bn));
        } catch (e) {
            return false;
        }
    }

    return false;
};

const cleanupPageResources = () => {
    // Cleanup Scripts
    activeViewScripts.forEach((scriptElement, scriptIdentifier) => {
        const isGlobal = isGlobalScript(scriptElement, scriptElement.src);

        if (!isGlobal) {
            // Page-specific: Entferne aus DOM und Cache
            if (scriptElement.src && scriptElement.src.startsWith('blob:')) {
                URL.revokeObjectURL(scriptElement.src);
                logger.debug(`[Router] Revoked Blob URL: ${scriptElement.src}`);
            }

            if (scriptElement.parentNode) {
                scriptElement.remove();
                logger.debug(`[Router] Removed page-specific script: ${scriptIdentifier || 'inline'}`);
            }

            // WICHTIG: Aus globalScriptCache entfernen falls dort
            if (scriptIdentifier) {
                globalScriptCache.delete(scriptIdentifier);
            }

            activeViewScripts.delete(scriptIdentifier);
        } else {
            logger.debug(`[Router] Keeping global script: ${scriptIdentifier || 'inline'}`);
        }
    });

    // Cleanup Page-specific Stylesheets
    activeViewStyles.forEach(styleElement => {
        if (styleElement.parentNode) {
            styleElement.remove();
            logger.debug(`[Router] Removed page-specific stylesheet: ${styleElement.href || 'inline'}`);
        }
    });
    activeViewStyles.clear();
};

const processScripts = (container) => {
    const scriptsInContent = Array.from(container.querySelectorAll("script"));

    scriptsInContent.forEach(oldScriptNode => {
        const scriptSrc = oldScriptNode.src;
        const scriptText = oldScriptNode.textContent;
        const isGlobal = isGlobalScript(oldScriptNode, scriptSrc);

        let scriptIdentifier = null;
        let shouldExecute = false;

        // Determine identifier
        if (scriptSrc) {
            try {
                scriptIdentifier = new URL(scriptSrc, window.location.origin).href;
            } catch (e) {
                scriptIdentifier = scriptSrc;
            }
        } else if (oldScriptNode.getAttribute('unsave') === 'true') {
            // Unsave inline scripts get unique blob identifier later
            scriptIdentifier = `inline_unsave_${Date.now()}_${Math.random()}`;
        }

        // Check if this is a MAIN_BUNDLE (skip completely, already loaded)
        if (scriptSrc) {
            try {
                const scriptUrl = new URL(scriptSrc, window.location.origin);
                const scriptFileName = scriptUrl.pathname.substring(scriptUrl.pathname.lastIndexOf('/') + 1);
                if (MAIN_BUNDLE_FILENAMES.some(bn => scriptFileName.startsWith(bn))) {
                    oldScriptNode.remove();
                    logger.debug(`[Router] Skipped main bundle: ${scriptFileName}`);
                    return;
                }
            } catch (e) { /* continue */ }
        }

        // Decide if script should execute
        if (isGlobal) {
            // Global scripts: nur einmal laden (via globalScriptCache)
            if (scriptIdentifier && globalScriptCache.has(scriptIdentifier)) {
                // Bereits geladen - check ob noch im DOM
                const existingScript = activeViewScripts.get(scriptIdentifier);
                if (existingScript && existingScript.parentNode) {
                    logger.debug(`[Router] Global script already active: ${scriptIdentifier}`);
                    oldScriptNode.remove();
                    return;
                }
            }
            shouldExecute = true;
        } else {
            // Page-specific scripts: IMMER ausführen
            shouldExecute = true;
        }

        if (!shouldExecute) {
            oldScriptNode.remove();
            return;
        }

        // Create and execute new script element
        const newScriptElement = document.createElement('script');
        Array.from(oldScriptNode.attributes).forEach(attr => {
            newScriptElement.setAttribute(attr.name, attr.value);
        });

        if (scriptSrc) {
            newScriptElement.src = scriptIdentifier;
        } else if (oldScriptNode.getAttribute('unsave') === 'true') {
            const blob = new Blob([scriptText], { type: 'application/javascript' });
            const blobUrl = URL.createObjectURL(blob);
            newScriptElement.src = blobUrl;
            scriptIdentifier = blobUrl;
            logger.debug(`[Router] Created blob for unsave script: ${blobUrl}`);
        } else {
            newScriptElement.text = scriptText;
        }

        // Store original content for inline script matching
        if (!scriptSrc) {
            newScriptElement._originalContent = scriptText;
        }

        // Setup load handlers for external scripts
        if (newScriptElement.src && !newScriptElement.src.startsWith('blob:')) {
            newScriptElement.onload = () => {
                if (isGlobal) {
                    globalScriptCache.add(scriptIdentifier);
                    logger.log(`[Router] Global script cached: ${scriptIdentifier}`);
                } else {
                    logger.log(`[Router] Page script loaded: ${scriptIdentifier}`);
                }
            };
            newScriptElement.onerror = () => {
                logger.error(`[Router] Script load error: ${scriptIdentifier}`);
                activeViewScripts.delete(scriptIdentifier);
                if (newScriptElement.parentNode) newScriptElement.remove();
            };
        }

        // Replace old with new and track
        oldScriptNode.parentNode.replaceChild(newScriptElement, oldScriptNode);
        activeViewScripts.set(scriptIdentifier || `inline_${Date.now()}`, newScriptElement);
    });
};

const processStyles = (container) => {
    // Process <link rel="stylesheet"> elements
    const linkElements = Array.from(container.querySelectorAll('link[rel="stylesheet"]'));

    linkElements.forEach(oldLink => {
        const href = oldLink.href;
        const isGlobal = oldLink.getAttribute('global') === 'true';

        if (!isGlobal) {
            // Track page-specific stylesheets for cleanup
            activeViewStyles.add(oldLink);
            logger.debug(`[Router] Tracking page-specific stylesheet: ${href}`);
        }
    });

    // Process inline <style> elements
    const styleElements = Array.from(container.querySelectorAll('style:not([global])'));
    styleElements.forEach(style => {
        activeViewStyles.add(style);
        logger.debug(`[Router] Tracking page-specific inline style`);
    });
};

const Router = {
    init: (rootElement, predefinedRoutes = []) => {
    if (!rootElement) {
        logger.error('[Router] Root element not provided.');
        return;
    }
    appRootElement = rootElement;

    // Reset all state
    globalScriptCache = new Set();
    activeViewScripts = new Map();
    activeViewStyles = new Set();
    currentPath = '';

    window.addEventListener('popstate', Router.handlePopState);
    document.addEventListener('click', Router.handleLinkClick);

    const initialUrl = window.location.pathname + window.location.search + window.location.hash;
    Router.navigateTo(initialUrl, true, true);
    logger.log('[Router] Initialized.', initialUrl);
},

    navigateTo: async (path, replace = false, isInitialLoad = false) => {
    if (path === currentPath && !isInitialLoad) {
        logger.log(`[Router] Already at path: ${path}`);
        return;
    }

    const fullUrl = new URL(path, config.get('baseFileUrl'));
    let cleanPath = fullUrl.pathname + fullUrl.search + fullUrl.hash;

    if (cleanPath.startsWith(config.get('baseFileUrl'))) {
        cleanPath = cleanPath.substring(config.get('baseFileUrl').length);
    }
    if (!cleanPath.startsWith('/')) cleanPath = '/' + cleanPath;

    logger.log(`[Router] Navigating from ${currentPath} to: ${cleanPath}`);
    events.emit('router:beforeNavigation', { from: currentPath, to: cleanPath });

    const loadingView = Loader ? Loader.show({ hideMainContent: false, text: `Routing to ${cleanPath}...` }) : null;

    // --- CLEANUP: Entferne page-specific resources ---
    if (!isInitialLoad) {
        cleanupPageResources();
    }

    try {
        let htmlContent;
        let contentSource = 'fetched';
        const cacheKey = ROUTER_CACHE_PREFIX + cleanPath;

        if (USE_SESSION_CACHE) {
            const cachedContent = sessionStorage.getItem(cacheKey);
            if (cachedContent) {
                htmlContent = cachedContent;
                contentSource = 'cache';
                logger.log(`[Router] Loaded from session cache: ${cleanPath}`);
            }
        }

        if (!htmlContent) {
            htmlContent = await api.fetchHtml(cleanPath);
            if (USE_SESSION_CACHE && htmlContent && !htmlContent.startsWith("HTTP error!")) {
                try {
                    sessionStorage.setItem(cacheKey, htmlContent);
                } catch (e) {
                    logger.warn(`[Router] SessionStorage quota exceeded: ${e.message}`);
                    sessionStorage.removeItem(cacheKey);
                }
            }
        }

        // Error handling (404, 401)
        if (htmlContent.startsWith("HTTP error! status: 404")) {
            logger.warn(`[Router] 404 for path: ${cleanPath}`);
            if (cleanPath !== '/web/assets/404.html') {
                await Router.navigateTo('/web/assets/404.html', true);
            } else {
                appRootElement.innerHTML = '<h1>404 - Page Not Found</h1>';
            }
            return;
        }
        if (htmlContent.startsWith("HTTP error! status: 401")) {
            logger.warn(`[Router] 401 for path: ${cleanPath}`);
            if (cleanPath !== '/web/assets/401.html') {
                await Router.navigateTo('/web/assets/401.html', false);
            } else {
                appRootElement.innerHTML = '<h1>401 - Unauthorized</h1>';
            }
            return;
        }

        // Root content check
        if (htmlContent.includes("<title>Simple</title>")) {
            const isRootPath = ['/', '', '/index.html', '/web', '/web/'].includes(cleanPath);
            if (!isInitialLoad && !isRootPath) {
                logger.warn(`[Router] Non-root path served root content: ${cleanPath}`);
                window.location.href = config.get('baseUrl', '/') + 'web/assets/404.html';
                return;
            } else if (isRootPath || isInitialLoad) {
                return;
            }
        }

        // Render content
        appRootElement.innerHTML = htmlContent;

        // Process scripts and styles
        processScripts(appRootElement);
        processStyles(appRootElement);

        // HTMX processing
        if (window.htmx) {
            window.htmx.process(appRootElement);
        }

        // History management
        if (!replace) {
            history.pushState({ path: cleanPath }, '', cleanPath);
        } else {
            history.replaceState({ path: cleanPath }, '', cleanPath);
        }
        currentPath = cleanPath;

        appRootElement.scrollTop = 0;
        events.emit('router:navigationSuccess', { path: cleanPath, contentSource });
        events.emit('router:contentProcessed', { path: cleanPath, element: appRootElement });

    } catch (error) {
        logger.error(`[Router] Navigation error for ${cleanPath}:`, error);
        events.emit('router:navigationError', { path: cleanPath, error });
        appRootElement.innerHTML = `<div class="p-4 bg-red-100 text-red-700">
            <p>Error loading page: ${cleanPath}.</p>
            <p>${error.message}</p>
        </div>`;
    } finally {
        if (loadingView && Loader) Loader.hide(loadingView);
    }
},

    handlePopState: (event) => {
        const path = event.state ? event.state.path : (window.location.pathname + window.location.search + window.location.hash);
        if (path) {
            logger.log(`[Router] Popstate event for path: ${path}`);
            Router.navigateTo(path, true); // true for replace, effectively just rendering
        }
    },

    handleLinkClick: (event) => {
        const link = event.target.closest('a');
        if (
            link &&
            typeof link.href === 'string' &&
            !link.hasAttribute('data-external-link') &&
            !link.hasAttribute('download')
        ) {
            let targetUrl;
            try {
                targetUrl = new URL(link.href, window.location.origin);
            } catch (e) {
                logger.warn(`[Router] Invalid URL in link: ${link.href}`, e);
                return;
            }

            const currentOrigin = window.location.origin;
            if (targetUrl.origin === currentOrigin) {
                event.preventDefault();

                const pathname = targetUrl.pathname || '/';
                const search = targetUrl.search || '';
                const hash = targetUrl.hash || '';

                const navigationPath = `${pathname}${search}${hash}`;

                Router.navigateTo(navigationPath)
                    .catch(err => logger.error("[Router] Navigation from link click failed:", err));
            } else {
                logger.log(`[Router] External link clicked: ${link.href}`);
            }
        }
    },

    getCurrentPath: () => currentPath,

    clearCache: (path) => {
    if (path) {
        const cacheKey = ROUTER_CACHE_PREFIX + path;
        sessionStorage.removeItem(cacheKey);
        logger.log(`[Router] Cleared cache for: ${path}`);
    } else {
        // Clear all session cache
        for (let i = sessionStorage.length - 1; i >= 0; i--) {
            const key = sessionStorage.key(i);
            if (key && key.startsWith(ROUTER_CACHE_PREFIX)) {
                sessionStorage.removeItem(key);
            }
        }
        // Clear script caches (but keep MAIN_BUNDLE references)
        globalScriptCache.clear();
        activeViewScripts.clear();
        activeViewStyles.clear();
        logger.log('[Router] Cleared all caches.');
    }
}
};

export default Router;
