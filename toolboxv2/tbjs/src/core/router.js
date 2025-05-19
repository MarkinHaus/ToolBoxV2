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

const ROUTER_CACHE_PREFIX = 'tb_router_cache_';
const USE_SESSION_CACHE = false; // Set to false to disable HTML caching

const MAIN_BUNDLE_FILENAMES = ['main.js', 'bundle.js', 'app.js', "vendor"];

const Router = {
    init: (rootElement, predefinedRoutes = []) => {
        if (!rootElement) {
            logger.error('[Router] Root element not provided or not found.');
            return;
        }
        appRootElement = rootElement;
        // TODO: Process predefinedRoutes if needed
        scriptCache = new Set();
        activeViewScripts = new Set();
        currentPath = ''; // Reset currentPath as well
        window.addEventListener('popstate', Router.handlePopState);
        document.addEventListener('click', Router.handleLinkClick);

        const initialUrl = window.location.pathname + window.location.search + window.location.hash;
        //if (initialUrl !== "/index.html"){
        //    console.log(initialUrl)
        //    Router.navigateTo("/web/assets/404.html", true, true);
        //    return;
        //}
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

        if (cleanPath === "/" || cleanPath === "" || cleanPath === "/web" || cleanPath === "/web/") {
            cleanPath = "/web/core0/index.html";
        }

        logger.log(`[Router] Navigating to: ${cleanPath}`);
        events.emit('router:beforeNavigation', { from: currentPath, to: cleanPath });

        const loadingView = Loader ? Loader.show() : null;

        // --- SCRIPT CLEANUP ---
        activeViewScripts.forEach(scriptElement => {
            if (scriptElement.src && scriptElement.src.startsWith('blob:')) {
                URL.revokeObjectURL(scriptElement.src);
                logger.debug(`[Router] Revoked Blob URL: ${scriptElement.src}`);
            }
            scriptElement.remove();
        });
        activeViewScripts.clear();
        // --- END SCRIPT CLEANUP ---

        try {
            let htmlContent;
            let contentSource = 'fetched'; // 'cache' or 'fetched'
            const cacheKey = ROUTER_CACHE_PREFIX + cleanPath;

            if (USE_SESSION_CACHE) {
                const cachedContent = sessionStorage.getItem(cacheKey);
                if (cachedContent) {
                    htmlContent = cachedContent;
                    contentSource = 'cache';
                    logger.log(`[Router] Loaded content for ${cleanPath} from session cache.`);
                }
            }

            if (!htmlContent) {
                htmlContent = await api.fetchHtml(cleanPath);
                if (USE_SESSION_CACHE && htmlContent && !htmlContent.startsWith("HTTP error!")) {
                    try {
                        sessionStorage.setItem(cacheKey, htmlContent);
                    } catch (e) {
                        logger.warn(`[Router] Failed to save to sessionStorage (quota likely exceeded): ${e.message}`);
                        sessionStorage.removeItem(cacheKey); // attempt to clear failed item
                    }
                }
            }

            if (htmlContent.startsWith("HTTP error! status: 404")) {
                logger.warn(`[Router] 404 for path: ${cleanPath}. Redirecting to 404 page.`);
                // Avoid loop if 404 page itself is missing
                if (cleanPath !== '/web/assets/404.html') {
                    await Router.navigateTo('/web/assets/404.html', true);
                } else {
                    appRootElement.innerHTML = '<h1>404 - Page Not Found (and 404.html is also missing)</h1>';
                }
                return;
            }
            if (htmlContent.startsWith("HTTP error! status: 401")) {
                 logger.warn(`[Router] 401 for path: ${cleanPath}. Redirecting to 401 page.`);
                 if (cleanPath !== '/web/assets/401.html') {
                    await Router.navigateTo('/web/assets/401.html', true);
                 } else {
                    appRootElement.innerHTML = '<h1>401 - Unauthorized (and 401.html is also missing)</h1>';
                 }
                return;
            }

            appRootElement.innerHTML = htmlContent;

            // --- SCRIPT PROCESSING (with guard for main bundles) ---
            const scriptsInNewContent = Array.from(appRootElement.querySelectorAll("script"));
            scriptsInNewContent.forEach(oldScriptNode => {
                const newScriptElement = document.createElement('script');
                Array.from(oldScriptNode.attributes).forEach(attr => {
                    newScriptElement.setAttribute(attr.name, attr.value);
                });

                let scriptShouldExecute = false;
                let scriptIdentifier;

                if (oldScriptNode.src) {
                    const scriptUrl = new URL(oldScriptNode.src, window.location.origin); // Resolve to absolute URL
                    const scriptPathName = scriptUrl.pathname.substring(scriptUrl.pathname.lastIndexOf('/') + 1);

                    // GUARD: Skip main application bundle scripts
                    if (MAIN_BUNDLE_FILENAMES.some(bn => scriptPathName.startsWith(bn))) {
                        // logger.warn(`[Router] SKIPPING execution of potential main bundle script found in loaded content: ${scriptUrl.href}`);
                        oldScriptNode.remove(); // Remove it to prevent browser default loading
                        return; // Go to the next script
                    }

                    newScriptElement.src = scriptUrl.href; // Use the full resolved URL
                    scriptIdentifier = newScriptElement.src;

                    if (!scriptCache.has(scriptIdentifier)) {
                        scriptShouldExecute = true;
                    } else {
                        logger.debug(`[Router] Script ${scriptIdentifier} already in global cache, skipping execution.`);
                    }
                } else if (oldScriptNode.getAttribute('unsave') === 'true') {
                    const blob = new Blob([oldScriptNode.textContent], { type: 'application/javascript' });
                    newScriptElement.src = URL.createObjectURL(blob);
                    scriptIdentifier = newScriptElement.src;
                    scriptShouldExecute = true;
                    logger.debug(`[Router] Processing 'unsave' script as Blob: ${scriptIdentifier}`);
                } else { // Inline script, not "unsave"
                    newScriptElement.text = oldScriptNode.textContent;
                    scriptShouldExecute = true;
                }

                if (scriptShouldExecute) {
                    oldScriptNode.parentNode.replaceChild(newScriptElement, oldScriptNode);
                    activeViewScripts.add(newScriptElement);

                    if (newScriptElement.src && !newScriptElement.src.startsWith('blob:')) {
                        newScriptElement.onload = () => {
                            if (!scriptCache.has(scriptIdentifier)) { // Double check before adding
                                scriptCache.add(scriptIdentifier);
                                logger.log(`[Router] Loaded and globally cached script: ${scriptIdentifier}`);
                            }
                        };
                        newScriptElement.onerror = () => {
                            logger.error(`[Router] Error loading script: ${scriptIdentifier}`);
                            activeViewScripts.delete(newScriptElement);
                            newScriptElement.remove();
                        };
                    }
                } else {
                     if (!MAIN_BUNDLE_FILENAMES.some(bn => oldScriptNode.src && oldScriptNode.src.endsWith(bn))) {
                        oldScriptNode.remove(); // Remove if not executed and not a main bundle (which was already handled)
                    }
                }
            });
            // --- END SCRIPT PROCESSING ---

            // Process other dynamic content like HTMX attributes, web components etc.
            // uiProcessDynamicContent(appRootElement, { scriptCache }); // This was the original call.
                                                                    // If it does more than scripts, it might be needed.
                                                                    // For now, assuming script handling above is sufficient.
            if (window.htmx) {
                window.htmx.process(appRootElement);
            }


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
            logger.error(`[Router] Error navigating to ${cleanPath}:`, error);
            events.emit('router:navigationError', { path: cleanPath, error });
            appRootElement.innerHTML = `<div class="p-4 bg-red-100 text-red-700">
                <p>Error loading page: ${cleanPath}.</p>
                <p>${error.message}</p>
                <p>Please try again or contact support.</p>
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
        if (link && link.href && !link.hasAttribute('data-external-link') && !link.hasAttribute('download')) {
            const targetUrl = new URL(link.href, window.location.origin); // Ensure link.href is absolute for comparison
            const currentOrigin = window.location.origin;

            if (targetUrl.origin === currentOrigin) {
                event.preventDefault();
                const navigationPath = targetUrl.pathname + targetUrl.search + targetUrl.hash;
                Router.navigateTo(navigationPath).catch(err => logger.error("[Router] Navigation from link click failed:", err));
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
            logger.log(`[Router] Cleared cache for path: ${path}`);
        } else {
            for (let i = 0; i < sessionStorage.length; i++) {
                const key = sessionStorage.key(i);
                if (key.startsWith(ROUTER_CACHE_PREFIX)) {
                    sessionStorage.removeItem(key);
                }
            }
            logger.log('[Router] Cleared all router page cache from sessionStorage.');
        }
    }
};

export default Router;
