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

const MAIN_BUNDLE_FILENAMES = ['main.js', 'bundle.js', 'app.js', "vendor", 'main-'];

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


        logger.log(`[Router] Navigating to: ${cleanPath}`);
        events.emit('router:beforeNavigation', { from: currentPath, to: cleanPath });

        const loadingView = Loader ? Loader.show() : null;

        // --- SCRIPT CLEANUP (Handles global attribute) ---
        const survivingGlobalScripts = new Set();
        activeViewScripts.forEach(scriptElement => {
            if (scriptElement.getAttribute('global') === 'true') {
                // This script is global. Keep its reference in survivingGlobalScripts.
                // Its DOM element might be removed by the subsequent innerHTML update if it was part of appRootElement.
                survivingGlobalScripts.add(scriptElement);
                logger.debug(`[Router] Global script ${scriptElement.src || 'inline (global attr)'} from previous view is noted. It will not be explicitly removed by cleanup.`);
            } else {
                // This is a page-specific script (no global="true" attribute). Attempt to remove and stop it.
                if (scriptElement.src && scriptElement.src.startsWith('blob:')) {
                    URL.revokeObjectURL(scriptElement.src);
                    logger.debug(`[Router] Revoked Blob URL for page-specific script: ${scriptElement.src}`);
                }
                // Remove the script element from the DOM if it's still there.
                if (scriptElement.parentNode) {
                    scriptElement.remove();
                    logger.debug(`[Router] Removed page-specific script DOM element: ${scriptElement.src || 'inline (page-specific)'}`);
                } else {
                    logger.debug(`[Router] Page-specific script ${scriptElement.src || 'inline (page-specific)'} was already removed from DOM or not attached.`);
                }
            }
        });
        // activeViewScripts will now track scripts for the new page.
        // It starts with surviving global scripts, and new scripts from the loaded content will be added.
        activeViewScripts = survivingGlobalScripts;
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
                    await Router.navigateTo('/web/assets/401.html', false);
                 } else {
                    appRootElement.innerHTML = '<h1>401 - Unauthorized (and 401.html is also missing)</h1>';
                 }
                return;
            }


            if (htmlContent.includes("<title>Simple</title>")){
                logger.warn(`[Router] Content for ${cleanPath} might be root content.`);
                // This check aims to prevent incorrect content rendering for sub-pages.
                // If it's the initial load and it's a root path, this content is expected.
                // Otherwise, if a non-root path gets this "Simple" title content, it's likely an issue.
                const isRootPath = cleanPath === "/" || cleanPath === "" || cleanPath === "/index.html" || cleanPath === "/web" || cleanPath === "/web/";
                if (!isInitialLoad && !isRootPath) {
                     logger.warn(`[Router] Content for non-root path ${cleanPath} unexpectedly contains '<title>Simple</title>'. Redirecting to 404 as a precaution.`);
                     window.location.href = config.get('baseUrl', '/') + 'web/assets/404.html'; // Hard redirect
                     return;
                } else if (isInitialLoad && !isRootPath && cleanPath !== '/web/assets/404.html' && cleanPath !== '/web/assets/401.html') {
                    // If initial load is for a specific sub-page but it serves index.html content
                    logger.warn(`[Router] Initial load for ${cleanPath} served root content. This might indicate a server misconfiguration if ${cleanPath} was meant to be a deep link.`);
                    // Potentially redirect to 404, or let it render if this is acceptable fallback for deep links on fresh load
                    // For now, let it render, but the warning is important.
                }else{
                    return;
                }
            }

            appRootElement.innerHTML = htmlContent;

            // --- SCRIPT PROCESSING (with guard for main bundles and improved global handling) ---
            const scriptsInNewContent = Array.from(appRootElement.querySelectorAll("script"));
            const tempNewActiveScripts = new Set(); // To collect scripts that actually run from this new content

            scriptsInNewContent.forEach(oldScriptNode => {
                const newScriptElement = document.createElement('script');
                Array.from(oldScriptNode.attributes).forEach(attr => {
                    newScriptElement.setAttribute(attr.name, attr.value);
                });

                let scriptShouldExecute = false; // Default: do not execute unless conditions met
                let scriptIdentifier; // Used for scriptCache key (src) or blob URL

                const isGlobalCandidate = newScriptElement.getAttribute('global') === 'true';
                const scriptTextContent = oldScriptNode.textContent; // Capture for inline/unsave

                // Store original text content on newScriptElement for potential matching if it's inline/unsave
                if (!oldScriptNode.src) {
                    newScriptElement.originalTextContent = scriptTextContent;
                }

                if (oldScriptNode.src) {
                    const scriptUrl = new URL(oldScriptNode.src, window.location.origin);
                    const scriptPathName = scriptUrl.pathname.substring(scriptUrl.pathname.lastIndexOf('/') + 1);

                    if (MAIN_BUNDLE_FILENAMES.some(bn => scriptPathName.startsWith(bn))) {
                        oldScriptNode.remove();
                        return; // Skip main bundles
                    }

                    newScriptElement.src = scriptUrl.href;
                    scriptIdentifier = newScriptElement.src;
                    let alreadyActiveGlobal = false;

                    if (isGlobalCandidate) {
                        for (const activeGlobalScript of activeViewScripts) { // Check against surviving globals
                            if (activeGlobalScript.src === scriptIdentifier) {
                                alreadyActiveGlobal = true;
                                // The activeGlobalScript (from previous view) is the one we want to keep.
                                // newScriptElement (from new HTML) is a duplicate declaration.
                                tempNewActiveScripts.add(activeGlobalScript); // Ensure it's tracked for this view
                                logger.debug(`[Router] Global script ${scriptIdentifier} instance from previous page carried over.`);
                                break;
                            }
                        }
                    }

                    if (alreadyActiveGlobal) {
                        oldScriptNode.remove(); // Remove duplicate declaration from new HTML
                        scriptShouldExecute = false;
                    } else if (!scriptCache.has(scriptIdentifier)) {
                        scriptShouldExecute = true;
                    } else {
                        logger.debug(`[Router] Script ${scriptIdentifier} (src) already in global cache, skipping execution.`);
                        oldScriptNode.remove(); // Remove if not executed due to cache
                        // If it was global and cached, it means it ran once. If not in activeViewScripts,
                        // its previous instance is gone. We don't re-run from cache alone if it's not an "active survivor".
                    }

                } else { // Inline or unsave script
                    let alreadyActiveGlobal = false;
                    if (isGlobalCandidate) {
                        for (const activeGlobalScript of activeViewScripts) { // Check against surviving globals
                            if (activeGlobalScript.originalTextContent === scriptTextContent &&
                                (oldScriptNode.hasAttribute('unsave') === activeGlobalScript.hasAttribute('unsave'))) {
                                alreadyActiveGlobal = true;
                                tempNewActiveScripts.add(activeGlobalScript); // Carry over existing instance
                                logger.debug(`[Router] Global inline/unsave script instance with identical content carried over.`);
                                break;
                            }
                        }
                    }

                    if (alreadyActiveGlobal) {
                        oldScriptNode.remove(); // Remove duplicate declaration
                        scriptShouldExecute = false;
                    } else { // Not an already active global instance
                        scriptShouldExecute = true; // Inline/unsave scripts execute if not a duplicate active global
                        if (oldScriptNode.getAttribute('unsave') === 'true') {
                            const blob = new Blob([scriptTextContent], { type: 'application/javascript' });
                            newScriptElement.src = URL.createObjectURL(blob);
                            scriptIdentifier = newScriptElement.src; // Blob URL
                            logger.debug(`[Router] Processing 'unsave' script as Blob: ${scriptIdentifier}`);
                        } else {
                            newScriptElement.text = scriptTextContent;
                            // scriptIdentifier remains undefined for plain inline
                        }
                    }
                }

                if (scriptShouldExecute) {
                    oldScriptNode.parentNode.replaceChild(newScriptElement, oldScriptNode);
                    tempNewActiveScripts.add(newScriptElement); // Track this newly executed script

                    if (newScriptElement.src && !newScriptElement.src.startsWith('blob:')) { // External, non-blob
                        newScriptElement.onload = () => {
                            if (!scriptCache.has(scriptIdentifier)) {
                                scriptCache.add(scriptIdentifier);
                                logger.log(`[Router] Loaded and globally cached script: ${scriptIdentifier}`);
                            }
                        };
                        newScriptElement.onerror = () => {
                            logger.error(`[Router] Error loading script: ${scriptIdentifier}`);
                            tempNewActiveScripts.delete(newScriptElement); // Untrack failed script
                            if(newScriptElement.parentNode) newScriptElement.remove();
                        };
                    }
                }
                // If scriptShouldExecute is false, oldScriptNode was already removed (or was a main bundle).
            });

            // Update activeViewScripts with all scripts now active for this view
            activeViewScripts = tempNewActiveScripts;
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
            logger.log(`[Router] Cleared cache for path: ${path}`);
        } else {
            for (let i = sessionStorage.length - 1; i >= 0; i--) { // Iterate backwards when removing
                const key = sessionStorage.key(i);
                if (key && key.startsWith(ROUTER_CACHE_PREFIX)) {
                    sessionStorage.removeItem(key);
                }
            }
            scriptCache.clear(); // Also clear the JS script src cache
            logger.log('[Router] Cleared all router page cache from sessionStorage and internal script cache.');
        }
    }
};

export default Router;
