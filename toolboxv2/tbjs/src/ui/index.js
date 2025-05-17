// tbjs/ui/index.js
// Aggregates and exports all UI modules and components.

import theme from './theme.js';
import htmxIntegration from './htmx-integration.js';
import * as effects from './effects.js'; // For page transitions etc.

// Import components
import Modal from './components/Modal/Modal.js';
import Toast from './components/Toast/Toast.js';
import Loader from './components/Loader/Loader.js';
import Button from './components/Button/Button.js';
import ThreeDeeBackground from './components/ThreeDeeBackground/ThreeDeeBackground.js';
import CookieBanner from './components/CookieBanner/CookieBanner.js';
import MarkdownRenderer from './components/MarkdownRenderer/MarkdownRenderer.js';
import AutocompleteWidget from './components/Autocomplete/Autocomplete.js'; // Renamed to avoid conflict with core util
import NavMenu from './components/NavMenu/NavMenu.js';

/**
 * Processes dynamic content added to the DOM (e.g., by HTMX or router).
 * Re-initializes tbjs UI components or applies global enhancements.
 * @param {HTMLElement} parentElement - The element containing the new content.
 * @param {object} [options={}] - Options for processing.
 * @param {boolean} [options.addScripts=true] - Whether to handle <script> tags.
 * @param {Set} [options.scriptCache] - A Set to track loaded script URLs.
 * Original: Parts of updateDome from original index.js
 */
function processDynamicContent(parentElement, options = {}) {
    if (!parentElement) return;

    const { addScripts = true, scriptCache = new Set() } = options;

    // 1. Initialize HTMX on the new content (if not handled by htmx:afterSwap already)
    window.htmx.process(parentElement);

    // 2. Handle <script> tags (carefully)
    // Original: script loading logic in updateDome from original index.js
    if (addScripts) {
        parentElement.querySelectorAll("script").forEach(script => {
            if (script.src && !scriptCache.has(script.src)) {
                // Basic filtering (you'll need to refine this from your original exclusion list)
                if (script.src.includes('/@vite/client') || script.src.includes('main.js')) { // Example: main.js is your app's entry
                    return;
                }
                const newScript = document.createElement("script");
                newScript.type = script.type || "application/javascript";
                if (script.id) newScript.id = script.id;
                if (script.async) newScript.async = true;
                if (script.defer) newScript.defer = true;

                if (script.src) {
                    newScript.src = script.src;
                    scriptCache.add(script.src);
                    // TB.logger.debug(`[UI] Loading dynamic script: ${script.src}`);
                    document.body.appendChild(newScript); // Or head, depending on script type
                } else if (script.textContent) {
                    // TB.logger.debug('[UI] Executing inline dynamic script.');
                    try {
                        // Careful with eval in production, consider alternatives if possible
                        // new Function(script.textContent)(); // Safer than eval
                         const tempScript = document.createElement('script');
                         tempScript.textContent = script.textContent;
                         document.body.appendChild(tempScript).parentNode.removeChild(tempScript);
                    } catch (e) {
                        // TB.logger.error('[UI] Error executing inline dynamic script:', e);
                    }
                }
            } else if (script.src && scriptCache.has(script.src)) {
                // TB.logger.debug(`[UI] Script already loaded, skipping: ${script.src}`);
            }
        });
    }

    // 3. Initialize any tbjs UI components that might be present in the new HTML
    //    This requires components to have a way to self-initialize or be discoverable.
    //    Example: TB.ui.Modal.initializeAllIn(parentElement);
    //    Example: parentElement.querySelectorAll('[data-tb-component="autocomplete"]').forEach(el => new TB.ui.AutocompleteWidget(el));

    // 4. Re-apply markdown rendering if new markdown content is added
    if (TB.ui.MarkdownRenderer && parentElement.querySelector('.markdown')) { // Assuming .markdown class
        TB.ui.MarkdownRenderer.renderAllIn(parentElement);
    }

    // TB.logger.debug('[UI] Processed dynamic content in:', parentElement);
}


export {
    theme,
    htmxIntegration,
    effects,
    processDynamicContent,
    // Components
    Modal,
    Toast,
    Loader,
    Button,
    ThreeDeeBackground,
    CookieBanner,
    MarkdownRenderer,
    AutocompleteWidget,
    NavMenu,
};
