// tbjs/ui/components/MarkdownRenderer/MarkdownRenderer.js
import TB from '../../../index.js';
// Assumes marked and highlight.js are loaded globally or imported
// import { marked } from 'marked';
// import { markedHighlight } from 'marked-highlight';
// import hljs from 'highlight.js/lib/core';
// Then register languages: hljs.registerLanguage('javascript', require('highlight.js/lib/languages/javascript'));

let markedInstance; // To store initialized marked with highlight

const MarkdownRenderer = {
    _isInitialized: false,

    async init() {
        if (MarkdownRenderer._isInitialized || typeof window.marked === 'undefined' || typeof window.hljs === 'undefined') {
            if (!MarkdownRenderer._isInitialized && (typeof window.marked === 'undefined' || typeof window.hljs === 'undefined')) {
                TB.logger.warn('[Markdown] `marked` or `hljs` not found globally. Dynamic import attempt or manual setup needed.');
                // You might try dynamic import here if they are not globally available
                // try {
                // const { marked } = await import('marked');
                // const { markedHighlight } = await import('marked-highlight');
                // const hljs = (await import('highlight.js/lib/core')).default;
                // /* import and register languages for hljs */
                // window.marked = marked; window.markedHighlight = markedHighlight; window.hljs = hljs;
                // } catch(e) { TB.logger.error('[Markdown] Failed to load dependencies.', e); return; }
            } else if (MarkdownRenderer._isInitialized) return;
             else return; // Not ready
        }

        try {
            markedInstance = window.marked.marked; // Access the function correctly
            const { markedHighlight } = window.markedHighlight; // If it's an object export

            markedInstance.use(markedHighlight({
                langPrefix: 'hljs language-',
                highlight(code, lang) {
                    const language = window.hljs.getLanguage(lang) ? lang : 'plaintext';
                    return window.hljs.highlight(code, { language }).value;
                }
            }));
            MarkdownRenderer._isInitialized = true;
            TB.logger.log('[Markdown] Renderer initialized with highlight.js.');
        } catch(e) {
            TB.logger.error('[Markdown] Error initializing marked with highlight:', e, window.marked, window.markedHighlight);
             // Fallback to basic marked if highlight setup fails
            if (window.marked && typeof window.marked.marked === 'function') {
                 markedInstance = window.marked.marked;
                 TB.logger.warn('[Markdown] Initialized with basic `marked` (no syntax highlighting).');
                 MarkdownRenderer._isInitialized = true;
            } else if (window.marked && typeof window.marked === 'function'){ // If marked is the function itself
                 markedInstance = window.marked;
                 TB.logger.warn('[Markdown] Initialized with basic `marked` (no syntax highlighting, direct function).');
                 MarkdownRenderer._isInitialized = true;
            }
        }
    },

    render: (markdownString) => {
        if (!MarkdownRenderer._isInitialized) MarkdownRenderer.init();
        if (!markedInstance) {
            TB.logger.warn('[Markdown] Not initialized, returning raw string.');
            return markdownString; // Or throw error
        }
        return markedInstance(markdownString);
    },

    renderElement: (element) => {
        if (element && element.matches('.markdown')) { // Assuming .markdown class for auto-rendering
            element.innerHTML = MarkdownRenderer.render(element.textContent || element.innerHTML); // Use textContent to avoid re-rendering HTML
            element.classList.add('prose', 'dark:prose-invert', 'max-w-none'); // Tailwind prose classes
            element.classList.remove('markdown'); // Prevent re-rendering
            element.dataset.tbMarkdownRendered = 'true';
        }
    },

    renderAllIn: (parentElement) => {
        if (!MarkdownRenderer._isInitialized) MarkdownRenderer.init();
        if (!markedInstance) return;

        parentElement.querySelectorAll('.markdown:not([data-tb-markdown-rendered="true"])').forEach(el => {
            MarkdownRenderer.renderElement(el);
        });
    },

    // For IntersectionObserver based rendering (original setupMutationObserver)
    // This is more complex and might be better handled by a dedicated "lazy content" module
    // or by simply calling renderAllIn after HTMX swaps.
    // For now, keep it simple: call renderAllIn from TB.ui.processDynamicContent
};

// Auto-init if marked and hljs are already global
if (typeof window !== 'undefined' && window.marked && window.hljs) {
    MarkdownRenderer.init();
}

export default MarkdownRenderer;
