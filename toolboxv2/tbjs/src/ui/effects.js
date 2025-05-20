// tbjs/ui/htmx-integration.js
// Manages integration with HTMX, especially event handling.
// Original: handleHtmxAfterRequest from original index.js, rendererPipeline from WorkerSocketRenderer.js (for HTMX triggered WS render)

import TB from '../index.js';

const HtmxIntegration = {
    init: () => {
        document.body.addEventListener('htmx:afterRequest', HtmxIntegration.handleAfterRequest);
        document.body.addEventListener('htmx:afterSwap', HtmxIntegration.handleAfterSwap);
        // Potentially listen to other HTMX events like htmx:beforeRequest, htmx:sendError etc.
        TB.logger.log('[HTMX] Integration initialized.');
    },

    handleAfterSwap: (event) => {
        // Content has been swapped into the DOM.
        // Process this new content for tbjs components or scripts.
        const targetElement = event.detail.target;
        if (targetElement) {
            TB.ui.processDynamicContent(targetElement, { scriptCache: TB.router.scriptCache }); // Pass router's script cache
        }
    },

    handleAfterRequest: async (event) => {
        // Original: handleHtmxAfterRequest from original index.js
        const xhr = event.detail.xhr;
        const target = event.detail.target;
        let responseContent = xhr.response;

        // If response is JSON (check Content-Type or try parsing)
        // Your backend should ideally set 'Content-Type: application/json' for API-like responses.
        const contentType = xhr.getResponseHeader('Content-Type');
        let jsonData;

        if (contentType && contentType.includes('application/json')) {
            try {
                jsonData = JSON.parse(responseContent);
            } catch (e) {
                TB.logger.warn('[HTMX] Failed to parse JSON response despite Content-Type:', responseContent, e);
            }
        } else if (typeof responseContent === 'string' && responseContent.trim().startsWith('{') && responseContent.trim().endsWith('}')) {
            // Fallback: try to parse if it looks like JSON
            try {
                jsonData = JSON.parse(responseContent);
            } catch (e) { /* Ignore if not JSON */ }
        }

        if (jsonData) {
            TB.logger.debug('[HTMX] Received JSON response:', jsonData);
            // Attempt to wrap it in your standard Result object
            const result = TB.api.utils.wrapApiResponse ? TB.api.utils.wrapApiResponse(jsonData, 'htmx') : jsonData; // Assuming wrapApiResponse exists in api/utils
                                                                                                        // Or use a simplified local wrapper if api.utils is not ready
            if(result.log) result.log();


            if (result.error && result.error !== TB.ToolBoxError.none) {
                TB.logger.error('[HTMX] Error in response:', result.info.help_text || result.error);
                if (TB.ui.Toast) TB.ui.Toast.showError(result.info.help_text || `Error: ${result.error}`);
                // Prevent HTMX from processing if it's a pure data error and not HTML
                if (!responseContent.trim().startsWith('<')) {
                     event.preventDefault(); // Might not be possible here, HTMX might have already processed.
                                          // Consider hx-swap="none" and handling swap manually for API-like JSON.
                }
                return;
            }

            // Specific handling for "REMOTE" or WebSocket render commands if applicable via HTMX
            // Original: rendererPipeline call in old handleHtmxAfterRequest
            if (result.result && result.result.data_to === TB.ToolBoxInterfaces.remote && result.result.data && result.result.data.render) {
                 TB.logger.log('[HTMX] Remote render command received via HTMX response.');
                 // This assumes your WebSocket rendererPipeline is now part of TB, e.g., TB.sse.handleRenderCommand
                 // Or it could be a specific event
                 TB.events.emit('ws:renderCommand', result.result.data); // or result.get() if Result object
                 // If this render command means no further HTMX processing for this response:
                 if (target && event.detail.elt) { // elt is the element that triggered request
                     // You might want to clear the target or indicate loading if WS will update it
                 }
            } else if (result.result && result.result.data && typeof result.result.data === 'string' && result.result.data.trim().startsWith('<')) {
                // If the processed JSON result *contains* HTML to be rendered by HTMX.
                // HTMX will typically handle this swap itself. We just log.
                TB.logger.debug('[HTMX] JSON response contains HTML. HTMX will swap.');
            }

            // If the JSON response itself is meant to update parts of the page not targeted by HTMX swap:
            TB.events.emit('htmx:jsonResponse', { detail: event.detail, data: result });

        } else {
            // Standard HTML response, HTMX will handle the swap.
            // After swap, `handleAfterSwap` will be called.
            TB.logger.debug('[HTMX] Received HTML response. Target:', target);
        }
    },
};

export default HtmxIntegration;
