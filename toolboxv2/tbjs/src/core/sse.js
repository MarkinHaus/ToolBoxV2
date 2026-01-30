// tbjs/core/sse.js
import TB from '../index.js'; // Or specific core modules if preferred, e.g., TB.events, TB.config

const SseManager = {
    connections: {}, // Store active EventSource connections by ID/URL

    connect: (url, options = {}) => {
        if (SseManager.connections[url] && SseManager.connections[url].readyState !== EventSource.CLOSED) {
            TB.logger.warn(`SSE: Already connected or connecting to ${url}`);
            return SseManager.connections[url];
        }

        let fullUrl;

        // TAURI OVERRIDE - Priorität 1
        if (TB.env.isTauri()) {
            const workerSseUrl = TB.env.getWorkerSseUrl();
            if (workerSseUrl) {
                if (url.startsWith('/')) {
                    fullUrl = `${workerSseUrl}${url}`;
                } else if (!url.startsWith('http://') && !url.startsWith('https://')) {
                    fullUrl = `${workerSseUrl}/${url}`;
                } else {
                    fullUrl = url;
                }
                TB.logger.debug(`[SSE] Tauri mode - using worker URL: ${fullUrl}`);
            } else {
                // Fallback zu localhost
                fullUrl = url.startsWith('/')
                    ? `http://localhost:5000${url}`
                    : url;
                TB.logger.warn(`[SSE] Tauri mode - no worker URL, using fallback: ${fullUrl}`);
            }
        }
        // WEB MODE - Standard-Verhalten
        else {
            fullUrl = url.startsWith('/')
                ? `${TB.config.get('baseApiUrl').replace('/api', '')}${url}`
                : url;
        }

        const eventSource = new EventSource(fullUrl, options.eventSourceOptions);
        SseManager.connections[url] = eventSource;

        eventSource.onopen = (event) => {
            TB.logger.log(`SSE: Connected to ${fullUrl}`);
            TB.events.emit(`sse:open:${url}`, event);
            if (options.onOpen) options.onOpen(event);
        };

        eventSource.onerror = (error) => {
            TB.logger.error(`SSE: Error with ${fullUrl}`, error);
            TB.events.emit(`sse:error:${url}`, error);
            if (options.onError) options.onError(error);
            eventSource.close();
            delete SseManager.connections[url];
        };

        eventSource.onmessage = (event) => {
            TB.logger.log(`SSE: Message from ${fullUrl}`, event.data);
            TB.events.emit(`sse:message:${url}`, { data: event.data, originEvent: event });
            if (options.onMessage) options.onMessage(event.data, event);
        };

        // Custom event listeners (unverändert)
        if (options.listeners && typeof options.listeners === 'object') {
            for (const eventName in options.listeners) {
                if (Object.hasOwnProperty.call(options.listeners, eventName)) {
                    const handler = options.listeners[eventName];
                    eventSource.addEventListener(eventName, (event) => {
                        let parsedData = event.data;
                        try {
                            if (typeof event.data === 'string' && (event.data.startsWith('{') || event.data.startsWith('['))) {
                                parsedData = JSON.parse(event.data);
                            }
                        } catch (e) {
                            TB.logger.warn(`SSE: Could not parse JSON for event '${eventName}' from ${url}`, event.data);
                        }
                        TB.logger.log(`SSE: Custom event '${eventName}' from ${url}`, parsedData);
                        TB.events.emit(`sse:event:${url}:${eventName}`, { data: parsedData, originEvent: event });
                        handler(parsedData, event);
                    });
                }
            }
        }

        return eventSource;
    },

    disconnect: (url) => {
        if (SseManager.connections[url]) {
            SseManager.connections[url].close();
            delete SseManager.connections[url];
            TB.logger.log(`SSE: Disconnected from ${url}`);
            TB.events.emit(`sse:close:${url}`);
        } else {
            TB.logger.warn(`SSE: No active connection found for ${url} to disconnect.`);
        }
    },

    disconnectAll: () => {
        for (const url in SseManager.connections) {
            SseManager.disconnect(url);
        }
    },

    getConnection: (url) => {
        return SseManager.connections[url];
    }
};

export default SseManager;
