// tbjs/core/sse.js
import TB from '../index.js'; // Or specific core modules if preferred, e.g., TB.events, TB.config

const SseManager = {
    connections: {}, // Store active EventSource connections by ID/URL

    connect: (url, options = {}) => {
        if (SseManager.connections[url] && SseManager.connections[url].readyState !== EventSource.CLOSED) {
            TB.logger.warn(`SSE: Already connected or connecting to ${url}`);
            return SseManager.connections[url];
        }

        const fullUrl = url.startsWith('/') ? `${TB.config.getBaseUrl()}${url}` : url;
        const eventSource = new EventSource(fullUrl, options.eventSourceOptions); // options.eventSourceOptions for withCredentials etc.

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
            // Optionally, close and attempt to reconnect based on strategy
            eventSource.close();
            delete SseManager.connections[url];
        };

        eventSource.onmessage = (event) => {
            // Generic message handler, emits a general event
            // It's often better to handle named events
            TB.logger.log(`SSE: Message from ${fullUrl}`, event.data);
            TB.events.emit(`sse:message:${url}`, { data: event.data, originEvent: event });
            if (options.onMessage) options.onMessage(event.data, event);
        };

        // Allow adding custom event listeners
        if (options.listeners && typeof options.listeners === 'object') {
            for (const eventName in options.listeners) {
                if (Object.hasOwnProperty.call(options.listeners, eventName)) {
                    const handler = options.listeners[eventName];
                    eventSource.addEventListener(eventName, (event) => {
                        let parsedData = event.data;
                        try {
                            // Attempt to parse if it looks like JSON
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
