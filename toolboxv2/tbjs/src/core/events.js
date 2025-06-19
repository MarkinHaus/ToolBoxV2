// tbjs/core/events.js
// A simple Pub/Sub event emitter.
// Original: No direct equivalent, but needed for decoupled modules.

import logger from './logger.js';

const EventBus = {
    _listeners: {},

    on: (eventName, callback) => {
        if (!EventBus._listeners[eventName]) {
            EventBus._listeners[eventName] = [];
        }
        EventBus._listeners[eventName].push(callback);
        // logger.debug(`[Events] Listener added for: ${eventName}`);
    },

    off: (eventName, callback) => {
        if (!EventBus._listeners[eventName]) {
            return;
        }
        EventBus._listeners[eventName] = EventBus._listeners[eventName].filter(
            listener => listener !== callback
        );
        // logger.debug(`[Events] Listener removed for: ${eventName}`);
    },

    emit: (eventName, data) => {
        if (!EventBus._listeners[eventName]) {
            return;
        }
        // logger.debug(`[Events] Emitting event: ${eventName}`, data);
        EventBus._listeners[eventName].forEach(listener => {
            try {
                listener(data);
            } catch (error) {
                logger.error(`[Events] Error in listener for ${eventName}:`, error);
            }
        });
    },

    once: (eventName, callback) => {
        const onceWrapper = (data) => {
            callback(data);
            EventBus.off(eventName, onceWrapper);
        };
        EventBus.on(eventName, onceWrapper);
    }
};

export default EventBus;
