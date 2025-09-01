// tbjs/core/ws.js

import TB from '../index.js';

const WsManager = {
    connection: null,
    url: null,
    reconnectInterval: 5000, // 5 Sekunden
    reconnectAttempts: 5,
    currentReconnects: 0,

    connect(url, options = {}) {
        if (this.connection && this.connection.readyState === WebSocket.OPEN && this.url === url) {
            TB.logger.warn(`WS: Already connected to ${url}`);
            return;
        }

        this.disconnect(); // Schließe bestehende Verbindungen, bevor eine neue aufgebaut wird
        this.url = url;
        const fullUrl = url.startsWith('ws') ? url : `ws://${window.location.host}${url}`;

        TB.logger.log(`WS: Connecting to ${fullUrl}...`);

        try {
            this.connection = new WebSocket(fullUrl);
        } catch (error) {
            TB.logger.error(`WS: Connection failed to initialize for ${fullUrl}`, error);
            if (options.onError) options.onError(error);
            TB.events.emit('ws:error', { error });
            return;
        }

        this.connection.onopen = (event) => {
            TB.logger.log(`WS: Connection opened to ${this.url}`);
            this.currentReconnects = 0; // Setze Wiederverbindungsversuche zurück
            if (options.onOpen) options.onOpen(event);
            TB.events.emit('ws:open', { event });
        };

        this.connection.onmessage = (event) => {
            let parsedData = event.data;
            try {
                if (typeof event.data === 'string') {
                    parsedData = JSON.parse(event.data);
                }
            } catch (e) {
                TB.logger.warn(`WS: Could not parse incoming message as JSON:`, event.data);
            }
            TB.logger.debug(`WS: Message received from ${this.url}`, parsedData);
            if (options.onMessage) options.onMessage(parsedData, event);
            // Globales Event für andere Module
            TB.events.emit('ws:message', { data: parsedData, originEvent: event });
            // Spezifisches Event basierend auf dem 'event'-Feld im Payload
            if (parsedData && parsedData.event) {
                 TB.events.emit(`ws:event:${parsedData.event}`, { data: parsedData.data, originEvent: event });
            }
        };

        this.connection.onerror = (error) => {
            TB.logger.error(`WS: Error with connection to ${this.url}`, error);
            if (options.onError) options.onError(error);
            TB.events.emit('ws:error', { error });
        };

        this.connection.onclose = (event) => {
            TB.logger.warn(`WS: Connection to ${this.url} closed. Code: ${event.code}, Reason: '${event.reason}'`);
            if (options.onClose) options.onClose(event);
            TB.events.emit('ws:close', { event });

            // Logik für automatische Wiederverbindung
            if (this.currentReconnects < this.reconnectAttempts && !event.wasClean) {
                this.currentReconnects++;
                TB.logger.log(`WS: Attempting to reconnect in ${this.reconnectInterval / 1000}s... (${this.currentReconnects}/${this.reconnectAttempts})`);
                setTimeout(() => this.connect(url, options), this.reconnectInterval);
            } else if (!event.wasClean) {
                 TB.logger.error(`WS: Reconnect attempts exhausted for ${this.url}.`);
            }
        };
    },

    send(payload) {
        if (!this.connection || this.connection.readyState !== WebSocket.OPEN) {
            TB.logger.error("WS: Cannot send message, connection is not open.");
            return false;
        }

        try {
            const message = JSON.stringify(payload);
            this.connection.send(message);
            TB.logger.debug("WS: Message sent:", payload);
            return true;
        } catch (error) {
            TB.logger.error("WS: Failed to send message:", error);
            return false;
        }
    },

    disconnect() {
        if (this.connection) {
            TB.logger.log(`WS: Manually disconnecting from ${this.url}`);
            // Deaktiviere die automatische Wiederverbindung vor dem Schließen
            this.reconnectAttempts = 0;
            this.connection.close(1000, "Client initiated disconnect");
            this.connection = null;
            this.url = null;
        }
    },

    getConnection() {
        return this.connection;
    }
};

export default WsManager;
