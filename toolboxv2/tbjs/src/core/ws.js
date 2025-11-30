// tbjs/core/ws.js - Erweiterte WebSocket-Client-Implementierung

import TB from '../index.js';

const WsManager = {
    connection: null,
    url: null,
    reconnectInterval: 5000,
    reconnectAttempts: 5,
    currentReconnects: 0,
    connectionContext: null, // NEU: Speichert Verbindungs-Kontext

    /**
     * Erweiterte Connect-Methode mit Kontext-Unterstützung
     * @param {string} url - WebSocket-URL
     * @param {Object} options - Verbindungsoptionen
     * @param {Object} options.context - Zusätzliche Kontext-Daten (z.B. room_id, user_token)
     */
    connect(url, options = {}) {
        if (this.connection && this.connection.readyState === WebSocket.OPEN && this.url === url) {
            TB.logger.warn(`WS: Already connected to ${url}`);
            return;
        }

        this.disconnect();
        this.url = url;

        // NEU: Kontext-Daten in URL-Parameter einbetten
        const contextParams = this._buildContextParams(options.context || {});
        const fullUrl = this._buildFullUrl(url, contextParams);

        TB.logger.log(`WS: Connecting to ${fullUrl}...`);

        try {
            this.connection = new WebSocket(fullUrl);
            this.connectionContext = options.context || {};
        } catch (error) {
            TB.logger.error(`WS: Connection failed to initialize for ${fullUrl}`, error);
            if (options.onError) options.onError(error);
            TB.events.emit('ws:error', { error });
            return;
        }

        // Event-Handler (unverändert, aber mit Kontext-Events)
        this.connection.onopen = (event) => {
            TB.logger.log(`WS: Connection opened to ${this.url}`);
            this.currentReconnects = 0;

            if (options.onOpen) options.onOpen(event);

            // NEU: Emit mit Kontext
            TB.events.emit('ws:open', {
                event,
                context: this.connectionContext
            });
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

            // Globales Event mit Kontext
            TB.events.emit('ws:message', {
                data: parsedData,
                originEvent: event,
                context: this.connectionContext  // NEU: Kontext mitliefern
            });

            // Spezifisches Event
            if (parsedData && parsedData.event) {
                TB.events.emit(`ws:event:${parsedData.event}`, {
                    data: parsedData.data,
                    originEvent: event,
                    context: this.connectionContext
                });
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

            // Wiederverbindungslogik
            if (this.currentReconnects < this.reconnectAttempts && !event.wasClean) {
                this.currentReconnects++;
                TB.logger.log(`WS: Attempting to reconnect in ${this.reconnectInterval / 1000}s... (${this.currentReconnects}/${this.reconnectAttempts})`);
                setTimeout(() => this.connect(url, options), this.reconnectInterval);
            } else if (!event.wasClean) {
                TB.logger.error(`WS: Reconnect attempts exhausted for ${this.url}.`);
            }
        };
    },

    /**
     * NEU: Kontext-Parameter in URL-String umwandeln
     * @private
     */
    _buildContextParams(context) {
        const params = new URLSearchParams();

        for (const [key, value] of Object.entries(context)) {
            if (value !== undefined && value !== null) {
                params.append(key, value.toString());
            }
        }

        return params.toString();
    },

    /**
     * NEU: Vollständige WebSocket-URL erstellen
     * @private
     */
    _buildFullUrl(url, contextParams) {
        let fullUrl;

        if (url.startsWith('ws://') || url.startsWith('wss://')) {
            fullUrl = url;
        } else {
            const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            fullUrl = `${protocol}${window.location.host}${url}`;
        }

        // Kontext-Parameter anhängen
        if (contextParams) {
            const separator = fullUrl.includes('?') ? '&' : '?';
            fullUrl = `${fullUrl}${separator}${contextParams}`;
        }

        return fullUrl;
    },

    /**
     * Nachricht mit optionalen Kontext-Daten senden
     */
    send(payload, includeContext = false) {
        if (!this.connection || this.connection.readyState !== WebSocket.OPEN) {
            TB.logger.error("WS: Cannot send message, connection is not open.");
            return false;
        }

        try {
            // NEU: Optional Kontext in Payload einbetten
            let messagePayload = payload;

            if (includeContext && this.connectionContext) {
                messagePayload = {
                    ...payload,
                    _context: this.connectionContext
                };
            }

            const message = JSON.stringify(messagePayload);
            this.connection.send(message);
            TB.logger.debug("WS: Message sent:", messagePayload);
            return true;
        } catch (error) {
            TB.logger.error("WS: Failed to send message:", error);
            return false;
        }
    },

    disconnect() {
        if (this.connection) {
            TB.logger.log(`WS: Manually disconnecting from ${this.url}`);
            this.reconnectAttempts = 0;
            this.connection.close(1000, "Client initiated disconnect");
            this.connection = null;
            this.url = null;
            this.connectionContext = null;  // NEU: Kontext löschen
        }
    },

    getConnection() {
        return this.connection;
    },

    /**
     * NEU: Gibt den aktuellen Verbindungs-Kontext zurück
     */
    getContext() {
        return this.connectionContext;
    },

    /**
     * NEU: Aktualisiert den Verbindungs-Kontext
     */
    updateContext(newContext) {
        this.connectionContext = {
            ...this.connectionContext,
            ...newContext
        };
        TB.logger.debug("WS: Context updated:", this.connectionContext);
    }
};

export default WsManager;
