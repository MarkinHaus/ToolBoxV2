
class ToolboxConnection {
    constructor() {
        if (!ToolboxConnection.instance) {
            this._events = new Map();
            this._connectToServer();
            ToolboxConnection.instance = this;
        }
        if (!this.client || this.client.destroyed) {
            this._connectToServer();
        }
        return ToolboxConnection.instance;
    }

    _connectToServer() {
        const path = 'tb_socket.sock';
        const net = require('net');
        this.client = net.createConnection({ path }, () => {
            console.log('Verbunden mit dem Server!');
        });

        this.client.on('data', (data) => {
            console.log('Daten empfangen:', data.toString());
            this._handleData(data.toString());
        });

        this.client.on('end', () => {
            console.log('Verbindung zum Server beendet');
        });
    }

    _handleData(data) {
        try {
            const event = JSON.parse(data);
            const eventID = event.event_id.ID;
            if (this._events.has(eventID)) {
                const callback = this._events.get(eventID);
                callback(event);
            }
        } catch (error) {
            console.error('Fehler beim Verarbeiten der Daten:', error);
        }
    }

    registerEvent(eventID, callback) {
        if (typeof callback === 'function') {
            this._events.set(eventID, callback);
        } else {
            console.error('Callback muss eine Funktion sein');
        }
    }

    sendEvent(event) {
        if (this.client && !this.client.destroyed) {
            const data = JSON.stringify(event);
            this.client.write(data);
        } else {
            console.error('Verbindung zum Server nicht aktiv oder unterbrochen');
        }
    }
}

// Stellen Sie sicher, dass die Klasse als Singleton behandelt wird
const instance = new ToolboxConnection();
Object.freeze(instance);

// Anhängen an das window-Objekt, um global verfügbar zu sein
if (typeof window !== 'undefined') {
    window.ToolboxConnection = ToolboxConnection;
}

module.exports = ToolboxConnection;
