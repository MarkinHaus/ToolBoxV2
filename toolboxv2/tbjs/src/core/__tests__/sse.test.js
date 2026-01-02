// tbjs/src/core/__tests__/sse.test.js
// Tests für das Server-Sent Events Modul

// Mock TB before importing SseManager
jest.mock('../index.js', () => ({
    logger: {
        log: jest.fn(),
        debug: jest.fn(),
        warn: jest.fn(),
        error: jest.fn(),
    },
    events: {
        emit: jest.fn(),
    },
    config: {
        get: jest.fn((key) => {
            if (key === 'baseApiUrl') return 'http://localhost/api';
            return undefined;
        }),
    },
}));

import SseManager from '../sse.js';
import TB from '../index.js';

// Mock EventSource
class MockEventSource {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 2;

    constructor(url, options = {}) {
        this.url = url;
        this.withCredentials = options.withCredentials || false;
        this.readyState = MockEventSource.CONNECTING;
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
        this._eventListeners = {};

        // Simulate async connection
        setTimeout(() => {
            this.readyState = MockEventSource.OPEN;
            if (this.onopen) this.onopen({ type: 'open' });
        }, 10);
    }

    addEventListener(eventName, handler) {
        if (!this._eventListeners[eventName]) {
            this._eventListeners[eventName] = [];
        }
        this._eventListeners[eventName].push(handler);
    }

    removeEventListener(eventName, handler) {
        if (this._eventListeners[eventName]) {
            this._eventListeners[eventName] = this._eventListeners[eventName].filter(h => h !== handler);
        }
    }

    close() {
        this.readyState = MockEventSource.CLOSED;
    }

    // Test helper: simulate receiving a message
    _receiveMessage(data) {
        if (this.onmessage) {
            this.onmessage({ data: typeof data === 'string' ? data : JSON.stringify(data) });
        }
    }

    // Test helper: simulate custom event
    _triggerEvent(eventName, data) {
        const eventData = typeof data === 'string' ? data : JSON.stringify(data);
        if (this._eventListeners[eventName]) {
            this._eventListeners[eventName].forEach(handler => {
                handler({ data: eventData, type: eventName });
            });
        }
    }

    // Test helper: simulate error
    _triggerError(error) {
        if (this.onerror) this.onerror(error);
    }
}

global.EventSource = MockEventSource;

describe('SseManager', () => {
    beforeEach(() => {
        // Reset SseManager state
        SseManager.connections = {};
        jest.clearAllMocks();
    });

    describe('connect', () => {
        it('should create an EventSource connection', () => {
            const connection = SseManager.connect('/sse/stream');

            expect(connection).toBeInstanceOf(MockEventSource);
            expect(SseManager.connections['/sse/stream']).toBe(connection);
        });

        it('should build full URL from relative path', () => {
            const connection = SseManager.connect('/sse/events');

            expect(connection.url).toContain('/sse/events');
        });

        it('should emit sse:open event when connected', (done) => {
            SseManager.connect('/sse/test', {
                onOpen: () => {
                    expect(TB.events.emit).toHaveBeenCalledWith('sse:open:/sse/test', expect.anything());
                    done();
                }
            });
        });

        it('should call onOpen callback when connected', (done) => {
            const onOpen = jest.fn(() => done());
            SseManager.connect('/sse/test', { onOpen });
        });

        it('should return existing connection if already connected', () => {
            const first = SseManager.connect('/sse/stream');
            const second = SseManager.connect('/sse/stream');

            expect(second).toBe(first);
            expect(TB.logger.warn).toHaveBeenCalledWith(expect.stringContaining('Already connected'));
        });

        it('should pass eventSourceOptions to EventSource', () => {
            const connection = SseManager.connect('/sse/auth', {
                eventSourceOptions: { withCredentials: true }
            });

            expect(connection.withCredentials).toBe(true);
        });
    });

    describe('message handling', () => {
        let connection;

        beforeEach(async () => {
            connection = SseManager.connect('/sse/messages');
            await new Promise(r => setTimeout(r, 20));
        });

        it('should emit sse:message event on incoming message', () => {
            connection._receiveMessage('test data');

            expect(TB.events.emit).toHaveBeenCalledWith('sse:message:/sse/messages', expect.objectContaining({
                data: 'test data'
            }));
        });

        it('should call onMessage callback', () => {
            const onMessage = jest.fn();
            SseManager.disconnect('/sse/messages');
            connection = SseManager.connect('/sse/messages', { onMessage });

            setTimeout(() => {
                connection._receiveMessage('callback test');
                expect(onMessage).toHaveBeenCalledWith('callback test', expect.anything());
            }, 20);
        });
    });

    describe('custom event listeners', () => {
        let connection;

        beforeEach(async () => {
            const customHandler = jest.fn();
            connection = SseManager.connect('/sse/custom', {
                listeners: {
                    'notification': customHandler,
                    'update': jest.fn()
                }
            });
            await new Promise(r => setTimeout(r, 20));
        });

        it('should register custom event listeners', () => {
            expect(connection._eventListeners['notification']).toBeDefined();
            expect(connection._eventListeners['update']).toBeDefined();
        });

        it('should emit specific event for custom events', () => {
            connection._triggerEvent('notification', { title: 'Test' });

            expect(TB.events.emit).toHaveBeenCalledWith(
                'sse:event:/sse/custom:notification',
                expect.objectContaining({
                    data: { title: 'Test' }
                })
            );
        });

        it('should parse JSON data in custom events', () => {
            connection._triggerEvent('update', '{"status": "ok"}');

            expect(TB.events.emit).toHaveBeenCalledWith(
                'sse:event:/sse/custom:update',
                expect.objectContaining({
                    data: { status: 'ok' }
                })
            );
        });

        it('should handle malformed JSON data gracefully', () => {
            // Trigger with data that starts with { but is not valid JSON
            connection._triggerEvent('notification', '{invalid json');

            expect(TB.logger.warn).toHaveBeenCalledWith(
                expect.stringContaining('Could not parse JSON'),
                expect.anything()
            );
        });

        it('should pass through plain text without parsing', () => {
            // Plain text that doesn't start with { or [ should not trigger JSON parsing
            connection._triggerEvent('notification', 'plain text');

            // Should emit event with the plain text data
            expect(TB.events.emit).toHaveBeenCalledWith(
                'sse:event:/sse/custom:notification',
                expect.objectContaining({
                    data: 'plain text'
                })
            );
        });
    });

    describe('error handling', () => {
        it('should emit sse:error event on error', async () => {
            const onError = jest.fn();
            const connection = SseManager.connect('/sse/error', { onError });
            await new Promise(r => setTimeout(r, 20));

            connection._triggerError(new Error('Connection failed'));

            expect(TB.events.emit).toHaveBeenCalledWith('sse:error:/sse/error', expect.anything());
            expect(onError).toHaveBeenCalled();
        });

        it('should close connection and remove from connections on error', async () => {
            const connection = SseManager.connect('/sse/error');
            await new Promise(r => setTimeout(r, 20));

            connection._triggerError(new Error('Test error'));

            expect(connection.readyState).toBe(MockEventSource.CLOSED);
            expect(SseManager.connections['/sse/error']).toBeUndefined();
        });
    });

    describe('disconnect', () => {
        beforeEach(async () => {
            SseManager.connect('/sse/disconnect-test');
            await new Promise(r => setTimeout(r, 20));
        });

        it('should close the connection', () => {
            const connection = SseManager.connections['/sse/disconnect-test'];
            SseManager.disconnect('/sse/disconnect-test');

            expect(connection.readyState).toBe(MockEventSource.CLOSED);
        });

        it('should remove connection from connections object', () => {
            SseManager.disconnect('/sse/disconnect-test');

            expect(SseManager.connections['/sse/disconnect-test']).toBeUndefined();
        });

        it('should emit sse:close event', () => {
            SseManager.disconnect('/sse/disconnect-test');

            expect(TB.events.emit).toHaveBeenCalledWith('sse:close:/sse/disconnect-test');
        });

        it('should handle disconnecting non-existent connection gracefully', () => {
            expect(() => SseManager.disconnect('/sse/non-existent')).not.toThrow();
            expect(TB.logger.warn).toHaveBeenCalledWith(expect.stringContaining('No active connection'));
        });
    });

    describe('disconnectAll', () => {
        beforeEach(async () => {
            SseManager.connect('/sse/stream1');
            SseManager.connect('/sse/stream2');
            SseManager.connect('/sse/stream3');
            await new Promise(r => setTimeout(r, 20));
        });

        it('should close all connections', () => {
            expect(Object.keys(SseManager.connections)).toHaveLength(3);

            SseManager.disconnectAll();

            expect(Object.keys(SseManager.connections)).toHaveLength(0);
        });

        it('should emit sse:close for each connection', () => {
            SseManager.disconnectAll();

            expect(TB.events.emit).toHaveBeenCalledWith('sse:close:/sse/stream1');
            expect(TB.events.emit).toHaveBeenCalledWith('sse:close:/sse/stream2');
            expect(TB.events.emit).toHaveBeenCalledWith('sse:close:/sse/stream3');
        });
    });

    describe('getConnection', () => {
        it('should return undefined for non-existent connection', () => {
            expect(SseManager.getConnection('/sse/non-existent')).toBeUndefined();
        });

        it('should return EventSource instance for existing connection', async () => {
            SseManager.connect('/sse/get-test');
            await new Promise(r => setTimeout(r, 20));

            expect(SseManager.getConnection('/sse/get-test')).toBeInstanceOf(MockEventSource);
        });
    });
});

