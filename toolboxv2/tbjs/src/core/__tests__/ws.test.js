// tbjs/src/core/__tests__/ws.test.js
// Tests für das WebSocket-Client Modul

// Mock TB before importing WsManager
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
}));

import WsManager from '../ws.js';
import TB from '../index.js';

// Mock WebSocket
class MockWebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    constructor(url) {
        this.url = url;
        this.readyState = MockWebSocket.CONNECTING;
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
        this.onclose = null;
        this._sentMessages = [];

        // Simulate async connection
        setTimeout(() => {
            this.readyState = MockWebSocket.OPEN;
            if (this.onopen) this.onopen({ type: 'open' });
        }, 10);
    }

    send(data) {
        if (this.readyState !== MockWebSocket.OPEN) {
            throw new Error('WebSocket is not open');
        }
        this._sentMessages.push(data);
    }

    close(code, reason) {
        this.readyState = MockWebSocket.CLOSED;
        if (this.onclose) {
            this.onclose({ code, reason, wasClean: true });
        }
    }

    // Test helper: simulate receiving a message
    _receiveMessage(data) {
        if (this.onmessage) {
            this.onmessage({ data: typeof data === 'string' ? data : JSON.stringify(data) });
        }
    }

    // Test helper: simulate error
    _triggerError(error) {
        if (this.onerror) this.onerror(error);
    }
}

global.WebSocket = MockWebSocket;

describe('WsManager', () => {
    beforeEach(() => {
        // Reset WsManager state
        WsManager.connection = null;
        WsManager.url = null;
        WsManager.currentReconnects = 0;
        WsManager.reconnectAttempts = 5;
        WsManager.connectionContext = null;
        jest.clearAllMocks();
    });

    describe('connect', () => {
        it('should create a WebSocket connection', async () => {
            WsManager.connect('ws://localhost:5001');

            expect(WsManager.connection).toBeInstanceOf(MockWebSocket);
            expect(WsManager.url).toBe('ws://localhost:5001');
        });

        it('should build full URL from relative path', () => {
            WsManager.connect('/ws/test');

            expect(WsManager.connection.url).toContain('/ws/test');
        });

        it('should append context parameters to URL', () => {
            WsManager.connect('ws://localhost:5001', {
                context: { room_id: '123', user_token: 'abc' }
            });

            expect(WsManager.connection.url).toContain('room_id=123');
            expect(WsManager.connection.url).toContain('user_token=abc');
        });

        it('should emit ws:open event when connected', (done) => {
            WsManager.connect('ws://localhost:5001', {
                onOpen: () => {
                    expect(TB.events.emit).toHaveBeenCalledWith('ws:open', expect.objectContaining({
                        context: {}
                    }));
                    done();
                }
            });
        });

        it('should call onOpen callback when connected', (done) => {
            const onOpen = jest.fn(() => done());
            WsManager.connect('ws://localhost:5001', { onOpen });
        });

        it('should not create duplicate connection to same URL', async () => {
            WsManager.connect('ws://localhost:5001');
            await new Promise(r => setTimeout(r, 20)); // Wait for connection

            const firstConnection = WsManager.connection;
            WsManager.connect('ws://localhost:5001');

            expect(WsManager.connection).toBe(firstConnection);
            expect(TB.logger.warn).toHaveBeenCalledWith(expect.stringContaining('Already connected'));
        });
    });

    describe('send', () => {
        beforeEach(async () => {
            WsManager.connect('ws://localhost:5001');
            await new Promise(r => setTimeout(r, 20)); // Wait for connection
        });

        it('should send JSON stringified message', () => {
            const result = WsManager.send({ type: 'ping' });

            expect(result).toBe(true);
            expect(WsManager.connection._sentMessages).toContain('{"type":"ping"}');
        });

        it('should include context when includeContext is true', () => {
            WsManager.connectionContext = { room_id: '123' };
            WsManager.send({ type: 'message' }, true);

            const sent = JSON.parse(WsManager.connection._sentMessages[0]);
            expect(sent._context).toEqual({ room_id: '123' });
        });

        it('should return false when connection is not open', () => {
            WsManager.disconnect();
            const result = WsManager.send({ type: 'test' });

            expect(result).toBe(false);
            expect(TB.logger.error).toHaveBeenCalled();
        });
    });

    describe('message handling', () => {
        beforeEach(async () => {
            WsManager.connect('ws://localhost:5001');
            await new Promise(r => setTimeout(r, 20));
        });

        it('should emit ws:message event on incoming message', () => {
            WsManager.connection._receiveMessage({ type: 'pong' });

            expect(TB.events.emit).toHaveBeenCalledWith('ws:message', expect.objectContaining({
                data: { type: 'pong' }
            }));
        });

        it('should emit specific event for messages with event field', () => {
            WsManager.connection._receiveMessage({ event: 'notification', data: { title: 'Test' } });

            expect(TB.events.emit).toHaveBeenCalledWith('ws:event:notification', expect.objectContaining({
                data: { title: 'Test' }
            }));
        });

        it('should handle non-JSON messages gracefully', () => {
            WsManager.connection._receiveMessage('plain text message');

            expect(TB.logger.warn).toHaveBeenCalledWith(
                expect.stringContaining('Could not parse'),
                expect.anything()
            );
        });

        it('should call onMessage callback', async () => {
            const onMessage = jest.fn();
            WsManager.disconnect();
            WsManager.connect('ws://localhost:5001', { onMessage });

            // Wait for connection
            await new Promise(r => setTimeout(r, 20));

            // Simulate message
            WsManager.connection._receiveMessage({ test: 'data' });
            expect(onMessage).toHaveBeenCalled();
        });
    });

    describe('disconnect', () => {
        beforeEach(async () => {
            WsManager.connect('ws://localhost:5001', { context: { room: 'test' } });
            await new Promise(r => setTimeout(r, 20));
        });

        it('should close the connection', () => {
            WsManager.disconnect();

            expect(WsManager.connection).toBeNull();
            expect(WsManager.url).toBeNull();
        });

        it('should clear connection context', () => {
            expect(WsManager.connectionContext).toEqual({ room: 'test' });
            WsManager.disconnect();
            expect(WsManager.connectionContext).toBeNull();
        });

        it('should emit ws:close event', () => {
            WsManager.disconnect();
            expect(TB.events.emit).toHaveBeenCalledWith('ws:close', expect.anything());
        });
    });

    describe('context management', () => {
        beforeEach(async () => {
            WsManager.connect('ws://localhost:5001', { context: { initial: 'value' } });
            await new Promise(r => setTimeout(r, 20));
        });

        it('getContext should return current context', () => {
            expect(WsManager.getContext()).toEqual({ initial: 'value' });
        });

        it('updateContext should merge new context', () => {
            WsManager.updateContext({ added: 'new' });

            expect(WsManager.getContext()).toEqual({
                initial: 'value',
                added: 'new'
            });
        });

        it('updateContext should override existing keys', () => {
            WsManager.updateContext({ initial: 'updated' });

            expect(WsManager.getContext().initial).toBe('updated');
        });
    });

    describe('getConnection', () => {
        it('should return null when not connected', () => {
            expect(WsManager.getConnection()).toBeNull();
        });

        it('should return WebSocket instance when connected', async () => {
            WsManager.connect('ws://localhost:5001');
            await new Promise(r => setTimeout(r, 20));

            expect(WsManager.getConnection()).toBeInstanceOf(MockWebSocket);
        });
    });

    describe('_buildContextParams', () => {
        it('should build URL params from context object', () => {
            const params = WsManager._buildContextParams({ a: '1', b: '2' });
            expect(params).toContain('a=1');
            expect(params).toContain('b=2');
        });

        it('should skip null and undefined values', () => {
            const params = WsManager._buildContextParams({ valid: 'yes', empty: null, undef: undefined });
            expect(params).toContain('valid=yes');
            expect(params).not.toContain('empty');
            expect(params).not.toContain('undef');
        });
    });

    describe('_buildFullUrl', () => {
        it('should return absolute WebSocket URL as-is', () => {
            const url = WsManager._buildFullUrl('ws://example.com/path', '');
            expect(url).toBe('ws://example.com/path');
        });

        it('should build full URL from relative path', () => {
            const url = WsManager._buildFullUrl('/ws/endpoint', '');
            expect(url).toContain('/ws/endpoint');
        });

        it('should append context params with correct separator', () => {
            const url1 = WsManager._buildFullUrl('ws://example.com/path', 'key=value');
            expect(url1).toBe('ws://example.com/path?key=value');

            const url2 = WsManager._buildFullUrl('ws://example.com/path?existing=1', 'key=value');
            expect(url2).toBe('ws://example.com/path?existing=1&key=value');
        });
    });
});

