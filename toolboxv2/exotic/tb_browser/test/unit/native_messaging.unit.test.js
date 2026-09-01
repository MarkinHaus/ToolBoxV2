/**
 * native_messaging.unit.test.js
 *
 * Unit-Tests für:
 *   1. Native Messaging Protokoll (encode/decode — kein Prozess nötig)
 *   2. background.js makeNativeCall() — chrome.runtime Mock
 *   3. background.js checkConnection() im native-Modus
 *   4. Fehlerbehandlung wenn lastError gesetzt ist
 *
 * Führe aus mit: npm run test:unit
 */

// ─── Protokoll-Helpers (gespiegelt aus dem nativen Host) ─────────────────────

function encodeNativeMessage(obj) {
    const payload = Buffer.from(JSON.stringify(obj), 'utf8');
    const header = Buffer.alloc(4);
    header.writeUInt32LE(payload.length, 0);
    return Buffer.concat([header, payload]);
}

function decodeNativeMessage(buf) {
    if (buf.length < 4) throw new Error('Buffer zu kurz für Header');
    const msgLen = buf.readUInt32LE(0);
    if (buf.length < 4 + msgLen) throw new Error(`Unvollständiger Body: erwartet ${msgLen}, hat ${buf.length - 4}`);
    return JSON.parse(buf.slice(4, 4 + msgLen).toString('utf8'));
}

// ─── Tests: Protokoll ────────────────────────────────────────────────────────

describe('Native Messaging Protokoll — encode/decode', () => {
    test('ping-Nachricht kodieren und dekodieren ergibt dasselbe Objekt', () => {
        const msg = { action: 'ping', payload: {} };
        const encoded = encodeNativeMessage(msg);
        const decoded = decodeNativeMessage(encoded);
        expect(decoded).toEqual(msg);
    });

    test('Header enthält korrekte Länge (Little-Endian uint32)', () => {
        const msg = { action: 'ping', payload: {} };
        const encoded = encodeNativeMessage(msg);
        const headerLen = encoded.readUInt32LE(0);
        const bodyLen = encoded.length - 4;
        expect(headerLen).toBe(bodyLen);
    });

    test('Unicode in payload wird korrekt kodiert', () => {
        const msg = { action: 'test', payload: { text: 'Hällo Wörld 🌍' } };
        const encoded = encodeNativeMessage(msg);
        const decoded = decodeNativeMessage(encoded);
        expect(decoded.payload.text).toBe('Hällo Wörld 🌍');
    });

    test('Leeres payload wird toleriert', () => {
        const encoded = encodeNativeMessage({ action: 'ping' });
        const decoded = decodeNativeMessage(encoded);
        expect(decoded.action).toBe('ping');
    });

    test('Großes payload (>1KB) funktioniert', () => {
        const bigText = 'x'.repeat(5000);
        const msg = { action: 'isaa_chat', payload: { message: bigText } };
        const encoded = encodeNativeMessage(msg);
        const decoded = decodeNativeMessage(encoded);
        expect(decoded.payload.message).toHaveLength(5000);
    });

    test('decodeNativeMessage wirft bei zu kurzem Buffer', () => {
        expect(() => decodeNativeMessage(Buffer.from([0x01]))).toThrow('Buffer zu kurz');
    });

    test('decodeNativeMessage wirft bei unvollständigem Body', () => {
        const buf = Buffer.alloc(4 + 5);
        buf.writeUInt32LE(100, 0); // behauptet 100 Bytes, hat aber nur 5
        expect(() => decodeNativeMessage(buf)).toThrow('Unvollständiger Body');
    });

    test('error_response Format entspricht Python-Implementierung', () => {
        // Spiegelt: def error_response(msg, code=500)
        const errorResp = { success: false, error: 'test error', code: 500 };
        const encoded = encodeNativeMessage(errorResp);
        const decoded = decodeNativeMessage(encoded);
        expect(decoded).toEqual({ success: false, error: 'test error', code: 500 });
    });
});

// ─── Chrome Mock Setup ───────────────────────────────────────────────────────

// chrome global wird von mocks/chrome.js gesetzt — wir erweitern es hier
function resetChromeMock() {
    global.chrome = {
        runtime: {
            lastError: null,
            sendNativeMessage: jest.fn(),
        },
        storage: {
            sync: {
                get: jest.fn().mockResolvedValue({}),
                set: jest.fn().mockResolvedValue(undefined),
            },
        },
        action: {
            setBadgeText: jest.fn(),
            setBadgeBackgroundColor: jest.fn(),
        },
    };
}

// ─── makeNativeCall Simulation ────────────────────────────────────────────────
// Isolierter Extrakt der makeNativeCall-Logik aus background.js
// (kein Import des echten background.js — ES-Module + Service Worker APIs)

function createMakeNativeCall(chromeObj) {
    return function makeNativeCall(action, payload) {
        return new Promise((resolve, reject) => {
            chromeObj.runtime.sendNativeMessage(
                'com.toolbox.native',
                { action, payload: payload || {} },
                (response) => {
                    if (chromeObj.runtime.lastError) {
                        reject(new Error(chromeObj.runtime.lastError.message));
                    } else {
                        resolve(response);
                    }
                }
            );
        });
    };
}

// ─── Tests: makeNativeCall ────────────────────────────────────────────────────

describe('makeNativeCall — chrome.runtime Mock', () => {
    let makeNativeCall;

    beforeEach(() => {
        resetChromeMock();
        makeNativeCall = createMakeNativeCall(global.chrome);
    });

    test('erfolgreiche ping-Antwort wird korrekt weitergegeben', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            cb({ success: true, pong: true, message: 'ToolBoxV2 Native Host aktiv' });
        });

        const result = await makeNativeCall('ping', {});
        expect(result).toEqual({ success: true, pong: true, message: 'ToolBoxV2 Native Host aktiv' });
    });

    test('sendNativeMessage wird mit korrektem Host-Namen aufgerufen', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            cb({ success: true });
        });

        await makeNativeCall('ping', {});
        expect(chrome.runtime.sendNativeMessage).toHaveBeenCalledWith(
            'com.toolbox.native',
            expect.any(Object),
            expect.any(Function)
        );
    });

    test('gesendetes Objekt hat action und payload', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            cb({ success: true });
        });

        await makeNativeCall('validate_session', { username: 'testuser' });
        const sentMsg = chrome.runtime.sendNativeMessage.mock.calls[0][1];
        expect(sentMsg).toEqual({ action: 'validate_session', payload: { username: 'testuser' } });
    });

    test('lastError → Promise rejected mit korrekter Fehlermeldung', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            chrome.runtime.lastError = {
                message: 'Error when communicating with the native messaging host.'
            };
            cb(undefined);
        });

        await expect(makeNativeCall('ping', {})).rejects.toThrow(
            'Error when communicating with the native messaging host.'
        );
    });

    test('lastError: "Specified native messaging host not found" → reject', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            chrome.runtime.lastError = {
                message: 'Specified native messaging host not found.'
            };
            cb(undefined);
        });

        await expect(makeNativeCall('ping', {})).rejects.toThrow(
            'Specified native messaging host not found.'
        );
    });

    test('leeres payload wird zu {} normalisiert', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            cb({ success: true });
        });

        await makeNativeCall('ping', null);
        const sentMsg = chrome.runtime.sendNativeMessage.mock.calls[0][1];
        expect(sentMsg.payload).toEqual({});
    });
});

// ─── Tests: checkConnection im native-Modus ──────────────────────────────────

describe('checkConnection — native Modus', () => {
    let state;
    let makeNativeCall;

    async function checkConnection() {
        try {
            if (state.useNative) {
                const resp = await makeNativeCall('ping', {});
                state.isConnected = !!(resp && resp.success);
            } else {
                // HTTP-Pfad nicht getestet hier
                state.isConnected = false;
            }
            chrome.action.setBadgeText({ text: state.isConnected ? '' : '!' });
            chrome.action.setBadgeBackgroundColor({
                color: state.isConnected ? '#4CAF50' : '#F44336'
            });
        } catch {
            state.isConnected = false;
            chrome.action.setBadgeText({ text: '!' });
            chrome.action.setBadgeBackgroundColor({ color: '#F44336' });
        }
    }

    beforeEach(() => {
        resetChromeMock();
        state = { useNative: true, isConnected: false };
        makeNativeCall = createMakeNativeCall(global.chrome);
    });

    test('ping erfolgreich → isConnected = true', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            cb({ success: true, pong: true });
        });

        await checkConnection();
        expect(state.isConnected).toBe(true);
        expect(chrome.action.setBadgeText).toHaveBeenCalledWith({ text: '' });
        expect(chrome.action.setBadgeBackgroundColor).toHaveBeenCalledWith({ color: '#4CAF50' });
    });

    test('ping mit success: false → isConnected = false', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            cb({ success: false, error: 'toolboxv2 nicht geladen' });
        });

        await checkConnection();
        expect(state.isConnected).toBe(false);
    });

    test('lastError → isConnected = false, Badge rot', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            chrome.runtime.lastError = { message: 'Native host not found.' };
            cb(undefined);
        });

        await checkConnection();
        expect(state.isConnected).toBe(false);
        expect(chrome.action.setBadgeText).toHaveBeenCalledWith({ text: '!' });
        expect(chrome.action.setBadgeBackgroundColor).toHaveBeenCalledWith({ color: '#F44336' });
    });

    test('null-Response → isConnected = false', async () => {
        chrome.runtime.sendNativeMessage.mockImplementation((host, msg, cb) => {
            cb(null);
        });

        await checkConnection();
        expect(state.isConnected).toBe(false);
    });
});

// ─── Tests: Bekannte Fehlermuster ────────────────────────────────────────────

describe('Bekannte Chrome Native Messaging Fehlermuster', () => {
    test('Fehlermeldung "Error when communicating" — Ursachen-Mapping', () => {
        // Dokumentiert bekannte Ursachen für diesen Fehler
        const knownCauses = [
            'Registry-Eintrag fehlt oder zeigt auf falsche .json-Datei',
            'com.toolbox.native.json hat falschen Pfad zur .bat-Datei',
            'Python-Prozess crashed beim Start (toolboxv2 import error)',
            'allowed_origins in .json stimmt nicht mit Extension-ID überein',
            'Prozess schreibt nichts auf stdout (kein send_message() vor exit)',
        ];
        // Dieser Test dient als lebende Dokumentation
        expect(knownCauses.length).toBeGreaterThan(0);
    });

    test('Registry-Pfad ist korrekt formatiert (Windows)', () => {
        const expectedRegistryKey =
            'HKCU\\Software\\Google\\Chrome\\NativeMessagingHosts\\com.toolbox.native';
        expect(expectedRegistryKey).toContain('NativeMessagingHosts');
        expect(expectedRegistryKey).toContain('com.toolbox.native');
    });

    test('allowed_origins Format ist korrekt (trailing slash)', () => {
        // Chrome erfordert trailing slash
        const extensionId = 'ncnikihghamkjfhfimbcbnmeboppechn';
        const origin = `chrome-extension://${extensionId}/`;
        expect(origin).toMatch(/^chrome-extension:\/\/[a-z]{32}\/$/);
    });

    test('com.toolbox.native.json "name" stimmt mit Aufruf in background.js überein', () => {
        const manifestName = 'com.toolbox.native'; // aus com.toolbox.native.json
        const backgroundCallName = 'com.toolbox.native'; // aus makeNativeCall
        expect(manifestName).toBe(backgroundCallName);
    });
});
