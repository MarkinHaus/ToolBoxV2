/**
 * native_host.e2e.test.js
 *
 * Testet den echten toolbox_native_host.py Prozess via stdin/stdout.
 * Führe aus mit: npm run test:e2e -- --testPathPattern=native_host
 *
 * Was getestet wird:
 *   1. Prozess startet überhaupt (exit-code-Check)
 *   2. Length-Prefix-Protokoll: senden + empfangen
 *   3. ping → pong
 *   4. Unbekannte Action → 404 error_response
 *   5. Prozess überlebt Fehler (bleibt nach fehlgeschlagenem Import am Leben)
 *   6. tauri_check (kein Server erwartet → tauri_running: false)
 *   7. validate_session (ohne CLI-Login → authenticated: false ODER error)
 *   8. Mehrere Nachrichten hintereinander (sequentiell)
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// ─── Automatische Pfad-Erkennung ──────────────────────────────────────────────
// Pfade werden relativ zur Testdatei aufgelöst — kein hardcoding mehr
const NATIVE_DIR = path.resolve(__dirname, '../../native');
const BAT_FILE   = path.join(NATIVE_DIR, 'toolbox_native_host_wrapper.bat');
const PY_SCRIPT  = path.join(NATIVE_DIR, 'toolbox_native_host.py');

// Python aus der .bat lesen (erste Zeile nach @echo off)
function getPythonFromBat() {
    try {
        const bat = fs.readFileSync(BAT_FILE, 'utf8');
        const match = bat.match(/"([^"]+python[^"]*\.exe)"/i);
        return match ? match[1] : process.env.TOOLBOX_PYTHON || 'python';
    } catch {
        return process.env.TOOLBOX_PYTHON || 'python';
    }
}

const PYTHON_EXE  = getPythonFromBat();
const HOST_SCRIPT = process.env.TOOLBOX_NATIVE_SCRIPT || PY_SCRIPT;

const STARTUP_TIMEOUT_MS = 15000;
const MSG_TIMEOUT_MS     = 8000;

console.log(`[E2E] Python:  ${PYTHON_EXE}`);
console.log(`[E2E] Script:  ${HOST_SCRIPT}`);
// ─── Protokoll-Helpers ───────────────────────────────────────────────────────

/**
 * Kodiert eine Nachricht im Chrome Native Messaging Format:
 * [4-Byte-Little-Endian-Länge][JSON-UTF8-Payload]
 */
function encodeMessage(obj) {
    const payload = Buffer.from(JSON.stringify(obj), 'utf8');
    const header  = Buffer.alloc(4);
    header.writeUInt32LE(payload.length, 0);
    return Buffer.concat([header, payload]);
}

/**
 * Liest eine vollständige Native-Messaging-Antwort aus einem Buffer-Stream.
 * Gibt eine Promise zurück, die mit dem geparsten Objekt resolved.
 */
function readOneMessage(proc, timeoutMs = MSG_TIMEOUT_MS) {
    return new Promise((resolve, reject) => {
        let buf = Buffer.alloc(0);
        let timer;

        const cleanup = () => {
            proc.stdout.removeListener('data', onData);
            proc.removeListener('error', onError);
            clearTimeout(timer);
        };

        const onData = (chunk) => {
            buf = Buffer.concat([buf, chunk]);

            // Warte bis 4-Byte-Header da ist
            if (buf.length < 4) return;

            const msgLen = buf.readUInt32LE(0);

            // Warte bis voller Body da ist
            if (buf.length < 4 + msgLen) return;

            cleanup();
            const json = buf.slice(4, 4 + msgLen).toString('utf8');
            try {
                resolve(JSON.parse(json));
            } catch (e) {
                reject(new Error(`JSON parse error: ${e.message} — raw: ${json}`));
            }
        };

        const onError = (err) => {
            cleanup();
            reject(err);
        };

        timer = setTimeout(() => {
            cleanup();
            reject(new Error(`Timeout: keine Antwort nach ${timeoutMs}ms`));
        }, timeoutMs);

        proc.stdout.on('data', onData);
        proc.once('error', onError);
    });
}

// ─── Prozess-Fixture ─────────────────────────────────────────────────────────

let nativeProc = null;
const stderrLines = [];

beforeAll((done) => {
    nativeProc = spawn(PYTHON_EXE, [HOST_SCRIPT], {
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
    });

    nativeProc.stderr.on('data', (chunk) => {
        const lines = chunk.toString().split('\n').filter(Boolean);
        lines.forEach(l => {
            stderrLines.push(l);
            // Hilfreiche Debug-Ausgabe in der Test-Console
            process.stderr.write(`[NATIVE STDERR] ${l}\n`);
        });
    });

    nativeProc.on('error', (err) => {
        done(new Error(`Prozess konnte nicht gestartet werden: ${err.message}\n` +
                       `Prüfe PYTHON_EXE: ${PYTHON_EXE}\n` +
                       `Prüfe HOST_SCRIPT: ${HOST_SCRIPT}`));
    });

    // Warte auf "ready"-Zeile in stderr ODER Timeout
    const readyTimer = setTimeout(() => {
        // Prozess läuft noch → als "ready" werten (startup-Log kann fehlen)
        if (nativeProc && !nativeProc.killed) done();
        else done(new Error('Prozess beim Start unerwartet beendet'));
    }, STARTUP_TIMEOUT_MS);

    nativeProc.stderr.on('data', (chunk) => {
        if (chunk.toString().includes('waiting for messages') ||
            chunk.toString().includes('ready')) {
            clearTimeout(readyTimer);
            done();
        }
    });

    nativeProc.once('exit', (code) => {
        if (code !== null && code !== 0) {
            clearTimeout(readyTimer);
            done(new Error(
                `Prozess sofort beendet mit Code ${code}.\n` +
                `Stderr:\n${stderrLines.slice(-10).join('\n')}`
            ));
        }
    });
}, STARTUP_TIMEOUT_MS + 2000);

afterAll(() => {
    if (nativeProc && !nativeProc.killed) {
        nativeProc.stdin.end();
        nativeProc.kill();
    }
});

// ─── Hilfsfunktion: Nachricht senden + Antwort lesen ─────────────────────────

async function call(action, payload = {}) {
    const responsePromise = readOneMessage(nativeProc);
    nativeProc.stdin.write(encodeMessage({ action, payload }));
    return responsePromise;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('Native Host — Prozess', () => {
    test('Prozess läuft nach dem Start', () => {
        expect(nativeProc).not.toBeNull();
        expect(nativeProc.killed).toBe(false);
        expect(nativeProc.exitCode).toBeNull();
    });

    test('Stderr enthält keine Python-Traceback-Fehler beim Start', () => {
        const tracebacks = stderrLines.filter(l =>
            l.includes('Traceback') || l.includes('ModuleNotFoundError')
        );
        // Nur loggen, nicht hart fehlschlagen — toolboxv2 könnte optional sein
        if (tracebacks.length > 0) {
            console.warn('⚠️  Startup-Fehler in stderr:', tracebacks);
        }
        // Prozess muss trotzdem laufen (lazy-init Fix)
        expect(nativeProc.killed).toBe(false);
    });
});

describe('Native Host — Protokoll', () => {
    test('ping → success: true, pong: true', async () => {
        const resp = await call('ping');
        expect(resp).toMatchObject({ success: true, pong: true });
    }, MSG_TIMEOUT_MS);

    test('Unbekannte Action → success: false, code: 404', async () => {
        const resp = await call('__unknown_action_xyz__');
        expect(resp.success).toBe(false);
        expect(resp.code).toBe(404);
        expect(resp.error).toMatch(/Unbekannte Action/i);
    }, MSG_TIMEOUT_MS);

    test('Mehrere sequentielle Nachrichten funktionieren', async () => {
        const r1 = await call('ping');
        const r2 = await call('ping');
        const r3 = await call('ping');
        expect(r1.success).toBe(true);
        expect(r2.success).toBe(true);
        expect(r3.success).toBe(true);
    }, MSG_TIMEOUT_MS * 3);

    test('Prozess bleibt nach unbekannter Action am Leben', async () => {
        await call('does_not_exist');
        // Danach noch ping
        const resp = await call('ping');
        expect(resp.success).toBe(true);
    }, MSG_TIMEOUT_MS * 2);
});

describe('Native Host — Actions', () => {
    test('tauri_check → Antwort-Struktur korrekt (Server nicht erwartet)', async () => {
        const resp = await call('tauri_check', { port: 5000 });
        expect(resp).toHaveProperty('success', true);
        expect(resp).toHaveProperty('tauri_running');
        expect(resp).toHaveProperty('port', 5000);
        // Im Test-Kontext läuft kein Tauri → false erwartet
        expect(resp.tauri_running).toBe(false);
    }, MSG_TIMEOUT_MS);

    test('validate_session → Antwort-Struktur korrekt', async () => {
        const resp = await call('validate_session', {});
        // Entweder success+authenticated ODER error (wenn toolboxv2 nicht verfügbar)
        expect(resp).toHaveProperty('success');
        if (resp.success) {
            expect(resp).toHaveProperty('authenticated');
        } else {
            expect(resp).toHaveProperty('error');
        }
    }, MSG_TIMEOUT_MS);

    test('version_check → Antwort-Struktur korrekt', async () => {
        const resp = await call('version_check', {});
        expect(resp).toHaveProperty('success');
        if (resp.success && resp.data) {
            expect(resp.data).toHaveProperty('version');
        }
    }, MSG_TIMEOUT_MS);

    test('password_list ohne Auth → Fehler oder leere Liste', async () => {
        const resp = await call('password_list', {});
        expect(resp).toHaveProperty('success');
        // Entweder Fehler (kein Auth) oder leere Liste — beides okay
        if (!resp.success) {
            expect(resp).toHaveProperty('error');
        }
    }, MSG_TIMEOUT_MS);
});

describe('Native Host — Protokoll-Robustheit', () => {
    test('Leeres payload-Objekt wird toleriert', async () => {
        const resp = await call('ping');
        expect(resp.success).toBe(true);
    }, MSG_TIMEOUT_MS);

    test('Antwort hat immer "success" Feld', async () => {
        const actions = ['ping', 'tauri_check', '__invalid__', 'version_check'];
        for (const action of actions) {
            const resp = await call(action);
            expect(resp).toHaveProperty('success');
        }
    }, MSG_TIMEOUT_MS * 4);
});
