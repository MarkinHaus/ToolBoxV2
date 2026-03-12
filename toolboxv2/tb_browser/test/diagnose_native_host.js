#!/usr/bin/env node
/**
 * diagnose_native_host.js
 *
 * Standalone-Diagnose-Script — kein Jest nötig.
 * Startet den Native Host direkt und prüft jeden Schritt.
 *
 * Ausführen: node diagnose_native_host.js
 *
 * Hilft bei: "Error when communicating with the native messaging host"
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const PYTHON_EXE    = 'C:\\Users\\Markin\\Workspace\\ToolBoxV2\\.venv\\Scripts\\python.exe';
const HOST_SCRIPT   = 'C:\\Users\\Markin\\Workspace\\ToolBoxV2\\toolboxv2\\tb_browser\\native\\toolbox_native_host.py';
const MANIFEST_JSON = 'C:\\Users\\Markin\\Workspace\\ToolBoxV2\\toolboxv2\\tb_browser\\native\\com.toolbox.native.json';
const EXTENSION_ID  = 'ncnikihghamkjfhfimbcbnmeboppechn';

const OK   = '✅';
const FAIL = '❌';
const WARN = '⚠️ ';

let passed = 0;
let failed = 0;

function check(label, condition, detail = '') {
    if (condition) {
        console.log(`${OK}  ${label}`);
        passed++;
    } else {
        console.log(`${FAIL} ${label}${detail ? '\n    → ' + detail : ''}`);
        failed++;
    }
}

function encodeMsg(obj) {
    const payload = Buffer.from(JSON.stringify(obj), 'utf8');
    const header  = Buffer.alloc(4);
    header.writeUInt32LE(payload.length, 0);
    return Buffer.concat([header, payload]);
}

async function main() {
    console.log('═══════════════════════════════════════════════════');
    console.log(' ToolBoxV2 Native Host Diagnose');
    console.log('═══════════════════════════════════════════════════\n');

    // ── SCHRITT 1: Dateien prüfen ────────────────────────────────────────────
    console.log('── Schritt 1: Dateien ──────────────────────────────');

    check('Python-Executable existiert',
        fs.existsSync(PYTHON_EXE),
        `Nicht gefunden: ${PYTHON_EXE}`
    );

    check('toolbox_native_host.py existiert',
        fs.existsSync(HOST_SCRIPT),
        `Nicht gefunden: ${HOST_SCRIPT}`
    );

    check('com.toolbox.native.json existiert',
        fs.existsSync(MANIFEST_JSON),
        `Nicht gefunden: ${MANIFEST_JSON}`
    );

    // ── SCHRITT 2: Manifest-Inhalt prüfen ────────────────────────────────────
    console.log('\n── Schritt 2: Manifest-Inhalt ──────────────────────');

    if (fs.existsSync(MANIFEST_JSON)) {
        let manifest;
        try {
            manifest = JSON.parse(fs.readFileSync(MANIFEST_JSON, 'utf8'));
            check('Manifest ist valides JSON', true);
        } catch (e) {
            check('Manifest ist valides JSON', false, e.message);
            manifest = null;
        }

        if (manifest) {
            check('name = "com.toolbox.native"',
                manifest.name === 'com.toolbox.native',
                `Ist: ${manifest.name}`
            );

            const batExists = fs.existsSync(manifest.path);
            check(`path zeigt auf existierende Datei`,
                batExists,
                `Pfad: ${manifest.path}`
            );

            const expectedOrigin = `chrome-extension://${EXTENSION_ID}/`;
            const hasOrigin = (manifest.allowed_origins || []).includes(expectedOrigin);
            check(`allowed_origins enthält ${expectedOrigin}`,
                hasOrigin,
                `Ist: ${JSON.stringify(manifest.allowed_origins)}`
            );
        }
    }

    // ── SCHRITT 3: Python-Import-Test ─────────────────────────────────────────
    console.log('\n── Schritt 3: Python Import-Test ───────────────────');

    try {
        const result = execSync(
            `"${PYTHON_EXE}" -c "import toolboxv2; print('ok')"`,
            { timeout: 10000, encoding: 'utf8' }
        );
        check('toolboxv2 importierbar', result.trim() === 'ok');
    } catch (e) {
        check('toolboxv2 importierbar', false,
            `Import-Fehler: ${e.stderr || e.message}`
        );
        console.log(`    ${WARN} Native Host wird lazy initialisieren und trotzdem funktionieren`);
    }

    // ── SCHRITT 4: Prozess-Kommunikation ──────────────────────────────────────
    console.log('\n── Schritt 4: Prozess-Kommunikation ────────────────');

    await new Promise((resolve) => {
        const proc = spawn(PYTHON_EXE, [HOST_SCRIPT], {
            stdio: ['pipe', 'pipe', 'pipe'],
            windowsHide: true,
        });

        const stderrBuf = [];
        proc.stderr.on('data', c => stderrBuf.push(c.toString()));

        // Timeout: Prozess started nicht
        const startTimeout = setTimeout(() => {
            check('Prozess startet', false, 'Timeout nach 10s');
            proc.kill();
            resolve();
        }, 10000);

        proc.on('error', (err) => {
            clearTimeout(startTimeout);
            check('Prozess startet', false, err.message);
            resolve();
        });

        proc.on('exit', (code) => {
            if (code !== null && code !== 0) {
                clearTimeout(startTimeout);
                check('Prozess startet', false,
                    `Sofort beendet mit Code ${code}\n    Stderr: ${stderrBuf.join('').slice(0, 300)}`
                );
                resolve();
            }
        });

        // Warte kurz dann ping senden
        setTimeout(() => {
            clearTimeout(startTimeout);
            check('Prozess startet', !proc.killed && proc.exitCode === null);

            // Ping senden
            let responseBuf = Buffer.alloc(0);
            const pingTimeout = setTimeout(() => {
                check('ping → Antwort erhalten', false, 'Keine Antwort nach 8s');
                proc.kill();
                resolve();
            }, 8000);

            proc.stdout.on('data', (chunk) => {
                responseBuf = Buffer.concat([responseBuf, chunk]);
                if (responseBuf.length >= 4) {
                    const msgLen = responseBuf.readUInt32LE(0);
                    if (responseBuf.length >= 4 + msgLen) {
                        clearTimeout(pingTimeout);
                        const json = responseBuf.slice(4, 4 + msgLen).toString('utf8');
                        let resp;
                        try {
                            resp = JSON.parse(json);
                            check('ping → gültiges JSON empfangen', true);
                        } catch (e) {
                            check('ping → gültiges JSON empfangen', false, `Parse-Fehler: ${e.message}`);
                            proc.kill();
                            resolve();
                            return;
                        }
                        check('ping → success: true', resp.success === true,
                            `Antwort: ${JSON.stringify(resp)}`
                        );
                        check('ping → pong: true', resp.pong === true);

                        proc.stdin.end();
                        proc.kill();
                        resolve();
                    }
                }
            });

            proc.stdin.write(encodeMsg({ action: 'ping', payload: {} }));

        }, 2000); // 2s Startup-Zeit für toolboxv2
    });

    // ── Zusammenfassung ────────────────────────────────────────────────────────
    console.log('\n═══════════════════════════════════════════════════');
    console.log(` Ergebnis: ${passed} bestanden, ${failed} fehlgeschlagen`);
    console.log('═══════════════════════════════════════════════════');

    if (failed > 0) {
        console.log('\n Mögliche Fixes:');
        console.log('  1. python toolbox_native_host.py --register ' + EXTENSION_ID);
        console.log('  2. Prüfe Registry: HKCU\\Software\\Google\\Chrome\\NativeMessagingHosts\\com.toolbox.native');
        console.log('  3. Stelle sicher dass toolboxv2 im venv installiert ist');
        console.log('  4. Führe: python toolbox_native_host.py --test');
    }

    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error('Unerwarteter Fehler:', e);
    process.exit(1);
});
