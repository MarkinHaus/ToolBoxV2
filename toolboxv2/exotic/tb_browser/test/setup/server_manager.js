'use strict';
/**
 * server_manager.js — Starts and monitors test backend servers
 *
 * Handles:
 * - Mock HTTP server (port 8080 = "local", port 5000 = "tauri")
 * - Tauri worker auto-start (subprocess via Python if available)
 * - Native host mock (stdin/stdout pipe)
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const http = require('http');

const MOCK_SERVER = path.join(__dirname, 'mock_server.js');

// ─── HTTP health check ────────────────────────────────────────────────────────
async function waitForPort(port, timeout = 10000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      await new Promise((resolve, reject) => {
        const req = http.get(`http://localhost:${port}/health`, (res) => {
          resolve(res.statusCode < 500);
        });
        req.on('error', reject);
        req.setTimeout(500, () => { req.destroy(); reject(new Error('timeout')); });
      });
      return true;
    } catch {
      await new Promise(r => setTimeout(r, 200));
    }
  }
  throw new Error(`Port ${port} not ready after ${timeout}ms`);
}

// ─── Start a mock server subprocess ──────────────────────────────────────────
function startMockServer(port) {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [MOCK_SERVER, String(port)], {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
    });

    proc.stdout.on('data', (d) => {
      if (process.env.VERBOSE) process.stdout.write(`[mock:${port}] ${d}`);
    });
    proc.stderr.on('data', (d) => {
      if (process.env.VERBOSE) process.stderr.write(`[mock:${port}] ${d}`);
    });
    proc.on('exit', (code) => {
      if (process.env.VERBOSE) console.log(`[mock:${port}] exited with code ${code}`);
    });

    waitForPort(port)
      .then(() => resolve(proc))
      .catch(reject);
  });
}

// ─── Try to start the real Tauri worker ──────────────────────────────────────
async function startTauriWorker(httpPort = 5000, wsPort = 5001) {
  // Look for tauri_integration.py relative to the workspace
  const candidates = [
    'C:\\Users\\Markin\\Workspace\\ToolBoxV2\\toolboxv2\\utils\\workers\\tauri_integration.py',
    '/srv/toolboxv2/toolboxv2/utils/workers/tauri_integration.py',
    path.join(process.env.HOME || '', 'Workspace/ToolBoxV2/toolboxv2/utils/workers/tauri_integration.py'),
  ];

  let scriptPath = null;
  for (const c of candidates) {
    try {
      require('fs').accessSync(c);
      scriptPath = c;
      break;
    } catch {}
  }

  if (!scriptPath) {
    console.warn('[TauriWorker] tauri_integration.py not found — falling back to mock server on', httpPort);
    return startMockServer(httpPort);
  }

  return new Promise((resolve, reject) => {
    const args = [scriptPath, `--http-port=${httpPort}`, `--ws-port=${wsPort}`];
    const proc = spawn('python', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
    });

    proc.stdout.on('data', d => {
      if (process.env.VERBOSE) process.stdout.write(`[tauri] ${d}`);
    });
    proc.stderr.on('data', d => {
      if (process.env.VERBOSE) process.stderr.write(`[tauri] ${d}`);
    });

    waitForPort(httpPort, 20000)
      .then(() => resolve(proc))
      .catch(async () => {
        proc.kill();
        console.warn('[TauriWorker] Real worker failed, falling back to mock');
        const mock = await startMockServer(httpPort);
        resolve(mock);
      });
  });
}

// ─── Native host mock ─────────────────────────────────────────────────────────
function startNativeHostMock() {
  // The native host communicates via stdin/stdout with Chrome Native Messaging protocol.
  // For E2E we patch chrome.runtime.sendNativeMessage instead.
  return {
    mock: true,
    responses: {
      ping:             { success: true },
      validate_session: { authenticated: true, username: 'test_user' },
      get_session_jwt:  { success: true, jwt: 'mock-jwt-token-abc123' },
    },
    stop: () => {},
  };
}

// ─── Global server registry ───────────────────────────────────────────────────
const _procs = new Map();

async function startAll(config = {}) {
  const {
    local  = true,   // port 8080
    tauri  = true,   // port 5000 (mock or real)
    native = true,   // native host mock
  } = config;

  if (local)  _procs.set('local',  await startMockServer(8080));
  if (tauri)  _procs.set('tauri',  await startMockServer(5000));
  const nativeMock = native ? startNativeHostMock() : null;
  if (nativeMock) _procs.set('native', nativeMock);

  console.log('[ServerManager] All test backends ready');
  return {
    local:  local  ? 'http://localhost:8080' : null,
    tauri:  tauri  ? 'http://localhost:5000' : null,
    native: nativeMock,
  };
}

function stopAll() {
  for (const [name, proc] of _procs) {
    try {
      if (proc && typeof proc.kill === 'function') proc.kill('SIGTERM');
      else if (proc && typeof proc.stop === 'function') proc.stop();
    } catch (e) {
      console.warn(`[ServerManager] Error stopping ${name}:`, e.message);
    }
  }
  _procs.clear();
}

// ─── Exports ──────────────────────────────────────────────────────────────────
module.exports = { startAll, stopAll, startMockServer, startTauriWorker, startNativeHostMock, waitForPort };

// ─── CLI: node server_manager.js ─────────────────────────────────────────────
if (require.main === module) {
  startAll().then(urls => {
    console.log('Backends:', urls);
    console.log('Press Ctrl+C to stop...');
  });
  process.on('SIGINT', () => { stopAll(); process.exit(0); });
}
