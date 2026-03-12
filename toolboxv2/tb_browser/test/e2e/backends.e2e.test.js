'use strict';
/**
 * E2E Tests — ToolBox Pro Extension (API integration)
 *
 * Tests all backend modes:
 *   - local  (mock on :8080)
 *   - tauri  (mock/real on :5000)
 *   - native (in-process mock)
 *   - remote (simplecore.app — skipped unless REMOTE_E2E=1)
 *
 * Uses node-fetch to hit the mock servers directly,
 * simulating what background.js / popup.js would do.
 */

const { startAll, stopAll, waitForPort } = require('../setup/server_manager');

// ─── fetch polyfill for Node 22 (already has global fetch, but just in case) ─
const nodeFetch = (...args) => import('node-fetch').then(m => m.default(...args)).catch(() => fetch(...args));

const MOCK_JWT  = 'mock-jwt-token-abc123';
const MOCK_USER = 'test_user';

let servers;

// ─── Setup: start all mock servers once ──────────────────────────────────────
beforeAll(async () => {
  servers = await startAll({ local: true, tauri: true, native: true });
}, 30000);

afterAll(() => stopAll());

// ─── Helper ───────────────────────────────────────────────────────────────────
async function api(base, endpoint, method = 'GET', body = null, jwt = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (jwt) headers['Authorization'] = `Bearer ${jwt}`;
  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${base}${endpoint}`, opts);
  return { status: res.status, data: await res.json() };
}

// ─── LOCAL BACKEND (port 8080) ────────────────────────────────────────────────
describe('Local backend (:8080)', () => {
  const BASE = 'http://localhost:8080';

  test('GET /health → healthy', async () => {
    const { status, data } = await api(BASE, '/health');
    expect(status).toBe(200);
    expect(data.status).toBe('healthy');
  });

  test('POST /validateSession — valid credentials', async () => {
    const { status, data } = await api(BASE, '/validateSession', 'POST', {
      Username: MOCK_USER, Jwt_claim: MOCK_JWT
    });
    expect(status).toBe(200);
    expect(data.result.data_info).toBe('Valid Session');
  });

  test('POST /validateSession — invalid credentials', async () => {
    const { data } = await api(BASE, '/validateSession', 'POST', {
      Username: 'wrong', Jwt_claim: 'wrong'
    });
    expect(data.result.data_info).toBe('Invalid Session');
  });

  test('GET /api/isaa/listAllAgents → agent list', async () => {
    const { status, data } = await api(BASE, '/api/isaa/listAllAgents', 'GET', null, MOCK_JWT);
    expect(status).toBe(200);
    expect(Array.isArray(data.result.data)).toBe(true);
    expect(data.result.data).toContain('speed');
  });

  test('POST /api/isaa/mini_task_completion → generates prompt', async () => {
    const { status, data } = await api(BASE, '/api/isaa/mini_task_completion', 'POST', {
      mini_task: 'Generate a code review prompt',
      agent_name: 'speed'
    }, MOCK_JWT);
    expect(status).toBe(200);
    const inner = JSON.parse(data.result.data);
    expect(inner).toHaveProperty('optimized_prompt');
    expect(inner).toHaveProperty('label');
  });

  test('POST /api/PasswordManager/list_passwords → password list', async () => {
    const { status, data } = await api(BASE, '/api/PasswordManager/list_passwords', 'POST', {}, MOCK_JWT);
    expect(status).toBe(200);
    expect(Array.isArray(data.result.data)).toBe(true);
    expect(data.result.data.length).toBeGreaterThan(0);
    expect(data.result.data[0]).toHaveProperty('username');
  });

  test('POST /api/PasswordManager/add_password → returns new entry', async () => {
    const { status, data } = await api(BASE, '/api/PasswordManager/add_password', 'POST', {
      title: 'TestSite', username: 'user1', password: 'secret', url: 'https://test.com'
    }, MOCK_JWT);
    expect(status).toBe(200);
    expect(data.result.data.title).toBe('TestSite');
  });

  test('GET /api/CloudM/openVersion → returns version string', async () => {
    const { status, data } = await api(BASE, '/api/CloudM/openVersion');
    expect(status).toBe(200);
    expect(data.result.data).toMatch(/\d+\.\d+\.\d+/);
  });

  test('GET /api/prompts/library → library with prompts', async () => {
    const { status, data } = await api(BASE, '/api/prompts/library', 'GET', null, MOCK_JWT);
    expect(status).toBe(200);
    expect(data.prompts).toBeDefined();
    expect(typeof data.prompts).toBe('object');
  });

  test('404 for unknown route', async () => {
    const { status } = await api(BASE, '/api/nonexistent');
    expect(status).toBe(404);
  });
});

// ─── TAURI BACKEND (port 5000) ────────────────────────────────────────────────
describe('Tauri backend (:5000)', () => {
  const BASE = 'http://localhost:5000';

  test('GET /health → healthy', async () => {
    const { status, data } = await api(BASE, '/health');
    expect(status).toBe(200);
    expect(data.status).toBe('healthy');
  });

  test('same API surface as local', async () => {
    const { data } = await api(BASE, '/api/isaa/listAllAgents', 'GET', null, MOCK_JWT);
    expect(Array.isArray(data.result.data)).toBe(true);
  });

  test('POST /api/CloudM/open_check_cli_auth → returns JWT', async () => {
    const { status, data } = await api(BASE, '/api/CloudM/open_check_cli_auth', 'POST', {
      session_id: 'test-session-123'
    });
    expect(status).toBe(200);
    expect(data.authenticated).toBe(true);
    expect(data.jwt_token).toBe(MOCK_JWT);
    expect(data.username).toBe(MOCK_USER);
  });

  test('port 5000 responds independently from port 8080', async () => {
    const [local, tauri] = await Promise.all([
      api('http://localhost:8080', '/health'),
      api('http://localhost:5000', '/health'),
    ]);
    expect(local.status).toBe(200);
    expect(tauri.status).toBe(200);
  });
});

// ─── NATIVE MOCK ─────────────────────────────────────────────────────────────
describe('Native backend (in-process mock)', () => {
  test('nativeHostMock has expected response keys', () => {
    expect(servers.native.mock).toBe(true);
    expect(servers.native.responses.ping.success).toBe(true);
    expect(servers.native.responses.validate_session.authenticated).toBe(true);
    expect(servers.native.responses.get_session_jwt.jwt).toBe(MOCK_JWT);
  });

  test('simulates ping response', () => {
    const resp = servers.native.responses['ping'];
    expect(resp.success).toBe(true);
  });

  test('simulates session validation', () => {
    const resp = servers.native.responses['validate_session'];
    expect(resp.authenticated).toBe(true);
    expect(resp.username).toBe(MOCK_USER);
  });

  test('simulates JWT retrieval', () => {
    const resp = servers.native.responses['get_session_jwt'];
    expect(resp.success).toBe(true);
    expect(resp.jwt).toBeTruthy();
  });
});

// ─── BACKEND SWITCHING SIMULATION ────────────────────────────────────────────
describe('Backend URL construction', () => {
  function getApiBase(backend, customUrl = '') {
    switch (backend) {
      case 'local':  return 'http://localhost:8080';
      case 'tauri':  return 'http://localhost:5000';
      case 'native': return null;
      case 'remote': return 'https://simplecore.app';
      case 'custom': return customUrl || 'http://localhost:8080';
      default:       return 'http://localhost:8080';
    }
  }

  function getWsUrl(apiBase) {
    if (!apiBase) return null;
    const isLocal = apiBase.includes('localhost') || apiBase.includes('127.0.0.1');
    return isLocal
      ? apiBase.replace(/^http/, 'ws').replace(/:\d+$/, '') + ':5001'
      : apiBase.replace(/^http/, 'ws');
  }

  test.each([
    ['local',  'http://localhost:8080', 'ws://localhost:5001'],
    ['tauri',  'http://localhost:5000', 'ws://localhost:5001'],
    ['remote', 'https://simplecore.app', 'wss://simplecore.app'],
    ['native', null, null],
  ])('backend=%s → apiBase=%s wsUrl=%s', (backend, expectedApi, expectedWs) => {
    const base = getApiBase(backend);
    expect(base).toBe(expectedApi);
    expect(getWsUrl(base)).toBe(expectedWs);
  });

  test('custom URL is passed through', () => {
    expect(getApiBase('custom', 'http://192.168.1.5:9000')).toBe('http://192.168.1.5:9000');
  });
});

// ─── AUTH FLOW SIMULATION ─────────────────────────────────────────────────────
describe('Auth flow integration', () => {
  const BASE = 'http://localhost:8080';

  test('full session check flow', async () => {
    // 1. Validate existing session
    const validateRes = await api(BASE, '/validateSession', 'POST', {
      Username: MOCK_USER, Jwt_claim: MOCK_JWT
    });
    expect(validateRes.data.result.data_info).toBe('Valid Session');
  });

  test('unauthenticated agent list still returns data (mock has no auth enforcement)', async () => {
    // In real server, missing JWT → 401; mock is permissive
    const { status } = await api(BASE, '/api/isaa/listAllAgents');
    expect(status).toBe(200);
  });
});

// ─── PROMPT SYNC SIMULATION ───────────────────────────────────────────────────
describe('Prompt library sync', () => {
  test('local backend syncs library', async () => {
    const { data } = await api('http://localhost:8080', '/api/prompts/library', 'GET', null, MOCK_JWT);
    expect(data.prompts.p1.label).toBe('Server Prompt');
  });

  test('tauri backend serves same library', async () => {
    const { data } = await api('http://localhost:5000', '/api/prompts/library', 'GET', null, MOCK_JWT);
    expect(data.prompts).toBeDefined();
  });
});

// ─── REMOTE BACKEND (optional, requires REMOTE_E2E=1) ────────────────────────
const REMOTE = process.env.REMOTE_E2E === '1';
describe.skip(!REMOTE ? 'Remote backend (simplecore.app) [SKIPPED — set REMOTE_E2E=1]' : 'Remote backend', () => {
  const BASE = 'https://simplecore.app';

  test('GET /health responds', async () => {
    const res = await fetch(`${BASE}/health`);
    expect(res.ok).toBe(true);
  });

  test('GET /api/CloudM/openVersion responds', async () => {
    const res = await fetch(`${BASE}/api/CloudM/openVersion`);
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data.result.data).toBeTruthy();
  });
});
