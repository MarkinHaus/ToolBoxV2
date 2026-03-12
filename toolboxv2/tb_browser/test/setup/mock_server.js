#!/usr/bin/env node
/**
 * mock_server.js — Minimal HTTP mock for ToolBoxV2 API
 * Simulates: health, validateSession, isaa, PasswordManager, CloudM
 *
 * Usage: node mock_server.js [port]
 */

const http = require('http');
const PORT = parseInt(process.argv[2] || '8080');

const MOCK_JWT  = 'mock-jwt-token-abc123';
const MOCK_USER = 'test_user';

const ROUTES = {
  'GET /health': () => ({
    status: 'healthy', worker_id: `mock_${PORT}`, pid: process.pid, timestamp: Date.now() / 1000
  }),

  'POST /validateSession': (body) => {
    const valid = body?.Jwt_claim === MOCK_JWT && body?.Username === MOCK_USER;
    return { result: { data_info: valid ? 'Valid Session' : 'Invalid Session', status: valid ? 'ok' : 'error' } };
  },

  'GET /api/CloudM/openVersion': () => ({
    result: { data: '3.0.0', status: 'ok' }
  }),

  'POST /api/CloudM/open_check_cli_auth': (body) => ({
    authenticated: true,
    username: MOCK_USER,
    jwt_token: MOCK_JWT
  }),

  'GET /api/isaa/listAllAgents': () => ({
    result: { data: ['speed', 'gpt4', 'claude'], status: 'ok' }
  }),

  'POST /api/isaa/mini_task_completion': (body) => ({
    result: {
      data: JSON.stringify({
        optimized_prompt: `Optimized: ${body?.mini_task?.slice(0, 50) || 'test'}`,
        label: 'Generated Prompt',
        description: 'Auto-generated'
      }),
      status: 'ok'
    }
  }),

  'POST /api/PasswordManager/list_passwords': () => ({
    result: {
      data: [
        { id: 'pw1', title: 'GitHub',   username: 'markin', url: 'https://github.com', has_totp: false },
        { id: 'pw2', title: 'Anthropic', username: 'markin', url: 'https://claude.ai', has_totp: true },
      ],
      status: 'ok'
    }
  }),

  'POST /api/PasswordManager/add_password': (body) => ({
    result: { data: { id: `pw_${Date.now()}`, ...body }, status: 'ok' }
  }),

  'POST /api/PasswordManager/get_password_by_url_username': (body) => ({
    result: { data: null, status: 'error' }
  }),

  'DELETE /api/PasswordManager/delete_password': () => ({
    result: { status: 'ok' }
  }),

  'GET /api/prompts/library': () => ({
    prompts: {
      p1: { id: 'p1', label: 'Server Prompt', content: 'From server', type: 'task', tags: [], pinned: false }
    },
    category_map: { _default: ['p1'] }
  }),
};

const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', chunk => body += chunk);
  req.on('end', () => {
    const key = `${req.method} ${req.url.split('?')[0]}`;
    const handler = ROUTES[key];

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Username');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    if (!handler) {
      res.writeHead(404);
      res.end(JSON.stringify({ error: `No mock for: ${key}` }));
      return;
    }

    let parsedBody = {};
    try { parsedBody = body ? JSON.parse(body) : {}; } catch {}

    try {
      const result = handler(parsedBody);
      res.writeHead(200);
      res.end(JSON.stringify(result));
    } catch (e) {
      res.writeHead(500);
      res.end(JSON.stringify({ error: e.message }));
    }
  });
});

server.listen(PORT, () => {
  console.log(`[MockServer] Listening on http://localhost:${PORT}`);
  // Signal to parent process that we're ready
  if (process.send) process.send({ type: 'ready', port: PORT });
});

process.on('SIGTERM', () => { server.close(); process.exit(0); });
process.on('SIGINT',  () => { server.close(); process.exit(0); });
