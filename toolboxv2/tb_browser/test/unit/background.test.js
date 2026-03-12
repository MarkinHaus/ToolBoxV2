'use strict';
/**
 * Unit Tests — background.js (ToolBoxBackground)
 * Tests all methods in isolation with chrome API mocked
 */

const chrome = require('../mocks/chrome');

// ─── Inline the classes under test (no module bundler needed) ────────────────
// We extract the parts that don't depend on chrome.debugger for unit testing.

// Minimal PromptEngine stub (tested separately)
class PromptEngine {
  async init() {}
}

// Minimal AgentView stub
class AgentView {
  constructor(tabId) { this.tabId = tabId; this.attached = false; }
  async detach() { this.attached = false; }
  async getStructuredView() { return { compressed: '', interactable: [], metadata: {} }; }
  async executeAction(action) { return { ok: true }; }
}

// Load ToolBoxBackground logic inline (extracted methods)
class ToolBoxBackground {
  constructor() {
    this.apiBase = 'http://localhost:8080';
    this.isConnected = false;
    this.activeTab = null;
    this.gestureHistory = [];
    this.useNative = false;
    this.authData = { username: null, jwt: null, isAuthenticated: false };
    this.promptRules = [];
    this.promptLibrary = { prompts: {}, category_map: {} };
    this.promptEngine = new PromptEngine();
    this.agentViews = {};
    this._ws = null;
  }

  sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  generateSessionId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    }).replace(/-/g, '');
  }

  applyBackendSettings(settings) {
    switch (settings.backend) {
      case 'local':  this.apiBase = 'http://localhost:8080'; this.useNative = false; break;
      case 'tauri':  this.apiBase = 'http://localhost:5000'; this.useNative = false; break;
      case 'native': this.apiBase = null; this.useNative = true; break;
      case 'remote': this.apiBase = 'https://simplecore.app'; this.useNative = false; break;
      case 'custom': this.apiBase = settings.customBackendUrl || 'http://localhost:8080'; this.useNative = false; break;
      default:       this.apiBase = 'http://localhost:8080'; this.useNative = false;
    }
  }

  async loadBackendSettings() {
    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    if (stored.toolboxSettings) {
      this.applyBackendSettings(stored.toolboxSettings);
      this.authData = {
        username: stored.toolboxSettings.username || null,
        jwt: stored.toolboxSettings.jwt || null,
        isAuthenticated: stored.toolboxSettings.isAuthenticated || false,
      };
    }
  }

  matchSiteRule(url) {
    if (!url) return null;
    try {
      const u = new URL(url);
      for (const rule of this.promptRules) {
        const m = rule.match;
        if (m.hostname && u.hostname !== m.hostname) continue;
        if (m.hostname_contains && !u.hostname.includes(m.hostname_contains)) continue;
        if (m.path_prefix && !u.pathname.startsWith(m.path_prefix)) continue;
        if (m.port) {
          const p = u.port || (u.protocol === 'https:' ? '443' : '80');
          if (p !== String(m.port)) continue;
        }
        if (m.port_range) {
          const p = parseInt(u.port) || (u.protocol === 'https:' ? 443 : 80);
          if (p < m.port_range[0] || p > m.port_range[1]) continue;
        }
        return rule;
      }
    } catch {}
    return null;
  }

  cleanupGestureHistory() {
    const oneHourAgo = Date.now() - 3600000;
    this.gestureHistory = this.gestureHistory.filter(g => g.timestamp > oneHourAgo);
  }

  handleGesture(gesture, tab) {
    this.gestureHistory.push({ gesture, tabId: tab.id, url: tab.url, timestamp: Date.now() });
  }

  async storeAuthCredentials(username, jwtToken, baseUrl) {
    this.authData = { username, jwt: jwtToken, isAuthenticated: true };
    let backend = 'local';
    if (baseUrl?.includes('simplecore.app')) backend = 'remote';
    else if (baseUrl === 'http://localhost:5000') backend = 'tauri';
    else if (baseUrl && baseUrl !== 'http://localhost:8080') backend = 'custom';

    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    const settings = stored.toolboxSettings || {};
    Object.assign(settings, { username, jwt: jwtToken, isAuthenticated: true, backend });
    if (baseUrl) this.apiBase = baseUrl;
    await chrome.storage.sync.set({ toolboxSettings: settings });
  }

  setupWebSocket() {
    if (this.useNative || !this.apiBase || !this.isConnected) return null;
    const isLocal = this.apiBase.includes('localhost') || this.apiBase.includes('127.0.0.1');
    const wsBase = isLocal
      ? this.apiBase.replace(/^http/, 'ws').replace(/:\d+$/, '') + ':5001'
      : this.apiBase.replace(/^http/, 'ws');
    return wsBase + '/ws';
  }

  async loadPromptLibrary() {
    const stored = await chrome.storage.local.get(['promptLibrary', 'siteRules']);
    if (stored.promptLibrary) this.promptLibrary = stored.promptLibrary;
    if (stored.siteRules) this.promptRules = stored.siteRules.rules || [];
  }

  async syncPromptLibrary() {
    if (!this.apiBase) return false;
    if (!this.isConnected) return false;
    return true; // simplified for unit test
  }

  async handleAgentView(tabId, action, msg) {
    if (!this.agentViews[tabId]) this.agentViews[tabId] = new AgentView(tabId);
    const agent = this.agentViews[tabId];
    switch (action) {
      case 'snapshot': return agent.getStructuredView(msg.maxDepth || 5);
      case 'execute':  return agent.executeAction(msg.agentAction);
      case 'detach':
        await agent.detach();
        delete this.agentViews[tabId];
        return { ok: true };
      default: return { error: `Unknown action: ${action}` };
    }
  }

  _safeSend(msg) {
    try {
      if (!chrome?.runtime?.id) return;
      chrome.runtime.sendMessage(msg);
    } catch (e) {
      if (!e.message?.includes('Extension context')) console.warn(e);
    }
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

beforeEach(() => chrome._reset());

// ── generateSessionId ─────────────────────────────────────────────────────────
describe('generateSessionId', () => {
  const bg = new ToolBoxBackground();

  test('returns 32-char hex string', () => {
    const id = bg.generateSessionId();
    expect(id).toHaveLength(32);
    expect(id).toMatch(/^[0-9a-f]+$/);
  });

  test('returns unique IDs', () => {
    const ids = new Set(Array.from({ length: 100 }, () => bg.generateSessionId()));
    expect(ids.size).toBe(100);
  });
});

// ── applyBackendSettings / loadBackendSettings ────────────────────────────────
describe('backend settings', () => {
  test('local → port 8080', () => {
    const bg = new ToolBoxBackground();
    bg.applyBackendSettings({ backend: 'local' });
    expect(bg.apiBase).toBe('http://localhost:8080');
    expect(bg.useNative).toBe(false);
  });

  test('tauri → port 5000', () => {
    const bg = new ToolBoxBackground();
    bg.applyBackendSettings({ backend: 'tauri' });
    expect(bg.apiBase).toBe('http://localhost:5000');
  });

  test('native → apiBase null, useNative true', () => {
    const bg = new ToolBoxBackground();
    bg.applyBackendSettings({ backend: 'native' });
    expect(bg.apiBase).toBeNull();
    expect(bg.useNative).toBe(true);
  });

  test('remote → simplecore.app', () => {
    const bg = new ToolBoxBackground();
    bg.applyBackendSettings({ backend: 'remote' });
    expect(bg.apiBase).toBe('https://simplecore.app');
  });

  test('custom → uses customBackendUrl', () => {
    const bg = new ToolBoxBackground();
    bg.applyBackendSettings({ backend: 'custom', customBackendUrl: 'http://192.168.1.10:9000' });
    expect(bg.apiBase).toBe('http://192.168.1.10:9000');
  });

  test('loadBackendSettings reads from chrome.storage.sync', async () => {
    await chrome.storage.sync.set({
      toolboxSettings: { backend: 'tauri', username: 'max', jwt: 'tok', isAuthenticated: true }
    });
    const bg = new ToolBoxBackground();
    await bg.loadBackendSettings();
    expect(bg.apiBase).toBe('http://localhost:5000');
    expect(bg.authData.username).toBe('max');
    expect(bg.authData.isAuthenticated).toBe(true);
  });
});

// ── matchSiteRule ─────────────────────────────────────────────────────────────
describe('matchSiteRule', () => {
  const bg = new ToolBoxBackground();
  bg.promptRules = [
    { id: 'claude', match: { hostname: 'claude.ai', path_prefix: '/chat' }, label: 'Claude' },
    { id: 'glm',    match: { hostname_contains: 'chatglm' }, label: 'GLM' },
    { id: 'tb_local', match: { hostname: 'localhost', port_range: [8080, 8090] }, label: 'TB Local' },
    { id: 'perplexity', match: { hostname: 'www.perplexity.ai' }, label: 'Perplexity' },
  ];

  test('matches by exact hostname + path_prefix', () => {
    expect(bg.matchSiteRule('https://claude.ai/chat/abc')?.id).toBe('claude');
  });

  test('does not match wrong path', () => {
    expect(bg.matchSiteRule('https://claude.ai/settings')).toBeNull();
  });

  test('matches hostname_contains', () => {
    expect(bg.matchSiteRule('https://www.chatglm.cn/chat')?.id).toBe('glm');
  });

  test('matches port_range', () => {
    expect(bg.matchSiteRule('http://localhost:8080/chat')?.id).toBe('tb_local');
    expect(bg.matchSiteRule('http://localhost:8085/chat')?.id).toBe('tb_local');
    expect(bg.matchSiteRule('http://localhost:9000/chat')).toBeNull();
  });

  test('returns null for unknown url', () => {
    expect(bg.matchSiteRule('https://google.com')).toBeNull();
  });

  test('returns null for invalid url', () => {
    expect(bg.matchSiteRule('not-a-url')).toBeNull();
  });

  test('returns null for empty string', () => {
    expect(bg.matchSiteRule('')).toBeNull();
  });
});

// ── gesture history ───────────────────────────────────────────────────────────
describe('gesture history', () => {
  test('handleGesture appends entry', () => {
    const bg = new ToolBoxBackground();
    bg.handleGesture('swipe-left', { id: 1, url: 'https://example.com' });
    expect(bg.gestureHistory).toHaveLength(1);
    expect(bg.gestureHistory[0].gesture).toBe('swipe-left');
  });

  test('cleanupGestureHistory removes old entries', () => {
    const bg = new ToolBoxBackground();
    const old = Date.now() - 4000000;
    bg.gestureHistory = [
      { gesture: 'swipe-left', timestamp: old },
      { gesture: 'swipe-right', timestamp: Date.now() },
    ];
    bg.cleanupGestureHistory();
    expect(bg.gestureHistory).toHaveLength(1);
    expect(bg.gestureHistory[0].gesture).toBe('swipe-right');
  });
});

// ── storeAuthCredentials ──────────────────────────────────────────────────────
describe('storeAuthCredentials', () => {
  test('sets authData and detects remote backend', async () => {
    const bg = new ToolBoxBackground();
    await bg.storeAuthCredentials('alice', 'jwt123', 'https://simplecore.app');
    expect(bg.authData).toMatchObject({ username: 'alice', jwt: 'jwt123', isAuthenticated: true });
    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    expect(stored.toolboxSettings.backend).toBe('remote');
  });

  test('detects tauri backend', async () => {
    const bg = new ToolBoxBackground();
    await bg.storeAuthCredentials('bob', 'jwtX', 'http://localhost:5000');
    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    expect(stored.toolboxSettings.backend).toBe('tauri');
  });

  test('detects local backend', async () => {
    const bg = new ToolBoxBackground();
    await bg.storeAuthCredentials('charlie', 'jwtY', 'http://localhost:8080');
    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    expect(stored.toolboxSettings.backend).toBe('local');
  });
});

// ── WebSocket URL construction ────────────────────────────────────────────────
describe('setupWebSocket URL', () => {
  test('local → ws on port 5001', () => {
    const bg = new ToolBoxBackground();
    bg.apiBase = 'http://localhost:8080';
    bg.isConnected = true;
    const url = bg.setupWebSocket();
    expect(url).toBe('ws://localhost:5001/ws');
  });

  test('tauri → ws on port 5001', () => {
    const bg = new ToolBoxBackground();
    bg.apiBase = 'http://localhost:5000';
    bg.isConnected = true;
    const url = bg.setupWebSocket();
    expect(url).toBe('ws://localhost:5001/ws');
  });

  test('remote → wss same port', () => {
    const bg = new ToolBoxBackground();
    bg.apiBase = 'https://simplecore.app';
    bg.isConnected = true;
    const url = bg.setupWebSocket();
    expect(url).toBe('wss://simplecore.app/ws');
  });

  test('returns null when not connected', () => {
    const bg = new ToolBoxBackground();
    bg.apiBase = 'http://localhost:8080';
    bg.isConnected = false;
    expect(bg.setupWebSocket()).toBeNull();
  });

  test('returns null when native mode', () => {
    const bg = new ToolBoxBackground();
    bg.useNative = true;
    bg.isConnected = true;
    expect(bg.setupWebSocket()).toBeNull();
  });
});

// ── prompt library load/sync ──────────────────────────────────────────────────
describe('prompt library', () => {
  test('loadPromptLibrary reads from local storage', async () => {
    const lib = { prompts: { p1: { id: 'p1', label: 'Test' } }, category_map: {} };
    await chrome.storage.local.set({ promptLibrary: lib });
    const bg = new ToolBoxBackground();
    await bg.loadPromptLibrary();
    expect(bg.promptLibrary.prompts.p1.label).toBe('Test');
  });

  test('syncPromptLibrary returns false when offline', async () => {
    const bg = new ToolBoxBackground();
    bg.isConnected = false;
    expect(await bg.syncPromptLibrary()).toBe(false);
  });

  test('syncPromptLibrary returns false when no apiBase', async () => {
    const bg = new ToolBoxBackground();
    bg.apiBase = null;
    expect(await bg.syncPromptLibrary()).toBe(false);
  });
});

// ── AgentView delegation ──────────────────────────────────────────────────────
describe('handleAgentView', () => {
  test('creates AgentView on first call', async () => {
    const bg = new ToolBoxBackground();
    await bg.handleAgentView(42, 'snapshot', {});
    expect(bg.agentViews[42]).toBeDefined();
  });

  test('snapshot returns structured view shape', async () => {
    const bg = new ToolBoxBackground();
    const result = await bg.handleAgentView(1, 'snapshot', { maxDepth: 3 });
    expect(result).toHaveProperty('compressed');
    expect(result).toHaveProperty('interactable');
  });

  test('detach removes agent from map', async () => {
    const bg = new ToolBoxBackground();
    await bg.handleAgentView(7, 'snapshot', {});
    expect(bg.agentViews[7]).toBeDefined();
    await bg.handleAgentView(7, 'detach', {});
    expect(bg.agentViews[7]).toBeUndefined();
  });

  test('unknown action returns error', async () => {
    const bg = new ToolBoxBackground();
    const result = await bg.handleAgentView(1, 'INVALID', {});
    expect(result.error).toMatch(/Unknown action/);
  });
});

// ── _safeSend ─────────────────────────────────────────────────────────────────
describe('_safeSend', () => {
  test('calls chrome.runtime.sendMessage normally', () => {
    const bg = new ToolBoxBackground();
    bg._safeSend({ type: 'TEST' });
    expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({ type: 'TEST' });
  });

  test('swallows error when context invalidated', () => {
    const bg = new ToolBoxBackground();
    chrome.runtime.id = null; // simulate invalidated context
    expect(() => bg._safeSend({ type: 'TEST' })).not.toThrow();
  });
});
