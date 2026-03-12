'use strict';
/**
 * Unit Tests — ToolBoxPopup (popup.js) — pure logic, no DOM
 */

const chrome = require('../mocks/chrome');

// ─── Extracted pure logic from popup.js ──────────────────────────────────────

class PopupLogic {
  constructor() {
    this.settings = { backend: 'local', customBackendUrl: '', username: null, jwt: null, isAuthenticated: false, agentName: 'speed' };
    this.apiBase = 'http://localhost:8080';
    this.isConnected = false;
    this._promptLib = null;
    this._allPrompts = [];
    this._generatedDescription = null;
    this.lastSelectedInput = null;
  }

  applySettings() {
    switch (this.settings.backend) {
      case 'local':   this.apiBase = 'http://localhost:8080'; break;
      case 'tauri':   this.apiBase = 'http://localhost:5000'; break;
      case 'native':  this.apiBase = null; break;
      case 'remote':  this.apiBase = 'https://simplecore.app'; break;
      case 'custom':  this.apiBase = this.settings.customBackendUrl || 'http://localhost:8080'; break;
      default:        this.apiBase = 'http://localhost:8080';
    }
  }

  async loadSettings() {
    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    if (stored.toolboxSettings) {
      this.settings = { ...this.settings, ...stored.toolboxSettings };
      this.applySettings();
    }
  }

  async saveSettings() {
    await chrome.storage.sync.set({ toolboxSettings: this.settings });
  }

  escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  async exportPrompts() {
    if (!this._promptLib) return null;
    return JSON.stringify(this._promptLib, null, 2);
  }

  async importPromptsFromText(jsonText) {
    const incoming = JSON.parse(jsonText);
    if (!incoming.prompts) throw new Error('Invalid format');
    if (!this._promptLib) this._promptLib = { prompts: {}, category_map: {} };
    Object.assign(this._promptLib.prompts, incoming.prompts);
    Object.entries(incoming.category_map || {}).forEach(([cat, ids]) => {
      const existing = this._promptLib.category_map[cat] || [];
      this._promptLib.category_map[cat] = [...new Set([...existing, ...ids])];
    });
    await chrome.storage.local.set({ promptLibrary: this._promptLib });
    this._allPrompts = Object.values(this._promptLib.prompts);
  }

  buildPromptObject(fields) {
    const { label, content, shortcut, site, model_hint, tags, pinned, editingId } = fields;
    const id = editingId || `custom_${Date.now()}`;
    const prompt = { id, label, content, type: 'task', pinned: !!pinned, tags: tags || [] };
    if (shortcut) prompt.shortcut = shortcut;
    if (site) prompt.site = site;
    if (model_hint) prompt.model_hint = model_hint;
    return prompt;
  }

  async savePromptToLib(promptObj) {
    if (!this._promptLib) this._promptLib = { prompts: {}, category_map: {} };
    this._promptLib.prompts[promptObj.id] = promptObj;
    if (promptObj.site) {
      const cat = promptObj.site.replace(/\./g, '_');
      if (!this._promptLib.category_map[cat]) this._promptLib.category_map[cat] = [];
      if (!this._promptLib.category_map[cat].includes(promptObj.id)) {
        this._promptLib.category_map[cat].push(promptObj.id);
      }
    }
    await chrome.storage.local.set({ promptLibrary: this._promptLib });
    this._allPrompts = Object.values(this._promptLib.prompts);
    return promptObj;
  }

  async deletePrompt(id) {
    if (!this._promptLib) return;
    delete this._promptLib.prompts[id];
    await chrome.storage.local.set({ promptLibrary: this._promptLib });
    this._allPrompts = Object.values(this._promptLib.prompts);
  }

  filterPrompts(prompts, query) {
    if (!query) return [...prompts].sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
    const q = query.toLowerCase();
    return prompts.filter(p =>
      p.label?.toLowerCase().includes(q) ||
      (p.description || '').toLowerCase().includes(q) ||
      (p.tags || []).some(t => t.includes(q)) ||
      (p.shortcut || '').includes(q)
    ).sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────
beforeEach(() => chrome._reset());

describe('applySettings', () => {
  const cases = [
    ['local',  'http://localhost:8080'],
    ['tauri',  'http://localhost:5000'],
    ['native', null],
    ['remote', 'https://simplecore.app'],
    ['custom', 'http://my.server:1234'],
  ];
  test.each(cases)('backend=%s → apiBase=%s', (backend, expected) => {
    const p = new PopupLogic();
    p.settings.backend = backend;
    if (backend === 'custom') p.settings.customBackendUrl = 'http://my.server:1234';
    p.applySettings();
    expect(p.apiBase).toBe(expected);
  });

  test('unknown backend defaults to localhost:8080', () => {
    const p = new PopupLogic();
    p.settings.backend = 'UNKNOWN';
    p.applySettings();
    expect(p.apiBase).toBe('http://localhost:8080');
  });
});

describe('loadSettings / saveSettings', () => {
  test('loads from sync storage', async () => {
    await chrome.storage.sync.set({ toolboxSettings: { backend: 'tauri', username: 'markin' } });
    const p = new PopupLogic();
    await p.loadSettings();
    expect(p.settings.username).toBe('markin');
    expect(p.apiBase).toBe('http://localhost:5000');
  });

  test('saveSettings persists to sync storage', async () => {
    const p = new PopupLogic();
    p.settings.username = 'test_user';
    await p.saveSettings();
    const stored = await chrome.storage.sync.get(['toolboxSettings']);
    expect(stored.toolboxSettings.username).toBe('test_user');
  });
});

describe('escapeHtml', () => {
  const p = new PopupLogic();
  test('escapes &', () => expect(p.escapeHtml('a&b')).toBe('a&amp;b'));
  test('escapes <', () => expect(p.escapeHtml('<b>')).toBe('&lt;b&gt;'));
  test('escapes "', () => expect(p.escapeHtml('"hi"')).toBe('&quot;hi&quot;'));
  test('escapes \'', () => expect(p.escapeHtml("it's")).toBe('it&#039;s'));
  test('safe strings unchanged', () => expect(p.escapeHtml('hello')).toBe('hello'));
});

describe('prompt library persistence', () => {
  test('savePromptToLib persists to local storage', async () => {
    const p = new PopupLogic();
    const prompt = p.buildPromptObject({ label: 'Test', content: 'Do it', tags: ['test'] });
    await p.savePromptToLib(prompt);
    const stored = await chrome.storage.local.get(['promptLibrary']);
    expect(stored.promptLibrary.prompts[prompt.id].label).toBe('Test');
  });

  test('savePromptToLib adds to category_map when site given', async () => {
    const p = new PopupLogic();
    const prompt = p.buildPromptObject({ label: 'Claude Prompt', content: 'X', site: 'claude.ai', tags: [] });
    await p.savePromptToLib(prompt);
    expect(p._promptLib.category_map['claude_ai']).toContain(prompt.id);
  });

  test('deletePrompt removes from storage', async () => {
    const p = new PopupLogic();
    const prompt = p.buildPromptObject({ label: 'Delete me', content: 'bye', tags: [] });
    await p.savePromptToLib(prompt);
    expect(p._allPrompts).toHaveLength(1);
    await p.deletePrompt(prompt.id);
    expect(p._allPrompts).toHaveLength(0);
    const stored = await chrome.storage.local.get(['promptLibrary']);
    expect(stored.promptLibrary.prompts[prompt.id]).toBeUndefined();
  });

  test('editing existing prompt updates in place', async () => {
    const p = new PopupLogic();
    const original = p.buildPromptObject({ label: 'Old', content: 'old content', tags: [] });
    await p.savePromptToLib(original);
    const updated = p.buildPromptObject({ label: 'New', content: 'new content', tags: [], editingId: original.id });
    await p.savePromptToLib(updated);
    expect(p._allPrompts).toHaveLength(1);
    expect(p._allPrompts[0].label).toBe('New');
  });

  test('importPrompts merges with existing', async () => {
    const p = new PopupLogic();
    await p.savePromptToLib(p.buildPromptObject({ label: 'Existing', content: 'e', tags: [] }));
    const incoming = {
      prompts: { imp1: { id: 'imp1', label: 'Imported', content: 'i', tags: [], type: 'task' } },
      category_map: {}
    };
    await p.importPromptsFromText(JSON.stringify(incoming));
    expect(p._allPrompts).toHaveLength(2);
  });

  test('importPrompts throws on invalid JSON', async () => {
    const p = new PopupLogic();
    await expect(p.importPromptsFromText('not json')).rejects.toThrow();
  });

  test('importPrompts throws when prompts key missing', async () => {
    const p = new PopupLogic();
    await expect(p.importPromptsFromText('{}')).rejects.toThrow('Invalid format');
  });

  test('exportPrompts returns valid JSON string', async () => {
    const p = new PopupLogic();
    await p.savePromptToLib(p.buildPromptObject({ label: 'X', content: 'x', tags: [] }));
    const json = await p.exportPrompts();
    const parsed = JSON.parse(json);
    expect(parsed.prompts).toBeDefined();
  });

  test('category_map deduplicates on repeated save', async () => {
    const p = new PopupLogic();
    const prompt = p.buildPromptObject({ label: 'X', content: 'x', site: 'test.com', tags: [] });
    await p.savePromptToLib(prompt);
    await p.savePromptToLib(prompt); // save again
    const ids = p._promptLib.category_map['test_com'];
    expect(ids.filter(id => id === prompt.id)).toHaveLength(1);
  });
});

describe('filterPrompts', () => {
  const prompts = [
    { id: 'a', label: 'Code Review', pinned: true, tags: ['code'], shortcut: 'cr', description: '' },
    { id: 'b', label: 'Summarize', pinned: false, tags: ['summary'], shortcut: 'sum', description: 'short summary' },
    { id: 'c', label: 'Translate', pinned: false, tags: ['lang'], shortcut: 'tr', description: '' },
  ];

  const p = new PopupLogic();

  test('empty query returns all, pinned first', () => {
    const result = p.filterPrompts(prompts, '');
    expect(result[0].id).toBe('a');
    expect(result).toHaveLength(3);
  });

  test('filters by label', () => {
    expect(p.filterPrompts(prompts, 'code')).toHaveLength(1);
  });

  test('filters by shortcut', () => {
    expect(p.filterPrompts(prompts, 'sum')).toHaveLength(1);
  });

  test('filters by tag', () => {
    expect(p.filterPrompts(prompts, 'lang')).toHaveLength(1);
  });

  test('filters by description', () => {
    expect(p.filterPrompts(prompts, 'short summary')).toHaveLength(1);
  });

  test('no match returns empty', () => {
    expect(p.filterPrompts(prompts, 'zzznomatch')).toHaveLength(0);
  });
});
