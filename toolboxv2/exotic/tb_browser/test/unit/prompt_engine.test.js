'use strict';
/**
 * Unit Tests — PromptEngine + SiteInjector
 */

const chrome = require('../mocks/chrome');

// ─── Inline PromptEngine (from prompt_engine.js, stripped of module.exports) ─

class PromptEngine {
  constructor() {
    this.rules = [];
    this.library = { prompts: {}, category_map: {} };
    this.currentMatch = null;
  }

  _testRule(match, u) {
    if (match.hostname && u.hostname !== match.hostname) return false;
    if (match.hostname_contains && !u.hostname.includes(match.hostname_contains)) return false;
    if (match.path_prefix && !u.pathname.startsWith(match.path_prefix)) return false;
    if (match.port) {
      const port = u.port || (u.protocol === 'https:' ? '443' : '80');
      if (port !== String(match.port)) return false;
    }
    if (match.port_range) {
      const port = parseInt(u.port) || (u.protocol === 'https:' ? 443 : 80);
      if (port < match.port_range[0] || port > match.port_range[1]) return false;
    }
    return true;
  }

  matchURL(url) {
    try {
      const u = new URL(url);
      for (const rule of this.rules) {
        if (this._testRule(rule.match, u)) {
          this.currentMatch = rule;
          return rule;
        }
      }
    } catch {}
    this.currentMatch = null;
    return null;
  }

  getPromptsForCategory(category) {
    const ids = this.library.category_map?.[category]
      || this.library.category_map?.['_default'] || [];
    return ids.map(id => this.library.prompts?.[id]).filter(Boolean)
      .sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
  }

  getPreprompts(rule) {
    return (rule.preprompts || []).map(id => this.library.prompts?.[id]).filter(Boolean);
  }

  buildInjectContent(rule, selectedPromptId, userText = '') {
    const preprompts = this.getPreprompts(rule);
    const selected = this.library.prompts?.[selectedPromptId];
    const parts = [];
    for (const pre of preprompts) { if (pre.auto_prepend) parts.push(pre.content); }
    if (selected) parts.push(selected.content);
    if (userText) parts.push(userText);
    return parts.join('\n\n');
  }

  search(query) {
    if (!query) return Object.values(this.library.prompts || {});
    const q = query.toLowerCase();
    return Object.values(this.library.prompts || {}).filter(p =>
      p.label.toLowerCase().includes(q)
      || (p.description || '').toLowerCase().includes(q)
      || (p.tags || []).some(t => t.includes(q))
      || (p.shortcut || '') === q
    );
  }
}

// ── Fixtures ──────────────────────────────────────────────────────────────────
const RULES = [
  { id: 'claude',  match: { hostname: 'claude.ai', path_prefix: '/chat' }, category: 'claude_chat', preprompts: ['pre1'], label: 'Claude' },
  { id: 'glm',    match: { hostname_contains: 'chatglm' }, category: 'glm_chat', preprompts: [], label: 'GLM' },
  { id: 'tb_local', match: { hostname: 'localhost', port_range: [8080, 8090] }, category: 'tb_local', preprompts: [], label: 'TB Local' },
];

const LIBRARY = {
  prompts: {
    pre1: { id: 'pre1', label: 'TB Context', type: 'preprompt', auto_prepend: true, content: 'CTX', tags: ['context'], pinned: true },
    p1:   { id: 'p1',   label: 'Code Review DE', type: 'task', shortcut: 'cr', content: 'Review!', tags: ['code', 'review'], pinned: false },
    p2:   { id: 'p2',   label: 'Summarize', type: 'task', shortcut: 'sum', content: 'Summarize!', tags: ['summary'] },
  },
  category_map: {
    claude_chat: ['pre1', 'p1'],
    _default: ['p2'],
  },
};

// ─── Tests ───────────────────────────────────────────────────────────────────
beforeEach(() => chrome._reset());

describe('PromptEngine.matchURL', () => {
  let engine;
  beforeEach(() => {
    engine = new PromptEngine();
    engine.rules = RULES;
  });

  test('matches exact hostname + path', () => {
    expect(engine.matchURL('https://claude.ai/chat/123')?.id).toBe('claude');
    expect(engine.currentMatch?.id).toBe('claude');
  });

  test('no match on wrong path', () => {
    expect(engine.matchURL('https://claude.ai/settings')).toBeNull();
    expect(engine.currentMatch).toBeNull();
  });

  test('matches hostname_contains', () => {
    expect(engine.matchURL('https://chatglm.cn/chat')?.id).toBe('glm');
  });

  test('matches port_range lower bound', () => {
    expect(engine.matchURL('http://localhost:8080')?.id).toBe('tb_local');
  });

  test('matches port_range upper bound', () => {
    expect(engine.matchURL('http://localhost:8089')?.id).toBe('tb_local');
  });

  test('no match outside port_range', () => {
    expect(engine.matchURL('http://localhost:9000')).toBeNull();
  });

  test('returns null for invalid URL', () => {
    expect(engine.matchURL('not-a-url')).toBeNull();
  });

  test('returns null for empty rules', () => {
    engine.rules = [];
    expect(engine.matchURL('https://claude.ai/chat')).toBeNull();
  });
});

describe('PromptEngine.getPromptsForCategory', () => {
  let engine;
  beforeEach(() => {
    engine = new PromptEngine();
    engine.library = LIBRARY;
  });

  test('returns prompts for known category, pinned first', () => {
    const prompts = engine.getPromptsForCategory('claude_chat');
    expect(prompts[0].id).toBe('pre1'); // pinned
    expect(prompts[1].id).toBe('p1');
  });

  test('falls back to _default for unknown category', () => {
    const prompts = engine.getPromptsForCategory('unknown');
    expect(prompts).toHaveLength(1);
    expect(prompts[0].id).toBe('p2');
  });

  test('returns empty array if category and _default missing', () => {
    engine.library = { prompts: {}, category_map: {} };
    expect(engine.getPromptsForCategory('anything')).toEqual([]);
  });
});

describe('PromptEngine.getPreprompts', () => {
  let engine;
  beforeEach(() => {
    engine = new PromptEngine();
    engine.library = LIBRARY;
  });

  test('returns prompt objects for rule preprompt ids', () => {
    const result = engine.getPreprompts({ preprompts: ['pre1'] });
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('pre1');
  });

  test('filters out missing ids', () => {
    const result = engine.getPreprompts({ preprompts: ['missing'] });
    expect(result).toEqual([]);
  });

  test('handles empty preprompts', () => {
    expect(engine.getPreprompts({ preprompts: [] })).toEqual([]);
  });
});

describe('PromptEngine.buildInjectContent', () => {
  let engine;
  beforeEach(() => {
    engine = new PromptEngine();
    engine.library = LIBRARY;
  });

  test('prepends auto_prepend prompts before selected', () => {
    const rule = { preprompts: ['pre1'] };
    const content = engine.buildInjectContent(rule, 'p1');
    expect(content).toBe('CTX\n\nReview!');
  });

  test('appends userText at end', () => {
    const rule = { preprompts: [] };
    const content = engine.buildInjectContent(rule, 'p1', 'My question');
    expect(content).toBe('Review!\n\nMy question');
  });

  test('skips preprompt if auto_prepend is false', () => {
    engine.library.prompts.pre1.auto_prepend = false;
    const rule = { preprompts: ['pre1'] };
    const content = engine.buildInjectContent(rule, 'p1');
    expect(content).toBe('Review!');
    engine.library.prompts.pre1.auto_prepend = true; // reset
  });

  test('returns only userText if no selected prompt', () => {
    const rule = { preprompts: [] };
    const content = engine.buildInjectContent(rule, 'NONEXISTENT', 'Hello');
    expect(content).toBe('Hello');
  });
});

describe('PromptEngine.search', () => {
  let engine;
  beforeEach(() => {
    engine = new PromptEngine();
    engine.library = LIBRARY;
  });

  test('returns all prompts for empty query', () => {
    expect(engine.search('')).toHaveLength(3);
  });

  test('matches by label', () => {
    const r = engine.search('code review');
    expect(r.every(p => p.label.toLowerCase().includes('code review'))).toBe(true);
  });

  test('matches by shortcut', () => {
    const r = engine.search('cr');
    expect(r.find(p => p.id === 'p1')).toBeDefined();
  });

  test('matches by tag', () => {
    const r = engine.search('summary');
    expect(r.find(p => p.id === 'p2')).toBeDefined();
  });

  test('case insensitive', () => {
    expect(engine.search('CODE')).toHaveLength(1);
  });

  test('returns empty for no match', () => {
    expect(engine.search('zzznomatch')).toHaveLength(0);
  });
});
