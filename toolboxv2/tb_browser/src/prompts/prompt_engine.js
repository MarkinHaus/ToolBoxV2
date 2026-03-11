// prompt_engine.js — TB Extension Prompt Library Core
// Handles: URL matching, prompt loading, category selection, injection prep

'use strict';

class PromptEngine {
  constructor() {
    this.rules = [];
    this.library = { prompts: {}, category_map: {} };
    this.currentMatch = null;
  }

  async init() {
    const stored = await chrome.storage.local.get(['promptLibrary', 'siteRules']);
    if (stored.promptLibrary) this.library = stored.promptLibrary;
    if (stored.siteRules) this.rules = stored.siteRules.rules || [];
    // Fallback: load bundled defaults
    if (!this.rules.length) await this._loadBundled();
  }

  async _loadBundled() {
    try {
      const [rulesRes, libRes] = await Promise.all([
        fetch(chrome.runtime.getURL('src/prompts/site_rules.json')),
        fetch(chrome.runtime.getURL('src/prompts/library.json'))
      ]);
      this.rules = (await rulesRes.json()).rules || [];
      this.library = await libRes.json();
    } catch (e) {
      console.error('[PromptEngine] Failed to load bundled defaults:', e);
    }
  }

  // URL → matching rule (password-manager style)
  matchURL(url) {
    try {
      const u = new URL(url);
      for (const rule of this.rules) {
        if (this._testRule(rule.match, u)) {
          this.currentMatch = rule;
          return rule;
        }
      }
    } catch (_) {}
    this.currentMatch = null;
    return null;
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

  // Get prompts for a category, ordered by pin + relevance
  getPromptsForCategory(category) {
    const ids = this.library.category_map?.[category]
      || this.library.category_map?.['_default']
      || [];
    return ids
      .map(id => this.library.prompts?.[id])
      .filter(Boolean)
      .sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
  }

  // Get preprompts for current match
  getPreprompts(rule) {
    return (rule.preprompts || [])
      .map(id => this.library.prompts?.[id])
      .filter(Boolean);
  }

  // Build final inject string: preprompts + selected prompt
  buildInjectContent(rule, selectedPromptId, userText = '') {
    const preprompts = this.getPreprompts(rule);
    const selected = this.library.prompts?.[selectedPromptId];
    const parts = [];

    for (const pre of preprompts) {
      if (pre.auto_prepend) parts.push(pre.content);
    }
    if (selected) parts.push(selected.content);
    if (userText) parts.push(userText);

    return parts.join('\n\n');
  }

  // Search prompts by query
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

  async syncFromServer(apiBase) {
    try {
      const res = await fetch(`${apiBase}/api/prompts/library`);
      if (!res.ok) return false;
      const data = await res.json();
      this.library = data;
      await chrome.storage.local.set({ promptLibrary: data, lastSync: Date.now() });
      return true;
    } catch (_) {
      return false;
    }
  }
}

// Content injection per site method
class SiteInjector {
  static inject(selector, text, method) {
    const el = this._find(selector);
    if (!el) return false;

    switch (method) {
      case 'prosemirror': return this._injectProsemirror(el, text);
      case 'react': return this._injectReact(el, text);
      case 'native_tb': return this._injectTB(text);
      case 'textarea':
      default: return this._injectTextarea(el, text);
    }
  }

  static _find(selector) {
    const selectors = selector.split(',').map(s => s.trim());
    for (const s of selectors) {
      const el = document.querySelector(s);
      if (el) return el;
    }
    return null;
  }

  static _injectProsemirror(el, text) {
    el.focus();
    // Select all existing content first
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(el);
    selection.removeAllRanges();
    selection.addRange(range);
    // Insert text (works for ProseMirror + contenteditable)
    document.execCommand('insertText', false, text);
    el.dispatchEvent(new InputEvent('input', { bubbles: true, data: text }));
    return true;
  }

  static _injectTextarea(el, text) {
    el.focus();
    el.value = text;
    el.dispatchEvent(new InputEvent('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    // Resize if needed
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
    return true;
  }

  static _injectReact(el, text) {
    // Override native value setter to bypass React controlled state
    el.focus();
    const proto = el.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (nativeSetter) {
      nativeSetter.call(el, text);
    } else {
      el.value = text;
    }
    el.dispatchEvent(new InputEvent('input', { bubbles: true, data: text }));
    return true;
  }

  static _injectTB(text) {
    // Route through background.js for TB-native handling
    chrome.runtime.sendMessage({ type: 'TB_INJECT_PROMPT', text });
    return true;
  }

  // Append instead of replace
  static append(selector, text, method) {
    const el = this._find(selector);
    if (!el) return false;
    const existing = el.value || el.innerText || el.textContent || '';
    const separator = existing.trim() ? '\n\n' : '';
    return this.inject(selector, existing + separator + text, method);
  }
}

// Export for popup and content scripts
if (typeof module !== 'undefined') {
  module.exports = { PromptEngine, SiteInjector };
} else {
  window.PromptEngine = PromptEngine;
  window.SiteInjector = SiteInjector;
}
