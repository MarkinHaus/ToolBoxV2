// content/injector.js — TB Extension AI Site Injector
// Runs on all matched AI sites via manifest content_scripts

(function () {
  'use strict';
  if (window.__tbInjectorLoaded) return;
  window.__tbInjectorLoaded = true;

  let currentRule = null;
  let panelEl = null;

  // Init: ask background for current URL rule
  chrome.runtime.sendMessage(
    { type: 'MATCH_SITE', url: location.href },
    (rule) => {
      if (!rule) return;
      currentRule = rule;
      waitForEditor(rule);
    }
  );

  // Keyboard shortcut Alt+P triggers panel toggle
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'TB_TOGGLE_PANEL' && currentRule) {
      toggleInlinePanel(currentRule);
    }
  });

  // Wait for the editor element to appear (SPAs load async)
  function waitForEditor(rule) {
    const selectors = rule.inject_selector.split(',').map(s => s.trim());
    let found = selectors.some(s => document.querySelector(s));
    if (found) {
      onEditorReady(rule);
      return;
    }
    const observer = new MutationObserver(() => {
      found = selectors.some(s => document.querySelector(s));
      if (found) {
        observer.disconnect();
        onEditorReady(rule);
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    // Timeout after 10s
    setTimeout(() => observer.disconnect(), 10000);
  }

  function onEditorReady(rule) {
    injectTBButton(rule);
    if (rule.auto_show_panel) {
      // Notify background: show badge
      chrome.runtime.sendMessage({ type: 'AI_SITE_DETECTED', rule });
    }
  }

  // Inject a small floating button near the textarea
  function injectTBButton(rule) {
    if (document.getElementById('tb-prompt-btn')) return;

    const btn = document.createElement('button');
    btn.id = 'tb-prompt-btn';
    btn.title = 'TB Prompt Library';
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/>
      <line x1="9" y1="9" x2="9" y2="21"/>
    </svg>`;
    Object.assign(btn.style, {
      position: 'fixed', bottom: '20px', right: '20px', zIndex: '99999',
      width: '36px', height: '36px', borderRadius: '8px',
      background: '#1a1a2e', border: '1px solid #4a4a8f',
      color: '#a0a0ff', cursor: 'pointer', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
      boxShadow: '0 2px 12px rgba(0,0,0,0.4)', transition: 'all 0.15s'
    });

    btn.addEventListener('mouseenter', () => { btn.style.background = '#2a2a4e'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = '#1a1a2e'; });
    btn.addEventListener('click', () => toggleInlinePanel(rule));

    document.body.appendChild(btn);
  }

  // Inline panel (alternative to opening popup)
  function toggleInlinePanel(rule) {
    if (panelEl) {
      panelEl.remove();
      panelEl = null;
      return;
    }
    chrome.runtime.sendMessage(
      { type: 'GET_PROMPTS_FOR_SITE', url: location.href },
      (data) => {
        if (!data) return;
        panelEl = buildPanel(rule, data.prompts, data.preprompts);
        document.body.appendChild(panelEl);
      }
    );
  }

  function buildPanel(rule, prompts, preprompts) {
    const panel = document.createElement('div');
    panel.id = 'tb-prompt-panel';
    Object.assign(panel.style, {
      position: 'fixed', bottom: '64px', right: '20px', zIndex: '99999',
      width: '300px', maxHeight: '420px',
      background: '#0f0f1a', border: '1px solid #3a3a6a',
      borderRadius: '12px', overflow: 'hidden',
      boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      fontFamily: 'system-ui, sans-serif', fontSize: '13px', color: '#d0d0f0'
    });

    panel.innerHTML = `
      <div style="padding:10px 12px;background:#1a1a2e;border-bottom:1px solid #3a3a6a;display:flex;align-items:center;gap:8px">
        <span>${rule.icon || '⚡'} ${rule.label}</span>
        <span style="margin-left:auto;font-size:11px;color:#6060aa">${rule.model_hint || ''}</span>
        <button id="tb-panel-close" style="background:none;border:none;color:#8080cc;cursor:pointer;font-size:16px">×</button>
      </div>
      <div style="padding:8px">
        <input id="tb-panel-search" placeholder="Suche..." style="width:100%;box-sizing:border-box;background:#1a1a2e;border:1px solid #3a3a6a;border-radius:6px;padding:5px 8px;color:#d0d0f0;font-size:12px;outline:none"/>
      </div>
      <div id="tb-panel-list" style="overflow-y:auto;max-height:300px;padding:0 8px 8px">
        ${prompts.map(p => `
          <div class="tb-prompt-item" data-id="${p.id}" style="padding:7px 10px;margin-bottom:4px;background:#1a1a2e;border-radius:6px;cursor:pointer;border:1px solid transparent;display:flex;align-items:center;gap:8px">
            <div style="flex:1">
              <div style="font-weight:500">${p.pinned ? '★ ' : ''}${p.label}</div>
              ${p.description ? `<div style="font-size:11px;color:#6060aa;margin-top:2px">${p.description}</div>` : ''}
            </div>
            <button class="tb-inject-btn" data-id="${p.id}" style="background:#2a2a6a;border:none;border-radius:4px;color:#a0a0ff;padding:3px 8px;cursor:pointer;font-size:11px;white-space:nowrap">→ Inject</button>
          </div>
        `).join('')}
      </div>
    `;

    // Events
    panel.querySelector('#tb-panel-close').addEventListener('click', () => {
      panel.remove(); panelEl = null;
    });

    panel.querySelector('#tb-panel-search').addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      panel.querySelectorAll('.tb-prompt-item').forEach(item => {
        const label = item.querySelector('div').innerText.toLowerCase();
        item.style.display = label.includes(q) ? '' : 'none';
      });
    });

    panel.querySelectorAll('.tb-inject-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const promptId = btn.dataset.id;
        chrome.runtime.sendMessage(
          { type: 'BUILD_INJECT', promptId, url: location.href },
          (content) => {
            if (!content) return;
            injectIntoEditor(rule, content);
            panel.remove(); panelEl = null;
          }
        );
      });
    });

    // Hover effects
    panel.querySelectorAll('.tb-prompt-item').forEach(item => {
      item.addEventListener('mouseenter', () => { item.style.borderColor = '#4a4a8f'; });
      item.addEventListener('mouseleave', () => { item.style.borderColor = 'transparent'; });
    });

    return panel;
  }

  function injectIntoEditor(rule, text) {
    const selectors = rule.inject_selector.split(',').map(s => s.trim());
    let el = null;
    for (const s of selectors) { el = document.querySelector(s); if (el) break; }
    if (!el) return;

    switch (rule.inject_method) {
      case 'prosemirror':
        el.focus();
        document.execCommand('selectAll');
        document.execCommand('insertText', false, text);
        el.dispatchEvent(new InputEvent('input', { bubbles: true }));
        break;

      case 'react': {
        el.focus();
        const proto = el.tagName === 'TEXTAREA'
          ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
        (setter || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value')?.set)
          ?.call(el, text);
        el.dispatchEvent(new InputEvent('input', { bubbles: true, data: text }));
        break;
      }

      case 'textarea':
      default:
        el.focus();
        el.value = text;
        el.dispatchEvent(new InputEvent('input', { bubbles: true }));
        el.style.height = 'auto';
        el.style.height = el.scrollHeight + 'px';
    }
  }

})();
