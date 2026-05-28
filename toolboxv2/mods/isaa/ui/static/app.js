/**
 * ISAA App — main controller.
 *
 * Views: 'welcome' (no active chat) | 'chat' (active chat).
 */
(function () {
  'use strict';

  const Store = window.ISAA.Store;
  const WS = window.ISAA.WS;
  const Chat = window.ISAA.Chat;

  async function get(p) {
    const r = await fetch(p, { credentials: 'same-origin' });
    if (!r.ok) throw new Error(`${r.status} ${p}`);
    return r.json();
  }
  async function post(p, body) {
    const r = await fetch(p, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${p}`);
    return r.json();
  }

  // ============================================================================
  // VIEW: WELCOME
  // ============================================================================

  async function renderWelcome() {
    const view = document.getElementById('view');
    document.querySelector('.isaa-body').dataset.view = 'welcome';
    let data;
    try { data = await get('/api/welcome'); } catch (_) { data = { greeting: 'Willkommen', time_interval: '' }; }
    const logo = document.getElementById('tpl-logo').content.cloneNode(true);
    view.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'welcome';
    wrap.appendChild(logo);
    wrap.insertAdjacentHTML('beforeend', `
      <div class="greeting">${escape(data.greeting || 'Willkommen')}</div>
      <div class="greeting-sub">${escape(data.time_interval || '')}</div>
    `);
    view.appendChild(wrap);

    // Pre-create default agent state
    Store.agents = (data.agents || []).map(name => ({ name }));
    Store.defaultAgent = data.default_agent || 'self';
  }

  // ============================================================================
  // VIEW: CHAT
  // ============================================================================

  function renderChat() {
    const view = document.getElementById('view');
    document.querySelector('.isaa-body').dataset.view = 'chat';
    let container = view.querySelector('.chat-container');
    if (!container) {
      view.innerHTML = '<div class="chat-container"></div>';
      container = view.querySelector('.chat-container');
      Chat.bindEvents(container);
      // Persist scroll on user scroll
      view.addEventListener('scroll', () => {
        Store.setScroll(view.scrollTop);
      }, { passive: true });
    }
    const wasAtBottom = view.scrollHeight - view.scrollTop - view.clientHeight < 50;
    if (!Store.frames.length) {
      // Empty active chat — show a placeholder so user sees the chat is alive
      container.innerHTML = `
        <div style="text-align:center;padding-top:14vh;color:var(--text-muted, rgba(255,255,255,0.5))">
          <div style="font-family:var(--font-mono, monospace);font-size:var(--text-xs, 9px);text-transform:uppercase;letter-spacing:2px">
            ${escape(Store.chatMeta?.title || 'New chat')} · ${escape(Store.chatMeta?.agent || '?')}
          </div>
          <div style="margin-top:var(--space-3, 12px);font-size:var(--text-sm, 11px)">Frag los — der Agent hört zu.</div>
          <div id="ws-status" style="margin-top:var(--space-5, 24px);font-family:var(--font-mono, monospace);font-size:var(--text-xs, 9px);opacity:0.5"></div>
        </div>
      `;
    } else {
      Chat.render(container);
      // Auto-scroll to bottom if we WERE at bottom or it's first render; else preserve.
      const view2 = document.getElementById('view');
      const savedScroll = Store.getScroll();
      if (wasAtBottom || !savedScroll) {
        Chat.scrollToBottom(container);
      } else {
        view2.scrollTop = savedScroll;
      }
    }
    const cancelBtn = document.getElementById('btn-cancel');
    const sendBtn = document.getElementById('btn-send');
    if (Store.isRunning) {
      cancelBtn.hidden = false;
      sendBtn.hidden = true;
      document.body.classList.add('is-running');
    } else {
      cancelBtn.hidden = true;
      sendBtn.hidden = false;
      document.body.classList.remove('is-running');
    }
  }

  function setWsStatus(text, color) {
    const el = document.getElementById('ws-status');
    if (!el) return;
    el.textContent = text;
    if (color) el.style.color = color;
  }

  // ============================================================================
  // CHAT MGMT
  // ============================================================================

  async function newChat() {
    const data = await post('/api/chats', { agent: Store.defaultAgent || 'self' });
    Store.setActiveChat(data.chat_id);
    await openChat(data.chat_id);
  }

  async function openChat(chatId) {
    let meta;
    try { meta = await get(`/api/chats/${chatId}`); }
    catch (e) { console.error('openChat failed', e); return; }
    Store.setActiveChat(chatId);
    Store.chatMeta = {
      chat_id: meta.chat_id, title: meta.title, agent: meta.agent,
      session_id: meta.session_id, run_id: meta.run_id, ui: meta.ui,
    };
    Store.resetFrames(meta.messages || []);
    Store.isRunning = !!meta.is_running;

    // Mount WS to this chat
    WS.close();
    WS.setHandlers({
      open: () => { console.log('[ws] open chat=', chatId); setWsStatus('● connected', 'var(--success)'); },
      close: () => { console.log('[ws] close chat=', chatId); setWsStatus('○ disconnected, reconnecting…', 'var(--warning)'); },
      frame: (f) => {
        // ---- System frames from the worker layer: handle, do NOT log to chat ----
        if (f.type === 'connected') {
          console.log('[ws] connected conn_id=', f.conn_id);
          setWsStatus('● connected', 'var(--success)');
          return;
        }
        if (f.type === 'sys.topology') {
          if (window.ISAA.UI) window.ISAA.UI.updateWorkerStatus(f.data || {});
          return;
        }
        if (f.type === 'pong' || f.type === 'resync_done') {
          return;
        }
        if (f.type === 'notification') {
          const d = f.data || {};
          if (window.ISAA.UI) window.ISAA.UI.toast(`${d.title ? d.title + ': ' : ''}${d.content || ''}`, d.level || 'info');
          return;
        }
        // ---- App frames ----
        if (f.type === 'rollback_done') {
          get(`/api/chats/${chatId}`).then(m => {
            Store.resetFrames(m.messages || []);
            renderChat();
          });
          return;
        }
        if (f.type === 'error') {
          // Auth/protocol errors are not chat content — surface as status + toast
          console.error('[ws] error frame:', f.message || f.error);
          const msg = f.message || f.error || 'error';
          setWsStatus('✕ ' + msg, 'var(--error)');
          if (f.code === 'ACCESS_DENIED' && window.ISAA.UI) {
            window.ISAA.UI.toast('WS Auth fehlgeschlagen: ' + msg, 'error', 6000);
          }
          return;
        }
        // Widget frames: route to Widgets module (task 18-23)
        if (window.ISAA.Widgets && window.ISAA.Widgets.handleFrame(f)) {
          Store.pushFrame(f);
          return;
        }
        Store.pushFrame(f);
        // Don't rebuild the chat view while in widget mode — it would wipe the grid.
        if (document.querySelector('.isaa-body').dataset.view !== 'widgets') renderChat();
      },
    });
    WS.connect(chatId);

    // Fix 1: switch to chat view (and play the slide animation) only once the
    // chat actually has messages — a freshly created/empty chat stays on Welcome
    // until the first message is sent from the text input.
    if (Store.frames.length || Store.isRunning) {
      renderChat();
    } else {
      await renderWelcome();
    }
  }

  // ============================================================================
  // SEND
  // ============================================================================

  function send() {
    const ta = document.getElementById('prompt');
    const text = ta.value.trim();
    const atts = Store.attachments.slice();
    if (!text && !atts.length) return;
    ta.value = '';
    Store.clearDraft();
    Store.clearAttachments();
    renderAttachments();
    autoresize(ta);
    (async () => {
      if (!Store.activeChatId) await newChat();
      const attMapped = atts.map(a => ({ upload_id: a.upload_id, vfs_path: a.vfs_path, name: a.name }));
      // Immediate feedback: optimistic user bubble + thinking state (Fix 3),
      // and switch to chat view → plays the welcome→chat animation (Fix 1).
      Store.isRunning = true;
      Store.pushFrame({ type: 'user_msg', text, attachments: attMapped, _optimistic: true });
      renderChat();
      // Reliable delivery even if the socket is still connecting (first message).
      WS.sendWhenReady({ op: 'send', text, attachments: attMapped.map(a => ({ upload_id: a.upload_id, vfs_path: a.vfs_path })) });
    })();
  }
  function cancel() {
    WS.send({ op: 'cancel' });
  }

  // ============================================================================
  // ATTACHMENTS
  // ============================================================================

  function renderAttachments() {
    const host = document.getElementById('attachments');
    host.innerHTML = Store.attachments.map(a =>
      `<span class="attachment-chip" data-id="${escape(a.upload_id)}">📎 ${escape(a.name)} <button data-rm>✕</button></span>`
    ).join('');
    host.querySelectorAll('.attachment-chip').forEach(chip => {
      chip.querySelector('[data-rm]').addEventListener('click', (e) => {
        e.stopPropagation();
        Store.removeAttachment(chip.dataset.id);
      });
    });
  }

  async function uploadFile(file) {
    if (!Store.activeChatId) await newChat();
    const fd = new FormData();
    fd.append('session_id', Store.activeChatId);
    fd.append('file', file);
    fd.append('path', '/uploads');
    let r;
    try {
      r = await fetch('/api/vfs/upload', { method: 'POST', credentials: 'same-origin', body: fd });
    } catch (e) { console.error('upload failed', e); return; }
    if (!r.ok) { console.error('upload status', r.status); return; }
    const data = await r.json();
    if (data.ok && data.upload_id) {
      Store.addAttachment({ upload_id: data.upload_id, vfs_path: data.vfs_path, name: file.name });
    }
  }

  // ============================================================================
  // INPUT BAR
  // ============================================================================

  function autoresize(ta) {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }

  function bindInput() {
    const ta = document.getElementById('prompt');
    const bubble = document.getElementById('bubble');
    const sendBtn = document.getElementById('btn-send');
    const cancelBtn = document.getElementById('btn-cancel');

    ta.addEventListener('input', () => {
      autoresize(ta);
      Store.setDraft(ta.value);
    });
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
    sendBtn.addEventListener('click', send);
    cancelBtn.addEventListener('click', cancel);

    // Restore draft
    ta.value = Store.getDraft();
    autoresize(ta);

    // Drag-drop on bubble
    bubble.addEventListener('dragover', (e) => { e.preventDefault(); bubble.classList.add('drag-over'); });
    bubble.addEventListener('dragleave', () => { bubble.classList.remove('drag-over'); });
    bubble.addEventListener('drop', async (e) => {
      e.preventDefault();
      bubble.classList.remove('drag-over');
      const files = Array.from(e.dataTransfer.files || []);
      for (const f of files) await uploadFile(f);
    });
  }

  // ============================================================================
  // STORE LISTENERS
  // ============================================================================

  function bindStore() {
    Store.on('attachments', renderAttachments);
    Store.on('step:toggled', () => {
      if (Store.activeChatId) renderChat();
    });
    Store.on('step:l2', () => {
      if (Store.activeChatId) renderChat();
    });
    // Auto-collapse sidebar when interacting with the main area
    const main = document.getElementById('main');
    if (main) {
      main.addEventListener('mousedown', () => {
        if (window.innerWidth <= 767) return;  // mobile uses overlay; don't auto-close
        if (Store.sidebarPanel) {
          window.ISAA.Sidebar.openPanel(null);
        }
      });
    }
  }

  // ============================================================================
  // INIT
  // ============================================================================

  function escape(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  }

  async function init() {
    bindInput();
    bindStore();
    window.ISAA.Sidebar.bind();
    startFooterPoll();
    bindTheme();
    bindMobileKeyboard();

    if (Store.activeChatId) {
      try {
        await openChat(Store.activeChatId);
      } catch (_) {
        Store.setActiveChat(null);
        await renderWelcome();
      }
    } else {
      await renderWelcome();
    }
  }

  // ============================================================================
  // MOBILE KEYBOARD HANDLING (task 28)
  // ============================================================================

  function bindMobileKeyboard() {
    // visualViewport: when keyboard appears on iOS/Android, adjust input bar
    if (!window.visualViewport) return;
    const inputBar = document.getElementById('input-bar');
    const update = () => {
      if (window.innerWidth > 767) return;  // desktop unaffected
      const vv = window.visualViewport;
      const keyboardOffset = window.innerHeight - vv.height - vv.offsetTop;
      if (keyboardOffset > 50) {
        // Keyboard is up — push input above it, hide mobile nav
        inputBar.style.bottom = `${keyboardOffset + 8}px`;
        document.getElementById('mobile-nav').style.transform = 'translateY(100%)';
      } else {
        inputBar.style.bottom = '';
        document.getElementById('mobile-nav').style.transform = '';
      }
    };
    window.visualViewport.addEventListener('resize', update);
    window.visualViewport.addEventListener('scroll', update);
  }

  // ============================================================================
  // RUNNING FOOTER POLL (task 15)
  // ============================================================================

  let _footerTimer = null;
  function startFooterPoll() {
    const poll = async () => {
      try {
        const items = await get('/api/running');
        renderFooter(items);
      } catch (_) { /* ignore */ }
    };
    poll();
    _footerTimer = setInterval(poll, 1500);
  }

  function renderFooter(items) {
    const el = document.getElementById('running-footer');
    if (!el) return;
    // Filter: only show footer if MORE than 1 (or any sub-agent active)
    const subagents = items.filter(i => i.kind === 'subagent');
    const shouldShow = items.length > 1 || subagents.length > 0;
    if (!shouldShow) {
      el.hidden = true;
      return;
    }
    el.hidden = false;
    el.innerHTML = items.map(i => {
      const label = i.kind === 'subagent' ? i.name || i.agent : i.agent;
      const meta = `iter ${i.iteration || 0}${i.context_pct ? ' · ' + Math.round(i.context_pct) + '%' : ''}`;
      const tooltip = i.last_thought ? `title="${escape(i.last_thought.slice(0, 400))}"` : '';
      return `<div class="rf-item" data-kind="${escape(i.kind)}" ${tooltip}>
        <span class="rf-dot"></span>
        <span class="rf-agent">${escape(label)}</span>
        <span>${escape(meta)}</span>
      </div>`;
    }).join('');
  }

  // ============================================================================
  // THEME TOGGLE (task 26)
  // ============================================================================

  function bindTheme() {
    // Restore saved
    const saved = localStorage.getItem('isaa.theme');
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.dataset.theme = saved;
    }
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + J toggles theme
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault();
        const cur = document.documentElement.dataset.theme || 'dark';
        const next = cur === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        localStorage.setItem('isaa.theme', next);
      }
      // Ctrl/Cmd + B toggles sidebar (close if open)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        const cur = Store.sidebarPanel;
        window.ISAA.Sidebar.openPanel(cur ? null : 'chats');
      }
      // Ctrl/Cmd + K opens quick-switcher
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openQuickSwitch();
      }
      // Escape closes quick-switcher
      if (e.key === 'Escape') {
        closeQuickSwitch();
      }
    });
  }

  // ============================================================================
  // QUICK SWITCHER (task 27)
  // ============================================================================

  async function openQuickSwitch() {
    let modal = document.getElementById('qs-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'qs-modal';
      modal.className = 'qs-modal-backdrop';
      modal.innerHTML = `
        <div class="qs-modal">
          <input id="qs-input" placeholder="Suche Chats… (Enter zum öffnen, Esc schließt)" autofocus>
          <ul id="qs-list"></ul>
        </div>
      `;
      document.body.appendChild(modal);
      modal.addEventListener('click', (e) => { if (e.target === modal) closeQuickSwitch(); });
    }
    let chats = [];
    try { chats = await get('/api/chats'); } catch (_) { chats = []; }
    const input = modal.querySelector('#qs-input');
    const list = modal.querySelector('#qs-list');
    let sel = 0;
    const render = (q = '') => {
      const filtered = chats.filter(c =>
        (c.title || '').toLowerCase().includes(q.toLowerCase()) ||
        (c.agent || '').toLowerCase().includes(q.toLowerCase())
      ).slice(0, 12);
      list.innerHTML = filtered.map((c, i) => `
        <li data-id="${escape(c.chat_id)}" data-sel="${i === sel ? 'true' : 'false'}">
          <span>${escape(c.title || 'Untitled')}</span>
          <span class="qs-meta">${escape(c.agent || '')}</span>
        </li>
      `).join('') || '<li style="opacity:0.5;cursor:default"><em>keine Treffer</em></li>';
      list.querySelectorAll('li[data-id]').forEach((li, i) => {
        li.addEventListener('click', () => { closeQuickSwitch(); window.ISAA.App.openChat(li.dataset.id); });
      });
      return filtered;
    };
    let filtered = render();
    input.value = '';
    input.focus();
    modal.dataset.open = 'true';
    input.onkeydown = (e) => {
      if (e.key === 'Enter') {
        const items = list.querySelectorAll('li[data-id]');
        if (items[sel]) { closeQuickSwitch(); window.ISAA.App.openChat(items[sel].dataset.id); }
      } else if (e.key === 'ArrowDown') {
        sel = Math.min(sel + 1, filtered.length - 1);
        filtered = render(input.value);
      } else if (e.key === 'ArrowUp') {
        sel = Math.max(sel - 1, 0);
        filtered = render(input.value);
      }
    };
    input.oninput = () => { sel = 0; filtered = render(input.value); };
  }

  function closeQuickSwitch() {
    const modal = document.getElementById('qs-modal');
    if (modal) modal.dataset.open = 'false';
  }

  window.ISAA = window.ISAA || {};
  window.ISAA.App = { init, newChat, openChat, rerender: renderChat };

  document.addEventListener('DOMContentLoaded', init);
})();
