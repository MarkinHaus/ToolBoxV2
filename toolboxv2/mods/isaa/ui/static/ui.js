/**
 * ISAA UI primitives — toast, confirm, prompt.
 * Replaces native window.alert/confirm/prompt with TBJS-Glass styled dialogs.
 */
(function () {
  'use strict';

  function escape(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  }

  // ---------- Toast ----------
  function ensureToastHost() {
    let host = document.getElementById('toast-host');
    if (!host) {
      host = document.createElement('div');
      host.id = 'toast-host';
      host.className = 'toast-host';
      document.body.appendChild(host);
    }
    return host;
  }

  function toast(message, level = 'info', duration = 3500) {
    const host = ensureToastHost();
    const el = document.createElement('div');
    el.className = 'toast';
    el.dataset.level = level;
    el.innerHTML = `<span class="toast-dot"></span><span class="toast-msg">${escape(message)}</span>`;
    host.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    const close = () => {
      el.classList.remove('show');
      setTimeout(() => el.remove(), 250);
    };
    el.addEventListener('click', close);
    if (duration > 0) setTimeout(close, duration);
    return close;
  }

  // ---------- Confirm ----------
  function confirm(message, { okLabel = 'OK', cancelLabel = 'Abbrechen', danger = false } = {}) {
    return new Promise((resolve) => {
      const backdrop = document.createElement('div');
      backdrop.className = 'dlg-backdrop';
      backdrop.innerHTML = `
        <div class="dlg">
          <div class="dlg-msg">${escape(message)}</div>
          <div class="dlg-actions">
            <button class="btn-secondary" data-act="cancel">${escape(cancelLabel)}</button>
            <button class="${danger ? 'btn-danger' : 'btn-primary'}" data-act="ok">${escape(okLabel)}</button>
          </div>
        </div>`;
      document.body.appendChild(backdrop);
      requestAnimationFrame(() => backdrop.classList.add('show'));
      const done = (val) => {
        backdrop.classList.remove('show');
        setTimeout(() => backdrop.remove(), 200);
        resolve(val);
      };
      backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) done(false);
        const act = e.target.dataset.act;
        if (act === 'ok') done(true);
        if (act === 'cancel') done(false);
      });
      const onKey = (e) => {
        if (e.key === 'Escape') { done(false); document.removeEventListener('keydown', onKey); }
        if (e.key === 'Enter') { done(true); document.removeEventListener('keydown', onKey); }
      };
      document.addEventListener('keydown', onKey);
    });
  }

  // ---------- Prompt ----------
  function prompt(message, defaultValue = '', { multiline = false, okLabel = 'OK' } = {}) {
    return new Promise((resolve) => {
      const backdrop = document.createElement('div');
      backdrop.className = 'dlg-backdrop';
      const field = multiline
        ? `<textarea class="dlg-input" rows="4">${escape(defaultValue)}</textarea>`
        : `<input class="dlg-input" value="${escape(defaultValue)}">`;
      backdrop.innerHTML = `
        <div class="dlg">
          <div class="dlg-msg">${escape(message)}</div>
          ${field}
          <div class="dlg-actions">
            <button class="btn-secondary" data-act="cancel">Abbrechen</button>
            <button class="btn-primary" data-act="ok">${escape(okLabel)}</button>
          </div>
        </div>`;
      document.body.appendChild(backdrop);
      const input = backdrop.querySelector('.dlg-input');
      requestAnimationFrame(() => { backdrop.classList.add('show'); input.focus(); input.select && input.select(); });
      const done = (val) => {
        backdrop.classList.remove('show');
        setTimeout(() => backdrop.remove(), 200);
        resolve(val);
      };
      backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) done(null);
        const act = e.target.dataset.act;
        if (act === 'ok') done(input.value);
        if (act === 'cancel') done(null);
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !multiline) { e.preventDefault(); done(input.value); }
        if (e.key === 'Escape') done(null);
      });
    });
  }

  // ---------- Worker status (sys.topology) ----------
  function updateWorkerStatus(topology) {
    let el = document.getElementById('worker-status');
    if (!el) {
      el = document.createElement('div');
      el.id = 'worker-status';
      el.className = 'worker-status';
      document.body.appendChild(el);
      el.addEventListener('click', () => el.classList.toggle('expanded'));
    }
    const workers = (topology && topology.workers) || {};
    const names = Object.keys(workers);
    const leader = topology && topology.leader;
    const allUp = names.length > 0;
    el.dataset.up = allUp ? 'true' : 'false';
    const summary = `${names.length} worker${names.length === 1 ? '' : 's'}`;
    const detail = names.map(n => {
      const w = workers[n];
      const isLeader = n === leader;
      return `<div class="ws-row">${isLeader ? '★ ' : ''}${escape(n)} · ${escape(w.worker_type || '?')} · ${Math.round(w.memory_mb || 0)}MB</div>`;
    }).join('');
    el.innerHTML = `<span class="ws-dot"></span><span class="ws-sum">${escape(summary)}</span><div class="ws-detail">${detail}</div>`;
  }

  window.ISAA = window.ISAA || {};
  window.ISAA.UI = { toast, confirm, prompt, updateWorkerStatus };
})();
