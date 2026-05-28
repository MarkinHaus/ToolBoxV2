/**
 * Sidebar — 5 panels: new-chat (direct), chats, vfs, agent, skills.
 */
(function () {
  'use strict';

  const Store = window.ISAA.Store;

  // ============================================================================
  // FETCH HELPERS
  // ============================================================================

  async function get(path) {
    const r = await fetch(path, { credentials: 'same-origin' });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  }
  async function post(path, body) {
    const r = await fetch(path, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  }
  async function put(path, body) {
    const r = await fetch(path, {
      method: 'PUT', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  }
  async function del(path) {
    const r = await fetch(path, { method: 'DELETE', credentials: 'same-origin' });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  }

  function escape(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[c]);
  }

  // ============================================================================
  // PANELS
  // ============================================================================

  async function renderChats(panel) {
    panel.innerHTML = '<h3 class="sb-panel-title">Chats</h3><ul class="sb-list" id="chat-list"><li>laden…</li></ul>';
    const list = panel.querySelector('#chat-list');
    let data;
    try { data = await get('/api/chats'); }
    catch (e) { list.innerHTML = `<li><em>Fehler: ${escape(e.message)}</em></li>`; return; }
    Store.chats = data;
    if (!data.length) {
      list.innerHTML = '<li><em>noch keine Chats</em></li>';
      return;
    }
    list.innerHTML = data.map(c => `
      <li data-chat-id="${escape(c.chat_id)}" data-active="${c.chat_id === Store.activeChatId ? 'true' : 'false'}">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1">${escape(c.title || 'Untitled')}</span>
        <span class="li-meta">${escape((c.agent || '').slice(0, 10))}</span>
        <button class="li-del" data-action="del" title="Löschen">✕</button>
      </li>
    `).join('');
    list.querySelectorAll('li').forEach(li => {
      li.addEventListener('click', async (e) => {
        const action = e.target.dataset.action;
        if (action === 'del') {
          e.stopPropagation();
          if (!await window.ISAA.UI.confirm('Chat löschen?', { danger: true, okLabel: 'Löschen' })) return;
          del(`/api/chats/${li.dataset.chatId}`).then(() => renderChats(panel));
          return;
        }
        window.ISAA.App.openChat(li.dataset.chatId);
      });
    });
  }

  async function renderVfs(panel) {
    if (!Store.activeChatId) {
      panel.innerHTML = '<h3 class="sb-panel-title">VFS</h3><p><em>Erst Chat öffnen</em></p>';
      return;
    }
    panel.innerHTML = `
      <h3 class="sb-panel-title">VFS</h3>
      <div style="display:flex;gap:4px;margin-bottom:8px">
        <button class="btn-secondary" id="vfs-refresh" style="flex:1">Reload</button>
        <button class="btn-secondary" id="vfs-zip" title="ZIP download">⬇ ZIP</button>
      </div>
      <ul class="vfs-tree" id="vfs-tree"><li><em>laden…</em></li></ul>
    `;
    const tree = panel.querySelector('#vfs-tree');
    const collapsed = new Set();
    const applyVis = () => {
      let hideBelow = null;
      tree.querySelectorAll('li[data-depth]').forEach(li => {
        const depth = +li.dataset.depth;
        if (hideBelow !== null && depth > hideBelow) { li.style.display = 'none'; return; }
        hideBelow = null;
        li.style.display = '';
        if (li.dataset.type === 'directory') {
          const isCol = collapsed.has(li.dataset.path);
          const chev = li.querySelector('.vfs-chev');
          if (chev) chev.textContent = isCol ? '▸' : '▾';
          if (isCol) hideBelow = depth;
        }
      });
    };
    const load = async () => {
      tree.innerHTML = '<li><em>laden…</em></li>';
      let data;
      try {
        data = await get(`/api/vfs/tree?session_id=${encodeURIComponent(Store.activeChatId)}&recursive=true`);
      } catch (e) { tree.innerHTML = `<li><em>Fehler: ${escape(e.message)}</em></li>`; return; }
      if (!data.success) { tree.innerHTML = `<li><em>${escape(data.error || 'Fehler')}</em></li>`; return; }
      const items = data.contents || [];
      if (!items.length) { tree.innerHTML = '<li><em>leer</em></li>'; return; }
      tree.innerHTML = items.map(item => {
        const indent = (item.depth || 0) * 12;
        const isDir = item.type === 'directory';
        const marker = isDir ? '<span class="vfs-chev">▾</span>' : '<span class="vfs-bullet">·</span>';
        return `<li class="${isDir ? 'vfs-dir' : ''}" data-path="${escape(item.path)}" data-type="${escape(item.type)}" data-depth="${item.depth || 0}" style="padding-left:${indent + 8}px">
          ${marker} ${escape(item.name)}
        </li>`;
      }).join('');
      tree.querySelectorAll('li').forEach(li => {
        li.addEventListener('click', () => {
          if (li.dataset.type === 'directory') {
            if (collapsed.has(li.dataset.path)) collapsed.delete(li.dataset.path);
            else collapsed.add(li.dataset.path);
            applyVis();
            return;
          }
          if (li.dataset.type !== 'file') return;
          openVfsFile(li.dataset.path);
        });
      });
      applyVis();
    };
    panel.querySelector('#vfs-refresh').addEventListener('click', load);
    panel.querySelector('#vfs-zip').addEventListener('click', () => {
      window.open(`/api/vfs/download_zip?session_id=${encodeURIComponent(Store.activeChatId)}&path=/`, '_blank');
    });
    load();
  }

  async function openVfsFile(path) {
    let res;
    try {
      res = await get(`/api/vfs/file?session_id=${encodeURIComponent(Store.activeChatId)}&path=${encodeURIComponent(path)}`);
    } catch (e) { window.ISAA.UI.toast('Fehler: ' + e.message, 'error'); return; }
    if (!res.success) { window.ISAA.UI.toast(res.error || 'Fehler', 'error'); return; }
    showFileModal(path, res.content || '');
  }

  function showFileModal(path, content) {
    let modal = document.getElementById('vfs-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'vfs-modal';
      modal.className = 'isaa-modal-backdrop';
      modal.innerHTML = `
        <div class="isaa-modal">
          <div class="isaa-modal-title" id="m-title"></div>
          <div class="isaa-modal-body"><textarea id="m-text" spellcheck="false"></textarea></div>
          <div class="isaa-modal-footer">
            <button class="btn-secondary" data-action="close">Schließen</button>
            <button class="btn-secondary" data-action="download">⬇ Download</button>
            <button class="btn-primary" data-action="save">Speichern</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.dataset.open = 'false';
        const a = e.target.dataset.action;
        const text = modal.querySelector('#m-text').value;
        const cur = modal.dataset.path;
        if (a === 'close') modal.dataset.open = 'false';
        else if (a === 'save') {
          put('/api/vfs/file', { session_id: Store.activeChatId, path: cur, content: text })
            .then(() => { modal.dataset.open = 'false'; });
        } else if (a === 'download') {
          window.open(`/api/vfs/download?session_id=${encodeURIComponent(Store.activeChatId)}&path=${encodeURIComponent(cur)}`, '_blank');
        }
      });
    }
    modal.dataset.path = path;
    modal.dataset.open = 'true';
    modal.querySelector('#m-title').textContent = path;
    modal.querySelector('#m-text').value = content;
  }

  async function renderAgent(panel) {
    panel.innerHTML = '<h3 class="sb-panel-title">Agent Config</h3><div id="agent-form"><em>laden…</em></div>';
    const wrap = panel.querySelector('#agent-form');
    const agents = await get('/api/agents');
    Store.agents = agents;
    if (!agents.length) { wrap.innerHTML = '<em>Keine Agents</em>'; return; }
    const activeName = Store.chatMeta ? Store.chatMeta.agent : (agents[0]?.name);
    wrap.innerHTML = `
      <div class="cfg-form">
        <div>
          <label>Agent</label>
          <select id="cfg-agent">${agents.map(a => `<option value="${escape(a.name)}" ${a.name === activeName ? 'selected' : ''}>${escape(a.name)}${a.is_running ? ' ●' : ''}</option>`).join('')}</select>
        </div>
        <div id="cfg-fields"><em>laden…</em></div>
      </div>
    `;
    const sel = panel.querySelector('#cfg-agent');
    const fieldsHost = panel.querySelector('#cfg-fields');
    const loadFor = async (name) => {
      fieldsHost.innerHTML = '<em>laden…</em>';
      let cfg;
      try { cfg = await get(`/api/agents/${encodeURIComponent(name)}/config`); }
      catch (e) { fieldsHost.innerHTML = `<em>Fehler: ${escape(e.message)}</em>`; return; }
      let personas = { active: cfg.active_persona || '', personas: [] };
      try {
        const sid = Store.activeChatId ? `?session_id=${encodeURIComponent(Store.activeChatId)}` : '';
        personas = await get(`/api/agents/${encodeURIComponent(name)}/personas${sid}`);
      } catch (_) {}
      const curPersona = cfg.active_persona || '';
      const pSeen = new Set();
      const pOpts = [`<option value="">— keine —</option>`];
      (personas.personas || []).forEach(p => {
        if (pSeen.has(p.name)) return; pSeen.add(p.name);
        const conf = (p.confidence != null) ? ` · ${Math.round(p.confidence * 100)}%` : '';
        pOpts.push(`<option value="${escape(p.name)}" ${p.name === curPersona ? 'selected' : ''}>${escape(p.name)} (${escape(p.source)}${conf})</option>`);
      });
      if (curPersona && !pSeen.has(curPersona)) pOpts.splice(1, 0, `<option value="${escape(curPersona)}" selected>${escape(curPersona)} (current)</option>`);
      const obs = cfg.obs || {};
      const web = cfg.web_config || {};
      const ctxb = cfg.context_config || {};
      fieldsHost.innerHTML = `
        <div><label>Fast Model</label><input id="f-fast" value="${escape(cfg.fast_llm_model || '')}"></div>
        <div><label>Complex Model</label><input id="f-complex" value="${escape(cfg.complex_llm_model || '')}"></div>
        <div><label>System Message</label><textarea id="f-sys" rows="6">${escape(cfg.system_message || '')}</textarea></div>
        <div><label>Temperature</label><input id="f-temp" type="number" step="0.1" min="0" max="2" value="${cfg.temperature ?? 0.7}"></div>
        <div><label>Max Parallel Tasks</label><input id="f-iter" type="number" value="${cfg.max_parallel_tasks ?? 3}"></div>
        <div><label>Active Persona</label>
          <select id="f-persona-sel">${pOpts.join('')}</select>
          <input id="f-persona" value="${escape(curPersona)}" placeholder="custom persona…" style="margin-top:4px">
        </div>
        <details class="cfg-advanced">
          <summary>Advanced</summary>
          <div><label>Max Tokens Output</label><input id="f-mto" type="number" value="${cfg.max_tokens_output ?? 2048}"></div>
          <div><label>Max Tokens Input</label><input id="f-mti" type="number" value="${cfg.max_tokens_input ?? 32768}"></div>
          <div><label>VFS Window Lines</label><input id="f-vfs" type="number" value="${cfg.vfs_max_window_lines ?? 250}"></div>
          <div><label><input id="f-fast-resp" type="checkbox" ${cfg.use_fast_response !== false ? 'checked' : ''}> Use Fast Response</label></div>
          <div><label><input id="f-stream" type="checkbox" ${cfg.stream !== false ? 'checked' : ''}> Stream</label></div>
          <div><label><input id="f-verbose" type="checkbox" ${cfg.verbose_logging ? 'checked' : ''}> Verbose Logging</label></div>
          <hr>
          <div><label><input id="f-obs-en" type="checkbox" ${obs.enabled !== false ? 'checked' : ''}> Observability</label></div>
          <div><label>Obs Max Runs</label><input id="f-obs-runs" type="number" value="${obs.max_runs ?? 3}"></div>
          <div><label>Obs Snapshot Interval</label><input id="f-obs-snap" type="number" value="${obs.snapshot_interval ?? 5}"></div>
          <div><label><input id="f-obs-audit" type="checkbox" ${obs.enable_audit !== false ? 'checked' : ''}> Enable Audit</label></div>
          <hr>
          <div><label><input id="f-web-en" type="checkbox" ${web.enable_web !== false ? 'checked' : ''}> Web Enabled</label></div>
          <div><label><input id="f-web-head" type="checkbox" ${web.web_headless ? 'checked' : ''}> Web Headless</label></div>
          <div><label>Web Single Site</label><input id="f-web-single" value="${escape(web.web_single_site || '')}"></div>
          <div><label>Web Trusted Sites (komma)</label><input id="f-web-trust" value="${escape((web.web_trusted_sites || []).join(', '))}"></div>
          <hr>
          <div><label>Ctx Max Ratio</label><input id="f-ctx-max" type="number" step="0.05" value="${ctxb.max_context_ratio ?? 0.85}"></div>
          <div><label>Ctx Immediate Offload</label><input id="f-ctx-imm" type="number" step="0.05" value="${ctxb.immediate_offload_ratio ?? 0.7}"></div>
          <div><label>Ctx Displacement Thr</label><input id="f-ctx-disp" type="number" step="0.05" value="${ctxb.displacement_threshold ?? 0.4}"></div>
          <div><label>Ctx Safety Margin</label><input id="f-ctx-safe" type="number" value="${ctxb.safety_margin_tokens ?? 500}"></div>
          <div><label>Ctx Heavy Hitter Min</label><input id="f-ctx-heavy" type="number" value="${ctxb.heavy_hitter_min_tokens ?? 1000}"></div>
        </details>
        <div class="btn-row">
          <button class="btn-primary" id="cfg-save">Speichern</button>
        </div>
        <div id="cfg-status" style="font-size:11px;color:var(--text-muted)"></div>
      `;
      const pSel = fieldsHost.querySelector('#f-persona-sel');
      const pInp = fieldsHost.querySelector('#f-persona');
      pSel.addEventListener('change', () => { pInp.value = pSel.value; });
      fieldsHost.querySelector('#cfg-save').addEventListener('click', async () => {
        const status = fieldsHost.querySelector('#cfg-status');
        status.textContent = 'speichere…';
        const num = (id, d) => { const v = parseFloat(fieldsHost.querySelector(id).value); return isNaN(v) ? d : v; };
        const int = (id, d) => { const v = parseInt(fieldsHost.querySelector(id).value); return isNaN(v) ? d : v; };
        const chk = (id) => fieldsHost.querySelector(id).checked;
        const body = {
          fast_llm_model: fieldsHost.querySelector('#f-fast').value,
          complex_llm_model: fieldsHost.querySelector('#f-complex').value,
          system_message: fieldsHost.querySelector('#f-sys').value,
          temperature: num('#f-temp', 0.7),
          max_parallel_tasks: int('#f-iter', 3),
          active_persona: pInp.value || null,
          max_tokens_output: int('#f-mto', 2048),
          max_tokens_input: int('#f-mti', 32768),
          vfs_max_window_lines: int('#f-vfs', 250),
          use_fast_response: chk('#f-fast-resp'),
          stream: chk('#f-stream'),
          verbose_logging: chk('#f-verbose'),
          obs: {
            enabled: chk('#f-obs-en'),
            max_runs: int('#f-obs-runs', 3),
            snapshot_interval: int('#f-obs-snap', 5),
            enable_audit: chk('#f-obs-audit'),
          },
          web_config: {
            enable_web: chk('#f-web-en'),
            web_headless: chk('#f-web-head'),
            web_single_site: fieldsHost.querySelector('#f-web-single').value || null,
            web_trusted_sites: fieldsHost.querySelector('#f-web-trust').value.split(',').map(s => s.trim()).filter(Boolean),
          },
          context_config: {
            max_context_ratio: num('#f-ctx-max', 0.85),
            immediate_offload_ratio: num('#f-ctx-imm', 0.7),
            displacement_threshold: num('#f-ctx-disp', 0.4),
            safety_margin_tokens: int('#f-ctx-safe', 500),
            heavy_hitter_min_tokens: int('#f-ctx-heavy', 1000),
          },
        };
        try {
          const r = await put(`/api/agents/${encodeURIComponent(name)}/config`, body);
          if (r.rebuild_required && r.rebuild_required.length) {
            status.textContent = `gespeichert; Rebuild nötig: ${r.rebuild_required.join(', ')}`;
          } else {
            status.textContent = 'gespeichert ' + (r.applied_hot?.length ? `(hot: ${r.applied_hot.join(', ')})` : '');
          }
        } catch (e) { status.textContent = 'Fehler: ' + e.message; }
      });
    };
    sel.addEventListener('change', () => loadFor(sel.value));
    loadFor(sel.value);
  }

  async function renderSkills(panel) {
    if (!Store.chatMeta) {
      panel.innerHTML = '<h3 class="sb-panel-title">Skills</h3><p><em>Erst Chat öffnen</em></p>';
      return;
    }
    const agentName = Store.chatMeta.agent || 'self';
    panel.innerHTML = `
      <h3 class="sb-panel-title">Skills · ${escape(agentName)}</h3>
      <div style="margin-bottom:8px;display:flex;gap:4px;flex-wrap:wrap">
        <button class="btn-secondary" id="sk-add" style="flex:1">+ Add</button>
        <button class="btn-secondary" id="sk-lib">Library</button>
      </div>
      <div style="margin-bottom:8px;display:flex;gap:4px;flex-wrap:wrap">
        <button class="btn-secondary" id="sk-export" style="flex:1">⬇ Export ZIP</button>
        <button class="btn-secondary" id="sk-import-zip">⬆ ZIP</button>
        <button class="btn-secondary" id="sk-import-gh">⬆ GitHub</button>
      </div>
      <input type="file" id="sk-file" accept=".zip" hidden>
      <div id="sk-list"><em>laden…</em></div>
    `;
    const list = panel.querySelector('#sk-list');
    const load = async () => {
      let skills;
      try { skills = await get(`/api/agents/${encodeURIComponent(agentName)}/skills`); }
      catch (e) { list.innerHTML = `<em>Fehler: ${escape(e.message)}</em>`; return; }
      if (!skills.length) { list.innerHTML = '<em>keine Skills</em>'; return; }
      list.innerHTML = skills.map(s => `
        <div class="skill-item" data-skill-id="${escape(s.id)}" data-source="${escape(s.source)}">
          <div class="skill-name">${escape(s.name)}</div>
          <div class="skill-triggers">${escape((s.triggers || []).join(', '))}</div>
        </div>
      `).join('');
      list.querySelectorAll('.skill-item').forEach(el => {
        el.addEventListener('click', () => editSkill(el, skills.find(x => x.id === el.dataset.skillId), agentName, load));
      });
    };
    panel.querySelector('#sk-add').addEventListener('click', async () => {
      const id = await window.ISAA.UI.prompt('Neue Skill-ID (slug):');
      if (!id) return;
      const name = await window.ISAA.UI.prompt('Anzeige-Name:', id);
      await post(`/api/agents/${encodeURIComponent(agentName)}/skills`, {
        id, name: name || id, triggers: [], instruction: '', tools_used: [], tool_groups: [],
      });
      load();
    });
    panel.querySelector('#sk-lib').addEventListener('click', async () => {
      const lib = await get('/api/skills/library');
      window.ISAA.UI.toast(`Library: ${lib.length} Skills verfügbar`, 'info');
    });
    panel.querySelector('#sk-export').addEventListener('click', () => {
      window.open(`/api/agents/${encodeURIComponent(agentName)}/skills/export`, '_blank');
    });
    const fileInput = panel.querySelector('#sk-file');
    panel.querySelector('#sk-import-zip').addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
      const f = fileInput.files[0];
      if (!f) return;
      const fd = new FormData();
      fd.append('file', f);
      try {
        const r = await fetch(`/api/agents/${encodeURIComponent(agentName)}/skills/import`, {
          method: 'POST', credentials: 'same-origin', body: fd,
        });
        const data = await r.json();
        window.ISAA.UI.toast(`Importiert: ${(data.imported || []).length}, Fehler: ${(data.errors || []).length}`, (data.errors || []).length ? 'warning' : 'success');
        load();
      } catch (e) { window.ISAA.UI.toast('Fehler: ' + e.message, 'error'); }
      fileInput.value = '';
    });
    panel.querySelector('#sk-import-gh').addEventListener('click', async () => {
      const url = await window.ISAA.UI.prompt('GitHub URL (Ordner /tree/, raw .json, oder .zip):');
      if (!url) return;
      try {
        const r = await post(`/api/agents/${encodeURIComponent(agentName)}/skills/import`, { github_url: url });
        window.ISAA.UI.toast(`Importiert: ${(r.imported || []).length}, Fehler: ${(r.errors || []).length}`, (r.errors || []).length ? 'warning' : 'success');
        load();
      } catch (e) { window.ISAA.UI.toast('Fehler: ' + e.message, 'error'); }
    });
    load();
  }

  // ============================================================================
  // TOOLS PANEL — MCP / CLI / list by category + toggle
  // ============================================================================

  async function renderTools(panel) {
    if (!Store.chatMeta) {
      panel.innerHTML = '<h3 class="sb-panel-title">Tools</h3><p><em>Erst Chat öffnen</em></p>';
      return;
    }
    const agentName = Store.chatMeta.agent || 'self';
    panel.innerHTML = `
      <h3 class="sb-panel-title">Tools · ${escape(agentName)}</h3>
      <div style="display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap">
        <button class="btn-secondary" id="t-cli" style="flex:1">+ CLI</button>
        <button class="btn-secondary" id="t-mcp" style="flex:1">+ MCP</button>
      </div>
      <div style="display:flex;gap:4px;margin-bottom:8px">
        <button class="btn-secondary" id="t-all-on" style="flex:1">Alle an</button>
        <button class="btn-secondary" id="t-all-off" style="flex:1">Alle aus</button>
      </div>
      <div id="t-list"><em>laden…</em></div>
    `;
    const list = panel.querySelector('#t-list');
    const load = async () => {
      let data;
      try { data = await get(`/api/agents/${encodeURIComponent(agentName)}/tools`); }
      catch (e) { list.innerHTML = `<em>Fehler: ${escape(e.message)}</em>`; return; }
      const cats = data.categories || {};
      const catNames = Object.keys(cats).sort();
      if (!catNames.length) { list.innerHTML = '<em>keine Tools</em>'; return; }
      let html = `<div style="font-family:var(--font-mono);font-size:9px;color:var(--text-muted);margin-bottom:8px">${data.total} tools</div>`;
      for (const cat of catNames) {
        const items = cats[cat];
        html += `<div class="tool-cat">
          <div class="tool-cat-head">${escape(cat)} <span class="li-meta">${items.length}</span></div>
          <div class="grid-table tool-table">`;
        for (const t of items) {
          html += `<div class="tool-row" data-tool="${escape(t.name)}">
            <span class="tool-tname" title="${escape(t.description)}">${escape(t.name)}</span>
            <span class="tool-tsrc">${escape(t.source)}</span>
            <label class="tool-toggle">
              <input type="checkbox" data-toggle ${t.enabled ? 'checked' : ''}>
            </label>
            ${t.source !== 'local' ? '<button class="li-del" data-del title="Entfernen">✕</button>' : '<span></span>'}
          </div>`;
        }
        html += `</div></div>`;
      }
      list.innerHTML = html;
      // bind toggles
      list.querySelectorAll('.tool-row').forEach(row => {
        const tname = row.dataset.tool;
        const cb = row.querySelector('[data-toggle]');
        if (cb) cb.addEventListener('change', async () => {
          await put(`/api/agents/${encodeURIComponent(agentName)}/tools/${encodeURIComponent(tname)}/toggle`, { enabled: cb.checked });
        });
        const del = row.querySelector('[data-del]');
        if (del) del.addEventListener('click', async () => {
          if (!await window.ISAA.UI.confirm(`Tool ${tname} entfernen?`, { danger: true, okLabel: 'Entfernen' })) return;
          await del2(`/api/agents/${encodeURIComponent(agentName)}/tools/${encodeURIComponent(tname)}`);
          load();
        });
      });
    };
    panel.querySelector('#t-all-on').addEventListener('click', async () => {
      await post(`/api/agents/${encodeURIComponent(agentName)}/tools/toggle_all`, { enabled: true });
      load();
    });
    panel.querySelector('#t-all-off').addEventListener('click', async () => {
      await post(`/api/agents/${encodeURIComponent(agentName)}/tools/toggle_all`, { enabled: false });
      load();
    });
    panel.querySelector('#t-cli').addEventListener('click', async () => {
      const name = await window.ISAA.UI.prompt('Tool-Name:'); if (!name) return;
      const executable = await window.ISAA.UI.prompt('Executable (z.B. python, git, tb):'); if (!executable) return;
      const sub = (await window.ISAA.UI.prompt('Sub-Command (z.B. status, oder leer):', '')) || '';
      try {
        const r = await post(`/api/agents/${encodeURIComponent(agentName)}/tools/cli`, {
          name, executable, cli_tool_executable: sub,
        });
        if (r.ok) { window.ISAA.UI.toast(`Tool ${name} hinzugefügt`, 'success'); load(); }
        else window.ISAA.UI.toast(r.error || 'Fehler', 'error');
      } catch (e) { window.ISAA.UI.toast('Fehler: ' + e.message, 'error'); }
    });
    panel.querySelector('#t-mcp').addEventListener('click', async () => {
      const server = await window.ISAA.UI.prompt('MCP Server-Name:'); if (!server) return;
      const cfgRaw = await window.ISAA.UI.prompt('Config JSON (z.B. {"command":"npx","args":["-y","@mcp/server-fs","/data"]} oder {"url":"https://..."}):', '', { multiline: true });
      if (!cfgRaw) return;
      let cfg;
      try { cfg = JSON.parse(cfgRaw); } catch (_) { window.ISAA.UI.toast('Ungültiges JSON', 'error'); return; }
      try {
        const r = await post(`/api/agents/${encodeURIComponent(agentName)}/tools/mcp`, { server_name: server, config: cfg });
        if (r.ok) { window.ISAA.UI.toast(`${r.registered} MCP-Tools registriert`, 'success'); load(); }
        else window.ISAA.UI.toast(r.error || 'Fehler', 'error');
      } catch (e) { window.ISAA.UI.toast('Fehler: ' + e.message, 'error'); }
    });
    load();
  }

  // delete helper (del is shadowed inside renderTools scope by param name)
  async function del2(path) {
    const r = await fetch(path, { method: 'DELETE', credentials: 'same-origin' });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  }

  function editSkill(el, skill, agentName, reload) {
    el.dataset.expanded = el.dataset.expanded === 'true' ? 'false' : 'true';
    let edit = el.querySelector('.skill-edit');
    if (el.dataset.expanded === 'false') {
      if (edit) edit.remove();
      return;
    }
    if (!edit) {
      edit = document.createElement('div');
      edit.className = 'skill-edit';
      edit.innerHTML = `
        <label>Triggers (komma-getrennt)</label>
        <input type="text" value="${escape((skill.triggers || []).join(', '))}">
        <label>Instruction</label>
        <textarea rows="6">${escape(skill.instruction || '')}</textarea>
        <label>Tools used (komma-getrennt)</label>
        <input type="text" value="${escape((skill.tools_used || []).join(', '))}">
        <div class="btn-row">
          <button class="btn-primary" data-action="save">Save</button>
          ${skill.source !== 'predefined' ? '<button class="btn-secondary" data-action="del">Delete</button>' : ''}
        </div>
      `;
      el.appendChild(edit);
      edit.addEventListener('click', async (e) => {
        e.stopPropagation();
        const a = e.target.dataset.action;
        if (a === 'save') {
          const inputs = edit.querySelectorAll('input, textarea');
          const body = {
            triggers: inputs[0].value.split(',').map(s => s.trim()).filter(Boolean),
            instruction: inputs[1].value,
            tools_used: inputs[2].value.split(',').map(s => s.trim()).filter(Boolean),
          };
          await put(`/api/agents/${encodeURIComponent(agentName)}/skills/${encodeURIComponent(skill.id)}`, body);
          reload();
        } else if (a === 'del') {
          if (!await window.ISAA.UI.confirm('Skill löschen?', { danger: true, okLabel: 'Löschen' })) return;
          await del(`/api/agents/${encodeURIComponent(agentName)}/skills/${encodeURIComponent(skill.id)}`);
          reload();
        }
      });
    }
  }

  // ============================================================================
  // PANEL DISPATCH
  // ============================================================================

  function openPanel(name) {
    const sidebar = document.getElementById('sidebar');
    const panel = document.getElementById('sb-panel');
    document.querySelectorAll('.sb-icon[data-panel]').forEach(b => {
      b.dataset.active = b.dataset.panel === name ? 'true' : 'false';
    });
    if (!name) {
      sidebar.dataset.expanded = 'false';
      document.body.parentElement.dataset.panelOpen = 'false';
      document.querySelector('.isaa-body').dataset.panelOpen = 'false';
      Store.setPanel(null);
      return;
    }
    sidebar.dataset.expanded = 'true';
    document.querySelector('.isaa-body').dataset.panelOpen = 'true';
    Store.setPanel(name);
    panel.innerHTML = '';
    if (name === 'chats') renderChats(panel);
    else if (name === 'vfs') renderVfs(panel);
    else if (name === 'agent') renderAgent(panel);
    else if (name === 'skills') renderSkills(panel);
    else if (name === 'tools') renderTools(panel);
  }

  function bind() {
    document.querySelectorAll('.sb-icon').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        const panel = btn.dataset.panel;
        if (action === 'new-chat') {
          window.ISAA.App.newChat();
          return;
        }
        if (panel) {
          const cur = Store.sidebarPanel;
          openPanel(cur === panel ? null : panel);
        }
      });
    });
    // Mobile bottom-nav uses same panel dispatch
    document.querySelectorAll('.mn-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        const panel = btn.dataset.panel;
        if (action === 'new-chat') {
          window.ISAA.App.newChat();
          return;
        }
        if (panel) {
          openMobilePanel(panel);
        }
      });
    });
    // Mobile-only close button (bottom-right) closes the open panel/Rubrik
    const mClose = document.getElementById('mobile-panel-close');
    if (mClose) mClose.addEventListener('click', closeMobilePanel);

    // Desktop: drag the sidebar's right edge to resize; width is persisted forever.
    const sidebarEl = document.getElementById('sidebar');
    const bodyEl = document.querySelector('.isaa-body');
    const resize = document.getElementById('sb-resize');
    const savedW = localStorage.getItem('isaa.sb_width');
    if (savedW && bodyEl) bodyEl.style.setProperty('--sb-width', savedW);
    if (resize && bodyEl && sidebarEl) {
      let dragging = false;
      const onMove = (e) => {
        if (!dragging) return;
        const w = Math.max(200, Math.min(640, e.clientX));
        bodyEl.style.setProperty('--sb-width', w + 'px');
      };
      const onUp = () => {
        if (!dragging) return;
        dragging = false;
        sidebarEl.style.transition = '';
        document.body.style.userSelect = '';
        const w = bodyEl.style.getPropertyValue('--sb-width').trim();
        if (w) localStorage.setItem('isaa.sb_width', w);
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };
      resize.addEventListener('pointerdown', (e) => {
        if (window.innerWidth <= 767) return;
        e.preventDefault();
        dragging = true;
        sidebarEl.style.transition = 'none';  // crisp drag, no width easing
        document.body.style.userSelect = 'none';
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
      });
    }

    // Restore last panel
    if (Store.sidebarPanel) openPanel(Store.sidebarPanel);
  }

  function closeMobilePanel() {
    const sidebar = document.getElementById('sidebar');
    sidebar.dataset.mobileOpen = 'false';
    document.querySelectorAll('.mn-btn').forEach(b => b.dataset.active = 'false');
    Store.setPanel(null);
  }

  function openMobilePanel(name) {
    const sidebar = document.getElementById('sidebar');
    if (sidebar.dataset.mobileOpen === 'true' && Store.sidebarPanel === name) {
      // close
      closeMobilePanel();
      return;
    }
    sidebar.dataset.mobileOpen = 'true';
    sidebar.dataset.expanded = 'true';
    document.querySelectorAll('.mn-btn').forEach(b => {
      b.dataset.active = b.dataset.panel === name ? 'true' : 'false';
    });
    Store.setPanel(name);
    const panel = document.getElementById('sb-panel');
    panel.innerHTML = '';
    if (name === 'chats') renderChats(panel);
    else if (name === 'vfs') renderVfs(panel);
    else if (name === 'agent') renderAgent(panel);
    else if (name === 'skills') renderSkills(panel);
    else if (name === 'tools') renderTools(panel);
  }

  Store.on('chat:changed', () => {
    // refresh active marker in chats panel if open
    if (Store.sidebarPanel === 'chats') {
      const panel = document.getElementById('sb-panel');
      renderChats(panel);
    }
  });

  window.ISAA = window.ISAA || {};
  window.ISAA.Sidebar = { openPanel, bind };
})();
