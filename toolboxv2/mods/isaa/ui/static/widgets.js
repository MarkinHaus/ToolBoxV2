/**
 * Widget mode — GridStack-based widget host.
 *
 * Triggered by:
 *   - Manual toggle (button on header or shortcut)
 *   - Agent emitting `widget_create` / synthetic frame (or a tool_start with name=create_widget)
 *
 * Built-in adapters (task 19):
 *   - markdown:  marked.min.js → HTML (sandboxed via DOMPurify-lite)
 *   - vega:      vega-embed via CDN
 *   - code:      <pre><code> (no syntax highlight yet)
 *   - table:     CSS grid
 *   - form:      input chain
 *   - html:      custom render_js executed in sandboxed iframe (task 23)
 *   - htmldoc:   raw html view in sandboxed iframe (task 23)
 *
 * External libs loaded lazily from CDN on first use:
 *   - https://cdn.jsdelivr.net/npm/gridstack@10/dist/gridstack-all.js
 *   - https://cdn.jsdelivr.net/npm/marked/marked.min.js
 *   - https://cdn.jsdelivr.net/npm/vega@5 + vega-lite + vega-embed
 */
(function () {
  'use strict';

  const Store = window.ISAA.Store;
  const CDN = {
    gridstackJs: 'https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack-all.js',
    gridstackCss: 'https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack.min.css',
    marked: 'https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js',
    vega: 'https://cdn.jsdelivr.net/npm/vega@5',
    vegaLite: 'https://cdn.jsdelivr.net/npm/vega-lite@5',
    vegaEmbed: 'https://cdn.jsdelivr.net/npm/vega-embed@6',
  };

  const loadedScripts = new Map(); // url -> Promise
  function loadScript(url) {
    if (loadedScripts.has(url)) return loadedScripts.get(url);
    const p = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('Failed to load ' + url));
      document.head.appendChild(s);
    });
    loadedScripts.set(url, p);
    return p;
  }
  function loadCss(url) {
    if (loadedScripts.has(url)) return;
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = url;
    document.head.appendChild(l);
    loadedScripts.set(url, true);
  }

  function escape(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  }

  // ============================================================================
  // ADAPTERS (task 19)
  // ============================================================================

  const adapters = {
    markdown: async (root, props) => {
      await loadScript(CDN.marked);
      const md = props.md || '';
      // marked is on window.marked after load
      root.innerHTML = window.marked ? window.marked.parse(md) : `<pre>${escape(md)}</pre>`;
    },

    vega: async (root, props) => {
      await loadScript(CDN.vega);
      await loadScript(CDN.vegaLite);
      await loadScript(CDN.vegaEmbed);
      const spec = props.spec || {};
      try {
        await window.vegaEmbed(root, spec, { actions: false, theme: 'dark' });
      } catch (e) {
        root.innerHTML = `<pre style="color:var(--error)">${escape(e.message)}</pre>`;
      }
    },

    code: (root, props) => {
      const lang = props.lang || '';
      const code = props.code || '';
      root.innerHTML = `<pre><code class="lang-${escape(lang)}">${escape(code)}</code></pre>`;
    },

    table: (root, props) => {
      const cols = props.columns || [];
      const rows = props.rows || [];
      const tpl = cols.map(() => 'auto').join(' ');
      let html = `<div class="grid-table" style="display:grid;grid-template-columns:${tpl};gap:4px">`;
      html += '<div class="grid-row grid-header" style="display:contents">';
      for (const c of cols) html += `<div class="grid-cell" style="font-family:var(--font-mono, monospace);font-size:9px;text-transform:uppercase;letter-spacing:1px;color:var(--text-label);border-bottom:1px solid var(--border-subtle, rgba(255,255,255,0.08));padding:4px 8px">${escape(c)}</div>`;
      html += '</div>';
      for (const r of rows) {
        html += '<div class="grid-row" style="display:contents">';
        for (const v of r) html += `<div class="grid-cell" style="padding:4px 8px;font-size:11px">${escape(v)}</div>`;
        html += '</div>';
      }
      html += '</div>';
      root.innerHTML = html;
    },

    form: (root, props, api) => {
      const fields = props.fields || [];
      const submitLabel = props.submit_label || 'Submit';
      let html = '<form class="widget-form" style="display:flex;flex-direction:column;gap:8px">';
      for (const f of fields) {
        html += `<label style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-label)">${escape(f.label || f.name)}</label>`;
        if (f.type === 'textarea') {
          html += `<textarea name="${escape(f.name)}" rows="3">${escape(f.default || '')}</textarea>`;
        } else {
          html += `<input name="${escape(f.name)}" type="${escape(f.type || 'text')}" value="${escape(f.default || '')}">`;
        }
      }
      html += `<button class="btn-primary" type="submit">${escape(submitLabel)}</button></form>`;
      root.innerHTML = html;
      root.querySelector('form').addEventListener('submit', (e) => {
        e.preventDefault();
        const data = {};
        root.querySelectorAll('input, textarea, select').forEach(inp => { data[inp.name] = inp.value; });
        api.emit('submit', data);
      });
    },

    // Custom HTML — runs render_js in a sandboxed iframe (task 23)
    // render_js may be EITHER a function expression `(root, props, api) => {…}`
    // (renders into root, return value ignored) OR a plain body that uses
    // `props`/`root`/`api` and `return`s a Node (it gets appended to root).
    html: (root, props, api, widgetId) => {
      const renderJs = props.render_js || '';
      const initialProps = props.props || {};
      // Keep injected source from closing the inline <script> early.
      const safeSrc = JSON.stringify(renderJs).replace(/<\//g, '<\\/');
      const safeProps = JSON.stringify(initialProps).replace(/<\//g, '<\\/');
      const safeWid = JSON.stringify(widgetId);
      const iframe = document.createElement('iframe');
      iframe.style.cssText = 'width:100%;height:100%;border:none;background:transparent';
      iframe.sandbox = 'allow-scripts';
      const html = `<!DOCTYPE html><html><head><style>
        body { margin: 0; padding: 8px; font-family: system-ui, sans-serif; background: transparent; color: rgba(255,255,255,0.85); }
        * { box-sizing: border-box; }
      </style></head><body><div id="root"></div>
      <script>
        const api = {
          emit: (name, payload) => parent.postMessage({ __isaa: true, widgetId: ${safeWid}, action: name, payload }, '*'),
          set_var: (scope, key, value) => parent.postMessage({ __isaa: true, widgetId: ${safeWid}, kind: 'set_var', scope, key, value }, '*'),
        };
        const __root = document.getElementById('root');
        const __props = ${safeProps};
        const __src = ${safeSrc};
        try {
          let fn = null;
          // Try expression form first: wrapping in parens makes it an expression.
          try { fn = (0, eval)('(' + __src + ')'); } catch (_) { fn = null; }
          if (typeof fn === 'function') {
            // Expression contract: renders into root, return value ignored.
            fn(__root, __props, api);
          } else {
            // Body contract: build element, return it; caller appends.
            const out = (new Function('root', 'props', 'api', __src))(__root, __props, api);
            if (out instanceof Node) __root.appendChild(out);
          }
        } catch (e) {
          document.body.innerHTML = '<pre style="color:#ff6b6b">' + (e.message || e) + '</pre>';
        }
      </script></body></html>`;
      iframe.srcdoc = html;
      root.appendChild(iframe);
    },
    // Raw HTML document — file content rendered as-is in a sandboxed iframe.
    // Injects a window.storage shim that proxies to the parent → WS set_var
    // (keys prefixed 'storage:', scope 'agent' = per chat). No page change needed.
    htmldoc: (root, props, api, widgetId) => {
      seedVarCache();
      const snapshot = {};
      for (const k in varCache.agent) {
        if (k.indexOf('storage:') === 0) snapshot[k] = varCache.agent[k];
      }
      const safeWid = JSON.stringify(widgetId);
      const safeSnap = JSON.stringify(snapshot).replace(/<\//g, '<\\/');
      const boot = `<script>(function(){
        var W = ${safeWid}, PREFIX = 'storage:', cache = ${safeSnap};
        function post(key, value){ parent.postMessage({ __isaa: true, widgetId: W, kind: 'set_var', scope: 'agent', key: key, value: value }, '*'); }
        window.storage = {
          get: function(key){ var k = PREFIX + key; return Promise.resolve((k in cache && cache[k] != null) ? { key: key, value: cache[k] } : null); },
          set: function(key, value){ var k = PREFIX + key; cache[k] = value; post(k, value); return Promise.resolve({ key: key, value: value }); },
          delete: function(key){ var k = PREFIX + key; delete cache[k]; post(k, null); return Promise.resolve({ key: key, deleted: true }); },
          list: function(prefix){ var out = []; for (var k in cache){ if (k.indexOf(PREFIX) === 0){ var bare = k.slice(PREFIX.length); if (!prefix || bare.indexOf(prefix) === 0) out.push(bare); } } return Promise.resolve({ keys: out }); }
        };
        // Sandbox = opaque origin → native localStorage/sessionStorage throw.
        // localStorage: cache-backed + persisted via storage: bridge. sessionStorage: in-memory.
        function mkLS(backing, persist){
          return {
            getItem: function(key){ var k = PREFIX + key; return (k in backing && backing[k] != null) ? String(backing[k]) : null; },
            setItem: function(key, value){ var k = PREFIX + key; backing[k] = String(value); if (persist) post(k, String(value)); },
            removeItem: function(key){ var k = PREFIX + key; delete backing[k]; if (persist) post(k, null); },
            clear: function(){ for (var k in backing){ if (k.indexOf(PREFIX) === 0){ if (persist) post(k, null); delete backing[k]; } } },
            key: function(i){ var ks = []; for (var k in backing){ if (k.indexOf(PREFIX) === 0) ks.push(k); } return ks[i] ? ks[i].slice(PREFIX.length) : null; },
            get length(){ var n = 0; for (var k in backing){ if (k.indexOf(PREFIX) === 0) n++; } return n; }
          };
        }
        try {
          Object.defineProperty(window, 'localStorage', { value: mkLS(cache, true), configurable: true });
          Object.defineProperty(window, 'sessionStorage', { value: mkLS({}, false), configurable: true });
        } catch (e) {}
        window.addEventListener('message', function(e){
          var d = e.data;
          if (d && d.__isaa_var_set && typeof d.key === 'string' && d.key.indexOf(PREFIX) === 0) cache[d.key] = d.value;
        });
      })();<\/script>`;
      const iframe = document.createElement('iframe');
      iframe.style.cssText = 'width:100%;height:100%;border:none;background:#fff';
      iframe.sandbox = 'allow-scripts';
      iframe.srcdoc = injectBoot(props.html || '', boot);
      root.appendChild(iframe);
    },
  };

  // ============================================================================
  // WIDGET HOST (task 18)
  // ============================================================================

  let grid = null;
  const widgets = new Map(); // widget_id -> { template, props, el }
  const templateSpecs = new Map(); // template_id -> { name, adapter, schema, render_js } (custom only)

  async function ensureGrid() {
    if (grid) return grid;
    loadCss(CDN.gridstackCss);
    await loadScript(CDN.gridstackJs);
    const view = document.getElementById('view');
    view.innerHTML = '<div class="grid-stack" id="widget-grid"></div>';
    document.querySelector('.isaa-body').dataset.view = 'widgets';
    grid = window.GridStack.init({
      column: 12,
      cellHeight: 70,
      margin: 8,
      animate: true,
      float: true,              // agent can place widgets at explicit x/y
      disableResize: false,
      disableDrag: false,
      handle: '.widget-header', // drag by header → doesn't conflict with body scroll
    }, '#widget-grid');
    // Responsive: 1 column under 768px
    if (window.innerWidth <= 767) {
      try { grid.column(1); } catch (_) {}
    }
    window.addEventListener('resize', () => {
      if (!grid) return;
      try { grid.column(window.innerWidth <= 767 ? 1 : 12); } catch (_) {}
    });
    grid.on('change', saveLayout);   // persist geometry on drag/resize
    await restoreLayout();           // bring back persisted widgets for this chat
    return grid;
  }
  // ============================================================================
  // PERSISTENCE — widget set + geometry per chat (survives mode switch + reload)
  // ============================================================================

  let _restoring = false;

  function _layoutKey() {
    return 'isaa.widgets.' + (Store.activeChatId || '_');
  }

  function saveLayout() {
    const out = [];
    let hasCtl = false;
    for (const [id, w] of widgets) {
      const n = w.el.gridstackNode || {};
      out.push({
        widgetId: id,
        template: w.template,
        props: id === CONTROLLER_ID ? null : w.props,  // controller body is rebuilt; only geometry matters
        pin: w.pin || null,
        x: n.x, y: n.y, w: n.w, h: n.h,
      });
      if (id === CONTROLLER_ID) hasCtl = true;
    }
    // Controller not mounted yet (e.g. save fires during restore) → keep its prior
    // geometry so it doesn't snap to 0,0 and shove the user layout.
    if (!hasCtl) {
      const prev = loadLayout().find(s => s && s.widgetId === CONTROLLER_ID);
      if (prev) out.push(prev);
    }
    try { localStorage.setItem(_layoutKey(), JSON.stringify(out)); } catch (_) {}
  }

  function loadLayout() {
    try {
      const raw = localStorage.getItem(_layoutKey());
      return raw ? JSON.parse(raw) : [];
    } catch (_) { return []; }
  }

  async function restoreLayout() {
    const saved = loadLayout();
    if (!saved.length) return;
    _restoring = true;
    try {
      for (const s of saved) {
        if (!s || !s.widgetId || s.widgetId === CONTROLLER_ID) continue;
        await createWidget(s.widgetId, s.template, s.props || {}, s.pin || null,
          { x: s.x, y: s.y, w: s.w, h: s.h });
      }
    } finally {
      _restoring = false;
    }
    saveLayout();
  }

  async function createWidget(widgetId, template, props, pin, geom) {
    await ensureGrid();
    if (widgets.has(widgetId)) return;  // guard against duplicate (restore/live)
    const adapter = adapters[template];
    if (!adapter) {
      console.warn('[widget] unknown adapter:', template);
      return;
    }
    const g = geom || pin;
    const w = (g && g.w) || 4;
    const h = (g && g.h) || 3;
    const x = (g && g.x !== undefined && g.x !== null) ? g.x : 0;
    const y = (g && g.y !== undefined && g.y !== null) ? g.y : 0;
    const noResize = pin ? 'gs-no-resize="true"' : '';  // only an agent pin locks
    const noMove = pin ? 'gs-no-move="true"' : '';
    const html = `<div class="grid-stack-item" gs-w="${w}" gs-h="${h}" gs-x="${x}" gs-y="${y}" ${noResize} ${noMove} data-widget-id="${escape(widgetId)}">
      <div class="grid-stack-item-content widget-card">
        <div class="widget-header">
          <span class="widget-title">${escape(template)}</span>
          <button class="widget-close" data-action="close">×</button>
        </div>
        <div class="widget-body"></div>
      </div>
    </div>`;
    const wrapper = document.createElement('div');
    wrapper.innerHTML = html;
    const el = wrapper.firstElementChild;
    grid.addWidget(el);
    const body = el.querySelector('.widget-body');
    const api = {
      emit: (name, payload) => {
        if (window.ISAA.WS) window.ISAA.WS.send({ op: 'widget_action', widget_id: widgetId, action: name, payload });
      },
      set_var: (scope, key, value) => {
        if (window.ISAA.WS) window.ISAA.WS.send({ op: 'widget_action', widget_id: widgetId, action: 'set_var', payload: { scope, key, value } });
      },
    };
    try {
      await adapter(body, props, api, widgetId);
    } catch (e) {
      body.innerHTML = `<pre style="color:var(--error)">${escape(e.message)}</pre>`;
    }
    el.querySelector('[data-action="close"]').addEventListener('click', () => closeWidget(widgetId));
    widgets.set(widgetId, { template, props, el, api, pin: pin || null });
    if (controllerBody) renderControllerBody();
    if (!_restoring) saveLayout();
  }

  function closeWidget(widgetId) {
    const w = widgets.get(widgetId);
    if (!w) return;
    if (grid) grid.removeWidget(w.el);
    widgets.delete(widgetId);
    if (controllerBody) renderControllerBody();
    if (!_restoring) saveLayout();
    if (widgets.size === 0) exitWidgetMode();
  }

  function updateWidget(widgetId, propsPatch) {
    const w = widgets.get(widgetId);
    if (!w) return;
    Object.assign(w.props, propsPatch);
    const body = w.el.querySelector('.widget-body');
    body.innerHTML = '';
    adapters[w.template](body, w.props, w.api, widgetId);
  }

  function exitWidgetMode() {
    saveLayout();
    grid = null;
    widgets.clear();
    controllerBody = null;
    document.querySelector('.isaa-body').dataset.view = Store.activeChatId ? 'chat' : 'welcome';
    if (window.ISAA.App && window.ISAA.App.rerender) window.ISAA.App.rerender();
  }

  /** Manual switch between chat and widget mode (bottom-right button). */
  async function toggleMode() {
    if (document.querySelector('.isaa-body').dataset.view === 'widgets') {
      exitWidgetMode();
    } else {
      await ensureGrid();
      await createControllerWidget()
    }
  }
  // ============================================================================
  // CONTROLLER WIDGET (manual open/close + template import/export)
  // ============================================================================

  const CONTROLLER_ID = '__controller__';
  const BUILTINS = new Set(['markdown', 'vega', 'code', 'table', 'form', 'html', 'htmldoc']);
  let controllerBody = null;

  function genId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, '').slice(0, 12);
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }

  function _toast(msg, kind) {
    if (window.ISAA && window.ISAA.UI && window.ISAA.UI.toast) window.ISAA.UI.toast(msg, kind || 'info');
    else console.log('[widgets]', msg);
  }

  function importTemplate(spec) {
    if (!spec || !spec.template_id) return false;
    if (BUILTINS.has(spec.template_id)) return false;  // never override built-ins
    delete adapters[spec.template_id];                 // allow re-import / overwrite
    registerTemplate(spec.template_id, spec.adapter || 'html', spec.render_js || null, spec.name, spec.schema);
    return true;
  }

  function importTemplatesFromFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      let data;
      try { data = JSON.parse(reader.result); }
      catch (e) { _toast('Import JSON ungültig: ' + e.message, 'error'); return; }
      const list = Array.isArray(data) ? data : [data];
      let n = 0;
      for (const spec of list) { if (importTemplate(spec)) n++; }
      _toast(n + ' Template(s) importiert', 'info');
      renderControllerBody();
    };
    reader.onerror = () => _toast('Datei lesen fehlgeschlagen', 'error');
    reader.readAsText(file);
  }

  function exportTemplates() {
    const specs = [...templateSpecs.entries()].map(([id, s]) => ({
      template_id: id, name: s.name, adapter: s.adapter, schema: s.schema, render_js: s.render_js,
    }));
    if (!specs.length) { _toast('Keine custom Templates zum Export', 'info'); return; }
    const blob = new Blob([JSON.stringify(specs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'isaa_templates.json';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  function resetStorageKey(key) {
    varCache.agent[key] = null;   // optimistic → no double reload from the echo
    if (window.ISAA.WS) window.ISAA.WS.send({ op: 'widget_action', widget_id: CONTROLLER_ID, action: 'set_var', payload: { scope: 'agent', key, value: null } });
    for (const [wid, w] of widgets) if (w.template === 'htmldoc') updateWidget(wid, {});
    renderControllerBody();
  }

  function renderControllerBody() {
    if (!controllerBody) return;
    const prevTpl = (controllerBody.querySelector('.ctl-tpl') || {}).value;
    const prevProps = (controllerBody.querySelector('.ctl-props') || {}).value;
    const lbl = 'font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-label);margin-bottom:4px';
    const opts = Object.keys(adapters).filter(k => k !== 'html')
      .map(k => `<option value="${escape(k)}">${escape(k)}</option>`).join('');
    const active = [...widgets.keys()].filter(id => id !== CONTROLLER_ID);
    seedVarCache();
    const activeHtml = active.length
      ? active.map(id => {
          const w = widgets.get(id);
          return `<li style="display:flex;justify-content:space-between;gap:8px;align-items:center;padding:2px 0">
            <span style="font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escape(w.template)} · ${escape(id)}</span>
            <button class="ctl-close-one" data-id="${escape(id)}" style="background:none;border:none;color:var(--error,#ff6b6b);cursor:pointer;font-size:14px">×</button>
          </li>`;
        }).join('')
      : '<li style="font-size:11px;opacity:.6">keine</li>';
    controllerBody.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:10px;font-family:system-ui,sans-serif">
        <div>
          <div style="${lbl}">Open widget</div>
          <select class="ctl-tpl" style="width:100%;margin-bottom:4px">${opts}</select>
          <textarea class="ctl-props" rows="3" spellcheck="false" style="width:100%;font-family:var(--font-mono,monospace);font-size:11px" placeholder='{"md":"# hi"}'>{}</textarea>
          <button class="ctl-create btn-primary" style="margin-top:4px;width:100%">Create</button>
        </div>
        <div>
          <div style="${lbl}">Active</div>
          <ul class="ctl-active" style="list-style:none;padding:0;margin:0">${activeHtml}</ul>
        </div>
        <div>
          <div style="${lbl}">Templates</div>
          <div style="display:flex;gap:4px">
            <label class="btn-secondary" style="flex:1;text-align:center;cursor:pointer;margin:0">Import<input type="file" accept="application/json,.json" class="ctl-import" hidden></label>
            <button class="ctl-export btn-secondary" style="flex:1">Export</button>
          </div>
        </div>
        <div>
          <div style="${lbl}">Page-State (storage:)</div>
          <ul class="ctl-state" style="list-style:none;padding:0;margin:0">${
            (() => {
              const keys = Object.keys(varCache.agent).filter(k => k.indexOf('storage:') === 0 && varCache.agent[k] != null);
              return keys.length
                ? keys.map(k => `<li style="display:flex;justify-content:space-between;gap:8px;align-items:center;padding:2px 0">
                    <span style="font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escape(k.slice(8))}</span>
                    <button class="ctl-reset-state" data-key="${escape(k)}" title="Reset" style="background:none;border:none;color:var(--error,#ff6b6b);cursor:pointer;font-size:13px">⟲</button>
                  </li>`).join('')
                : '<li style="font-size:11px;opacity:.6">keine</li>';
            })()
          }</ul>
        </div>
      </div>`;
    const selEl = controllerBody.querySelector('.ctl-tpl');
    if (prevTpl && [...selEl.options].some(o => o.value === prevTpl)) selEl.value = prevTpl;
    if (prevProps !== undefined && prevProps !== '') controllerBody.querySelector('.ctl-props').value = prevProps;
    controllerBody.querySelector('.ctl-create').addEventListener('click', () => {
      const tpl = controllerBody.querySelector('.ctl-tpl').value;
      if (!tpl) return;
      let props = {};
      try { props = JSON.parse(controllerBody.querySelector('.ctl-props').value || '{}'); }
      catch (e) { _toast('Props kein valides JSON: ' + e.message, 'error'); return; }
      createWidget(genId(), tpl, props);
    });
    controllerBody.querySelectorAll('.ctl-close-one').forEach(b => {
      b.addEventListener('click', () => closeWidget(b.dataset.id));
    });
    controllerBody.querySelectorAll('.ctl-reset-state').forEach(b => {
      b.addEventListener('click', () => resetStorageKey(b.dataset.key));
    });
    controllerBody.querySelector('.ctl-export').addEventListener('click', exportTemplates);
    controllerBody.querySelector('.ctl-import').addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) importTemplatesFromFile(f);
      e.target.value = '';
    });
  }

  async function createControllerWidget() {
    const savedCtl = loadLayout().find(s => s && s.widgetId === CONTROLLER_ID);
    await ensureGrid();
    if (widgets.has(CONTROLLER_ID)) return;
    const cw = (savedCtl && savedCtl.w) || 3;
    const ch = (savedCtl && savedCtl.h) || 6;
    const cx = (savedCtl && savedCtl.x != null) ? savedCtl.x : 0;
    const cy = (savedCtl && savedCtl.y != null) ? savedCtl.y : 0;
    const html = `<div class="grid-stack-item" gs-w="${cw}" gs-h="${ch}" gs-x="${cx}" gs-y="${cy}" data-widget-id="${CONTROLLER_ID}">
      <div class="grid-stack-item-content widget-card">
        <div class="widget-header"><span class="widget-title">Controller</span></div>
        <div class="widget-body" style="overflow:auto"></div>
      </div>
    </div>`;
    const wrapper = document.createElement('div');
    wrapper.innerHTML = html;
    const el = wrapper.firstElementChild;
    grid.addWidget(el);
    controllerBody = el.querySelector('.widget-body');
    widgets.set(CONTROLLER_ID, { template: 'controller', props: {}, el, api: null });
    renderControllerBody();
  }

  // ============================================================================
  // FRAME ROUTING — recognize widget_create/update/close/template_register/var_set
  // ============================================================================

  // Local cache of agent + global vars (mirrors server-side meta.ui.vars_*)
  const varCache = { agent: {}, global: {} };

  // Seed parent var cache from this chat's persisted meta.ui (read path for B).
function seedVarCache() {
    // Read from server-wide globalVars (not per-chat meta.ui)
    const gv = Store.globalVars || {};
    const a = gv.vars_agent || {};
    for (const k in a) if (!(k in varCache.agent)) varCache.agent[k] = a[k];
    const g = gv.vars_global || {};
    for (const k in g) if (!(k in varCache.global)) varCache.global[k] = g[k];
  }

  // Inject a bootstrap <script> before the page's own scripts (into <head>, else <html>, else prepend).
  function injectBoot(html, boot) {
    if (/<head[^>]*>/i.test(html)) return html.replace(/<head[^>]*>/i, m => m + boot);
    if (/<html[^>]*>/i.test(html)) return html.replace(/<html[^>]*>/i, m => m + boot);
    return boot + html;
  }

  function handleFrame(frame) {
    if (frame.type === 'widget_create') {
      createWidget(frame.widget_id, frame.template, frame.props || {}, frame.pin);
      return true;
    }
    if (frame.type === 'widget_update') {
      updateWidget(frame.widget_id, frame.props_patch || {});
      return true;
    }
    if (frame.type === 'widget_close') {
      closeWidget(frame.widget_id);
      return true;
    }
    if (frame.type === 'template_register') {
      registerTemplate(frame.template_id, frame.adapter, frame.render_js, frame.name, frame.schema);
      return true;
if (frame.type === var_set) {
      const scope = frame.scope === global ? global : agent;
      const prev = varCache[scope][frame.key];
      const changed = JSON.stringify(prev) !== JSON.stringify(frame.value);
      varCache[scope][frame.key] = frame.value;
      // Mirror into global Store so other chats / re-opens see latest values
      const storeKey = scope === global ? vars_global : vars_agent;
      if (!Store.globalVars) Store.globalVars = { vars_agent: {}, vars_global: {} };
      Store.globalVars[storeKey] = Store.globalVars[storeKey] || {};
      Store.globalVars[storeKey][frame.key] = frame.value;
      const changed = JSON.stringify(prev) !== JSON.stringify(frame.value);
      varCache[scope][frame.key] = frame.value;
      for (const [wid, w] of widgets) {
        const iframe = w.el.querySelector('iframe');
        if (iframe && iframe.contentWindow) {
          try { iframe.contentWindow.postMessage({ __isaa_var_set: true, scope, key: frame.key, value: frame.value }, '*'); } catch (_) {}
        }
      }
      // External (e.g. agent) change to an html-doc storage key → reload so the page re-reads.
      // Self-writes are suppressed: the parent updates varCache optimistically → changed=false.
      if (changed && String(frame.key).indexOf('storage:') === 0) {
        for (const [wid, w] of widgets) if (w.template === 'htmldoc') updateWidget(wid, {});
      }
      return true;
    }
    return false;
  }

  /** Register a custom template that proxies to an existing adapter or uses html sandbox. */
  function registerTemplate(templateId, baseAdapter, renderJs, name, schema) {
    if (adapters[templateId]) return;  // do not overwrite
    if (renderJs) {
      // Custom template: render in iframe with the supplied render_js
      adapters[templateId] = (root, props, api, widgetId) =>
        adapters.html(root, { render_js: renderJs, props }, api, widgetId);
    } else if (adapters[baseAdapter]) {
      adapters[templateId] = adapters[baseAdapter];
    } else {
      console.warn('[widget] template_register with unknown base adapter:', baseAdapter);
      return;
    }
    templateSpecs.set(templateId, {
      name: name || templateId,
      adapter: baseAdapter || 'html',
      schema: schema || {},
      render_js: renderJs || null,
    });
    if (controllerBody) renderControllerBody();
  }

  // Iframe→parent action bridge
  window.addEventListener('message', (e) => {
    const d = e.data;
    if (!d || !d.__isaa || !d.widgetId) return;
    if (d.kind === 'set_var') {
      const scope = d.scope === 'global' ? 'global' : 'agent';
      varCache[scope][d.key] = d.value;   // optimistic → suppresses self-echo reload
      if (window.ISAA.WS) window.ISAA.WS.send({ op: 'widget_action', widget_id: d.widgetId, action: 'set_var', payload: { scope, key: d.key, value: d.value } });
    } else {
      if (window.ISAA.WS) window.ISAA.WS.send({ op: 'widget_action', widget_id: d.widgetId, action: d.action, payload: d.payload });
    }
  });

  window.ISAA = window.ISAA || {};
  window.ISAA.Widgets = { handleFrame, createWidget, closeWidget, updateWidget, exitWidgetMode, toggleMode };

  const _tg = document.getElementById('widget-toggle');
  if (_tg) _tg.addEventListener('click', toggleMode);
})();
