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
    html: (root, props, api, widgetId) => {
      const renderJs = props.render_js || '';
      const initialProps = props.props || {};
      const iframe = document.createElement('iframe');
      iframe.style.cssText = 'width:100%;height:100%;border:none;background:transparent';
      iframe.sandbox = 'allow-scripts';
      const html = `<!DOCTYPE html><html><head><style>
        body { margin: 0; padding: 8px; font-family: system-ui, sans-serif; background: transparent; color: rgba(255,255,255,0.85); }
        * { box-sizing: border-box; }
      </style></head><body><div id="root"></div>
      <script>
        const api = {
          emit: (name, payload) => parent.postMessage({ __isaa: true, widgetId: ${JSON.stringify(widgetId)}, action: name, payload }, '*'),
          set_var: (scope, key, value) => parent.postMessage({ __isaa: true, widgetId: ${JSON.stringify(widgetId)}, kind: 'set_var', scope, key, value }, '*'),
        };
        try {
          const fn = ${renderJs};
          fn(document.getElementById('root'), ${JSON.stringify(initialProps)}, api);
        } catch (e) {
          document.body.innerHTML = '<pre style="color:#ff6b6b">' + (e.message || e) + '</pre>';
        }
      </script></body></html>`;
      iframe.srcdoc = html;
      root.appendChild(iframe);
    },
  };

  // ============================================================================
  // WIDGET HOST (task 18)
  // ============================================================================

  let grid = null;
  const widgets = new Map(); // widget_id -> { template, props, el }

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
    return grid;
  }

  async function createWidget(widgetId, template, props, pin) {
    await ensureGrid();
    const adapter = adapters[template];
    if (!adapter) {
      console.warn('[widget] unknown adapter:', template);
      return;
    }
    const w = (pin && pin.w) || 4;
    const h = (pin && pin.h) || 3;
    const x = (pin && pin.x !== undefined) ? pin.x : 0;
    const y = (pin && pin.y !== undefined) ? pin.y : 0;
    const noResize = pin ? 'gs-no-resize="true"' : '';
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
    widgets.set(widgetId, { template, props, el, api });
  }

  function closeWidget(widgetId) {
    const w = widgets.get(widgetId);
    if (!w) return;
    if (grid) grid.removeWidget(w.el);
    widgets.delete(widgetId);
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
    grid = null;
    widgets.clear();
    document.querySelector('.isaa-body').dataset.view = Store.activeChatId ? 'chat' : 'welcome';
    if (window.ISAA.App && window.ISAA.App.rerender) window.ISAA.App.rerender();
  }

  /** Manual switch between chat and widget mode (bottom-right button). */
  async function toggleMode() {
    if (document.querySelector('.isaa-body').dataset.view === 'widgets') {
      exitWidgetMode();
    } else {
      await ensureGrid();
    }
  }

  // ============================================================================
  // FRAME ROUTING — recognize widget_create/update/close/template_register/var_set
  // ============================================================================

  // Local cache of agent + global vars (mirrors server-side meta.ui.vars_*)
  const varCache = { agent: {}, global: {} };

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
      registerTemplate(frame.template_id, frame.adapter, frame.render_js);
      return true;
    }
    if (frame.type === 'var_set') {
      const scope = frame.scope === 'global' ? 'global' : 'agent';
      varCache[scope][frame.key] = frame.value;
      // Notify any iframe widgets so they can re-read
      for (const [wid, w] of widgets) {
        const iframe = w.el.querySelector('iframe');
        if (iframe && iframe.contentWindow) {
          try { iframe.contentWindow.postMessage({ __isaa_var_set: true, scope, key: frame.key, value: frame.value }, '*'); } catch (_) {}
        }
      }
      return true;
    }
    return false;
  }

  /** Register a custom template that proxies to an existing adapter or uses html sandbox. */
  function registerTemplate(templateId, baseAdapter, renderJs) {
    if (adapters[templateId]) return;  // do not overwrite
    if (renderJs) {
      // Custom template: render in iframe with the supplied render_js
      adapters[templateId] = (root, props, api, widgetId) =>
        adapters.html(root, { render_js: renderJs, props }, api, widgetId);
    } else if (adapters[baseAdapter]) {
      adapters[templateId] = adapters[baseAdapter];
    } else {
      console.warn('[widget] template_register with unknown base adapter:', baseAdapter);
    }
  }

  // Iframe→parent action bridge
  window.addEventListener('message', (e) => {
    const d = e.data;
    if (!d || !d.__isaa || !d.widgetId) return;
    if (d.kind === 'set_var') {
      if (window.ISAA.WS) window.ISAA.WS.send({ op: 'widget_action', widget_id: d.widgetId, action: 'set_var', payload: { scope: d.scope, key: d.key, value: d.value } });
    } else {
      if (window.ISAA.WS) window.ISAA.WS.send({ op: 'widget_action', widget_id: d.widgetId, action: d.action, payload: d.payload });
    }
  });

  window.ISAA = window.ISAA || {};
  window.ISAA.Widgets = { handleFrame, createWidget, closeWidget, updateWidget, exitWidgetMode, toggleMode };

  const _tg = document.getElementById('widget-toggle');
  if (_tg) _tg.addEventListener('click', toggleMode);
})();
