#!/usr/bin/env python3
"""
toolboxv2/utils/workers/fast/local_ui.py - Local Default UI (Static + Auto-Mount)

PRIMARY JOBS:
1. Asset-Server: Serves dist/ statically (including index.html/mainPagen.html).
2. Auto-Mount-Hub: Features mount themselves via feature.yaml deklaratively.

Refactored 2026-09-01: Removed HTMX legacy code.
"""

import asyncio
import importlib
import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from toolboxv2 import Result, get_app, tb_root_dir
from toolboxv2.utils.workers.fast.endpoint import dist_dir
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.extras.blobs import BlobFile

app = FastTB(title="ToolBox Local")
app.serve_login_assets = True
app.inject_style = False
# Mount dist/ as /dist
app.mount_dist()

# =============================================================================
# Helpers
# =============================================================================

def _ensure_local(request: ParsedRequest) -> Optional[Result]:
    """Refuse anything that isn't loopback."""
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
        return Result.default_user_error(
            info="This UI is local-only. Use the public site for remote access.",
            exec_code=403,
        )
    return None

def _auto_mount_features(fast_app: FastTB):
    """Scan all features and mount their UIs based on feature.yaml."""
    features_dir = tb_root_dir / "features"
    if not features_dir.exists():
        return

    for fyaml in features_dir.glob("*/feature.yaml"):
        try:
            with open(fyaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            mount_cfg = data.get("mount")
            if mount_cfg and isinstance(mount_cfg, dict):
                prefix = mount_cfg.get("prefix")
                module_path = mount_cfg.get("module")
                app_var = mount_cfg.get("app_var", "app")

                if prefix and module_path:
                    try:
                        # Attempt to load the module
                        mod = importlib.import_module(module_path)
                        sub_app_raw = getattr(mod, app_var)

                        # Handle factory functions
                        if callable(sub_app_raw) and not isinstance(sub_app_raw, FastTB):
                            sub_app = sub_app_raw()
                        else:
                            sub_app = sub_app_raw

                        if not isinstance(sub_app, FastTB):
                             print(f"[local_ui] Warning: {module_path}.{app_var} is not a FastTB instance")
                             continue

                        # mount_app expects (other_app, prefix, source)
                        fast_app.mount_app(sub_app, prefix, source=f"feature:{data.get('name', 'unknown')}")
                        print(f"[local_ui] Mounted {data.get('name', 'unknown')} at {prefix}")
                    except Exception as e:
                        print(f"[local_ui] Failed to mount feature {data.get('name')}: {e}")
        except Exception:
            continue

def get_installation_state() -> Dict[str, Any]:
    """Get installation and first-start state (BlobFile + plaintext mirror).

    FIX (bug-tauri-firstrun): BlobFile ist Backend-abhaengig (MOBILE/SQLite unter
    TAURI_ENV, sonst dev/Server) — der State kann je Start-Kontext verschwinden.
    Reihenfolge: BlobFile -> plaintext-Mirror (tb_root_dir/system/) ->
    Installations-Heuristik. Nie blind first_run=True (Wizard-Schleife).
    """
    try:
        with BlobFile("system/installation_state.json", "r") as f:
            state = f.read_json()
            if state: return state
    except Exception:
        pass
    try:
        from toolboxv2.utils.workers.fast.wizard_api import _load_state_with_mirror
        state = _load_state_with_mirror()
        if state:
            return state
    except Exception:
        pass
    try:
        from toolboxv2.utils.workers.fast.wizard_api import _looks_installed
        installed = _looks_installed()
    except Exception:
        installed = False
    return {"installed": installed, "first_run": not installed}

# =============================================================================
# Routes
# =============================================================================

@app.get("/", auth=False)
async def root(request: ParsedRequest):
    err = _ensure_local(request)
    if err is not None: return err

    # Check if first run
    state = get_installation_state()
    if state.get("first_run", True):
        return await welcome(request)

    # SPA Fallback / Entry point
    d = dist_dir()
    if d:
        # Priority for mainPagen.html if it exists, otherwise index.html
        for name in ["mainPagen.html", "index.html"]:
            if (d / name).exists():
                from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
                return FastTBHandler._serve_static_file(str(d / name))

    return "ToolBox Local UI - dist/index.html not found."

@app.get("/welcome", auth=False)
async def welcome(request: ParsedRequest):
    """First-run setup hub: profile choice + config steps + GUI/dist install.

    Wizard-API: GET/POST /api/wizard/* (siehe wizard_api.py) - nur lokale Geraete.
    """
    state = get_installation_state()
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Welcome to ToolBox - Setup</title>
    <link rel="stylesheet" href="/dist/tbjs/main.css">
    <script src="/dist/tbjs/main.js"></script>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 860px; margin: 0 auto; padding: 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
        .card { border: 1px solid rgba(128,128,128,.3); border-radius: 10px; padding: 14px; cursor: pointer; }
        .card:hover { border-color: #4a9eff; }
        .card.active { border-color: #4a9eff; background: rgba(74,158,255,.08); }
        .card h3 { margin: 0 0 6px; font-size: 1em; }
        .card p { margin: 0; font-size: .85em; opacity: .7; }
        section { margin: 28px 0; }
        h2 { font-size: 1.1em; }
        label { display: block; margin: 10px 0 2px; font-size: .9em; }
        input, select { width: 100%; box-sizing: border-box; padding: 6px; border-radius: 6px;
                        border: 1px solid rgba(128,128,128,.4); background: transparent; color: inherit; }
        button { margin: 16px 8px 0 0; padding: 8px 18px; border-radius: 8px;
                 border: 1px solid #4a9eff; background: #4a9eff; color: #fff; cursor: pointer; }
        button.secondary { background: transparent; color: inherit; border-color: rgba(128,128,128,.5); }
        #status { margin-top: 12px; font-size: .9em; white-space: pre-wrap; }
        .err { color: #ff6b6b; } .ok { color: #51cf6e; }
    </style>
</head>
<body class="glass-v3">
    <h1>Welcome to ToolBox</h1>
    <div id="app"></div>
    <script>
    let wizardState = null, profiles = [];

    async function api(path, opts) {
        const r = await fetch(path, opts ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(opts)} : undefined);
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || r.status);
        return data;
    }
    const el = (html) => { const d = document.createElement('div'); d.innerHTML = html; return d.firstElementChild; };

    async function loadProfiles() {
        profiles = (await api('/api/wizard/profiles')).profiles;
        const wrap = document.getElementById('profile-grid');
        wrap.innerHTML = '';
        profiles.forEach(p => {
            const c = el(`<div class="card" data-id="${p.id}"><h3>${p.label}</h3><p>${p.desc}</p></div>`);
            c.onclick = () => chooseProfile(p.id);
            wrap.appendChild(c);
        });
    }

    async function chooseProfile(id) {
        try {
            const res = await api('/api/wizard/profile', {profile: id});
            document.querySelectorAll('#profile-grid .card').forEach(c =>
                c.classList.toggle('active', c.dataset.id === id));
            wizardState = res;
            renderSteps();
            setStatus(`Profile: ${id} (init preset: ${res.init_preset})`, 'ok');
        } catch (e) { setStatus(String(e), 'err'); }
    }

    function renderSteps() {
        const wrap = document.getElementById('steps');
        wrap.innerHTML = '';
        wizardState.steps.forEach(s => {
            if (s.type === 'llm') { wrap.appendChild(llmCard()); return; }
            if (s.type === 'features') { wrap.appendChild(featuresCard()); return; }
            const card = el(`<div class="card" style="cursor:default"><h3>${s.title}</h3></div>`);
            s.fields.forEach(f => card.appendChild(fieldInput(f)));
            const b = el('<button class="secondary">Apply</button>');
            b.onclick = () => applyStep(s);
            card.appendChild(b);
            wrap.appendChild(card);
        });
        wrap.appendChild(el('<button id="save-btn">Save configuration</button>'));
        document.getElementById('save-btn').onclick = save;
        wrap.appendChild(el('<button class="secondary" id="gui-btn">Install GUI app</button>'));
        document.getElementById('gui-btn').onclick = installGui;
        wrap.appendChild(el('<button class="secondary" id="dist-btn">Fetch web dist (server)</button>'));
        document.getElementById('dist-btn').onclick = showDist;
    }

    function fieldInput(f) {
        const w = el('<div></div>');
        w.appendChild(el(`<label>${f.label}</label>`));
        let inp;
        if (f.type === 'bool') {
            inp = el(`<select data-f="${f.name}"><option ${f.value?'selected':''}>true</option><option ${f.value?'':'selected'}>false</option></select>`);
        } else if (f.type === 'choice' || f.type === 'multi') {
            inp = el(`<select data-f="${f.name}" ${f.type==='multi'?'multiple size="3"':''}>` +
                f.choices.map(c => `<option ${f.type!=='multi'&&f.value===c?'selected':''}>${c}</option>`).join('') + '</select>');
        } else {
            inp = el(`<input data-f="${f.name}" type="${f.type==='password'?'password':'text'}" value="${f.value ?? ''}">`);
        }
        w.appendChild(inp);
        return w;
    }

    function collectValues(card) {
        const values = {};
        card.querySelectorAll('[data-f]').forEach(i => {
            values[i.dataset.f] = i.type === 'text' || i.type === 'password' ? i.value
                : i.multiple ? [...i.selectedOptions].map(o => o.value)
                : i.value === 'true';
        });
        return values;
    }

    async function applyStep(s) {
        const values = collectValues(event.target.closest('.card'));
        try {
            const res = await api('/api/wizard/step', {step: s.id, values});
            wizardState = res;
            setStatus(`Step '${s.id}' saved to draft`, 'ok');
        } catch (e) { setStatus(String(e), 'err'); }
    }

    function llmCard() {
        const card = el('<div class="card" style="cursor:default"><h3>LLM Providers</h3></div>');
        api('/api/wizard/llm').then(d => {
            const sel = el('<select id="llm-prov"></select>');
            d.providers.forEach(p => sel.appendChild(el(`<option value="${p.id}">${p.name}${p.configured?' ✓':''}</option>`)));
            card.appendChild(el('<label>Provider</label>'));
            card.appendChild(sel);
            card.appendChild(el('<label>API key</label>'));
            card.appendChild(el('<input id="llm-key" type="password" placeholder="sk-...">'));
            card.appendChild(el('<label>Base URL (optional)</label>'));
            card.appendChild(el('<input id="llm-url" type="text" placeholder="http://localhost:20128/v1">'));
            const b1 = el('<button class="secondary">Validate</button>');
            b1.onclick = async () => {
                try {
                    const r = await api('/api/wizard/llm/validate', {
                        provider: sel.value, api_key: document.getElementById('llm-key').value,
                        base_url: document.getElementById('llm-url').value});
                    setStatus(r.valid ? `Valid. Models: ${r.models.join(', ')}` : `Invalid: ${r.message}`,
                              r.valid ? 'ok' : 'err');
                } catch (e) { setStatus(String(e), 'err'); }
            };
            const b2 = el('<button class="secondary">Save key</button>');
            b2.onclick = async () => {
                const prov = d.providers.find(p => p.id === sel.value);
                const values = {};
                if (prov.key_vars.length) values[prov.key_vars[0]] = document.getElementById('llm-key').value;
                if (prov.url_var) values[prov.url_var] = document.getElementById('llm-url').value;
                try { await api('/api/wizard/llm', {provider: sel.value, values}); setStatus('Key saved to .env draft', 'ok'); }
                catch (e) { setStatus(String(e), 'err'); }
            };
            card.appendChild(b1); card.appendChild(b2);
        }).catch(e => card.appendChild(el(`<p>${e}</p>`)));
        return card;
    }

    function featuresCard() {
        const card = el('<div class="card" style="cursor:default"><h3>Modules</h3></div>');
        api('/api/wizard/features').then(d => {
            if (!d.features.length) { card.appendChild(el('<p>None available</p>')); return; }
            const sel = el('<select multiple size="4" id="feat-sel"></select>');
            d.features.forEach(f => { if (!f.installed) sel.appendChild(el(`<option value="${f.id}">${f.id}</option>`)); });
            card.appendChild(sel);
            const b = el('<button class="secondary">Install selected</button>');
            b.onclick = async () => {
                const feats = [...sel.selectedOptions].map(o => o.value);
                try { setStatus(JSON.stringify((await api('/api/wizard/features', {features: feats})).results), 'ok'); }
                catch (e) { setStatus(String(e), 'err'); }
            };
            card.appendChild(b);
        });
        return card;
    }

    async function installGui() {
        try {
            await api('/api/wizard/gui', {});
            const poll = setInterval(async () => {
                const s = await api('/api/wizard/gui');
                if (!s.running) { clearInterval(poll); setStatus(s.error ? `GUI install failed: ${s.error}` :
                    `GUI installed: ${s.app_path} (v${s.version})`, s.error ? 'err' : 'ok'); }
            }, 1500);
            setStatus('GUI download started...');
        } catch (e) { setStatus(String(e), 'err'); }
    }

    async function showDist() {
        try {
            const d = await api('/api/wizard/dist');
            if (!d.assets.length) { setStatus('No dist assets. Available: ' + (d.all_asset_names||[]).join(', '), 'err'); return; }
            const a = d.assets[0];
            await api('/api/wizard/dist', {url: a.url, name: a.name});
            setStatus(`Downloading ${a.name} -> ${d.dist_dir} (background)`);
        } catch (e) { setStatus(String(e), 'err'); }
    }

    async function save() {
        try {
            const r = await api('/api/wizard/save', {apply: true});
            setStatus(`Saved: ${r.manifest}\\nGenerated: ${(r.generated||[]).join(', ') || 'none'}\\nSetup complete - reloading...`, 'ok');
            setTimeout(() => location.href = '/', 2500);
        } catch (e) { setStatus(String(e), 'err'); }
    }

    function setStatus(msg, cls) { const s = document.getElementById('status'); s.textContent = msg; s.className = cls || ''; }

    (async () => {
        const app = document.getElementById('app');
        app.appendChild(el(`<section><h2>1. Choose your profile</h2><div class="grid" id="profile-grid"></div></section>`));
        app.appendChild(el(`<section><h2>2. Configuration</h2><div class="grid" id="steps"></div></section>`));
        app.appendChild(el('<div id="status"></div>'));
        wizardState = await api('/api/wizard/state');
        if (wizardState.profile) renderSteps();
        await loadProfiles();
        localStorage.setItem('tb_installation_state', JSON.stringify(wizardState));
    })();
    </script>
</body>
</html>"""

@app.get("/install", auth=False)
async def install(request: ParsedRequest):
    """Manage-Hub (W1b): Wizard auf bereits installierter TB.

    Erreichbar aus Tauri-Launch-Routing (W1a) + Tray. first_run ->
    302 auf /welcome (Bootstrap gehoert dorthin, nicht hier).
    """
    err = _ensure_local(request)
    if err is not None:
        return err
    state = get_installation_state()
    if state.get("first_run", True):
        return 302, {"Location": "/welcome"}, b""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ToolBox Manager</title>
    <link rel="stylesheet" href="/dist/tbjs/main.css">
    <script src="/dist/tbjs/main.js"></script>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 0 auto; padding: 24px; }}
        .card {{ border: 1px solid rgba(128,128,128,.3); border-radius: 10px; padding: 14px; margin: 12px 0; }}
        .card h3 {{ margin: 0 0 8px; font-size: 1em; }}
        select, input {{ width: 100%; box-sizing: border-box; padding: 6px; border-radius: 6px;
                         border: 1px solid rgba(128,128,128,.4); background: transparent; color: inherit; }}
        button {{ margin: 10px 8px 0 0; padding: 8px 18px; border-radius: 8px;
                 border: 1px solid #4a9eff; background: #4a9eff; color: #fff; cursor: pointer; }}
        button.secondary {{ background: transparent; color: inherit; border-color: rgba(128,128,128,.5); }}
        #status {{ margin-top: 12px; font-size: .9em; white-space: pre-wrap; }}
        .err {{ color: #ff6b6b; }} .ok {{ color: #51cf6e; }}
        .muted {{ opacity: .65; font-size: .88em; }}
    </style>
</head>
<body class="glass-v3">
    <h1>ToolBox Manager</h1>
    <p class="muted">Installierte ToolBox verwalten. Ersteinrichtung: <a href="/welcome">Setup-Assistent</a></p>
    <div id="app"></div>
    <script>
    const INSTALL_STATE = {json.dumps(state)};
    localStorage.setItem('tb_installation_state', JSON.stringify(INSTALL_STATE));

    async function api(path, opts) {{
        const r = await fetch(path, opts ? {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(opts)}} : undefined);
        const data = await r.json().catch(() => ({{}}));
        if (!r.ok) throw new Error(data.error || r.status);
        return data;
    }}
    const el = (html) => {{ const d = document.createElement('div'); d.innerHTML = html; return d.firstElementChild; }};
    function setStatus(msg, cls) {{ const s = document.getElementById('status'); s.textContent = msg; s.className = cls || ''; }}

    async function profileCard() {{
        const card = el('<div class="card"><h3>Profil</h3></div>');
        try {{
            const [d, st] = await Promise.all([api('/api/wizard/profiles'), api('/api/wizard/state')]);
            const sel = el('<select id="mgr-profile"></select>');
            d.profiles.forEach(p => sel.appendChild(el(`<option value="${{p.id}}" ${{p.id===st.profile?'selected':''}}>${{p.label}}</option>`)));
            const b = el('<button>Profil setzen</button>');
            b.onclick = async () => {{
                try {{
                    const res = await api('/api/wizard/profile', {{profile: sel.value}});
                    setStatus(`Profil: ${{sel.value}} (init preset: ${{res.init_preset}}) - Konfiguration im Setup-Assistent.`, 'ok');
                }} catch (e) {{ setStatus(String(e), 'err'); }}
            }};
            card.appendChild(sel); card.appendChild(b);
        }} catch (e) {{ card.appendChild(el(`<p>${{e}}</p>`)); }}
        return card;
    }}

    async function featuresCard() {{
        const card = el('<div class="card"><h3>Module</h3></div>');
        try {{
            const d = await api('/api/wizard/features');
            if (!d.features.length) {{ card.appendChild(el('<p class="muted">Keine optionalen Module verfuegbar.</p>')); return card; }}
            const sel = el('<select multiple size="4"></select>');
            d.features.forEach(f => {{ if (!f.installed) sel.appendChild(el(`<option value="${{f.id}}">${{f.id}}</option>`)); }});
            const b = el('<button class="secondary">Ausgewaehlte installieren</button>');
            b.onclick = async () => {{
                const feats = [...sel.selectedOptions].map(o => o.value);
                if (!feats.length) return setStatus('Nichts ausgewaehlt.', 'err');
                try {{ setStatus(JSON.stringify((await api('/api/wizard/features', {{features: feats}})).results), 'ok'); }}
                catch (e) {{ setStatus(String(e), 'err'); }}
            }};
            card.appendChild(sel); card.appendChild(b);
        }} catch (e) {{ card.appendChild(el(`<p>${{e}}</p>`)); }}
        return card;
    }}

    function guiCard() {{
        const card = el('<div class="card"><h3>Desktop-GUI</h3></div>');
        const b = el('<button class="secondary">GUI-App installieren</button>');
        b.onclick = async () => {{
            try {{
                await api('/api/wizard/gui', {{}});
                const poll = setInterval(async () => {{
                    const s = await api('/api/wizard/gui');
                    if (!s.running) {{
                        clearInterval(poll);
                        setStatus(s.error ? `GUI-Install fehlgeschlagen: ${{s.error}}` : `GUI installiert: ${{s.app_path}} (v${{s.version}})`, s.error ? 'err' : 'ok');
                    }}
                }}, 1500);
                setStatus('GUI-Download gestartet...');
            }} catch (e) {{ setStatus(String(e), 'err'); }}
        }};
        card.appendChild(b);
        return card;
    }}

    function distCard() {{
        const card = el('<div class="card"><h3>Web-Dist (Server)</h3></div>');
        const b = el('<button class="secondary">Dist laden</button>');
        b.onclick = async () => {{
            try {{
                const d = await api('/api/wizard/dist');
                if (!d.assets.length) {{ setStatus('Keine Dist-Assets. Verfuegbar: ' + (d.all_asset_names||[]).join(', '), 'err'); return; }}
                const a = d.assets[0];
                await api('/api/wizard/dist', {{url: a.url, name: a.name}});
                setStatus(`Download ${{a.name}} -> ${{d.dist_dir}} (Hintergrund)`);
            }} catch (e) {{ setStatus(String(e), 'err'); }}
        }};
        card.appendChild(b);
        return card;
    }}

    (async () => {{
        const app = document.getElementById('app');
        app.appendChild(await profileCard());
        app.appendChild(await featuresCard());
        app.appendChild(guiCard());
        app.appendChild(distCard());
        app.appendChild(el(`<p class="muted">State: first_run=${{INSTALL_STATE.first_run}}, installed=${{INSTALL_STATE.installed}}</p>`));
        app.appendChild(el('<div id="status"></div>'));
    }})();
    </script>
</body>
</html>"""

# =============================================================================
# Wizard-API (MUSS vor dem spa_fallback-Catch-all registriert werden)
# =============================================================================
from toolboxv2.utils.workers.fast.wizard_api import register_wizard_routes

register_wizard_routes(app)

# /live sub-mount (Dashboard-Redirect auf Worker, muss vor Catch-all)
from toolboxv2.utils.workers.fast.live_proxy import register_live_routes

register_live_routes(app)


@app.get("/{path:path}", auth=False)
async def spa_fallback(request: ParsedRequest, path: str):
    """Catch-all for SPA routing."""
    d = dist_dir()
    if d:
        target = d / path
        if target.is_file():
             from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
             return FastTBHandler._serve_static_file(str(target))

        # SPA Fallback to index.html
        if (d / "index.html").exists():
            from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
            return FastTBHandler._serve_static_file(str(d / "index.html"))

    return Result.default_user_error(info=f"Not found: {path}", exec_code=404)

# =============================================================================
# Initialization
# =============================================================================

_auto_mount_features(app)

# Wire-in helper
def get_handler():
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    return FastTBHandler(app)

if __name__ == "__main__":
    from waitress import serve
    h = get_handler()
    print("Starting local_ui on http://127.0.0.1:8467")
    serve(h.as_wsgi_app(enable_ws=False), host="127.0.0.1", port=8467)
