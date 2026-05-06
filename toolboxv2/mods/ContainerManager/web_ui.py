"""
ContainerManager — FastTB Web UI

Single-page FastTB dashboard for Docker container management.

Usage:
    # Inside HTTPWorker (production):
    from toolboxv2.mods.ContainerManager.web_ui import container_ui_app
    worker.run(fast_tb_app=container_ui_app)

    # Standalone:
    uvicorn toolboxv2.mods.ContainerManager.web_ui:container_ui_app --port 9000
"""

from toolboxv2.utils.workers.fast_tb import FastTB

container_ui_app = FastTB(title="ContainerManager")


# ============================================================================
# API proxy endpoints — existing
# ============================================================================

@container_ui_app.get("/cm/api/containers")
async def api_list_containers(user_id: str = "", admin_key: str = "", all: str = "false"):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import list_containers
    result = await list_containers(
        app=get_app(), user_id=user_id or None,
        admin_key=admin_key or None, all=(all == "true")
    )
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.get("/cm/api/container/{container_id}")
async def api_get_container(container_id: str, admin_key: str = ""):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import get_container
    result = await get_container(
        app=get_app(), container_id=container_id, admin_key=admin_key or None
    )
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.post("/cm/api/containers")
async def api_create_container(
    container_type: str = "cli_v4", user_id: str = "",
    admin_key: str = "", image: str = "", command: str = "",
    memory_limit: str = "", cpu_limit: str = "", ssh_public_key: str = ""
):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import create_container
    result = await create_container(
        app=get_app(), container_type=container_type, user_id=user_id or None,
        admin_key=admin_key or None, image=image or None,
        command=command or None, memory_limit=memory_limit or None,
        cpu_limit=cpu_limit or None, ssh_public_key=ssh_public_key or None,
    )
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.post("/cm/api/container/{container_id}/start")
async def api_start(container_id: str, admin_key: str = ""):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import start_container
    result = await start_container(app=get_app(), container_id=container_id, admin_key=admin_key or None)
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.post("/cm/api/container/{container_id}/stop")
async def api_stop(container_id: str, admin_key: str = ""):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import stop_container
    result = await stop_container(app=get_app(), container_id=container_id, admin_key=admin_key or None)
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.post("/cm/api/container/{container_id}/restart")
async def api_restart(container_id: str, admin_key: str = ""):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import restart_container
    result = await restart_container(app=get_app(), container_id=container_id, admin_key=admin_key or None)
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.post("/cm/api/container/{container_id}/delete")
async def api_delete(container_id: str, admin_key: str = "", force: str = "false"):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import delete_container
    result = await delete_container(
        app=get_app(), container_id=container_id,
        admin_key=admin_key or None, force=(force == "true")
    )
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.get("/cm/api/container/{container_id}/logs")
async def api_logs(container_id: str, admin_key: str = "", tail: int = 100):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import container_logs
    result = await container_logs(
        app=get_app(), container_id=container_id,
        admin_key=admin_key or None, tail=tail
    )
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


# ============================================================================
# NEW API proxy endpoints — T1.1–T2.1
# ============================================================================

@container_ui_app.get("/cm/api/docker-health")
async def api_docker_health(admin_key: str = ""):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import docker_health
    result = await docker_health(app=get_app(), admin_key=admin_key or None)
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.get("/cm/api/all-containers")
async def api_all_containers(admin_key: str = ""):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import list_all_docker_containers
    result = await list_all_docker_containers(app=get_app(), admin_key=admin_key or None)
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.post("/cm/api/container/{container_id}/update")
async def api_update(container_id: str, admin_key: str = "", pull: str = "true"):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import update_container
    result = await update_container(
        app=get_app(), container_id=container_id,
        admin_key=admin_key or None, pull=(pull == "true")
    )
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.post("/cm/api/container/{container_id}/reconcile")
async def api_reconcile(container_id: str, admin_key: str = ""):
    from toolboxv2 import get_app
    from toolboxv2.mods.ContainerManager import reconcile_status
    result = await reconcile_status(
        app=get_app(), container_id=container_id,
        admin_key=admin_key or None
    )
    if result.is_error():
        return (400, {"error": str(result.info)})
    return result.get()


@container_ui_app.get("/cm/api/topology")
async def api_topology(admin_key: str = ""):
    """Return network topology: containers + Docker networks for graph rendering."""
    from toolboxv2.mods.ContainerManager.docker_ops import get_docker_ops
    ops = get_docker_ops()
    if not ops.is_available():
        return {"networks": [], "containers": [], "docker_available": False}

    from toolboxv2.mods.ContainerManager import check_admin_key
    if not check_admin_key(admin_key or ""):
        return (400, {"error": "Invalid admin key"})

    all_containers = ops.list_all_containers()
    all_networks = ops.list_networks()

    containers = []
    for c in all_containers:
        containers.append({
            "container_id": c.container_id[:12],
            "container_id_full": c.container_id,
            "name": c.name,
            "status": c.status,
            "is_tb_managed": c.is_tb_managed,
            "networks": c.networks,
        })

    networks = []
    for n in all_networks:
        networks.append({
            "network_id": n.network_id,
            "name": n.name,
            "driver": n.driver,
            "containers": n.containers,
        })

    return {
        "containers": containers,
        "networks": networks,
        "docker_available": True,
    }


# ============================================================================
# HTML UI
# ============================================================================

@container_ui_app.get("/cm")
async def serve_ui():
    return _DASHBOARD_HTML


@container_ui_app.get("/cm/")
async def serve_ui_slash():
    return _DASHBOARD_HTML


# ============================================================================
# HTML TEMPLATE — full rewrite
# ============================================================================

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ContainerManager — Control Center</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root {
  --raw-primary: 55% 0.18 230;
  --raw-success: 65% 0.2 145;
  --raw-warning: 75% 0.18 85;
  --raw-error:   55% 0.22 25;

  --primary: oklch(var(--raw-primary));
  --success: oklch(var(--raw-success));
  --warning: oklch(var(--raw-warning));
  --error:   oklch(var(--raw-error));

  --term-bg: #000000;
  --term-bg-raised: #0a0a0f;
  --term-bg-sunken: #030306;
  --term-border: rgba(255,255,255,0.12);
  --term-selection: color-mix(in oklch, var(--primary) 30%, transparent);

  --term-fg: rgba(255,255,255,0.92);
  --term-fg-dim: rgba(255,255,255,0.5);
  --term-fg-muted: rgba(255,255,255,0.3);

  --font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;
  --term-text-h1: 16px;
  --term-text-h2: 14px;
  --term-text-sm: 11px;
  --term-text-xs: 10px;
  --term-text-base: 12px;
}

html,body { height:100%; background:var(--term-bg); color:var(--term-fg); font-family:var(--font-mono); font-size:var(--term-text-base); line-height:1.4; overflow:hidden; }

.app { display:grid; grid-template-rows:auto auto 1fr auto; height:100vh; }

/* Header */
.header { padding:8px 16px; border-bottom:1px solid var(--term-border); display:flex; align-items:center; gap:16px; background:var(--term-bg); }
.header-title { font-size:var(--term-text-h1); font-weight:500; }
.header-title::before { content:"# "; color:var(--term-fg-muted); }
.auth-bar { display:flex; align-items:center; gap:8px; margin-left:auto; }
.auth-bar input { font:inherit; font-size:var(--term-text-xs); background:var(--term-bg-sunken); border:1px solid var(--term-border); color:var(--term-fg); padding:2px 8px; width:180px; border-radius:0; caret-color:var(--primary); }
.auth-bar input:focus { outline:none; border-color:var(--primary); }
.auth-bar label { color:var(--term-fg-muted); font-size:var(--term-text-xs); }

/* Docker Health Banner */
.docker-health-banner { padding:4px 16px; font-size:var(--term-text-xs); text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid var(--term-border); }
.docker-health-banner.online { background:color-mix(in oklch, var(--success) 15%, var(--term-bg)); color:var(--success); }
.docker-health-banner.offline { background:color-mix(in oklch, var(--error) 20%, var(--term-bg)); color:var(--error); }

/* Tabs */
.tabs { display:flex; border-bottom:1px solid var(--term-border); background:var(--term-bg); }
.tab { padding:6px 16px; color:var(--term-fg-dim); background:transparent; border:none; border-bottom:2px solid transparent; margin-bottom:-1px; cursor:pointer; font:inherit; font-size:var(--term-text-sm); transition:all 100ms linear; }
.tab:hover { color:var(--term-fg); }
.tab.active { color:var(--primary); border-bottom-color:var(--primary); }
.tab.active::before { content:"> "; }

.main { overflow-y:auto; padding:16px; }

/* Status Bar */
.status-bar { height:22px; background:var(--term-bg-raised); border-top:1px solid var(--term-border); font-size:var(--term-text-xs); color:var(--term-fg-dim); display:flex; align-items:center; padding:0 12px; gap:12px; }
.seg+.seg::before { content:"│"; color:var(--term-border); margin-right:12px; }

/* Buttons */
.btn { font:inherit; font-size:var(--term-text-xs); font-weight:500; padding:4px 10px; background:transparent; color:var(--term-fg); border:1px solid var(--term-border); border-radius:0; cursor:pointer; transition:all 100ms linear; }
.btn::before { content:"[ "; color:var(--term-fg-muted); }
.btn::after  { content:" ]"; color:var(--term-fg-muted); }
.btn:hover { border-color:var(--primary); color:var(--primary); }
.btn:hover::before,.btn:hover::after { color:var(--primary); }
.btn:active { background:var(--primary); color:var(--term-bg); }
.btn-sm { padding:2px 6px; font-size:var(--term-text-xs); }
.btn-success:hover { border-color:var(--success); color:var(--success); }
.btn-success:hover::before,.btn-success:hover::after { color:var(--success); }
.btn-warning:hover { border-color:var(--warning); color:var(--warning); }
.btn-warning:hover::before,.btn-warning:hover::after { color:var(--warning); }
.btn-danger:hover { border-color:var(--error); color:var(--error); }
.btn-danger:hover::before,.btn-danger:hover::after { color:var(--error); }
.btn-primary { background:var(--primary); color:var(--term-bg); border-color:var(--primary); }
.btn-primary::before,.btn-primary::after { color:var(--term-bg); }

/* Grid Table */
.tbl { font-size:var(--term-text-sm); border:1px solid var(--term-border); background:var(--term-bg-raised); }
.tbl-hdr { display:grid; grid-template-columns:var(--cols); background:var(--term-bg); border-bottom:1px solid var(--term-border); }
.tbl-hdr .c { padding:6px 12px; color:var(--term-fg-muted); text-transform:uppercase; letter-spacing:1px; font-size:var(--term-text-xs); }
.tbl-row { display:grid; grid-template-columns:var(--cols); border-bottom:1px solid var(--term-border); border-left:2px solid transparent; transition:background 80ms linear; cursor:pointer; }
.tbl-row:hover { background:var(--term-selection); border-left-color:var(--primary); }
.tbl-row .c { padding:4px 12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* Status Glyphs */
.st-run { color:var(--success); }  .st-run::before { content:"● "; }
.st-stop { color:var(--term-fg-muted); }  .st-stop::before { content:"○ "; }
.st-start { color:var(--warning); }  .st-start::before { content:"◐ "; }
.st-err { color:var(--error); }  .st-err::before { content:"✕ "; }

/* Category Tag */
.cat-tb { font-size:var(--term-text-xs); color:var(--primary); border:1px solid var(--primary); padding:1px 4px; }
.cat-ext { font-size:var(--term-text-xs); color:var(--term-fg-muted); border:1px solid var(--term-border); padding:1px 4px; }

/* Forms */
.field { margin-bottom:8px; }
.field label { display:block; color:var(--term-fg-dim); font-size:var(--term-text-xs); margin-bottom:2px; text-transform:uppercase; letter-spacing:1px; }
.field input,.field select,.field textarea { width:100%; font:inherit; font-size:var(--term-text-base); color:var(--term-fg); background:var(--term-bg-sunken); border:1px solid var(--term-border); border-radius:0; padding:6px 10px; caret-color:var(--primary); }
.field input:focus,.field select:focus,.field textarea:focus { outline:none; border-color:var(--primary); background:#000; }
.field select option { background:var(--term-bg); }

/* Stats */
.stats-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:12px; margin-bottom:16px; }
.stat-card { background:var(--term-bg-raised); border:1px solid var(--term-border); padding:12px; }
.stat-label { font-size:var(--term-text-xs); color:var(--term-fg-muted); text-transform:uppercase; letter-spacing:1px; }
.stat-val { font-size:var(--term-text-h1); margin-top:4px; }

/* Cards */
.card { background:var(--term-bg-raised); border:1px solid var(--term-border); padding:12px 16px; margin-bottom:12px; }
.card-title { font-size:var(--term-text-sm); color:var(--term-fg-dim); text-transform:uppercase; letter-spacing:1px; padding-bottom:8px; margin-bottom:8px; border-bottom:1px solid var(--term-border); }

/* Topology */
.topo-wrap { position:relative; width:100%; height:calc(100vh - 200px); min-height:400px; }
#topo { width:100%; height:100%; background:var(--term-bg); }

/* Log Viewer */
.log-area { background:var(--term-bg-sunken); border:1px solid var(--term-border); padding:8px; font-size:var(--term-text-xs); line-height:1.5; overflow-y:auto; max-height:400px; white-space:pre-wrap; color:var(--term-fg-dim); }

/* Modal */
.modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.85); z-index:1000; display:flex; align-items:center; justify-content:center; }
.modal { background:var(--term-bg); border:1px solid var(--primary); min-width:400px; max-width:90vw; }
.modal-hdr { padding:8px 12px; background:var(--primary); color:var(--term-bg); font-size:var(--term-text-sm); font-weight:500; display:flex; justify-content:space-between; }
.modal-body { padding:16px; }
.modal-footer { padding:8px 12px; border-top:1px solid var(--term-border); display:flex; gap:8px; justify-content:flex-end; }
.modal-close { cursor:pointer; background:none; border:none; color:var(--term-bg); font:inherit; }

/* Toast */
.toast-area { position:fixed; top:12px; right:12px; z-index:1100; display:flex; flex-direction:column; gap:8px; }
.toast { font-size:var(--term-text-sm); background:var(--term-bg); border:1px solid var(--term-border); border-left:3px solid var(--toast-color,var(--primary)); padding:8px 12px; min-width:240px; animation:toast-in 200ms linear; }
.toast-title { color:var(--toast-color,var(--primary)); font-size:var(--term-text-xs); text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
@keyframes toast-in { from{opacity:0;transform:translateX(20px)} to{opacity:1;transform:translateX(0)} }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--term-bg); }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.15); }
::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,0.25); }

@media(max-width:640px) {
  .stats-grid { grid-template-columns:1fr 1fr; }
  .tbl-row,.tbl-hdr { font-size:var(--term-text-xs); }
  .auth-bar input { width:120px; }
}
</style>
</head>
<body>

<div class="app" id="app">
  <div>
    <div class="header">
      <span class="header-title">ContainerManager</span>
      <div class="auth-bar">
        <label>ADMIN_KEY</label>
        <input type="password" id="adminKey" placeholder="key..." autocomplete="off">
      </div>
    </div>
    <div class="docker-health-banner offline" id="dockerHealth">● DOCKER STATUS: checking...</div>
    <div class="tabs">
      <button class="tab active" data-tab="dashboard" onclick="switchTab('dashboard',this)">dashboard</button>
      <button class="tab" data-tab="topology" onclick="switchTab('topology',this)">topology</button>
      <button class="tab" data-tab="create" onclick="switchTab('create',this)">create</button>
      <button class="tab" data-tab="logs" onclick="switchTab('logs',this)">logs</button>
    </div>
  </div>

  <div class="main" id="main">

    <!-- Dashboard -->
    <div id="tab-dashboard">
      <div class="stats-grid" id="statsGrid">
        <div class="stat-card"><div class="stat-label">TOTAL</div><div class="stat-val" id="statTotal" style="color:var(--primary)">—</div></div>
        <div class="stat-card"><div class="stat-label">RUNNING</div><div class="stat-val" id="statRunning" style="color:var(--success)">—</div></div>
        <div class="stat-card"><div class="stat-label">STOPPED</div><div class="stat-val" id="statStopped" style="color:var(--term-fg-muted)">—</div></div>
        <div class="stat-card"><div class="stat-label">ERRORS</div><div class="stat-val" id="statErrors" style="color:var(--error)">—</div></div>
        <div class="stat-card"><div class="stat-label">TB MANAGED</div><div class="stat-val" id="statTB" style="color:var(--primary)">—</div></div>
        <div class="stat-card"><div class="stat-label">EXTERNAL</div><div class="stat-val" id="statExt" style="color:var(--term-fg-dim)">—</div></div>
      </div>

      <div class="tbl" style="--cols: 90px 160px 140px 70px 70px 1fr">
        <div class="tbl-hdr">
          <div class="c">STATUS</div><div class="c">NAME</div><div class="c">IMAGE</div>
          <div class="c">PORT</div><div class="c">CAT</div><div class="c">ACTIONS</div>
        </div>
        <div id="containerRows"></div>
      </div>
    </div>

    <!-- Topology -->
    <div id="tab-topology" style="display:none">
      <div class="topo-wrap"><canvas id="topo"></canvas></div>
    </div>

    <!-- Create -->
    <div id="tab-create" style="display:none">
      <div class="card" style="max-width:500px">
        <div class="card-title">NEW CONTAINER</div>
        <div class="field"><label>User ID</label><input id="cUserId" placeholder="usr_..."></div>
        <div class="field"><label>Type</label>
          <select id="cType">
            <option value="cli_v4">cli_v4 — Persistent CLI</option>
            <option value="isaa">isaa — ISAA Agent System</option>
            <option value="custom">custom</option>
          </select>
        </div>
        <div class="field"><label>Image (optional)</label><input id="cImage" placeholder="toolboxv2:latest"></div>
        <div class="field"><label>Command (optional)</label><input id="cCmd" placeholder="override entrypoint"></div>
        <div class="field"><label>Memory Limit</label><input id="cMem" placeholder="512m" value="512m"></div>
        <div class="field"><label>CPU Limit</label><input id="cCpu" placeholder="0.5" value="0.5"></div>
        <div class="field"><label>SSH Public Key (optional)</label><textarea id="cSsh" rows="2" placeholder="ssh-ed25519 AAAAC3..."></textarea></div>
        <div style="margin-top:12px"><button class="btn btn-primary" onclick="createContainer()">deploy</button></div>
      </div>
    </div>

    <!-- Logs -->
    <div id="tab-logs" style="display:none">
      <div style="display:flex;gap:8px;margin-bottom:8px;align-items:center">
        <label style="color:var(--term-fg-muted);font-size:var(--term-text-xs)">CONTAINER_ID</label>
        <input id="logCid" style="font:inherit;font-size:var(--term-text-base);background:var(--term-bg-sunken);border:1px solid var(--term-border);color:var(--term-fg);padding:4px 8px;width:200px;border-radius:0" placeholder="abc123...">
        <button class="btn" onclick="fetchLogs()">fetch</button>
        <label style="color:var(--term-fg-muted);font-size:var(--term-text-xs);margin-left:8px">TAIL</label>
        <input id="logTail" style="font:inherit;font-size:var(--term-text-base);background:var(--term-bg-sunken);border:1px solid var(--term-border);color:var(--term-fg);padding:4px 8px;width:60px;border-radius:0" value="100">
      </div>
      <div class="log-area" id="logOutput">$ waiting for input...</div>
    </div>

  </div>

  <div class="status-bar">
    <span class="seg" id="sbDocker">○ docker: unknown</span>
    <span class="seg" id="sbCount">containers: —</span>
    <span class="seg" id="sbPoll">poll: —</span>
    <span class="seg" id="sbTime"></span>
  </div>
</div>

<div class="toast-area" id="toasts"></div>

<div class="modal-overlay" id="detailModal" style="display:none" onclick="if(event.target===this)closeModal()">
  <div class="modal" style="min-width:500px">
    <div class="modal-hdr"><span id="modalTitle">Container Details</span><button class="modal-close" onclick="closeModal()">✕</button></div>
    <div class="modal-body" id="modalBody"></div>
    <div class="modal-footer" id="modalFooter"></div>
  </div>
</div>

<script>
// ======================================================================
// STATE
// ======================================================================
const KEY = () => document.getElementById('adminKey').value;
const Q = (p) => `admin_key=${encodeURIComponent(KEY())}${p?'&'+p:''}`;

let allContainers = [];     // from /all-containers — TB + external
let dockerOnline = false;
let reconcileIdx = 0;
let pollTimer = null;
let fullRefreshTimer = null;

const POLL_VISIBLE = 5000;
const POLL_HIDDEN = 60000;
const FULL_REFRESH_VISIBLE = 30000;

// ======================================================================
// HELPERS
// ======================================================================
function toast(msg, type='info') {
  const t = document.createElement('div');
  t.className = 'toast';
  const colors = {success:'var(--success)',error:'var(--error)',warning:'var(--warning)',info:'var(--primary)'};
  t.style.setProperty('--toast-color', colors[type]||colors.info);
  t.innerHTML = '<div class="toast-title">'+type+'</div>'+msg;
  document.getElementById('toasts').appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

async function api(url, opts={}) {
  try {
    const r = await fetch(url, opts);
    const d = await r.json();
    if (d.error) { toast(d.error, 'error'); return null; }
    return d;
  } catch(e) { toast(e.message, 'error'); return null; }
}

function stClass(s) {
  if (s==='running') return 'st-run';
  if (s==='exited'||s==='stopped') return 'st-stop';
  if (s==='starting'||s==='restarting'||s==='created'||s==='paused') return 'st-start';
  return 'st-err';
}

// ======================================================================
// TABS
// ======================================================================
function switchTab(name, btn) {
  document.querySelectorAll('[id^="tab-"]').forEach(e=>e.style.display='none');
  document.querySelectorAll('.tab').forEach(e=>e.classList.remove('active'));
  document.getElementById('tab-'+name).style.display='';
  btn.classList.add('active');
  if (name==='topology') loadTopology();
  if (name==='dashboard') fullRefresh();
}

// ======================================================================
// DOCKER HEALTH
// ======================================================================
async function checkDockerHealth() {
  const d = await api('/cm/api/docker-health?'+Q());
  const el = document.getElementById('dockerHealth');
  const sb = document.getElementById('sbDocker');
  if (!d) {
    el.className = 'docker-health-banner offline';
    el.textContent = '✕ DOCKER STATUS: unreachable — all statuses stale';
    sb.innerHTML = '<span style="color:var(--error)">✕ docker: error</span>';
    dockerOnline = false;
    return;
  }
  dockerOnline = d.docker_available;
  if (dockerOnline) {
    el.className = 'docker-health-banner online';
    el.textContent = '● DOCKER ONLINE — '+d.total_containers+' containers ('+d.tb_managed+' managed, '+d.external+' external)';
    sb.innerHTML = '<span style="color:var(--success)">● docker: online</span>';
  } else {
    el.className = 'docker-health-banner offline';
    el.textContent = '✕ DOCKER OFFLINE — all statuses stale';
    sb.innerHTML = '<span style="color:var(--error)">✕ docker: offline</span>';
  }
}

// ======================================================================
// DASHBOARD — full refresh from /all-containers
// ======================================================================
async function fullRefresh() {
  await checkDockerHealth();
  const d = await api('/cm/api/all-containers?'+Q());
  if (!d) return;
  allContainers = d.containers || [];
  renderDashboard();
}

function renderDashboard() {
  const cs = allContainers;
  const run = cs.filter(c=>c.status==='running').length;
  const stop = cs.filter(c=>c.status==='exited'||c.status==='stopped').length;
  const err = cs.filter(c=>!['running','exited','stopped','created'].includes(c.status)).length;
  const tb = cs.filter(c=>c.is_tb_managed).length;
  const ext = cs.length - tb;

  document.getElementById('statTotal').textContent = cs.length;
  document.getElementById('statRunning').textContent = run;
  document.getElementById('statStopped').textContent = stop;
  document.getElementById('statErrors').textContent = err;
  document.getElementById('statTB').textContent = tb;
  document.getElementById('statExt').textContent = ext;
  document.getElementById('sbCount').textContent = 'containers: '+cs.length+' ('+run+' running)';

  const rows = cs.map(c => {
    const isTB = c.is_tb_managed;
    const cid = c.container_id_full || c.container_id;
    const displayId = c.container_id;
    const cat = isTB ? '<span class="cat-tb">TB</span>' : '<span class="cat-ext">EXT</span>';

    // TB containers get full actions; external get none
    let actions = '';
    if (isTB) {
      actions = [
        '<button class="btn btn-sm btn-success" onclick="event.stopPropagation();doAction(&quot;'+cid+'&quot;,&quot;start&quot;)">start</button>',
        '<button class="btn btn-sm" onclick="event.stopPropagation();doAction(&quot;'+cid+'&quot;,&quot;stop&quot;)">stop</button>',
        '<button class="btn btn-sm" onclick="event.stopPropagation();doAction(&quot;'+cid+'&quot;,&quot;restart&quot;)">restart</button>',
        '<button class="btn btn-sm btn-warning" onclick="event.stopPropagation();doUpdate(&quot;'+cid+'&quot;)">update</button>',
        '<button class="btn btn-sm btn-danger" onclick="event.stopPropagation();doAction(&quot;'+cid+'&quot;,&quot;delete&quot;)">del</button>',
      ].join('');
    } else {
      actions = '<span style="color:var(--term-fg-muted);font-size:var(--term-text-xs)">external</span>';
    }

    // Format port display
    let portStr = '—';
    if (c.ports && typeof c.ports === 'object') {
      const vals = Object.values(c.ports).filter(p=>p);
      if (vals.length) portStr = vals.join(',');
    }

    return '<div class="tbl-row" data-cid="'+cid+'" onclick="showDetail(&quot;'+cid+'&quot;)">' +
      '<div class="c"><span class="'+stClass(c.status)+'">'+c.status+'</span></div>' +
      '<div class="c">'+c.name+'</div>' +
      '<div class="c" style="color:var(--term-fg-dim)">'+c.image+'</div>' +
      '<div class="c">'+portStr+'</div>' +
      '<div class="c">'+cat+'</div>' +
      '<div class="c" style="display:flex;gap:3px;flex-wrap:wrap">'+actions+'</div>' +
      '</div>';
  }).join('');

  document.getElementById('containerRows').innerHTML = rows ||
    '<div style="padding:12px;color:var(--term-fg-muted)">no containers found — enter admin key above</div>';
}

// ======================================================================
// ACTIONS
// ======================================================================
async function doAction(cid, action) {
  if (action==='delete' && !confirm('Delete container '+cid.substring(0,12)+'?')) return;
  const force = action==='delete' ? '&force=true' : '';
  await api('/cm/api/container/'+cid+'/'+action+'?'+Q()+force, {method:'POST'});
  toast(action+' → '+cid.substring(0,12), 'success');
  fullRefresh();
}

async function doUpdate(cid) {
  if (!confirm('Update '+cid.substring(0,12)+' to latest image? This will restart the container.')) return;
  const d = await api('/cm/api/container/'+cid+'/update?'+Q('pull=true'), {method:'POST'});
  if (d) {
    toast('Updated: '+d.old_container_id+' → '+d.new_container_id, 'success');
    // container_id changed — must full refresh
    fullRefresh();
  }
}

// ======================================================================
// DETAIL MODAL
// ======================================================================
async function showDetail(cid) {
  const d = await api('/cm/api/container/'+cid+'?'+Q());
  if (!d) return;
  const c = d.container||{};
  const s = d.stats;
  document.getElementById('modalTitle').textContent = c.container_name||cid;

  let statsHtml = '';
  if (s) {
    statsHtml = '<div class="stats-grid" style="margin-top:12px">' +
      '<div class="stat-card"><div class="stat-label">CPU</div><div class="stat-val">'+(s.cpu_percent||0).toFixed(1)+'%</div></div>' +
      '<div class="stat-card"><div class="stat-label">MEM</div><div class="stat-val">'+(s.memory_mb||0).toFixed(0)+' MB</div></div>' +
      '<div class="stat-card"><div class="stat-label">NET RX</div><div class="stat-val">'+((s.network_rx_bytes||0)/1024/1024).toFixed(1)+' MB</div></div>' +
      '<div class="stat-card"><div class="stat-label">NET TX</div><div class="stat-val">'+((s.network_tx_bytes||0)/1024/1024).toFixed(1)+' MB</div></div>' +
      '</div>';
  }

  document.getElementById('modalBody').innerHTML =
    '<div style="display:grid;grid-template-columns:12ch 1fr;gap:4px 12px;font-size:var(--term-text-sm)">' +
    '<span style="color:var(--term-fg-muted)">ID</span><span>'+c.container_id+'</span>' +
    '<span style="color:var(--term-fg-muted)">TYPE</span><span>'+(c.container_type||'—')+'</span>' +
    '<span style="color:var(--term-fg-muted)">IMAGE</span><span>'+(c.image||'—')+'</span>' +
    '<span style="color:var(--term-fg-muted)">USER</span><span>'+(c.user_id||'—')+'</span>' +
    '<span style="color:var(--term-fg-muted)">PORT</span><span>'+(c.port||'—')+'</span>' +
    '<span style="color:var(--term-fg-muted)">SSH</span><span>'+(c.ssh_port||'none')+'</span>' +
    '<span style="color:var(--term-fg-muted)">STATUS</span><span class="'+stClass(d.docker_status||c.status)+'">'+(d.docker_status||c.status)+'</span>' +
    '<span style="color:var(--term-fg-muted)">VOLUME</span><span>'+(c.volume_name||'—')+'</span>' +
    '<span style="color:var(--term-fg-muted)">RESTARTS</span><span>'+(c.restart_count||0)+'</span>' +
    '</div>' + statsHtml;

  document.getElementById('modalFooter').innerHTML =
    '<button class="btn btn-sm" onclick="logFromModal(&quot;'+c.container_id+'&quot;)">view logs</button>' +
    '<button class="btn btn-sm" onclick="closeModal()">close</button>';
  document.getElementById('detailModal').style.display='';
}

function logFromModal(cid) {
  closeModal();
  switchTab('logs', document.querySelector('[data-tab="logs"]'));
  document.getElementById('logCid').value = cid;
  fetchLogs();
}
function closeModal() { document.getElementById('detailModal').style.display='none'; }

// ======================================================================
// CREATE
// ======================================================================
async function createContainer() {
  const params = new URLSearchParams({
    admin_key: KEY(),
    user_id: document.getElementById('cUserId').value,
    container_type: document.getElementById('cType').value,
    image: document.getElementById('cImage').value,
    command: document.getElementById('cCmd').value,
    memory_limit: document.getElementById('cMem').value,
    cpu_limit: document.getElementById('cCpu').value,
    ssh_public_key: document.getElementById('cSsh').value,
  });
  const d = await api('/cm/api/containers?'+params, {method:'POST'});
  if (d) {
    toast('Container created: '+d.container_id, 'success');
    switchTab('dashboard', document.querySelector('[data-tab="dashboard"]'));
  }
}

// ======================================================================
// LOGS
// ======================================================================
async function fetchLogs() {
  const cid = document.getElementById('logCid').value;
  const tail = document.getElementById('logTail').value||100;
  if (!cid) { toast('enter container ID', 'warning'); return; }
  const d = await api('/cm/api/container/'+cid+'/logs?'+Q('tail='+tail));
  if (d) document.getElementById('logOutput').textContent = d.logs || '(empty)';
}

// ======================================================================
// TOPOLOGY — real data from /cm/api/topology
// ======================================================================
async function loadTopology() {
  const d = await api('/cm/api/topology?'+Q());
  if (!d || !d.docker_available) {
    drawTopologyEmpty();
    return;
  }
  drawTopology(d.containers, d.networks);
}

function drawTopologyEmpty() {
  const canvas = document.getElementById('topo');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  canvas.style.width = rect.width+'px';
  canvas.style.height = rect.height+'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = "12px 'IBM Plex Mono', monospace";
  ctx.textAlign = 'center';
  ctx.fillText('Docker offline — no topology data', rect.width/2, rect.height/2);
}

function drawTopology(containers, networks) {
  const canvas = document.getElementById('topo');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  canvas.style.width = rect.width+'px';
  canvas.style.height = rect.height+'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const W = rect.width, H = rect.height;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);

  if (!containers.length) {
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = "12px 'IBM Plex Mono', monospace";
    ctx.textAlign = 'center';
    ctx.fillText('No containers running', W/2, H/2);
    return;
  }

  const success = '#44cc66';
  const dim = 'rgba(255,255,255,0.3)';
  const fg = 'rgba(255,255,255,0.85)';
  const primary = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim()||'#4466ff';

  // Position containers in a circle
  const cx = W/2, cy = H/2, radius = Math.min(W, H) * 0.32;
  const nodes = containers.map((c, i) => {
    const angle = (i / containers.length) * Math.PI * 2 - Math.PI/2;
    return {
      ...c,
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
      r: 14,
      color: c.status==='running' ? success : (c.is_tb_managed ? dim : 'rgba(255,255,255,0.1)'),
    };
  });

  // Build network membership map: network_name -> [container_id_full, ...]
  const netMap = {};
  networks.forEach(n => { netMap[n.name] = n.containers || []; });

  // Draw edges: containers in same network get a line
  ctx.lineWidth = 1;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i+1; j < nodes.length; j++) {
      // Check if they share a network
      const shared = nodes[i].networks && nodes[j].networks &&
        nodes[i].networks.some(n => nodes[j].networks.includes(n));
      if (shared) {
        const a = nodes[i], b = nodes[j];
        ctx.strokeStyle = (a.status==='running'&&b.status==='running')
          ? 'rgba(68,204,102,0.25)' : 'rgba(255,255,255,0.06)';
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
  }

  // Draw nodes
  nodes.forEach(n => {
    ctx.fillStyle = '#0a0a0f';
    ctx.strokeStyle = n.color;
    ctx.lineWidth = n.is_tb_managed ? 2 : 1;
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI*2); ctx.fill(); ctx.stroke();

    ctx.fillStyle = n.is_tb_managed ? fg : dim;
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = 'center';
    ctx.fillText(n.name.substring(0,18), n.x, n.y + n.r + 14);
  });

  // Legend
  ctx.fillStyle = dim;
  ctx.font = "10px 'IBM Plex Mono', monospace";
  ctx.textAlign = 'left';
  ctx.fillText('# network topology — '+containers.length+' container(s), '+networks.length+' network(s)', 12, 16);
}

// ======================================================================
// ROUND-ROBIN RECONCILE POLLING
// ======================================================================
async function reconcileNext() {
  if (!allContainers.length) return;
  // Only reconcile TB-managed containers
  const tbContainers = allContainers.filter(c => c.is_tb_managed);
  if (!tbContainers.length) return;

  if (reconcileIdx >= tbContainers.length) reconcileIdx = 0;
  const c = tbContainers[reconcileIdx];
  reconcileIdx++;

  const cid = c.container_id_full || c.container_id;
  const d = await api('/cm/api/container/'+cid+'/reconcile?'+Q(), {method:'POST'});
  if (d) {
    // Update in-place
    const idx = allContainers.findIndex(x => (x.container_id_full||x.container_id) === cid);
    if (idx !== -1 && allContainers[idx].status !== d.status) {
      allContainers[idx].status = d.status;
      renderDashboard();
    }
    // Update docker status from response
    if (d.docker_available !== undefined) {
      const wasOnline = dockerOnline;
      dockerOnline = d.docker_available;
      if (wasOnline !== dockerOnline) checkDockerHealth();
    }
  }
  document.getElementById('sbPoll').textContent = 'poll: '+new Date().toLocaleTimeString('de-DE');
}

// ======================================================================
// INTELLIGENT POLLING — visibility-based
// ======================================================================
function startPolling() {
  stopPolling();
  const interval = document.hidden ? POLL_HIDDEN : POLL_VISIBLE;
  pollTimer = setInterval(reconcileNext, interval);
  if (!document.hidden) {
    fullRefreshTimer = setInterval(fullRefresh, FULL_REFRESH_VISIBLE);
  }
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  if (fullRefreshTimer) { clearInterval(fullRefreshTimer); fullRefreshTimer = null; }
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopPolling();
    pollTimer = setInterval(reconcileNext, POLL_HIDDEN);
  } else {
    stopPolling();
    fullRefresh();
    startPolling();
  }
});

// ======================================================================
// CLOCK & INIT
// ======================================================================
function tick() {
  document.getElementById('sbTime').textContent = new Date().toLocaleTimeString('de-DE');
}
setInterval(tick, 1000); tick();

fullRefresh();
startPolling();

window.addEventListener('resize', () => {
  if (document.getElementById('tab-topology').style.display !== 'none') loadTopology();
});
</script>
</body>
</html>"""
