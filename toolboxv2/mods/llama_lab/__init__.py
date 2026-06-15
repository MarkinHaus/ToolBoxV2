# file: toolboxv2/mods/llama_lab/__init__.py
"""llama_lab — interactive ToolBoxV2 mod to run local LLM / embedding /
multimodal / Omni models on llama.cpp efficiently for agent harnesses.

Covers: (1) backend-aware llama.cpp install, (2) HF GGUF browser + manager
with per-card hardware run-estimation, (3) optional llama-bench autotune,
(4) hosting via models.ini with single/mass serving modes + llama-swap router.

Reuses the existing CLI helpers (menu_select_async, Spinner, Style). Served
endpoints are plain OpenAI-compatible llama-server, ready to be registered as a
provider in the existing llm_router / Gateway.

Run:  tb -c llama_lab cli
"""

import json
import os
import platform
from pathlib import Path

from toolboxv2 import get_app

from toolboxv2.utils.clis.cli_input import menu_select_async
from toolboxv2 import Spinner, Style

from . import backend as be
from . import bench as bn
from . import hub
from . import hw as hwmod
from . import serve as sv

export = get_app(from_="LlamaLab.Export").tb
Name = "llama_lab"
version = get_app(from_="LlamaLab.Export").version


# --------------------------------------------------------------- config ----

def _data_dir(app) -> Path:
    for attr in ("data_dir", "appdata", "app_data", "info_dir"):
        d = getattr(app, attr, None)
        if d:
            return Path(d) / "llama_lab"
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", str(Path.home()))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "toolboxv2" / "llama_lab"


def _cfg_path(app) -> Path:
    return _data_dir(app) / "config.json"


def _load_cfg(app) -> dict:
    p = _cfg_path(app)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"bin_dir": "", "backend": "", "models_dir": "",
            "ini_path": str(_data_dir(app) / "models.ini")}


def _save_cfg(app, cfg: dict):
    p = _cfg_path(app)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))


def _apply_models_dir(cfg: dict):
    """Point both the HF downloader and llama.cpp's own -hf cache at models_dir."""
    if cfg.get("models_dir"):
        os.environ["LLAMA_CACHE"] = cfg["models_dir"]


def _ask(prompt: str, default: str = "") -> str:
    raw = input(Style.CYAN(f"  {prompt} ") + (Style.GREY(f"[{default}] ") if default else ""))
    return raw.strip() or default


async def _confirm(prompt: str) -> bool:
    return await menu_select_async([("y", "yes"), ("n", "no")], title=prompt) == "y"


# ----------------------------------------------------------- install flow ----

async def _flow_install(app, cfg):
    info = hwmod.probe()
    print(Style.GREEN2("  " + hwmod.summary(info)))
    method = await menu_select_async(
        [("prebuilt", "Prebuilt release (recommended)"),
         ("source", "Build from source / fork"),
         ("brew", "Homebrew (macOS)")],
        title="Install llama.cpp", hint="binaries come per-backend; pick the matching one")
    if method is None:
        return
    backends = [("cuda", "CUDA (NVIDIA)"), ("hip", "HIP/ROCm (AMD)"),
                ("vulkan", "Vulkan (AMD/Intel/any)"), ("sycl", "SYCL (Intel)"),
                ("metal", "Metal (Apple)"), ("cpu", "CPU only")]
    backend = await menu_select_async(
        backends, title="Backend",
        start=[b[0] for b in backends].index(info.suggested_backend)
        if info.suggested_backend in [b[0] for b in backends] else 0,
        hint=f"detected suggestion: {info.suggested_backend}")
    if backend is None:
        return
    dest = _data_dir(app) / "llama.cpp"
    try:
        with Spinner("installing llama.cpp", symbols="d") as sp:
            if method == "brew":
                bin_dir = be.install_brew(sp)
            elif method == "source":
                repo = _ask("fork repo URL (blank=official):")
                ref = _ask("branch/tag (blank=default):")
                bin_dir = be.install_source(backend, dest, repo, ref, sp)
            else:
                bin_dir = be.install_prebuilt(info, backend, dest, sp)
        ver = be.verify(bin_dir)
    except Exception as e:
        print(Style.RED(f"  install failed: {e}"))
        return
    cfg["bin_dir"], cfg["backend"] = str(bin_dir), backend
    _save_cfg(app, cfg)
    print(Style.GREEN(f"  ok: {ver}\n  bin: {bin_dir}"))


# ------------------------------------------------------------- model flow ----

def _add_model_to_ini(cfg, name, model_path, modality, mmproj=None):
    ini = Path(cfg["ini_path"])
    cp = sv.load_ini(ini)
    if not cp["*"]:
        cp["*"].update(sv.default_section())
    sec = sv.default_section(modality)
    sec["model"] = str(model_path)
    sec["modality"] = modality
    if mmproj:
        sec["mmproj"] = str(mmproj)
    cp[name] = {k: v for k, v in sec.items() if k not in cp["*"] or k in ("model", "mmproj", "modality")}
    sv.save_ini(cp, ini)


async def _flow_browse(app, cfg):
    info = hwmod.probe()
    query = _ask("HF search (e.g. 'qwen3 gguf', 'embed', 'omni'):")
    with Spinner("searching HF", symbols="d"):
        repos = hub.search(query, limit=20)
    if not repos:
        print(Style.YELLOW("  no GGUF repos found"))
        return
    repo_id = await menu_select_async([(r, r) for r in repos], title="Models")
    if repo_id is None:
        return
    with Spinner("loading model card", symbols="d"):
        c = hub.card(repo_id)
    print(Style.BOLD(f"  {c.repo_id}") +
          Style.GREY(f"  ↓{c.downloads}  ♥{c.likes}  [{c.modality}]  {' '.join(c.tags[:6])}"))
    opts = []
    for fname, size in c.gguf_files:
        est = hub.estimate(size, fname, info)
        opts.append(((fname, size),
                     f"{fname}  {size/1e9:.1f}GB  ->  {est['tier']} "
                     f"(need ~{est['need_gb']}GB, bound: {est['bound']}){est['note']}"))
    if not opts:
        print(Style.YELLOW("  repo has no standalone GGUF files"))
        return
    pick = await menu_select_async(opts, title="GGUF files — fit on THIS hardware",
                                   hint="🟢 full GPU · 🟡/🟠 offload · 🔴 won't fit")
    if pick is None:
        return
    fname, _ = pick
    if not await _confirm(f"download {repo_id}/{fname}?"):
        return
    try:
        with Spinner("downloading", symbols="b") as sp:
            path = hub.download(repo_id, fname, with_mmproj=c.has_mmproj, spinner=sp)
    except Exception as e:
        print(Style.RED(f"  download failed: {e}"))
        return
    print(Style.GREEN(f"  saved: {path}"))
    if await _confirm("add to models.ini?"):
        name = _ask("section name:", default=repo_id.split("/")[-1].lower())
        mm = next((p for p in Path(path).parent.glob("*mmproj*.gguf")), None)
        _add_model_to_ini(cfg, name, path, c.modality, mm)
        print(Style.GREEN(f"  added [{name}] to {cfg['ini_path']}"))


async def _flow_installed(app, cfg):
    items = hub.installed()
    if not items:
        print(Style.YELLOW("  no installed GGUFs"))
        return
    opts = [((p, s), f"{p.name}  {s/1e9:.1f}GB  {p.parent}") for p, s in items]
    pick = await menu_select_async(opts, title="Installed models", hint="select to manage")
    if pick is None:
        return
    path, _ = pick
    action = await menu_select_async(
        [("ini", "Add to models.ini"), ("rm", "Remove file"), ("back", "Back")],
        title=path.name)
    if action == "rm" and await _confirm(f"delete {path.name}?"):
        print(Style.GREEN("  removed") if hub.remove(path) else Style.RED("  failed"))
    elif action == "ini":
        name = _ask("section name:", default=path.stem.lower())
        mm = next((p for p in path.parent.glob("*mmproj*.gguf")), None)
        _add_model_to_ini(cfg, name, path, "omni" if mm else "text", mm)
        print(Style.GREEN(f"  added [{name}]"))


async def _flow_models(app, cfg):
    while True:
        loc = cfg.get("models_dir") or str(hub.cache_dir())
        ch = await menu_select_async(
            [("browse", "Browse / download from HF"),
             ("installed", "Installed models (manage / remove)"),
             ("location", f"Download location  ({loc})"),
             ("back", "Back")], title="Models")
        if ch in (None, "back"):
            return
        if ch == "location":
            new = _ask("models dir (blank = default cache):", default=cfg.get("models_dir"))
            cfg["models_dir"] = new
            _save_cfg(app, cfg)
            _apply_models_dir(cfg)
            print(Style.GREEN(f"  download location: {new or hub.cache_dir()}"))
            continue
        await (_flow_browse if ch == "browse" else _flow_installed)(app, cfg)


# -------------------------------------------------------------- bench flow ----

async def _flow_bench(app, cfg):
    if not cfg.get("bin_dir"):
        print(Style.YELLOW("  install llama.cpp first"))
        return
    items = hub.installed()
    if not items:
        print(Style.YELLOW("  no installed GGUFs to bench"))
        return
    pick = await menu_select_async([((p, s), f"{p.name}  {s/1e9:.1f}GB") for p, s in items],
                                   title="Benchmark which model?")
    if pick is None:
        return
    gguf = pick[0]
    info = hwmod.probe()
    try:
        with Spinner("llama-bench sweep (threads + ubatch)", symbols="e"):
            res = bn.autotune(Path(cfg["bin_dir"]), gguf, info)
    except Exception as e:
        print(Style.RED(f"  bench failed: {e}"))
        return
    print(Style.GREEN(f"  threads={res['threads']} ubatch={res['ubatch']}  "
                      f"(tg {res['tg_ts']} t/s, pp {res['pp_ts']} t/s)"))
    if await _confirm("write these into a models.ini section?"):
        name = _ask("section name:", default=gguf.stem.lower())
        cp = sv.load_ini(Path(cfg["ini_path"]))
        if name not in cp:
            cp[name] = {"model": str(gguf), "modality": "text"}
        cp[name]["threads"] = str(res["threads"])
        cp[name]["ubatch-size"] = str(res["ubatch"])
        sv.save_ini(cp, Path(cfg["ini_path"]))
        print(Style.GREEN(f"  updated [{name}]"))


# -------------------------------------------------------------- serve flow ----

async def _flow_serve(app, cfg):
    if not cfg.get("bin_dir"):
        print(Style.YELLOW("  install llama.cpp first"))
        return
    cp = sv.load_ini(Path(cfg["ini_path"]))
    models = [s for s in cp.sections() if s != "*"]
    if not models:
        print(Style.YELLOW(f"  models.ini empty ({cfg['ini_path']}) — add a model first"))
        return
    mode = await menu_select_async(
        [("single", "single — one instance, max prefill+token speed"),
         ("mass", "mass — N slots, many concurrent requests")],
        title="Serving mode")
    if mode is None:
        return
    parallel = 4
    if mode == "mass":
        parallel = int(_ask("parallel slots:", default="4") or "4")

    data_dir = _data_dir(app)
    host = _ask("host:", default="0.0.0.0")
    port = int(_ask("port:", default="8080") or "8080")
    bin_dir = Path(cfg["bin_dir"])

    if len(models) > 1 and sv.has_swap():
        if await _confirm("multiple models + llama-swap found — start ROUTER (auto load/unload)?"):
            swap_cfg = data_dir / "llama-swap.yaml"
            sv.write_swap_config(bin_dir, cp, swap_cfg, mode=mode, parallel=parallel)
            rec = sv.start_router(cp, data_dir, swap_cfg, host, port)
            print(Style.GREEN(f"  router on http://{host}:{port}  (model=<section> in payload)"
                              f"  pid={rec['pid']}"))
            return

    name = await menu_select_async([(m, f"{m}  [{sv.merged(cp, m).get('modality','text')}]")
                                    for m in models], title="Model to serve")
    if name is None:
        return
    rec = sv.start(bin_dir, cp, name, data_dir, host, port, mode, parallel)
    print(Style.GREEN(f"  [{name}] serving on http://{host}:{port}/v1  "
                      f"(mode={mode}, pid={rec['pid']})"))


async def _flow_running(app, cfg):
    live = sv.running(_data_dir(app))
    if not live:
        print(Style.YELLOW("  nothing running"))
        return
    pick = await menu_select_async(
        [(n, f"{n}  :{r['port']}  pid={r['pid']}  mode={r['mode']}") for n, r in live.items()],
        title="Running servers", hint="select to stop")
    if pick is None:
        return
    if await _confirm(f"stop {pick}?"):
        print(Style.GREEN("  stopped") if sv.stop(_data_dir(app), pick) else Style.RED("  failed"))


# ------------------------------------------------------------- entrypoints ----

@export(name="cli", mod_name=Name, version=version, helper="Interactive local-model lab")
async def cli(app):
    cfg = _load_cfg(app)
    _apply_models_dir(cfg)
    print(Style.VIOLET2("  llama_lab — local LLM / embed / multimodal / omni on llama.cpp"))
    while True:
        ch = await menu_select_async(
            [("install", "Install / update llama.cpp"),
             ("models", "Browse / manage models (HF)"),
             ("bench", "Benchmark a model (autotune flags)"),
             ("serve", "Serve (single / mass / router)"),
             ("running", "Running servers"),
             ("exit", "Exit")],
            title="llama_lab", hint=f"backend: {cfg.get('backend') or 'not installed'}")
        if ch in (None, "exit"):
            print(Style.GREY("  bye"))
            return
        flow = {"install": _flow_install, "models": _flow_models, "bench": _flow_bench,
                "serve": _flow_serve, "running": _flow_running}[ch]
        await flow(app, cfg)
        cfg = _load_cfg(app)


@export(name="serve", mod_name=Name, version=version, helper="Headless start a models.ini section")
async def _serve(app, name: str, mode: str = "single", port: int = 8080,
                host: str = "0.0.0.0", parallel: int = 4):
    cfg = _load_cfg(app)
    _apply_models_dir(cfg)
    if not cfg.get("bin_dir"):
        return {"error": "llama.cpp not installed; run `tb -c llama_lab cli`"}
    cp = sv.load_ini(Path(cfg["ini_path"]))
    if name not in cp.sections():
        return {"error": f"no section [{name}] in {cfg['ini_path']}"}
    rec = sv.start(Path(cfg["bin_dir"]), cp, name, _data_dir(app), host, port, mode, parallel)
    return {"serving": name, "url": f"http://{host}:{port}/v1", "pid": rec["pid"], "mode": mode}


@export(name="status", mod_name=Name, version=version, helper="List running local servers")
async def status(app):
    return sv.running(_data_dir(app))


@export(name="web", mod_name=Name, version=version, helper="Run the FastTB web app (gateway/playground/admin)")
async def web(app, host: str = "0.0.0.0", port: int = 8000):
    """Start the llama_lab web app standalone (waitress). For production mount
    `webapp.app` into HTTPWorker via worker.run(fast_tb_app=...)."""
    from .webapp import run
    import asyncio
    print(Style.GREEN(f"  llama_lab web on http://{host}:{port}  (/, /user/, /admin/, /playground/, /docs/)"))
    await asyncio.to_thread(run, host, port)
    return {"served": f"http://{host}:{port}"}
