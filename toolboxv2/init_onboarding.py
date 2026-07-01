"""
Bulletproof programmatic onboarding for ToolBoxV2.

One entry point::

    from toolboxv2 import init
    app = init(profile="mini", headless=True)   # ready to login + chat, no CLI/web

Guarantees on first call:
  1. A persistent JWT/cookie secret exists (auto-generated, never crashes login).
  2. Missing env vars are filled from env-template with sane defaults.
  3. A manifest for the chosen profile exists (desktop / server / colab / mini).
  4. Offline-by-default DB for headless profiles -> no MinIO retry storm.

Everything is overridable: pass env={...} or manifest=<path|dict>.

ponytail: this whole module is the lazy seam. It does NOT refactor the App's
data-dir handling or the runner system -- it only sets env + writes config
*before* the first get_app(), which is the one ordering that matters.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional, Mapping, Union

from toolboxv2 import tb_root_dir

# --- profile aliases -> real ProfileType + sane infra defaults -----------------
# ponytail: friendly names map onto the existing enum; no new enum value (that
# would ripple through the runner map). Add a profile here, not in the runner.
_PROFILE_PRESETS: dict[str, dict] = {
    # headless, SQLite-only, no services -> Colab / CI / notebooks
    "mini":    {"profile": "local",    "db": "LC", "offline": True,  "services": []},
    "colab":   {"profile": "local",    "db": "LC", "offline": True,  "services": []},
    # desktop: gui+cli+tray, local dict DB, workers available
    "desktop": {"profile": "consumer", "db": "LC", "offline": True,  "services": ["workers"]},
    # server: full stack, services + autostart expected
    "server":  {"profile": "server",   "db": "RR", "offline": False, "services": ["workers", "db"]},
}

# --- env keys we guarantee a usable value for (minimal "good defaults") --------
# Only the ones that crash or badly degrade if empty. Everything else is left to
# env-template / the user. ponytail: minimal set, grow only when something breaks.
_REQUIRED_DEFAULTS: dict[str, str] = {
    "TB_ENV": "development",
    "IS_OFFLINE_DB": "true",
    "DB_MODE_KEY": "LC",
    "APP_BASE_URL": "http://localhost:8000",

}

_SECRET_KEYS = ("TB_JWT_SECRET", "TB_COOKIE_SECRET")


def _writable_env_path() -> Path:
    """First writable location for a persisted .env, in priority order."""
    candidates = []
    if os.getenv("TB_DATA_DIR"):
        candidates.append(Path(os.environ["TB_DATA_DIR"]) / ".env")
    candidates.append(tb_root_dir.parent / ".env")   # repo / editable install
    candidates.append(Path.cwd() / ".env")           # site-packages fallback
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            # touch-test writability without clobbering existing content
            if p.exists():
                if os.access(p, os.W_OK):
                    return p
            else:
                p.touch()
                return p
        except OSError:
            continue
    return Path.cwd() / ".env"  # ponytail: last resort, cwd is ~always writable


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.split("#", 1)[0].strip().strip('"').strip("'")
    return out


def _append_env(path: Path, kv: Mapping[str, str]) -> None:
    if not kv:
        return
    existing = _read_env_file(path)
    new = {k: v for k, v in kv.items() if k not in existing}
    if not new:
        return
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size and not path.read_text(encoding="utf-8").endswith("\n"):
            f.write("\n")
        for k, v in new.items():
            f.write(f'{k}="{v}"\n')


def _template_defaults() -> dict[str, str]:
    """Pull non-empty defaults straight out of env-template (if shipped).

    ponytail: skip security/secret keys -- those must stay user-controlled.
    TB_R_KEY in particular feeds device-key derivation; injecting the template
    value silently re-keys the device and bricks an existing device.enc.
    """
    tpl = tb_root_dir.parent / "env-template"
    if not tpl.exists():
        return {}
    _skip = {"TB_R_KEY", "TB_JWT_SECRET", "TB_COOKIE_SECRET", "TOKEN_SECRET",
             "CLUSTER_SECRET", "ADMIN_UI_PASSWORD", "STRIPE_SECRET_KEY"}
    out: dict[str, str] = {}
    for line in tpl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k in _skip:
            continue
        v = v.split("#", 1)[0].strip().strip('"').strip("'")
        if v:  # only fill what the template actually has a value for
            out[k] = v
    return out


def ensure_secret(persist: bool = True) -> str:
    """Return a stable JWT/cookie secret; generate + persist on first use.

    ponytail: single source of truth. config.get_jwt_secret() falls back here,
    so login never crashes on a fresh install. 64 hex bytes = 512 bits.
    """
    for k in _SECRET_KEYS:
        if os.getenv(k):
            return os.environ[k]

    env_path = _writable_env_path()
    persisted = _read_env_file(env_path)
    for k in _SECRET_KEYS:
        if persisted.get(k):
            os.environ.setdefault(k, persisted[k])
            return persisted[k]

    secret = secrets.token_hex(64)
    os.environ["TB_JWT_SECRET"] = secret
    os.environ.setdefault("TB_COOKIE_SECRET", secret)
    if persist:
        _append_env(env_path, {"TB_JWT_SECRET": secret, "TB_COOKIE_SECRET": secret})
    return secret


def _resolve_preset(profile: str) -> dict:
    p = (profile or "mini").lower()
    if p not in _PROFILE_PRESETS:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {sorted(_PROFILE_PRESETS)}"
        )
    return _PROFILE_PRESETS[p]


def build_manifest(profile: str = "mini", save: bool = True,
                   path: Optional[Union[str, Path]] = None):
    """Create (and optionally save) a profile manifest. Returns the TBManifest."""
    from toolboxv2.utils.manifest import ManifestLoader, TBManifest

    preset = _resolve_preset(profile)
    manifest = TBManifest(
        manifest_version="1.0.0",
        app={"name": "ToolBoxV2", "environment": "development",
             "profile": preset["profile"]},
        mods={
            "installed": {"CloudM": "^0.1.0", "DB": "^0.0.3"},
            "init_modules": ["CloudM", "DB"],
            "open_modules": ["CloudM.AuthHelper", "CloudM.Auth"],
        },
        database={"mode": preset["db"]},
        services={"enabled": preset["services"]},
    )
    if save:
        loader = ManifestLoader(str(path) if path else str(tb_root_dir))
        if path:
            loader._manifest_path = Path(path)
        loader.save(manifest)
    return manifest


_FAILSAFE_DONE = False


def manifest_exists() -> bool:
    """True if a real tb-manifest.yaml is present at the active location."""
    return (tb_root_dir / "tb-manifest.yaml").exists()


def prepare_mini_failsafe() -> bool:
    """Idempotent, non-recursive 'mini headless' prepare for bare get_app().

    Fires only when NO manifest exists yet -- e.g. a user calls get_app()
    straight from their own code/menu before any onboarding ran. Sets the
    secret + env defaults + a mini manifest so the App boots offline and login
    works. Does NOT construct the App (the caller does that) -> no recursion.

    Returns True if it ran the prepare, False if it was a no-op.
    ponytail: cheapest possible guard. One bool, one file check.
    """
    global _FAILSAFE_DONE
    if _FAILSAFE_DONE or manifest_exists():
        return False
    _FAILSAFE_DONE = True
    try:
        init(profile="mini", headless=True, create_app=False)
    except Exception:
        # Never let onboarding prep break get_app(); App has its own defaults.
        return False
    return True


def _open_browser_later(url: str, delay: float = 1.2) -> None:
    """Open the browser shortly after the server starts (best-effort)."""
    import threading, webbrowser
    threading.Timer(delay, lambda: webbrowser.open(url)).start()


def _start_tray(app, url: str) -> bool:
    """Start the system-tray icon if pystray is available. Best-effort."""
    try:
        import pystray  # noqa: F401
    except Exception:
        return False
    try:
        import os as _os
        _os.environ.setdefault("TB_TRAY_URL", url)
        from toolboxv2.utils.extras.fallback_tray import run_fallback_tray
        import threading
        threading.Thread(target=run_fallback_tray, args=(app,), daemon=True).start()
        return True
    except Exception:
        return False


def _serve_local_ui(host: str = "127.0.0.1", port: int = 5000) -> bool:
    """Serve the local web UI (FastTB) on a loopback port. Blocks. Returns
    False immediately if its deps are missing so the caller can fall through.
    """
    try:
        from waitress import serve
        from toolboxv2.utils.workers.fast.local_ui import app as local_ui_app
        from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    except Exception:
        return False
    handler = FastTBHandler(local_ui_app)
    serve(handler.as_wsgi_app(enable_ws=False), host=host, port=port)
    return True


def launch_ui(app, prefer: str = "auto", host: str = "127.0.0.1",
              port: int = 5000) -> str:
    """headless=False entry: route the user into a usable UI with a fallback
    chain. Returns which surface was launched.

    Chain (prefer='auto'):
      1. Tauri desktop app   -- if an installed binary is found (tb gui).
      2. Local web UI        -- FastTB served on loopback + browser + tray.
      3. Local CLI dashboard -- terminal, always available.

    ponytail: no new UI is built. Each rung reuses an existing entry and falls
    through on absence (missing binary / missing waitress / no display).
    """
    prefer = (prefer or "auto").lower()

    # 1) Tauri (desktop). Only if a binary is actually installed -- never block
    #    on a multi-minute download here; that's an explicit `tb gui` action.
    if prefer in ("auto", "tauri", "gui"):
        try:
            from toolboxv2.utils.clis.tauri_cli import get_installed_app_path, run_app
            if get_installed_app_path():
                run_app(with_worker=True, http_port=port, ws_port=port + 1,
                        download_if_missing=False)
                return "tauri"
        except Exception:
            pass

    # 2) Local web UI + tray + browser.
    if prefer in ("auto", "web", "browser"):
        url = f"http://{host}:{port}"
        _start_tray(app, url)
        _open_browser_later(url)
        if _serve_local_ui(host, port):   # blocks until Ctrl+C
            return "web"
        # waitress missing -> fall through to CLI.

    # 3) Local CLI dashboard -- always works.
    try:
        from toolboxv2.utils.clis.local_cli import main as cli_main
        import asyncio
        res = cli_main()
        if asyncio.iscoroutine(res):
            asyncio.get_event_loop().run_until_complete(res)
    except Exception:
        pass
    return "cli"


def _open_browser_later_noop():  # kept for symmetry / future tests
    pass


def init(profile: str = "mini",
         headless: bool = True,
         env: Optional[Mapping[str, str]] = None,
         manifest: Optional[Union[str, Path, dict]] = None,
         create_app: bool = True,
         persist_secret: bool = True):
    """Bulletproof one-call onboarding.

    Args:
        profile: mini | colab | desktop | server.
        headless: no CLI/web; just return a ready App (or None if create_app=False).
        env: extra/override env vars, applied before the App boots.
        manifest: a path/dict to use instead of the profile default.
        create_app: if False, only prepare env+secret+manifest and return None.
        persist_secret: write the generated secret to .env (set False for ephemeral CI).

    Returns:
        The booted App singleton, or None when create_app=False.
    """
    preset = _resolve_preset(profile)
    global _FAILSAFE_DONE
    _FAILSAFE_DONE = True   # an explicit init() satisfies the get_app fail-safe

    # 1) secret first -- the one hard crash without it.
    ensure_secret(persist=persist_secret)

    # 2) env defaults: template -> required -> profile -> caller override.
    #    os.environ.setdefault keeps anything the user already exported.
    layered: dict[str, str] = {}
    layered.update(_template_defaults())
    layered.update(_REQUIRED_DEFAULTS)
    layered["IS_OFFLINE_DB"] = "true" if preset["offline"] else "false"
    layered["DB_MODE_KEY"] = preset["db"]
    if env:
        layered.update({k: str(v) for k, v in env.items()})
    for k, v in layered.items():
        if env and k in env:
            os.environ[k] = str(v)        # explicit caller override wins
        else:
            os.environ.setdefault(k, str(v))

    # 3) manifest: custom or profile default.
    from toolboxv2.utils.manifest import ManifestLoader, TBManifest
    loader = ManifestLoader(str(tb_root_dir))
    if isinstance(manifest, dict):
        loader.save(TBManifest.model_validate(manifest))
    elif manifest:
        # caller-provided file: load + re-save into the active location.
        loader.save(loader.load(str(manifest)))
    elif not (tb_root_dir / "tb-manifest.yaml").exists():
        build_manifest(profile, save=True)

    if not create_app:
        return None

    # 4) boot the singleton with env already in place.
    from toolboxv2 import get_app
    app = get_app(f"init.{profile}")

    # 5) headless=False -> route the user into a UI (Tauri -> web -> CLI).
    #    Blocks until the chosen surface exits.
    if not headless:
        launch_ui(app)

    return app


# ----------------------------------------------------------------------------- #
def _demo() -> None:
    """Runnable self-check: assert the invariants init() promises."""
    import tempfile

    # isolate: ephemeral data dir, clean secret env
    tmp = tempfile.mkdtemp(prefix="tb_init_demo_")
    os.environ["TB_DATA_DIR"] = tmp
    for k in _SECRET_KEYS:
        os.environ.pop(k, None)

    # presets resolve, unknown rejected
    assert _resolve_preset("colab")["offline"] is True
    try:
        _resolve_preset("nope"); assert False, "should reject unknown profile"
    except ValueError:
        pass

    # secret autogen + persistence
    s1 = ensure_secret(persist=True)
    assert len(s1) == 128 and os.environ["TB_JWT_SECRET"] == s1
    env_file = Path(tmp) / ".env"
    assert env_file.exists() and "TB_JWT_SECRET" in env_file.read_text()
    # second call is stable (reads back, no new secret)
    os.environ.pop("TB_JWT_SECRET"); os.environ.pop("TB_COOKIE_SECRET", None)
    s2 = ensure_secret(persist=True)
    assert s2 == s1, "secret must be stable across calls"

    # env layering: profile offline flag applied, caller override wins
    init(profile="colab", create_app=False, env={"APP_BASE_URL": "http://x:1"},
         persist_secret=False)
    assert os.environ["IS_OFFLINE_DB"] == "true"
    assert os.environ["APP_BASE_URL"] == "http://x:1"

    # manifest builder produces a valid profile mapping
    m = build_manifest("server", save=False)
    assert m.app.profile.value == "server"
    assert m.database.mode.value == "RR"

    print("ponytail init self-check: OK")


if __name__ == "__main__":
    _demo()
