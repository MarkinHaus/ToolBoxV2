"""T3/F6/F7: Nutzerzustand-Pfade fuer Wheel-Installationen.

Stdlib-only, KEINE toolboxv2-Imports (wird von __init__.py ganz oben benutzt).
Regeln (docs/devdocs/taskmap_stable_tauri_2026-11.md, Abschnitt T3):
- TB_INSTALL_DIR  -> Basis fuer .data/.config bei paketierten Installationen
- TB_MANIFEST_DIR -> Basis fuer tb-manifest.yaml (Overrides alles)
- Source-Checkout -> alles bleibt im Paketordner (tb_root_dir)
"""
import os
import sys
from pathlib import Path


def is_packaged_install() -> bool:
    """True, wenn dieses Paket aus site-packages/dist-packages laeuft."""
    return any(p in ("site-packages", "dist-packages")
               for p in Path(__file__).resolve().parts)


def _default_user_dir() -> Path:
    """Exakt wie installer.ps1 Get-DefaultPath / installer.sh appdata_path."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "toolboxv2"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "toolboxv2"
    return Path.home() / ".local" / "share" / "toolboxv2"


def _installed_root():
    """Install dir derived from where this package lives.

    The installer (installer.sh / installer.ps1) puts the package under
    <install>/uv-tools/... (uv mode) or <install>/.venv/... (venv mode) and
    writes <install>/install.manifest. That marker is required, so a plain
    project venv is never mistaken for an installation.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name in ("uv-tools", ".venv"):
            root = parent.parent
            return root if (root / "install.manifest").is_file() else None
    return None


def user_state_dir() -> Path:
    """THE install dir of a packaged install (one rule for installer + core).

    TB_INSTALL_DIR wins, then the directory the installer actually used
    (derived from the package location, so a custom path chosen in the
    installer is found without any env var), then the per-user default.
    """
    env = os.environ.get("TB_INSTALL_DIR", "").strip()
    if env:
        return Path(env)
    return _installed_root() or _default_user_dir()


def _source_root() -> Path:
    """Source checkout: the package dir (tb_root_dir seam keeps tests patchable)."""
    tb = sys.modules.get("toolboxv2")
    root = getattr(tb, "tb_root_dir", None)
    return Path(root) if root is not None else Path(__file__).parent


def env_file() -> Path:
    """THE one .env path.

    packaged install -> <install dir>/.env (the file the installer creates)
    source checkout  -> repository root .env
    TB_DATA_DIR and cwd are NOT .env locations.
    """
    if is_packaged_install():
        return user_state_dir() / ".env"
    # source checkout: _state_dir.py lives in <repo>/toolboxv2/ -> repo root
    return Path(__file__).resolve().parent.parent / ".env"


def state_root() -> Path:
    """THE one root that holds .data/, .config/ and .info/.

    TB_DATA_DIR (explicit, e.g. the Docker volume) -> packaged: install dir
    -> source checkout: the package dir (unchanged historic layout).
    Exactly one level of .data/.config/.info is appended below this root.
    """
    env = os.environ.get("TB_DATA_DIR", "").strip()
    if env:
        return Path(env)
    if is_packaged_install():
        return user_state_dir()
    return _source_root()


def data_root() -> Path:
    """<state root>/.data -- holds <app id>/, blob_cache/, cli_sessions/."""
    return state_root() / ".data"


def info_dir() -> Path:
    """<state root>/.info -- pids, device key, services."""
    return state_root() / ".info"


def ensure_user_data_env() -> None:
    """F6-Hook: bei Paket-Installationen .data/.config in den Nutzerordner lenken.

    utils/toolbox.py haengt selbst ".data" an TB_DATA_DIR an.
    """
    if os.environ.get("TB_DATA_DIR"):
        return
    if not is_packaged_install():
        return
    d = user_state_dir()
    os.environ["TB_DATA_DIR"] = str(d)
    os.makedirs(d, exist_ok=True)


def manifest_base_dir() -> Path:
    env = os.environ.get("TB_MANIFEST_DIR", "").strip()
    if env:
        return Path(env)
    if is_packaged_install():
        return user_state_dir()
    return _source_root()
