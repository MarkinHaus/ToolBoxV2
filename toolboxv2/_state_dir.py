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


def user_state_dir() -> Path:
    env = os.environ.get("TB_INSTALL_DIR", "").strip()
    return Path(env) if env else _default_user_dir()


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
    # Source-Checkout: ueber den tb_root_dir-Seam gehen, damit Tests
    # mock.patch("toolboxv2.tb_root_dir") weiterhin greift.
    tb = sys.modules.get("toolboxv2")
    root = getattr(tb, "tb_root_dir", None)
    if root is not None:
        return Path(root)
    return Path(__file__).parent
