"""Top-level package for ToolBox."""
import os
import sys
from pathlib import Path


try:
    from .utils.system.main_tool import MainTool, get_version_from_pyproject
except ImportError:
    MainTool = get_version_from_pyproject = None

__author__ = "Markin Hausmanns"
__email__ = 'Markinhausmanns@gmail.com'

__init_cwd__ = init_cwd = Path.cwd()
__tb_root_dir__ = tb_root_dir = Path(__file__).parent
os.makedirs(__tb_root_dir__ / 'dist', exist_ok=True)
__version__ = get_version_from_pyproject() if get_version_from_pyproject is not None else "0.1.25"
ToolBox_over: str = "root"
from dotenv import load_dotenv
load_dotenv(str(__tb_root_dir__.parent/".env"))
# =============================================================================
# Phase 1 — Feature Loader  (unpack ZIPs → .installed markers)
# =============================================================================
try:
    from .feature_loader import ensure_features_loaded, get_feature_status
    _feature_load_results = ensure_features_loaded()
except ImportError:
    _feature_load_results = {}
    ensure_features_loaded = None
    get_feature_status = None

# =============================================================================
# Phase 2 — Feature Manager  (reads feature.yaml enabled-flags + env override)
# =============================================================================
_feature_manager = None
FeatureManager = None

try:
    from .utils.system.feature_manager import FeatureManager
    _features_dir = tb_root_dir / "features"
    if _features_dir.exists():
        _feature_manager = FeatureManager(features_dir=str(_features_dir))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# _feature_enabled()
#
# Fallback ohne FeatureManager: core/cli=True, alles andere=False.
# Mit FM: liest feature.yaml + TB_ENABLED_FEATURES env-override.
# ---------------------------------------------------------------------------
_CORE_DEFAULTS = frozenset({"core", "cli", "mini"})

def _feature_enabled(name: str) -> bool:
    """
    Prüfe ob Feature aktiv ist.

    Auflösungsreihenfolge:
      1. FeatureManager (feature.yaml + env TB_ENABLED_FEATURES)
      2. Fallback: core/cli/mini=True, alles andere=False
    """
    if _feature_manager is None:
        return name in _CORE_DEFAULTS
    return _feature_manager.enabled(name)


# =============================================================================
# _auto_install_deps()
#
# Installiert fehlende pip-Pakete für ein Feature.
# Wird nur aufgerufen wenn auto_install Flag aktiv.
# =============================================================================
def _auto_install_deps(feature_name: str) -> bool:
    """
    Installiere fehlende pip/uv Dependencies für ein Feature.

    Liest deps aus FeatureManager.features[name].dependencies.
    Nutzt uv wenn verfügbar, sonst pip.

    Returns:
        True wenn Installation erfolgreich.
    """
    if _feature_manager is None:
        return False
    feature = _feature_manager.features.get(feature_name)
    if not feature or not feature.dependencies:
        return False

    import subprocess
    deps = feature.dependencies

    # uv verfügbar?
    try:
        r = subprocess.run(
            [sys.executable, "-m", "uv", "--version"],
            capture_output=True, text=True
        )
        use_uv = r.returncode == 0
    except Exception:
        use_uv = False

    cmd = (
        [sys.executable, "-m", "uv", "pip", "install"] if use_uv
        else [sys.executable, "-m", "pip", "install"]
    )
    cmd.extend(deps)

    print(
        f"\n[toolboxv2] Auto-installing missing deps for '{feature_name}':\n"
        f"  {' '.join(deps)}\n"
        f"  Using: {'uv' if use_uv else 'pip'}",
        file=sys.stderr,
    )

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(
            f"[toolboxv2] ✓ Dependencies for '{feature_name}' installed.",
            file=sys.stderr,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[toolboxv2] ✗ Auto-install failed for '{feature_name}':\n"
            f"  {e.stderr.strip() if e.stderr else e}",
            file=sys.stderr,
        )
        return False


def _should_auto_install(feature_name: str) -> bool:
    """
    Prüfe ob Auto-Install für dieses Feature konfiguriert ist.

    Auflösung (Manifest → per-feature override → global flag):
      1. manifest.features.{name}.auto_install  (per-feature)
      2. manifest.features.auto_install_deps     (global)
      3. Fallback: False
    """
    try:
        from .utils.manifest.loader import ManifestLoader
        loader = ManifestLoader(tb_root_dir)
        if loader.exists():
            manifest = loader.load(resolve_env=False)
            return manifest.features.should_auto_install(feature_name)
    except Exception:
        pass
    return False


# =============================================================================
# _guarded_import()
#
# Importiert ein Symbol nur wenn das zugehörige Feature aktiv ist.
#
# Fehlerbehandlung:
#   - Feature disabled       → None, kein Lärm
#   - ImportError + auto_install aktiv → deps installieren, retry, dann weiter
#   - ImportError + auto_install inaktiv → klare Fehlermeldung + sys.exit(1)
# =============================================================================
def _guarded_import(
    module_path: str,
    attr: str,
    feature: str,
    *,
    abort_on_error: bool = False,
    package: str = None,
):
    """
    Importiere `attr` aus `module_path` nur wenn `feature` aktiv ist.

    Args:
        module_path:    Vollständiger Modul-Pfad, z.B. "toolboxv2.utils.workers"
        attr:           Attribut-Name, z.B. "cli_http_worker". Leer → ganzes Modul.
        feature:        Feature-Name, z.B. "web"
        abort_on_error: True → sys.exit(1) wenn kein auto_install und Import schlägt fehl.
                        False → None zurückgeben (Standard für top-level init).
        package:        Für relative Imports.

    Returns:
        Importiertes Objekt / Modul oder None.
    """
    if not _feature_enabled(feature):
        return None

    import importlib

    def _do_import():
        mod = importlib.import_module(module_path, package=package)
        return getattr(mod, attr) if attr else mod

    # 1. Versuch
    try:
        return _do_import()
    except ImportError as e:
        first_error = e

    # 2. Auto-Install-Entscheidung
    if _should_auto_install(feature):
        if _auto_install_deps(feature):
            # Retry nach Installation
            try:
                # Modul-Cache leeren damit neu importiert wird
                for key in list(sys.modules.keys()):
                    if module_path.split(".")[0] in key:
                        del sys.modules[key]
                return _do_import()
            except ImportError as e:
                # Auch nach Install fehlgeschlagen → abort
                print(
                    f"\n[toolboxv2] ✗ Feature '{feature}' failed after auto-install.\n"
                    f"  Module:  {module_path}.{attr}\n"
                    f"  Error:   {e}\n"
                    f"  Action:  tb manifest disable {feature}\n",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            # Install fehlgeschlagen → abort
            print(
                f"\n[toolboxv2] ✗ Feature '{feature}': auto-install failed.\n"
                f"  Fix:  pip install toolboxv2[{feature}]\n"
                f"  Then: tb manifest enable {feature}\n",
                file=sys.stderr,
            )
            sys.exit(1)

    # 3. Kein Auto-Install → Feature deaktivieren + Fehlermeldung
    if _feature_manager is not None and feature in _feature_manager.features:
        try:
            _feature_manager.features[feature].enabled = False
            _feature_manager._persist_feature_yaml(feature)
            _feature_manager._log(
                "warning",
                f"Auto-disabled '{feature}': import failed ({first_error}). "
                f"Fix: pip install toolboxv2[{feature}]",
            )
        except Exception:
            pass

    if abort_on_error:
        print(
            f"\n[toolboxv2] ✗ Feature '{feature}' enabled but dependencies are missing.\n"
            f"  Module:  {module_path}.{attr}\n"
            f"  Error:   {first_error}\n"
            f"  Fix:     pip install toolboxv2[{feature}]\n"
            f"  Or:      tb manifest disable {feature}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return None


# =============================================================================
# Core Imports  (feature=core — immer)
# =============================================================================

try:
    from .utils.toolbox import App
except ImportError as e:
    import traceback; traceback.print_exc()
    App = None

try:
    from .utils.singelton_class import Singleton
except ImportError:
    Singleton = None

try:
    from .utils.system.file_handler import FileHandler
except ImportError:
    FileHandler = None

try:
    from .utils.extras.Style import Style, Spinner, remove_styles
except ImportError:
    Style = Spinner = remove_styles = None

try:
    from .utils.system.types import (
        AppArgs, AppType, MainToolType,
        ToolBoxError, ToolBoxInfo, ToolBoxInterfaces,
        ToolBoxResult, ToolBoxResultBM,
    )
except ImportError:
    (AppArgs, MainToolType, AppType, ToolBoxError,
     ToolBoxInterfaces, ToolBoxResult, ToolBoxInfo, ToolBoxResultBM) = [None] * 8

try:
    from .utils.system.tb_logger import get_logger, setup_logging
except ImportError:
    get_logger = setup_logging = None

try:
    from .utils.system.getting_and_closing_app import get_app
except ImportError:
    class _DummyApp:
        id = "toolbox-main"
        def __str__(self): return f"<App id='{self.id}'>"
        __repr__ = __str__
    def get_app(): return _DummyApp()

try:
    from .utils.system.types import Result, ApiResult, RequestData
except ImportError:
    Result = ApiResult = RequestData = None

try:
    from .utils.security.cryp import Code
except ImportError:
    Code = None

try:
    from .utils.system import all_functions_enums as TBEF
except ImportError:
    TBEF = {}

try:
    from .flows import flows_dict
except ImportError:
    flows_dict = {}

# =============================================================================
# Desktop Guard  (show_console — feature=desktop, PyQt6)
# =============================================================================

show_console = _guarded_import(
    "toolboxv2.utils.extras.show_and_hide_console",
    "show_console",
    feature="desktop",
    # abort_on_error=False: kein crash wenn PyQt6 fehlt, nur None
)

# =============================================================================
# Mods  (core-mods immer, feature-mods bedingt)
# =============================================================================

MODS_ERROR = None
try:
    import toolboxv2.mods
    from toolboxv2.mods import *
except ImportError as e:
    MODS_ERROR = e
except Exception as e:
    print(f"WARNING: mods import error: {e}", file=sys.stderr)
    MODS_ERROR = e

# =============================================================================
# Optionale Extras
# =============================================================================

try:
    from .utils.tbx.setup import TBxSetup
except ImportError:
    TBxSetup = None

try:
    from .utils.extras.profiler import profile_code
except ImportError:
    profile_code = None

# =============================================================================
# __all__
# =============================================================================

__all__ = [
    "App", "ToolBox_over", "MainTool", "FileHandler",
    "Style", "Spinner", "remove_styles",
    "AppArgs", "setup_logging", "get_logger",
    "flows_dict", "get_app", "TBEF",
    "Result", "ApiResult", "RequestData", "Code",
    "init_cwd", "tb_root_dir", "__init_cwd__", "__version__",
    "MainToolType", "AppType", "ToolBoxError", "ToolBoxInterfaces",
    "ToolBoxResult", "ToolBoxInfo", "ToolBoxResultBM",
    # Feature API
    "_feature_enabled", "_feature_manager", "_guarded_import",
    "_auto_install_deps", "_should_auto_install",
    # Conditional (None wenn Feature disabled)
    "show_console",
    "TBxSetup", "profile_code",
]
