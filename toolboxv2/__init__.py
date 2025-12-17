"""Top-level package for ToolBox."""
import os
import sys

# Suppress print statements during import in PyO3 environment
_suppress_output = os.environ.get('PYTHONIOENCODING') == 'utf-8'
if _suppress_output:
    import io
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

try:
    from .utils.toolbox import App
except ImportError as e:
    print(e)
    import traceback
    print(traceback.format_exc())
    App = None
    print("⚠️ Missing: utils.toolbox.App")

try:
    from .utils.singelton_class import Singleton
except ImportError:
    Singleton = None
    print("⚠️ Missing: utils.singelton_class.Singleton")

try:
    from .utils.system.main_tool import MainTool, get_version_from_pyproject
except ImportError:
    MainTool = get_version_from_pyproject = None
    print("⚠️ Missing: utils.system.main_tool")

try:
    from .utils.system.file_handler import FileHandler
except ImportError:
    FileHandler = None
    print("⚠️ Missing: utils.system.file_handler")

try:
    from .utils.extras.Style import Style
except ImportError:
    Style = None
    print("⚠️ Missing: utils.extras.Style.Style")

try:
    from .utils.extras.Style import Spinner
except ImportError:
    Spinner = None
    print("⚠️ Missing: utils.extras.Style.Spinner")

try:
    from .utils.extras.Style import remove_styles
except ImportError:
    remove_styles = None
    print("⚠️ Missing: utils.extras.Style.remove_styles")

try:
    from .utils.system.types import (
        AppArgs,
        AppType,
        MainToolType,
        ToolBoxError,
        ToolBoxInfo,
        ToolBoxInterfaces,
        ToolBoxResult,
        ToolBoxResultBM,
    )
except ImportError:
    (AppArgs, MainToolType, AppType,
     ToolBoxError, ToolBoxInterfaces, ToolBoxResult,
    ToolBoxInfo, ToolBoxResultBM) = [None] * 8
    print("⚠️ Missing: utils.system.types.AppArgs")

try:
    from .utils.extras.show_and_hide_console import show_console
except ImportError:
    show_console = None
    print("⚠️ Missing: utils.extras.show_and_hide_console.show_console")

try:
    from .utils.system.tb_logger import get_logger, setup_logging
except ImportError:
    get_logger = setup_logging = None
    print("⚠️ Missing: utils.system.tb_logger")

try:
    from .utils.system.getting_and_closing_app import get_app
except ImportError:
    # Fallback for PyO3 environment where imports may fail
    class _DummyApp:
        def __init__(self):
            self.id = "toolbox-main"

        def __str__(self):
            return f"<App id='{self.id}'>"

        def __repr__(self):
            return self.__str__()

    def get_app():
        return _DummyApp()

    print("⚠️ Missing: utils.system.getting_and_closing_app.get_app (using fallback)")

try:
    from .utils.system.types import Result
except ImportError:
    Result = None
    print("⚠️ Missing: utils.system.types.Result")

try:
    from .utils.system.types import ApiResult, RequestData
except ImportError:
    ApiResult = RequestData = None
    print("⚠️ Missing: utils.system.types.ApiResult/RequestData")

try:
    from .utils.security.cryp import Code
except ImportError:
    Code = None
    print("⚠️ Missing: utils.security.cryp.Code")

try:
    from .utils.system import all_functions_enums as TBEF
except ImportError:
    TBEF = {}
    print("⚠️ Missing: utils.system.all_functions_enums")

try:
    from .flows import flows_dict
except ImportError:
    flows_dict = {}
    print("⚠️ Missing: flows.flows_dict")


try:
    MODS_ERROR = None
    import toolboxv2.mods
    from toolboxv2.mods import *
except ImportError as e:
    MODS_ERROR = e
except Exception as e:
    print(f"WARNING ERROR IN LIBRARY MODULE´S details : {e}")
    MODS_ERROR = e

try:
    TBX = None
    from .utils.tbx.setup import TBxSetup
except ImportError as e:
    TBX = e

__author__ = """Markin Hausmanns"""
__email__ = 'Markinhausmanns@gmail.com'


from pathlib import Path

__init_cwd__ = init_cwd = Path.cwd()

__tb_root_dir__ = tb_root_dir = Path(__file__).parent
os.chdir(__tb_root_dir__)
os.makedirs(__tb_root_dir__/'dist', exist_ok=True)
__version__ = get_version_from_pyproject() if get_version_from_pyproject is not None else "0.1.25"

ToolBox_over: str = "root"
__all__ = [
    "App",
    "ToolBox_over",
    "MainTool",
    "FileHandler",
    "Style",
    "Spinner",
    "remove_styles",
    "AppArgs",
    "setup_logging",
    "get_logger",
    "flows_dict",
    "mods",
    "get_app",
    "TBEF",
    "Result",
    "ApiResult",
    "RequestData",
    "Code",
    "show_console",
    "init_cwd",
    "tb_root_dir",

    "MainToolType",
    "AppType",
    "ToolBoxError",
    "ToolBoxInterfaces",
    "ToolBoxResult",
    "ToolBoxInfo",
    "ToolBoxResultBM",
    "__init_cwd__",
    "TBxSetup",
]

# Restore stdout/stderr after import
if _suppress_output:
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
