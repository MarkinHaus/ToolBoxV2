"""Top-level package for ToolBox."""
import os

from yaml import safe_load

try:
    from .utils.toolbox import App
except ImportError:
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
    get_app = None
    print("⚠️ Missing: utils.system.getting_and_closing_app.get_app")

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

__author__ = """Markin Hausmanns"""
__email__ = 'Markinhausmanns@gmail.com'


from pathlib import Path
from platform import node, system

__cwd__ = cwd = Path.cwd()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
__version__ = get_version_from_pyproject()

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
    "RequestData",  # ADD
    "Code",
    "show_console",  # ADD
    "cwd",

    "MainToolType",
    "AppType",
    "ToolBoxError",
    "ToolBoxInterfaces",
    "ToolBoxResult",
    "ToolBoxInfo",
    "ToolBoxResultBM",
]
