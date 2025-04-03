"""Top-level package for ToolBox."""
import os

from yaml import safe_load

from .utils.toolbox import App
from .utils.singelton_class import Singleton
from .utils.system.main_tool import MainTool, get_version_from_pyproject
from .utils.system.file_handler import FileHandler
from .utils.extras.Style import Style
from .utils.extras.Style import Spinner
from .utils.extras.Style import remove_styles
from .utils.system.types import AppArgs
from .utils.extras.show_and_hide_console import show_console
from .utils.system.tb_logger import get_logger, setup_logging
from .utils.system.getting_and_closing_app import get_app
from .utils.system.types import Result
from .utils.system.types import ApiResult, RequestData
from .utils.security.cryp import Code
from .utils.system import all_functions_enums as TBEF
from .flows import flows_dict

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



__version__ = get_version_from_pyproject("pyproject.toml")

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
    "utils",
    "get_app",
    "TBEF",
    "Result",
    "ApiResult",
    "Code",
]
