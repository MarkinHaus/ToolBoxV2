"""Top-level package for ToolBox."""
import os

from yaml import safe_load

from .utils.toolbox import App
from .utils.singelton_class import Singleton
from .utils.system.main_tool import MainTool
from .utils.system.file_handler import FileHandler
from .utils.extras.Style import Style
from .utils.extras.Style import Spinner
from .utils.extras.Style import remove_styles
from .utils.system.types import AppArgs
from .utils.extras.show_and_hide_console import show_console
from .utils.system.tb_logger import get_logger, setup_logging
from .utils.system.getting_and_closing_app import get_app
from .utils.system.types import Result
from .utils.system.types import ApiResult
from .utils.security.cryp import Code
from .utils.system import all_functions_enums as tbef
from .runabel import runnable_dict

try:
    MODS_ERROR = None
    import toolboxv2.mods
    from toolboxv2.mods import *
except ImportError as e:
    MODS_ERROR = e

__author__ = """Markin Hausmanns"""
__email__ = 'Markinhausmanns@gmail.com'

with open(os.getenv('CONFIG_FILE', f'{os.path.abspath(__file__).replace("__init__.py", "")}toolbox.yaml'),
          'r') as config_file:
    _version = safe_load(config_file)
    __version__ = _version.get('main', {}).get('version', '-.-.-')

__all__ = [
    "App",
    "MainTool",
    "FileHandler",
    "Style",
    "Spinner",
    "remove_styles",
    "AppArgs",
    "setup_logging",
    "get_logger",
    "runnable_dict",
    "mods",
    "utils",
    "get_app",
    "tbef",
    "Result",
    "Code",
]

ToolBox_over: str = "root"
