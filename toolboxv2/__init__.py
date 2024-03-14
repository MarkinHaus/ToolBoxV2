"""Top-level package for ToolBox."""
import os

from yaml import safe_load

from .utils import (Style, remove_styles, Spinner, FileHandler, App, get_app,
                             setup_logging, get_logger, MainTool,
                             Result, AppArgs, Code)
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

print(os.system('dir'))

with open(os.getenv('CONFIG_FILE', f'{os.path.abspath(__file__).replace("__init__.py", "")}toolbox.yaml'), 'r') as config_file:
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
