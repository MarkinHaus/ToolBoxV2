"""Top-level package for ToolBox."""
from toolboxv2.utils import (Style, remove_styles, Spinner, FileHandler, App, get_app,
                             setup_logging, get_logger, MainTool,
                             all_functions_enums as tbef,
                             Result, AppArgs, Code)
from toolboxv2.runabel import runnable_dict

# try:
#     MODS_ERROR = None
#     import toolboxv2.mods
#     from toolboxv2.mods import *
# except ImportError as e:
#     MODS_ERROR = e

__author__ = """Markin Hausmanns"""
__email__ = 'Markinhausmanns@gmail.com'
__version__ = '0.1.8'
__all__ = [
    "__version__",
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
