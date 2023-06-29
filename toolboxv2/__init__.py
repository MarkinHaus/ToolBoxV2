"""Top-level package for ToolBox."""
from toolboxv2.utils.Style import Style, remove_styles, Spinner
from toolboxv2.utils.file_handler import FileHandler
from toolboxv2.utils.toolbox import App, AppArgs
from toolboxv2.utils.tb_logger import setup_logging, get_logger

from toolboxv2.utils.main_tool import MainTool

from toolboxv2.runabel import runnable_dict

__author__ = """Markin Hausmanns"""
__email__ = 'Markinhausmanns@gmail.com'
__version__ = '0.0.3'
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
    ]

ToolBox_over: str = "root"

