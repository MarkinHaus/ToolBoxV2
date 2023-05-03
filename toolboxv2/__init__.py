"""Top-level package for ToolBox."""
from toolboxv2.util.Style import Style, remove_styles
from toolboxv2.util.file_handler import FileHandler
from toolboxv2.util.toolbox import App, AppArgs
from toolboxv2.util.tb_logger import setup_logging, get_logger

from toolboxv2.main_tool import MainTool

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
    "remove_styles",
    "AppArgs",
    "setup_logging",
    "get_logger",
    "runnable_dict",
    ]

ToolBox_ovner = "root"
