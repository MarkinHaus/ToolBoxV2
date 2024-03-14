import os

from yaml import safe_load

from .toolbox import App

from .singelton_class import Singleton
from .security.cryp import Code

from .system.file_handler import FileHandler
from .system.tb_logger import setup_logging, get_logger
from .system.main_tool import MainTool
from .system import all_functions_enums as tbef
from .system.types import Result, AppArgs, ApiResult
from .system.getting_and_closing_app import get_app

from .extras.Style import Style, remove_styles, Spinner
from .extras.show_and_hide_console import show_console

__all__ = [
    "App",
    "Singleton",
    "MainTool",
    "FileHandler",
    "Style",
    "Spinner",
    "remove_styles",
    "AppArgs",
    "show_console",
    "setup_logging",
    "get_logger",
    "get_app",
    "tbef",
    "Result",
    "ApiResult",
    "Code",
]
