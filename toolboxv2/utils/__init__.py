import os

from yaml import safe_load

from .extras.show_and_hide_console import show_console
from .extras.Style import Spinner, Style, remove_styles
from .security.cryp import Code
from .singelton_class import Singleton
from .system import all_functions_enums as TBEF
from .system.file_handler import FileHandler
from .system.getting_and_closing_app import get_app
from .system.main_tool import MainTool
from .system.tb_logger import get_logger, setup_logging
from .system.types import ApiResult, AppArgs, Result
from .toolbox import App
from .tbx import  install_support, setup as tbx_setup
from .tbx.install_support import function_runner as system_tbx_support
from .tbx.setup import function_runner as language_ide_extension

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
    "TBEF",
    "Result",
    "ApiResult",
    "Code",
    "install_support",
    "tbx_setup",
    "language_ide_extension",
    "system_tbx_support",
]
