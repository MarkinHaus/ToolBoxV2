import asyncio
import contextlib
import logging
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock


def async_test(coro):
    """Decorator to run async tests with unittest."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


# --- PYTEST STDOUT SCHUTZ ---
# Verhindert, dass prompt_toolkit Pytests Capture-System zerstört
if "pytest" in sys.modules:
    try:
        import prompt_toolkit.patch_stdout


        @contextlib.contextmanager
        def dummy_patch_stdout(*args, **kwargs):
            yield  # Tut einfach gar nichts und lässt sys.stdout in Ruhe


        prompt_toolkit.patch_stdout.patch_stdout = dummy_patch_stdout
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Internal helper — must be called BEFORE app.exit()
# ---------------------------------------------------------------------------

def _detach_all_logging_handlers():
    """Trennt alle Logger sicher ab, ohne Dateien zu öffnen."""
    try:
        import logging
        null_handler = logging.NullHandler()

        # Root Logger bereinigen
        root = logging.getLogger()
        root.handlers = [null_handler]

        # Alle anderen Logger bereinigen
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.handlers = [null_handler]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _is_mock(obj) -> bool:
    """Check if object is a Mock or has mock-contaminated attributes."""
    if isinstance(obj, (MagicMock, Mock)):
        return True
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(obj, attr_name)
            if isinstance(attr, (MagicMock, Mock)):
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Core reset function
# ---------------------------------------------------------------------------

def reset_app_singleton(force_exit: bool = False):
    """
    Reset ALL singletons and global state to ensure test isolation.

    Cleans up:
    - registered_apps global list
    - Singleton._instances (App, Session, ProxyApp, DaemonApp, …)
    - Mock-contaminated instances

    IMPORTANT: always detaches logging handlers before calling app.exit()
    to avoid corrupting pytest's FDCapture temporary files on Windows.

    Args:
        force_exit: If True, exit real App instances and clear ALL singletons.
                    Recommended for full test isolation.
    """
    # 1. Reset registered_apps
    try:
        from toolboxv2.utils.system.getting_and_closing_app import registered_apps

        if registered_apps[0] is not None:
            app = registered_apps[0]
            is_mock = _is_mock(app)

            if is_mock or force_exit:
                if not isinstance(app, (MagicMock, Mock)) and hasattr(app, "exit"):
                    try:
                        # ── CRITICAL ────────────────────────────────────────
                        # Detach logging handlers BEFORE app.exit().
                        # app.exit() triggers logging.shutdown() which closes
                        # every handler's stream.  On Windows, pytest's
                        # FDCapture redirects FD 1/2 to NamedTemporaryFiles;
                        # if those streams are closed here, every later
                        # tmpfile.seek(0) raises ValueError.
                        # ─────────────────────────────────────────────────────
                        _detach_all_logging_handlers()
                        app.exit()
                    except Exception:
                        pass
                registered_apps[0] = None
    except ImportError:
        pass

    # 2. Reset Singleton instances
    try:
        from toolboxv2.utils.singelton_class import Singleton

        if force_exit:
            Singleton._instances.clear()
            Singleton._args.clear()
            Singleton._kwargs.clear()
        else:
            for cls in list(Singleton._instances.keys()):
                if _is_mock(Singleton._instances[cls]):
                    del Singleton._instances[cls]
                    Singleton._args.pop(cls, None)
                    Singleton._kwargs.pop(cls, None)
    except ImportError:
        pass

    # 3. Reset Session singleton explicitly (may hold open aiohttp connections)
    try:
        from toolboxv2.utils.singelton_class import Singleton
        from toolboxv2.utils.system.session import Session

        if Session in Singleton._instances and force_exit:
            del Singleton._instances[Session]
            Singleton._args.pop(Session, None)
            Singleton._kwargs.pop(Session, None)
    except (ImportError, KeyError):
        pass

    # 4. Reload mock-contaminated modules
    if force_exit:
        try:
            import importlib

            import toolboxv2
            from toolboxv2.utils.system import types as types_module

            if isinstance(getattr(types_module, "Result", None), (MagicMock, Mock, AsyncMock)):
                importlib.reload(types_module)

            if isinstance(getattr(toolboxv2, "Result", None), (MagicMock, Mock, AsyncMock)):
                importlib.reload(types_module)
                toolboxv2.Result = types_module.Result

            if isinstance(getattr(toolboxv2, "App", None), (MagicMock, Mock, AsyncMock)):
                from toolboxv2.utils import toolbox
                importlib.reload(toolbox)
                toolboxv2.App = toolbox.App

            if isinstance(getattr(toolboxv2, "MainTool", None), (MagicMock, Mock, AsyncMock)):
                from toolboxv2.utils.system import main_tool
                importlib.reload(main_tool)
                toolboxv2.MainTool = main_tool.MainTool

            if isinstance(getattr(toolboxv2, "FileHandler", None), (MagicMock, Mock, AsyncMock)):
                from toolboxv2.utils.system import file_handler
                importlib.reload(file_handler)
                toolboxv2.FileHandler = file_handler.FileHandler

        except (ImportError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# Base test classes
# ---------------------------------------------------------------------------

class IsolatedTestCase(unittest.TestCase):
    """
    Base test class with automatic App/Singleton isolation per test.

    Usage:
        class MyTests(IsolatedTestCase):
            def test_something(self):
                app = get_app(from_="test", name="test")
                ...
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        reset_app_singleton(force_exit=True)

    def setUp(self):
        super().setUp()
        reset_app_singleton(force_exit=True)

    def tearDown(self):
        super().tearDown()
        reset_app_singleton(force_exit=True)

    @classmethod
    def tearDownClass(cls):
        reset_app_singleton(force_exit=True)
        super().tearDownClass()


class PersistentAppTestCase(unittest.TestCase):
    """
    Base test class for tests that share one App across all tests in the class.

    Use when App creation is expensive (loading mods, etc.) and tests are
    designed to run sequentially against the same instance.

    Usage:
        class MyModTests(PersistentAppTestCase):
            @classmethod
            def setUpClass(cls):
                super().setUpClass()
                cls.app = get_app(from_="test", name="test")
                cls.app.load_mod("my_mod")

            @classmethod
            def tearDownClass(cls):
                if hasattr(cls, "app") and cls.app:
                    cls.app.exit()
                super().tearDownClass()
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        reset_app_singleton(force_exit=True)

    @classmethod
    def tearDownClass(cls):
        reset_app_singleton(force_exit=True)
        super().tearDownClass()
