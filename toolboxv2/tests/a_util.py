import asyncio
import unittest
from unittest.mock import MagicMock, Mock, AsyncMock


def async_test(coro):
    """Decorator to run async tests with unittest."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


class IsolatedTestCase(unittest.TestCase):
    """
    Base test class that provides automatic App/Singleton isolation.

    This class ensures that each test starts with a clean slate by resetting
    all singletons and global state in setUp and tearDown.

    Works with both pytest and unittest runners.

    Usage:
        class MyTests(IsolatedTestCase):
            def test_something(self):
                app = get_app(from_="test", name="test")
                # ... test code ...

    For tests that need a persistent App across all tests in the class,
    use PersistentAppTestCase instead.
    """

    @classmethod
    def setUpClass(cls):
        """Reset singletons before any tests in this class run."""
        super().setUpClass()
        reset_app_singleton(force_exit=True)

    def setUp(self):
        """Reset singletons before each test."""
        super().setUp()
        reset_app_singleton(force_exit=True)

    def tearDown(self):
        """Reset singletons after each test."""
        super().tearDown()
        reset_app_singleton(force_exit=True)

    @classmethod
    def tearDownClass(cls):
        """Final cleanup after all tests in this class."""
        reset_app_singleton(force_exit=True)
        super().tearDownClass()


class PersistentAppTestCase(unittest.TestCase):
    """
    Base test class for tests that need a persistent App across all tests.

    Use this when:
    - Tests share expensive setup (loading mods, etc.)
    - Tests need to verify state changes across multiple operations
    - Performance is critical and App creation is slow

    The App is created once in setUpClass and cleaned up in tearDownClass.
    Individual tests do NOT reset the App between runs.

    Usage:
        class MyModTests(PersistentAppTestCase):
            @classmethod
            def setUpClass(cls):
                super().setUpClass()
                cls.app = get_app(from_="test", name="test")
                cls.app.load_mod("my_mod")

            @classmethod
            def tearDownClass(cls):
                if hasattr(cls, 'app') and cls.app:
                    cls.app.exit()
                super().tearDownClass()

            def test_something(self):
                result = self.app.run_any(...)
    """

    @classmethod
    def setUpClass(cls):
        """Reset singletons before creating the persistent App."""
        super().setUpClass()
        reset_app_singleton(force_exit=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up the persistent App."""
        reset_app_singleton(force_exit=True)
        super().tearDownClass()


def _is_mock(obj) -> bool:
    """Check if object is a Mock or has mock attributes."""
    if isinstance(obj, (MagicMock, Mock)):
        return True
    # Check for contaminated objects (real objects with mock attributes)
    for attr_name in dir(obj):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(obj, attr_name)
            if isinstance(attr, (MagicMock, Mock)):
                return True
        except Exception:
            pass
    return False


def reset_app_singleton(force_exit: bool = False):
    """
    Reset ALL singletons and global state to ensure test isolation.

    This function cleans up:
    - registered_apps global list
    - Singleton._instances for App, Session, ProxyApp, DaemonApp
    - Any mock-contaminated instances

    Args:
        force_exit: If True, exit real App instances and clear ALL singletons.
                   Use True for full test isolation (recommended).
                   Use False to only clear mocks (legacy behavior).

    Example:
        @classmethod
        def setUpClass(cls):
            reset_app_singleton(force_exit=True)  # Clean slate
            cls.app = get_app(from_="test", name="test")
    """
    # 1. Reset registered_apps global
    try:
        from toolboxv2.utils.system.getting_and_closing_app import registered_apps
        if registered_apps[0] is not None:
            app = registered_apps[0]
            is_mock = _is_mock(app)

            if is_mock or force_exit:
                # Exit real apps properly
                if not isinstance(app, (MagicMock, Mock)) and hasattr(app, 'exit'):
                    try:
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
            # Clear ALL singletons for full isolation
            Singleton._instances.clear()
            Singleton._args.clear()
            Singleton._kwargs.clear()
        else:
            # Only clear mocks (legacy behavior)
            for cls in list(Singleton._instances.keys()):
                if _is_mock(Singleton._instances[cls]):
                    del Singleton._instances[cls]
                    Singleton._args.pop(cls, None)
                    Singleton._kwargs.pop(cls, None)
    except ImportError:
        pass

    # 3. Reset Session singleton explicitly (it may have open connections)
    try:
        from toolboxv2.utils.system.session import Session
        from toolboxv2.utils.singelton_class import Singleton
        if Session in Singleton._instances and force_exit:
            session = Singleton._instances[Session]
            # Close aiohttp session if open
            if hasattr(session, '_session') and session._session is not None:
                try:
                    if not session._session.closed:
                        # Can't await here, just mark for cleanup
                        pass
                except Exception:
                    pass
            del Singleton._instances[Session]
            Singleton._args.pop(Session, None)
            Singleton._kwargs.pop(Session, None)
    except (ImportError, KeyError):
        pass

    # 4. Reload modules if they've been contaminated by mocks
    if force_exit:
        try:
            import importlib

            # Check if Result is a mock
            from toolboxv2.utils.system import types as types_module
            if hasattr(types_module, 'Result'):
                result_cls = types_module.Result
                if isinstance(result_cls, (MagicMock, Mock, AsyncMock)):
                    # Reload the module to get the real Result class
                    importlib.reload(types_module)

            # Also check toolboxv2 package level
            import toolboxv2
            if hasattr(toolboxv2, 'Result'):
                result_cls = toolboxv2.Result
                if isinstance(result_cls, (MagicMock, Mock, AsyncMock)):
                    # Reload to restore real Result
                    importlib.reload(types_module)
                    # Update the reference in toolboxv2
                    toolboxv2.Result = types_module.Result

            # Check if App is a mock
            if hasattr(toolboxv2, 'App'):
                app_cls = toolboxv2.App
                if isinstance(app_cls, (MagicMock, Mock, AsyncMock)):
                    # Reload the app module to get the real App class
                    from toolboxv2.utils import toolbox
                    importlib.reload(toolbox)
                    toolboxv2.App = toolbox.App

            if hasattr(toolboxv2, 'MainTool'):
                mt_cls = toolboxv2.MainTool
                if isinstance(mt_cls, (MagicMock, Mock, AsyncMock)):
                    # Reload the app module to get the real App class
                    from toolboxv2.utils.system import main_tool
                    importlib.reload(main_tool)
                    toolboxv2.MainTool = main_tool.MainTool

            if hasattr(toolboxv2, 'FileHandler'):
                fh_cls = toolboxv2.FileHandler
                if isinstance(fh_cls, (MagicMock, Mock, AsyncMock)):
                    # Reload the app module to get the real App class
                    from toolboxv2.utils.system import file_handler
                    importlib.reload(file_handler)
                    toolboxv2.FileHandler = file_handler.FileHandler
        except (ImportError, AttributeError):
            pass
