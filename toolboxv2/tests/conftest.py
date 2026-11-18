"""
conftest.py for toolboxv2/tests/

Provides:
- Automatic App singleton isolation between tests
- Serial execution marker for xdist (tests that share state)
"""
import contextlib
import os
import tempfile
import pytest
import sys


# HARD ISOLATION: the test suite must NEVER touch real user data (.data / global VFS / appdata).
# Set at conftest import time — before the first get_app() resolves data_dir and appdata.
# appdata kommt aus XDG_CONFIG_HOME (Linux/macOS) bzw. APPDATA (Windows), siehe utils/toolbox.py.
from toolboxv2.tests.schreibschutz import Schreibschutz  # noqa: E402

_SCHREIBSCHUTZ = Schreibschutz()          # echte Ordner, bestimmt VOR dem Umbiegen
if not os.environ.get("TB_TEST_DATA_LOCK"):
    os.environ["TB_DATA_DIR"] = tempfile.mkdtemp(prefix="tb_test_data_")
    _iso = tempfile.mkdtemp(prefix="tb_test_home_")
    for _var, _sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "share"),
                       ("XDG_STATE_HOME", "state"), ("APPDATA", "appdata"), ("LOCALAPPDATA", "local")):
        os.environ[_var] = os.path.join(_iso, _sub)
    os.environ["TB_TEST_DATA_LOCK"] = "1"


def pytest_sessionfinish(session, exitstatus):
    """Am Ende: hat irgendein Test echte Zustandsordner veraendert, scheitert der Lauf (Regel 6.3)."""
    text = _SCHREIBSCHUTZ.unterschiede()
    if text:
        session.config.get_terminal_writer().line(f"SCHREIBSCHUTZ: {text}", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

# Files that create one real App for ALL their tests in setUpClass.
# The reset fixture must not interfere with them.
PERSISTENT_APP_TEST_FILES = {
    "test_toolbox.py",
    "test_toolboxv2.py",
}


@pytest.fixture(autouse=True)
def reset_app_state_before_test(request):
    """
    Reset the App singleton before and after every test for full isolation.

    Skipped for files in PERSISTENT_APP_TEST_FILES — those manage their
    own App lifecycle via setUpClass / tearDownClass.
    """
    test_file = request.fspath.basename if hasattr(request, "fspath") else ""

    if test_file in PERSISTENT_APP_TEST_FILES:
        yield
        return

    from toolboxv2.tests.a_util import reset_app_singleton
    reset_app_singleton(force_exit=True)
    yield
    reset_app_singleton(force_exit=True)


@pytest.fixture(autouse=True)
def ensure_open_event_loop():
    """Sync tests that call asyncio.get_event_loop().run_until_complete(...) need
    an open current loop. Earlier tests may have closed or unset it."""
    import asyncio
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loop = asyncio.get_event_loop_policy().get_event_loop()
        stale = loop.is_closed()
    except RuntimeError:
        stale = True
    if stale:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield


def pytest_configure(config):
    """Register the 'serial' marker for xdist compatibility."""
    config.addinivalue_line(
        "markers",
        "serial: mark test class to run in a single xdist worker (no distribution)",
    )

@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    """
    Hook into xdist scheduling: tests marked @pytest.mark.serial
    are grouped onto a single worker instead of being distributed.

    Falls back to default scheduling if xdist is not active.
    """
    try:
        from xdist.scheduler import LoadScheduling

        class SerialAwareScheduling(LoadScheduling):
            def _assign_work_unit(self, node):
                # Default behavior — xdist handles grouping via
                # pytest_collection_modifyitems below
                return super()._assign_work_unit(node)

        return SerialAwareScheduling(config, log)
    except ImportError:
        return None


def pytest_collection_modifyitems(items):
    """
    Group all @serial-marked tests under one xdist group
    so they land on the same worker.
    """
    for item in items:
        if item.get_closest_marker("serial"):
            # xdist groups by this attribute — same string = same worker
            item._nodeid = item.nodeid  # preserve original
            if not hasattr(item, "fixturenames"):
                continue
            # Force same xdist group
            item.add_marker(pytest.mark.xdist_group("serial_tests"))
