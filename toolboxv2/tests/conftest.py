"""
conftest.py for toolboxv2/tests/

Provides automatic App singleton isolation between tests.

Tests that manage their own App lifecycle (e.g. via setUpClass) are
exempted via PERSISTENT_APP_TEST_FILES.
"""
import contextlib

import pytest

import sys

# These files create one real App for ALL their tests in setUpClass.
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

