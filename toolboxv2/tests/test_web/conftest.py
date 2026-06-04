"""
conftest.py for toolboxv2/tests/web_test/

E2E test configuration:
- Server lifecycle management
- Authenticated browser sessions
- Test environment setup
"""

import asyncio
import os
from typing import AsyncGenerator

try:
    import pytest
except ImportError:
    pytest = None

try:
    from toolboxv2.tests.test_web.web_util import AsyncWebTestFramework
except ImportError:
    AsyncWebTestFramework = None

try:
    from tests.test_web.test_web import (
        TEST_SERVER_BASE_URL,
        TEST_SERVER_PORT,
        TEST_WEB_BASE_URL,
        TEST_WEB_PORT,
        is_server_running,
        start_test_server,
        stop_test_server,
        setup_auth_session,
        TEST_USERS,
    )
except ImportError:
    TEST_SERVER_PORT = 8000
    TEST_SERVER_BASE_URL = f"http://localhost:{TEST_SERVER_PORT}"
    TEST_WEB_PORT = int(os.getenv("TEST_WEB_PORT", 80))
    TEST_WEB_BASE_URL = f"http://localhost:{TEST_WEB_PORT}"
    TEST_USERS = {}

    def is_server_running():
        try:
            import requests
            return requests.get(f"{TEST_SERVER_BASE_URL}/health", timeout=2).status_code == 200
        except Exception:
            return False

    def start_test_server(wait_time=10):
        return False

    def stop_test_server():
        return False

    async def setup_auth_session(framework, user_key="testUser"):
        return False


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Server availability
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def server_url() -> str:
    return TEST_SERVER_BASE_URL


@pytest.fixture(scope="session")
def check_server_available():
    """Skip all tests in the session if the server is not running."""
    if not is_server_running():
        pytest.skip(
            f"Test server not running at {TEST_SERVER_BASE_URL}. "
            "Start with: python -m toolboxv2 workers start"
        )


@pytest.fixture(scope="module")
def require_server(check_server_available):
    """Module-scoped guard — skips if server is unavailable."""
    if not is_server_running():
        pytest.skip(f"Server not available at {TEST_SERVER_BASE_URL}")


# ---------------------------------------------------------------------------
# Browser contexts
# ---------------------------------------------------------------------------

@pytest.fixture
async def browser_context() -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Fresh desktop browser context per test."""
    async with AsyncWebTestFramework(headless=True) as tf:
        await tf.create_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (ToolBoxV2 E2E Test)",
        )
        yield tf


@pytest.fixture
async def browser_context_mobile() -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Mobile viewport browser context."""
    async with AsyncWebTestFramework(headless=True) as tf:
        await tf.create_context(
            viewport={"width": 375, "height": 667},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
        )
        yield tf


@pytest.fixture
async def browser_context_tablet() -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Tablet viewport browser context."""
    async with AsyncWebTestFramework(headless=True) as tf:
        await tf.create_context(
            viewport={"width": 768, "height": 1024},
            user_agent="Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)",
        )
        yield tf


# ---------------------------------------------------------------------------
# Authenticated sessions
# ---------------------------------------------------------------------------

@pytest.fixture
async def authenticated_session(
    browser_context: AsyncWebTestFramework,
    require_server,
) -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Authenticated browser session for testUser."""
    if not await setup_auth_session(browser_context, "testUser"):
        pytest.skip("Could not setup authenticated session")
    yield browser_context


@pytest.fixture
async def admin_session(
    browser_context: AsyncWebTestFramework,
    require_server,
) -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Authenticated browser session for admin."""
    if not await setup_auth_session(browser_context, "admin"):
        pytest.skip("Could not setup admin session")
    yield browser_context


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def test_user_email() -> str:
    return TEST_USERS.get("testUser", {}).get("email", "test@example.com")


@pytest.fixture
def admin_email() -> str:
    return TEST_USERS.get("admin", {}).get("email", "admin@example.com")


@pytest.fixture
def screenshot_dir(tmp_path):
    """Temporary directory for test screenshots."""
    path = tmp_path / "screenshots"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture
def test_state_dir():
    """Persistent directory for test state files."""
    state_dir = os.path.join("tests", "test_states")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir
