# toolboxv2/tests/web_test/conftest.py
"""
ToolBoxV2 E2E Test Configuration

Pytest fixtures for:
- Server lifecycle management
- Authenticated browser sessions
- Test environment setup
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

from toolboxv2.tests.web_util import AsyncWebTestFramework

# Import from test_web with fallback defaults
try:
    from toolboxv2.tests.test_web import (
        TEST_SERVER_BASE_URL,
        TEST_SERVER_PORT,
        TEST_WEB_BASE_URL,
        TEST_WEB_PORT,
        is_server_running,
        start_test_server,
        stop_test_server,
        setup_clerk_session,
        TEST_USERS,
    )
except ImportError:
    # Fallback defaults if imports fail
    TEST_SERVER_PORT = 8000
    TEST_SERVER_BASE_URL = f"http://localhost:{TEST_SERVER_PORT}"
    # Web interface served by nginx on port 80
    TEST_WEB_PORT = int(os.getenv("TEST_WEB_PORT", 80))
    TEST_WEB_BASE_URL = f"http://localhost:{TEST_WEB_PORT}"
    TEST_USERS = {}

    def is_server_running():
        try:
            import requests
            response = requests.get(f"{TEST_SERVER_BASE_URL}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def start_test_server(wait_time=10):
        return False

    def stop_test_server():
        return False

    async def setup_clerk_session(framework, user_key="testUser"):
        return False


# =================== Event Loop Configuration ===================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =================== Server Management Fixtures ===================

@pytest.fixture(scope="session")
def server_url() -> str:
    """Return the test server base URL"""
    return TEST_SERVER_BASE_URL


@pytest.fixture(scope="session")
def check_server_available():
    """
    Session-scoped fixture to check if server is available.
    Skips all tests if server is not running.
    """
    if not is_server_running():
        pytest.skip(
            f"Test server not running at {TEST_SERVER_BASE_URL}. "
            f"Start with: python -m toolboxv2 workers start"
        )


@pytest.fixture(scope="module")
def require_server(check_server_available):
    """Module-scoped fixture that requires server to be running"""
    if not is_server_running():
        pytest.skip(f"Server not available at {TEST_SERVER_BASE_URL}")


# =================== Browser Framework Fixtures ===================

@pytest.fixture
async def browser_context() -> AsyncGenerator[AsyncWebTestFramework, None]:
    """
    Provide a fresh browser context for each test.
    Automatically handles setup and teardown.
    """
    async with AsyncWebTestFramework(headless=True) as tf:
        await tf.create_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (ToolBoxV2 E2E Test)"
        )
        yield tf


@pytest.fixture
async def browser_context_mobile() -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Provide a mobile viewport browser context"""
    async with AsyncWebTestFramework(headless=True) as tf:
        await tf.create_context(
            viewport={"width": 375, "height": 667},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
        )
        yield tf


@pytest.fixture
async def browser_context_tablet() -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Provide a tablet viewport browser context"""
    async with AsyncWebTestFramework(headless=True) as tf:
        await tf.create_context(
            viewport={"width": 768, "height": 1024},
            user_agent="Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
        )
        yield tf


# =================== Authenticated Session Fixtures ===================

@pytest.fixture
async def authenticated_session(
    browser_context: AsyncWebTestFramework,
    require_server
) -> AsyncGenerator[AsyncWebTestFramework, None]:
    """
    Provide an authenticated browser session.
    Uses testUser credentials from TEST_USERS.
    """
    success = await setup_clerk_session(browser_context, "testUser")
    if not success:
        pytest.skip("Could not setup authenticated session")
    yield browser_context


@pytest.fixture
async def admin_session(
    browser_context: AsyncWebTestFramework,
    require_server
) -> AsyncGenerator[AsyncWebTestFramework, None]:
    """Provide an admin authenticated browser session"""
    success = await setup_clerk_session(browser_context, "admin")
    if not success:
        pytest.skip("Could not setup admin session")
    yield browser_context


# =================== Test Data Fixtures ===================

@pytest.fixture
def test_user_email() -> str:
    """Return test user email"""
    return TEST_USERS.get("testUser", {}).get("email", "test@example.com")


@pytest.fixture
def admin_email() -> str:
    """Return admin user email"""
    return TEST_USERS.get("admin", {}).get("email", "admin@example.com")


# =================== Utility Fixtures ===================

@pytest.fixture
def screenshot_dir(tmp_path):
    """Provide a temporary directory for screenshots"""
    screenshot_path = tmp_path / "screenshots"
    screenshot_path.mkdir(exist_ok=True)
    return screenshot_path


@pytest.fixture
def test_state_dir():
    """Provide the test state directory path"""
    state_dir = os.path.join("tests", "test_states")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir

