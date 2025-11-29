# toolboxv2/tests/test_web.py
"""
ToolBox V2 - E2E Web Test Utilities
Updated for Clerk Authentication and API CLI Management

Usage:
    - Manages web server lifecycle via API CLI
    - Creates test users via Clerk
    - Provides session state management for E2E tests
    - Supports both async and sync test execution
"""

import asyncio
import json
import os
import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

from toolboxv2 import get_app
from toolboxv2.tests.a_util import async_test
from toolboxv2.tests.web_util import AsyncWebTestFramework, WebTestFramework


# =================== Configuration ===================

TEST_SERVER_PORT = 8080
TEST_SERVER_HOST = "localhost"
TEST_SERVER_BASE_URL = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}"

# Clerk Test Users (configure in .env or here)
TEST_USERS = {
    "admin": {
        "email": os.getenv("TEST_ADMIN_EMAIL", "test-admin@example.com"),
        "clerk_user_id": os.getenv("TEST_ADMIN_CLERK_ID"),
    },
    "testUser": {
        "email": os.getenv("TEST_USER_EMAIL", "test-user@example.com"),
        "clerk_user_id": os.getenv("TEST_USER_CLERK_ID"),
    },
    "loot": {
        "email": os.getenv("TEST_LOOT_EMAIL", "test-loot@example.com"),
        "clerk_user_id": os.getenv("TEST_LOOT_CLERK_ID"),
    },
}

# Session storage for reuse across tests
_test_sessions: dict[str, dict] = {}


# =================== Server Management via API CLI ===================


def is_server_running() -> bool:
    """Check if the test server is running"""
    try:
        import requests

        response = requests.get(f"{TEST_SERVER_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def start_test_server(wait_time: int = 10) -> bool:
    """
    Start the test server using API CLI

    Returns:
        bool: True if server started successfully
    """
    app = get_app(name="test")

    if is_server_running():
        app.logger.info("Test server already running")
        return True

    app.logger.info("Starting test server via API CLI...")

    try:
        # Use API CLI to start server
        result = subprocess.run(
            ["tb", "api", "start", "--version", "test"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            app.logger.error(f"Failed to start server: {result.stderr}")
            return False

        # Wait for server to be ready
        app.logger.info(f"Waiting {wait_time}s for server to start...")
        time.sleep(wait_time)

        # Verify server is running
        if is_server_running():
            app.logger.info("✓ Test server started successfully")
            return True
        else:
            app.logger.error("Server started but health check failed")
            return False

    except subprocess.TimeoutExpired:
        app.logger.error("Server start timeout")
        return False
    except Exception as e:
        app.logger.error(f"Error starting server: {e}")
        return False


def stop_test_server() -> bool:
    """
    Stop the test server using API CLI

    Returns:
        bool: True if server stopped successfully
    """
    app = get_app(name="test")

    if not is_server_running():
        app.logger.info("Test server not running")
        return True

    app.logger.info("Stopping test server via API CLI...")

    try:
        result = subprocess.run(
            ["tb", "api", "stop"], capture_output=True, text=True, timeout=15
        )

        if result.returncode != 0:
            app.logger.warning(f"Server stop returned error: {result.stderr}")

        # Wait and verify
        time.sleep(2)

        if not is_server_running():
            app.logger.info("✓ Test server stopped successfully")
            return True
        else:
            app.logger.warning("Server may still be running")
            return False

    except Exception as e:
        app.logger.error(f"Error stopping server: {e}")
        return False


# =================== Clerk Authentication Helpers ===================


async def clerk_authenticate_user(
    email: str, clerk_user_id: Optional[str] = None
) -> Optional[str]:
    """
    Authenticate a test user via Clerk and get session token.

    This simulates the Clerk auth flow. In real tests, you might need to:
    1. Use Clerk's test mode
    2. Pre-create test users in Clerk Dashboard
    3. Use magic links or test tokens

    Args:
        email: User email address
        clerk_user_id: Optional Clerk user ID if known

    Returns:
        Session token or None if authentication failed
    """
    app = get_app(name="test")

    try:
        # Option 1: If you have Clerk test tokens
        test_token = os.getenv(
            f"CLERK_TEST_TOKEN_{email.replace('@', '_').replace('.', '_').upper()}"
        )
        if test_token:
            app.logger.info(f"Using test token for {email}")
            return test_token

        # Option 2: Use AuthClerk's CLI flow (if available)
        from toolboxv2.mods.CloudM.AuthClerk import cli_request_code, cli_verify_code

        # Request verification code
        result = await cli_request_code(app=app, email=email)

        if result.is_error():
            app.logger.error(f"Failed to request code for {email}: {result.info}")
            return None

        cli_session_id = result.get().get("cli_session_id")

        # In real tests, you'd need to get the code from email or test inbox
        # For now, we'll assume test mode bypasses this
        app.logger.warning(
            f"⚠️  Manual step required: Enter verification code for {email}"
        )
        app.logger.info(f"CLI Session ID: {cli_session_id}")

        # You could implement a test code retriever here
        # For example, using a test email service API

        return None  # Placeholder - implement based on your test setup

    except ImportError:
        app.logger.error("AuthClerk module not available")
        return None
    except Exception as e:
        app.logger.error(f"Authentication error for {email}: {e}")
        return None


def create_clerk_session_cookie(session_token: str) -> dict:
    """
    Create Clerk session cookie for browser context

    Args:
        session_token: Clerk session token

    Returns:
        Cookie dict for Playwright
    """
    return {
        "name": "__session",
        "value": session_token,
        "domain": TEST_SERVER_HOST,
        "path": "/",
        "httpOnly": True,
        "secure": False,  # Set True for HTTPS
        "sameSite": "Lax",
    }


async def setup_clerk_session(
    framework: AsyncWebTestFramework, user_key: str = "testUser"
) -> bool:
    """
    Setup Clerk authenticated session in test framework

    Args:
        framework: AsyncWebTestFramework instance
        user_key: Key from TEST_USERS dict

    Returns:
        bool: True if session setup successful
    """
    user_info = TEST_USERS.get(user_key)

    if not user_info:
        framework.logger.error(f"Unknown test user: {user_key}")
        return False

    # Check if we have a cached session
    if user_key in _test_sessions:
        cached = _test_sessions[user_key]
        if time.time() - cached.get("created_at", 0) < 3600:  # 1 hour validity
            framework.logger.info(f"Using cached session for {user_key}")

            # Add cookie to context
            await framework.context.add_cookies(
                [create_clerk_session_cookie(cached["token"])]
            )
            return True

    # Authenticate and get new token
    framework.logger.info(f"Authenticating {user_key} via Clerk...")

    token = await clerk_authenticate_user(
        email=user_info["email"], clerk_user_id=user_info.get("clerk_user_id")
    )

    if not token:
        framework.logger.error(f"Failed to authenticate {user_key}")
        return False

    # Cache session
    _test_sessions[user_key] = {
        "token": token,
        "created_at": time.time(),
        "email": user_info["email"],
    }

    # Add cookie to context
    await framework.context.add_cookies([create_clerk_session_cookie(token)])

    framework.logger.info(f"✓ Session setup complete for {user_key}")
    return True


# =================== Test State Management ===================


def get_test_state_path(state_name: str) -> Path:
    """Get path for test state file"""
    app = get_app(name="test")
    state_dir = Path(app.config.get("test_state_dir", "tests/test_states"))
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{state_name}_state.json"


async def save_authenticated_state(
    framework: AsyncWebTestFramework, state_name: str, user_key: str = "testUser"
) -> bool:
    """
    Save authenticated browser state for reuse

    Args:
        framework: Test framework instance
        state_name: Name for the saved state
        user_key: User key from TEST_USERS

    Returns:
        bool: True if save successful
    """
    try:
        # Setup session if not already done
        await setup_clerk_session(framework, user_key)

        # Navigate to main page to verify auth
        await framework.navigate(f"{TEST_SERVER_BASE_URL}/web/mainContent.html")
        await asyncio.sleep(2)

        # Save state
        await framework.save_state(state_name)

        framework.logger.info(f"✓ Saved authenticated state: {state_name}")
        return True

    except Exception as e:
        framework.logger.error(f"Failed to save state {state_name}: {e}")
        return False


async def load_authenticated_state(
    framework: AsyncWebTestFramework, state_name: str
) -> bool:
    """
    Load previously saved authenticated state

    Args:
        framework: Test framework instance
        state_name: Name of saved state

    Returns:
        bool: True if load successful
    """
    try:
        success = await framework.load_state(state_name)

        if success:
            framework.logger.info(f"✓ Loaded authenticated state: {state_name}")
        else:
            framework.logger.warning(f"Failed to load state: {state_name}")

        return success

    except Exception as e:
        framework.logger.error(f"Error loading state {state_name}: {e}")
        return False


# =================== Test Execution Helpers ===================


async def ensure_test_environment_ready(
    start_server: bool = True, stop_existing: bool = False
) -> bool:
    """
    Ensure test environment is ready

    Args:
        start_server: Whether to start server if not running
        stop_existing: Whether to stop existing server first

    Returns:
        bool: True if environment ready
    """
    app = get_app(name="test")

    # Check if tests should run
    if not app.local_test:
        app.logger.info("Skipping tests - local_test not enabled")
        return False

    # Stop existing server if requested
    if stop_existing and is_server_running():
        app.logger.info("Stopping existing server...")
        if not stop_test_server():
            app.logger.error("Failed to stop existing server")
            return False

    # Start server if needed
    if start_server and not is_server_running():
        if not start_test_server():
            app.logger.error("Failed to start test server")
            return False

    # Verify server is accessible
    if not is_server_running():
        app.logger.error("Test server not accessible")
        return False

    app.logger.info("✓ Test environment ready")
    return True


async def run_test_suite(
    test_functions: List[callable],
    state_name: str,
    user_key: str = "testUser",
    headless: bool = True,
    save_state: bool = True,
) -> Tuple[bool, List]:
    """
    Run a suite of test functions with shared authenticated state

    Args:
        test_functions: List of test functions to run
        state_name: Name for session state
        user_key: Test user to authenticate as
        headless: Run browser in headless mode
        save_state: Whether to save state after tests

    Returns:
        Tuple of (all_passed, results)
    """
    if not await ensure_test_environment_ready():
        return False, []

    results = []
    all_passed = True

    async with AsyncWebTestFramework(headless=headless) as tf:
        # Create browser context
        await tf.create_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (ToolBox E2E Test)",
        )

        # Try to load existing state
        loaded = await load_authenticated_state(tf, state_name)

        if not loaded:
            # Create new authenticated state
            tf.logger.info(f"Creating new authenticated state for {user_key}...")

            if not await setup_clerk_session(tf, user_key):
                tf.logger.error("Failed to setup session")
                return False, []

            # Navigate to verify auth
            await tf.navigate(f"{TEST_SERVER_BASE_URL}/web/mainContent.html")
            await asyncio.sleep(2)

            if save_state:
                await tf.save_state(state_name)

        # Run test functions
        for test_func in test_functions:
            tf.logger.info(f"Running test: {test_func.__name__}")

            try:
                test_result = await tf.run_tests(test_func, evaluation=False)
                results.extend(test_result)

                # Check if test passed
                if test_result and isinstance(test_result[0], list):
                    for passed, _ in test_result[0]:
                        if not passed:
                            all_passed = False

            except Exception as e:
                tf.logger.error(f"Test {test_func.__name__} failed: {e}")
                all_passed = False
                results.append([(False, str(e))])

    return all_passed, results


# =================== Example Test Functions ===================
# These should be moved to separate test files


async def test_login_page(tf: AsyncWebTestFramework):
    """Test login page accessibility"""
    return [
        {"type": "goto", "url": f"{TEST_SERVER_BASE_URL}/web/assets/login.html"},
        {"type": "sleep", "time": 2},
        {"type": "test", "selector": "#clerk-login"},
        {"type": "screenshot", "path": "login_page.png"},
    ]


async def test_main_content_authenticated(tf: AsyncWebTestFramework):
    """Test main content page with authentication"""
    return [
        {"type": "goto", "url": f"{TEST_SERVER_BASE_URL}/web/mainContent.html"},
        {"type": "sleep", "time": 2},
        {"type": "test", "selector": "#inputField.inputField"},
        {"type": "test", "selector": "[data-user-authenticated]"},
        {"type": "screenshot", "path": "main_content_authenticated.png"},
    ]


async def test_api_endpoint(tf: AsyncWebTestFramework):
    """Test API endpoint access"""
    return [
        {
            "type": "goto",
            "url": f"{TEST_SERVER_BASE_URL}/api/CloudM.UserAccountManager/get_current_user",
        },
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "api_response.png"},
    ]


# =================== CLI Test Runners ===================


@async_test
async def run_authenticated_tests(headless: bool = True, user_key: str = "testUser"):
    """
    Run authenticated test suite

    Usage:
        python -m toolboxv2.tests.test_web
    """
    test_functions = [test_main_content_authenticated, test_api_endpoint]

    passed, results = await run_test_suite(
        test_functions=test_functions,
        state_name=f"{user_key}_session",
        user_key=user_key,
        headless=headless,
    )

    print(f"\n{'=' * 60}")
    print(f"Test Results: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"Total tests: {len(results)}")
    print(f"{'=' * 60}\n")

    return passed


@async_test
async def run_unauthenticated_tests(headless: bool = True):
    """Run tests that don't require authentication"""
    test_functions = [test_login_page]

    if not await ensure_test_environment_ready():
        return False

    async with AsyncWebTestFramework(headless=headless) as tf:
        await tf.create_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (ToolBox E2E Test)",
        )

        results = await tf.run_tests(*test_functions, evaluation=False)

        all_passed = all(
            passed
            for result in results
            for passed, _ in (result if isinstance(result, list) else [result])
        )

        print(f"\n{'=' * 60}")
        print(f"Unauthenticated Tests: {'✓ PASSED' if all_passed else '✗ FAILED'}")
        print(f"{'=' * 60}\n")

        return all_passed


# =================== Main Entry Point ===================

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    headless = "--no-headless" not in sys.argv
    user = "testUser"

    if "--user" in sys.argv:
        idx = sys.argv.index("--user")
        if idx + 1 < len(sys.argv):
            user = sys.argv[idx + 1]

    # Run tests
    print("Starting E2E Test Suite...")
    print(f"Headless: {headless}")
    print(f"User: {user}\n")

    # Run unauthenticated tests
    unauth_passed = run_unauthenticated_tests(headless=headless)

    # Run authenticated tests
    auth_passed = run_authenticated_tests(headless=headless, user_key=user)

    # Exit with appropriate code
    sys.exit(0 if (unauth_passed and auth_passed) else 1)
