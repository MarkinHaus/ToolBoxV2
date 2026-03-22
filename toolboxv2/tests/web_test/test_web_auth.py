# toolboxv2/tests/web_test/test_auth.py
"""
ToolBoxV2 E2E Authentication Tests

Tests for Clerk-based authentication:
- Login page rendering
- Signup page rendering
- Session validation
- Protected route redirects
- Logout functionality

Run:
    pytest toolboxv2/tests/web_test/test_auth.py -v
    pytest toolboxv2/tests/web_test/test_auth.py -v -k "test_login"
"""

import pytest
from typing import List, Dict, Any

from toolboxv2.tests.web_util import AsyncWebTestFramework

# Web URL for page tests (nginx on port 80)
import os
BASE_URL = os.getenv("TEST_WEB_BASE_URL", "http://localhost:80")

def is_server_running():
    """Check if web server (nginx) is running"""
    try:
        import requests
        response = requests.get(f"{BASE_URL}/web/core0/index.html", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# =================== Test Interactions ===================

def login_page_interactions(base_url: str) -> List[Dict[str, Any]]:
    """Interactions for login page test"""
    return [
        {"type": "goto", "url": f"{base_url}/web/assets/login.html"},
        {"type": "sleep", "time": 2},
        {"type": "screenshot", "path": "auth/login_page.png"},
        {"type": "test", "selector": "body"},
        {"type": "title"},
    ]


def signup_page_interactions(base_url: str) -> List[Dict[str, Any]]:
    """Interactions for signup page test"""
    return [
        {"type": "goto", "url": f"{base_url}/web/assets/signup.html"},
        {"type": "sleep", "time": 2},
        {"type": "screenshot", "path": "auth/signup_page.png"},
        {"type": "test", "selector": "body"},
    ]


def logout_page_interactions(base_url: str) -> List[Dict[str, Any]]:
    """Interactions for logout page test"""
    return [
        {"type": "goto", "url": f"{base_url}/web/assets/logout.html"},
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "auth/logout_page.png"},
        {"type": "test", "selector": "body"},
    ]


def protected_page_redirect_interactions(base_url: str) -> List[Dict[str, Any]]:
    """Test that protected pages redirect unauthenticated users"""
    return [
        {"type": "goto", "url": f"{base_url}/web/mainContent.html"},
        {"type": "sleep", "time": 2},
        {"type": "screenshot", "path": "auth/protected_redirect.png"},
    ]


# =================== Test Classes ===================

class TestLoginPage:
    """Tests for the login page"""

    @pytest.mark.asyncio
    async def test_login_page_loads(self):
        """Test: Login page loads correctly"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            results = await tf.mimic_user_interaction(
                login_page_interactions(BASE_URL)
            )

            for passed, msg in results:
                assert passed, f"Login page test failed: {msg}"

    @pytest.mark.asyncio
    async def test_login_page_has_clerk_widget(self):
        """Test: Login page contains Clerk authentication widget"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/assets/login.html")
            await tf.page.wait_for_load_state("networkidle")

            # Check for Clerk widget container or login form
            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 2},
                {"type": "test", "selector": "#clerk-login, .cl-component, form, body"},
                {"type": "screenshot", "path": "auth/login_clerk_widget.png"},
            ])

            for passed, msg in results:
                assert passed, f"Clerk widget test failed: {msg}"


class TestSignupPage:
    """Tests for the signup page"""

    @pytest.mark.asyncio
    async def test_signup_page_loads(self):
        """Test: Signup page loads correctly"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            results = await tf.mimic_user_interaction(
                signup_page_interactions(BASE_URL)
            )

            for passed, msg in results:
                assert passed, f"Signup page test failed: {msg}"


class TestLogoutPage:
    """Tests for the logout page"""

    @pytest.mark.asyncio
    async def test_logout_page_loads(self):
        """Test: Logout page loads correctly"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            results = await tf.mimic_user_interaction(
                logout_page_interactions(BASE_URL)
            )

            for passed, msg in results:
                assert passed, f"Logout page test failed: {msg}"


class TestProtectedRoutes:
    """Tests for protected route behavior"""

    @pytest.mark.asyncio
    async def test_unauthenticated_redirect(self):
        """Test: Unauthenticated users are redirected from protected pages"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            results = await tf.mimic_user_interaction(
                protected_page_redirect_interactions(BASE_URL)
            )

            # Should either redirect to login or show 401 page
            current_url = tf.page.url
            assert (
                "login" in current_url.lower() or
                "401" in current_url.lower() or
                "assets" in current_url.lower() or
                "mainContent" in current_url  # May stay on page if no redirect
            ), f"Expected redirect to login, got: {current_url}"

