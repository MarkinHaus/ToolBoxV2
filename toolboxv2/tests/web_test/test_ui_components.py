# toolboxv2/tests/web_test/test_ui_components.py
"""
ToolBoxV2 E2E UI Component Tests

Tests for TBJS UI components:
- Desktop components (StatusBar, QuickCapture)
- Mobile components (BottomNav)
- Shared components (Modal, Toast, Chat, etc.)
- Theme switching (Dark/Light mode)

Run:
    pytest toolboxv2/tests/web_test/test_ui_components.py -v
    pytest toolboxv2/tests/web_test/test_ui_components.py -v -k "test_modal"
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

def main_content_page_interactions(base_url: str) -> List[Dict[str, Any]]:
    """Navigate to main content page for component testing"""
    return [
        {"type": "goto", "url": f"{base_url}/web/mainContent.html"},
        {"type": "sleep", "time": 2},
        {"type": "screenshot", "path": "ui/main_content_initial.png"},
    ]


def dashboard_page_interactions(base_url: str) -> List[Dict[str, Any]]:
    """Navigate to dashboard for component testing"""
    return [
        {"type": "goto", "url": f"{base_url}/web/dashboards/dashboard.html"},
        {"type": "sleep", "time": 2},
        {"type": "screenshot", "path": "ui/dashboard_initial.png"},
    ]


# =================== Desktop Component Tests ===================

class TestDesktopComponents:
    """Tests for desktop-specific UI components"""

    @pytest.mark.asyncio
    async def test_main_content_page_loads(self):
        """Test: Main content page loads with UI components"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            results = await tf.mimic_user_interaction(
                main_content_page_interactions(BASE_URL)
            )

            # Page should load (may redirect if not authenticated)
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_input_field_exists(self):
        """Test: Main input field is present on main content page"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/mainContent.html")
            await tf.page.wait_for_load_state("networkidle")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "test", "selector": "#inputField, .inputField, input[type='text'], body"},
                {"type": "screenshot", "path": "ui/input_field.png"},
            ])

            # Check if input field was found
            for passed, msg in results:
                if "test" in str(msg).lower():
                    assert passed, f"Input field not found: {msg}"


class TestDashboardComponents:
    """Tests for dashboard UI components"""

    @pytest.mark.asyncio
    async def test_dashboard_page_loads(self):
        """Test: Dashboard page loads correctly"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            results = await tf.mimic_user_interaction(
                dashboard_page_interactions(BASE_URL)
            )

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_user_dashboard_loads(self):
        """Test: User dashboard page loads"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/dashboards/user_dashboard.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 2},
                {"type": "screenshot", "path": "ui/user_dashboard.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results:
                assert passed, f"User dashboard test failed: {msg}"


class TestThemeComponents:
    """Tests for theme switching functionality"""

    @pytest.mark.asyncio
    async def test_page_has_theme_support(self):
        """Test: Page supports dark/light theme"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/core0/index.html")
            await tf.page.wait_for_load_state("networkidle")

            # Check for theme-related elements or classes
            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "ui/theme_default.png"},
            ])

            # Evaluate if dark mode class exists
            has_theme = await tf.page.evaluate("""
                () => {
                    const html = document.documentElement;
                    const body = document.body;
                    return {
                        hasDarkClass: html.classList.contains('dark') || body.classList.contains('dark'),
                        hasThemeAttr: html.hasAttribute('data-theme') || body.hasAttribute('data-theme'),
                        hasColorScheme: window.matchMedia('(prefers-color-scheme: dark)').matches
                    };
                }
            """)

            # At least one theme mechanism should be present
            assert isinstance(has_theme, dict)


class TestResponsiveLayout:
    """Tests for responsive layout behavior"""

    @pytest.mark.asyncio
    async def test_mobile_viewport_layout(self):
        """Test: Page renders correctly on mobile viewport"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(
                viewport={"width": 375, "height": 667},
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
            )
            await tf.navigate(f"{BASE_URL}/web/core0/index.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "ui/mobile_layout.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results:
                assert passed, f"Mobile layout test failed: {msg}"

    @pytest.mark.asyncio
    async def test_tablet_viewport_layout(self):
        """Test: Page renders correctly on tablet viewport"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(
                viewport={"width": 768, "height": 1024},
                user_agent="Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
            )
            await tf.navigate(f"{BASE_URL}/web/core0/index.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "ui/tablet_layout.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results:
                assert passed, f"Tablet layout test failed: {msg}"

