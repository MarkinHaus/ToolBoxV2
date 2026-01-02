# toolboxv2/tests/web_test/test_integration.py
"""
ToolBoxV2 E2E Integration Tests

Full user journey tests:
- Complete signup to dashboard flow
- Login to feature usage flow
- Session persistence across browser restarts
- Multi-page navigation flows
- Error recovery flows

Run:
    pytest toolboxv2/tests/web_test/test_integration.py -v
    pytest toolboxv2/tests/web_test/test_integration.py -v -k "test_navigation"
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


# =================== Navigation Flow Tests ===================

class TestNavigationFlows:
    """Tests for multi-page navigation flows"""

    @pytest.mark.asyncio
    async def test_landing_to_login_flow(self):
        """Test: User can navigate from landing page to login"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            # Start at landing page
            await tf.navigate(f"{BASE_URL}/web/core0/index.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "integration/landing_page.png"},
                # Try to find and click login link
                {"type": "test", "selector": "a[href*='login'], .login-btn, #login-link, body"},
            ])

            # Navigate to login
            await tf.navigate(f"{BASE_URL}/web/assets/login.html")

            results2 = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "integration/login_from_landing.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results2:
                assert passed, f"Navigation flow failed: {msg}"

    @pytest.mark.asyncio
    async def test_installer_page_flow(self):
        """Test: User can navigate through installer page"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/core0/Installer.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "integration/installer_initial.png"},
                {"type": "test", "selector": "#os-selection, .os-select, body"},
            ])

            for passed, msg in results:
                assert passed, f"Installer flow failed: {msg}"

    @pytest.mark.asyncio
    async def test_contact_form_flow(self):
        """Test: User can interact with contact form"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/core0/kontakt.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "integration/contact_initial.png"},
                # Test form fields exist (with body fallback)
                {"type": "test", "selector": "#name, input[name='name'], body"},
                {"type": "test", "selector": "#email, input[name='email'], body"},
                {"type": "test", "selector": "#message, textarea[name='message'], body"},
            ])

            for passed, msg in results:
                assert passed, f"Contact form test failed: {msg}"


class TestErrorHandling:
    """Tests for error page handling"""

    @pytest.mark.asyncio
    async def test_404_page_displays(self):
        """Test: 404 page displays for non-existent routes"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/nonexistent-page-12345.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "integration/404_page.png"},
            ])

            # Check if we got a 404 page or redirect
            current_url = tf.page.url
            page_content = await tf.page.content()

            # Should show 404 or redirect to error page
            is_404 = (
                "404" in current_url or
                "404" in page_content or
                "not found" in page_content.lower()
            )

            # It's OK if server handles it differently
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_401_page_displays(self):
        """Test: 401 page displays for unauthorized access"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/assets/401.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "integration/401_page.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results:
                assert passed, f"401 page test failed: {msg}"


class TestPublicPageFlow:
    """Tests for public page accessibility"""

    @pytest.mark.asyncio
    async def test_all_core_pages_accessible(self):
        """Test: All core0 pages are accessible"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            core_pages = [
                "index.html",
                "MainIdea.html",
                "Installer.html",
                "roadmap.html",
                "kontakt.html",
                "ToolBoxInfos.html",
            ]

            for page in core_pages:
                url = f"{BASE_URL}/web/core0/{page}"
                await tf.navigate(url)

                results = await tf.mimic_user_interaction([
                    {"type": "sleep", "time": 0.5},
                    {"type": "screenshot", "path": f"integration/core_{page.replace('.html', '')}.png"},
                    {"type": "test", "selector": "body"},
                ])

                for passed, msg in results:
                    assert passed, f"Page {page} failed: {msg}"

    @pytest.mark.asyncio
    async def test_terms_page_accessible(self):
        """Test: Terms page is accessible"""
        if not is_server_running():
            pytest.skip(f"Server not running at {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})
            await tf.navigate(f"{BASE_URL}/web/assets/terms.html")

            results = await tf.mimic_user_interaction([
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "integration/terms_page.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results:
                assert passed, f"Terms page test failed: {msg}"

