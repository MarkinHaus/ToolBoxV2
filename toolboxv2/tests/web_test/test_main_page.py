# toolboxv2/tests/web_test/test_main_page.py
"""
ToolBoxV2 - E2E Web Page Tests

Pytest-konforme Tests für die Web-Seiten.
Nutzt das bestehende WebTestFramework aus web_util.py.

Ausführung:
    pytest toolboxv2/tests/web_test/test_main_page.py -v
    pytest toolboxv2/tests/web_test/test_main_page.py -v -k "test_index"
"""

import pytest
from typing import List, Dict, Any

from toolboxv2.tests.a_util import async_test
from toolboxv2.tests.web_util import AsyncWebTestFramework
from toolboxv2.tests.test_web import (
    TEST_SERVER_BASE_URL,
    is_server_running,
)


# =================== Konfiguration ===================

# Basis-URL für Tests (kann überschrieben werden)
BASE_URL = TEST_SERVER_BASE_URL


def _server_available() -> bool:
    """Prüft ob der Test-Server erreichbar ist"""
    return is_server_running()


# =================== Interaktions-Definitionen ===================
# Diese Funktionen definieren die Test-Interaktionen im Framework-Format


def main_page_interactions() -> List[Dict[str, Any]]:
    """
    Interaktionen für die ToolBoxV2 Hauptseite

    Returns:
        Liste von Interaktions-Dictionaries
    """
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/core0/index.html"},
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "main_page/initial.png"},
        {"type": "test", "selector": "body"},
    ]


def installer_page_interactions() -> List[Dict[str, Any]]:
    """
    Interaktionen für die Installer-Seite

    Returns:
        Liste von Interaktions-Dictionaries
    """
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/core0/Installer.html"},
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "installer/initial.png"},
        {"type": "test", "selector": "#os-selection"},
    ]


def contact_page_interactions() -> List[Dict[str, Any]]:
    """
    Interaktionen für die Kontakt-Seite

    Returns:
        Liste von Interaktions-Dictionaries
    """
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/core0/kontakt.html"},
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "contact/initial.png"},
        {"type": "test", "selector": "#name"},
        {"type": "test", "selector": "#email"},
        {"type": "test", "selector": "#subject"},
        {"type": "test", "selector": "#message"},
    ]


def login_page_interactions() -> List[Dict[str, Any]]:
    """
    Interaktionen für die Login-Seite

    Returns:
        Liste von Interaktions-Dictionaries
    """
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/assets/login.html"},
        {"type": "sleep", "time": 2},
        {"type": "screenshot", "path": "auth/login.png"},
        {"type": "test", "selector": "body"},
    ]


def signup_page_interactions() -> List[Dict[str, Any]]:
    """
    Interaktionen für die Signup-Seite

    Returns:
        Liste von Interaktions-Dictionaries
    """
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/assets/signup.html"},
        {"type": "sleep", "time": 2},
        {"type": "screenshot", "path": "auth/signup.png"},
        {"type": "test", "selector": "body"},
    ]


# =================== Pytest Tests ===================


@pytest.fixture(scope="module")
def check_server():
    """Fixture: Prüft ob Server verfügbar ist"""
    if not _server_available():
        pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")


class TestPublicPages:
    """Tests für öffentlich zugängliche Seiten"""

    @async_test
    async def test_index_page_loads(self, check_server):
        """Test: Index-Seite lädt korrekt"""
        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(main_page_interactions())

            # Alle Interaktionen müssen erfolgreich sein
            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @async_test
    async def test_installer_page_loads(self, check_server):
        """Test: Installer-Seite lädt korrekt"""
        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(installer_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @async_test
    async def test_contact_page_loads(self, check_server):
        """Test: Kontakt-Seite lädt und Formular-Elemente existieren"""
        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(contact_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"


class TestAuthPages:
    """Tests für Authentifizierungs-Seiten"""

    @async_test
    async def test_login_page_loads(self, check_server):
        """Test: Login-Seite lädt korrekt"""
        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(login_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @async_test
    async def test_signup_page_loads(self, check_server):
        """Test: Signup-Seite lädt korrekt"""
        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(signup_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

