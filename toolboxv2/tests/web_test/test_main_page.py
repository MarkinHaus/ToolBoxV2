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

from toolboxv2.tests.web_util import AsyncWebTestFramework

# Import server config
try:
    from toolboxv2.tests.test_web import (
        TEST_SERVER_BASE_URL,
        TEST_WEB_BASE_URL,
        is_server_running,
    )
except ImportError:
    TEST_SERVER_BASE_URL = "http://localhost:8000"
    TEST_WEB_BASE_URL = "http://localhost:80"
    def is_server_running():
        return False


# =================== Konfiguration ===================

# Basis-URL für Web-Seiten (nginx/tauri auf Port 80)
BASE_URL = TEST_WEB_BASE_URL
# API-URL für Backend-Aufrufe (Rust workers auf Port 8000)
API_URL = TEST_SERVER_BASE_URL


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


class TestPublicPages:
    """Tests für öffentlich zugängliche Seiten"""

    @pytest.mark.asyncio
    async def test_index_page_loads(self):
        """Test: Index-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(main_page_interactions())

            # Alle Interaktionen müssen erfolgreich sein
            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @pytest.mark.asyncio
    async def test_installer_page_loads(self):
        """Test: Installer-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(installer_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @pytest.mark.asyncio
    async def test_contact_page_loads(self):
        """Test: Kontakt-Seite lädt und Formular-Elemente existieren"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(contact_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"


class TestAuthPages:
    """Tests für Authentifizierungs-Seiten"""

    @pytest.mark.asyncio
    async def test_login_page_loads(self):
        """Test: Login-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(login_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @pytest.mark.asyncio
    async def test_signup_page_loads(self):
        """Test: Signup-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(signup_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"


# =================== Additional Page Tests ===================

def roadmap_page_interactions() -> List[Dict[str, Any]]:
    """Interaktionen für die Roadmap-Seite"""
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/core0/roadmap.html"},
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "pages/roadmap.png"},
        {"type": "test", "selector": "body"},
    ]


def main_idea_page_interactions() -> List[Dict[str, Any]]:
    """Interaktionen für die MainIdea-Seite"""
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/core0/MainIdea.html"},
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "pages/main_idea.png"},
        {"type": "test", "selector": "body"},
    ]


def toolbox_infos_page_interactions() -> List[Dict[str, Any]]:
    """Interaktionen für die ToolBoxInfos-Seite"""
    return [
        {"type": "goto", "url": f"{BASE_URL}/web/core0/ToolBoxInfos.html"},
        {"type": "sleep", "time": 1},
        {"type": "screenshot", "path": "pages/toolbox_infos.png"},
        {"type": "test", "selector": "body"},
    ]


class TestAdditionalPages:
    """Tests für zusätzliche öffentliche Seiten"""

    @pytest.mark.asyncio
    async def test_roadmap_page_loads(self):
        """Test: Roadmap-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(roadmap_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @pytest.mark.asyncio
    async def test_main_idea_page_loads(self):
        """Test: MainIdea-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(main_idea_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @pytest.mark.asyncio
    async def test_toolbox_infos_page_loads(self):
        """Test: ToolBoxInfos-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction(toolbox_infos_page_interactions())

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"


class TestErrorPages:
    """Tests für Fehlerseiten"""

    @pytest.mark.asyncio
    async def test_401_page_loads(self):
        """Test: 401-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction([
                {"type": "goto", "url": f"{BASE_URL}/web/assets/401.html"},
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "errors/401.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

    @pytest.mark.asyncio
    async def test_404_page_loads(self):
        """Test: 404-Seite lädt korrekt"""
        if not _server_available():
            pytest.skip(f"Test-Server nicht erreichbar: {BASE_URL}")

        async with AsyncWebTestFramework(headless=True) as tf:
            await tf.create_context(viewport={"width": 1280, "height": 720})

            results = await tf.mimic_user_interaction([
                {"type": "goto", "url": f"{BASE_URL}/web/assets/404.html"},
                {"type": "sleep", "time": 1},
                {"type": "screenshot", "path": "errors/404.png"},
                {"type": "test", "selector": "body"},
            ])

            for passed, msg in results:
                assert passed, f"Interaktion fehlgeschlagen: {msg}"

