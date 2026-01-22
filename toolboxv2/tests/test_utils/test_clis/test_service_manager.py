"""
Unit Tests für den Service Manager

Verwendet unittest (NICHT pytest!)
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestServiceDefinition(unittest.TestCase):
    """Tests für ServiceDefinition Dataclass"""

    def test_service_definition_creation(self):
        """Test dass ServiceDefinition korrekt erstellt wird"""
        from toolboxv2.utils.clis.service_manager import ServiceDefinition

        svc = ServiceDefinition(
            name="test_service",
            description="Test Service Description",
            category="core",
            module="test.module",
            entry_point="main",
            is_async=True,
            runner_key="test"
        )

        self.assertEqual(svc.name, "test_service")
        self.assertEqual(svc.description, "Test Service Description")
        self.assertEqual(svc.category, "core")
        self.assertEqual(svc.module, "test.module")
        self.assertEqual(svc.entry_point, "main")
        self.assertTrue(svc.is_async)
        self.assertEqual(svc.runner_key, "test")

    def test_service_definition_defaults(self):
        """Test dass Defaults korrekt gesetzt werden"""
        from toolboxv2.utils.clis.service_manager import ServiceDefinition

        svc = ServiceDefinition(
            name="test",
            description="Test",
            category="extension",
            module="test",
            entry_point="run"
        )

        self.assertFalse(svc.is_async)
        self.assertIsNone(svc.runner_key)


class TestServiceRegistry(unittest.TestCase):
    """Tests für ServiceRegistry Singleton"""

    def test_registry_is_singleton(self):
        """Test dass Registry ein Singleton ist"""
        from toolboxv2.utils.clis.service_manager import ServiceRegistry

        reg1 = ServiceRegistry()
        reg2 = ServiceRegistry()

        self.assertIs(reg1, reg2)

    def test_registry_has_builtin_services(self):
        """Test dass Built-in Services registriert sind"""
        from toolboxv2.utils.clis.service_manager import ServiceRegistry

        registry = ServiceRegistry()

        # Core services
        self.assertIsNotNone(registry.get("workers"))
        self.assertIsNotNone(registry.get("db"))
        self.assertIsNotNone(registry.get("user"))
        self.assertIsNotNone(registry.get("run"))

        # Infrastructure services
        self.assertIsNotNone(registry.get("config"))
        self.assertIsNotNone(registry.get("session"))
        self.assertIsNotNone(registry.get("broker"))

        # Extension services
        self.assertIsNotNone(registry.get("p2p"))
        self.assertIsNotNone(registry.get("mcp"))

    def test_registry_get_by_category(self):
        """Test get_by_category Methode"""
        from toolboxv2.utils.clis.service_manager import ServiceRegistry

        registry = ServiceRegistry()

        core_services = registry.get_by_category("core")
        self.assertTrue(len(core_services) >= 4)

        infra_services = registry.get_by_category("infrastructure")
        self.assertTrue(len(infra_services) >= 5)

    def test_registry_list_names(self):
        """Test list_names Methode"""
        from toolboxv2.utils.clis.service_manager import ServiceRegistry

        registry = ServiceRegistry()
        names = registry.list_names()

        self.assertIn("workers", names)
        self.assertIn("db", names)
        self.assertIn("mcp", names)


class TestServiceStartResult(unittest.TestCase):
    """Tests für ServiceStartResult Dataclass"""

    def test_start_result_success(self):
        """Test erfolgreicher Start"""
        from toolboxv2.utils.clis.service_manager import ServiceStartResult

        result = ServiceStartResult(
            name="test",
            success=True,
            pid=12345
        )

        self.assertTrue(result.success)
        self.assertEqual(result.pid, 12345)
        self.assertIsNone(result.error)

    def test_start_result_failure(self):
        """Test fehlgeschlagener Start"""
        from toolboxv2.utils.clis.service_manager import ServiceStartResult

        result = ServiceStartResult(
            name="test",
            success=False,
            error="Process crashed"
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.pid)
        self.assertEqual(result.error, "Process crashed")


class TestServiceManagerConfig(unittest.TestCase):
    """Tests für ServiceManager Konfiguration"""

    def setUp(self):
        """Setup temporäres Verzeichnis für Tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "services.json"
        self.pids_dir = Path(self.temp_dir) / "pids"
        self.pids_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_manager(self):
        """Erstelle Manager mit gemockten Pfaden"""
        from toolboxv2.utils.clis.service_manager import ServiceManager

        manager = ServiceManager.__new__(ServiceManager)
        manager.pids_dir = self.pids_dir
        manager.config_path = self.config_path
        return manager

    def test_load_config_empty(self):
        """Test load_config wenn keine Config existiert"""
        manager = self._create_manager()
        config = manager.load_config()

        self.assertEqual(config, {"services": {}})

    def test_save_and_load_config(self):
        """Test save_config und load_config"""
        manager = self._create_manager()

        test_config = {
            "services": {
                "workers": {"auto_start": True},
                "db": {"auto_start": False}
            }
        }

        manager.save_config(test_config)
        loaded = manager.load_config()

        self.assertEqual(loaded, test_config)

    def test_configure_service(self):
        """Test configure_service Methode"""
        manager = self._create_manager()

        manager.configure_service("workers", auto_start=True)
        config = manager.load_config()

        self.assertTrue(config["services"]["workers"]["auto_start"])

        manager.configure_service("workers", auto_start=False, auto_restart=True)
        config = manager.load_config()

        self.assertFalse(config["services"]["workers"]["auto_start"])
        self.assertTrue(config["services"]["workers"]["auto_restart"])

    def test_get_auto_start_services(self):
        """Test get_auto_start_services Methode"""
        manager = self._create_manager()

        manager.configure_service("workers", auto_start=True)
        manager.configure_service("db", auto_start=False)
        manager.configure_service("mcp", auto_start=True)

        auto_start = manager.get_auto_start_services()

        self.assertIn("workers", auto_start)
        self.assertIn("mcp", auto_start)
        self.assertNotIn("db", auto_start)


class TestServiceManagerPID(unittest.TestCase):
    """Tests für PID-File Handling"""

    def setUp(self):
        """Setup temporäres Verzeichnis für Tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "services.json"
        self.pids_dir = Path(self.temp_dir) / "pids"
        self.pids_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_manager(self):
        """Erstelle Manager mit gemockten Pfaden"""
        from toolboxv2.utils.clis.service_manager import ServiceManager

        manager = ServiceManager.__new__(ServiceManager)
        manager.pids_dir = self.pids_dir
        manager.config_path = self.config_path
        return manager

    def test_is_service_running_no_pid_file(self):
        """Test is_service_running wenn kein PID-File existiert"""
        manager = self._create_manager()

        running, pid = manager.is_service_running("nonexistent")

        self.assertFalse(running)
        self.assertIsNone(pid)

    def test_is_service_running_stale_pid(self):
        """Test is_service_running mit stale PID"""
        manager = self._create_manager()

        # Schreibe ungültige PID
        pid_file = self.pids_dir / "stale.pid"
        pid_file.write_text("999999999")  # Sehr hohe PID, existiert nicht

        running, pid = manager.is_service_running("stale")

        self.assertFalse(running)
        self.assertIsNone(pid)
        # PID-File sollte gelöscht worden sein
        self.assertFalse(pid_file.exists())

    def test_is_service_running_invalid_pid_file(self):
        """Test is_service_running mit ungültigem PID-File"""
        manager = self._create_manager()

        pid_file = self.pids_dir / "invalid.pid"
        pid_file.write_text("not_a_number")

        running, pid = manager.is_service_running("invalid")

        self.assertFalse(running)
        self.assertIsNone(pid)


class TestServiceManagerStatus(unittest.TestCase):
    """Tests für Status-Methoden"""

    def setUp(self):
        """Setup temporäres Verzeichnis für Tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "services.json"
        self.pids_dir = Path(self.temp_dir) / "pids"
        self.pids_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_manager(self):
        """Erstelle Manager mit gemockten Pfaden"""
        from toolboxv2.utils.clis.service_manager import ServiceManager

        manager = ServiceManager.__new__(ServiceManager)
        manager.pids_dir = self.pids_dir
        manager.config_path = self.config_path
        return manager

    def test_get_all_status_empty(self):
        """Test get_all_status ohne konfigurierte Services"""
        manager = self._create_manager()

        # Ohne Registry-Services
        status = manager.get_all_status(include_registry=False)
        self.assertEqual(status, {})

    def test_get_all_status_with_registry(self):
        """Test get_all_status mit Registry-Services"""
        manager = self._create_manager()

        status = manager.get_all_status(include_registry=True)

        # Sollte Registry-Services enthalten
        self.assertIn("workers", status)
        self.assertIn("db", status)
        self.assertIn("mcp", status)

    def test_get_service_info(self):
        """Test get_service_info Methode"""
        manager = self._create_manager()

        info = manager.get_service_info("workers")

        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "workers")
        self.assertEqual(info["category"], "core")
        self.assertFalse(info["running"])
        self.assertIn("description", info)

    def test_get_service_info_unknown(self):
        """Test get_service_info für unbekannten Service"""
        manager = self._create_manager()

        info = manager.get_service_info("unknown_service_xyz")

        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()

