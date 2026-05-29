"""
Tests für Feature State Sync und Pack/Unpack System

Verwendet unittest (kein pytest!)
"""
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml


class TestFeatureStateElement(unittest.TestCase):
    """Tests für FeatureStateElement Dataclass"""

    def test_feature_state_element_creation(self):
        """Test FeatureStateElement erstellen"""
        from toolboxv2.utils.system.state_system import FeatureStateElement

        elem = FeatureStateElement(
            name="test",
            version="1.0.0",
            enabled=True,
            shasum="abc123",
            source="local",
        )

        self.assertEqual(elem.name, "test")
        self.assertEqual(elem.version, "1.0.0")
        self.assertTrue(elem.enabled)
        self.assertEqual(elem.shasum, "abc123")
        self.assertEqual(elem.source, "local")
        self.assertEqual(elem.dependencies, [])
        self.assertEqual(elem.requires, [])

    def test_feature_state_element_str(self):
        """Test FeatureStateElement String Darstellung"""
        from toolboxv2.utils.system.state_system import FeatureStateElement

        elem_enabled = FeatureStateElement(name="web", version="0.1.0", enabled=True)
        elem_disabled = FeatureStateElement(name="cli", version="0.2.0", enabled=False)

        self.assertIn("✓", str(elem_enabled))
        self.assertIn("web", str(elem_enabled))
        self.assertIn("✗", str(elem_disabled))
        self.assertIn("cli", str(elem_disabled))

    def test_feature_state_element_defaults(self):
        """Test FeatureStateElement Default-Werte"""
        from toolboxv2.utils.system.state_system import FeatureStateElement

        elem = FeatureStateElement(name="minimal")

        self.assertEqual(elem.version, "0.0.0")
        self.assertFalse(elem.enabled)
        self.assertEqual(elem.shasum, "")
        self.assertEqual(elem.source, "local")


class TestTbStateWithFeatures(unittest.TestCase):
    """Tests für TbState mit Features"""

    def test_tb_state_features_field(self):
        """Test TbState hat features Feld"""
        from toolboxv2.utils.system.state_system import TbState, FeatureStateElement

        state = TbState(
            utils={},
            mods={},
            installable={},
            runnable={},
            api={},
            app={},
            features={
                "core": FeatureStateElement(name="core", enabled=True),
                "web": FeatureStateElement(name="web", enabled=False),
            }
        )

        self.assertIn("core", state.features)
        self.assertIn("web", state.features)
        self.assertTrue(state.features["core"].enabled)
        self.assertFalse(state.features["web"].enabled)

    def test_tb_state_features_in_str(self):
        """Test Features werden in __str__ angezeigt"""
        from toolboxv2.utils.system.state_system import TbState, FeatureStateElement

        state = TbState(
            utils={},
            mods={},
            installable={},
            runnable={},
            api={},
            app={},
            features={
                "test": FeatureStateElement(name="test", enabled=True),
            }
        )

        output = str(state)
        self.assertIn("Features", output)
        self.assertIn("test", output)


class TestProcessFeatures(unittest.TestCase):
    """Tests für process_features Funktion"""

    def setUp(self):
        """Erstelle temporäres Features-Verzeichnis"""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_dir.mkdir()

    def tearDown(self):
        """Lösche temporäres Verzeichnis"""
        shutil.rmtree(self.temp_dir)

    def test_process_features_empty(self):
        """Test process_features mit leerem Verzeichnis"""
        from toolboxv2.utils.system.state_system import process_features

        features = process_features(str(self.features_dir))
        self.assertEqual(features, {})

    def test_process_features_single(self):
        """Test process_features mit einem Feature"""
        from toolboxv2.utils.system.state_system import process_features

        # Erstelle Feature
        feature_dir = self.features_dir / "test_feature"
        feature_dir.mkdir()

        feature_yaml = feature_dir / "feature.yaml"
        feature_yaml.write_text(yaml.dump({
            "version": "1.2.3",
            "enabled": True,
            "dependencies": ["requests"],
            "requires": ["core"],
        }))

        features = process_features(str(self.features_dir))

        self.assertIn("test_feature", features)
        self.assertEqual(features["test_feature"].version, "1.2.3")
        self.assertTrue(features["test_feature"].enabled)
        self.assertEqual(features["test_feature"].dependencies, ["requests"])
        self.assertEqual(features["test_feature"].requires, ["core"])
        self.assertNotEqual(features["test_feature"].shasum, "")

    def test_process_features_multiple(self):
        """Test process_features mit mehreren Features"""
        from toolboxv2.utils.system.state_system import process_features

        for name in ["core", "web", "cli"]:
            feature_dir = self.features_dir / name
            feature_dir.mkdir()
            (feature_dir / "feature.yaml").write_text(yaml.dump({
                "version": "0.1.0",
                "enabled": name == "core",
            }))

        features = process_features(str(self.features_dir))

        self.assertEqual(len(features), 3)
        self.assertTrue(features["core"].enabled)
        self.assertFalse(features["web"].enabled)


class TestFeatureManagerStateSync(unittest.TestCase):
    """Tests für FeatureManager State Sync"""

    def setUp(self):
        """Setup mit temporärem Verzeichnis"""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_dir.mkdir()

        # Erstelle Test-Feature
        core_dir = self.features_dir / "core"
        core_dir.mkdir()
        (core_dir / "feature.yaml").write_text(yaml.dump({
            "name": "core",
"version": "0.1.25",
"enabled": True,
"immutable": True,
"description": "Core ToolBox — App runtime, types, logging, CloudM, DB, core flows",
"files":[
  # ── App Kernel ─────────────────────────────────────────────
  "utils/toolbox.py",
  "utils/singelton_class.py",
  "utils/__init__.py",
  "__genv__.py",
  "feature_loader.py",
  "feature_loader_registry.py",
  "setup_helper.py",
  # ── System ─────────────────────────────────────────────────
  "utils/system/*",
  "utils/manifest/*",
  "utils/security/*",
  "utils/tbx/*",
  "utils/brodcast/*",
  # ── Extras (core-subset) ────────────────────────────────────
  "utils/extras/Style.py",
  "utils/extras/blobs.py",
  "utils/extras/__init__.py",
  "utils/extras/db/*",
  "utils/extras/code_analyzer/*",
  "utils/extras/depsy/*",
  "utils/extras/install/*",
  "utils/extras/notification.py",
  "utils/extras/keword_matcher.py",
  "utils/extras/registry_client.py",
  "utils/extras/reqbuilder.py",
  "utils/extras/gist_control.py",
  "utils/extras/dev_runner.py",
  "utils/extras/gateway_live_client.py",
  "utils/extras/helper_test_functions.py",
  "utils/extras/mkdocs.py",
  # ── Core Flows ─────────────────────────────────────────────
   "flows/__init__.py",
   "flows/bg.py",
   "flows/core0.py",
   "flows/pyshell.py",
   "flows/minicli.py",
   "flows/adaptive_prompt_system.py",
   "flows/auto_git_commit.py",
   "flows/toolbox_admin.py",
  # ── Core Mods ──────────────────────────────────────────────
  "mods/CloudM/*",
  "mods/DB/*",
  "mods/__init__.py"],
"imports":[
  "toolboxv2.utils.toolbox",
  "toolboxv2.flows",
  "toolboxv2.mods.CloudM",
  "toolboxv2.mods.DB" ],
"dependencies": [
   "radon",
   "qrcode",
   "watchfiles",
   "gitpython",
   "redis",
   "xmltodict",
   "minio",
   "prompt-toolkit",
   "opentelemetry-api",
   "opentelemetry-sdk",
   "docker" ],
"requires": [],
"commands": [],

        }))

    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)
        # Reset Singleton
        from toolboxv2.utils.system.feature_manager import FeatureManager
        FeatureManager._instances = {}

    def test_sync_with_state(self):
        """Test sync_with_state übernimmt enabled Status"""
        from toolboxv2.utils.system.feature_manager import FeatureManager
        from toolboxv2.utils.system.state_system import TbState, FeatureStateElement

        fm = FeatureManager(features_dir=str(self.features_dir))

        # State mit enabled=True
        state = TbState(
            utils={}, mods={}, installable={},
            runnable={}, api={}, app={},
            features={
                "core": FeatureStateElement(name="core", enabled=True, version="0.1.0"),
            }
        )

        fm.sync_with_state(state)

        # Jetzt sollte core enabled sein
        self.assertTrue(fm.features["core"].enabled)

    def test_export_to_state(self):
        """Test export_to_state gibt korrekte Daten zurück"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))
        fm.features["core"].enabled = True

        exported = fm.export_to_state()

        self.assertIn("core", exported)
        self.assertEqual(exported["core"]["name"], "core")
        self.assertTrue(exported["core"]["enabled"])
        fm.features["core"].enabled = False


class TestFeatureManagerPackUnpack(unittest.TestCase):
    """Tests für FeatureManager Pack/Unpack"""

    def setUp(self):
        """Setup mit temporären Verzeichnissen"""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_dir.mkdir()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()

        # Erstelle Test-Feature
        test_dir = self.features_dir / "packtest"
        test_dir.mkdir()
        (test_dir / "feature.yaml").write_text(yaml.dump({
            "name": "core",
"version": "0.1.25",
"enabled": True,
"immutable": True,
"description": "Core ToolBox — App runtime, types, logging, CloudM, DB, core flows",
"files":[
  # ── App Kernel ─────────────────────────────────────────────
  "utils/toolbox.py",
  "utils/singelton_class.py",
  "utils/__init__.py",
  "__genv__.py",
  "feature_loader.py",
  "feature_loader_registry.py",
  "setup_helper.py",
  # ── System ─────────────────────────────────────────────────
  "utils/system/*",
  "utils/manifest/*",
  "utils/security/*",
  "utils/tbx/*",
  "utils/brodcast/*",
  # ── Extras (core-subset) ────────────────────────────────────
  "utils/extras/Style.py",
  "utils/extras/blobs.py",
  "utils/extras/__init__.py",
  "utils/extras/db/*",
  "utils/extras/code_analyzer/*",
  "utils/extras/depsy/*",
  "utils/extras/install/*",
  "utils/extras/notification.py",
  "utils/extras/registry_client.py",
  "utils/extras/reqbuilder.py",
  "utils/extras/gist_control.py",
  "utils/extras/dev_runner.py",
  "utils/extras/gateway_live_client.py",
  "utils/extras/helper_test_functions.py",
  "utils/extras/mkdocs.py",
  # ── Core Flows ─────────────────────────────────────────────
   "flows/__init__.py",
   "flows/bg.py",
   "flows/core0.py",
   "flows/pyshell.py",
   "flows/minicli.py",
   "flows/adaptive_prompt_system.py",
   "flows/auto_git_commit.py",
   "flows/toolbox_admin.py",
  # ── Core Mods ──────────────────────────────────────────────
  "mods/CloudM/*",
  "mods/DB/*",
  "mods/__init__.py"],
"imports":[
  "toolboxv2.utils.toolbox",
  "toolboxv2.flows",
  "toolboxv2.mods.CloudM",
  "toolboxv2.mods.DB" ],
"dependencies": [
   "radon",
   "qrcode",
   "watchfiles",
   "gitpython",
   "redis",
   "xmltodict",
   "minio",
   "prompt-toolkit",
   "opentelemetry-api",
   "opentelemetry-sdk",
   "docker" ],
"requires": [],
"commands": [],

        }))

    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)
        from toolboxv2.utils.system.feature_manager import FeatureManager
        FeatureManager._instances = {}

    def test_pack_feature_creates_zip(self):
        """Test pack_feature erstellt ZIP-Datei"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))

        zip_path = fm.pack_feature("packtest", output_path=str(self.output_dir))

        self.assertIsNotNone(zip_path)
        self.assertTrue(Path(zip_path).exists())
        self.assertTrue(zip_path.endswith(".zip"))

    def test_pack_feature_contains_files(self):
        """Test gepacktes Feature enthält benötigte Dateien"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))
        zip_path = fm.pack_feature("packtest", output_path=str(self.output_dir))

        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            self.assertIn("feature.yaml", names)
            self.assertIn("requirements.txt", names)
            self.assertIn("_metadata.yaml", names)

    def test_pack_feature_metadata(self):
        """Test Metadata im gepackten Feature"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))
        zip_path = fm.pack_feature("packtest", output_path=str(self.output_dir))

        with zipfile.ZipFile(zip_path, 'r') as zf:
            metadata = yaml.safe_load(zf.read("_metadata.yaml").decode("utf-8"))
            self.assertEqual(metadata["feature_name"], "packtest")
            self.assertIn("packed_at", metadata)

    def test_unpack_feature(self):
        """Test unpack_feature entpackt korrekt"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))

        # Pack
        zip_path = fm.pack_feature("packtest", output_path=str(self.output_dir))

        # Lösche Original
        shutil.rmtree(self.features_dir / "packtest")
        del fm.features["packtest"]

        # Unpack
        feature_name = fm.unpack_feature(zip_path, target_dir=str(self.features_dir))

        self.assertEqual(feature_name, "packtest")
        self.assertTrue((self.features_dir / "packtest" / "feature.yaml").exists())

    def test_unpack_creates_backup(self):
        """Test unpack_feature erstellt Backup bei existierendem Feature"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))
        zip_path = fm.pack_feature("packtest", output_path=str(self.output_dir))

        # Unpack über existierendes Feature
        fm.unpack_feature(zip_path, target_dir=str(self.features_dir))

        # Backup sollte existieren
        backup_path = self.features_dir / "packtest.backup"
        self.assertTrue(backup_path.exists())

    def test_pack_unknown_feature_returns_none(self):
        """Test pack_feature mit unbekanntem Feature gibt None zurück"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))
        result = fm.pack_feature("nonexistent")

        self.assertIsNone(result)

    def test_unpack_invalid_zip_returns_none(self):
        """Test unpack_feature mit ungültigem ZIP gibt None zurück"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))

        # Erstelle ungültiges ZIP (ohne feature.yaml)
        invalid_zip = self.output_dir / "invalid.zip"
        with zipfile.ZipFile(invalid_zip, 'w') as zf:
            zf.writestr("random.txt", "not a feature")

        result = fm.unpack_feature(str(invalid_zip))

        self.assertIsNone(result)


class TestListPackedFeatures(unittest.TestCase):
    """Tests für list_packed_features"""

    def setUp(self):
        """Setup"""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_dir.mkdir()
        self.packages_dir = Path(self.temp_dir) / "packages"
        self.packages_dir.mkdir()

    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)
        from toolboxv2.utils.system.feature_manager import FeatureManager
        FeatureManager._instances = {}

    def test_list_packed_features_empty(self):
        """Test list_packed_features mit leerem Verzeichnis"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))
        packages = fm.list_packed_features(search_dir=str(self.packages_dir))

        self.assertEqual(packages, [])

    def test_list_packed_features_finds_packages(self):
        """Test list_packed_features findet Packages"""
        from toolboxv2.utils.system.feature_manager import FeatureManager

        fm = FeatureManager(features_dir=str(self.features_dir))

        # Erstelle Test-Package
        pkg_path = self.packages_dir / "tbv2-feature-test-1.0.0.zip"
        with zipfile.ZipFile(pkg_path, 'w') as zf:
            zf.writestr("feature.yaml", yaml.dump({"version": "1.0.0"}))
            zf.writestr("_metadata.yaml", yaml.dump({
                "feature_name": "test",
                "version": "1.0.0",
                "packed_at": "2025-01-29T12:00:00",
            }))

        packages = fm.list_packed_features(search_dir=str(self.packages_dir))

        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0]["feature_name"], "test")
        self.assertEqual(packages[0]["version"], "1.0.0")


if __name__ == "__main__":
    unittest.main()
