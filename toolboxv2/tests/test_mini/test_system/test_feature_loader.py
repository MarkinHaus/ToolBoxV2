"""
Tests für Feature Loader System

Verwendet unittest (kein pytest!)
"""
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml


class TestFeatureDetection(unittest.TestCase):
    """Tests für Feature Detection basierend auf installierten Packages"""

    def test_detect_no_extras(self):
        """Test dass ohne extras nur core benötigt wird"""
        from toolboxv2.feature_loader import get_required_features, CORE_FEATURE
        
        with patch('toolboxv2.feature_loader.detect_installed_extras', return_value=set()):
            required = get_required_features()
            self.assertEqual(required, {CORE_FEATURE})

    def test_detect_web_extra(self):
        """Test Web-Feature Detection"""
        from toolboxv2.feature_loader import get_required_features
        
        with patch('toolboxv2.feature_loader.detect_installed_extras', return_value={'web'}):
            required = get_required_features()
            self.assertIn('core', required)
            self.assertIn('web', required)

    def test_detect_multiple_extras(self):
        """Test Multiple Extras Detection"""
        from toolboxv2.feature_loader import get_required_features
        
        with patch('toolboxv2.feature_loader.detect_installed_extras', return_value={'web', 'cli'}):
            required = get_required_features()
            self.assertIn('core', required)
            self.assertIn('web', required)
            self.assertIn('cli', required)


class TestFeatureLoader(unittest.TestCase):
    """Tests für Feature Loader Funktionen"""

    def setUp(self):
        """Setup mit temporärem Verzeichnis"""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_packed_dir = Path(self.temp_dir) / "features_packed"
        self.features_dir.mkdir()
        self.features_packed_dir.mkdir()

    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)

    def test_is_feature_installed_with_marker(self):
        """Test is_feature_installed mit .installed Marker"""
        from toolboxv2 import feature_loader
        
        # Patch get_features_dir
        with patch.object(feature_loader, 'get_features_dir', return_value=self.features_dir):
            # Ohne Marker
            feature_dir = self.features_dir / "test"
            feature_dir.mkdir()
            (feature_dir / "feature.yaml").write_text("name: test")
            
            self.assertFalse(feature_loader.is_feature_installed("test"))
            
            # Mit Marker
            (feature_dir / ".installed").touch()
            self.assertTrue(feature_loader.is_feature_installed("test"))

    def test_is_feature_packed(self):
        """Test is_feature_packed"""
        from toolboxv2 import feature_loader
        
        with patch.object(feature_loader, 'get_features_packed_dir', return_value=self.features_packed_dir):
            # Ohne ZIP
            self.assertFalse(feature_loader.is_feature_packed("web"))
            
            # Mit ZIP
            (self.features_packed_dir / "tbv2-feature-web-0.1.25.zip").touch()
            self.assertTrue(feature_loader.is_feature_packed("web"))

    def test_unpack_feature(self):
        """Test Feature Entpacken"""
        from toolboxv2 import feature_loader
        
        # Erstelle Test-ZIP
        zip_path = self.features_packed_dir / "tbv2-feature-test-1.0.0.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("feature.yaml", yaml.dump({
                "name": "test",
                "version": "1.0.0",
                "files": [],
            }))
            zf.writestr("requirements.txt", "requests>=2.0")
        
        # Patches
        with patch.object(feature_loader, 'get_features_dir', return_value=self.features_dir), \
             patch.object(feature_loader, 'get_features_packed_dir', return_value=self.features_packed_dir), \
             patch.object(feature_loader, 'get_package_root', return_value=Path(self.temp_dir)):
            
            success = feature_loader.unpack_feature("test")
            
            self.assertTrue(success)
            self.assertTrue((self.features_dir / "test" / "feature.yaml").exists())
            self.assertTrue((self.features_dir / "test" / "requirements.txt").exists())
            self.assertTrue((self.features_dir / "test" / ".installed").exists())

    def test_unpack_feature_with_files(self):
        """Test Feature Entpacken mit Dateien"""
        from toolboxv2 import feature_loader
        
        # Erstelle Test-ZIP mit Dateien
        zip_path = self.features_packed_dir / "tbv2-feature-test-1.0.0.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("feature.yaml", yaml.dump({
                "name": "test",
                "version": "1.0.0",
                "files": ["mods/TestMod/*"],
            }))
            zf.writestr("files/mods/TestMod/__init__.py", "# Test Module")
            zf.writestr("files/mods/TestMod/main.py", "print('hello')")
        
        # Patches
        package_root = Path(self.temp_dir)
        with patch.object(feature_loader, 'get_features_dir', return_value=self.features_dir), \
             patch.object(feature_loader, 'get_features_packed_dir', return_value=self.features_packed_dir), \
             patch.object(feature_loader, 'get_package_root', return_value=package_root):
            
            success = feature_loader.unpack_feature("test")
            
            self.assertTrue(success)
            # Prüfe dass Dateien entpackt wurden
            self.assertTrue((package_root / "mods" / "TestMod" / "__init__.py").exists())
            self.assertTrue((package_root / "mods" / "TestMod" / "main.py").exists())


class TestListAvailableFeatures(unittest.TestCase):
    """Tests für list_available_features"""

    def setUp(self):
        """Setup"""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_packed_dir = Path(self.temp_dir) / "features_packed"
        self.features_dir.mkdir()
        self.features_packed_dir.mkdir()

    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)

    def test_list_mixed_sources(self):
        """Test Liste mit installierten und gepackten Features"""
        from toolboxv2 import feature_loader
        
        # Installiertes Feature
        core_dir = self.features_dir / "core"
        core_dir.mkdir()
        (core_dir / "feature.yaml").write_text("name: core")
        
        # Gepacktes Feature
        (self.features_packed_dir / "tbv2-feature-web-0.1.25.zip").touch()
        
        with patch.object(feature_loader, 'get_features_dir', return_value=self.features_dir), \
             patch.object(feature_loader, 'get_features_packed_dir', return_value=self.features_packed_dir):
            
            available = feature_loader.list_available_features()
            
            self.assertIn("core", available)
            self.assertIn("web", available)


class TestEnsureFeaturesLoaded(unittest.TestCase):
    """Tests für ensure_features_loaded"""

    def setUp(self):
        """Setup"""
        self.temp_dir = tempfile.mkdtemp()
        self.features_dir = Path(self.temp_dir) / "features"
        self.features_packed_dir = Path(self.temp_dir) / "features_packed"
        self.features_dir.mkdir()
        self.features_packed_dir.mkdir()

    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)

    def test_ensure_loads_required_features(self):
        """Test dass benötigte Features geladen werden"""
        from toolboxv2 import feature_loader
        
        # Core als Source
        core_dir = self.features_dir / "core"
        core_dir.mkdir()
        (core_dir / "feature.yaml").write_text("name: core")
        
        # Web als ZIP
        zip_path = self.features_packed_dir / "tbv2-feature-web-0.1.25.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("feature.yaml", yaml.dump({"name": "web", "version": "0.1.25"}))
        
        with patch.object(feature_loader, 'get_features_dir', return_value=self.features_dir), \
             patch.object(feature_loader, 'get_features_packed_dir', return_value=self.features_packed_dir), \
             patch.object(feature_loader, 'get_package_root', return_value=Path(self.temp_dir)), \
             patch.object(feature_loader, 'get_required_features', return_value={'core', 'web'}):
            
            results = feature_loader.ensure_features_loaded()
            
            self.assertTrue(results.get('core'))
            self.assertTrue(results.get('web'))
            # Prüfe .installed Marker
            self.assertTrue((self.features_dir / "core" / ".installed").exists())
            self.assertTrue((self.features_dir / "web" / ".installed").exists())


class TestGetFeatureStatus(unittest.TestCase):
    """Tests für get_feature_status"""

    def test_status_shows_all_info(self):
        """Test dass Status alle Infos enthält"""
        from toolboxv2 import feature_loader
        
        with patch.object(feature_loader, 'is_feature_installed', return_value=True), \
             patch.object(feature_loader, 'is_feature_packed', return_value=False), \
             patch.object(feature_loader, 'is_feature_available', return_value=True), \
             patch.object(feature_loader, 'get_required_features', return_value={'core'}):
            
            status = feature_loader.get_feature_status()
            
            self.assertIn('core', status)
            self.assertIn('installed', status['core'])
            self.assertIn('packed', status['core'])
            self.assertIn('required', status['core'])
            self.assertIn('available', status['core'])


if __name__ == "__main__":
    unittest.main()
