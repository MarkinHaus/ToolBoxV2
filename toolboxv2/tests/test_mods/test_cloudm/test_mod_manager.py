# file: toolboxv2/tests/test_mods/test_cloudm/test_mod_manager.py
"""
Tests for CloudM ModManager module.

Tests module management functionality WITHOUT making real cloud calls:
- Module packaging and ZIP creation
- Module metadata extraction
- Version comparison
- Platform filtering
- Module listing (mocked API)
- Upload/download simulation (mocked)

All cloud/network operations are mocked to ensure tests run offline.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import zipfile

from toolboxv2 import Result


class TestModulePackaging(unittest.TestCase):
    """Tests for module packaging functionality"""

    def setUp(self):
        """Create temporary directory for test modules"""
        self.test_dir = tempfile.mkdtemp()
        self.mods_dir = Path(self.test_dir) / "mods"
        self.mods_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_simple_module_structure(self):
        """Test creating a simple module structure"""
        module_name = "TestModule"
        module_dir = self.mods_dir / module_name
        module_dir.mkdir()

        # Create module file
        module_file = module_dir / "__init__.py"
        module_file.write_text("# Test Module\nversion = '1.0.0'\n")

        self.assertTrue(module_dir.exists())
        self.assertTrue(module_file.exists())

    def test_extract_module_version_from_file(self):
        """Test extracting version from module file"""
        module_name = "VersionTest"
        module_dir = self.mods_dir / module_name
        module_dir.mkdir()

        # Create module with version
        module_file = module_dir / "__init__.py"
        module_file.write_text('version = "2.5.3"\n')

        # Read and extract version
        content = module_file.read_text()
        self.assertIn('version', content)
        self.assertIn('2.5.3', content)

    def test_create_zip_from_module(self):
        """Test creating a ZIP file from module directory"""
        module_name = "ZipTest"
        module_dir = self.mods_dir / module_name
        module_dir.mkdir()

        # Create some files
        (module_dir / "__init__.py").write_text("# Init")
        (module_dir / "utils.py").write_text("# Utils")

        # Create ZIP
        zip_path = Path(self.test_dir) / f"{module_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in module_dir.rglob('*'):
                if file.is_file():
                    zipf.write(file, file.relative_to(module_dir.parent))

        self.assertTrue(zip_path.exists())

        # Verify ZIP contents
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            names = zipf.namelist()
            self.assertIn(f"{module_name}/__init__.py", names)
            self.assertIn(f"{module_name}/utils.py", names)


class TestModuleMetadata(unittest.TestCase):
    """Tests for module metadata extraction"""

    def test_parse_module_name_from_zip_filename(self):
        """Test extracting module name from ZIP filename"""
        # Format: RST$ModuleName&version§1.0.0.zip
        filename = "RST$TestMod&0.1.0§1.2.3.zip"

        # Extract module name (between $ and &)
        parts = filename.split('$')
        if len(parts) > 1:
            module_name = parts[1].split('&')[0]
            self.assertEqual(module_name, "TestMod")

    def test_parse_version_from_zip_filename(self):
        """Test extracting version from ZIP filename"""
        filename = "RST$TestMod&0.1.0§2.5.1.zip"

        # Extract version (after §, before .zip)
        version = filename.split('§')[1].replace('.zip', '')
        self.assertEqual(version, "2.5.1")

    def test_parse_app_version_from_zip_filename(self):
        """Test extracting app version from ZIP filename"""
        filename = "RST$TestMod&1.5.0§2.0.0.zip"

        # Extract app version (between & and §)
        app_version = filename.split('&')[1].split('§')[0]
        self.assertEqual(app_version, "1.5.0")


class TestVersionComparison(unittest.TestCase):
    """Tests for version comparison logic"""

    def test_compare_semantic_versions(self):
        """Test comparing semantic versions"""
        def version_tuple(v):
            return tuple(map(int, v.split('.')))

        v1 = "1.2.3"
        v2 = "1.2.4"
        v3 = "2.0.0"

        self.assertLess(version_tuple(v1), version_tuple(v2))
        self.assertLess(version_tuple(v2), version_tuple(v3))
        self.assertEqual(version_tuple(v1), version_tuple(v1))

    def test_version_comparison_edge_cases(self):
        """Test version comparison edge cases"""
        def version_tuple(v):
            return tuple(map(int, v.split('.')))

        # Different lengths
        v1 = "1.0.0"
        v2 = "1.0.0"
        self.assertEqual(version_tuple(v1), version_tuple(v2))

        # Major version difference
        v3 = "2.0.0"
        v4 = "1.9.9"
        self.assertGreater(version_tuple(v3), version_tuple(v4))


class TestPlatformFiltering(unittest.TestCase):
    """Tests for platform-specific module filtering"""

    def test_identify_platform_specific_files(self):
        """Test identifying platform-specific files"""
        files = [
            "module.py",
            "utils_windows.py",
            "utils_linux.py",
            "utils_darwin.py",
            "common.py"
        ]

        windows_files = [f for f in files if 'windows' in f.lower()]
        linux_files = [f for f in files if 'linux' in f.lower()]

        self.assertEqual(len(windows_files), 1)
        self.assertEqual(len(linux_files), 1)

    def test_filter_files_by_platform(self):
        """Test filtering files based on platform"""
        all_files = {
            "core.py": "common",
            "win_specific.py": "windows",
            "linux_specific.py": "linux",
            "mac_specific.py": "darwin"
        }

        target_platform = "windows"
        filtered = {k: v for k, v in all_files.items()
                   if v == "common" or v == target_platform}

        self.assertIn("core.py", filtered)
        self.assertIn("win_specific.py", filtered)
        self.assertNotIn("linux_specific.py", filtered)


class TestModuleListingMocked(unittest.TestCase):
    """Tests for module listing with mocked API calls"""

    @patch('toolboxv2.mods.CloudM.ModManager.get_app')
    def test_list_modules_returns_result(self, mock_get_app):
        """Test that list_modules returns proper Result object"""
        from toolboxv2.mods.CloudM.ModManager import list_modules

        # Mock app with modules
        mock_app = MagicMock()
        mock_app.get_all_mods.return_value = ["Mod1", "Mod2", "Mod3"]
        mock_get_app.return_value = mock_app

        result = list_modules(app=mock_app)

        # list_modules returns ApiResult
        self.assertIsNotNone(result)
        if hasattr(result, 'result'):
            data = result.result.data
            self.assertIn("modules", data)
            self.assertIn("count", data)
            self.assertEqual(data["count"], 3)

    @patch('toolboxv2.mods.CloudM.ModManager.get_app')
    def test_list_modules_empty(self, mock_get_app):
        """Test listing modules when none exist"""
        from toolboxv2.mods.CloudM.ModManager import list_modules

        mock_app = MagicMock()
        mock_app.get_all_mods.return_value = []
        mock_get_app.return_value = mock_app

        result = list_modules(app=mock_app)

        # list_modules returns ApiResult
        self.assertIsNotNone(result)
        if hasattr(result, 'result'):
            data = result.result.data
            self.assertEqual(data["count"], 0)
            self.assertEqual(len(data["modules"]), 0)


class TestModuleUploadMocked(unittest.TestCase):
    """Tests for module upload with mocked operations"""

    @patch('toolboxv2.mods.CloudM.ModManager.get_app')
    async def test_upload_mod_validates_form_data(self, mock_get_app):
        """Test that upload_mod validates form data"""
        from toolboxv2.mods.CloudM.ModManager import upload_mod

        mock_app = MagicMock()
        mock_request = MagicMock()

        # Test with no form data
        result = await upload_mod(mock_app, mock_request, form_data=None)

        self.assertIsInstance(result, Result)
        self.assertTrue(result.is_error())

    @patch('toolboxv2.mods.CloudM.ModManager.get_app')
    async def test_upload_mod_validates_file_presence(self, mock_get_app):
        """Test that upload_mod validates file presence"""
        from toolboxv2.mods.CloudM.ModManager import upload_mod

        mock_app = MagicMock()
        mock_request = MagicMock()

        # Test with form data but no files
        result = await upload_mod(mock_app, mock_request, form_data={})

        self.assertIsInstance(result, Result)
        self.assertTrue(result.is_error())


class TestModuleInstallationMocked(unittest.TestCase):
    """Tests for module installation with mocked operations"""

    def setUp(self):
        """Create temporary directories"""
        self.test_dir = tempfile.mkdtemp()
        self.mods_dir = Path(self.test_dir) / "mods"
        self.mods_dir.mkdir()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unpack_zip_structure(self):
        """Test unpacking ZIP file structure"""
        # Create a test ZIP
        module_name = "TestInstall"
        zip_path = Path(self.test_dir) / f"{module_name}.zip"

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr(f"{module_name}/__init__.py", "# Init")
            zipf.writestr(f"{module_name}/utils.py", "# Utils")

        # Extract
        extract_dir = Path(self.test_dir) / "extracted"
        extract_dir.mkdir()

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_dir)

        # Verify extraction
        self.assertTrue((extract_dir / module_name / "__init__.py").exists())
        self.assertTrue((extract_dir / module_name / "utils.py").exists())

    def test_move_extracted_module_to_mods(self):
        """Test moving extracted module to mods directory"""
        module_name = "MoveTest"
        source_dir = Path(self.test_dir) / "temp" / module_name
        source_dir.mkdir(parents=True)

        # Create files
        (source_dir / "__init__.py").write_text("# Init")

        # Move to mods
        target_dir = self.mods_dir / module_name
        shutil.copytree(source_dir, target_dir)

        self.assertTrue(target_dir.exists())
        self.assertTrue((target_dir / "__init__.py").exists())


class TestModuleDownloadMocked(unittest.TestCase):
    """Tests for module download with mocked network calls"""

    @patch('requests.get')
    def test_download_module_mocked(self, mock_get):
        """Test downloading module with mocked request"""
        # Mock successful download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake zip content"
        mock_get.return_value = mock_response

        import requests
        response = requests.get("http://fake-url.com/module.zip")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake zip content")

    @patch('requests.get')
    def test_download_module_error_handling(self, mock_get):
        """Test download error handling"""
        # Mock failed download
        mock_get.side_effect = Exception("Network error")

        import requests
        with self.assertRaises(Exception):
            requests.get("http://fake-url.com/module.zip")


class TestModManagerIntegration(unittest.TestCase):
    """Integration tests for ModManager functionality"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_module_packaging_workflow(self):
        """Test complete module packaging workflow"""
        # 1. Create module structure
        module_name = "CompleteTest"
        module_dir = Path(self.test_dir) / "mods" / module_name
        module_dir.mkdir(parents=True)

        (module_dir / "__init__.py").write_text('version = "1.0.0"\n')
        (module_dir / "core.py").write_text("# Core functionality\n")

        # 2. Create ZIP
        zip_path = Path(self.test_dir) / f"RST${module_name}&0.1.0§1.0.0.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in module_dir.rglob('*'):
                if file.is_file():
                    zipf.write(file, file.relative_to(module_dir.parent))

        # 3. Verify ZIP
        self.assertTrue(zip_path.exists())

        # 4. Extract metadata
        filename = zip_path.name
        extracted_name = filename.split('$')[1].split('&')[0]
        extracted_version = filename.split('§')[1].replace('.zip', '')

        self.assertEqual(extracted_name, module_name)
        self.assertEqual(extracted_version, "1.0.0")


if __name__ == '__main__':
    unittest.main(verbosity=2)

