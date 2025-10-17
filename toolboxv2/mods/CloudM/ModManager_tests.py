
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from toolboxv2 import App, get_app

from .ModManager import (
    create_and_pack_module,
    increment_version,
    install_from_zip,
    unpack_and_move_module,
)

# The user wants the test function to be exported.
Name = 'CloudM'
export = get_app(f"{Name}.Test.Export").tb


@export(test_only=True)
def run_mod_manager_tests(app: App):
    """
    This function will be automatically discovered and run by the test runner.
    It uses the standard unittest framework to run tests.
    """
    print("Running ModManager Tests...")
    # We pass the app instance to the test class so it can be used if needed.
    TestModManager.app = app
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestModManager))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        # Raise an exception to signal failure to the toolboxv2 test runner
        raise AssertionError(f"ModManager tests failed: {result.errors} {result.failures}")
    print("ModManager tests passed successfully.")
    return True


class TestModManager(unittest.TestCase):
    app: App = None

    def test_increment_version(self):
        """Tests the version increment logic."""
        print("\nTesting increment_version...")
        self.assertEqual(increment_version("v0.0.1"), "v0.0.2")
        self.assertEqual(increment_version("v0.0.99", max_value=99), "v0.1.0")
        self.assertEqual(increment_version("v0.99.99", max_value=99), "v1.0.0")
        self.assertEqual(increment_version("v98"), "v99")
        with self.assertRaises(ValueError, msg="Should fail if 'v' is missing"):
            print(increment_version("0.0.1"))
        print("increment_version tests passed.")

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.original_cwd = os.getcwd()
        self.test_dir = tempfile.mkdtemp(prefix="mod_manager_test_")

        # The functions in ModManager use relative paths like './mods' and './mods_sto'
        # We'll create these inside our temp directory and chdir into it.
        os.chdir(self.test_dir)
        os.makedirs("mods", exist_ok=True)
        os.makedirs("mods_sto", exist_ok=True)
        os.makedirs("source_module", exist_ok=True)

    def tearDown(self):
        """Clean up the temporary environment after each test."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_pack_unpack_cycle(self):
        """Tests the full cycle of creating, packing, and unpacking a module."""
        print("\nTesting create_pack_unpack_cycle...")
        module_name = "MyTestMod"
        module_version = "v0.1.0"

        # 1. Create a dummy module structure inside the temp 'source_module' dir
        source_path = Path("source_module")
        module_source_path = source_path / module_name
        module_source_path.mkdir()
        (module_source_path / "main.py").write_text("print('hello from my test mod')")
        (module_source_path / "data.txt").write_text("some test data")

        # 2. Call create_and_pack_module
        # The 'path' argument is the parent directory of the module directory.
        zip_path_str = create_and_pack_module(
            path=str(source_path),
            module_name=module_name,
            version=module_version
        )
        self.assertTrue(zip_path_str, "create_and_pack_module should return a path.")
        zip_path = Path(zip_path_str)

        # 3. Assert the zip file was created in the correct location ('./mods_sto')
        self.assertTrue(zip_path.exists(), f"Zip file should exist at {zip_path}")
        self.assertEqual(zip_path.parent.name, "mods_sto")

        # 4. Call unpack_and_move_module
        # We unpack into the './mods' directory.
        unpacked_name = unpack_and_move_module(
            zip_path=str(zip_path),
            base_path="mods"
        )

        # 5. Assert the module was unpacked correctly
        self.assertEqual(unpacked_name, module_name)
        unpacked_dir = Path("mods") / module_name
        self.assertTrue(unpacked_dir.is_dir(), "Unpacked module directory should exist.")

        # Verify content
        self.assertTrue((unpacked_dir / "main.py").exists())
        self.assertEqual((unpacked_dir / "main.py").read_text(), "print('hello from my test mod')")
        self.assertTrue((unpacked_dir / "data.txt").exists())
        self.assertEqual((unpacked_dir / "data.txt").read_text(), "some test data")

        # Verify that the tbConfig.yaml was created and has correct info
        config_path = unpacked_dir / "tbConfig.yaml"
        self.assertTrue(config_path.exists())
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.assertEqual(config.get("module_name"), module_name)
        self.assertEqual(config.get("version"), module_version)

        print("create_pack_unpack_cycle tests passed.")

    def test_install_from_zip(self):
        """Tests the install_from_zip helper function."""
        print("\nTesting install_from_zip...")
        module_name = "MyInstallTestMod"
        module_version = "v0.1.1"

        # 1. Create a dummy module and zip it
        source_path = Path("source_module")
        module_source_path = source_path / module_name
        module_source_path.mkdir()
        (module_source_path / "main.py").write_text("pass")
        zip_path_str = create_and_pack_module(
            path=str(source_path),
            module_name=module_name,
            version=module_version
        )
        zip_path = Path(zip_path_str)
        zip_name = zip_path.name

        # 2. Mock the app object needed by install_from_zip
        mock_app = lambda :None
        mock_app.start_dir = self.test_dir

        # 3. Call install_from_zip
        result = install_from_zip(mock_app, zip_name, no_dep=True)

        # 4. Assert the installation was successful
        self.assertTrue(result)
        unpacked_dir = Path("mods") / module_name
        self.assertTrue(unpacked_dir.is_dir())
        self.assertTrue((unpacked_dir / "main.py").exists())
        print("install_from_zip tests passed.")


