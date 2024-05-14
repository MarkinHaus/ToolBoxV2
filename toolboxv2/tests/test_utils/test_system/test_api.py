import unittest

import packaging

from toolboxv2 import get_app
from toolboxv2.utils.system import get_state_from_app
from toolboxv2.utils.system.api import find_highest_zip_version_entry


class TestFindHighestZipVersionEntry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        get_state_from_app(get_app(name="test-debug"))

    def test_find_highest_zip_version_entry_without_target_app_version(self):
        # Test case without target app version
        get_app(name="test-debug")
        result = find_highest_zip_version_entry(name="example")
        # Add assertions to check the result
        self.assertIsNone(result)
        # Add more specific assertions based on the expected behavior

    def test_find_highest_zip_version_entry_with_target_app_version_matching(self):
        # Test case with target app version matching the entry's app version
        tv = get_app(name="test-debug").get_mod("DB").version

        result = find_highest_zip_version_entry(name="DB")
        # Add assertions to check the result
        print("RES :", result, tv)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("version", result.keys())
        self.assertEqual(result.get("version"), [ get_app(name="test-debug").version, tv])
        # Add more specific assertions based on the expected behavior

    def test_find_highest_zip_version_entry_with_target_app_version_not_matching(self):
        # Test case with target app version not matching the entry's app version
        get_app(name="test-debug")
        with self.assertRaises(packaging.version.InvalidVersion):
            result = find_highest_zip_version_entry(name="DB", target_app_version="-0.0.0")


    def test_find_highest_zip_version_entry_with_invalid_filepath(self):
        # Test case with an invalid filepath
        get_app(name="test-debug")
        with self.assertRaises(FileNotFoundError):
            find_highest_zip_version_entry(name="example", filepath="invalid.yaml")
