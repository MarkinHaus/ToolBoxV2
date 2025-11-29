import unittest

from toolboxv2 import get_app
from toolboxv2.utils.system import get_state_from_app
from toolboxv2.utils.system.state_system import find_highest_zip_version_entry


class TestFindHighestZipVersionEntry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        get_state_from_app(get_app(name="test-debug"))

    def test_find_highest_zip_version_entry_without_target_app_version(self):
        # Test case without target app version
        get_app(name="test-debug")
        result = find_highest_zip_version_entry(name="example")
        # Add assertions to check the result
        self.assertEqual(result, {})
        # Add more specific assertions based on the expected behavior

    def test_find_highest_zip_version_entry_with_invalid_filepath(self):
        # Test case with an invalid filepath
        get_app(name="test-debug")
        with self.assertRaises(FileNotFoundError):
            find_highest_zip_version_entry(name="example", filepath="invalid.yaml")
