import os
import unittest
from unittest.mock import MagicMock, patch

from toolboxv2.utils.system.state_system import (
    DefaultFilesFormatElement,
    TbState,
    calculate_shasum,
    get_state_from_app,
    process_files,
)


class TestTbState(unittest.TestCase):

    def test_default_files_format_element_str(self):
        # Test the __str__ method of DefaultFilesFormatElement
        element = DefaultFilesFormatElement(version="1.0", shasum="12345", provider="git", url="https://example.com")
        expected_str = "version='1.0'shasum='12345'provider='git'url='https://example.com'|"
        self.assertEqual(str(element), expected_str)

    def test_calculate_shasum(self):
        # Test the calculate_shasum function
        # Mocking file read operation
        with patch('builtins.open', MagicMock()), \
             patch('toolboxv2.utils.system.state_system.hashlib.sha256', MagicMock()) as mock_sha256:
            mock_hexdigest = MagicMock(return_value="abc123")
            mock_sha256.return_value = MagicMock(hexdigest=mock_hexdigest)

            shasum = calculate_shasum("test_file.txt")

            self.assertEqual(shasum, "abc123")
            mock_sha256.assert_called_once()
            mock_hexdigest.assert_called_once()

    def test_process_files(self):
        # Test the process_files function
        # Mocking os.walk to simulate directory traversal
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [("", [], [""])]

            state = process_files("")
            self.assertIsInstance(state, TbState)
            self.assertEqual(len(state.utils), 0)
            self.assertEqual(len(state.mods), 0)
            self.assertEqual(len(state.installable), 0)
            self.assertEqual(len(state.runnable), 0)
            self.assertEqual(len(state.api), 0)
            self.assertEqual(len(state.app), 0)

    @patch('toolboxv2.utils.system.state_system.yaml.dump')
    def test_get_state_from_app(self, mock_yaml_dump):
        # Test the get_state_from_app function
        mock_app = MagicMock()
        mock_app.start_dir = "/root"
        mock_app.version = "1.0"
        # mock_app.get_all_mods.return_value = {"mod1": MagicMock(version="1.0"), "mod2": MagicMock(version="2.0")}

        state = get_state_from_app(mock_app)

        self.assertIsInstance(state, TbState)
        self.assertEqual(len(state.utils), 0)
        self.assertEqual(len(state.mods), 0)  # Depending on the mock_app configuration
        self.assertEqual(len(state.installable), 0)
        self.assertEqual(len(state.runnable), 0)
        self.assertEqual(len(state.api), 0)
        self.assertEqual(len(state.app), 0)
        mock_yaml_dump.assert_called_once()
        os.remove("tbState.yaml")
