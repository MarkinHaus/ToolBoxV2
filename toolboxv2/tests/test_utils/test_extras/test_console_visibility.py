import unittest
import ctypes
import sys

# Import the function to be tested
from toolboxv2.utils.extras.show_and_hide_console import show_console, TBRUNNER_console_viabel


class TestShowConsole(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Ensure we're only running these tests on Windows
        if sys.platform != 'win32':
            self.skipTest("Console visibility tests are Windows-specific")

        # Store the initial console visibility state
        self.initial_state = TBRUNNER_console_viabel

    def tearDown(self):
        """Restore the initial console visibility state after each test."""
        global TBRUNNER_console_viabel
        TBRUNNER_console_viabel = self.initial_state

    def test_show_console_initially_hidden(self):
        """Test showing console when it's initially hidden."""
        # Attempt to show console
        result = show_console(show=True)

        # Verify the result
        self.assertFalse(result)
        self.assertTrue(TBRUNNER_console_viabel)

    def test_show_console_initially_visible(self):
        """Test showing console when it's already visible."""
        # Ensure console is visible
        global TBRUNNER_console_viabel
        TBRUNNER_console_viabel = True

        # Attempt to show console again
        result = show_console(show=True)

        # Verify the result
        self.assertFalse(result)
        self.assertTrue(TBRUNNER_console_viabel)

    def test_hide_console_initially_visible(self):
        """Test hiding console when it's initially visible."""
        # Ensure console is visible
        # Attempt to hide console
        result = show_console(show=False)

        self.assertFalse(result)
        result = show_console(show=True)

        # Verify the result
        self.assertTrue(result)
        self.assertTrue(TBRUNNER_console_viabel)

    def test_hide_console_initially_hidden(self):
        """Test hiding console when it's already hidden."""
        # Ensure console is hidden
        result = show_console(show=False)

        # Verify the result
        self.assertFalse(result)
        self.assertTrue(TBRUNNER_console_viabel)

    def test_console_visibility_state_tracking(self):
        """Test that the console visibility state is tracked correctly."""

        # Hide console
        show_console(show=False)
        self.assertTrue(TBRUNNER_console_viabel)

    def test_console_visibility_idempotency(self):
        """Test that repeated show/hide operations work as expected."""
        global TBRUNNER_console_viabel

        # Multiple show operations
        TBRUNNER_console_viabel = False
        show_console(show=True)
        first_result = show_console(show=True)

        # Verify
        self.assertFalse(first_result)
        self.assertFalse(TBRUNNER_console_viabel)

        # Multiple hide operations
        TBRUNNER_console_viabel = True
        show_console(show=False)
        first_result = show_console(show=False)

        # Verify
        self.assertFalse(first_result)

    def test_windows_specific_error_handling(self):
        """Test error handling when Windows-specific functions are not available."""
        # Mock Windows-specific functions to simulate error
        original_GetConsoleWindow = ctypes.windll.kernel32.GetConsoleWindow
        original_ShowWindow = ctypes.windll.user32.ShowWindow

        try:
            # Replace with functions that raise an exception
            def mock_get_console_window():
                raise Exception("Simulated Windows API error")

            def mock_show_window(hwnd, flag):
                raise Exception("Simulated Windows API error")

            ctypes.windll.kernel32.GetConsoleWindow = mock_get_console_window
            ctypes.windll.user32.ShowWindow = mock_show_window

            # Attempt to show/hide console
            result_show = show_console(show=True)
            result_hide = show_console(show=False)

            # Verify error handling
            self.assertFalse(result_show)
            self.assertFalse(result_hide)

        finally:
            # Restore original functions
            ctypes.windll.kernel32.GetConsoleWindow = original_GetConsoleWindow
            ctypes.windll.user32.ShowWindow = original_ShowWindow


if __name__ == '__main__':
    unittest.main()
