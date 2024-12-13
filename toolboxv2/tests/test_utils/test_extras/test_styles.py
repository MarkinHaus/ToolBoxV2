import unittest
import re
import json
import sys
import io
from contextlib import contextmanager, redirect_stdout

# Import the functions and classes to be tested
from toolboxv2.utils.extras.Style import (
    Style,
    remove_styles,
    print_to_console,
    extract_json_strings,
    extract_python_code,
    JSONExtractor,
    Spinner
)


class TestStyleUtilities(unittest.TestCase):
    def test_style_color_methods(self):
        """Test various color methods of the Style class."""
        # Test that each color method returns a string with color codes
        color_methods = [
            'BLUE', 'BLACK', 'RED', 'GREEN', 'YELLOW', 'MAGENTA',
            'CYAN', 'WHITE', 'Bold', 'Underline', 'Reversed',
            'ITALIC', 'BLINK', 'BLINK2'
        ]

        for method_name in color_methods:
            method = getattr(Style, method_name)
            colored_text = method("test")

            # Check that the color method adds color codes
            self.assertTrue('\33[' in colored_text,
                            f"{method_name} method did not add color codes")

            # Verify the text is wrapped correctly
            self.assertTrue(colored_text.startswith('\33['),
                            f"{method_name} method incorrect color code placement")
            self.assertTrue(colored_text.endswith('\33[0m'),
                            f"{method_name} method should end with reset code")

    def test_remove_styles(self):
        """Test removing style codes from text."""
        styled_text = Style.RED("Hello") + Style.BLUE(" World")

        # Test basic removal
        plain_text = remove_styles(styled_text)
        self.assertEqual(plain_text, "Hello World")

        # Test with info extraction
        plain_text_with_info, styles = remove_styles(styled_text, infos=True)
        self.assertEqual(plain_text_with_info, "Hello World")
        self.assertIn("RED", styles)
        self.assertIn("BLUE", styles)

    def test_print_to_console(self):
        """Test print_to_console function."""

        # Capture stdout
        @contextmanager
        def capture_stdout():
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            try:
                yield captured_output
            finally:
                sys.stdout = old_stdout

        # Test with a simple string
        with capture_stdout() as output:
            print_to_console("Test Title", "\33[32m", "Hello World")

        # Check that something was printed
        printed_text = output.getvalue()
        self.assertIn("Hello", printed_text)
        self.assertIn("World", printed_text)

        # Test with a list input
        with capture_stdout() as output:
            print_to_console("Test Title", "\33[32m", ["Hello", "World"])

        printed_text = output.getvalue()
        self.assertIn("Hello", printed_text)
        self.assertIn("World", printed_text)

    def test_json_extraction(self):
        """Test JSON string extraction."""
        test_text = 'Some text {"key": "value"} more text {"another": "json"}'

        extracted_jsons = extract_json_strings(test_text)

        self.assertEqual(len(extracted_jsons), 2)
        self.assertIn('{"key": "value"}', extracted_jsons)
        self.assertIn('{"another": "json"}', extracted_jsons)

    def test_python_code_extraction(self):
        """Test Python code block extraction."""
        test_text = """
Some text before
```python
def hello():
    print("World")
```
More text
```python
x = 10
y = 20
```
        """

        extracted_code = extract_python_code(test_text)

        self.assertEqual(len(extracted_code), 2)
        print(extracted_code)
        self.assertIn('def hello():\n    print("World")', extracted_code)
        self.assertIn('x = 10\ny = 20', extracted_code)

    def test_spinner(self):
        """Test Spinner functionality."""
        import io
        import sys

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Test basic spinner
        try:
            with Spinner(message="Testing", delay=0.01, time_in_s=0.1):
                # Do some work
                pass
        except Exception as e:
            self.fail(f"Spinner raised an unexpected exception: {e}")
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

        # Check that something was printed
        output = captured_output.getvalue()
        self.assertTrue(len(output) > 0)

    def test_style_color_demo(self):
        """Test color demo method."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            Style().color_demo()
        except Exception as e:
            self.fail(f"Color demo raised an unexpected exception: {e}")
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

        # Check that something was printed
        output = captured_output.getvalue()
        self.assertTrue(len(output) > 0)


if __name__ == '__main__':
    unittest.main()
