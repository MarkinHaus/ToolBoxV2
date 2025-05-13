import unittest

from toolboxv2 import Result
from toolboxv2.utils.system.types import (
    CallingObject,
    ToolBoxError,
    ToolBoxInterfaces,
    analyze_data,
)


class TestResult(unittest.TestCase):

    def test_default_result(self):
        # Test case for default result
        result = Result.default()
        self.assertEqual(result.error, ToolBoxError.none)
        self.assertEqual(result.result.data_to, ToolBoxInterfaces.native)

    def test_ok_result(self):
        # Test case for OK result
        data = {"key": "value"}
        result = Result.ok(data=data, data_info="Info")
        self.assertEqual(result.error, ToolBoxError.none)
        self.assertEqual(result.result.data_to, ToolBoxInterfaces.native)
        self.assertEqual(result.result.data, data)
        self.assertEqual(result.result.data_info, "Info")

    def test_future_result(self):
        # Test case for future result
        data = {"key": "value"}
        result = Result.future(data=data, data_info="Info")
        self.assertEqual(result.error, ToolBoxError.none)
        self.assertEqual(result.result.data_to, ToolBoxInterfaces.future)
        self.assertEqual(result.result.data, data)
        self.assertEqual(result.result.data_info, "Info")

    def test_custom_error_result(self):
        # Test case for custom error result
        data = {"key": "value"}
        result = Result.custom_error(data=data, data_info="Info", info="Custom Error", exec_code=-1)
        self.assertEqual(result.error, ToolBoxError.custom_error)
        self.assertEqual(result.result.data_to, ToolBoxInterfaces.native)
        self.assertEqual(result.result.data, data)
        self.assertEqual(result.result.data_info, "Info")
        self.assertEqual(result.info.exec_code, -1)
        self.assertEqual(result.info.help_text, "Custom Error")


class TestCallingObject(unittest.TestCase):

    def test_empty_calling_object(self):
        # Test case for creating an empty CallingObject
        calling_object = CallingObject.empty()
        self.assertEqual(calling_object.module_name, "")
        self.assertEqual(calling_object.function_name, "")
        self.assertIsNone(calling_object.args)
        self.assertIsNone(calling_object.kwargs)

    def test_str_representation_with_args_and_kwargs(self):
        # Test case for string representation with args and kwargs
        calling_object = CallingObject(module_name="Module", function_name="Function", args=["arg1", "arg2"],
                                       kwargs={"key1": "val1", "key2": "val2"})
        self.assertEqual(str(calling_object), "Module Function arg1 arg2 key1-val1 key2-val2")

    def test_str_representation_with_args_only(self):
        # Test case for string representation with args only
        calling_object = CallingObject(module_name="Module", function_name="Function", args=["arg1", "arg2"])
        self.assertEqual(str(calling_object), "Module Function arg1 arg2")

    def test_str_representation_without_args_and_kwargs(self):
        # Test case for string representation without args and kwargs
        calling_object = CallingObject(module_name="Module", function_name="Function")
        self.assertEqual(str(calling_object), "Module Function")

    # Add more test cases to cover other methods and scenarios


class TestAnalyzeData(unittest.TestCase):

    def test_analyze_data_with_valid_input(self):
        # Test case for analyzing valid data
        data = {
            "module1": {"functions_run": 10, "functions_fatal_error": 2, "error": 1, "functions_sug": 7,
                        "coverage": [20, 30], "callse": {"func1": ["Error1", "Error2"], "func2": ["Error3"]}},
            "module2": {"functions_run": 5, "functions_fatal_error": 1, "error": 0, "functions_sug": 4,
                        "coverage": [10, 15], "callse": {"func3": ["Error4"]}}
        }
        expected_output = ("Modul: module1\n"
                           "  Funktionen ausgeführt: 10\n"
                           "  Funktionen mit Fatalen Fehler: 2\n"
                           "  Funktionen mit Fehler: 1\n"
                           "  Funktionen erfolgreich: 7\n"
                           "  coverage: 1.50\n"
                           "  Fehler:\n"
                           "    - func1, Fehler: Error1\n"
                           "    - func1, Fehler: Error2\n"
                           "    - func2, Fehler: Error3\n"
                           "Modul: module2\n"
                           "  Funktionen ausgeführt: 5\n"
                           "  Funktionen mit Fatalen Fehler: 1\n"
                           "  Funktionen mit Fehler: 0\n"
                           "  Funktionen erfolgreich: 4\n"
                           "  coverage: 1.50\n"
                           "  Fehler:\n"
                           "    - func3, Fehler: Error4")
        self.assertEqual(analyze_data(data), expected_output)

    # Add more test cases to cover other scenarios

