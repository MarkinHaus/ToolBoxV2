import os
import tempfile
import unittest
from dataclasses import dataclass
from enum import Enum, auto

from toolboxv2.utils.system.ipy_completer import (
    create_completions_from_classes,
    extract_class_info,
    get_dataclasses_from_file,
    nested_dict_autocomplete,
)


# Create a test dataclass and enum for testing
@dataclass
class SampleDataClass(Enum):
    name: str = auto()
    age: int = auto()
    active: bool = False

@dataclass
class SampleEnum(Enum):
    OPTION_A = auto()
    OPTION_B = auto()
    OPTION_C = auto()


class TestIPyCompleter(unittest.TestCase):

    def test_extract_class_info(self):
        # Test extract_class_info with an Enum
        class_info = extract_class_info(SampleEnum)

        self.assertIn('OPTION_A', class_info)
        self.assertIn('OPTION_B', class_info)
        self.assertIn('OPTION_C', class_info)

        self.assertEqual(class_info['OPTION_A']['annotations'], None)

    def test_create_completions_from_classes(self):
        # Test create_completions_from_classes
        classes = [SampleEnum, SampleDataClass]
        completions = create_completions_from_classes(classes)

        # Check for Enum completions
        self.assertIn('SampleEnum', completions)
        self.assertEqual(set(completions['SampleEnum']), {'OPTION_A', 'OPTION_B', 'OPTION_C'})

        # Check for DataClass completions
        self.assertIn('SampleDataClass', completions)
        self.assertEqual(set(completions['SampleDataClass']), {'name', 'age', 'active'})

    def test_nested_dict_autocomplete(self):
        # Test nested_dict_autocomplete with IPython Completer
        classes = [SampleEnum, SampleDataClass]
        completer = nested_dict_autocomplete(classes).complete

        # Test Enum completions
        def test_completion(text, expected_completions):
            completions = []
            state = 0
            while True:
                completion = completer(text, state)
                if completion is None:
                    break
                completions.append(completion.text)
                state += 1

            self.assertTrue(any(comp in expected_completions for comp in completions),
                            f"No matching completion found for {text}. Got: {completions}")

        # Test various completion scenarios
        test_completion('SampleEnum', ['OPTION_A', 'OPTION_B', 'OPTION_C'])
        test_completion('SampleEnum.OPTION', ['OPTION_A', 'OPTION_B', 'OPTION_C'])
        test_completion('SampleDataClass', ['name', 'age', 'active'])
        test_completion('SampleDataClass.n', ['name'])

    def test_get_dataclasses_from_file(self):
        # Test get_dataclasses_from_file
        # Create a temporary file with test dataclasses and enums
        temp_dir = tempfile.mkdtemp()
        test_file_path = os.path.join(temp_dir, 'test_classes.py')

        with open(test_file_path, 'w') as f:
            f.write('''
from dataclasses import dataclass
from enum import Enum, auto

@dataclass
class SampleDataClass(Enum):
    name: str = auto()
    age: int = auto()
    active: bool = False

@dataclass
class SampleEnum(Enum):
    OPTION_A = auto()
    OPTION_B = auto()
    OPTION_C = auto()
        ''')
        dataclasses = get_dataclasses_from_file(test_file_path)

        self.assertEqual(len(dataclasses), 2)
        self.assertEqual(dataclasses[0].__name__, 'SampleDataClass')



if __name__ == '__main__':
    unittest.main()
