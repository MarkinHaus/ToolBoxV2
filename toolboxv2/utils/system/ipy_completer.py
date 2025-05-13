import importlib.util
import inspect
from dataclasses import is_dataclass
from typing import Any

from IPython.core.completer import Completer, IPCompleter, SimpleCompletion


def extract_class_info(cls) -> dict[str, Any]:
    class_info = {}
    for name, member in cls.__members__.items():
        class_info[name] = {
            "value": member.value,
            "annotations": cls.__annotations__.get(name, None)
        }
    return class_info


def create_completions_from_classes(classes: list[type]) -> dict[str, list[str]]:
    completions = {}
    for cls in classes:
        class_name = cls.__name__
        class_info = extract_class_info(cls)
        completions[class_name] = [name for name in class_info]
    return completions


def nested_dict_autocomplete(classes: list[type]):
    completions = create_completions_from_classes(classes)

    # print(completions)
    class ClassAttributeCompleter(Completer):
        def __init__(self, completions_, **kwargs):
            super().__init__(**kwargs)
            self.completions = completions_

        def complete(self, text, state):

            if isinstance(text, IPCompleter):
                print(text, dir(text))
                text = text.matchers
            parts = text.split('.')
            options = self.completions.get(parts[0], [])
            if len(parts) > 1:
                options = [opt for opt in options if opt.startswith(parts[1])]
            try:
                return SimpleCompletion(options[state])
            except IndexError:
                return None

    return ClassAttributeCompleter(completions)


def get_dataclasses_from_file(file_path: str) -> list[type]:
    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract all classes from the module
    dataclasses = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if the class is defined in the current module
        if obj.__module__ == module.__name__:
            # Check if the class is a dataclass
            if is_dataclass(obj):
                dataclasses.append(obj)

    return dataclasses


def get_completer(st_dir="."):
    # Example usage
    classes = get_dataclasses_from_file(st_dir + "/utils/system/all_functions_enums.py")
    return nested_dict_autocomplete(classes).complete
