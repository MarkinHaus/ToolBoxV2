from typing import Any, List
from faker import Faker

fake = Faker()


def generate_test_cases(sig) -> List[dict]:
    params = list(sig.parameters.values())

    if len(params) != 0:
        print(dir(params[0].annotation))

    test_cases = []
    # Edge Cases
    edge_kwargs = {param.name: generate_edge_value(param.annotation) for param in params if
                   param.name not in ['self', 'app', 'state']}
    test_cases.append(edge_kwargs)
    # Normal Cases
    normal_kwargs = {param.name: generate_normal_value(param.annotation) for param in params if
                     param.name not in ['self', 'app', 'state']}
    test_cases.append(normal_kwargs)
    # Default Values Cases
    default_kwargs = {param.name: param.default if param.default is not param.empty else None for param in params if
                      param.name not in ['self', 'app', 'state']}
    if any(default_kwargs.values()):  # Prüfen, ob es Standardwerte gibt
        test_cases.append(default_kwargs)

    # Custom Objects
    # custom_kwargs = {param.name: generate_custom_object(param.annotation) for param in params}
    # test_cases.append(custom_kwargs)

    return test_cases


def generate_edge_value(param_type: Any) -> Any:
    """
    Generiert Edge-Case-Werte basierend auf dem Parametertyp.
    """
    if param_type in [int, float]:
        return -999  # Beispiel für negative Zahlen
    elif param_type == str:
        return fake.word() * 100  # Lange zufällige Strings
    # Fügen Sie hier weitere Bedingungen für andere Datentypen hinzu
    return None


def generate_normal_value(param_type: Any) -> Any:
    """
    Generiert normale Werte basierend auf dem Parametertyp.
    """
    if param_type in [int, float]:
        return fake.random_int(min=0, max=100)  # Zufällige normale Zahlen
    elif param_type == str:
        return fake.word()  # Zufälliges Wort
    # Fügen Sie hier weitere Bedingungen für andere Datentypen hinzu
    return None


def generate_custom_object(param_type: Any) -> Any:
    # Hier können Sie benutzerdefinierte Logik basierend auf dem Parametertyp implementieren
    # Beispiel: Wenn der Parametertyp eine bestimmte Klasse ist, erzeugen Sie eine Instanz dieser Klasse
    return None
