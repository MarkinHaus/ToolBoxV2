

import yaml
from packaging import version


def find_highest_zip_version_entry(name, target_app_version=None, filepath='tbState.yaml'):
    """
    Findet den Eintrag mit der höchsten ZIP-Version für einen gegebenen Namen und eine optionale Ziel-App-Version in einer YAML-Datei.

    :param name: Der Name des gesuchten Eintrags.
    :param target_app_version: Die Zielversion der App als String (optional).
    :param filepath: Der Pfad zur YAML-Datei.
    :return: Den Eintrag mit der höchsten ZIP-Version innerhalb der Ziel-App-Version oder None, falls nicht gefunden.
    """
    highest_zip_ver = None
    highest_entry = None

    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

        for key, value in data.get('installable', {}).items():
            # Prüfe, ob der Name im Schlüssel enthalten ist
            if name in key:
                app_ver, zip_ver = value['version']
                # Wenn eine Ziel-App-Version angegeben ist, vergleiche sie
                if target_app_version is None or version.parse(app_ver) == version.parse(target_app_version):
                    current_zip_ver = version.parse(zip_ver)
                    if highest_zip_ver is None or current_zip_ver > highest_zip_ver:
                        highest_zip_ver = current_zip_ver
                        highest_entry = value

    return highest_entry
