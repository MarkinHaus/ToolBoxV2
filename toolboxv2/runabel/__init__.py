import os
import importlib.util

# Erstelle ein leeres Wörterbuch

def runnable_dict():
    runnable_dict_ = {}

    # Erhalte den Pfad zum aktuellen Verzeichnis
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Iteriere über alle Dateien im Verzeichnis
    for file_name in os.listdir(dir_path):
        # Überprüfe, ob die Datei eine Python-Datei ist
        if file_name == "__init__.py":
            pass
        elif file_name.endswith('.py'):
            # Entferne die Erweiterung ".py" aus dem Dateinamen
            name = os.path.splitext(file_name)[0]
            # Lade das Modul
            spec = importlib.util.spec_from_file_location(name, os.path.join(dir_path, file_name))
            module = importlib.util.module_from_spec(spec)
            #try:
            spec.loader.exec_module(module)
            #except Exception as e:
            #    print("Error loading module ")
            #    print(e)

            # Füge das Modul der Dictionary hinzu
            if hasattr(module, 'run') and callable(module.run) and hasattr(module, 'NAME'):
                runnable_dict_[module.NAME.lower()] = module.run
    return runnable_dict_
