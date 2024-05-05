import importlib.util
import os
import time


# Erstelle ein leeres Wörterbuch

def runnable_dict(s='.py'):
    runnable_dict_ = {}

    # Erhalte den Pfad zum aktuellen Verzeichnis
    dir_path = os.path.dirname(os.path.realpath(__file__))
    to = time.perf_counter()
    # Iteriere über alle Dateien im Verzeichnis
    for file_name in os.listdir(dir_path):
        # Überprüfe, ob die Datei eine Python-Datei ist
        if file_name == "__init__.py":
            pass
        elif file_name.endswith('.py') and s in file_name:
            # Entferne die Erweiterung ".py" aus dem Dateinamen
            name = os.path.splitext(file_name)[0]
            # print("Ent", name)
            # Lade das Modul
            spec = importlib.util.spec_from_file_location(name, os.path.join(dir_path, file_name))
            module = importlib.util.module_from_spec(spec)
            # try:
            spec.loader.exec_module(module)
            # except Exception as e:
            #    print("Error loading module ")
            #    print(e)

            # Füge das Modul der Dictionary hinzu
            if hasattr(module, 'run') and callable(module.run) and hasattr(module, 'NAME'):
                # print("Collecing :", module.NAME)
                runnable_dict_[module.NAME] = module.run
    print(f"Getting all runnable took {time.perf_counter() - to:.2f} for {len(runnable_dict_.keys())} elements")
    return runnable_dict_
