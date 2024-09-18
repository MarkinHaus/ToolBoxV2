import importlib.util
import os
import time

from ..utils.extras.gist_control import GistLoader


def runnable_dict(s='.py', remote=False, dir_path=None, runnable_dict_=None):

    if runnable_dict_ is None:
        runnable_dict_ = {}

    # Erhalte den Pfad zum aktuellen Verzeichnis
    if dir_path is None:
        for ex_path in os.getenv("EXTERNAL_PATH_RUNNABELS", '').split(','):
            if not ex_path or len(ex_path) == 0:
                continue
            runnable_dict(s,remote,ex_path,runnable_dict_)
        dir_path = os.path.dirname(os.path.realpath(__file__))
    to = time.perf_counter()
    # Iteriere über alle Dateien im Verzeichnis
    for file_name in os.listdir(dir_path):
        # Überprüfe, ob die Datei eine Python-Datei ist
        if file_name == "__init__.py":
            pass
        elif not remote and file_name.endswith('.py') and s in file_name:
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
        elif remote and s in file_name and file_name.endswith('.gist'):
            # print("Loading from Gist :", file_name)
            name_f = os.path.splitext(file_name)[0]
            name = name_f.split('.')[0]
            # publisher = name_f.split('.')[1]
            url = name_f.split('.')[-1]
            # print("Ent", name)
            # Lade das Modul
            print(f"Gist Name: {name}, URL: {url}")
            module = GistLoader(f"{name}/{url}").load_module(name)
            #try:
            #    module = GistLoader(f"{name}/{url}")
            #except Exception as e:
            #    print(f"Error loading module {name} from github {url}")
            #    print(e)
            #    continue

            # Füge das Modul der Dictionary hinzu
            print(f"{hasattr(module, 'run')} and {callable(module.run)} and {hasattr(module, 'NAME')}")
            if hasattr(module, 'run') and callable(module.run) and hasattr(module, 'NAME'):
                # print("Collecing :", module.NAME)
                runnable_dict_[module.NAME] = module.run

    print(f"Getting all runnable took {time.perf_counter() - to:.2f} for {len(runnable_dict_.keys())} elements")
    return runnable_dict_
