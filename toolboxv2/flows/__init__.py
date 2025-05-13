import importlib.util
import os
import time

from toolboxv2 import Spinner

from ..utils.extras.gist_control import GistLoader


def flows_dict(s='.py', remote=False, dir_path=None, flows_dict_=None):

    if flows_dict_ is None:
        flows_dict_ = {}
    with Spinner("Loading flows"):
        # Erhalte den Pfad zum aktuellen Verzeichnis
        if dir_path is None:
            for ex_path in os.getenv("EXTERNAL_PATH_RUNNABELS", '').split(','):
                if not ex_path or len(ex_path) == 0:
                    continue
                flows_dict(s,remote,ex_path,flows_dict_)
            dir_path = os.path.dirname(os.path.realpath(__file__))
        to = time.perf_counter()
        # Iteriere über alle Dateien im Verzeichnis
        files = os.listdir(dir_path)
        l_files = len(files)
        for i, file_name in enumerate(files):
            with Spinner(f"{file_name} {i}/{l_files}"):
                # Überprüfe, ob die Datei eine Python-Datei ist
                if file_name == "__init__.py":
                    pass

                elif remote and s in file_name and file_name.endswith('.gist'):
                    # print("Loading from Gist :", file_name)
                    name_f = os.path.splitext(file_name)[0]
                    name = name_f.split('.')[0]
                    # publisher = name_f.split('.')[1]
                    url = name_f.split('.')[-1]
                    # print("Ent", name)
                    # Lade das Modul
                    print(f"Gist Name: {name}, URL: {url}")
                    try:
                        module = GistLoader(f"{name}/{url}").load_module(name)
                    #try:
                    #    module = GistLoader(f"{name}/{url}")
                    except Exception as e:
                        print(f"Error loading module {name} from github {url}")
                        print(e)
                        continue

                    # Füge das Modul der Dictionary hinzu
                    print(f"{hasattr(module, 'run')} and {callable(module.run)} and {hasattr(module, 'NAME')}")
                    if hasattr(module, 'run') and callable(module.run) and hasattr(module, 'NAME'):
                        # print("Collecing :", module.NAME)
                        flows_dict_[module.NAME] = module.run
                elif file_name.endswith('.py') and s in file_name:
                    name = os.path.splitext(file_name)[0]
                    # print("Loading :", name)
                    # Lade das Modul
                    spec = importlib.util.spec_from_file_location(name, os.path.join(dir_path, file_name))
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                    except Exception as e:
                        print("Error loading module ", name)
                        print(e)
                        continue

                    # Füge das Modul der Dictionary hinzu
                    if hasattr(module, 'run') and callable(module.run) and hasattr(module, 'NAME'):
                        # print("Collecing :", module.NAME)
                        flows_dict_[module.NAME] = module.run

        print(f"Getting all flows took {time.perf_counter() - to:.2f} for {len(flows_dict_.keys())} elements")
        return flows_dict_
