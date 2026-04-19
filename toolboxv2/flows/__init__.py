import importlib.util
import os
import time
import re

from toolboxv2 import Spinner

from ..utils.extras.gist_control import GistLoader

def normalize_flow_name(name: str) -> str:
    """myFlow, my-flow, my_flow, MyFlow → 'my_flow'"""
    # camelCase/PascalCase: insert _ before uppercase groups
    name = re.sub(r'([A-Z]+)', lambda m: '_' + m.group(0).lower(), name)
    name = name.replace('-', '_').lower()
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def flows_dict(s='.py', remote=False, dir_path=None, flows_dict_=None, ui=False):
    if s:
        s = normalize_flow_name(s)
    if flows_dict_ is None:
        flows_dict_ = {}
    with Spinner("Loading flows"):
        # Erhalte den Pfad zum aktuellen Verzeichnis
        if dir_path is None:
            for ex_path in os.getenv("EXTERNAL_PATH_RUNNABLE", '').split(','):
                if not ex_path or len(ex_path) == 0:
                    continue
                flows_dict(s,remote,ex_path,flows_dict_)
            dir_path = os.path.dirname(os.path.realpath(__file__))
        to = time.perf_counter()
        # Iteriere über alle Dateien im Verzeichnis
        files = os.listdir(dir_path)
        l_files = len(files)
        for i, file_name in enumerate(files):
            if not file_name:
                continue
            with Spinner(f"{file_name} {i}/{l_files}"):
                if file_name == "__init__.py":
                    pass

                elif remote and s in file_name and file_name.endswith('.gist'):
                    name_f = os.path.splitext(file_name)[0]
                    name = name_f.split('.')[0]
                    url = name_f.split('.')[-1]
                    try:
                        module = GistLoader(f"{name}/{url}").load_module(name)
                    except Exception as e:
                        continue

                    if not ui:
                        if (
                            hasattr(module, "run")
                            and callable(module.run)
                            and hasattr(module, "NAME")
                        ):
                            func = module.run
                    else:
                        if (
                            hasattr(module, "ui")
                            and callable(module.ui)
                            and hasattr(module, "NAME")
                        ):
                            func = module.ui
                    key = normalize_flow_name(module.NAME)
                    flows_dict_[key] = func
                    fname_key = normalize_flow_name(name)
                    if fname_key != key:
                        flows_dict_[fname_key] = func
                elif file_name.endswith('.py') and s in normalize_flow_name(file_name):
                    name = os.path.splitext(file_name)[0]
                    spec = importlib.util.spec_from_file_location(name, os.path.join(dir_path, file_name))
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                    except Exception:
                        continue

                    # Füge das Modul der Dictionary hinzu
                    if not ui:
                        if (
                            hasattr(module, "run")
                            and callable(module.run)
                            and hasattr(module, "NAME")
                        ):
                            flows_dict_[module.NAME] = module.run
                            if normalize_flow_name(module.NAME) != module.NAME:
                                flows_dict_[normalize_flow_name(module.NAME)] = module.run
                    else:
                        if (
                            hasattr(module, "ui")
                            and callable(module.ui)
                            and hasattr(module, "NAME")
                        ):
                            flows_dict_[module.NAME] = { 'ui':module.ui, 'icon': getattr(module, "ICON", "apps"), 'auth': getattr(module, "AUTH", False), 'bg_img_url': getattr(module, "BG_IMG_URL", None) }
                            if normalize_flow_name(module.NAME) != module.NAME:
                                flows_dict_[normalize_flow_name(module.NAME)] = { 'ui':module.ui, 'icon': getattr(module, "ICON", "apps"), 'auth': getattr(module, "AUTH", False), 'bg_img_url': getattr(module, "BG_IMG_URL", None) }

        return flows_dict_
