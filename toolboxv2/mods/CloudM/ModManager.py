import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from typing import Optional
from packaging import version as p_version

import yaml
from tqdm import tqdm

from toolboxv2 import get_app, App, __version__, Spinner
from toolboxv2.utils.system.api import find_highest_zip_version_entry
from toolboxv2.utils.system.types import ToolBoxInterfaces, Result

Name = 'CloudM'
export = get_app(f"{Name}.Export").tb
version = '0.0.1'
default_export = export(mod_name=Name, version=version, interface=ToolBoxInterfaces.native, test=False)


def download_files(urls, directory, desc, print_func, filename=None):
    """ Hilfsfunktion zum Herunterladen von Dateien. """
    for url in tqdm(urls, desc=desc):
        if filename is None:
            filename = os.path.basename(url)
        print_func(f"Download {filename}")
        print_func(f"{url} -> {directory}/{filename}")
        os.makedirs(directory, exist_ok=True)
        urllib.request.urlretrieve(url, f"{directory}/{filename}")
    return f"{directory}/{filename}"


def handle_requirements(requirements_url, module_name, print_func):
    """ Verarbeitet und installiert Requirements. """
    if requirements_url:
        requirements_filename = f"{module_name}-requirements.txt"
        print_func(f"Download requirements {requirements_filename}")
        urllib.request.urlretrieve(requirements_url, requirements_filename)

        print_func("Install requirements")
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", requirements_filename])

        os.remove(requirements_filename)


@export(mod_name=Name, api=True, interface=ToolBoxInterfaces.remote, test=False)
def list_modules(app: App = None):
    if app is None:
        app = get_app("cm.list_modules")
    return app.get_all_mods()


def create_and_pack_module(path, module_name='', version='-.-.-', additional_dirs=None, yaml_data=None):
    """
    Erstellt ein Python-Modul und packt es in eine ZIP-Datei.

    Args:
        path (str): Pfad zum Ordner oder zur Datei, die in das Modul aufgenommen werden soll.
        additional_dirs (dict): Zusätzliche Verzeichnisse, die hinzugefügt werden sollen.
        version (str): Version des Moduls.
        module_name (str): Name des Moduls.

    Returns:
        str: Pfad zur erstellten ZIP-Datei.
    """
    if additional_dirs is None:
        additional_dirs = {}
    if yaml_data is None:
        yaml_data = {}

    os.makedirs("./mods_sto/temp/", exist_ok=True)

    base_path = os.path.dirname(path)
    module_path = os.path.join(base_path, module_name)

    if not os.path.exists(module_path):
        module_path += '.py'

    temp_dir = tempfile.mkdtemp(dir=os.path.join("./mods_sto", "temp"))
    zip_file_name = f"RST${module_name}&{__version__}§{version}.zip"
    zip_path = f"./mods_sto/{zip_file_name}"

    # Modulverzeichnis erstellen, falls es nicht existiert
    if not os.path.exists(module_path):
        return False

    if os.path.isdir(module_path):
        # tbConfig.yaml erstellen
        config_path = os.path.join(module_path, "tbConfig.yaml")
        with open(config_path, 'w') as config_file:
            yaml.dump({"version": version, "module_name": module_name, "zip": zip_file_name, **yaml_data}, config_file)

        bundle_dependencies(module_path, config_path)
    # Datei oder Ordner in das Modulverzeichnis kopieren
    if os.path.isdir(module_path):
        shutil.copytree(module_path, os.path.join(temp_dir, os.path.basename(module_path)), dirs_exist_ok=True)
    else:
        shutil.copy2(module_path, temp_dir)
        config_path = os.path.join(temp_dir, f"{module_name}.yaml")
        with open(config_path, 'w') as config_file:
            yaml.dump({"version": version, "module_name": module_name, **yaml_data}, config_file)
        bundle_dependencies(temp_dir, config_path)
    # Zusätzliche Verzeichnisse hinzufügen
    for dir_name, dir_paths in additional_dirs.items():
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]
        for dir_path in dir_paths:
            full_path = os.path.join(temp_dir, dir_name)
            if os.path.isdir(dir_path):
                shutil.copytree(dir_path, full_path, dirs_exist_ok=True)
            elif os.path.isfile(dir_path):
                # Stellen Sie sicher, dass das Zielverzeichnis existiert
                os.makedirs(full_path, exist_ok=True)
                # Kopieren Sie die Datei statt des Verzeichnisses
                shutil.copy2(dir_path, full_path)
            else:
                print(f"Der Pfad {dir_path} ist weder ein Verzeichnis noch eine Datei.")

    # Modul in eine ZIP-Datei packen
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, temp_dir))

    # Temperatures Modulverzeichnis löschen
    shutil.rmtree(temp_dir)

    return zip_path


def uninstall_module(path, module_name='', version='-.-.-', additional_dirs=None, yaml_data=None):
    """
    Deinstalliert ein Python-Modul, indem es das Modulverzeichnis oder die ZIP-Datei entfernt.

    Args:
        path (str): Pfad zum Ordner oder zur Datei, die in das Modul aufgenommen werden soll.
        additional_dirs (dict): Zusätzliche Verzeichnisse, die hinzugefügt werden sollen.
        version (str): Version des Moduls.
        module_name (str): Name des Moduls.

    """
    if additional_dirs is None:
        additional_dirs = {}
    if yaml_data is None:
        yaml_data = {}

    os.makedirs("./mods_sto/temp/", exist_ok=True)

    base_path = os.path.dirname(path)
    module_path = os.path.join(base_path, module_name)
    zip_path = f"./mods_sto/RST${module_name}&{__version__}§{version}.zip"

    # Modulverzeichnis erstellen, falls es nicht existiert
    if not os.path.exists(module_path):
        print("Module %s already uninstalled")
        return False

    # Datei oder Ordner in das Modulverzeichnis kopieren
    shutil.rmtree(module_path)

    # Zusätzliche Verzeichnisse hinzufügen
    for dir_name, dir_paths in additional_dirs.items():
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]
        for dir_path in dir_paths:
            shutil.rmtree(dir_path)
            print(f"Der Pfad {dir_path} wurde entfernt")

    # Ursprüngliches Modulverzeichnis löschen
    shutil.rmtree(zip_path)


def unpack_and_move_module(zip_path, base_path='./mods', module_name=''):
    """
    Entpackt eine ZIP-Datei und verschiebt die Inhalte an die richtige Stelle.

    Args:
        zip_path (str): Pfad zur ZIP-Datei, die entpackt werden soll.
        base_path (str): Basispfad, unter dem das Modul gespeichert werden soll.
        module_name (str): Name des Moduls, der aus dem ZIP-Dateinamen extrahiert oder als Argument übergeben werden kann.
    """
    if not module_name:
        # Extrahiere den Modulnamen aus dem ZIP-Pfad, wenn nicht explizit angegeben
        module_name = os.path.basename(zip_path).split('$')[1].split('&')[0]

    module_path = os.path.join(base_path, module_name)

    os.makedirs("./mods_sto/temp/", exist_ok=True)
    # Temporäres Verzeichnis für das Entpacken erstellen
    temp_dir = tempfile.mkdtemp(dir=os.path.join("./mods_sto", "temp"))

    # ZIP-Datei entpacken
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Sicherstellen, dass das Zielverzeichnis existiert
    if os.path.isdir(module_name):
        os.makedirs(module_path, exist_ok=True)

    shutil.move(os.path.join(temp_dir, module_name), module_path)

    # Inhalte aus dem temporären Verzeichnis in das Zielverzeichnis verschieben
    for item in os.listdir(temp_dir):
        s = os.path.join(temp_dir, item)
        d = os.path.join("./", item)
        if os.path.isdir(s):
            shutil.move(s, d)
        else:
            shutil.copy2(s, d)

    # Temporäres Verzeichnis löschen
    shutil.rmtree(temp_dir)

    print(f"Modul {module_name} wurde erfolgreich nach {module_path} entpackt und verschoben.")
    return module_name


@export(mod_name=Name, name="make_install", test=False)
async def make_installer(app: Optional[App], module_name: str, base="./mods", upload=None):
    if app is None:
        app = get_app(f"{Name}.installer")

    if module_name not in app.get_all_mods():
        return "module not found"
    with Spinner("test loading module"):
        app.save_load(module_name)
    mod = app.get_mod(module_name)
    version_ = version
    if mod is not None:
        version_ = mod.version
    with Spinner("create and pack module"):
        zip_path = create_and_pack_module(f"{base}/{module_name}", module_name, version_)
    if upload or 'y' in input("uploade zip file ?"):
        with Spinner("Uploading file"):
            res = await app.session.upload_file(zip_path, '/installer/upload-file/')
        print(res)
        if isinstance(res, dict):
            if res.get('res', '').startswith('Successfully uploaded'):
                return Result.ok(res)
            return Result.default_user_error(res)
    return Result.ok(zip_path)


@export(mod_name=Name, name="uninstall", test=False)
def uninstaller(app: Optional[App], module_name: str):
    if app is None:
        app = get_app(f"{Name}.installer")

    if module_name not in app.get_all_mods():
        return "module not found"

    version_ = app.get_mod(module_name).version

    if 'y' in input("uploade zip file ?"):
        pass
    don = uninstall_module(f"./mods/{module_name}", module_name, version_)

    return don


@export(mod_name=Name, name="upload", test=False)
async def upload(app: Optional[App], module_name: str):
    if app is None:
        app = get_app(f"{Name}.installer")

    zip_path = \
        find_highest_zip_version_entry(module_name, filepath=f'{app.start_dir}/tbState.yaml').get('url', '').split(
            'mods_sto')[-1]

    if upload or 'y' in input(f"uploade zip file {zip_path} ?"):
        await app.session.upload_file(zip_path, '/installer/upload-file/')


@export(mod_name=Name, name="install", test=False)
async def installer(app: Optional[App], module_name: str):
    if app is None:
        app = get_app(f"{Name}.installer")

    if not app.session.valid:
        return None

    response = await app.session.fetch("/installer/get?name=" + module_name, method="GET")
    json_response = await response.json()
    response = json.loads(json_response)

    do_install = True

    if "provider" not in response.keys() and "url" not in response.keys():
        print("Error while fetching mod data", response)
        do_install = False

    if response['provider'] != "SimpleCore":
        print("Error while fetching mod data")
        do_install = False

    if module_name not in response['url']:
        print("404 mod not found")
        do_install = False

    if not do_install:
        return
    mod_url = response['url'].replace(app.session.base, "/")
    print(f"mod url is : {app.session.base + mod_url}")
    if not await app.session.download_file(response['result']['data'], app.start_dir + '/mods_sto'):
        print("failed to download mod")
        print("optional download it ur self and put the zip in the mods_sto folder")
        if 'y' not in input("Done ? will start set up from the mods_sto folder").lower():
            return
    os.rename(app.start_dir + '/mods_sto/' + mod_url.split('/')[-1].replace("$", '').replace("&", '').replace("§", ''),
              app.start_dir + '/mods_sto/' + mod_url.split('/')[-1])
    report = install_from_zip(app, mod_url.split('/')[-1])
    if not report:
        print("Set up error")
    return


@export(mod_name=Name, name="update_all", test=False)
async def update_all_mods(app, ignor_app_version=True):
    if app is None:
        app = get_app(f"{Name}.update_all")
    all_mods = app.get_all_mods()

    async def pipeline(name):
        mod_info = await app.session.fetch("/install/get?name="+name)
        json_response = await mod_info.json()
        mod_version = [__version__, app.get_mod(name).version]
        remote_mod_version = json.loads(json_response).get("version", [__version__, '0'])
        if len(remote_mod_version) == 1:
            remote_mod_version = [remote_mod_version[0], '0']

        local_app_version, local_mod_version = (p_version.parse(mod_version[0]),
                                                p_version.parse(mod_version[1]))
        remote_app_version, remote_mod_version = (p_version.parse(remote_mod_version[0]),
                                                  p_version.parse(remote_mod_version[1]))

        if not ignor_app_version and local_app_version < remote_app_version:
            app.run_any("CloudM", "update_core")
            exit(0)

        if local_mod_version < remote_mod_version:
            await installer(app, name)

    [await pipeline(mod) for mod in all_mods]


@export(mod_name=Name, name="build_all", test=False)
async def update_all_mods(app, base="./mods", upload=True):
    if app is None:
        app = get_app(f"{Name}.update_all")
    all_mods = app.get_all_mods()

    async def pipeline(name):
        res = await make_installer(app, name, base, upload)
        return res

    res = [await pipeline(mod) for mod in all_mods]
    for r in res:
        print(set(r))


def install_from_zip(app, zip_name, no_dep=True, auto_dep=False):
    zip_path = f"{app.start_dir}/mods_sto/{zip_name}"
    _name = unpack_and_move_module(zip_path, f"{app.start_dir}/mods")
    if not no_dep and os.path.exists(f"{app.start_dir}/mods/{_name}/tbConfig.yaml"):
            install_dependencies(f"{app.start_dir}/mods/{_name}/tbConfig.yaml", auto_dep)
    return True


#  =================== v2 functions =================

def run_command(command, cwd=None):
    """Führt einen Befehl aus und gibt den Output zurück."""
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
                            encoding='cp850')
    return result.stdout


def test_bundle_dependencies():
    _ = get_app("test_bundle_dependencies", "test")
    print("Testing bundle", _.id)
    dep = bundle_dependencies(r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\SchedulerManager.py",
                              return_dependencies=True)
    print(dep)


def bundle_dependencies(start_directory, output_file="dependencies.yaml", return_dependencies=False):
    dependencies = set()

    # Durchlaufen des Startverzeichnisses und Identifizieren von Abhängigkeiten
    # Hier ist ein vereinfachtes Beispiel, das externe Importe identifiziert
    # und zu den Abhängigkeiten hinzufügt
    def _(root_, file_):
        with open(os.path.join(root_, file_), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('import') or line.startswith('from'):
                    parts = line.split()
                    if len(parts) > 1 and parts[0] in ['import', 'from']:
                        dependencies.add(parts[1].split('.')[0])

    if start_directory.endswith('.py'):
        _("", start_directory)
    else:
        for root, dirs, files in os.walk(start_directory):
            for file in files:
                if file.endswith('.py'):
                    _(root, file)
    if "toolboxv2" in dependencies:
        dependencies.remove("toolboxv2")
    if return_dependencies:
        return list(dependencies)
    # Schreiben der Abhängigkeiten in YAML-Datei
    with open(output_file, 'a') as f:
        yaml.dump({"dependencies": list(dependencies)}, f)


def install_dependencies(yaml_file, do=False):
    with open(yaml_file, 'r') as f:
        dependencies = yaml.safe_load(f)

    if "dependencies" in dependencies:
        dependencies = dependencies["dependencies"]

    # Installation der Abhängigkeiten mit pip
    print(f"Dependency :", dependencies)
    for dependency in dependencies:
        if not do:
            if u_ip := input(f"{dependency} | install skipp exit [I/s/e]"):
                if u_ip == "s":
                    continue
                if u_ip == "e":
                    break
        subprocess.call(['pip', 'install', dependency])


def uninstall_dependencies(yaml_file):
    with open(yaml_file, 'r') as f:
        dependencies = yaml.safe_load(f)

    # Installation der Abhängigkeiten mit pip
    for dependency in dependencies:
        subprocess.call(['pip', 'uninstall', dependency])


if __name__ == "__main__":
    app_ = get_app()
    print(app_.get_all_mods())
    for module_ in app_.get_all_mods():  # ['dockerEnv', 'email_waiting_list',  'MinimalHtml', 'SchedulerManager', 'SocketManager', 'WebSocketManager', 'welcome']:
        print(f"Building module {module_}")
        make_installer(app_, module_, upload=False)
        time.sleep(0.1)
    # zip_path = create_and_pack_module("./mods/audio", "audio", "0.0.5")
    # print(zip_path)
    # unpack_and_move_module("./mods_sto/RST$audio&0.1.9§0.0.5.zip")
# # Beispielverwendung TODO
# archive_path = '/pfad/zum/archiv'
# temp_path = '/pfad/zum/temp'
# module_path = '/pfad/zum/modul'
# module_name = 'MeinModul'
#
# # Initialisiere und baue ein neues Modul
# initialize_module_repository(module_path)
# build_and_archive_module(module_path, archive_path)
#
# # Extrahiere, aktualisiere und rearchiviere ein existierendes Modul
# temp_module_path = extract_and_prepare_module(archive_path, module_name, temp_path)
# # Hier würden Sie Änderungen am Modul vornehmen
# update_and_rearchive_module(temp_module_path, archive_path)
