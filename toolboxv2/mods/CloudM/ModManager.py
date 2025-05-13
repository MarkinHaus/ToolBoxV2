import asyncio
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml
from packaging import version as pv
from packaging.version import Version
from tqdm import tqdm

from toolboxv2 import App, Spinner, __version__, get_app
from toolboxv2.utils.extras.reqbuilder import generate_requirements
from toolboxv2.utils.system.api import find_highest_zip_version
from toolboxv2.utils.system.state_system import get_state_from_app
from toolboxv2.utils.system.types import Result, ToolBoxInterfaces

Name = 'CloudM'
export = get_app(f"{Name}.Export").tb
version = "0.0.3"
default_export = export(mod_name=Name, version=version, interface=ToolBoxInterfaces.native, test=False)
mv = None


def increment_version(version_str: str, max_value: int = 99) -> str:
    """
    Inkrementiert eine Versionsnummer im Format "vX.Y.Z".

    Args:
        version_str (str): Die aktuelle Versionsnummer, z. B. "v0.0.1".
        max_value (int): Die maximale Zahl pro Stelle (default: 99).

    Returns:
        str: Die inkrementierte Versionsnummer.
    """
    if not version_str.startswith("v"):
        raise ValueError("Die Versionsnummer muss mit 'v' beginnen, z. B. 'v0.0.1'.")

    # Entferne das führende 'v' und parse die Versionsnummer
    version_core = version_str[1:]
    try:
        version = Version(version_core)
    except ValueError as e:
        raise ValueError(f"Ungültige Versionsnummer: {version_core}") from e

    # Extrahiere die Versionsteile und konvertiere sie zu einer Liste
    parts = list(version.release)

    # Inkrementiere die letzte Stelle
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] < max_value:
            parts[i] += 1
            break
        else:
            parts[i] = 0
            # Schleife fährt fort, um die nächsthöhere Stelle zu inkrementieren
    else:
        # Wenn alle Stellen auf "max_value" sind, füge eine neue Stelle hinzu
        parts.insert(0, 1)

    # Baue die neue Version
    new_version = "v" + ".".join(map(str, parts))
    return new_version


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

    module_path = os.path.join(path, module_name)
    print("module_pathmodule_pathmodule_path", module_path)
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
            yaml.dump({"version": version, "module_name": module_name,
                       "dependencies_file": f"./mods/{module_name}/requirements.txt",
                       "zip": zip_file_name, **yaml_data}, config_file)

        generate_requirements(module_path, os.path.join(module_path, "requirements.txt"))
    # Datei oder Ordner in das Modulverzeichnis kopieren
    if os.path.isdir(module_path):
        shutil.copytree(module_path, os.path.join(temp_dir, os.path.basename(module_path)), dirs_exist_ok=True)
    else:
        shutil.copy2(module_path, temp_dir)
        config_path = os.path.join(temp_dir, f"{module_name}.yaml")
        with open(config_path, 'w') as config_file:
            yaml.dump({"version": version, "dependencies_file": f"./mods/{module_name}/requirements.txt",
                       "module_name": module_name, **yaml_data}, config_file)
        generate_requirements(temp_dir, os.path.join(temp_dir, "requirements.txt"))
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
        for root, _dirs, files in os.walk(temp_dir):
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
    for _dir_name, dir_paths in additional_dirs.items():
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]
        for dir_path in dir_paths:
            shutil.rmtree(dir_path)
            print(f"Der Pfad {dir_path} wurde entfernt")

    # Ursprüngliches Modulverzeichnis löschen
    shutil.rmtree(zip_path)


def unpack_and_move_module(zip_path: str, base_path: str = './mods', module_name: str = '') -> str:
    """
    Entpackt eine ZIP-Datei und verschiebt die Inhalte an die richtige Stelle.
    Überschreibt existierende Dateien für Update-Unterstützung.

    Args:
        zip_path (str): Pfad zur ZIP-Datei, die entpackt werden soll
        base_path (str): Basispfad, unter dem das Modul gespeichert werden soll
        module_name (str): Name des Moduls (optional, wird sonst aus ZIP-Namen extrahiert)

    Returns:
        str: Name des installierten Moduls
    """
    # Konvertiere Pfade zu Path-Objekten für bessere Handhabung
    zip_path = Path(zip_path)
    base_path = Path(base_path)

    # Extrahiere Modulnamen falls nicht angegeben
    if not module_name:
        module_name = zip_path.name.split('$')[1].split('&')[0]

    module_path = base_path / module_name
    temp_base = Path('./mods_sto/temp')

    try:
        # Erstelle temporäres Verzeichnis
        temp_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temp_base)) as temp_dir:
            temp_dir = Path(temp_dir)

            with Spinner(f"Extracting {zip_path.name}"):
                # Entpacke ZIP-Datei
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

            # Behandle Modul-Verzeichnis
            source_module = temp_dir / module_name
            if source_module.exists():
                with Spinner(f"Installing module to {module_path}"):
                    if module_path.exists():
                        # Lösche existierendes Modul-Verzeichnis für sauberes Update
                        shutil.rmtree(module_path)
                    # Verschiebe neues Modul-Verzeichnis
                    shutil.copytree(source_module, module_path, dirs_exist_ok=True)

            # Behandle zusätzliche Dateien im Root
            with Spinner("Installing additional files"):
                for item in temp_dir.iterdir():
                    if item.name == module_name:
                        continue

                    target = Path('./') / item.name
                    if item.is_dir():
                        with Spinner(f"Installing directory {item.name}"):
                            if target.exists():
                                shutil.rmtree(target)
                            shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        with Spinner(f"Installing file {item.name}"):
                            shutil.copy2(item, target)

            print(f"Successfully installed/updated module {module_name} to {module_path}")
            return module_name

    except Exception as e:
        print(f"Error during installation: {str(e)}")
        # Cleanup bei Fehler
        if module_path.exists():
            shutil.rmtree(module_path)
        raise


@export(mod_name=Name, name="make_install", test=False)
async def make_installer(app: App | None, module_name: str, base="./mods", upload=None):
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
        zip_path = create_and_pack_module(base, module_name, version_)
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
def uninstaller(app: App | None, module_name: str):
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
async def upload(app: App | None, module_name: str):
    if app is None:
        app = get_app(f"{Name}.installer")

    zip_path = find_highest_zip_version(module_name)
       # find_highest_zip_version_entry(module_name, filepath=f'{app.start_dir}/tbState.yaml').get('url', '').split(
        #    'mods_sto')[-1]

    if upload or 'y' in input(f"uploade zip file {zip_path} ?"):
        await app.session.upload_file(zip_path, '/installer/upload-file/')


@export(mod_name=Name, name="install", test=False)
async def installer(app: App | None, module_name: str, build_state=True):
    """
    Installiert oder aktualisiert ein Modul basierend auf der Remote-Version.
    """
    if app is None:
        app = get_app(f"{Name}.installer")

    if not app.session.valid and not await app.session.login():
        return Result.default_user_error("Please login with CloudM login")

    # Hole nur die höchste verfügbare Version vom Server
    response = await app.session.fetch(f"/installer/version?name={module_name}", method="GET")
    remote_version : str = await response.text()
    remote_version = remote_version.split('"')[1]
    if remote_version == "None":
        remote_version = None
    # Finde lokale Version
    local_version = find_highest_zip_version(
        module_name, version_only=True
    )

    if not local_version and not remote_version:
        return Result.default_user_error(f"404 mod {module_name} not found")

    # Vergleiche Versionen
    local_ver = pv.parse(local_version) if local_version else pv.parse("0.0.0")
    remote_ver = pv.parse(remote_version)

    app.print(f"Mod versions - Local: {local_ver}, Remote: {remote_ver}")

    if remote_ver > local_ver:
        # Konstruiere die URL direkt aus Modulname und Version
        mod_url = f"/installer/mods_sto/RST${module_name}&{app.version}§{remote_version}.zip"
        download_path = Path(app.start_dir) / 'mods_sto'

        app.print(f"Fetching Mod from {app.session.base+mod_url}")
        if not await app.session.download_file(mod_url, str(download_path)):
            app.print("Failed to download mod")
            if 'y' not in input("Download manually and place in mods_sto folder. Done? (y/n) ").lower():
                return Result.default_user_error("Installation cancelled")

        # Korrigiere Dateinamen
        zip_name = mod_url.split('/')[-1]
        clean_name = zip_name.replace("$", '').replace("&", '').replace("§", '')
        with contextlib.suppress(FileExistsError):
            os.rename(
                str(download_path / clean_name),
                str(download_path / zip_name)
            )

        with Spinner("Installing from zip"):
            report = install_from_zip(app, zip_name)

        if not report:
            return Result.default_user_error("Setup error occurred")

        if build_state:
            get_state_from_app(app)

        return report

    app.print("Module is already up to date")
    return Result.ok()


@export(mod_name=Name, name="update_all", test=False)
async def update_all_mods(app):
    """
    Aktualisiert alle installierten Module mit minimalen Server-Anfragen.
    """
    if app is None:
        app = get_app(f"{Name}.update_all")

    all_mods = app.get_all_mods()
    update_tasks = []

    async def check_and_update(mod_name: str):
        # Hole nur die Version vom Server
        version_response = await app.session.fetch(f"/installer/version/{mod_name}")
        remote_version = await version_response.text()

        if not remote_version or remote_version == "None":
            app.print(f"Could not fetch version for {mod_name}")
            return

        local_mod = app.get_mod(mod_name)
        if not local_mod:
            app.print(f"Local mod {mod_name} not found")
            return
        remote_version = remote_version.spilt('"')[1]
        if pv.parse(remote_version) > pv.parse(local_mod.version):
            await installer(app, mod_name, build_state=False)

    # Erstelle Update-Tasks für alle Module
    for mod in all_mods:
        update_tasks.append(check_and_update(mod))

    # Führe Updates parallel aus
    await asyncio.gather(*update_tasks)

    # Aktualisiere den State einmal am Ende
    get_state_from_app(app)


@export(mod_name=Name, name="build_all", test=False)
async def update_all_mods(app, base="mods", upload=True):
    if app is None:
        app = get_app(f"{Name}.update_all")
    all_mods = app.get_all_mods()

    async def pipeline(name):
        res = await make_installer(app, name, os.path.join('.', base), upload)
        return res

    res = [await pipeline(mod) for mod in all_mods]
    for r in res:
        print(r)


def install_from_zip(app, zip_name, no_dep=True, auto_dep=False):
    zip_path = f"{app.start_dir}/mods_sto/{zip_name}"
    with Spinner(f"unpack_and_move_module {zip_path[-30:]}"):
        _name = unpack_and_move_module(zip_path, f"{app.start_dir}/mods")
    if not no_dep and os.path.exists(f"{app.start_dir}/mods/{_name}/tbConfig.yaml"):
        with Spinner(f"install_dependencies {_name}"):
            install_dependencies(f"{app.start_dir}/mods/{_name}/tbConfig.yaml", auto_dep)
    return True


#  =================== v2 functions =================

def run_command(command, cwd=None):
    """Führt einen Befehl aus und gibt den Output zurück."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True,
                            encoding='cp850')
    return result.stdout



def install_dependencies(yaml_file, do=False):
    with open(yaml_file) as f:
        dependencies = yaml.safe_load(f)

    if "dependencies_file" in dependencies:
        dependencies_file = dependencies["dependencies_file"]

        # Installation der Abhängigkeiten mit pip
        print("Dependency :", dependencies_file)
        subprocess.call(['pip', 'install', '-r', dependencies_file])


def uninstall_dependencies(yaml_file):
    with open(yaml_file) as f:
        dependencies = yaml.safe_load(f)

    # Installation der Abhängigkeiten mit pip
    for dependency in dependencies:
        subprocess.call(['pip', 'uninstall', dependency])


if __name__ == "__main__":
    app_ = get_app('Manager')
    print(app_.get_all_mods())
    for module_ in app_.get_all_mods():  # ['dockerEnv', 'email_waiting_list',  'MinimalHtml', 'SchedulerManager', 'SocketManager', 'WebSocketManager', 'welcome']:
        print(f"Building module {module_}")
        make_installer(app_, module_, upload=False)
        time.sleep(0.1)
    # zip_path = create_and_pack_module("./mods/audio", "audio", "0.0.5")
    # print(zip_path)
    # unpack_and_move_module("./mods_sto/RST$audio&0.1.9§0.0.5.zip")

