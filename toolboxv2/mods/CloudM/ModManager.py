import sys
import urllib.request
import json
import zipfile

import yaml
from tqdm import tqdm
from bs4 import BeautifulSoup
import tempfile

from toolboxv2 import get_app, App, __version__
from toolboxv2.utils.extras.Style import extract_json_strings
from toolboxv2.utils.system.types import ToolBoxInterfaces

import os
import subprocess
import shutil
from zipfile import ZipFile


Name = 'CloudM.ModManager'
export = get_app(f"{Name}.Export").tb
version = '0.0.1'
default_export = export(mod_name=Name, version=version, interface=ToolBoxInterfaces.native, test=False)


def unpack_mod(zip_path, extract_to):
    """Entpackt eine ZIP-Datei in ein Verzeichnis."""
    with ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)


def build_and_archive_module(module_path, archive_path, ver=version):
    """Baut das Modul und speichert es als ZIP im Archiv."""
    # Bauen des Moduls
    run_command(['python', '-m', 'build'], cwd=module_path)

    # Erstellen des ZIP-Archivs
    dist_path = os.path.join(module_path, 'dist')
    module_name = os.path.basename(module_path)
    zip_file_path = os.path.join(archive_path, f"RST${module_name}&{__version__}§{ver}.zip")
    with ZipFile(zip_file_path, 'w') as zipf:
        for file in os.listdir(dist_path):
            zipf.write(os.path.join(dist_path, file), arcname=file)

    return zip_file_path


def extract_and_prepare_module(archive_path, module_name, temp_path, ver=version):
    """Extrahiert ein Modul aus dem Archiv und bereitet es vor."""
    zip_file_path = os.path.join(archive_path, f"RST${module_name}&{__version__}§{ver}.zip")
    temp_module_path = os.path.join(temp_path, module_name)
    with ZipFile(zip_file_path, 'r') as zipf:
        zipf.extractall(temp_module_path)
    return temp_module_path


# shutil.rmtree(temp_path)

def install_zip_from_url(zip_url, target_directory, print_func):
    """
    Lädt eine ZIP-Datei von einer gegebenen URL herunter und entpackt sie in ein Zielverzeichnis.

    Args:
        zip_url (str): URL der ZIP-Datei, die heruntergeladen werden soll.
        target_directory (str): Pfad des Verzeichnisses, in das die ZIP-Datei entpackt werden soll.
        print_func (function): Funktion zum Ausgeben von Debug-Informationen.

    Returns:
        None
    """
    # Stellen Sie sicher, dass das Zielverzeichnis existiert
    os.makedirs(target_directory, exist_ok=True)

    # Temporäre Datei für den Download erstellen
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        print_func(f"Downloading ZIP from {zip_url}")
        # ZIP-Datei herunterladen
        urllib.request.urlretrieve(zip_url, tmp_file.name)
        print_func(f"ZIP downloaded successfully to {tmp_file.name}")

    # ZIP-Datei entpacken
    with ZipFile(tmp_file.name, 'r') as zip_ref:
        print_func(f"Extracting ZIP to {target_directory}")
        zip_ref.extractall(target_directory)
        print_func("ZIP extraction completed")

    # Temporäre Datei löschen
    os.remove(tmp_file.name)
    print_func("Temporary file removed")


def download_files(urls, directory, desc, print_func, filename=None):
    """ Hilfsfunktion zum Herunterladen von Dateien. """
    for url in tqdm(urls, desc=desc):
        if filename is None:
            filename = os.path.basename(url)
        print_func(f"Download {filename}")
        print_func(f"{url} -> {directory}\\{filename}")
        os.makedirs(directory, exist_ok=True)
        urllib.request.urlretrieve(url, f"{directory}\\{filename}")
    return f"{directory}\\{filename}"


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


@default_export
def delete_package(url):
    if isinstance(url, list):
        for i in url:
            if i.strip().startswith('http'):
                url = i
                break
    with urllib.request.urlopen(url) as response:
        res = response \
            .read()
        soup = BeautifulSoup(res, 'html.parser')
        data = json.loads(extract_json_strings(soup.text)[0].replace('\n', ''))

    for mod_url in tqdm(data["mods"], desc="Mods löschen"):
        filename = os.path.basename(mod_url)
        file_path = os.path.join("mods", filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    for runnable_url in tqdm(data["runnable"], desc="Runnables löschen"):
        filename = os.path.basename(runnable_url)
        file_path = os.path.join("runnable", filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    additional_dir_path = os.path.join("mods",
                                       os.path.basename(data["additional-dirs"]))
    if os.path.exists(additional_dir_path):
        shutil.rmtree(additional_dir_path)

    # Herunterladen der Requirements-Datei
    requirements_url = data["requirements"]
    requirements_filename = f"{data['Name']}-requirements.txt"
    urllib.request.urlretrieve(requirements_url, requirements_filename)

    # Deinstallieren der Requirements mit pip
    with tempfile.NamedTemporaryFile(mode="w",
                                     delete=False) as temp_requirements_file:
        with open(requirements_filename) as original_requirements_file:
            for line in original_requirements_file:
                package_name = line.strip().split("==")[0]
                temp_requirements_file.write(f"{package_name}\n")

        temp_requirements_file.flush()
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", "-r",
            temp_requirements_file.name
        ])

    # Löschen der heruntergeladenen Requirements-Datei
    os.remove(requirements_filename)
    os.remove(temp_requirements_file.name)


@export(mod_name=Name, api=True, interface=ToolBoxInterfaces.remote, test=False)
def list_modules(app: App = None):
    if app is None:
        app = get_app("cm.list_modules")
    return list(map(lambda x: '/api/installer/' + x + '-installer.json', app.get_all_mods()))


def create_and_pack_module(path, additional_dirs, version, module_name):
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
    base_path = os.path.dirname(path)
    module_path = os.path.join(base_path, module_name)
    zip_path = f"RST${module_name}&{__version__}§{version}.zip"

    # Modulverzeichnis erstellen, falls es nicht existiert
    os.makedirs(module_path, exist_ok=True)

    # Datei oder Ordner in das Modulverzeichnis kopieren
    if os.path.isdir(path):
        shutil.copytree(path, os.path.join(module_path, os.path.basename(path)), dirs_exist_ok=True)
    else:
        shutil.copy2(path, module_path)

    # Zusätzliche Verzeichnisse hinzufügen
    for dir_name, dir_paths in additional_dirs.items():
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]
        for dir_path in dir_paths:
            full_path = os.path.join(module_path, dir_name)
            shutil.copytree(dir_path, full_path, dirs_exist_ok=True)

    # tbConfig.yaml erstellen
    config_path = os.path.join(module_path, "tbConfig.yaml")
    with open(config_path, 'w') as config_file:
        yaml.dump({"version": version, "module_name": module_name}, config_file)

    # Modul in eine ZIP-Datei packen
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(module_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, base_path))

    # Ursprüngliches Modulverzeichnis löschen
    shutil.rmtree(module_path)

    return zip_path


#  =================== v2 functions =================

def run_command(command, cwd=None):
    """Führt einen Befehl aus und gibt den Output zurück."""
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return result.stdout


if __name__ == "__main__":

    # package module
    get_app()
    zip_path = create_and_pack_module(r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\welcome.py", {
        "runnable": [r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\runabel\readchar_buldin_style_cli.py"],
        ".":[r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\init.config"]
    }, "0.0.1", "Welcome")
    print(zip_path)
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
