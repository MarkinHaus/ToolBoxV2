import os
import sys
import subprocess
import urllib.request
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import shutil
import tempfile

from toolboxv2 import get_app, App
from toolboxv2.utils.Style import extract_json_strings
from toolboxv2.utils.types import ToolBoxInterfaces

Name = 'CloudM.ModManager'
export = get_app(f"{Name}.Export").tb
version = '0.0.1'
default_export = export(mod_name=Name, version=version, interface=ToolBoxInterfaces.native, test=False)


@default_export
def installer(url, debug=False):
    """
    Installiert Module und Abhängigkeiten basierend auf einer URL.

    Args:
        url (str or list): URL oder Liste von URLs zum Herunterladen der Installationsdaten.
        debug (bool): Aktiviert Debug-Ausgaben, wenn True.

    Returns:
        None
    """

    def print_debug(*args, **kwargs):
        """ Hilfsfunktion für Debug-Ausgaben. """
        if debug:
            print(*args, **kwargs)

    # URL-Verarbeitung
    if isinstance(url, list):
        url = next((u for u in url if u.strip().startswith('http')), None)

    if not url:
        raise ValueError("Keine gültige URL gefunden")

    # Daten herunterladen und verarbeiten
    with urllib.request.urlopen(url) as response:
        res = response.read()
        print_debug("Collecting installer file data")
        soup = BeautifulSoup(res, 'html.parser')
        data = json.loads(next(extract_json_strings(soup.text), '{}').replace('\n', ''))
        print_debug(f"data collected successfully data : {data}")

    # Verzeichnisse erstellen
    os.makedirs("mods", exist_ok=True)
    os.makedirs("runable", exist_ok=True)

    # Module und Runnables herunterladen
    download_files(data.get("mods", []), "mods", "Mods herunterladen", print_debug)
    download_files(data.get("runnable", []), "runable", "Runnables herunterladen", print_debug)

    # Zusätzliche Verzeichnisse
    handle_additional_dirs(data.get("additional-dirs"), print_debug)

    # Requirements installieren
    handle_requirements(data.get("requirements"), data.get("Name"), print_debug)

    print_debug("Installation abgeschlossen")


def download_files(urls, directory, desc, print_func):
    """ Hilfsfunktion zum Herunterladen von Dateien. """
    for url in tqdm(urls, desc=desc):
        filename = os.path.basename(url)
        print_func(f"Download {filename}")
        urllib.request.urlretrieve(url, f"{directory}/{filename}")


def handle_additional_dirs(additional_dirs_url, print_func):
    """ Verarbeitet zusätzliche Verzeichnisse. """
    if additional_dirs_url:
        print_func(f"Download additional dirs {additional_dirs_url}")
        shutil.unpack_archive(additional_dirs_url, "/")


def handle_requirements(requirements_url, module_name, print_func):
    """ Verarbeitet und installiert Requirements. """
    if requirements_url:
        requirements_filename = f"{module_name}-requirements.txt"
        print_func(f"Download requirements {requirements_filename}")
        urllib.request.urlretrieve(requirements_url, requirements_filename)

        print_func("Install requirements")
        subprocess.check_call(
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
