"""The Task of the State System is :
 1 Kep trak of the current state of the ToolBox and its dependency's
 2 tracks the shasum of all mod and runnabael
 3 the version of all mod

 The state :
 {"utils":{"file_name": {"version":##,"shasum"}}
 ,"mods":{"file_name": {"version":##,"shasum":##,"src-url":##}}
 ,"runnable":{"file_name": {"version":##,"shasum":##,"src-url":##}}
 ,"api":{"file_name": {"version":##,"shasum"}}
 ,"app":{"file_name": {"version":##,"shasum":##,"src-url":##}}
 }

 trans form state from on to an other.
 """
import hashlib
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from ..extras.Style import Spinner
from .getting_and_closing_app import get_app


@dataclass
class DefaultFilesFormatElement:
    version: str = "-1"
    shasum: str = "-1"
    provider: str = "-1"
    url: str = "-1"

    def __str__(self):
        return f"{self.version=}{self.shasum=}{self.provider=}{self.url=}|".replace("self.", "")


@dataclass
class TbState:
    utils: dict[str, DefaultFilesFormatElement]
    mods: dict[str, DefaultFilesFormatElement]
    installable: dict[str, DefaultFilesFormatElement]
    runnable: dict[str, DefaultFilesFormatElement]
    api: dict[str, DefaultFilesFormatElement]
    app: dict[str, DefaultFilesFormatElement]

    def __str__(self):
        fstr = "Utils\n"
        for name, item in self.utils.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "Mods\n"
        for name, item in self.mods.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "Mods installable\n"
        for name, item in self.installable.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "runnable\n"
        for name, item in self.runnable.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "api\n"
        for name, item in self.api.items():
            fstr += f"  {name} | {str(item)}\n"
        fstr += "app\n"
        for name, item in self.app.items():
            fstr += f"  {name} | {str(item)}\n"
        return fstr


def calculate_shasum(file_path: str) -> str:
    BUF_SIZE = 65536

    sha_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        buf = file.read(BUF_SIZE)
        while len(buf) > 0:
            sha_hash.update(buf)
            buf = file.read(BUF_SIZE)

    return sha_hash.hexdigest()


def process_files(directory: str) -> TbState:
    utils = {}
    mods = {}
    runnable = {}
    installable = {}
    api = {}
    app = {}
    scann_dirs = ["utils", "mods", "runnable", "api", "app", "mods_sto"]
    for s_dir in scann_dirs:
        for root, _dirs, files in os.walk(directory+'/'+s_dir):
            for file_name in files:
                if file_name.endswith(".zip") and 'mods_sto' in root:
                    file_path = os.path.join(root, file_name)
                    shasum = calculate_shasum(file_path)

                    element = DefaultFilesFormatElement()
                    element.shasum = shasum
                    installable[file_name] = element

                if file_name.endswith(".py"):
                    file_path = os.path.join(root, file_name)
                    shasum = calculate_shasum(file_path)

                    element = DefaultFilesFormatElement()
                    element.shasum = shasum

                    if 'utils' in root:
                        utils[file_name] = element
                    elif 'mods' in root:
                        mods[file_name] = element
                    elif 'runnable' in root:
                        runnable[file_name] = element
                    elif 'api' in root:
                        api[file_name] = element
                    elif 'app' in root:
                        app[file_name] = element

    return TbState(
        utils=utils,
        mods=mods,
        installable=installable,
        runnable=runnable,
        api=api,
        app=app,
    )


def get_state_from_app(app, simple_core_hub_url="https://simplecore.app/",
                       github_url="https://github.com/MarkinHaus/ToolBoxV2/tree/master/toolboxv2/"):
    if simple_core_hub_url[-1] != '/':
        simple_core_hub_url += '/'

    if github_url[-1] != '/':
        github_url += '/'

    with Spinner("Scanning files"):
        state: TbState = process_files(app.start_dir)

    with tqdm(total=6, unit='chunk', desc='Building State data') as pbar:
        # and unit information
        # current time being units ar installed and managed via GitHub
        version = app.version
        pbar.write("working on utils files")
        for file_name, file_data in state.utils.items():
            file_data.provider = "git"
            file_data.version = version
            file_data.url = github_url + "utils/" + file_name
        pbar.update()
        pbar.write("working on api files")
        for file_name, file_data in state.api.items():
            file_data.provider = "git"
            file_data.version = version
            file_data.url = github_url + "api/" + file_name
        pbar.update()
        pbar.write("working on app files")
        for file_name, file_data in state.app.items():
            file_data.provider = "git"
            file_data.version = version
            file_data.url = github_url + "app/" + file_name

        # and mods information
        # current time being mods ar installed and managed via SimpleCoreHub.com
        all_mods = app.get_all_mods()
        pbar.update()
        pbar.write("working on mods files")
        for file_name, file_data in state.mods.items():
            file_data.provider = "SimpleCore"

            module_name = file_name.replace(".py", "")
            if module_name in all_mods:
                try:
                    file_data.version = app.get_mod(module_name).version if file_name != "__init__.py" else version
                except Exception:
                    file_data.version = "dependency"
            else:
                file_data.version = "legacy"

            file_data.url = simple_core_hub_url + "mods/" + file_name
        pbar.update()
        pbar.write("working on installable files")
        for file_name, file_data in state.installable.items():
            file_data.provider = "SimpleCore"
            try:
                file_data.version = file_name.replace(".py", "").replace(".zip", "").split("&")[-1].split("§")
            except Exception:
                file_data.version = "dependency"

            file_data.url = simple_core_hub_url + "installer/download/mods_sto/" + file_name
        pbar.update()
        pbar.write("Saving State Data")
        with open("tbState.yaml", "w") as config_file:
            yaml.dump(asdict(state), config_file)
        pbar.update()
    return state




def find_highest_zip_version_entry(name, target_app_version=None, filepath='tbState.yaml'):
    """
    Findet den Eintrag mit der höchsten ZIP-Version für einen gegebenen Namen und eine optionale Ziel-App-Version in einer YAML-Datei.

    :param name: Der Name des gesuchten Eintrags.
    :param target_app_version: Die Zielversion der App als String (optional).
    :param filepath: Der Pfad zur YAML-Datei.
    :return: Den Eintrag mit der höchsten ZIP-Version innerhalb der Ziel-App-Version oder None, falls nicht gefunden.
    """
    import yaml

    from packaging import version

    highest_zip_ver = None
    highest_entry = {}

    with open(filepath) as file:
        data = yaml.safe_load(file)
        # print(data)
        app_ver_h = None
        for key, value in list(data.get('installable', {}).items())[::-1]:
            # Prüfe, ob der Name im Schlüssel enthalten ist

            if name in key:
                v = value['version']
                if len(v) == 1:
                    app_ver = v[0].split('v')[-1]
                    zip_ver = "0.0.0"
                else:
                    app_ver, zip_ver = v
                    app_ver = app_ver.split('v')[-1]
                app_ver = version.parse(app_ver)
                # Wenn eine Ziel-App-Version angegeben ist, vergleiche sie
                if target_app_version is None or app_ver == version.parse(target_app_version):
                    current_zip_ver = version.parse(zip_ver)
                    # print(current_zip_ver, highest_zip_ver)

                    if highest_zip_ver is None or current_zip_ver > highest_zip_ver:
                        highest_zip_ver = current_zip_ver
                        highest_entry = value

                    if app_ver_h is None or app_ver > app_ver_h:
                        app_ver_h = app_ver
                        highest_zip_ver = current_zip_ver
                        highest_entry = value
    return highest_entry


def find_highest_zip_version(name_filter: str, app_version: str = None, root_dir: str = "mods_sto", version_only=False) -> str:
    """
    Findet die höchste verfügbare ZIP-Version in einem Verzeichnis basierend auf einem Namensfilter.

    Args:
        root_dir (str): Wurzelverzeichnis für die Suche
        name_filter (str): Namensfilter für die ZIP-Dateien
        app_version (str, optional): Aktuelle App-Version für Kompatibilitätsprüfung

    Returns:
        str: Pfad zur ZIP-Datei mit der höchsten Version oder None wenn keine gefunden
    """

    from packaging import version

    # Kompiliere den Regex-Pattern für die Dateinamen
    pattern = fr"{name_filter}&v[0-9.]+§([0-9.]+)\.zip$"

    highest_version = None
    highest_version_file = None

    # Durchsuche das Verzeichnis
    root_path = Path(root_dir)
    for file_path in root_path.rglob("*.zip"):
        if "RST$"+name_filter not in str(file_path):
            continue
        match = re.search(pattern, str(file_path).split("RST$")[-1].strip())
        if match:
            zip_version = match.group(1)

            # Prüfe App-Version Kompatibilität falls angegeben
            if app_version:
                file_app_version = re.search(r"&v([0-9.]+)§", str(file_path)).group(1)
                if version.parse(file_app_version) > version.parse(app_version):
                    continue

            # Vergleiche Versionen
            current_version = version.parse(zip_version)
            if highest_version is None or current_version > highest_version:
                highest_version = current_version
                highest_version_file = str(file_path)
    if version_only:
        return str(highest_version)
    return highest_version_file


def detect_os_and_arch():
    """Detect the current operating system and architecture."""
    current_os = platform.system().lower()  # e.g., 'windows', 'linux', 'darwin'
    machine = platform.machine().lower()  # e.g., 'x86_64', 'amd64'
    return current_os, machine


def query_executable_url(current_os, machine):
    """
    Query a remote URL for a matching executable based on OS and architecture.
    The file name is built dynamically based on parameters.
    """
    base_url = "https://example.com/downloads"  # Replace with the actual URL
    # Windows executables have .exe extension
    if current_os == "windows":
        file_name = f"server_{current_os}_{machine}.exe"
    else:
        file_name = f"server_{current_os}_{machine}"
    full_url = f"{base_url}/{file_name}"
    return full_url, file_name


def download_executable(url, file_name):
    """Attempt to download the executable from the provided URL."""
    try:
        import requests
    except ImportError:
        print("The 'requests' library is required. Please install it via pip install requests")
        sys.exit(1)

    print(f"Attempting to download executable from {url}...")
    try:
        response = requests.get(url, stream=True)
    except Exception as e:
        print(f"Download error: {e}")
        return None

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # Make the file executable on non-Windows systems
        if platform.system().lower() != "windows":
            os.chmod(file_name, 0o755)
        return file_name
    else:
        print("Download failed. Status code:", response.status_code)
        return None


if __name__ == "__main__":
    # Provide the directory to search for Python files
    app = get_app('state.system')
    app.load_all_mods_in_file()
    state = get_state_from_app(app)
    print(state)
    # def get_state_from_app(app: App):
    #    """"""
