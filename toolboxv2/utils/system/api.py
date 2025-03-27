from pathlib import Path
from packaging import version
import re

def find_highest_zip_version_entry(name, target_app_version=None, filepath='tbState.yaml'):
    """
    Findet den Eintrag mit der höchsten ZIP-Version für einen gegebenen Namen und eine optionale Ziel-App-Version in einer YAML-Datei.

    :param name: Der Name des gesuchten Eintrags.
    :param target_app_version: Die Zielversion der App als String (optional).
    :param filepath: Der Pfad zur YAML-Datei.
    :return: Den Eintrag mit der höchsten ZIP-Version innerhalb der Ziel-App-Version oder None, falls nicht gefunden.
    """
    import yaml
    highest_zip_ver = None
    highest_entry = {}

    with open(filepath, 'r') as file:
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


import os
import platform
import subprocess
import sys


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
        file_name = f"app_{current_os}_{machine}.exe"
    else:
        file_name = f"app_{current_os}_{machine}"
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


def run_executable(file_path):
    """Run the executable file."""
    try:
        print("Running it.")
        subprocess.run([os.path.abspath(file_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute {file_path}: {e}")


def check_and_run_local_release():
    """Search for a pre-built release executable in the src-core folder and run it if found."""
    src_core_path = os.path.join(".", "src-core")
    if os.path.isdir(src_core_path):
        # Define the path to the expected release executable, assuming a Cargo project structure
        expected_name = "simple-core-server.exe" if platform.system().lower() == "windows" else "simple-core-server"
        release_path = os.path.join(src_core_path, expected_name)
        if os.path.isfile(release_path):
            print("Found pre-built release executable.")
            run_executable(release_path)
            return True
        release_path = os.path.join(src_core_path, "target", "release", expected_name)
        if os.path.isfile(release_path):
            print("Found pre-built release executable.")
            # Move the executable from target/release to src_core_path for easier access next time
            dest_path = os.path.join(src_core_path, expected_name)
            try:
                import shutil
                shutil.copy2(release_path, dest_path)
                print(f"Copied executable to {dest_path} for easier access next time")
            except Exception as e:
                print(f"Failed to copy executable: {e}")
                return False
            run_executable(dest_path)
            return True
    return False


def check_cargo_installed():
    """Check if Cargo (Rust package manager) is installed on the system."""
    try:
        subprocess.run(["cargo", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def build_cargo_project():
    """Build the Cargo project in release mode located in the src-core folder."""
    print("Building in release mode...")
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=os.path.join(".", "src-core"), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Cargo build failed:", e)
        return False


def main_api_runner():
    """Non blocking!"""
    if check_and_run_local_release():
        return
    # Step 1: Detect current OS and machine architecture
    current_os, machine = detect_os_and_arch()
    print(f"Detected OS: {current_os}, Architecture: {machine}")

    # Step 2: Attempt to download executable from remote URL
    url, file_name = query_executable_url(current_os, machine)
    downloaded_exe = download_executable(url, file_name)

    if downloaded_exe:
        print("Downloaded executable. Executing it...")
        run_executable(downloaded_exe)
        return

    # Step 3: Fallback: Check for local pre-built release executable in src-core folder
    print("Remote executable not found. Searching local 'src-core' folder...")
    if check_and_run_local_release():
        return
    else:
        print("Pre-built release executable not found locally.")

        # Step 4: If executable not found locally, check if Cargo is installed
        if check_cargo_installed():
            print("Cargo is installed. Proceeding with build.")
            if build_cargo_project():
                # After successful build, try running the release executable again
                if check_and_run_local_release():
                    return
                else:
                    print("Release executable missing even after build.")
            else:
                print("Failed to build the Cargo project.")
        else:
            print("Cargo is not installed. Please install Cargo to build the project.")

