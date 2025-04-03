import json
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


from pathlib import Path
from packaging import version
import re
import os
import platform
import subprocess
import sys
import shutil
import time


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


def build_cargo_project(debug=False):
    """Build the Cargo project, optionally in debug mode."""
    mode = "debug" if debug else "release"
    args = ["cargo", "build"]
    if not debug:
        args.append("--release")

    print(f"Building in {mode} mode...")
    try:
        subprocess.run(args, cwd=os.path.join(".", "src-core"), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Cargo build failed: {e}")
        return False


def run_with_hot_reload():
    """Run the Cargo project with hot reloading."""
    src_core_path = os.path.join(".", "src-core")

    # Check if cargo-watch is installed
    try:
        subprocess.run(["cargo", "watch", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        print("cargo-watch is not installed. Installing now...")
        try:
            subprocess.run(["cargo", "install", "cargo-watch"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install cargo-watch: {e}")
            print("Running without hot reload")
            return run_in_debug_mode()

    print("Running with hot reload in debug mode...")
    try:
        subprocess.run(["cargo", "watch", "-x", "run"], cwd=src_core_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Hot reload execution failed: {e}")
        return False


def run_in_debug_mode():
    """Run the Cargo project in debug mode."""
    src_core_path = os.path.join(".", "src-core")
    print("Running in debug mode...")
    try:
        subprocess.run(["cargo", "run"], cwd=src_core_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Debug execution failed: {e}")
        return False


def remove_release_executable():
    """Removes the release executable."""
    src_core_path = os.path.join(".", "src-core")
    expected_name = "simple-core-server.exe" if platform.system().lower() == "windows" else "simple-core-server"

    # Remove from src-core root
    direct_path = os.path.join(src_core_path, expected_name)
    if os.path.exists(direct_path):
        try:
            os.remove(direct_path)
            print(f"Removed release executable: {direct_path}")
        except Exception as e:
            print(f"Failed to remove {direct_path}: {e}")

    # Remove from target/release
    release_path = os.path.join(src_core_path, "target", "release", expected_name)
    if os.path.exists(release_path):
        try:
            os.remove(release_path)
            print(f"Removed release executable: {release_path}")
        except Exception as e:
            print(f"Failed to remove {release_path}: {e}")

    return True


def cleanup_build_files():
    """Cleans up build files."""
    src_core_path = os.path.join(".", "src-core")
    target_path = os.path.join(src_core_path, "target")

    if os.path.exists(target_path):
        try:
            print(f"Cleaning up build files in {target_path}...")
            # First try using cargo clean
            try:
                subprocess.run(["cargo", "clean"], cwd=src_core_path, check=True)
                print("Successfully cleaned up build files with cargo clean")
            except subprocess.CalledProcessError:
                # If cargo clean fails, manually remove directories
                print("Cargo clean failed, manually removing build directories...")
                for item in os.listdir(target_path):
                    item_path = os.path.join(target_path, item)
                    if os.path.isdir(item_path) and item != ".rustc_info.json":
                        shutil.rmtree(item_path)
                        print(f"Removed {item_path}")
            return True
        except Exception as e:
            print(f"Failed to clean up build files: {e}")
            return False
    else:
        print(f"Build directory {target_path} not found")
        return True


import os
import sys
import subprocess
import tarfile
import shutil

def is_uv_installed():
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def get_uv_site_packages():
    """Find the site-packages directory for a uv-managed virtual environment."""
    try:
        site_packages = subprocess.check_output(["uv", "info", "--json"], text=True)
        import json
        data = json.loads(site_packages)
        return data["venv"]["site_packages"]
    except Exception as e:
        print(f"Error finding uv site-packages: {e}")
        return None

def create_dill_archive(site_packages, output_file="python312.dill"):
    """Package dill and all dependencies into a single .dill archive."""
    try:
        temp_dir = "/tmp/dill_package"
        os.makedirs(temp_dir, exist_ok=True)

        # Copy only necessary packages
        packages = ["dill"]
        for package in packages:
            package_path = os.path.join(site_packages, package)
            if os.path.exists(package_path):
                shutil.copytree(package_path, os.path.join(temp_dir, package), dirs_exist_ok=True)
            else:
                print(f"Warning: {package} not found in site-packages.")

        # Create the .dill archive
        with tarfile.open(output_file, "w:gz") as tar:
            tar.add(temp_dir, arcname=".")

        print(f"Successfully created {output_file}")

        # Clean up
        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"Error creating .dill archive: {e}")

def add_py_dill():
    if not is_uv_installed():
        print("uv is not installed. Please install uv before running this script.")
        return
    print(f"VIRTUAL_ENV=$ {os.getenv('VIRTUAL_ENV')}")
    site_packages = os.getenv("PY_SITE_PACKAGES")
    if not site_packages:
        print("Could not determine site-packages path. Is this a uv environment?")
        return

    print(f"Packaging dill from {site_packages}...")
    create_dill_archive(site_packages, output_file=os.getenv("PY_DILL"))



def main_api_runner(debug=False):
    """
    Main function to run the API server.
    When debug=True, enables hot reloading and runs in debug mode.

    Non blocking!
    """
    if not os.path.exists(os.getenv("PY_DILL", '.')):
        add_py_dill()
    if is_uv_installed():
        print(f"VIRTUAL_ENV=$ {os.getenv('VIRTUAL_ENV')} {os.getenv("PY_SITE_PACKAGES")}")
        os.environ["VIRTUAL_ENV"] = os.getenv('UV_BASE_ENV', os.getenv('VIRTUAL_ENV'))
        # os.environ["PY_SITE_PACKAGES"] = os.getenv('PY_SITE_PACKAGES')
    if debug:
        print("Starting in DEBUG mode with hot reloading enabled...")
        if check_cargo_installed():
            run_with_hot_reload()
        else:
            print("Cargo is not installed. Hot reloading requires Cargo.")
        return

    # Release mode flow
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
            if build_cargo_project(debug=False):
                # After successful build, try running the release executable again
                if check_and_run_local_release():
                    return
                else:
                    print("Release executable missing even after build.")
            else:
                print("Failed to build the Cargo project.")
        else:
            print("Cargo is not installed. Please install Cargo to build the project.")

def cli_api_runner():

    import argparse

    parser = argparse.ArgumentParser(description="API Runner and Build Utilities")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with hot reloading')
    parser.add_argument('--clean', action='store_true', help='Clean up build files')
    parser.add_argument('--remove-exe', action='store_true', help='Remove release executables')

    args = parser.parse_args()

    if args.clean:
        cleanup_build_files()

    if args.remove_exe:
        remove_release_executable()

    if not (args.clean or args.remove_exe):
        main_api_runner(debug=args.debug)

if __name__ == "__main__":
    # Example usage of the utility functions
    cli_api_runner()
