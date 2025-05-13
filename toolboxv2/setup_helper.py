import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from toolboxv2.utils.system.api import (
    detect_os_and_arch,
    download_executable,
    query_executable_url,
)


def input_with_validation(prompt, valid_options=None):
    while True:
        user_input = input(prompt).strip().lower()
        if valid_options is None or user_input in valid_options:
            return user_input
        print("Ungültige Eingabe. Bitte wählen Sie eine gültige Option.")

def is_installed(tool):
    import shutil
    return shutil.which(tool) is not None

def ask_choice(prompt, choices):
    print(prompt)
    for idx, choice in enumerate(choices, 1):
        print(f"{idx}. {choice}")
    while True:
        try:
            selection = int(input("Wähle eine Option: "))
            if 1 <= selection <= len(choices):
                return choices[selection - 1]
        except ValueError:
            pass
        print("Ungültige Auswahl.")

def select_mode():
    return ask_choice("Wähle den Modus:", ["dev", "user"])

def select_python_interpreter():
    found = []
    for cmd in ["python3.11", "python3", "python", "coda", "uv"]:
        if is_installed(cmd):
            found.append(cmd)
    if not found:
        print("❌ Keine Python-Installation gefunden.")
        sys.exit(1)
    return ask_choice("Wähle einen Python-Interpreter/Manager:", found)

def setup_uv_api_env():
    helper = Path("uv_api_python_helper.py")
    if not helper.exists():
        print("❌ uv_api_python_helper.py nicht gefunden.")
        sys.exit(1)
    print("⚙️  Konfiguriere API-Umgebung via uv_helper...")
    run_command(f"{sys.executable} {helper}")

def install_dev_tools():
    print("🔧 Installiere Dev-Tools...")
    install_tools_parallel(["cargo", "node", "docker"], max_threads=3)

# === Platform groups ===
PLATFORMS = {
    "Windows": ["winget", "scoop", "choco"],
    "Linux": ["apt", "dnf", "pacman", "asdf"],
    "Darwin": ["brew", "asdf"],
}

# === Tool install templates ===
TEMPLATES = {
    "cargo": {
        "winget": "winget install Rustlang.Rustup",
        "scoop": "scoop install rust",
        "choco": "choco install rust",
        "apt": "sudo apt install -y cargo",
        "dnf": "sudo dnf install -y cargo",
        "pacman": "sudo pacman -S --noconfirm cargo",
        "brew": "brew install rust",
        "asdf": "asdf plugin-add rust || true && asdf install rust latest",
    },
    "node": {
        "winget": "winget install OpenJS.NodeJS",
        "scoop": "scoop install nodejs",
        "choco": "choco install nodejs",
        "apt": "sudo apt install -y nodejs",
        "dnf": "sudo dnf install -y nodejs",
        "pacman": "sudo pacman -S --noconfirm nodejs npm",
        "brew": "brew install node",
        "asdf": "asdf plugin-add nodejs || true && asdf install nodejs latest",
    },
    "docker": {
        "winget": "winget install Docker.DockerDesktop",
        "scoop": "scoop install docker",
        "choco": "choco install docker-desktop",
        "apt": "sudo apt install -y docker.io",
        "dnf": "sudo dnf install -y docker",
        "pacman": "sudo pacman -S --noconfirm docker",
        "brew": "brew install --cask docker",
        "asdf": "asdf plugin-add docker || true && asdf install docker latest",
    }
}

# === Binaries for detection ===
BIN_MAP = {
    "winget": "winget",
    "scoop": "scoop",
    "choco": "choco",
    "apt": "apt",
    "dnf": "dnf",
    "pacman": "pacman",
    "brew": "brew",
    "asdf": "asdf"
}

# === System info ===
def get_current_managers():
    system = platform.system()
    return PLATFORMS.get(system, [])

# === Build manager list dynamically ===
def get_managers_for_tool(tool):
    available_managers = get_current_managers()
    tool_cmds = TEMPLATES.get(tool, {})
    return [
        {
            "name": mgr,
            "bin": BIN_MAP[mgr],
            "install_cmd": tool_cmds[mgr]
        }
        for mgr in available_managers if mgr in tool_cmds
    ]


# === Command runner ===
def run_command(command, cwd=None, silent=False):
    if cwd is None:
        from toolboxv2 import cwd as _cwd
        cwd = _cwd
    try:
        subprocess.run(command, cwd=cwd, shell=True, check=True,
                       stdout=subprocess.PIPE if silent else None)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei: {command} → {e}")
        return False

# === Try installing with available manager ===
def install_with_manager(tool):
    for mgr in get_managers_for_tool(tool):
        if shutil.which(mgr["bin"]):
            print(f"⚙️  Installing {tool} using {mgr['name']}...")
            success = run_command(mgr["install_cmd"])
            return tool, success
    print(f"⛔ Kein verfügbarer Installer für {tool}")
    return tool, False

# === Parallel installer ===
def install_tools_parallel(tools, max_threads=3):
    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(install_with_manager, tool): tool for tool in tools}
        for future in as_completed(futures):
            tool, success = future.result()
            status = "✅" if success else "❌"
            print(f"{status} Installation abgeschlossen: {tool}")
            results.append((tool, success))
    return results


def install_all_npm_deps():
    print("📦 Installiere npm-Abhängigkeiten...")
    from toolboxv2 import cwd as _cwd
    tb_root = _cwd
    web_dir = os.path.join(tb_root, "web")

    success = run_command("npm install", cwd=tb_root)
    success_web = run_command("npm install", cwd=web_dir)

    return success and success_web

def build_web_dist():
    print("🛠 Baue Web-Distribution...")
    from toolboxv2 import cwd as _cwd
    tb_root = _cwd
    return run_command("npm run build", cwd=tb_root)

def build_tauri():
    print("🖥 Baue Tauri-App...")

    from toolboxv2 import cwd as _cwd
    tb_root = _cwd
    tauri_prefix = os.path.join(tb_root, "simple-core")
    build_cmd = "npm run tauriB"

    # Trigger build
    success = run_command(build_cmd, cwd=tb_root)
    if not success:
        return False

    # Detect platform-specific release path
    target_dir = os.path.join(
        tauri_prefix,
        "src-tauri",
        "target",
        "release",
        "bundle"
    )

    # Define binary extensions for different systems
    system = platform.system()
    binary_ext = {
        "Windows": ".exe",
        "Darwin": ".app",
        "Linux": ""  # usually ELF
    }

    # Try to locate the bundle
    found = False
    for root, _dirs, files in os.walk(target_dir):
        for f in files:
            if f.endswith(binary_ext[system]):
                print(f"📦 Gefundene Release-Datei: {f}")
                shutil.copy(os.path.join(root, f), tb_root)
                found = True
                break
        if found:
            break

    if not found:
        print("⚠️ Konnte die Release-Datei nicht finden.")

    return True

def full_build_pipeline(dev_mode=False):
    if not install_all_npm_deps():
        print("❌ Fehler bei der Installation der npm-Pakete.")
        return

    if not build_web_dist():
        print("❌ Fehler beim Erstellen der Webdist.")
        return

    if dev_mode:
        print("⚙️ DEV-Modus erkannt: Baue Desktop-App...")
        if not build_tauri():
            print("⚠️ Desktop-Build fehlgeschlagen.")
        else:
            print("✅ Desktop-App gebaut und in toolboxv2 abgelegt.")

    print("✅ Build abgeschlossen.")

def setup_main():
    mode = select_mode()
    python_choice = select_python_interpreter()

    if python_choice == "uv":
        print("⚠️ Die API benötigt eine separate native Python-Umgebung, uv reicht nicht aus.")
        setup_uv_api_env()

    if mode == "dev":
        install_dev_tools()
    full_build_pipeline(mode == "dev")

    if mode == "user":
        current_os, machine = detect_os_and_arch()
        print(f"Detected OS: {current_os}, Architecture: {machine}")
        url, file_name = query_executable_url(current_os, machine) # install rust server
        _ = download_executable(url, file_name) # install tauri app
        if current_os == "windows":
            file_name = f"app_{current_os}_{machine}.exe"
        else:
            file_name = f"app_{current_os}_{machine}"
        _ = download_executable(url, file_name) # install tauri app

    print("✅ Setup abgeschlossen.")
    print("run tb --ipy / tb gui / tb api start")

if __name__ == "__main__":
    setup_main()

