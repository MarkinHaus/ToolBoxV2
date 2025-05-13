import json
import os
import pathlib
import time
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, WebSocket

from toolboxv2 import get_app
from toolboxv2.utils.system.api import find_highest_zip_version_entry
from toolboxv2.utils.system.state_system import TbState, get_state_from_app

router = APIRouter(
    prefix="/installer",
)


async def ws_send(data, websocket=None):
    time.sleep(0.001)
    await websocket.send_text(data)


@router.websocket("/generate_download_zip")
async def generate_download_zip(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()
    if isinstance(data, str):
        data = json.loads(data)

    ClientInfos = {'os': data.get('os', 'None')}
    app = get_app("Instalelr_for_user")
    """
    {
        'installationFolder': './',
        'targetVersion': null,
        'addLootUser': false,
        'stoFolder': './.data',
        'addDB': false,
        'DBtype': 'local',
        'DBUri': null,
        'DBUserName': null,
        'DBKey': null,
        'startBgRunnerSetup': false,
        'DefaultMods': ['DB', 'CloudM', 'welcome', 'EventManager', 'WidgetsProvider', 'api_manager',
            'cicd', 'cli_functions', 'SchedulerManager', 'SocketManager'],
        'connectTO': {
            'remote': false,
            'localP0': false,
        },
        'autoStart': false,
        "MODS": 'additional',
        'Install': ['DB', 'CloudM', 'welcome', 'EventManager', 'WidgetsProvider', 'api_manager',
            'cicd', 'cli_functions', 'SchedulerManager', 'SocketManager'],
    }

    """
    # Verarbeiten der Daten...
    # Senden von Nachrichten zurück an den Client
    target_version = data.get('targetVersion', app.version)
    await ws_send("Crate installation instructions yml data", websocket=websocket)
    await ws_send(f"root @> Testing Version: {target_version}", websocket=websocket)
    await ws_send("root @> Creating Mod bundle ...", websocket=websocket)
    installation_data = {
        "core": "",
        "mods": [],
        "extras": {},
        "dependency": []
    }
    for mod_name in data.get("Install", []):
        await ws_send(f"root @> searching for mod {mod_name}", websocket=websocket)
        mod_data = find_highest_zip_version_entry(mod_name, target_version)
        if mod_data is None:
            await ws_send(f"mods @> {mod_name} 404 not Found !!")
            continue
        installation_data["mods"].append(mod_data)
        await ws_send(f"mods @> added {mod_name} ", websocket=websocket)
        await ws_send(f"mods @> {mod_name} data: shasum = {mod_data.get('shasum')} url = {mod_data.get('url')}",
                      websocket=websocket)
        await ws_send(f"mods @> {mod_name} infos: version= {mod_data.get('version')}", websocket=websocket)
        await ws_send(f"mods @> {mod_name} infos: provider = {mod_data.get('provider')}", websocket=websocket)
    await ws_send("root @> adding Core", websocket=websocket)
    if target_version is None:
        target_version = '0.1.14'
    if dir_f := data.get("installationFolder"):
        installation_data['core'] = f"pip install --target={dir_f} ToolBoxV2==" + target_version
    else:
        installation_data['core'] = "pip install ToolBoxV2==" + target_version
    await ws_send("root @> adding Extras", websocket=websocket)
    if data.get("DBtype") == 'local_redis':
        installation_data['extras']['redis'] = 'local'

    await ws_send("root @> adding Dependencies", websocket=websocket)
    if "diffuser" in data.get("Install"):
        installation_data['dependency'].append("cuda")
        await ws_send("root @> adding cuda", websocket=websocket)

    if data.get("Ollama"):
        installation_data['dependency'].append("ollama")
        await ws_send("root @> adding ollama", websocket=websocket)

    if "isaa" in data.get("Install"):
        installation_data['dependency'].append("isaa")
        await ws_send("root @> adding isaa", websocket=websocket)

    await ws_send("Data crated successfully:", websocket=websocket)
    # Senden des Download-Links als letzte Nachricht
    await ws_send(f"Data: {json.dumps(installation_data, indent=2)}", websocket=websocket)

    urls = []
    for mods_data in installation_data["mods"]:
        urls.append(mods_data.get('url'))
    if ClientInfos.get('os') == "Windows":
        end = 'bat'
        script = """@echo off
setlocal EnableDelayedExpansion

:: Metadaten hinzufügen
set "Author=Markin Hausmanns"
set "WebPage=Simplecore.app"
set "ToolboxInstaller=Toolbox Windows Installer"

:: Benutzereingaben
echo Willkommen zum ToolboxV2 Installer.

:: Überprfen der Python-Version und Installation, falls notwendig
set "PythonFound=NO"
for /L %%i in (9,1,11) do (
    py -3.%%i --version > NUL 2>&1 && set "PythonFound=YES" && set "PythonVersion=3.%%i" && goto PythonFound
)

:PythonFound
if "%PythonFound%"=="NO" (
    echo Python 3.11 wird installiert...
    :: Python 3.11 Installer herunterladen und installieren
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe' -OutFile 'python-3.11.0-amd64.exe'"
    start /wait python-3.11.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
    del python-3.11.0-amd64.exe
) else (
    echo Gefundene Python-Version: %PythonVersion%
)

:: Überprüfen, ob ToolboxV2 bereits installiert ist
pip show ToolboxV2 > NUL 2>&1
if %ERRORLEVEL% == 0 (
    echo Eine Version von ToolboxV2 ist bereits installiert.
    set /p UpdateToolbox="Moechten Sie ToolboxV2 aktualisieren? (ja/nein): "
    if /i "!UpdateToolbox!"=="ja" (
        echo Aktualisiere ToolboxV2...
        pip install ToolboxV2 --upgrade
	goto :EndScript
    )

    set /p ReInstallToolbox="Moechten Sie ToolboxV2 neu aufsetzen? (ja/nein): "
    if /i "!ReInstallToolbox!"=="ja" (
        echo ToolboxV2... wird deinstallirt
        pip uninstall ToolboxV2
    ) else (
        echo beende das programm
	goto :EndScript
    )

    set /p InstallToolbox="Moechten Sie ToolboxV2 jetzt installieren? (ja/nein): "
    if /i "!InstallToolbox!"=="nein" (
        echo beende das programm
	goto :EndScript
    )
)

:: ToolboxV2 installieren
echo Installiere ToolboxV2...
""" + installation_data['core'] + """


ToolboxV2ModuleDir=$(python -c "import os, ToolboxV2; print(os.path.dirname(ToolboxV2.__file__))")
ModsStoDir="$ToolboxV2ModuleDir/mods_sto"
mkdir -p "$ModsStoDir"

urls=""" + str(tuple(urls)) + """"

for url in "${urls[@]}"; do
    echo "Lade herunter und installiere Modul von: $url"
    # Dateiname aus der URL extrahieren
    filename=$(basename "$url")
    # ZIP-Datei herunterladen
    curl -o "$filename" "$url"
    # ZIP-Datei in den mods_sto-Ordner verschieben
    mv "$filename" "$ModsStoDir"
done

:: ToolboxV2 testen
echo Teste ToolboxV2...
ToolboxV2 -v -fg
ToolboxV2 -fg -i main

echo if toolboxv2 wos not found this is no problem simply run 'ToolboxV2 -fg -i main' to complete the setup

echo Installation

:EndScript
echo abgeschlossen.
pause"""
    elif ClientInfos.get('os') == "IOS" or ClientInfos.get('os') == "Android":
        await websocket.send_text("Sorry IOS is Not Nativ Supportet. we start the WebApp Installer")
        await websocket.send_text("ServiceWorker::Active")
        await websocket.close()
        return
    else:
        end = 'sh'
        script = """#!/bin/bash

# Metadaten hinzufügen
Author="Markin Hausmanns"
WebPage="Simplecore.app"
ToolboxInstaller="Toolbox Linux/Mac/Termux Installer"

# Benutzereingaben
echo "Willkommen zum ToolboxV2 Installer."

# Überprüfen der Python-Version und Installation, falls notwendig
PythonFound="NO"
for i in {9..11}; do
    if command -v python3.$i &>/dev/null; then
        PythonFound="YES"
        PythonVersion="3.$i"
        break
    fi
done

if [ "$PythonFound" == "NO" ]; then
    echo "Python 3.11 wird installiert..."
    # Python 3.11 Installer herunterladen und installieren
    curl -o python-3.11.0-amd64.exe https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe
    sudo chmod +x python-3.11.0-amd64.exe
    sudo ./python-3.11.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
    rm python-3.11.0-amd64.exe
else
    echo "Gefundene Python-Version: $PythonVersion"
fi

# Überprüfen, ob ToolboxV2 bereits installiert ist
if pip show ToolboxV2 &>/dev/null; then
    echo "Eine Version von ToolboxV2 ist bereits installiert."
    read -p "Moechten Sie ToolboxV2 aktualisieren? (ja/nein): " UpdateToolbox
    if [ "$UpdateToolbox" == "ja" ]; then
        echo "Aktualisiere ToolboxV2..."
        pip install ToolboxV2 --upgrade
    else
        read -p "Moechten Sie ToolboxV2 neu aufsetzen? (ja/nein): " ReInstallToolbox
        if [ "$ReInstallToolbox" == "ja" ]; then
            echo "ToolboxV2 wird deinstalliert..."
            pip uninstall ToolboxV2
        else
            echo "Beende das Programm."
            exit 0
        fi
    fi
    read -p "Moechten Sie ToolboxV2 jetzt installieren? (ja/nein): " InstallToolbox
    if [ "$InstallToolbox" == "nein" ]; then
        echo "Beende das Programm."
        exit 0
    fi
fi

# ToolboxV2 installieren
# Hier Code einfügen, um ToolboxV2 zu installieren

# Optionale Custom-Flag-Eingabe
echo "Installiere ToolboxV2..."
""" + installation_data['core'] + """


ToolboxV2ModuleDir=$(python -c "import os, ToolboxV2; print(os.path.dirname(ToolboxV2.__file__))")
ModsStoDir="$ToolboxV2ModuleDir/mods_sto"
mkdir -p "$ModsStoDir"

urls=""" + str(tuple(urls)) + """"

for url in "${urls[@]}"; do
    echo "Lade herunter und installiere Modul von: $url"
    # Dateiname aus der URL extrahieren
    filename=$(basename "$url")
    # ZIP-Datei herunterladen
    curl -o "$filename" "$url"
    # ZIP-Datei in den mods_sto-Ordner verschieben
    mv "$filename" "$ModsStoDir"
done

ToolboxV2 -v -fg
ToolboxV2 -fg -i main

echo "Installation abgeschlossen. if toolboxv2 wos not found this is no problem simply run 'ToolboxV2 -fg -i main' to complete the setup"
read -p " END " T"""

    await ws_send("Script crated successfully", websocket=websocket)
    for line in script.split('\n'):
        time.sleep(0.01)
        await websocket.send_text(line)

    print(f"Full script : {script}")
    custom_script_name = f"ReSimpleToolBoxV{target_version}-{str(uuid.uuid4())[:4]}." + end
    await ws_send("saving custom script", websocket=websocket)
    await ws_send("Crating Installation link", websocket=websocket)
    with open(f"./installer/{custom_script_name}", "w") as script_file:
        script_file.write(script)

    await ws_send(f"Link: /installer/download/installer\\{custom_script_name}", websocket=websocket)
    await ws_send("Press the Download button to Download the script", websocket=websocket)
    await websocket.close()


def save_mod_snapshot(app, mod_name, provider=None, tb_state: TbState or None = None):
    if app is None:
        app = get_app(from_="Api.start.installer")
    if provider is None:
        provider = app.config_fh.get_file_handler("provider::")
    if provider is None:
        raise ValueError("No provider specified")
    if tb_state is None:
        tb_state: TbState = get_state_from_app(app, simple_core_hub_url=provider)
    mod_data = tb_state.mods.get(mod_name)
    if mod_data is None:
        mod_data = tb_state.mods.get(mod_name + ".py")

    if mod_data is None:
        app.print(f"Valid ar : {list(tb_state.installable.keys())}")
        return list(tb_state.installable.keys())

    if not os.path.exists("./installer"):
        os.mkdir("./installer")

    json_data = {"Name": mod_name,
                 "mods": [mod_data.url],
                 "runnable": None,
                 "requirements": None,
                 "additional-dirs": None,
                 mod_name: {
                     "version": mod_data.version,
                     "shasum": mod_data.shasum,
                     "provider": mod_data.provider,
                     "url": mod_data.url
                 }}
    installer_path = f"./installer/{mod_name}-installer.json"
    if os.path.exists(installer_path):
        with open(installer_path) as installer_file:
            file_data: dict = json.loads(installer_file.read())
            if len(file_data.get('mods', [])) > 1:
                file_data['mods'].append(mod_data.url)
            file_data[mod_name] = json_data[mod_name]

            json_data = file_data

    with open(installer_path, "w") as installer_file:
        json.dump(json_data, installer_file)

    return json_data


@router.post("/upload-file/")
async def create_upload_file(file: UploadFile):
    # if tb_app.debug:
    # Ensure the target directory exists
    target_dir = "./mods_sto/"
    os.makedirs(target_dir, exist_ok=True)

    # Check if the file has a valid name
    if file.filename.startswith("RST%24") and file.filename.endswith(".zip"):
        try:
            f_name = file.filename.replace("%24", "$").replace("%26", "&").replace("%C2%A7", "§")
            file_path = os.path.join(target_dir, f_name)

            # Save the file in chunks to avoid memory overload
            with open(file_path, 'wb') as f:
                while contents := file.file.read(1024 * 1024):
                    f.write(contents)
            return {"res": f"Successfully uploaded {f_name}"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"There was an error uploading the file: {str(e)}")

        finally:
            file.file.close()

    else:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {file.filename}")


from nicegui import ui


class FileBrowser:
    ALLOWED_DIRECTORIES: set[str] = {"mods_sto", "flows", "static", "apps"}

    def __init__(self, start_dir: str):
        self.static_dir = pathlib.Path(start_dir).resolve()
        self.current_container = None

    def is_path_allowed(self, file_path: pathlib.Path) -> bool:
        """Check if the path is within allowed directories."""
        if not file_path.is_relative_to(self.static_dir):
            return False

        relative_parts = file_path.parts[len(self.static_dir.parts):]
        return any(part in self.ALLOWED_DIRECTORIES for part in relative_parts)

    async def download_file(self, file_path: pathlib.Path) -> None:
        """Handle file download."""
        if not file_path.is_file() or not self.is_path_allowed(file_path):
            ui.notify('Access denied or file not found', type='negative')
            return

        # Use NiceGUI's download function
        await ui.download(str(file_path))

    def refresh_view(self, path: pathlib.Path) -> None:
        """Refresh the file browser view."""
        if self.current_container:
            self.current_container.clear()

        with self.current_container:
            # Add header with current path
            ui.label(f'Current directory: {path.relative_to(self.static_dir)}').classes('text-h6')

            # Add parent directory link if not at root
            if path != self.static_dir and path.parent.is_relative_to(self.static_dir):
                with ui.row().classes('w-full items-center'):
                    ui.button('..', on_click=lambda p=path.parent: self.refresh_view(p)) \
                        .classes('bg-blue-100 px-4 py-2 rounded')

            # List directories first
            for item in sorted(path.iterdir()):
                if not self.is_path_allowed(item):
                    continue

                with ui.row().classes('w-full items-center gap-2'):
                    if item.is_dir():
                        ui.button(f'📁 {item.name}/',
                                  on_click=lambda p=item: self.refresh_view(p)) \
                            .classes('bg-blue-100 px-4 py-2 rounded')
                    else:
                        ui.label(f'📄 {item.name}').classes('flex-grow')
                        ui.button('Download',
                                  on_click=lambda p=item: self.download_file(p)) \
                            .classes('bg-green-100 px-4 py-2 rounded')


def register():
    @ui.page('/installer')
    def main():
        """Main page setup."""
        browser = FileBrowser('.')  # Replace with your actual start directory

        # Create main container
        with (ui.card().
                  classes('w-full max-w-3xl mx-auto p-4').
                  style("background-color: var(--background-color) !important")):
            ui.label('File Browser').classes('text-h4 mb-4')
            browser.current_container = ui.column().classes('w-full gap-2')
            browser.refresh_view(browser.static_dir)

    return main
