import os
import shutil

from toolboxv2 import get_app, tbef, Spinner
from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope, EventID

Name = 'cicd'
export = get_app("cicd.Export").tb
default_export = export(mod_name=Name)
version = '0.0.3'
spec = ''

"""
Architecture :: State transform via Running Scenarios

:: States ::
 '' dev0
 '' P0/S0
 '' PN/SN
:: Phases ::
-> setup
-> build
-> deploy

:: Scenarios ::
[Ich bin] 'und' [ich,du werde]
 -> meine aktionen um den neuen zustand zu erreichen

 dev0 '' P0/S0
  -> test
  -> build
  -> test
  -> deploy

 P0/S0 '' PN/SN
  -> deploy

"""


# Update Core

def update_core(flags):
    """
    use pipy uploader script
    """


def install_dependencies(web_row_path):
    # Prüfen Sie, ob das Befehlsprogramm vorhanden ist
    os.chdir(web_row_path)

    def command_exists(cmd):
        return shutil.which(cmd) is not None

    # Installieren von Bun, falls nicht vorhanden
    if not command_exists("bun"):
        os.system("npm install -g bun")

        # Installation von fehlenden Modulen
    os.system("bun install")

    # Aktualisieren von Bun
    os.system("bun update")

    # Installation oder Aktualisierung von Abhängigkeiten aus package.json
    os.system("bun install")
    os.chdir("..")


def downloaded(payload: EventID):
    app = get_app("Event saving new web data")
    print("downloaded", payload)
    # if isinstance(payload.payload, str):
    print("payload.payload", payload.payload)
    #    payload.payload = json.loads(payload.payload)
    if 'DESKTOP-CI57V1L' not in payload.get_path()[-1]:
        return "Invalid payload"
    app.run_any(tbef.SOCKETMANAGER.RECEIVE_AND_DECOMPRESS_FILE_AS_SERVER, save_path="./web",
                listening_port=payload.payload['port'])
    print("Don installing modules")
    with Spinner("installing web dependencies"):
        install_dependencies("./web")
    return "Done installing"


def downloaded_mod(payload: EventID):
    app = get_app("Event saving new web data")
    print("downloaded", payload)
    # if isinstance(payload.payload, str):
    print("payload.payload", payload.payload)
    #    payload.payload = json.loads(payload.payload)
    if 'DESKTOP-CI57V1L' not in payload.get_path()[-1]:
        return "Invalid payload"
    app.run_any(tbef.SOCKETMANAGER.RECEIVE_AND_DECOMPRESS_FILE_AS_SERVER, save_path=payload.payload['filename'],
                listening_port=payload.payload['port'])
    return "Done uploading"


def copy_files(src_dir, dest_dir, exclude_dirs):
    for root, dirs, files in os.walk(src_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, os.path.relpath(src_file_path, src_dir))
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copy2(src_file_path, dest_file_path)


@export(mod_name=Name, test=False)
def web_get(app):
    if app is None:
        app = get_app(f"{Name}.web_update")
        # register download event
    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.NAME)
    if ev.identification != "P0|S0":
        ev.identification = "P0|S0"
    dw_event = ev.make_event_from_fuction(downloaded,
                                          "receive-web-data-port-s0",
                                          source_types=SourceTypes.P,
                                          scope=Scope.global_network,
                                          threaded=True)
    ev.register_event(dw_event)


@export(mod_name=Name, test=False)
def mods_get(app):
    if app is None:
        app = get_app(f"{Name}.web_update")
        # register download event
    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.NAME)
    if ev.identification != "P0|S0":
        ev.identification = "P0|S0"
    mods_event = ev.make_event_from_fuction(downloaded_mod,
                                            "receive-mod-module-filename-name-s0",
                                            source_types=SourceTypes.P,
                                            scope=Scope.global_network,
                                            threaded=True)
    ev.register_event(mods_event)


@export(mod_name=Name, state=False, test=False)
def build_edv_blob():
    src_dir = r"C:\Users\Markin\Workspace\ToolBoxV2"
    dest_dir = r"C:\Users\Markin\Workspace\ReSimpleToolBox"
    exclude_dirs = [".config", ".data", ".git", ".github", ".idea", ".info", ".pytest_cache",
                    "build", "dist", "logs", "ToolBoxV2.egg-info", ".editorconfig",
                    ".env", ".gitignore", "auto_uploade_pypi_twine.bat", "Pipfile.lock", "Pipfile",
                    "toolboxv2\\.config", "toolboxv2\\.data", "toolboxv2\\.info",
                    "toolboxv2\\__pycache__", "toolboxv2\\api\\__pycache__", "toolboxv2\\.data", "toolboxv2\\.info",
                    "toolboxv2\\installer\\dist\\UiInstallerTB", "toolboxv2\\mods\\__pycache__", "toolboxv2\\mods_dev",
                    "toolboxv2\\mods_sto", "toolboxv2\\runabel\\__pycache__", "toolboxv2\\tests\\__pycache__",
                    "toolboxv2\\utils\\__pycache__", "toolboxv2\\web", "toolboxv2\\web_row",
                    "toolboxv2\\litellm_uuid.txt",
                    "toolboxv2\\token.pickle", "toolboxv2\\token-0.pickle", "toolboxv2\\token-1.pickle",
                    "toolboxv2\\token-main.pickle",
                    "node_modules", "src-tauri"]
    copy_files(src_dir, dest_dir, exclude_dirs)


@export(mod_name=Name, test=False)
def send_web(app):
    if app is None:
        app = get_app(f"{Name}.web_update")

    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.NAME)
    if ev.identification not in "PN":
        ev.identification = "PN-" + ev.identification
    ev.connect_to_remote()  # add_client_route("P0", ('139.162.136.35', 6568))
    # source = input("Surece: ")
    # e_id = input("evid")
    res = ev.trigger_event(EventID.crate("app.main-localhost:S0", "receive-web-data-s0",
                                         payload={'keyOneTime': 'event',
                                                  'port': 6560}))
    print(res)
    src_dir = "./web"
    dest_dir = "./web_row"
    exclude_dirs = [".idea", "node_modules"]
    copy_files(src_dir, dest_dir, exclude_dirs)
    app.run_any(tbef.SOCKETMANAGER.SEND_FILE_TO_SEVER, filepath='./web_row', host='139.162.136.35', port=6560)


@export(mod_name=Name, test=False)
def send_mod_all_in_one(app):
    if app is None:
        app = get_app(f"{Name}.web_update")
    send_mod_build(app)
    print(send_mod_start_sver_event(app))
    return send_mod_uploade_data(app)


@export(mod_name=Name, test=False)
def send_mod_build(app):
    if app is None:
        app = get_app(f"{Name}.web_update")

    with Spinner("Preparing Mods"):
        for mod_name in app.get_all_mods():
            app.run_any(tbef.CLOUDM.MAKE_INSTALL, module_name=mod_name)


@export(mod_name=Name, test=False)
def send_mod_start_sver_event(app):
    if app is None:
        app = get_app(f"{Name}.web_update")
    ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.NAME)
    if ev.identification != "PN":
        ev.identification = "PN"
    ev.connect_to_remote()

    res = ev.trigger_event(EventID.crate("app.main-localhost:S0", "receive-mod-module-filename-name-s0",
                                         payload={'filename': "./mods_sto", 'port': 6561}))

    return res


@export(mod_name=Name, test=False)
def send_mod_uploade_data(app):
    if app is None:
        app = get_app(f"{Name}.web_update")
    res = app.run_any(tbef.SOCKETMANAGER.SEND_FILE_TO_SEVER, filepath="./mods_sto", host='139.162.136.35', port=6561)
    return res
