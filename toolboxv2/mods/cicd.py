"""
Main task of this Mod :
0) from 0 device to base TB
1) from 0 device to devEnv
2) from devEnv0 to core0 HMR 0 downtime updates
steps :
    0) :
     * shell - setup python and env
     * shell    # on linux and mac tet install python3-venv
     * shell    # using a mini forward script as base installation calld tb taht runs the toolboxv2 in the venv
     * shell - install ex dependency (node, docker, ...)
     * shell    # using winget or apt or co ...
     * shell - install ToolBoxV2 from git or pip installation dependent
     * shell - init auth connection / login to core0
     * shell    # in mini forward script save and crate session + pars to venv
      #0 installation from mod_sto
        - core from pip or git 0 exras (isaa qn dyt ttt diff alert cyr bkt shm auto)
      #0 test local
    1) :
     - installation devModStorage from core0
     - setup files (dev mods)
     - test local Core
     -> install (MOD-NAME)
        install from .yaml file
         specs: name
                version
                dependencies
                extras
   2) :
     -# development (?ISAA?)
        -~ uploade to devModStorage
       -> test local
       -> test remote
     -> move to mod_sto
     -> install on to remote
"""
import os
import shutil
from toolboxv2 import get_app, TBEF, Spinner

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


def downloaded(payload):
    app = get_app("Event saving new web data")
    print("downloaded", payload)
    # if isinstance(payload.payload, str):
    print("payload.payload", payload.payload)
    #    payload.payload = json.loads(payload.payload)
    if 'DESKTOP-CI57V1L' not in payload.get_path()[-1]:
        return "Invalid payload"
    app.run_any(TBEF.SOCKETMANAGER.RECEIVE_AND_DECOMPRESS_FILE_AS_SERVER, save_path="./web",
                listening_port=payload.payload['port'])
    print("Don installing modules")
    with Spinner("installing web dependencies"):
        install_dependencies("./web")
    return "Done installing"


def downloaded_mod(payload):
    app = get_app("Event saving new web data")
    print("downloaded", payload)
    # if isinstance(payload.payload, str):
    print("payload.payload", payload.payload)
    #    payload.payload = json.loads(payload.payload)
    if 'DESKTOP-CI57V1L' not in payload.get_path()[-1]:
        return "Invalid payload"
    app.run_any(TBEF.SOCKETMANAGER.RECEIVE_AND_DECOMPRESS_FILE_AS_SERVER, save_path=payload.payload['filename'],
                listening_port=payload.payload['port'])
    return "Done uploading"


def copy_files(src_dir, dest_dir, exclude_dirs, include=None):
    for root, dirs, files in os.walk(src_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        dirs[:] = [d for d in dirs if include is None or d in include]

        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, os.path.relpath(src_file_path, src_dir))
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copy2(src_file_path, dest_file_path)


@export(mod_name=Name, test=False, helper="init row web data updates Event")
def web_get(app):
    from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope
    if app is None:
        app = get_app(f"{Name}.web_update")
        # register download event
    ev: EventManagerClass = app.run_any(TBEF.EVENTMANAGER.NAME)
    if ev.identification != "P0|S0":
        ev.identification = "P0|S0"
    dw_event = ev.make_event_from_fuction(downloaded,
                                          "receive-web-data-port-s0",
                                          source_types=SourceTypes.P,
                                          scope=Scope.global_network,
                                          threaded=True)
    ev.register_event(dw_event)


@export(mod_name=Name, test=False, helper="init mods updates Event")
def mods_get(app):
    from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope
    if app is None:
        app = get_app(f"{Name}.web_update")
        # register download event
    ev: EventManagerClass = app.run_any(TBEF.EVENTMANAGER.NAME)
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
    include = [
        "./setup.py",
        "requirements_leg.txt",
        "./README.md",
        "./setup.cfg",
        "./MANIFEST.in",
        "./toolboxv2/mods",
        "./toolboxv2/runabel",
        "./toolboxv2/tests",
        "./toolboxv2/web",
        "./toolboxv2/utils",
        "./toolboxv2/__init__.py",
        "./toolboxv2/__main__.py",
        "./toolboxv2/favicon.ico",
        "./toolboxv2/index.html",
        "./toolboxv2/index.js",
        "./toolboxv2/package.json",
        "./toolboxv2/tbState.yaml",
        "./toolboxv2/toolbox.yaml",
        "./toolboxv2/mods_sto",
    ]
    copy_files(src_dir, dest_dir, exclude_dirs, include)


@export(mod_name=Name, test=False)
def send_web(app):
    from toolboxv2.mods.EventManager.module import EventManagerClass, EventID
    if app is None:
        app = get_app(f"{Name}.web_update")

    ev: EventManagerClass = app.run_any(TBEF.EVENTMANAGER.NAME)
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
    app.run_any(TBEF.SOCKETMANAGER.SEND_FILE_TO_SEVER, filepath='./web_row', host='139.162.136.35', port=6560)


@export(mod_name=Name, test=False)
def send_mod_all_in_one(app):
    if app is None:
        app = get_app(f"{Name}.web_update")
    build_mod(app)
    print(send_mod_start_sver_event(app))
    return send_mod_uploade_data(app)


@export(mod_name=Name, test=False)
def build_mod(app, name=None):
    if app is None:
        app = get_app(f"{Name}.web_update")

    with Spinner("Preparing Mods"):
        if name:
            app.run_any(TBEF.CLOUDM.MAKE_INSTALL, module_name=name)
            return
        for mod_name in app.get_all_mods():
            app.run_any(TBEF.CLOUDM.MAKE_INSTALL, module_name=mod_name)


@export(mod_name=Name, test=False)
def send_mod_start_sver_event(app):
    from toolboxv2.mods.EventManager.module import EventManagerClass, EventID
    if app is None:
        app = get_app(f"{Name}.web_update")
    ev: EventManagerClass = app.run_any(TBEF.EVENTMANAGER.NAME)
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
    res = app.run_any(TBEF.SOCKETMANAGER.SEND_FILE_TO_SEVER, filepath="./mods_sto", host='139.162.136.35', port=6561)
    return res


@export(mod_name=Name, test=False, state=False)
def buildDoLive():
    return os.system("docker build --tag 'livetb' .")


@export(mod_name=Name, test=False, state=False)
def runDoLive():
    return os.system("docker run 'livetb'")


@export(mod_name=Name, test=False, state=False)
def doLogs():
    return os.system("docker logs -f 'livetb'")
