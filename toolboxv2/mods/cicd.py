import json
import threading
import time

from toolboxv2 import get_app, App, Result, tbef, Spinner
from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope, EventID
from toolboxv2.utils.system.types import ToolBoxInterfaces
from fastapi import Request
import os
import shutil

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


def copy_files(src_dir, dest_dir, exclude_dirs):
    for root, dirs, files in os.walk(src_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, os.path.relpath(src_file_path, src_dir))
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copy2(src_file_path, dest_file_path)


@export(mod_name=Name)
def web_update(app, t):
    if app is None:
        app = get_app(f"{Name}.web_update")
    if 's' in t:
        # register download event
        ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.NAME)
        ev.identification = "P0|S0"
        dw_event = ev.make_event_from_fuction(downloaded, "receive-web-data-s0",
                                              source_types=SourceTypes.P,
                                              scope=Scope.global_network,
                                              threaded=True)
        ev.register_event(dw_event)
    else:
        ev: EventManagerClass = app.run_any(tbef.EVENTMANAGER.NAME)
        ev.identification = "PN"
        ev.connect_to_remote()  # add_client_route("P0", ('139.162.136.35', 6568))
        # source = input("Surece: ")
        # e_id = input("evid")
        res = ev.trigger_event(EventID.crate("app.main-localhost:S0", "receive-web-data-s0",
                                             payload={'keyOneTime': 'event',
                                                      'port': 6560}))
        print(res)
        src_dir = "./web"
        dest_dir = "./web_row"
        exclude_dirs = [".idea", "node_modules", "src-tauri"]
        copy_files(src_dir, dest_dir, exclude_dirs)
        app.run_any(tbef.SOCKETMANAGER.SEND_FILE_TO_SEVER, filepath='./web_row', host='139.162.136.35', port=6560)
