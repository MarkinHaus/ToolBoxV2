import threading
import time

from toolboxv2 import get_app, App, Result, tbef
from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope, EventID
from toolboxv2.utils.system.types import ToolBoxInterfaces
from fastapi import Request

Name = 'cicd'
export = get_app("cicd.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'
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


def downloaded(payload):
    app = get_app("Event saving new web data")
    print(payload)
    app.run_any(tbef.SOCKETMANAGER.RECEIVE_AND_DECOMPRESS_FILE_AS_SERVER, save_path="./web", listening_port=payload)
    return "listening on Port " + payload


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
        source = input("Surece")
        e_id = input("Surece")
        res = ev.trigger_event(EventID.crate(source+":S0", "receive-web-data-s0"))
        print(res)
        app.run_any(tbef.SOCKETMANAGER.SEND_FILE_TO_SEVER, filepath='./web', host='139.162.136.35', port=6568)
