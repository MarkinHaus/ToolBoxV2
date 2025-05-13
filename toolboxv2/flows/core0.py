from datetime import datetime

from toolboxv2 import TBEF, App, AppArgs, ToolBox_over, get_app
from toolboxv2.utils.extras.blobs import BlobFile

try:
    from toolboxv2.mods.EventManager.module import (
        EventID,
        EventManagerClass,
        Scope,
        SourceTypes,
    )
    Online = True
except ImportError as e:
    SocketType = None
    EventManagerClass, SourceTypes, Scope = None, None, None
    Online = False
    print(f"Chat runner is missing modules Pleas install {e}")

NAME = 'core0'


def save_db_to_blob(app):
    stamp = datetime.now().strftime('%m#%d#%Y.%H:%M:%S')
    app.print(f"Saving DB Data {stamp}")
    db_data = app.run_any(TBEF.DB.GET, quary='*', get_results=True)
    if db_data.is_error():
        app.run_any(TBEF.DB.EDIT_CLI, mode='RR')
        db_data = app.run_any(TBEF.DB.GET, quary='*', get_results=True)
    if db_data.is_error():
        app.print("Error getting Data")
        return
    with BlobFile(f"DB#Backup/{ToolBox_over}/{stamp}/data.row", 'w') as f:
        f.write(db_data.get())
    app.print(f"Data Saved volumen : {len(db_data.get())}")


async def get_connection_point(payload):
    app = get_app("Event get_connection_point")
    payload_key = payload.payload['key']
    ev: EventManagerClass = app.get_mod("EventManager").get_manager()
    try:
        rute_data = list(ev.routes.keys())[list(ev.routes.values()).index(payload_key)]
        return rute_data
    except ValueError:
        return "Invalid payload"


async def run(app: App, args: AppArgs):
    import schedule
    app.print("Starting core 0")

    # app.run_any(TBEF.SCHEDULERMANAGER.INIT)
    await app.a_run_any(TBEF.SCHEDULERMANAGER.ADD,
                        job_data={
                            "job_id": "system#Backup#Database",
                            "second": 0,
                            "func": None,
                            "job": schedule.every(2).days.at("04:00").do(save_db_to_blob, app),
                            "time_passer": None,
                            "object_name": "save_db_to_blob",
                            "receive_job": False,
                            "save": True,
                            "max_live": False,
                            "args": (app,)
                        })
    ev: EventManagerClass = app.get_mod("EventManager").get_manager()
    service_event = ev.make_event_from_fuction(get_connection_point,
                                               "get-connection-point",
                                               source_types=SourceTypes.AP,
                                               scope=Scope.global_network,
                                               threaded=True)
    await ev.register_event(service_event)
    app.print("Service P2P Online")
    await app.run_flows("cli")


if __name__ == "__main__":
    import os

    # os.system(f"toolboxv2 --test --debug")
    os.system(f"tb -bgr -p 42869 -n core0 -l -m {NAME}")
