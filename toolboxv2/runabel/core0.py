from datetime import datetime

from toolboxv2 import App, AppArgs, tbef, ToolBox_over
from toolboxv2.utils.extras.blobs import BlobFile

NAME = 'core0'


def save_db_to_blob(app):
    stamp = datetime.now().strftime('%m#%d#%Y.%H:%M:%S')
    app.print(f"Saving DB Data {stamp}")
    db_data = app.run_any(tbef.DB.GET, quary='*', get_results=True)
    if db_data.is_error():
        app.run_any(tbef.DB.EDIT_CLI, mode='RR')
        db_data = app.run_any(tbef.DB.GET, quary='*', get_results=True)
    if db_data.is_error():
        app.print("Error getting Data")
        return
    with BlobFile(f"DB#Backup/{ToolBox_over}/{stamp}/data.row", 'w') as f:
        f.write(db_data.get())
    app.print(f"Data Saved volumen : {len(db_data.get())}")


async def run(app: App, args: AppArgs):
    import schedule
    app.print("Starting core 0")

    # app.run_any(tbef.SCHEDULERMANAGER.INIT)
    await app.a_run_any(tbef.SCHEDULERMANAGER.ADD,
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
    await app.a_idle()


if __name__ == "__main__":
    import os

    # os.system(f"toolboxv2 --test --debug")
    os.system(f"tb -bgr -p 42869 -n core0 -l -m {NAME}")
