import asyncio

from toolboxv2 import App, AppArgs

NAME = 'bgws'


async def run(app: App, app_args:AppArgs):
    from toolboxv2.utils.extras.blobs import BlobFile
    app.print("Running...")


    with BlobFile("HRMODS/dev", mode='r') as f:
        modules = f.read_json().get("modules", [])
    for mods in modules:
        app.print(f"ADDING :  {mods}")
        app.watch_mod(mods)

    app.get_mod("DB").initialize_database()
    app.sprint("DB initialized")

    await asyncio.gather(*[
        app.get_mod("isaa"),
        app.load_all_mods_in_file(),

        app.get_mod("WebSocketManager").start_server(host=app_args.host, port=app_args.port-100),
        app.daemon_app.connect(app)
    ])
