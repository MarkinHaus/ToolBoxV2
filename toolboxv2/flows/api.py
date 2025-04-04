import asyncio

from toolboxv2 import App
from toolboxv2.utils.system.api import main_api_runner

NAME = 'api'

async def run(app:App, app_args):
    main_api_runner(app.debug)

    async def helper():
        if hasattr(app, 'daemon_app'):
            await app.daemon_app.connect(app)
        else:
            await app.a_idle()

    app.get_mod("DB").initialize_database()
    app.sprint("DB initialized")
    print(f"Running Ws on", app_args.port-100)
    await asyncio.gather(*[
        app.get_mod("isaa"),
        app.load_all_mods_in_file(),

        app.get_mod("WebSocketManager").start_server(host=app_args.host, port=app_args.port-100),
        helper()
    ])
