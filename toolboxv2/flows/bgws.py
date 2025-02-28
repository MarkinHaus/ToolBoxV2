import asyncio

import requests

from toolboxv2 import AppArgs, App

NAME = 'bgws'


async def run(app: App, app_args:AppArgs):
    app.print("Running...")
    await asyncio.gather(*[
        app.get_mod("WebSocketManager").start_server(host=app_args.host, port=app_args.port-100),
        app.daemon_app.connect(app)
    ])
