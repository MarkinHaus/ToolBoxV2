import asyncio
import atexit
import os
import time

from ..extras.Style import Style
from .tb_logger import get_logger
from .types import AppArgs, AppType

registered_apps: list[AppType | None] = [None]


def override_main_app(app):
    global registered_apps
    if registered_apps[0] is not None:
        if time.time() - registered_apps[0].called_exit[1] > 30:
            raise PermissionError("Permission denied because of overtime fuction override_main_app sud only be called "
                                  f"once and ontime overtime {time.time() - registered_apps[0].called_exit[1]}")

    registered_apps[0] = app

    return registered_apps[0]



def get_app(from_=None, name=None, args=AppArgs().default(), app_con=None, sync=False) -> AppType:
    global registered_apps
    # name = None
    # inspect caller
    # from inspect import getouterframes, currentframe
    # print(f"get app requested from: {getouterframes(currentframe(), 2)[1].filename}::{getouterframes(currentframe(), 2)[1].lineno}")

    # print(f"get app requested from: {from_} withe name: {name}")
    logger = get_logger()
    logger.info(Style.GREYBG(f"get app requested from: {from_}"))
    if registered_apps[0] is not None:
        return registered_apps[0]

    if app_con is None:
        try:
            from ... import App
        except ImportError:
            try:
                from ..toolbox import App
            except ImportError:
                from toolboxv2 import App

        app_con = App
    app = app_con(name, args=args) if name else app_con()
    logger.info(Style.Bold(f"App instance, returned ID: {app.id}"))

    registered_apps[0] = app
    return app


async def a_get_proxy_app(app, host="localhost", port=6587, key="remote@root", timeout=12):
    from os import getenv

    from toolboxv2.utils.proxy.proxy_app import ProxyApp
    app.print("INIT PROXY APP")
    _ = await ProxyApp(app, host, port, timeout=timeout)
    time.sleep(0.2)
    _.print("PROXY APP START VERIFY")
    await _.verify({'key': getenv('TB_R_KEY', key)})
    time.sleep(0.1)
    _.print("PROXY APP CONNECTED")
    return override_main_app(_)


@atexit.register
def save_closing_app():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(a_save_closing_app())
    except RuntimeError:
        if registered_apps[0] is None:
            return
        registered_apps[0].exit()


async def a_save_closing_app():
    if registered_apps[0] is None:
        return

    app = registered_apps[0]
    if app.start_dir != "test":
        os.chdir(app.start_dir)

    pid_file = f"{app.start_dir}\\.info\\{app.args_sto.modi}-{app.REFIX}.pid"
    if os.path.exists(pid_file):
        os.remove(pid_file)

    if not app.alive:
        await app.a_exit()
        app.print(Style.Bold(Style.ITALIC("- end -")))
        return

    if not app.called_exit[0] and time.time() - app.called_exit[1] < 8:
        await app.a_exit()
        app.print(Style.Bold(Style.ITALIC("- Fast exit -")))
        return

    if not app.called_exit[0]:
        app.print(Style.Bold(Style.ITALIC("- auto exit -")))
        await app.a_exit()

    if app.called_exit[0] and time.time() - app.called_exit[1] > 15:
        app.print(Style.Bold(Style.ITALIC(f"- zombie sice|{time.time() - app.called_exit[1]:.2f}s kill -")))
        await app.a_exit()

    app.print(Style.Bold(Style.ITALIC("- completed -")))
    registered_apps[0] = None
