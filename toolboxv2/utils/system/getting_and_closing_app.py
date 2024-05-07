import asyncio
import atexit
import time
from typing import List, Optional

from .tb_logger import get_logger
from .types import AppArgs, AppType

from ..extras.Style import Style

registered_apps: List[Optional[AppType]] = [None]


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
    # print(f"get app requested from: {from_}")
    logger = get_logger()
    logger.info(Style.GREYBG(f"get app requested from: {from_}"))
    if registered_apps[0] is not None:
        return registered_apps[0]

    if app_con is None:
        from ... import App
        app_con = App
    if name:
        app = app_con(name, args=args)
    else:
        app = app_con()
    logger.info(Style.Bold(f"App instance, returned ID: {app.id}"))
    if from_ != "InitialStartUp" and not hasattr(app, 'loop') and not sync:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        print("ADDET A LOOP")
        app.loop = loop
    registered_apps[0] = app
    return app


@atexit.register
def save_closing_app():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    loop.run_until_complete(a_save_closing_app())


async def a_save_closing_app():
    if registered_apps[0] is None:
        return

    app = registered_apps[0]
    if not app.alive:
        await app.a_exit()
        app.print(Style.Bold(Style.ITALIC("- end -")))
        return

    if not app.called_exit[0] and time.time() - app.called_exit[1] < 8:
        await app.a_exit()
        app.print(Style.Bold(Style.ITALIC("- Killing a kid is not ok -")))
        return

    if not app.called_exit[0]:
        app.print(Style.Bold(Style.ITALIC("- auto exit -")))
        await app.a_exit()

    if app.called_exit[0] and time.time() - app.called_exit[1] > 15:
        app.print(Style.Bold(Style.ITALIC(f"- zombie sice|{time.time() - app.called_exit[1]:.2f}s kill -")))
        await app.a_exit()

    if hasattr(app, 'loop'):
        if not app.loop.is_closed():
            app.loop.close()

    app.print(Style.Bold(Style.ITALIC("- completed -")))
    registered_apps[0] = None
