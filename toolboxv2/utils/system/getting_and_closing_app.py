import asyncio
import atexit
import logging
import os
import sys
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

    # Fast-path: registered app short-circuits BEFORE any logging/frames cost.
    # (get_app is called dozens of times per agent run - logging must never
    #  be paid on the hot path.)
    _app = registered_apps[0]
    if _app is not None:
        logger = get_logger()
        if logger.isEnabledFor(logging.DEBUG):
            if from_ is None:
                f = sys._getframe(1)  # 1 frame walk instead of getouterframes x2
                from_ = f"{f.f_code.co_filename}::{f.f_lineno}"
            logger.info(Style.GREYBG(f"get app requested from: {from_}"))
        return _app

    # Cold path: no app registered yet - full logging + boot.
    logger = get_logger()
    caller_src = "set debug mod"
    if from_ is None and logger.isEnabledFor(logging.DEBUG):
        f = sys._getframe(1)
        caller_src = f"{f.f_code.co_filename}::{f.f_lineno}"
    logger.info(Style.GREYBG(f"get app requested from: {from_ if from_ is not None else caller_src}"))

    # Fail-safe: someone called get_app() straight from their own code/menu
    # before any onboarding ran. If no manifest exists, prepare 'mini headless'
    # (secret + offline env + mini manifest) so the App boots cleanly and login
    # works. Non-recursive: prepare only, this function builds the App below.
    try:
        from toolboxv2.init_onboarding import prepare_mini_failsafe
        prepare_mini_failsafe()
    except Exception:
        pass  # never block app creation on onboarding prep

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
    # Coroutine erst hier erstellen, damit sie nicht vorzeitig "verloren" geht
    if "unittest" in sys.modules or os.environ.get("TOOLBOX_TESTING") == "true" or "test" in sys.argv:
        return

    if registered_apps[0] is not None:
        registered_apps[0].exit()

async def a_save_closing_app():
    if registered_apps[0] is None:
        return

    app = registered_apps[0]
    if app.start_dir != "test":
        os.chdir(app.start_dir)

    pid_file = f"{app.start_dir}\\.info\\pids\\{app.args_sto.modi}-{app.REFIX}.pid"
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
