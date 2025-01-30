from urllib.request import Request

import asyncio
import os

from toolboxv2 import App, get_app
from toolboxv2.mods.WhatsAppTb.client import runner

from toolboxv2.mods.WhatsAppTb.server import AppManager

Name = "WhatsAppTb"
version = "0.0.1"
managers = []


@get_app().tb(mod_name="WhatsAppTb", name="exit", exit_f=True)
def on_exit(*a):
    if len(managers) < 1:
        return
    managers[0].stop_all_instances()


@get_app().tb(mod_name="WhatsAppTb", name="init", initial=True)
def on_start(app: App):
    if app is None:
        app = get_app("init-whatsapp")

    if os.getenv("WHATSAPP_API_TOKEN") is None or os.getenv("WHATSAPP_PHONE_NUMBER_ID") is None:
        print("WhatsAppTb No main instance init pleas set envs")
    print("WhatsAppTb1")
    if len(managers) > 1:
        return
    print("WhatsAppTb2")
    manager = AppManager(start_port=8050, port_range=10, em=app.get_mod("EventManager"))
    print("WhatsAppTb2.1")
    managers.append(manager)
    print("WhatsAppTb2.2")
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "0000d004567cs890d987")
    print("WhatsAppTb2.3", verify_token)
    manager.add_instance(
        "main",
        token=os.getenv("WHATSAPP_API_TOKEN"),
        phone_number_id={"key": os.getenv("WHATSAPP_PHONE_NUMBER_ID")},
        verify_token=verify_token
    )
    print("WhatsAppTb3")
    def start(WHATSAPP_PHONE_NUMBER_ID=None, RECIPIENT_PHONE=None):
        waa = runner(app, os.getenv("WHATSAPP_PHONE_NUMBER_ID"), to=os.getenv("RECIPIENT_PHONE"))
        print("WhatsAppTb6")
        return "Online"
    try:
        from toolboxv2.mods.FastApi.fast_nice import NiceGUIManager
        nm = NiceGUIManager(None,...)
        if nm.init:
            nm.register_gui("WhatsAppTb", manager.create_manager_ui(start_assistant=start))
            print("WhatsAppTb4")
        else:
            print("No ui")
    except ImportError:
        print("No ui")

    print("WhatsAppTb5 ")
