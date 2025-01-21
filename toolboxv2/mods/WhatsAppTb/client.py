
import json
from dataclasses import dataclass
from typing import Callable

from whatsapp import WhatsApp

from toolboxv2.mods.EventManager.module import EventManagerClass, SourceTypes, Scope, EventID
from toolboxv2.mods.WhatsAppTb.utils import ProgressMessenger, emoji_set_thermometer, emoji_set_work_phases


@dataclass
class WhClient:
    messenger: WhatsApp
    disconnect: Callable
    progress_messenger0: ProgressMessenger
    progress_messenger1: ProgressMessenger
    progress_messenger2: ProgressMessenger


async def connect(app, callback, **kwargs):
    skey = app.config_fh.generate_symmetric_key()
    messenger = WhatsApp(**kwargs)

    async def disconnect():
        ev: EventManagerClass = app.get_mod("EventManager").get_manager()

        return await ev.trigger_event(
            EventID.crate(f"{source_id}:S0", "whatsapp-disconnection-point", payload={'key': key}))

    # Progress Messenger Configuration
    emoji_set_loading = ["🔄", "🌀", "⏳", "⌛", "🔃"]  # Custom Loading Emoji Set
    progress_messenger0 = ProgressMessenger(messenger, "", emoji_set=emoji_set_loading)
    progress_messenger1 = ProgressMessenger(messenger, "", emoji_set=emoji_set_thermometer)
    progress_messenger2 = ProgressMessenger(messenger, "", emoji_set=emoji_set_work_phases)
    whc = WhClient(messenger=messenger, disconnect=disconnect,
                   progress_messenger0=progress_messenger0,
                   progress_messenger1=progress_messenger1,
                   progress_messenger2=progress_messenger2)

    async def on_massage(payload):
        data = payload.payload.get('data')
        print(data)
        data = app.config_fh.decrypt_symmetric(data, skey, to_str=True)
        print(data)
        data = json.loads(data)
        print(data)
        await callback(whc, data)

    ev: EventManagerClass = app.get_mod("EventManager", spec="client").get_manager()
    ev.identification = "client"
    service_event = ev.make_event_from_fuction(on_massage,
                                               "on-message",
                                               source_types=SourceTypes.AP,
                                               scope=Scope.global_network,
                                               threaded=True)
    await ev.register_event(service_event)
    await ev.identity_post_setter()

    await ev.connect_to_remote(host="localhost")

    from .server import AppManager
    key = app.config_fh.one_way_hash(kwargs.get("phone_number_id", {}).get("key"), "WhatsappAppManager",
                                     AppManager.pepper)

    source_id = f"client.{app.id}"
    res = await ev.trigger_event(EventID.crate(f"{source_id}:S0", "whatsapp-connection-point",
                                               payload={
                                                   'key': key,
                                                   'source_id': source_id,
                                                   'sKey': skey}))

    print(res)

    # stop_flag = threading.Event()
    # message_id = progress_messenger0.send_initial_message(mode="loading")
    # progress_messenger0.start_loading_in_background(stop_flag)

    # Simulate work, then stop loading
    # time.sleep(10)  # Simulate work duration
    # stop_flag.set()

    return whc
