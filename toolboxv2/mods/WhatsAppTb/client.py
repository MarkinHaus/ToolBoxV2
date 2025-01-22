import json
from dataclasses import dataclass
from typing import Callable

from whatsapp import WhatsApp

from toolboxv2.mods.WhatsAppTb.utils import ProgressMessenger, emoji_set_thermometer, emoji_set_work_phases
from toolboxv2.mods.WhatsAppTb.server import AppManager


@dataclass
class WhClient:
    messenger: WhatsApp
    disconnect: Callable
    s_callbacks: Callable
    progress_messenger0: ProgressMessenger
    progress_messenger1: ProgressMessenger
    progress_messenger2: ProgressMessenger


def connect(app, phone_number_id):
    key = app.config_fh.one_way_hash(phone_number_id, "WhatsappAppManager",
                                     AppManager.pepper)

    messenger, s_callbacks = AppManager().online(key)

    emoji_set_loading = ["🔄", "🌀", "⏳", "⌛", "🔃"]  # Custom Loading Emoji Set
    progress_messenger0 = ProgressMessenger(messenger, "", emoji_set=emoji_set_loading)
    progress_messenger1 = ProgressMessenger(messenger, "", emoji_set=emoji_set_thermometer)
    progress_messenger2 = ProgressMessenger(messenger, "", emoji_set=emoji_set_work_phases)
    whc = WhClient(messenger=messenger,
                   s_callbacks=s_callbacks,
                   disconnect=AppManager().offline(key),
                   progress_messenger0=progress_messenger0,
                   progress_messenger1=progress_messenger1,
                   progress_messenger2=progress_messenger2)

    # stop_flag = threading.Event()
    # message_id = progress_messenger0.send_initial_message(mode="loading")
    # progress_messenger0.start_loading_in_background(stop_flag)

    # Simulate work, then stop loading
    # time.sleep(10)  # Simulate work duration
    # stop_flag.set()

    return whc
