import json

from nicegui import ui
from datetime import datetime
from typing import Dict
from threading import Thread, Event
import time
import signal
import sys
from whatsapp import WhatsApp, Message
from toolboxv2 import Singleton, Code
import asyncio
import logging

from toolboxv2.mods.EventManager.module import EventID, SourceTypes, Scope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppManager(metaclass=Singleton):
    pepper = "pepper0"

    def __init__(self, start_port: int = 8000, port_range: int = 10, em=None):
        self.instances: Dict[str, Dict] = {}
        self.start_port = start_port
        self.port_range = port_range
        self.threads: Dict[str, Thread] = {}
        self.stop_events: Dict[str, Event] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.last_messages: Dict[str, datetime] = {}
        self.keys: Dict[str, str] = {}
        self.forwarders: Dict[str, Dict] = {}

        if em is None:
            from toolboxv2 import get_app
            em = get_app().get_mod("EventManager")
        from toolboxv2.mods import EventManager
        self.event_manager: EventManager = em.get_manager()

        # Set up signal handlers for graceful shutdown
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
        except Exception:
            pass

    def offline(self, key):

        def mark_as_offline():
            if key is None:
                return "Invalid key"
            if key not in self.keys:
                return "Invalid key 404"
            self.forwarders[self.keys[key]]['send'] = None
            return 'done'

        return mark_as_offline

    def online(self, key):

        def mark_as_online():
            if key is None:
                return "Invalid key"
            if key not in self.keys:
                return "Invalid key 404"
            return self.instances[self.keys[key]]

        def set_callbacks(callback, e_callback=None):
            if callback is not None:
                self.forwarders[self.keys[key]]['send'] = callback
            if e_callback is not None:
                self.forwarders[self.keys[key]]['sende'] = e_callback
            self.forwarders[self.keys[key]]['send'](Message(id=0, instance=None, content="test", to="0000", data={'none':'none'}))

        return mark_as_online(), set_callbacks

    def get_next_available_port(self) -> int:
        """Find the next available port in the range."""
        used_ports = {instance['port'] for instance in self.instances.values()}
        for port in range(self.start_port, self.start_port + self.port_range):
            if port not in used_ports:
                return port
        raise RuntimeError("No available ports in range")

    def add_instance(self, instance_id: str, **kwargs):
        """
        Add a new app instance to the manager with automatic port assignment.
        """
        if instance_id in self.instances:
            raise ValueError(f"Instance {instance_id} already exists")

        port = self.get_next_available_port()
        app_instance = WhatsApp(**kwargs)
        print("app_instance", app_instance.verify_token)
        self.instances[instance_id] = {
            'app': app_instance,
            'port': port,
            'kwargs': kwargs,
            'retry_count': 0,
            'max_retries': 3,
            'retry_delay': 5
        }
        self.keys[instance_id] = Code.one_way_hash(kwargs.get("phone_number_id", {}).get("key"), "WhatsappAppManager",
                                                   self.pepper)

        # Set up message handlers
        @app_instance.on_message
        async def message_handler(message):
            await self.on_message(instance_id, message)

        @app_instance.on_event
        async def event_handler(event):
            await self.on_event(instance_id, event)

        @app_instance.on_verification
        async def verification_handler(verification):
            await self.on_verification(instance_id, verification)

        # Create stop event for this instance
        self.stop_events[instance_id] = Event()

    def run_instance(self, instance_id: str):
        """Run a single instance in a separate thread with error handling and automatic restart."""
        instance_data = self.instances[instance_id]
        stop_event = self.stop_events[instance_id]

        while not stop_event.is_set():
            try:
                logger.info(f"Starting instance {instance_id} on port {instance_data['port']}")
                instance_data['app'].run(host='0.0.0.0', port=instance_data['port'])

            except Exception as e:
                logger.error(f"Error in instance {instance_id}: {str(e)}")
                instance_data['retry_count'] += 1

                if instance_data['retry_count'] > instance_data['max_retries']:
                    logger.error(f"Max retries exceeded for instance {instance_id}")
                    break

                logger.info(f"Restarting instance {instance_id} in {instance_data['retry_delay']} seconds...")
                time.sleep(instance_data['retry_delay'])

                # Recreate the instance
                instance_data['app'] = WhatsApp(**instance_data['kwargs'])
                continue

    async def on_message(self, instance_id: str, message: Message):
        """Handle and forward incoming messages."""
        logger.info(f"Message from instance {instance_id}: {message}")
        if instance_id in self.forwarders and 'send' in self.forwarders[instance_id]:
            self.forwarders[instance_id]['send'](message)

    async def on_event(self, instance_id: str, event):
        """Handle events."""
        logger.info(f"Event from instance {instance_id}: {event}")
        if instance_id in self.forwarders and 'sende' in self.forwarders[instance_id]:
            self.forwarders[instance_id]['sende'](event)

    async def on_verification(self, instance_id: str, verification):
        """Handle verification events."""
        logger.info(f"Verification from instance {instance_id}: {verification}")

    def run_all_instances(self):
        """Start all instances in separate daemon threads."""
        # Start message forwarder

        # Start all instances
        for instance_id in self.instances:
            thread = Thread(
                target=self.run_instance,
                args=(instance_id,),
                daemon=True,
                name=f"WhatsApp-{instance_id}"
            )
            self.threads[instance_id] = thread
            thread.start()

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, stopping all instances...")
        self.stop_all_instances()
        sys.exit(0)

    def stop_all_instances(self):
        """Stop all running instances gracefully."""
        for instance_id in self.stop_events:
            self.stop_events[instance_id].set()

        for thread in self.threads.values():
            thread.join(timeout=5)

    def create_manager_ui(self):
        """Create a NiceGUI interface for the WhatsApp App Manager."""

        def ui_manager():
            # Add last message timestamp tracking to manager
            # Enhance message handler to track timestamps
            original_on_message = self.on_message

            async def enhanced_on_message(instance_id: str, message):
                self.last_messages[instance_id] = datetime.now()
                await original_on_message(instance_id, message)

            self.on_message = enhanced_on_message

            def create_instance_card(instance_id: str):
                """Create a card for a single WhatsApp instance."""
                with ui.card().classes('w-full p-4 mb-4'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(f'Instance: {instance_id}').classes('text-xl font-bold')
                        status_label = ui.label('Status: Running').classes('text-green-500')

                        async def update_status():
                            while True:
                                is_running = (
                                    instance_id in self.threads
                                    and self.threads[instance_id].is_alive()
                                )
                                status_label.text = f'Status: {"Running" if is_running else "Stopped"}'
                                status_label.classes(
                                    'text-green-500' if is_running else 'text-red-500',
                                )
                                await asyncio.sleep(1)

                        ui.timer(0.1, lambda: asyncio.create_task(update_status()))

                    with ui.row().classes('w-full mt-2'):
                        ui.label(f'Port: {self.instances[instance_id]["port"]}')
                        last_msg = ui.label('Last Message: Never')

                        async def update_last_message():
                            while True:
                                timestamp = self.last_messages.get(instance_id)
                                last_msg.text = (
                                    f'Last Message: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
                                    if timestamp else 'Last Message: Never'
                                )
                                await asyncio.sleep(1)

                        ui.timer(0.1, lambda: asyncio.create_task(update_last_message()))

                    with ui.row().classes('w-full mt-4 gap-2'):
                        async def restart_instance():
                            if instance_id in self.threads:
                                self.stop_events[instance_id].set()
                                self.threads[instance_id].join(timeout=5)
                            thread = Thread(
                                target=self.run_instance,
                                args=(instance_id,),
                                daemon=True,
                                name=f"WhatsApp-{instance_id}"
                            )
                            self.threads[instance_id] = thread
                            thread.start()

                        async def stop_instance():
                            if instance_id in self.stop_events:
                                self.stop_events[instance_id].set()
                                if instance_id in self.threads:
                                    self.threads[instance_id].join(timeout=5)

                        ui.button('Restart', on_click=restart_instance).props('color=warning')
                        ui.button('Stop', on_click=stop_instance).props('color=negative')

            # Main UI Layout
            with ui.column().classes('w-full max-w-3xl mx-auto p-4'):
                ui.label('WhatsApp Instance Manager').classes('text-2xl font-bold mb-6')

                # Instance Creation Form
                with ui.card().classes('w-full p-4 mb-6'):
                    ui.label('Add New Instance').classes('text-xl font-bold mb-4')

                    instance_id = ui.input('Instance ID').classes('w-full')
                    token = ui.input('Token').classes('w-full')
                    phone_id = ui.input('Phone Number ID').classes('w-full')

                    async def add_new_instance():
                        try:
                            self.add_instance(
                                instance_id.value,
                                token=token.value,
                                phone_number_id={"key": phone_id.value}
                            )
                            thread = Thread(
                                target=self.run_instance,
                                args=(instance_id.value,),
                                daemon=True,
                                name=f"WhatsApp-{instance_id.value}"
                            )
                            self.threads[instance_id.value] = thread
                            thread.start()

                            # Refresh instances display
                            instances_container.clear()
                            for inst_id in self.instances:
                                create_instance_card(inst_id)

                            # Clear form
                            instance_id.value = ''
                            token.value = ''
                            phone_id.value = ''

                        except Exception as e:
                            ui.notify(f'Error adding instance: {str(e)}', type='negative')

                    ui.button('Add Instance', on_click=add_new_instance).props('color=positive')

                # Instances Display
                instances_container = ui.column().classes('w-full')
                for instance_id in self.instances:
                    create_instance_card(instance_id)

        return ui_manager  # Return the ui object for registration
