import os
import signal
import sys
import threading
import time

#from nicegui import ui
from datetime import datetime
from threading import Event, Thread

try:
    from whatsapp import Message, WhatsApp
except ImportError:
    print("NO Whatsapp installed")
    def WhatsApp():
        return None
    def Message():
        return None
import asyncio
import logging

from toolboxv2 import Code, Singleton

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppManager(metaclass=Singleton):
    pepper = "pepper0"

    def __init__(self, start_port: int = 8000, port_range: int = 10, em=None):
        self.instances: dict[str, dict] = {}
        self.start_port = start_port
        self.port_range = port_range
        self.threads: dict[str, Thread] = {}
        self.stop_events: dict[str, Event] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.last_messages: dict[str, datetime] = {}
        self.keys: dict[str, str] = {}
        self.forwarders: dict[str, dict] = {}
        self.runner = lambda :None

        if em is None:
            from toolboxv2 import get_app
            em = get_app().get_mod("EventManager")
        from toolboxv2.mods import EventManager
        self.event_manager: EventManager = em.get_manager()

        # Set up signal handlers for graceful shutdown
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self.signal_handler)
                signal.signal(signal.SIGTERM, self.signal_handler)
        except Exception:
            pass

    def offline(self, instance_id):

        def mark_as_offline():
            self.forwarders[instance_id]['send'] = None
            return 'done'

        return mark_as_offline

    def online(self, instance_id):

        def mark_as_online():
            return self.instances[instance_id]['app']

        def set_callbacks(callback, e_callback=None):
            if callback is not None:
                self.forwarders[instance_id]['send'] = callback
            if e_callback is not None:
                self.forwarders[instance_id]['sende'] = e_callback

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

        self.instances[instance_id] = {
            'app': app_instance,
            'port': port,
            'kwargs': kwargs,
            'phone_number_id': kwargs.get("phone_number_id", {}),
            'retry_count': 0,
            'max_retries': 3,
            'retry_delay': 5
        }
        self.keys[instance_id] = Code.one_way_hash(kwargs.get("phone_number_id", {}).get("key"), "WhatsappAppManager",
                                                   self.pepper)
        self.forwarders[instance_id] = {}

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

        # Create stop event for this instance Error parsing message1:
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
            await self.forwarders[instance_id]['send'](message)

    async def on_event(self, instance_id: str, event):
        """Handle events."""
        logger.info(f"Event from instance {instance_id}: {event}")
        if instance_id in self.forwarders and 'sende' in self.forwarders[instance_id] and self.forwarders[instance_id]['sende'] is not None:
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

    def create_manager_ui(self, start_assistant):
        """Enhanced WhatsApp Manager UI with instance configuration controls"""
        self.runner = start_assistant
        def ui_manager():
            # Track instance states and messages
            original_on_message = self.on_message

            async def enhanced_on_message(instance_id: str, message):
                self.last_messages[instance_id] = datetime.now()
                await original_on_message(instance_id, message)

            self.on_message = enhanced_on_message

            def create_instance_card(instance_id: str):
                """Interactive instance control card"""
                config = self.instances[instance_id]
                with ui.card().classes('w-full p-4 mb-4 bg-gray-50 dark:bg-gray-800').style("background-color: var(--background-color) !important"):
                    # Header Section
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(f'📱 {instance_id}').classes('text-xl font-bold')

                        # Status Indicator
                        ui.label().bind_text_from(
                            self.threads, instance_id,
                            lambda x: 'Running' if x and x.is_alive() else 'Stopped'
                        )

                    # Configuration Display
                    with ui.grid(columns=2).classes('w-full mt-4 gap-2'):

                        ui.label('port:').classes('font-bold')
                        ui.label(config['port'])

                        ui.label('Last Activity:').classes('font-bold')
                        ui.label().bind_text_from(
                            self.last_messages, instance_id,
                            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if x else 'Never'
                        )

                    # Action Controls
                    with ui.row().classes('w-full mt-4 gap-2'):
                        with ui.button(icon='settings', on_click=lambda: edit_dialog.open()).props('flat'):
                            ui.tooltip('Configure')

                        with ui.button(icon='refresh', color='orange',
                                       on_click=lambda: self.restart_instance(instance_id)):
                            ui.tooltip('Restart')

                        with ui.button(icon='stop', color='red',
                                       on_click=lambda: self.stop_instance(instance_id)):
                            ui.tooltip('Stop')

                    # Edit Configuration Dialog
                    with ui.dialog() as edit_dialog, ui.card().classes('p-4 gap-4'):
                        new_key = ui.input('API Key', value=config['phone_number_id'].get('key', ''))
                        new_number = ui.input('Phone Number', value=config['phone_number_id'].get('number', ''))

                        with ui.row().classes('w-full justify-end'):
                            ui.button('Cancel', on_click=edit_dialog.close)
                            ui.button('Save', color='primary', on_click=lambda: (
                                self.update_instance_config(
                                    instance_id,
                                    new_key.value,
                                    new_number.value
                                ),
                                edit_dialog.close()
                            ))

            # Main UI Layout
            with ui.column().classes('w-full max-w-4xl mx-auto p-4'):
                ui.label('WhatsApp Instance Manager').classes('text-2xl font-bold mb-6')

                # Add Instance Section
                with ui.expansion('➕ Add New Instance', icon='add').classes('w-full'):
                    with ui.card().classes('w-full p-4 mt-2'):
                        instance_id = ui.input('Instance ID').classes('w-full')
                        token = ui.input('API Token').classes('w-full')
                        phone_key = ui.input('Phone Number Key').classes('w-full')
                        phone_number = ui.input('Phone Number').classes('w-full')

                        with ui.row().classes('w-full justify-end gap-2'):
                            ui.button('Clear', on_click=lambda: (
                                instance_id.set_value(''),
                                token.set_value(''),
                                phone_key.set_value(''),
                                phone_number.set_value('')
                            ))
                            ui.button('Create', color='positive', on_click=lambda: (
                                self.add_update_instance(
                                    instance_id.value,
                                    token.value,
                                    phone_key.value,
                                    phone_number.value
                                ),
                                instances_container.refresh()
                            ))

                # Instances Display
                instances_container = ui.column().classes('w-full')
                with instances_container:
                    for instance_id in self.instances:
                        create_instance_card(instance_id)

        return ui_manager

    # Add to manager class
    def add_update_instance(self, instance_id, token, phone_key, phone_number):
        """Add or update instance configuration"""
        if instance_id in self.instances:
            self.stop_instance(instance_id)
            del self.instances[instance_id]

        self.add_instance(
            instance_id,
            token=token,
            phone_number_id={
                'key': phone_key,
                'number': phone_number
            },
            verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN")
        )
        self.start_instance(instance_id)

    def update_instance_config(self, instance_id, new_key, new_number):
        """Update existing instance configuration"""
        if instance_id in self.instances:
            self.instances[instance_id]['phone_number_id'] = {
                'key': new_key,
                'number': new_number
            }
            self.restart_instance(instance_id)

    def restart_instance(self, instance_id):
        """Safe restart of instance"""
        self.stop_instance(instance_id)
        self.start_instance(instance_id)

    def stop_instance(self, instance_id):
        """Graceful stop of instance"""
        if instance_id in self.threads:
            self.stop_events[instance_id].set()
            self.threads[instance_id].join(timeout=5)
            del self.threads[instance_id]

    def start_instance(self, instance_id):
        """Start instance thread"""
        print("Starting Istance")

        self.stop_events[instance_id] = threading.Event()
        self.threads[instance_id] = threading.Thread(
            target=self.run_instance,
            args=(instance_id,),
            daemon=True
        )
        self.threads[instance_id].start()
        print("Running starter", self.runner())

