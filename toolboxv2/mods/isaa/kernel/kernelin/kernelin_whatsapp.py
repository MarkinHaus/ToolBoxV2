"""
ProA Kernel WhatsApp Interface
================================

Production-ready WhatsApp interface for the Enhanced ProA Kernel with:
- Auto-persistence (save/load on start/stop)
- Full media support (images, documents, audio, video)
- Message formatting (bold, italic, code)
- Typing indicators
- Read receipts
- Contact management
"""

import asyncio
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter

try:
    from whatsapp import Message, WhatsApp
except ImportError:
    print("⚠️ WhatsApp library not installed. Install with: pip install whatsapp")
    WhatsApp = None
    Message = None

from toolboxv2 import App, get_app
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import Signal as KernelSignal, SignalType, KernelConfig


class WhatsAppOutputRouter:
    """WhatsApp-specific output router with media support"""

    def __init__(self, messenger: WhatsApp):
        self.messenger = messenger
        self.active_chats: Dict[str, dict] = {}  # phone_number -> chat info

    def _format_message(self, text: str, bold: bool = False, italic: bool = False, code: bool = False) -> str:
        """Format WhatsApp message with markdown"""
        if bold:
            text = f"*{text}*"
        if italic:
            text = f"_{text}_"
        if code:
            text = f"```{text}```"
        return text

    async def send_response(self, user_id: str, content: str, metadata: dict = None):
        """Send agent response to WhatsApp user"""
        try:
            # Send typing indicator
            await self.messenger.send_typing(user_id)

            # Format and send message
            formatted_content = self._format_message(content, bold=False)
            await self.messenger.send_message(user_id, formatted_content)

            # Mark as read
            if metadata and metadata.get("message_id"):
                await self.messenger.mark_as_read(metadata["message_id"])

        except Exception as e:
            print(f"❌ Error sending WhatsApp response: {e}")

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Send notification to WhatsApp user"""
        try:
            # High priority notifications are bold
            formatted_content = self._format_message(
                f"🔔 {content}",
                bold=(priority >= 7)
            )
            await self.messenger.send_message(user_id, formatted_content)

        except Exception as e:
            print(f"❌ Error sending WhatsApp notification: {e}")

    async def send_error(self, user_id: str, error: str, metadata: dict = None):
        """Send error message to WhatsApp user"""
        try:
            formatted_error = self._format_message(f"❌ Error: {error}", italic=True)
            await self.messenger.send_message(user_id, formatted_error)

        except Exception as e:
            print(f"❌ Error sending WhatsApp error: {e}")

    async def send_media(self, user_id: str, media_path: str, media_type: str = "document", caption: str = None):
        """
        Send media to WhatsApp user

        Args:
            user_id: Phone number
            media_path: Path to media file
            media_type: Type of media (document, image, audio, video)
            caption: Optional caption for media
        """
        try:
            if media_type == "image":
                await self.messenger.send_image(user_id, media_path, caption=caption)
            elif media_type == "audio":
                await self.messenger.send_audio(user_id, media_path)
            elif media_type == "video":
                await self.messenger.send_video(user_id, media_path, caption=caption)
            else:  # document
                await self.messenger.send_document(user_id, media_path, caption=caption)

        except Exception as e:
            print(f"❌ Error sending WhatsApp media: {e}")


class WhatsAppKernel:
    """WhatsApp-based ProA Kernel with auto-persistence and media support"""

    def __init__(
        self,
        agent,
        app: App,
        phone_number_id: str,
        token: str,
        instance_id: str = "default",
        auto_save_interval: int = 300
    ):
        """
        Initialize WhatsApp Kernel

        Args:
            agent: FlowAgent instance
            app: ToolBoxV2 App instance
            phone_number_id: WhatsApp Business phone number ID
            token: WhatsApp API token
            instance_id: Instance identifier for multi-instance support
            auto_save_interval: Auto-save interval in seconds (default: 5 minutes)
        """
        if WhatsApp is None:
            raise ImportError("WhatsApp library not installed")

        self.agent = agent
        self.app = app
        self.instance_id = instance_id
        self.auto_save_interval = auto_save_interval
        self.running = False
        self.save_path = self._get_save_path()

        # Initialize WhatsApp messenger
        self.messenger = WhatsApp(phone_number_id=phone_number_id, token=token)

        # Initialize kernel with WhatsApp output router
        config = KernelConfig(
            heartbeat_interval=30.0,
            idle_threshold=600.0,  # 10 minutes for WhatsApp
            proactive_cooldown=120.0,  # 2 minutes cooldown
            max_proactive_per_hour=5  # Less proactive on WhatsApp
        )

        self.output_router = WhatsAppOutputRouter(self.messenger)
        self.kernel = Kernel(
            agent=agent,
            config=config,
            output_router=self.output_router
        )

        print(f"✓ WhatsApp Kernel initialized (instance: {instance_id})")

    def _get_save_path(self) -> Path:
        """Get save file path"""
        save_dir = Path(self.app.data_dir) / 'Agents' / 'kernel' / self.agent.amd.name / 'whatsapp'
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"whatsapp_kernel_{self.instance_id}.pkl"

    async def _auto_save_loop(self):
        """Auto-save kernel state periodically"""
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))
                print(f"💾 Auto-saved WhatsApp kernel at {datetime.now().strftime('%H:%M:%S')}")

    async def start(self):
        """Start the WhatsApp kernel"""
        self.running = True

        # Load previous state if exists
        if self.save_path.exists():
            print("📂 Loading previous WhatsApp session...")
            await self.kernel.load_from_file(str(self.save_path))

        # Start kernel
        await self.kernel.start()

        # Inject kernel prompt to agent
        self.kernel.inject_kernel_prompt_to_agent()

        # Start auto-save loop
        asyncio.create_task(self._auto_save_loop())

        # Setup WhatsApp message handler
        self.messenger.on_message = self.handle_message

        print(f"✓ WhatsApp Kernel started (instance: {self.instance_id})")

    async def stop(self):
        """Stop the WhatsApp kernel"""
        if not self.running:
            return

        self.running = False
        print("💾 Saving WhatsApp session...")

        # Save final state
        await self.kernel.save_to_file(str(self.save_path))

        # Stop kernel
        await self.kernel.stop()

        print("✓ WhatsApp Kernel stopped")

    async def handle_message(self, message: Message):
        """Handle incoming WhatsApp message"""
        try:
            # Extract message details
            sender = message.from_number
            message_type = message.type
            message_id = message.id

            # Handle different message types
            if message_type == "text":
                content = message.text

                # Send signal to kernel
                signal = KernelSignal(
                    type=SignalType.USER_INPUT,
                    id=sender,
                    content=content,
                    metadata={
                        "interface": "whatsapp",
                        "message_id": message_id,
                        "message_type": message_type
                    }
                )
                await self.kernel.process_signal(signal)

            elif message_type in ["image", "document", "audio", "video"]:
                # Download media
                media_path = await self._download_media(message)

                # Send signal with media info
                signal = KernelSignal(
                    type=SignalType.USER_INPUT,
                    id=sender,
                    content=f"[{message_type.upper()}] {message.caption or 'No caption'} [media:{media_path}]",
                    metadata={
                        "interface": "whatsapp",
                        "message_id": message_id,
                        "message_type": message_type,
                        "media_path": media_path,
                        "media_url": message.media_url
                    }
                )
                await self.kernel.process_signal(signal)

            else:
                # Unsupported message type
                await self.output_router.send_error(
                    sender,
                    f"Unsupported message type: {message_type}"
                )

        except Exception as e:
            print(f"❌ Error handling WhatsApp message: {e}")

    async def _download_media(self, message: Message) -> Optional[str]:
        """Download media from WhatsApp message"""
        try:
            # Create media directory
            media_dir = Path(self.app.data_dir) / 'Agents' / 'kernel' / 'whatsapp_media'
            media_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = self._get_media_extension(message.type, message.mime_type)
            filename = f"{message.from_number}_{timestamp}.{extension}"
            media_path = media_dir / filename

            # Download media
            media_data = await self.messenger.download_media(message.media_url)
            with open(media_path, 'wb') as f:
                f.write(media_data)

            return str(media_path)

        except Exception as e:
            print(f"❌ Error downloading media: {e}")
            return None

    def _get_media_extension(self, media_type: str, mime_type: str) -> str:
        """Get file extension from media type and MIME type"""
        extensions = {
            "image": {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif"},
            "audio": {"audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/wav": "wav"},
            "video": {"video/mp4": "mp4", "video/3gpp": "3gp"},
            "document": {"application/pdf": "pdf", "application/msword": "doc"}
        }

        return extensions.get(media_type, {}).get(mime_type, "bin")


# ===== MODULE REGISTRATION =====

Name = "isaa.KernelWhatsApp"
version = "1.0.0"
app = get_app(Name)
export = app.tb

# Global kernel instances (multi-instance support)
_kernel_instances: Dict[str, WhatsAppKernel] = {}


@export(mod_name=Name, version=version, initial=True)
async def init_kernel_whatsapp(app: App):
    """Initialize the WhatsApp Kernel module"""
    # Get WhatsApp configuration from environment
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    token = os.getenv("WHATSAPP_API_TOKEN")

    if not phone_number_id or not token:
        return {
            "success": False,
            "error": "WhatsApp credentials not configured. Set WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_API_TOKEN"
        }

    # Create default instance if credentials are available
    try:
        await add_kernel_instance(
            app=app,
            instance_id="main",
            phone_number_id=phone_number_id,
            token=token
        )
        return {"success": True, "info": f"KernelWhatsApp initialized with instance 'main'"}
    except Exception as e:
        return {"success": False, "error": f"Failed to initialize: {str(e)}"}


@export(mod_name=Name, version=version, exit_f=True)
async def on_exit_whatsapp():
    """Cleanup on module exit"""
    global _kernel_instances

    print("🛑 Stopping all WhatsApp kernel instances...")
    for instance_id in list(_kernel_instances.keys()):
        await remove_kernel_instance(instance_id)

    print("✓ All WhatsApp kernel instances stopped")


@export(mod_name=Name, version=version)
async def add_kernel_instance(
    app: App,
    instance_id: str,
    phone_number_id: str,
    token: str,
    auto_save_interval: int = 300
) -> dict:
    """
    Add a new WhatsApp kernel instance

    Args:
        app: ToolBoxV2 App instance
        instance_id: Unique identifier for this instance
        phone_number_id: WhatsApp Business phone number ID
        token: WhatsApp API token
        auto_save_interval: Auto-save interval in seconds

    Returns:
        dict with success status and info
    """
    global _kernel_instances

    if instance_id in _kernel_instances:
        return {
            "success": False,
            "error": f"Instance '{instance_id}' already exists"
        }

    try:
        # Get ISAA and create agent
        isaa = app.get_mod("isaa")
        builder = isaa.get_agent_builder(f"WhatsAppKernelAssistant_{instance_id}")
        builder.with_system_message(
            "You are a helpful WhatsApp assistant. Provide clear, concise responses. "
            "Use WhatsApp formatting when appropriate (*bold*, _italic_, ```code```)."
        )
        #uilder.with_models(
        #   fast_llm_model="openrouter/anthropic/claude-3-haiku",
        #   complex_llm_model="openrouter/openai/gpt-4o"
        #

        await isaa.register_agent(builder)
        agent = await isaa.get_agent(f"WhatsAppKernelAssistant_{instance_id}")
        agent.set_progress_callback(ProgressiveTreePrinter().progress_callback)
        # Create kernel instance
        kernel = WhatsAppKernel(
            agent=agent,
            app=app,
            phone_number_id=phone_number_id,
            token=token,
            instance_id=instance_id,
            auto_save_interval=auto_save_interval
        )

        # Start kernel
        await kernel.start()

        # Store instance
        _kernel_instances[instance_id] = kernel

        return {
            "success": True,
            "info": f"WhatsApp kernel instance '{instance_id}' created and started",
            "instance_id": instance_id
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create instance: {str(e)}"
        }


@export(mod_name=Name, version=version)
async def remove_kernel_instance(instance_id: str) -> dict:
    """
    Remove a WhatsApp kernel instance

    Args:
        instance_id: Instance identifier to remove

    Returns:
        dict with success status and info
    """
    global _kernel_instances

    if instance_id not in _kernel_instances:
        return {
            "success": False,
            "error": f"Instance '{instance_id}' not found"
        }

    try:
        kernel = _kernel_instances[instance_id]
        await kernel.stop()
        del _kernel_instances[instance_id]

        return {
            "success": True,
            "info": f"WhatsApp kernel instance '{instance_id}' stopped and removed"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to remove instance: {str(e)}"
        }


@export(mod_name=Name, version=version)
def list_kernel_instances() -> dict:
    """
    List all active WhatsApp kernel instances

    Returns:
        dict with list of instance IDs and their status
    """
    global _kernel_instances

    instances = {}
    for instance_id, kernel in _kernel_instances.items():
        instances[instance_id] = {
            "running": kernel.running,
            "save_path": str(kernel.save_path),
            "auto_save_interval": kernel.auto_save_interval
        }

    return {
        "success": True,
        "instances": instances,
        "total": len(instances)
    }


@export(mod_name=Name, version=version)
async def get_kernel_status(instance_id: str) -> dict:
    """
    Get status of a specific WhatsApp kernel instance

    Args:
        instance_id: Instance identifier

    Returns:
        dict with kernel status
    """
    global _kernel_instances

    if instance_id not in _kernel_instances:
        return {
            "success": False,
            "error": f"Instance '{instance_id}' not found"
        }

    kernel = _kernel_instances[instance_id]
    status = kernel.kernel.to_dict()

    return {
        "success": True,
        "instance_id": instance_id,
        "status": status
    }


"""
ProA Kernel WhatsApp Advanced Interface
=========================================

High-End Production-Ready WhatsApp Interface für den ProA Kernel.
Features:
- Echtzeit-Sprachnachrichten-Transkription (via Groq Whisper)
- Interaktive Nachrichten (Buttons, Listen, Templates)
- Gruppen-Kontext-Management & Broadcasts
- Medien-Handling (Bild, Audio, Dokumente)
- Auto-Persistence
- Vollständige Integration in das ToolBoxV2 Ökosystem

Voraussetzungen:
    pip install whatsapp-python groq
"""

import asyncio
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter

# Bibliotheken Import
try:
    from whatsapp import WhatsApp, Message
    from whatsapp.api.modules.interactive import Interactive
except ImportError:
    print("⚠️ WhatsApp library not installed. Install with: pip install whatsapp-python")
    WhatsApp = None
    Message = None

# Groq für Transkription
try:
    from groq import Groq

    GROQ_SUPPORT = True
except ImportError:
    GROQ_SUPPORT = False
    print("⚠️ Groq not installed. Voice transcription disabled. Install with: pip install groq")

from toolboxv2 import App, get_app
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter
from toolboxv2.mods.isaa.kernel.kernelin.tools.whatsapp_tools import WhatsAppKernelTools

# Logging Setup
logger = logging.getLogger("WhatsAppKernel")


class WhatsAppOutputRouter(IOutputRouter):
    """Erweiterter Output-Router mit Support für Interaktive Nachrichten & Medien"""

    def __init__(self, messenger: WhatsApp):
        self.messenger = messenger

    def _format_text(self, text: str, bold: bool = False, italic: bool = False, code: bool = False) -> str:
        """WhatsApp Markdown Formatierung"""
        if bold: text = f"*{text}*"
        if italic: text = f"_{text}_"
        if code: text = f"```{text}```"
        return text

    async def send_response(self, user_id: str, content: str, role: str = "assistant", metadata: dict = None):
        """Sendet eine Antwort an den Nutzer (Text oder Interaktiv)"""
        try:
            # Typing Indicator
            # Hinweis: whatsapp-python mark_as_read ist oft synchron, hier wrappen wir es
            try:
                self.messenger.mark_as_read(metadata.get("message_id"))
            except:
                pass

            # Check auf Interaktive Elemente im Metadata
            if metadata and metadata.get("interactive"):
                await self._send_interactive(user_id, content, metadata["interactive"])
            else:
                # Standard Text
                self.messenger.send_message(
                    message=content,
                    recipient_id=user_id,
                    preview_url=True
                )

        except Exception as e:
            logger.error(f"❌ Error sending WhatsApp response: {e}")

    async def _send_interactive(self, user_id: str, content: str, interactive_data: dict):
        """Sendet Buttons oder Listen"""
        try:
            itype = interactive_data.get("type")

            if itype == "button":
                # Buttons erstellen
                self.messenger.send_button(
                    recipient_id=user_id,
                    body=content,
                    buttons=interactive_data.get("buttons", []),
                    header=interactive_data.get("header"),
                    footer=interactive_data.get("footer")
                )

            elif itype == "list":
                # Liste erstellen
                self.messenger.send_list(
                    recipient_id=user_id,
                    button=interactive_data.get("button_text", "Menu"),
                    rows=interactive_data.get("rows", []),
                    title=interactive_data.get("title", "Optionen"),
                    body=content
                )

        except Exception as e:
            logger.error(f"❌ Error sending interactive message: {e}")
            # Fallback auf Text
            self.messenger.send_message(message=f"{content}\n\n(Optionen konnten nicht angezeigt werden)",
                                        recipient_id=user_id)

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Sendet eine Proactive Notification"""
        try:
            prefix = "🔔" if priority < 8 else "🚨"
            formatted = f"{prefix} *Benachrichtigung*\n\n{content}"
            self.messenger.send_message(message=formatted, recipient_id=user_id)
        except Exception as e:
            logger.error(f"❌ Error sending notification: {e}")

    async def send_media(self, user_id: str, media_path: str, media_type: str = "document", caption: str = None):
        """Sendet Medien"""
        try:
            if media_type == "image":
                self.messenger.send_image(image=media_path, recipient_id=user_id, caption=caption)
            elif media_type == "audio":
                self.messenger.send_audio(audio=media_path, recipient_id=user_id)
            elif media_type == "video":
                self.messenger.send_video(video=media_path, recipient_id=user_id, caption=caption)
            else:
                self.messenger.send_document(document=media_path, recipient_id=user_id, caption=caption)
        except Exception as e:
            logger.error(f"❌ Error sending media: {e}")


class WhatsAppKernel:
    """
    Advanced WhatsApp Kernel mit Voice-Transkription und Gruppen-Logik
    """

    def __init__(
        self,
        agent,
        app: App,
        phone_number_id: str,
        token: str,
        instance_id: str = "default",
        auto_save_interval: int = 300
    ):
        if WhatsApp is None:
            raise ImportError("WhatsApp library not installed")

        self.agent = agent
        self.app = app
        self.instance_id = instance_id
        self.auto_save_interval = auto_save_interval
        self.running = False
        self.save_path = self._get_save_path()

        # WhatsApp API Client
        self.messenger = WhatsApp(token=token, phone_number_id=phone_number_id)

        # Groq Client für Transkription
        self.groq_client = None
        if GROQ_SUPPORT and os.getenv("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            print("✓ Groq Whisper enabled for WhatsApp voice notes")

        # Kernel Konfiguration
        config = KernelConfig(
            heartbeat_interval=30.0,
            idle_threshold=600.0,
            proactive_cooldown=120.0,
            max_proactive_per_hour=10
        )

        self.output_router = WhatsAppOutputRouter(self.messenger)
        self.kernel = Kernel(
            agent=agent,
            config=config,
            output_router=self.output_router
        )

        # Tools initialisieren
        self.wa_tools = WhatsAppKernelTools(
            messenger=self.messenger,
            kernel=self.kernel,
            output_router=self.output_router
        )

        # Webhook Handler Setup (Muss von externem Server aufgerufen werden,
        # hier simulieren wir die Struktur für Integration in Flask/FastAPI)

        print(f"✓ WhatsApp Advanced Kernel initialized (instance: {instance_id})")

    def _get_save_path(self) -> Path:
        save_dir = Path(self.app.data_dir) / 'Agents' / 'kernel' / self.agent.amd.name / 'whatsapp'
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"wa_kernel_{self.instance_id}.pkl"

    async def start(self):
        """Startet den Kernel"""
        self.running = True

        if self.save_path.exists():
            await self.kernel.load_from_file(str(self.save_path))

        await self.kernel.start()
        self.kernel.inject_kernel_prompt_to_agent()

        # Tools exportieren
        await self.wa_tools.export_to_agent()

        asyncio.create_task(self._auto_save_loop())
        print(f"✓ WhatsApp Kernel started. Webhook endpoint ready.")

    async def stop(self):
        self.running = False
        await self.kernel.save_to_file(str(self.save_path))
        await self.kernel.stop()
        print("✓ WhatsApp Kernel stopped")

    async def _auto_save_loop(self):
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))

    # ===== MESSAGE HANDLING =====

    async def handle_webhook_payload(self, data: dict):
        """
        Haupt-Eingangspunkt für Webhook-Daten von Meta/WhatsApp Cloud API.
        Muss vom Webserver (Flask/FastAPI) aufgerufen werden.
        """
        try:
            # Extrahiere Nachrichten aus dem komplexen JSON
            if not data.get("entry"): return

            for entry in data["entry"]:
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    contacts = value.get("contacts", [])

                    # Kontakt-Info speichern (Name des Nutzers)
                    if contacts:
                        self._update_contact_info(contacts[0])

                    for msg in messages:
                        await self._process_single_message(msg)

        except Exception as e:
            logger.error(f"❌ Error processing webhook: {e}")
            traceback.print_exc()

    def _update_contact_info(self, contact: dict):
        """Speichert Nutzernamen für Kontext"""
        wa_id = contact.get("wa_id")
        profile = contact.get("profile", {})
        name = profile.get("name")
        if wa_id and name:
            # Wir könnten das in den Kernel Memory injecten
            # Hier speichern wir es temporär oder über ContextStore
            pass

    async def _process_single_message(self, msg: dict):
        """Verarbeitet eine einzelne Nachricht (Text, Audio, Interaktiv)"""
        sender_id = msg.get("from")
        msg_type = msg.get("type")
        msg_id = msg.get("id")

        # Metadaten aufbauen
        metadata = {
            "interface": "whatsapp",
            "message_id": msg_id,
            "timestamp": msg.get("timestamp"),
            "user_name": "",  # Könnte aus contacts geholt werden
            "is_group": False  # Cloud API handhabt Gruppen anders, meist 1:1
        }

        content = ""
        signal_type = SignalType.USER_INPUT

        # 1. TEXT
        if msg_type == "text":
            content = msg["text"]["body"]

        # 2. AUDIO (Voice Notes)
        elif msg_type == "audio" and self.groq_client:
            audio_id = msg["audio"]["id"]
            content = await self._handle_voice_note(audio_id, sender_id)
            if content:
                metadata["transcription"] = True
                content = f"[Voice Transcription] {content}"
            else:
                return  # Fehler bei Transkription

        # 3. INTERACTIVE (Button Replies)
        elif msg_type == "interactive":
            interactive = msg["interactive"]
            if interactive["type"] == "button_reply":
                content = interactive["button_reply"]["title"]
                metadata["button_id"] = interactive["button_reply"]["id"]
            elif interactive["type"] == "list_reply":
                content = interactive["list_reply"]["title"]
                metadata["list_id"] = interactive["list_reply"]["id"]
                metadata["description"] = interactive["list_reply"].get("description", "")

        # 4. IMAGE/DOCUMENT
        elif msg_type in ["image", "document"]:
            media_id = msg[msg_type]["id"]
            caption = msg[msg_type].get("caption", "")
            media_url = self.messenger.get_media_url(media_id)
            content = f"[{msg_type.upper()}] {caption} (Media ID: {media_id})"
            metadata["media_url"] = media_url
            metadata["caption"] = caption

        else:
            # Unbekannter Typ
            return

        # Signal senden
        if content:
            signal = KernelSignal(
                type=signal_type,
                id=sender_id,
                content=content,
                metadata=metadata
            )
            await self.kernel.process_signal(signal)

    async def _handle_voice_note(self, media_id: str, sender_id: str) -> Optional[str]:
        """Lädt Audio herunter und transkribiert mit Groq"""
        try:
            # 1. URL holen
            media_url = self.messenger.get_media_url(media_id)

            # 2. Download (Wrapper-Funktion oder Requests)
            # Annahme: self.messenger hat download_media, sonst requests nutzen
            import requests
            # Hinweis: Benötigt Auth Token im Header
            headers = {"Authorization": f"Bearer {self.messenger.token}"}
            response = requests.get(media_url, headers=headers)

            if response.status_code == 200:
                # Temp file speichern
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

                # 3. Transkribieren mit Groq
                with open(tmp_path, "rb") as file:
                    transcription = self.groq_client.audio.transcriptions.create(
                        file=(tmp_path, file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="json"
                    )

                # Cleanup
                os.unlink(tmp_path)
                return transcription.text

            return None

        except Exception as e:
            logger.error(f"Transkriptionsfehler: {e}")
            return None


# ===== MODULE REGISTRATION =====

Name = "isaa.KernelWhatsAppAdvanced"
version = "2.0.0"
app = get_app(Name)
export = app.tb
_kernel_instances: Dict[str, WhatsAppKernel] = {}


@export(mod_name=Name, version=version, initial=True)
async def init_wa_advanced(app: App):
    phone_id = os.getenv("WHATSAPP_PHONE_ID")
    token = os.getenv("WHATSAPP_TOKEN")

    if not phone_id or not token:
        return {"success": False, "error": "Env vars WHATSAPP_PHONE_ID or WHATSAPP_TOKEN missing"}

    # Agent erstellen
    isaa = app.get_mod("isaa")
    builder = isaa.get_agent_builder("WhatsAppProAgent")
    builder.with_system_message(
        "Du bist ein intelligenter WhatsApp-Assistent. Nutze Buttons und Listen für Interaktionen.")
    await isaa.register_agent(builder)
    agent = await isaa.get_agent("WhatsAppProAgent")

    kernel = WhatsAppKernel(agent, app, phone_id, token)
    await kernel.start()
    _kernel_instances["main"] = kernel

    return {"success": True, "info": "WhatsApp Advanced Kernel active"}


# Für externe Webhook-Integration (z.B. FastAPI Route)
@export(mod_name=Name, version=version)
async def feed_webhook_data(data: dict, instance_id: str = "main"):
    """Funktion, die vom Webserver aufgerufen wird, um Daten in den Kernel zu speisen"""
    if instance_id in _kernel_instances:
        await _kernel_instances[instance_id].handle_webhook_payload(data)
        return {"status": "processed"}
    return {"status": "error", "message": "instance not found"}
