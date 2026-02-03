"""
Discord Interface V2 - Modulares Agent-Kommunikations-System
=============================================================

Einfach, modular, wiederverwendbar.

Kernprinzipien:
1. Nur Kommunikation - keine Business-Logik
2. Context-Awareness: Wer schreibt von wo
3. Auto-Router: Antworten gehen automatisch zurück zum Ursprung
4. VFS Adressbuch: Agent entscheidet bewusst wohin

Architektur:
- DiscordInterface: Hauptklasse, verbindet Bot mit Agent
- MessageContext: Enthält alle Infos über eine Nachricht
- AddressBook (VFS): /discord/servers, /discord/dms, /discord/active
- AutoRouter: Routet Antworten automatisch
- AudioHandler: TTS/STT Integration

Author: Markin / ToolBoxV2
Version: 2.1.0
"""

import asyncio
import io
import os
import json
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING
from enum import Enum, auto

try:
    from groq import AsyncGroq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    AsyncGroq = None

try:
    import discord
    from discord.ext import commands
except ImportError:
    raise ImportError("pip install discord.py>=2.6.4")

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent


# =============================================================================
# AUDIO INTEGRATION - TTS/STT aus ToolBoxV2
# =============================================================================

# TTS Backend imports (lazy loaded)
_TTS_AVAILABLE = False
_STT_AVAILABLE = False

def _load_tts():
    """Lazy load TTS module"""
    global _TTS_AVAILABLE
    try:
        from toolboxv2.mods.isaa.base.audio_io.Tts import (
            synthesize, TTSConfig, TTSBackend, TTSResult
        )
        _TTS_AVAILABLE = True
        return synthesize, TTSConfig, TTSBackend
    except ImportError:
        try:
            # Fallback: direkter Import
            from Tts import synthesize, TTSConfig, TTSBackend
            _TTS_AVAILABLE = True
            return synthesize, TTSConfig, TTSBackend
        except ImportError:
            return None, None, None

def _load_stt():
    """Lazy load STT module"""
    global _STT_AVAILABLE
    try:
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            transcribe, STTConfig, STTBackend, STTResult
        )
        _STT_AVAILABLE = True
        return transcribe, STTConfig, STTBackend
    except ImportError:
        try:
            from Stt import transcribe, STTConfig, STTBackend
            _STT_AVAILABLE = True
            return transcribe, STTConfig, STTBackend
        except ImportError:
            return None, None, None


# =============================================================================
# MESSAGE CONTEXT - Alles was der Agent über eine Nachricht wissen muss
# =============================================================================

class MessageSource(Enum):
    """Woher kommt die Nachricht"""
    TEXT_CHANNEL = auto()
    DM = auto()
    VOICE_CHANNEL = auto()
    THREAD = auto()


@dataclass
class AttachmentInfo:
    """Strukturierte Attachment-Information mit Media-Type Detection"""
    path: str
    filename: str
    content_type: Optional[str]
    size: int
    transcription: Optional[str] = None  # Für Audio

    @property
    def media_type(self) -> str:
        """Bestimmt den Medientyp basierend auf content_type und Extension"""
        if self.content_type:
            ct = self.content_type.lower()
            if "image" in ct:
                return "image"
            if "audio" in ct:
                return "audio"
            if "video" in ct:
                return "video"
            if "pdf" in ct:
                return "pdf"

        # Fallback auf Extension
        ext = Path(self.filename).suffix.lower()
        ext_map = {
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image',
            '.gif': 'image', '.webp': 'image', '.bmp': 'image',
            '.mp3': 'audio', '.wav': 'audio', '.ogg': 'audio',
            '.m4a': 'audio', '.flac': 'audio',
            '.mp4': 'video', '.avi': 'video', '.mov': 'video',
            '.mkv': 'video', '.webm': 'video',
            '.pdf': 'pdf',
        }
        return ext_map.get(ext, "file")

    @property
    def is_native_llm_compatible(self) -> bool:
        """Prüft ob das Format direkt vom LLM verarbeitet werden kann"""
        return self.media_type == "image"

    def to_media_tag(self) -> str:
        """Generiert den [media:path] Tag für den Agent"""
        return f"[media:{self.path}]"

    def to_context_line(self) -> str:
        """Generiert eine kontextuelle Beschreibung für den Agent"""
        size_kb = self.size / 1024
        size_str = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb / 1024:.1f}MB"

        line = f"  - {self.filename} ({self.media_type}, {size_str})"

        if self.is_native_llm_compatible:
            line += " → Native verarbeitbar"
        else:
            line += " → Erfordert Tool-Verarbeitung"

        return line


@dataclass
class MessageContext:
    """
    Vollständiger Context einer eingehenden Nachricht.
    Wird dem Agent als strukturierte Info übergeben.
    """
    # Quelle
    source: MessageSource
    source_address: str  # z.B. "discord://server:123/channel:456" oder "discord://dm:789"

    # User Info
    user_id: int
    user_name: str
    user_display_name: str
    is_bot: bool = False

    # Server Info (None bei DMs)
    guild_id: Optional[int] = None
    guild_name: Optional[str] = None

    # Channel Info
    channel_id: int = 0
    channel_name: str = ""

    # Message Content
    content: str = ""
    message_id: int = 0
    timestamp: str = ""

    # Mentions
    mentioned_bot: bool = False
    mentioned_users: list[int] = field(default_factory=list)

    # Attachments
    attachments: list[AttachmentInfo] = field(default_factory=list)

    # Reply Context
    is_reply: bool = False
    reply_to_message_id: Optional[int] = None

    # Voice Context (wenn im Voice Channel)
    voice_channel_id: Optional[int] = None
    voice_channel_name: Optional[str] = None

    # Special Flags
    wants_audio_response: bool = False  # #audio am Ende

    def to_agent_context(self) -> str:
        """Formatiert den Context für den Agent System Prompt"""
        lines = [
            f"[Discord Message Context]",
            f"From: {self.user_display_name} (@{self.user_name})",
            f"Source: {self.source.name}",
            f"Address: {self.source_address}",
        ]

        if self.guild_name:
            lines.append(f"Server: {self.guild_name}")
            lines.append(f"Channel: #{self.channel_name}")
        else:
            lines.append("Channel: Direct Message")

        if self.mentioned_bot:
            lines.append("Note: You were directly mentioned")

        if self.is_reply:
            lines.append(f"Note: This is a reply to message {self.reply_to_message_id}")

        if self.voice_channel_name:
            lines.append(f"Voice: User is in #{self.voice_channel_name} (ID: {self.voice_channel_id})")

        if self.wants_audio_response:
            lines.append("Response Mode: AUDIO (convert response to TTS)")

        if self.attachments:
            lines.append("")
            lines.append("[Attachments]")

            native_media = []
            non_native_media = []
            transcriptions = []

            for att in self.attachments:
                # Kategorisiere Attachments
                if att.is_native_llm_compatible:
                    native_media.append(att)
                else:
                    non_native_media.append(att)

                # Sammle Transkriptionen
                if att.transcription:
                    transcriptions.append((att.filename, att.transcription))

            # Native Media (Bilder) - Direkt als [media:] Tags
            if native_media:
                lines.append("Native Media (direkt verarbeitbar):")
                for att in native_media:
                    lines.append(f"  {att.to_media_tag()}")
                    lines.append(f"    ↳ {att.filename} ({att.media_type})")

            # Non-Native Media - Mit Hinweis auf erforderliche Tools
            if non_native_media:
                lines.append("Weitere Dateien (erfordern Tool-Verarbeitung):")
                for att in non_native_media:
                    lines.append(att.to_context_line())
                    lines.append(f"    Pfad: {att.path}")

                    # Spezifische Hinweise je nach Typ
                    if att.media_type == "audio":
                        lines.append(
                            "    → Audio bereits transkribiert (siehe unten)" if att.transcription else "    → Nutze Audio-Transkription für Inhalt")
                    elif att.media_type == "video":
                        lines.append("    → Nutze Video-Frame-Extraktion oder Transkription")
                    elif att.media_type == "pdf":
                        lines.append("    → Nutze PDF-Text-Extraktion für Inhalt")
                    else:
                        lines.append(f"    → Dateityp '{att.media_type}' - manuelle Verarbeitung erforderlich")

            # Audio Transkriptionen
            if transcriptions:
                lines.append("")
                lines.append("[Audio Transcriptions]")
                for filename, text in transcriptions:
                    lines.append(f"  {filename}:")
                    # Transkription einrücken
                    for line in text.split('\n'):
                        lines.append(f"    {line}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisiert für VFS Storage"""
        return {
            "source": self.source.name,
            "source_address": self.source_address,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_display_name": self.user_display_name,
            "guild_id": self.guild_id,
            "guild_name": self.guild_name,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "mentioned_bot": self.mentioned_bot,
            "wants_audio_response": self.wants_audio_response,
        }


# =============================================================================
# ADDRESS BOOK - VFS Integration für bewusste Kommunikation
# =============================================================================

class AddressBook:
    """
    Verwaltet das VFS Adressbuch für Discord.

    Struktur:
    /discord/
        servers/{guild_id}/
            info.json           # Server-Metadaten
            channels/{channel_id}/
                session.json    # Conversation State
                history.json    # Letzte Nachrichten
        dms/{user_id}/
            session.json
            history.json
        active.json             # Aktive Conversations
        contacts.json           # Bekannte User
    """

    def __init__(self, vfs: Any):
        self.vfs = vfs
        self._ensure_structure()

    def _ensure_structure(self):
        """Erstellt die Basis-Verzeichnisstruktur"""
        dirs = [
            "/discord",
            "/discord/servers",
            "/discord/dms",
        ]
        for d in dirs:
            if not self.vfs._is_directory(d):
                self.vfs.mkdir(d)

        # Initiale Files
        if not self.vfs._is_file("/discord/active.json"):
            self.vfs.create("/discord/active.json", json.dumps({
                "conversations": {},
                "last_updated": datetime.now().isoformat()
            }))

        if not self.vfs._is_file("/discord/contacts.json"):
            self.vfs.create("/discord/contacts.json", json.dumps({
                "users": {},
                "last_updated": datetime.now().isoformat()
            }))

    def register_server(self, guild_id: int, guild_name: str, channels: list[dict]):
        """Registriert einen Server im Adressbuch"""
        server_dir = f"/discord/servers/{guild_id}"

        if not self.vfs._is_directory(server_dir):
            self.vfs.mkdir(server_dir)
            self.vfs.mkdir(f"{server_dir}/channels")

        self.vfs.write(f"{server_dir}/info.json", json.dumps({
            "id": guild_id,
            "name": guild_name,
            "channels": channels,
            "registered_at": datetime.now().isoformat()
        }))

    def register_channel(self, guild_id: int, channel_id: int, channel_name: str):
        """Registriert einen Channel"""
        channel_dir = f"/discord/servers/{guild_id}/channels/{channel_id}"

        if not self.vfs._is_directory(channel_dir):
            self.vfs.mkdir(channel_dir)

        if not self.vfs._is_file(f"{channel_dir}/session.json"):
            self.vfs.create(f"{channel_dir}/session.json", json.dumps({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
            }))

    def register_dm(self, user_id: int, user_name: str):
        """Registriert einen DM Channel"""
        dm_dir = f"/discord/dms/{user_id}"

        if not self.vfs._is_directory(dm_dir):
            self.vfs.mkdir(dm_dir)

        if not self.vfs._is_file(f"{dm_dir}/session.json"):
            self.vfs.create(f"{dm_dir}/session.json", json.dumps({
                "user_id": user_id,
                "user_name": user_name,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
            }))

    def update_contact(self, user_id: int, user_name: str, display_name: str, guild_id: Optional[int] = None):
        """Aktualisiert einen Kontakt"""
        contacts_result = self.vfs.read("/discord/contacts.json")
        if not contacts_result.get("success"):
            return

        contacts = json.loads(contacts_result["content"])

        user_key = str(user_id)
        if user_key not in contacts["users"]:
            contacts["users"][user_key] = {
                "id": user_id,
                "name": user_name,
                "display_name": display_name,
                "guilds": [],
                "first_seen": datetime.now().isoformat(),
            }

        contacts["users"][user_key]["name"] = user_name
        contacts["users"][user_key]["display_name"] = display_name
        contacts["users"][user_key]["last_seen"] = datetime.now().isoformat()

        if guild_id and guild_id not in contacts["users"][user_key]["guilds"]:
            contacts["users"][user_key]["guilds"].append(guild_id)

        contacts["last_updated"] = datetime.now().isoformat()
        self.vfs.write("/discord/contacts.json", json.dumps(contacts, indent=2))

    def update_active_conversation(self, ctx: MessageContext):
        """Markiert eine Conversation als aktiv"""
        active_result = self.vfs.read("/discord/active.json")
        if not active_result.get("success"):
            return

        active = json.loads(active_result["content"])

        conv_key = ctx.source_address
        active["conversations"][conv_key] = {
            "address": ctx.source_address,
            "user_id": ctx.user_id,
            "user_name": ctx.user_display_name,
            "channel_name": ctx.channel_name if ctx.guild_id else "DM",
            "guild_name": ctx.guild_name,
            "last_message": ctx.timestamp,
            "source": ctx.source.name,
        }
        active["last_updated"] = datetime.now().isoformat()

        self.vfs.write("/discord/active.json", json.dumps(active, indent=2))

    def get_active_conversations(self) -> dict:
        """Holt alle aktiven Conversations"""
        result = self.vfs.read("/discord/active.json")
        if result.get("success"):
            return json.loads(result["content"])
        return {"conversations": {}}

    def get_channel_address(self, guild_id: int, channel_id: int) -> str:
        """Generiert eine Discord-Adresse"""
        return f"discord://server:{guild_id}/channel:{channel_id}"

    def get_dm_address(self, user_id: int) -> str:
        """Generiert eine DM-Adresse"""
        return f"discord://dm:{user_id}"

    def parse_address(self, address: str) -> dict:
        """Parst eine Discord-Adresse"""
        result = {"type": None, "guild_id": None, "channel_id": None, "user_id": None}

        if address.startswith("discord://dm:"):
            result["type"] = "dm"
            result["user_id"] = int(address.split(":")[-1])
        elif address.startswith("discord://server:"):
            result["type"] = "channel"
            parts = address.replace("discord://server:", "").split("/channel:")
            result["guild_id"] = int(parts[0])
            if len(parts) > 1:
                result["channel_id"] = int(parts[1])

        return result

    def search(self, query: str) -> list[dict]:
        """
        Durchsucht das Adressbuch.

        Args:
            query: Suchbegriff (User-Name, Channel-Name, Server-Name)

        Returns:
            Liste von Matches mit Adresse und Info
        """
        results = []
        query_lower = query.lower()

        # Contacts durchsuchen
        contacts_result = self.vfs.read("/discord/contacts.json")
        if contacts_result.get("success"):
            contacts = json.loads(contacts_result["content"])
            for user_id, user_info in contacts.get("users", {}).items():
                if query_lower in user_info.get("name", "").lower() or \
                   query_lower in user_info.get("display_name", "").lower():
                    results.append({
                        "type": "user",
                        "address": self.get_dm_address(int(user_id)),
                        "name": user_info.get("display_name"),
                        "username": user_info.get("name"),
                    })

        # Server/Channels durchsuchen
        servers_result = self.vfs.ls("/discord/servers")
        if servers_result.get("success"):
            for item in servers_result.get("contents", []):
                if item["type"] == "directory":
                    guild_id = int(item["name"])
                    info_result = self.vfs.read(f"/discord/servers/{guild_id}/info.json")
                    if info_result.get("success"):
                        info = json.loads(info_result["content"])
                        if query_lower in info.get("name", "").lower():
                            results.append({
                                "type": "server",
                                "address": f"discord://server:{guild_id}",
                                "name": info.get("name"),
                            })
                        for ch in info.get("channels", []):
                            if query_lower in ch.get("name", "").lower():
                                results.append({
                                    "type": "channel",
                                    "address": self.get_channel_address(guild_id, ch["id"]),
                                    "name": f"#{ch['name']}",
                                    "server": info.get("name"),
                                })

        return results


# =============================================================================
# AUTO ROUTER - Routet Antworten automatisch zurück
# =============================================================================


@dataclass
class DiscordConfig:
    """Discord transport configuration"""

    token: str
    admin_whitelist: list[int] = field(default_factory=list)
    command_prefix: str = "!"  # Ignored - no commands, just for bot init

    # Voice settings
    enable_voice: bool = True
    voice_language: str = "de"
    silence_threshold_ms: int = 1500
    min_audio_length_ms: int = 500

    # Media settings
    temp_dir: str = "/tmp/discord_media"
    max_attachment_size_mb: int = 25

    # TTS settings (output)
    tts_provider: str = "local"  # "local", "elevenlabs", "google"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"


# =============================================================================
# MEDIA HANDLER
# =============================================================================


class MediaHandler:
    """Handles media downloads and processing"""

    def __init__(self, config: DiscordConfig):
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Groq client for transcription
        self._groq: Optional[AsyncGroq] = None
        if GROQ_AVAILABLE:
            groq_key = os.environ.get("GROQ_API_KEY")
            if groq_key:
                self._groq = AsyncGroq(api_key=groq_key)

    async def download_attachment(self, attachment: discord.Attachment) -> Optional[str]:
        """Download attachment to temp file, return path"""
        if attachment.size > self.config.max_attachment_size_mb * 1024 * 1024:
            return None

        # Generate unique filename
        ext = Path(attachment.filename).suffix or ".bin"
        filename = f"{int(time.time())}_{attachment.id}{ext}"
        filepath = self.temp_dir / filename

        try:
            await attachment.save(filepath)
            return str(filepath)
        except Exception as e:
            print(f"[Discord] Failed to download attachment: {e}")
            return None

    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file using Groq Whisper"""
        if not self._groq:
            print("[Discord] Groq not available for transcription")
            return None

        try:
            with open(audio_path, "rb") as audio_file:
                transcription = await self._groq.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language=self.config.voice_language,
                )
            return transcription.text
        except Exception as e:
            print(f"[Discord] Transcription failed: {e}")
            return None

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old temp files"""
        cutoff = time.time() - (max_age_hours * 3600)
        for filepath in self.temp_dir.iterdir():
            if filepath.stat().st_mtime < cutoff:
                try:
                    filepath.unlink()
                except Exception:
                    pass


class AutoRouter:
    """
    Routet Agent-Antworten automatisch zum richtigen Ziel.

    Logik:
    1. Agent antwortet ohne explizites Ziel → zurück zum Ursprung
    2. Agent spezifiziert Ziel via Tool → dorthin senden
    3. Bei #audio → TTS und Audio-File senden
    """

    def __init__(self, bot: commands.Bot, media_handler: MediaHandler):
        self.bot = bot
        self.media_handler = media_handler
        self._pending_context: Optional[MessageContext] = None

    def set_pending_context(self, ctx: MessageContext):
        """Setzt den aktuellen Context für Auto-Routing"""
        self._pending_context = ctx

    def clear_pending_context(self):
        """Löscht den pending Context"""
        self._pending_context = None

    async def route_response(
        self,
        content: str,
        target_address: Optional[str] = None,
        reply_to: Optional[int] = None,
        as_audio: bool = False,
    ) -> dict:
        """
        Routet eine Antwort.

        Args:
            content: Die Antwort
            target_address: Optional explizites Ziel (discord://...)
            reply_to: Optional Message ID für Reply
            as_audio: TTS Response?

        Returns:
            Result dict
        """
        # Ziel bestimmen
        if target_address:
            target = self._parse_address(target_address)
        elif self._pending_context:
            target = {
                "type": "auto",
                "channel_id": self._pending_context.channel_id,
                "user_id": self._pending_context.user_id if not self._pending_context.guild_id else None,
            }
            # Auto-Reply wenn Kontext vorhanden
            if not reply_to and self._pending_context.message_id:
                reply_to = self._pending_context.message_id
            # Audio wenn gewünscht
            if self._pending_context.wants_audio_response:
                as_audio = True
        else:
            return {"success": False, "error": "No target and no pending context"}

        # Channel holen
        try:
            if target.get("user_id") and not target.get("channel_id"):
                # DM
                user = await self.bot.fetch_user(target["user_id"])
                channel = await user.create_dm()
            else:
                channel = self.bot.get_channel(target["channel_id"])
                if not channel:
                    channel = await self.bot.fetch_channel(target["channel_id"])

            if not channel:
                return {"success": False, "error": f"Channel not found"}

            # Reference für Reply
            reference = None
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(reply_to)
                    reference = ref_msg
                except:
                    pass

            # Audio Response?
            if as_audio:
                return await self._send_audio_response(channel, content, reference)

            # Text Response
            return await self._send_text_response(channel, content, reference)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_text_response(
        self,
        channel: discord.TextChannel,
        content: str,
        reference: Optional[discord.Message] = None,
    ) -> dict:
        """Sendet Text-Response, splittet bei Bedarf"""
        if len(content) > 2000:
            chunks = [content[i:i+1990] for i in range(0, len(content), 1990)]
            for i, chunk in enumerate(chunks):
                prefix = f"({i+1}/{len(chunks)}) " if len(chunks) > 1 else ""
                msg = await channel.send(prefix + chunk, reference=reference if i == 0 else None)
                await asyncio.sleep(0.3)
            return {"success": True, "message_id": msg.id, "chunks": len(chunks)}
        else:
            msg = await channel.send(content, reference=reference)
            return {"success": True, "message_id": msg.id}

    async def _send_audio_response(
        self,
        channel: discord.TextChannel,
        content: str,
        reference: Optional[discord.Message] = None,
    ) -> dict:
        """Synthetisiert Audio und sendet als File"""
        try:
            # TTS
            audio_bytes = await self.media_handler.synthesize_speech(content)

            if not audio_bytes:
                # Fallback: Text senden wenn TTS fehlschlägt
                return await self._send_text_response(
                    channel,
                    f"🔊 *[TTS unavailable]*\n\n{content}",
                    reference
                )

            # Audio File erstellen
            audio_file = discord.File(
                io.BytesIO(audio_bytes),
                filename=f"response_{int(time.time())}.wav"
            )

            # Kurze Text-Zusammenfassung (optional)
            summary = content[:200] + "..." if len(content) > 200 else content

            msg = await channel.send(
                content=f"🔊 *Audio Response*",
                file=audio_file,
                reference=reference,
            )

            return {
                "success": True,
                "message_id": msg.id,
                "audio": True,
                "audio_size": len(audio_bytes),
            }

        except Exception as e:
            print(f"[AutoRouter] Audio response failed: {e}")
            # Fallback: Text
            return await self._send_text_response(channel, content, reference)

    def _parse_address(self, address: str) -> dict:
        """Parst eine Discord-Adresse zu Routing-Info"""
        result = {"type": None, "channel_id": None, "user_id": None}

        if address.startswith("discord://dm:"):
            result["type"] = "dm"
            result["user_id"] = int(address.split(":")[-1])
        elif address.startswith("discord://server:"):
            result["type"] = "channel"
            parts = address.replace("discord://server:", "").split("/channel:")
            if len(parts) > 1:
                result["channel_id"] = int(parts[1])

        return result


# =============================================================================
# MEDIA HANDLER - Attachments verarbeiten + TTS/STT
# =============================================================================

class MediaHandler:
    """Verarbeitet Discord Attachments und Audio I/O"""

    def __init__(
        self,
        temp_dir: str = "/tmp/discord_media",
        tts_backend: str = "groq",  # "piper", "groq", "elevenlabs"
        tts_voice: str = "autumn",  # Backend-spezifisch
        stt_backend: str = "groq",  # "faster_whisper", "groq"
        language: str = "de",
    ):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = 25

        # TTS/STT Config
        self.tts_backend = tts_backend
        self.tts_voice = tts_voice
        self.stt_backend = stt_backend
        self.language = language

    async def process_attachment(self, attachment: discord.Attachment) -> Optional[dict]:
        """
        Lädt ein Attachment herunter und gibt Info zurück.

        Returns:
            Dict mit path, filename, content_type, size oder None
        """
        if attachment.size > self.max_size_mb * 1024 * 1024:
            return None

        ext = Path(attachment.filename).suffix or ".bin"
        filename = f"{int(time.time())}_{attachment.id}{ext}"
        filepath = self.temp_dir / filename

        try:
            await attachment.save(filepath)
            return {
                "path": str(filepath),
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "size": attachment.size,
            }
        except Exception as e:
            print(f"[MediaHandler] Download failed: {e}")
            return None

    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transkribiert Audio via STT Backend"""
        transcribe, STTConfig, STTBackend = _load_stt()

        if transcribe is None:
            # Fallback: Groq direkt
            return await self._transcribe_groq_direct(audio_path)

        try:
            # Backend auswählen
            if self.stt_backend == "groq":
                config = STTConfig(
                    backend=STTBackend.GROQ_WHISPER,
                    language=self.language,
                )
            else:
                config = STTConfig(
                    backend=STTBackend.FASTER_WHISPER,
                    language=self.language,
                    device="cpu",
                    compute_type="int8",
                )

            result = transcribe(audio_path, config=config)
            return result.text

        except Exception as e:
            print(f"[MediaHandler] STT failed: {e}")
            return await self._transcribe_groq_direct(audio_path)

    async def _transcribe_groq_direct(self, audio_path: str) -> Optional[str]:
        """Fallback: Direkte Groq API"""
        try:
            from groq import AsyncGroq

            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                return None

            client = AsyncGroq(api_key=groq_key)

            with open(audio_path, "rb") as f:
                transcription = await client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    language=self.language,
                )
            return transcription.text
        except Exception as e:
            print(f"[MediaHandler] Groq STT failed: {e}")
            return None

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Synthetisiert Text zu Audio via TTS Backend"""
        synthesize, TTSConfig, TTSBackend = _load_tts()

        if synthesize is None:
            # Fallback: Groq direkt
            return await self._synthesize_groq_direct(text)

        try:
            # Backend auswählen
            if self.tts_backend == "groq":
                config = TTSConfig(
                    backend=TTSBackend.GROQ_TTS,
                    voice=self.tts_voice,
                    language=self.language,
                )
            elif self.tts_backend == "elevenlabs":
                config = TTSConfig(
                    backend=TTSBackend.ELEVENLABS,
                    voice=self.tts_voice,
                    language=self.language,
                )
            else:  # piper
                # Piper voice format: de_DE-thorsten-medium
                voice = self.tts_voice
                if not "_" in voice:
                    voice = f"{self.language}_{'DE' if self.language == 'de' else 'US'}-{voice}-medium"
                config = TTSConfig(
                    backend=TTSBackend.PIPER,
                    voice=voice,
                    language=self.language,
                )

            result = synthesize(text, config=config)
            return result.audio

        except Exception as e:
            print(f"[MediaHandler] TTS failed: {e}")
            return await self._synthesize_groq_direct(text)

    async def _synthesize_groq_direct(self, text: str) -> Optional[bytes]:
        """Fallback: Direkte Groq TTS API"""
        try:
            from groq import AsyncGroq

            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                print("[MediaHandler] No GROQ_API_KEY set")
                return None

            client = AsyncGroq(api_key=groq_key)

            response = await client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice=self.tts_voice or "autumn",
                input=text,
                response_format="wav",
            )

            # Response ist ein Iterator über Bytes
            audio_bytes = b""
            async for chunk in response:
                audio_bytes += chunk

            return audio_bytes

        except Exception as e:
            print(f"[MediaHandler] Groq TTS failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup_old(self, max_age_hours: int = 24):
        """Löscht alte Temp-Dateien"""
        cutoff = time.time() - (max_age_hours * 3600)
        for f in self.temp_dir.iterdir():
            if f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                except:
                    pass

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        format: str = "wav"
    ) -> Optional[str]:
        """
        Transkribiert Audio aus Bytes (kein Disk I/O).

        Args:
            audio_bytes: Raw audio data
            format: Audio format (wav, mp3, etc.)

        Returns:
            Transcribed text or None
        """
        transcribe, STTConfig, STTBackend = _load_stt()

        # Groq/Cloud APIs: BytesIO direkt
        if self.stt_backend == "groq":
            return await self._transcribe_groq_bytes(audio_bytes, format)

        # Lokale Backends (faster_whisper): brauchen leider temp file
        # faster-whisper kann kein BytesIO, nur file paths
        if transcribe is not None:
            return await self._transcribe_local_bytes(audio_bytes, format, transcribe, STTConfig, STTBackend)

        # Fallback
        return await self._transcribe_groq_bytes(audio_bytes, format)

    async def _transcribe_groq_bytes(
        self,
        audio_bytes: bytes,
        format: str = "wav"
    ) -> Optional[str]:
        """Groq API mit BytesIO - kein Disk I/O"""
        try:
            from groq import AsyncGroq

            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                return None

            client = AsyncGroq(api_key=groq_key)

            # BytesIO mit filename für format detection
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{format}"

            transcription = await client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language=self.language,
            )
            return transcription.text

        except Exception as e:
            print(f"[MediaHandler] Groq bytes STT failed: {e}")
            return None

    async def _transcribe_local_bytes(
        self,
        audio_bytes: bytes,
        format: str,
        transcribe,
        STTConfig,
        STTBackend
    ) -> Optional[str]:
        """
        Lokales Backend (faster_whisper) - minimaler temp file.
        faster-whisper unterstützt kein BytesIO, daher tmpfs/ramfs nutzen.
        """
        import tempfile

        # /dev/shm ist RAM-backed auf Linux - quasi in-memory
        tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{format}",
                dir=tmp_dir,  # RAM wenn verfügbar
                delete=True
            ) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()

                config = STTConfig(
                    backend=STTBackend.FASTER_WHISPER,
                    language=self.language,
                    device="cpu",
                    compute_type="int8",
                )

                result = transcribe(tmp.name, config=config)
                return result.text

        except Exception as e:
            print(f"[MediaHandler] Local bytes STT failed: {e}")
            return await self._transcribe_groq_bytes(audio_bytes, format)


# =============================================================================
# DISCORD INTERFACE - Hauptklasse
# =============================================================================

class DiscordInterface:
    """
    Modulares Discord Interface für FlowAgent.

    Usage:
        interface = DiscordInterface(agent, token)
        await interface.start()

    Der Agent erhält:
    - MessageContext bei jeder Nachricht
    - Zugriff auf AddressBook via VFS (/discord/...)
    - Tools zum bewussten Senden an andere Ziele

    Features:
    - Text Responses (automatisch)
    - Audio Responses (#audio am Ende)
    - Audio Input Transcription
    - VFS-basiertes Adressbuch
    """

    def __init__(
        self,
        agent: "FlowAgent",
        token: str,
        respond_to_mentions_only: bool = True,
        admin_ids: Optional[list[int]] = None,
        tts_backend: str = "groq",  # "piper", "groq", "elevenlabs"
        tts_voice: str = "autumn",
        stt_backend: str = "groq",  # "faster_whisper", "groq"
        language: str = "de",
    ):
        self.agent = agent
        self.token = token
        self.respond_to_mentions_only = respond_to_mentions_only
        self.admin_ids = admin_ids or []

        # Bot Setup
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        intents.voice_states = True

        self.bot = commands.Bot(
            command_prefix="!",
            intents=intents,
            help_command=None,
        )

        # Components mit TTS/STT Config
        self.media_handler = MediaHandler(
            tts_backend=tts_backend,
            tts_voice=tts_voice,
            stt_backend=stt_backend,
            language=language,
        )
        self.router = AutoRouter(self.bot, self.media_handler)
        self.address_book: Optional[AddressBook] = None  # Init nach Agent Session

        # Register Events
        self._register_events()

        # Register Agent Tools
        self._register_agent_tools()

    def _register_events(self):
        """Registriert Discord Event Handler"""

        @self.bot.event
        async def on_ready():
            print(f"[Discord] Bot ready: {self.bot.user}")
            print(f"[Discord] Guilds: {[g.name for g in self.bot.guilds]}")

            # Init AddressBook mit Agent VFS
            session = await self.agent.session_manager.get_or_create("discord")
            self.address_book = AddressBook(session.vfs)

            # Register all Servers
            for guild in self.bot.guilds:
                channels = [
                    {"id": ch.id, "name": ch.name, "type": str(ch.type)}
                    for ch in guild.channels
                    if isinstance(ch, discord.TextChannel)
                ]
                self.address_book.register_server(guild.id, guild.name, channels)

            print(f"[Discord] AddressBook initialized at /discord in VFS")

        @self.bot.event
        async def on_message(message: discord.Message):
            # Ignore self
            if message.author.bot:
                return

            # Build Context
            ctx = await self._build_message_context(message)

            # Check if should respond
            if not self._should_respond(ctx):
                # Trotzdem im Adressbuch tracken
                if self.address_book:
                    self.address_book.update_contact(
                        ctx.user_id, ctx.user_name, ctx.user_display_name, ctx.guild_id
                    )
                    self.address_book.update_active_conversation(ctx)
                return

            # Process
            await self._handle_message(ctx, message)

    async def _build_message_context(self, message: discord.Message) -> MessageContext:
        """
        Baut einen vollständigen MessageContext mit strukturierten Attachments.

        Attachments werden als AttachmentInfo-Objekte gespeichert, die:
        - Media-Type Detection enthalten
        - Native LLM-Kompatibilität prüfen
        - [media:] Tags generieren können
        """

        # Source bestimmen
        if isinstance(message.channel, discord.DMChannel):
            source = MessageSource.DM
            source_address = f"discord://dm:{message.author.id}"
            channel_name = "DM"
        elif isinstance(message.channel, discord.Thread):
            source = MessageSource.THREAD
            source_address = f"discord://server:{message.guild.id}/channel:{message.channel.id}"
            channel_name = message.channel.name
        else:
            source = MessageSource.TEXT_CHANNEL
            source_address = f"discord://server:{message.guild.id}/channel:{message.channel.id}"
            channel_name = message.channel.name

        # Content analysieren
        content = message.content
        wants_audio = content.strip().endswith("#audio")
        if wants_audio:
            content = content.rsplit("#audio", 1)[0].strip()

        # Attachments verarbeiten
        attachments: list[AttachmentInfo] = []

        for att in message.attachments:
            att_data = await self.media_handler.process_attachment(att)
            if att_data:
                attachment_info = AttachmentInfo(
                    path=att_data["path"],
                    filename=att_data["filename"],
                    content_type=att_data["content_type"],
                    size=att_data["size"],
                )

                # Audio transkribieren
                if attachment_info.media_type == "audio":
                    transcription = await self.media_handler.transcribe_audio(attachment_info.path)
                    if transcription:
                        attachment_info.transcription = transcription

                attachments.append(attachment_info)

        # Voice Channel Check
        voice_channel_id = None
        voice_channel_name = None
        if message.guild and message.author.voice:
            voice_channel_id = message.author.voice.channel.id
            voice_channel_name = message.author.voice.channel.name

        return MessageContext(
            source=source,
            source_address=source_address,
            user_id=message.author.id,
            user_name=message.author.name,
            user_display_name=message.author.display_name,
            guild_id=message.guild.id if message.guild else None,
            guild_name=message.guild.name if message.guild else None,
            channel_id=message.channel.id,
            channel_name=channel_name,
            content=content,
            message_id=message.id,
            timestamp=message.created_at.isoformat(),
            mentioned_bot=self.bot.user in message.mentions if self.bot.user else False,
            mentioned_users=[u.id for u in message.mentions],
            attachments=attachments,
            is_reply=message.reference is not None,
            reply_to_message_id=message.reference.message_id if message.reference else None,
            voice_channel_id=voice_channel_id,
            voice_channel_name=voice_channel_name,
            wants_audio_response=wants_audio,
        )

    def _should_respond(self, ctx: MessageContext) -> bool:
        """Entscheidet ob der Bot antworten soll"""

        # DMs: immer antworten
        if ctx.source == MessageSource.DM:
            return True

        # Mentions: immer antworten
        if ctx.mentioned_bot:
            return True

        # Wenn respond_to_mentions_only aktiv, nicht antworten
        if self.respond_to_mentions_only:
            return False

        return True

    async def _handle_message(self, ctx: MessageContext, message: discord.Message):
        """Verarbeitet eine Nachricht und ruft den Agent auf"""

        # Typing indicator
        async with message.channel.typing():
            # Update AddressBook
            if self.address_book:
                self.address_book.update_contact(
                    ctx.user_id, ctx.user_name, ctx.user_display_name, ctx.guild_id
                )
                self.address_book.update_active_conversation(ctx)

                if ctx.guild_id:
                    self.address_book.register_channel(
                        ctx.guild_id, ctx.channel_id, ctx.channel_name
                    )
                else:
                    self.address_book.register_dm(ctx.user_id, ctx.user_name)

            # Prepare Auto-Router
            self.router.set_pending_context(ctx)

            try:
                # Build Agent Input
                agent_input = f"{ctx.to_agent_context()}\n\nMessage:\n{ctx.content}"

                # Remove bot mention from content
                if self.bot.user:
                    agent_input = agent_input.replace(f"<@{self.bot.user.id}>", "").strip()

                # Call Agent
                response = await self.agent.a_run(
                    query=agent_input,
                    session_id=f"discord_{ctx.source_address.replace('://', '_').replace('/', '_')}",
                )

                # Route Response
                if response:
                    result = await self.router.route_response(
                        content=response,
                        as_audio=ctx.wants_audio_response,
                    )

                    if not result.get("success"):
                        print(f"[Discord] Failed to route response: {result.get('error')}")

            except Exception as e:
                print(f"[Discord] Error handling message: {e}")
                import traceback
                traceback.print_exc()

                await message.channel.send(
                    "⚠️ I encountered an error processing your message. Please try again."
                )

            finally:
                self.router.clear_pending_context()

    def _register_agent_tools(self):
        """Registriert Discord-spezifische Tools beim Agent"""

        interface = self  # Closure reference

        async def discord_send_message(
            target_address: str,
            content: str,
            reply_to: Optional[int] = None,
            as_audio: bool = False,
        ) -> str:
            """
            Send a message to a specific Discord address.

            Args:
                target_address: Discord address (e.g. "discord://server:123/channel:456" or "discord://dm:789")
                content: Message content
                reply_to: Optional message ID to reply to
                as_audio: If True, send as TTS audio file

            Returns:
                Result JSON with success status
            """
            result = await interface.router.route_response(
                content=content,
                target_address=target_address,
                reply_to=reply_to,
                as_audio=as_audio,
            )
            return json.dumps(result)

        async def discord_search_address(query: str) -> str:
            """
            Search the Discord address book for users, servers, or channels.

            Args:
                query: Search term (user name, channel name, server name)

            Returns:
                JSON list of matching addresses
            """
            if not interface.address_book:
                return json.dumps({"error": "AddressBook not initialized"})

            results = interface.address_book.search(query)
            return json.dumps(results)

        async def discord_get_active_conversations() -> str:
            """
            Get all currently active Discord conversations.

            Returns:
                JSON dict of active conversations
            """
            if not interface.address_book:
                return json.dumps({"error": "AddressBook not initialized"})

            return json.dumps(interface.address_book.get_active_conversations())

        async def discord_get_channel_history(
            channel_id: int,
            limit: int = 10,
        ) -> str:
            """
            Get recent message history from a channel.

            Args:
                channel_id: Discord channel ID
                limit: Number of messages (max 50)

            Returns:
                JSON list of recent messages
            """
            try:
                channel = interface.bot.get_channel(channel_id)
                if not channel:
                    channel = await interface.bot.fetch_channel(channel_id)

                if not channel:
                    return json.dumps({"error": f"Channel {channel_id} not found"})

                messages = []
                async for msg in channel.history(limit=min(limit, 50)):
                    messages.append({
                        "id": msg.id,
                        "author": msg.author.display_name,
                        "author_id": msg.author.id,
                        "content": msg.content[:500],  # Truncate
                        "timestamp": msg.created_at.isoformat(),
                    })

                return json.dumps(messages)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Register Tools
        self.agent.add_tool(
            discord_send_message,
            "discord_send_message",
            description="Send a message to a specific Discord address. Use as_audio=True to send as voice message. Use this to reply to different channels or users than the current conversation.",
            category=["discord", "communication"],
        )

        self.agent.add_tool(
            discord_search_address,
            "discord_search_address",
            description="Search for Discord users, channels, or servers by name. Returns addresses you can use with discord_send_message.",
            category=["discord", "lookup"],
        )

        self.agent.add_tool(
            discord_get_active_conversations,
            "discord_get_active",
            description="Get all active Discord conversations you're currently engaged in.",
            category=["discord", "context"],
        )

        self.agent.add_tool(
            discord_get_channel_history,
            "discord_get_history",
            description="Get recent message history from a Discord channel. Useful for context about what others have said.",
            category=["discord", "context"],
        )

        # VFS-basierte Tools sind automatisch über die Standard VFS Tools verfügbar
        # Der Agent kann /discord/ im VFS erkunden mit vfs_ls, vfs_read etc.
        print("[Discord] Agent tools registered: discord_send_message, discord_search_address, discord_get_active, discord_get_history")
        print("[Discord] Agent can also explore /discord/ folder via VFS tools")

    async def start(self):
        """Startet den Discord Bot"""
        print("[Discord] Starting interface...")
        await self.bot.start(self.token)

    async def stop(self):
        """Stoppt den Discord Bot"""
        print("[Discord] Stopping interface...")
        await self.bot.close()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_discord_interface(
    agent: "FlowAgent",
    token: Optional[str] = None,
    respond_to_mentions_only: bool = True,
    admin_ids: Optional[list[int]] = None,
    tts_backend: str = "groq",  # "piper", "groq", "elevenlabs"
    tts_voice: str = "autumn",
    stt_backend: str = "groq",  # "faster_whisper", "groq"
    language: str = "de",
) -> DiscordInterface:
    """
    Factory für DiscordInterface.

    Args:
        agent: FlowAgent Instanz
        token: Discord Bot Token (oder DISCORD_TOKEN env var)
        respond_to_mentions_only: Nur auf @mentions in Channels antworten
        admin_ids: Liste von Admin User IDs
        tts_backend: TTS Backend ("piper", "groq", "elevenlabs")
        tts_voice: Voice ID für TTS (backend-spezifisch)
        stt_backend: STT Backend ("faster_whisper", "groq")
        language: Sprache für TTS/STT (z.B. "de", "en")

    Returns:
        Konfiguriertes DiscordInterface

    TTS Voices:
        groq: [autumn diana hannah austin daniel troy]
        elevenlabs: "21m00Tcm4TlvDq8ikWAM" (Rachel), etc.
        piper: "de_DE-thorsten-medium", "en_US-amy-medium", etc.

    Example:
        interface = create_discord_interface(
            agent=my_agent,
            tts_backend="groq",
            tts_voice="Zola-PlayAI",  # Weibliche Stimme
            language="de",
        )
        await interface.start()
    """
    token = token or os.environ.get("DISCORD_TOKEN")
    if not token:
        raise ValueError("Discord token required (pass token or set DISCORD_TOKEN env)")

    return DiscordInterface(
        agent=agent,
        token=token,
        respond_to_mentions_only=respond_to_mentions_only,
        admin_ids=admin_ids,
        tts_backend=tts_backend,
        tts_voice=tts_voice,
        stt_backend=stt_backend,
        language=language,
    )
