"""
Discord Transport Layer for ProA Kernel
Version: 1.0.0

A DUMB transport layer that:
- Converts Discord events → Kernel Signals
- Routes Kernel responses → Discord messages/voice
- Contains NO business logic

Dependencies: discord.py, groq (for voice transcription)
"""

import asyncio
import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    import discord
    from discord import VoiceClient
    from discord.ext import commands
except ImportError:
    print("pip install discord.py")
    discord = lambda: None
    discord.Attachment = None
    discord.Forbidden = Exception
    discord.TextChannel = str
    commands = None
    VoiceClient = None

from toolboxv2.mods.isaa.kernel.types import IOutputRouter, Signal, SignalType, UserState

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.kernel.instace import Kernel

# Optional voice dependencies
try:
    from groq import AsyncGroq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    AsyncGroq = None

try:
    import pyttsx3

    TTS_LOCAL_AVAILABLE = True
except ImportError:
    TTS_LOCAL_AVAILABLE = False

# =============================================================================
# CONFIGURATION
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


# =============================================================================
# VOICE HANDLER
# =============================================================================


class VoiceHandler:
    """Handles Discord voice channel interactions"""

    def __init__(
        self, bot: commands.Bot, config: DiscordConfig, media_handler: MediaHandler
    ):
        self.bot = bot
        self.config = config
        self.media_handler = media_handler

        # Voice state
        self._voice_clients: dict[int, VoiceClient] = {}  # guild_id -> VoiceClient
        self._audio_buffers: dict[int, bytearray] = {}  # user_id -> audio buffer
        self._last_audio_time: dict[int, float] = {}  # user_id -> timestamp
        self._silence_tasks: dict[int, asyncio.Task] = {}

        # Callback for transcribed audio
        self._on_transcription: Optional[callable] = None

    def set_transcription_callback(self, callback: callable):
        """Set callback for when audio is transcribed"""
        self._on_transcription = callback

    async def join_channel(self, channel: discord.VoiceChannel) -> bool:
        """Join a voice channel"""
        try:
            if channel.guild.id in self._voice_clients:
                await self._voice_clients[channel.guild.id].disconnect()

            vc = await channel.connect()
            self._voice_clients[channel.guild.id] = vc

            # Start listening (requires voice_recv)
            # Note: discord.py doesn't have built-in voice receive
            # This would require a custom sink or library like discord-ext-voice-recv

            return True
        except Exception as e:
            print(f"[Discord] Failed to join voice channel: {e}")
            return False

    async def leave_channel(self, guild_id: int):
        """Leave voice channel"""
        if guild_id in self._voice_clients:
            await self._voice_clients[guild_id].disconnect()
            del self._voice_clients[guild_id]

    def get_connected_channel(self, guild_id: int) -> Optional[discord.VoiceChannel]:
        """Get currently connected voice channel for guild"""
        vc = self._voice_clients.get(guild_id)
        if vc and vc.is_connected():
            return vc.channel
        return None

    def is_connected(self, guild_id: int) -> bool:
        """Check if connected to voice in guild"""
        vc = self._voice_clients.get(guild_id)
        return vc is not None and vc.is_connected()

    async def speak_text(self, guild_id: int, text: str):
        """Convert text to speech and play in voice channel"""
        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return

        # Generate TTS audio
        audio_path = await self._generate_tts(text)
        if not audio_path:
            return

        try:
            # Play audio
            source = discord.FFmpegPCMAudio(audio_path)
            vc.play(source)

            # Wait for playback to finish
            while vc.is_playing():
                await asyncio.sleep(0.1)
        finally:
            # Cleanup
            try:
                Path(audio_path).unlink()
            except Exception:
                pass

    async def _generate_tts(self, text: str) -> Optional[str]:
        """Generate TTS audio file"""
        if self.config.tts_provider == "local" and TTS_LOCAL_AVAILABLE:
            return await self._tts_local(text)
        elif self.config.tts_provider == "elevenlabs" and self.config.elevenlabs_api_key:
            return await self._tts_elevenlabs(text)
        return None

    async def _tts_local(self, text: str) -> Optional[str]:
        """Generate TTS using local pyttsx3"""
        try:
            import pyttsx3

            engine = pyttsx3.init()

            filepath = self.media_handler.temp_dir / f"tts_{int(time.time())}.wav"
            engine.save_to_file(text, str(filepath))
            engine.runAndWait()

            return str(filepath)
        except Exception as e:
            print(f"[Discord] Local TTS failed: {e}")
            return None

    async def _tts_elevenlabs(self, text: str) -> Optional[str]:
        """Generate TTS using ElevenLabs API"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.elevenlabs_voice_id}",
                    headers={
                        "xi-api-key": self.config.elevenlabs_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    },
                )

                if response.status_code == 200:
                    filepath = self.media_handler.temp_dir / f"tts_{int(time.time())}.mp3"
                    filepath.write_bytes(response.content)
                    return str(filepath)
        except Exception as e:
            print(f"[Discord] ElevenLabs TTS failed: {e}")
        return None


# =============================================================================
# DISCORD OUTPUT ROUTER
# =============================================================================


class DiscordOutputRouter(IOutputRouter):
    """Routes Kernel outputs to Discord"""

    def __init__(
        self,
        bot: commands.Bot,
        voice_handler: VoiceHandler,
        default_channel_id: Optional[int] = None,
    ):
        self.bot = bot
        self.voice_handler = voice_handler
        self.default_channel_id = default_channel_id

        # User -> Channel mapping (last interaction channel)
        self._user_channels: dict[str, int] = {}
        # User -> Guild mapping for voice
        self._user_guilds: dict[str, int] = {}
        # User voice mode preference
        self._user_voice_mode: dict[str, bool] = {}

    def register_user_channel(
        self, user_id: str, channel_id: int, guild_id: Optional[int] = None
    ):
        """Register which channel a user last interacted in"""
        self._user_channels[user_id] = channel_id
        if guild_id:
            self._user_guilds[user_id] = guild_id

    def set_voice_mode(self, user_id: str, enabled: bool):
        """Set whether user prefers voice responses"""
        self._user_voice_mode[user_id] = enabled

    async def send_response(
        self, user_id: str, content: str, role: str = "assistant", metadata: dict = None
    ):
        """Send response to user via Discord"""
        metadata = metadata or {}
        channel_id = (
            metadata.get("channel_id")
            or self._user_channels.get(user_id)
            or self.default_channel_id
        )
        print(
            f"[Discord] Sending response to {channel_id} {user_id}",
            metadata.get("channel_id"),
            self._user_channels.get(user_id),
            self.default_channel_id,
        )
        if not channel_id:
            print(f"[Discord] No channel for user {user_id}")
            return

        # Try get_channel first (works for guild channels)
        channel = self.bot.get_channel(channel_id)

        # If not found, try fetch_channel (works for DMs and uncached channels)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except Exception as e:
                print(f"[Discord] Failed to fetch channel {channel_id}: {e}")
                return

        if not channel:
            print(f"[Discord] Channel {channel_id} not found")
            return

        # Check voice mode
        guild_id = self._user_guilds.get(user_id)
        use_voice = self._user_voice_mode.get(user_id, False)

        if use_voice and guild_id and self.voice_handler.is_connected(guild_id):
            # Speak via voice
            await self.voice_handler.speak_text(guild_id, content)
            # Also send text as backup
            await self._send_text(channel, content)
        else:
            # Text only
            await self._send_text(channel, content)

    async def send_notification(
        self, user_id: str, content: str, priority: int = 5, metadata: dict = None
    ):
        """Send proactive notification"""
        metadata = metadata or {}
        channel_id = (
            metadata.get("channel_id")
            or self._user_channels.get(user_id)
            or self.default_channel_id
        )

        print(
            f"[Discord] Sending response to {channel_id} {user_id}",
            metadata.get("channel_id"),
            self._user_channels.get(user_id),
            self.default_channel_id,
        )
        if not channel_id:
            print(f"[Discord] No channel for user {user_id}")
            return

        # Try get_channel first, then fetch_channel for DMs
        channel = self.bot.get_channel(channel_id)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except Exception as e:
                print(f"[Discord] Failed to fetch channel {channel_id} - {e}")
                return

        if not channel:
            print(f"[Discord] No channel for user {user_id}!")
            return

        # Add priority indicator
        prefix = "🔴" if priority >= 8 else "🟡" if priority >= 5 else "🟢"
        formatted = f"{prefix} **Notification:** {content}"

        await self._send_text(channel, formatted)

    async def _send_text(self, channel: discord.TextChannel, content: str):
        """Send text message, splitting if necessary"""
        if len(content) <= 2000:
            await channel.send(content)
        else:
            # Split into chunks
            chunks = [content[i : i + 1990] for i in range(0, len(content), 1990)]
            for i, chunk in enumerate(chunks):
                prefix = f"({i + 1}/{len(chunks)}) " if len(chunks) > 1 else ""
                await channel.send(prefix + chunk)
                await asyncio.sleep(0.5)  # Rate limit safety

    async def send_file(
        self,
        user_id: str,
        filepath: str,
        filename: Optional[str] = None,
        content: str = "",
    ):
        """Send file to user"""
        channel_id = self._user_channels.get(user_id) or self.default_channel_id
        if not channel_id:
            return

        channel = self.bot.get_channel(channel_id)
        if not channel:
            return

        try:
            file = discord.File(filepath, filename=filename or Path(filepath).name)
            await channel.send(content=content, file=file)
        except Exception as e:
            print(f"[Discord] Failed to send file: {e}")


# =============================================================================
# DISCORD TRANSPORT
# =============================================================================


class DiscordTransport:
    """
    Discord Transport Layer for ProA Kernel

    Responsibilities:
    1. Convert Discord events → Kernel Signals
    2. Route Kernel outputs → Discord
    3. Handle voice channels and TTS
    4. Manage media attachments

    NO business logic - just transport!
    """

    def __init__(
        self, config: DiscordConfig, kernel: "Kernel", identity_map: Optional[dict] = None
    ):
        self.config = config
        self.kernel = kernel
        self.identity_map = identity_map or {}

        # Setup bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        intents.members = True

        self.bot = commands.Bot(
            command_prefix=config.command_prefix,
            intents=intents,
            help_command=None,  # No help command
        )

        # Setup handlers
        self.media_handler = MediaHandler(config)
        self.voice_handler = VoiceHandler(self.bot, config, self.media_handler)
        self.output_router = DiscordOutputRouter(self.bot, self.voice_handler)

        # Register events
        self._register_events()

    def _register_events(self):
        """Register Discord event handlers"""

        @self.bot.event
        async def on_ready():
            print(f"[Discord] Bot ready: {self.bot.user}")
            print(f"[Discord] Whitelisted users: {self.config.admin_whitelist}")

        @self.bot.event
        async def on_message(message: discord.Message):
            await self._handle_message(message)

        @self.bot.event
        async def on_voice_state_update(
            member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
        ):
            await self._handle_voice_state(member, before, after)

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        if not self.config.admin_whitelist:
            return True  # No whitelist = allow all
        return user_id in self.config.admin_whitelist

    def _resolve_user_id(self, discord_id: int) -> str:
        """Resolve Discord ID to unified user ID"""
        # Check identity map
        discord_key = f"discord:{discord_id}"
        if discord_key in self.identity_map:
            return self.identity_map[discord_key]

        # Default: use discord ID
        return f"discord_{discord_id}"

    async def _handle_message(self, message: discord.Message):
        """Handle incoming Discord message"""
        # Ignore bot messages
        if message.author.bot:
            return

        # Check authorization
        if not self._is_authorized(message.author.id):
            return  # Silent ignore

        user_id = self._resolve_user_id(message.author.id)

        # Register channel for responses
        self.output_router.register_user_channel(
            user_id, message.channel.id, message.guild.id if message.guild else None
        )

        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "discord",
            "channel_id": message.channel.id,
            "guild_id": message.guild.id if message.guild else None,
            "message_id": message.id,
            "author_name": str(message.author),
            "voice_input": False,
            "attachments": [],
        }

        # Check if bot is in voice channel
        if message.guild:
            vc_channel = self.voice_handler.get_connected_channel(message.guild.id)
            if vc_channel:
                metadata["bot_voice_channel"] = vc_channel.name
                metadata["bot_voice_channel_id"] = vc_channel.id

        # Handle attachments
        attachment_paths = []
        for attachment in message.attachments:
            path = await self.media_handler.download_attachment(attachment)
            if path:
                attachment_paths.append(path)
                metadata["attachments"].append(
                    {
                        "path": path,
                        "filename": attachment.filename,
                        "content_type": attachment.content_type,
                        "size": attachment.size,
                    }
                )

        # Build content
        content = message.content
        if attachment_paths:
            attachment_info = "\n".join(
                [f"[System: User uploaded file at {p}]" for p in attachment_paths]
            )
            content = f"{content}\n\n{attachment_info}" if content else attachment_info

        # Skip empty messages
        if not content.strip():
            return

        # Send to kernel
        try:
            await self.kernel.handle_user_input(
                user_id=user_id, content=content, metadata=metadata
            )
        except Exception as e:
            print(f"[Discord] Kernel error: {e}")
            await message.channel.send(
                "⚠️ I'm having trouble thinking right now. Please try again."
            )

    async def _handle_voice_state(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ):
        """Handle voice state changes"""
        if member.bot:
            return

        if not self._is_authorized(member.id):
            return

        user_id = self._resolve_user_id(member.id)

        # User joined voice channel
        if after.channel and (not before.channel or before.channel != after.channel):
            # Update state monitor
            await self.kernel.set_user_location(
                user_id, f"discord_voice:{after.channel.name}"
            )

            # Auto-join if configured and same channel as user
            # (Optional: could be controlled by user preference)

        # User left voice channel
        if before.channel and not after.channel:
            await self.kernel.set_user_location(user_id, "discord_text")

    async def start(self):
        """Start the Discord bot"""
        print("[Discord] Starting transport...")
        await self.bot.start(self.config.token)

    async def stop(self):
        """Stop the Discord bot"""
        print("[Discord] Stopping transport...")

        # Leave all voice channels
        for guild_id in list(self.voice_handler._voice_clients.keys()):
            await self.voice_handler.leave_channel(guild_id)

        await self.bot.close()

    def get_router(self) -> DiscordOutputRouter:
        """Get the output router for kernel integration"""
        return self.output_router


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_discord_transport(
    kernel: "Kernel",
    token: str,
    admin_ids: list[int],
    identity_map: Optional[dict] = None,
    **config_kwargs,
) -> DiscordTransport:
    """
    Factory function to create Discord transport.

    Args:
        kernel: ProA Kernel instance
        token: Discord bot token
        admin_ids: List of authorized Discord user IDs
        identity_map: Optional mapping of discord IDs to unified user IDs
        **config_kwargs: Additional DiscordConfig options

    Returns:
        Configured DiscordTransport instance
    """
    config = DiscordConfig(token=token, admin_whitelist=admin_ids, **config_kwargs)

    return DiscordTransport(config, kernel, identity_map)


# =============================================================================
# STANDALONE RUNNER (for testing)
# =============================================================================


async def run_discord_standalone(kernel: "Kernel", token: str, admin_ids: list[int]):
    """Run Discord transport standalone (for testing)"""
    transport = create_discord_transport(kernel, token, admin_ids)

    # Register router with kernel
    # Note: In production, use MultiChannelRouter
    kernel.output_router = transport.get_router()

    try:
        await asyncio.gather(kernel.start(), transport.start())
    except KeyboardInterrupt:
        pass
    finally:
        await transport.stop()
        await kernel.stop()
