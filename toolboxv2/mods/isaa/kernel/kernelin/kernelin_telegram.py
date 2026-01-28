"""
Telegram Transport Layer for ProA Kernel
Version: 1.0.0

A DUMB transport layer that:
- Converts Telegram updates → Kernel Signals
- Routes Kernel responses → Telegram messages
- Contains NO business logic

Dependencies: python-telegram-bot, groq (for voice transcription)
"""

import asyncio
import html
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    from telegram import Bot, Message, Update
    from telegram.constants import ChatAction, ParseMode
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        ContextTypes,
        MessageHandler,
        filters,
    )
except ImportError as e:
    print("Install telegram via pip install telegram")
    Update = None
    Bot = None
    Message = None
    Application = None
    ApplicationBuilder = None
    ContextTypes = None
    MessageHandler = None
    filters = None
    ParseMode = None
    ChatAction = None

from toolboxv2.mods.isaa.kernel.types import IOutputRouter, Signal, SignalType, UserState

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.kernel.instace import Kernel

# Optional dependencies
try:
    from groq import AsyncGroq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    AsyncGroq = None


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TelegramConfig:
    """Telegram transport configuration"""

    token: str
    admin_whitelist: list[int] = field(default_factory=list)

    # Voice settings
    voice_language: str = "de"

    # Media settings
    temp_dir: str = "/tmp/telegram_media"
    max_file_size_mb: int = 20

    # Message settings
    parse_mode: str = "MarkdownV2"  # or "HTML"
    disable_web_preview: bool = True


# =============================================================================
# MARKDOWN V2 ESCAPER
# =============================================================================


class MarkdownV2Escaper:
    """
    Handles MarkdownV2 escaping for Telegram.

    MarkdownV2 requires escaping of special characters:
    _ * [ ] ( ) ~ ` > # + - = | { } . !
    """

    SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!"

    @classmethod
    def escape(cls, text: str) -> str:
        """Escape text for MarkdownV2"""
        # Escape all special characters
        for char in cls.SPECIAL_CHARS:
            text = text.replace(char, f"\\{char}")
        return text

    @classmethod
    def escape_code(cls, text: str) -> str:
        """Escape text inside code blocks (only ` and \)"""
        text = text.replace("\\", "\\\\")
        text = text.replace("`", "\\`")
        return text

    @classmethod
    def format_response(cls, text: str) -> str:
        """
        Format agent response for MarkdownV2.
        Preserves code blocks and escapes the rest.
        """
        # Pattern for code blocks
        code_pattern = r"```(\w*)\n?(.*?)```"
        inline_code_pattern = r"`([^`]+)`"

        result = []
        last_end = 0

        # Handle multi-line code blocks first
        for match in re.finditer(code_pattern, text, re.DOTALL):
            # Escape text before code block
            before = text[last_end : match.start()]
            result.append(cls.escape(before))

            # Format code block
            lang = match.group(1) or ""
            code = match.group(2)
            escaped_code = cls.escape_code(code)
            result.append(f"```{lang}\n{escaped_code}```")

            last_end = match.end()

        # Handle remaining text
        remaining = text[last_end:]

        # Handle inline code in remaining text
        inline_result = []
        inline_last_end = 0
        for match in re.finditer(inline_code_pattern, remaining):
            # Escape text before inline code
            before = remaining[inline_last_end : match.start()]
            inline_result.append(cls.escape(before))

            # Format inline code
            code = match.group(1)
            escaped_code = cls.escape_code(code)
            inline_result.append(f"`{escaped_code}`")

            inline_last_end = match.end()

        # Add remaining text after last inline code
        if inline_last_end < len(remaining):
            inline_result.append(cls.escape(remaining[inline_last_end:]))

        result.append("".join(inline_result))

        return "".join(result)


# =============================================================================
# MEDIA HANDLER
# =============================================================================


class TelegramMediaHandler:
    """Handles media downloads and processing"""

    def __init__(self, config: TelegramConfig, bot: Bot):
        self.config = config
        self.bot = bot
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Groq client for transcription
        self._groq: Optional[AsyncGroq] = None
        if GROQ_AVAILABLE:
            groq_key = os.environ.get("GROQ_API_KEY")
            if groq_key:
                self._groq = AsyncGroq(api_key=groq_key)

    async def download_voice(self, voice_file_id: str) -> Optional[str]:
        """Download voice message to temp file"""
        try:
            file = await self.bot.get_file(voice_file_id)

            # Telegram voice messages are OGG format
            filename = f"voice_{int(time.time())}_{voice_file_id[-8:]}.ogg"
            filepath = self.temp_dir / filename

            await file.download_to_drive(filepath)
            return str(filepath)
        except Exception as e:
            print(f"[Telegram] Failed to download voice: {e}")
            return None

    async def download_photo(
        self, photo_file_id: str, filename_hint: str = ""
    ) -> Optional[str]:
        """Download photo to temp file"""
        try:
            file = await self.bot.get_file(photo_file_id)

            ext = ".jpg"  # Telegram photos are usually JPEG
            filename = f"photo_{int(time.time())}_{photo_file_id[-8:]}{ext}"
            filepath = self.temp_dir / filename

            await file.download_to_drive(filepath)
            return str(filepath)
        except Exception as e:
            print(f"[Telegram] Failed to download photo: {e}")
            return None

    async def download_document(self, document) -> Optional[str]:
        """Download document to temp file"""
        try:
            if document.file_size > self.config.max_file_size_mb * 1024 * 1024:
                return None

            file = await self.bot.get_file(document.file_id)

            # Preserve original filename
            original_name = document.file_name or f"doc_{document.file_id[-8:]}"
            filename = f"{int(time.time())}_{original_name}"
            filepath = self.temp_dir / filename

            await file.download_to_drive(filepath)
            return str(filepath)
        except Exception as e:
            print(f"[Telegram] Failed to download document: {e}")
            return None

    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file using Groq Whisper"""
        if not self._groq:
            print("[Telegram] Groq not available for transcription")
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
            print(f"[Telegram] Transcription failed: {e}")
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
# TELEGRAM OUTPUT ROUTER
# =============================================================================


class TelegramOutputRouter(IOutputRouter):
    """Routes Kernel outputs to Telegram"""

    def __init__(self, bot: Bot, config: TelegramConfig):
        self.bot = bot
        self.config = config

        # User -> Chat ID mapping
        self._user_chats: dict[str, int] = {}
        # Typing indicators
        self._typing_tasks: dict[int, asyncio.Task] = {}

    def register_user_chat(self, user_id: str, chat_id: int):
        """Register user's chat ID"""
        self._user_chats[user_id] = chat_id

    async def send_response(
        self, user_id: str, content: str, role: str = "assistant", metadata: dict = None
    ):
        """Send response to user via Telegram"""
        metadata = metadata or {}
        chat_id = metadata.get("chat_id") or self._user_chats.get(user_id)

        if not chat_id:
            print(f"[Telegram] No chat ID for user {user_id}")
            return

        # Stop typing indicator
        self._cancel_typing(chat_id)

        # Format and send
        await self._send_text(chat_id, content)

    async def send_notification(
        self, user_id: str, content: str, priority: int = 5, metadata: dict = None
    ):
        """Send proactive notification"""
        metadata = metadata or {}
        chat_id = metadata.get("chat_id") or self._user_chats.get(user_id)

        if not chat_id:
            return

        # Add priority indicator
        prefix = "🔴" if priority >= 8 else "🟡" if priority >= 5 else "🟢"
        formatted = f"{prefix} *Notification:* {content}"

        await self._send_text(chat_id, formatted)

    async def _send_text(self, chat_id: int, content: str):
        """Send text message with proper formatting"""
        try:
            # Escape for MarkdownV2 if using that mode
            if self.config.parse_mode == "MarkdownV2":
                formatted = MarkdownV2Escaper.format_response(content)
            else:
                formatted = content

            # Split long messages
            if len(formatted) <= 4096:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=formatted,
                    parse_mode=self.config.parse_mode
                    if self.config.parse_mode != "MarkdownV2"
                    else ParseMode.MARKDOWN_V2,
                    disable_web_page_preview=self.config.disable_web_preview,
                )
            else:
                # Split into chunks (simple split, not preserving formatting)
                chunks = [formatted[i : i + 4000] for i in range(0, len(formatted), 4000)]
                for i, chunk in enumerate(chunks):
                    prefix = f"({i + 1}/{len(chunks)}) " if len(chunks) > 1 else ""
                    try:
                        await self.bot.send_message(
                            chat_id=chat_id,
                            text=prefix + chunk,
                            parse_mode=None,  # Plain text for chunked
                            disable_web_page_preview=True,
                        )
                    except Exception:
                        # Fallback to plain text
                        await self.bot.send_message(
                            chat_id=chat_id, text=prefix + chunk, parse_mode=None
                        )
                    await asyncio.sleep(0.5)
        except Exception as e:
            print(f"[Telegram] Failed to send message: {e}")
            # Fallback: send as plain text
            try:
                await self.bot.send_message(
                    chat_id=chat_id, text=content, parse_mode=None
                )
            except Exception as e2:
                print(f"[Telegram] Fallback also failed: {e2}")

    async def send_file(self, user_id: str, filepath: str, caption: str = ""):
        """Send file to user"""
        chat_id = self._user_chats.get(user_id)
        if not chat_id:
            return

        try:
            with open(filepath, "rb") as f:
                await self.bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    caption=caption[:1024] if caption else None,
                )
        except Exception as e:
            print(f"[Telegram] Failed to send file: {e}")

    async def send_photo(self, user_id: str, filepath: str, caption: str = ""):
        """Send photo to user"""
        chat_id = self._user_chats.get(user_id)
        if not chat_id:
            return

        try:
            with open(filepath, "rb") as f:
                await self.bot.send_photo(
                    chat_id=chat_id, photo=f, caption=caption[:1024] if caption else None
                )
        except Exception as e:
            print(f"[Telegram] Failed to send photo: {e}")

    async def start_typing(self, chat_id: int):
        """Start typing indicator"""
        self._cancel_typing(chat_id)

        async def typing_loop():
            try:
                while True:
                    await self.bot.send_chat_action(chat_id, ChatAction.TYPING)
                    await asyncio.sleep(4)  # Typing indicator lasts ~5s
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._typing_tasks[chat_id] = asyncio.create_task(typing_loop())

    def _cancel_typing(self, chat_id: int):
        """Cancel typing indicator"""
        if chat_id in self._typing_tasks:
            self._typing_tasks[chat_id].cancel()
            del self._typing_tasks[chat_id]


# =============================================================================
# TELEGRAM TRANSPORT
# =============================================================================


class TelegramTransport:
    """
    Telegram Transport Layer for ProA Kernel

    Responsibilities:
    1. Convert Telegram updates → Kernel Signals
    2. Route Kernel outputs → Telegram messages
    3. Handle voice notes (transcription)
    4. Handle photos and documents

    NO business logic - just transport!
    """

    def __init__(
        self,
        config: TelegramConfig,
        kernel: "Kernel",
        identity_map: Optional[dict] = None,
    ):
        self.config = config
        self.kernel = kernel
        self.identity_map = identity_map or {}

        # Build application
        self.app = ApplicationBuilder().token(config.token).build()
        self.bot = self.app.bot

        # Setup handlers
        self.media_handler = TelegramMediaHandler(config, self.bot)
        self.output_router = TelegramOutputRouter(self.bot, config)

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register Telegram message handlers"""
        # Text messages
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        # Voice messages
        self.app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice)
        )

        # Photos
        self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))

        # Documents
        self.app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))

        # Video notes (circular videos)
        self.app.add_handler(MessageHandler(filters.VIDEO_NOTE, self._handle_video_note))

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        if not self.config.admin_whitelist:
            return True
        return user_id in self.config.admin_whitelist

    def _resolve_user_id(self, telegram_id: int) -> str:
        """Resolve Telegram ID to unified user ID"""
        telegram_key = f"telegram:{telegram_id}"
        if telegram_key in self.identity_map:
            return self.identity_map[telegram_key]
        return f"telegram_{telegram_id}"

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        message = update.message
        if not message or not message.text:
            return

        if not self._is_authorized(message.from_user.id):
            return  # Silent ignore

        user_id = self._resolve_user_id(message.from_user.id)

        # Register chat for responses
        self.output_router.register_user_chat(user_id, message.chat_id)

        # Start typing indicator
        await self.output_router.start_typing(message.chat_id)

        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "telegram",
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "author_name": message.from_user.full_name,
            "username": message.from_user.username,
            "voice_input": False,
            "attachments": [],
        }

        # Send to kernel
        try:
            await self.kernel.handle_user_input(
                user_id=user_id, content=message.text, metadata=metadata
            )
        except Exception as e:
            print(f"[Telegram] Kernel error: {e}")
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text(
                "⚠️ I'm having trouble thinking right now. Please try again."
            )

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages"""
        message = update.message
        if not message:
            return

        voice = message.voice or message.audio
        if not voice:
            return

        if not self._is_authorized(message.from_user.id):
            return

        user_id = self._resolve_user_id(message.from_user.id)
        self.output_router.register_user_chat(user_id, message.chat_id)

        # Start typing
        await self.output_router.start_typing(message.chat_id)

        # Download and transcribe
        audio_path = await self.media_handler.download_voice(voice.file_id)
        if not audio_path:
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text("⚠️ Failed to process voice message.")
            return

        transcription = await self.media_handler.transcribe_audio(audio_path)
        if not transcription:
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text("⚠️ Failed to transcribe voice message.")
            return

        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "telegram",
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "author_name": message.from_user.full_name,
            "voice_input": True,
            "original_audio_path": audio_path,
            "attachments": [],
        }

        # Send to kernel
        try:
            await self.kernel.handle_user_input(
                user_id=user_id, content=transcription, metadata=metadata
            )
        except Exception as e:
            print(f"[Telegram] Kernel error: {e}")
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text("⚠️ I'm having trouble thinking right now.")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages"""
        message = update.message
        if not message or not message.photo:
            return

        if not self._is_authorized(message.from_user.id):
            return

        user_id = self._resolve_user_id(message.from_user.id)
        self.output_router.register_user_chat(user_id, message.chat_id)

        await self.output_router.start_typing(message.chat_id)

        # Get largest photo
        photo = message.photo[-1]
        photo_path = await self.media_handler.download_photo(photo.file_id)

        if not photo_path:
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text("⚠️ Failed to download photo.")
            return

        # Build content
        caption = message.caption or ""
        content = (
            f"{caption}\n\n[System: User uploaded image at {photo_path}]"
            if caption
            else f"[System: User uploaded image at {photo_path}]"
        )

        metadata = {
            "user_id": user_id,
            "source": "telegram",
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "author_name": message.from_user.full_name,
            "voice_input": False,
            "attachments": [
                {
                    "path": photo_path,
                    "type": "photo",
                    "width": photo.width,
                    "height": photo.height,
                }
            ],
        }

        try:
            await self.kernel.handle_user_input(
                user_id=user_id, content=content, metadata=metadata
            )
        except Exception as e:
            print(f"[Telegram] Kernel error: {e}")
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text("⚠️ I'm having trouble processing this image.")

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document messages"""
        message = update.message
        if not message or not message.document:
            return

        if not self._is_authorized(message.from_user.id):
            return

        user_id = self._resolve_user_id(message.from_user.id)
        self.output_router.register_user_chat(user_id, message.chat_id)

        await self.output_router.start_typing(message.chat_id)

        doc_path = await self.media_handler.download_document(message.document)

        if not doc_path:
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text(
                "⚠️ Failed to download document (too large or error)."
            )
            return

        # Build content
        caption = message.caption or ""
        doc_name = message.document.file_name or "document"
        content = (
            f"{caption}\n\n[System: User uploaded document '{doc_name}' at {doc_path}]"
            if caption
            else f"[System: User uploaded document '{doc_name}' at {doc_path}]"
        )

        metadata = {
            "user_id": user_id,
            "source": "telegram",
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "author_name": message.from_user.full_name,
            "voice_input": False,
            "attachments": [
                {
                    "path": doc_path,
                    "type": "document",
                    "filename": doc_name,
                    "mime_type": message.document.mime_type,
                    "size": message.document.file_size,
                }
            ],
        }

        try:
            await self.kernel.handle_user_input(
                user_id=user_id, content=content, metadata=metadata
            )
        except Exception as e:
            print(f"[Telegram] Kernel error: {e}")
            self.output_router._cancel_typing(message.chat_id)
            await message.reply_text("⚠️ I'm having trouble processing this document.")

    async def _handle_video_note(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle video notes (circular videos)"""
        message = update.message
        if not message or not message.video_note:
            return

        if not self._is_authorized(message.from_user.id):
            return

        user_id = self._resolve_user_id(message.from_user.id)
        self.output_router.register_user_chat(user_id, message.chat_id)

        # For now, just acknowledge - could extract audio for transcription
        await message.reply_text(
            "📹 Video note received. I can see that you sent a video, but I can't process video content directly yet."
        )

    async def start(self):
        """Start the Telegram bot"""
        print("[Telegram] Starting transport...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        print("[Telegram] Transport running")

    async def stop(self):
        """Stop the Telegram bot"""
        print("[Telegram] Stopping transport...")
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
        print("[Telegram] Transport stopped")

    def get_router(self) -> TelegramOutputRouter:
        """Get the output router for kernel integration"""
        return self.output_router

    async def send_direct_message(self, chat_id: int, text: str):
        """
        Send a direct message to a chat (for proactive notifications from Kernel).
        This bypasses the output router for direct Kernel->Telegram communication.
        """
        try:
            await self.bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            print(f"[Telegram] Direct message failed: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_telegram_transport(
    kernel: "Kernel",
    token: str,
    admin_ids: list[int],
    identity_map: Optional[dict] = None,
    **config_kwargs,
) -> TelegramTransport:
    """
    Factory function to create Telegram transport.

    Args:
        kernel: ProA Kernel instance
        token: Telegram bot token
        admin_ids: List of authorized Telegram user IDs
        identity_map: Optional mapping of telegram IDs to unified user IDs
        **config_kwargs: Additional TelegramConfig options

    Returns:
        Configured TelegramTransport instance
    """
    config = TelegramConfig(token=token, admin_whitelist=admin_ids, **config_kwargs)

    return TelegramTransport(config, kernel, identity_map)


# =============================================================================
# STANDALONE RUNNER (for testing)
# =============================================================================


async def run_telegram_standalone(kernel: "Kernel", token: str, admin_ids: list[int]):
    """Run Telegram transport standalone (for testing)"""
    transport = create_telegram_transport(kernel, token, admin_ids)

    # Register router with kernel
    kernel.output_router = transport.get_router()

    try:
        await kernel.start()
        await transport.start()

        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await transport.stop()
        await kernel.stop()
