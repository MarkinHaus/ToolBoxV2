"""Telegram Interface for ISAA.

Adapted from discord_interface.py — same security model, same session isolation.
Uses ISAA_Host.run_agent_monitored() for agent execution (streaming + dashboard).

Architecture:
    ISAA_Host (icli) → run_agent_monitored(agent_name, query, session_id)
    moderator agent (public, safe-mode) ← non-admin users
    self_agent (full access)             ← admin users via /agent switch
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.flows.isaa.icli import ISAA_Host

logger = logging.getLogger(__name__)

# ─── Lazy import telegram ─────────────────────────────────────────────
def _import_telegram():
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
        )
        return Update, Application, CommandHandler, ContextTypes, MessageHandler, filters
    except ImportError:
        return None, None, None, None, None, None


# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════

class MessageSource(Enum):
    PRIVATE = "private"
    GROUP = "group"


@dataclass
class MessageContext:
    """Normalized message context — platform-agnostic."""
    user_id: int
    user_name: str
    user_display_name: str
    chat_id: int
    chat_name: str
    content: str
    source: MessageSource
    mentioned_bot: bool
    source_address: str  # "telegram://chat:12345"
    raw_update: Any = None

    def to_agent_context(self) -> str:
        return (
            f"[Telegram] {self.user_display_name} ({self.user_name}) "
            f"in {self.chat_name}: {self.content[:200]}"
        )

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "source": self.source.value,
            "source_address": self.source_address,
        }


# ═══════════════════════════════════════════════════════════════════════
# AddressBook
# ═══════════════════════════════════════════════════════════════════════

class AddressBook:
    """VFS-backed address book for Telegram contacts and chats."""

    def __init__(self, vfs: Any = None):
        self._vfs = vfs
        self._contacts: dict[int, dict] = {}
        self._chats: dict[int, dict] = {}
        self._active: dict[str, dict] = {}

    def register_chat(self, chat_id: int, chat_name: str, chat_type: str):
        self._chats[chat_id] = {"chat_id": chat_id, "name": chat_name, "type": chat_type}

    def register_user(self, user_id: int, user_name: str, display_name: str = None):
        self._contacts[user_id] = {
            "user_id": user_id,
            "username": user_name,
            "display_name": display_name or user_name,
        }

    def update_active_conversation(self, ctx: MessageContext):
        self._active[ctx.source_address] = {**ctx.to_dict(), "last_active": time.time()}

    def get_active_conversations(self) -> dict:
        return self._active

    def search(self, query: str) -> list[dict]:
        q = query.lower()
        results = []
        for uid, info in self._contacts.items():
            if q in info.get("username", "").lower() or q in info.get("display_name", "").lower():
                results.append({"type": "user", **info})
        for cid, info in self._chats.items():
            if q in info.get("name", "").lower():
                results.append({"type": "chat", **info})
        return results


# ═══════════════════════════════════════════════════════════════════════
# TelegramInterface
# ═══════════════════════════════════════════════════════════════════════

class TelegramInterface:
    """Telegram bot interface for ISAA.

    Uses ISAA_Host.run_agent_monitored() for agent execution.
    Security model (identical to Discord):
    1. moderator agent: public, safe-mode (only telegram_* + safelist tools)
    2. self agent: full access, admin-only
    3. Session isolation per user/chat
    """

    MODERATOR_SAFE_TOOLS: set[str] = {"search_web"}

    def __init__(
        self,
        icli_host: "ISAA_Host",
        token: str,
        moderator_agent_name: str = "moderator",
        self_agent_name: str = "self",
        respond_to_groups_only_when_mentioned: bool = True,
        admin_ids: Optional[list[int]] = None,
        moderator_safelist: Optional[set[str]] = None,
        bot_username: Optional[str] = None,
    ):
        self.icli = icli_host
        self.token = token
        self.moderator_agent_name = moderator_agent_name
        self.self_agent_name = self_agent_name
        self.respond_to_groups_only_when_mentioned = respond_to_groups_only_when_mentioned
        self.admin_ids = admin_ids or []
        self.bot_username = bot_username
        self.moderator_safelist = (
            moderator_safelist if moderator_safelist is not None
            else set(self.MODERATOR_SAFE_TOOLS)
        )
        self.user_prefs: dict[int, dict[str, str]] = {}
        self.address_book = AddressBook()
        self._application = None

        # Apply safe mode to moderator agent
        self._apply_moderator_safe_mode()

    # ─── Security: Moderator Safe Mode ────────────────────────────────

    def _apply_moderator_safe_mode(self):
        """Wrap moderator agent's init_session_tools so that after each session-init
        every tool whose name lacks 'telegram' and is not in the safelist is unregistered.

        Identical mechanism to DiscordInterface._apply_moderator_safe_mode.
        """
        safelist = self.moderator_safelist

        async def _apply():
            try:
                agent = await self.icli.isaa_tools.get_agent(self.moderator_agent_name)
                orig_init = agent.init_session_tools

                def safe_init(session):
                    orig_init(session)
                    for name in list(agent.tool_manager.list_names()):
                        if "telegram" in name.lower() or name in safelist:
                            continue
                        agent.remove_tool(name)

                agent.init_session_tools = safe_init
                logger.info(
                    f"[Telegram] Moderator safe-mode active: only telegram* tools "
                    f"+ safelist {sorted(safelist)} survive session init."
                )
            except Exception as e:
                logger.warning(f"[Telegram] Could not apply safe-mode yet: {e}")

        # Schedule safe-mode application (agent may not exist yet)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_apply())
            else:
                loop.run_until_complete(_apply())
        except RuntimeError:
            asyncio.ensure_future(_apply())

    # ─── Session Isolation ────────────────────────────────────────────

    def _isolated_session_id(self, ctx: MessageContext) -> str:
        return f"tg_{ctx.chat_id}_{ctx.user_id}"

    def _resolve_route(self, ctx: MessageContext) -> tuple[str, str]:
        """Determine which agent + session to route the message to."""
        pref = self.user_prefs.get(ctx.user_id, {})
        use_self = pref.get("agent") == "self" and ctx.user_id in self.admin_ids
        agent_name = self.self_agent_name if use_self else self.moderator_agent_name
        use_default = ctx.user_id in self.admin_ids and pref.get("session") == "default"
        session_id = "default" if use_default else self._isolated_session_id(ctx)
        return agent_name, session_id

    # ─── Response Decision ────────────────────────────────────────────

    def _should_respond(self, ctx: MessageContext) -> bool:
        if ctx.source == MessageSource.PRIVATE:
            return True
        if ctx.mentioned_bot:
            return True
        if self.respond_to_groups_only_when_mentioned:
            return False
        return True

    # ─── Admin Commands ───────────────────────────────────────────────

    def _handle_admin_command(self, ctx: MessageContext) -> Optional[str]:
        parts = ctx.content.strip().split()
        cmd = parts[0].lower().removeprefix("/") if parts else ""

        if cmd == "agent":
            if len(parts) < 2:
                pref = self.user_prefs.get(ctx.user_id, {})
                current = pref.get("agent", "moderator")
                return f"Current agent: {current}. Usage: /agent [moderator|self]"
            choice = parts[1].lower()
            if choice == "self" and ctx.user_id in self.admin_ids:
                self.user_prefs.setdefault(ctx.user_id, {})["agent"] = "self"
                return "Switched to self-agent (full access)."
            elif choice == "moderator":
                self.user_prefs.setdefault(ctx.user_id, {})["agent"] = "moderator"
                return "Switched to moderator agent (safe mode)."
            return "Invalid choice. Use: /agent [moderator|self]"

        elif cmd == "session":
            if len(parts) < 2:
                pref = self.user_prefs.get(ctx.user_id, {})
                current = pref.get("session", "isolated")
                return f"Current session: {current}. Usage: /session [isolated|default]"
            choice = parts[1].lower()
            if choice in ("isolated", "default"):
                self.user_prefs.setdefault(ctx.user_id, {})["session"] = choice
                return f"Session mode: {choice}"
            return "Invalid choice. Use: /session [isolated|default]"

        elif cmd == "whoami":
            is_admin = ctx.user_id in self.admin_ids
            return (
                f"User: {ctx.user_display_name} ({ctx.user_name})\n"
                f"ID: {ctx.user_id}\n"
                f"Admin: {'yes' if is_admin else 'no'}\n"
                f"Chat: {ctx.chat_name} ({ctx.chat_id})"
            )

        return None

    # ─── Message Handling (uses run_agent_monitored) ──────────────────

    async def _build_message_context(self, update) -> MessageContext:
        msg = update.message or update.edited_message
        user = update.effective_user
        chat = update.effective_chat

        content = msg.text or msg.caption or ""

        mentioned = False
        if self.bot_username:
            mentioned = f"@{self.bot_username}" in content
        if chat.type == "private":
            mentioned = True

        chat_type = MessageSource.PRIVATE if chat.type == "private" else MessageSource.GROUP

        return MessageContext(
            user_id=user.id,
            user_name=user.username or str(user.id),
            user_display_name=user.full_name or user.username or str(user.id),
            chat_id=chat.id,
            chat_name=chat.title or user.full_name or str(chat.id),
            content=content,
            source=chat_type,
            mentioned_bot=mentioned,
            source_address=f"telegram://chat:{chat.id}",
            raw_update=update,
        )

    async def _handle_message(self, ctx: MessageContext) -> str:
        """Process a message via run_agent_monitored and collect the response."""
        # Update address book
        self.address_book.register_user(ctx.user_id, ctx.user_name, ctx.user_display_name)
        self.address_book.register_chat(ctx.chat_id, ctx.chat_name, ctx.source.value)
        self.address_book.update_active_conversation(ctx)

        # Admin commands (private chats only)
        if (
            ctx.source == MessageSource.PRIVATE
            and ctx.content.strip().startswith("/")
            and ctx.user_id in self.admin_ids
        ):
            reply = self._handle_admin_command(ctx)
            if reply is not None:
                return reply

        # Resolve agent + session
        agent_name, session_id = self._resolve_route(ctx)

        # Build context prefix for the agent
        context_prefix = ctx.to_agent_context()

        # Use run_agent_monitored for streaming agent execution
        full_response: list[str] = []

        try:
            async for chunk in self.icli.run_agent_monitored(
                agent_name=agent_name,
                query=ctx.content,
                kind="telegram",
                session_id=session_id,
                take_focus=False,
                system_prompt_override=None,
            ):
                chunk_type = chunk.get("type", "")
                if chunk_type == "text":
                    full_response.append(chunk.get("content", ""))
                elif chunk_type == "final":
                    content = chunk.get("content", "")
                    if content:
                        full_response.append(content)
                elif chunk_type == "error":
                    full_response.append(f"[Error: {chunk.get('message', '')}]")
        except Exception as e:
            logger.error(f"[Telegram] run_agent_monitored error: {e}")
            return f"Error processing message: {e}"

        return "".join(full_response).strip() or "[no response]"

    # ─── Agent Tool Registration ──────────────────────────────────────

    async def _register_agent_tools(self):
        """Register Telegram-specific tools on the moderator agent."""
        interface = self

        async def telegram_send_message(
            target_chat_id: int,
            content: str,
            reply_to_message_id: Optional[int] = None,
        ) -> str:
            """Send a message to a specific Telegram chat."""
            try:
                app = interface._application
                if app is None:
                    return json.dumps({"error": "Bot not running"})
                chunks = [content[i:i+4096] for i in range(0, len(content), 4096)]
                results = []
                for chunk_text in chunks:
                    msg = await app.bot.send_message(
                        chat_id=target_chat_id,
                        text=chunk_text,
                        reply_to_message_id=reply_to_message_id,
                    )
                    results.append({"message_id": msg.message_id})
                return json.dumps({"success": True, "messages": results})
            except Exception as e:
                return json.dumps({"error": str(e)})

        async def telegram_search_address(query: str) -> str:
            """Search for Telegram users or chats by name."""
            return json.dumps(interface.address_book.search(query))

        async def telegram_get_active_conversations() -> str:
            """Get all active Telegram conversations."""
            return json.dumps(interface.address_book.get_active_conversations())

        # Register on moderator agent
        try:
            agent = await self.icli.isaa_tools.get_agent(self.moderator_agent_name)
            agent.add_tool(
                telegram_send_message,
                "telegram_send_message",
                description="Send a message to a specific Telegram chat.",
                category=["telegram", "communication"],
            )
            agent.add_tool(
                telegram_search_address,
                "telegram_search_address",
                description="Search for Telegram users or chats by name.",
                category=["telegram", "lookup"],
            )
            agent.add_tool(
                telegram_get_active_conversations,
                "telegram_get_active",
                description="Get all active Telegram conversations.",
                category=["telegram", "context"],
            )
            logger.info("[Telegram] Registered telegram_* tools on moderator agent")
        except Exception as e:
            logger.warning(f"[Telegram] Could not register tools: {e}")

    # ─── Bot Lifecycle ────────────────────────────────────────────────

    async def start(self):
        """Start the Telegram bot (blocking)."""
        telegram_modules = _import_telegram()
        Update, Application, CommandHandler, ContextTypes, MessageHandler, filters = telegram_modules
        if Application is None:
            raise ImportError("python-telegram-bot not installed. Run: pip install python-telegram-bot")

        # Register tools before starting
        await self._register_agent_tools()

        app = Application.builder().token(self.token).build()
        self._application = app

        me = await app.bot.get_me()
        self.bot_username = me.username
        logger.info(f"[Telegram] Bot started as @{self.bot_username}")

        async def handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
            ctx = await self._build_message_context(update)
            if not self._should_respond(ctx):
                return
            # Strip mention prefix
            if self.bot_username:
                ctx.content = ctx.content.replace(f"@{self.bot_username}", "").strip()

            response = await self._handle_message(ctx)
            if response:
                # Split long responses (Telegram limit: 4096)
                chunks = [response[i:i+4096] for i in range(0, len(response), 4096)]
                for chunk_text in chunks:
                    await update.message.reply_text(chunk_text)

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_update))
        app.add_handler(CommandHandler("agent", handle_update))
        app.add_handler(CommandHandler("session", handle_update))
        app.add_handler(CommandHandler("whoami", handle_update))
        app.add_handler(CommandHandler("start", handle_update))

        await app.run_polling()

    async def stop(self):
        """Stop the Telegram bot."""
        logger.info("[Telegram] Stopping interface...")
        if self._application:
            await self._application.stop()


# ═══════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════

def create_telegram_interface(
    icli_host: "ISAA_Host",
    token: Optional[str] = None,
    moderator_agent_name: str = "moderator",
    self_agent_name: str = "self",
    respond_to_groups_only_when_mentioned: bool = True,
    admin_ids: Optional[list[int]] = None,
    moderator_safelist: Optional[set[str]] = None,
) -> TelegramInterface:
    """Factory for TelegramInterface.

    Args:
        icli_host: ISAA_Host instance (provides run_agent_monitored)
        token: Telegram Bot Token (or TELEGRAM_BOT_TOKEN env var)
        moderator_agent_name: Name of the moderator agent in ISAA_Host
        self_agent_name: Name of the self agent (admin-only)
        respond_to_groups_only_when_mentioned: Only respond in groups when @mentioned
        admin_ids: List of admin Telegram user IDs
        moderator_safelist: Extra tools allowed on moderator (beyond telegram_*)

    Returns:
        Configured TelegramInterface

    Example:
        interface = create_telegram_interface(
            icli_host=my_icli,
            admin_ids=[123456789],
        )
        await interface.start()
    """
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Telegram token required (pass token or set TELEGRAM_BOT_TOKEN env)")

    if admin_ids is None and os.environ.get("TELEGRAM_ADMIN_IDS"):
        try:
            admin_ids = [int(x.strip()) for x in os.environ.get("TELEGRAM_ADMIN_IDS").split(",") if x.strip()]
        except ValueError:
            logger.warning("[Telegram] Invalid TELEGRAM_ADMIN_IDS env value")

    return TelegramInterface(
        icli_host=icli_host,
        token=token,
        moderator_agent_name=moderator_agent_name,
        self_agent_name=self_agent_name,
        respond_to_groups_only_when_mentioned=respond_to_groups_only_when_mentioned,
        admin_ids=admin_ids,
        moderator_safelist=moderator_safelist,
    )
