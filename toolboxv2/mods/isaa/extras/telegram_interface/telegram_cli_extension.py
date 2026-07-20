"""CLI Extension for Telegram Integration.

Adapted from discord_cli_extension.py — provides /telegram commands in ICLI.
Uses ISAA_Host.run_agent_monitored() for agent execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.flows.isaa.icli import ISAA_Host

from toolboxv2 import get_logger
logger = get_logger()


class TelegramCliExtension:
    """CLI Extension for Telegram Integration.

    Provides /telegram commands in the ICLI:
    /telegram connect [token]   — Start the Telegram bot
    /telegram disconnect        — Stop the Telegram bot
    /telegram status            — Show connection status
    /telegram chats             — List known chats
    /telegram send <chat_id> <msg> — Send a message
    /telegram search <query>    — Search address book
    /telegram active            — Show active conversations
    /telegram admin [add|remove] <id> — Manage admin list
    """

    def __init__(self, icli_host: "ISAA_Host"):
        self.icli = icli_host
        self._interface = None
        self._task = None

    @property
    def is_connected(self) -> bool:
        return self._interface is not None and self._interface._application is not None

    async def cmd_connect(self, token: Optional[str] = None) -> str:
        if self.is_connected:
            return "Already connected. Use /telegram disconnect first."

        from .telegram_interface import create_telegram_interface

        try:
            self._interface = create_telegram_interface(
                icli_host=self.icli,
                token=token,
            )

            def _on_done(fut):
                exc = fut.exception() if not fut.cancelled() else None
                if exc:
                    logger.error("[Telegram] bot task ended with error: %s", exc, exc_info=exc)
                    self._interface = None

            self._task = asyncio.create_task(self._interface.start())
            self._task.add_done_callback(_on_done)
            return "Telegram bot connecting... Check /telegram status (errors appear in the log)."
        except Exception as e:
            self._interface = None
            logger.exception("[Telegram] connect failed")
            return f"Failed to connect: {e}"

    async def cmd_disconnect(self) -> str:
        if not self.is_connected:
            return "Not connected."
        try:
            await self._interface.stop()
            if self._task:
                self._task.cancel()
            self._interface = None
            self._task = None
            return "Telegram bot disconnected."
        except Exception as e:
            return f"Error disconnecting: {e}"

    async def cmd_status(self) -> str:
        if not self._interface:
            return "Not connected."
        lines = ["Telegram Status:"]
        lines.append(f"  Connected: {self.is_connected}")
        if self._interface.bot_username:
            lines.append(f"  Bot: @{self._interface.bot_username}")
        lines.append(f"  Admin IDs: {self._interface.admin_ids}")
        lines.append(f"  Safe-mode: active (only telegram* + {sorted(self._interface.moderator_safelist)})")
        lines.append(f"  Group mention-only: {self._interface.respond_to_groups_only_when_mentioned}")
        active = self._interface.address_book.get_active_conversations()
        lines.append(f"  Active conversations: {len(active)}")
        return "\n".join(lines)

    async def cmd_chats(self) -> str:
        if not self._interface:
            return "Not connected."
        chats = self._interface.address_book._chats
        if not chats:
            return "No chats known yet."
        lines = ["Known chats:"]
        for cid, info in chats.items():
            lines.append(f"  {info['name']} (ID: {cid}, type: {info['type']})")
        return "\n".join(lines)

    async def cmd_send(self, chat_id_str: str, message: str) -> str:
        if not self._interface:
            return "Not connected."
        try:
            chat_id = int(chat_id_str)
            result = await self._interface._application.bot.send_message(
                chat_id=chat_id, text=message,
            )
            return f"Message sent (ID: {result.message_id})"
        except ValueError:
            return f"Invalid chat_id: {chat_id_str}"
        except Exception as e:
            return f"Error sending: {e}"

    async def cmd_search(self, query: str) -> str:
        if not self._interface:
            return "Not connected."
        results = self._interface.address_book.search(query)
        if not results:
            return f"No results for '{query}'."
        lines = [f"Search results for '{query}':"]
        for r in results:
            rtype = r.get("type", "unknown")
            if rtype == "user":
                lines.append(f"  User: {r.get('display_name')} (@{r.get('username')}) ID: {r.get('user_id')}")
            elif rtype == "chat":
                lines.append(f"  Chat: {r.get('name')} ID: {r.get('chat_id')}")
        return "\n".join(lines)

    async def cmd_active(self) -> str:
        if not self._interface:
            return "Not connected."
        active = self._interface.address_book.get_active_conversations()
        if not active:
            return "No active conversations."
        lines = ["Active conversations:"]
        for addr, info in active.items():
            lines.append(f"  {addr} — {info.get('user_name', '?')} in {info.get('chat_name', '?')}")
        return "\n".join(lines)

    async def cmd_admin(self, action: str, user_id_str: str) -> str:
        if not self._interface:
            return "Not connected."
        try:
            uid = int(user_id_str)
        except ValueError:
            return f"Invalid user ID: {user_id_str}"
        if action == "add":
            if uid not in self._interface.admin_ids:
                self._interface.admin_ids.append(uid)
                return f"Added {uid} to admin list."
            return f"{uid} is already an admin."
        elif action == "remove":
            if uid in self._interface.admin_ids:
                self._interface.admin_ids.remove(uid)
                return f"Removed {uid} from admin list."
            return f"{uid} is not an admin."
        return "Usage: /telegram admin [add|remove] <user_id>"

    async def cmd_setup(self) -> str:
        """How to create a Telegram bot and get the token + your user id."""
        return (
            "Telegram Bot Setup\n"
            "──────────────────\n"
            "1) Create the bot (get the TOKEN):\n"
            "   • Open Telegram, search for @BotFather (the verified one).\n"
            "   • Send /newbot → pick a name and a username ending in 'bot'.\n"
            "   • BotFather replies with a token like 123456789:ABCdef...\n"
            "   • Put it in env TELEGRAM_BOT_TOKEN, or run: /telegram connect <token>\n"
            "\n"
            "2) Get YOUR Telegram user id (for the admin list):\n"
            "   • Message @userinfobot (or @RawDataBot) → it replies with your numeric id.\n"
            "   • Or DM your own bot after connecting and send /whoami.\n"
            "   • Add it: /telegram admin add <your_id>  (or env TELEGRAM_ADMIN_IDS=111,222)\n"
            "\n"
            "3) Optional bot tuning via @BotFather:\n"
            "   • /setprivacy → Disable  (so the bot can read all group messages)\n"
            "   • /setjoingroups, /setcommands as needed.\n"
            "\n"
            "Then: /telegram connect   →   message your bot   →   it answers via the agent.\n"
            "Multimedia (photo/voice/audio/video/document) is downloaded and passed to the agent natively."
        )

    async def handle_command(self, args: str) -> str:
        parts = args.strip().split(maxsplit=2)
        if not parts:
            return (
                "Telegram Commands:\n"
                "  setup            — How to create a bot + get token/id\n"
                "  connect [token]  — Start bot\n"
                "  disconnect       — Stop bot\n"
                "  status           — Connection status\n"
                "  chats            — List known chats\n"
                "  send <id> <msg>  — Send message\n"
                "  search <query>   — Search contacts\n"
                "  active           — Active conversations\n"
                "  admin [add|rm] <id> — Manage admins"
            )
        cmd = parts[0].lower()
        if cmd in ("setup", "help"):
            return await self.cmd_setup()
        if cmd == "connect":
            return await self.cmd_connect(parts[1] if len(parts) > 1 else None)
        elif cmd == "disconnect":
            return await self.cmd_disconnect()
        elif cmd == "status":
            return await self.cmd_status()
        elif cmd == "chats":
            return await self.cmd_chats()
        elif cmd == "send":
            if len(parts) < 3:
                return "Usage: /telegram send <chat_id> <message>"
            return await self.cmd_send(parts[1], parts[2])
        elif cmd == "search":
            if len(parts) < 2:
                return "Usage: /telegram search <query>"
            return await self.cmd_search(parts[1])
        elif cmd == "active":
            return await self.cmd_active()
        elif cmd == "admin":
            if len(parts) < 3:
                return "Usage: /telegram admin [add|remove] <user_id>"
            return await self.cmd_admin(parts[1], parts[2])
        return f"Unknown command: {cmd}. Use /telegram for help."
