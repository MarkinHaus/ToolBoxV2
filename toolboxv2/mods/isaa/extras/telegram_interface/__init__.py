"""Telegram Interface for ISAA — adapted from Discord Interface.

Provides a Telegram bot that exposes a FlowAgent (moderator) to Telegram chats.
Security: moderator safe-mode (tool whitelisting) identical to Discord.
"""
from .telegram_interface import (
    TelegramInterface,
    MessageContext,
    MessageSource,
    create_telegram_interface,
)
from .telegram_cli_extension import TelegramCliExtension

__all__ = [
    "TelegramInterface",
    "MessageContext",
    "MessageSource",
    "create_telegram_interface",
    "TelegramCliExtension",
]
