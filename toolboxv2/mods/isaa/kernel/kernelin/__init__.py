from .kernelin_cli import main as CLIKernel
try:
    import discord
    from .kernelin_discord import DiscordKernel
except ImportError:
    DiscordKernel = None
    print("⚠️ Discord not installed. Discord kernel disabled.")
try:
    import telegram
    from .kernelin_telegram import run_telegram_standalone
except ImportError:
    run_telegram_standalone = None
    print("⚠️ Telegram not installed. Telegram kernel disabled.")
from .kernelin_whatsapp import WhatsAppKernel
