from .kernelin_cli import main as CLIKernel
try:
    import discord
    from .kernelin_discord import DiscordKernel
except ImportError:
    DiscordKernel = None
    from toolboxv2.utils.clis.cli_printing import c_print
    c_print("⚠️ Discord not installed. Discord kernel disabled.")
try:
    import telegram
    from .kernelin_telegram import run_telegram_standalone
except ImportError:
    run_telegram_standalone = None
    from toolboxv2.utils.clis.cli_printing import c_print
    c_print("⚠️ Telegram not installed. Telegram kernel disabled.")
from .kernelin_whatsapp import WhatsAppKernel
