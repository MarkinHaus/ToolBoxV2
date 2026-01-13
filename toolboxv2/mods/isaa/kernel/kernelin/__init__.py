from .kernelin_cli import main as CLIKernel
try:
    import discord
    from .kernelin_discord import DiscordKernel
except ImportError:
    DiscordKernel = None
    print("⚠️ Discord not installed. Discord kernel disabled.")
from .kernelin_whatsapp import WhatsAppKernel
