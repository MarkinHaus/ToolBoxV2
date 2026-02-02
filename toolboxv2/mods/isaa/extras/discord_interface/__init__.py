"""
Discord Interface Package for ToolBoxV2/ISAA
=============================================

Modulares, einfaches Discord Interface System.

Components:
- DiscordInterface: Hauptklasse für Bot-Agent-Verbindung
- MessageContext: Vollständiger Message Context
- AddressBook: VFS-basiertes Adressbuch
- AutoRouter: Automatisches Response Routing
- VFS Search: Suchfunktion für VFS

Quick Start:
    from discord_interface import create_discord_interface

    interface = create_discord_interface(agent, token)
    await interface.start()
"""

try:
    """
    Discord Interface Package for ToolBoxV2/ISAA
    =============================================

    Modulares, einfaches Discord Interface System.

    Components:
    - DiscordInterface: Hauptklasse für Bot-Agent-Verbindung
    - MessageContext: Vollständiger Message Context
    - AddressBook: VFS-basiertes Adressbuch
    - AutoRouter: Automatisches Response Routing
    - VFS Search: Suchfunktion für VFS
    - Voice Mode: Voice Channel TTS/STT

    Quick Start:
        from discord_interface import create_discord_interface, create_voice_mode

        interface = create_discord_interface(agent, token)
        voice = create_voice_mode(interface)  # Optional: Voice support
        await interface.start()
    """

    from .discord_interface import (
        # Main Classes
        DiscordInterface,
        MessageContext,
        MessageSource,
        AddressBook,
        AutoRouter,
        MediaHandler,

        # Factory
        create_discord_interface,
    )

    from .vfs_search import (
        # Search Classes
        VFSSearchMixin,
        SearchMode,
        SearchResult,

        # Functions
        add_search_to_vfs,
        register_vfs_search_tools,
    )

    from .discord_cli_extension import (
        DiscordCLIExtension,
        get_completer_dict,
    )

    from .voice_mode import (
        # Voice Classes
        VoiceHandler,
        VoiceMessage,
        VoiceConversation,
        VoiceModeExtension,
        UserAudioSink,

        # Factory
        create_voice_mode,

        # CLI
        get_voice_cli_commands,
        handle_voice_cli_command,

        # Constants
        VOICE_RECV_AVAILABLE,
    )

    __version__ = "2.1.0"
    __all__ = [
        # Discord Interface
        "DiscordInterface",
        "MessageContext",
        "MessageSource",
        "AddressBook",
        "AutoRouter",
        "MediaHandler",
        "create_discord_interface",

        # VFS Search
        "VFSSearchMixin",
        "SearchMode",
        "SearchResult",
        "add_search_to_vfs",
        "register_vfs_search_tools",

        # CLI Extension
        "DiscordCLIExtension",
        "get_completer_dict",

        # Voice Mode
        "VoiceHandler",
        "VoiceMessage",
        "VoiceConversation",
        "VoiceModeExtension",
        "UserAudioSink",
        "create_voice_mode",
        "get_voice_cli_commands",
        "handle_voice_cli_command",
        "VOICE_RECV_AVAILABLE",
    ]

    from .integration_example import run_standalone_discord_bot
except ImportError:
    print("⚠️ integration_example not found. Make sure discord_interface.py is available.")
