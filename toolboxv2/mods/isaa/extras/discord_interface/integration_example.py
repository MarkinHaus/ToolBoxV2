"""
Integration Example - Discord + CLI + VFS Search
=================================================

Zeigt wie alle Module zusammenspielen.

Dieses File ist ein Beispiel für die Integration in cli_v4.py
"""

import asyncio
import os
from typing import Optional


# =============================================================================
# OPTION 1: Standalone Discord Bot
# =============================================================================

async def run_standalone_discord_bot():
    """
    Startet den Discord Bot standalone ohne CLI.
    Nützlich für Server-Deployment.
    """
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.module import Tools as IsaaTool

    # ToolBoxV2 App
    app = get_app("discord-bot")

    # ISAA Tools initialisieren
    isaa = IsaaTool(app)

    # Agent Builder
    builder = isaa.get_agent_builder(
        name="discord_agent",
        add_base_tools=True,
    )

    # Agent konfigurieren
    builder.with_persona(
        name="Discord Assistant",
        description="Helpful assistant on Discord",
        style="friendly and concise",
    )

    # Agent registrieren
    await isaa.register_agent(builder)
    agent = await isaa.get_agent("discord_agent")

    # VFS Search hinzufügen
    from vfs_search import add_search_to_vfs, register_vfs_search_tools
    session = await agent.session_manager.get_or_create("discord")
    add_search_to_vfs(session.vfs)
    register_vfs_search_tools(agent, session.vfs)

    # Discord Interface erstellen
    from discord_interface import create_discord_interface

    interface = create_discord_interface(
        agent=agent,
        token=os.environ["DISCORD_TOKEN"],
        respond_to_mentions_only=True,
    )

    # Starten
    print("Starting Discord Bot...")
    try:
        await interface.start()
    except KeyboardInterrupt:
        await interface.stop()
        await agent.close()


# =============================================================================
# OPTION 2: CLI Integration Patch
# =============================================================================

def patch_cli_for_discord(host):
    """
    Patcht eine ISAA_Host Instanz für Discord Support.

    Usage in cli_v4.py nach Host-Erstellung:
        host = ISAA_Host(app)
        patch_cli_for_discord(host)
        await host.run()

    Args:
        host: ISAA_Host Instanz
    """
    from toolboxv2.mods.isaa.extras.discord_interface.discord_cli_extension import DiscordCLIExtension, get_completer_dict

    # Extension erstellen
    host.discord_ext = DiscordCLIExtension(host)

    # Original _handle_command speichern
    original_handle_command = host._handle_command

    async def patched_handle_command(user_input: str):
        """Erweiterter Command Handler mit Discord Support"""
        parts = user_input.strip().split()
        command = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        if command == "/discord":
            await host.discord_ext.handle_command(args)
            return

        # Original Handler aufrufen
        await original_handle_command(user_input)

    # Patch anwenden
    host._handle_command = patched_handle_command

    # Completer erweitern
    original_build_completer = host._build_completer

    def patched_build_completer():
        completer = original_build_completer()
        completer["/discord"] = get_completer_dict()
        return completer

    host._build_completer = patched_build_completer

    print("✓ Discord extension patched into CLI")


# =============================================================================
# OPTION 3: Minimale Integration
# =============================================================================

async def minimal_discord_integration():
    """
    Minimale Integration - nur das Nötigste.
    Zeigt die Kernkomponenten.
    """
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.builder import AgentModelData
    from discord_interface import DiscordInterface, MessageContext

    # Agent erstellen (vereinfacht)
    amd = AgentModelData(
        name="minimal_bot",
        persona="You are a helpful Discord bot.",
        fast_llm_model="gpt-4o-mini",
        complex_llm_model="gpt-4o",
    )
    agent = FlowAgent(amd)

    # Discord Interface
    interface = DiscordInterface(
        agent=agent,
        token=os.environ["DISCORD_TOKEN"],
        respond_to_mentions_only=True,
    )

    # Custom Handler für spezielle Logik
    @interface.bot.event
    async def on_message(message):
        # Custom Pre-Processing hier
        if message.content.startswith("!ping"):
            await message.channel.send("Pong!")
            return

        # Standard Handler aufrufen
        await interface._handle_message(
            await interface._build_message_context(message),
            message
        )

    await interface.start()


# =============================================================================
# OPTION 4: Multi-Platform (CLI + Discord parallel)
# =============================================================================

async def multi_platform_setup():
    """
    Läuft CLI und Discord parallel.
    Der gleiche Agent bedient beide Interfaces.
    """
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.module import Tools as IsaaTool

    app = get_app("multi-platform")
    isaa = IsaaTool(app)

    # Shared Agent
    builder = isaa.get_agent_builder("shared_agent", add_base_tools=True)
    await isaa.register_agent(builder)
    agent = await isaa.get_agent("shared_agent")

    # VFS Search für alle Sessions
    from vfs_search import add_search_to_vfs, register_vfs_search_tools
    session = await agent.session_manager.get_or_create("default")
    add_search_to_vfs(session.vfs)
    register_vfs_search_tools(agent, session.vfs)

    # Discord Interface (Background)
    from discord_interface import create_discord_interface

    discord_interface = create_discord_interface(
        agent=agent,
        respond_to_mentions_only=True,
    )

    # Discord in Background Task
    discord_task = app.run_bg_task_advanced(discord_interface.start)

    # CLI in Foreground
    # (Hier würde normalerweise die CLI laufen)
    print("Agent läuft auf CLI und Discord!")
    print("Discord: Antwortet auf @mentions")
    print("CLI: Direkte Eingabe hier")

    # Simple CLI Loop
    while True:
        try:
            user_input = input("\n[CLI] > ")
            if user_input.lower() == "quit":
                break

            response = await agent.a_run(
                query=user_input,
                session_id="cli_session",
            )
            print(f"\n{response}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    # Cleanup
    discord_task.cancel()
    await discord_interface.stop()
    await agent.close()


# =============================================================================
# QUICK START GUIDE
# =============================================================================

QUICK_START = """
# Discord Interface Quick Start Guide

## 1. Environment Setup
```bash
export DISCORD_TOKEN="your-bot-token"
export GROQ_API_KEY="your-groq-key"  # Optional, für Audio Transcription
```

## 2. Standalone Bot
```python
from integration_example import run_standalone_discord_bot
import asyncio

asyncio.run(run_standalone_discord_bot())
```

## 3. CLI Integration
```python
# In cli_v4.py, nach Host-Erstellung:
from integration_example import patch_cli_for_discord

host = ISAA_Host(app)
patch_cli_for_discord(host)
await host.run()

# Dann in der CLI:
# /discord connect
# /discord status
# /discord channels
```

## 4. Key Features

### Auto-Router
Antworten gehen automatisch zum Ursprung zurück.
Der Agent muss nicht explizit sagen wohin.

### Context Awareness
Der Agent erhält bei jeder Nachricht:
- Wer hat geschrieben
- Woher (Server/Channel oder DM)
- Ob @mentioned wurde
- Voice Channel Status
- Attachments

### VFS Address Book
Im Agent VFS unter /discord/:
- /discord/servers/{id}/info.json
- /discord/dms/{user_id}/session.json
- /discord/active.json
- /discord/contacts.json

### Agent Tools
- discord_send_message(address, content)
- discord_search_address(query)
- discord_get_active()
- discord_get_history(channel_id)

### VFS Search (Bonus)
- vfs_search(query, mode="both")
- vfs_find("*.py")
- vfs_grep("TODO", file_pattern="*.py")
"""

if __name__ == "__main__":
    print(QUICK_START)

    # Uncomment to run:
    #
    import dotenv
    # dotenv.load_dotenv(r"C:\Users\Markin\Workspace\ToolBoxV2\.env")
    asyncio.run(multi_platform_setup())
    # asyncio.run(run_standalone_discord_bot())
