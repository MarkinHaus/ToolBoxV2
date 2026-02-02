"""
Session Manager V2 Extension - VFS Features Integration
========================================================

Erweiterung für SessionManager um VFS Features automatisch
bei Session-Erstellung zu initialisieren.

Usage:
    # In flow_agent.py oder beim SessionManager Setup:
    from vfs_extensions import register_vfs_tools, init_session_with_vfs_features

    # Nach Session-Erstellung:
    await init_session_with_vfs_features(session)

    # Tools beim Agent registrieren:
    register_vfs_tools(agent)

Author: Markin / ToolBoxV2
Version: 2.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.session_manager import SessionManager
    from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSessionV2
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent


# =============================================================================
# PATCHED SESSION MANAGER
# =============================================================================

def patch_session_manager(session_manager: "SessionManager") -> "SessionManager":
    """
    Patcht den SessionManager um VFS Features automatisch zu aktivieren.

    Der gepatchte SessionManager:
    - Mountet /global/ automatisch bei neuen Sessions
    - Registriert Sessions beim GlobalVFS Manager
    - Initialisiert Sharing Support

    Args:
        session_manager: Zu patchender SessionManager

    Returns:
        Gepatchter SessionManager
    """
    from toolboxv2.mods.isaa.base.patch.power_vfs import init_session_with_vfs_features

    # Original get_or_create speichern
    original_get_or_create = session_manager.get_or_create

    async def patched_get_or_create(
        session_id: str,
        max_history: int = None,
        rule_config_path: str = None,
        enable_lsp: bool = None,
        enable_docker: bool = None,
        docker_config=None,
        # VFS Extension Options
        mount_global: bool = True,
        global_readonly: bool = False,
    ) -> "AgentSessionV2":
        """
        Erweiterte get_or_create mit VFS Features.

        Neue Args:
            mount_global: /global/ automatisch mounten (default: True)
            global_readonly: /global/ nur lesend (default: False)
        """
        # Original aufrufen
        session = await original_get_or_create(
            session_id,
            max_history,
            rule_config_path,
            enable_lsp,
            enable_docker,
            docker_config,
        )

        # VFS Features initialisieren (nur bei neuen Sessions)
        if not hasattr(session, "_vfs_features_initialized"):
            await init_session_with_vfs_features(
                session,
                mount_global=mount_global,
                global_readonly=global_readonly,
            )
            session._vfs_features_initialized = True

        return session

    # Patch anwenden
    session_manager.get_or_create = patched_get_or_create
    session_manager._vfs_patched = True

    return session_manager


def patch_flow_agent(agent: "FlowAgent") -> "FlowAgent":
    """
    Patcht einen FlowAgent um VFS Tools automatisch zu registrieren.

    Der gepatchte Agent:
    - Hat alle VFS Core Tools (ls, read, write, etc.)
    - Hat Global VFS Tools (global_read, global_write, etc.)
    - Hat Sharing Tools (share_create, share_mount, etc.)
    - Hat Search Tools (vfs_search, vfs_find, vfs_grep)
    - Hat Execute Tools (wenn Docker enabled)

    Args:
        agent: Zu patchender FlowAgent

    Returns:
        Gepatchter FlowAgent
    """
    from toolboxv2.mods.isaa.base.patch.power_vfs import register_vfs_tools

    if hasattr(agent, "_vfs_tools_registered"):
        return agent

    # Session Manager patchen
    if hasattr(agent, "session_manager") and not getattr(agent.session_manager, "_vfs_patched", False):
        patch_session_manager(agent.session_manager)

    # Tools registrieren
    result = register_vfs_tools(
        agent,
        enable_global=True,
        enable_sharing=True,
        enable_search=True,
        enable_execute=agent.session_manager.enable_docker if hasattr(agent, "session_manager") else False,
    )

    agent._vfs_tools_registered = True
    agent._vfs_tools_info = result

    print(f"[VFS] Registered {result['count']} VFS tools: {', '.join(result['registered'])}")

    return agent


# =============================================================================
# INTEGRATION EXAMPLES
# =============================================================================

INTEGRATION_EXAMPLE = """
# ============================================================================
# Option 1: Automatische Integration beim Agent-Start
# ============================================================================

from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from session_manager_patch import patch_flow_agent

# Agent erstellen
agent = FlowAgent(name="MyAgent", enable_docker=True)

# Patchen - registriert alle VFS Tools
patch_flow_agent(agent)

# Jetzt kann der Agent alle VFS Tools nutzen:
# - vfs_ls, vfs_read, vfs_write, vfs_edit, vfs_delete
# - vfs_mkdir, vfs_mv, vfs_cp, vfs_mount, vfs_unmount
# - global_read, global_write, global_ls, global_delete
# - share_create, share_mount, share_grant, share_list
# - vfs_search, vfs_find, vfs_grep
# - vfs_execute, vfs_run_code, docker_shell (wenn Docker enabled)


# ============================================================================
# Option 2: Manuelle Integration
# ============================================================================

from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from vfs_extensions import register_vfs_tools, init_session_with_vfs_features

# Agent erstellen
agent = FlowAgent(name="MyAgent", enable_docker=True)

# VFS Tools registrieren (alle oder selektiv)
register_vfs_tools(
    agent,
    enable_global=True,     # /global/ Tools
    enable_sharing=True,    # Share Tools
    enable_search=True,     # Search Tools
    enable_execute=True,    # Docker Execute Tools
)

# Session mit VFS Features erstellen
session = await agent.session_manager.get_or_create("my_session")
await init_session_with_vfs_features(
    session,
    mount_global=True,       # /global/ mounten
    global_readonly=False,   # Schreibzugriff auf /global/
)


# ============================================================================
# Option 3: Nur bestimmte Features
# ============================================================================

from vfs_extensions import (
    register_vfs_tools,
    get_global_vfs,
    get_sharing_manager,
)

# Nur Core + Search Tools
register_vfs_tools(
    agent,
    enable_global=False,
    enable_sharing=False,
    enable_search=True,
    enable_execute=False,
)

# Global VFS manuell verwenden
global_vfs = get_global_vfs()
global_vfs.write_file("config/settings.json", '{"key": "value"}')
content = global_vfs.read_file("config/settings.json")

# Sharing manuell verwenden
sharing = get_sharing_manager()
result = sharing.create_share(session.vfs, "/project/src", readonly=True)
share_id = result["share_id"]

# Anderen Agent Zugriff geben
sharing.grant_access_agent(share_id, "OtherAgent")


# ============================================================================
# VFS Struktur nach Initialisierung
# ============================================================================

# /
# ├── .system/           (readonly)
# │   ├── state.json
# │   ├── active_rules.txt
# │   └── help.txt
# ├── global/            (alle Sessions, auto-mounted)
# │   ├── config/
# │   └── shared_data/
# ├── shared/            (gemountete Shares)
# │   └── {share_id}/
# └── project/           (Session-spezifisch)
#     └── ...


# ============================================================================
# Agent Tool Verwendung
# ============================================================================

# Im Agent Prompt kann der Agent diese Tools nutzen:

# Dateien auflisten
await agent.execute_tool("vfs_ls", path="/project", recursive=True)

# Datei lesen
content = await agent.execute_tool("vfs_read", path="/project/main.py")

# Datei schreiben
await agent.execute_tool("vfs_write", path="/project/new_file.py", content="print('hello')")

# Suchen
results = await agent.execute_tool("vfs_search", query="TODO", mode="content")

# Global schreiben (alle Agents können lesen)
await agent.execute_tool("global_write", path="config.json", content='{"shared": true}')

# Share erstellen
share_result = await agent.execute_tool("share_create", vfs_path="/project/src")

# Im Docker ausführen
output = await agent.execute_tool("docker_shell", command="pip install requests && python main.py")
"""

if __name__ == "__main__":
    print(INTEGRATION_EXAMPLE)
