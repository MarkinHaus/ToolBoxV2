"""
AgentSession V2 - Enhanced Session with VFS V2 and Docker Integration

Extends AgentSession with:
- VirtualFileSystemV2 (directories, file types, LSP)
- Docker execution environment
- Web app display

Author: FlowAgent V2
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from toolboxv2.mods.isaa.base.Agent.docker_vfs import DockerConfig, DockerVFS
from toolboxv2.mods.isaa.base.Agent.lsp_manager import LSPManager

# Import V2 components
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2
from toolboxv2 import get_logger

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.rule_set import RuleResult, RuleSet
    from toolboxv2.mods.isaa.base.Agent.skills import Skill, SkillsManager

# Global VFS Manager import
from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

logger = get_logger()


# =============================================================================
# HELPER
# =============================================================================


def retrieval_to_llm_context_compact(data, max_entries=5):
    """Format retrieval results for LLM context"""
    lines = []
    for _data in data:
        result = _data["result"]
        lines.append("\nMemory: " + _data["memory"])

        for item in result.overview[:max_entries]:
            relevance = float(item.get("relevance_score", 0))

            for chunk in item.get("main_chunks", []):
                meta = chunk.get("metadata", {})
                role = meta.get("role", "unk")
                text = chunk.get("text", "").strip()
                concepts = ",".join(meta.get("concepts", []))

                lines.append(f"{role}: [{text}]| is: {concepts} | r={relevance:.2f}")
        lines.append("\n")
    return "\n".join(lines)


# =============================================================================
# AGENT SESSION V2
# =============================================================================


class AgentSessionV2:
    """
    Enhanced AgentSession with VFS V2 and Docker integration.

    Features:
    - VirtualFileSystemV2 with directories, file types, LSP
    - DockerVFS for isolated command execution
    - ChatSession integration for RAG and history
    - RuleSet for situation-aware behavior
    - Tool restrictions per session
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        memory_instance: Any,
        max_history: int = 100,
        vfs_max_window_lines: int = 250,
        rule_config_path: str | None = None,
        summarizer: Callable | None = None,
        # V2 additions
        enable_lsp: bool = True,
        enable_docker: bool = True,
        docker_config: DockerConfig | None = None,
        toolboxv2_wheel_path: str | None = None,
        skills_manager: SkillsManager | None = None,
    ):
        """
        Initialize AgentSessionV2.

        Args:
            session_id: Unique session identifier
            agent_name: Name of the parent agent
            memory_instance: AISemanticMemory instance for ChatSession
            max_history: Maximum conversation history length
            vfs_max_window_lines: Max lines to show per VFS file
            rule_config_path: Optional path to RuleSet config
            summarizer: Optional async function for VFS summaries
            enable_lsp: Enable LSP integration for code diagnostics
            enable_docker: Enable Docker execution environment
            docker_config: Docker configuration
            toolboxv2_wheel_path: Path to ToolboxV2 wheel for Docker
            skills_manager: SkillsManager instance
        """
        self.session_id = session_id
        self.agent_name = agent_name
        self._memory = memory_instance

        self.tools_initialized = False

        # Timestamps
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        # Metadata
        self.metadata: dict[str, Any] = {}

        # Tool restrictions: tool_name -> allowed
        self.tool_restrictions: dict[str, bool] = {}

        # Initialize components
        self._chat_session = None
        self._max_history = max_history

        # LSP Manager (optional)
        self._lsp_manager: LSPManager | None = None
        if enable_lsp:
            self._lsp_manager = LSPManager(auto_install=True)

        # VFS V2 - session specific with directories and file types
        self.vfs = VirtualFileSystemV2(
            session_id=session_id,
            agent_name=agent_name,
            max_window_lines=vfs_max_window_lines,
            summarizer=summarizer,
            lsp_manager=self._lsp_manager,
        )

        # Docker VFS (optional)
        self._docker_vfs: DockerVFS | None = None
        self._docker_enabled = enable_docker

        if enable_docker:
            docker_cfg = docker_config or DockerConfig()
            if toolboxv2_wheel_path:
                docker_cfg.toolboxv2_wheel_path = toolboxv2_wheel_path

            self._docker_vfs = DockerVFS(vfs=self.vfs, config=docker_cfg)

        # RuleSet - session specific
        from toolboxv2.mods.isaa.base.Agent.rule_set import (
            RuleSet,
            create_default_ruleset,
        )

        self.rule_set: RuleSet = create_default_ruleset(config_path=rule_config_path)
        from toolboxv2.mods.isaa.base.Agent.skills import SkillsManager

        self.skills: SkillsManager = skills_manager or SkillsManager(
            agent_name=agent_name, memory_instance=memory_instance
        )
        # Sync RuleSet to VFS
        self._sync_ruleset_to_vfs()

        # State
        self._initialized = False
        self._closed = False

    async def initialize(self):
        """
        Async initialization - must be called after __init__.

        Automatically mounts /global/ and shared mount points to the VFS.
        """
        if self._initialized:
            return

        # Create ChatSession
        from toolboxv2.mods.isaa.extras.session import ChatSession

        space_name = f"ChatSession/{self.agent_name}.{self.session_id}.unified"
        self._chat_session = ChatSession(
            self._memory, max_length=self._max_history, space_name=space_name
        )

        # Initialize VFS Features - Mount /global/ automatically
        try:
            global_vfs = get_global_vfs()

            # Mount /global/ directory with auto-sync
            mount_result = self.vfs.mount(
                local_path=global_vfs.local_path,
                vfs_path="/global",
                readonly=False,
                auto_sync=True,
            )

            if mount_result.get("success", True):
                logger.info(
                    f"[{self.session_id}] Successfully mounted /global/ from {global_vfs.local_path}"
                )
                global_vfs.register_vfs(self.vfs)
            else:
                logger.warning(
                    f"[{self.session_id}] Failed to mount /global/: {mount_result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"[{self.session_id}] Error during VFS global mount initialization: {e}"
            )

        self.tools_initialized = False
        self._initialized = True

    def _ensure_initialized(self):
        """Ensure session is initialized"""
        if not self._initialized:
            raise RuntimeError(
                f"AgentSessionV2 '{self.session_id}' not initialized. "
                "Call 'await session.initialize()' first."
            )

    def _update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

    def _sync_ruleset_to_vfs(self):
        """Sync RuleSet content to VFS active_rules file"""
        if self.rule_set.is_dirty():
            content = self.rule_set.build_vfs_content()
            self.vfs.set_rules_file(content)
            self.rule_set.mark_clean()

    # =========================================================================
    # CHAT METHODS
    # =========================================================================

    def clear_history(self):
        """Clear conversation history"""
        self._ensure_initialized()
        self._update_activity()
        self._chat_session.clear_history()

    async def add_message(self, message: dict, **kwargs):
        """Add message to conversation history"""
        self._ensure_initialized()
        self._update_activity()
        await self._chat_session.add_message(message, **kwargs)

    async def get_reference(self, text: str, concepts=False, **kwargs) -> str:
        """Query RAG for relevant context"""
        self._ensure_initialized()
        self._update_activity()
        kwargs["row"] = True
        res = await self._chat_session.get_reference(text, **kwargs)
        return (
            res
            if concepts
            else retrieval_to_llm_context_compact(
                res, max_entries=kwargs.get("max_entries", 5)
            )
        )

    def get_history(self, last_n: int | None = None) -> list[dict]:
        """Get conversation history"""
        self._ensure_initialized()
        if last_n is None:
            return self._chat_session.history.copy()
        return self._chat_session.get_past_x(last_n)

    def get_history_for_llm(self, last_n: int = 10) -> list[dict]:
        """Get history formatted for LLM context"""
        self._ensure_initialized()
        return self._chat_session.get_start_with_last_user(last_n)

    # =========================================================================
    # VFS METHODS (V2)
    # =========================================================================

    def vfs_create(self, path: str, content: str = "") -> dict:
        """Create VFS file"""
        self._update_activity()
        return self.vfs.create(path, content)

    def vfs_read(self, path: str) -> dict:
        """Read VFS file"""
        return self.vfs.read(path)

    def vfs_write(self, path: str, content: str) -> dict:
        """Write VFS file"""
        self._update_activity()
        return self.vfs.write(path, content)

    def vfs_open(self, path: str, line_start: int = 1, line_end: int = -1) -> dict:
        """Open VFS file"""
        self._update_activity()
        return self.vfs.open(path, line_start, line_end)

    async def vfs_close(self, path: str) -> dict:
        """Close VFS file with summary"""
        self._update_activity()
        return await self.vfs.close(path)

    def vfs_list(self) -> dict:
        """List VFS files"""
        return self.vfs.list_files()

    def vfs_mkdir(self, path: str, parents: bool = False) -> dict:
        """Create directory"""
        self._update_activity()
        return self.vfs.mkdir(path, parents)

    def vfs_rmdir(self, path: str, force: bool = False) -> dict:
        """Remove directory"""
        self._update_activity()
        return self.vfs.rmdir(path, force)

    def vfs_mv(self, source: str, destination: str) -> dict:
        """Move/rename file or directory"""
        self._update_activity()
        return self.vfs.mv(source, destination)

    def vfs_ls(self, path: str = "/", recursive: bool = False) -> dict:
        """List directory contents"""
        return self.vfs.ls(path, recursive)

    async def vfs_diagnostics(self, path: str) -> dict:
        """Get LSP diagnostics for a file"""
        return await self.vfs.get_diagnostics(path)

    def build_vfs_context(self) -> str:
        """Build VFS context for LLM"""
        self._sync_ruleset_to_vfs()
        return self.vfs.build_context_string()

    # =========================================================================
    # DOCKER METHODS
    # =========================================================================

    async def docker_run_command(
        self,
        command: str,
        timeout: int = 300,
        sync_before: bool = True,
        sync_after: bool = True,
    ) -> dict:
        """
        Run a command in the Docker container.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds
            sync_before: Sync VFS to container before execution
            sync_after: Sync container to VFS after execution

        Returns:
            Result dict with stdout, stderr, exit_code
        """
        if not self._docker_vfs:
            return {"success": False, "error": "Docker not enabled for this session"}

        self._update_activity()
        return await self._docker_vfs.run_command(
            command, timeout, sync_before, sync_after
        )

    async def docker_start_web_app(
        self, entrypoint: str, port: int = 8080, env: dict[str, str] | None = None
    ) -> dict:
        """Start a web app in the Docker container"""
        if not self._docker_vfs:
            return {"success": False, "error": "Docker not enabled for this session"}

        self._update_activity()
        return await self._docker_vfs.start_web_app(entrypoint, port, env)

    async def docker_stop_web_app(self) -> dict:
        """Stop running web app"""
        if not self._docker_vfs:
            return {"success": False, "error": "Docker not enabled"}
        return await self._docker_vfs.stop_web_app()

    async def docker_get_logs(self, lines: int = 100) -> dict:
        """Get web app logs"""
        if not self._docker_vfs:
            return {"success": False, "error": "Docker not enabled"}
        return await self._docker_vfs.get_app_logs(lines)

    def docker_status(self) -> dict:
        """Get Docker container status"""
        if not self._docker_vfs:
            return {"enabled": False}
        return {"enabled": True, **self._docker_vfs.get_status()}

    # =========================================================================
    # RULESET METHODS
    # =========================================================================

    def get_current_rule_set(self) -> dict:
        """Get current rule set state"""
        return self.rule_set.get_current_rule_set()

    def rule_on_action(self, action: str, context: dict | None = None) -> "RuleResult":
        """Evaluate if action is allowed"""
        from toolboxv2.mods.isaa.base.Agent.rule_set import RuleResult

        return self.rule_set.rule_on_action(action, context)

    def set_situation(self, situation: str, intent: str):
        """Set current situation and intent"""
        self.rule_set.set_situation(situation, intent)
        self._sync_ruleset_to_vfs()
        self._update_activity()

    def suggest_situation(self, situation: str, intent: str) -> dict:
        """Suggest situation (agent confirms)"""
        return self.rule_set.suggest_situation(situation, intent)

    def confirm_suggestion(self) -> bool:
        """Confirm pending situation suggestion"""
        result = self.rule_set.confirm_suggestion()
        if result:
            self._sync_ruleset_to_vfs()
        return result

    def clear_situation(self):
        """Clear current situation"""
        self.rule_set.clear_situation()
        self._sync_ruleset_to_vfs()

    # =========================================================================
    # TOOL RESTRICTIONS
    # =========================================================================

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if tool is allowed in this session"""
        return self.tool_restrictions.get(tool_name, True)

    def set_tool_restriction(self, tool_name: str, allowed: bool):
        """Set tool restriction"""
        self.tool_restrictions[tool_name] = allowed
        self._update_activity()

    def get_restrictions(self) -> dict[str, bool]:
        """Get all tool restrictions"""
        return self.tool_restrictions.copy()

    def reset_restrictions(self):
        """Reset all tool restrictions"""
        self.tool_restrictions.clear()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def close(self):
        """Close session - persist VFS and save state"""
        if self._closed:
            return

        # Close all open VFS files
        for path, f in list(self.vfs.files.items()):
            if f.state == "open" and not f.readonly:
                await self.vfs.close(path)

        # Stop Docker container
        if self._docker_vfs:
            await self._docker_vfs.destroy_container()

        # Stop LSP servers
        if self._lsp_manager:
            await self._lsp_manager.stop_all_servers()

        # Save ChatSession
        if self._chat_session:
            self._chat_session.on_exit()

        self._closed = True

    async def cleanup(self):
        """Clean up resources"""
        await self.close()

        # Clear VFS
        self.vfs.files.clear()
        self.vfs.directories.clear()
        self.vfs._init_root()
        self.vfs._init_system_files()

        # Clear rule set state
        self.rule_set.clear_situation()

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize session for checkpoint"""
        self._chat_session.on_exit() if self._chat_session else None

        checkpoint = {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "tool_restrictions": self.tool_restrictions,
            "vfs": self.vfs.to_checkpoint(),
            "rule_set": self.rule_set.to_checkpoint(),
            "skills": self.skills.to_checkpoint(),
            "chat_history": self._chat_session.history if self._chat_session else [],
            "max_history": self._max_history,
            "kb": self._chat_session.mem.save_memory(self._chat_session.space_name, None)
            if self._chat_session
            else None,
            # V2 additions
            "version": 2,
            "docker_enabled": self._docker_enabled,
            "lsp_enabled": self._lsp_manager is not None,
        }

        # Include Docker history if enabled
        if self._docker_vfs:
            checkpoint["docker_history"] = self._docker_vfs.to_checkpoint()

        return checkpoint

    @classmethod
    async def from_checkpoint(
        cls,
        data: dict,
        memory_instance: Any,
        summarizer: Callable | None = None,
        docker_config: DockerConfig | None = None,
    ) -> "AgentSessionV2":
        """Restore session from checkpoint"""
        session = cls(
            session_id=data["session_id"],
            agent_name=data["agent_name"],
            memory_instance=memory_instance,
            max_history=data.get("max_history", 100),
            summarizer=summarizer,
            enable_lsp=data.get("lsp_enabled", True),
            enable_docker=data.get("docker_enabled", False),
            docker_config=docker_config,
        )

        # Restore timestamps
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])

        # Restore metadata
        session.metadata = data.get("metadata", {})

        # Restore tool restrictions
        session.tool_restrictions = data.get("tool_restrictions", {})

        # Restore VFS
        session.vfs.from_checkpoint(data.get("vfs", {}))

        # Restore RuleSet
        session.rule_set.from_checkpoint(data.get("rule_set", {}))
        if data.get("skills"):
            session.skills.from_checkpoint(data["skills"])

        # Initialize ChatSession
        await session.initialize()

        # Restore chat history
        if session._chat_session and data.get("chat_history"):
            session._chat_session.history = data["chat_history"]

        # Restore knowledge base
        if (
            session._chat_session
            and data.get("kb")
            and session._chat_session.get_volume() == 0
        ):
            session._chat_session.mem.load_memory(
                session._chat_session.space_name, data["kb"]
            )

        # Restore Docker history
        if session._docker_vfs and data.get("docker_history"):
            session._docker_vfs.from_checkpoint(data["docker_history"])

        session._sync_ruleset_to_vfs()

        return session

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_stats(self) -> dict:
        """Get session statistics"""
        stats = {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "version": 2,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "age_seconds": (datetime.now() - self.created_at).total_seconds(),
            "idle_seconds": (datetime.now() - self.last_activity).total_seconds(),
            "history_length": len(self._chat_session.history)
            if self._chat_session
            else 0,
            "vfs_files": len(self.vfs.files),
            "vfs_directories": len(self.vfs.directories),
            "vfs_open_files": sum(
                1 for f in self.vfs.files.values() if f.state == "open"
            ),
            "tool_restrictions": len(self.tool_restrictions),
            "active_rules": len(self.rule_set.get_active_rules()),
            "skills": self.skills.get_stats(),
            "current_situation": self.rule_set.current_situation,
            "current_intent": self.rule_set.current_intent,
            "lsp_enabled": self._lsp_manager is not None,
            "docker_enabled": self._docker_enabled,
        }

        if self._docker_vfs:
            stats["docker_status"] = self._docker_vfs.get_status()

        return stats

    def __repr__(self) -> str:
        status = (
            "closed"
            if self._closed
            else ("initialized" if self._initialized else "created")
        )
        features = []
        if self._lsp_manager:
            features.append("LSP")
        if self._docker_enabled:
            features.append("Docker")
        features_str = f" [{', '.join(features)}]" if features else ""
        return f"<AgentSessionV2 {self.session_id} [{status}]{features_str}>"


AgentSession = AgentSessionV2
