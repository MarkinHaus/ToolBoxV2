"""
SessionManager - Manages all AgentSessions for FlowAgent

Provides:
- Lazy loading of memory instance
- Session lifecycle management
- Bulk operations on sessions

Author: FlowAgent V2
"""

from datetime import datetime
from typing import Any, Callable

from toolboxv2.mods.isaa.base.Agent.agent_session import AgentSession


class SessionManager:
    """
    Manages all sessions for a FlowAgent instance.

    Features:
    - Lazy loading of AISemanticMemory
    - Session creation/retrieval/cleanup
    - Auto-cleanup of inactive sessions
    """

    def __init__(
        self,
        agent_name: str,
        default_max_history: int = 100,
        vfs_max_window_lines: int = 250,
        rule_config_path: str | None = None,
        summarizer: Callable | None = None,
        auto_cleanup_hours: float | None = None
    ):
        """
        Initialize SessionManager.

        Args:
            agent_name: Name of parent agent
            default_max_history: Default history length for new sessions
            vfs_max_window_lines: Max VFS window lines
            rule_config_path: Default RuleSet config path
            summarizer: Summarizer function for VFS
            auto_cleanup_hours: Auto-cleanup sessions older than this
        """
        self.agent_name = agent_name
        self.default_max_history = default_max_history
        self.vfs_max_window_lines = vfs_max_window_lines
        self.rule_config_path = rule_config_path
        self._summarizer = summarizer
        self.auto_cleanup_hours = auto_cleanup_hours

        # Session storage
        self.sessions: dict[str, AgentSession] = {}

        # Memory instance (lazy loaded)
        self._memory_instance = None

        # Stats
        self._total_sessions_created = 0

    def _get_memory(self) -> Any:
        """Lazy load AISemanticMemory"""
        if self._memory_instance is None:
            from toolboxv2 import get_app
            res = get_app().get_mod("isaa")
            if not hasattr(res, "get_memory") and hasattr(res, "get"):
                res = res.get()
            self._memory_instance = res.get_memory()
        return self._memory_instance

    def _ensure_memory(self):
        """Ensure memory is loaded"""
        self._get_memory()

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    async def get_or_create(
        self,
        session_id: str,
        max_history: int | None = None,
        rule_config_path: str | None = None
    ) -> AgentSession:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier
            max_history: Override default max history
            rule_config_path: Override default rule config

        Returns:
            AgentSession instance (initialized)
        """
        # Return existing
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if not session._initialized:
                await session.initialize()
            return session

        # Create new
        self._ensure_memory()

        session = AgentSession(
            session_id=session_id,
            agent_name=self.agent_name,
            memory_instance=self._memory_instance,
            max_history=max_history or self.default_max_history,
            vfs_max_window_lines=self.vfs_max_window_lines,
            rule_config_path=rule_config_path or self.rule_config_path,
            summarizer=self._summarizer
        )

        await session.initialize()

        self.sessions[session_id] = session
        self._total_sessions_created += 1

        return session

    def get(self, session_id: str) -> AgentSession | None:
        """Get session by ID (None if not exists)"""
        return self.sessions.get(session_id)

    def exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self.sessions

    async def close_session(self, session_id: str) -> bool:
        """
        Close and remove a session.

        Args:
            session_id: Session to close

        Returns:
            True if session was closed
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        await session.close()
        del self.sessions[session_id]

        return True

    async def close_all(self):
        """Close all sessions"""
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)

    def list_sessions(self) -> list[str]:
        """List all session IDs"""
        return list(self.sessions.keys())

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    def get_all_active(self) -> list[AgentSession]:
        """Get all active (initialized) sessions"""
        return [s for s in self.sessions.values() if s._initialized and not s._closed]

    async def cleanup_inactive(self, max_idle_hours: float | None = None) -> int:
        """
        Clean up sessions that have been idle too long.

        Args:
            max_idle_hours: Max idle time (uses auto_cleanup_hours if None)

        Returns:
            Number of sessions cleaned up
        """
        threshold = max_idle_hours or self.auto_cleanup_hours
        if threshold is None:
            return 0

        now = datetime.now()
        to_cleanup = []

        for session_id, session in self.sessions.items():
            idle_hours = (now - session.last_activity).total_seconds() / 3600
            if idle_hours > threshold:
                to_cleanup.append(session_id)

        for session_id in to_cleanup:
            await self.close_session(session_id)

        return len(to_cleanup)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize all sessions for checkpoint"""
        return {
            'agent_name': self.agent_name,
            'default_max_history': self.default_max_history,
            'vfs_max_window_lines': self.vfs_max_window_lines,
            'rule_config_path': self.rule_config_path,
            'total_sessions_created': self._total_sessions_created,
            'sessions': {
                session_id: session.to_checkpoint()
                for session_id, session in self.sessions.items()
                if session._initialized
            }
        }

    async def from_checkpoint(self, data: dict):
        """
        Restore sessions from checkpoint.

        Args:
            data: Checkpoint data
        """
        self._ensure_memory()

        # Restore config
        self.default_max_history = data.get('default_max_history', self.default_max_history)
        self.vfs_max_window_lines = data.get('vfs_max_window_lines', self.vfs_max_window_lines)
        self.rule_config_path = data.get('rule_config_path', self.rule_config_path)
        self._total_sessions_created = data.get('total_sessions_created', 0)

        # Restore sessions
        for session_id, session_data in data.get('sessions', {}).items():
            try:
                session = await AgentSession.from_checkpoint(
                    data=session_data,
                    memory_instance=self._memory_instance,
                    summarizer=self._summarizer
                )
                self.sessions[session_id] = session
            except Exception as e:
                print(f"[SessionManager] Failed to restore session {session_id}: {e}")

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get session manager statistics"""
        active_count = len(self.get_all_active())
        total_history = sum(
            len(s._chat_session.history) if s._chat_session else 0
            for s in self.sessions.values()
        )

        return {
            'agent_name': self.agent_name,
            'total_sessions': len(self.sessions),
            'active_sessions': active_count,
            'total_sessions_created': self._total_sessions_created,
            'total_history_messages': total_history,
            'memory_loaded': self._memory_instance is not None,
            'session_ids': list(self.sessions.keys())
        }

    def __repr__(self) -> str:
        return f"<SessionManager {self.agent_name} [{len(self.sessions)} sessions]>"
