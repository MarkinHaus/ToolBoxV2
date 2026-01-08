"""
CheckpointManager - Complete persistence for FlowAgent

Provides:
- Full agent state serialization
- Auto-load on initialization
- Checkpoint rotation and cleanup

Author: FlowAgent V2
"""

import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent


@dataclass
class AgentCheckpoint:
    """Complete agent checkpoint data"""

    # Version info
    version: str = "2.0"
    timestamp: datetime = field(default_factory=datetime.now)

    # Agent config
    agent_name: str = ""
    agent_config: dict = field(default_factory=dict)

    # Sessions (full state)
    sessions_data: dict = field(default_factory=dict)

    # Tools (metadata only, functions restored separately)
    tool_registry_data: dict = field(default_factory=dict)

    # Statistics
    statistics: dict = field(default_factory=dict)

    # Bind state
    bind_state: dict | None = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        parts = []

        if self.sessions_data:
            parts.append(f"{len(self.sessions_data)} sessions")

        if self.tool_registry_data:
            tool_count = len(self.tool_registry_data.get('tools', {}))
            parts.append(f"{tool_count} tools")

        if self.statistics:
            cost = self.statistics.get('total_cost', 0)
            if cost > 0:
                parts.append(f"${cost:.4f} spent")

        return "; ".join(parts) if parts else "Empty checkpoint"


class CheckpointManager:
    """
    Manages agent checkpoints for persistence and recovery.

    Features:
    - Auto-load latest checkpoint on init
    - Full state serialization
    - Checkpoint rotation (keep N newest)
    """

    def __init__(
        self,
        agent: 'FlowAgent',
        checkpoint_dir: str | None = None,
        auto_load: bool = True,
        max_checkpoints: int = 5,
        max_age_hours: int = 168  # 1 week
    ):
        """
        Initialize CheckpointManager.

        Args:
            agent: Parent FlowAgent instance
            checkpoint_dir: Directory for checkpoints (auto-detected if None)
            auto_load: Auto-load latest checkpoint on init
            max_checkpoints: Maximum checkpoints to keep
            max_age_hours: Max age before auto-cleanup
        """
        self.agent = agent
        self.max_checkpoints = max_checkpoints
        self.max_age_hours = max_age_hours

        # Determine checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            from toolboxv2 import get_app
            self.checkpoint_dir = os.path.join(
                str(get_app().data_dir),
                'Agents',
                'checkpoint',
                agent.amd.name
            )

        # Ensure directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # State
        self.last_checkpoint: datetime | None = None
        self._loaded_checkpoint: AgentCheckpoint | None = None

        # Auto-load if enabled
        if auto_load:
            self._auto_load_sync()

    def _auto_load_sync(self):
        """Synchronous auto-load for use in __init__"""
        try:
            latest = self._find_latest_checkpoint()
            if latest:
                self._loaded_checkpoint = self._load_checkpoint_file(latest)
                print(f"[CheckpointManager] Loaded checkpoint: {latest}\n{self._loaded_checkpoint.get_summary()}")
        except Exception as e:
            print(f"[CheckpointManager] Auto-load failed: {e}")

    def _find_latest_checkpoint(self) -> str | None:
        """Find latest valid checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith('.pkl'):
                continue

            filepath = os.path.join(self.checkpoint_dir, filename)
            try:
                # Extract timestamp from filename
                if filename.startswith('agent_checkpoint_'):
                    ts_str = filename.replace('agent_checkpoint_', '').replace('.pkl', '')
                    file_time = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                elif filename == 'final_checkpoint.pkl':
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                else:
                    continue

                # Check age
                age_hours = (datetime.now() - file_time).total_seconds() / 3600
                if age_hours <= self.max_age_hours:
                    checkpoints.append((filepath, file_time))
            except Exception:
                continue

        if not checkpoints:
            return None

        # Return newest
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]

    def _load_checkpoint_file(self, filepath: str) -> AgentCheckpoint:
        """Load checkpoint from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    # =========================================================================
    # CHECKPOINT CREATION
    # =========================================================================

    async def create(self) -> AgentCheckpoint:
        """
        Create checkpoint from current agent state.

        Returns:
            AgentCheckpoint with full state
        """
        checkpoint = AgentCheckpoint(
            timestamp=datetime.now(),
            agent_name=self.agent.amd.name,
        )

        # Agent config (AMD)
        checkpoint.agent_config = {
            'name': self.agent.amd.name,
            'fast_llm_model': self.agent.amd.fast_llm_model,
            'complex_llm_model': self.agent.amd.complex_llm_model,
            'system_message': self.agent.amd.system_message,
            'temperature': self.agent.amd.temperature,
            'max_tokens': self.agent.amd.max_tokens,
            'max_input_tokens': self.agent.amd.max_input_tokens,
            'vfs_max_window_lines': self.agent.amd.vfs_max_window_lines,
        }

        # Persona config if present
        if self.agent.amd.persona:
            checkpoint.agent_config['persona'] = {
                'name': self.agent.amd.persona.name,
                'style': self.agent.amd.persona.style,
                'tone': self.agent.amd.persona.tone,
                'personality_traits': self.agent.amd.persona.personality_traits,
                'custom_instructions': self.agent.amd.persona.custom_instructions,
            }

        # Sessions
        if hasattr(self.agent, 'session_manager') and self.agent.session_manager:
            checkpoint.sessions_data = self.agent.session_manager.to_checkpoint()

        # Tool registry
        if hasattr(self.agent, 'tool_manager') and self.agent.tool_manager:
            checkpoint.tool_registry_data = self.agent.tool_manager.to_checkpoint()

        # Statistics
        checkpoint.statistics = {
            'total_tokens_in': self.agent.total_tokens_in,
            'total_tokens_out': self.agent.total_tokens_out,
            'total_cost': self.agent.total_cost_accumulated,
            'total_llm_calls': self.agent.total_llm_calls,
        }

        # Bind state
        if hasattr(self.agent, 'bind_manager') and self.agent.bind_manager:
            checkpoint.bind_state = self.agent.bind_manager.to_checkpoint()

        # Metadata
        checkpoint.metadata = {
            'created_by': 'CheckpointManager',
            'agent_version': '2.0',
            'checkpoint_version': checkpoint.version,
        }

        return checkpoint

    async def save(self, checkpoint: AgentCheckpoint | None = None, filename: str | None = None) -> str:
        """
        Save checkpoint to file.

        Args:
            checkpoint: Checkpoint to save (creates new if None)
            filename: Custom filename (auto-generated if None)

        Returns:
            Filepath of saved checkpoint
        """
        if checkpoint is None:
            checkpoint = await self.create()

        if filename is None:
            timestamp = checkpoint.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"agent_checkpoint_{timestamp}.pkl"

        filepath = os.path.join(self.checkpoint_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        self.last_checkpoint = checkpoint.timestamp

        # Auto-cleanup old checkpoints
        await self.cleanup_old()

        return filepath

    async def save_current(self) -> str:
        """Shortcut to create and save checkpoint"""
        return await self.save()

    # =========================================================================
    # CHECKPOINT RESTORATION
    # =========================================================================

    async def load_latest(self) -> AgentCheckpoint | None:
        """
        Load the latest checkpoint.

        Returns:
            AgentCheckpoint or None if not found
        """
        # Use already loaded if available
        if self._loaded_checkpoint:
            return self._loaded_checkpoint

        latest = self._find_latest_checkpoint()
        if latest:
            return self._load_checkpoint_file(latest)

        return None

    async def restore(
        self,
        checkpoint: AgentCheckpoint | None = None,
        restore_sessions: bool = True,
        restore_tools: bool = True,
        restore_statistics: bool = True,
        function_registry: dict[str, Callable] | None = None
    ) -> dict[str, Any]:
        """
        Restore agent state from checkpoint.

        Args:
            checkpoint: Checkpoint to restore (loads latest if None)
            restore_sessions: Restore session data
            restore_tools: Restore tool registry
            restore_statistics: Restore statistics
            function_registry: Dict mapping tool names to functions

        Returns:
            Restoration statistics
        """
        if checkpoint is None:
            checkpoint = await self.load_latest()

        if checkpoint is None:
            return {'success': False, 'error': 'No checkpoint found'}

        stats = {
            'success': True,
            'checkpoint_timestamp': checkpoint.timestamp.isoformat(),
            'restored_components': [],
            'errors': []
        }

        try:
            # Restore agent config (selective)
            if checkpoint.agent_config:
                # Only restore safe config values
                safe_fields = ['temperature', 'max_tokens', 'max_input_tokens']
                for field in safe_fields:
                    if field in checkpoint.agent_config:
                        setattr(self.agent.amd, field, checkpoint.agent_config[field])

                stats['restored_components'].append('agent_config')

            # Restore sessions
            if restore_sessions and checkpoint.sessions_data:
                if hasattr(self.agent, 'session_manager') and self.agent.session_manager:
                    await self.agent.session_manager.from_checkpoint(checkpoint.sessions_data)
                    stats['restored_components'].append(f'sessions')
                    stats['sessions_restored'] = len(checkpoint.sessions_data.get('sessions', {}))
                    for name, session in self.agent.session_manager.sessions.items():
                        await session.initialize()
                        history_len = len(session._chat_session.history) if session._chat_session else 0
                        stats['restored_components'].append(f'session:{name}_{history_len}')
                        if not session.rule_set.tool_groups:
                            from toolboxv2.mods.isaa.base.Agent.rule_set import auto_group_tools_by_name_pattern
                            auto_group_tools_by_name_pattern(
                                tool_manager=self.agent.tool_manager,
                                rule_set=session.rule_set
                            )

            # Restore tools
            if restore_tools and checkpoint.tool_registry_data:
                if hasattr(self.agent, 'tool_manager') and self.agent.tool_manager:
                    self.agent.tool_manager.from_checkpoint(
                        checkpoint.tool_registry_data,
                        function_registry=function_registry
                    )
                    stats['restored_components'].append('tools')
                    stats['tools_restored'] = len(checkpoint.tool_registry_data.get('tools', {}))

            # Restore statistics
            if restore_statistics and checkpoint.statistics:
                self.agent.total_tokens_in = checkpoint.statistics.get('total_tokens_in', 0)
                self.agent.total_tokens_out = checkpoint.statistics.get('total_tokens_out', 0)
                self.agent.total_cost_accumulated = checkpoint.statistics.get('total_cost', 0.0)
                self.agent.total_llm_calls = checkpoint.statistics.get('total_llm_calls', 0)
                stats['restored_components'].append('statistics')

            # Note: Bind state restoration requires both agents to be present
            # This is handled separately in BindManager

        except Exception as e:
            stats['success'] = False
            stats['errors'].append(str(e))
            import traceback
            traceback.print_exc()

        return stats

    async def auto_restore(
        self,
        function_registry: dict[str, Callable] | None = None
    ) -> dict[str, Any]:
        """
        Auto-restore from latest checkpoint if available.
        Should be called after agent initialization.

        Returns:
            Restoration statistics or empty dict if no checkpoint
        """
        if self._loaded_checkpoint:
            return await self.restore(
                checkpoint=self._loaded_checkpoint,
                function_registry=function_registry
            )

        return {'success': False, 'error': 'No checkpoint loaded'}

    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================

    def list_checkpoints(self, max_age_hours: int | None = None) -> list[dict]:
        """
        List available checkpoints.

        Args:
            max_age_hours: Filter by max age (uses default if None)

        Returns:
            List of checkpoint info dicts
        """
        max_age = max_age_hours or self.max_age_hours

        if not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith('.pkl'):
                continue

            filepath = os.path.join(self.checkpoint_dir, filename)
            try:
                file_stat = os.stat(filepath)
                file_size = file_stat.st_size
                modified_time = datetime.fromtimestamp(file_stat.st_mtime)

                # Extract timestamp
                if filename.startswith('agent_checkpoint_'):
                    ts_str = filename.replace('agent_checkpoint_', '').replace('.pkl', '')
                    checkpoint_time = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    checkpoint_type = "regular"
                elif filename == 'final_checkpoint.pkl':
                    checkpoint_time = modified_time
                    checkpoint_type = "final"
                else:
                    continue

                age_hours = (datetime.now() - checkpoint_time).total_seconds() / 3600

                if age_hours <= max_age:
                    # Try to get summary
                    summary = "Unknown"
                    try:
                        cp = self._load_checkpoint_file(filepath)
                        summary = cp.get_summary()
                    except Exception:
                        pass

                    checkpoints.append({
                        'filepath': filepath,
                        'filename': filename,
                        'type': checkpoint_type,
                        'timestamp': checkpoint_time.isoformat(),
                        'age_hours': round(age_hours, 1),
                        'size_kb': round(file_size / 1024, 1),
                        'summary': summary
                    })

            except Exception:
                continue

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

        return checkpoints

    async def cleanup_old(self, keep_count: int | None = None) -> dict[str, Any]:
        """
        Delete old checkpoints, keeping newest N.

        Args:
            keep_count: Number to keep (uses max_checkpoints if None)

        Returns:
            Cleanup statistics
        """
        keep = keep_count or self.max_checkpoints

        checkpoints = self.list_checkpoints(max_age_hours=self.max_age_hours * 2)

        deleted = 0
        freed_kb = 0
        errors = []

        # Delete excess checkpoints (keep newest)
        for cp in checkpoints[keep:]:
            if cp['type'] == 'final':
                continue  # Never delete final checkpoint

            try:
                os.remove(cp['filepath'])
                deleted += 1
                freed_kb += cp['size_kb']
            except Exception as e:
                errors.append(f"Failed to delete {cp['filename']}: {e}")

        return {
            'deleted': deleted,
            'freed_kb': round(freed_kb, 1),
            'remaining': min(keep, len(checkpoints)),
            'errors': errors
        }

    async def delete_checkpoint(self, filename: str) -> bool:
        """Delete a specific checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filepath):
            return False

        try:
            os.remove(filepath)
            return True
        except Exception:
            return False

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_stats(self) -> dict:
        """Get checkpoint manager statistics"""
        checkpoints = self.list_checkpoints()
        total_size = sum(cp['size_kb'] for cp in checkpoints)

        return {
            'checkpoint_dir': self.checkpoint_dir,
            'total_checkpoints': len(checkpoints),
            'total_size_kb': round(total_size, 1),
            'max_checkpoints': self.max_checkpoints,
            'max_age_hours': self.max_age_hours,
            'last_checkpoint': self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            'has_loaded_checkpoint': self._loaded_checkpoint is not None
        }

    def __repr__(self) -> str:
        count = len(self.list_checkpoints())
        return f"<CheckpointManager {self.agent.amd.name} [{count} checkpoints]>"
