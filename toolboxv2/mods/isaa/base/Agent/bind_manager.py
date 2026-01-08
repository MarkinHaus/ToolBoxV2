"""
BindManager - Agent binding with live-sync via VFS

Provides:
- Public mode: All bound agents share one sync file
- Private mode: 1-to-1 bindings with separate sync files
- Live synchronization between agents

Author: FlowAgent V2
"""

import asyncio
import json
import weakref
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent


@dataclass
class SyncEntry:
    """Single sync log entry"""
    id: str
    timestamp: datetime
    source_agent: str
    action: str                        # 'message', 'tool_result', 'state_update', etc.
    data: Any
    acknowledged: bool = False
    acknowledged_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'source_agent': self.source_agent,
            'action': self.action,
            'data': self.data,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SyncEntry':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class BindConfig:
    """Configuration for a binding"""
    binding_id: str
    mode: str                          # 'public' or 'private'
    partner_name: str                  # Partner agent name
    sync_filename: str                 # VFS filename for sync
    created_at: datetime = field(default_factory=datetime.now)
    last_sync: datetime | None = None
    
    # Stats
    messages_sent: int = 0
    messages_received: int = 0


class BindManager:
    """
    Manages agent-to-agent bindings with live synchronization.
    
    Modes:
    - Public: All bound agents share one sync file, everyone sees everything
    - Private: 1-to-1 bindings, each pair has separate sync file
    
    Sync happens via VFS files that both agents can read/write.
    """

    def __init__(self, agent: 'FlowAgent'):
        """
        Initialize BindManager.
        
        Args:
            agent: Parent FlowAgent instance
        """
        self.agent = agent
        self.agent_name = agent.amd.name
        
        # Bindings: partner_name -> BindConfig
        self.bindings: dict[str, BindConfig] = {}
        
        # Partner references (weak to avoid circular refs)
        self._partners: dict[str, weakref.ref] = {}
        
        # Sync state
        self._sync_lock = asyncio.Lock()
        self._last_poll: dict[str, datetime] = {}
        
        # Public binding group (if in public mode)
        self._public_binding_id: str | None = None
        self._public_sync_filename: str | None = None

    def _generate_sync_filename(self, partner_name: str, mode: str) -> str:
        """Generate sync filename for binding"""
        if mode == 'public':
            # All public bindings use same file
            return f"_bind_sync_public_{self._public_binding_id}.json"
        else:
            # Private binding has unique file per pair
            names = sorted([self.agent_name, partner_name])
            return f"_bind_sync_private_{names[0]}_{names[1]}.json"

    def _get_sync_file_content(self, filename: str, session_id: str) -> list[SyncEntry]:
        """Read sync entries from VFS file"""
        session = self.agent.session_manager.get(session_id)
        if not session:
            return []
        
        result = session.vfs.read(filename)
        if not result['success']:
            return []
        
        try:
            data = json.loads(result['content'])
            return [SyncEntry.from_dict(e) for e in data.get('entries', [])]
        except Exception:
            return []

    def _write_sync_file(self, filename: str, entries: list[SyncEntry], session_id: str):
        """Write sync entries to VFS file"""
        session = self.agent.session_manager.get(session_id)
        if not session:
            return
        
        data = {
            'last_updated': datetime.now().isoformat(),
            'entries': [e.to_dict() for e in entries]
        }
        
        content = json.dumps(data, indent=2, ensure_ascii=False)
        
        if session.vfs.files.get(filename):
            session.vfs.write(filename, content)
        else:
            session.vfs.create(filename, content)

    # =========================================================================
    # BINDING OPERATIONS
    # =========================================================================

    async def bind(
        self,
        partner: 'FlowAgent',
        mode: str = 'public',
        session_id: str = 'default'
    ) -> BindConfig:
        """
        Bind to another agent.
        
        Args:
            partner: Partner FlowAgent to bind with
            mode: 'public' (all see all) or 'private' (1-to-1)
            session_id: Session to use for sync file
            
        Returns:
            BindConfig for this binding
        """
        import uuid
        
        partner_name = partner.amd.name
        
        # Check if already bound
        if partner_name in self.bindings:
            return self.bindings[partner_name]
        
        # Generate binding ID
        if mode == 'public':
            # Use existing public binding ID or create new
            if not self._public_binding_id:
                self._public_binding_id = f"pub_{uuid.uuid4().hex[:8]}"
            binding_id = self._public_binding_id
        else:
            binding_id = f"priv_{uuid.uuid4().hex[:8]}"
        
        # Create config
        sync_filename = self._generate_sync_filename(partner_name, mode)
        
        config = BindConfig(
            binding_id=binding_id,
            mode=mode,
            partner_name=partner_name,
            sync_filename=sync_filename
        )
        
        # Store binding
        self.bindings[partner_name] = config
        self._partners[partner_name] = weakref.ref(partner)
        
        if mode == 'public':
            self._public_sync_filename = sync_filename
        
        # Initialize sync file
        session = await self.agent.session_manager.get_or_create(session_id)
        self._write_sync_file(sync_filename, [], session_id)
        
        # Reciprocal binding on partner (if partner has BindManager)
        if hasattr(partner, 'bind_manager') and partner.bind_manager:
            if self.agent_name not in partner.bind_manager.bindings:
                partner_config = BindConfig(
                    binding_id=binding_id,
                    mode=mode,
                    partner_name=self.agent_name,
                    sync_filename=sync_filename
                )
                partner.bind_manager.bindings[self.agent_name] = partner_config
                partner.bind_manager._partners[self.agent_name] = weakref.ref(self.agent)
                
                if mode == 'public':
                    partner.bind_manager._public_binding_id = binding_id
                    partner.bind_manager._public_sync_filename = sync_filename
        
        return config

    def unbind(self, partner_name: str) -> bool:
        """
        Unbind from a partner agent.
        
        Args:
            partner_name: Name of partner to unbind
            
        Returns:
            True if unbound successfully
        """
        if partner_name not in self.bindings:
            return False
        
        config = self.bindings[partner_name]
        
        # Remove from partner if still referenced
        partner_ref = self._partners.get(partner_name)
        if partner_ref:
            partner = partner_ref()
            if partner and hasattr(partner, 'bind_manager'):
                if self.agent_name in partner.bind_manager.bindings:
                    del partner.bind_manager.bindings[self.agent_name]
                if self.agent_name in partner.bind_manager._partners:
                    del partner.bind_manager._partners[self.agent_name]
        
        # Clean up local state
        del self.bindings[partner_name]
        if partner_name in self._partners:
            del self._partners[partner_name]
        
        # If was public binding and no more bindings, clear public state
        if config.mode == 'public' and not any(
            b.mode == 'public' for b in self.bindings.values()
        ):
            self._public_binding_id = None
            self._public_sync_filename = None
        
        return True

    def unbind_all(self):
        """Unbind from all partners"""
        for partner_name in list(self.bindings.keys()):
            self.unbind(partner_name)

    def is_bound_to(self, partner_name: str) -> bool:
        """Check if bound to partner"""
        return partner_name in self.bindings

    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================

    async def write_sync(
        self,
        action: str,
        data: Any,
        target_partner: str | None = None,
        session_id: str = 'default'
    ):
        """
        Write sync entry for partners to read.
        
        Args:
            action: Action type ('message', 'tool_result', 'state_update')
            data: Data to sync
            target_partner: Specific partner (None = all in public mode)
            session_id: Session for VFS
        """
        import uuid
        
        async with self._sync_lock:
            entry = SyncEntry(
                id=f"sync_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                source_agent=self.agent_name,
                action=action,
                data=data
            )
            
            # Determine which bindings to update
            if target_partner:
                targets = [target_partner] if target_partner in self.bindings else []
            else:
                # All bindings (in public mode, just one file)
                targets = list(self.bindings.keys())
            
            # Group by sync file
            files_to_update: dict[str, list[str]] = {}
            for partner_name in targets:
                config = self.bindings[partner_name]
                if config.sync_filename not in files_to_update:
                    files_to_update[config.sync_filename] = []
                files_to_update[config.sync_filename].append(partner_name)
            
            # Update each sync file
            for filename, partners in files_to_update.items():
                entries = self._get_sync_file_content(filename, session_id)
                entries.append(entry)
                
                # Keep only last 100 entries
                if len(entries) > 100:
                    entries = entries[-100:]
                
                self._write_sync_file(filename, entries, session_id)
                
                # Update stats
                for partner_name in partners:
                    self.bindings[partner_name].messages_sent += 1
                    self.bindings[partner_name].last_sync = datetime.now()

    async def read_sync(
        self,
        partner_name: str | None = None,
        since: datetime | None = None,
        unacknowledged_only: bool = True,
        session_id: str = 'default'
    ) -> list[SyncEntry]:
        """
        Read sync entries from partners.
        
        Args:
            partner_name: Specific partner (None = all)
            since: Only entries after this time
            unacknowledged_only: Only unacknowledged entries
            session_id: Session for VFS
            
        Returns:
            List of SyncEntry objects
        """
        results = []
        
        # Determine which files to read
        if partner_name:
            if partner_name not in self.bindings:
                return []
            filenames = [self.bindings[partner_name].sync_filename]
        else:
            # Unique filenames from all bindings
            filenames = list(set(b.sync_filename for b in self.bindings.values()))
        
        for filename in filenames:
            entries = self._get_sync_file_content(filename, session_id)
            
            for entry in entries:
                # Skip own messages
                if entry.source_agent == self.agent_name:
                    continue
                
                # Filter by time
                if since and entry.timestamp <= since:
                    continue
                
                # Filter by acknowledgment
                if unacknowledged_only:
                    if entry.acknowledged or self.agent_name in entry.acknowledged_by:
                        continue
                
                results.append(entry)
        
        # Update stats
        for partner_name in self.bindings:
            self.bindings[partner_name].messages_received += len(results)
        
        return results

    async def acknowledge_sync(
        self,
        entry_id: str,
        session_id: str = 'default'
    ):
        """
        Acknowledge a sync entry.
        
        Args:
            entry_id: Entry ID to acknowledge
            session_id: Session for VFS
        """
        async with self._sync_lock:
            # Find and update in all sync files
            for config in self.bindings.values():
                entries = self._get_sync_file_content(config.sync_filename, session_id)
                
                for entry in entries:
                    if entry.id == entry_id:
                        if self.agent_name not in entry.acknowledged_by:
                            entry.acknowledged_by.append(self.agent_name)
                        
                        # Mark as acknowledged if all partners have acked
                        # (simplified: mark if this agent acked)
                        entry.acknowledged = True
                        
                        self._write_sync_file(config.sync_filename, entries, session_id)
                        return

    async def poll_sync(
        self,
        session_id: str = 'default'
    ) -> dict[str, list[SyncEntry]]:
        """
        Poll for new sync entries from all partners.
        
        Returns:
            Dict mapping partner_name -> list of new entries
        """
        results: dict[str, list[SyncEntry]] = {}
        
        for partner_name, config in self.bindings.items():
            since = self._last_poll.get(partner_name)
            
            entries = await self.read_sync(
                partner_name=partner_name,
                since=since,
                unacknowledged_only=True,
                session_id=session_id
            )
            
            if entries:
                results[partner_name] = entries
            
            self._last_poll[partner_name] = datetime.now()
        
        return results

    # =========================================================================
    # QUERIES
    # =========================================================================

    def list_bindings(self) -> list[BindConfig]:
        """Get all bindings"""
        return list(self.bindings.values())

    def get_binding(self, partner_name: str) -> BindConfig | None:
        """Get binding for specific partner"""
        return self.bindings.get(partner_name)

    def get_partner(self, partner_name: str) -> 'FlowAgent | None':
        """Get partner agent reference (may be None if GC'd)"""
        ref = self._partners.get(partner_name)
        if ref:
            return ref()
        return None

    def get_sync_history(
        self,
        partner_name: str,
        last_n: int = 20,
        session_id: str = 'default'
    ) -> list[SyncEntry]:
        """Get sync history with a partner"""
        if partner_name not in self.bindings:
            return []
        
        config = self.bindings[partner_name]
        entries = self._get_sync_file_content(config.sync_filename, session_id)
        
        return entries[-last_n:]

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize bindings for checkpoint"""
        return {
            'agent_name': self.agent_name,
            'public_binding_id': self._public_binding_id,
            'public_sync_filename': self._public_sync_filename,
            'bindings': {
                name: {
                    'binding_id': config.binding_id,
                    'mode': config.mode,
                    'partner_name': config.partner_name,
                    'sync_filename': config.sync_filename,
                    'created_at': config.created_at.isoformat(),
                    'last_sync': config.last_sync.isoformat() if config.last_sync else None,
                    'messages_sent': config.messages_sent,
                    'messages_received': config.messages_received
                }
                for name, config in self.bindings.items()
            }
        }

    def from_checkpoint(self, data: dict, partner_agents: dict[str, 'FlowAgent'] | None = None):
        """
        Restore bindings from checkpoint.
        
        Note: This only restores binding configs. Actual partner references
        must be re-established by calling bind() again or providing partner_agents.
        """
        self._public_binding_id = data.get('public_binding_id')
        self._public_sync_filename = data.get('public_sync_filename')
        
        partner_agents = partner_agents or {}
        
        for name, config_data in data.get('bindings', {}).items():
            config = BindConfig(
                binding_id=config_data['binding_id'],
                mode=config_data['mode'],
                partner_name=config_data['partner_name'],
                sync_filename=config_data['sync_filename'],
                messages_sent=config_data.get('messages_sent', 0),
                messages_received=config_data.get('messages_received', 0)
            )
            
            if config_data.get('created_at'):
                config.created_at = datetime.fromisoformat(config_data['created_at'])
            if config_data.get('last_sync'):
                config.last_sync = datetime.fromisoformat(config_data['last_sync'])
            
            self.bindings[name] = config
            
            # Restore partner reference if provided
            if name in partner_agents:
                self._partners[name] = weakref.ref(partner_agents[name])

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_stats(self) -> dict:
        """Get binding statistics"""
        total_sent = sum(b.messages_sent for b in self.bindings.values())
        total_received = sum(b.messages_received for b in self.bindings.values())
        
        return {
            'agent_name': self.agent_name,
            'total_bindings': len(self.bindings),
            'public_bindings': sum(1 for b in self.bindings.values() if b.mode == 'public'),
            'private_bindings': sum(1 for b in self.bindings.values() if b.mode == 'private'),
            'total_messages_sent': total_sent,
            'total_messages_received': total_received,
            'partners': list(self.bindings.keys())
        }

    def __repr__(self) -> str:
        return f"<BindManager {self.agent_name} [{len(self.bindings)} bindings]>"
