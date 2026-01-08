"""
AgentSession - Session-isolated context for FlowAgent

Provides:
- ChatSession integration for RAG and history
- Session-specific VFS with RuleSet integration
- Tool restrictions per session
- Complete lifecycle management

Author: FlowAgent V2
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.rule_set import RuleSet, RuleResult


# =============================================================================
# VFS FILE
# =============================================================================

@dataclass
class VFSFile:
    """Represents a file in the Virtual File System."""
    filename: str
    content: str
    state: str = "closed"              # "open" or "closed"
    view_start: int = 0
    view_end: int = -1
    mini_summary: str = ""
    readonly: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# VIRTUAL FILE SYSTEM
# =============================================================================

class VirtualFileSystem:
    """
    Virtual File System for token-efficient context management.

    Features:
    - open/closed states (only open files show in context)
    - Windowing (show only specific line ranges)
    - System files (read-only, auto-updated)
    - Auto-summary on close
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        max_window_lines: int = 250,
        summarizer: Callable[[str], str] | None = None
    ):
        self.session_id = session_id
        self.agent_name = agent_name
        self.max_window_lines = max_window_lines
        self._summarizer = summarizer

        self.files: dict[str, VFSFile] = {}
        self._dirty = True

        self._init_system_files()

    def _init_system_files(self):
        """Initialize read-only system files"""
        self.files["system_context"] = VFSFile(
            filename="system_context",
            content=self._build_system_context(),
            state="open",
            readonly=True
        )

    def _build_system_context(self) -> str:
        """Build system context content"""
        now = datetime.now()
        return f"""# System Context
Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}
Agent: {self.agent_name}
Session: {self.session_id}
"""

    def update_system_context(self):
        """Refresh system context"""
        if "system_context" in self.files:
            self.files["system_context"].content = self._build_system_context()
            self.files["system_context"].updated_at = datetime.now().isoformat()
            self._dirty = True

    def set_rules_file(self, content: str):
        """Set the active_rules file content (from RuleSet)"""
        if "active_rules" not in self.files:
            self.files["active_rules"] = VFSFile(
                filename="active_rules",
                content=content,
                state="open",
                readonly=True
            )
        else:
            self.files["active_rules"].content = content
            self.files["active_rules"].updated_at = datetime.now().isoformat()
        self._dirty = True

    # -------------------------------------------------------------------------
    # FILE OPERATIONS
    # -------------------------------------------------------------------------

    def create(self, filename: str, content: str = "") -> dict:
        """Create a new file"""
        if filename in self.files and self.files[filename].readonly:
            return {"success": False, "error": f"Cannot overwrite system file: {filename}"}

        self.files[filename] = VFSFile(filename=filename, content=content, state="closed")
        self._dirty = True
        return {"success": True, "message": f"Created '{filename}' ({len(content)} chars)"}

    def read(self, filename: str) -> dict:
        """Read file content"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}
        return {"success": True, "content": self.files[filename].content}

    def write(self, filename: str, content: str) -> dict:
        """Write/overwrite file content"""
        if filename in self.files and self.files[filename].readonly:
            return {"success": False, "error": f"Read-only: {filename}"}

        if filename not in self.files:
            return self.create(filename, content)

        self.files[filename].content = content
        self.files[filename].updated_at = datetime.now().isoformat()
        self._dirty = True
        return {"success": True, "message": f"Updated '{filename}'"}

    def append(self, filename: str, content: str) -> dict:
        """Append to file"""
        if filename not in self.files:
            return self.create(filename, content)

        if self.files[filename].readonly:
            return {"success": False, "error": f"Read-only: {filename}"}

        self.files[filename].content += content
        self.files[filename].updated_at = datetime.now().isoformat()
        self._dirty = True
        return {"success": True, "message": f"Appended to '{filename}'"}

    def edit(self, filename: str, line_start: int, line_end: int, new_content: str) -> dict:
        """Edit file by replacing lines (1-indexed)"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.readonly:
            return {"success": False, "error": f"Read-only: {filename}"}

        lines = f.content.split('\n')
        start_idx = max(0, line_start - 1)
        end_idx = min(len(lines), line_end)

        new_lines = new_content.split('\n')
        lines = lines[:start_idx] + new_lines + lines[end_idx:]

        f.content = '\n'.join(lines)
        f.updated_at = datetime.now().isoformat()
        self._dirty = True
        return {"success": True, "message": f"Edited {filename} lines {line_start}-{line_end}"}

    def insert_lines(self, filename: str, after_line: int, content: str) -> dict:
        """Insert lines after specified line (1-indexed, 0 = at beginning)"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.readonly:
            return {"success": False, "error": f"Read-only: {filename}"}

        lines = f.content.split('\n')
        insert_idx = max(0, min(after_line, len(lines)))

        new_lines = content.split('\n')
        lines = lines[:insert_idx] + new_lines + lines[insert_idx:]

        f.content = '\n'.join(lines)
        f.updated_at = datetime.now().isoformat()
        self._dirty = True
        return {"success": True, "message": f"Inserted {len(new_lines)} lines after line {after_line}"}

    def delete_lines(self, filename: str, line_start: int, line_end: int) -> dict:
        """Delete lines from file (1-indexed, inclusive)"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.readonly:
            return {"success": False, "error": f"Read-only: {filename}"}

        lines = f.content.split('\n')
        start_idx = max(0, line_start - 1)
        end_idx = min(len(lines), line_end)

        deleted_count = end_idx - start_idx
        lines = lines[:start_idx] + lines[end_idx:]

        f.content = '\n'.join(lines)
        f.updated_at = datetime.now().isoformat()
        self._dirty = True
        return {"success": True, "message": f"Deleted {deleted_count} lines ({line_start}-{line_end})"}

    def replace_text(self, filename: str, old_text: str, new_text: str, count: int = 1) -> dict:
        """Find and replace text in file"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.readonly:
            return {"success": False, "error": f"Read-only: {filename}"}

        if old_text not in f.content:
            return {"success": False, "error": f"Text not found in {filename}"}

        f.content = f.content.replace(old_text, new_text, count)
        f.updated_at = datetime.now().isoformat()
        self._dirty = True
        return {"success": True, "message": f"Replaced {count} occurrence(s)"}

    def get_file_info(self, filename: str) -> dict:
        """Get file metadata without content"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        return {
            "success": True,
            "filename": filename,
            "state": f.state,
            "readonly": f.readonly,
            "size": len(f.content),
            "lines": len(f.content.splitlines()),
            "summary": f.mini_summary if f.state == "closed" else None,
            "created_at": f.created_at,
            "updated_at": f.updated_at,
            "view_range": (f.view_start + 1, f.view_end) if f.state == "open" else None
        }

    def list_closed_with_summaries(self) -> list[dict]:
        """List closed files with their summaries"""
        closed = []
        for name, f in self.files.items():
            if f.state == "closed" and not f.readonly:
                closed.append({
                    "filename": name,
                    "size": len(f.content),
                    "lines": len(f.content.splitlines()),
                    "summary": f.mini_summary or f"[{len(f.content)} chars]"
                })
        return closed

    def count_open_files(self) -> int:
        """Count currently open files (excluding system files)"""
        return sum(1 for f in self.files.values() if f.state == "open" and not f.readonly)

    def delete(self, filename: str) -> dict:
        """Delete a file"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        if self.files[filename].readonly:
            return {"success": False, "error": f"Cannot delete system file: {filename}"}

        del self.files[filename]
        self._dirty = True
        return {"success": True, "message": f"Deleted '{filename}'"}

    # -------------------------------------------------------------------------
    # OPEN/CLOSE OPERATIONS
    # -------------------------------------------------------------------------

    def open(self, filename: str, line_start: int = 1, line_end: int = -1) -> dict:
        """Open file (make content visible in context)"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        f.state = "open"
        f.view_start = max(0, line_start - 1)
        f.view_end = line_end

        lines = f.content.split('\n')
        end = line_end if line_end > 0 else len(lines)
        visible = lines[f.view_start:end]

        self._dirty = True
        return {
            "success": True,
            "message": f"Opened '{filename}' (lines {line_start}-{end})",
            "preview": '\n'.join(visible[:5]) + ("..." if len(visible) > 5 else "")
        }

    async def close(self, filename: str) -> dict:
        """Close file (create summary, remove from context)"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.readonly:
            return {"success": False, "error": f"Cannot close system file: {filename}"}

        # Generate summary
        if len(f.content) > 100 and self._summarizer:
            try:
                summary = self._summarizer(f.content[:2000])
                if hasattr(summary, '__await__'):
                    summary = await summary
                f.mini_summary = str(summary).strip()
            except Exception:
                f.mini_summary = f"[{len(f.content)} chars, {len(f.content.splitlines())} lines]"
        else:
            f.mini_summary = f"[{len(f.content)} chars]"

        f.state = "closed"
        self._dirty = True
        return {"success": True, "summary": f.mini_summary}

    def view(self, filename: str, line_start: int = 1, line_end: int = -1) -> dict:
        """View/adjust visible window"""
        if filename not in self.files:
            return {"success": False, "error": f"File not found: {filename}"}

        f = self.files[filename]
        if f.state != "open":
            return self.open(filename, line_start, line_end)

        f.view_start = max(0, line_start - 1)
        f.view_end = line_end

        lines = f.content.split('\n')
        end = line_end if line_end > 0 else len(lines)

        self._dirty = True
        return {"success": True, "content": '\n'.join(lines[f.view_start:end])}

    def list_files(self) -> dict:
        """List all files with metadata"""
        listing = []
        for name, f in self.files.items():
            info = {
                "filename": name,
                "state": f.state,
                "readonly": f.readonly,
                "size": len(f.content),
                "lines": len(f.content.splitlines())
            }
            if f.state == "closed" and f.mini_summary:
                info["summary"] = f.mini_summary
            listing.append(info)
        return {"success": True, "files": listing}

    # -------------------------------------------------------------------------
    # LOCAL FILE OPERATIONS (Safe load/save to real filesystem)
    # -------------------------------------------------------------------------

    def load_from_local(
        self,
        local_path: str,
        vfs_name: str | None = None,
        allowed_dirs: list[str] | None = None,
        max_size_bytes: int = 1024 * 1024  # 1MB default
    ) -> dict:
        """
        Safely load a local file into VFS.

        Args:
            local_path: Path to local file
            vfs_name: Name in VFS (default: basename of local_path)
            allowed_dirs: List of allowed directories (security)
            max_size_bytes: Maximum file size to load

        Returns:
            Result dict with success status
        """
        try:
            resolved_path = os.path.abspath(os.path.expanduser(local_path))
        except Exception as e:
            return {"success": False, "error": f"Invalid path: {e}"}

        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                allowed_resolved = os.path.abspath(os.path.expanduser(allowed_dir))
                if resolved_path.startswith(allowed_resolved):
                    allowed = True
                    break
            if not allowed:
                return {"success": False, "error": f"Path not in allowed directories: {resolved_path}"}

        if not os.path.exists(resolved_path):
            return {"success": False, "error": f"File not found: {resolved_path}"}

        if not os.path.isfile(resolved_path):
            return {"success": False, "error": f"Not a file: {resolved_path}"}

        file_size = os.path.getsize(resolved_path)
        if file_size > max_size_bytes:
            return {"success": False, "error": f"File too large: {file_size} bytes (max: {max_size_bytes})"}

        if vfs_name is None:
            vfs_name = os.path.basename(resolved_path)

        try:
            with open(resolved_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return {"success": False, "error": f"Read error: {e}"}

        result = self.create(vfs_name, content)

        if result['success']:
            return {
                "success": True,
                "vfs_name": vfs_name,
                "source_path": resolved_path,
                "size_bytes": len(content),
                "lines": len(content.splitlines())
            }

        return result

    def save_to_local(
        self,
        vfs_name: str,
        local_path: str,
        allowed_dirs: list[str] | None = None,
        overwrite: bool = False,
        create_dirs: bool = True
    ) -> dict:
        """
        Safely save a VFS file to local filesystem.

        Args:
            vfs_name: Name of file in VFS
            local_path: Destination path
            allowed_dirs: List of allowed directories (security)
            overwrite: Allow overwriting existing files
            create_dirs: Create parent directories if needed

        Returns:
            Result dict with success status
        """
        if vfs_name not in self.files:
            return {"success": False, "error": f"VFS file not found: {vfs_name}"}

        vfs_file = self.files[vfs_name]

        try:
            resolved_path = os.path.abspath(os.path.expanduser(local_path))
        except Exception as e:
            return {"success": False, "error": f"Invalid path: {e}"}

        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                allowed_resolved = os.path.abspath(os.path.expanduser(allowed_dir))
                if resolved_path.startswith(allowed_resolved):
                    allowed = True
                    break
            if not allowed:
                return {"success": False, "error": f"Path not in allowed directories: {resolved_path}"}

        if os.path.exists(resolved_path) and not overwrite:
            return {"success": False, "error": f"File exists (use overwrite=True): {resolved_path}"}

        parent_dir = os.path.dirname(resolved_path)
        if parent_dir and not os.path.exists(parent_dir):
            if create_dirs:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except Exception as e:
                    return {"success": False, "error": f"Cannot create directory: {e}"}
            else:
                return {"success": False, "error": f"Parent directory does not exist: {parent_dir}"}

        try:
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.write(vfs_file.content)
        except Exception as e:
            return {"success": False, "error": f"Write error: {e}"}

        return {
            "success": True,
            "vfs_name": vfs_name,
            "saved_path": resolved_path,
            "size_bytes": len(vfs_file.content),
            "lines": len(vfs_file.content.splitlines())
        }

    # -------------------------------------------------------------------------
    # CONTEXT BUILDING
    # -------------------------------------------------------------------------

    def build_context_string(self) -> str:
        """Build VFS context string for LLM"""
        self.update_system_context()

        parts = ["=== VFS (Virtual File System) ==="]

        # Order: system_context, active_rules, then others
        ordered = []
        if "system_context" in self.files:
            ordered.append(("system_context", self.files["system_context"]))
        if "active_rules" in self.files:
            ordered.append(("active_rules", self.files["active_rules"]))

        for name, f in self.files.items():
            if name not in ("system_context", "active_rules"):
                ordered.append((name, f))

        for name, f in ordered:
            if f.state == "open":
                lines = f.content.split('\n')
                end = f.view_end if f.view_end > 0 else len(lines)
                visible = lines[f.view_start:end]

                if len(visible) > self.max_window_lines:
                    visible = visible[:self.max_window_lines]
                    parts.append(f"\n[{name}] OPEN (lines {f.view_start + 1}-{f.view_start + self.max_window_lines}, truncated):")
                else:
                    parts.append(f"\n[{name}] OPEN (lines {f.view_start + 1}-{end}):")
                parts.append('\n'.join(visible))
            else:
                summary = f.mini_summary or f"[{len(f.content)} chars]"
                parts.append(f"\n• {name} [closed]: {summary}")

        return '\n'.join(parts)

    # -------------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------------

    def to_checkpoint(self) -> dict:
        """Serialize VFS for checkpoint"""
        return {
            'session_id': self.session_id,
            'agent_name': self.agent_name,
            'max_window_lines': self.max_window_lines,
            'files': {
                name: asdict(f) for name, f in self.files.items()
                if not f.readonly  # Don't save system files
            }
        }

    def from_checkpoint(self, data: dict):
        """Restore VFS from checkpoint"""
        for name, file_data in data.get('files', {}).items():
            self.files[name] = VFSFile(**file_data)
        self._dirty = True


# =============================================================================
# AGENT SESSION
# =============================================================================
def retrieval_to_llm_context_compact(data, max_entries=5):
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

                lines.append(
                    f"{role}: [{text}]| is: {concepts} | r={relevance:.2f}"
                )
        lines.append("\n")
    return "\n".join(lines)

class AgentSession:
    """
    Session-isolated context encapsulating:
    - ChatSession for RAG and conversation history
    - VirtualFileSystem for token-efficient file management
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
        summarizer: Callable | None = None
    ):
        """
        Initialize AgentSession.

        Args:
            session_id: Unique session identifier
            agent_name: Name of the parent agent
            memory_instance: AISemanticMemory instance for ChatSession
            max_history: Maximum conversation history length
            vfs_max_window_lines: Max lines to show per VFS file
            rule_config_path: Optional path to RuleSet config
            summarizer: Optional async function for VFS summaries
        """
        self.session_id = session_id
        self.agent_name = agent_name
        self._memory = memory_instance

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

        # VFS - session specific
        self.vfs = VirtualFileSystem(
            session_id=session_id,
            agent_name=agent_name,
            max_window_lines=vfs_max_window_lines,
            summarizer=summarizer
        )

        # RuleSet - session specific
        from toolboxv2.mods.isaa.base.Agent.rule_set import RuleSet, create_default_ruleset
        self.rule_set: RuleSet = create_default_ruleset(config_path=rule_config_path)

        # Sync RuleSet to VFS
        self._sync_ruleset_to_vfs()

        # State
        self._initialized = False
        self._closed = False

    async def initialize(self):
        """Async initialization - must be called after __init__"""
        if self._initialized:
            return

        # Create ChatSession
        from toolboxv2.mods.isaa.extras.session import ChatSession

        space_name = f"ChatSession/{self.agent_name}.{self.session_id}.unified"
        self._chat_session = ChatSession(
            self._memory,
            max_length=self._max_history,
            space_name=space_name
        )

        self._initialized = True

    def _ensure_initialized(self):
        """Ensure session is initialized"""
        if not self._initialized:
            raise RuntimeError(
                f"AgentSession '{self.session_id}' not initialized. "
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

    async def add_message(self, message: dict, **kwargs):
        """
        Add message to conversation history.

        Args:
            message: Dict with 'role' and 'content'
            **kwargs: Additional metadata for the message
        """
        self._ensure_initialized()
        self._update_activity()

        await self._chat_session.add_message(message, **kwargs)

    async def get_reference(self, text: str, concepts=False, **kwargs) -> str:
        """
        Query RAG for relevant context.

        Args:
            text: Query text
            **kwargs: Additional query parameters

        Returns:
            Relevant context string
        """
        self._ensure_initialized()
        self._update_activity()
        kwargs["row"] = True
        res = await self._chat_session.get_reference(text, **kwargs)
        return res if concepts else retrieval_to_llm_context_compact(res, max_entries=kwargs.get("max_entries", 5))

    def get_history(self, last_n: int | None = None) -> list[dict]:
        """
        Get conversation history.

        Args:
            last_n: Number of recent messages (None = all)

        Returns:
            List of message dicts
        """
        self._ensure_initialized()

        if last_n is None:
            return self._chat_session.history.copy()
        return self._chat_session.get_past_x(last_n)

    def get_history_for_llm(self, last_n: int = 10) -> list[dict]:
        """
        Get history formatted for LLM context.

        Args:
            last_n: Number of recent messages

        Returns:
            List of messages with role and content
        """
        self._ensure_initialized()

        history = self._chat_session.get_start_with_last_user(last_n)

        # Ensure proper format
        formatted = []
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role.startswith('s'):
                role = 'system'
            elif role.startswith('u'):
                role = 'user'
            elif role.startswith('a'):
                role = 'assistant'

            formatted.append({'role': role, 'content': content})

        return formatted

    # =========================================================================
    # VFS METHODS
    # =========================================================================

    def vfs_create(self, filename: str, content: str = "") -> dict:
        """Create VFS file"""
        self._update_activity()
        return self.vfs.create(filename, content)

    def vfs_read(self, filename: str) -> dict:
        """Read VFS file"""
        return self.vfs.read(filename)

    def vfs_write(self, filename: str, content: str) -> dict:
        """Write VFS file"""
        self._update_activity()
        return self.vfs.write(filename, content)

    def vfs_open(self, filename: str, line_start: int = 1, line_end: int = -1) -> dict:
        """Open VFS file"""
        self._update_activity()
        return self.vfs.open(filename, line_start, line_end)

    async def vfs_close(self, filename: str) -> dict:
        """Close VFS file with summary"""
        self._update_activity()
        return await self.vfs.close(filename)

    def vfs_list(self) -> dict:
        """List VFS files"""
        return self.vfs.list_files()

    def build_vfs_context(self) -> str:
        """Build VFS context for LLM"""
        self._sync_ruleset_to_vfs()
        return self.vfs.build_context_string()

    # =========================================================================
    # RULESET METHODS
    # =========================================================================

    def get_current_rule_set(self) -> dict:
        """Get current rule set state"""
        return self.rule_set.get_current_rule_set()

    def rule_on_action(self, action: str, context: dict | None = None) -> 'RuleResult':
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
        # Default: allowed unless explicitly restricted
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
        """
        Close session - persist VFS and save state.
        Should be called when session ends.
        """
        if self._closed:
            return

        # Close all open VFS files
        for filename, f in list(self.vfs.files.items()):
            if f.state == "open" and not f.readonly:
                await self.vfs.close(filename)

        # Save ChatSession
        if self._chat_session:
            self._chat_session.on_exit()

        self._closed = True

    async def cleanup(self):
        """Clean up resources"""
        await self.close()

        # Clear VFS
        self.vfs.files.clear()
        self.vfs._init_system_files()

        # Clear rule set state
        self.rule_set.clear_situation()

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize session for checkpoint"""
        self._chat_session.on_exit() if self._chat_session else None
        return {
            'session_id': self.session_id,
            'agent_name': self.agent_name,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'metadata': self.metadata,
            'tool_restrictions': self.tool_restrictions,
            'vfs': self.vfs.to_checkpoint(),
            'rule_set': self.rule_set.to_checkpoint(),
            'chat_history': self._chat_session.history if self._chat_session else [],
            'max_history': self._max_history,
            'kb': self._chat_session.mem.save_memory(self._chat_session.space_name, None) if self._chat_session else None
        }

    @classmethod
    async def from_checkpoint(
        cls,
        data: dict,
        memory_instance: Any,
        summarizer: Callable | None = None
    ) -> 'AgentSession':
        """
        Restore session from checkpoint.

        Args:
            data: Checkpoint data
            memory_instance: AISemanticMemory instance
            summarizer: Optional summarizer function

        Returns:
            Restored AgentSession
        """
        session = cls(
            session_id=data['session_id'],
            agent_name=data['agent_name'],
            memory_instance=memory_instance,
            max_history=data.get('max_history', 100),
            summarizer=summarizer
        )

        # Restore timestamps
        session.created_at = datetime.fromisoformat(data['created_at'])
        session.last_activity = datetime.fromisoformat(data['last_activity'])

        # Restore metadata
        session.metadata = data.get('metadata', {})

        # Restore tool restrictions
        session.tool_restrictions = data.get('tool_restrictions', {})

        # Restore VFS
        session.vfs.from_checkpoint(data.get('vfs', {}))

        # Restore RuleSet
        session.rule_set.from_checkpoint(data.get('rule_set', {}))

        # Initialize ChatSession
        await session.initialize()

        # Restore chat history
        if session._chat_session and data.get('chat_history'):
            session._chat_session.history = data['chat_history']

        # Restore knowledge base
        if session._chat_session and data.get('kb') and session._chat_session.get_volume() == 0:
            session._chat_session.mem.load_memory(session._chat_session.space_name, data['kb'])

        session._sync_ruleset_to_vfs()

        return session

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_stats(self) -> dict:
        """Get session statistics"""
        return {
            'session_id': self.session_id,
            'agent_name': self.agent_name,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'age_seconds': (datetime.now() - self.created_at).total_seconds(),
            'idle_seconds': (datetime.now() - self.last_activity).total_seconds(),
            'history_length': len(self._chat_session.history) if self._chat_session else 0,
            'vfs_files': len(self.vfs.files),
            'vfs_open_files': sum(1 for f in self.vfs.files.values() if f.state == "open"),
            'tool_restrictions': len(self.tool_restrictions),
            'active_rules': len(self.rule_set.get_active_rules()),
            'current_situation': self.rule_set.current_situation,
            'current_intent': self.rule_set.current_intent
        }

    def __repr__(self) -> str:
        status = "closed" if self._closed else ("initialized" if self._initialized else "created")
        return f"<AgentSession {self.session_id} [{status}]>"
