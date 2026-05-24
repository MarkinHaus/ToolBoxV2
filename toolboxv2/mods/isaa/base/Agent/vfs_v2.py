"""
VirtualFileSystem V2 - Enhanced VFS with Directories, FileTypes, and LSP Integration

Features:
- Hierarchical directory structure (mkdir, rmdir, mv, ls)
- File type detection with LSP integration
- Executable flag for runnable files
- Token-efficient context management

Author: FlowAgent V2
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.lsp_manager import Diagnostic, LSPManager


# =============================================================================
# FILE TYPES
# =============================================================================


class FileCategory(Enum):
    """High-level file categories"""

    CODE = auto()
    WEB = auto()
    DATA = auto()
    CONFIG = auto()
    DOCS = auto()
    BINARY = auto()
    UNKNOWN = auto()


@dataclass
class FileTypeInfo:
    """Information about a file type"""

    extension: str
    category: FileCategory
    language_id: str  # LSP language identifier
    mime_type: str
    is_executable: bool = False
    lsp_server: str | None = None  # e.g., "pylsp", "typescript-language-server"
    icon: str = "📄"
    description: str = ""


# File type registry
FILE_TYPES: dict[str, FileTypeInfo] = {
    # Python
    ".py": FileTypeInfo(
        ".py", FileCategory.CODE, "python", "text/x-python", True, "pylsp", "🐍", "Python"
    ),
    ".pyw": FileTypeInfo(
        ".pyw",
        FileCategory.CODE,
        "python",
        "text/x-python",
        True,
        "pylsp",
        "🐍",
        "Python (Windows)",
    ),
    ".pyi": FileTypeInfo(
        ".pyi",
        FileCategory.CODE,
        "python",
        "text/x-python",
        False,
        "pylsp",
        "🐍",
        "Python Stub",
    ),
    # JavaScript/TypeScript
    ".js": FileTypeInfo(
        ".js",
        FileCategory.CODE,
        "javascript",
        "application/javascript",
        True,
        "typescript-language-server",
        "📜",
        "JavaScript",
    ),
    ".mjs": FileTypeInfo(
        ".mjs",
        FileCategory.CODE,
        "javascript",
        "application/javascript",
        True,
        "typescript-language-server",
        "📜",
        "JavaScript Module",
    ),
    ".jsx": FileTypeInfo(
        ".jsx",
        FileCategory.CODE,
        "javascriptreact",
        "text/jsx",
        True,
        "typescript-language-server",
        "⚛️",
        "React JSX",
    ),
    ".ts": FileTypeInfo(
        ".ts",
        FileCategory.CODE,
        "typescript",
        "application/typescript",
        True,
        "typescript-language-server",
        "📘",
        "TypeScript",
    ),
    ".tsx": FileTypeInfo(
        ".tsx",
        FileCategory.CODE,
        "typescriptreact",
        "text/tsx",
        True,
        "typescript-language-server",
        "⚛️",
        "React TSX",
    ),
    # Rust
    ".rs": FileTypeInfo(
        ".rs",
        FileCategory.CODE,
        "rust",
        "text/x-rust",
        True,
        "rust-analyzer",
        "🦀",
        "Rust",
    ),
    # Go
    ".go": FileTypeInfo(
        ".go", FileCategory.CODE, "go", "text/x-go", True, "gopls", "🐹", "Go"
    ),
    # C/C++
    ".c": FileTypeInfo(
        ".c", FileCategory.CODE, "c", "text/x-c", True, "clangd", "🔧", "C"
    ),
    ".h": FileTypeInfo(
        ".h", FileCategory.CODE, "c", "text/x-c", False, "clangd", "🔧", "C Header"
    ),
    ".cpp": FileTypeInfo(
        ".cpp", FileCategory.CODE, "cpp", "text/x-c++", True, "clangd", "🔧", "C++"
    ),
    ".hpp": FileTypeInfo(
        ".hpp",
        FileCategory.CODE,
        "cpp",
        "text/x-c++",
        False,
        "clangd",
        "🔧",
        "C++ Header",
    ),
    ".cc": FileTypeInfo(
        ".cc", FileCategory.CODE, "cpp", "text/x-c++", True, "clangd", "🔧", "C++"
    ),
    # Web
    ".html": FileTypeInfo(
        ".html",
        FileCategory.WEB,
        "html",
        "text/html",
        False,
        "vscode-html-language-server",
        "🌐",
        "HTML",
    ),
    ".htm": FileTypeInfo(
        ".htm",
        FileCategory.WEB,
        "html",
        "text/html",
        False,
        "vscode-html-language-server",
        "🌐",
        "HTML",
    ),
    ".css": FileTypeInfo(
        ".css",
        FileCategory.WEB,
        "css",
        "text/css",
        False,
        "vscode-css-language-server",
        "🎨",
        "CSS",
    ),
    ".scss": FileTypeInfo(
        ".scss",
        FileCategory.WEB,
        "scss",
        "text/x-scss",
        False,
        "vscode-css-language-server",
        "🎨",
        "SCSS",
    ),
    ".less": FileTypeInfo(
        ".less",
        FileCategory.WEB,
        "less",
        "text/x-less",
        False,
        "vscode-css-language-server",
        "🎨",
        "Less",
    ),
    ".vue": FileTypeInfo(
        ".vue", FileCategory.WEB, "vue", "text/x-vue", False, "volar", "💚", "Vue"
    ),
    ".svelte": FileTypeInfo(
        ".svelte",
        FileCategory.WEB,
        "svelte",
        "text/x-svelte",
        False,
        "svelte-language-server",
        "🔥",
        "Svelte",
    ),
    # Data
    ".json": FileTypeInfo(
        ".json",
        FileCategory.DATA,
        "json",
        "application/json",
        False,
        "vscode-json-language-server",
        "📋",
        "JSON",
    ),
    ".jsonc": FileTypeInfo(
        ".jsonc",
        FileCategory.DATA,
        "jsonc",
        "application/json",
        False,
        "vscode-json-language-server",
        "📋",
        "JSON with Comments",
    ),
    ".xml": FileTypeInfo(
        ".xml", FileCategory.DATA, "xml", "application/xml", False, None, "📋", "XML"
    ),
    ".csv": FileTypeInfo(
        ".csv", FileCategory.DATA, "csv", "text/csv", False, None, "📊", "CSV"
    ),
    ".tsv": FileTypeInfo(
        ".tsv",
        FileCategory.DATA,
        "tsv",
        "text/tab-separated-values",
        False,
        None,
        "📊",
        "TSV",
    ),
    # Config
    ".yaml": FileTypeInfo(
        ".yaml",
        FileCategory.CONFIG,
        "yaml",
        "text/yaml",
        False,
        "yaml-language-server",
        "⚙️",
        "YAML",
    ),
    ".yml": FileTypeInfo(
        ".yml",
        FileCategory.CONFIG,
        "yaml",
        "text/yaml",
        False,
        "yaml-language-server",
        "⚙️",
        "YAML",
    ),
    ".toml": FileTypeInfo(
        ".toml", FileCategory.CONFIG, "toml", "text/x-toml", False, "taplo", "⚙️", "TOML"
    ),
    ".ini": FileTypeInfo(
        ".ini", FileCategory.CONFIG, "ini", "text/x-ini", False, None, "⚙️", "INI"
    ),
    ".env": FileTypeInfo(
        ".env",
        FileCategory.CONFIG,
        "dotenv",
        "text/plain",
        False,
        None,
        "🔐",
        "Environment",
    ),
    ".editorconfig": FileTypeInfo(
        ".editorconfig",
        FileCategory.CONFIG,
        "editorconfig",
        "text/plain",
        False,
        None,
        "⚙️",
        "EditorConfig",
    ),
    # Docs
    ".md": FileTypeInfo(
        ".md",
        FileCategory.DOCS,
        "markdown",
        "text/markdown",
        False,
        None,
        "📝",
        "Markdown",
    ),
    ".mdx": FileTypeInfo(
        ".mdx", FileCategory.DOCS, "mdx", "text/mdx", False, None, "📝", "MDX"
    ),
    ".rst": FileTypeInfo(
        ".rst",
        FileCategory.DOCS,
        "restructuredtext",
        "text/x-rst",
        False,
        None,
        "📝",
        "reStructuredText",
    ),
    ".txt": FileTypeInfo(
        ".txt",
        FileCategory.DOCS,
        "plaintext",
        "text/plain",
        False,
        None,
        "📄",
        "Plain Text",
    ),
    # Shell
    ".sh": FileTypeInfo(
        ".sh",
        FileCategory.CODE,
        "shellscript",
        "text/x-shellscript",
        True,
        "bash-language-server",
        "🐚",
        "Shell Script",
    ),
    ".bash": FileTypeInfo(
        ".bash",
        FileCategory.CODE,
        "shellscript",
        "text/x-shellscript",
        True,
        "bash-language-server",
        "🐚",
        "Bash Script",
    ),
    ".zsh": FileTypeInfo(
        ".zsh",
        FileCategory.CODE,
        "shellscript",
        "text/x-shellscript",
        True,
        None,
        "🐚",
        "Zsh Script",
    ),
    ".ps1": FileTypeInfo(
        ".ps1",
        FileCategory.CODE,
        "powershell",
        "text/x-powershell",
        True,
        None,
        "💠",
        "PowerShell",
    ),
    # SQL
    ".sql": FileTypeInfo(
        ".sql", FileCategory.CODE, "sql", "text/x-sql", True, None, "🗃️", "SQL"
    ),
    # Docker
    "Dockerfile": FileTypeInfo(
        "Dockerfile",
        FileCategory.CONFIG,
        "dockerfile",
        "text/x-dockerfile",
        False,
        "dockerfile-language-server",
        "🐳",
        "Dockerfile",
    ),
    ".dockerfile": FileTypeInfo(
        ".dockerfile",
        FileCategory.CONFIG,
        "dockerfile",
        "text/x-dockerfile",
        False,
        "dockerfile-language-server",
        "🐳",
        "Dockerfile",
    ),
    # TB (ToolBoxV2 Language)
    ".tb": FileTypeInfo(
        ".tb", FileCategory.CODE, "tb", "text/x-tb", True, None, "🧰", "TB Language"
    ),
}

_UNKNOWN_TYPE_CACHE: dict[str, FileTypeInfo] = {}

def get_file_type(filename: str) -> FileTypeInfo:
    """Get file type info — with caching for unknown extensions."""
    # Check exact filename match first
    if filename in FILE_TYPES:
        return FILE_TYPES[filename]

    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower in FILE_TYPES:
        return FILE_TYPES[ext_lower]

    # Cache unknown types by extension instead of creating new objects
    if ext_lower not in _UNKNOWN_TYPE_CACHE:
        _UNKNOWN_TYPE_CACHE[ext_lower] = FileTypeInfo(
            ext_lower or "",
            FileCategory.UNKNOWN,
            "plaintext",
            "text/plain",
            False,
            None,
            "📄",
            "Unknown",
        )
    return _UNKNOWN_TYPE_CACHE[ext_lower]

# =============================================================================
# VFS FILE
# =============================================================================


@dataclass(slots=True)
class VFSDirectory:
    """Represents a directory in the Virtual File System"""

    name: str
    created_at: str = ""
    updated_at: str = ""
    readonly: bool = False


class FileBackingType(Enum):
    """Wo lebt die Datei wirklich?"""

    MEMORY = auto()  # Nur im VFS (wie bisher)
    SHADOW = auto()  # Lazy-loaded vom lokalen FS
    MODIFIED = auto()  # Shadow, aber mit lokalen Änderungen (dirty)


@dataclass(slots=True)
class ShadowMount:
    """Ein gemounteter lokaler Ordner"""

    vfs_path: str  # z.B. "/project"
    local_path: str  # z.B. "/home/user/myproject"
    allowed_extensions: list[str] | None = None
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "__pycache__",
            "*.pyc",
            ".git",
            "node_modules",
            ".venv",
            "*.log",
        ]
    )
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    readonly: bool = False
    auto_sync: bool = True  # Änderungen sofort schreiben


@dataclass(slots=True)
class VFSFile:
    """Erweiterte VFS-Datei mit Shadow-Support"""

    filename: str
    backing_type: FileBackingType = FileBackingType.MEMORY

    # Content Management
    _content: str | None = None  # None = not loaded yet
    _content_hash: str | None = None  # Für dirty detection

    # Shadow-specific
    local_path: str | None = None  # Backing file path
    local_mtime: float | None = None  # Für change detection

    # Metadata (immer verfügbar, auch bei Shadow)
    size_bytes: int = 0
    line_count: int = 0
    file_type: FileTypeInfo | None = None

    # State
    state: str = "closed"
    view_start: int = 0
    view_end: int = -1
    mini_summary: str = ""
    readonly: bool = False
    is_dirty: bool = False  # Hat ungespeicherte Änderungen
    show_full: bool = False  # Gesamten Inhalt anzeigen, max_window_lines ignorieren

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    @property
    def size(self) -> int:
        return self.size_bytes if not self.is_loaded else len(self.content)

    @property
    def content(self) -> str:
        """Lazy content access - lädt bei Shadow on-demand"""
        if self._content is None and self.backing_type == FileBackingType.SHADOW:
            ContentNotLoadedError = Exception
            # print(f"File not opened: {self.filename}")
            # return f"open file ({self.local_path or self.filename}) first"
            raise ContentNotLoadedError(f"File not opened: {self.filename}")
        return self._content or ""

    @content.setter
    def content(self, value: str):
        self._content = value
        if self.backing_type == FileBackingType.SHADOW:
            self.backing_type = FileBackingType.MODIFIED
            self.is_dirty = True

    @property
    def is_loaded(self) -> bool:
        return self._content is not None

    """Represents a file in the Virtual File System"""

    # V2 additions
    is_executable: bool = False
    lsp_enabled: bool = False
    diagnostics: list[dict] = field(default_factory=list)  # LSP diagnostics cache

    def __post_init__(self):
        """Initialize file type info"""
        if self.file_type is None:
            self.file_type = get_file_type(self.filename)
            self.is_executable = self.file_type.is_executable
            self.lsp_enabled = self.file_type.lsp_server is not None


@dataclass
class VFSDelta:
    """One tracked VFS mutation with full content for revert."""
    index: int                              # monotonic per VFS instance
    step_id: int                            # set by obs via vfs._current_step_id
    timestamp: float
    path: str
    action: str                             # create|write|edit|delete|append|mv|mkdir|rmdir
    before_content: str | None = None       # None for create/mkdir
    after_content: str | None = None        # None for delete/rmdir
    old_path: str | None = None             # mv only: source path before move
    edit_range: tuple[int, int] | None = None  # edit only: (line_start, line_end)
    before_meta: dict | None = None         # {size_bytes, line_count, backing_type}

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "path": self.path,
            "action": self.action,
            "before_content": self.before_content,
            "after_content": self.after_content,
            "old_path": self.old_path,
            "edit_range": list(self.edit_range) if self.edit_range else None,
            "before_meta": self.before_meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VFSDelta":
        er = d.get("edit_range")
        return cls(
            index=d.get("index", 0),
            step_id=d.get("step_id", 0),
            timestamp=d.get("timestamp", 0.0),
            path=d.get("path", ""),
            action=d.get("action", ""),
            before_content=d.get("before_content"),
            after_content=d.get("after_content"),
            old_path=d.get("old_path"),
            edit_range=tuple(er) if er else None,
            before_meta=d.get("before_meta"),
        )
# =============================================================================
# VIRTUAL FILE SYSTEM V2
# =============================================================================


def unescape_string(text: str) -> str:
    """Delegiert an _decode_content — einzige Quelle der Wahrheit."""
    from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import (
        _decode_content,
        _strip_quotes,
    )

    return _decode_content(_strip_quotes(text))


class VirtualFileSystemV2:
    """
    Virtual File System V2 with hierarchical directories and LSP integration.

    Features:
    - Hierarchical directory structure
    - open/closed states (only open files show in context)
    - Windowing (show only specific line ranges)
    - System files (read-only, auto-updated)
    - File type detection with LSP support
    - Auto-summary on close
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        max_window_lines: int = 3000,
        summarizer: Callable[[str], str] | None = None,
        lsp_manager: "LSPManager | None" = None,
    ):
        self._context_cache = None
        self.session_id = session_id
        self.agent_name = agent_name
        self.max_window_lines = max_window_lines
        self._summarizer = summarizer
        self._lsp_manager = lsp_manager

        # Storage: path -> VFSFile or VFSDirectory
        self.files: dict[str, VFSFile] = {}
        self.directories: dict[str, VFSDirectory] = {}

        self.mounts: dict[str, ShadowMount] = {}
        self._shadow_index: dict[str, str] = {}

        self._dirty = True

        # ── Delta tracking ─────────────────────────────────────────
        self._change_log: list[VFSDelta] = []
        self._delta_counter: int = 0
        self._current_step_id: int = 0
        self._suppress_deltas: bool = False
        self.on_change: Callable[[VFSDelta], None] | None = None

        # Initialize root and system files
        self._init_root()
        self._init_system_files()

    # =========================================================================
    # DELTA TRACKING + REVERT
    # =========================================================================

    def _safe_get_content(self, path: str) -> str | None:
        """Get file content without raising. Returns None if unavailable."""
        path = self._normalize_path(path)
        f = self.files.get(path)
        if f is None:
            return None
        if isinstance(f, VFSFile):
            return f._content  # None if shadow not loaded — that's fine
        return getattr(f, "content", None)

    def _safe_get_meta(self, path: str) -> dict | None:
        """Capture file metadata before mutation."""
        path = self._normalize_path(path)
        f = self.files.get(path)
        if f is None:
            return None
        return {
            "size_bytes": getattr(f, "size_bytes", 0),
            "line_count": getattr(f, "line_count", 0),
            "backing_type": f.backing_type.name if isinstance(f, VFSFile) else "MEMORY",
            "is_dirty": getattr(f, "is_dirty", False),
        }

    def _emit_delta(
        self,
        path: str,
        action: str,
        before_content: str | None = None,
        after_content: str | None = None,
        old_path: str | None = None,
        edit_range: tuple[int, int] | None = None,
        before_meta: dict | None = None,
    ) -> VFSDelta | None:
        """Record a VFS mutation. Skipped when _suppress_deltas is True."""
        if self._suppress_deltas:
            return None
        import time as _time
        delta = VFSDelta(
            index=self._delta_counter,
            step_id=self._current_step_id,
            timestamp=_time.time(),
            path=path,
            action=action,
            before_content=before_content,
            after_content=after_content,
            old_path=old_path,
            edit_range=edit_range,
            before_meta=before_meta,
        )
        self._delta_counter += 1
        self._change_log.append(delta)
        if self.on_change is not None:
            try:
                self.on_change(delta)
            except Exception:
                pass
        return delta

    def _apply_revert(self, delta: VFSDelta) -> dict:
        """Apply the inverse of a single delta. Auto-syncs to disk."""
        self._suppress_deltas = True
        try:
            if delta.action == "create":
                if self._is_file(delta.path):
                    return self.delete(delta.path)
                return {"success": True, "message": f"Already gone: {delta.path}"}

            elif delta.action in ("write", "edit", "append"):
                if delta.before_content is None:
                    # File didn't exist before this mutation
                    if self._is_file(delta.path):
                        return self.delete(delta.path)
                    return {"success": True, "message": f"Already gone: {delta.path}"}
                return self.write(delta.path, delta.before_content)

            elif delta.action == "delete":
                if delta.before_content is not None:
                    result = self.create(delta.path, delta.before_content)
                    if not result.get("success") and "already exists" in result.get("error", "").lower():
                        return self.write(delta.path, delta.before_content)
                    return result
                return {"success": True, "message": "No content to restore"}

            elif delta.action == "mv":
                if delta.old_path and self._path_exists(delta.path):
                    return self.mv(delta.path, delta.old_path)
                return {"success": False, "error": f"Cannot revert mv: {delta.path} → {delta.old_path}"}

            elif delta.action == "mkdir":
                if self._is_directory(delta.path):
                    return self.rmdir(delta.path)
                return {"success": True, "message": f"Already gone: {delta.path}"}

            elif delta.action == "rmdir":
                if not self._is_directory(delta.path):
                    return self.mkdir(delta.path, parents=True)
                return {"success": True, "message": f"Already exists: {delta.path}"}

            return {"success": False, "error": f"Unknown action: {delta.action}"}
        finally:
            self._suppress_deltas = False

    def revert_delta(self, delta_index: int) -> dict:
        """Revert a single delta by its index."""
        for d in self._change_log:
            if d.index == delta_index:
                return self._apply_revert(d)
        return {"success": False, "error": f"Delta {delta_index} not found in change_log"}

    def revert_step(self, step_id: int) -> dict:
        """Revert ALL deltas from a given step, in reverse order."""
        deltas = [d for d in self._change_log if d.step_id == step_id]
        if not deltas:
            return {"success": False, "error": f"No deltas for step {step_id}"}
        results = []
        for d in reversed(deltas):
            r = self._apply_revert(d)
            results.append({"index": d.index, "path": d.path, "action": d.action, **r})
        ok = all(r.get("success", False) for r in results)
        return {"success": ok, "reverted": len(results), "details": results}

    def revert_file(self, path: str, to_step: int | None = None) -> dict:
        """Revert all deltas for a specific file path, optionally only since to_step."""
        path = self._normalize_path(path)
        deltas = [
            d for d in self._change_log
            if d.path == path and (to_step is None or d.step_id >= to_step)
        ]
        if not deltas:
            return {"success": False, "error": f"No deltas for {path}"}
        results = []
        for d in reversed(deltas):
            r = self._apply_revert(d)
            results.append({"index": d.index, "action": d.action, **r})
        ok = all(r.get("success", False) for r in results)
        return {"success": ok, "reverted": len(results), "details": results}

    def revert_dir(self, prefix: str, to_step: int | None = None) -> dict:
        """Revert all deltas under a directory prefix."""
        prefix = self._normalize_path(prefix)
        if not prefix.endswith("/"):
            prefix += "/"
        deltas = [
            d for d in self._change_log
            if (d.path.startswith(prefix) or d.path == prefix.rstrip("/"))
               and (to_step is None or d.step_id >= to_step)
        ]
        if not deltas:
            return {"success": False, "error": f"No deltas under {prefix}"}
        results = []
        for d in reversed(deltas):
            r = self._apply_revert(d)
            results.append({"index": d.index, "path": d.path, "action": d.action, **r})
        ok = all(r.get("success", False) for r in results)
        return {"success": ok, "reverted": len(results), "details": results}

    def revert_all(self, to_step: int | None = None) -> dict:
        """Revert all deltas, optionally only those since to_step."""
        deltas = [
            d for d in self._change_log
            if to_step is None or d.step_id >= to_step
        ]
        if not deltas:
            return {"success": False, "error": "No deltas to revert"}
        results = []
        for d in reversed(deltas):
            r = self._apply_revert(d)
            results.append({"index": d.index, "path": d.path, "action": d.action, **r})
        ok = all(r.get("success", False) for r in results)
        return {"success": ok, "reverted": len(results), "details": results}

    def get_change_log(
        self, path: str | None = None, step_id: int | None = None
    ) -> list[dict]:
        """Query change log. Filters optional."""
        out = self._change_log
        if path is not None:
            path = self._normalize_path(path)
            out = [d for d in out if d.path == path]
        if step_id is not None:
            out = [d for d in out if d.step_id == step_id]
        return [d.to_dict() for d in out]

    def inject_deltas(self, deltas: list[dict]) -> int:
        """Inject deltas from obs persistence (run_*.json / live_*.jsonl).
        Used by obs.load_and_revert_from_run(). Returns count injected."""
        count = 0
        for dd in deltas:
            delta = VFSDelta.from_dict(dd)
            delta.index = self._delta_counter
            self._delta_counter += 1
            self._change_log.append(delta)
            count += 1
        return count

    def clear_change_log(self) -> None:
        """Clear all recorded deltas."""
        self._change_log.clear()

    def _init_root(self):
        """Initialize root directory"""
        self.directories["/"] = VFSDirectory(name="/", readonly=True)

    def _build_vfs_guide(self) -> str:
        """Build the VFS usage guide that is injected as /vfs_guide.md."""
        return (
            r"""# VFS — Schnellreferenz

## Die zwei Kern-Tools

| Tool | Zweck |
|------|-------|
| `vfs_shell(reason, command)` | Alle Datei-Operationen (lesen, schreiben, suchen, navigieren) |
| `vfs_view(path, ...)` | Kontext-Fenster steuern — was du im nächsten Prompt **siehst** |

---

## Kontext-Fenster — Konzept

Offene Dateien erscheinen in **jedem folgenden Prompt** (= Token-Kosten!).
Geschlossene Dateien sind unsichtbar — nur Metadaten bleiben erhalten.

**Regel**: Öffne immer nur den Bereich, der für deine aktuelle Aufgabe direkt relevant ist.

---

## Fokussierter Recherche-Workflow (x und y finden)

```
# 1. x suchen
vfs_shell("find specif section to work focussed and persist.", "grep -rn 'ClassX' /src")
# → /src/models.py:42:class ClassX:

# 2. Auf x einzoomen  →  models.py erscheint ab jetzt im Kontext
vfs_view("/src/models.py", scroll_to="ClassX", context_lines=60)

# 3. y suchen
vfs_shell("initial find locations and information abut method_y","grep -rn 'method_y' /src")
# → /src/services.py:88:    def method_y(self):

# 4. y zum Kontext hinzufügen  →  jetzt sind BEIDE Abschnitte sichtbar
vfs_view("/src/services.py", scroll_to="method_y", context_lines=40)

# 5. Antwort geben — du siehst genau x und y, nichts Überflüssiges

# 6. Aufräumen  →  alles schließen, neu starten
vfs_view("/src/neue_datei.py", scroll_to="...", close_others=True)
# ODER einzeln:
vfs_shell("I done wit working on models and do not need any references to this files or ist information's","close /src/models.py")
vfs_shell("focusing on the next section this file is project relayed i'm now working on an web research as requested by the user","close /src/services.py")
```

---

## vfs_shell Referenz

```
# Navigation
ls [-la] [-R] [path]         tree [path] [-L depth]      pwd

# Lesen
cat <path>                   head -n N <path>            tail -n N <path>
wc -l <path>                 stat <path>

# Suchen
find [path] -name "*.py"     find [path] -type f -name pattern
grep -rn "pattern" /src      grep -in "pattern" /file.py

# Schreiben (klein)
touch <path>
echo "content" > <path>      echo "line" >> <path>

# Heredoc — für Content mit beliebigen Quotes/Sonderzeichen (bevorzugt)
write <path> <<'_VFS_END_a799'
content with '''triple quotes''' and "mixed" quotes
_VFS_END_a799
# Funktioniert auch mit: write_mini, write_chunk, edit, echo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WICHTIG: SCHREIB-LIMITS & CHUNK-PROTOKOLL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ein einzelner Tool-Call darf max ~40 Zeilen Content enthalten.
Grund: JSON-Tool-Calls können bei Längenbeschränkung abbrechen.
Ein abgebrochener Call = keine Wirkung. Kein partial-write.

REGEL:
  < 40 Zeilen  →  write <path> "..."   (ein Call)
  ≥ 40 Zeilen  →  write_chunk          (ein Call pro Block)

CHUNK-PROTOKOLL:
  write_chunk <path> 0 <N> "<block_0_content>"   ← erzeugt/überschreibt Datei
  write_chunk <path> 1 <N> "<block_1_content>"   ← hängt an
  ...
  write_chunk <path> N-1 <N> "<block_N-1>"       ← finalisiert

Nach Abbruch / Wiederaufnahme:
  write_chunk_status <path>   →  zeigt welche Blöcke fehlen
  Dann nur die fehlenden Blöcke erneut senden.

Content-Encoding:
  "line1\nline2"    ← echter Newline im JSON-String (STANDARD)
  NIEMALS: \\n      ← erzeugt LITERAL Backslash+n im File, KEINEN Zeilenumbruch!
  NIEMALS: \\\\n    ← erzeugt ebenfalls Backslash+n im File!
  NIEMALS: \\|      ← das ürde nun | als literal anshen was Fasch ist!
  NIEMALS: \|      ← das ürde nun | als literal anshen was Fasch ist!

WICHTIG: Der JSON-Parser wandelt \n in echte Newlines um BEVOR
der Command das VFS erreicht. Das System decodiert NICHT nochmal.
Nur echte Newlines (JSON-Standard) funktionieren als Zeilenumbruch.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Das LLM muss echte Newlines via JSON senden

# Zeilen ersetzen (präzises Editieren)
edit <path> <start> <end> "neuer inhalt"
⚠️ **edit schreibt Content 1:1 — keine Escape-Verarbeitung.**
Mehrzeilige Replacements: echte Newlines im JSON-String verwenden.

# Verzeichnisse
mkdir -p /src/components      rm -rf /old_dir/

# Dateien
rm /file.py                   mv /old.py /new.py          cp /src.py /dst.py

# Kontext
close <path>                  (entfernt Datei aus dem Kontext-Fenster)

# Ausführen
exec <path> [args...]

## grep — Hinweise für zuverlässige Suche

**Backend:** grep nutzt ripgrep (`rg`) für alle disk-backed Bereiche (Mounts, `/global/`, `/shared/`).
Das eliminiert Memory-Spikes bei großen Projekten. Reine In-Memory-Files werden per Python-Regex durchsucht.
Die Syntax bleibt identisch — das Backend ist transparent.

**Exakte Patterns:** grep ist Regex-basiert. Spaces, Sonderzeichen und Groß/Kleinschreibung
zählen. `grep "x=y+z"` matcht NICHT `x = y + z`.

| Wenn kein Match | Lösung |
|-----------------|--------|
| Spaces im Code | Kürzeres Pattern: `grep "lastPlatX"` statt `grep "lastPlatX=px+pw"` |
| Groß/Klein | `-i` Flag: `grep -in "debug" /src` |
| Sonderzeichen `+.*[]()` | Escapen: `grep "x\+y"` oder kürzeres Pattern ohne Sonderzeichen |
| Alternation (`\|` oder `\\|`) | **NIEMALS** `\\|` oder `\|` — nutze `|` direkt: `grep -rn 'class|def |async def ' /path` |

**⚠️ ALTERNATION — HÄUFIGSTER FEHLER:**
VFS-grep nutzt ERE/PCRE-Syntax. `|` ist direkt Alternation, KEIN Escaping nötig.
- ✅ `grep -rn 'class|def |async def ' /src`
- ✅ `grep -rn 'minio|redis|host_network' /src`
- ❌ `grep -n 'class\|def \|async def ' /file`  ← matcht NICHTS, `\|` ist Literal !!
- ❌ `grep -n 'class\\|def ' /file`  ← matcht NICHTS, selbes Problem eifac nein nicht machen tu so las ob duch nach dem str "def |class" suchst genau so und nicht ander s !!!!!!!!!!


**grep in Pipes vs. direkt:**
- `grep -rn "pattern" /src` → durchsucht Dateien via ripgrep, Output: `datei:zeile:match`
- `cat /f.py | grep "pattern"` → durchsucht Text-Stream (Python), Output: `match` (kein Dateiname)
- `-l` (nur Dateinamen) funktioniert nur bei direktem grep, NICHT in Pipes

**Kein Match ≠ Tool-Fehler.** Wenn grep rc=1 zurückgibt, existiert das Pattern nicht exakt so im Code.
Prüfe Whitespace und verwende ein kürzeres, eindeutiges Pattern.
```
---


## Chunk-Protokoll (Beispiel für 120 Zeilen Datei)

Wenn eine Datei zu groß für einen Call ist, teile sie in Blöcke à ~40 Zeilen:

1. `vfs_shell("init big file", "write_chunk /src/big.py 0 3 '...zeile 1-40...'")`
2. `vfs_shell("add chunk 1", "write_chunk /src/big.py 1 3 '...zeile 41-80...'")`
3. `vfs_shell("finalize", "write_chunk /src/big.py 2 3 '...zeile 81-120...'")`

pythonBei Verbindungsabbruch: `write_chunk_status /src/big.py` prüft, welche Indizes fehlen.

## Heredoc-Syntax (empfohlen für komplexen Content)

Wenn Content Quotes, Triple-Quotes oder Sonderzeichen enthält → Heredoc:
write /src/app.py <<'_VFS_END_a799'
class App:
TEMPLATE = """
            + r'''"""<html>{content}</html>"""
def run(self):
print('it's working')
_VFS_END_a799

- Bevorzugter Delimiter: `_VFS_END_a799`
- Delimiter darf NICHT im Content vorkommen
- Content wird 1:1 übernommen, keine Escape-Verarbeitung
- Ohne Heredoc wird normal geparst; bei Fehler → Hint in der Fehlermeldung

---
```

## Mehrere Befehle in einem Aufruf (Batch-Syntax)

`vfs_shell` versteht Shell-Operatoren um mehrere Befehle in **einem einzigen Aufruf** zu kombinieren.

| Operator | Semantik | Beispiel |
|----------|----------|---------|
| `;` | Immer ausführen (Sequenz) | `mkdir /out; touch /out/f.txt` |
| `&&` | Nur wenn vorheriger **erfolgreich** | `mkdir /out && write /out/f.py "x=1"` |
| `||` | Nur wenn vorheriger **fehlgeschlagen** | `cat /cfg.py || write /cfg.py "DEBUG=True"` |
| `|` | Pipe — stdout links wird stdin rechts | `cat /f.py | grep def | wc -l` |

**Pipe-fähige Rechts-Befehle:** `grep [-invC]`, `wc [-l|-w|-c]`, `head [-n N]`, `tail [-n N]`, `sort [-r]`, `uniq`

```
# Typische Workflows

# Verzeichnis anlegen und Datei schreiben (nur wenn mkdir klappt)
"mkdir -p /src/utils && write /src/utils/helper.py 'def f(): pass'"

# Fallback: Datei lesen, falls nicht vorhanden anlegen
"cat /config.py || write /config.py 'DEBUG = True'"

# Drei Schritte in einem Aufruf
"mkdir /out; write /out/app.py 'x=1'; cat /out/app.py"

# Pipeline: alle Klassen zählen
"grep -rn 'class ' /src | grep -v '#' | wc -l"

# Pipeline: nur Dateinamen mit Matches
"grep -rl 'TODO' /src | sort"

# Sync nach manuellen Bulk-Schreiboperationen
"sync"
```

# Alternation — ERE-Syntax direkt (ohne \|):
grep -rn 'minio|redis|external|host_network' /tb/.../__init__.py

# Oder einzeln:
grep -rn 'minio' /path && grep -rn 'redis' /path

# Statt find -exec:
grep -rn 'create_container|network|docker' /tb/toolboxv2/mods/ContainerManager


> ⚠️ **Wichtig — Zeilenumbrüche (`\n`) sind kein Separator.**
> Echter Newline in Datei-Inhalten bleibt erhalten:
> `write /f.py "class Foo:\n    pass"` → eine Datei, kein Batch.
> Für Batches immer `;` oder `&&` verwenden.
```

---

## vfs_view Parameter

```
vfs_view(
    path         : str,         # Dateipfad (Pflicht)
    line_start   : int = 1,     # Startzeile — wird ignoriert wenn scroll_to gesetzt
    line_end     : int = -1,    # Endzeile   — wird ignoriert wenn scroll_to gesetzt
    scroll_to    : str = None,  # Pattern → erste Übereinstimmung finden & zentrieren
    context_lines: int = 40,    # Zeilen um den Treffer anzeigen
    close_others : bool = False # Alle anderen offenen Dateien zuerst schließen
)
```

---

## Mount-Punkte

| Pfad | Beschreibung |
|------|-------------|
| `/global/` | Geteilt zwischen allen Sessions (persistent auf Disk) |
| `/shared/{id}/` | Cross-Session/Agent Shares  →  `vfs_share_*` Tools |
| `/project/` | Typischer Einhängepunkt für lokale Projektordner  →  `vfs_mount` |


**⚠️ ALTERNATION — HÄUFIGSTER FEHLER:**
VFS-grep nutzt ERE/PCRE-Syntax. `|` ist direkt Alternation, KEIN Escaping nötig.
- ✅ `grep -rn 'class|def |async def ' /src`
- ✅ `grep -rn 'minio|redis|host_network' /src`
- ❌ `grep -n 'class\|def \|async def ' /file`  ← matcht NICHTS, `\|` ist Literal
- ❌ `grep -n 'class\\|def ' /file`  ← matcht NICHTS, selbes Problem
'''
        )

    def _init_system_files(self):
        """Initialize read-only system files"""
        self.files["/system_context.md"] = VFSFile(
            filename="system_context.md",
            _content=self._build_system_context(),
            state="open",
            readonly=True,
            show_full=True,
        )

        self.files["/vfs_guide.md"] = VFSFile(
            filename="vfs_guide.md",
            _content=self._build_vfs_guide(),
            state="open",
            readonly=True,
            show_full=True,
        )

    def _build_system_context(self) -> str:
        """Build system context content"""
        now = datetime.now()
        return f"""# System Context
Current Time: {now.strftime("%Y-%m-%d %H:%M:%S")}
Agent: {self.agent_name}
Session: {self.session_id}
"""

    def update_system_context(self):
        """Refresh system context"""
        if "/system_context.md" in self.files:
            self.files["/system_context.md"].content = self._build_system_context()
            self.files["/system_context.md"].updated_at = datetime.now().isoformat()
            self._dirty = True

    def set_rules_file(self, content: str):
        """Set the active_rules.md file content (from RuleSet)"""
        path = "/active_rules.md"
        if path not in self.files:
            self.files[path] = VFSFile(
                filename="active_rules.md", _content=content, state="open", readonly=True, show_full=True
            )
        else:
            self.files[path].content = content
            self.files[path].updated_at = datetime.now().isoformat()
            self.files[path].show_full = True
        self._dirty = True

    def set_memory_index_file(self, content: str, agent_name="default"):
        """Set the memory_index.md file content (from RuleSet)"""
        path = "/memory_index.md"
        global_path = "/global/default/memory_index.md"
        if path not in self.files:
            self.write(global_path, content)
            self.open(global_path)
            if global_path in self.files:
                self.files[global_path].show_full = True
        else:
            self.files[path].content = content
            self.files[path].updated_at = datetime.now().isoformat()
            self.files[path].show_full = True
        self._dirty = True

    # =========================================================================
    # SYSTEM FILE MANAGEMENT (Permanent Read-Only Local Files)
    # =========================================================================

    def add_system_file(
        self,
        local_path: str,
        vfs_path: str | None = None,
        auto_refresh: bool = False,
        open_state: bool = False,
    ) -> dict:
        """
        Füge eine lokale Datei als permanente Read-Only System-Datei zum VFS hinzu.

        Die Datei wird im VFS als read-only markiert und bleibt über die gesamte
        Session hinweg verfügbar. Mit auto_refresh wird der Inhalt bei jedem Zugriff
        neu vom lokalen Dateisystem geladen.

        Args:
            local_path: Pfad zur lokalen Datei
            vfs_path: VFS-Pfad (default: /<filename>)
            auto_refresh: Bei True wird der Inhalt bei jedem Zugriff neu geladen

        Returns:
            Result dict mit success status
        """
        try:
            local_path = os.path.abspath(os.path.expanduser(local_path))

            if not os.path.exists(local_path):
                return {"success": False, "error": f"File not found: {local_path}"}

            if not os.path.isfile(local_path):
                return {"success": False, "error": f"Not a file: {local_path}"}

            filename = os.path.basename(local_path)

            if vfs_path is None:
                vfs_path = f"/{filename}"
            else:
                vfs_path = self._normalize_path(vfs_path)

            # Check if already exists
            if vfs_path in self.files:
                return {"success": False, "error": f"Path already exists: {vfs_path}"}

            # Ensure parent directory exists
            parent = self._get_parent_path(vfs_path)
            if parent != "/" and not self._is_directory(parent):
                self.mkdir(parent, parents=True)

            # Read file content
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            file_size = os.path.getsize(local_path)
            file_type = get_file_type(filename)

            # Create system file entry
            self.files[vfs_path] = VFSFile(
                filename=filename,
                _content=content,
                state="open" if open_state else "closed",
                readonly=True,
                local_path=local_path,  # Store reference for refresh
                size_bytes=file_size,
                line_count=len(content.splitlines()),
                file_type=file_type,
            )

            # Store auto-refresh preference
            if auto_refresh:
                self.files[vfs_path].__dict__["_auto_refresh"] = True

            self._dirty = True

            return {
                "success": True,
                "vfs_path": vfs_path,
                "local_path": local_path,
                "size_bytes": file_size,
                "lines": len(content.splitlines()),
                "auto_refresh": auto_refresh,
                "message": f"Added system file: {vfs_path} ← {local_path}",
            }

        except Exception as e:
            return {"success": False, "error": f"Error adding system file: {e}"}

    def remove_system_file(self, vfs_path: str) -> dict:
        """
        Entferne eine System-Datei aus dem VFS.

        Args:
            vfs_path: VFS-Pfad der zu entfernenden Datei

        Returns:
            Result dict mit success status
        """
        vfs_path = self._normalize_path(vfs_path)

        if vfs_path not in self.files:
            return {"success": False, "error": f"File not found: {vfs_path}"}

        f = self.files[vfs_path]

        # Allow removal of readonly system files (but protect core system files)
        protected_files = ["/system_context.md", "/active_rules.md"]
        if vfs_path in protected_files:
            return {
                "success": False,
                "error": f"Cannot remove protected system file: {vfs_path}",
            }

        # Store info for response
        local_path = getattr(f, "local_path", None)
        filename = f.filename

        del self.files[vfs_path]
        self._dirty = True

        msg = f"Removed system file: {vfs_path}"
        if local_path:
            msg += f" (was: {local_path})"

        return {"success": True, "message": msg, "filename": filename}

    def refresh_system_file(self, vfs_path: str) -> dict:
        """
        Lade den Inhalt einer System-Datei neu vom lokalen Dateisystem.

        Args:
            vfs_path: VFS-Pfad der zu aktualisierenden Datei

        Returns:
            Result dict mit success status
        """
        vfs_path = self._normalize_path(vfs_path)

        if vfs_path not in self.files:
            return {"success": False, "error": f"File not found: {vfs_path}"}

        f = self.files[vfs_path]

        local_path = getattr(f, "local_path", None)
        if not local_path:
            return {"success": False, "error": f"File has no local backing: {vfs_path}"}

        if not os.path.exists(local_path):
            return {"success": False, "error": f"Local file not found: {local_path}"}

        try:
            with open(local_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()

            f._content = content
            f.size_bytes = len(content.encode("utf-8"))
            f.line_count = len(content.splitlines())
            f.updated_at = datetime.now().isoformat()

            self._dirty = True

            return {
                "success": True,
                "message": f"Refreshed: {vfs_path}",
                "size_bytes": f.size_bytes,
                "lines": f.line_count,
            }

        except Exception as e:
            return {"success": False, "error": f"Error reading file: {e}"}

    def list_system_files(self) -> dict:
        """
        Liste alle System-Dateien (readonly Dateien mit lokalem Backing) auf.

        Returns:
            Dict mit Liste der System-Dateien
        """
        system_files = []
        for path, f in self.files.items():
            if f.readonly:
                info = {
                    "path": path,
                    "filename": f.filename,
                    "local_path": getattr(f, "local_path", None),
                    "size": f.size,
                    "lines": f.line_count if f.line_count >= 0 else "unknown",
                    "auto_refresh": getattr(f, "_auto_refresh", False),
                    "file_type": f.file_type.description if f.file_type else "Unknown",
                }
                system_files.append(info)

        return {"success": True, "system_files": system_files, "count": len(system_files)}

    def wipe(self):
        """
        HARD RESET: Completely erases the file system state.
        Used for benchmarking to ensure 0% state leakage between runs.
        """
        # 1. Clear Memory Files
        self.files.clear()

        # 2. Clear Directory Structure
        self.directories.clear()

        # 3. Unmount all shadow mounts (except global if needed)
        # We iterate a copy of keys to avoid modification errors
        for mount_point in list(self.mounts.keys()):
            if mount_point != "/global":  # Preserve global tools
                self.unmount(mount_point, save_changes=False)
        # 4.clears change log
        self._change_log.clear()
        self._delta_counter = 0

        # 5. Re-initialize Root & System Files
        self._init_root()
        self._init_system_files()

        if hasattr(self, "lsp_manager") and self.lsp_manager:
            # We don't stop the server (too slow), but we clear diagnostics
            pass

    # =========================================================================
    # PATH UTILITIES
    # =========================================================================

    def _normalize_path(self, path: str) -> str:
        """Normalize path to absolute POSIX-style path"""
        if not path:
            return "/"

        # Ensure path starts with /
        if not path.startswith("/"):
            path = "/" + path

        # Normalize using PurePosixPath
        normalized = str(PurePosixPath(path))

        # Remove trailing slash except for root
        if normalized != "/" and normalized.endswith("/"):
            normalized = normalized[:-1]

        return normalized

    def _get_parent_path(self, path: str) -> str:
        """Get parent directory path"""
        path = self._normalize_path(path)
        if path == "/":
            return "/"
        return str(PurePosixPath(path).parent)

    def _get_basename(self, path: str) -> str:
        """Get filename/dirname from path"""
        path = self._normalize_path(path)
        return PurePosixPath(path).name or "/"

    def _path_exists(self, path: str) -> bool:
        """Check if path exists (file or directory)"""
        path = self._normalize_path(path)
        return path in self.files or path in self.directories

    def _is_file(self, path: str) -> bool:
        """Check if path is a file"""
        return self._normalize_path(path) in self.files

    def _is_directory(self, path: str) -> bool:
        """Check if path is a directory"""
        return self._normalize_path(path) in self.directories

    def _ensure_parent_exists(self, path: str) -> dict | None:
        """Ensure parent directory exists, return error dict if not"""
        parent = self._get_parent_path(path)
        if parent != "/" and not self._is_directory(parent):
            return {
                "success": False,
                "error": f"Parent directory does not exist: {parent}",
            }
        return None

    # =========================================================================
    # DIRECTORY OPERATIONS
    # =========================================================================

    def mkdir(self, path: str, parents: bool = False) -> dict:
        """
        Create a directory.

        Args:
            path: Directory path to create
            parents: If True, create parent directories as needed

        Returns:
            Result dict with success status
        """
        path = self._normalize_path(path)

        if path == "/":
            return {"success": False, "error": "Cannot create root directory"}

        if self._path_exists(path):
            return {"success": False, "error": f"Path already exists: {path}"}

        parent = self._get_parent_path(path)

        if not self._is_directory(parent):
            if parents:
                # Recursively create parents
                result = self.mkdir(parent, parents=True)
                if not result["success"]:
                    return result
            else:
                return {
                    "success": False,
                    "error": f"Parent directory does not exist: {parent}",
                }

        # Check if under a mount
        mount = self._get_mount_for_path(path)
        local_path = None

        if mount:
            rel_path = path[len(mount.vfs_path) :].lstrip("/")
            local_path = os.path.join(mount.local_path, rel_path.replace("/", os.sep))

            try:
                os.makedirs(local_path, exist_ok=True if parents else False)
            except OSError as e:
                return {"success": False, "error": f"Cannot create local directory: {e}"}

        # Create directory
        self.directories[path] = VFSDirectory(
            name=self._get_basename(path), readonly=mount.readonly if mount else False
        )
        self._dirty = True
        self._emit_delta(path, "mkdir")
        return {"success": True, "message": f"Created directory: {path}"}

    def rmdir(self, path: str, force: bool = False) -> dict:
        """
        Remove a directory.

        Args:
            path: Directory path to remove
            force: If True, remove non-empty directories recursively

        Returns:
            Result dict with success status
        """
        path = self._normalize_path(path)

        if path == "/":
            return {"success": False, "error": "Cannot remove root directory"}

        if not self._is_directory(path):
            return {"success": False, "error": f"Not a directory: {path}"}

        if self.directories[path].readonly:
            return {
                "success": False,
                "error": f"Cannot remove readonly directory: {path}",
            }

        # Check if directory is empty
        contents = self._list_directory_contents(path)

        if contents and not force:
            return {
                "success": False,
                "error": f"Directory not empty: {path} (use force=True to remove)",
            }

        if force and contents:
            # Recursively remove contents
            for item in contents:
                item_path =os.path.join(path, item['name'])
                if item["type"] == "directory":
                    result = self.rmdir(item_path, force=True)
                else:
                    result = self.delete(item_path)
                if not result["success"]:
                    return result
        if not contents or force:
            # Sicherheitsnetz: alle verbliebenen Sub-Directories + sich selbst entfernen
            dirs_to_remove = [
                p
                for p in list(self.directories.keys())
                if p.startswith(path + "/") or p == path
            ]
            for d in dirs_to_remove:
                del self.directories[d]
        self._dirty = True
        self._emit_delta(path, "rmdir")
        return {"success": True, "message": f"Removed directory: {path}"}

    def _list_directory_contents(self, path: str) -> list[dict]:
        """List contents of a directory"""
        path = self._normalize_path(path)
        contents = []

        # Find all direct children
        prefix = path if path == "/" else path + "/"

        # Check directories
        for dir_path in self.directories:
            if dir_path == path:
                continue
            if dir_path.startswith(prefix):
                # Check if it's a direct child
                relative = dir_path[len(prefix) :]
                if "/" not in relative:
                    contents.append(
                        {"name": relative, "type": "directory", "path": dir_path}
                    )

        # Check files
        for file_path in self.files:
            if file_path.startswith(prefix):
                relative = file_path[len(prefix) :]
                if "/" not in relative:
                    f = self.files[file_path]
                    contents.append(
                        {
                            "name": relative,
                            "type": "file",
                            "path": file_path,
                            "size": f.size,
                            "state": f.state,
                            "file_type": f.file_type.description
                            if f.file_type
                            else "Unknown",
                        }
                    )

        return sorted(contents, key=lambda x: (x["type"] != "directory", x["name"]))

    def ls(
        self, path: str = "/", recursive: bool = False, show_hidden: bool = False
    ) -> dict:
        """
        List directory contents.

        Args:
            path: Directory path to list
            recursive: If True, list recursively
            show_hidden: If True, show hidden files (starting with .)

        Returns:
            Result dict with directory contents
        """
        path = self._normalize_path(path)

        if not self._is_directory(path):
            if self._is_file(path):
                return {"success": False, "error": f"Not a directory: {path}"}
            return {"success": False, "error": f"Path not found: {path}"}

        def list_recursive(dir_path: str, depth: int = 0) -> list[dict]:
            items = []
            contents = self._list_directory_contents(dir_path)

            for item in contents:
                if not show_hidden and item["name"].startswith("."):
                    continue

                item["depth"] = depth
                items.append(item)

                if recursive and item["type"] == "directory":
                    items.extend(list_recursive(item["path"], depth + 1))

            return items

        contents = (
            list_recursive(path) if recursive else self._list_directory_contents(path)
        )

        if not show_hidden:
            contents = [c for c in contents if not c["name"].startswith(".")]

        return {
            "success": True,
            "path": path,
            "contents": contents,
            "total_items": len(contents),
        }

    def mv(self, source: str, destination: str) -> dict:
        """
        Move/rename a file or directory.

        Args:
            source: Source path
            destination: Destination path

        Returns:
            Result dict with success status
        """
        source = self._normalize_path(source)
        _before_content = self._safe_get_content(source)
        _before_meta = self._safe_get_meta(source)
        destination = self._normalize_path(destination)


        if not self._path_exists(source):
            return {"success": False, "error": f"Source not found: {source}"}

        if source == "/":
            return {"success": False, "error": "Cannot move root directory"}

        # Check if source is readonly
        if self._is_file(source) and self.files[source].readonly:
            return {"success": False, "error": f"Cannot move readonly file: {source}"}
        if self._is_directory(source) and self.directories[source].readonly:
            return {
                "success": False,
                "error": f"Cannot move readonly directory: {source}",
            }

        # If destination is existing directory, move into it
        if self._is_directory(destination):
            destination = os.path.join(destination, self._get_basename(source))

        if self._path_exists(destination):
            return {
                "success": False,
                "error": f"Destination already exists: {destination}",
            }

        # Ensure destination parent exists
        error = self._ensure_parent_exists(destination)
        if error:
            return error

        # Perform move
        if self._is_file(source):
            self.files[destination] = self.files[source]
            self.files[destination].filename = self._get_basename(destination)
            self.files[destination].updated_at = datetime.now().isoformat()
            # Update file type based on new name
            self.files[destination].file_type = get_file_type(
                self._get_basename(destination)
            )
            del self.files[source]
        else:
            # Move directory and all contents
            old_prefix = source if source == "/" else source + "/"
            new_prefix = destination if destination == "/" else destination + "/"

            # Collect paths to move
            dirs_to_move = [(source, destination)]
            files_to_move = []

            for dir_path in list(self.directories.keys()):
                if dir_path.startswith(old_prefix):
                    new_path = new_prefix + dir_path[len(old_prefix) :]
                    dirs_to_move.append((dir_path, new_path))

            for file_path in list(self.files.keys()):
                if file_path.startswith(old_prefix):
                    new_path = new_prefix + file_path[len(old_prefix) :]
                    files_to_move.append((file_path, new_path))

            # Move directories
            for old_path, new_path in dirs_to_move:
                self.directories[new_path] = self.directories[old_path]
                self.directories[new_path].name = self._get_basename(new_path)
                self.directories[new_path].updated_at = datetime.now().isoformat()
                del self.directories[old_path]

            # Move files
            for old_path, new_path in files_to_move:
                self.files[new_path] = self.files[old_path]
                self.files[new_path].filename = self._get_basename(new_path)
                self.files[new_path].updated_at = datetime.now().isoformat()
                del self.files[old_path]

        self._dirty = True
        self._emit_delta(destination, "mv", _before_content, _before_content,
                         old_path=source, before_meta=_before_meta)
        return {"success": True, "message": f"Moved {source} to {destination}"}

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    def create(self, path: str, content: str = "") -> dict:
        """
        Create a new file with mount-aware local file creation.

        Note: content is written 1:1 — no escape-unescaping. Agent-input from
        vfs_shell is already decoded via _decode_content before reaching here.
        """
        path = self._normalize_path(path)
        _before_content = self._safe_get_content(path)
        _before_meta = self._safe_get_meta(path)

        if self._path_exists(path):
            if self._is_file(path) and self.files[path].readonly:
                return {
                    "success": False,
                    "error": f"Cannot overwrite system file: {path}",
                }
            if self._is_directory(path):
                return {"success": False, "error": f"Path is a directory: {path}"}

        filename = self._get_basename(path)

        # Check mount first (needed for readonly-check before any side effects)
        mount = self._get_mount_for_path(path)

        # Readonly-Check — FIX: war komplett fehlend
        if mount and mount.readonly:
            return {"success": False, "error": f"Mount is read-only: {mount.vfs_path}"}

        # Ensure parent exists — FIX: auto-create mit parents=True
        # (vorher: _ensure_parent_exists ohne parents → Subdir-Erstellung schlug fehl)
        parent = self._get_parent_path(path)
        if parent != "/" and not self._is_directory(parent):
            mkdir_result = self.mkdir(parent, parents=True)
            if not mkdir_result["success"]:
                return mkdir_result

        local_path = None
        backing_type = FileBackingType.MEMORY

        if mount:
            rel_path = path[len(mount.vfs_path) :].lstrip("/")
            local_path = os.path.join(mount.local_path, rel_path.replace("/", os.sep))
            backing_type = FileBackingType.SHADOW

        self.files[path] = VFSFile(
            filename=filename,
            _content=content,
            state="closed",
            local_path=local_path,
            backing_type=backing_type,
        )

        # If mount: create local file + shadow_index eintragen
        if mount and local_path:
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as e:
                return {"success": False, "error": f"Cannot create local file: {e}"}

            # BUG #5 FIX: shadow_index NUR für mount-Dateien, NUR wenn local_path gesetzt
            self._shadow_index[path] = local_path

        self._dirty = True
        self._emit_delta(path, "create", _before_content, content, before_meta=_before_meta)
        return {
            "success": True,
            "message": f"Created '{path}' ({len(content)} chars)",
            "file_type": self.files[path].file_type.description,
        }

    def read(self, path: str, max_chars: int = 25000) -> dict:
        """
        Read file content with auto-load for shadow files and auto-refresh for system files.
        """
        path = self._normalize_path(path)

        if not self._is_file(path):
            if self._is_directory(path):
                return {"success": False, "error": f"Cannot read directory: {path}"}
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]
        if (
            isinstance(f, VFSFile)
            and f.local_path
            and not f.is_dirty
            and f.backing_type in (FileBackingType.SHADOW, FileBackingType.MODIFIED)
        ):
            try:
                disk_mtime = os.path.getmtime(f.local_path)
                if f.local_mtime is None or disk_mtime > f.local_mtime:
                    reload_result = self._load_shadow_content(path)
                    if reload_result.get("success"):
                        f.backing_type = FileBackingType.SHADOW
            except OSError:
                pass
        # EBENE 3: Für /global/-Pfade zuerst den Shared-Store prüfen.
        # Das ist der schnellste Pfad für Multi-Agent-Reads im selben Prozess —
        # keine Disk-IO, keine Poll-Wartezeit.
        # EBENE 3: Für Shared-Mount-Pfade zuerst den Shared-Store prüfen.
        # Gilt für /global/ UND für CoderAgent-Worktrees etc. die via
        # GlobalVFSManager.register_shared_mount() registriert wurden.
        if isinstance(f, VFSFile) and not f.is_dirty:
            shared_info = self._get_shared_store_info(path)
            if shared_info is not None:
                mount_key, relative, _ = shared_info
                try:
                    from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

                    gvfs = get_global_vfs()
                    if mount_key == "global":
                        shared = gvfs.get_shared(relative)
                    else:
                        shared = gvfs.shared_read(mount_key, relative)
                    if shared is not None:
                        f._content = shared["content"]
                        f.local_mtime = shared["mtime"]
                        f.size_bytes = len(shared["content"])
                        f.line_count = len(shared["content"].splitlines())
                        content = shared["content"]
                        if len(content) > max_chars * 2.1:
                            return {
                                "success": True,
                                "content": (
                                    content[:max_chars]
                                    + f"\n\n... [TRUNCATED: File is {len(content)} chars. "
                                    "Use 'view' for specific lines] ..."
                                    + content[-max_chars:]
                                ),
                                "truncated": True,
                                "total_chars": len(content),
                            }
                        return {"success": True, "content": content}
                except Exception as e:
                    print(e)
        # Auto-refresh for system files with auto-refresh enabled
        if (
            isinstance(f, VFSFile)
            and f.readonly
            and getattr(f, "_auto_refresh", False)
            and getattr(f, "local_path", None)
        ):
            refresh_result = self._load_shadow_content(path)
            if not refresh_result["success"]:
                return refresh_result

        # Auto-load for shadow files that were never loaded
        elif (
            isinstance(f, VFSFile)
            and f.backing_type == FileBackingType.SHADOW
            and not f.is_loaded
        ):
            load_result = self._load_shadow_content(path)
            if not load_result["success"]:
                return load_result

        # FIX #3: Auto-refresh for already-loaded shadow/modified files.
        # Detects external changes (another agent wrote to disk).
        # Does NOT refresh dirty files — agent has unsynced changes.
        elif (
            isinstance(f, VFSFile)
            and f.local_path
            and not f.is_dirty
            and f.backing_type in (FileBackingType.SHADOW, FileBackingType.MODIFIED)
            and f.is_loaded
        ):
            try:
                disk_mtime = os.path.getmtime(f.local_path)
                if f.local_mtime is None or disk_mtime > f.local_mtime:
                    # Disk has newer version — reload
                    reload_result = self._load_shadow_content(path)
                    if reload_result.get("success"):
                        # Restore MODIFIED→SHADOW since we just took disk as truth
                        f.backing_type = FileBackingType.SHADOW
            except OSError:
                # File disappeared from disk — surface as error
                return {
                    "success": False,
                    "error": f"Backing file missing: {f.local_path}",
                    "hint": "File was deleted externally. Use vfs_refresh_mount to clean up.",
                }

        content = f.content
        if len(content) > max_chars * 2.1:
            return {
                "success": True,
                "content": (
                    content[:max_chars]
                    + f"\n\n... [TRUNCATED: File is {len(content)} chars. "
                    "Use 'view' for specific lines] ...\n\n" + content[-max_chars:]
                ),
                "truncated": True,
                "total_chars": len(content),
            }

        return {"success": True, "content": content}

    # =========================================================================
    # OVERRIDE: WRITE WITH AUTO-SYNC
    # =========================================================================

    def write(self, path: str, content: str) -> dict:
        """
        Write file - synct Shadow-Dateien automatisch.
        """
        path = self._normalize_path(path)
        _before_content = self._safe_get_content(path)
        _before_meta = self._safe_get_meta(path)

        if self._is_directory(path):
            return {"success": False, "error": f"Cannot write to directory: {path}"}

        # EBENE 3: Writes auf Shared-Mounts (inkl. /global/ und Worktrees)
        # gehen durch den Shared-Store → instant visible für alle Agents
        # im selben Prozess.
        shared_info = self._get_shared_store_info(path)
        if shared_info is not None:
            mount_key, relative, local_base = shared_info
            try:
                from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

                gvfs = get_global_vfs()
                if mount_key == "global":
                    result = gvfs.write_file(relative, content, author=self.agent_name)
                else:
                    result = gvfs.shared_write(
                        mount_key,
                        relative,
                        content,
                        local_base=local_base,
                        author=self.agent_name,
                    )
                if result.get("success"):
                    f = self.files.get(path)
                    if isinstance(f, VFSFile):
                        f._content = content
                        f.is_dirty = False
                        f.backing_type = FileBackingType.SHADOW
                        f.size_bytes = len(content)
                        f.updated_at = datetime.now().isoformat()
                    else:
                        filename = self._get_basename(path)
                        local_path_on_disk = str(Path(local_base) / relative)
                        parent = self._get_parent_path(path)
                        if parent != "/" and not self._is_directory(parent):
                            self.mkdir(parent, parents=True)
                        self.files[path] = VFSFile(
                            filename=filename,
                            _content=content,
                            backing_type=FileBackingType.SHADOW,
                            local_path=local_path_on_disk,
                            local_mtime=os.path.getmtime(local_path_on_disk)
                            if os.path.exists(local_path_on_disk)
                            else None,
                            size_bytes=len(content),
                            line_count=len(content.splitlines()),
                            is_dirty=False,
                        )
                        self._shadow_index[path] = local_path_on_disk
                    self._dirty = True
                    self._emit_delta(path, "write", _before_content, content, before_meta=_before_meta)
                    return {
                        "success": True,
                        "message": f"Updated '{path}' via shared store",
                        "version": result.get("version"),
                    }
                    # Shared-store returned explicit failure — surface it.
                return {
                    "success": False,
                    "error": f"shared_write failed: {result.get('error', 'unknown')}",
                }
            except Exception as e:
                # Real exception (permission, disk, path-traversal) — do not
                # silently fall through. Return so the agent sees the error.
                return {
                    "success": False,
                    "error": f"shared_write exception: {type(e).__name__}: {e}",
                }

        if self._is_file(path):
            f = self.files[path]

            if f.readonly:
                return {"success": False, "error": f"Read-only: {path}"}

            # FIX #4: Optimistic concurrency check.
            # Detect external disk changes before overwriting.
            if isinstance(f, VFSFile) and f.local_path and not f.is_dirty:
                try:
                    disk_mtime = os.path.getmtime(f.local_path)
                    if f.local_mtime is not None and disk_mtime > f.local_mtime:
                        # Disk has newer version — another writer modified it.
                        # Reload transparently since we have no local changes
                        # to lose. This gives the agent the latest baseline.
                        reload_result = self._load_shadow_content(path)
                        if reload_result.get("success"):
                            f.backing_type = FileBackingType.SHADOW
                except OSError:
                    pass  # Disk file gone — write will re-create it

            # FIX #4b: Hard conflict — agent has dirty content AND disk changed.
            # Do not silently overwrite. Return a conflict error so the agent
            # can decide (force-write by deleting-and-recreating, merge manually,
            # or abort).
            if isinstance(f, VFSFile) and f.local_path and f.is_dirty:
                try:
                    disk_mtime = os.path.getmtime(f.local_path)
                    if f.local_mtime is not None and disk_mtime > f.local_mtime:
                        return {
                            "success": False,
                            "error": f"Write conflict: {path} was modified externally",
                            "conflict": True,
                            "hint": (
                                "Another agent or process modified this file on disk "
                                "while you had unsynced changes. To resolve: "
                                "(a) use vfs_shell('reason','rm <path>') then re-create with your content "
                                "to force-overwrite, or "
                                "(b) vfs_shell('reason','cat <path>') to see current disk content first."
                            ),
                            "disk_mtime": disk_mtime,
                            "local_mtime": f.local_mtime,
                        }
                except OSError:
                    pass

            # Update content
            if isinstance(f, VFSFile):
                f._content = content
                f.is_dirty = True

                if f.backing_type == FileBackingType.SHADOW:
                    f.backing_type = FileBackingType.MODIFIED

                # Auto-sync if enabled
                mount = self._get_mount_for_path(path)
                if mount and mount.auto_sync and f.local_path:
                    sync_result = self._sync_to_local(path)

                    f.updated_at = datetime.now().isoformat()
                    self._dirty = True

                    if sync_result["success"]:
                        self._emit_delta(path, "write", _before_content, content, before_meta=_before_meta)
                        return {
                            "success": True,
                            "message": f"Updated and synced '{path}'",
                            "synced_to": f.local_path,
                            "is_dirty": False,
                        }
            else:
                f.is_dirty = True
                f.content = content

            f.updated_at = datetime.now().isoformat()
            self._dirty = True

            self._emit_delta(path, "write", _before_content, content, before_meta=_before_meta)
            return {"success": True, "message": f"Updated '{path}'"}

        # Create new file
        return self.create(path, content)

    def _sync_to_local(self, path: str) -> dict:
        """
        Sync einer modifizierten Datei zurück auf Disk.
        """
        f = self.files.get(path)

        if not isinstance(f, VFSFile) or not f.local_path:
            return {"success": False, "error": "Not a shadow file"}

        if f._content is None:
            return {"success": False, "error": "No content to sync"}

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(f.local_path), exist_ok=True)

            with open(f.local_path, "w", encoding="utf-8") as file:
                file.write(f._content)

            f.local_mtime = os.path.getmtime(f.local_path)
            f.is_dirty = False
            f.backing_type = FileBackingType.SHADOW
            return {"success": True, "synced_to": f.local_path}

        except Exception as e:
            from toolboxv2 import get_app

            get_app().debug_rains(e)
            return {"success": False, "error": f"Sync error: {e}"}

    def _get_mount_for_path(self, path: str) -> ShadowMount | None:
        """Find mount that contains this path"""
        for mount_path, mount in self.mounts.items():
            if path.startswith(mount_path):
                return mount
        return None

    def _get_shared_store_info(self, path: str) -> tuple[str, str, str] | None:
        """
        Prüft ob *path* zu einem Shared-Mount gehört.

        Returns:
            (mount_key, relative_path, local_base) — oder None wenn nicht shared.
        """
        # Spezialfall /global/ — immer shared via GlobalVFSManager
        if path.startswith("/global/"):
            try:
                from toolboxv2.mods.isaa.base.patch.power_vfs import (
                    GLOBAL_VFS_PATH,
                    get_global_vfs,
                )

                relative = path[len(GLOBAL_VFS_PATH) :].lstrip("/")
                if not relative:
                    return None
                gvfs = get_global_vfs()
                return ("global", relative, str(gvfs.data_dir))
            except Exception as e:
                print(e)
                return None

        # Generische Shared-Mounts (Worktrees etc.)
        mount = self._get_mount_for_path(path)
        if mount is None:
            return None
        try:
            from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

            gvfs = get_global_vfs()
            mount_key = gvfs.get_mount_key_for(mount.local_path)
            if mount_key is None:
                return None
            # relative_path = path minus mount.vfs_path
            relative = path[len(mount.vfs_path) :].lstrip("/")
            if not relative:
                return None
            return (mount_key, relative, mount.local_path)
        except Exception as e:
            print(e)
            return None

    def sync_all(self) -> dict:
        """
        Sync alle dirty files.
        """
        synced = []
        errors = []

        for path, f in self.files.items():
            if isinstance(f, VFSFile) and f.is_dirty and f.local_path:
                result = self._sync_to_local(path)
                if result["success"]:
                    synced.append(path)
                else:
                    errors.append(f"{path}: {result['error']}")

        return {"success": len(errors) == 0, "synced": synced, "errors": errors}

    def append(self, path: str, content: str) -> dict:
        """Append to file - mit Shadow auto-sync"""
        path = self._normalize_path(path)
        _before_content = self._safe_get_content(path)
        _before_meta = self._safe_get_meta(path)

        if not self._is_file(path):
            return self.create(path, content)

        # EBENE 3: Shared-Store Support für Append
        shared_info = self._get_shared_store_info(path)
        if shared_info is not None:
            mount_key, relative, local_base = shared_info
            try:
                from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

                gvfs = get_global_vfs()

                # Wir holen den aktuellen Stand via VFS Read (liest direkt aus RAM-Store)
                current = self.read(path)
                new_content = (
                    current["content"] if current.get("success") else ""
                ) + content

                if mount_key == "global":
                    result = gvfs.write_file(
                        relative, new_content, author=self.agent_name
                    )
                else:
                    result = gvfs.shared_write(
                        mount_key,
                        relative,
                        new_content,
                        local_base=local_base,
                        author=self.agent_name,
                    )

                if result.get("success"):
                    f = self.files.get(path)
                    if isinstance(f, VFSFile):
                        f._content = new_content
                        f.is_dirty = False
                        f.backing_type = FileBackingType.SHADOW
                        f.size_bytes = len(new_content)
                        f.updated_at = datetime.now().isoformat()
                    self._dirty = True
                    return {
                        "success": True,
                        "message": f"Appended to '{path}' via shared store",
                    }
                return {
                    "success": False,
                    "error": f"shared_write failed: {result.get('error')}",
                }
            except Exception as e:
                return {"success": False, "error": f"shared_append exception: {e}"}

        f = self.files[path]

        if f.readonly:
            return {"success": False, "error": f"Read-only: {path}"}

        # Shadow: erst laden falls nötig
        if (
            isinstance(f, VFSFile)
            and f.backing_type == FileBackingType.SHADOW
            and not f.is_loaded
        ):
            load_result = self._load_shadow_content(path)
            if not load_result["success"]:
                return load_result

        # Append
        if isinstance(f, VFSFile):
            f._content = (f._content or "") + content
            f.is_dirty = True
            if f.backing_type == FileBackingType.SHADOW:
                f.backing_type = FileBackingType.MODIFIED

            # Auto-sync — BUG #4 FIX: Rückgabe prüfen und Fehler weitergeben
            mount = self._get_mount_for_path(path)
            if mount and mount.auto_sync and f.local_path:
                sync_result = self._sync_to_local(path)
                if not sync_result["success"]:
                    return {
                        "success": False,
                        "error": f"Sync failed: {sync_result['error']}",
                    }
        else:
            f.content += content

        f.updated_at = datetime.now().isoformat()
        self._dirty = True
        _after_content = self._safe_get_content(path)
        self._emit_delta(path, "append", _before_content, _after_content, before_meta=_before_meta)
        return {"success": True, "message": f"Appended to '{path}'"}

    def edit(self, path: str, line_start: int, line_end: int, new_content: str) -> dict:
        """Edit file by replacing lines (1-indexed) - mit Shadow auto-sync"""
        path = self._normalize_path(path)
        _before_content = self._safe_get_content(path)
        _before_meta = self._safe_get_meta(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]

        if f.readonly:
            return {"success": False, "error": f"Read-only: {path}"}

        # Shadow: erst laden falls nötig
        if (
            isinstance(f, VFSFile)
            and f.backing_type == FileBackingType.SHADOW
            and not f.is_loaded
        ):
            load_result = self._load_shadow_content(path)
            if not load_result["success"]:
                return load_result

        # Edit
        content = f._content if isinstance(f, VFSFile) else f.content
        lines = content.split("\n")
        start_idx = max(0, line_start - 1)
        end_idx = min(len(lines), line_end)

        new_lines = new_content.split("\n")
        lines = lines[:start_idx] + new_lines + lines[end_idx:]
        new_full_content = "\n".join(lines)

        if isinstance(f, VFSFile):
            f._content = new_full_content
            f.is_dirty = True
            if f.backing_type == FileBackingType.SHADOW:
                f.backing_type = FileBackingType.MODIFIED

            mount = self._get_mount_for_path(path)
            if mount and mount.auto_sync and f.local_path:
                sync_result = self._sync_to_local(path)
                if not sync_result["success"]:
                    return {
                        "success": False,
                        "error": f"Sync failed: {sync_result['error']}",
                    }

                # FIX: Update Shared-Store nach edit, sonst überschreibt
                # read() den editierten Content mit dem alten Store-Stand.
                shared_info = self._get_shared_store_info(path)
                if shared_info is not None:
                    mount_key, relative, local_base = shared_info
                    try:
                        from toolboxv2.mods.isaa.base.patch.power_vfs import (
                            get_global_vfs,
                        )

                        gvfs = get_global_vfs()
                        if mount_key == "global":
                            gvfs.write_file(
                                relative, new_full_content, author=self.agent_name
                            )
                        else:
                            gvfs.shared_write(
                                mount_key,
                                relative,
                                new_full_content,
                                local_base=local_base,
                                author=self.agent_name,
                            )
                    except Exception as e:
                        print(f"Error editing file: {e}")
                        pass  # Disk is already written, shared-store is best-effort
        else:
            f.content = new_full_content

        f.updated_at = datetime.now().isoformat()
        self._dirty = True
        _after_content = self._safe_get_content(path)
        self._emit_delta(path, "edit", _before_content, _after_content,
                         edit_range=(line_start, line_end), before_meta=_before_meta)
        return {
            "success": True,
            "message": f"Edited {path} lines {line_start}-{line_end}",
        }

    def delete(self, path: str) -> dict:
        """Delete a file - löscht auch lokale Shadow-Datei"""
        path = self._normalize_path(path)
        _before_content = self._safe_get_content(path)
        _before_meta = self._safe_get_meta(path)
        if not self._is_file(path):
            if self._is_directory(path):
                return self.rmdir(path)
            return {"success": False, "error": f"File not found: {path}"}

        # EBENE 3: /global/-Deletes gehen durch den Shared-Store
        shared_info = self._get_shared_store_info(path)
        if shared_info is not None:
            mount_key, relative, local_base = shared_info
            try:
                from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

                gvfs = get_global_vfs()
                if mount_key == "global":
                    result = gvfs.delete_file(relative)
                else:
                    result = gvfs.shared_delete(
                        mount_key, relative, local_base=local_base
                    )
                if result.get("success"):
                    self.files.pop(path, None)
                    self._shadow_index.pop(path, None)
                    self._dirty = True
                    self._emit_delta(path, "delete", _before_content, None, before_meta=_before_meta)
                    return {
                        "success": True,
                        "message": f"Deleted '{path}' via shared store",
                    }
            except Exception as e:
                print(e)
                pass

        f = self.files[path]

        if f.readonly:
            return {"success": False, "error": f"Cannot delete system file: {path}"}

        # Shadow: auch lokal löschen
        local_deleted = False
        if isinstance(f, VFSFile) and f.local_path:
            mount = self._get_mount_for_path(path)
            if mount and mount.auto_sync and not mount.readonly:
                try:
                    if os.path.exists(f.local_path):
                        os.remove(f.local_path)
                        local_deleted = True
                except OSError as e:
                    return {"success": False, "error": f"Cannot delete local file: {e}"}

            # Aus shadow index entfernen
            self._shadow_index.pop(path, None)

        del self.files[path]
        self._dirty = True

        msg = f"Deleted '{path}'"
        if local_deleted:
            msg += " (and local file)"

        self._emit_delta(path, "delete", _before_content, None, before_meta=_before_meta)
        return {"success": True, "message": msg}

    # =========================================================================
    # OPEN/CLOSE OPERATIONS
    # =========================================================================

    def open(self, path: str, line_start: int = 1, line_end: int = -1) -> dict:
        """Open file (make content visible in context)"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]

        # Lazy load for shadow files
        if isinstance(f, VFSFile) and f.backing_type == FileBackingType.SHADOW:
            load_result = self._load_shadow_content(path)
            if not load_result["success"]:
                return load_result

        f.state = "open"
        f.view_start = max(0, line_start - 1)
        f.view_end = line_end

        lines = f.content.split("\n")
        end = line_end if line_end > 0 else len(lines)
        visible = lines[f.view_start : end]

        self._dirty = True

        return {
            "success": True,
            "message": f"Opened '{path}' (lines {line_start}-{end})",
            "preview": "\n".join(visible[:5]) + ("..." if len(visible) > 5 else ""),
            "file_type": f.file_type.description if f.file_type else "Unknown",
        }

    def _load_shadow_content(self, path: str) -> dict:
        """
        Lädt Content einer Shadow-Datei vom Disk.
        """
        f = self.files[path]

        if not isinstance(f, VFSFile) or not f.local_path:
            return {"success": False, "error": "Not a shadow file"}

        if not os.path.exists(f.local_path):
            return {"success": False, "error": f"Backing file missing: {f.local_path}"}

        try:
            # Check if file changed on disk
            current_mtime = os.path.getmtime(f.local_path)

            with open(f.local_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()

            f._content = content
            f.local_mtime = current_mtime
            f.line_count = len(content.splitlines())
            f.size_bytes = len(content.encode("utf-8"))

            return {"success": True, "loaded_bytes": f.size_bytes}

        except Exception as e:
            return {"success": False, "error": f"Load error: {e}"}

    def _sync_from_local(self, path: str) -> dict:
        """
        Sync einer Datei vom lokalen Filesystem ins VFS.
        """
        f = self.files.get(path)

        if not isinstance(f, VFSFile) or not f.local_path:
            return {"success": False, "error": "Not a shadow file"}

        if not os.path.exists(f.local_path):
            return {"success": False, "error": f"Backing file missing: {f.local_path}"}

        try:
            current_mtime = os.path.getmtime(f.local_path)

            # Check if changed
            if (
                f.local_mtime
                and current_mtime == f.local_mtime
                and f._content is not None
            ):
                return {"success": True, "skipped": "Unchanged"}

            # Load content
            with open(f.local_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()

            f._content = content
            f.local_mtime = current_mtime
            f.line_count = len(content.splitlines())
            f.size_bytes = len(content.encode("utf-8"))

            # Falls Datei offen war, schließen um Context neu zu bauen
            if f.state == "open":
                f.state = "closed"

            return {"success": True, "loaded_bytes": f.size_bytes}

        except Exception as e:
            return {"success": False, "error": f"Sync from local error: {e}"}

    async def close(self, path: str, do_summary=True) -> dict:
        """Close file (create summary, remove from context)"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]
        if f.readonly:
            return {"success": False, "error": f"Cannot close system file: {path}"}

        # Generate summary

        f.mini_summary = f"[{f.size} chars]"
        if do_summary:
            if f.size > 100 and self._summarizer:
                try:
                    summary = self._summarizer(f.content[:2000])
                    if hasattr(summary, "__await__"):
                        summary = await summary
                    f.mini_summary = str(summary).strip()
                except Exception as e:
                    print(e)
                    f.mini_summary = (
                        f"[{f.size} chars, {len(f.content.splitlines())} lines]"
                    )

        f.state = "closed"
        if isinstance(f, VFSFile):
            if f.backing_type == FileBackingType.SHADOW and not f.is_dirty:
                f._content = None  # Gibt den String sofort für den Garbage Collector frei!
                f.size_bytes = 0  # Optional, wird beim nächsten read() neu gesetzt

        self._dirty = True

        return {"success": True, "summary": f.mini_summary}

    def view(self, path: str, line_start: int = 1, line_end: int = -1) -> dict:
        """View/adjust visible window"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]
        if f.state != "open":
            return self.open(path, line_start, line_end)

        f.view_start = max(0, line_start - 1)
        f.view_end = line_end

        lines = f.content.split("\n")
        end = line_end if line_end > 0 else len(lines)

        self._dirty = True

        return {"success": True, "content": "\n".join(lines[f.view_start : end])}

    def list_files(self) -> dict:
        """List all files with metadata (legacy compatibility)"""
        listing = []
        for path, f in self.files.items():
            info = {
                "path": path,
                "filename": f.filename,
                "state": f.state,
                "readonly": f.readonly,
                "size": f.size,
                "lines": len(f.content.splitlines()),
                "file_type": f.file_type.description if f.file_type else "Unknown",
                "is_executable": f.is_executable,
                "lsp_enabled": f.lsp_enabled,
            }
            if f.state == "closed" and f.mini_summary:
                info["summary"] = f.mini_summary
            listing.append(info)

        return {"success": True, "files": listing}

    # =========================================================================
    # LSP INTEGRATION
    # =========================================================================

    async def get_diagnostics(self, path: str, force_refresh: bool = False) -> dict:
        """
        Get LSP diagnostics for a file.

        Args:
            path: File path
            force_refresh: If True, refresh diagnostics from LSP server

        Returns:
            Result dict with diagnostics
        """
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]

        if f.file_type is None:
            f.__post_init__()

        # Auto-load shadow files that haven't been read yet
        if not f.is_loaded and f.backing_type == FileBackingType.SHADOW:
            load_result = self._load_shadow_content(path)
            if not load_result.get("success"):
                return {
                    "success": False,
                    "error": f"Cannot load file content: {load_result.get('error', 'unknown')}",
                }

        if not f.lsp_enabled:
            return {
                "success": True,
                "diagnostics": [],
                "message": "LSP not available for this file type",
            }

        if not self._lsp_manager:
            return {
                "success": True,
                "diagnostics": [],
                "message": "LSP manager not configured",
            }

        # Get diagnostics from LSP manager
        if force_refresh or not f.diagnostics:
            try:
                language_id = f.file_type.language_id if f.file_type else "plaintext"
                diagnostics = await self._lsp_manager.get_diagnostics(
                    path,
                    f.content,
                    language_id,
                )
                f.diagnostics = [
                    d.to_dict() if hasattr(d, "to_dict") else d for d in diagnostics
                ]
            except Exception as e:
                return {"success": False, "error": f"LSP error: {str(e)}"}

        return {
            "success": True,
            "diagnostics": f.diagnostics,
            "errors": len([d for d in f.diagnostics if d.get("severity") == "error"]),
            "warnings": len([d for d in f.diagnostics if d.get("severity") == "warning"]),
            "hints": len(
                [d for d in f.diagnostics if d.get("severity") in ("hint", "information")]
            ),
        }

    def get_file_info(self, path: str) -> dict:
        """Get file metadata without content"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            if self._is_directory(path):
                d = self.directories[path]
                return {
                    "success": True,
                    "path": path,
                    "type": "directory",
                    "name": d.name,
                    "readonly": d.readonly,
                    "created_at": d.created_at,
                    "updated_at": d.updated_at,
                }
            return {"success": False, "error": f"Path not found: {path}"}

        f = self.files[path]

        # Guard: shadow files that are not yet loaded have no content.
        # Use size_bytes directly and report lines as -1 (unknown).
        if isinstance(f, VFSFile) and not f.is_loaded:
            lines_count = -1
            file_size = f.size_bytes
        else:
            try:
                lines_count = len(f.content.splitlines())
                file_size = f.size
            except Exception as e:
                print(e)
                lines_count = -1
                file_size = f.size_bytes

        return {
            "success": True,
            "path": path,
            "type": "file",
            "filename": f.filename,
            "state": f.state,
            "readonly": f.readonly,
            "size": file_size,
            "lines": lines_count,
            "summary": f.mini_summary if f.state == "closed" else None,
            "created_at": f.created_at,
            "updated_at": f.updated_at,
            "view_range": (f.view_start + 1, f.view_end) if f.state == "open" else None,
            "file_type": f.file_type.description if f.file_type else "Unknown",
            "category": f.file_type.category.name if f.file_type else "UNKNOWN",
            "is_executable": f.is_executable,
            "lsp_enabled": f.lsp_enabled,
            "lsp_server": f.file_type.lsp_server if f.file_type else None,
            "icon": f.file_type.icon if f.file_type else "📄",
        }

    # =========================================================================
    # LOCAL FILE OPERATIONS
    # =========================================================================

    def load_from_local(
        self,
        local_path: str,
        vfs_path: str | None = None,
        allowed_dirs: list[str] | None = None,
        max_size_bytes: int = 1024 * 1024,
    ) -> dict:
        """Safely load a local file into VFS"""
        try:
            resolved_path = os.path.abspath(os.path.expanduser(local_path))
        except Exception as e:
            return {"success": False, "error": f"Invalid path: {e}"}

        if allowed_dirs:
            allowed = any(
                resolved_path.startswith(os.path.abspath(os.path.expanduser(d)))
                for d in allowed_dirs
            )
            if not allowed:
                return {
                    "success": False,
                    "error": f"Path not in allowed directories: {resolved_path}",
                }

        if not os.path.exists(resolved_path):
            return {"success": False, "error": f"File not found: {resolved_path}"}

        if not os.path.isfile(resolved_path):
            return {"success": False, "error": f"Not a file: {resolved_path}"}

        file_size = os.path.getsize(resolved_path)
        if file_size > max_size_bytes:
            return {
                "success": False,
                "error": f"File too large: {file_size} bytes (max: {max_size_bytes})",
            }

        if vfs_path is None:
            vfs_path = "/" + os.path.basename(resolved_path)

        try:
            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return {"success": False, "error": f"Read error: {e}"}

        result = self.create(vfs_path, content)

        if result["success"]:
            return {
                "success": True,
                "vfs_path": vfs_path,
                "source_path": resolved_path,
                "size_bytes": len(content),
                "lines": len(content.splitlines()),
                "file_type": self.files[
                    self._normalize_path(vfs_path)
                ].file_type.description,
            }

        return result

    def save_to_local(
        self,
        vfs_path: str,
        local_path: str,
        allowed_dirs: list[str] | None = None,
        overwrite: bool = False,
        create_dirs: bool = True,
    ) -> dict:
        """Safely save a VFS file to local filesystem"""
        vfs_path = self._normalize_path(vfs_path)

        if not self._is_file(vfs_path):
            return {"success": False, "error": f"VFS file not found: {vfs_path}"}

        vfs_file = self.files[vfs_path]

        try:
            resolved_path = os.path.abspath(os.path.expanduser(local_path))
        except Exception as e:
            return {"success": False, "error": f"Invalid path: {e}"}

        if allowed_dirs:
            allowed = any(
                resolved_path.startswith(os.path.abspath(os.path.expanduser(d)))
                for d in allowed_dirs
            )
            if not allowed:
                return {
                    "success": False,
                    "error": f"Path not in allowed directories: {resolved_path}",
                }

        if os.path.exists(resolved_path) and not overwrite:
            return {
                "success": False,
                "error": f"File exists (use overwrite=True): {resolved_path}",
            }

        parent_dir = os.path.dirname(resolved_path)
        if parent_dir and not os.path.exists(parent_dir):
            if create_dirs:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except Exception as e:
                    return {"success": False, "error": f"Cannot create directory: {e}"}
            else:
                return {
                    "success": False,
                    "error": f"Parent directory does not exist: {parent_dir}",
                }

        try:
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(vfs_file.content)
        except Exception as e:
            return {"success": False, "error": f"Write error: {e}"}

        return {
            "success": True,
            "vfs_path": vfs_path,
            "saved_path": resolved_path,
            "size_bytes": len(vfs_file.content),
            "lines": len(vfs_file.content.splitlines()),
        }

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def file_tree_string(self, as_list=False, max_depth=4):

        parts = ["=== VFS (Virtual File System) ==="]

        if self.mounts:
            parts.append("\n📂 Mounts:")
            for vfs_path, mount in self.mounts.items():
                parts.append(f"  {vfs_path} → {mount.local_path}")
        # Build directory tree summary
        dir_tree = self._build_tree_string(max_depth=max_depth)
        if dir_tree:
            parts.append("\n📁 Structure:")
            parts.append(dir_tree)

        return "\n".join(parts) if not as_list else parts

    def build_context_string(self) -> str:
        """Build VFS context string for LLM"""
        self.update_system_context()
        if not self._dirty and hasattr(self, '_context_cache'):
            return self._context_cache
        self._dirty = False
        parts = self.file_tree_string(as_list=True)

        # Order: system_context, active_rules, then others
        ordered = []
        if "/system_context.md" in self.files:
            ordered.append(("/system_context.md", self.files["/system_context.md"]))
        if "/active_rules.md" in self.files:
            ordered.append(("/active_rules.md", self.files["/active_rules.md"]))

        for path, f in self.files.items():
            if path not in ("/system_context.md", "/active_rules.md"):
                ordered.append((path, f))

        for path, f in ordered:
            if f.state == "open" and (not isinstance(f, VFSFile) or f.is_loaded):
                lines = f.content.split("\n")
                end = f.view_end if f.view_end > 0 else len(lines)
                visible = lines[f.view_start : end]
                total = len(lines)

                icon = f.file_type.icon if f.file_type else "📄"

                # Dirty indicator
                dirty = " [MODIFIED]" if isinstance(f, VFSFile) and f.is_dirty else ""

                if getattr(f, "show_full", False):
                    parts.append(
                        f"\n{icon} [{path}]{dirty} (full content, {total=}):"
                    )
                    parts.append("\n".join(lines))
                else:
                    if len(visible) > self.max_window_lines:
                        visible = visible[: self.max_window_lines]
                        parts.append(
                            f"\n{icon} [{path}]{dirty} (lines {f.view_start + 1}-{f.view_start + self.max_window_lines}, truncated, {total=}):"
                        )
                    else:
                        parts.append(
                            f"\n{icon} [{path}]{dirty} (lines {f.view_start + 1}-{end}, {total=}):"
                        )
                    parts.append("\n".join(visible))
        closed_count = sum(1 for f in self.files.values() if f.state == "closed")
        if closed_count > 0:
            parts.append(f"\n📋 {closed_count} closed files available")
        result = "\n".join(parts)
        self._context_cache = result
        return result

    def _build_tree_string(
        self, path: str = "/", prefix: str = "", max_depth: int = 3
    ) -> str:
        """Build a tree representation of the directory structure"""
        if max_depth <= 0:
            return ""

        lines = []
        contents = self._list_directory_contents(path)

        for i, item in enumerate(contents):
            is_last = i == len(contents) - 1
            current_prefix = "└── " if is_last else "├── "

            if item["type"] == "directory":
                icon = "📁"
                lines.append(f"{prefix}{current_prefix}{icon} {item['name']}/")

                child_prefix = prefix + ("    " if is_last else "│   ")
                subtree = self._build_tree_string(
                    item["path"], child_prefix, max_depth - 1
                )
                if subtree:
                    lines.append(subtree)
            else:
                f = self.files.get(item["path"])
                icon = f.file_type.icon if f and f.file_type else "📄"
                state = "[OPEN]" if f and f.state == "open" else ""
                lines.append(
                    f"{prefix}{current_prefix}{icon} {item['name']} {state}".rstrip()
                )

        return "\n".join(lines)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize VFS for checkpoint"""
        from toolboxv2.mods.isaa.base.patch.power_vfs import GLOBAL_VFS_PATH

        files_to_save = {}
        for path, f in self.files.items():
            if f.readonly:
                continue
            # Fix 2: Skip unmodified shadow files -> Saves ~99% of memory and disk space
            if f.backing_type == FileBackingType.SHADOW and not f.is_dirty and f.state == "closed":
                continue

            # Fix 1: Manual dict mapping -> Prevents 120MB+ DeepCopy leak from asdict()
            files_to_save[path] = {
                "filename": f.filename,
                "backing_type": f.backing_type,
                "_content": f._content,
                "local_path": f.local_path,
                "local_mtime": f.local_mtime,
                "size_bytes": f.size_bytes,
                "line_count": f.line_count,
                "state": f.state,
                "view_start": f.view_start,
                "view_end": f.view_end,
                "mini_summary": f.mini_summary,
                "readonly": f.readonly,
                "is_dirty": f.is_dirty,
                "show_full": getattr(f, "show_full", False),
                "created_at": f.created_at,
                "updated_at": f.updated_at,
                "is_executable": f.is_executable,
                "lsp_enabled": f.lsp_enabled,
                "diagnostics": f.diagnostics,
                "file_type": None,
            }

        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "max_window_lines": self.max_window_lines,
            "directories": {
                path: asdict(d) for path, d in self.directories.items() if not d.readonly
            },
            "files": files_to_save,
            "mounts": {
                vfs_path: {
                    "local_path": mount.local_path,
                    "readonly": mount.readonly,
                    "auto_sync": mount.auto_sync,
                    "allowed_extensions": mount.allowed_extensions,
                    "exclude_patterns": list(mount.exclude_patterns),
                }
                for vfs_path, mount in self.mounts.items()
                if vfs_path != GLOBAL_VFS_PATH  # Global wird separat gemountet
            },
        }

    def from_checkpoint(self, data: dict):
        """Restore VFS from checkpoint — skip re-scanning existing mounts."""
        # Restore directories
        for path, dir_data in data.get("directories", {}).items():
            self.directories[path] = VFSDirectory(**dir_data)

        # Restore files
        for path, file_data in data.get("files", {}).items():
            file_data.pop("file_type", None)
            file_data.pop("diagnostics", None)
            if "content" in file_data:
                file_data["_content"] = file_data.pop("content")
            self.files[path] = VFSFile(**file_data)

        # Restore mounts — but DON'T re-scan if files already loaded
        for vfs_path, mount_data in data.get("mounts", {}).items():
            if os.path.isdir(mount_data["local_path"]):
                # Check if we already have files under this mount from the checkpoint
                mount_prefix = vfs_path if vfs_path.endswith("/") else vfs_path + "/"
                has_files = any(
                    p == vfs_path or p.startswith(mount_prefix)
                    for p in self.files
                )
                if has_files:
                    # Mount exists in checkpoint — register mount metadata only,
                    # skip the expensive _scan_mount() since files are already loaded.
                    # _scan_mount will run on next explicit rescan/sync call.
                    self._register_mount_metadata(mount_data, vfs_path)
                else:
                    # No files from checkpoint — do a fresh scan
                    self.mount(**mount_data, vfs_path=vfs_path)

        self._dirty = True

    def _register_mount_metadata(self, mount_data: dict, vfs_path: str):
        """Register mount config without scanning. For checkpoint restore."""
        # Store mount config so rescan() can find it later.
        # Actual implementation depends on how self.mounts is structured.
        # Minimal version:
        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import ShadowMount
        mount = ShadowMount(
            vfs_path=vfs_path,
            **{k: v for k, v in mount_data.items() if k != 'vfs_path'}
        )
        self.mounts[vfs_path] = mount

    # =========================================================================
    # EXECUTABLE FILE HELPERS
    # =========================================================================

    def get_executable_files(self) -> list[dict]:
        """Get list of executable files"""
        executables = []
        for path, f in self.files.items():
            if f.is_executable and not f.readonly:
                executables.append(
                    {
                        "path": path,
                        "filename": f.filename,
                        "language": f.file_type.language_id if f.file_type else "unknown",
                        "size": f.size,
                    }
                )
        return executables

    def can_execute(self, path: str) -> bool:
        """Check if file can be executed"""
        path = self._normalize_path(path)
        if not self._is_file(path):
            return False
        return self.files[path].is_executable

    # =========================================================================
    # MOUNT OPERATIONS
    # =========================================================================

    def mount(
        self,
        local_path: str,
        vfs_path: str = "/project",
        allowed_extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        readonly: bool = False,
        auto_sync: bool = True,
    ) -> dict:
        """
        Mount einen lokalen Ordner als Shadow ins VFS.

        Scannt nur Metadata, lädt keine Dateiinhalte!

        Args:
            local_path: Lokaler Ordner
            vfs_path: Mount-Punkt im VFS
            allowed_extensions: Nur diese Extensions (None = alle)
            exclude_patterns: Ausschluss-Patterns
            readonly: Keine Schreiboperationen erlauben
            auto_sync: Änderungen sofort auf Disk schreiben

        Returns:
            Result mit Statistiken
        """
        local_path = os.path.abspath(os.path.expanduser(local_path))
        vfs_path = self._normalize_path(vfs_path)

        if not os.path.isdir(local_path):
            return {"success": False, "error": f"Not a directory: {local_path}"}

        if vfs_path in self.mounts:
            return {"success": False, "error": f"Already mounted: {vfs_path}"}

        # Create mount
        mount = ShadowMount(
            vfs_path=vfs_path,
            local_path=local_path,
            allowed_extensions=allowed_extensions,
            exclude_patterns=exclude_patterns
            or ShadowMount(vfs_path=vfs_path, local_path=local_path).exclude_patterns,
            readonly=readonly,
            auto_sync=auto_sync,
        )

        # Scan and index (metadata only!)
        stats = self._scan_mount(mount)

        self.mounts[vfs_path] = mount
        self._dirty = True

        # EBENE 2: Subscribe to the poll registry for external-change detection.
        # Other agents / editors / git modifying local_path will be reflected
        # lazily in this VFS via invalidation.
        try:
            from toolboxv2.mods.isaa.base.patch.mount_poll_registry import (
                get_mount_poll_registry,
            )

            get_mount_poll_registry().subscribe(
                local_path=local_path,
                vfs=self,
                exclude_patterns=list(mount.exclude_patterns),
            )
        except Exception as _poll_err:
            # Polling is a nice-to-have; don't fail the mount if the
            # registry is unavailable (e.g. during tests).
            import logging

            logging.getLogger("vfs.poll").warning(
                f"Could not subscribe to poll registry: {_poll_err}"
            )

        return {
            "success": True,
            "mount_point": vfs_path,
            "local_path": local_path,
            "files_indexed": stats["files"],
            "dirs_indexed": stats["dirs"],
            "total_size": stats["total_size"],
            "scan_time_ms": stats["scan_time_ms"],
        }

    def _scan_mount(self, mount: ShadowMount) -> dict:
        """
        Schneller Metadata-Scan ohne Content-Loading.

        Erstellt Shadow-Einträge für alle Dateien.
        """
        import fnmatch
        import time

        start = time.perf_counter()
        stats = {"files": 0, "dirs": 0, "total_size": 0, "skipped": 0, "removed": 0, "seen": 0}

        # Pre-split exclude patterns
        _exclude_exact: set[str] = set()
        _exclude_globs: list[str] = []
        for p in mount.exclude_patterns:
            if "*" in p or "?" in p or "[" in p:
                _exclude_globs.append(p)
            else:
                _exclude_exact.add(p)
        _allowed_ext = frozenset(mount.allowed_extensions) if mount.allowed_extensions else None

        def should_include(name: str, is_dir: bool) -> bool:
            if name in _exclude_exact:
                return False
            for pat in _exclude_globs:
                if fnmatch.fnmatch(name, pat):
                    return False
            if not is_dir and _allowed_ext is not None:
                _, ext = os.path.splitext(name)
                if ext.lower() not in _allowed_ext:
                    return False
            return True

        # Collect paths we visit — but reuse strings from self.files where possible
        # Instead of building a seen_file_paths set with N new f-strings,
        # we mark visited paths in a set of ids or use a compact marker.
        visited_files: set[str] = set()
        visited_dirs: set[str] = set()

        mount_vfs = mount.vfs_path  # cache attribute lookup

        for root, dirs, files in os.walk(mount.local_path):
            dirs[:] = [d for d in dirs if should_include(d, True)]

            rel_path = os.path.relpath(root, mount.local_path)
            if rel_path == ".":
                vfs_dir = mount_vfs
            else:
                vfs_dir = mount_vfs + "/" + rel_path.replace(os.sep, "/")

            # Directory entry
            if vfs_dir not in self.directories:
                self.directories[vfs_dir] = VFSDirectory(
                    name=os.path.basename(vfs_dir) or mount_vfs,
                    readonly=mount.readonly,
                )
                stats["dirs"] += 1
            visited_dirs.add(vfs_dir)

            # File entries
            vfs_dir_slash = vfs_dir + "/"
            for filename in files:
                if not should_include(filename, False):
                    stats["skipped"] += 1
                    continue

                local_file = os.path.join(root, filename)
                # Reuse the string from self.files if it already exists,
                # otherwise build it once
                vfs_file_path = vfs_dir_slash + filename

                try:
                    existing = self.files.get(vfs_file_path)
                    if existing is not None and isinstance(existing, VFSFile):
                        file_stat = os.stat(local_file)
                        if file_stat.st_size > mount.max_file_size:
                            visited_files.add(vfs_file_path)
                            stats["skipped"] += 1
                            continue
                        if existing.is_dirty:
                            visited_files.add(vfs_file_path)
                            stats["seen"] += 1
                            stats["files"] += 1
                            continue
                        if (existing.is_loaded and existing.local_mtime is not None
                            and file_stat.st_mtime > existing.local_mtime):
                            existing._content = None
                            existing.backing_type = FileBackingType.SHADOW
                        existing.local_mtime = file_stat.st_mtime
                        existing.size_bytes = file_stat.st_size
                        stats["seen"] += 1
                        stats["files"] += 1
                        stats["total_size"] += file_stat.st_size
                        visited_files.add(vfs_file_path)
                        continue

                    file_stat = os.stat(local_file)
                    if file_stat.st_size > mount.max_file_size:
                        visited_files.add(vfs_file_path)
                        stats["skipped"] += 1
                        continue

                    file_type = get_file_type(filename)
                    self.files[vfs_file_path] = VFSFile(
                        filename=filename,
                        backing_type=FileBackingType.SHADOW,
                        _content=None,
                        local_path=local_file,
                        local_mtime=file_stat.st_mtime,
                        size_bytes=file_stat.st_size,
                        line_count=-1,
                        file_type=file_type,
                        readonly=mount.readonly,
                    )
                    self._shadow_index[vfs_file_path] = local_file
                    visited_files.add(vfs_file_path)
                    stats["files"] += 1
                    stats["total_size"] += file_stat.st_size
                except (OSError, IOError):
                    stats["skipped"] += 1

        # Zombie cleanup — only for this mount's files
        mount_prefix = mount_vfs if mount_vfs.endswith("/") else mount_vfs + "/"
        to_remove_files = []
        for path in self.files:
            if not (path == mount_vfs or path.startswith(mount_prefix)):
                continue
            if path in visited_files:
                continue
            f = self.files[path]
            if not isinstance(f, VFSFile) or f.is_dirty or f.readonly:
                continue
            to_remove_files.append(path)

        for path in to_remove_files:
            del self.files[path]
            self._shadow_index.pop(path, None)
            stats["removed"] += 1

        to_remove_dirs = []
        for path in self.directories:
            if not (path == mount_vfs or path.startswith(mount_prefix)):
                continue
            if path in visited_dirs:
                continue
            d = self.directories[path]
            if d.readonly:
                continue
            has_remaining = any(p.startswith(path + "/") for p in self.files) or any(
                p.startswith(path + "/") for p in self.directories if p != path
            )
            if not has_remaining:
                to_remove_dirs.append(path)

        for path in to_remove_dirs:
            del self.directories[path]
            stats["removed"] += 1

        self._dirty = True
        stats["scan_time_ms"] = (time.perf_counter() - start) * 1000
        return stats

    def unmount(self, vfs_path: str, save_changes: bool = True) -> dict:
        """
        Unmount und optional alle Änderungen speichern.
        """
        vfs_path = self._normalize_path(vfs_path)

        if vfs_path not in self.mounts:
            return {"success": False, "error": f"Not mounted: {vfs_path}"}

        mount = self.mounts[vfs_path]
        # EBENE 2: Unsubscribe from poll registry
        try:
            from toolboxv2.mods.isaa.base.patch.mount_poll_registry import (
                get_mount_poll_registry,
            )

            get_mount_poll_registry().unsubscribe(mount.local_path, self)
        except Exception as e:
            print(e)
            pass

        # Save dirty files if requested
        saved = []
        if save_changes and not mount.readonly:
            for path, f in list(self.files.items()):
                if path.startswith(vfs_path) and isinstance(f, VFSFile):
                    if f.is_dirty and f.local_path:
                        result = self._sync_to_local(path)
                        if result["success"]:
                            saved.append(path)

        # Remove all entries under mount point
        paths_to_remove = [p for p in self.files if p.startswith(vfs_path)]
        for p in paths_to_remove:
            del self.files[p]
            self._shadow_index.pop(p, None)

        dirs_to_remove = [p for p in self.directories if p.startswith(vfs_path)]
        for p in dirs_to_remove:
            del self.directories[p]

        del self.mounts[vfs_path]
        self._dirty = True

        return {"success": True, "unmounted": vfs_path, "files_saved": saved}

    def refresh_mount(self, vfs_path: str, skip_rescan: bool = False) -> dict:
        """Re-scan mount for new/deleted files and sync content from local FS.

            Args:
                vfs_path: Mount point to refresh
                skip_rescan: If True, skip the _scan_mount() call (used when
                             the poll registry already detected and applied changes)
            """
        vfs_path = self._normalize_path(vfs_path)

        if vfs_path not in self.mounts:
            return {"success": False, "error": f"Not mounted: {vfs_path}"}

        mount = self.mounts[vfs_path]

        synced = []
        errors = []

        for path, f in list(self.files.items()):
            if path.startswith(vfs_path) and isinstance(f, VFSFile):
                if f.is_dirty:
                    continue
                if f.local_path:
                    result = self._sync_from_local(path)
                    if result.get("success"):
                        if "skipped" not in result:
                            synced.append(path)
                    else:
                        errors.append(f"{path}: {result.get('error', 'unknown')}")

        stats = {"files": 0, "dirs": 0}
        if not skip_rescan:
            stats = self._scan_mount(mount)

        return {
            "success": len(errors) == 0,
            "files_indexed": stats.get("files", 0),
            "content_synced": synced,
            "errors": errors,
        }

    def execute(
        self, path: str, args: list[str] | None = None, timeout: float = 30.0
    ) -> dict:
        """
        Execute a file if executable.

        Args:
            path: VFS path to executable file
            args: Command line arguments
            timeout: Execution timeout in seconds

        Returns:
            Result with stdout, stderr, return_code
        """
        import subprocess

        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]

        if not f.is_executable:
            return {"success": False, "error": f"File not executable: {path}"}

        # Shadow: nutze lokalen Pfad direkt
        if isinstance(f, VFSFile) and f.local_path and os.path.exists(f.local_path):
            # Sync falls dirty
            if f.is_dirty:
                sync_result = self._sync_to_local(path)
                if not sync_result["success"]:
                    return sync_result
            exec_path = f.local_path
        else:
            # Memory file: in temp speichern
            import tempfile

            ext = f.file_type.extension if f.file_type else ""

            content = f._content if isinstance(f, VFSFile) else f.content

            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as tmp:
                tmp.write(content)
                exec_path = tmp.name

        # Determine interpreter
        interpreter = []
        if f.file_type:
            lang = f.file_type.language_id
            if lang == "python":
                interpreter = [sys.executable]
            elif lang == "javascript":
                interpreter = ["node"]
            elif lang == "typescript":
                interpreter = ["npx", "ts-node"]
            elif lang == "shellscript":
                interpreter = ["bash"]
            elif lang == "rust":
                # Rust muss kompiliert werden - nicht direkt ausführbar
                return {"success": False, "error": "Rust files must be compiled first"}

        cmd = interpreter + [exec_path] + (args or [])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(exec_path) if exec_path.startswith("/") else None,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Execution timed out after {timeout}s"}
        except FileNotFoundError as e:
            return {"success": False, "error": f"Interpreter not found: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Execution error: {e}"}
        finally:
            # Cleanup temp file if created
            if not (isinstance(f, VFSFile) and f.local_path):
                try:
                    os.unlink(exec_path)
                except:
                    pass

    def __del__(self):
        """Safety net: ensure poll registry doesn't hold dangling refs."""
        try:
            from toolboxv2.mods.isaa.base.patch.mount_poll_registry import (
                get_mount_poll_registry,
            )

            get_mount_poll_registry().unsubscribe_all(self)
        except Exception as e:
            print(e)
            pass  # Destructor must never raise


import os


def sync_obsidian_vault(
    vfs: VirtualFileSystemV2, local_vault_path: str, vfs_path: str = "/obsidian"
) -> dict:
    """
    Bindet einen lokalen Obsidian-Vault bi-direktional in das VFS ein.

    Funktionsweise:
    1. VFS -> Disk: Aktiviert 'auto_sync', sodass Schreibvorgänge im VFS sofort gespeichert werden.
    2. Disk -> VFS: Führt 'refresh_mount' aus, um externe Änderungen aus Obsidian zu laden.
    3. Filtert Obsidian-Systemordner (.obsidian, .trash) automatisch heraus.
    """

    # 1. Pfad normalisieren und prüfen
    local_path = os.path.abspath(os.path.expanduser(local_vault_path))
    if not os.path.isdir(local_path):
        return {"success": False, "error": f"Obsidian vault not found at: {local_path}"}

    # 2. Bestehende ungespeicherte VFS-Änderungen auf Disk schreiben (Safety First)
    vfs.sync_all()

    # 3. Mount prüfen oder erstellen
    is_new_mount = False
    if vfs_path not in vfs.mounts:
        # Obsidian-spezifische Excludes, damit die KI nicht die Config zerschießt
        obsidian_excludes = [".obsidian", ".trash", ".git", "*.lock", ".DS_Store"]

        # Mount erstellen (Metadata Scan)
        mount_result = vfs.mount(
            local_path=local_path,
            vfs_path=vfs_path,
            exclude_patterns=obsidian_excludes,
            readonly=False,
            auto_sync=True,  # WICHTIG: VFS schreibt sofort auf Disk
        )

        if not mount_result["success"]:
            return mount_result
        is_new_mount = True

    # 4. Refresh durchführen (Disk -> VFS Synchronisation)
    # Scannt nach neuen Dateien im Explorer und lädt geänderte Inhalte neu
    refresh_result = vfs.refresh_mount(vfs_path)

    status_msg = "Mounted and synced" if is_new_mount else "Refreshed sync"

    return {
        "success": True,
        "message": f"{status_msg}: {local_path} <-> {vfs_path}",
        "stats": {
            "files_indexed": refresh_result.get("files_indexed", 0),
            "updated_from_disk": len(refresh_result.get("content_synced", [])),
            "errors": refresh_result.get("errors", []),
        },
    }


VirtualFileSystem = VirtualFileSystemV2
