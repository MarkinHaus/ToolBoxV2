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


def get_file_type(filename: str) -> FileTypeInfo:
    """Get file type info from filename"""
    # Check exact filename match first (e.g., Dockerfile)
    if filename in FILE_TYPES:
        return FILE_TYPES[filename]

    # Check extension
    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower in FILE_TYPES:
        return FILE_TYPES[ext_lower]

    # Default unknown type
    return FileTypeInfo(
        ext_lower or "",
        FileCategory.UNKNOWN,
        "plaintext",
        "text/plain",
        False,
        None,
        "📄",
        "Unknown",
    )


# =============================================================================
# VFS FILE
# =============================================================================


@dataclass
class VFSDirectory:
    """Represents a directory in the Virtual File System"""

    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    readonly: bool = False


class FileBackingType(Enum):
    """Wo lebt die Datei wirklich?"""

    MEMORY = auto()  # Nur im VFS (wie bisher)
    SHADOW = auto()  # Lazy-loaded vom lokalen FS
    MODIFIED = auto()  # Shadow, aber mit lokalen Änderungen (dirty)


@dataclass
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
    max_file_size: int = 1024 * 1024  # 1MB
    readonly: bool = False
    auto_sync: bool = True  # Änderungen sofort schreiben


@dataclass
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

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

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


# =============================================================================
# VIRTUAL FILE SYSTEM V2
# =============================================================================


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
        max_window_lines: int = 250,
        summarizer: Callable[[str], str] | None = None,
        lsp_manager: "LSPManager | None" = None,
    ):
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

        # Initialize root and system files
        self._init_root()
        self._init_system_files()

    def _init_root(self):
        """Initialize root directory"""
        self.directories["/"] = VFSDirectory(name="/", readonly=True)

    def _init_system_files(self):
        """Initialize read-only system files"""
        self.files["/system_context"] = VFSFile(
            filename="system_context",
            _content=self._build_system_context(),
            state="open",
            readonly=True,
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
        if "/system_context" in self.files:
            self.files["/system_context"].content = self._build_system_context()
            self.files["/system_context"].updated_at = datetime.now().isoformat()
            self._dirty = True

    def set_rules_file(self, content: str):
        """Set the active_rules file content (from RuleSet)"""
        path = "/active_rules"
        if path not in self.files:
            self.files[path] = VFSFile(
                filename="active_rules", _content=content, state="open", readonly=True
            )
        else:
            self.files[path].content = content
            self.files[path].updated_at = datetime.now().isoformat()
        self._dirty = True

    # =========================================================================
    # SYSTEM FILE MANAGEMENT (Permanent Read-Only Local Files)
    # =========================================================================

    def add_system_file(
        self,
        local_path: str,
        vfs_path: str | None = None,
        auto_refresh: bool = False,
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
                state="closed",  # Start closed, user can open if needed
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
        protected_files = ["/system_context", "/active_rules"]
        if vfs_path in protected_files:
            return {"success": False, "error": f"Cannot remove protected system file: {vfs_path}"}

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

        # 4. Re-initialize Root & System Files
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
            rel_path = path[len(mount.vfs_path):].lstrip('/')
            local_path = os.path.join(mount.local_path, rel_path.replace('/', os.sep))

            try:
                os.makedirs(local_path, exist_ok=True if parents else False)
            except OSError as e:
                return {"success": False, "error": f"Cannot create local directory: {e}"}

        # Create directory
        self.directories[path] = VFSDirectory(
            name=self._get_basename(path),
            readonly=mount.readonly if mount else False
        )
        self._dirty = True

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
                item_path = f"{path}/{item['name']}"
                if item["type"] == "directory":
                    result = self.rmdir(item_path, force=True)
                else:
                    result = self.delete(item_path)
                if not result["success"]:
                    return result

        del self.directories[path]
        self._dirty = True

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
            destination = f"{destination}/{self._get_basename(source)}"

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
        return {"success": True, "message": f"Moved {source} to {destination}"}

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    def create(self, path: str, content: str = "") -> dict:
        """
        Create a new file with mount-aware local file creation.
        """
        path = self._normalize_path(path)

        if self._path_exists(path):
            if self._is_file(path) and self.files[path].readonly:
                return {
                    "success": False,
                    "error": f"Cannot overwrite system file: {path}",
                }
            if self._is_directory(path):
                return {"success": False, "error": f"Path is a directory: {path}"}

        # Ensure parent exists
        error = self._ensure_parent_exists(path)
        if error:
            return error

        filename = self._get_basename(path)

        # Check if under a mount
        mount = self._get_mount_for_path(path)
        local_path = None
        backing_type = FileBackingType.MEMORY

        if mount:
            rel_path = path[len(mount.vfs_path):].lstrip('/')
            local_path = os.path.join(mount.local_path, rel_path.replace('/', os.sep))
            backing_type = FileBackingType.MODIFIED

        self.files[path] = VFSFile(
            filename=filename,
            _content=content,
            state="closed",
            local_path=local_path,
            backing_type=backing_type,
        )

        # If mount: create local file
        if mount and local_path:
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as e:
                return {"success": False, "error": f"Cannot create local file: {e}"}

        self._dirty = True

        return {
            "success": True,
            "message": f"Created '{path}' ({len(content)} chars)",
            "file_type": self.files[path].file_type.description,
        }

    def read(self, path: str) -> dict:
        """
        Read file content with auto-load for shadow files and auto-refresh for system files.
        """
        path = self._normalize_path(path)

        if not self._is_file(path):
            if self._is_directory(path):
                return {"success": False, "error": f"Cannot read directory: {path}"}
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]

        # Auto-refresh for system files with auto-refresh enabled
        if (isinstance(f, VFSFile) and f.readonly
            and getattr(f, "_auto_refresh", False)
            and getattr(f, "local_path", None)):
            refresh_result = self._load_shadow_content(path)
            if not refresh_result["success"]:
                return refresh_result

        # Auto-load for shadow files
        elif isinstance(f, VFSFile) and f.backing_type == FileBackingType.SHADOW and not f.is_loaded:
            load_result = self._load_shadow_content(path)
            if not load_result["success"]:
                return load_result

        return {"success": True, "content": f.content}

    # =========================================================================
    # OVERRIDE: WRITE WITH AUTO-SYNC
    # =========================================================================

    def write(self, path: str, content: str) -> dict:
        """
        Write file - synct Shadow-Dateien automatisch.
        """
        path = self._normalize_path(path)

        if self._is_directory(path):
            return {"success": False, "error": f"Cannot write to directory: {path}"}

        if self._is_file(path):
            f = self.files[path]

            if f.readonly:
                return {"success": False, "error": f"Read-only: {path}"}

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
                    if sync_result["success"]:
                        f.is_dirty = False
                        return {
                            "success": True,
                            "message": f"Updated and synced '{path}'",
                            "synced_to": f.local_path,
                        }
            else:
                f.content = content

            f.updated_at = datetime.now().isoformat()
            self._dirty = True

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

        if not f._content:
            return {"success": False, "error": "No content to sync"}

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(f.local_path), exist_ok=True)

            with open(f.local_path, "w", encoding="utf-8") as file:
                file.write(f._content)

            f.local_mtime = os.path.getmtime(f.local_path)
            f.is_dirty = False

            return {"success": True, "synced_to": f.local_path}

        except Exception as e:
            return {"success": False, "error": f"Sync error: {e}"}

    def _get_mount_for_path(self, path: str) -> ShadowMount | None:
        """Find mount that contains this path"""
        for mount_path, mount in self.mounts.items():
            if path.startswith(mount_path):
                return mount
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

        if not self._is_file(path):
            return self.create(path, content)

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

            # Auto-sync
            mount = self._get_mount_for_path(path)
            if mount and mount.auto_sync and f.local_path:
                self._sync_to_local(path)
        else:
            f.content += content

        f.updated_at = datetime.now().isoformat()
        self._dirty = True

        return {"success": True, "message": f"Appended to '{path}'"}

    def edit(self, path: str, line_start: int, line_end: int, new_content: str) -> dict:
        """Edit file by replacing lines (1-indexed) - mit Shadow auto-sync"""
        path = self._normalize_path(path)

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

            # Auto-sync
            mount = self._get_mount_for_path(path)
            if mount and mount.auto_sync and f.local_path:
                self._sync_to_local(path)
        else:
            f.content = new_full_content

        f.updated_at = datetime.now().isoformat()
        self._dirty = True

        return {
            "success": True,
            "message": f"Edited {path} lines {line_start}-{line_end}",
        }

    def delete(self, path: str) -> dict:
        """Delete a file - löscht auch lokale Shadow-Datei"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            if self._is_directory(path):
                return self.rmdir(path)
            return {"success": False, "error": f"File not found: {path}"}

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
            if f.local_mtime and current_mtime == f.local_mtime:
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
                f.is_loaded = True  # Content ist geladen

            return {"success": True, "loaded_bytes": f.size_bytes}

        except Exception as e:
            return {"success": False, "error": f"Sync from local error: {e}"}

    async def close(self, path: str) -> dict:
        """Close file (create summary, remove from context)"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]
        if f.readonly:
            return {"success": False, "error": f"Cannot close system file: {path}"}

        # Generate summary
        if f.size > 100 and self._summarizer:
            try:
                summary = self._summarizer(f.content[:2000])
                if hasattr(summary, "__await__"):
                    summary = await summary
                f.mini_summary = str(summary).strip()
            except Exception:
                f.mini_summary = (
                    f"[{f.size} chars, {len(f.content.splitlines())} lines]"
                )
        else:
            f.mini_summary = f"[{f.size} chars]"

        f.state = "closed"
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
                diagnostics = await self._lsp_manager.get_diagnostics(
                    path,
                    f.content,
                    f.file_type.language_id if f.file_type else "plaintext",
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
        return {
            "success": True,
            "path": path,
            "type": "file",
            "filename": f.filename,
            "state": f.state,
            "readonly": f.readonly,
            "size": f.size,
            "lines": len(f.content.splitlines()),
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

    def build_context_string(self) -> str:
        """Build VFS context string for LLM"""
        self.update_system_context()

        parts = ["=== VFS (Virtual File System) ==="]

        if self.mounts:
            parts.append("\n📂 Mounts:")
            for vfs_path, mount in self.mounts.items():
                parts.append(f"  {vfs_path} → {mount.local_path}")
        # Build directory tree summary
        dir_tree = self._build_tree_string()
        if dir_tree:
            parts.append("\n📁 Structure:")
            parts.append(dir_tree)

        # Order: system_context, active_rules, then others
        ordered = []
        if "/system_context" in self.files:
            ordered.append(("/system_context", self.files["/system_context"]))
        if "/active_rules" in self.files:
            ordered.append(("/active_rules", self.files["/active_rules"]))

        for path, f in self.files.items():
            if path not in ("/system_context", "/active_rules"):
                ordered.append((path, f))

        for path, f in ordered:
            if f.state == "open" and (not isinstance(f, VFSFile) or f.is_loaded):
                lines = f.content.split("\n")
                end = f.view_end if f.view_end > 0 else len(lines)
                visible = lines[f.view_start : end]

                icon = f.file_type.icon if f.file_type else "📄"

                # Dirty indicator
                dirty = " [MODIFIED]" if isinstance(f, VFSFile) and f.is_dirty else ""

                if len(visible) > self.max_window_lines:
                    visible = visible[: self.max_window_lines]
                    parts.append(
                        f"\n{icon} [{path}]{dirty} (lines {f.view_start + 1}-{f.view_start + self.max_window_lines}, truncated):"
                    )
                else:
                    parts.append(
                        f"\n{icon} [{path}]{dirty} (lines {f.view_start + 1}-{end}):"
                    )
                parts.append("\n".join(visible))
        closed_count = sum(1 for f in self.files.values() if f.state == "closed")
        if closed_count > 0:
            parts.append(f"\n📋 {closed_count} closed files available")

        return "\n".join(parts)

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
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "max_window_lines": self.max_window_lines,
            "directories": {
                path: asdict(d) for path, d in self.directories.items() if not d.readonly
            },
            "files": {
                path: {
                    **asdict(f),
                    "file_type": None,  # Don't serialize FileTypeInfo, reconstruct on load
                }
                for path, f in self.files.items()
                if not f.readonly
            },
        }

    def from_checkpoint(self, data: dict):
        """Restore VFS from checkpoint"""
        # Restore directories
        for path, dir_data in data.get("directories", {}).items():
            self.directories[path] = VFSDirectory(**dir_data)

        # Restore files
        for path, file_data in data.get("files", {}).items():
            file_data.pop("file_type", None)  # Remove if present
            file_data.pop("diagnostics", None)  # Remove diagnostics, will be refreshed
            if "content" in file_data:
                file_data["_content"] = file_data.pop("content")
            self.files[path] = VFSFile(**file_data)

        self._dirty = True

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
            exclude_patterns=exclude_patterns or ShadowMount(
            vfs_path=vfs_path,
            local_path=local_path).exclude_patterns,
            readonly=readonly,
            auto_sync=auto_sync,
        )

        # Scan and index (metadata only!)
        stats = self._scan_mount(mount)

        self.mounts[vfs_path] = mount
        self._dirty = True

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
        stats = {"files": 0, "dirs": 0, "total_size": 0, "skipped": 0}

        def should_include(path: str, is_dir: bool) -> bool:
            name = os.path.basename(path)
            for pattern in mount.exclude_patterns:
                if fnmatch.fnmatch(name, pattern):
                    return False
            if not is_dir and mount.allowed_extensions:
                ext = os.path.splitext(name)[1].lower()
                if ext not in mount.allowed_extensions:
                    return False
            return True

        # Walk directory tree
        for root, dirs, files in os.walk(mount.local_path):
            # Filter directories in-place
            dirs[:] = [d for d in dirs if should_include(os.path.join(root, d), True)]

            # Calculate VFS path for current directory
            rel_path = os.path.relpath(root, mount.local_path)
            if rel_path == ".":
                vfs_dir = mount.vfs_path
            else:
                vfs_dir = f"{mount.vfs_path}/{rel_path.replace(os.sep, '/')}"

            # Create directory entry
            if vfs_dir not in self.directories:
                self.directories[vfs_dir] = VFSDirectory(
                    name=os.path.basename(vfs_dir) or mount.vfs_path,
                    readonly=mount.readonly,
                )
                stats["dirs"] += 1

            # Create shadow file entries (metadata only!)
            for filename in files:
                local_file = os.path.join(root, filename)

                if not should_include(local_file, False):
                    stats["skipped"] += 1
                    continue

                try:
                    file_stat = os.stat(local_file)

                    # Skip too large files
                    if file_stat.st_size > mount.max_file_size:
                        stats["skipped"] += 1
                        continue

                    vfs_file_path = f"{vfs_dir}/{filename}"

                    # Create shadow entry (NO content loading!)
                    file_type = get_file_type(filename)

                    self.files[vfs_file_path] = VFSFile(
                        filename=filename,
                        backing_type=FileBackingType.SHADOW,
                        _content=None,  # NOT LOADED
                        local_path=local_file,
                        local_mtime=file_stat.st_mtime,
                        size_bytes=file_stat.st_size,
                        line_count=-1,  # Unknown until loaded
                        file_type=file_type,
                        readonly=mount.readonly,
                    )

                    self._shadow_index[vfs_file_path] = local_file
                    stats["files"] += 1
                    stats["total_size"] += file_stat.st_size

                except (OSError, IOError):
                    stats["skipped"] += 1

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

    def refresh_mount(self, vfs_path: str) -> dict:
        """
        Re-scan mount für neue/gelöschte Dateien und sync Content vom lokalen FS.
        """
        vfs_path = self._normalize_path(vfs_path)

        if vfs_path not in self.mounts:
            return {"success": False, "error": f"Not mounted: {vfs_path}"}

        mount = self.mounts[vfs_path]

        # Sync Content vom lokalen FS
        synced = []
        errors = []

        for path, f in list(self.files.items()):
            if path.startswith(vfs_path) and isinstance(f, VFSFile):
                # Dirty-Dateien überspringen (VFS-Änderungen haben Priorität)
                if f.is_dirty:
                    continue

                if f.local_path:
                    result = self._sync_from_local(path)
                    if result.get("success"):
                        if "skipped" not in result:
                            synced.append(path)
                    else:
                        errors.append(f"{path}: {result.get('error', 'unknown')}")

        # Re-scan für neue/gelöschte Dateien
        stats = self._scan_mount(mount)

        return {
            "success": len(errors) == 0,
            "files_indexed": stats["files"],
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


import os


def sync_obsidian_vault(vfs: VirtualFileSystemV2, local_vault_path: str, vfs_path: str = "/obsidian") -> dict:
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
        obsidian_excludes = [
            ".obsidian",
            ".trash",
            ".git",
            "*.lock",
            ".DS_Store"
        ]

        # Mount erstellen (Metadata Scan)
        mount_result = vfs.mount(
            local_path=local_path,
            vfs_path=vfs_path,
            exclude_patterns=obsidian_excludes,
            readonly=False,
            auto_sync=True  # WICHTIG: VFS schreibt sofort auf Disk
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
            "errors": refresh_result.get("errors", [])
        }
    }

VirtualFileSystem = VirtualFileSystemV2
