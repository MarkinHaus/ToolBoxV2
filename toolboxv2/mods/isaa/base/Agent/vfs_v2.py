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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.lsp_manager import LSPManager, Diagnostic


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
    ".py": FileTypeInfo(".py", FileCategory.CODE, "python", "text/x-python", True, "pylsp", "🐍", "Python"),
    ".pyw": FileTypeInfo(".pyw", FileCategory.CODE, "python", "text/x-python", True, "pylsp", "🐍", "Python (Windows)"),
    ".pyi": FileTypeInfo(".pyi", FileCategory.CODE, "python", "text/x-python", False, "pylsp", "🐍", "Python Stub"),

    # JavaScript/TypeScript
    ".js": FileTypeInfo(".js", FileCategory.CODE, "javascript", "application/javascript", True, "typescript-language-server", "📜", "JavaScript"),
    ".mjs": FileTypeInfo(".mjs", FileCategory.CODE, "javascript", "application/javascript", True, "typescript-language-server", "📜", "JavaScript Module"),
    ".jsx": FileTypeInfo(".jsx", FileCategory.CODE, "javascriptreact", "text/jsx", True, "typescript-language-server", "⚛️", "React JSX"),
    ".ts": FileTypeInfo(".ts", FileCategory.CODE, "typescript", "application/typescript", True, "typescript-language-server", "📘", "TypeScript"),
    ".tsx": FileTypeInfo(".tsx", FileCategory.CODE, "typescriptreact", "text/tsx", True, "typescript-language-server", "⚛️", "React TSX"),

    # Rust
    ".rs": FileTypeInfo(".rs", FileCategory.CODE, "rust", "text/x-rust", True, "rust-analyzer", "🦀", "Rust"),

    # Go
    ".go": FileTypeInfo(".go", FileCategory.CODE, "go", "text/x-go", True, "gopls", "🐹", "Go"),

    # C/C++
    ".c": FileTypeInfo(".c", FileCategory.CODE, "c", "text/x-c", True, "clangd", "🔧", "C"),
    ".h": FileTypeInfo(".h", FileCategory.CODE, "c", "text/x-c", False, "clangd", "🔧", "C Header"),
    ".cpp": FileTypeInfo(".cpp", FileCategory.CODE, "cpp", "text/x-c++", True, "clangd", "🔧", "C++"),
    ".hpp": FileTypeInfo(".hpp", FileCategory.CODE, "cpp", "text/x-c++", False, "clangd", "🔧", "C++ Header"),
    ".cc": FileTypeInfo(".cc", FileCategory.CODE, "cpp", "text/x-c++", True, "clangd", "🔧", "C++"),

    # Web
    ".html": FileTypeInfo(".html", FileCategory.WEB, "html", "text/html", False, "vscode-html-language-server", "🌐", "HTML"),
    ".htm": FileTypeInfo(".htm", FileCategory.WEB, "html", "text/html", False, "vscode-html-language-server", "🌐", "HTML"),
    ".css": FileTypeInfo(".css", FileCategory.WEB, "css", "text/css", False, "vscode-css-language-server", "🎨", "CSS"),
    ".scss": FileTypeInfo(".scss", FileCategory.WEB, "scss", "text/x-scss", False, "vscode-css-language-server", "🎨", "SCSS"),
    ".less": FileTypeInfo(".less", FileCategory.WEB, "less", "text/x-less", False, "vscode-css-language-server", "🎨", "Less"),
    ".vue": FileTypeInfo(".vue", FileCategory.WEB, "vue", "text/x-vue", False, "volar", "💚", "Vue"),
    ".svelte": FileTypeInfo(".svelte", FileCategory.WEB, "svelte", "text/x-svelte", False, "svelte-language-server", "🔥", "Svelte"),

    # Data
    ".json": FileTypeInfo(".json", FileCategory.DATA, "json", "application/json", False, "vscode-json-language-server", "📋", "JSON"),
    ".jsonc": FileTypeInfo(".jsonc", FileCategory.DATA, "jsonc", "application/json", False, "vscode-json-language-server", "📋", "JSON with Comments"),
    ".xml": FileTypeInfo(".xml", FileCategory.DATA, "xml", "application/xml", False, None, "📋", "XML"),
    ".csv": FileTypeInfo(".csv", FileCategory.DATA, "csv", "text/csv", False, None, "📊", "CSV"),
    ".tsv": FileTypeInfo(".tsv", FileCategory.DATA, "tsv", "text/tab-separated-values", False, None, "📊", "TSV"),

    # Config
    ".yaml": FileTypeInfo(".yaml", FileCategory.CONFIG, "yaml", "text/yaml", False, "yaml-language-server", "⚙️", "YAML"),
    ".yml": FileTypeInfo(".yml", FileCategory.CONFIG, "yaml", "text/yaml", False, "yaml-language-server", "⚙️", "YAML"),
    ".toml": FileTypeInfo(".toml", FileCategory.CONFIG, "toml", "text/x-toml", False, "taplo", "⚙️", "TOML"),
    ".ini": FileTypeInfo(".ini", FileCategory.CONFIG, "ini", "text/x-ini", False, None, "⚙️", "INI"),
    ".env": FileTypeInfo(".env", FileCategory.CONFIG, "dotenv", "text/plain", False, None, "🔐", "Environment"),
    ".editorconfig": FileTypeInfo(".editorconfig", FileCategory.CONFIG, "editorconfig", "text/plain", False, None, "⚙️", "EditorConfig"),

    # Docs
    ".md": FileTypeInfo(".md", FileCategory.DOCS, "markdown", "text/markdown", False, None, "📝", "Markdown"),
    ".mdx": FileTypeInfo(".mdx", FileCategory.DOCS, "mdx", "text/mdx", False, None, "📝", "MDX"),
    ".rst": FileTypeInfo(".rst", FileCategory.DOCS, "restructuredtext", "text/x-rst", False, None, "📝", "reStructuredText"),
    ".txt": FileTypeInfo(".txt", FileCategory.DOCS, "plaintext", "text/plain", False, None, "📄", "Plain Text"),

    # Shell
    ".sh": FileTypeInfo(".sh", FileCategory.CODE, "shellscript", "text/x-shellscript", True, "bash-language-server", "🐚", "Shell Script"),
    ".bash": FileTypeInfo(".bash", FileCategory.CODE, "shellscript", "text/x-shellscript", True, "bash-language-server", "🐚", "Bash Script"),
    ".zsh": FileTypeInfo(".zsh", FileCategory.CODE, "shellscript", "text/x-shellscript", True, None, "🐚", "Zsh Script"),
    ".ps1": FileTypeInfo(".ps1", FileCategory.CODE, "powershell", "text/x-powershell", True, None, "💠", "PowerShell"),

    # SQL
    ".sql": FileTypeInfo(".sql", FileCategory.CODE, "sql", "text/x-sql", True, None, "🗃️", "SQL"),

    # Docker
    "Dockerfile": FileTypeInfo("Dockerfile", FileCategory.CONFIG, "dockerfile", "text/x-dockerfile", False, "dockerfile-language-server", "🐳", "Dockerfile"),
    ".dockerfile": FileTypeInfo(".dockerfile", FileCategory.CONFIG, "dockerfile", "text/x-dockerfile", False, "dockerfile-language-server", "🐳", "Dockerfile"),

    # TB (ToolBoxV2 Language)
    ".tb": FileTypeInfo(".tb", FileCategory.CODE, "tb", "text/x-tb", True, None, "🧰", "TB Language"),
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
    return FileTypeInfo(ext_lower or "", FileCategory.UNKNOWN, "plaintext", "text/plain", False, None, "📄", "Unknown")


# =============================================================================
# VFS FILE
# =============================================================================

@dataclass
class VFSFile:
    """Represents a file in the Virtual File System"""
    filename: str
    content: str
    state: str = "closed"              # "open" or "closed"
    view_start: int = 0
    view_end: int = -1
    mini_summary: str = ""
    readonly: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # V2 additions
    file_type: FileTypeInfo | None = None
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
class VFSDirectory:
    """Represents a directory in the Virtual File System"""
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    readonly: bool = False


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
        lsp_manager: 'LSPManager | None' = None
    ):
        self.session_id = session_id
        self.agent_name = agent_name
        self.max_window_lines = max_window_lines
        self._summarizer = summarizer
        self._lsp_manager = lsp_manager

        # Storage: path -> VFSFile or VFSDirectory
        self.files: dict[str, VFSFile] = {}
        self.directories: dict[str, VFSDirectory] = {}

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
        if "/system_context" in self.files:
            self.files["/system_context"].content = self._build_system_context()
            self.files["/system_context"].updated_at = datetime.now().isoformat()
            self._dirty = True

    def set_rules_file(self, content: str):
        """Set the active_rules file content (from RuleSet)"""
        path = "/active_rules"
        if path not in self.files:
            self.files[path] = VFSFile(
                filename="active_rules",
                content=content,
                state="open",
                readonly=True
            )
        else:
            self.files[path].content = content
            self.files[path].updated_at = datetime.now().isoformat()
        self._dirty = True

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
            return {"success": False, "error": f"Parent directory does not exist: {parent}"}
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
                return {"success": False, "error": f"Parent directory does not exist: {parent}"}

        # Create directory
        self.directories[path] = VFSDirectory(name=self._get_basename(path))
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
            return {"success": False, "error": f"Cannot remove readonly directory: {path}"}

        # Check if directory is empty
        contents = self._list_directory_contents(path)

        if contents and not force:
            return {"success": False, "error": f"Directory not empty: {path} (use force=True to remove)"}

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
                relative = dir_path[len(prefix):]
                if "/" not in relative:
                    contents.append({
                        "name": relative,
                        "type": "directory",
                        "path": dir_path
                    })

        # Check files
        for file_path in self.files:
            if file_path.startswith(prefix):
                relative = file_path[len(prefix):]
                if "/" not in relative:
                    f = self.files[file_path]
                    contents.append({
                        "name": relative,
                        "type": "file",
                        "path": file_path,
                        "size": len(f.content),
                        "state": f.state,
                        "file_type": f.file_type.description if f.file_type else "Unknown"
                    })

        return sorted(contents, key=lambda x: (x["type"] != "directory", x["name"]))

    def ls(self, path: str = "/", recursive: bool = False, show_hidden: bool = False) -> dict:
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

        contents = list_recursive(path) if recursive else self._list_directory_contents(path)

        if not show_hidden:
            contents = [c for c in contents if not c["name"].startswith(".")]

        return {
            "success": True,
            "path": path,
            "contents": contents,
            "total_items": len(contents)
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
            return {"success": False, "error": f"Cannot move readonly directory: {source}"}

        # If destination is existing directory, move into it
        if self._is_directory(destination):
            destination = f"{destination}/{self._get_basename(source)}"

        if self._path_exists(destination):
            return {"success": False, "error": f"Destination already exists: {destination}"}

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
            self.files[destination].file_type = get_file_type(self._get_basename(destination))
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
                    new_path = new_prefix + dir_path[len(old_prefix):]
                    dirs_to_move.append((dir_path, new_path))

            for file_path in list(self.files.keys()):
                if file_path.startswith(old_prefix):
                    new_path = new_prefix + file_path[len(old_prefix):]
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
        """Create a new file"""
        path = self._normalize_path(path)

        if self._path_exists(path):
            if self._is_file(path) and self.files[path].readonly:
                return {"success": False, "error": f"Cannot overwrite system file: {path}"}
            if self._is_directory(path):
                return {"success": False, "error": f"Path is a directory: {path}"}

        # Ensure parent exists
        error = self._ensure_parent_exists(path)
        if error:
            return error

        filename = self._get_basename(path)
        self.files[path] = VFSFile(filename=filename, content=content, state="closed")
        self._dirty = True

        return {"success": True, "message": f"Created '{path}' ({len(content)} chars)", "file_type": self.files[path].file_type.description}

    def read(self, path: str) -> dict:
        """Read file content"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            if self._is_directory(path):
                return {"success": False, "error": f"Cannot read directory: {path}"}
            return {"success": False, "error": f"File not found: {path}"}

        return {"success": True, "content": self.files[path].content}

    def write(self, path: str, content: str) -> dict:
        """Write/overwrite file content"""
        path = self._normalize_path(path)

        if self._is_directory(path):
            return {"success": False, "error": f"Cannot write to directory: {path}"}

        if self._is_file(path):
            if self.files[path].readonly:
                return {"success": False, "error": f"Read-only: {path}"}
            self.files[path].content = content
            self.files[path].updated_at = datetime.now().isoformat()
            self._dirty = True
            return {"success": True, "message": f"Updated '{path}'"}

        return self.create(path, content)

    def append(self, path: str, content: str) -> dict:
        """Append to file"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return self.create(path, content)

        if self.files[path].readonly:
            return {"success": False, "error": f"Read-only: {path}"}

        self.files[path].content += content
        self.files[path].updated_at = datetime.now().isoformat()
        self._dirty = True

        return {"success": True, "message": f"Appended to '{path}'"}

    def edit(self, path: str, line_start: int, line_end: int, new_content: str) -> dict:
        """Edit file by replacing lines (1-indexed)"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]
        if f.readonly:
            return {"success": False, "error": f"Read-only: {path}"}

        lines = f.content.split('\n')
        start_idx = max(0, line_start - 1)
        end_idx = min(len(lines), line_end)

        new_lines = new_content.split('\n')
        lines = lines[:start_idx] + new_lines + lines[end_idx:]

        f.content = '\n'.join(lines)
        f.updated_at = datetime.now().isoformat()
        self._dirty = True

        return {"success": True, "message": f"Edited {path} lines {line_start}-{line_end}"}

    def delete(self, path: str) -> dict:
        """Delete a file"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            if self._is_directory(path):
                return self.rmdir(path)
            return {"success": False, "error": f"File not found: {path}"}

        if self.files[path].readonly:
            return {"success": False, "error": f"Cannot delete system file: {path}"}

        del self.files[path]
        self._dirty = True

        return {"success": True, "message": f"Deleted '{path}'"}

    # =========================================================================
    # OPEN/CLOSE OPERATIONS
    # =========================================================================

    def open(self, path: str, line_start: int = 1, line_end: int = -1) -> dict:
        """Open file (make content visible in context)"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]
        f.state = "open"
        f.view_start = max(0, line_start - 1)
        f.view_end = line_end

        lines = f.content.split('\n')
        end = line_end if line_end > 0 else len(lines)
        visible = lines[f.view_start:end]

        self._dirty = True

        return {
            "success": True,
            "message": f"Opened '{path}' (lines {line_start}-{end})",
            "preview": '\n'.join(visible[:5]) + ("..." if len(visible) > 5 else ""),
            "file_type": f.file_type.description if f.file_type else "Unknown"
        }

    async def close(self, path: str) -> dict:
        """Close file (create summary, remove from context)"""
        path = self._normalize_path(path)

        if not self._is_file(path):
            return {"success": False, "error": f"File not found: {path}"}

        f = self.files[path]
        if f.readonly:
            return {"success": False, "error": f"Cannot close system file: {path}"}

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

        lines = f.content.split('\n')
        end = line_end if line_end > 0 else len(lines)

        self._dirty = True

        return {"success": True, "content": '\n'.join(lines[f.view_start:end])}

    def list_files(self) -> dict:
        """List all files with metadata (legacy compatibility)"""
        listing = []
        for path, f in self.files.items():
            info = {
                "path": path,
                "filename": f.filename,
                "state": f.state,
                "readonly": f.readonly,
                "size": len(f.content),
                "lines": len(f.content.splitlines()),
                "file_type": f.file_type.description if f.file_type else "Unknown",
                "is_executable": f.is_executable,
                "lsp_enabled": f.lsp_enabled
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
            return {"success": True, "diagnostics": [], "message": "LSP not available for this file type"}

        if not self._lsp_manager:
            return {"success": True, "diagnostics": [], "message": "LSP manager not configured"}

        # Get diagnostics from LSP manager
        if force_refresh or not f.diagnostics:
            try:
                diagnostics = await self._lsp_manager.get_diagnostics(
                    path,
                    f.content,
                    f.file_type.language_id if f.file_type else "plaintext"
                )
                f.diagnostics = [d.to_dict() if hasattr(d, 'to_dict') else d for d in diagnostics]
            except Exception as e:
                return {"success": False, "error": f"LSP error: {str(e)}"}

        return {
            "success": True,
            "diagnostics": f.diagnostics,
            "errors": len([d for d in f.diagnostics if d.get("severity") == "error"]),
            "warnings": len([d for d in f.diagnostics if d.get("severity") == "warning"]),
            "hints": len([d for d in f.diagnostics if d.get("severity") in ("hint", "information")])
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
                    "updated_at": d.updated_at
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
            "size": len(f.content),
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
            "icon": f.file_type.icon if f.file_type else "📄"
        }

    # =========================================================================
    # LOCAL FILE OPERATIONS
    # =========================================================================

    def load_from_local(
        self,
        local_path: str,
        vfs_path: str | None = None,
        allowed_dirs: list[str] | None = None,
        max_size_bytes: int = 1024 * 1024
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
                return {"success": False, "error": f"Path not in allowed directories: {resolved_path}"}

        if not os.path.exists(resolved_path):
            return {"success": False, "error": f"File not found: {resolved_path}"}

        if not os.path.isfile(resolved_path):
            return {"success": False, "error": f"Not a file: {resolved_path}"}

        file_size = os.path.getsize(resolved_path)
        if file_size > max_size_bytes:
            return {"success": False, "error": f"File too large: {file_size} bytes (max: {max_size_bytes})"}

        if vfs_path is None:
            vfs_path = "/" + os.path.basename(resolved_path)

        try:
            with open(resolved_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return {"success": False, "error": f"Read error: {e}"}

        result = self.create(vfs_path, content)

        if result['success']:
            return {
                "success": True,
                "vfs_path": vfs_path,
                "source_path": resolved_path,
                "size_bytes": len(content),
                "lines": len(content.splitlines()),
                "file_type": self.files[self._normalize_path(vfs_path)].file_type.description
            }

        return result

    def save_to_local(
        self,
        vfs_path: str,
        local_path: str,
        allowed_dirs: list[str] | None = None,
        overwrite: bool = False,
        create_dirs: bool = True
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
            "vfs_path": vfs_path,
            "saved_path": resolved_path,
            "size_bytes": len(vfs_file.content),
            "lines": len(vfs_file.content.splitlines())
        }

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def build_context_string(self) -> str:
        """Build VFS context string for LLM"""
        self.update_system_context()

        parts = ["=== VFS (Virtual File System) ==="]

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
            if f.state == "open":
                lines = f.content.split('\n')
                end = f.view_end if f.view_end > 0 else len(lines)
                visible = lines[f.view_start:end]

                icon = f.file_type.icon if f.file_type else "📄"

                if len(visible) > self.max_window_lines:
                    visible = visible[:self.max_window_lines]
                    parts.append(f"\n{icon} [{path}] OPEN (lines {f.view_start + 1}-{f.view_start + self.max_window_lines}, truncated):")
                else:
                    parts.append(f"\n{icon} [{path}] OPEN (lines {f.view_start + 1}-{end}):")
                parts.append('\n'.join(visible))
            else:
                icon = f.file_type.icon if f.file_type else "📄"
                summary = f.mini_summary or f"[{len(f.content)} chars]"
                parts.append(f"\n• {icon} {path} [closed]: {summary}")

        return '\n'.join(parts)

    def _build_tree_string(self, path: str = "/", prefix: str = "", max_depth: int = 3) -> str:
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
                subtree = self._build_tree_string(item["path"], child_prefix, max_depth - 1)
                if subtree:
                    lines.append(subtree)
            else:
                f = self.files.get(item["path"])
                icon = f.file_type.icon if f and f.file_type else "📄"
                state = "[OPEN]" if f and f.state == "open" else ""
                lines.append(f"{prefix}{current_prefix}{icon} {item['name']} {state}".rstrip())

        return '\n'.join(lines)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize VFS for checkpoint"""
        return {
            'session_id': self.session_id,
            'agent_name': self.agent_name,
            'max_window_lines': self.max_window_lines,
            'directories': {
                path: asdict(d) for path, d in self.directories.items()
                if not d.readonly
            },
            'files': {
                path: {
                    **asdict(f),
                    'file_type': None  # Don't serialize FileTypeInfo, reconstruct on load
                } for path, f in self.files.items()
                if not f.readonly
            }
        }

    def from_checkpoint(self, data: dict):
        """Restore VFS from checkpoint"""
        # Restore directories
        for path, dir_data in data.get('directories', {}).items():
            self.directories[path] = VFSDirectory(**dir_data)

        # Restore files
        for path, file_data in data.get('files', {}).items():
            file_data.pop('file_type', None)  # Remove if present
            file_data.pop('diagnostics', None)  # Remove diagnostics, will be refreshed
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
                executables.append({
                    "path": path,
                    "filename": f.filename,
                    "language": f.file_type.language_id if f.file_type else "unknown",
                    "size": len(f.content)
                })
        return executables

    def can_execute(self, path: str) -> bool:
        """Check if file can be executed"""
        path = self._normalize_path(path)
        if not self._is_file(path):
            return False
        return self.files[path].is_executable


VirtualFileSystem = VirtualFileSystemV2
