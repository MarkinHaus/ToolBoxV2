"""
VFS Extensions V2 - Global Sharing, Search, Execute, Agent Tools
================================================================

Erweiterungen für VirtualFileSystemV2:

1. Global Folder (/global/) - Alle Sessions können lesen/schreiben
2. Session Sharing - Ordner zwischen Sessions teilen
3. Agent Sharing - Ordner zwischen Agents teilen (via ToolBoxV2 Data Dir)
4. Search - VFS durchsuchen (Filename, Content, Regex)
5. Execute - Scripts im Docker ausführen
6. Agent Tools - Alle VFS Funktionen als Tools mit Flags

Architektur:
- GlobalVFS: Singleton für /global/
- VFSSharing: Mixin für Sharing-Funktionen
- VFSSearch: Mixin für Suche
- VFSExecute: Mixin für Script-Ausführung
- register_vfs_tools(): Registriert alle Tools beim Agent

Author: Markin / ToolBoxV2
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2, VFSFile, VFSDirectory, ShadowMount
    from toolboxv2.mods.isaa.base.Agent.docker_vfs import DockerVFS
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSessionV2


# =============================================================================
# CONSTANTS
# =============================================================================

def get_toolboxv2_data_dir() -> Path:
    """Get the ToolBoxV2 data directory for VFS sharing"""
    # Try to get from app
    try:
        from toolboxv2 import get_app
        app = get_app()
        data_dir = Path(app.data_dir) / "Agents" / "VFS"
    except:
        # Fallback
        data_dir = Path.home() / ".toolboxv2" / "data" / "Agents" / "VFS"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


GLOBAL_VFS_PATH = "/global"
SHARED_VFS_PATH = "/shared"  # Für Session/Agent Shares


# =============================================================================
# GLOBAL VFS MANAGER (Singleton)
# =============================================================================

class GlobalVFSManager:
    """
    Singleton Manager für den globalen VFS Ordner.

    Alle Sessions/Agents können auf /global/ zugreifen.
    Daten werden in toolboxv2/.data/{app.id}/Agents/VFS/global/ persistiert.
    """

    _instance: Optional["GlobalVFSManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.data_dir = get_toolboxv2_data_dir() / "global"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Registry: welche VFS Instanzen haben /global/ gemountet
        self._mounted_vfs: dict[str, "VirtualFileSystemV2"] = {}

        self._initialized = True

    @property
    def local_path(self) -> str:
        """Lokaler Pfad für den globalen Ordner"""
        return str(self.data_dir)

    def register_vfs(self, vfs: "VirtualFileSystemV2"):
        """Registriert ein VFS für den globalen Mount"""
        key = f"{vfs.agent_name}:{vfs.session_id}"
        self._mounted_vfs[key] = vfs

    def unregister_vfs(self, vfs: "VirtualFileSystemV2"):
        """Entfernt ein VFS aus der Registry"""
        key = f"{vfs.agent_name}:{vfs.session_id}"
        self._mounted_vfs.pop(key, None)

    def get_all_mounted(self) -> list["VirtualFileSystemV2"]:
        """Gibt alle registrierten VFS Instanzen zurück"""
        return list(self._mounted_vfs.values())

    def write_file(self, relative_path: str, content: str) -> dict:
        """
        Schreibt eine Datei in den globalen Ordner.

        Args:
            relative_path: Pfad relativ zu /global/ (z.B. "data/config.json")
            content: Dateiinhalt

        Returns:
            Result dict
        """
        # Sicherheitscheck
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        file_path = self.data_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.write_text(content, encoding="utf-8")
            return {
                "success": True,
                "path": f"{GLOBAL_VFS_PATH}/{relative_path}",
                "local_path": str(file_path),
                "size": len(content),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, relative_path: str) -> dict:
        """
        Liest eine Datei aus dem globalen Ordner.

        Args:
            relative_path: Pfad relativ zu /global/

        Returns:
            Result dict mit content
        """
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        file_path = self.data_dir / relative_path

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {relative_path}"}

        try:
            content = file_path.read_text(encoding="utf-8")
            return {
                "success": True,
                "path": f"{GLOBAL_VFS_PATH}/{relative_path}",
                "content": content,
                "size": len(content),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self, relative_path: str = "") -> dict:
        """Listet Dateien im globalen Ordner"""
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        dir_path = self.data_dir / relative_path if relative_path else self.data_dir

        if not dir_path.exists():
            return {"success": False, "error": f"Directory not found: {relative_path}"}

        items = []
        for item in dir_path.iterdir():
            items.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
                "path": f"{GLOBAL_VFS_PATH}/{relative_path}/{item.name}".replace("//", "/"),
            })

        return {
            "success": True,
            "path": f"{GLOBAL_VFS_PATH}/{relative_path}".rstrip("/"),
            "items": items,
            "count": len(items),
        }

    def delete_file(self, relative_path: str) -> dict:
        """Löscht eine Datei aus dem globalen Ordner"""
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        file_path = self.data_dir / relative_path

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {relative_path}"}

        try:
            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
            return {"success": True, "deleted": f"{GLOBAL_VFS_PATH}/{relative_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance
_global_vfs = None


def get_global_vfs() -> GlobalVFSManager:
    """Get the global VFS manager singleton"""
    global _global_vfs
    if _global_vfs is None:
        _global_vfs = GlobalVFSManager()
    return _global_vfs


# =============================================================================
# VFS SHARING
# =============================================================================

@dataclass
class ShareInfo:
    """Information über einen geteilten Ordner"""

    share_id: str  # Unique identifier
    source_agent: str
    source_session: str
    source_path: str  # Pfad im Quell-VFS
    target_path: str  # Lokaler Pfad im ToolBoxV2 Data Dir
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    readonly: bool = False
    expires_at: Optional[str] = None

    # Sharing targets
    shared_with_sessions: list[str] = field(default_factory=list)
    shared_with_agents: list[str] = field(default_factory=list)


class VFSSharingManager:
    """
    Manager für VFS Ordner-Sharing zwischen Sessions und Agents.

    Sharing funktioniert über:
    1. Export: Ordner wird in toolboxv2/.data/.../VFS/shares/{share_id}/ kopiert
    2. Import: Andere Sessions/Agents mounten den Share

    Die Daten werden automatisch synchronisiert (via auto_sync Mount).
    """

    def __init__(self):
        self.data_dir = get_toolboxv2_data_dir() / "shares"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.shares_index_file = self.data_dir / "index.json"
        self.shares: dict[str, ShareInfo] = {}

        self._load_shares_index()

    def _load_shares_index(self):
        """Lädt den Shares Index"""
        if self.shares_index_file.exists():
            try:
                data = json.loads(self.shares_index_file.read_text())
                for share_id, info in data.items():
                    self.shares[share_id] = ShareInfo(**info)
            except:
                self.shares = {}

    def _save_shares_index(self):
        """Speichert den Shares Index"""
        data = {
            share_id: {
                "share_id": info.share_id,
                "source_agent": info.source_agent,
                "source_session": info.source_session,
                "source_path": info.source_path,
                "target_path": info.target_path,
                "created_at": info.created_at,
                "readonly": info.readonly,
                "expires_at": info.expires_at,
                "shared_with_sessions": info.shared_with_sessions,
                "shared_with_agents": info.shared_with_agents,
            }
            for share_id, info in self.shares.items()
        }
        self.shares_index_file.write_text(json.dumps(data, indent=2))

    def _generate_share_id(self, agent: str, session: str, path: str) -> str:
        """Generiert eine eindeutige Share ID"""
        content = f"{agent}:{session}:{path}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def create_share(
        self,
        vfs: "VirtualFileSystemV2",
        vfs_path: str,
        readonly: bool = False,
        expires_hours: Optional[float] = None,
    ) -> dict:
        """
        Erstellt einen Share für einen VFS Ordner.

        Der Ordner wird in das ToolBoxV2 Data Directory kopiert
        und kann von anderen Sessions/Agents gemountet werden.

        Args:
            vfs: Quell-VFS
            vfs_path: Pfad im VFS (z.B. "/project/src")
            readonly: Nur Lesezugriff für Empfänger
            expires_hours: Optional Ablaufzeit in Stunden

        Returns:
            Result dict mit share_id
        """
        # Normalisiere Pfad
        vfs_path = vfs_path.rstrip("/")
        if not vfs_path.startswith("/"):
            vfs_path = "/" + vfs_path

        # Prüfe ob Pfad existiert
        if vfs_path not in vfs.directories:
            return {"success": False, "error": f"Directory not found: {vfs_path}"}

        # Generiere Share ID
        share_id = self._generate_share_id(vfs.agent_name, vfs.session_id, vfs_path)

        # Ziel-Pfad im Data Dir
        target_path = self.data_dir / share_id
        target_path.mkdir(parents=True, exist_ok=True)

        # Kopiere Dateien aus dem VFS
        files_copied = 0
        for file_path, vfs_file in vfs.files.items():
            if file_path.startswith(vfs_path + "/") or file_path == vfs_path:
                # Berechne relativen Pfad
                rel_path = file_path[len(vfs_path):].lstrip("/")
                if not rel_path:
                    rel_path = vfs_file.filename

                dest_file = target_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Kopiere Inhalt
                try:
                    # Lade Content wenn nötig (Shadow files)
                    content = vfs_file.content if vfs_file.is_loaded else ""
                    if not vfs_file.is_loaded and vfs_file.local_path:
                        content = Path(vfs_file.local_path).read_text(encoding="utf-8")

                    dest_file.write_text(content, encoding="utf-8")
                    files_copied += 1
                except Exception as e:
                    print(f"[Sharing] Failed to copy {file_path}: {e}")

        # Ablaufzeit berechnen
        expires_at = None
        if expires_hours:
            from datetime import timedelta
            expires_at = (datetime.now() + timedelta(hours=expires_hours)).isoformat()

        # Share Info erstellen
        share_info = ShareInfo(
            share_id=share_id,
            source_agent=vfs.agent_name,
            source_session=vfs.session_id,
            source_path=vfs_path,
            target_path=str(target_path),
            readonly=readonly,
            expires_at=expires_at,
        )

        self.shares[share_id] = share_info
        self._save_shares_index()

        return {
            "success": True,
            "share_id": share_id,
            "source_path": vfs_path,
            "target_path": str(target_path),
            "files_copied": files_copied,
            "readonly": readonly,
            "expires_at": expires_at,
        }

    def grant_access_session(self, share_id: str, session_id: str) -> dict:
        """Gibt einer Session Zugriff auf einen Share"""
        if share_id not in self.shares:
            return {"success": False, "error": f"Share not found: {share_id}"}

        share = self.shares[share_id]
        if session_id not in share.shared_with_sessions:
            share.shared_with_sessions.append(session_id)
            self._save_shares_index()

        return {"success": True, "share_id": share_id, "granted_to": session_id}

    def grant_access_agent(self, share_id: str, agent_name: str) -> dict:
        """Gibt einem Agent Zugriff auf einen Share"""
        if share_id not in self.shares:
            return {"success": False, "error": f"Share not found: {share_id}"}

        share = self.shares[share_id]
        if agent_name not in share.shared_with_agents:
            share.shared_with_agents.append(agent_name)
            self._save_shares_index()

        return {"success": True, "share_id": share_id, "granted_to": agent_name}

    def can_access(self, share_id: str, agent_name: str, session_id: str) -> bool:
        """Prüft ob ein Agent/Session auf einen Share zugreifen kann"""
        if share_id not in self.shares:
            return False

        share = self.shares[share_id]

        # Quelle hat immer Zugriff
        if share.source_agent == agent_name and share.source_session == session_id:
            return True

        # Agent-Level Zugriff
        if agent_name in share.shared_with_agents:
            return True

        # Session-Level Zugriff
        if session_id in share.shared_with_sessions:
            return True

        return False

    def get_share_info(self, share_id: str) -> Optional[ShareInfo]:
        """Gibt Share Info zurück"""
        return self.shares.get(share_id)

    def list_shares_for_agent(self, agent_name: str) -> list[ShareInfo]:
        """Listet alle Shares die ein Agent sehen kann"""
        result = []
        for share in self.shares.values():
            if share.source_agent == agent_name or agent_name in share.shared_with_agents:
                result.append(share)
        return result

    def mount_share(self, vfs: "VirtualFileSystemV2", share_id: str, mount_point: str = None) -> dict:
        """
        Mountet einen Share in ein VFS.

        Args:
            vfs: Ziel-VFS
            share_id: Share ID
            mount_point: Optional Mount-Punkt (default: /shared/{share_id})

        Returns:
            Result dict
        """
        if share_id not in self.shares:
            return {"success": False, "error": f"Share not found: {share_id}"}

        share = self.shares[share_id]

        # Access Check
        if not self.can_access(share_id, vfs.agent_name, vfs.session_id):
            return {"success": False, "error": "Access denied"}

        # Check expiration
        if share.expires_at:
            if datetime.fromisoformat(share.expires_at) < datetime.now():
                return {"success": False, "error": "Share has expired"}

        # Mount Point bestimmen
        if mount_point is None:
            mount_point = f"{SHARED_VFS_PATH}/{share_id}"

        # Mount via VFS mount() Funktion
        result = vfs.mount(
            local_path=share.target_path,
            vfs_path=mount_point,
            readonly=share.readonly,
            auto_sync=not share.readonly,  # Sync nur wenn nicht readonly
        )

        if result.get("success"):
            result["share_id"] = share_id
            result["source"] = f"{share.source_agent}:{share.source_session}:{share.source_path}"

        return result

    def delete_share(self, share_id: str, agent_name: str, session_id: str) -> dict:
        """Löscht einen Share (nur vom Ersteller)"""
        if share_id not in self.shares:
            return {"success": False, "error": f"Share not found: {share_id}"}

        share = self.shares[share_id]

        # Nur Ersteller kann löschen
        if share.source_agent != agent_name or share.source_session != session_id:
            return {"success": False, "error": "Only share owner can delete"}

        # Lösche Dateien
        try:
            shutil.rmtree(share.target_path)
        except:
            pass

        del self.shares[share_id]
        self._save_shares_index()

        return {"success": True, "deleted": share_id}


# Global sharing manager
_sharing_manager = None


def get_sharing_manager() -> VFSSharingManager:
    """Get the VFS sharing manager singleton"""
    global _sharing_manager
    if _sharing_manager is None:
        _sharing_manager = VFSSharingManager()
    return _sharing_manager


# =============================================================================
# VFS SEARCH
# =============================================================================

class SearchMode(Enum):
    """Suchmodus"""
    FILENAME = auto()  # Nur Dateinamen
    CONTENT = auto()  # Nur Dateiinhalt
    BOTH = auto()  # Beides


@dataclass
class SearchResult:
    """Ein Suchergebnis"""
    path: str
    filename: str
    match_type: str  # "filename", "content"
    line_number: Optional[int] = None
    line_content: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    score: float = 1.0


def search_vfs(
    vfs: "VirtualFileSystemV2",
    query: str,
    path: str = "/",
    mode: SearchMode = SearchMode.BOTH,
    case_sensitive: bool = False,
    regex: bool = False,
    max_results: int = 50,
    file_extensions: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
    include_content_context: bool = True,
    context_lines: int = 2,
) -> list[SearchResult]:
    """
    Durchsucht das VFS nach Dateien und Inhalten.

    Args:
        vfs: VirtualFileSystemV2 Instanz
        query: Suchbegriff oder Regex
        path: Start-Pfad für die Suche
        mode: FILENAME, CONTENT, oder BOTH
        case_sensitive: Groß-/Kleinschreibung beachten
        regex: Query als Regex interpretieren
        max_results: Maximale Anzahl Ergebnisse
        file_extensions: Nur diese Extensions (z.B. [".py", ".js"])
        exclude_patterns: Ausschluss-Patterns (fnmatch)
        include_content_context: Kontext-Zeilen bei Content-Matches
        context_lines: Anzahl Kontext-Zeilen

    Returns:
        Liste von SearchResult
    """
    results: list[SearchResult] = []

    # Prepare pattern
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            return []  # Invalid regex
    else:
        if not case_sensitive:
            query = query.lower()

    exclude_patterns = exclude_patterns or []

    def matches_query(text: str) -> bool:
        """Prüft ob Text dem Query entspricht"""
        if regex:
            return bool(pattern.search(text))
        else:
            check_text = text if case_sensitive else text.lower()
            return query in check_text

    def should_include(file_path: str, filename: str) -> bool:
        """Prüft ob Datei durchsucht werden soll"""
        # Extension filter
        if file_extensions:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in file_extensions:
                return False

        # Exclude patterns
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(file_path, pattern):
                return False

        return True

    # Durchsuche Dateien
    for file_path, vfs_file in vfs.files.items():
        if len(results) >= max_results:
            break

        # Pfad-Filter
        if not file_path.startswith(path):
            continue

        # Include/Exclude Filter
        if not should_include(file_path, vfs_file.filename):
            continue

        # Filename Search
        if mode in (SearchMode.FILENAME, SearchMode.BOTH):
            if matches_query(vfs_file.filename):
                results.append(SearchResult(
                    path=file_path,
                    filename=vfs_file.filename,
                    match_type="filename",
                    score=1.0,
                ))
                if len(results) >= max_results:
                    break

        # Content Search
        if mode in (SearchMode.CONTENT, SearchMode.BOTH):
            try:
                # Lade Content wenn nötig
                if vfs_file.is_loaded:
                    content = vfs_file.content
                elif vfs_file.local_path and os.path.exists(vfs_file.local_path):
                    content = Path(vfs_file.local_path).read_text(encoding="utf-8", errors="ignore")
                else:
                    continue

                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    if matches_query(line):
                        # Context
                        context_before = None
                        context_after = None
                        if include_content_context:
                            start = max(0, line_num - 1 - context_lines)
                            end = min(len(lines), line_num + context_lines)
                            context_before = "\n".join(lines[start:line_num - 1]) if start < line_num - 1 else None
                            context_after = "\n".join(lines[line_num:end]) if line_num < end else None

                        results.append(SearchResult(
                            path=file_path,
                            filename=vfs_file.filename,
                            match_type="content",
                            line_number=line_num,
                            line_content=line[:200],  # Truncate long lines
                            context_before=context_before,
                            context_after=context_after,
                            score=0.8,
                        ))

                        if len(results) >= max_results:
                            break
            except Exception:
                continue

        if len(results) >= max_results:
            break

    return results


def find_files(
    vfs: "VirtualFileSystemV2",
    pattern: str,
    path: str = "/",
) -> list[str]:
    """
    Findet Dateien per Glob-Pattern.

    Args:
        vfs: VirtualFileSystemV2 Instanz
        pattern: Glob-Pattern (z.B. "*.py", "test_*.js")
        path: Start-Pfad

    Returns:
        Liste von Dateipfaden
    """
    results = []

    for file_path in vfs.files.keys():
        if not file_path.startswith(path):
            continue

        filename = os.path.basename(file_path)
        if fnmatch.fnmatch(filename, pattern):
            results.append(file_path)

    return results


def grep_vfs(
    vfs: "VirtualFileSystemV2",
    pattern: str,
    file_pattern: str = "*",
    path: str = "/",
    context_lines: int = 0,
) -> list[dict]:
    """
    Grep-ähnliche Suche im VFS.

    Args:
        vfs: VirtualFileSystemV2 Instanz
        pattern: Regex Pattern für Content
        file_pattern: Glob für Dateinamen
        path: Start-Pfad
        context_lines: Anzahl Kontext-Zeilen

    Returns:
        Liste von Match-Dicts
    """
    results = []

    try:
        regex = re.compile(pattern)
    except re.error:
        return []

    for file_path, vfs_file in vfs.files.items():
        if not file_path.startswith(path):
            continue

        if not fnmatch.fnmatch(vfs_file.filename, file_pattern):
            continue

        try:
            if vfs_file.is_loaded:
                content = vfs_file.content
            elif vfs_file.local_path and os.path.exists(vfs_file.local_path):
                content = Path(vfs_file.local_path).read_text(encoding="utf-8", errors="ignore")
            else:
                continue

            lines = content.split("\n")
            for line_num, line in enumerate(lines, 1):
                if regex.search(line):
                    result = {
                        "file": file_path,
                        "line": line_num,
                        "match": line[:200],
                    }

                    if context_lines > 0:
                        start = max(0, line_num - 1 - context_lines)
                        end = min(len(lines), line_num + context_lines)
                        result["context"] = lines[start:end]

                    results.append(result)
        except Exception:
            continue

    return results


# =============================================================================
# VFS EXECUTE
# =============================================================================

async def execute_file(
    vfs: "VirtualFileSystemV2",
    docker_vfs: "DockerVFS",
    file_path: str,
    args: list[str] = None,
    timeout: int = 300,
    sync_after: bool = True,
) -> dict:
    """
    Führt eine Datei im Docker Container aus.

    Args:
        vfs: VirtualFileSystemV2 Instanz
        docker_vfs: DockerVFS Instanz
        file_path: Pfad zur Datei im VFS
        args: Argumente für das Script
        timeout: Timeout in Sekunden
        sync_after: VFS nach Ausführung synchronisieren

    Returns:
        Result dict mit stdout, stderr, exit_code
    """
    if file_path not in vfs.files:
        return {"success": False, "error": f"File not found: {file_path}"}

    vfs_file = vfs.files[file_path]

    if not vfs_file.is_executable:
        return {"success": False, "error": f"File is not executable: {file_path}"}

    # Bestimme Interpreter
    file_type = vfs_file.file_type
    interpreter = None

    if file_type:
        if file_type.language_id == "python":
            interpreter = "python"
        elif file_type.language_id in ("javascript", "javascriptreact"):
            interpreter = "node"
        elif file_type.language_id in ("typescript", "typescriptreact"):
            interpreter = "npx ts-node"
        elif file_type.language_id == "shellscript":
            interpreter = "bash"
        elif file_type.language_id == "rust":
            # Rust muss kompiliert werden
            return {"success": False, "error": "Rust files need to be compiled first"}

    if not interpreter:
        return {"success": False,
                "error": f"No interpreter for file type: {file_type.language_id if file_type else 'unknown'}"}

    # Build command
    # Der file_path im VFS entspricht dem Pfad im Docker Container unter /workspace
    workspace_path = file_path.lstrip("/")  # /project/main.py -> project/main.py

    args_str = " ".join(args) if args else ""
    command = f"{interpreter} {workspace_path} {args_str}".strip()

    # Execute
    result = await docker_vfs.run_command(
        command=command,
        timeout=timeout,
        sync_before=True,
        sync_after=sync_after,
    )

    result["file"] = file_path
    result["command"] = command

    return result


async def execute_code(
    docker_vfs: "DockerVFS",
    code: str,
    language: str = "python",
    timeout: int = 60,
) -> dict:
    """
    Führt Code-Snippet direkt aus (ohne Datei zu erstellen).

    Args:
        docker_vfs: DockerVFS Instanz
        code: Code zum Ausführen
        language: Programmiersprache
        timeout: Timeout in Sekunden

    Returns:
        Result dict
    """
    if language == "python":
        # Escape für Shell
        escaped_code = code.replace("'", "'\"'\"'")
        command = f"python -c '{escaped_code}'"
    elif language in ("javascript", "js"):
        escaped_code = code.replace("'", "'\"'\"'")
        command = f"node -e '{escaped_code}'"
    elif language in ("bash", "sh"):
        escaped_code = code.replace("'", "'\"'\"'")
        command = f"bash -c '{escaped_code}'"
    else:
        return {"success": False, "error": f"Unsupported language: {language}"}

    return await docker_vfs.run_command(
        command=command,
        timeout=timeout,
        sync_before=False,
        sync_after=False,
    )


# =============================================================================
# AGENT TOOLS REGISTRATION
# =============================================================================

def register_vfs_tools(
    agent: "FlowAgent",
    enable_global: bool = True,
    enable_sharing: bool = True,
    enable_search: bool = True,
    enable_execute: bool = True,
) -> dict:
    """
    Registriert alle VFS Tools beim Agent.

    Args:
        agent: FlowAgent Instanz
        enable_global: Global VFS Tools aktivieren
        enable_sharing: Sharing Tools aktivieren
        enable_search: Search Tools aktivieren
        enable_execute: Execute Tools aktivieren

    Returns:
        Dict mit registrierten Tools
    """
    registered = []

    # =========================================================================
    # CORE VFS TOOLS
    # =========================================================================

    async def vfs_ls(path: str = "/", recursive: bool = False, max_depth: int = 2) -> str:
        """
        List files and directories in the VFS.

        Args:
            path: Directory path to list (default: root)
            recursive: Show subdirectories recursively
            max_depth: Maximum depth for recursive listing

        Returns:
            Formatted directory listing
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        if recursive:
            return vfs.get_tree(path, max_depth=max_depth)
        else:
            return vfs.ls(path)

    async def vfs_read(path: str, start_line: int = None, end_line: int = None) -> str:
        """
        Read a file from VFS. Opens the file if not already open.

        Args:
            path: File path in VFS
            start_line: Start line (1-based, optional)
            end_line: End line (optional, -1 for end of file)

        Returns:
            File content or windowed view
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        # Open file if needed
        if path not in vfs.files or vfs.files[path].state != "open":
            result = await vfs.open(path)
            if not result.get("success", True) and "error" in result:
                return f"Error: {result['error']}"

        # Window if specified
        if start_line is not None:
            result = vfs.window(path, start_line, end_line or -1)
            if isinstance(result, dict) and "error" in result:
                return f"Error: {result['error']}"

        return vfs.get_file_content(path)

    async def vfs_write(path: str, content: str, create_dirs: bool = True) -> str:
        """
        Write content to a file in VFS.

        Args:
            path: File path in VFS
            content: Content to write
            create_dirs: Create parent directories if they don't exist

        Returns:
            Result message
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        # Create directories if needed
        if create_dirs:
            dir_path = os.path.dirname(path)
            if dir_path and dir_path != "/":
                vfs.mkdir(dir_path, parents=True)

        result = vfs.create(path, content)
        if result.get("success"):
            return f"Written {len(content)} chars to {path}"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    async def vfs_edit(
        path: str,
        old_str: str,
        new_str: str,
    ) -> str:
        """
        Replace a string in a file. The old_str must appear exactly once.

        Args:
            path: File path in VFS
            old_str: String to replace (must be unique in file)
            new_str: Replacement string

        Returns:
            Result message
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        result = vfs.str_replace(path, old_str, new_str)
        if result.get("success"):
            return f"Replaced in {path}"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    async def vfs_delete(path: str) -> str:
        """
        Delete a file or directory from VFS.

        Args:
            path: Path to delete

        Returns:
            Result message
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        if path in vfs.directories:
            result = vfs.rmdir(path, recursive=True)
        else:
            result = vfs.delete(path)

        if result.get("success"):
            return f"Deleted {path}"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    async def vfs_mkdir(path: str, parents: bool = True) -> str:
        """
        Create a directory in VFS.

        Args:
            path: Directory path
            parents: Create parent directories if needed

        Returns:
            Result message
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        result = vfs.mkdir(path, parents=parents)
        if result.get("success"):
            return f"Created directory {path}"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    async def vfs_mv(src: str, dest: str) -> str:
        """
        Move/rename a file or directory.

        Args:
            src: Source path
            dest: Destination path

        Returns:
            Result message
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        result = vfs.mv(src, dest)
        if result.get("success"):
            return f"Moved {src} to {dest}"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    async def vfs_cp(src: str, dest: str) -> str:
        """
        Copy a file or directory.

        Args:
            src: Source path
            dest: Destination path

        Returns:
            Result message
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        result = vfs.cp(src, dest)
        if result.get("success"):
            return f"Copied {src} to {dest}"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    async def vfs_mount(
        local_path: str,
        vfs_path: str = "/project",
        readonly: bool = False,
        auto_sync: bool = True,
    ) -> str:
        """
        Mount a local directory into VFS.

        Args:
            local_path: Local filesystem path
            vfs_path: Mount point in VFS
            readonly: Mount as read-only
            auto_sync: Automatically sync changes to disk

        Returns:
            Result message with statistics
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        result = vfs.mount(local_path, vfs_path, readonly=readonly, auto_sync=auto_sync)
        if result.get("success"):
            return f"Mounted {local_path} at {vfs_path}: {result.get('files_indexed', 0)} files, {result.get('dirs_indexed', 0)} dirs"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    async def vfs_unmount(vfs_path: str, save_changes: bool = True) -> str:
        """
        Unmount a directory from VFS.

        Args:
            vfs_path: Mount point to unmount
            save_changes: Save pending changes before unmounting

        Returns:
            Result message
        """
        session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
        vfs = session.vfs

        result = vfs.unmount(vfs_path, save_changes=save_changes)
        if result.get("success"):
            return f"Unmounted {vfs_path}"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    # Register core tools
    agent.add_tool(vfs_ls, "vfs_ls", "List files and directories in VFS", ["vfs", "filesystem"])
    agent.add_tool(vfs_read, "vfs_read", "Read file content from VFS", ["vfs", "filesystem"])
    agent.add_tool(vfs_write, "vfs_write", "Write content to file in VFS", ["vfs", "filesystem"])
    agent.add_tool(vfs_edit, "vfs_edit", "Replace string in file (str_replace)", ["vfs", "filesystem"])
    agent.add_tool(vfs_delete, "vfs_delete", "Delete file or directory from VFS", ["vfs", "filesystem"])
    agent.add_tool(vfs_mkdir, "vfs_mkdir", "Create directory in VFS", ["vfs", "filesystem"])
    agent.add_tool(vfs_mv, "vfs_mv", "Move/rename file or directory", ["vfs", "filesystem"])
    agent.add_tool(vfs_cp, "vfs_cp", "Copy file or directory", ["vfs", "filesystem"])
    agent.add_tool(vfs_mount, "vfs_mount", "Mount local directory into VFS", ["vfs", "filesystem"])
    agent.add_tool(vfs_unmount, "vfs_unmount", "Unmount directory from VFS", ["vfs", "filesystem"])

    registered.extend(
        ["vfs_ls", "vfs_read", "vfs_write", "vfs_edit", "vfs_delete", "vfs_mkdir", "vfs_mv", "vfs_cp", "vfs_mount",
         "vfs_unmount"])

    # =========================================================================
    # GLOBAL VFS TOOLS
    # =========================================================================

    if enable_global:
        async def global_read(path: str) -> str:
            """
            Read a file from the global shared directory (/global/).
            All sessions and agents can read from this directory.

            Args:
                path: Path relative to /global/ (e.g., "data/config.json")

            Returns:
                File content
            """
            global_vfs = get_global_vfs()
            result = global_vfs.read_file(path)
            if result.get("success"):
                return result["content"]
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        async def global_write(path: str, content: str) -> str:
            """
            Write a file to the global shared directory (/global/).
            All sessions and agents can write to this directory.

            Args:
                path: Path relative to /global/ (e.g., "data/config.json")
                content: Content to write

            Returns:
                Result message
            """
            global_vfs = get_global_vfs()
            result = global_vfs.write_file(path, content)
            if result.get("success"):
                return f"Written to {result['path']}"
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        async def global_ls(path: str = "") -> str:
            """
            List files in the global shared directory.

            Args:
                path: Path relative to /global/ (default: root)

            Returns:
                Directory listing
            """
            global_vfs = get_global_vfs()
            result = global_vfs.list_files(path)
            if result.get("success"):
                items = result["items"]
                lines = [f"{result['path']}:"]
                for item in items:
                    icon = "📁" if item["is_dir"] else "📄"
                    lines.append(f"  {icon} {item['name']}")
                return "\n".join(lines)
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        async def global_delete(path: str) -> str:
            """
            Delete a file from the global shared directory.

            Args:
                path: Path relative to /global/

            Returns:
                Result message
            """
            global_vfs = get_global_vfs()
            result = global_vfs.delete_file(path)
            if result.get("success"):
                return f"Deleted {result['deleted']}"
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        agent.add_tool(global_read, "global_read", "Read file from global shared directory", ["vfs", "global"])
        agent.add_tool(global_write, "global_write", "Write file to global shared directory", ["vfs", "global"])
        agent.add_tool(global_ls, "global_ls", "List files in global shared directory", ["vfs", "global"])
        agent.add_tool(global_delete, "global_delete", "Delete file from global shared directory", ["vfs", "global"])

        registered.extend(["global_read", "global_write", "global_ls", "global_delete"])

    # =========================================================================
    # SHARING TOOLS
    # =========================================================================

    if enable_sharing:
        async def share_create(
            vfs_path: str,
            readonly: bool = False,
            expires_hours: float = None,
        ) -> str:
            """
            Create a share for a VFS directory.
            Other sessions or agents can then mount this share.

            Args:
                vfs_path: Path in VFS to share (e.g., "/project/src")
                readonly: Make share read-only for others
                expires_hours: Optional expiration time in hours

            Returns:
                Share ID that can be used to mount the share
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
            vfs = session.vfs

            sharing = get_sharing_manager()
            result = sharing.create_share(vfs, vfs_path, readonly, expires_hours)

            if result.get("success"):
                return f"Share created: {result['share_id']}\nFiles: {result['files_copied']}\nMount with: share_mount('{result['share_id']}')"
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        async def share_mount(share_id: str, mount_point: str = None) -> str:
            """
            Mount a shared directory from another session or agent.

            Args:
                share_id: The share ID from share_create
                mount_point: Optional custom mount point (default: /shared/{share_id})

            Returns:
                Result message
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
            vfs = session.vfs

            sharing = get_sharing_manager()
            result = sharing.mount_share(vfs, share_id, mount_point)

            if result.get("success"):
                return f"Mounted share at {result['mount_point']}"
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        async def share_grant(share_id: str, target: str, target_type: str = "agent") -> str:
            """
            Grant access to a share.

            Args:
                share_id: The share ID
                target: Agent name or session ID to grant access
                target_type: "agent" or "session"

            Returns:
                Result message
            """
            sharing = get_sharing_manager()

            if target_type == "agent":
                result = sharing.grant_access_agent(share_id, target)
            else:
                result = sharing.grant_access_session(share_id, target)

            if result.get("success"):
                return f"Access granted to {target}"
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        async def share_list() -> str:
            """
            List all available shares for the current agent.

            Returns:
                List of shares
            """
            sharing = get_sharing_manager()
            shares = sharing.list_shares_for_agent(agent.name)

            if not shares:
                return "No shares available"

            lines = ["Available shares:"]
            for share in shares:
                owner = f"{share.source_agent}:{share.source_session}"
                lines.append(f"  {share.share_id}: {share.source_path} (from {owner})")

            return "\n".join(lines)

        agent.add_tool(share_create, "share_create", "Create a share for a VFS directory", ["vfs", "sharing"])
        agent.add_tool(share_mount, "share_mount", "Mount a shared directory", ["vfs", "sharing"])
        agent.add_tool(share_grant, "share_grant", "Grant access to a share", ["vfs", "sharing"])
        agent.add_tool(share_list, "share_list", "List available shares", ["vfs", "sharing"])

        registered.extend(["share_create", "share_mount", "share_grant", "share_list"])

    # =========================================================================
    # SEARCH TOOLS
    # =========================================================================

    if enable_search:
        async def vfs_search(
            query: str,
            path: str = "/",
            mode: str = "both",
            case_sensitive: bool = False,
            regex: bool = False,
            max_results: int = 20,
        ) -> str:
            """
            Search for files and content in VFS.

            Args:
                query: Search term or regex pattern
                path: Start path for search
                mode: "filename", "content", or "both"
                case_sensitive: Case-sensitive search
                regex: Interpret query as regex
                max_results: Maximum number of results

            Returns:
                Search results
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
            vfs = session.vfs

            mode_map = {
                "filename": SearchMode.FILENAME,
                "content": SearchMode.CONTENT,
                "both": SearchMode.BOTH,
            }

            results = search_vfs(
                vfs, query, path,
                mode=mode_map.get(mode, SearchMode.BOTH),
                case_sensitive=case_sensitive,
                regex=regex,
                max_results=max_results,
            )

            if not results:
                return f"No results for '{query}'"

            lines = [f"Found {len(results)} results for '{query}':"]
            for r in results:
                if r.match_type == "filename":
                    lines.append(f"  📄 {r.path}")
                else:
                    lines.append(f"  📄 {r.path}:{r.line_number}: {r.line_content}")

            return "\n".join(lines)

        async def vfs_find(pattern: str, path: str = "/") -> str:
            """
            Find files by glob pattern (like Unix find).

            Args:
                pattern: Glob pattern (e.g., "*.py", "test_*.js")
                path: Start path

            Returns:
                List of matching files
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
            vfs = session.vfs

            results = find_files(vfs, pattern, path)

            if not results:
                return f"No files matching '{pattern}'"

            return f"Found {len(results)} files:\n" + "\n".join(f"  {f}" for f in results)

        async def vfs_grep(
            pattern: str,
            file_pattern: str = "*",
            path: str = "/",
            context: int = 0,
        ) -> str:
            """
            Search file contents with regex (like Unix grep).

            Args:
                pattern: Regex pattern to search for
                file_pattern: Glob pattern for files to search
                path: Start path
                context: Number of context lines

            Returns:
                Grep results
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")
            vfs = session.vfs

            results = grep_vfs(vfs, pattern, file_pattern, path, context)

            if not results:
                return f"No matches for '{pattern}'"

            lines = [f"Found {len(results)} matches:"]
            for r in results:
                lines.append(f"  {r['file']}:{r['line']}: {r['match']}")

            return "\n".join(lines)

        agent.add_tool(vfs_search, "vfs_search", "Search files and content in VFS", ["vfs", "search"])
        agent.add_tool(vfs_find, "vfs_find", "Find files by glob pattern", ["vfs", "search"])
        agent.add_tool(vfs_grep, "vfs_grep", "Grep file contents with regex", ["vfs", "search"])

        registered.extend(["vfs_search", "vfs_find", "vfs_grep"])

    # =========================================================================
    # EXECUTE TOOLS
    # =========================================================================

    if enable_execute:
        async def vfs_execute(
            file_path: str,
            args: str = "",
            timeout: int = 300,
        ) -> str:
            """
            Execute a script file in Docker container.
            Supports Python, JavaScript, TypeScript, Shell scripts.

            Args:
                file_path: Path to executable file in VFS
                args: Command line arguments (space-separated)
                timeout: Execution timeout in seconds

            Returns:
                Execution result with stdout/stderr
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")

            if not session._docker_vfs:
                return "Error: Docker not enabled for this session"

            args_list = args.split() if args else []
            result = await execute_file(
                session.vfs,
                session._docker_vfs,
                file_path,
                args_list,
                timeout,
            )

            if result.get("success"):
                output = f"Exit code: {result.get('exit_code', 0)}\n"
                if result.get("stdout"):
                    output += f"--- stdout ---\n{result['stdout']}\n"
                if result.get("stderr"):
                    output += f"--- stderr ---\n{result['stderr']}\n"
                return output
            else:
                return f"Error: {result.get('error', 'Execution failed')}"

        async def vfs_run_code(
            code: str,
            language: str = "python",
            timeout: int = 60,
        ) -> str:
            """
            Execute a code snippet directly (without creating a file).

            Args:
                code: Code to execute
                language: "python", "javascript", "bash"
                timeout: Execution timeout in seconds

            Returns:
                Execution result
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")

            if not session._docker_vfs:
                return "Error: Docker not enabled for this session"

            result = await execute_code(
                session._docker_vfs,
                code,
                language,
                timeout,
            )

            if result.get("success"):
                output = f"Exit code: {result.get('exit_code', 0)}\n"
                if result.get("stdout"):
                    output += f"{result['stdout']}"
                if result.get("stderr"):
                    output += f"\n[stderr] {result['stderr']}"
                return output
            else:
                return f"Error: {result.get('error', 'Execution failed')}"

        async def docker_shell(command: str, timeout: int = 300) -> str:
            """
            Run a shell command in the Docker container.
            VFS files are synced before and after execution.

            Args:
                command: Shell command to execute
                timeout: Timeout in seconds

            Returns:
                Command output
            """
            session = await agent.session_manager.get_or_create(agent._current_session_id or "default")

            if not session._docker_vfs:
                return "Error: Docker not enabled for this session"

            result = await session._docker_vfs.run_command(
                command=command,
                timeout=timeout,
            )

            if result.get("success"):
                output = result.get("stdout", "")
                if result.get("stderr"):
                    output += f"\n[stderr] {result['stderr']}"
                return output or "(no output)"
            else:
                return f"Error: {result.get('error', 'Command failed')}\n{result.get('stderr', '')}"

        agent.add_tool(vfs_execute, "vfs_execute", "Execute script file in Docker", ["vfs", "execute", "docker"])
        agent.add_tool(vfs_run_code, "vfs_run_code", "Execute code snippet in Docker", ["vfs", "execute", "docker"])
        agent.add_tool(docker_shell, "docker_shell", "Run shell command in Docker", ["vfs", "execute", "docker"])

        registered.extend(["vfs_execute", "vfs_run_code", "docker_shell"])

    return {
        "registered": registered,
        "count": len(registered),
        "categories": {
            "core": ["vfs_ls", "vfs_read", "vfs_write", "vfs_edit", "vfs_delete", "vfs_mkdir", "vfs_mv", "vfs_cp",
                     "vfs_mount", "vfs_unmount"],
            "global": ["global_read", "global_write", "global_ls", "global_delete"] if enable_global else [],
            "sharing": ["share_create", "share_mount", "share_grant", "share_list"] if enable_sharing else [],
            "search": ["vfs_search", "vfs_find", "vfs_grep"] if enable_search else [],
            "execute": ["vfs_execute", "vfs_run_code", "docker_shell"] if enable_execute else [],
        },
    }


# =============================================================================
# SESSION INITIALIZATION HELPER
# =============================================================================

async def init_session_with_vfs_features(
    session: "AgentSessionV2",
    mount_global: bool = True,
    global_readonly: bool = False,
) -> dict:
    """
    Initialisiert eine Session mit allen VFS Features.

    - Mountet /global/ automatisch
    - Registriert GlobalVFS
    - Lädt verfügbare Shares

    Args:
        session: AgentSessionV2 Instanz
        mount_global: /global/ automatisch mounten
        global_readonly: /global/ nur lesend mounten

    Returns:
        Init-Status dict
    """
    result = {
        "session_id": session.session_id,
        "features": [],
    }

    # Mount global directory
    if mount_global:
        global_vfs = get_global_vfs()
        global_vfs.register_vfs(session.vfs)

        mount_result = session.vfs.mount(
            local_path=global_vfs.local_path,
            vfs_path=GLOBAL_VFS_PATH,
            readonly=global_readonly,
            auto_sync=not global_readonly,
        )

        if mount_result.get("success"):
            result["features"].append("global_vfs")
            result["global_mount"] = mount_result

    # Load available shares
    sharing = get_sharing_manager()
    available_shares = sharing.list_shares_for_agent(session.agent_name)
    result["available_shares"] = len(available_shares)
    result["features"].append("sharing")

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Global VFS
    "GlobalVFSManager",
    "get_global_vfs",
    "GLOBAL_VFS_PATH",

    # Sharing
    "VFSSharingManager",
    "get_sharing_manager",
    "ShareInfo",
    "SHARED_VFS_PATH",

    # Search
    "SearchMode",
    "SearchResult",
    "search_vfs",
    "find_files",
    "grep_vfs",

    # Execute
    "execute_file",
    "execute_code",

    # Agent Tools
    "register_vfs_tools",

    # Session Init
    "init_session_with_vfs_features",
]
