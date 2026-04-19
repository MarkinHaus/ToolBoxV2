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
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSessionV2
    from toolboxv2.mods.isaa.base.Agent.docker_vfs import DockerVFS
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
        ShadowMount,
        VFSDirectory,
        VFSFile,
        VirtualFileSystemV2,
    )


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

        # EBENE 3: In-Memory Shared Store für /global/
        # relative_path (z.B. "data/config.json") → {content, mtime, author, version}
        # Alle VFS-Instanzen die /global/ gemountet haben, sehen denselben Dict.
        # Verhindert Disk-Roundtrip für Multi-Agent-Reads im selben Prozess.
        self._shared_store: dict[str, dict] = {}
        self._store_lock = threading.RLock()

        # Monotone Version für Change-Detection ohne mtime-Granularität
        self._version_counter = 0

        # Subscribers für reaktive Benachrichtigung (optional)
        # key: relative_path or "*" for all | value: list of callables(event_dict)
        self._subscribers: dict[str, list] = {}

        # Ebene 3b: Registry für zusätzliche Shared-Mounts (nicht /global/)
        # local_path → {"mount_key": str, "hydrated": bool}
        self._extra_mounts: dict[str, dict] = {}

        # Initial-Load: Disk-Content in Shared-Store ziehen
        self._hydrate_from_disk()

        self._initialized = True

    # =====================================================================
    # EBENE 3: Shared-Store Internals
    # =====================================================================

    def _hydrate_from_disk(self) -> None:
        """Beim Start: vorhandene Disk-Dateien in den Shared-Store laden."""
        if not self.data_dir.exists():
            return
        with self._store_lock:
            for root, _, files in os.walk(self.data_dir):
                for filename in files:
                    full = os.path.join(root, filename)
                    try:
                        rel = os.path.relpath(full, self.data_dir).replace(os.sep, "/")
                        stat = os.stat(full)
                        content = Path(full).read_text(encoding="utf-8", errors="replace")
                        self._version_counter += 1
                        self._shared_store[rel] = {
                            "content": content,
                            "mtime": stat.st_mtime,
                            "author": None,
                            "version": self._version_counter,
                        }
                    except (OSError, UnicodeError):
                        continue

    def _bump_version(self) -> int:
        with self._store_lock:
            self._version_counter += 1
            return self._version_counter

    def _notify(self, event: dict) -> None:
        """Broadcast an alle passenden Subscribers. Exceptions werden geschluckt."""
        path = event.get("path", "")
        targets = []
        with self._store_lock:
            if path in self._subscribers:
                targets.extend(self._subscribers[path])
            if "*" in self._subscribers:
                targets.extend(self._subscribers["*"])
        for cb in targets:
            try:
                cb(event)
            except Exception:
                pass

    def subscribe(self, path: str, callback) -> None:
        """
        Registriere Callback für Shared-Store-Events.

        path: relativer /global/-Pfad (z.B. "tasks.md") oder "*" für alle
        callback: callable(event_dict) — event hat keys: type, path, author, version
        """
        with self._store_lock:
            self._subscribers.setdefault(path, []).append(callback)

    def unsubscribe(self, path: str, callback) -> None:
        with self._store_lock:
            if path in self._subscribers:
                try:
                    self._subscribers[path].remove(callback)
                except ValueError:
                    pass
                if not self._subscribers[path]:
                    del self._subscribers[path]

    def get_shared(self, relative_path: str) -> dict | None:
        """Thread-safe getter auf den Shared-Store. Gibt Copy zurück."""
        with self._store_lock:
            entry = self._shared_store.get(relative_path)
            return dict(entry) if entry else None

    def has_shared(self, relative_path: str) -> bool:
        with self._store_lock:
            return relative_path in self._shared_store

    def invalidate_vfs_caches(self, relative_path: str) -> None:
        """
        Markiere die entsprechende Datei in allen gemounteten VFS-Instanzen
        als invalid (Content = None). Nächstes read() zieht aus Shared-Store.
        """
        vfs_file_path = f"{GLOBAL_VFS_PATH}/{relative_path}"
        for vfs in list(self._mounted_vfs.values()):
            f = vfs.files.get(vfs_file_path)
            if f is None:
                continue
            if getattr(f, "is_dirty", False):
                continue  # Agent-Arbeit nicht überschreiben
            try:
                f._content = None
                from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType
                f.backing_type = FileBackingType.SHADOW
                vfs._dirty = True
            except Exception:
                pass

    # =====================================================================
    # EBENE 3b: Generic Shared-Mount API (für CoderAgent Worktrees etc.)
    # =====================================================================

    def register_shared_mount(
        self,
        local_path: str,
        mount_key: str | None = None,
        hydrate: bool = True,
    ) -> str:
        """
        Registriert einen lokalen Pfad als Shared-Mount.

        Mehrere VFS-Instanzen die denselben local_path mounten, teilen
        dann den Content-Cache im Shared-Store — Reads/Writes gehen durch
        den gemeinsamen RAM-Store, kein Disk-Roundtrip im selben Prozess.

        Args:
            local_path: Absoluter lokaler Pfad
            mount_key: Identifier (default: hash des Pfads)
            hydrate: Wenn True, werden vorhandene Disk-Dateien sofort geladen

        Returns:
            mount_key (zum Referenzieren in get/write/read Operations)
        """
        abs_path = str(Path(local_path).resolve())
        if mount_key is None:
            mount_key = "mnt-" + hashlib.sha256(abs_path.encode()).hexdigest()[:12]

        with self._store_lock:
            if abs_path in self._extra_mounts:
                return self._extra_mounts[abs_path]["mount_key"]
            self._extra_mounts[abs_path] = {
                "mount_key": mount_key,
                "hydrated": False,
            }

        if hydrate:
            self._hydrate_extra_mount(abs_path)

        return mount_key

    def unregister_shared_mount(self, local_path: str) -> None:
        """Entferne Shared-Mount + alle zugehörigen Store-Einträge."""
        abs_path = str(Path(local_path).resolve())
        with self._store_lock:
            info = self._extra_mounts.pop(abs_path, None)
            if info is None:
                return
            prefix = f"{info['mount_key']}::"
            keys_to_drop = [k for k in self._shared_store if k.startswith(prefix)]
            for k in keys_to_drop:
                del self._shared_store[k]

    def get_mount_key_for(self, local_path: str) -> str | None:
        """Finde mount_key für einen lokalen Pfad. None wenn nicht registriert."""
        abs_path = str(Path(local_path).resolve())
        with self._store_lock:
            info = self._extra_mounts.get(abs_path)
            return info["mount_key"] if info else None

    def _hydrate_extra_mount(self, abs_path: str) -> None:
        """Lade vorhandene Disk-Dateien eines Extra-Mounts in den Store."""
        info = self._extra_mounts.get(abs_path)
        if not info:
            return
        mount_key = info["mount_key"]
        base = Path(abs_path)
        if not base.exists():
            return

        exclude = {".git", "__pycache__", "node_modules", ".venv", "venv",
                   ".pytest_cache", ".mypy_cache"}

        with self._store_lock:
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in exclude]
                for filename in files:
                    full = Path(root) / filename
                    try:
                        rel = str(full.relative_to(base)).replace(os.sep, "/")
                        stat = full.stat()
                        if stat.st_size > 1024 * 1024 * 5:  # Skip >5MB
                            continue
                        content = full.read_text(encoding="utf-8", errors="replace")
                        self._version_counter += 1
                        self._shared_store[f"{mount_key}::{rel}"] = {
                            "content": content,
                            "mtime": stat.st_mtime,
                            "author": None,
                            "version": self._version_counter,
                        }
                    except (OSError, UnicodeError):
                        continue
            info["hydrated"] = True

    def shared_write(
        self,
        mount_key: str,
        relative_path: str,
        content: str,
        local_base: str,
        author: str | None = None,
    ) -> dict:
        """
        Write in einen generischen Shared-Mount.

        Analog zu write_file() aber für nicht-global-Mounts.
        Schreibt auf Disk UND in den Store, broadcasted, invalidiert Caches.
        """
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        full_path = Path(local_base) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            full_path.write_text(content, encoding="utf-8")
            stat = full_path.stat()

            store_key = f"{mount_key}::{relative_path}"
            with self._store_lock:
                self._version_counter += 1
                version = self._version_counter
                self._shared_store[store_key] = {
                    "content": content,
                    "mtime": stat.st_mtime,
                    "author": author,
                    "version": version,
                }

            # Invalidate VFS caches für alle Instanzen die diesen Mount haben
            self._invalidate_extra_mount_caches(mount_key, relative_path)

            # Broadcast
            self._notify({
                "type": "write",
                "mount_key": mount_key,
                "path": relative_path,
                "author": author,
                "version": version,
            })

            return {"success": True, "version": version, "path": relative_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def shared_read(self, mount_key: str, relative_path: str) -> dict | None:
        """Read aus generischem Shared-Mount. None wenn kein Store-Hit."""
        store_key = f"{mount_key}::{relative_path}"
        with self._store_lock:
            entry = self._shared_store.get(store_key)
            return dict(entry) if entry else None

    def shared_delete(
        self,
        mount_key: str,
        relative_path: str,
        local_base: str,
    ) -> dict:
        """Delete aus generischem Shared-Mount."""
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        full_path = Path(local_base) / relative_path

        try:
            if full_path.exists():
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                else:
                    full_path.unlink()

            store_key = f"{mount_key}::{relative_path}"
            with self._store_lock:
                self._shared_store.pop(store_key, None)
                # Sub-pfade bei Dir-Delete
                prefix = f"{mount_key}::{relative_path}/"
                to_drop = [k for k in self._shared_store if k.startswith(prefix)]
                for k in to_drop:
                    del self._shared_store[k]
                self._version_counter += 1
                version = self._version_counter

            self._invalidate_extra_mount_caches(mount_key, relative_path, is_delete=True)

            self._notify({
                "type": "delete",
                "mount_key": mount_key,
                "path": relative_path,
                "version": version,
            })

            return {"success": True, "deleted": relative_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _invalidate_extra_mount_caches(
        self,
        mount_key: str,
        relative_path: str,
        is_delete: bool = False,
    ) -> None:
        """
        Alle VFS-Instanzen die einen Mount mit passendem local_path haben,
        bekommen den entsprechenden File-Cache invalidiert bzw. entfernt.
        """
        # Finde abs_path zu mount_key
        target_abs = None
        with self._store_lock:
            for abs_path, info in self._extra_mounts.items():
                if info["mount_key"] == mount_key:
                    target_abs = abs_path
                    break
        if target_abs is None:
            return

        # Iteriere über alle registrierten VFS-Instanzen
        for vfs in list(self._mounted_vfs.values()):
            # Finde den Mount-Point in diesem VFS der auf target_abs zeigt
            for vfs_path, mount in list(vfs.mounts.items()):
                try:
                    if str(Path(mount.local_path).resolve()) != target_abs:
                        continue
                except Exception:
                    continue

                vfs_file_path = f"{vfs_path.rstrip('/')}/{relative_path}"
                f = vfs.files.get(vfs_file_path)

                if is_delete:
                    if f is not None and not getattr(f, "is_dirty", False):
                        vfs.files.pop(vfs_file_path, None)
                        vfs._shadow_index.pop(vfs_file_path, None)
                        vfs._dirty = True
                else:
                    if f is None or getattr(f, "is_dirty", False):
                        continue
                    try:
                        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType
                        f._content = None
                        f.backing_type = FileBackingType.SHADOW
                        vfs._dirty = True
                    except Exception:
                        pass
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

    def write_file(self, relative_path: str, content: str, author: str | None = None) -> dict:
        """
        Schreibt eine Datei in den globalen Ordner.

        EBENE 3: Schreibt in Shared-Store (RAM) UND auf Disk.
        Andere Agents sehen die Änderung sofort über den Shared-Store,
        ohne auf den nächsten Poll-Tick warten zu müssen.

        Args:
            relative_path: Pfad relativ zu /global/ (z.B. "data/config.json")
            content: Dateiinhalt
            author: Optional Agent-Name für Audit-Trail

        Returns:
            Result dict
        """
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        file_path = self.data_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Disk-Write zuerst (Crash-Safety — RAM ohne Disk wäre verloren)
            file_path.write_text(content, encoding="utf-8")
            stat = file_path.stat()

            # 2. Shared-Store updaten unter Lock
            with self._store_lock:
                self._version_counter += 1
                version = self._version_counter
                self._shared_store[relative_path] = {
                    "content": content,
                    "mtime": stat.st_mtime,
                    "author": author,
                    "version": version,
                }

            # 3. VFS-Caches anderer Agents invalidieren
            self.invalidate_vfs_caches(relative_path)

            # 4. Subscribers benachrichtigen
            self._notify({
                "type": "write",
                "path": relative_path,
                "author": author,
                "version": version,
            })

            return {
                "success": True,
                "path": f"{GLOBAL_VFS_PATH}/{relative_path}",
                "local_path": str(file_path),
                "size": len(content),
                "version": version,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, relative_path: str) -> dict:
        """
        Liest eine Datei aus dem globalen Ordner.

        EBENE 3: Shared-Store wird zuerst geprüft (In-Process-Cache).
        Disk ist Fallback für uninitialisierte Dateien.
        """
        if ".." in relative_path:
            return {"success": False, "error": "Path traversal not allowed"}

        # 1. Shared-Store zuerst (schneller, keine Disk-IO)
        with self._store_lock:
            entry = self._shared_store.get(relative_path)
            if entry is not None:
                return {
                    "success": True,
                    "path": f"{GLOBAL_VFS_PATH}/{relative_path}",
                    "content": entry["content"],
                    "size": len(entry["content"]),
                    "version": entry.get("version"),
                    "author": entry.get("author"),
                }

        # 2. Fallback: von Disk laden und in Store einfügen
        file_path = self.data_dir / relative_path
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {relative_path}"}

        try:
            content = file_path.read_text(encoding="utf-8")
            stat = file_path.stat()
            with self._store_lock:
                self._version_counter += 1
                version = self._version_counter
                self._shared_store[relative_path] = {
                    "content": content,
                    "mtime": stat.st_mtime,
                    "author": None,
                    "version": version,
                }
            return {
                "success": True,
                "path": f"{GLOBAL_VFS_PATH}/{relative_path}",
                "content": content,
                "size": len(content),
                "version": version,
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
            items.append(
                {
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "path": f"{GLOBAL_VFS_PATH}/{relative_path}/{item.name}".replace(
                        "//", "/"
                    ),
                }
            )

        return {
            "success": True,
            "path": f"{GLOBAL_VFS_PATH}/{relative_path}".rstrip("/"),
            "items": items,
            "count": len(items),
        }

    def delete_file(self, relative_path: str) -> dict:
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
            # EBENE 3: Shared-Store aufräumen + Broadcast
            with self._store_lock:
                removed = self._shared_store.pop(relative_path, None)
                # Auch alle Sub-Pfade wenn es ein Dir war
                to_drop = [
                    k for k in self._shared_store
                    if k.startswith(relative_path + "/")
                ]
                for k in to_drop:
                    del self._shared_store[k]
                self._version_counter += 1
                version = self._version_counter

            self._notify({
                "type": "delete",
                "path": relative_path,
                "author": None,
                "version": version,
            })

            # VFS-Sync: alle registrierten Instanzen updaten
            vfs_path = f"{GLOBAL_VFS_PATH}/{relative_path}"
            for vfs in self._mounted_vfs.values():
                # Files unter dem Pfad rauswerfen
                for p in [p for p in list(vfs.files.keys()) if p.startswith(vfs_path)]:
                    del vfs.files[p]
                    vfs._shadow_index.pop(p, None)
                # Dirs rauswerfen
                for p in [p for p in list(vfs.directories.keys())
                          if p.startswith(vfs_path + "/") or p == vfs_path]:
                    del vfs.directories[p]
                vfs._dirty = True

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
                rel_path = file_path[len(vfs_path) :].lstrip("/")
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

    def mount_share(
        self, vfs: "VirtualFileSystemV2", share_id: str, mount_point: str = None
    ) -> dict:
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
            result["source"] = (
                f"{share.source_agent}:{share.source_session}:{share.source_path}"
            )

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

        if filename in ["vfs_guide.md"]:
            return False

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
                results.append(
                    SearchResult(
                        path=file_path,
                        filename=vfs_file.filename,
                        match_type="filename",
                        score=1.0,
                    )
                )
                if len(results) >= max_results:
                    break

        # Content Search
        if mode in (SearchMode.CONTENT, SearchMode.BOTH):
            try:
                # Lade Content wenn nötig
                if vfs_file.is_loaded:
                    content = vfs_file.content
                elif vfs_file.local_path and os.path.exists(vfs_file.local_path):
                    content = Path(vfs_file.local_path).read_text(
                        encoding="utf-8", errors="ignore"
                    )
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
                            context_before = (
                                "\n".join(lines[start : line_num - 1])
                                if start < line_num - 1
                                else None
                            )
                            context_after = (
                                "\n".join(lines[line_num:end]) if line_num < end else None
                            )

                        results.append(
                            SearchResult(
                                path=file_path,
                                filename=vfs_file.filename,
                                match_type="content",
                                line_number=line_num,
                                line_content=line[:200],  # Truncate long lines
                                context_before=context_before,
                                context_after=context_after,
                                score=0.8,
                            )
                        )

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

        if vfs_file.filename in ["vfs_guide.md"]:
            continue

        try:
            if vfs_file.is_loaded:
                content = vfs_file.content
            elif vfs_file.local_path and os.path.exists(vfs_file.local_path):
                content = Path(vfs_file.local_path).read_text(
                    encoding="utf-8", errors="ignore"
                )
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
# AGENT TOOLS REGISTRATION - REMOVED
# =============================================================================

# NOTE: The register_vfs_tools() function has been removed as part of the
# architecture refactoring. VFS tools are now directly registered in
# FlowAgent.init_session_tools() in flow_agent.py.
#
# This eliminates tool bloat and monkey-patching patterns. The VFS operates
# transparently through mount points (/global/, /shared/, etc.) without
# requiring special tools like global_read, global_write, etc.
#
# Session initialization with VFS features (mounting /global/, etc.) is now
# handled directly in AgentSessionV2.initialize() in agent_session_v2.py.


# =============================================================================
# SESSION INITIALIZATION HELPER - REMOVED
# =============================================================================

# NOTE: The init_session_with_vfs_features() function has been removed.
# Session initialization with VFS features has been moved to
# AgentSessionV2.initialize() in agent_session_v2.py.
#
# This eliminates the need for monkey-patching and external initialization.
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
]
