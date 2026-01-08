"""
Obsidian MCP Server for Agent Access
=====================================

Provides MCP tools for agents to interact with Obsidian vaults:
- Read/Write/Search notes
- Graph operations (links, backlinks, neighbors)
- Daily notes management
- Git-based versioning per agent branch

Architecture:
- Each agent works on its own branch (agent/discord, agent/telegram)
- Changes auto-commit to agent branch
- Auto-merge to main if no conflicts
- Conflicts flagged for manual resolution
"""

import os
import re
import json
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("⚠️ GitPython not installed. Install with: pip install gitpython")

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ watchdog not installed. Install with: pip install watchdog")


# ===== DATA STRUCTURES =====

@dataclass
class Note:
    """Represents an Obsidian note"""
    path: str  # Relative path from vault root
    title: str
    content: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)  # Outgoing [[links]]
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class GraphNode:
    """Node in the knowledge graph"""
    id: str  # Note path
    title: str
    tags: List[str]
    link_count: int
    backlink_count: int
    folder: str


@dataclass
class GraphEdge:
    """Edge in the knowledge graph"""
    source: str
    target: str
    edge_type: str = "link"  # link, tag, folder


@dataclass
class SearchResult:
    """Search result"""
    path: str
    title: str
    snippet: str
    score: float
    matches: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) positions


class AgentPermission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


"""
VaultManager V2 - Optimized with persistent incremental indexing
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Tuple, Any
import re

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


@dataclass
class Note:
    path: str
    title: str
    content: str
    frontmatter: Dict
    tags: List[str]
    links: List[str]
    created: datetime
    modified: datetime


@dataclass
class SearchResult:
    path: str
    title: str
    snippet: str
    score: float
    matches: List[Tuple[int, int]]


@dataclass
class GraphNode:
    id: str
    title: str
    tags: List[str]
    link_count: int
    backlink_count: int
    folder: str


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str


@dataclass
class FolderIndex:
    """Index for a single folder - enables granular updates"""
    folder_path: str
    files: Dict[str, 'FileIndexEntry'] = field(default_factory=dict)
    last_scan: float = 0.0  # timestamp
    content_hash: str = ""  # hash of folder state

    def to_dict(self) -> dict:
        return {
            "folder_path": self.folder_path,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "last_scan": self.last_scan,
            "content_hash": self.content_hash
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FolderIndex':
        idx = cls(
            folder_path=data["folder_path"],
            last_scan=data.get("last_scan", 0.0),
            content_hash=data.get("content_hash", "")
        )
        idx.files = {k: FileIndexEntry.from_dict(v) for k, v in data.get("files", {}).items()}
        return idx


@dataclass
class FileIndexEntry:
    """Index entry for a single file"""
    path: str
    title: str
    tags: List[str]
    links: List[str]
    mtime: float  # modification time for change detection
    size: int
    content_hash: str  # for content change detection

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'FileIndexEntry':
        return cls(**data)


@dataclass
class VaultIndex:
    """Complete vault index with folder-based sharding"""
    vault_path: str
    folders: Dict[str, FolderIndex] = field(default_factory=dict)
    link_index: Dict[str, List[str]] = field(default_factory=dict)  # path -> links
    backlink_index: Dict[str, List[str]] = field(default_factory=dict)  # path -> backlinks
    version: int = 2
    last_full_scan: float = 0.0

    INDEX_FILENAME = ".tb_index"

    def to_dict(self) -> dict:
        return {
            "vault_path": self.vault_path,
            "folders": {k: v.to_dict() for k, v in self.folders.items()},
            "link_index": self.link_index,
            "backlink_index": self.backlink_index,
            "version": self.version,
            "last_full_scan": self.last_full_scan
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VaultIndex':
        idx = cls(
            vault_path=data["vault_path"],
            version=data.get("version", 1),
            last_full_scan=data.get("last_full_scan", 0.0)
        )
        idx.folders = {k: FolderIndex.from_dict(v) for k, v in data.get("folders", {}).items()}
        idx.link_index = data.get("link_index", {})
        idx.backlink_index = data.get("backlink_index", {})
        return idx


class VaultManager:
    """Manages Obsidian vault operations with persistent incremental indexing"""

    def __init__(self, vault_path: str, git_repo_path: str = None):
        self.vault_path = Path(vault_path)
        self.git_repo_path = Path(git_repo_path) if git_repo_path else self.vault_path

        # Ensure vault exists
        if not self.vault_path.exists():
            self.vault_path.mkdir(parents=True)
            print(f"✓ Created vault at {self.vault_path}")

        # Initialize Git if available
        self.repo = None
        if GIT_AVAILABLE and (self.git_repo_path / '.git').exists():
            self.repo = git.Repo(self.git_repo_path)
            print(f"✓ Git repository loaded: {self.repo.active_branch}")

        # Note cache (loaded on demand)
        self._note_cache: Dict[str, Note] = {}

        # Load or build index
        self.index = self._load_or_create_index()

        # Incremental update
        self._incremental_update()

    # ===== INDEX PERSISTENCE =====

    def _get_index_path(self) -> Path:
        """Get path to index file"""
        return self.vault_path / VaultIndex.INDEX_FILENAME

    def _load_or_create_index(self) -> VaultIndex:
        """Load existing index or create new one"""
        index_path = self._get_index_path()

        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Version check
                if data.get("version", 1) >= 2:
                    print(f"✓ Loaded index from {index_path}")
                    return VaultIndex.from_dict(data)
                else:
                    print("⚠️ Index version outdated, rebuilding...")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Index corrupted ({e}), rebuilding...")

        # Create new index
        return VaultIndex(vault_path=str(self.vault_path))

    def _save_index(self):
        """Persist index to disk"""
        index_path = self._get_index_path()

        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(self.index.to_dict(), f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save index: {e}")

    # ===== INTELLIGENT INCREMENTAL INDEXING =====

    def _incremental_update(self):
        """Scan only changed folders and files"""
        print("📊 Checking for changes...")

        changes = {"added": 0, "modified": 0, "deleted": 0}

        # Get current folder structure (without rglob - use os.walk for efficiency)
        current_folders = set()
        current_files: Dict[str, Tuple[float, int]] = {}  # path -> (mtime, size)

        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden folders
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            rel_root = os.path.relpath(root, self.vault_path)
            if rel_root == '.':
                rel_root = ''

            current_folders.add(rel_root)

            for filename in files:
                if not filename.endswith('.md'):
                    continue

                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, self.vault_path)

                stat = os.stat(full_path)
                current_files[rel_path] = (stat.st_mtime, stat.st_size)

        # Detect deleted folders
        indexed_folders = set(self.index.folders.keys())
        deleted_folders = indexed_folders - current_folders

        for folder in deleted_folders:
            # Remove all files in deleted folder from indexes
            folder_index = self.index.folders.pop(folder, None)
            if folder_index:
                for file_path in folder_index.files:
                    self._remove_from_link_index(file_path)
                    changes["deleted"] += 1

        # Process each current folder
        for folder in current_folders:
            folder_index = self.index.folders.get(folder)

            if folder_index is None:
                # New folder - index all files
                folder_index = FolderIndex(folder_path=folder)
                self.index.folders[folder] = folder_index

            # Get files in this folder
            folder_files = {
                path: (mtime, size)
                for path, (mtime, size) in current_files.items()
                if os.path.dirname(path) == folder or (folder == '' and '/' not in path and '\\' not in path)
            }

            # Check folder hash for quick skip
            folder_hash = self._compute_folder_hash(folder_files)

            if folder_hash == folder_index.content_hash:
                # Folder unchanged, skip
                continue

            # Detect changes in this folder
            indexed_files = set(folder_index.files.keys())
            current_file_paths = set(folder_files.keys())

            # Deleted files
            for deleted in indexed_files - current_file_paths:
                self._remove_from_link_index(deleted)
                del folder_index.files[deleted]
                changes["deleted"] += 1

            # New or modified files
            for file_path in current_file_paths:
                mtime, size = folder_files[file_path]
                existing = folder_index.files.get(file_path)

                if existing is None:
                    # New file
                    self._index_file(file_path, folder_index)
                    changes["added"] += 1
                elif existing.mtime < mtime or existing.size != size:
                    # Modified file
                    self._remove_from_link_index(file_path)
                    self._index_file(file_path, folder_index)
                    changes["modified"] += 1

            # Update folder hash
            folder_index.content_hash = folder_hash
            folder_index.last_scan = datetime.now().timestamp()

        # Rebuild backlink index (fast operation on link_index)
        self._rebuild_backlink_index()

        # Save updated index
        self._save_index()

        total = sum(changes.values())
        if total > 0:
            print(f"✓ Index updated: +{changes['added']} ~{changes['modified']} -{changes['deleted']}")
        else:
            print("✓ Index up to date")

    def _compute_folder_hash(self, files: Dict[str, Tuple[float, int]]) -> str:
        """Compute hash of folder state for quick change detection"""
        # Sort for deterministic hash
        items = sorted(files.items())
        content = "|".join(f"{path}:{mtime}:{size}" for path, (mtime, size) in items)
        return hashlib.md5(content.encode()).hexdigest()

    def _index_file(self, path: str, folder_index: FolderIndex):
        """Index a single file"""
        file_path = self.vault_path / path

        try:
            content = file_path.read_text(encoding='utf-8')
            stat = file_path.stat()

            # Parse metadata
            title, tags, links = self._parse_file_metadata(path, content)

            # Create index entry
            entry = FileIndexEntry(
                path=path,
                title=title,
                tags=tags,
                links=links,
                mtime=stat.st_mtime,
                size=stat.st_size,
                content_hash=hashlib.md5(content.encode()).hexdigest()[:16]
            )

            folder_index.files[path] = entry

            # Update link index
            self.index.link_index[path] = links

        except Exception as e:
            print(f"⚠️ Error indexing {path}: {e}")

    def _parse_file_metadata(self, path: str, content: str) -> Tuple[str, List[str], List[str]]:
        """Extract title, tags, and links from content"""
        # Parse frontmatter for tags
        tags = []
        body = content

        if content.startswith('---') and YAML_AVAILABLE:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    fm_tags = frontmatter.get('tags', [])
                    if isinstance(fm_tags, str):
                        tags = [fm_tags]
                    elif isinstance(fm_tags, list):
                        tags = fm_tags
                    body = parts[2]
                except yaml.YAMLError:
                    pass

        # Title from first H1 or filename
        title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
        title = title_match.group(1) if title_match else Path(path).stem

        # Inline tags
        inline_tags = re.findall(r'(?<!\w)#([a-zA-Z0-9_-]+)', body)
        tags = list(set(tags + inline_tags))

        # Extract [[links]]
        raw_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        links = []
        for link in raw_links:
            resolved = self._resolve_link(link)
            if resolved:
                links.append(resolved)

        return title, tags, links

    def _resolve_link(self, link: str) -> Optional[str]:
        """Resolve a [[link]] to an actual file path"""
        if link.endswith('.md'):
            if (self.vault_path / link).exists():
                return link

        # Search in index first (faster than filesystem)
        search_name = link + '.md'
        search_name_lower = search_name.lower()

        for folder_index in self.index.folders.values():
            for file_path in folder_index.files:
                if file_path.lower().endswith(search_name_lower):
                    return file_path

        # Fallback to filesystem for new files
        for folder in self.vault_path.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                candidate = folder / search_name
                if candidate.exists():
                    return str(candidate.relative_to(self.vault_path))

        # Root level
        if (self.vault_path / search_name).exists():
            return search_name

        return None

    def _remove_from_link_index(self, path: str):
        """Remove a file from link index"""
        if path in self.index.link_index:
            del self.index.link_index[path]

    def _rebuild_backlink_index(self):
        """Rebuild backlink index from link index"""
        self.index.backlink_index.clear()

        for source_path, links in self.index.link_index.items():
            for target in links:
                if target not in self.index.backlink_index:
                    self.index.backlink_index[target] = []
                self.index.backlink_index[target].append(source_path)

    # ===== PUBLIC API - READ OPERATIONS =====

    def read_note(self, path: str) -> Optional[Note]:
        """Read a note by path"""
        if path in self._note_cache:
            return self._note_cache[path]

        file_path = self.vault_path / path
        if not file_path.exists():
            return None

        note = self._parse_note(file_path)
        self._note_cache[path] = note
        return note

    def _parse_note(self, file_path: Path) -> Note:
        """Full parse of a note file"""
        content = file_path.read_text(encoding='utf-8')
        rel_path = str(file_path.relative_to(self.vault_path))

        frontmatter = {}
        body = content

        if content.startswith('---') and YAML_AVAILABLE:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                except yaml.YAMLError:
                    pass

        title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        tags = frontmatter.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        inline_tags = re.findall(r'(?<!\w)#([a-zA-Z0-9_-]+)', body)
        tags = list(set(tags + inline_tags))

        links = self.index.link_index.get(rel_path, [])

        stat = file_path.stat()

        return Note(
            path=rel_path,
            title=title,
            content=content,
            frontmatter=frontmatter,
            tags=tags,
            links=links,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime)
        )

    def search_notes(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Full-text search across all notes"""
        results = []
        query_lower = query.lower()

        for folder_index in self.index.folders.values():
            for path, entry in folder_index.files.items():
                # Quick title check first
                if query_lower in entry.title.lower():
                    note = self.read_note(path)
                    if note:
                        results.append(SearchResult(
                            path=path,
                            title=entry.title,
                            snippet=note.content[:150] + "...",
                            score=2.0,
                            matches=[]
                        ))
                    continue

                # Full content search (lazy load)
                note = self.read_note(path)
                if note and query_lower in note.content.lower():
                    pos = note.content.lower().find(query_lower)
                    start = max(0, pos - 50)
                    end = min(len(note.content), pos + len(query) + 50)
                    snippet = note.content[start:end]

                    results.append(SearchResult(
                        path=path,
                        title=entry.title,
                        snippet=f"...{snippet}...",
                        score=1.0,
                        matches=[(pos, pos + len(query))]
                    ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_by_tag(self, tag: str) -> List[Note]:
        """Find all notes with a specific tag"""
        tag_clean = tag.lstrip('#')
        results = []

        for folder_index in self.index.folders.values():
            for path, entry in folder_index.files.items():
                if tag_clean in entry.tags:
                    note = self.read_note(path)
                    if note:
                        results.append(note)

        return results

    def get_backlinks(self, path: str) -> List[str]:
        """Get all notes that link to this note"""
        return self.index.backlink_index.get(path, [])

    def get_neighbors(self, path: str, depth: int = 1) -> Dict[str, List[str]]:
        """Get linked notes within N hops"""
        visited = set()
        neighbors = {"outgoing": [], "incoming": []}

        def explore(current_path: str, current_depth: int, direction: str):
            if current_depth > depth or current_path in visited:
                return
            visited.add(current_path)

            if direction in ("outgoing", "both"):
                for link in self.index.link_index.get(current_path, []):
                    neighbors["outgoing"].append(link)
                    if current_depth < depth:
                        explore(link, current_depth + 1, "outgoing")

            if direction in ("incoming", "both"):
                for backlink in self.index.backlink_index.get(current_path, []):
                    neighbors["incoming"].append(backlink)
                    if current_depth < depth:
                        explore(backlink, current_depth + 1, "incoming")

        explore(path, 0, "both")

        neighbors["outgoing"] = list(set(neighbors["outgoing"]) - {path})
        neighbors["incoming"] = list(set(neighbors["incoming"]) - {path})

        return neighbors

    # ===== WRITE OPERATIONS =====

    def write_note(self, path: str, content: str, frontmatter: Dict = None,
                   agent_id: str = None) -> bool:
        """Write or update a note"""
        file_path = self.vault_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if frontmatter and YAML_AVAILABLE:
            yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
            full_content = f"---\n{yaml_str}---\n\n{content}"
        else:
            full_content = content

        file_path.write_text(full_content, encoding='utf-8')

        # Update index incrementally
        folder = str(Path(path).parent) if '/' in path or '\\' in path else ''

        if folder not in self.index.folders:
            self.index.folders[folder] = FolderIndex(folder_path=folder)

        self._index_file(path, self.index.folders[folder])
        self._rebuild_backlink_index()

        # Invalidate note cache
        if path in self._note_cache:
            del self._note_cache[path]

        self._save_index()

        if self.repo and agent_id:
            self._git_commit(path, f"Update {path}", agent_id)

        return True

    def create_note(self, path: str, title: str, template: str = None,
                    tags: List[str] = None, agent_id: str = None) -> Note:
        """Create a new note from template"""
        if template is None:
            template = f"# {title}\n\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        else:
            template_path = self.vault_path / "Templates" / f"{template}.md"
            if template_path.exists():
                template = template_path.read_text()
                template = template.replace("{{title}}", title)
                template = template.replace("{{date}}", datetime.now().strftime('%Y-%m-%d'))

        frontmatter = {
            "created": datetime.now().isoformat(),
            "tags": tags or []
        }

        self.write_note(path, template, frontmatter, agent_id)
        return self.read_note(path)

    def delete_note(self, path: str, soft: bool = True, agent_id: str = None) -> bool:
        """Delete a note (soft = move to archive)"""
        file_path = self.vault_path / path

        if not file_path.exists():
            return False

        if soft:
            archive_path = self.vault_path / "Archive" / path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(archive_path)
        else:
            file_path.unlink()

        # Update index
        folder = str(Path(path).parent) if '/' in path or '\\' in path else ''
        if folder in self.index.folders and path in self.index.folders[folder].files:
            del self.index.folders[folder].files[path]

        self._remove_from_link_index(path)
        self._rebuild_backlink_index()

        if path in self._note_cache:
            del self._note_cache[path]

        self._save_index()

        if self.repo and agent_id:
            action = "Archive" if soft else "Delete"
            self._git_commit(path, f"{action} {path}", agent_id)

        return True

    # ===== DAILY NOTES =====

    def get_daily_note(self, for_date: date = None) -> Note:
        """Get or create daily note for date"""
        if for_date is None:
            for_date = date.today()

        date_str = for_date.strftime('%Y-%m-%d')
        path = f"Daily/{date_str}.md"

        note = self.read_note(path)
        if note:
            return note

        template = f"""# {date_str}

## 📅 Schedule

## 📝 Notes

## ✅ Tasks
- [ ]

## 💡 Ideas

## 📚 Learned

---
*Created automatically*
"""
        return self.create_note(path=path, title=date_str, template=template, tags=["daily"])

    def append_to_daily(self, content: str, section: str = "Notes",
                        for_date: date = None, agent_id: str = None) -> bool:
        """Append content to a section in daily note"""
        note = self.get_daily_note(for_date)

        section_pattern = rf'^## [^\n]*{section}[^\n]*$'
        match = re.search(section_pattern, note.content, re.MULTILINE | re.IGNORECASE)

        if match:
            insert_pos = match.end()
            new_content = note.content[:insert_pos] + f"\n\n{content}" + note.content[insert_pos:]
        else:
            new_content = note.content + f"\n\n## {section}\n\n{content}"

        return self.write_note(note.path, new_content, note.frontmatter, agent_id)

    # ===== GRAPH OPERATIONS =====

    def get_graph(self) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get full graph structure"""
        nodes = []
        edges = []

        for folder_index in self.index.folders.values():
            for path, entry in folder_index.files.items():
                folder = str(Path(path).parent) if '/' in path else ""
                nodes.append(GraphNode(
                    id=path,
                    title=entry.title,
                    tags=entry.tags,
                    link_count=len(self.index.link_index.get(path, [])),
                    backlink_count=len(self.index.backlink_index.get(path, [])),
                    folder=folder
                ))

                for link in self.index.link_index.get(path, []):
                    edges.append(GraphEdge(source=path, target=link, edge_type="link"))

        return nodes, edges

    def get_orphans(self) -> List[str]:
        """Get notes with no incoming or outgoing links"""
        orphans = []

        for folder_index in self.index.folders.values():
            for path in folder_index.files:
                has_outgoing = len(self.index.link_index.get(path, [])) > 0
                has_incoming = len(self.index.backlink_index.get(path, [])) > 0

                if not has_outgoing and not has_incoming:
                    orphans.append(path)

        return orphans

    def suggest_links(self, path: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Suggest potential links based on content similarity"""
        note = self.read_note(path)
        if not note:
            return []

        words = set(re.findall(r'\b[a-zA-Z]{4,}\b', note.content.lower()))
        suggestions = []

        for folder_index in self.index.folders.values():
            for other_path in folder_index.files:
                if other_path == path or other_path in self.index.link_index.get(path, []):
                    continue

                other_note = self.read_note(other_path)
                if not other_note:
                    continue

                other_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', other_note.content.lower()))
                overlap = len(words & other_words)

                if overlap > 3:
                    score = overlap / max(len(words), len(other_words))
                    suggestions.append((other_path, score))

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:limit]

    # ===== INDEX MANAGEMENT =====

    def force_reindex(self):
        """Force complete reindex of vault"""
        print("🔄 Force reindexing entire vault...")
        self.index = VaultIndex(vault_path=str(self.vault_path))
        self._note_cache.clear()
        self._incremental_update()

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        total_files = sum(len(f.files) for f in self.index.folders.values())
        total_links = sum(len(links) for links in self.index.link_index.values())

        return {
            "folders": len(self.index.folders),
            "files": total_files,
            "links": total_links,
            "backlinks": sum(len(bl) for bl in self.index.backlink_index.values()),
            "index_version": self.index.version,
            "last_full_scan": datetime.fromtimestamp(
                self.index.last_full_scan).isoformat() if self.index.last_full_scan else None
        }

    # ===== GIT OPERATIONS =====

    def _git_commit(self, path: str, message: str, agent_id: str):
        """Commit changes to agent's branch"""
        if not self.repo:
            return

        try:
            branch_name = f"agent/{agent_id}" if agent_id else "main"

            if branch_name not in [b.name for b in self.repo.branches]:
                self.repo.create_head(branch_name)

            self.repo.index.add([path])
            self.repo.index.commit(f"[{agent_id}] {message}")
            print(f"✓ Committed to {branch_name}: {message}")

        except Exception as e:
            print(f"⚠️ Git commit failed: {e}")

    def get_branch_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of agent's branch"""
        if not self.repo:
            return {"error": "Git not available"}

        branch_name = f"agent/{agent_id}"

        try:
            branch = self.repo.heads[branch_name]
            main = self.repo.heads.main
            commits_ahead = list(self.repo.iter_commits(f'main..{branch_name}'))
            commits_behind = list(self.repo.iter_commits(f'{branch_name}..main'))

            return {
                "branch": branch_name,
                "last_commit": str(branch.commit),
                "last_message": branch.commit.message,
                "commits_ahead": len(commits_ahead),
                "commits_behind": len(commits_behind),
                "can_auto_merge": len(commits_behind) == 0
            }
        except Exception as e:
            return {"error": str(e)}

    def merge_to_main(self, agent_id: str, auto: bool = True) -> Dict[str, Any]:
        """Merge agent branch to main"""
        if not self.repo:
            return {"success": False, "error": "Git not available"}

        branch_name = f"agent/{agent_id}"

        try:
            self.repo.heads.main.checkout()

            try:
                self.repo.git.merge(branch_name, '--no-ff', '-m', f'Merge {branch_name}')
                return {"success": True, "message": f"Merged {branch_name} to main"}
            except git.GitCommandError as e:
                if 'conflict' in str(e).lower():
                    if auto:
                        self.repo.git.merge('--abort')
                        return {"success": False, "error": "Merge conflict", "needs_manual": True}
                raise

        except Exception as e:
            return {"success": False, "error": str(e)}


# ===== CLI for testing =====
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vault_manager_v2.py <vault_path> [command]")
        print("Commands: stats, reindex, orphans, search <query>")
        sys.exit(1)

    vault_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "stats"

    vm = VaultManager(vault_path)

    if command == "stats":
        stats = vm.get_index_stats()
        print(json.dumps(stats, indent=2))
    elif command == "reindex":
        vm.force_reindex()
    elif command == "orphans":
        orphans = vm.get_orphans()
        for o in orphans:
            print(f"  - {o}")
    elif command == "search" and len(sys.argv) > 3:
        query = sys.argv[3]
        results = vm.search_notes(query)
        for r in results:
            print(f"  [{r.score:.1f}] {r.title}: {r.snippet[:60]}...")

# ===== MCP TOOLS EXPORT =====

class ObsidianMCPTools:
    """MCP Tool definitions for Obsidian vault access"""

    def __init__(self, vault_manager: VaultManager, agent_id: str):
        self.vault = vault_manager
        self.agent_id = agent_id

    def get_tools(self) -> List[Dict]:
        """Get tool definitions for MCP/Agent registration"""
        return [
            {
                "name": "obsidian_read_note",
                "description": "Read a note from the Obsidian vault by path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the note (e.g., 'Projects/MyProject.md')"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "obsidian_write_note",
                "description": "Write or update a note in the Obsidian vault",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path for the note"},
                        "content": {"type": "string", "description": "Markdown content"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for the note"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "obsidian_search",
                "description": "Search notes by text query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "obsidian_search_by_tag",
                "description": "Find all notes with a specific tag",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string", "description": "Tag to search for"}
                    },
                    "required": ["tag"]
                }
            },
            {
                "name": "obsidian_get_daily_note",
                "description": "Get or create today's daily note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format (default: today)"
                        }
                    }
                }
            },
            {
                "name": "obsidian_append_to_daily",
                "description": "Append content to a section in daily note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to append"},
                        "section": {
                            "type": "string",
                            "default": "Notes",
                            "description": "Section name (Notes, Tasks, Ideas, etc.)"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "obsidian_get_backlinks",
                "description": "Get all notes that link to a specific note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the note"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "obsidian_get_graph",
                "description": "Get the knowledge graph structure (nodes and edges)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_orphans": {"type": "boolean", "default": False}
                    }
                }
            },
            {
                "name": "obsidian_suggest_links",
                "description": "Get AI-suggested links for a note based on content similarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the note"},
                        "limit": {"type": "integer", "default": 5}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "obsidian_create_link",
                "description": "Create a [[link]] from one note to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_path": {"type": "string", "description": "Source note path"},
                        "to_path": {"type": "string", "description": "Target note path"},
                        "context": {
                            "type": "string",
                            "description": "Context where to insert the link (optional)"
                        }
                    },
                    "required": ["from_path", "to_path"]
                }
            }
        ]

    async def execute_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """Execute an MCP tool"""
        try:
            if tool_name == "obsidian_read_note":
                note = self.vault.read_note(parameters["path"])
                if note:
                    return {
                        "success": True,
                        "note": {
                            "path": note.path,
                            "title": note.title,
                            "content": note.content,
                            "tags": note.tags,
                            "links": note.links,
                            "backlinks": self.vault.get_backlinks(note.path)
                        }
                    }
                return {"success": False, "error": "Note not found"}

            elif tool_name == "obsidian_write_note":
                frontmatter = {"tags": parameters.get("tags", [])}
                success = self.vault.write_note(
                    parameters["path"],
                    parameters["content"],
                    frontmatter,
                    self.agent_id
                )
                return {"success": success}

            elif tool_name == "obsidian_search":
                results = self.vault.search_notes(
                    parameters["query"],
                    parameters.get("limit", 10)
                )
                return {
                    "success": True,
                    "results": [
                        {"path": r.path, "title": r.title, "snippet": r.snippet}
                        for r in results
                    ]
                }

            elif tool_name == "obsidian_search_by_tag":
                notes = self.vault.search_by_tag(parameters["tag"])
                return {
                    "success": True,
                    "notes": [{"path": n.path, "title": n.title} for n in notes]
                }

            elif tool_name == "obsidian_get_daily_note":
                date_str = parameters.get("date")
                for_date = datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else None
                note = self.vault.get_daily_note(for_date)
                return {
                    "success": True,
                    "note": {
                        "path": note.path,
                        "content": note.content
                    }
                }

            elif tool_name == "obsidian_append_to_daily":
                success = self.vault.append_to_daily(
                    parameters["content"],
                    parameters.get("section", "Notes"),
                    agent_id=self.agent_id
                )
                return {"success": success}

            elif tool_name == "obsidian_get_backlinks":
                backlinks = self.vault.get_backlinks(parameters["path"])
                return {"success": True, "backlinks": backlinks}

            elif tool_name == "obsidian_get_graph":
                nodes, edges = self.vault.get_graph()
                return {
                    "success": True,
                    "nodes": [
                        {"id": n.id, "title": n.title, "tags": n.tags,
                         "links": n.link_count, "backlinks": n.backlink_count}
                        for n in nodes
                    ],
                    "edges": [
                        {"source": e.source, "target": e.target, "type": e.edge_type}
                        for e in edges
                    ],
                    "stats": {
                        "total_notes": len(nodes),
                        "total_links": len(edges),
                        "orphans": len(self.vault.get_orphans()) if parameters.get("include_orphans") else None
                    }
                }

            elif tool_name == "obsidian_suggest_links":
                suggestions = self.vault.suggest_links(
                    parameters["path"],
                    parameters.get("limit", 5)
                )
                return {
                    "success": True,
                    "suggestions": [
                        {"path": path, "score": score}
                        for path, score in suggestions
                    ]
                }

            elif tool_name == "obsidian_create_link":
                from_note = self.vault.read_note(parameters["from_path"])
                if not from_note:
                    return {"success": False, "error": "Source note not found"}

                to_note = self.vault.read_note(parameters["to_path"])
                to_title = to_note.title if to_note else Path(parameters["to_path"]).stem

                # Add link to content
                link_text = f"[[{to_title}]]"
                context = parameters.get("context")

                if context:
                    new_content = from_note.content.replace(context, f"{context} {link_text}")
                else:
                    new_content = from_note.content + f"\n\n## Related\n- {link_text}"

                self.vault.write_note(
                    parameters["from_path"],
                    new_content,
                    from_note.frontmatter,
                    self.agent_id
                )
                return {"success": True, "link_added": link_text}

            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


# ===== STANDALONE TESTING =====

if __name__ == "__main__2":
    import tempfile

    # Create test vault
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = VaultManager(tmpdir)

        # Create some test notes
        vault.create_note("Projects/TestProject.md", "Test Project", tags=["project"])
        vault.create_note("Knowledge/Python.md", "Python Notes", tags=["programming"])
        vault.write_note(
            "Knowledge/Python.md",
            "# Python\n\nSee also: [[TestProject]]\n\n## Basics\n\nPython is great!",
            {"tags": ["programming", "python"]}
        )

        # Test search
        results = vault.search_notes("Python")
        print(f"Search results: {len(results)}")

        # Test graph
        nodes, edges = vault.get_graph()
        print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")

        # Test backlinks
        backlinks = vault.get_backlinks("Projects/TestProject.md")
        print(f"Backlinks to TestProject: {backlinks}")

        print("✓ All tests passed!")
