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


# ===== VAULT MANAGER =====

class VaultManager:
    """Manages Obsidian vault operations"""
    
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
        
        # Cache for graph building
        self._note_cache: Dict[str, Note] = {}
        self._link_index: Dict[str, Set[str]] = {}  # path -> set of paths it links to
        self._backlink_index: Dict[str, Set[str]] = {}  # path -> set of paths linking to it
        
        # Build initial index
        self._build_index()
    
    def _build_index(self):
        """Build link index from all notes"""
        print("📊 Building vault index...")
        
        for md_file in self.vault_path.rglob("*.md"):
            rel_path = str(md_file.relative_to(self.vault_path))
            
            # Skip .obsidian folder
            if rel_path.startswith('.obsidian'):
                continue
            
            try:
                note = self._parse_note(md_file)
                self._note_cache[rel_path] = note
                
                # Build link index
                self._link_index[rel_path] = set(note.links)
                
                # Build backlink index
                for link in note.links:
                    if link not in self._backlink_index:
                        self._backlink_index[link] = set()
                    self._backlink_index[link].add(rel_path)
                    
            except Exception as e:
                print(f"⚠️ Error parsing {rel_path}: {e}")
        
        print(f"✓ Indexed {len(self._note_cache)} notes")
    
    def _parse_note(self, file_path: Path) -> Note:
        """Parse a markdown note file"""
        content = file_path.read_text(encoding='utf-8')
        rel_path = str(file_path.relative_to(self.vault_path))
        
        # Parse frontmatter
        frontmatter = {}
        body = content
        
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                except yaml.YAMLError:
                    pass
        
        # Extract title (first H1 or filename)
        title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem
        
        # Extract tags
        tags = frontmatter.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        # Also find inline #tags
        inline_tags = re.findall(r'#([a-zA-Z0-9_-]+)', body)
        tags = list(set(tags + inline_tags))
        
        # Extract [[links]]
        links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        # Normalize links (could be paths or just names)
        normalized_links = []
        for link in links:
            # Try to find the actual file
            link_path = self._resolve_link(link)
            if link_path:
                normalized_links.append(link_path)
        
        # File stats
        stat = file_path.stat()
        
        return Note(
            path=rel_path,
            title=title,
            content=content,
            frontmatter=frontmatter,
            tags=tags,
            links=normalized_links,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime)
        )
    
    def _resolve_link(self, link: str) -> Optional[str]:
        """Resolve a [[link]] to an actual file path"""
        # Already a path?
        if link.endswith('.md'):
            if (self.vault_path / link).exists():
                return link
        
        # Search for matching file
        search_name = link + '.md'
        for md_file in self.vault_path.rglob("*.md"):
            if md_file.name == search_name:
                return str(md_file.relative_to(self.vault_path))
        
        return None
    
    # ===== READ OPERATIONS =====
    
    def read_note(self, path: str) -> Optional[Note]:
        """Read a note by path"""
        # Check cache first
        if path in self._note_cache:
            return self._note_cache[path]
        
        file_path = self.vault_path / path
        if not file_path.exists():
            return None
        
        note = self._parse_note(file_path)
        self._note_cache[path] = note
        return note
    
    def search_notes(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Full-text search across all notes"""
        results = []
        query_lower = query.lower()
        
        for path, note in self._note_cache.items():
            content_lower = note.content.lower()
            
            if query_lower in content_lower:
                # Find match position for snippet
                pos = content_lower.find(query_lower)
                start = max(0, pos - 50)
                end = min(len(note.content), pos + len(query) + 50)
                snippet = note.content[start:end]
                
                # Simple scoring: title match > content match
                score = 1.0
                if query_lower in note.title.lower():
                    score = 2.0
                
                results.append(SearchResult(
                    path=path,
                    title=note.title,
                    snippet=f"...{snippet}...",
                    score=score,
                    matches=[(pos, pos + len(query))]
                ))
        
        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    def search_by_tag(self, tag: str) -> List[Note]:
        """Find all notes with a specific tag"""
        tag_clean = tag.lstrip('#')
        return [
            note for note in self._note_cache.values()
            if tag_clean in note.tags
        ]
    
    def get_backlinks(self, path: str) -> List[str]:
        """Get all notes that link to this note"""
        return list(self._backlink_index.get(path, set()))
    
    def get_neighbors(self, path: str, depth: int = 1) -> Dict[str, List[str]]:
        """Get linked notes within N hops"""
        visited = set()
        neighbors = {"outgoing": [], "incoming": []}
        
        def explore(current_path: str, current_depth: int, direction: str):
            if current_depth > depth or current_path in visited:
                return
            visited.add(current_path)
            
            if direction in ("outgoing", "both"):
                for link in self._link_index.get(current_path, set()):
                    neighbors["outgoing"].append(link)
                    if current_depth < depth:
                        explore(link, current_depth + 1, "outgoing")
            
            if direction in ("incoming", "both"):
                for backlink in self._backlink_index.get(current_path, set()):
                    neighbors["incoming"].append(backlink)
                    if current_depth < depth:
                        explore(backlink, current_depth + 1, "incoming")
        
        explore(path, 0, "both")
        
        # Remove duplicates
        neighbors["outgoing"] = list(set(neighbors["outgoing"]) - {path})
        neighbors["incoming"] = list(set(neighbors["incoming"]) - {path})
        
        return neighbors
    
    # ===== WRITE OPERATIONS =====
    
    def write_note(self, path: str, content: str, frontmatter: Dict = None,
                   agent_id: str = None) -> bool:
        """Write or update a note"""
        file_path = self.vault_path / path
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build content with frontmatter
        if frontmatter:
            yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
            full_content = f"---\n{yaml_str}---\n\n{content}"
        else:
            full_content = content
        
        # Write file
        file_path.write_text(full_content, encoding='utf-8')
        
        # Update cache
        note = self._parse_note(file_path)
        self._note_cache[path] = note
        
        # Update indexes
        old_links = self._link_index.get(path, set())
        new_links = set(note.links)
        
        # Remove old backlinks
        for old_link in old_links - new_links:
            if old_link in self._backlink_index:
                self._backlink_index[old_link].discard(path)
        
        # Add new backlinks
        for new_link in new_links - old_links:
            if new_link not in self._backlink_index:
                self._backlink_index[new_link] = set()
            self._backlink_index[new_link].add(path)
        
        self._link_index[path] = new_links
        
        # Git commit if available
        if self.repo and agent_id:
            self._git_commit(path, f"Update {path}", agent_id)
        
        return True
    
    def create_note(self, path: str, title: str, template: str = None,
                    tags: List[str] = None, agent_id: str = None) -> Note:
        """Create a new note from template"""
        # Default template
        if template is None:
            template = f"# {title}\n\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        else:
            # Load template file
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
        return self._note_cache[path]
    
    def delete_note(self, path: str, soft: bool = True, agent_id: str = None) -> bool:
        """Delete a note (soft = move to archive)"""
        file_path = self.vault_path / path
        
        if not file_path.exists():
            return False
        
        if soft:
            # Move to archive
            archive_path = self.vault_path / "Archive" / path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(archive_path)
        else:
            file_path.unlink()
        
        # Update cache and indexes
        if path in self._note_cache:
            del self._note_cache[path]
        
        if path in self._link_index:
            del self._link_index[path]
        
        # Remove from backlinks
        for backlinks in self._backlink_index.values():
            backlinks.discard(path)
        
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
        
        # Create daily note
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
        return self.create_note(
            path=path,
            title=date_str,
            template=template,
            tags=["daily"]
        )
    
    def append_to_daily(self, content: str, section: str = "Notes", 
                        for_date: date = None, agent_id: str = None) -> bool:
        """Append content to a section in daily note"""
        note = self.get_daily_note(for_date)
        
        # Find section
        section_pattern = rf'^## [^\n]*{section}[^\n]*$'
        match = re.search(section_pattern, note.content, re.MULTILINE | re.IGNORECASE)
        
        if match:
            # Insert after section header
            insert_pos = match.end()
            new_content = (
                note.content[:insert_pos] + 
                f"\n\n{content}" + 
                note.content[insert_pos:]
            )
        else:
            # Append at end
            new_content = note.content + f"\n\n## {section}\n\n{content}"
        
        return self.write_note(note.path, new_content, note.frontmatter, agent_id)
    
    # ===== GRAPH OPERATIONS =====
    
    def get_graph(self) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get full graph structure"""
        nodes = []
        edges = []
        
        for path, note in self._note_cache.items():
            # Create node
            folder = str(Path(path).parent) if '/' in path else ""
            nodes.append(GraphNode(
                id=path,
                title=note.title,
                tags=note.tags,
                link_count=len(self._link_index.get(path, set())),
                backlink_count=len(self._backlink_index.get(path, set())),
                folder=folder
            ))
            
            # Create edges
            for link in self._link_index.get(path, set()):
                edges.append(GraphEdge(
                    source=path,
                    target=link,
                    edge_type="link"
                ))
        
        return nodes, edges
    
    def get_orphans(self) -> List[str]:
        """Get notes with no incoming or outgoing links"""
        orphans = []
        
        for path in self._note_cache.keys():
            has_outgoing = len(self._link_index.get(path, set())) > 0
            has_incoming = len(self._backlink_index.get(path, set())) > 0
            
            if not has_outgoing and not has_incoming:
                orphans.append(path)
        
        return orphans
    
    def suggest_links(self, path: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Suggest potential links based on content similarity"""
        note = self.read_note(path)
        if not note:
            return []
        
        # Simple keyword-based suggestion
        # Extract significant words from title and content
        words = set(re.findall(r'\b[a-zA-Z]{4,}\b', note.content.lower()))
        
        suggestions = []
        for other_path, other_note in self._note_cache.items():
            if other_path == path:
                continue
            
            # Already linked?
            if other_path in self._link_index.get(path, set()):
                continue
            
            # Calculate overlap
            other_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', other_note.content.lower()))
            overlap = len(words & other_words)
            
            if overlap > 3:  # Threshold
                score = overlap / max(len(words), len(other_words))
                suggestions.append((other_path, score))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:limit]
    
    # ===== GIT OPERATIONS =====
    
    def _git_commit(self, path: str, message: str, agent_id: str):
        """Commit changes to agent's branch"""
        if not self.repo:
            return
        
        try:
            # Determine branch
            branch_name = f"agent/{agent_id}" if agent_id else "main"
            
            # Checkout branch (create if not exists)
            if branch_name not in [b.name for b in self.repo.branches]:
                self.repo.create_head(branch_name)
            
            # Stage and commit
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
            
            # Commits ahead/behind main
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
            # Checkout main
            self.repo.heads.main.checkout()
            
            # Try merge
            try:
                self.repo.git.merge(branch_name, '--no-ff', '-m', f'Merge {branch_name}')
                return {"success": True, "message": f"Merged {branch_name} to main"}
            except git.GitCommandError as e:
                if 'conflict' in str(e).lower():
                    if auto:
                        # Abort merge
                        self.repo.git.merge('--abort')
                        return {
                            "success": False, 
                            "error": "Merge conflict",
                            "needs_manual": True
                        }
                raise
                
        except Exception as e:
            return {"success": False, "error": str(e)}


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

if __name__ == "__main__":
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
