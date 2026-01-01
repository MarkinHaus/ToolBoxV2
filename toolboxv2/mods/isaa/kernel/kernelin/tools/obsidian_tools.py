"""
Obsidian Tools for Discord/Telegram Kernels
=============================================

Quick-access tools that wrap the Obsidian MCP Server for use in chat interfaces.
"""

from typing import Any, Dict, List, Optional
from datetime import date, datetime
from pathlib import Path

from toolboxv2.mods.isaa.kernel.kernelin.obsidian import VaultManager, ObsidianMCPTools


class ObsidianKernelTools:
    """
    High-level tools for Discord/Telegram kernel integration.
    
    Provides simple commands like:
    - /capture [text] -> Daily note
    - /note [title] [content] -> New note
    - /search [query] -> Search vault
    - /link [from] [to] -> Create link
    """
    
    def __init__(self, vault_path: str, agent_id: str):
        self.vault = VaultManager(vault_path)
        self.mcp_tools = ObsidianMCPTools(self.vault, agent_id)
        self.agent_id = agent_id
    
    # ===== QUICK CAPTURE =====
    
    async def capture(self, text: str, section: str = "Notes", 
                      tags: List[str] = None) -> Dict[str, Any]:
        """
        Quick capture to today's daily note.
        
        Usage: /capture This is my idea #project #important
        """
        # Extract inline tags from text
        import re
        inline_tags = re.findall(r'#(\w+)', text)
        text_clean = re.sub(r'\s*#\w+', '', text).strip()
        
        all_tags = list(set((tags or []) + inline_tags))
        
        # Format entry
        timestamp = datetime.now().strftime('%H:%M')
        entry = f"- [{timestamp}] {text_clean}"
        if all_tags:
            entry += f" #{' #'.join(all_tags)}"
        
        # Append to daily note
        success = self.vault.append_to_daily(
            content=entry,
            section=section,
            agent_id=self.agent_id
        )
        
        return {
            "success": success,
            "captured": text_clean,
            "section": section,
            "tags": all_tags,
            "daily_note": f"Daily/{date.today().strftime('%Y-%m-%d')}.md"
        }
    
    # ===== NOTE CREATION =====
    
    async def create_note(self, title: str, content: str = "", 
                         folder: str = "Inbox", tags: List[str] = None,
                         template: str = None) -> Dict[str, Any]:
        """
        Create a new note.
        
        Usage: /note "My Project" "Initial ideas for the project" folder=Projects
        """
        # Sanitize title for filename
        import re
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
        path = f"{folder}/{safe_title}.md"
        
        note = self.vault.create_note(
            path=path,
            title=title,
            template=template,
            tags=tags,
            agent_id=self.agent_id
        )
        
        if content:
            # Add content after title
            full_content = note.content + f"\n\n{content}"
            self.vault.write_note(path, full_content, note.frontmatter, self.agent_id)
        
        return {
            "success": True,
            "path": path,
            "title": title,
            "tags": tags or []
        }
    
    # ===== SEARCH =====
    
    async def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search notes by text.
        
        Usage: /search python async
        """
        results = self.vault.search_notes(query, limit)
        
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": [
                {
                    "path": r.path,
                    "title": r.title,
                    "snippet": r.snippet
                }
                for r in results
            ]
        }
    
    async def search_tag(self, tag: str) -> Dict[str, Any]:
        """
        Find notes by tag.
        
        Usage: /tag project
        """
        notes = self.vault.search_by_tag(tag)
        
        return {
            "success": True,
            "tag": tag,
            "count": len(notes),
            "notes": [
                {"path": n.path, "title": n.title}
                for n in notes
            ]
        }
    
    # ===== READ =====
    
    async def read_note(self, path: str) -> Dict[str, Any]:
        """
        Read a note.
        
        Usage: /read Projects/MyProject.md
        """
        note = self.vault.read_note(path)
        
        if not note:
            return {"success": False, "error": "Note not found"}
        
        return {
            "success": True,
            "path": note.path,
            "title": note.title,
            "content": note.content,
            "tags": note.tags,
            "links": note.links,
            "backlinks": self.vault.get_backlinks(path)
        }
    
    # ===== GRAPH =====
    
    async def get_related(self, path: str) -> Dict[str, Any]:
        """
        Get related notes (links + backlinks).
        
        Usage: /related Projects/MyProject.md
        """
        neighbors = self.vault.get_neighbors(path, depth=1)
        suggestions = self.vault.suggest_links(path, limit=3)
        
        return {
            "success": True,
            "path": path,
            "links_to": neighbors["outgoing"],
            "linked_from": neighbors["incoming"],
            "suggested": [s[0] for s in suggestions]
        }
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get vault graph statistics.
        
        Usage: /graph
        """
        nodes, edges = self.vault.get_graph()
        orphans = self.vault.get_orphans()
        
        # Top linked notes
        top_linked = sorted(nodes, key=lambda n: n.backlink_count, reverse=True)[:5]
        
        # Tag distribution
        all_tags = {}
        for node in nodes:
            for tag in node.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1
        top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "success": True,
            "stats": {
                "total_notes": len(nodes),
                "total_links": len(edges),
                "orphan_notes": len(orphans),
                "average_links": len(edges) / len(nodes) if nodes else 0
            },
            "top_linked": [
                {"path": n.id, "title": n.title, "backlinks": n.backlink_count}
                for n in top_linked
            ],
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
            "orphans": orphans[:5]  # First 5 orphans
        }
    
    # ===== DAILY =====
    
    async def get_daily(self, date_str: str = None) -> Dict[str, Any]:
        """
        Get or create daily note.
        
        Usage: /daily or /daily 2024-01-15
        """
        if date_str:
            for_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            for_date = None
        
        note = self.vault.get_daily_note(for_date)
        
        return {
            "success": True,
            "path": note.path,
            "content": note.content
        }
    
    # ===== LINKS =====
    
    async def create_link(self, from_path: str, to_path: str) -> Dict[str, Any]:
        """
        Create link between notes.
        
        Usage: /link Projects/A.md Projects/B.md
        """
        result = await self.mcp_tools.execute_tool("obsidian_create_link", {
            "from_path": from_path,
            "to_path": to_path
        })
        return result
    
    # ===== EXPORT FOR AGENT =====
    
    def get_tools_for_agent(self) -> List[Dict]:
        """Get tool definitions for agent registration"""
        return [
            {
                "name": "vault_capture",
                "description": "Quick capture text to today's daily note. Supports inline #tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to capture"},
                        "section": {"type": "string", "default": "Notes", 
                                   "description": "Section in daily note"}
                    },
                    "required": ["text"]
                },
                "handler": self.capture
            },
            {
                "name": "vault_create_note",
                "description": "Create a new note in the vault",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Note title"},
                        "content": {"type": "string", "description": "Note content"},
                        "folder": {"type": "string", "default": "Inbox"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["title"]
                },
                "handler": self.create_note
            },
            {
                "name": "vault_search",
                "description": "Search notes in the vault",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                },
                "handler": self.search
            },
            {
                "name": "vault_read",
                "description": "Read a note from the vault",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Note path"}
                    },
                    "required": ["path"]
                },
                "handler": self.read_note
            },
            {
                "name": "vault_graph",
                "description": "Get vault graph statistics and top notes",
                "parameters": {"type": "object", "properties": {}},
                "handler": self.get_graph_stats
            },
            {
                "name": "vault_related",
                "description": "Get notes related to a specific note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Note path"}
                    },
                    "required": ["path"]
                },
                "handler": self.get_related
            },
            {
                "name": "vault_daily",
                "description": "Get today's daily note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_str": {"type": "string", "description": "Date (YYYY-MM-DD)"}
                    }
                },
                "handler": self.get_daily
            }
        ]
