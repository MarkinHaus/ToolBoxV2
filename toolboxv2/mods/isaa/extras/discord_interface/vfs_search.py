"""
VFS Search Extension - Suchfunktion für VirtualFileSystemV2
============================================================

Fügt search() Methode zum VFS hinzu.

Usage:
    from vfs_search import add_search_to_vfs

    # VFS erweitern
    add_search_to_vfs(vfs_instance)

    # Oder als Mixin
    class MyVFS(VirtualFileSystemV2, VFSSearchMixin):
        pass
"""

import re
from typing import Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """Suchmodus"""
    FILENAME = "filename"  # Nur Dateinamen
    CONTENT = "content"  # Nur Inhalt
    BOTH = "both"  # Beides


@dataclass
class SearchResult:
    """Ein Suchergebnis"""
    path: str
    filename: str
    match_type: str  # "filename" oder "content"
    line_number: Optional[int] = None  # Bei Content-Match
    line_content: Optional[str] = None  # Die matchende Zeile
    score: float = 0.0  # Relevanz-Score

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "filename": self.filename,
            "match_type": self.match_type,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "score": self.score,
        }


class VFSSearchMixin:
    """
    Mixin-Klasse die search() zum VFS hinzufügt.

    Kann als Mixin verwendet werden oder die Methoden können
    direkt zu einer VFS-Instanz hinzugefügt werden.
    """

    def search(
        self,
        query: str,
        path: str = "/",
        mode: SearchMode = SearchMode.BOTH,
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 50,
        file_extensions: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> dict:
        """
        Durchsucht das VFS nach Dateien und Inhalten.

        Args:
            query: Suchbegriff oder Regex-Pattern
            path: Startverzeichnis (default: "/")
            mode: Suchmodus (FILENAME, CONTENT, BOTH)
            case_sensitive: Groß/Kleinschreibung beachten
            regex: Query als Regex interpretieren
            max_results: Maximale Anzahl Ergebnisse
            file_extensions: Nur diese Extensions durchsuchen (z.B. [".py", ".md"])
            exclude_patterns: Pfade die diese Patterns enthalten überspringen

        Returns:
            Dict mit success, results (Liste von SearchResult dicts), total_matches
        """
        results: list[SearchResult] = []

        # Query vorbereiten
        if regex:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(query, flags)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex: {e}"}

            def matches(text: str) -> bool:
                return bool(pattern.search(text))

            def find_in_line(line: str) -> Optional[str]:
                match = pattern.search(line)
                return match.group(0) if match else None
        else:
            if not case_sensitive:
                query_lower = query.lower()

                def matches(text: str) -> bool:
                    return query_lower in text.lower()
            else:
                def matches(text: str) -> bool:
                    return query in text

            def find_in_line(line: str) -> Optional[str]:
                if matches(line):
                    return query
                return None

        # Exclude Pattern Check
        exclude_patterns = exclude_patterns or []

        def should_exclude(filepath: str) -> bool:
            for pattern in exclude_patterns:
                if pattern in filepath:
                    return True
            return False

        # Extension Filter
        def check_extension(filename: str) -> bool:
            if not file_extensions:
                return True
            for ext in file_extensions:
                if filename.endswith(ext):
                    return True
            return False

        # Rekursiv durchsuchen
        def search_directory(dir_path: str):
            if len(results) >= max_results:
                return

            # Liste Verzeichnisinhalt
            ls_result = self.ls(dir_path, recursive=False, show_hidden=True)
            if not ls_result.get("success"):
                return

            for item in ls_result.get("contents", []):
                if len(results) >= max_results:
                    return

                item_path = item["path"]

                # Exclude Check
                if should_exclude(item_path):
                    continue

                if item["type"] == "directory":
                    search_directory(item_path)
                else:
                    # Datei
                    filename = item["name"]

                    # Extension Check
                    if not check_extension(filename):
                        continue

                    # Filename Search
                    if mode in (SearchMode.FILENAME, SearchMode.BOTH):
                        if matches(filename):
                            results.append(SearchResult(
                                path=item_path,
                                filename=filename,
                                match_type="filename",
                                score=1.0 if query.lower() == filename.lower() else 0.8,
                            ))

                    # Content Search
                    if mode in (SearchMode.CONTENT, SearchMode.BOTH):
                        # Datei lesen
                        read_result = self.read(item_path)
                        if read_result.get("success"):
                            content = read_result.get("content", "")

                            # Zeilenweise durchsuchen
                            for line_num, line in enumerate(content.split("\n"), 1):
                                if len(results) >= max_results:
                                    break

                                match_text = find_in_line(line)
                                if match_text:
                                    results.append(SearchResult(
                                        path=item_path,
                                        filename=filename,
                                        match_type="content",
                                        line_number=line_num,
                                        line_content=line.strip()[:200],  # Truncate
                                        score=0.5,
                                    ))

        # Suche starten
        try:
            search_directory(self._normalize_path(path))
        except Exception as e:
            return {"success": False, "error": f"Search error: {e}"}

        # Ergebnisse sortieren (höchster Score zuerst)
        results.sort(key=lambda r: (-r.score, r.path))

        return {
            "success": True,
            "results": [r.to_dict() for r in results],
            "total_matches": len(results),
            "query": query,
            "mode": mode.value,
        }

    def find_files(
        self,
        pattern: str,
        path: str = "/",
        max_results: int = 100,
    ) -> dict:
        """
        Sucht Dateien nach Filename-Pattern (glob-style).

        Args:
            pattern: Glob-Pattern (z.B. "*.py", "test_*.md")
            path: Startverzeichnis
            max_results: Maximale Anzahl

        Returns:
            Dict mit success und files (Liste von Pfaden)
        """
        import fnmatch

        files = []

        def search_dir(dir_path: str):
            if len(files) >= max_results:
                return

            ls_result = self.ls(dir_path, recursive=False, show_hidden=True)
            if not ls_result.get("success"):
                return

            for item in ls_result.get("contents", []):
                if len(files) >= max_results:
                    return

                if item["type"] == "directory":
                    search_dir(item["path"])
                else:
                    if fnmatch.fnmatch(item["name"], pattern):
                        files.append(item["path"])

        try:
            search_dir(self._normalize_path(path))
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "files": files,
            "total": len(files),
            "pattern": pattern,
        }

    def grep(
        self,
        pattern: str,
        path: str = "/",
        file_pattern: str = "*",
        context_lines: int = 0,
        max_results: int = 100,
    ) -> dict:
        """
        Grep-ähnliche Suche in Dateien.

        Args:
            pattern: Regex-Pattern zum Suchen
            path: Startverzeichnis
            file_pattern: Glob für Dateinamen (z.B. "*.py")
            context_lines: Anzahl Zeilen vor/nach Match
            max_results: Maximale Anzahl Matches

        Returns:
            Dict mit success und matches
        """
        import fnmatch

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {"success": False, "error": f"Invalid pattern: {e}"}

        matches = []

        def search_file(file_path: str, filename: str):
            if len(matches) >= max_results:
                return

            read_result = self.read(file_path)
            if not read_result.get("success"):
                return

            lines = read_result.get("content", "").split("\n")

            for i, line in enumerate(lines):
                if len(matches) >= max_results:
                    return

                if regex.search(line):
                    # Context sammeln
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)

                    context = {
                        "before": lines[start:i],
                        "match": line,
                        "after": lines[i + 1:end],
                    } if context_lines > 0 else None

                    matches.append({
                        "path": file_path,
                        "filename": filename,
                        "line_number": i + 1,
                        "line": line.strip()[:500],
                        "context": context,
                    })

        def search_dir(dir_path: str):
            if len(matches) >= max_results:
                return

            ls_result = self.ls(dir_path, recursive=False, show_hidden=True)
            if not ls_result.get("success"):
                return

            for item in ls_result.get("contents", []):
                if len(matches) >= max_results:
                    return

                if item["type"] == "directory":
                    search_dir(item["path"])
                else:
                    if fnmatch.fnmatch(item["name"], file_pattern):
                        search_file(item["path"], item["name"])

        try:
            search_dir(self._normalize_path(path))
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "matches": matches,
            "total": len(matches),
            "pattern": pattern,
        }


def add_search_to_vfs(vfs: Any) -> Any:
    """
    Fügt Search-Methoden zu einer bestehenden VFS-Instanz hinzu.

    Args:
        vfs: VirtualFileSystemV2 Instanz

    Returns:
        Die gleiche VFS-Instanz mit neuen Methoden

    Usage:
        from vfs_search import add_search_to_vfs

        vfs = VirtualFileSystemV2()
        add_search_to_vfs(vfs)

        results = vfs.search("TODO", mode=SearchMode.CONTENT)
    """
    import types

    # Methoden binden
    vfs.search = types.MethodType(VFSSearchMixin.search, vfs)
    vfs.find_files = types.MethodType(VFSSearchMixin.find_files, vfs)
    vfs.grep = types.MethodType(VFSSearchMixin.grep, vfs)

    return vfs


# =============================================================================
# AGENT TOOL REGISTRATION
# =============================================================================

def register_vfs_search_tools(agent: Any, vfs: Any):
    """
    Registriert VFS Search als Agent Tools.

    Args:
        agent: FlowAgent Instanz
        vfs: VFS Instanz (mit search erweitert)
    """

    # Sicherstellen dass search existiert
    if not hasattr(vfs, 'search'):
        add_search_to_vfs(vfs)

    def vfs_search(
        query: str,
        path: str = "/",
        mode: str = "both",
        case_sensitive: bool = False,
        max_results: int = 20,
    ) -> str:
        """
        Search files and content in the virtual filesystem.

        Args:
            query: Search term
            path: Starting directory (default: "/")
            mode: "filename", "content", or "both"
            case_sensitive: Match case (default: False)
            max_results: Maximum results (default: 20)

        Returns:
            JSON with search results
        """
        import json

        mode_map = {
            "filename": SearchMode.FILENAME,
            "content": SearchMode.CONTENT,
            "both": SearchMode.BOTH,
        }

        result = vfs.search(
            query=query,
            path=path,
            mode=mode_map.get(mode, SearchMode.BOTH),
            case_sensitive=case_sensitive,
            max_results=max_results,
        )

        return json.dumps(result, indent=2)

    def vfs_find(pattern: str, path: str = "/") -> str:
        """
        Find files by glob pattern.

        Args:
            pattern: Glob pattern (e.g. "*.py", "test_*.md")
            path: Starting directory

        Returns:
            JSON list of matching file paths
        """
        import json
        result = vfs.find_files(pattern=pattern, path=path)
        return json.dumps(result, indent=2)

    def vfs_grep(pattern: str, path: str = "/", file_pattern: str = "*") -> str:
        """
        Search for regex pattern in file contents (like grep).

        Args:
            pattern: Regex pattern to search
            path: Starting directory
            file_pattern: Glob for filenames (e.g. "*.py")

        Returns:
            JSON with matching lines
        """
        import json
        result = vfs.grep(
            pattern=pattern,
            path=path,
            file_pattern=file_pattern,
        )
        return json.dumps(result, indent=2)

    # Tools registrieren
    agent.add_tool(
        vfs_search,
        "vfs_search",
        description="Search for files and content in the virtual filesystem. Use mode='filename' to search only filenames, 'content' for content, or 'both'.",
        category=["vfs", "search"],
    )

    agent.add_tool(
        vfs_find,
        "vfs_find",
        description="Find files by glob pattern. Example: vfs_find('*.py') finds all Python files.",
        category=["vfs", "search"],
    )

    agent.add_tool(
        vfs_grep,
        "vfs_grep",
        description="Search for regex pattern in file contents, similar to grep. Returns matching lines with line numbers.",
        category=["vfs", "search"],
    )
