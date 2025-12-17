"""
Markdown Documentation System - Refactored v2.1
================================================
Modular, async, memory-efficient documentation management.

Fixes in v2.1:
- Inverted Index for O(1) keyword lookups
- Proper error logging instead of swallowing
- JS/TS support via RegexAnalyzer

Architecture:
- DataModels: __slots__ dataclasses for minimal RAM
- DocParser: State-machine parser (code-block aware)
- CodeAnalyzer: AST-based extraction with visitor pattern
- JSTSAnalyzer: Regex-based JS/TS extraction
- IndexManager: Thread-safe persistence with atomic writes
- ContextEngine: Inverted index for fast lookups
- DocsSystem: Facade orchestrating all components
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Iterator, Any, Callable, Coroutine
from collections import defaultdict

from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS - Memory Efficient with __slots__
# =============================================================================


class ChangeType(Enum):
    ADDED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RENAMED = auto()


@dataclass(slots=True)
class DocSection:
    """Documentation section with minimal memory footprint."""

    section_id: str
    file_path: str
    title: str
    content: str
    level: int
    line_start: int
    line_end: int
    content_hash: str
    last_modified: float
    source_refs: tuple = ()
    tags: tuple = ()
    doc_style: str = "markdown"


@dataclass(slots=True)
class CodeElement:
    """Code element (class/function/method)."""

    name: str
    element_type: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    content_hash: str
    language: str = "python"
    docstring: Optional[str] = None
    parent_class: Optional[str] = None


@dataclass(slots=True)
class FileChange:
    """Git file change."""

    file_path: str
    change_type: ChangeType
    old_path: Optional[str] = None


@dataclass
class InvertedIndex:
    """Inverted index for fast keyword lookups."""

    # keyword -> set of section_ids
    keyword_to_sections: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # tag -> set of section_ids
    tag_to_sections: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # file_path -> set of section_ids
    file_to_sections: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # name -> set of element_ids (for code)
    name_to_elements: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # type -> set of element_ids
    type_to_elements: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # file -> set of element_ids
    file_to_elements: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def clear(self):
        """Clear all indexes."""
        self.keyword_to_sections.clear()
        self.tag_to_sections.clear()
        self.file_to_sections.clear()
        self.name_to_elements.clear()
        self.type_to_elements.clear()
        self.file_to_elements.clear()


@dataclass
class DocsIndex:
    """Complete documentation index."""

    sections: Dict[str, DocSection] = field(default_factory=dict)
    code_elements: Dict[str, CodeElement] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    inverted: InvertedIndex = field(default_factory=InvertedIndex)
    last_git_commit: Optional[str] = None
    last_indexed: float = field(default_factory=time.time)
    version: str = "2.1"

# =============================================================================
# NEW DATA MODELS & TYPES
# =============================================================================

class ContextBundle(dict):
    """
    Token-optimized context dictionary for LLMs.
    Structure:
    {
        "intent": str,
        "focus_files": { path: content },
        "definitions": [ { signature, docstring, ... } ],
        "graph": {
            "upstream": [ { name, file, type } ],   # Dependencies (Imports)
            "downstream": [ { name, file, usage } ] # Usage (Callers)
        },
        "documentation": [ { title, content_snippet, relevance } ]
    }
    """

# =============================================================================
# DOC PARSER - State Machine for Robust Parsing
# =============================================================================


class ParserState(Enum):
    """Parser states for state machine."""

    NORMAL = auto()
    CODE_BLOCK = auto()
    FRONTMATTER = auto()


class DocParser:
    """
    State-machine based document parser.
    Supports: Markdown, RST-style, YAML frontmatter, code-block aware.
    """

    PATTERN_ATX = re.compile(r"^(#{1,6})\s+(.+)$")
    CODE_FENCE = re.compile(r"^(`{3,}|~{3,})")
    FRONTMATTER = re.compile(r"^---\s*$")
    TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_-]{1,30})(?:\s|$)")
    REF_PATTERN = re.compile(r"`([^`]+\.py(?::[^`]+)?)`")

    __slots__ = ("_cache",)

    def __init__(self):
        self._cache: Dict[str, Tuple[float, List[DocSection]]] = {}

    def parse(self, file_path: Path, use_cache: bool = True) -> List[DocSection]:
        """Parse document file into sections."""
        path_str = str(file_path)

        try:
            mtime = file_path.stat().st_mtime
        except OSError as e:
            logger.warning(f"Cannot stat file {file_path}: {e}")
            return []

        if use_cache and path_str in self._cache:
            cached_mtime, cached_sections = self._cache[path_str]
            if cached_mtime == mtime:
                return cached_sections

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.warning(f"Cannot read file {file_path}: {e}")
            return []

        if not content.strip():
            return []

        style = self._detect_style(content)
        sections = self._parse_with_state_machine(file_path, content, style, mtime)

        self._cache[path_str] = (mtime, sections)
        return sections

    def _detect_style(self, content: str) -> str:
        """Auto-detect documentation style."""
        lines = content[:2000].split("\n")

        has_atx = any(self.PATTERN_ATX.match(line) for line in lines[:50])
        has_rst = any(re.match(r"^[=\-~]{3,}\s*$", line) for line in lines[:50])
        has_frontmatter = lines[0].strip() == "---" if lines else False

        if has_frontmatter:
            return "yaml_md"
        if has_rst and not has_atx:
            return "rst"
        return "markdown"

    def _parse_with_state_machine(
        self, file_path: Path, content: str, style: str, mtime: float
    ) -> List[DocSection]:
        """State machine parser - handles code blocks correctly."""
        sections: List[DocSection] = []
        lines = content.split("\n")

        state = ParserState.NORMAL
        fence_char = ""
        fence_len = 0

        current_title: Optional[str] = None
        current_level = 0
        current_lines: List[str] = []
        section_start = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            if state == ParserState.NORMAL:
                if i == 0 and self.FRONTMATTER.match(line):
                    state = ParserState.FRONTMATTER
                    i += 1
                    continue

                fence_match = self.CODE_FENCE.match(line)
                if fence_match:
                    fence_char = fence_match.group(1)[0]
                    fence_len = len(fence_match.group(1))
                    state = ParserState.CODE_BLOCK
                    if current_title:
                        current_lines.append(line)
                    i += 1
                    continue

                header = self._extract_header(line, lines, i, style)
                if header:
                    title, level, skip_lines = header

                    if current_title is not None:
                        section = self._create_section(
                            file_path,
                            current_title,
                            current_level,
                            current_lines,
                            section_start,
                            i - 1,
                            mtime,
                            style,
                        )
                        if section:
                            sections.append(section)

                    current_title = title
                    current_level = level
                    current_lines = []
                    section_start = i
                    i += skip_lines
                    continue

                if current_title is not None:
                    current_lines.append(line)

            elif state == ParserState.CODE_BLOCK:
                if current_title:
                    current_lines.append(line)

                if (
                    line.startswith(fence_char * fence_len)
                    and len(line.strip()) <= fence_len + 1
                ):
                    state = ParserState.NORMAL

            elif state == ParserState.FRONTMATTER:
                if self.FRONTMATTER.match(line):
                    state = ParserState.NORMAL

            i += 1

        if current_title is not None:
            section = self._create_section(
                file_path,
                current_title,
                current_level,
                current_lines,
                section_start,
                len(lines) - 1,
                mtime,
                style,
            )
            if section:
                sections.append(section)

        return sections

    def _extract_header(
        self, line: str, lines: List[str], idx: int, style: str
    ) -> Optional[Tuple[str, int, int]]:
        """Extract header from line(s). Returns (title, level, lines_to_skip)."""
        match = self.PATTERN_ATX.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip().rstrip("#").strip()
            return (title, level, 1) if title else None

        if idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if re.match(r"^={3,}\s*$", next_line) and line.strip():
                return (line.strip(), 1, 2)
            if re.match(r"^-{3,}\s*$", next_line) and line.strip():
                return (line.strip(), 2, 2)

        if style == "rst" and idx + 2 < len(lines):
            if re.match(r"^[=\-~`]{3,}$", line):
                title = lines[idx + 1].strip()
                underline = lines[idx + 2] if idx + 2 < len(lines) else ""
                if title and re.match(r"^[=\-~`]{3,}$", underline):
                    level = {"=": 1, "-": 2, "~": 3}.get(line[0], 2)
                    return (title, level, 3)

        return None

    def _create_section(
        self,
        file_path: Path,
        title: str,
        level: int,
        content_lines: List[str],
        line_start: int,
        line_end: int,
        mtime: float,
        style: str,
    ) -> Optional[DocSection]:
        """Create DocSection from parsed data."""
        content = "\n".join(content_lines).strip()
        if len(content) < 5:
            return None

        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        tags = tuple(set(self.TAG_PATTERN.findall(content)))
        refs = tuple(set(self.REF_PATTERN.findall(content)))

        return DocSection(
            section_id=f"{file_path.name}#{title}",
            file_path=str(file_path),
            title=title,
            content=content,
            level=level,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
            last_modified=mtime,
            source_refs=refs,
            tags=tags,
            doc_style=style,
        )

    def clear_cache(self):
        """Clear parser cache."""
        self._cache.clear()


# =============================================================================
# CODE ANALYZER - AST-based with Visitor Pattern (Python)
# =============================================================================


class CodeAnalyzer:
    """Efficient AST-based code analyzer using visitor pattern for Python."""

    __slots__ = ("_cache",)

    def __init__(self):
        self._cache: Dict[str, Tuple[float, List[CodeElement]]] = {}

    def analyze(self, file_path: Path, use_cache: bool = True) -> List[CodeElement]:
        """Analyze Python file for code elements."""
        path_str = str(file_path)

        try:
            mtime = file_path.stat().st_mtime
        except OSError as e:
            logger.warning(f"Cannot stat Python file {file_path}: {e}")
            return []

        if use_cache and path_str in self._cache:
            cached_mtime, cached = self._cache[path_str]
            if cached_mtime == mtime:
                return cached

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
            elements = list(self._visit(tree, file_path))
            self._cache[path_str] = (mtime, elements)
            return elements
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e.msg} at line {e.lineno}")
            return []
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error analyzing {file_path}: {e}")
            return []

    def _visit(self, tree: ast.AST, file_path: Path) -> Iterator[CodeElement]:
        """Visit AST nodes once, extracting all elements."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                yield self._class_element(node, file_path)

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield self._method_element(item, node.name, file_path)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if this function is a method (inside a class)
                is_method = False
                for p in ast.walk(tree):
                    if isinstance(p, ast.ClassDef):
                        body = getattr(p, "body", None)
                        # Ensure body is iterable (list)
                        if isinstance(body, list) and node in body:
                            is_method = True
                            break
                if not is_method:
                    yield self._function_element(node, file_path)

    def _class_element(self, node: ast.ClassDef, file_path: Path) -> CodeElement:
        """Create CodeElement for class."""
        bases = ", ".join(self._get_name(b) for b in node.bases[:3])
        sig = f"class {node.name}({bases})" if bases else f"class {node.name}"

        return CodeElement(
            name=node.name,
            element_type="class",
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            signature=sig,
            language="python",
            docstring=ast.get_docstring(node),
            content_hash=self._hash_node(node),
        )

    def _function_element(self, node: ast.FunctionDef, file_path: Path) -> CodeElement:
        """Create CodeElement for function."""
        return CodeElement(
            name=node.name,
            element_type="function",
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            signature=self._get_signature(node),
            language="python",
            docstring=ast.get_docstring(node),
            content_hash=self._hash_node(node),
        )

    def _method_element(
        self, node: ast.FunctionDef, parent: str, file_path: Path
    ) -> CodeElement:
        """Create CodeElement for method."""
        return CodeElement(
            name=node.name,
            element_type="method",
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            signature=self._get_signature(node),
            language="python",
            docstring=ast.get_docstring(node),
            parent_class=parent,
            content_hash=self._hash_node(node),
        )

    @staticmethod
    def _get_signature(node: ast.FunctionDef) -> str:
        """Extract function signature."""
        args = [a.arg for a in node.args.args[:5]]
        if len(node.args.args) > 5:
            args.append("...")
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"

    @staticmethod
    def _get_name(node: ast.expr) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return "?"

    @staticmethod
    def _hash_node(node: ast.AST) -> str:
        """Hash AST node content."""
        try:
            return hashlib.md5(ast.unparse(node).encode()).hexdigest()[:12]
        except:
            return hashlib.md5(str(node.lineno).encode()).hexdigest()[:12]

    def clear_cache(self):
        """Clear analyzer cache."""
        self._cache.clear()


# =============================================================================
# JS/TS ANALYZER - Regex-based for JavaScript/TypeScript
# =============================================================================


class JSTSAnalyzer:
    """Regex-based analyzer for JavaScript and TypeScript files."""

    # Patterns for JS/TS constructs
    PATTERNS = {
        "class": re.compile(
            r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+[\w,\s]+)?\s*\{",
            re.MULTILINE,
        ),
        "function": re.compile(
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE
        ),
        "arrow_const": re.compile(
            r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w+)?\s*=>",
            re.MULTILINE,
        ),
        "method": re.compile(
            r"^\s+(?:async\s+)?(?:static\s+)?(?:private\s+|public\s+|protected\s+)?(\w+)\s*\(([^)]*)\)\s*(?::\s*[\w<>\[\]|]+)?\s*\{",
            re.MULTILINE,
        ),
        "interface": re.compile(
            r"^(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?\s*\{",
            re.MULTILINE,
        ),
        "type": re.compile(r"^(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE),
        "jsdoc": re.compile(r"/\*\*\s*([\s\S]*?)\s*\*/", re.MULTILINE),
    }

    __slots__ = ("_cache",)

    def __init__(self):
        self._cache: Dict[str, Tuple[float, List[CodeElement]]] = {}

    def analyze(self, file_path: Path, use_cache: bool = True) -> List[CodeElement]:
        """Analyze JS/TS file for code elements."""
        path_str = str(file_path)

        try:
            mtime = file_path.stat().st_mtime
        except OSError as e:
            logger.warning(f"Cannot stat JS/TS file {file_path}: {e}")
            return []

        if use_cache and path_str in self._cache:
            cached_mtime, cached = self._cache[path_str]
            if cached_mtime == mtime:
                return cached

        try:
            content = file_path.read_text(encoding="utf-8")
            elements = self._extract_elements(content, file_path)
            self._cache[path_str] = (mtime, elements)
            return elements
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error analyzing JS/TS {file_path}: {e}")
            return []

    def _extract_elements(self, content: str, file_path: Path) -> List[CodeElement]:
        """Extract code elements from JS/TS content."""
        elements = []
        lines = content.split("\n")
        language = "typescript" if file_path.suffix == ".ts" else "javascript"

        # Extract JSDoc comments for later matching
        jsdocs = {}
        for match in self.PATTERNS["jsdoc"].finditer(content):
            end_pos = match.end()
            line_num = content[:end_pos].count("\n") + 1
            jsdocs[line_num] = self._clean_jsdoc(match.group(1))

        # Extract classes
        for match in self.PATTERNS["class"].finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            name = match.group(1)
            extends = match.group(2)
            sig = f"class {name}" + (f" extends {extends}" if extends else "")

            elements.append(
                CodeElement(
                    name=name,
                    element_type="class",
                    file_path=str(file_path),
                    line_start=line_num,
                    line_end=self._find_block_end(lines, line_num - 1),
                    signature=sig,
                    language=language,
                    docstring=jsdocs.get(line_num - 1),
                    content_hash=hashlib.md5(match.group(0).encode()).hexdigest()[:12],
                )
            )

        # Extract functions
        for match in self.PATTERNS["function"].finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            name = match.group(1)
            params = match.group(2).strip()

            elements.append(
                CodeElement(
                    name=name,
                    element_type="function",
                    file_path=str(file_path),
                    line_start=line_num,
                    line_end=self._find_block_end(lines, line_num - 1),
                    signature=f"function {name}({params})",
                    language=language,
                    docstring=jsdocs.get(line_num - 1),
                    content_hash=hashlib.md5(match.group(0).encode()).hexdigest()[:12],
                )
            )

        # Extract arrow functions (const)
        for match in self.PATTERNS["arrow_const"].finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            name = match.group(1)

            elements.append(
                CodeElement(
                    name=name,
                    element_type="function",
                    file_path=str(file_path),
                    line_start=line_num,
                    line_end=line_num,  # Arrow functions are usually single expression
                    signature=f"const {name} = () =>",
                    language=language,
                    docstring=jsdocs.get(line_num - 1),
                    content_hash=hashlib.md5(match.group(0).encode()).hexdigest()[:12],
                )
            )

        # Extract interfaces (TypeScript)
        if language == "typescript":
            for match in self.PATTERNS["interface"].finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                name = match.group(1)

                elements.append(
                    CodeElement(
                        name=name,
                        element_type="interface",
                        file_path=str(file_path),
                        line_start=line_num,
                        line_end=self._find_block_end(lines, line_num - 1),
                        signature=f"interface {name}",
                        language=language,
                        docstring=jsdocs.get(line_num - 1),
                        content_hash=hashlib.md5(match.group(0).encode()).hexdigest()[
                            :12
                        ],
                    )
                )

            # Extract type aliases
            for match in self.PATTERNS["type"].finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                name = match.group(1)

                elements.append(
                    CodeElement(
                        name=name,
                        element_type="type",
                        file_path=str(file_path),
                        line_start=line_num,
                        line_end=line_num,
                        signature=f"type {name}",
                        language=language,
                        docstring=jsdocs.get(line_num - 1),
                        content_hash=hashlib.md5(match.group(0).encode()).hexdigest()[
                            :12
                        ],
                    )
                )

        return elements

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a code block by matching braces."""
        brace_count = 0
        started = False

        for i in range(start_idx, min(start_idx + 500, len(lines))):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    started = True
                elif char == "}":
                    brace_count -= 1
                    if started and brace_count == 0:
                        return i + 1

        return start_idx + 1

    @staticmethod
    def _clean_jsdoc(doc: str) -> str:
        """Clean JSDoc comment content."""
        lines = doc.split("\n")
        cleaned = []
        for line in lines:
            line = re.sub(r"^\s*\*\s?", "", line).strip()
            if line and not line.startswith("@"):
                cleaned.append(line)
        return " ".join(cleaned)[:500] if cleaned else None

    def clear_cache(self):
        """Clear analyzer cache."""
        self._cache.clear()


# =============================================================================
# INDEX MANAGER - Thread-safe Persistence with Inverted Index
# =============================================================================


class IndexManager:
    """Thread-safe index management with atomic writes and inverted indexing."""

    __slots__ = ("index_path", "index", "_lock", "_executor", "_dirty")

    # Stop words to exclude from inverted index
    STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "here",
            "there",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "themselves",
        }
    )

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = DocsIndex()
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="idx")
        self._dirty = False

    async def load(self) -> DocsIndex:
        """Load index from disk."""
        async with self._lock:
            if not self.index_path.exists():
                return self.index

            data = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._sync_load
            )
            if data:
                self.index = self._deserialize(data)
                self._rebuild_inverted_index()
            return self.index

    def _sync_load(self) -> Optional[dict]:
        """Synchronous load (runs in thread)."""
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load index: {e}")
            return None

    async def save(self, force: bool = False):
        """Save index with atomic write pattern."""
        if not self._dirty and not force:
            return

        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._sync_save
            )
            self._dirty = False

    def _sync_save(self):
        """Synchronous atomic save (runs in thread)."""
        data = self._serialize()
        temp_path = self.index_path.with_suffix(".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
            os.replace(temp_path, self.index_path)
        except OSError as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def _serialize(self) -> dict:
        """Serialize index to dict (inverted index is rebuilt on load)."""
        return {
            "version": self.index.version,
            "last_git_commit": self.index.last_git_commit,
            "last_indexed": self.index.last_indexed,
            "file_hashes": self.index.file_hashes,
            "sections": {
                sid: {
                    "section_id": s.section_id,
                    "file_path": s.file_path,
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                    "content_hash": s.content_hash,
                    "last_modified": s.last_modified,
                    "source_refs": list(s.source_refs),
                    "tags": list(s.tags),
                    "doc_style": s.doc_style,
                }
                for sid, s in self.index.sections.items()
            },
            "code_elements": {
                eid: {
                    "name": e.name,
                    "element_type": e.element_type,
                    "file_path": e.file_path,
                    "line_start": e.line_start,
                    "line_end": e.line_end,
                    "signature": e.signature,
                    "content_hash": e.content_hash,
                    "language": e.language,
                    "docstring": e.docstring,
                    "parent_class": e.parent_class,
                }
                for eid, e in self.index.code_elements.items()
            },
        }

    def _deserialize(self, data: dict) -> DocsIndex:
        """Deserialize dict to index."""
        index = DocsIndex()
        index.version = data.get("version", "2.1")
        index.last_git_commit = data.get("last_git_commit")
        index.last_indexed = data.get("last_indexed", time.time())
        index.file_hashes = data.get("file_hashes", {})

        for sid, s in data.get("sections", {}).items():
            index.sections[sid] = DocSection(
                section_id=s["section_id"],
                file_path=s["file_path"],
                title=s["title"],
                content=s["content"],
                level=s["level"],
                line_start=s["line_start"],
                line_end=s["line_end"],
                content_hash=s["content_hash"],
                last_modified=s["last_modified"],
                source_refs=tuple(s.get("source_refs", [])),
                tags=tuple(s.get("tags", [])),
                doc_style=s.get("doc_style", "markdown"),
            )

        for eid, e in data.get("code_elements", {}).items():
            index.code_elements[eid] = CodeElement(
                name=e["name"],
                element_type=e["element_type"],
                file_path=e["file_path"],
                line_start=e["line_start"],
                line_end=e["line_end"],
                signature=e["signature"],
                content_hash=e["content_hash"],
                language=e.get("language", "python"),
                docstring=e.get("docstring"),
                parent_class=e.get("parent_class"),
            )

        return index

    def _rebuild_inverted_index(self):
        """Rebuild inverted index from loaded data."""
        self.index.inverted.clear()

        for sid, section in self.index.sections.items():
            self._index_section(sid, section)

        for eid, element in self.index.code_elements.items():
            self._index_element(eid, element)

        logger.debug(
            f"Rebuilt inverted index: {len(self.index.inverted.keyword_to_sections)} keywords"
        )

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into searchable keywords."""
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text.lower())
        return {w for w in words if w not in self.STOP_WORDS and len(w) <= 50}

    def _index_section(self, section_id: str, section: DocSection):
        """Add section to inverted index."""
        # Index keywords from title and content
        keywords = self._tokenize(f"{section.title} {section.content[:1000]}")
        for keyword in keywords:
            self.index.inverted.keyword_to_sections[keyword].add(section_id)

        # Index tags
        for tag in section.tags:
            self.index.inverted.tag_to_sections[tag.lower()].add(section_id)

        # Index by file
        self.index.inverted.file_to_sections[section.file_path].add(section_id)

    def _index_element(self, element_id: str, element: CodeElement):
        """Add code element to inverted index."""
        # Index by name (and name parts for camelCase/snake_case)
        name_parts = re.findall(
            r"[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", element.name
        )
        for part in name_parts:
            self.index.inverted.name_to_elements[part.lower()].add(element_id)
        self.index.inverted.name_to_elements[element.name.lower()].add(element_id)

        # Index by type
        self.index.inverted.type_to_elements[element.element_type].add(element_id)

        # Index by file
        self.index.inverted.file_to_elements[element.file_path].add(element_id)

    def _unindex_section(self, section_id: str, section: DocSection):
        """Remove section from inverted index."""
        keywords = self._tokenize(f"{section.title} {section.content[:1000]}")
        for keyword in keywords:
            self.index.inverted.keyword_to_sections[keyword].discard(section_id)

        for tag in section.tags:
            self.index.inverted.tag_to_sections[tag.lower()].discard(section_id)

        self.index.inverted.file_to_sections[section.file_path].discard(section_id)

    def _unindex_element(self, element_id: str, element: CodeElement):
        """Remove code element from inverted index."""
        name_parts = re.findall(
            r"[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", element.name
        )
        for part in name_parts:
            self.index.inverted.name_to_elements[part.lower()].discard(element_id)
        self.index.inverted.name_to_elements[element.name.lower()].discard(element_id)

        self.index.inverted.type_to_elements[element.element_type].discard(element_id)
        self.index.inverted.file_to_elements[element.file_path].discard(element_id)

    def mark_dirty(self):
        self._dirty = True

    def update_section(self, section: DocSection):
        """Update or add section with inverted index update."""
        old_section = self.index.sections.get(section.section_id)
        if old_section:
            self._unindex_section(section.section_id, old_section)

        self.index.sections[section.section_id] = section
        self._index_section(section.section_id, section)
        self._dirty = True

    def update_element(self, element_id: str, element: CodeElement):
        """Update or add code element with inverted index update."""
        old_element = self.index.code_elements.get(element_id)
        if old_element:
            self._unindex_element(element_id, old_element)

        self.index.code_elements[element_id] = element
        self._index_element(element_id, element)
        self._dirty = True

    def remove_file(self, file_path: str):
        """Remove all entries for a file."""
        # Remove sections
        sections_to_remove = list(
            self.index.inverted.file_to_sections.get(file_path, set())
        )
        for sid in sections_to_remove:
            if sid in self.index.sections:
                self._unindex_section(sid, self.index.sections[sid])
                del self.index.sections[sid]

        # Remove elements
        elements_to_remove = list(
            self.index.inverted.file_to_elements.get(file_path, set())
        )
        for eid in elements_to_remove:
            if eid in self.index.code_elements:
                self._unindex_element(eid, self.index.code_elements[eid])
                del self.index.code_elements[eid]

        self.index.file_hashes.pop(file_path, None)
        self._dirty = True


# =============================================================================
# CONTEXT ENGINE - Fast Search with Inverted Index
# =============================================================================


class ContextEngine:
    """Fast context lookups using inverted index for O(1) keyword search."""

    __slots__ = ("_index_mgr", "_query_cache", "_cache_ttl")

    def __init__(self, index_manager: IndexManager, cache_ttl: float = 300.0):
        self._index_mgr = index_manager
        self._query_cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl

    def search_sections(
        self,
        query: Optional[str] = None,
        file_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_results: int = 25,
    ) -> List[DocSection]:
        """Fast section search using inverted index."""
        cache_key = f"s:{query}:{file_path}:{tags}:{max_results}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        inverted = self._index_mgr.index.inverted
        candidate_ids: Optional[Set[str]] = None

        # Filter by query keywords using inverted index - O(k) where k = keyword count
        # Use same tokenization as indexing + OR semantics with ranking
        if query:
            # Tokenize query the same way as content is tokenized
            query_terms = self._index_mgr._tokenize(query)
            # Also add raw words for flexibility (handles short words)
            raw_terms = set(query.lower().split())
            all_terms = query_terms | raw_terms

            # Use OR semantics: collect all matching sections with match counts
            term_match_counts: Dict[str, int] = defaultdict(int)
            for term in all_terms:
                term_ids = inverted.keyword_to_sections.get(term, set())
                for sid in term_ids:
                    term_match_counts[sid] += 1

            if term_match_counts:
                # Sort by match count (more matches = better)
                sorted_ids = sorted(term_match_counts.keys(),
                                   key=lambda x: term_match_counts[x], reverse=True)
                candidate_ids = set(sorted_ids)

        # Filter by tags using inverted index - O(t) where t = tag count
        if tags:
            tag_ids: Set[str] = set()
            for tag in tags:
                tag_ids |= inverted.tag_to_sections.get(tag.lower(), set())
            if candidate_ids is None:
                candidate_ids = tag_ids
            else:
                candidate_ids &= tag_ids

        # Filter by file path using inverted index - O(1)
        if file_path:
            file_ids = inverted.file_to_sections.get(file_path, set())
            # Also check partial path match
            if not file_ids:
                file_ids = set()
                for fp, ids in inverted.file_to_sections.items():
                    if file_path in fp:
                        file_ids |= ids
            if candidate_ids is None:
                candidate_ids = file_ids
            else:
                candidate_ids &= file_ids

        # If no filters, return all (but limit)
        if candidate_ids is None:
            candidate_ids = set(self._index_mgr.index.sections.keys())

        # Fetch actual sections
        results = []
        for sid in candidate_ids:
            if len(results) >= max_results:
                break
            section = self._index_mgr.index.sections.get(sid)
            if section:
                results.append(section)

        self._set_cached(cache_key, results)
        return results

    def search_elements(
        self,
        name: Optional[str] = None,
        element_type: Optional[str] = None,
        file_path: Optional[str] = None,
        max_results: int = 25,
    ) -> List[CodeElement]:
        """Fast code element search using inverted index."""
        cache_key = f"e:{name}:{element_type}:{file_path}:{max_results}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        inverted = self._index_mgr.index.inverted
        candidate_ids: Optional[Set[str]] = None

        # Filter by name using inverted index - O(1)
        # Track exact matches separately for prioritization
        exact_match_ids: Set[str] = set()
        if name:
            name_lower = name.lower()
            # First get exact matches (highest priority)
            exact_match_ids = inverted.name_to_elements.get(name_lower, set()).copy()
            candidate_ids = exact_match_ids.copy() if exact_match_ids else None
            # Also check partial matches (lower priority)
            for indexed_name, ids in inverted.name_to_elements.items():
                if indexed_name != name_lower and (name_lower in indexed_name or indexed_name in name_lower):
                    if candidate_ids is None:
                        candidate_ids = ids.copy()
                    else:
                        candidate_ids |= ids

        # Filter by type using inverted index - O(1)
        if element_type:
            type_ids = inverted.type_to_elements.get(element_type, set())
            if candidate_ids is None:
                candidate_ids = type_ids.copy()
            else:
                candidate_ids &= type_ids

        # Filter by file path using inverted index - O(1)
        if file_path:
            # Normalize file path for matching (handle both / and \)
            file_path_normalized = file_path.replace("\\", "/").lower()
            file_ids = inverted.file_to_elements.get(file_path, set())
            if not file_ids:
                file_ids = set()
                for fp, ids in inverted.file_to_elements.items():
                    # Normalize indexed path for comparison
                    fp_normalized = fp.replace("\\", "/").lower()
                    # Check if the query path is contained in the indexed path
                    # or if the indexed path ends with the query path
                    if (file_path_normalized in fp_normalized or
                        fp_normalized.endswith(file_path_normalized) or
                        file_path_normalized.endswith(fp_normalized.split("/")[-1])):
                        file_ids |= ids
            if candidate_ids is None:
                candidate_ids = file_ids
            else:
                candidate_ids &= file_ids

        # If no filters, return all (but limit)
        if candidate_ids is None:
            candidate_ids = set(self._index_mgr.index.code_elements.keys())

        # Fetch actual elements with smart ranking
        all_matches = []

        for eid in candidate_ids:
            element = self._index_mgr.index.code_elements.get(eid)
            if element:
                all_matches.append(element)

        # Sort by relevance score
        def score_element(elem: CodeElement) -> tuple:
            """Score element for ranking. Higher = better. Returns tuple for multi-key sort."""
            file_path = elem.file_path.replace("\\", "/").lower()
            elem_name = elem.name.lower()
            query_name = name.lower() if name else ""

            # Exact name match is highest priority
            exact_match = 1 if elem_name == query_name else 0

            # Prefer source files over test files
            is_test = 1 if "/test" in file_path or "_test" in file_path else 0

            # Prefer Python files when searching for Python-like names
            # (classes with CamelCase, functions with snake_case)
            is_python = 1 if file_path.endswith(".py") else 0

            # Prefer core source directories over mods/flows/clis
            # utils/system and utils/extras are primary sources
            is_core = 0
            if "/utils/system/" in file_path:
                is_core = 4  # Highest priority - core system definitions
            elif "/utils/extras/" in file_path:
                is_core = 4  # Highest priority - core extras definitions
            elif "/utils/" in file_path and "/clis/" not in file_path:
                is_core = 3  # High priority - other utility code
            elif "/mods/" in file_path:
                is_core = 2  # Medium priority - module code (actual definitions)
            elif "/src-core/" in file_path:
                is_core = 1  # Lower priority - compiled/bridge code
            elif "/clis/" in file_path or "/flows/" in file_path:
                is_core = 0  # Lowest - CLI wrappers and flows (usually imports, not definitions)
            else:
                is_core = 1

            # Prefer shorter file paths (usually more fundamental)
            path_depth = file_path.count("/")

            # Return tuple: (exact_match, not_test, is_python, is_core, -path_depth)
            # Higher values = better match
            return (exact_match, 1 - is_test, is_python, is_core, -path_depth)

        all_matches.sort(key=score_element, reverse=True)

        results = all_matches[:max_results]

        self._set_cached(cache_key, results)
        return results

    def get_context_for_element(self, element_id: str) -> dict:
        """Get comprehensive context for a code element."""
        element = self._index_mgr.index.code_elements.get(element_id)
        if not element:
            return {}

        related_docs = []
        for section in self._index_mgr.index.sections.values():
            if (
                element_id in section.source_refs
                or element.name in section.title
                or element.name in section.content[:300]
            ):
                related_docs.append(
                    {
                        "section_id": section.section_id,
                        "title": section.title,
                        "relevance": self._calc_relevance(element, section),
                    }
                )

        related_docs.sort(key=lambda x: x["relevance"], reverse=True)

        related_elements = []
        for eid, e in self._index_mgr.index.code_elements.items():
            if eid == element_id:
                continue
            if e.file_path == element.file_path:
                if (
                    e.parent_class == element.parent_class
                    or e.name == element.parent_class
                ):
                    related_elements.append(eid)

        return {
            "element": {
                "id": element_id,
                "name": element.name,
                "type": element.element_type,
                "signature": element.signature,
                "file": element.file_path,
                "language": element.language,
                "lines": (element.line_start, element.line_end),
            },
            "documentation": related_docs[:5],
            "related_elements": related_elements[:10],
        }

    def _calc_relevance(self, element: CodeElement, section: DocSection) -> float:
        score = 0.0
        if element.name in section.title:
            score += 5.0
        if element.name in section.source_refs:
            score += 3.0
        if element.name in section.content:
            score += 1.0
        if element.file_path in section.file_path:
            score += 2.0
        return score

    def _get_cached(self, key: str) -> Optional[Any]:
        if key in self._query_cache:
            ts, value = self._query_cache[key]
            if time.time() - ts < self._cache_ttl:
                return value
            del self._query_cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        if len(self._query_cache) > 100:
            oldest = min(self._query_cache.items(), key=lambda x: x[1][0])
            del self._query_cache[oldest[0]]
        self._query_cache[key] = (time.time(), value)

    def clear_cache(self):
        self._query_cache.clear()

    # Add new logic for Graph-based Context
    def get_context_for_task(
        self, files: List[str], intent: str, max_tokens: int = 8000
    ) -> ContextBundle:
        """
        Generates a graph-based context bundle optimized for an LLM task.

        1. Loads code elements for focus files.
        2. Resolves Upstream (what these files need).
        3. Resolves Downstream (what uses these files).
        4. Finds relevant docs based on code entities AND intent.
        """
        # Normalize paths
        focus_paths = {str(Path(f).resolve()) for f in files}
        relative_paths = [str(Path(f)) for f in files]

        # 1. Analyze Focus Files
        focus_elements = []
        focus_names = set()

        for eid, elem in self._index_mgr.index.code_elements.items():
            # Check if element belongs to focus files (absolute or relative match)
            if any(str(Path(elem.file_path).resolve()) == fp for fp in focus_paths):
                focus_elements.append(elem)
                focus_names.add(elem.name)

        # 2. Build Dependency Graph (Just-In-Time)
        upstream = self._resolve_upstream(focus_elements)
        downstream = self._resolve_downstream(focus_names, exclude_paths=focus_paths)

        # 3. Find Relevant Documentation
        # Combine intent keywords + focus element names for doc search
        search_query = f"{intent} {' '.join(focus_names)}"
        docs = self.search_sections(query=search_query, max_results=10)

        # Filter docs: prioritize those explicitly referencing focus files/elements
        relevant_docs = []
        for doc in docs:
            score = 0
            # Higher score if doc references our code
            if any(name in doc.content for name in focus_names):
                score += 5
            if any(path in doc.file_path for path in relative_paths):
                score += 5
            # Base score from intent match
            score += 1

            relevant_docs.append(
                {
                    "title": doc.title,
                    "file": doc.file_path,
                    "content": self._truncate_content(
                        doc.content, 500
                    ),  # Token efficient
                    "score": score,
                }
            )

        relevant_docs.sort(key=lambda x: x["score"], reverse=True)

        # 4. Assemble Bundle (Token Optimization)
        bundle = ContextBundle(
            {
                "task_intent": intent,
                "focus_code": {
                    # We assume file content reading happens in System or here if needed
                    # Here we just list the analyzed elements to save tokens vs full file
                    fp: [
                        e.signature
                        for e in focus_elements
                        if str(Path(e.file_path)) == str(Path(fp))
                    ]
                    for fp in relative_paths
                },
                "context_graph": {
                    "upstream_dependencies": [
                        {"name": u.name, "file": u.file_path, "type": u.element_type}
                        for u in upstream[:10]  # Limit for tokens
                    ],
                    "downstream_usages": [
                        {
                            "name": d["element"].name,
                            "file": d["element"].file_path,
                            "context": "caller",
                        }
                        for d in downstream[:10]
                    ],
                },
                "relevant_docs": relevant_docs[:5],
            }
        )

        return bundle

    def _resolve_upstream(
        self, focus_elements: List[CodeElement]
    ) -> List[CodeElement]:
        """
        Find dependencies: What do the focus elements call/inherit/use?
        Strategy: Look for known element names inside the focus file content.
        """
        dependencies = []
        # Get all known names in the index (excluding the focus elements themselves)
        all_known_names = self._index_mgr.index.inverted.name_to_elements

        # We need the content of the focus files to check for usage
        # This is a simplified check. A full AST traversal for calls is better but expensive.
        for elem in focus_elements:
            try:
                # Read specific lines of the element
                path = Path(elem.file_path)
                if not path.exists():
                    continue

                # Naive: Read full file (cached by OS usually), extract lines
                # Optimization: In a real persistent system, cache content or AST Analysis result
                lines = path.read_text(encoding="utf-8").splitlines()
                code_snippet = "\n".join(lines[elem.line_start - 1 : elem.line_end])

                # Check which known global names appear in this snippet
                # Tokenization similar to Inverted Index building
                tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code_snippet))

                for token in tokens:
                    if token in all_known_names and token != elem.name:
                        # Found a dependency! Get the element definition
                        # Resolve ambiguous names (multiple files might have 'utils')
                        # Heuristic: Prefer same directory or utils
                        possible_ids = all_known_names[token]
                        for eid in possible_ids:
                            dep_elem = self._index_mgr.index.code_elements.get(eid)
                            if dep_elem and dep_elem.file_path != elem.file_path:
                                dependencies.append(dep_elem)
                                break  # Take first match for now
            except Exception:
                continue

        # Deduplicate
        unique_deps = {e.content_hash: e for e in dependencies}
        return list(unique_deps.values())

    def _resolve_downstream(
        self, focus_names: Set[str], exclude_paths: Set[str]
    ) -> List[dict]:
        """
        Find usage: Who calls/uses the focus elements?
        Strategy: Search inverted index or file contents for focus_names.
        """
        usages = []

        # Use Inverted Index for fast candidate finding
        # keyword_to_sections tracks Docs, but we need Code usage.
        # We iterate over other code elements and check their definitions/bodies?
        # Too slow.

        # Fast path: Check specific files that likely import these modules
        # (This implies we need an Import Graph, which we approximate here)

        for name in focus_names:
            # We look for files containing this name textually
            # This relies on the FileScanner or IndexManager having a "files_containing_token" map
            # Since we don't have that in v2.1, we iterate code elements names (definitions)
            # and check if they *contain* our name? No.

            # Fallback: Scan known code elements to see if their *signatures* or *docstrings*
            # mention the focus name (e.g. type hinting `def foo(bar: FocusClass)`)

            for eid, elem in self._index_mgr.index.code_elements.items():
                if str(Path(elem.file_path).resolve()) in exclude_paths:
                    continue

                # Check signature for type usage or docstring for references
                if (name in elem.signature) or (
                    elem.docstring and name in elem.docstring
                ):
                    usages.append({"element": elem, "match": "signature_or_doc"})

        return usages

    def _truncate_content(self, content: str, limit: int) -> str:
        """Helper to keep context bundle small."""
        if len(content) <= limit:
            return content
        return content[:limit] + "... (truncated)"


# =============================================================================
# FILE SCANNER - Efficient File Discovery
# =============================================================================

def iter_files(root: Path, suffixes: Set[str], exclude_dirs: Set[str]):
    stack = [root] if isinstance(root, Path) else root

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in exclude_dirs:
                            stack.append(Path(entry.path))
                    else:
                        if Path(entry.name).suffix in suffixes:
                            yield Path(entry.path)
        except PermissionError:
            pass

class FileScanner:
    """Fast file discovery with filtering."""

    DEFAULT_EXCLUDES = frozenset(
        {
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".tox",
            ".next",
            ".nuxt",
            "target",
            ".gradle",
            ".idea",
            ".vscode",
            ".coverage",
            "coverage",
            ".cache",
            "temp",
            "tmp",
        }
    )

    __slots__ = ("root", "docs_root", "include_dirs", "exclude_dirs", "_file_cache")

    def __init__(
        self,
        root: Path,
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[Set[str]] = None,
        docs_root: Optional[Path] = None,
    ):
        self.root = root
        self.docs_root = docs_root
        self.include_dirs = include_dirs
        self.exclude_dirs = exclude_dirs or self.DEFAULT_EXCLUDES
        self._file_cache: Optional[Tuple[float, List[Path]]] = None

    def scan(self, extensions: Set[str], use_cache: bool = True, show_tqdm: bool = True
    ) -> List[Path]:
        """Scan for files with given extensions."""
        if use_cache and self._file_cache:
            cache_time, cached_files = self._file_cache
            if time.time() - cache_time < 60:
                return [f for f in cached_files if f.suffix in extensions]

        files = []
        search_roots = self._get_search_roots()

        for search_root in (search_roots if not show_tqdm else tqdm(search_roots, desc="Scanning files", unit="dir", total=len(search_roots))):
            print(search_root)
            for path in iter_files(search_root, extensions, self.exclude_dirs):
                if path.is_file() and self._should_include(path):
                    files.append(path)

        self._file_cache = (time.time(), files)
        return [f for f in files if f.suffix in extensions]

    def _get_search_roots(self) -> List[Path]:
        if not self.include_dirs:
            return [self.root, self.docs_root]

        roots = [self.docs_root]
        for include in self.include_dirs:
            path = self.root / include
            if path.exists() and path.is_dir():
                roots.append(path)

        return roots or [self.root, self.docs_root]

    def _should_include(self, path: Path) -> bool:
        """Check if file should be included (exclude only check)."""
        parts = path.parts
        return not any(exc in parts for exc in self.exclude_dirs)

    def get_file_hash(self, path: Path) -> str:
        try:
            stat = path.stat()
            return hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[
                :12
            ]
        except OSError:
            return ""

    def clear_cache(self):
        self._file_cache = None


# =============================================================================
# GIT TRACKER - Async Change Detection
# =============================================================================


class GitTracker:
    """Async git change detection."""

    __slots__ = ("root",)

    def __init__(self, root: Path):
        self.root = root

    async def get_commit_hash(self) -> Optional[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                cwd=self.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode().strip() if proc.returncode == 0 else None
        except (asyncio.TimeoutError, FileNotFoundError):
            return None

    async def get_changes(self, since_commit: Optional[str] = None) -> List[FileChange]:
        try:
            cmd = (
                ["git", "diff", "--name-status", f"{since_commit}..HEAD"]
                if since_commit
                else ["git", "ls-files"]
            )

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)

            if proc.returncode != 0:
                return []

            return self._parse_changes(stdout.decode(), bool(since_commit))
        except (asyncio.TimeoutError, FileNotFoundError):
            return []

    def _parse_changes(self, output: str, has_status: bool) -> List[FileChange]:
        changes = []

        for line in output.strip().split("\n")[:500]:
            if not line:
                continue

            if has_status:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                status, path = parts[0], parts[-1]
                change_type = {
                    "A": ChangeType.ADDED,
                    "M": ChangeType.MODIFIED,
                    "D": ChangeType.DELETED,
                }.get(status[0], ChangeType.MODIFIED)
                old_path = parts[1] if status.startswith("R") and len(parts) > 2 else None
                changes.append(FileChange(path, change_type, old_path))
            else:
                changes.append(FileChange(line.strip(), ChangeType.ADDED))

        return changes


# =============================================================================
# DOCS SYSTEM - Main Facade
# =============================================================================


class DocsSystem:
    """Main documentation system facade with multi-language support."""

    # Supported file extensions
    DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".txt"}
    PYTHON_EXTENSIONS = {".py", ".pyw"}
    JSTS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

    def __init__(
        self,
        project_root: Path,
        docs_root: Path,
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[Set[str]] = None,
        extensions: Optional[Dict[str, Set[str]]] = None,
    ):
        self.project_root = project_root
        self.docs_root = docs_root
        if extensions:
            for k, v in extensions.items():
                if k == "doc":
                    self.DOC_EXTENSIONS = v
                elif k == "python":
                    self.PYTHON_EXTENSIONS = v
                elif k == "jsts":
                    self.JSTS_EXTENSIONS = v

        self.scanner = FileScanner(project_root, include_dirs, exclude_dirs, docs_root=docs_root)
        self.doc_parser = DocParser()
        self.code_analyzer = CodeAnalyzer()
        self.jsts_analyzer = JSTSAnalyzer()
        self.index_mgr = IndexManager(docs_root / ".docs_index.json")
        self.context = ContextEngine(self.index_mgr)
        self.git = GitTracker(project_root)

        self.docs_root.mkdir(exist_ok=True)

    async def initialize(self, force_rebuild: bool = False, show_tqdm=False) -> dict:
        """Initialize or load documentation index."""
        start = time.perf_counter()

        if not force_rebuild:
            await self.index_mgr.load()
            if self.index_mgr.index.sections or self.index_mgr.index.code_elements:
                return {
                    "status": "loaded",
                    "sections": len(self.index_mgr.index.sections),
                    "elements": len(self.index_mgr.index.code_elements),
                    "time_ms": (time.perf_counter() - start) * 1000,
                }

        await self._build_index(show_tqdm=show_tqdm)
        await self.index_mgr.save(force=True)

        return {
            "status": "rebuilt",
            "sections": len(self.index_mgr.index.sections),
            "elements": len(self.index_mgr.index.code_elements),
            "time_ms": (time.perf_counter() - start) * 1000,
        }

    async def read(
        self,
        query: Optional[str] = None,
        section_id: Optional[str] = None,
        file_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_results: int = 25,
        format_type: str = "structured",
    ) -> dict:
        """Read documentation sections."""
        start = time.perf_counter()

        if not self.index_mgr.index.sections:
            await self.index_mgr.load()

        if section_id:
            section = self.index_mgr.index.sections.get(section_id)
            if not section:
                return {"error": f"Section not found: {section_id}"}
            return self._format_sections([section], format_type, start)

        sections = self.context.search_sections(query, file_path, tags, max_results)
        return self._format_sections(sections, format_type, start)

    async def write(self, action: str, **kwargs) -> dict:
        """Write/modify documentation."""
        start = time.perf_counter()

        handlers = {
            "create_file": self._handle_create_file,
            "add_section": self._handle_add_section,
            "update_section": self._handle_update_section,
            "delete_section": self._handle_delete_section,
        }

        handler = handlers.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}"}

        result = await handler(**kwargs)
        result["time_ms"] = (time.perf_counter() - start) * 1000
        await self.index_mgr.save()
        return result

    async def lookup_code(
        self,
        name: Optional[str] = None,
        element_type: Optional[str] = None,
        file_path: Optional[str] = None,
        language: Optional[str] = None,
        include_code: bool = False,
        max_results: int = 25,
    ) -> dict:
        """Look up code elements across all languages."""
        start = time.perf_counter()

        if not self.index_mgr.index.code_elements:
            await self.index_mgr.load()

        elements = self.context.search_elements(
            name, element_type, file_path, max_results
        )

        # Filter by language if specified
        if language:
            elements = [e for e in elements if e.language == language]

        results = []
        for elem in elements:
            elem_data = {
                "name": elem.name,
                "type": elem.element_type,
                "signature": elem.signature,
                "file": elem.file_path,
                "lines": (elem.line_start, elem.line_end),
                "language": elem.language,
                "parent": elem.parent_class,
                "docstring": elem.docstring[:200] if elem.docstring else None,
            }
            if include_code:
                elem_data["code"] = self._extract_code(elem)
            results.append(elem_data)

        return {
            "results": results,
            "count": len(results),
            "time_ms": (time.perf_counter() - start) * 1000,
        }

    async def get_suggestions(self, max_suggestions: int = 20) -> dict:
        """Get documentation improvement suggestions."""
        start = time.perf_counter()

        if not self.index_mgr.index.code_elements:
            await self.index_mgr.load()

        suggestions = []
        documented_names = set()
        for section in self.index_mgr.index.sections.values():
            documented_names.update(section.source_refs)
            documented_names.add(section.title.lower())

        for eid, elem in self.index_mgr.index.code_elements.items():
            if elem.name.startswith("_"):
                continue
            if (
                eid not in documented_names
                and elem.name.lower() not in documented_names
                and not elem.docstring
            ):
                priority = "high" if elem.element_type == "class" else "medium"
                suggestions.append(
                    {
                        "type": "missing_docs",
                        "element": elem.name,
                        "element_type": elem.element_type,
                        "language": elem.language,
                        "file": elem.file_path,
                        "priority": priority,
                    }
                )

        unclear_markers = {"todo", "fixme", "tbd", "placeholder"}
        for sid, section in self.index_mgr.index.sections.items():
            content_lower = section.content.lower()
            if (
                any(m in content_lower for m in unclear_markers)
                or len(section.content) < 50
            ):
                suggestions.append(
                    {
                        "type": "unclear_section",
                        "section_id": sid,
                        "title": section.title,
                        "priority": "low",
                    }
                )

        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order[x["priority"]])

        return {
            "suggestions": suggestions[:max_suggestions],
            "total": len(suggestions),
            "time_ms": (time.perf_counter() - start) * 1000,
        }

    async def sync(self) -> dict:
        """Sync index with file system changes."""
        start = time.perf_counter()

        changes = await self.git.get_changes(self.index_mgr.index.last_git_commit)
        updated = 0

        for change in changes:
            path = self.project_root / change.file_path

            if change.change_type == ChangeType.DELETED:
                self.index_mgr.remove_file(str(path))
                updated += 1
                continue

            if not path.exists():
                continue

            new_hash = self.scanner.get_file_hash(path)
            old_hash = self.index_mgr.index.file_hashes.get(str(path))

            if new_hash != old_hash:
                await self._update_file(path)
                self.index_mgr.index.file_hashes[str(path)] = new_hash
                updated += 1

        self.index_mgr.index.last_git_commit = await self.git.get_commit_hash()
        self.index_mgr.index.last_indexed = time.time()

        if updated:
            await self.index_mgr.save()
            self.context.clear_cache()

        return {
            "changes_detected": len(changes),
            "files_updated": updated,
            "time_ms": (time.perf_counter() - start) * 1000,
        }

    async def _build_index(self, show_tqdm: bool = True):
        """Build complete index from scratch."""
        self.index_mgr.index = DocsIndex()

        # Scan and parse markdown files
        md_files = self.scanner.scan(self.DOC_EXTENSIONS, show_tqdm=show_tqdm)
        for md_file in (md_files if not show_tqdm else tqdm(md_files, desc="Indexing docs", unit="file", total=len(md_files))):
            sections = self.doc_parser.parse(md_file)
            for section in sections:
                self.index_mgr.update_section(section)
            self.index_mgr.index.file_hashes[str(md_file)] = self.scanner.get_file_hash(
                md_file
            )

        # Scan and analyze Python files
        py_files = self.scanner.scan(self.PYTHON_EXTENSIONS, show_tqdm=show_tqdm, use_cache=False)
        for py_file in (py_files if not show_tqdm else tqdm(py_files, desc="Indexing py code", unit="file", total=len(py_files))):
            elements = self.code_analyzer.analyze(py_file)
            for elem in elements:
                eid = (
                    f"{elem.file_path}:{elem.parent_class}.{elem.name}"
                    if elem.parent_class
                    else f"{elem.file_path}:{elem.name}"
                )
                self.index_mgr.update_element(eid, elem)
            self.index_mgr.index.file_hashes[str(py_file)] = self.scanner.get_file_hash(
                py_file
            )

        # Scan and analyze JS/TS files
        jsts_files = self.scanner.scan(self.JSTS_EXTENSIONS, show_tqdm=show_tqdm, use_cache=False)
        for jsts_file in (jsts_files if not show_tqdm else tqdm(jsts_files, desc="Indexing js code", unit="file", total=len(jsts_files))):
            elements = self.jsts_analyzer.analyze(jsts_file)
            for elem in elements:
                eid = (
                    f"{elem.file_path}:{elem.parent_class}.{elem.name}"
                    if elem.parent_class
                    else f"{elem.file_path}:{elem.name}"
                )
                self.index_mgr.update_element(eid, elem)
            self.index_mgr.index.file_hashes[str(jsts_file)] = self.scanner.get_file_hash(
                jsts_file
            )

        self.index_mgr.index.last_git_commit = await self.git.get_commit_hash()
        self.index_mgr.index.last_indexed = time.time()

        logger.info(
            f"Built index: {len(self.index_mgr.index.sections)} sections, "
            f"{len(self.index_mgr.index.code_elements)} code elements"
        )

    async def _update_file(self, path: Path):
        """Update index for a single file."""
        self.index_mgr.remove_file(str(path))

        if path.suffix in self.DOC_EXTENSIONS:
            sections = self.doc_parser.parse(path, use_cache=False)
            for section in sections:
                self.index_mgr.update_section(section)
        elif path.suffix in self.PYTHON_EXTENSIONS:
            elements = self.code_analyzer.analyze(path, use_cache=False)
            for elem in elements:
                eid = (
                    f"{elem.file_path}:{elem.parent_class}.{elem.name}"
                    if elem.parent_class
                    else f"{elem.file_path}:{elem.name}"
                )
                self.index_mgr.update_element(eid, elem)
        elif path.suffix in self.JSTS_EXTENSIONS:
            elements = self.jsts_analyzer.analyze(path, use_cache=False)
            for elem in elements:
                eid = (
                    f"{elem.file_path}:{elem.parent_class}.{elem.name}"
                    if elem.parent_class
                    else f"{elem.file_path}:{elem.name}"
                )
                self.index_mgr.update_element(eid, elem)

    def _format_sections(
        self, sections: List[DocSection], format_type: str, start: float
    ) -> dict:
        """Format sections for output."""
        if format_type == "markdown":
            output = []
            for s in sections[:20]:
                output.append(f"{'#' * s.level} {s.title}\n")
                output.append(s.content[:1000])
                output.append("")
            return {
                "content": "\n".join(output),
                "count": len(sections),
                "time_ms": (time.perf_counter() - start) * 1000,
            }

        return {
            "sections": [
                {
                    "id": s.section_id,
                    "title": s.title,
                    "content": s.content[:1000],
                    "file": s.file_path,
                    "level": s.level,
                    "tags": list(s.tags),
                    "refs": list(s.source_refs)[:5],
                }
                for s in sections[:20]
            ],
            "count": len(sections),
            "time_ms": (time.perf_counter() - start) * 1000,
        }

    def _extract_code(self, elem: CodeElement) -> str:
        """Extract code block for element."""
        try:
            path = Path(elem.file_path)
            lines = path.read_text(encoding="utf-8").split("\n")
            return "\n".join(lines[elem.line_start - 1 : elem.line_end])
        except:
            return ""

    async def _handle_create_file(self, file_path: str, content: str = "") -> dict:
        full_path = self.docs_root / file_path
        if full_path.exists():
            return {"error": f"File exists: {file_path}"}

        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not content:
            title = Path(file_path).stem.replace("_", " ").title()
            content = f"# {title}\n\nDocumentation for {title}.\n"

        full_path.write_text(content, encoding="utf-8")
        sections = self.doc_parser.parse(full_path, use_cache=False)
        for section in sections:
            self.index_mgr.update_section(section)

        return {"status": "created", "file": str(full_path), "sections": len(sections)}

    async def _handle_add_section(
        self,
        file_path: str,
        title: str,
        content: str,
        level: int = 2,
        position: str = "end",
    ) -> dict:
        full_path = self.docs_root / file_path
        section_md = f"\n{'#' * level} {title}\n\n{content}\n"

        if full_path.exists():
            existing = full_path.read_text(encoding="utf-8")
            new_content = (
                section_md + existing if position == "start" else existing + section_md
            )
        else:
            new_content = section_md
            full_path.parent.mkdir(parents=True, exist_ok=True)

        full_path.write_text(new_content, encoding="utf-8")

        self.index_mgr.remove_file(str(full_path))
        sections = self.doc_parser.parse(full_path, use_cache=False)
        for section in sections:
            self.index_mgr.update_section(section)

        return {"status": "added", "section": f"{file_path}#{title}"}

    async def _handle_update_section(self, section_id: str, content: str) -> dict:
        section = self.index_mgr.index.sections.get(section_id)
        if not section:
            return {"error": f"Section not found: {section_id}"}

        path = Path(section.file_path)
        lines = path.read_text(encoding="utf-8").split("\n")

        header = "#" * section.level + " " + section.title
        new_lines = [header, "", content, ""]
        lines[section.line_start : section.line_end + 1] = new_lines
        path.write_text("\n".join(lines), encoding="utf-8")

        self.index_mgr.remove_file(str(path))
        sections = self.doc_parser.parse(path, use_cache=False)
        for s in sections:
            self.index_mgr.update_section(s)

        return {"status": "updated", "section": section_id}

    async def _handle_delete_section(self, section_id: str) -> dict:
        section = self.index_mgr.index.sections.get(section_id)
        if not section:
            return {"error": f"Section not found: {section_id}"}

        path = Path(section.file_path)
        lines = path.read_text(encoding="utf-8").split("\n")
        del lines[section.line_start : section.line_end + 1]
        path.write_text("\n".join(lines), encoding="utf-8")

        self.index_mgr.remove_file(str(path))
        sections = self.doc_parser.parse(path, use_cache=False)
        for s in sections:
            self.index_mgr.update_section(s)

        return {"status": "deleted", "section": section_id}

    async def get_task_context(self, files: List[str], intent: str) -> dict:
        """
        New Endpoint: Get optimized context for a specific editing task.

        Args:
            files: List of file paths relevant to the task.
            intent: Description of what the user wants to do (e.g., "Add logging to auth").

        Returns:
            ContextBundle dictionary ready for LLM injection.
        """
        start = time.perf_counter()

        # Ensure index is loaded
        if not self.index_mgr.index.code_elements:
            await self.index_mgr.load()

        # Offload graph analysis to thread as it involves I/O and regex
        loop = asyncio.get_running_loop()
        bundle = await loop.run_in_executor(
            self.index_mgr._executor, self.context.get_context_for_task, files, intent
        )

        # Wrap in result dict
        return {
            "result": bundle,
            "meta": {
                "analyzed_files": len(files),
                "time_ms": (time.perf_counter() - start) * 1000,
            },
        }


# =============================================================================
# APP INTEGRATION
# =============================================================================


def create_docs_system(
    project_root: str = ".",
    docs_root: str = "../docs",
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
) -> DocsSystem:
    """Factory function for DocsSystem."""
    return DocsSystem(
        project_root=Path(project_root).resolve(),
        docs_root=Path(docs_root).resolve(),
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )


def add_to_app(
    app, docs_root: str = "../docs", include_dirs: Optional[List[str]] = None
) -> DocsSystem:
    """Add docs system to ToolBoxV2 app."""
    system = DocsSystem(
        project_root=Path.cwd(),
        docs_root=Path(docs_root).resolve(),
        include_dirs=include_dirs or ["toolboxv2", "flows", "mods", "utils", "docs"],
    )

    app.docs_reader = system.read
    app.docs_writer = system.write
    app.docs_lookup = system.lookup_code
    app.docs_suggestions = system.get_suggestions
    app.docs_sync = system.sync
    app.docs_init = system.initialize
    app.get_task_context = system.get_task_context

    return system
