
import asyncio
import os
import json
import hashlib
import ast
import subprocess
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

import yaml
from ..system.tb_logger import get_logger
from ..system.types import AppType, Result

logger = get_logger()


class ChangeType(Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class DocSection:
    """Represents a documentation section with change tracking"""
    section_id: str
    file_path: str
    title: str
    content: str
    level: int
    line_start: int
    line_end: int
    source_refs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    hash_signature: str = ""
    content_hash: str = ""
    last_modified: datetime = field(default_factory=datetime.now)
    change_detected: bool = False


@dataclass
class CodeElement:
    """Represents a code element (class, function, etc.)"""
    name: str
    element_type: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str] = None
    hash_signature: str = ""
    parent_class: Optional[str] = None


@dataclass
class DocsIndex:
    """Complete documentation index with section-level tracking"""
    sections: Dict[str, DocSection] = field(default_factory=dict)
    code_elements: Dict[str, CodeElement] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    section_hashes: Dict[str, str] = field(default_factory=dict)
    last_git_commit: Optional[str] = None
    last_indexed: datetime = field(default_factory=datetime.now)
    version: str = "1.1"



@dataclass
class FileChange:
    """Represents a file change detected by git"""
    file_path: str
    change_type: ChangeType
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    old_path: Optional[str] = None  # for renamed files


@dataclass
class ImportReference:
    """Represents an import/dependency reference"""
    source_file: str
    target_file: str
    import_name: str
    line_number: int
    import_type: str  # 'python', 'js', 'relative', 'absolute'


class TOCEntry:
    """Represents a table of contents entry"""

    def __init__(self, title: str, file_path: str, level: int, line_number: int):
        self.title = title
        self.file_path = file_path
        self.level = level
        self.line_number = line_number
        self.has_implementation = False
        self.is_unclear = False
        self.source_refs: List[str] = []

class DocsAnalyzer:
    """Analyzes documentation quality and completeness"""

    def __init__(self, index: DocsIndex, project_root: Path):
        self.index = index
        self.project_root = project_root

    def find_unclear_sections(self) -> List[str]:
        """Find sections with unclear or placeholder content"""
        unclear_indicators = [
            "todo", "fixme", "placeholder", "coming soon", "not implemented",
            "tbd", "under construction", "work in progress", "missing",
            "add content here", "fill this", "example here"
        ]

        unclear_sections = []

        for section_id, section in self.index.sections.items():
            content_lower = section.content.lower()

            # Check for unclear indicators
            if any(indicator in content_lower for indicator in unclear_indicators):
                unclear_sections.append(section_id)
                continue

            # Check for very short content (likely incomplete)
            if len(section.content.strip()) < 50:
                unclear_sections.append(section_id)
                continue

            # Check for sections with only code blocks (no explanation)
            code_blocks = re.findall(r'```[\s\S]*?```', section.content)
            text_without_code = re.sub(r'```[\s\S]*?```', '', section.content).strip()
            if code_blocks and len(text_without_code) < 20:
                unclear_sections.append(section_id)

        return unclear_sections

    def find_missing_implementations(self) -> List[Dict]:
        """Find TOC entries that don't have corresponding implementations"""
        missing = []

        for section_id, section in self.index.sections.items():
            # Look for function/class references that don't exist in code
            potential_refs = re.findall(r'`([A-Za-z_][A-Za-z0-9_.]*)`', section.content)

            for ref in potential_refs:
                if '.' in ref:  # Looks like a class.method reference
                    if not any(ref in element_id for element_id in self.index.code_elements.keys()):
                        missing.append({
                            "section_id": section_id,
                            "missing_ref": ref,
                            "type": "code_reference",
                            "content_context": section.content[:200] + "..."
                        })

        return missing

class GitChangeDetector:
    """Detects changes in the repository since last index update"""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def get_current_commit_hash(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def get_changed_files(self, since_commit: Optional[str] = None) -> List[FileChange]:
        """Get list of changed files since given commit with timeout"""
        changes = []

        try:
            if since_commit:
                cmd = ["git", "diff", "--name-status", f"{since_commit}..HEAD"]
            else:
                cmd = ["git", "ls-files"]

            # Add timeout to subprocess
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=20  # 20 second timeout for git operations
            )

            if result.returncode != 0:
                logger.warning(f"Git command failed with return code {result.returncode}")
                return changes

            # Process output with limits
            lines = result.stdout.strip().split('\n')[:1000]  # Limit to 1000 files

            for line in lines:
                if not line:
                    continue

                if since_commit:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        status = parts[0]
                        file_path = parts[1]

                        if status == 'A':
                            change_type = ChangeType.ADDED
                        elif status == 'M':
                            change_type = ChangeType.MODIFIED
                        elif status == 'D':
                            change_type = ChangeType.DELETED
                        elif status.startswith('R'):
                            change_type = ChangeType.RENAMED
                            old_path = parts[1] if len(parts) > 2 else None
                            file_path = parts[2] if len(parts) > 2 else parts[1]
                        else:
                            continue

                        changes.append(FileChange(
                            file_path=file_path,
                            change_type=change_type,
                            old_path=old_path if change_type == ChangeType.RENAMED else None
                        ))
                else:
                    changes.append(FileChange(
                        file_path=line.strip(),
                        change_type=ChangeType.ADDED
                    ))

        except subprocess.TimeoutExpired:
            logger.error("Git operation timed out after 20 seconds")
        except Exception as e:
            logger.error(f"Error detecting git changes: {e}")

        return changes


class ImportAnalyzer:
    """Analyzes imports and dependencies in Python and JS files"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def analyze_python_imports(self, file_path: Path) -> List[ImportReference]:
        """Analyze Python imports"""
        imports = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportReference(
                            source_file=str(file_path),
                            target_file=self._resolve_python_import(alias.name),
                            import_name=alias.name,
                            line_number=node.lineno,
                            import_type='python'
                        ))

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        target_file = self._resolve_python_import(f"{module}.{alias.name}")
                        imports.append(ImportReference(
                            source_file=str(file_path),
                            target_file=target_file,
                            import_name=f"{module}.{alias.name}",
                            line_number=node.lineno,
                            import_type='relative' if node.level > 0 else 'absolute'
                        ))

        except Exception as e:
            logger.error(f"Error analyzing Python imports in {file_path}: {e}")

        return imports

    def analyze_js_imports(self, file_path: Path) -> List[ImportReference]:
        """Analyze JavaScript/TypeScript imports"""
        imports = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Regex patterns for different import types
            patterns = [
                r'import\s+(?:{\s*([^}]+)\s*}|\*\s+as\s+(\w+)|(\w+))\s+from\s+["\']([^"\']+)["\']',
                r'const\s+(?:{\s*([^}]+)\s*}|(\w+))\s*=\s*require\(["\']([^"\']+)["\']\)',
                r'import\(["\']([^"\']+)["\']\)'
            ]

            for i, line in enumerate(content.split('\n'), 1):
                for pattern in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        groups = match.groups()
                        target = groups[-1]  # Last group is always the module path

                        imports.append(ImportReference(
                            source_file=str(file_path),
                            target_file=self._resolve_js_import(file_path.parent, target),
                            import_name=target,
                            line_number=i,
                            import_type='js'
                        ))

        except Exception as e:
            logger.error(f"Error analyzing JS imports in {file_path}: {e}")

        return imports

    def _resolve_python_import(self, import_name: str) -> str:
        """Resolve Python import to file path"""
        # Try to find the actual file
        parts = import_name.split('.')

        # Check in project root
        potential_paths = [
            self.project_root / '/'.join(parts) / '__init__.py',
            self.project_root / f"{'/'.join(parts)}.py",
        ]

        for path in potential_paths:
            if path.exists():
                return str(path)

        return import_name  # Return original if not found

    def _resolve_js_import(self, current_dir: Path, import_path: str) -> str:
        """Resolve JS import to file path"""
        if import_path.startswith('.'):
            # Relative import
            resolved = (current_dir / import_path).resolve()

            # Try different extensions
            for ext in ['.js', '.ts', '.jsx', '.tsx', '.json']:
                if resolved.with_suffix(ext).exists():
                    return str(resolved.with_suffix(ext))

            # Try index files
            for ext in ['.js', '.ts']:
                index_file = resolved / f"index{ext}"
                if index_file.exists():
                    return str(index_file)

        return import_path  # Return original if not found


class CodeElementExtractor:
    """Extracts code elements from source files"""

    def extract_python_elements(self, file_path: Path) -> List[CodeElement]:
        """Extract elements from Python file"""
        elements = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    elements.append(CodeElement(
                        name=node.name,
                        element_type='class',
                        file_path=str(file_path),
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        signature=f"class {node.name}",
                        docstring=ast.get_docstring(node),
                        hash_signature=self._hash_node(node, content)
                    ))

                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            elements.append(CodeElement(
                                name=item.name,
                                element_type='method',
                                file_path=str(file_path),
                                line_start=item.lineno,
                                line_end=getattr(item, 'end_lineno', item.lineno),
                                signature=self._get_function_signature(item),
                                docstring=ast.get_docstring(item),
                                parent_class=node.name,
                                hash_signature=self._hash_node(item, content)
                            ))

                elif isinstance(node, ast.FunctionDef):
                    elements.append(CodeElement(
                        name=node.name,
                        element_type='function',
                        file_path=str(file_path),
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        signature=self._get_function_signature(node),
                        docstring=ast.get_docstring(node),
                        hash_signature=self._hash_node(node, content)
                    ))

        except Exception as e:
            logger.error(f"Error extracting Python elements from {file_path}: {e}")

        return elements

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature string"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        return f"def {node.name}({', '.join(args)})"

    def _hash_node(self, node: ast.AST, content: str) -> str:
        """Generate hash for AST node"""
        try:
            node_source = ast.unparse(node)
            return hashlib.md5(node_source.encode()).hexdigest()
        except:
            # Fallback: use line-based hash
            lines = content.split('\n')
            start = getattr(node, 'lineno', 1) - 1
            end = getattr(node, 'end_lineno', start + 1)
            node_content = '\n'.join(lines[start:end])
            return hashlib.md5(node_content.encode()).hexdigest()


class MarkdownParser:
    """Parses markdown files and extracts sections"""

    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse markdown file into sections"""
        sections = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            current_section = None
            section_content = []
            line_start = 0

            for i, line in enumerate(lines):
                # Check for header
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

                if header_match:
                    # Save previous section
                    if current_section:
                        sections.append(self._create_section(
                            file_path, current_section, section_content,
                            line_start, i - 1
                        ))

                    # Start new section
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    current_section = (title, level)
                    section_content = []
                    line_start = i

                elif current_section:
                    section_content.append(line)

            # Save last section
            if current_section:
                sections.append(self._create_section(
                    file_path, current_section, section_content,
                    line_start, len(lines) - 1
                ))

        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {e}")

        return sections

    def _create_section(self, file_path: Path, section_info: Tuple[str, int],
                        content_lines: List[str], line_start: int, line_end: int) -> DocSection:
        """Create DocSection from parsed data"""
        title, level = section_info
        content = '\n'.join(content_lines).strip()

        # Extract source references from content
        source_refs = self._extract_source_refs(content)

        # Extract tags
        tags = self._extract_tags(content)

        section_id = f"{file_path.name}#{title}"

        return DocSection(
            section_id=section_id,
            file_path=str(file_path),
            title=title,
            content=content,
            level=level,
            line_start=line_start,
            line_end=line_end,
            source_refs=source_refs,
            tags=tags,
            hash_signature=hashlib.md5(content.encode()).hexdigest(),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
        )

    def _extract_source_refs(self, content: str) -> List[str]:
        """Extract source code references from markdown content"""
        refs = []

        # Look for code references in various formats
        patterns = [
            r'`([^`]+\.py:[^`]+)`',  # `file.py:Class.method`
            r'\[([^\]]+)\]\([^)]*\.py[^)]*\)',  # [text](file.py)
            r'```python[^`]*?([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)[^`]*?```'  # code blocks
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            refs.extend(matches)

        return list(set(refs))  # Remove duplicates

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from markdown content"""
        tags = []

        # Look for tags in various formats
        tag_patterns = [
            r'Tags?:\s*([^\n]+)',  # Tags: tag1, tag2
            r'#([a-zA-Z][a-zA-Z0-9_-]*)',  # #hashtag
        ]

        for pattern in tag_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if ',' in match:
                    tags.extend([tag.strip() for tag in match.split(',')])
                else:
                    tags.append(match.strip())

        return list(set(tags))

    def parse_file_incremental(self, file_path: Path, existing_sections: Dict[str, DocSection] = None) -> Tuple[
        List[DocSection], List[str]]:
        """Parse file with section-level change detection"""
        if existing_sections is None:
            existing_sections = {}

        new_sections = []
        changed_sections = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            current_section = None
            section_content = []
            line_start = 0

            for i, line in enumerate(lines):
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

                if header_match:
                    # Process previous section
                    if current_section:
                        section = self._create_section_with_hash(
                            file_path, current_section, section_content,
                            line_start, i - 1
                        )

                        # Check if section changed
                        existing_section = existing_sections.get(section.section_id)
                        if self._section_changed(section, existing_section):
                            section.change_detected = True
                            changed_sections.append(section.section_id)

                        new_sections.append(section)

                    # Start new section
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    current_section = (title, level)
                    section_content = []
                    line_start = i

                elif current_section:
                    section_content.append(line)

            # Handle last section
            if current_section:
                section = self._create_section_with_hash(
                    file_path, current_section, section_content,
                    line_start, len(lines) - 1
                )

                existing_section = existing_sections.get(section.section_id)
                if self._section_changed(section, existing_section):
                    section.change_detected = True
                    changed_sections.append(section.section_id)

                new_sections.append(section)

        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {e}")

        return new_sections, changed_sections

    def _create_section_with_hash(self, file_path: Path, section_info: Tuple[str, int],
                                  content_lines: List[str], line_start: int, line_end: int) -> DocSection:
        """Create DocSection with precise hash tracking"""
        title, level = section_info
        content = '\n'.join(content_lines).strip()

        # Create separate hashes for different aspects
        content_hash = hashlib.md5(content.encode()).hexdigest()
        title_hash = hashlib.md5(title.encode()).hexdigest()
        combined_hash = hashlib.md5(f"{title}:{content}:{line_start}:{line_end}".encode()).hexdigest()

        source_refs = self._extract_source_refs(content)
        tags = self._extract_tags(content)
        section_id = f"{file_path.name}#{title}"

        return DocSection(
            section_id=section_id,
            file_path=str(file_path),
            title=title,
            content=content,
            level=level,
            line_start=line_start,
            line_end=line_end,
            source_refs=source_refs,
            tags=tags,
            hash_signature=combined_hash,
            content_hash=content_hash,
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
            change_detected=False
        )

    def _section_changed(self, new_section: DocSection, existing_section: Optional[DocSection]) -> bool:
        """Check if section actually changed"""
        if not existing_section:
            return True  # New section

        # Quick hash comparison
        if new_section.content_hash == existing_section.content_hash:
            return False

        # Title change
        if new_section.title != existing_section.title:
            return True

        # Significant content change (not just whitespace)
        if new_section.hash_signature != existing_section.hash_signature:
            return True

        return False


class DocsIndexer:
    """Main indexer that builds and maintains the complete documentation index"""

    def __init__(self, project_root: Path, docs_root: Path,
                 include_dirs: List[str] = None, exclude_dirs: List[str] = None):
        self.project_root = project_root
        self.docs_root = docs_root
        self.git_detector = GitChangeDetector(project_root)
        self.import_analyzer = ImportAnalyzer(project_root)
        self.code_extractor = CodeElementExtractor()
        self.md_parser = MarkdownParser()
        self.index_file = docs_root / '.docs_index.json'

        # Directory filters
        self.include_dirs = include_dirs or ["toolboxv2", "src", "lib", "docs"]
        self.exclude_dirs = exclude_dirs or [
            "__pycache__", ".git", "node_modules", ".venv", "venv", "env",
            ".pytest_cache", ".mypy_cache", "dist", "build", ".tox",
            "coverage_html_report", ".coverage", ".next", ".nuxt",
            "target", "bin", "obj", ".gradle", ".idea", ".vscode"
        ]

    async def update_index_precise(self, current_index: DocsIndex,
                                   force_full_scan: bool = False,
                                   max_files_per_batch: int = 10) -> Tuple[DocsIndex, List[str], Dict[str, List[str]]]:
        """Update index with precise section-level tracking"""
        logger.info("Starting precise index update...")

        update_notes = []
        section_changes = {}  # file_path -> [changed_section_ids]

        if force_full_scan:
            return await self._full_scan_with_sections(current_index, max_files_per_batch)

        # Quick git change detection with timeout
        try:
            changes = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.git_detector.get_changed_files, current_index.last_git_commit
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning("Git detection timed out, using cached index")
            return current_index, ["Git timeout - using cached index"], {}

        # Process changes in batches
        changed_files = [c for c in changes if self._should_include_file(Path(c.file_path))]

        for i in range(0, len(changed_files), max_files_per_batch):
            batch = changed_files[i:i + max_files_per_batch]

            for change in batch:
                file_path = Path(change.file_path)

                if change.change_type == ChangeType.DELETED:
                    removed_sections = self._remove_file_from_index(current_index, str(file_path))
                    update_notes.append(f"Removed file: {file_path} ({len(removed_sections)} sections)")

                elif change.change_type in [ChangeType.ADDED, ChangeType.MODIFIED]:
                    if not file_path.exists():
                        continue

                    # Quick hash check first
                    new_hash = self._get_file_hash(file_path)
                    old_hash = current_index.file_hashes.get(str(file_path))

                    if new_hash == old_hash:
                        continue  # No actual change

                    # Section-level update for markdown files
                    if file_path.suffix == '.md' and file_path.is_relative_to(self.docs_root):
                        changed_section_ids = await self._update_markdown_sections(
                            current_index, file_path
                        )
                        if changed_section_ids:
                            section_changes[str(file_path)] = changed_section_ids
                            update_notes.append(f"Updated {len(changed_section_ids)} sections in {file_path.name}")

                    # Code files - update entire file (faster for code)
                    elif file_path.suffix in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                        self._update_file_in_index(current_index, file_path)
                        update_notes.append(f"Updated code file: {file_path}")

                    # Update file hash
                    current_index.file_hashes[str(file_path)] = new_hash

            # Yield control between batches
            await asyncio.sleep(0.01)

        # Update timestamps
        current_index.last_git_commit = self.git_detector.get_current_commit_hash()
        current_index.last_indexed = datetime.now()

        return current_index, update_notes, section_changes

    async def _update_markdown_sections(self, index: DocsIndex, file_path: Path) -> List[str]:
        """Update only changed sections in a markdown file"""
        try:
            # Get existing sections for this file
            existing_sections = {
                sid: section for sid, section in index.sections.items()
                if section.file_path == str(file_path)
            }

            # Parse with change detection
            new_sections, changed_section_ids = self.md_parser.parse_file_incremental(
                file_path, existing_sections
            )

            # Update only changed sections
            for section in new_sections:
                if section.change_detected or section.section_id not in index.sections:
                    index.sections[section.section_id] = section
                    # Update section hash tracking
                    index.section_hashes[section.section_id] = section.hash_signature

            # Remove sections that no longer exist
            current_section_ids = {s.section_id for s in new_sections}
            to_remove = [sid for sid in existing_sections.keys() if sid not in current_section_ids]

            for sid in to_remove:
                if sid in index.sections:
                    del index.sections[sid]
                if sid in index.section_hashes:
                    del index.section_hashes[sid]
                changed_section_ids.append(f"REMOVED:{sid}")

            return changed_section_ids

        except Exception as e:
            logger.error(f"Error updating markdown sections in {file_path}: {e}")
            return []

    def build_initial_index(self, file_extensions: List[str] = None) -> DocsIndex:
        """Build complete index for the first time with directory filtering"""
        logger.info("Building initial documentation index...")

        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.md']

        index = DocsIndex()

        # Get current git commit
        index.last_git_commit = self.git_detector.get_current_commit_hash()

        # Get filtered file list
        target_files = self._get_filtered_files(file_extensions)

        logger.info(f"Processing {len(target_files)} files in {len(self.include_dirs)} directories")

        # Process each file type
        for file_path in target_files:
            logger.debug(f"Indexing {file_path}")
            try:
                if file_path.suffix == '.py':
                    self._index_python_file(file_path, index)
                elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
                    self._index_js_file(file_path, index)
                elif file_path.suffix == '.md' and file_path.is_relative_to(self.docs_root):
                    self._index_md_file(file_path, index)

                # Store file hash for change detection
                index.file_hashes[str(file_path)] = self._get_file_hash(file_path)

            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                continue

        index.last_indexed = datetime.now()

        logger.info(f"Initial index built: {len(index.code_elements)} code elements, "
                    f"{len(index.sections)} doc sections, {len(index.import_refs)} import maps")

        return index

    def update_index(self, current_index: DocsIndex,
                     force_full_scan: bool = False) -> Tuple[DocsIndex, List[str]]:
        """Update index based on detected changes or force full scan"""
        logger.info("Updating documentation index...")

        update_notes = []

        if force_full_scan:
            logger.info("Performing force full scan...")
            # Get all current files
            current_files = set(self._get_filtered_files(['.py', '.js', '.ts', '.jsx', '.tsx', '.md']))

            # Check each file for changes
            for file_path in current_files:
                new_hash = self._get_file_hash(file_path)
                old_hash = current_index.file_hashes.get(str(file_path))

                if new_hash != old_hash:
                    self._update_file_in_index(current_index, file_path)
                    update_notes.append(f"Updated (force scan): {file_path}")
                    current_index.file_hashes[str(file_path)] = new_hash

            # Remove files that no longer exist
            existing_files = {str(f) for f in current_files}
            to_remove = [f for f in current_index.file_hashes.keys() if f not in existing_files]

            for file_path in to_remove:
                self._remove_file_from_index(current_index, file_path)
                update_notes.append(f"Removed (no longer exists): {file_path}")

        else:
            # Git-based change detection
            changes = self.git_detector.get_changed_files(current_index.last_git_commit)

            for change in changes:
                file_path = Path(change.file_path)

                # Skip if not in our target directories
                if not self._should_include_file(file_path):
                    continue

                if change.change_type == ChangeType.DELETED:
                    self._remove_file_from_index(current_index, str(file_path))
                    update_notes.append(f"Removed (git): {file_path}")

                elif change.change_type == ChangeType.RENAMED:
                    if change.old_path:
                        self._remove_file_from_index(current_index, change.old_path)
                        update_notes.append(f"Removed (renamed from): {change.old_path}")

                    if file_path.exists():
                        self._update_file_in_index(current_index, file_path)
                        update_notes.append(f"Added (renamed to): {file_path}")
                        current_index.file_hashes[str(file_path)] = self._get_file_hash(file_path)

                elif change.change_type in [ChangeType.ADDED, ChangeType.MODIFIED]:
                    if not file_path.exists():
                        continue

                    # Verify actual change with hash comparison
                    new_hash = self._get_file_hash(file_path)
                    old_hash = current_index.file_hashes.get(str(file_path))

                    if new_hash != old_hash:
                        self._update_file_in_index(current_index, file_path)
                        update_notes.append(f"Updated (git {change.change_type.value}): {file_path}")
                        current_index.file_hashes[str(file_path)] = new_hash

        # Update git commit and timestamp
        current_index.last_git_commit = self.git_detector.get_current_commit_hash()
        current_index.last_indexed = datetime.now()

        if update_notes:
            logger.info(f"Index updated with {len(update_notes)} changes")
        else:
            logger.info("No changes detected in index update")

        return current_index, update_notes

    def _get_filtered_files(self, extensions: List[str]) -> List[Path]:
        """Get list of files matching extensions and directory filters"""
        files = []

        # If include_dirs is specified, only search in those directories
        search_dirs = []
        for include_dir in self.include_dirs:
            search_path = self.project_root / include_dir
            if search_path.exists() and search_path.is_dir():
                search_dirs.append(search_path)

        # If no include dirs exist, search entire project root
        if not search_dirs:
            search_dirs = [self.project_root]

        for search_dir in search_dirs:
            for ext in extensions:
                pattern = f"**/*{ext}"
                for file_path in search_dir.rglob(pattern):
                    if self._should_include_file(file_path):
                        files.append(file_path)

        return list(set(files))  # Remove duplicates

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included based on directory filters"""
        file_str = str(file_path)

        # Check exclude patterns
        for exclude_dir in self.exclude_dirs:
            if exclude_dir in file_str:
                return False

        # If include_dirs specified, file must be in one of them
        if self.include_dirs:
            for include_dir in self.include_dirs:
                include_path = self.project_root / include_dir
                try:
                    if file_path.is_relative_to(include_path):
                        return True
                except ValueError:
                    continue
            return False

        return True

    def _update_file_in_index(self, index: DocsIndex, file_path: Path):
        """Update index for a specific file"""
        # Remove old entries first
        self._remove_file_from_index(index, str(file_path))

        # Add new entries based on file type
        try:
            if file_path.suffix == '.py':
                self._index_python_file(file_path, index)
            elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
                self._index_js_file(file_path, index)
            elif file_path.suffix == '.md' and file_path.is_relative_to(self.docs_root):
                self._index_md_file(file_path, index)
        except Exception as e:
            logger.error(f"Error updating file {file_path}: {e}")

    def _index_python_file(self, file_path: Path, index: DocsIndex):
        """Index a Python file"""
        # Extract code elements
        elements = self.code_extractor.extract_python_elements(file_path)
        for element in elements:
            element_id = f"{element.file_path}:{element.name}"
            if element.parent_class:
                element_id = f"{element.file_path}:{element.parent_class}.{element.name}"
            index.code_elements[element_id] = element

        # Analyze imports
        imports = self.import_analyzer.analyze_python_imports(file_path)
        if imports:
            index.import_refs[str(file_path)] = imports

    def _index_js_file(self, file_path: Path, index: DocsIndex):
        """Index a JavaScript/TypeScript file"""
        imports = self.import_analyzer.analyze_js_imports(file_path)
        if imports:
            index.import_refs[str(file_path)] = imports

    def _index_md_file(self, file_path: Path, index: DocsIndex):
        """Index a markdown file"""
        sections = self.md_parser.parse_file(file_path)
        for section in sections:
            index.sections[section.section_id] = section

    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _remove_file_from_index(self, index: DocsIndex, file_path: str):
        """Remove all references to a file from index"""
        # Remove code elements
        to_remove = [k for k, v in index.code_elements.items() if v.file_path == file_path]
        for key in to_remove:
            del index.code_elements[key]

        # Remove doc sections
        to_remove = [k for k, v in index.sections.items() if v.file_path == file_path]
        for key in to_remove:
            del index.sections[key]

        # Remove import refs
        if file_path in index.import_refs:
            del index.import_refs[file_path]

        # Remove file hash
        if file_path in index.file_hashes:
            del index.file_hashes[file_path]


class SectionManager:
    """Manages precise operations on documentation sections"""

    def __init__(self, docs_root: Path):
        self.docs_root = docs_root

    def create_new_file(self, file_path: str, initial_content: str = "") -> Result:
        """Create a new markdown file"""
        try:
            full_path = self.docs_root / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if full_path.exists():
                return Result.default_user_error(f"File already exists: {file_path}")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(initial_content)

            return Result.ok({"file_path": str(full_path), "action": "created"})

        except Exception as e:
            return Result.default_user_error(f"Error creating file: {e}")

    def add_section(self, file_path: str, section: DocSection, position: Optional[str] = None) -> Result:
        """Add a new section to a markdown file"""
        try:
            full_path = self.docs_root / file_path

            if not full_path.exists():
                # Create new file
                content = f"{'#' * section.level} {section.title}\n\n{section.content}\n\n"
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return Result.ok({"action": "created_file_with_section"})

            # Read existing content
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Find insertion point
            insert_index = len(lines)  # Default: append at end

            if position:
                if position == "top":
                    insert_index = 0
                elif position.startswith("after:"):
                    target_section = position[6:]
                    for i, line in enumerate(lines):
                        if line.strip().endswith(target_section):
                            # Find end of this section
                            for j in range(i + 1, len(lines)):
                                if re.match(r'^#{1,6}\s+', lines[j]):
                                    insert_index = j
                                    break
                            break

            # Insert new section
            new_content = f"{'#' * section.level} {section.title}\n\n{section.content}\n\n"
            lines.insert(insert_index, new_content)

            # Write back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return Result.ok({"action": "section_added", "position": insert_index})

        except Exception as e:
            return Result.default_user_error(f"Error adding section: {e}")

    def update_section(self, section: DocSection, new_content: str) -> Result:
        """Update an existing section"""
        try:
            file_path = Path(section.file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Replace content between line_start and line_end
            new_section_content = f"{'#' * section.level} {section.title}\n\n{new_content}\n\n"
            new_lines = new_section_content.split('\n')

            # Replace the section
            lines[section.line_start:section.line_end + 1] = [line + '\n' for line in new_lines]

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return Result.ok({"action": "section_updated"})

        except Exception as e:
            return Result.default_user_error(f"Error updating section: {e}")

    def delete_section(self, section: DocSection) -> Result:
        """Delete a section from a file"""
        try:
            file_path = Path(section.file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Remove lines for this section
            del lines[section.line_start:section.line_end + 1]

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return Result.ok({"action": "section_deleted"})

        except Exception as e:
            return Result.default_user_error(f"Error deleting section: {e}")


class MarkdownDocsSystem:
    """Production-ready unified markdown documentation system"""

    def __init__(self, app: AppType, docs_root: str = "../docs", source_root: str = ".",
                 include_dirs: List[str] = None, exclude_dirs: List[str] = None):
        self.app = app
        self.docs_root = Path(docs_root)
        self.source_root = Path(source_root)
        self.project_root = Path.cwd()

        # Directory filters
        self.include_dirs = include_dirs or ["toolboxv2", "flows", "mods", "utils", "tbjs", "tests", "tcm", "docs"]
        self.exclude_dirs = exclude_dirs or [
            "__pycache__", ".git", "node_modules", ".venv", "venv", "env", "python_env",
            ".pytest_cache", ".mypy_cache", "dist", "build", ".tox", "coverage_html_report",
            ".coverage", ".next", ".nuxt", "target", "bin", "obj", ".gradle", ".idea",
            ".vscode", "temp", "tmp", "logs", ".cache", "coverage", ".data", ".config",
            ".info", "web", "simple-core", "src-core"
        ]

        # Index management
        self.index_file = self.docs_root / '.docs_index.json'
        self.current_index: Optional[DocsIndex] = None

        # Ensure directories exist
        self.docs_root.mkdir(exist_ok=True)

        # Internal cache for performance
        self._search_cache = {}
        self._cache_timeout = 300  # 5 minutes

    # ==================== CORE INDEX MANAGEMENT ====================


    def _load_index(self, minimal: bool = False, force_reload: bool = False) -> DocsIndex:
        """Unified index loading with proper incremental loading"""
        # Return cached index if available and not forcing reload
        if not force_reload and self.current_index and not minimal:
            return self.current_index

        if not self.index_file.exists():
            logger.info("No index file found, will build new index")
            return DocsIndex()

        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            index = DocsIndex()
            index.last_indexed = datetime.fromisoformat(
                data.get('last_indexed', datetime.now().isoformat())
            )
            index.last_git_commit = data.get('last_git_commit')
            index.version = data.get('version', '1.1')
            index.file_hashes = data.get('file_hashes', {})
            index.section_hashes = data.get('section_hashes', {})

            # Load sections with optional truncation for performance
            sections_data = data.get('sections', {})
            section_limit = 200 if minimal else None
            section_count = 0

            for section_id, section_data in sections_data.items():
                if minimal and section_count >= section_limit:
                    break

                content = section_data['content']
                if minimal:
                    content = content[:800]  # Truncate for speed

                index.sections[section_id] = DocSection(
                    section_id=section_data['section_id'],
                    file_path=section_data['file_path'],
                    title=section_data['title'],
                    content=content,
                    level=section_data['level'],
                    line_start=section_data['line_start'],
                    line_end=section_data['line_end'],
                    source_refs=section_data.get('source_refs', [])[:5 if minimal else None],
                    tags=section_data.get('tags', [])[:3 if minimal else None],
                    hash_signature=section_data.get('hash_signature', ''),
                    content_hash=section_data.get('content_hash', ''),
                    last_modified=datetime.fromisoformat(
                        section_data.get('last_modified', datetime.now().isoformat()))
                )
                section_count += 1

            # Load code elements only if not minimal
            if not minimal:
                for element_id, element_data in data.get('code_elements', {}).items():
                    index.code_elements[element_id] = CodeElement(
                        name=element_data['name'],
                        element_type=element_data['element_type'],
                        file_path=element_data['file_path'],
                        line_start=element_data['line_start'],
                        line_end=element_data['line_end'],
                        signature=element_data['signature'],
                        docstring=element_data.get('docstring'),
                        hash_signature=element_data.get('hash_signature', ''),
                        parent_class=element_data.get('parent_class')
                    )

            # Cache full index for future use
            if not minimal:
                self.current_index = index

            logger.info(
                f"Loaded {'minimal' if minimal else 'full'} index: {len(index.sections)} sections, {len(index.code_elements)} elements")
            return index

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return DocsIndex()

    def _save_index(self, index: DocsIndex):
        """Optimized index saving"""
        try:
            data = {
                'version': index.version,
                'last_git_commit': index.last_git_commit,
                'last_indexed': index.last_indexed.isoformat(),
                'file_hashes': index.file_hashes,
                'section_hashes': index.section_hashes,
                'sections': {},
                'code_elements': {}
            }

            # Convert sections efficiently
            for section_id, section in index.sections.items():
                section_dict = asdict(section)
                section_dict['last_modified'] = section.last_modified.isoformat()
                data['sections'][section_id] = section_dict

            # Convert code elements efficiently
            for element_id, element in index.code_elements.items():
                data['code_elements'][element_id] = asdict(element)

            # Write with minimal formatting for speed
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'), ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    async def _save_index_async(self, index: DocsIndex):
        """Async wrapper for index saving"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._save_index, index)
        except Exception as e:
            logger.error(f"Async index save failed: {e}")

    # ==================== CORE FUNCTIONALITY ====================

    async def docs_reader(self,
                          query: Optional[str] = None,
                          section_id: Optional[str] = None,
                          file_path: Optional[str] = None,
                          tags: Optional[List[str]] = None,
                          include_source_refs: bool = True,
                          format_type: str = "structured",
                          max_results: int = 25) -> Result:
        """Ultra-fast unified docs reader with proper index loading"""
        try:
            start_time = time.time()

            # Load index if needed - try cached first, then saved, then build
            if not self.current_index:
                # First try to load from saved file
                self.current_index = self._load_index(minimal=False)

                # If no sections found, suggest running initial_docs_parse
                if not self.current_index.sections:
                    if self.index_file.exists():
                        logger.warning("Index file exists but contains no documentation sections")
                        # Try to find any markdown files
                        md_files = list(self.docs_root.rglob('*.md'))
                        if md_files:
                            logger.info(f"Found {len(md_files)} markdown files, re-parsing...")
                            # Re-parse markdown files
                            for md_file in md_files[:10]:  # Limit for quick check
                                sections = self._parse_markdown_file(md_file)
                                for section in sections:
                                    self.current_index.sections[section.section_id] = section

                            if self.current_index.sections:
                                logger.info(f"Re-parsed {len(self.current_index.sections)} sections")
                                await self._save_index_async(self.current_index)

                    if not self.current_index.sections:
                        return Result.default_user_error(
                            "No documentation sections found. Run initial_docs_parse() to build the documentation index, "
                            "or create some .md files in the docs directory."
                        )

            # Check cache for repeated queries
            cache_key = f"{query}:{section_id}:{file_path}:{tags}:{format_type}:{max_results}"
            if cache_key in self._search_cache:
                cache_entry = self._search_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self._cache_timeout:
                    return Result.ok(cache_entry['result'])

            # Fast path for specific section
            if section_id:
                if section_id in self.current_index.sections:
                    section = self.current_index.sections[section_id]
                    result = self._format_single_section(section, include_source_refs, format_type)
                    return result
                else:
                    return Result.default_user_error(f"Section not found: {section_id}")

            # Search and format results
            matching_sections = await self._search_sections(query, file_path, tags, max_results, start_time)
            result_data = self._format_sections(matching_sections, include_source_refs, format_type)

            # Cache the result
            self._search_cache[cache_key] = {
                'result': result_data,
                'timestamp': time.time()
            }

            return Result.ok(result_data)

        except Exception as e:
            logger.error(f"Error in docs_reader: {e}")
            return Result.default_user_error(f"Error reading documentation: {e}")

    async def docs_writer(self,
                          action: str,
                          file_path: Optional[str] = None,
                          section_title: Optional[str] = None,
                          content: Optional[str] = None,
                          source_file: Optional[str] = None,
                          auto_generate: bool = False,
                          position: Optional[str] = None,
                          level: int = 2) -> Result:
        """Unified optimized docs writer"""
        try:
            if not self.current_index:
                self.current_index = self._load_index(minimal=True)

            result = {"action": action, "timestamp": datetime.now().isoformat()}

            if action == "update_section":
                return await self._update_section(file_path, section_title, content,
                                                  source_file, auto_generate)
            elif action == "add_section":
                return await self._add_section(file_path, section_title, content,
                                               source_file, auto_generate, position, level)
            elif action == "create_file":
                return await self._create_file(file_path, content, source_file, auto_generate)
            elif action == "generate_from_code":
                return await self._generate_from_code(source_file, file_path, auto_generate)
            else:
                return Result.default_user_error(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error in docs_writer: {e}")
            return Result.default_user_error(f"Error writing documentation: {e}")

    # ==================== ADVANCED OPERATIONS ====================

    async def get_update_suggestions(self, force_scan: bool = False,
                                     max_suggestions: int = 50) -> Result:
        """Get prioritized documentation update suggestions"""
        try:
            if not self.current_index:
                self.current_index = self._load_index()

            # Quick scan for changes if needed
            if force_scan:
                await self._quick_update_index()

            suggestions = []

            # Find undocumented code elements
            undocumented = self._find_undocumented_elements()
            for element in undocumented[:max_suggestions // 2]:
                priority = self._assess_priority(element)
                suggestions.append({
                    "type": "missing_documentation",
                    "element_name": element.name,
                    "element_type": element.element_type,
                    "file_path": element.file_path,
                    "priority": priority,
                    "action": "generate_from_code" if element.element_type == "class" else "add_section"
                })

            # Find unclear sections
            unclear = self._find_unclear_sections()
            for section_id in unclear[:max_suggestions // 2]:
                section = self.current_index.sections[section_id]
                suggestions.append({
                    "type": "unclear_documentation",
                    "section_id": section_id,
                    "title": section.title,
                    "file_path": section.file_path,
                    "priority": "medium",
                    "action": "update_section"
                })

            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))

            return Result.ok({
                "suggestions": suggestions[:max_suggestions],
                "total_found": len(suggestions),
                "undocumented_elements": len(undocumented),
                "unclear_sections": len(unclear)
            })

        except Exception as e:
            logger.error(f"Error getting update suggestions: {e}")
            return Result.default_user_error(f"Error analyzing updates: {e}")

    async def auto_update_docs(self, dry_run: bool = False, max_updates: int = 5,
                               timeout: int = 30) -> Result:
        """Automatically update documentation with timeout protection"""
        try:
            # Get suggestions
            suggestions_result = await asyncio.wait_for(
                self.get_update_suggestions(max_suggestions=max_updates * 2),
                timeout=10.0
            )

            if suggestions_result.is_error():
                return suggestions_result

            suggestions = suggestions_result.get("suggestions")[:max_updates]

            if dry_run:
                return Result.ok({
                    "dry_run": True,
                    "would_update": len(suggestions),
                    "suggestions": [s["type"] + ": " + s.get("element_name", s.get("title", ""))
                                    for s in suggestions]
                })

            results = []
            start_time = time.time()

            for suggestion in suggestions:
                if time.time() - start_time > timeout:
                    break

                try:
                    if suggestion["action"] == "generate_from_code":
                        result = await asyncio.wait_for(
                            self.docs_writer(
                                action="generate_from_code",
                                source_file=suggestion["file_path"],
                                auto_generate=True
                            ), timeout=10.0
                        )
                    elif suggestion["action"] == "add_section":
                        result = await asyncio.wait_for(
                            self.docs_writer(
                                action="add_section",
                                file_path=f"{Path(suggestion['file_path']).stem}.md",
                                section_title=suggestion["element_name"],
                                source_file=suggestion["file_path"],
                                auto_generate=True
                            ), timeout=8.0
                        )
                    else:
                        continue

                    results.append({
                        "suggestion": suggestion["type"],
                        "status": "success" if result.is_ok() else "error",
                        "details": str(result.error) if result.is_error() else "completed"
                    })

                except asyncio.TimeoutError:
                    results.append({
                        "suggestion": suggestion["type"],
                        "status": "timeout",
                        "details": "Operation timed out"
                    })

                await asyncio.sleep(0.1)  # Brief pause between operations

            return Result.ok({
                "processed": len(results),
                "successful": len([r for r in results if r["status"] == "success"]),
                "results": results,
                "execution_time": f"{time.time() - start_time:.2f}s"
            })

        except Exception as e:
            logger.error(f"Error in auto_update_docs: {e}")
            return Result.default_user_error(f"Auto-update failed: {e}")

    async def initial_docs_parse(self, update_index: bool = True, force_rebuild: bool = False) -> Result:
        """Parse existing documentation and build initial index with proper saving"""
        try:
            logger.info("Starting initial documentation parse...")

            # Check if we can use existing index
            if not force_rebuild and self.index_file.exists() and not update_index:
                self.current_index = self._load_index(minimal=False)
                if self.current_index.sections or self.current_index.code_elements:
                    logger.info("Using existing index")
                    return Result.ok({
                        "total_sections": len(self.current_index.sections),
                        "total_code_elements": len(self.current_index.code_elements),
                        "linked_sections": len([s for s in self.current_index.sections.values() if s.source_refs]),
                        "completion_rate": f"{(len([s for s in self.current_index.sections.values() if s.source_refs]) / max(len(self.current_index.sections), 1) * 100):.1f}%",
                        "used_cached": True
                    })

            if update_index or force_rebuild:
                # Build comprehensive index
                logger.info("Building comprehensive index from source files...")
                self.current_index = await asyncio.get_event_loop().run_in_executor(
                    None, self._build_full_index
                )
            else:
                self.current_index = self._load_index(minimal=False)

            # Ensure we have some documentation sections
            if not self.current_index.sections:
                logger.info("No documentation sections found, scanning for markdown files...")
                # Look for markdown files in docs and project root
                search_paths = [self.docs_root]
                if self.docs_root != self.project_root:
                    search_paths.append(self.project_root)

                for search_path in search_paths:
                    for md_file in search_path.rglob('*.md'):
                        if self._should_include_file(md_file):
                            sections = self._parse_markdown_file(md_file)
                            for section in sections:
                                self.current_index.sections[section.section_id] = section

                    # Don't search too deep
                    if len(self.current_index.sections) > 0:
                        break

                logger.info(f"Found {len(self.current_index.sections)} documentation sections")

            # Link documentation sections to code elements
            linked_count = await self._link_docs_to_code()

            # Save updated index with proper error handling
            try:
                await self._save_index_async(self.current_index)
                logger.info("Index saved successfully")
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                # Continue anyway, we have the index in memory

            completion_rate = (
                    linked_count / max(len(self.current_index.sections), 1) * 100) if self.current_index.sections else 0

            return Result.ok({
                "total_sections": len(self.current_index.sections),
                "total_code_elements": len(self.current_index.code_elements),
                "linked_sections": linked_count,
                "completion_rate": f"{completion_rate:.1f}%",
                "index_file": str(self.index_file),
                "docs_root": str(self.docs_root)
            })

        except Exception as e:
            logger.error(f"Error in initial_docs_parse: {e}")
            return Result.default_user_error(f"Failed to parse docs: {e}")

    # ==================== INTERNAL HELPER METHODS ====================

    async def _search_sections(self, query: Optional[str], file_path: Optional[str],
                               tags: Optional[List[str]], max_results: int, start_time: float) -> List[DocSection]:
        """Optimized section search with early termination"""
        matching_sections = []
        search_terms = set(query.lower().split()) if query else set()

        for section in self.current_index.sections.values():
            if len(matching_sections) >= max_results:
                break

            # Timeout protection
            if time.time() - start_time > 3.0:
                break

            # Quick filters
            if file_path and file_path not in section.file_path:
                continue

            if tags and not any(tag in section.tags for tag in tags):
                continue

            if search_terms:
                search_content = f"{section.title} {section.content[:200]}".lower()
                if not any(term in search_content for term in search_terms):
                    continue

            matching_sections.append(section)

        return matching_sections

    def _format_sections(self, sections: List[DocSection], include_source_refs: bool,
                         format_type: str) -> dict:
        """Unified section formatting"""
        if format_type == "markdown":
            output = []
            for section in sections[:20]:
                output.append(f"{'#' * section.level} {section.title}")
                output.append("")
                content = section.content[:500] + ("..." if len(section.content) > 500 else "")
                output.append(content)
                if include_source_refs and section.source_refs:
                    output.append(f"\n**References:** {', '.join(section.source_refs[:3])}")
                output.append("")
            return "\n".join(output)

        elif format_type == "json":
            return [self._section_to_dict(s, include_source_refs) for s in sections[:20]]

        else:  # structured
            return {
                "sections": [self._section_to_dict(s, include_source_refs) for s in sections[:20]],
                "metadata": {
                    "total_available": len(sections),
                    "returned_sections": min(len(sections), 20),
                    "truncated": len(sections) > 20
                }
            }

    def _format_single_section(self, section: DocSection, include_source_refs: bool,
                               format_type: str) -> Result:
        """Format a single section efficiently"""
        if format_type == "markdown":
            content = f"{'#' * section.level} {section.title}\n\n{section.content}"
            if include_source_refs and section.source_refs:
                content += f"\n\n**References:** {', '.join(section.source_refs[:5])}"
            return Result.ok(content)

        elif format_type == "json":
            return Result.ok([self._section_to_dict(section, include_source_refs)])

        else:  # structured
            return Result.ok({
                "sections": [self._section_to_dict(section, include_source_refs)],
                "metadata": {"total_sections": 1, "query_type": "specific_section"}
            })

    def _section_to_dict(self, section: DocSection, include_source_refs: bool) -> dict:
        """Convert section to dictionary efficiently"""
        data = {
            "id": section.section_id,
            "title": section.title,
            "content": section.content[:500],  # Limit for performance
            "file_path": section.file_path,
            "level": section.level,
            "tags": section.tags[:3]
        }
        if include_source_refs:
            data["source_refs"] = section.source_refs[:3]
        return data

    async def _update_section(self, file_path: str, section_title: str, content: str,
                              source_file: str, auto_generate: bool) -> Result:
        """Update existing section"""
        if not file_path or not section_title:
            return Result.default_user_error("file_path and section_title required")

        section_id = f"{Path(file_path).name}#{section_title}"
        if section_id not in self.current_index.sections:
            return Result.default_user_error(f"Section not found: {section_id}")

        section = self.current_index.sections[section_id]

        if auto_generate and source_file:
            content = await self._generate_content(section_title, source_file)

        if not content:
            return Result.default_user_error("content required")

        # Update file
        success = self._update_section_in_file(section, content)
        if not success:
            return Result.default_user_error("Failed to update file")

        # Update index
        section.content = content
        section.content_hash = hashlib.md5(content.encode()).hexdigest()
        section.last_modified = datetime.now()

        asyncio.create_task(self._save_index_async(self.current_index))

        return Result.ok({"action": "section_updated", "section_id": section_id})

    async def _add_section(self, file_path: str, section_title: str, content: str,
                           source_file: str, auto_generate: bool, position: str, level: int) -> Result:
        """Add new section to file"""
        if not file_path or not section_title:
            return Result.default_user_error("file_path and section_title required")

        if auto_generate and source_file:
            content = await self._generate_content(section_title, source_file)

        if not content:
            content = f"Content for {section_title}\n\nTODO: Add documentation here."

        # Create section
        section = DocSection(
            section_id=f"{Path(file_path).name}#{section_title}",
            file_path=str(self.docs_root / file_path),
            title=section_title,
            content=content,
            level=level,
            line_start=0,
            line_end=0,
            hash_signature=hashlib.md5(content.encode()).hexdigest(),
            content_hash=hashlib.md5(content.encode()).hexdigest()
        )

        # Add to file
        success = self._add_section_to_file(file_path, section, position)
        if not success:
            return Result.default_user_error("Failed to add section to file")

        # Update index
        self.current_index.sections[section.section_id] = section
        asyncio.create_task(self._save_index_async(self.current_index))

        return Result.ok({"action": "section_added", "section_id": section.section_id})

    async def _create_file(self, file_path: str, content: str, source_file: str,
                           auto_generate: bool) -> Result:
        """Create new documentation file"""
        if not file_path:
            return Result.default_user_error("file_path required")

        full_path = self.docs_root / file_path

        if full_path.exists():
            return Result.default_user_error(f"File already exists: {file_path}")

        if auto_generate and source_file:
            content = await self._generate_file_content(source_file)

        if not content:
            title = Path(file_path).stem.replace('_', ' ').title()
            content = f"# {title}\n\nDocumentation for {title}.\n\n"

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')

        return Result.ok({"action": "file_created", "file_path": str(full_path)})

    async def _generate_from_code(self, source_file: str, file_path: str,
                                  auto_generate: bool) -> Result:
        """Generate documentation from source code"""
        if not source_file:
            return Result.default_user_error("source_file required")

        if not Path(source_file).exists():
            return Result.default_user_error(f"Source file not found: {source_file}")

        if not file_path:
            file_path = f"{Path(source_file).stem}.md"

        content = await self._generate_file_content(source_file) if auto_generate else ""

        return await self._create_file(file_path, content, source_file, False)

    async def _generate_content(self, title: str, source_file: str) -> str:
        """Generate content using AI with timeout protection"""
        try:
            isaa = self.app.get_mod("isaa")
            agent = await isaa.get_agent("docwriter")

            if not agent:
                return f"# {title}\n\nAI agent not available. Please add content manually."

            with open(source_file, 'r', encoding='utf-8') as f:
                source_content = f.read()[:2000]  # Limit source size

            prompt = f"""Generate concise documentation for "{title}" from this code:

```python
{source_content}
```

Requirements: Clear, concise, markdown format, no section header."""

            result = await asyncio.wait_for(
                agent.a_run_llm_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model_preference="fast"
                ), timeout=8.0
            )
            return result

        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return f"# {title}\n\nError generating content: {e}"

    async def _generate_file_content(self, source_file: str) -> str:
        """Generate complete file documentation"""
        try:
            isaa = self.app.get_mod("isaa")
            agent = await isaa.get_agent("docwriter")

            if not agent:
                return f"# {Path(source_file).stem}\n\nAI agent not available."

            # Extract code elements
            elements = self._extract_code_elements(Path(source_file))

            with open(source_file, 'r', encoding='utf-8') as f:
                source_content = f.read()[:3000]  # Limit source size

            prompt = f"""Generate comprehensive documentation for: {source_file}

Code:
```python
{source_content}
```

Elements: {len(elements)} classes/functions found

Create complete markdown with headers, overview, and examples."""

            result = await asyncio.wait_for(
                agent.a_run_llm_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model_preference="fast"
                ), timeout=12.0
            )
            return result

        except Exception as e:
            logger.error(f"Error generating file content: {e}")
            return f"# {Path(source_file).stem}\n\nError generating documentation: {e}"

    def _build_full_index(self) -> DocsIndex:
        """Build comprehensive index from scratch with better progress tracking"""
        logger.info("Building full documentation index...")

        index = DocsIndex()
        index.last_git_commit = self._get_git_commit()

        # Get target files
        target_files = self._get_target_files()
        logger.info(f"Processing {len(target_files)} files")

        code_files_processed = 0
        md_files_processed = 0

        for i, file_path in enumerate(target_files):
            try:
                if file_path.suffix == '.py':
                    elements = self._extract_code_elements(file_path)
                    for element in elements:
                        element_id = f"{element.file_path}:{element.name}"
                        if element.parent_class:
                            element_id = f"{element.file_path}:{element.parent_class}.{element.name}"
                        index.code_elements[element_id] = element
                    code_files_processed += 1

                elif file_path.suffix.lower() == '.md':
                    sections = self._parse_markdown_file(file_path)
                    for section in sections:
                        index.sections[section.section_id] = section
                    if sections:
                        md_files_processed += 1

                # Store file hash
                index.file_hashes[str(file_path)] = self._get_file_hash(file_path)

                # Progress logging
                if i % 50 == 0 and i > 0:
                    logger.info(f"Progress: {i}/{len(target_files)} files processed")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        index.last_indexed = datetime.now()
        logger.info(f"Index built: {len(index.code_elements)} elements from {code_files_processed} Python files, "
                    f"{len(index.sections)} sections from {md_files_processed} markdown files")

        return index

    async def _quick_update_index(self):
        """Quick index update for changed files only"""
        try:
            changes = await asyncio.get_event_loop().run_in_executor(
                None, self._get_git_changes
            )

            for change in changes[:20]:  # Limit changes processed
                file_path = Path(change)
                if not self._should_include_file(file_path):
                    continue

                if not file_path.exists():
                    self._remove_file_from_index(file_path)
                    continue

                new_hash = self._get_file_hash(file_path)
                old_hash = self.current_index.file_hashes.get(str(file_path))

                if new_hash != old_hash:
                    self._update_file_in_index(file_path)
                    self.current_index.file_hashes[str(file_path)] = new_hash

            self.current_index.last_indexed = datetime.now()

        except Exception as e:
            logger.error(f"Error in quick index update: {e}")

    async def _link_docs_to_code(self) -> int:
        """Link documentation sections to code elements"""
        linked_count = 0

        for section_id, section in self.current_index.sections.items():
            matches = self._find_code_matches(section)
            if matches:
                section.source_refs.extend(matches)
                section.source_refs = list(set(section.source_refs))  # Remove duplicates
                linked_count += 1

        return linked_count

    def _find_code_matches(self, section: DocSection) -> List[str]:
        """Find code elements that match a documentation section"""
        matches = []
        title_words = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', section.title.lower()))
        content_words = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', section.content.lower()))

        for element_id, element in self.current_index.code_elements.items():
            score = 0

            # Direct matches
            if element.name.lower() in title_words:
                score += 10
            if element.name.lower() in content_words:
                score += 5

            # File correlation
            if Path(element.file_path).stem.lower() in Path(section.file_path).stem.lower():
                score += 3

            if score >= 5:
                matches.append(element_id)

        return matches

    def _find_undocumented_elements(self) -> List[CodeElement]:
        """Find code elements without documentation"""
        undocumented = []

        for element_id, element in self.current_index.code_elements.items():
            has_docs = any(
                element_id in section.source_refs or
                element.name in section.content or
                element.name in section.title
                for section in self.current_index.sections.values()
            )

            if not has_docs:
                undocumented.append(element)

        return undocumented

    def _find_unclear_sections(self) -> List[str]:
        """Find sections with unclear content"""
        unclear = []
        unclear_indicators = ['todo', 'fixme', 'placeholder', 'coming soon', 'tbd']

        for section_id, section in self.current_index.sections.items():
            content_lower = section.content.lower()

            if (any(indicator in content_lower for indicator in unclear_indicators) or
                len(section.content.strip()) < 50):
                unclear.append(section_id)

        return unclear

    def _assess_priority(self, element: CodeElement) -> str:
        """Assess documentation priority for code element"""
        score = 0

        # Type weights
        type_scores = {'class': 5, 'function': 3, 'method': 2}
        score += type_scores.get(element.element_type, 1)

        # Complexity
        if element.signature and element.signature.count(',') > 2:
            score += 2

        # Public vs private
        if not element.name.startswith('_'):
            score += 2

        # No docstring
        if not element.docstring:
            score += 2

        return "high" if score >= 8 else "medium" if score >= 5 else "low"

    # ==================== FILE OPERATIONS ====================

    def _update_section_in_file(self, section: DocSection, new_content: str) -> bool:
        """Update section content in file"""
        try:
            file_path = Path(section.file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Replace section content
            new_section = f"{'#' * section.level} {section.title}\n\n{new_content}\n\n"
            new_lines = new_section.split('\n')

            # Update the lines
            lines[section.line_start:section.line_end + 1] = [line + '\n' for line in new_lines]

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return True
        except Exception as e:
            logger.error(f"Error updating section in file: {e}")
            return False

    def _add_section_to_file(self, file_path: str, section: DocSection, position: str) -> bool:
        """Add section to file"""
        try:
            full_path = self.docs_root / file_path

            if not full_path.exists():
                # Create new file
                content = f"{'#' * section.level} {section.title}\n\n{section.content}\n\n"
                full_path.write_text(content, encoding='utf-8')
                return True

            # Add to existing file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            new_section = f"\n{'#' * section.level} {section.title}\n\n{section.content}\n\n"

            if position == "top":
                content = new_section + content
            else:
                content = content + new_section

            full_path.write_text(content, encoding='utf-8')
            return True

        except Exception as e:
            logger.error(f"Error adding section to file: {e}")
            return False

    # ==================== UTILITY METHODS ====================

    def _get_target_files(self) -> List[Path]:
        """Get filtered list of target files"""
        files = []
        extensions = ['.py', '.md']

        search_dirs = [self.project_root / d for d in self.include_dirs if (self.project_root / d).exists()]
        if not search_dirs:
            search_dirs = [self.project_root]

        for search_dir in search_dirs:
            for ext in extensions:
                for file_path in search_dir.rglob(f"*{ext}"):
                    if self._should_include_file(file_path):
                        files.append(file_path)

        return list(set(files))

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        file_str = str(file_path)

        # Exclude patterns
        if any(exclude in file_str for exclude in self.exclude_dirs):
            return False

        # Include patterns
        if self.include_dirs:
            return any(file_path.is_relative_to(self.project_root / include_dir)
                       for include_dir in self.include_dirs
                       if (self.project_root / include_dir).exists())

        return True

    def _get_file_hash(self, file_path: Path) -> str:
        """Get file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _get_git_changes(self) -> List[str]:
        """Get changed files from git"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip().split('\n') if result.returncode == 0 else []
        except Exception:
            return []

    def _extract_code_elements(self, file_path: Path) -> List[CodeElement]:
        """Extract code elements from Python file"""
        elements = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    elements.append(CodeElement(
                        name=node.name,
                        element_type='class',
                        file_path=str(file_path),
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        signature=f"class {node.name}",
                        docstring=ast.get_docstring(node),
                        hash_signature=hashlib.md5(f"class {node.name}".encode()).hexdigest()
                    ))

                    # Add methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            elements.append(CodeElement(
                                name=item.name,
                                element_type='method',
                                file_path=str(file_path),
                                line_start=item.lineno,
                                line_end=getattr(item, 'end_lineno', item.lineno),
                                signature=f"def {item.name}",
                                docstring=ast.get_docstring(item),
                                parent_class=node.name,
                                hash_signature=hashlib.md5(f"def {item.name}".encode()).hexdigest()
                            ))

                elif isinstance(node, ast.FunctionDef):
                    elements.append(CodeElement(
                        name=node.name,
                        element_type='function',
                        file_path=str(file_path),
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        signature=f"def {node.name}",
                        docstring=ast.get_docstring(node),
                        hash_signature=hashlib.md5(f"def {node.name}".encode()).hexdigest()
                    ))

        except Exception as e:
            logger.error(f"Error extracting elements from {file_path}: {e}")

        return elements

    def _parse_markdown_file(self, file_path: Path) -> List[DocSection]:
        """Enhanced markdown file parsing with better error handling"""
        sections = []

        try:
            # Skip non-markdown files and hidden files
            if not file_path.suffix.lower() == '.md' or file_path.name.startswith('.'):
                return sections

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return sections

            lines = content.split('\n')
            current_section = None
            section_content = []
            line_start = 0

            for i, line in enumerate(lines):
                # Look for markdown headers with improved regex
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())

                if header_match:
                    # Save previous section if exists
                    if current_section:
                        section = self._create_doc_section(
                            file_path, current_section, section_content, line_start, i - 1
                        )
                        if section:  # Only add non-empty sections
                            sections.append(section)

                    # Start new section
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()

                    # Skip empty titles
                    if not title:
                        continue

                    current_section = (title, level)
                    section_content = []
                    line_start = i

                elif current_section:
                    section_content.append(line)

            # Save last section
            if current_section:
                section = self._create_doc_section(
                    file_path, current_section, section_content, line_start, len(lines) - 1
                )
                if section:
                    sections.append(section)

            if sections:
                logger.debug(f"Parsed {len(sections)} sections from {file_path}")

        except Exception as e:
            logger.error(f"Error parsing markdown {file_path}: {e}")

        return sections

    def _create_doc_section(self, file_path: Path, section_info: Tuple[str, int],
                            content_lines: List[str], line_start: int, line_end: int) -> Optional[DocSection]:
        """Create DocSection with validation"""
        try:
            title, level = section_info
            content = '\n'.join(content_lines).strip()

            # Skip sections with no meaningful content
            if len(content) < 10 and not any(c.isalnum() for c in content):
                return None

            # Extract tags and references with safer regex
            tags = []
            source_refs = []

            try:
                tags = re.findall(r'#([a-zA-Z][a-zA-Z0-9_-]*)', content)
                source_refs = re.findall(r'`([^`]+\.py:[^`]+)`', content)
            except Exception:
                pass  # If regex fails, continue with empty lists

            section_id = f"{file_path.name}#{title}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            combined_hash = hashlib.md5(f"{title}:{content}:{line_start}".encode()).hexdigest()

            return DocSection(
                section_id=section_id,
                file_path=str(file_path),
                title=title,
                content=content,
                level=level,
                line_start=line_start,
                line_end=line_end,
                source_refs=source_refs,
                tags=tags,
                hash_signature=combined_hash,
                content_hash=content_hash,
                last_modified=datetime.fromtimestamp(
                    file_path.stat().st_mtime) if file_path.exists() else datetime.now()
            )
        except Exception as e:
            logger.error(f"Error creating doc section: {e}")
            return None

    def _remove_file_from_index(self, file_path: Path):
        """Remove all references to a file from index"""
        file_str = str(file_path)

        # Remove sections
        to_remove = [k for k, v in self.current_index.sections.items() if v.file_path == file_str]
        for key in to_remove:
            del self.current_index.sections[key]

        # Remove code elements
        to_remove = [k for k, v in self.current_index.code_elements.items() if v.file_path == file_str]
        for key in to_remove:
            del self.current_index.code_elements[key]

        # Remove file hash
        if file_str in self.current_index.file_hashes:
            del self.current_index.file_hashes[file_str]

    def _update_file_in_index(self, file_path: Path):
        """Update index for a specific file"""
        # Remove old entries
        self._remove_file_from_index(file_path)

        # Add new entries
        try:
            if file_path.suffix == '.py':
                elements = self._extract_code_elements(file_path)
                for element in elements:
                    element_id = f"{element.file_path}:{element.name}"
                    if element.parent_class:
                        element_id = f"{element.file_path}:{element.parent_class}.{element.name}"
                    self.current_index.code_elements[element_id] = element

            elif file_path.suffix == '.md' and file_path.is_relative_to(self.docs_root):
                sections = self._parse_markdown_file(file_path)
                for section in sections:
                    self.current_index.sections[section.section_id] = section

        except Exception as e:
            logger.error(f"Error updating file {file_path}: {e}")


    def source_code_lookup(self,
                           element_name: Optional[str] = None,
                           file_path: Optional[str] = None,
                           element_type: Optional[str] = None,
                           max_results: int = 25,
                           return_code_block: bool = True) -> Result:
        """Look up source code elements with option to return single method code blocks"""
        try:
            if not self.current_index:
                self.current_index = self._load_index(minimal=False)

            matches = []

            for element_id, element in self.current_index.code_elements.items():
                if len(matches) >= max_results:
                    break

                # Apply filters
                if element_name and element_name.lower() not in element.name.lower():
                    continue
                if file_path and file_path not in element.file_path:
                    continue
                if element_type and element.element_type != element_type:
                    continue

                # Get code block for this specific element
                code_block = ""
                if return_code_block:
                    code_block = self._extract_single_code_block(element)

                # Find related docs
                related_docs = []
                for section_id, section in self.current_index.sections.items():
                    if len(related_docs) >= 3:  # Limit related docs
                        break
                    if (element_id in section.source_refs or
                        element.name in section.content[:200] or
                        element.name in section.title):
                        related_docs.append({
                            "section_id": section_id,
                            "title": section.title,
                            "file_path": section.file_path
                        })

                match_data = {
                    "element_id": element_id,
                    "name": element.name,
                    "type": element.element_type,
                    "signature": element.signature,
                    "file_path": element.file_path,
                    "line_start": element.line_start,
                    "line_end": element.line_end,
                    "parent_class": element.parent_class,
                    "docstring": element.docstring[:300] if element.docstring else None,
                    "related_documentation": related_docs
                }

                if return_code_block and code_block:
                    match_data["code_block"] = code_block

                matches.append(match_data)

            return Result.ok({
                "matches": matches,
                "total_matches": len(matches),
                "total_available": len(self.current_index.code_elements),
                "truncated": len(matches) >= max_results
            })

        except Exception as e:
            logger.error(f"Error in source code lookup: {e}")
            return Result.default_user_error(f"Error looking up source code: {e}")

    def _extract_single_code_block(self, element: CodeElement) -> str:
        """Extract single method/class code block from source file"""
        try:
            with open(element.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Get lines for this element only
            start_line = max(0, element.line_start - 1)  # Convert to 0-based
            end_line = min(len(lines), element.line_end)

            element_lines = lines[start_line:end_line]

            # Clean up the code block
            if element_lines:
                # Remove common leading whitespace
                import textwrap
                code_block = textwrap.dedent(''.join(element_lines))

                # Add language hint for syntax highlighting
                return f"```python\n{code_block}```"

            return ""

        except Exception as e:
            logger.error(f"Error extracting code block for {element.name}: {e}")
            return ""

    # Add the missing method to the app registration
    def add_source_code_lookup_to_app(self):
        """Add source code lookup method to app"""
        if hasattr(self, 'app'):
            self.app.source_code_lookup = self.source_code_lookup


# Update the app registration function
def add_to_app(app: AppType, include_dirs: List[str] = None, exclude_dirs: List[str] = None) -> MarkdownDocsSystem:
    """Add optimized markdown docs system to app"""
    from toolboxv2 import tb_root_dir

    docs_system = MarkdownDocsSystem(
        app,
        docs_root=str(tb_root_dir.parent / "docs"),
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs
    )

    # Register core functions
    app.docs_reader = docs_system.docs_reader
    app.docs_writer = docs_system.docs_writer
    app.get_update_suggestions = docs_system.get_update_suggestions
    app.auto_update_docs = docs_system.auto_update_docs
    app.initial_docs_parse = docs_system.initial_docs_parse
    app.source_code_lookup = docs_system.source_code_lookup

    return docs_system

