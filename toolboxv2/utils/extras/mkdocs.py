# coding: utf-8
import os
import json
import hashlib
import ast
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

import yaml
from tqdm import tqdm
from ..system.tb_logger import get_logger
from ..system.types import AppType, Result

logger = get_logger()


class ChangeType(Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


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


@dataclass
class CodeElement:
    """Represents a code element (class, function, etc.)"""
    name: str
    element_type: str  # 'class', 'function', 'variable', 'module'
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str] = None
    hash_signature: str = ""
    parent_class: Optional[str] = None


@dataclass
class DocSection:
    """Represents a documentation section"""
    section_id: str
    file_path: str
    title: str
    content: str
    level: int  # header level (1, 2, 3, etc.)
    line_start: int
    line_end: int
    source_refs: List[str] = field(default_factory=list)  # references to code elements
    tags: List[str] = field(default_factory=list)
    hash_signature: str = ""
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class DocsIndex:
    """Complete documentation index"""
    sections: Dict[str, DocSection] = field(default_factory=dict)
    code_elements: Dict[str, CodeElement] = field(default_factory=dict)
    import_refs: Dict[str, List[ImportReference]] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    last_git_commit: Optional[str] = None
    last_indexed: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


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
        """Get list of changed files since given commit"""
        changes = []

        try:
            if since_commit:
                # Get changes since specific commit
                cmd = ["git", "diff", "--name-status", f"{since_commit}..HEAD"]
            else:
                # Get all tracked files as new
                cmd = ["git", "ls-files"]

            result = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)

            if result.returncode != 0:
                return changes

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                if since_commit:
                    # Parse git diff output
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
                    # All files are "new"
                    changes.append(FileChange(
                        file_path=line.strip(),
                        change_type=ChangeType.ADDED
                    ))

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
        for file_path in tqdm(target_files, desc="Indexing files"):
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
        for include_dir in tqdm(self.include_dirs, desc="Searching include directories", total=len(self.include_dirs)):
            search_path = self.project_root / include_dir
            if search_path.exists() and search_path.is_dir():
                search_dirs.append(search_path)

        # If no include dirs exist, search entire project root
        if not search_dirs:
            search_dirs = [self.project_root]

        for search_dir in tqdm(search_dirs, desc="Searching for files", total=len(search_dirs)):
            print(f"Searching {search_dir} for files with extensions {extensions}")
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
    """Main bidirectional markdown documentation system"""

    def __init__(self, app: AppType, docs_root: str = "../docs", source_root: str = ".",
                 include_dirs: List[str] = None, exclude_dirs: List[str] = None):
        self.app = app
        self.docs_root = Path(docs_root)
        self.source_root = Path(source_root)
        self.project_root = Path.cwd()
        print(f"abs project_root: {self.project_root.absolute()}")
        print(f"abs docs_root   : {self.docs_root.absolute()}")
        print(f"abs source_root : {self.source_root.absolute()}")

        # Initialize components with directory filters
        self.indexer = DocsIndexer(
            self.project_root,
            self.docs_root,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs
        )
        self.section_manager = SectionManager(self.docs_root)

        # Index management
        self.index_file = self.docs_root / '.docs_index.json'
        self.current_index: Optional[DocsIndex] = None

        # Ensure directories exist
        self.docs_root.mkdir(exist_ok=True)

    def _load_index(self) -> DocsIndex:
        """Load index from file or create new one"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert back to dataclass objects
                index = DocsIndex()
                index.last_git_commit = data.get('last_git_commit')
                index.last_indexed = datetime.fromisoformat(data.get('last_indexed', datetime.now().isoformat()))
                index.version = data.get('version', '1.0')
                index.file_hashes = data.get('file_hashes', {})

                # Restore sections
                for section_id, section_data in data.get('sections', {}).items():
                    index.sections[section_id] = DocSection(
                        section_id=section_data['section_id'],
                        file_path=section_data['file_path'],
                        title=section_data['title'],
                        content=section_data['content'],
                        level=section_data['level'],
                        line_start=section_data['line_start'],
                        line_end=section_data['line_end'],
                        source_refs=section_data.get('source_refs', []),
                        tags=section_data.get('tags', []),
                        hash_signature=section_data.get('hash_signature', ''),
                        last_modified=datetime.fromisoformat(
                            section_data.get('last_modified', datetime.now().isoformat()))
                    )

                # Restore code elements
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

                # Restore import refs
                for file_path, imports_data in data.get('import_refs', {}).items():
                    imports = []
                    for import_data in imports_data:
                        imports.append(ImportReference(
                            source_file=import_data['source_file'],
                            target_file=import_data['target_file'],
                            import_name=import_data['import_name'],
                            line_number=import_data['line_number'],
                            import_type=import_data['import_type']
                        ))
                    index.import_refs[file_path] = imports

                return index

            except Exception as e:
                logger.error(f"Error loading index: {e}")

        # Build initial index if file doesn't exist or loading failed
        return self.indexer.build_initial_index()

    def _save_index(self, index: DocsIndex):
        """Save index to file"""
        try:
            # Convert to serializable format
            data = {
                'version': index.version,
                'last_git_commit': index.last_git_commit,
                'last_indexed': index.last_indexed.isoformat(),
                'file_hashes': index.file_hashes,
                'sections': {},
                'code_elements': {},
                'import_refs': {}
            }

            # Convert sections
            for section_id, section in index.sections.items():
                data['sections'][section_id] = asdict(section)
                data['sections'][section_id]['last_modified'] = section.last_modified.isoformat()

            # Convert code elements
            for element_id, element in index.code_elements.items():
                data['code_elements'][element_id] = asdict(element)

            # Convert import refs
            for file_path, imports in index.import_refs.items():
                data['import_refs'][file_path] = [asdict(imp) for imp in imports]

            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    async def docs_reader(self,
                          query: Optional[str] = None,
                          section_id: Optional[str] = None,
                          file_path: Optional[str] = None,
                          tags: Optional[List[str]] = None,
                          include_source_refs: bool = True,
                          format_type: str = "structured") -> Result:
        """
        Read documentation using only the index (no AI required)

        Args:
            query: Text to search for in titles and content
            section_id: Specific section ID to retrieve
            file_path: Filter by specific file path
            tags: Filter by tags
            include_source_refs: Include source code references
            format_type: Output format ('structured', 'markdown', 'json')
        """
        try:
            # Load or update index
            if not self.current_index:
                self.current_index = self._load_index()
            else:
                # Check for updates
                self.current_index, update_notes = self.indexer.update_index(self.current_index)
                if update_notes:
                    self._save_index(self.current_index)

            # Filter sections based on criteria
            matching_sections = []

            for sid, section in self.current_index.sections.items():
                # App filters
                if section_id and sid != section_id:
                    continue

                if file_path and section.file_path != file_path:
                    continue

                if tags and not any(tag in section.tags for tag in tags):
                    continue

                if query:
                    # Simple text search in title and content
                    search_text = f"{section.title} {section.content}".lower()
                    if query.lower() not in search_text:
                        continue

                matching_sections.append(section)

            # Format output based on type
            if format_type == "markdown":
                output = []
                for section in matching_sections:
                    output.append(f"{'#' * section.level} {section.title}")
                    output.append("")
                    output.append(section.content)
                    if include_source_refs and section.source_refs:
                        output.append("")
                        output.append("**Source References:**")
                        for ref in section.source_refs:
                            output.append(f"- `{ref}`")
                    output.append("")

                return Result.ok("\n".join(output))

            elif format_type == "json":
                output = []
                for section in matching_sections:
                    section_data = asdict(section)
                    section_data['last_modified'] = section.last_modified.isoformat()
                    if not include_source_refs:
                        section_data.pop('source_refs', None)
                    output.append(section_data)

                return Result.ok(output)

            else:  # structured
                output = {
                    "sections": [],
                    "metadata": {
                        "total_sections": len(self.current_index.sections),
                        "matching_sections": len(matching_sections),
                        "last_indexed": self.current_index.last_indexed.isoformat(),
                        "git_commit": self.current_index.last_git_commit
                    }
                }

                for section in matching_sections:
                    section_data = {
                        "id": section.section_id,
                        "title": section.title,
                        "content": section.content,
                        "file_path": section.file_path,
                        "level": section.level,
                        "tags": section.tags
                    }

                    if include_source_refs:
                        section_data["source_refs"] = section.source_refs

                        # Include referenced code elements
                        referenced_elements = []
                        for ref in section.source_refs:
                            if ref in self.current_index.code_elements:
                                elem = self.current_index.code_elements[ref]
                                referenced_elements.append({
                                    "name": elem.name,
                                    "type": elem.element_type,
                                    "signature": elem.signature,
                                    "file_path": elem.file_path,
                                    "line_start": elem.line_start,
                                    "parent_class": elem.parent_class
                                })
                        section_data["referenced_elements"] = referenced_elements

                    output["sections"].append(section_data)

                return Result.ok(output)

        except Exception as e:
            logger.error(f"Error in docs_reader: {e}")
            return Result.default_user_error(f"Error reading documentation: {e}")

    async def docs_writer(self,
                          action: str,  # 'create_file', 'add_section', 'update_section', 'generate_from_code'
                          file_path: Optional[str] = None,
                          section_title: Optional[str] = None,
                          content: Optional[str] = None,
                          source_file: Optional[str] = None,
                          auto_generate: bool = False,
                          position: Optional[str] = None,
                          level: int = 2) -> Result:
        """
        Write documentation with precise programmatic control

        Args:
            action: Type of action to perform
            file_path: Target documentation file
            section_title: Title for new/updated section
            content: Content for the section (if not auto-generating)
            source_file: Source code file to generate docs from
            auto_generate: Use AI to generate content
            position: Where to place new sections ('top', 'bottom', 'after:SectionName')
            level: Header level for new sections
        """
        try:
            # Load current index
            if not self.current_index:
                self.current_index = self._load_index()

            result = {}

            if action == "create_file":
                if not file_path:
                    return Result.default_user_error("file_path required for create_file")

                initial_content = content or f"# {Path(file_path).stem.replace('_', ' ').title()}\n\n"
                create_result = self.section_manager.create_new_file(file_path, initial_content)

                if create_result.is_error():
                    return create_result

                result.update(create_result.data)

            elif action == "add_section":
                if not file_path or not section_title:
                    return Result.default_user_error("file_path and section_title required for add_section")

                # Generate content if requested
                if auto_generate and source_file:
                    content = await self._generate_section_content(section_title, source_file)

                if not content:
                    content = f"Content for {section_title}\n\nTODO: Add documentation here."

                # Create section object
                section = DocSection(
                    section_id=f"{file_path}#{section_title}",
                    file_path=str(self.docs_root / file_path),
                    title=section_title,
                    content=content,
                    level=level,
                    line_start=0,  # Will be updated by section_manager
                    line_end=0,
                    tags=[],
                    hash_signature=hashlib.md5(content.encode()).hexdigest()
                )

                add_result = self.section_manager.add_section(file_path, section, position)
                if add_result.is_error():
                    return add_result

                result.update(add_result.data)

            elif action == "update_section":
                if not section_title or not file_path:
                    return Result.default_user_error("section_title and file_path required for update_section")

                # Find existing section
                section_id = f"{Path(file_path).name}#{section_title}"
                if section_id not in self.current_index.sections:
                    return Result.default_user_error(f"Section not found: {section_id}")

                section = self.current_index.sections[section_id]

                # Generate new content if requested
                if auto_generate and source_file:
                    content = await self._generate_section_content(section_title, source_file)

                if not content:
                    return Result.default_user_error("content required for update_section")

                update_result = self.section_manager.update_section(section, content)
                if update_result.is_error():
                    return update_result

                result.update(update_result.data)

            elif action == "generate_from_code":
                if not source_file:
                    return Result.default_user_error("source_file required for generate_from_code")

                source_path = Path(source_file)
                if not source_path.exists():
                    return Result.default_user_error(f"Source file not found: {source_file}")

                # Determine target file path
                if not file_path:
                    file_path = f"{source_path.stem}.md"

                # Generate documentation for the entire file
                generated_content = await self._generate_file_documentation(source_path)

                # Create or update file
                create_result = self.section_manager.create_new_file(file_path, generated_content)
                if create_result.is_error():
                    return create_result

                result.update(create_result.data)

            else:
                return Result.default_user_error(f"Unknown action: {action}")

            # Update index after changes
            self.current_index, update_notes = self.indexer.update_index(self.current_index)
            self._save_index(self.current_index)

            result["index_updates"] = update_notes
            result["timestamp"] = datetime.now().isoformat()

            return Result.ok(result)

        except Exception as e:
            logger.error(f"Error in docs_writer: {e}")
            return Result.default_user_error(f"Error writing documentation: {e}")

    async def _generate_section_content(self, title: str, source_file: str) -> str:
        """Generate content for a section using AI"""
        try:
            isaa = self.app.get_mod("isaa")
            agent = await isaa.get_agent("mcp-agent")  # or another configured agent

            if not agent:
                return f"# {title}\n\nAI agent not available. Please add content manually."

            # Read source file
            with open(source_file, 'r', encoding='utf-8') as f:
                source_content = f.read()

            prompt = f"""
Generate documentation content for the section "{title}" based on the following source code:

File: {source_file}
```python
{source_content}
```

Requirements:
- Focus on the specific aspect indicated by the title "{title}"
- Write clear, concise documentation suitable for both developers and LLMs
- Include code examples if relevant
- Use markdown formatting
- Do not include the section header (it will be added automatically)
"""

            result = await agent.run(prompt)
            return result.data if hasattr(result, 'data') else str(result)

        except Exception as e:
            logger.error(f"Error generating section content: {e}")
            return f"# {title}\n\nError generating content: {e}\n\nTODO: Add documentation manually."

    async def _generate_file_documentation(self, source_path: Path) -> str:
        """Generate complete documentation for a source file"""
        try:
            isaa = self.app.get_mod("isaa")
            agent = await isaa.get_agent("mcp-agent")

            if not agent:
                return f"# {source_path.stem}\n\nAI agent not available. Please add documentation manually."

            # Get code elements for this file
            elements = self.indexer.code_extractor.extract_python_elements(source_path)

            # Read source content
            with open(source_path, 'r', encoding='utf-8') as f:
                source_content = f.read()

            prompt = f"""
Generate comprehensive markdown documentation for the Python file: {source_path}

Source Code:
```python
{source_content}
```

Code Elements Found:
{chr(10).join([f"- {elem.element_type}: {elem.name} (line {elem.line_start})" for elem in elements])}

Requirements:
- Create a complete markdown document with proper headers
- Include file overview and purpose
- Document all classes and their methods
- Document standalone functions
- Include usage examples where appropriate
- Make it suitable for both developers and LLMs
- Use clear, professional language
"""

            result = await agent.run(prompt)
            return result.data if hasattr(result, 'data') else str(result)

        except Exception as e:
            logger.error(f"Error generating file documentation: {e}")
            return f"# {source_path.stem}\n\nError generating documentation: {e}"

    async def get_update_suggestions(self, force_scan: bool = False,
                                     priority_filter: List[str] = None) -> Result:
        """Get list of suggested documentation updates with enhanced filtering"""
        try:
            if not self.current_index:
                self.current_index = self._load_index()

            # Update index to detect changes
            updated_index, update_notes = self.indexer.update_index(
                self.current_index,
                force_full_scan=force_scan
            )

            suggestions = []

            # Analyze changes and suggest documentation updates
            for note in update_notes:
                suggestion = self._analyze_update_note(note, updated_index)
                if suggestion:
                    suggestions.append(suggestion)

            # Filter by priority if specified
            if priority_filter:
                suggestions = [s for s in suggestions if s['priority'] in priority_filter]

            # Sort by priority: high, medium, low
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            suggestions.sort(key=lambda x: priority_order.get(x['priority'], 3))

            # Save updated index
            self.current_index = updated_index
            self._save_index(self.current_index)

            return Result.ok({
                "suggestions": suggestions,
                "total_suggestions": len(suggestions),
                "update_notes": update_notes,
                "force_scan_used": force_scan,
                "index_stats": {
                    "code_elements": len(updated_index.code_elements),
                    "doc_sections": len(updated_index.sections),
                    "import_refs": len(updated_index.import_refs)
                }
            })

        except Exception as e:
            logger.error(f"Error getting update suggestions: {e}")
            return Result.default_user_error(f"Error analyzing updates: {e}")

    def _analyze_update_note(self, note: str, index: DocsIndex) -> Optional[Dict]:
        """Analyze an update note and create suggestion"""
        suggestion = None

        # Extract file path and action from note
        if "Updated" in note or "Added" in note:
            # Extract file path
            parts = note.split(": ", 1)
            if len(parts) > 1:
                file_path = parts[1].strip()
                path_obj = Path(file_path)

                if path_obj.suffix == '.py':
                    # Check for existing documentation
                    doc_file = f"{path_obj.stem}.md"
                    has_docs = any(
                        section.file_path.endswith(doc_file)
                        for section in index.sections.values()
                    )

                    # Analyze code complexity to determine priority
                    priority = self._assess_documentation_priority(file_path, index)

                    if has_docs:
                        suggestion = {
                            "type": "update_existing",
                            "source_file": file_path,
                            "doc_file": doc_file,
                            "suggestion": f"Update documentation for {path_obj.name} - code has changed",
                            "action": "update_section",
                            "priority": priority,
                            "change_type": "code_modified"
                        }
                    else:
                        suggestion = {
                            "type": "create_new",
                            "source_file": file_path,
                            "doc_file": doc_file,
                            "suggestion": f"Create documentation for undocumented file {path_obj.name}",
                            "action": "generate_from_code",
                            "priority": "high" if priority == "high" else "medium",
                            "change_type": "new_file"
                        }

        elif "Removed" in note:
            parts = note.split(": ", 1)
            if len(parts) > 1:
                file_path = parts[1].strip()
                path_obj = Path(file_path)

                if path_obj.suffix == '.py':
                    doc_file = f"{path_obj.stem}.md"
                    suggestion = {
                        "type": "cleanup",
                        "source_file": file_path,
                        "doc_file": doc_file,
                        "suggestion": f"Consider removing/updating docs for deleted file {path_obj.name}",
                        "action": "manual_review",
                        "priority": "low",
                        "change_type": "file_deleted"
                    }

        return suggestion

    def _assess_documentation_priority(self, file_path: str, index: DocsIndex) -> str:
        """Assess priority for documentation based on code complexity"""
        # Count code elements in this file
        elements = [
            elem for elem in index.code_elements.values()
            if elem.file_path == file_path
        ]

        # Count classes and functions
        classes = len([e for e in elements if e.element_type == 'class'])
        functions = len([e for e in elements if e.element_type in ['function', 'method']])

        # Check if file has docstrings
        has_docstrings = any(elem.docstring for elem in elements)

        # Priority assessment
        if classes > 2 or functions > 5:
            return "high"
        elif classes > 0 or functions > 2 or not has_docstrings:
            return "medium"
        else:
            return "low"

    async def auto_update_docs(self, dry_run: bool = False,
                               max_updates: int = 10,
                               priority_filter: List[str] = None,
                               force_scan: bool = False) -> Result:
        """Automatically update documentation based on suggestions"""
        try:
            # Get suggestions
            suggestions_result = await self.get_update_suggestions(
                force_scan=force_scan,
                priority_filter=priority_filter
            )

            if suggestions_result.is_error():
                return suggestions_result

            suggestions = suggestions_result.data['suggestions'][:max_updates]

            if dry_run:
                return Result.ok({
                    "dry_run": True,
                    "would_update": len(suggestions),
                    "suggestions": suggestions
                })

            results = []

            for suggestion in suggestions:
                try:
                    if suggestion['action'] == 'generate_from_code':
                        result = await self.docs_writer(
                            action="generate_from_code",
                            source_file=suggestion['source_file'],
                            file_path=suggestion['doc_file'],
                            auto_generate=True
                        )

                    elif suggestion['action'] == 'update_section':
                        # Find main class/function to update
                        elements = [
                            elem for elem in self.current_index.code_elements.values()
                            if elem.file_path == suggestion['source_file']
                        ]

                        if elements:
                            main_element = max(elements, key=lambda x: 1 if x.element_type == 'class' else 0)

                            result = await self.docs_writer(
                                action="update_section",
                                file_path=suggestion['doc_file'],
                                section_title=main_element.name,
                                source_file=suggestion['source_file'],
                                auto_generate=True
                            )
                        else:
                            continue

                    else:
                        # Skip manual review items
                        continue

                    if result.is_ok():
                        results.append({
                            "suggestion": suggestion,
                            "result": "success",
                            "details": result.data
                        })
                    else:
                        results.append({
                            "suggestion": suggestion,
                            "result": "error",
                            "error": result.error
                        })

                except Exception as e:
                    results.append({
                        "suggestion": suggestion,
                        "result": "error",
                        "error": str(e)
                    })

            successful_updates = len([r for r in results if r['result'] == 'success'])

            return Result.ok({
                "total_suggestions": len(suggestions),
                "processed": len(results),
                "successful_updates": successful_updates,
                "results": results
            })

        except Exception as e:
            logger.error(f"Error in auto_update_docs: {e}")
            return Result.default_user_error(f"Auto-update failed: {e}")

    async def source_code_lookup(self,
                                 element_name: Optional[str] = None,
                                 file_path: Optional[str] = None,
                                 element_type: Optional[str] = None) -> Result:
        """Look up source code elements using the index"""
        try:
            if not self.current_index:
                self.current_index = self._load_index()

            matches = []

            for element_id, element in self.current_index.code_elements.items():
                if element_name and element_name.lower() not in element.name.lower():
                    continue

                if file_path and file_path not in element.file_path:
                    continue

                if element_type and element.element_type != element_type:
                    continue

                # Find related documentation
                related_docs = []
                for section_id, section in self.current_index.sections.items():
                    if element_id in section.source_refs or element.name in section.content:
                        related_docs.append({
                            "section_id": section_id,
                            "title": section.title,
                            "file_path": section.file_path
                        })

                matches.append({
                    "element_id": element_id,
                    "name": element.name,
                    "type": element.element_type,
                    "signature": element.signature,
                    "file_path": element.file_path,
                    "line_start": element.line_start,
                    "line_end": element.line_end,
                    "parent_class": element.parent_class,
                    "docstring": element.docstring,
                    "related_documentation": related_docs
                })

            return Result.ok({
                "matches": matches,
                "total_matches": len(matches)
            })

        except Exception as e:
            logger.error(f"Error in source code lookup: {e}")
            return Result.default_user_error(f"Error looking up source code: {e}")

    async def initial_docs_parse(self, update_index: bool = True) -> Result:
        """Parse existing documentation and TOCs, complete the index"""
        try:
            logger.info("Starting initial documentation parse...")

            # Load current index
            if not self.current_index:
                self.current_index = self._load_index()

            toc_entries = []
            nav_structure = {}

            # Parse mkdocs.yml if exists
            mkdocs_file = self.project_root / "mkdocs.yml"
            if mkdocs_file.exists():
                with open(mkdocs_file, 'r', encoding='utf-8') as f:
                    mkdocs_config = yaml.safe_load(f)

                nav_structure = self._extract_nav_structure(mkdocs_config.get('nav', []))

            # Scan all markdown files for TOC structures
            glob = self.docs_root.rglob('*.md')
            for md_file in glob:
                file_tocs = self._extract_file_tocs(md_file)
                toc_entries.extend(file_tocs)

            # Cross-reference TOCs with code elements
            print(f"len(toc_entries) {len(toc_entries)} {toc_entries} {glob}")
            linked_entries = 0
            for toc_entry in tqdm(toc_entries, desc="Linking TOC entries to code", total=len(toc_entries)):
                # Try to find matching code elements
                potential_matches = self._find_code_matches(toc_entry.title)
                if potential_matches:
                    toc_entry.source_refs = potential_matches
                    toc_entry.has_implementation = True
                    linked_entries += 1

            # Update index if requested
            if update_index:
                # Force re-index all files to ensure completeness
                self.current_index, update_notes = self.indexer.update_index(
                    self.current_index,
                    force_full_scan=True
                )
                self._save_index(self.current_index)

            result_data = {
                "total_toc_entries": len(toc_entries),
                "linked_entries": linked_entries,
                "nav_structure": nav_structure,
                "unlinked_entries": [
                    {
                        "title": entry.title,
                        "file": entry.file_path,
                        "level": entry.level
                    }
                    for entry in toc_entries if not entry.has_implementation
                ],
                "index_updated": update_index,
                "completion_rate": f"{(linked_entries / len(toc_entries) * 100):.1f}%" if toc_entries else "0%"
            }

            logger.info(f"Parsed {len(toc_entries)} TOC entries, linked {linked_entries}")

            return Result.ok(result_data)

        except Exception as e:
            logger.error(f"Error in initial_docs_parse: {e}")
            return Result.default_user_error(f"Failed to parse existing docs: {e}")

    async def auto_adapt_docs_to_index(self, create_missing: bool = True,
                                       update_existing: bool = True) -> Result:
        """Automatically adapt documentation to match current code index"""
        try:
            if not self.current_index:
                self.current_index = self._load_index()

            adaptations = []

            # Find code elements without documentation
            undocumented_elements = []
            items = self.current_index.code_elements.items()
            for element_id, element in tqdm(items, desc="Finding undocumented elements", total=len(items)):
                has_docs = any(
                    element.name in section.content or element_id in section.source_refs
                    for section in self.current_index.sections.values()
                )

                if not has_docs:
                    undocumented_elements.append(element)

            # Create documentation for undocumented elements
            if create_missing:
                for element in tqdm(undocumented_elements[:10], desc="Creating missing documentation", total=10):  # Limit to avoid overwhelming
                    doc_file = f"{Path(element.file_path).stem}.md"

                    result = await self.docs_writer(
                        action="add_section" if (self.docs_root / doc_file).exists() else "create_file",
                        file_path=doc_file,
                        section_title=element.name,
                        source_file=element.file_path,
                        auto_generate=True,
                        level=2 if element.element_type == "method" else 1
                    )

                    if result.is_ok():
                        adaptations.append({
                            "action": "created_docs",
                            "element": element.name,
                            "file": doc_file,
                            "type": element.element_type
                        })

            # Update existing documentation that's outdated
            if update_existing:
                items = self.current_index.sections.items()
                for section_id, section in tqdm(items, desc="Updating outdated documentation", total=len(items)):
                    if section.source_refs:
                        # Check if referenced code has changed
                        needs_update = False
                        for ref in section.source_refs:
                            if ref in self.current_index.code_elements:
                                element = self.current_index.code_elements[ref]
                                # Simple hash comparison to detect changes
                                if element.hash_signature and section.hash_signature:
                                    if element.hash_signature not in section.content:
                                        needs_update = True
                                        break

                        if needs_update:
                            # Update the section
                            result = await self.docs_writer(
                                action="update_section",
                                file_path=Path(section.file_path).name,
                                section_title=section.title,
                                source_file=section.source_refs[0].split(':')[0] if section.source_refs else None,
                                auto_generate=True
                            )

                            if result.is_ok():
                                adaptations.append({
                                    "action": "updated_docs",
                                    "section": section.title,
                                    "file": section.file_path,
                                    "reason": "code_changed"
                                })

            return Result.ok({
                "total_adaptations": len(adaptations),
                "undocumented_elements": len(undocumented_elements),
                "adaptations": adaptations
            })

        except Exception as e:
            logger.error(f"Error in auto_adapt_docs_to_index: {e}")
            return Result.default_user_error(f"Failed to adapt docs: {e}")

    async def find_unclear_and_missing(self, analyze_tocs: bool = True) -> Result:
        """Find unclear documentation and missing implementations from TOC sections"""
        try:
            if not self.current_index:
                self.current_index = self._load_index()

            analyzer = DocsAnalyzer(self.current_index, self.project_root)

            # Find unclear sections
            unclear_sections = analyzer.find_unclear_sections()

            # Find missing implementations
            missing_implementations = analyzer.find_missing_implementations()

            # Analyze TOC structure if requested
            toc_issues = []
            if analyze_tocs:
                items = self.docs_root.rglob('*.md')
                for md_file in items:
                    file_issues = self._analyze_toc_structure(md_file)
                    toc_issues.extend(file_issues)

            # Find orphaned sections (docs without corresponding code)
            orphaned_sections = []
            items = self.current_index.sections.items()
            for section_id, section in items:
                if section.source_refs:
                    has_valid_refs = any(
                        ref in self.current_index.code_elements
                        for ref in section.source_refs
                    )
                    if not has_valid_refs:
                        orphaned_sections.append({
                            "section_id": section_id,
                            "title": section.title,
                            "file_path": section.file_path,
                            "invalid_refs": section.source_refs
                        })

            result_data = {
                "unclear_sections": [
                    {
                        "section_id": sid,
                        "title": self.current_index.sections[sid].title,
                        "file_path": self.current_index.sections[sid].file_path,
                        "content_preview": self.current_index.sections[sid].content[:100] + "..."
                    }
                    for sid in unclear_sections
                ],
                "missing_implementations": missing_implementations,
                "toc_issues": toc_issues,
                "orphaned_sections": orphaned_sections,
                "summary": {
                    "total_unclear": len(unclear_sections),
                    "total_missing": len(missing_implementations),
                    "total_toc_issues": len(toc_issues),
                    "total_orphaned": len(orphaned_sections)
                }
            }

            return Result.ok(result_data)

        except Exception as e:
            logger.error(f"Error in find_unclear_and_missing: {e}")
            return Result.default_user_error(f"Failed to analyze docs: {e}")

    async def rebuild_clean_docs(self, keep_unclear: bool = True, keep_missing: bool = True,
                                 keep_level: int = 1, update_mkdocs: bool = True) -> Result:
        """Rebuild and clean documentation with options to preserve content"""
        try:
            if not self.current_index:
                self.current_index = self._load_index()

            # Analyze current state
            analysis_result = await self.find_unclear_and_missing()
            if analysis_result.is_error():
                return analysis_result

            analysis = analysis_result.data

            # Create backup
            backup_dir = self.docs_root / f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)

            for md_file in self.docs_root.rglob('*.md'):
                if not md_file.name.startswith('.'):
                    backup_file = backup_dir / md_file.relative_to(self.docs_root)
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    backup_file.write_text(md_file.read_text(), encoding='utf-8')

            rebuilt_files = []
            cleaned_sections = []

            # Group sections by file
            files_sections = {}
            for section_id, section in self.current_index.sections.items():
                file_path = section.file_path
                if file_path not in files_sections:
                    files_sections[file_path] = []
                files_sections[file_path].append(section)

            # Process each file
            for file_path, sections in files_sections.items():
                if not Path(file_path).exists():
                    continue

                new_content = []
                file_name = Path(file_path).name

                # Add file header
                title = file_name.replace('.md', '').replace('_', ' ').title()
                new_content.append(f"# {title}\n")

                # Process sections based on rules
                for section in sorted(sections, key=lambda x: x.line_start):
                    section_id = section.section_id
                    should_keep = True

                    # Check if section should be removed
                    if not keep_unclear and section_id in [s["section_id"] for s in analysis["unclear_sections"]]:
                        should_keep = False
                        cleaned_sections.append({"section": section.title, "reason": "unclear", "file": file_name})

                    if not keep_missing and section_id in [s["section_id"] for s in analysis["orphaned_sections"]]:
                        should_keep = False
                        cleaned_sections.append({"section": section.title, "reason": "missing_impl", "file": file_name})

                    # Check level filtering
                    if section.level > keep_level + 1:  # Allow one level deeper than specified
                        should_keep = False
                        cleaned_sections.append({"section": section.title, "reason": "too_deep", "file": file_name})

                    if should_keep:
                        # Add section with proper formatting
                        header_level = min(section.level, keep_level + 2)
                        new_content.append(f"{'#' * header_level} {section.title}\n")
                        new_content.append(f"{section.content}\n")

                # Write cleaned file
                Path(file_path).write_text('\n'.join(new_content), encoding='utf-8')
                rebuilt_files.append(file_name)

            # Update mkdocs.yml if requested
            mkdocs_updated = False
            if update_mkdocs:
                mkdocs_result = self._update_mkdocs_config()
                mkdocs_updated = mkdocs_result.is_ok()

            # Rebuild index after cleaning
            self.current_index = self.indexer.build_initial_index()
            self._save_index(self.current_index)

            return Result.ok({
                "rebuilt_files": rebuilt_files,
                "cleaned_sections": cleaned_sections,
                "backup_location": str(backup_dir),
                "mkdocs_updated": mkdocs_updated,
                "settings": {
                    "keep_unclear": keep_unclear,
                    "keep_missing": keep_missing,
                    "keep_level": keep_level
                },
                "summary": {
                    "files_processed": len(rebuilt_files),
                    "sections_cleaned": len(cleaned_sections),
                    "backup_created": True
                }
            })

        except Exception as e:
            logger.error(f"Error in rebuild_clean_docs: {e}")
            return Result.default_user_error(f"Failed to rebuild docs: {e}")

    def _extract_nav_structure(self, nav_list: List) -> Dict:
        """Extract navigation structure from mkdocs nav"""
        structure = {}

        def process_nav_item(item, parent_key="root"):
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str):
                        # File reference
                        structure[f"{parent_key}.{key}"] = value
                    elif isinstance(value, list):
                        # Nested structure
                        for subitem in value:
                            process_nav_item(subitem, f"{parent_key}.{key}")
            elif isinstance(item, str):
                # Direct file reference
                structure[f"{parent_key}.{item}"] = item

        for item in nav_list:
            process_nav_item(item)

        return structure

    def _extract_file_tocs(self, file_path: Path) -> List[TOCEntry]:
        """Extract table of contents from a markdown file"""
        toc_entries = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()

                    toc_entries.append(TOCEntry(
                        title=title,
                        file_path=str(file_path),
                        level=level,
                        line_number=i + 1
                    ))

        except Exception as e:
            logger.error(f"Error extracting TOC from {file_path}: {e}")

        return toc_entries

    def _find_code_matches(self, title: str) -> List[str]:
        """Find code elements that match a documentation title"""
        matches = []
        title_lower = title.lower()

        for element_id, element in self.current_index.code_elements.items():
            if element.name.lower() in title_lower or title_lower in element.name.lower():
                matches.append(element_id)

            # Check signature matches
            if hasattr(element, 'signature') and element.signature:
                if any(word in element.signature.lower() for word in title_lower.split()):
                    matches.append(element_id)

        return matches

    def _analyze_toc_structure(self, file_path: Path) -> List[Dict]:
        """Analyze TOC structure for issues"""
        issues = []
        toc_entries = self._extract_file_tocs(file_path)

        if not toc_entries:
            return issues

        # Check for proper header hierarchy
        prev_level = 0
        for entry in toc_entries:
            if entry.level > prev_level + 1:
                issues.append({
                    "type": "header_jump",
                    "file": str(file_path),
                    "line": entry.line_number,
                    "title": entry.title,
                    "description": f"Header level jumps from {prev_level} to {entry.level}"
                })
            prev_level = entry.level

        return issues

    def _update_mkdocs_config(self) -> Result:
        """Update mkdocs.yml based on current documentation structure"""
        try:
            mkdocs_file = self.project_root / "mkdocs.yml"

            if mkdocs_file.exists():
                with open(mkdocs_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                config = {
                    'site_name': 'Project Documentation',
                    'theme': {'name': 'material'}
                }

            # Build new nav structure based on existing docs
            nav_structure = []

            # Group sections by file
            file_sections = {}
            for section_id, section in self.current_index.sections.items():
                file_name = Path(section.file_path).name
                if file_name not in file_sections:
                    file_sections[file_name] = []
                file_sections[file_name].append(section)

            # Create nav entries
            for file_name in sorted(file_sections.keys()):
                sections = file_sections[file_name]
                display_name = file_name.replace('.md', '').replace('_', ' ').title()

                # Only add files that have substantial content
                total_content = sum(len(s.content) for s in sections)
                if total_content > 100:  # Minimum content threshold
                    nav_structure.append({display_name: file_name})

            config['nav'] = nav_structure

            # Write updated config
            with open(mkdocs_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            return Result.ok({"nav_entries": len(nav_structure)})

        except Exception as e:
            return Result.default_user_error(f"Failed to update mkdocs.yml: {e}")


def add_to_app(app: AppType, include_dirs: List[str] = None,
               exclude_dirs: List[str] = None) -> MarkdownDocsSystem:
    """Add markdown docs system to app with directory configuration"""

    # Default include directories for common project structures
    if include_dirs is None:
        include_dirs = ["toolboxv2", "flows", "mods", "utils", "tbjs","tests", "tcm", "docs"]

    # Enhanced exclude directories
    if exclude_dirs is None:
        exclude_dirs = [
            "__pycache__", ".git", "node_modules", ".venv", "venv", "env", "python_env",
            ".pytest_cache", ".mypy_cache", "dist", "build", ".tox",
            "coverage_html_report", ".coverage", ".next", ".nuxt",
            "target", "bin", "obj", ".gradle", ".idea", ".vscode",
            "temp", "tmp", "logs", ".cache", "coverage", ".data", ".config", ".info", "web", "simple-core", "src-core",
        ]
    from toolboxv2 import tb_root_dir
    docs_system = MarkdownDocsSystem(
        app,
        docs_root=str(tb_root_dir.parent / "docs"),
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs
    )

    # Register functions
    app.docs_reader = docs_system.docs_reader
    app.docs_writer = docs_system.docs_writer
    app.get_update_suggestions = docs_system.get_update_suggestions
    app.source_code_lookup = docs_system.source_code_lookup
    app.auto_update_docs = docs_system.auto_update_docs

    app.initial_docs_parse = docs_system.initial_docs_parse
    app.auto_adapt_docs_to_index = docs_system.auto_adapt_docs_to_index
    app.find_unclear_and_missing = docs_system.find_unclear_and_missing
    app.rebuild_clean_docs = docs_system.rebuild_clean_docs

    return docs_system
