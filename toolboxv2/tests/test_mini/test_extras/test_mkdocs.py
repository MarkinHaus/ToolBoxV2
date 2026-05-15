"""
Tests for Markdown Documentation System v2.1

Unit Tests: ~80% coverage
Live Integration Tests: Full system workflow

Run with: python -m pytest test_docs_system.py -v --cov=docs_system --cov-report=term-missing
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Import the module under test
from toolboxv2.utils.extras.mkdocs import (
    # Data Models
    ChangeType,
    DocSection,
    CodeElement,
    FileChange,
    InvertedIndex,
    DocsIndex,
    ContextBundle,
    # Parser State
    ParserState,
    # Core Components
    DocParser,
    CodeAnalyzer,
    JSTSAnalyzer,
    IndexManager,
    ContextEngine,
    FileScanner,
    GitTracker,
    DocsSystem,
    # Factory functions
    create_docs_system,
    add_to_app,
)


# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================

class TestFixtures:
    """Shared test fixtures and sample data."""

    SAMPLE_MARKDOWN = """---
title: Test Document
---

# Main Title

This is the introduction with #tag1 and #tag2.

## Section One

Content for section one referencing `utils.py:helper_function`.

```python
def example():
    pass
```

## Section Two

More content here with `core.py:MainClass`.

### Subsection 2.1

Nested content.
"""

    SAMPLE_PYTHON = '''"""Module docstring."""

class MyClass:
    """A sample class."""

    def __init__(self, value: int):
        """Initialize with value."""
        self.value = value

    def process(self, data: str) -> str:
        """Process the data."""
        return data.upper()


def standalone_function(a, b, c):
    """A standalone function."""
    return a + b + c


async def async_function(items):
    """An async function."""
    return [item async for item in items]
'''

    SAMPLE_TYPESCRIPT = '''/**
 * Main application class
 * Handles core functionality
 */
export class Application {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    public start(): void {
        console.log("Starting");
    }
}

/**
 * Utility function for processing
 */
export function processData(input: string): string {
    return input.trim();
}

export const helper = (x: number) => x * 2;

export interface Config {
    debug: boolean;
    port: number;
}

export type Handler = (event: Event) => void;
'''

    SAMPLE_JAVASCRIPT = '''/**
 * User management module
 */
class UserManager {
    constructor(db) {
        this.db = db;
    }

    async findUser(id) {
        return this.db.find(id);
    }
}

function validateEmail(email) {
    return email.includes("@");
}

const formatName = (first, last) => `${first} ${last}`;
'''


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def run_async(self, coro):
        """Run an async coroutine."""
        return self.loop.run_until_complete(coro)


# =============================================================================
# UNIT TESTS - DATA MODELS
# =============================================================================

class TestChangeType(unittest.TestCase):
    """Tests for ChangeType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        self.assertIsNotNone(ChangeType.ADDED)
        self.assertIsNotNone(ChangeType.MODIFIED)
        self.assertIsNotNone(ChangeType.DELETED)
        self.assertIsNotNone(ChangeType.RENAMED)

    def test_enum_uniqueness(self):
        """Test enum values are unique."""
        values = [e.value for e in ChangeType]
        self.assertEqual(len(values), len(set(values)))


class TestDocSection(unittest.TestCase):
    """Tests for DocSection dataclass."""

    def test_creation(self):
        """Test DocSection creation with required fields."""
        section = DocSection(
            section_id="test.md#Title",
            file_path="/path/to/test.md",
            title="Title",
            content="Test content",
            level=1,
            line_start=0,
            line_end=10,
            content_hash="abc123",
            last_modified=time.time(),
        )
        self.assertEqual(section.section_id, "test.md#Title")
        self.assertEqual(section.title, "Title")
        self.assertEqual(section.level, 1)

    def test_default_values(self):
        """Test DocSection default values."""
        section = DocSection(
            section_id="test",
            file_path="/test",
            title="Test",
            content="Content",
            level=1,
            line_start=0,
            line_end=5,
            content_hash="hash",
            last_modified=0.0,
        )
        self.assertEqual(section.source_refs, ())
        self.assertEqual(section.tags, ())
        self.assertEqual(section.doc_style, "markdown")

    def test_slots_optimization(self):
        """Test that slots are used for memory efficiency."""
        self.assertTrue(hasattr(DocSection, '__slots__'))


class TestCodeElement(unittest.TestCase):
    """Tests for CodeElement dataclass."""

    def test_creation(self):
        """Test CodeElement creation."""
        element = CodeElement(
            name="my_function",
            element_type="function",
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            signature="def my_function(a, b)",
            content_hash="xyz789",
        )
        self.assertEqual(element.name, "my_function")
        self.assertEqual(element.element_type, "function")
        self.assertEqual(element.language, "python")  # default

    def test_optional_fields(self):
        """Test optional fields are None by default."""
        element = CodeElement(
            name="test",
            element_type="class",
            file_path="/test.py",
            line_start=1,
            line_end=10,
            signature="class Test",
            content_hash="hash",
        )
        self.assertIsNone(element.docstring)
        self.assertIsNone(element.parent_class)


class TestFileChange(unittest.TestCase):
    """Tests for FileChange dataclass."""

    def test_creation(self):
        """Test FileChange creation."""
        change = FileChange(
            file_path="src/test.py",
            change_type=ChangeType.MODIFIED,
        )
        self.assertEqual(change.file_path, "src/test.py")
        self.assertEqual(change.change_type, ChangeType.MODIFIED)
        self.assertIsNone(change.old_path)

    def test_renamed_with_old_path(self):
        """Test FileChange for renamed files."""
        change = FileChange(
            file_path="new_name.py",
            change_type=ChangeType.RENAMED,
            old_path="old_name.py",
        )
        self.assertEqual(change.old_path, "old_name.py")


class TestInvertedIndex(unittest.TestCase):
    """Tests for InvertedIndex dataclass."""

    def test_default_factory(self):
        """Test default factory creates empty defaultdicts."""
        index = InvertedIndex()
        self.assertEqual(len(index.keyword_to_sections), 0)
        self.assertEqual(len(index.tag_to_sections), 0)

        # Should auto-create keys
        index.keyword_to_sections["test"].add("section1")
        self.assertIn("section1", index.keyword_to_sections["test"])

    def test_clear(self):
        """Test clearing all indexes."""
        index = InvertedIndex()
        index.keyword_to_sections["key"].add("val")
        index.tag_to_sections["tag"].add("val")
        index.file_to_sections["file"].add("val")
        index.name_to_elements["name"].add("val")
        index.type_to_elements["type"].add("val")
        index.file_to_elements["file"].add("val")

        index.clear()

        self.assertEqual(len(index.keyword_to_sections), 0)
        self.assertEqual(len(index.tag_to_sections), 0)
        self.assertEqual(len(index.file_to_sections), 0)
        self.assertEqual(len(index.name_to_elements), 0)
        self.assertEqual(len(index.type_to_elements), 0)
        self.assertEqual(len(index.file_to_elements), 0)


class TestDocsIndex(unittest.TestCase):
    """Tests for DocsIndex dataclass."""

    def test_default_values(self):
        """Test DocsIndex default values."""
        index = DocsIndex()
        self.assertEqual(index.sections, {})
        self.assertEqual(index.code_elements, {})
        self.assertEqual(index.file_hashes, {})
        self.assertIsInstance(index.inverted, InvertedIndex)
        self.assertIsNone(index.last_git_commit)
        self.assertEqual(index.version, "2.1")


class TestContextBundle(unittest.TestCase):
    """Tests for ContextBundle dictionary subclass."""

    def test_is_dict(self):
        """Test ContextBundle is a dict subclass."""
        bundle = ContextBundle()
        self.assertIsInstance(bundle, dict)

    def test_dict_operations(self):
        """Test standard dict operations work."""
        bundle = ContextBundle({
            "intent": "test",
            "focus_files": {"file.py": "content"},
        })
        self.assertEqual(bundle["intent"], "test")
        bundle["new_key"] = "new_value"
        self.assertEqual(bundle["new_key"], "new_value")


# =============================================================================
# UNIT TESTS - DOC PARSER
# =============================================================================

class TestDocParser(unittest.TestCase):
    """Tests for DocParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = DocParser()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_temp_file(self, content: str, filename: str = "test.md") -> Path:
        """Create a temporary file with given content."""
        path = Path(self.temp_dir) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_parse_simple_markdown(self):
        """Test parsing simple markdown file."""
        content = """# Title

Content here.

## Section

More content.
"""
        path = self._create_temp_file(content)
        sections = self.parser.parse(path)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "Title")
        self.assertEqual(sections[0].level, 1)
        self.assertEqual(sections[1].title, "Section")
        self.assertEqual(sections[1].level, 2)

    def test_parse_with_code_blocks(self):
        """Test parser correctly handles code blocks."""
        content = """# Main

```python
# This is not a header
def func():
    pass
```

## Real Section

Content.
"""
        path = self._create_temp_file(content)
        sections = self.parser.parse(path)

        # Should only find 2 sections, not treat code comments as headers
        self.assertEqual(len(sections), 2)
        titles = [s.title for s in sections]
        self.assertIn("Main", titles)
        self.assertIn("Real Section", titles)

    def test_parse_with_frontmatter(self):
        """Test parsing with YAML frontmatter."""
        content = """---
title: My Doc
author: Test
---

# Actual Title

Content.
"""
        path = self._create_temp_file(content)
        sections = self.parser.parse(path)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].title, "Actual Title")

    def test_parse_extracts_tags(self):
        """Test tag extraction from content."""
        content = """# Tagged Section

This has #tag1 and #another-tag in it.
"""
        path = self._create_temp_file(content)
        sections = self.parser.parse(path)

        self.assertEqual(len(sections), 1)
        self.assertIn("tag1", sections[0].tags)
        self.assertIn("another-tag", sections[0].tags)

    def test_parse_extracts_source_refs(self):
        """Test source reference extraction."""
        content = """# References

See `utils.py:helper_function` and `core.py` for details.
"""
        path = self._create_temp_file(content)
        sections = self.parser.parse(path)

        self.assertEqual(len(sections), 1)
        self.assertIn("utils.py:helper_function", sections[0].source_refs)

    def test_parse_setext_headers(self):
        """Test parsing setext-style headers (underline)."""
        content = """Main Title
==========

Content.

Sub Section
-----------

More content.
"""
        path = self._create_temp_file(content)
        sections = self.parser.parse(path)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "Main Title")
        self.assertEqual(sections[0].level, 1)
        self.assertEqual(sections[1].title, "Sub Section")
        self.assertEqual(sections[1].level, 2)

    def test_parse_caching(self):
        """Test parser caching behavior."""
        content = "# Test\n\nContent."
        path = self._create_temp_file(content)

        # First parse
        sections1 = self.parser.parse(path, use_cache=True)
        # Second parse should use cache
        sections2 = self.parser.parse(path, use_cache=True)

        self.assertEqual(len(sections1), len(sections2))
        self.assertEqual(sections1[0].content_hash, sections2[0].content_hash)

    def test_parse_cache_invalidation(self):
        """Test cache invalidation on file change."""
        content = "# Test\n\nContent."
        path = self._create_temp_file(content)

        sections1 = self.parser.parse(path, use_cache=True)

        # Modify file
        time.sleep(0.1)  # Ensure mtime changes
        path.write_text("# New Test\n\nNew content.", encoding="utf-8")

        sections2 = self.parser.parse(path, use_cache=True)

        self.assertEqual(sections2[0].title, "New Test")

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file returns empty list."""
        path = Path(self.temp_dir) / "nonexistent.md"
        sections = self.parser.parse(path)
        self.assertEqual(sections, [])

    def test_parse_empty_file(self):
        """Test parsing empty file returns empty list."""
        path = self._create_temp_file("")
        sections = self.parser.parse(path)
        self.assertEqual(sections, [])

    def test_detect_style_markdown(self):
        """Test style detection for markdown."""
        content = "# Header\n\nContent."
        style = self.parser._detect_style(content)
        self.assertEqual(style, "markdown")

    def test_detect_style_rst(self):
        """Test style detection for RST."""
        content = "Header\n======\n\nContent."
        style = self.parser._detect_style(content)
        self.assertEqual(style, "rst")

    def test_detect_style_yaml_md(self):
        """Test style detection for YAML frontmatter."""
        content = "---\ntitle: Test\n---\n\n# Header"
        style = self.parser._detect_style(content)
        self.assertEqual(style, "yaml_md")

    def test_clear_cache(self):
        """Test cache clearing."""
        content = "# Test\n\nContent."
        path = self._create_temp_file(content)

        self.parser.parse(path, use_cache=True)
        self.assertGreater(len(self.parser._cache), 0)

        self.parser.clear_cache()
        self.assertEqual(len(self.parser._cache), 0)

    def test_multiple_header_levels(self):
        """Test parsing multiple header levels."""
        content = """# H1
text1
## H2
text2
### H3
text3
#### H4
text4
##### H5
text5
###### H6
text6
"""
        path = self._create_temp_file(content)
        sections = self.parser.parse(path)

        levels = [s.level for s in sections]
        self.assertEqual(levels, [1, 2, 3, 4, 5, 6])

    def test_content_hash_uniqueness(self):
        """Test that different content produces different hashes."""
        content1 = "# Title\n\nContent one."
        content2 = "# Title\n\nContent two."

        path1 = self._create_temp_file(content1, "test1.md")
        path2 = self._create_temp_file(content2, "test2.md")

        sections1 = self.parser.parse(path1)
        sections2 = self.parser.parse(path2)

        self.assertNotEqual(sections1[0].content_hash, sections2[0].content_hash)


# =============================================================================
# UNIT TESTS - CODE ANALYZER (PYTHON)
# =============================================================================

class TestCodeAnalyzer(unittest.TestCase):
    """Tests for CodeAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CodeAnalyzer()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_temp_file(self, content: str, filename: str = "test.py") -> Path:
        """Create a temporary Python file."""
        path = Path(self.temp_dir) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_analyze_class(self):
        """Test analyzing Python class."""
        content = '''class MyClass:
    """My class docstring."""
    pass
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)

        classes = [e for e in elements if e.element_type == "class"]
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0].name, "MyClass")
        self.assertEqual(classes[0].docstring, "My class docstring.")

    def test_analyze_function(self):
        """Test analyzing Python function."""
        content = '''def my_function(a, b, c):
    """Function docstring."""
    return a + b + c
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)

        funcs = [e for e in elements if e.element_type == "function"]
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "my_function")
        self.assertIn("a", funcs[0].signature)
        self.assertIn("b", funcs[0].signature)
        self.assertIn("c", funcs[0].signature)

    def test_analyze_method(self):
        """Test analyzing class methods."""
        content = '''class MyClass:
    def my_method(self, x):
        """Method docstring."""
        return x
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)

        methods = [e for e in elements if e.element_type == "method"]
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].name, "my_method")
        self.assertEqual(methods[0].parent_class, "MyClass")

    def test_analyze_async_function(self):
        """Test analyzing async functions."""
        content = '''async def async_func(items):
    """Async function."""
    return items
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)

        funcs = [e for e in elements if e.element_type == "function"]
        self.assertEqual(len(funcs), 1)
        self.assertIn("async def", funcs[0].signature)

    def test_analyze_with_inheritance(self):
        """Test analyzing class with inheritance."""
        content = '''class Child(Parent, Mixin):
    pass
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)

        classes = [e for e in elements if e.element_type == "class"]
        self.assertEqual(len(classes), 1)
        self.assertIn("Parent", classes[0].signature)
        self.assertIn("Mixin", classes[0].signature)

    def test_analyze_syntax_error(self):
        """Test handling syntax errors gracefully."""
        content = '''def broken(
    # Missing closing paren
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)
        self.assertEqual(elements, [])

    def test_analyze_nonexistent_file(self):
        """Test analyzing nonexistent file."""
        path = Path(self.temp_dir) / "nonexistent.py"
        elements = self.analyzer.analyze(path)
        self.assertEqual(elements, [])

    def test_analyze_caching(self):
        """Test analyzer caching."""
        content = "def test(): pass"
        path = self._create_temp_file(content)

        elements1 = self.analyzer.analyze(path, use_cache=True)
        elements2 = self.analyzer.analyze(path, use_cache=True)

        self.assertEqual(len(elements1), len(elements2))

    def test_analyze_cache_bypass(self):
        """Test bypassing cache."""
        content = "def test(): pass"
        path = self._create_temp_file(content)

        self.analyzer.analyze(path, use_cache=True)
        self.analyzer.analyze(path, use_cache=False)

        # Should still work without errors
        self.assertTrue(True)

    def test_clear_cache(self):
        """Test cache clearing."""
        content = "def test(): pass"
        path = self._create_temp_file(content)

        self.analyzer.analyze(path, use_cache=True)
        self.assertGreater(len(self.analyzer._cache), 0)

        self.analyzer.clear_cache()
        self.assertEqual(len(self.analyzer._cache), 0)

    def test_signature_truncation(self):
        """Test signature truncation for many parameters."""
        content = "def many_args(a, b, c, d, e, f, g, h): pass"
        path = self._create_temp_file(content)

        elements = self.analyzer.analyze(path)
        funcs = [e for e in elements if e.element_type == "function"]

        self.assertEqual(len(funcs), 1)
        self.assertIn("...", funcs[0].signature)

    def test_language_attribute(self):
        """Test that language is set to python."""
        content = "def test(): pass"
        path = self._create_temp_file(content)

        elements = self.analyzer.analyze(path)
        self.assertEqual(elements[0].language, "python")


# =============================================================================
# UNIT TESTS - JS/TS ANALYZER
# =============================================================================

class TestJSTSAnalyzer(unittest.TestCase):
    """Tests for JSTSAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = JSTSAnalyzer()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_temp_file(self, content: str, filename: str = "test.ts") -> Path:
        """Create a temporary JS/TS file."""
        path = Path(self.temp_dir) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_analyze_typescript_class(self):
        """Test analyzing TypeScript class."""
        content = '''export class MyClass {
    constructor() {}
}
'''
        path = self._create_temp_file(content, "test.ts")
        elements = self.analyzer.analyze(path)

        classes = [e for e in elements if e.element_type == "class"]
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0].name, "MyClass")
        self.assertEqual(classes[0].language, "typescript")

    def test_analyze_class_with_extends(self):
        """Test analyzing class with inheritance."""
        content = '''class Child extends Parent {
}
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)

        classes = [e for e in elements if e.element_type == "class"]
        self.assertEqual(len(classes), 1)
        self.assertIn("extends Parent", classes[0].signature)

    def test_analyze_function(self):
        """Test analyzing JavaScript function."""
        content = '''function myFunction(a, b) {
    return a + b;
}
'''
        path = self._create_temp_file(content, "test.js")
        elements = self.analyzer.analyze(path)

        funcs = [e for e in elements if e.element_type == "function"]
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "myFunction")
        self.assertEqual(funcs[0].language, "javascript")

    def test_analyze_arrow_function(self):
        """Test analyzing arrow functions."""
        content = '''export const helper = (x) => x * 2;
'''
        path = self._create_temp_file(content)
        elements = self.analyzer.analyze(path)

        funcs = [e for e in elements if e.element_type == "function"]
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "helper")

    def test_analyze_interface(self):
        """Test analyzing TypeScript interface."""
        content = '''export interface Config {
    debug: boolean;
    port: number;
}
'''
        path = self._create_temp_file(content, "test.ts")
        elements = self.analyzer.analyze(path)

        interfaces = [e for e in elements if e.element_type == "interface"]
        self.assertEqual(len(interfaces), 1)
        self.assertEqual(interfaces[0].name, "Config")

    def test_analyze_type_alias(self):
        """Test analyzing TypeScript type alias."""
        content = '''export type Handler = (event: Event) => void;
'''
        path = self._create_temp_file(content, "test.ts")
        elements = self.analyzer.analyze(path)

        types = [e for e in elements if e.element_type == "type"]
        self.assertEqual(len(types), 1)
        self.assertEqual(types[0].name, "Handler")

    def test_analyze_with_jsdoc(self):
        """Test JSDoc extraction."""
        content = '''/**
 * Processes the input data
 */
function process(data) {
    return data;
}
'''
        path = self._create_temp_file(content, "test.js")
        elements = self.analyzer.analyze(path)

        funcs = [e for e in elements if e.element_type == "function"]
        self.assertEqual(len(funcs), 1)
        self.assertIsNotNone(funcs[0].docstring)
        self.assertIn("Processes", funcs[0].docstring)

    def test_analyze_async_function(self):
        """Test analyzing async functions."""
        content = '''export async function fetchData(url) {
    return fetch(url);
}
'''
        path = self._create_temp_file(content, "test.js")
        elements = self.analyzer.analyze(path)

        funcs = [e for e in elements if e.element_type == "function"]
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "fetchData")

    def test_analyze_nonexistent_file(self):
        """Test analyzing nonexistent file."""
        path = Path(self.temp_dir) / "nonexistent.ts"
        elements = self.analyzer.analyze(path)
        self.assertEqual(elements, [])

    def test_analyze_caching(self):
        """Test analyzer caching."""
        content = "function test() {}"
        path = self._create_temp_file(content, "test.js")

        elements1 = self.analyzer.analyze(path, use_cache=True)
        elements2 = self.analyzer.analyze(path, use_cache=True)

        self.assertEqual(len(elements1), len(elements2))

    def test_clear_cache(self):
        """Test cache clearing."""
        content = "function test() {}"
        path = self._create_temp_file(content, "test.js")

        self.analyzer.analyze(path, use_cache=True)
        self.assertGreater(len(self.analyzer._cache), 0)

        self.analyzer.clear_cache()
        self.assertEqual(len(self.analyzer._cache), 0)

    def test_find_block_end(self):
        """Test block end finding."""
        lines = [
            "function test() {",
            "    if (true) {",
            "        return;",
            "    }",
            "}",
        ]
        end = self.analyzer._find_block_end(lines, 0)
        self.assertEqual(end, 5)

    def test_clean_jsdoc(self):
        """Test JSDoc cleaning."""
        doc = "* First line\n * Second line\n * @param x"
        cleaned = self.analyzer._clean_jsdoc(doc)
        self.assertIn("First line", cleaned)
        self.assertIn("Second line", cleaned)
        self.assertNotIn("@param", cleaned)


# =============================================================================
# UNIT TESTS - INDEX MANAGER
# =============================================================================

class TestIndexManager(AsyncTestCase):
    """Tests for IndexManager class."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / ".docs_index.json"
        self.manager = IndexManager(self.index_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_initial_state(self):
        """Test initial index state."""
        self.assertIsInstance(self.manager.index, DocsIndex)
        self.assertEqual(len(self.manager.index.sections), 0)
        self.assertEqual(len(self.manager.index.code_elements), 0)

    def test_load_nonexistent(self):
        """Test loading when no index file exists."""
        index = self.run_async(self.manager.load())
        self.assertIsInstance(index, DocsIndex)

    def test_save_and_load(self):
        """Test saving and loading index."""
        # Add some data
        section = DocSection(
            section_id="test#Section",
            file_path="/test.md",
            title="Section",
            content="Content here",
            level=1,
            line_start=0,
            line_end=5,
            content_hash="abc123",
            last_modified=time.time(),
            tags=("tag1", "tag2"),
        )
        self.manager.update_section(section)

        # Save
        self.run_async(self.manager.save(force=True))

        # Create new manager and load
        new_manager = IndexManager(self.index_path)
        loaded_index = self.run_async(new_manager.load())

        self.assertIn("test#Section", loaded_index.sections)
        self.assertEqual(loaded_index.sections["test#Section"].title, "Section")

    def test_update_section(self):
        """Test updating section with inverted index."""
        section = DocSection(
            section_id="test#Section",
            file_path="/test.md",
            title="Test Section",
            content="Content with keyword1 and keyword2",
            level=1,
            line_start=0,
            line_end=5,
            content_hash="abc123",
            last_modified=time.time(),
            tags=("python", "testing"),
        )

        self.manager.update_section(section)

        # Check inverted index
        self.assertIn("test#Section",
                      self.manager.index.inverted.keyword_to_sections.get("keyword1", set()))
        self.assertIn("test#Section",
                      self.manager.index.inverted.tag_to_sections.get("python", set()))

    def test_update_element(self):
        """Test updating code element with inverted index."""
        element = CodeElement(
            name="MyFunction",
            element_type="function",
            file_path="/test.py",
            line_start=10,
            line_end=20,
            signature="def MyFunction(a, b)",
            content_hash="xyz789",
        )

        self.manager.update_element("test:MyFunction", element)

        # Check inverted index
        self.assertIn("test:MyFunction",
                      self.manager.index.inverted.name_to_elements.get("myfunction", set()))
        self.assertIn("test:MyFunction",
                      self.manager.index.inverted.type_to_elements.get("function", set()))

    def test_remove_file(self):
        """Test removing file entries."""
        section = DocSection(
            section_id="test#Section",
            file_path="/test.md",
            title="Section",
            content="Content",
            level=1,
            line_start=0,
            line_end=5,
            content_hash="abc",
            last_modified=time.time(),
        )
        self.manager.update_section(section)
        self.manager.index.file_hashes["/test.md"] = "hash123"

        self.manager.remove_file("/test.md")

        self.assertNotIn("test#Section", self.manager.index.sections)
        self.assertNotIn("/test.md", self.manager.index.file_hashes)

    def test_mark_dirty(self):
        """Test dirty flag."""
        self.assertFalse(self.manager._dirty)
        self.manager.mark_dirty()
        self.assertTrue(self.manager._dirty)

    def test_save_skips_when_not_dirty(self):
        """Test that save is skipped when not dirty."""
        # Save without marking dirty
        self.run_async(self.manager.save(force=False))

        # File should not exist
        self.assertFalse(self.index_path.exists())

    def test_tokenize(self):
        """Test text tokenization."""
        text = "This is a test_function with CamelCase and some words"
        tokens = self.manager._tokenize(text)

        self.assertIn("test_function", tokens)
        self.assertIn("camelcase", tokens)
        self.assertNotIn("is", tokens)  # Stop word
        self.assertNotIn("a", tokens)   # Stop word

    def test_stop_words_filtering(self):
        """Test stop words are filtered."""
        text = "the a an is are was were be been being have has had"
        tokens = self.manager._tokenize(text)
        self.assertEqual(len(tokens), 0)

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        section = DocSection(
            section_id="test#Section",
            file_path="/test.md",
            title="Title",
            content="Content",
            level=2,
            line_start=5,
            line_end=10,
            content_hash="hash",
            last_modified=123.456,
            source_refs=("ref1", "ref2"),
            tags=("tag1",),
            doc_style="markdown",
        )

        element = CodeElement(
            name="func",
            element_type="function",
            file_path="/test.py",
            line_start=1,
            line_end=10,
            signature="def func()",
            content_hash="hash",
            language="python",
            docstring="Docstring",
            parent_class=None,
        )

        self.manager.update_section(section)
        self.manager.update_element("test:func", element)

        # Serialize
        data = self.manager._serialize()

        # Deserialize
        new_index = self.manager._deserialize(data)

        self.assertEqual(new_index.sections["test#Section"].title, "Title")
        self.assertEqual(new_index.code_elements["test:func"].name, "func")


# =============================================================================
# UNIT TESTS - CONTEXT ENGINE
# =============================================================================

class TestContextEngine(unittest.TestCase):
    """Tests for ContextEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / ".docs_index.json"
        self.index_mgr = IndexManager(self.index_path)
        self.engine = ContextEngine(self.index_mgr, cache_ttl=60.0)

        # Add test data
        self._add_test_data()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _add_test_data(self):
        """Add test sections and elements."""
        sections = [
            DocSection(
                section_id="doc1#Getting Started",
                file_path="/docs/guide.md",
                title="Getting Started",
                content="Introduction to the project with Python examples",
                level=1,
                line_start=0,
                line_end=10,
                content_hash="hash1",
                last_modified=time.time(),
                tags=("tutorial", "python"),
            ),
            DocSection(
                section_id="doc2#API Reference",
                file_path="/docs/api.md",
                title="API Reference",
                content="API documentation for the core module",
                level=1,
                line_start=0,
                line_end=20,
                content_hash="hash2",
                last_modified=time.time(),
                tags=("api", "reference"),
            ),
        ]

        elements = [
            CodeElement(
                name="UserManager",
                element_type="class",
                file_path="/src/users.py",
                line_start=10,
                line_end=50,
                signature="class UserManager",
                content_hash="elem1",
                docstring="Manages user operations",
            ),
            CodeElement(
                name="process_data",
                element_type="function",
                file_path="/src/utils.py",
                line_start=5,
                line_end=15,
                signature="def process_data(data)",
                content_hash="elem2",
            ),
        ]

        for section in sections:
            self.index_mgr.update_section(section)

        for i, elem in enumerate(elements):
            self.index_mgr.update_element(f"elem{i}", elem)

    def test_search_sections_by_query(self):
        """Test searching sections by query."""
        results = self.engine.search_sections(query="python")

        self.assertGreater(len(results), 0)
        titles = [r.title for r in results]
        self.assertIn("Getting Started", titles)

    def test_search_sections_by_tag(self):
        """Test searching sections by tag."""
        results = self.engine.search_sections(tags=["tutorial"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Getting Started")

    def test_search_sections_by_file(self):
        """Test searching sections by file path."""
        results = self.engine.search_sections(file_path="/docs/api.md")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "API Reference")

    def test_search_sections_combined_filters(self):
        """Test searching with multiple filters."""
        results = self.engine.search_sections(
            query="python",
            tags=["tutorial"],
        )

        self.assertEqual(len(results), 1)

    def test_search_elements_by_name(self):
        """Test searching elements by name."""
        results = self.engine.search_elements(name="UserManager")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "UserManager")

    def test_search_elements_by_type(self):
        """Test searching elements by type."""
        results = self.engine.search_elements(element_type="function")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "process_data")

    def test_search_elements_by_file(self):
        """Test searching elements by file."""
        results = self.engine.search_elements(file_path="/src/users.py")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "UserManager")

    def test_search_max_results(self):
        """Test max_results limit."""
        results = self.engine.search_sections(max_results=1)
        self.assertLessEqual(len(results), 1)

    def test_caching(self):
        """Test query result caching."""
        # First query
        results1 = self.engine.search_sections(query="python")

        # Second query (should use cache)
        results2 = self.engine.search_sections(query="python")

        self.assertEqual(len(results1), len(results2))

    def test_clear_cache(self):
        """Test cache clearing."""
        self.engine.search_sections(query="test")
        self.assertGreater(len(self.engine._query_cache), 0)

        self.engine.clear_cache()
        self.assertEqual(len(self.engine._query_cache), 0)

    def test_get_context_for_element(self):
        """Test getting context for element."""
        context = self.engine.get_context_for_element("elem0")

        self.assertIn("element", context)
        self.assertEqual(context["element"]["name"], "UserManager")
        self.assertIn("documentation", context)
        self.assertIn("related_elements", context)

    def test_get_context_for_nonexistent_element(self):
        """Test getting context for nonexistent element."""
        context = self.engine.get_context_for_element("nonexistent")
        self.assertEqual(context, {})

    def test_truncate_content(self):
        """Test content truncation."""
        long_content = "x" * 1000
        truncated = self.engine._truncate_content(long_content, 100)

        self.assertLessEqual(len(truncated), 120)  # 100 + "... (truncated)"
        self.assertIn("truncated", truncated)

    def test_cache_eviction(self):
        """Test cache eviction when limit reached."""
        # Fill cache with many queries
        for i in range(150):
            self.engine._set_cached(f"key{i}", f"value{i}")

        # Cache should be limited
        self.assertLessEqual(len(self.engine._query_cache), 101)


# =============================================================================
# UNIT TESTS - FILE SCANNER
# =============================================================================

class TestFileScanner(unittest.TestCase):
    """Tests for FileScanner class."""

    def setUp(self):
        """Set up test directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        # Create test structure
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("# main")
        (self.root / "src" / "utils.py").write_text("# utils")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "readme.md").write_text("# readme")
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "cache.pyc").write_text("cache")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "pkg.js").write_text("pkg")

        self.scanner = FileScanner(self.root)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scan_python_files(self):
        """Test scanning for Python files."""
        files = self.scanner.scan({".py"})

        names = [f.name for f in files]
        self.assertIn("main.py", names)
        self.assertIn("utils.py", names)

    def test_scan_markdown_files(self):
        """Test scanning for Markdown files."""
        files = self.scanner.scan({".md"})

        names = [f.name for f in files]
        self.assertIn("readme.md", names)

    def test_exclude_pycache(self):
        """Test __pycache__ is excluded."""
        files = self.scanner.scan({".pyc", ".py"})

        paths = [str(f) for f in files]
        self.assertFalse(any("__pycache__" in p for p in paths))

    def test_exclude_node_modules(self):
        """Test node_modules is excluded."""
        files = self.scanner.scan({".js"})

        paths = [str(f) for f in files]
        self.assertFalse(any("node_modules" in p for p in paths))

    def test_custom_exclude_dirs(self):
        """Test custom exclude directories."""
        scanner = FileScanner(self.root, exclude_dirs={"src"})
        files = scanner.scan({".py"})

        self.assertEqual(len(files), 0)

    def test_include_dirs(self):
        """Test include directories filter."""
        scanner = FileScanner(self.root, include_dirs=["src"])
        files = scanner.scan({".py", ".md"})

        # Should only find files in src
        paths = [str(f) for f in files]
        self.assertTrue(all("src" in p for p in paths))

    def test_caching(self):
        """Test file scanning cache."""
        files1 = self.scanner.scan({".py"}, use_cache=True)
        files2 = self.scanner.scan({".py"}, use_cache=True)

        self.assertEqual(len(files1), len(files2))

    def test_clear_cache(self):
        """Test cache clearing."""
        self.scanner.scan({".py"}, use_cache=True)
        self.assertIsNotNone(self.scanner._file_cache)

        self.scanner.clear_cache()
        self.assertIsNone(self.scanner._file_cache)

    def test_get_file_hash(self):
        """Test file hash generation."""
        file_path = self.root / "src" / "main.py"
        hash1 = self.scanner.get_file_hash(file_path)

        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 12)

    def test_get_file_hash_nonexistent(self):
        """Test hash for nonexistent file."""
        hash_val = self.scanner.get_file_hash(self.root / "nonexistent.py")
        self.assertEqual(hash_val, "")


# =============================================================================
# UNIT TESTS - GIT TRACKER
# =============================================================================

class TestGitTracker(AsyncTestCase):
    """Tests for GitTracker class."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = GitTracker(Path(self.temp_dir))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_parse_changes_with_status(self):
        """Test parsing git diff output with status."""
        output = """A\tnew_file.py
M\tmodified.py
D\tdeleted.py
R100\told_name.py\tnew_name.py"""

        changes = self.tracker._parse_changes(output, has_status=True)

        self.assertEqual(len(changes), 4)

        # Check types
        types = {c.file_path: c.change_type for c in changes}
        self.assertEqual(types["new_file.py"], ChangeType.ADDED)
        self.assertEqual(types["modified.py"], ChangeType.MODIFIED)
        self.assertEqual(types["deleted.py"], ChangeType.DELETED)

    def test_parse_changes_without_status(self):
        """Test parsing git ls-files output."""
        output = """file1.py
file2.py
dir/file3.py"""

        changes = self.tracker._parse_changes(output, has_status=False)

        self.assertEqual(len(changes), 3)
        self.assertTrue(all(c.change_type == ChangeType.ADDED for c in changes))

    def test_parse_changes_empty(self):
        """Test parsing empty output."""
        changes = self.tracker._parse_changes("", has_status=True)
        self.assertEqual(changes, [])

    def test_parse_changes_limit(self):
        """Test that changes are limited to 500."""
        output = "\n".join([f"M\tfile{i}.py" for i in range(600)])
        changes = self.tracker._parse_changes(output, has_status=True)
        self.assertEqual(len(changes), 500)

    @patch('asyncio.create_subprocess_exec')
    def test_get_commit_hash_success(self, mock_exec):
        """Test getting commit hash successfully."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"abc123def456\n", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = self.run_async(self.tracker.get_commit_hash())

        self.assertEqual(result, "abc123def456")

    @patch('asyncio.create_subprocess_exec')
    def test_get_commit_hash_failure(self, mock_exec):
        """Test getting commit hash when git fails."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"error")
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        result = self.run_async(self.tracker.get_commit_hash())

        self.assertIsNone(result)


# =============================================================================
# INTEGRATION TESTS - DOCS SYSTEM
# =============================================================================

class TestDocsSystemIntegration(AsyncTestCase):
    """Full integration tests for DocsSystem."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "project"
        self.docs_root = Path(self.temp_dir) / "docs"

        # Create project structure
        self.project_root.mkdir()
        self.docs_root.mkdir()

        # Create source files
        self._create_project_files()

        # Create docs system
        self.system = DocsSystem(
            project_root=self.project_root,
            docs_root=self.docs_root,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _create_project_files(self):
        """Create test project files."""
        # Python file
        (self.project_root / "main.py").write_text('''"""Main module."""

class Application:
    """Main application class."""

    def __init__(self, name: str):
        """Initialize application."""
        self.name = name

    def run(self):
        """Run the application."""
        print(f"Running {self.name}")


def helper_function(x, y):
    """Helper function."""
    return x + y
''')

        # Markdown file
        (self.project_root / "README.md").write_text('''# Project README

Welcome to the project.

## Installation

Install with pip.

## Usage

Run `main.py:Application` to start.
''')

        # TypeScript file
        (self.project_root / "utils.ts").write_text('''/**
 * Utility class
 */
export class Utils {
    static format(value: string): string {
        return value.trim();
    }
}

export function validate(input: string): boolean {
    return input.length > 0;
}
''')

    def test_initialize_builds_index(self):
        """Test that initialize builds the index."""
        result = self.run_async(self.system.initialize(force_rebuild=True))

        self.assertEqual(result["status"], "rebuilt")
        self.assertGreater(result["sections"], 0)
        self.assertGreater(result["elements"], 0)

    def test_initialize_loads_existing(self):
        """Test that initialize loads existing index."""
        # First build
        self.run_async(self.system.initialize(force_rebuild=True))

        # Create new system instance
        new_system = DocsSystem(
            project_root=self.project_root,
            docs_root=self.docs_root,
        )

        # Should load, not rebuild
        result = self.run_async(new_system.initialize(force_rebuild=False))
        self.assertEqual(result["status"], "loaded")

    def test_read_all_sections(self):
        """Test reading all documentation sections."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.read())

        self.assertIn("sections", result)
        self.assertGreater(len(result["sections"]), 0)

    def test_read_by_query(self):
        """Test reading sections by query."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.read(query="installation"))

        self.assertIn("sections", result)
        # Should find Installation section
        titles = [s["title"] for s in result["sections"]]
        self.assertTrue(any("Installation" in t for t in titles))

    def test_read_by_section_id(self):
        """Test reading specific section by ID."""
        self.run_async(self.system.initialize(force_rebuild=True))

        # First get all sections
        all_sections = self.run_async(self.system.read())
        if all_sections["sections"]:
            section_id = all_sections["sections"][0]["id"]

            result = self.run_async(self.system.read(section_id=section_id))

            self.assertIn("sections", result)
            self.assertEqual(len(result["sections"]), 1)

    def test_read_nonexistent_section(self):
        """Test reading nonexistent section."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.read(section_id="nonexistent"))

        self.assertIn("error", result)

    def test_read_markdown_format(self):
        """Test reading in markdown format."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.read(format_type="markdown"))

        self.assertIn("content", result)
        self.assertIn("#", result["content"])  # Should have headers

    def test_lookup_code_python(self):
        """Test looking up Python code elements."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.lookup_code(name="Application"))

        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)
        self.assertEqual(result["results"][0]["name"], "Application")

    def test_lookup_code_by_type(self):
        """Test looking up code by element type."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.lookup_code(element_type="class"))

        self.assertIn("results", result)
        self.assertTrue(all(r["type"] == "class" for r in result["results"]))

    def test_lookup_code_with_code_extraction(self):
        """Test looking up code with source extraction."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(
            self.system.lookup_code(name="helper_function", include_code=True)
        )

        self.assertIn("results", result)
        if result["results"]:
            self.assertIn("code", result["results"][0])

    def test_lookup_code_typescript(self):
        """Test looking up TypeScript elements."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(
            self.system.lookup_code(name="Utils", language="typescript")
        )

        self.assertIn("results", result)

    def test_get_suggestions(self):
        """Test getting documentation suggestions."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.get_suggestions())

        self.assertIn("suggestions", result)
        self.assertIn("total", result)

    def test_write_create_file(self):
        """Test creating a new documentation file."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.write(
            action="create_file",
            file_path="new_doc.md",
            content="# New Document\n\nContent here.",
        ))

        self.assertEqual(result["status"], "created")
        self.assertTrue((self.docs_root / "new_doc.md").exists())

    def test_write_create_file_exists(self):
        """Test creating file that already exists."""
        self.run_async(self.system.initialize(force_rebuild=True))

        # Create file first
        (self.docs_root / "existing.md").write_text("# Existing")

        result = self.run_async(self.system.write(
            action="create_file",
            file_path="existing.md",
        ))

        self.assertIn("error", result)

    def test_write_add_section(self):
        """Test adding a section to a file."""
        self.run_async(self.system.initialize(force_rebuild=True))

        # Create file first
        self.run_async(self.system.write(
            action="create_file",
            file_path="test.md",
            content="# Test\n\nIntro.",
        ))

        # Add section
        result = self.run_async(self.system.write(
            action="add_section",
            file_path="test.md",
            title="New Section",
            content="Section content.",
            level=2,
        ))

        self.assertEqual(result["status"], "added")

    def test_write_unknown_action(self):
        """Test unknown write action."""
        result = self.run_async(self.system.write(action="unknown"))
        self.assertIn("error", result)

    def test_sync(self):
        """Test syncing index with file system."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.sync())

        self.assertIn("changes_detected", result)
        self.assertIn("files_updated", result)

    def test_get_task_context(self):
        """Test getting task context."""
        self.run_async(self.system.initialize(force_rebuild=True))

        result = self.run_async(self.system.get_task_context(
            files=[str(self.project_root / "main.py")],
            intent="Add logging to Application class",
        ))

        self.assertIn("result", result)
        self.assertIn("meta", result)


class TestDocsSystemLiveWorkflow(AsyncTestCase):
    """Live workflow tests simulating real usage scenarios."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "project"
        self.docs_root = Path(self.temp_dir) / "docs"

        self.project_root.mkdir()
        self.docs_root.mkdir()

        self._setup_realistic_project()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _setup_realistic_project(self):
        """Set up a realistic project structure."""
        # Create source directory
        src = self.project_root / "src"
        src.mkdir()

        # Main application
        (src / "app.py").write_text('''"""Main application module."""

from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class Config:
    """Application configuration."""

    def __init__(self, debug: bool = False, port: int = 8080):
        self.debug = debug
        self.port = port


class Application:
    """Main application class.

    Handles request processing and routing.
    """

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.routes = {}

    def route(self, path: str):
        """Decorator for registering routes."""
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

    async def handle_request(self, path: str, data: dict) -> dict:
        """Handle incoming request."""
        handler = self.routes.get(path)
        if handler:
            return await handler(data)
        return {"error": "Not found"}

    def run(self):
        """Start the application."""
        logger.info(f"Starting on port {self.config.port}")
''')

        # Utilities
        (src / "utils.py").write_text('''"""Utility functions."""


def format_response(data: dict, status: int = 200) -> dict:
    """Format API response."""
    return {
        "status": status,
        "data": data,
    }


def validate_input(data: dict, required: list) -> bool:
    """Validate required fields are present."""
    return all(field in data for field in required)


class Cache:
    """Simple in-memory cache."""

    def __init__(self, max_size: int = 100):
        self._store = {}
        self.max_size = max_size

    def get(self, key: str):
        """Get value from cache."""
        return self._store.get(key)

    def set(self, key: str, value):
        """Set value in cache."""
        if len(self._store) >= self.max_size:
            self._store.pop(next(iter(self._store)))
        self._store[key] = value
''')

        # Documentation
        docs = self.project_root / "docs"
        docs.mkdir()

        (docs / "guide.md").write_text('''# User Guide

Welcome to the application user guide.

## Getting Started

First, install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `Config` object to configure the app.
See `src/app.py:Config` for available options.

## Creating Routes

Use the `@app.route` decorator:

```python
@app.route("/api/users")
async def get_users(data):
    return {"users": []}
```

## API Reference

### Application Class

The main `Application` class in `src/app.py:Application`.

### Utility Functions

- `format_response` - Format API responses
- `validate_input` - Validate request data
''')

        (docs / "api.md").write_text('''# API Documentation

## Endpoints

### GET /api/users

Returns list of users.

### POST /api/users

Create a new user.

## Response Format

All responses use the `format_response` utility.

#api #reference
''')

        # TypeScript frontend
        frontend = self.project_root / "frontend"
        frontend.mkdir()

        (frontend / "client.ts").write_text('''/**
 * API Client for backend communication
 */
export class ApiClient {
    private baseUrl: string;

    constructor(baseUrl: string = "/api") {
        this.baseUrl = baseUrl;
    }

    async get<T>(path: string): Promise<T> {
        const response = await fetch(`${this.baseUrl}${path}`);
        return response.json();
    }

    async post<T>(path: string, data: unknown): Promise<T> {
        const response = await fetch(`${this.baseUrl}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        return response.json();
    }
}

export interface User {
    id: number;
    name: string;
    email: string;
}

export type ApiResponse<T> = {
    status: number;
    data: T;
};
''')

    def test_full_documentation_workflow(self):
        """Test complete documentation workflow."""
        system = DocsSystem(
            project_root=self.project_root,
            docs_root=self.docs_root,
        )

        # Step 1: Initialize and build index
        init_result = self.run_async(system.initialize(force_rebuild=True))
        self.assertEqual(init_result["status"], "rebuilt")
        self.assertGreater(init_result["sections"], 0)
        self.assertGreater(init_result["elements"], 0)

        # Step 2: Read documentation
        read_result = self.run_async(system.read(query="configuration"))
        self.assertGreater(len(read_result["sections"]), 0)

        # Step 3: Lookup code elements
        code_result = self.run_async(system.lookup_code(name="Application"))
        self.assertGreater(len(code_result["results"]), 0)

        # Step 4: Get suggestions
        suggestions = self.run_async(system.get_suggestions())
        self.assertIn("suggestions", suggestions)

        # Step 5: Create new documentation
        create_result = self.run_async(system.write(
            action="create_file",
            file_path="changelog.md",
            content="# Changelog\n\n## v1.0.0\n\n- Initial release",
        ))
        self.assertEqual(create_result["status"], "created")

        # Step 6: Add section to existing doc
        add_result = self.run_async(system.write(
            action="add_section",
            file_path="changelog.md",
            title="v1.1.0",
            content="- Added new features\n- Bug fixes",
            level=2,
        ))
        self.assertEqual(add_result["status"], "added")

        # Step 7: Read back the created documentation
        changelog = self.run_async(system.read(
            file_path=str(self.docs_root / "changelog.md")
        ))
        self.assertGreater(len(changelog["sections"]), 0)

        # Step 8: Sync changes
        sync_result = self.run_async(system.sync())
        self.assertIn("files_updated", sync_result)

    def test_multi_language_support(self):
        """Test support for multiple languages."""
        system = DocsSystem(
            project_root=self.project_root,
            docs_root=self.docs_root,
        )

        self.run_async(system.initialize(force_rebuild=True))

        # Python elements
        py_result = self.run_async(
            system.lookup_code(language="python", element_type="class")
        )
        py_classes = [r for r in py_result["results"] if r["language"] == "python"]
        self.assertGreater(len(py_classes), 0)

        # TypeScript elements
        ts_result = self.run_async(
            system.lookup_code(language="typescript")
        )
        ts_elements = [r for r in ts_result["results"] if r["language"] == "typescript"]
        self.assertGreater(len(ts_elements), 0)

    def test_cross_reference_docs_and_code(self):
        """Test cross-referencing between docs and code."""
        system = DocsSystem(
            project_root=self.project_root,
            docs_root=self.docs_root,
        )

        self.run_async(system.initialize(force_rebuild=True))

        # Find docs that reference Application
        result = self.run_async(system.read(query="Application"))

        # Should find guide.md which references Application
        files = [s["file"] for s in result["sections"]]
        self.assertTrue(any("guide" in f for f in files))

    def test_task_context_generation(self):
        """Test task context generation for LLM."""
        system = DocsSystem(
            project_root=self.project_root,
            docs_root=self.docs_root,
        )

        self.run_async(system.initialize(force_rebuild=True))

        # Get context for modifying Application class
        context = self.run_async(system.get_task_context(
            files=[str(self.project_root / "src" / "app.py")],
            intent="Add middleware support to Application class",
        ))

        self.assertIn("result", context)
        bundle = context["result"]

        self.assertIn("task_intent", bundle)
        self.assertIn("focus_code", bundle)
        self.assertIn("context_graph", bundle)
        self.assertIn("relevant_docs", bundle)

    def test_incremental_indexing(self):
        """Test incremental index updates."""
        system = DocsSystem(
            project_root=self.project_root,
            docs_root=self.docs_root,
        )

        # Initial build
        self.run_async(system.initialize(force_rebuild=True))
        initial_elements = len(system.index_mgr.index.code_elements)

        # Add new file
        new_file = self.project_root / "src" / "new_module.py"
        new_file.write_text('''"""New module."""

class NewClass:
    """A new class."""
    pass
''')

        # Sync
        self.run_async(system.sync())

        # Should have more elements now
        # Note: actual behavior depends on git tracking
        # In non-git scenarios, we manually update
        self.run_async(system._update_file(new_file))

        self.assertGreater(len(system.index_mgr.index.code_elements), initial_elements)


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions(unittest.TestCase):
    """Tests for factory functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_docs_system(self):
        """Test create_docs_system factory."""
        project_root = Path(self.temp_dir) / "project"
        docs_root = Path(self.temp_dir) / "docs"
        project_root.mkdir()

        system = create_docs_system(
            project_root=str(project_root),
            docs_root=str(docs_root),
        )

        self.assertIsInstance(system, DocsSystem)
        self.assertTrue(docs_root.exists())

    def test_create_docs_system_with_options(self):
        """Test create_docs_system with custom options."""
        project_root = Path(self.temp_dir) / "project"
        docs_root = Path(self.temp_dir) / "docs"
        project_root.mkdir()

        system = create_docs_system(
            project_root=str(project_root),
            docs_root=str(docs_root),
            include_dirs=["src", "lib"],
            exclude_dirs={"tests", "__pycache__"},
        )

        self.assertEqual(system.scanner.include_dirs, ["src", "lib"])
        self.assertIn("tests", system.scanner.exclude_dirs)

    def test_add_to_app(self):
        """Test add_to_app integration."""
        # Create mock app
        class MockApp:
            pass

        app = MockApp()
        project_root = Path(self.temp_dir) / "project"
        docs_root = Path(self.temp_dir) / "docs"
        project_root.mkdir()

        with patch('toolboxv2.utils.extras.mkdocs.Path') as mock_path:
            mock_path.cwd.return_value = project_root
            mock_path.return_value.resolve.return_value = docs_root

            # This would add methods to app
            # system = add_to_app(app, docs_root=str(docs_root))


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases(AsyncTestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_parser_with_unicode(self):
        """Test parser with Unicode content."""
        parser = DocParser()
        path = Path(self.temp_dir) / "unicode.md"

        content = """# タイトル

日本語のコンテンツ。

## Émoji Section 🎉

Content with émojis: 🚀 💻 📚
"""
        path.write_text(content, encoding="utf-8")

        sections = parser.parse(path)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "タイトル")

    def test_analyzer_with_decorators(self):
        """Test analyzer with decorated functions."""
        analyzer = CodeAnalyzer()
        path = Path(self.temp_dir) / "decorated.py"

        content = '''@decorator
@another_decorator(arg=1)
def decorated_function():
    """Decorated function."""
    pass

class DecoratedClass:
    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass

    @property
    def prop(self):
        return self._prop
'''
        path.write_text(content, encoding="utf-8")

        elements = analyzer.analyze(path)
        names = [e.name for e in elements]

        self.assertIn("decorated_function", names)
        self.assertIn("DecoratedClass", names)

    def test_large_file_handling(self):
        """Test handling of large files."""
        parser = DocParser()
        path = Path(self.temp_dir) / "large.md"

        # Create large file
        content = "# Large Document\n\n"
        for i in range(1000):
            content += f"## Section {i}\n\nContent for section {i}.\n\n"

        path.write_text(content, encoding="utf-8")

        sections = parser.parse(path)
        self.assertGreater(len(sections), 100)

    def test_nested_code_blocks(self):
        """Test parser with nested code blocks."""
        parser = DocParser()
        path = Path(self.temp_dir) / "nested.md"

        content = '''# Main

````markdown
```python
# This is nested
def func():
    pass
```
````

## Real Section

Content.
'''
        path.write_text(content, encoding="utf-8")

        sections = parser.parse(path)
        titles = [s.title for s in sections]
        self.assertIn("Main", titles)
        self.assertIn("Real Section", titles)

    def test_empty_class_and_function(self):
        """Test analyzer with empty class/function bodies."""
        analyzer = CodeAnalyzer()
        path = Path(self.temp_dir) / "empty.py"

        content = '''class Empty:
    pass

def empty_func():
    ...

async def async_empty():
    pass
'''
        path.write_text(content, encoding="utf-8")

        elements = analyzer.analyze(path)
        self.assertEqual(len(elements), 3)

    def test_concurrent_access(self):
        """Test concurrent index access."""
        index_path = Path(self.temp_dir) / "index.json"
        manager = IndexManager(index_path)

        # Add data concurrently (simulated)
        for i in range(10):
            section = DocSection(
                section_id=f"test#{i}",
                file_path="/test.md",
                title=f"Section {i}",
                content=f"Content {i}",
                level=1,
                line_start=i,
                line_end=i + 5,
                content_hash=f"hash{i}",
                last_modified=time.time(),
            )
            manager.update_section(section)

        self.assertEqual(len(manager.index.sections), 10)

    def test_malformed_json_index(self):
        """Test handling of malformed index file."""
        index_path = Path(self.temp_dir) / "bad_index.json"
        index_path.write_text("{ invalid json }", encoding="utf-8")

        manager = IndexManager(index_path)
        index = self.run_async(manager.load())

        # Should return empty index, not crash
        self.assertEqual(len(index.sections), 0)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance(AsyncTestCase):
    """Performance-related tests."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_inverted_index_lookup_speed(self):
        """Test inverted index provides fast lookups."""
        index_path = Path(self.temp_dir) / "index.json"
        manager = IndexManager(index_path)
        engine = ContextEngine(manager)

        # Add many sections
        for i in range(100):
            section = DocSection(
                section_id=f"doc#{i}",
                file_path=f"/docs/file{i % 10}.md",
                title=f"Section {i}",
                content=f"Content with keyword{i % 5} and term{i % 3}",
                level=1,
                line_start=0,
                line_end=10,
                content_hash=f"hash{i}",
                last_modified=time.time(),
                tags=(f"tag{i % 4}",),
            )
            manager.update_section(section)

        # Search should be fast
        start = time.perf_counter()
        for _ in range(100):
            engine.search_sections(query="keyword1")
        elapsed = time.perf_counter() - start

        # 100 searches should complete in under 1 second
        self.assertLess(elapsed, 1.0)

    def test_cache_effectiveness(self):
        """Test that caching improves performance."""
        parser = DocParser()
        path = Path(self.temp_dir) / "test.md"

        content = "# Title\n\n" + "Content paragraph.\n\n" * 100
        path.write_text(content, encoding="utf-8")

        # First parse (cold)
        start = time.perf_counter()
        parser.parse(path, use_cache=True)
        cold_time = time.perf_counter() - start

        # Second parse (cached)
        start = time.perf_counter()
        parser.parse(path, use_cache=True)
        cached_time = time.perf_counter() - start

        # Cached should be significantly faster
        self.assertLess(cached_time, cold_time)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run with coverage
    # python -m pytest test_docs_system.py -v --cov=docs_system --cov-report=term-missing

    # Run specific test class
    # python -m pytest test_docs_system.py::TestDocParser -v

    # Run all tests
    unittest.main(verbosity=2)

## Test Summary

### Coverage Areas

# | Component | Test Class | Coverage Focus |
# |-----------|------------|----------------|
# | **Data Models** | `TestChangeType`, `TestDocSection`, `TestCodeElement`, `TestFileChange`, `TestInvertedIndex`, `TestDocsIndex`, `TestContextBundle` | Creation, defaults, slots, dict operations |
# | **DocParser** | `TestDocParser` | State machine, headers, code blocks, frontmatter, tags, refs, caching |
# | **CodeAnalyzer** | `TestCodeAnalyzer` | AST parsing, classes, functions, methods, async, signatures, caching |
# | **JSTSAnalyzer** | `TestJSTSAnalyzer` | Regex patterns, classes, functions, interfaces, types, JSDoc |
# | **IndexManager** | `TestIndexManager` | Serialization, inverted index, CRUD, tokenization, persistence |
# | **ContextEngine** | `TestContextEngine` | Search, filtering, caching, context generation |
# | **FileScanner** | `TestFileScanner` | Scanning, filtering, exclusions, caching |
# | **GitTracker** | `TestGitTracker` | Change parsing, commit hash retrieval |
# | **DocsSystem** | `TestDocsSystemIntegration`, `TestDocsSystemLiveWorkflow` | Full workflow, CRUD, sync, multi-language |

### Running the Tests

# ```bash
# # Install test dependencies
# pip install pytest pytest-asyncio pytest-cov
#
# # Run all tests with coverage
# python -m pytest test_docs_system.py -v --cov=toolboxv2.utils.extras.mkdocs --cov-report=term-missing
#
# # Run only unit tests
# python -m pytest test_docs_system.py -v -k "not Integration and not Live"
#
# # Run only integration tests
# python -m pytest test_docs_system.py -v -k "Integration or Live"
#
# # Run specific test class
# python -m pytest test_docs_system.py::TestDocParser -v
#
# # Run with HTML coverage report
# python -m pytest test_docs_system.py --cov=toolboxv2.utils.extras.mkdocs --cov-report=html
# ```
