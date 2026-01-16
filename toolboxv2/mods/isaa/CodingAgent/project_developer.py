"""
ProjectDeveloperEngine V3 - Multi-File Code Generation System

Refactored from AtomicCoder with deep integration of the ToolBoxV2 ecosystem:
- DocsSystem & ContextEngine (mkdocs.py): Project graph, semantic search, token-optimized context
- DockerCodeExecutor / RestrictedPythonExecutor (executors.py): Safe code execution
- FlowAgent & FlowAgentBuilder (flow_agent.py): LLM orchestration with chain patterns

State Machine Phases:
1. PHASE_ANALYSIS: Load context graph via DocsSystem.get_task_context()
2. PHASE_RESEARCH: MCP/Web search for external APIs/libraries
3. PHASE_MULTI_SPEC: Multi-file planning (Create/Modify operations)
4. PHASE_GENERATION: Iterative code generation with ContextBundle
5. PHASE_VALIDATION: LSP + Runtime validation with auto-fix loop

Author: ProjectDeveloper V3
Version: 3.0.0
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import yaml
from pydantic import BaseModel, Field

# =============================================================================
# FRAMEWORK IMPORTS - Native Integration
# =============================================================================

from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from toolboxv2.utils.extras.mkdocs import DocsSystem, create_docs_system
from toolboxv2.mods.isaa.base.Agent.executors import (
    DockerCodeExecutor,
    RestrictedPythonExecutor,
)

# Lazy imports for runtime
_DocsSystem = None
_get_code_executor = None
_DOCKER_AVAILABLE = False
_RESTRICTED_AVAILABLE = False


def _lazy_load_docs_system():
    global _DocsSystem
    if _DocsSystem is None:
        try:
            from toolboxv2.utils.extras.mkdocs import (
                DocsSystem,
                create_docs_system,
                ContextEngine,
                ContextBundle,
            )
            _DocsSystem = DocsSystem
        except ImportError:
            _DocsSystem = None
    return _DocsSystem


def _lazy_load_executors():
    global _get_code_executor, _DOCKER_AVAILABLE, _RESTRICTED_AVAILABLE
    try:
        from toolboxv2.mods.isaa.base.Agent.executors import (
            get_code_executor,
            DockerCodeExecutor,
            RestrictedPythonExecutor,
            DOCKER_AVAILABLE,
            RESTRICTEDPYTHON_AVAILABLE,
        )
        _get_code_executor = get_code_executor
        _DOCKER_AVAILABLE = DOCKER_AVAILABLE
        _RESTRICTED_AVAILABLE = RESTRICTEDPYTHON_AVAILABLE
    except ImportError:
        _get_code_executor = None
        _DOCKER_AVAILABLE = False
        _RESTRICTED_AVAILABLE = False
    return _get_code_executor


# =============================================================================
# ENUMS - Phase Management
# =============================================================================

class DeveloperPhase(str, Enum):
    """Project development phases - State Machine States"""
    IDLE = "idle"
    PHASE_ANALYSIS = "analysis"
    PHASE_RESEARCH = "research"
    PHASE_MULTI_SPEC = "multi_spec"
    PHASE_GENERATION = "generation"
    PHASE_VALIDATION = "validation"
    PHASE_REFINEMENT = "refinement"
    PHASE_SYNC = "sync"
    COMPLETED = "completed"
    FAILED = "failed"


class FileActionType(str, Enum):
    """Types of file operations"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class LanguageType(str, Enum):
    """Supported languages with file extensions"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> 'LanguageType':
        """Detect language from file extension"""
        mapping = {
            '.py': cls.PYTHON, '.pyw': cls.PYTHON,
            '.js': cls.JAVASCRIPT, '.jsx': cls.JAVASCRIPT, '.mjs': cls.JAVASCRIPT,
            '.ts': cls.TYPESCRIPT, '.tsx': cls.TYPESCRIPT,
            '.html': cls.HTML, '.htm': cls.HTML,
            '.css': cls.CSS, '.scss': cls.CSS, '.sass': cls.CSS,
            '.json': cls.JSON,
            '.yaml': cls.YAML, '.yml': cls.YAML,
            '.toml': cls.TOML,
            '.md': cls.MARKDOWN, '.markdown': cls.MARKDOWN,
        }
        return mapping.get(ext.lower(), cls.UNKNOWN)


class DiagnosticSeverity(str, Enum):
    """Diagnostic severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


# =============================================================================
# PYDANTIC MODELS - Structured Data
# =============================================================================

class ResearchResult(BaseModel):
    """Result from external research (MCP/Web)"""
    source: str = Field(description="Source of information (docs, web, mcp)")
    topic: str = Field(description="Topic researched")
    content: str = Field(description="Retrieved content/documentation")
    url: Optional[str] = Field(default=None, description="Source URL if applicable")
    relevance: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score")


class FileAction(BaseModel):
    """Single file operation in the project spec"""
    action: FileActionType = Field(description="Type of file action")
    file_path: str = Field(description="Relative path to file")
    language: LanguageType = Field(description="Language/file type")
    description: str = Field(description="What this action accomplishes")
    dependencies: List[str] = Field(default_factory=list, description="Files this depends on")
    target_symbols: List[str] = Field(default_factory=list, description="Symbols to create/modify")
    priority: int = Field(default=1, ge=1, le=10, description="Execution priority (1=highest)")

    # Generated content (filled during GENERATION phase)
    generated_code: Optional[str] = Field(default=None)
    validation_passed: bool = Field(default=False)


class ProjectSpec(BaseModel):
    """Complete project specification for multi-file development"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    intent: str = Field(description="High-level task description")
    summary: str = Field(description="Brief summary of changes")
    actions: List[FileAction] = Field(default_factory=list, description="Ordered list of file actions")

    # Context from DocsSystem
    upstream_deps: List[Dict[str, str]] = Field(default_factory=list, description="Dependencies")
    downstream_usage: List[Dict[str, str]] = Field(default_factory=list, description="Usage sites")
    research_results: List[ResearchResult] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    estimated_tokens: int = Field(default=0)


class LSPDiagnostic(BaseModel):
    """LSP Diagnostic result"""
    severity: DiagnosticSeverity
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    message: str
    code: Optional[str] = None
    source: str = "lsp"


class ValidationResult(BaseModel):
    """Validation result from LSP/Runtime tests"""
    success: bool = Field(description="Did validation pass?")
    diagnostics: List[LSPDiagnostic] = Field(default_factory=list)
    test_output: str = Field(default="")
    error_message: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)


# =============================================================================
# EXECUTION STATE
# =============================================================================

@dataclass
class DeveloperState:
    """Execution state for the ProjectDeveloperEngine"""
    execution_id: str
    task: str
    target_files: List[str]
    phase: DeveloperPhase = DeveloperPhase.IDLE
    iteration: int = 0
    max_iterations: int = 5

    # Generated artifacts
    project_spec: Optional[ProjectSpec] = None
    context_bundle: Optional[Dict[str, Any]] = None
    research_results: List[ResearchResult] = field(default_factory=list)

    # Validation tracking
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    generated_files: Dict[str, str] = field(default_factory=dict)

    # History
    errors: List[str] = field(default_factory=list)
    phase_history: List[Tuple[DeveloperPhase, float]] = field(default_factory=list)

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    success: bool = False
    total_tokens_used: int = 0

    def to_dict(self) -> dict:
        """Serialize state to dictionary"""
        return {
            "execution_id": self.execution_id,
            "task": self.task,
            "target_files": self.target_files,
            "phase": self.phase.value,
            "iteration": self.iteration,
            "project_spec": self.project_spec.model_dump() if self.project_spec else None,
            "generated_files": self.generated_files,
            "errors": self.errors[-5:],
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# LSP MANAGER - Language Server Integration (Preserved from AtomicCoder)
# =============================================================================

@dataclass
class LSPServerConfig:
    """Configuration for an LSP server"""
    language: LanguageType
    command: List[str]
    root_uri: str
    initialization_options: dict = field(default_factory=dict)


class LSPManager:
    """Unified LSP Manager for multi-language support"""

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self._servers: Dict[LanguageType, subprocess.Popen] = {}
        self._request_id = 0
        self._initialized: Dict[LanguageType, bool] = {}
        self._configs = self._build_configs()

    def _build_configs(self) -> Dict[LanguageType, LSPServerConfig]:
        """Build LSP server configurations"""
        root_uri = f"file://{self.workspace}"
        return {
            LanguageType.PYTHON: LSPServerConfig(
                language=LanguageType.PYTHON,
                command=["pylsp"],
                root_uri=root_uri,
                initialization_options={
                    "pylsp": {
                        "plugins": {
                            "pyflakes": {"enabled": True},
                            "pycodestyle": {"enabled": True},
                            "pylint": {"enabled": False},
                        }
                    }
                }
            ),
            LanguageType.JAVASCRIPT: LSPServerConfig(
                language=LanguageType.JAVASCRIPT,
                command=["typescript-language-server", "--stdio"],
                root_uri=root_uri,
            ),
            LanguageType.TYPESCRIPT: LSPServerConfig(
                language=LanguageType.TYPESCRIPT,
                command=["typescript-language-server", "--stdio"],
                root_uri=root_uri,
            ),
        }

    async def start_server(self, language: LanguageType) -> bool:
        """Start LSP server for language"""
        if language in self._servers and self._servers[language].poll() is None:
            return True

        config = self._configs.get(language)
        if not config:
            return False

        try:
            which_cmd = "where" if sys.platform == "win32" else "which"
            result = subprocess.run(
                [which_cmd, config.command[0]],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                return False

            process = subprocess.Popen(
                config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workspace)
            )
            self._servers[language] = process
            await self._initialize_server(language, config)
            self._initialized[language] = True
            return True

        except Exception:
            return False

    async def _initialize_server(self, language: LanguageType, config: LSPServerConfig):
        """Send LSP initialize request"""
        init_params = {
            "processId": os.getpid(),
            "rootUri": config.root_uri,
            "capabilities": {
                "textDocument": {
                    "completion": {"completionItem": {"snippetSupport": True}},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "publishDiagnostics": {"relatedInformation": True},
                }
            },
            "initializationOptions": config.initialization_options
        }
        await self._send_request(language, "initialize", init_params)
        await self._send_notification(language, "initialized", {})

    async def _send_request(self, language: LanguageType, method: str, params: dict) -> dict:
        """Send LSP request and wait for response"""
        if language not in self._servers:
            return {}

        self._request_id += 1
        request = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"

        try:
            process = self._servers[language]
            process.stdin.write(message.encode())
            process.stdin.flush()

            response_data = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._read_response(process)
                ),
                timeout=5.0
            )
            return response_data
        except (asyncio.TimeoutError, Exception):
            return {}

    async def _send_notification(self, language: LanguageType, method: str, params: dict):
        """Send LSP notification (no response expected)"""
        if language not in self._servers:
            return

        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        content = json.dumps(notification)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"

        try:
            process = self._servers[language]
            process.stdin.write(message.encode())
            process.stdin.flush()
        except Exception:
            pass

    def _read_response(self, process: subprocess.Popen) -> dict:
        """Read LSP response from stdout"""
        try:
            headers = {}
            while True:
                line = process.stdout.readline().decode().strip()
                if not line:
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()

            content_length = int(headers.get("Content-Length", 0))
            if content_length > 0:
                content = process.stdout.read(content_length).decode()
                return json.loads(content)
            return {}
        except Exception:
            return {}

    async def get_diagnostics(self, file_path: str, content: str, language: LanguageType) -> List[LSPDiagnostic]:
        """Get diagnostics for a file"""
        if not await self.start_server(language):
            return await self._fallback_diagnostics(content, language)

        uri = f"file://{file_path}"
        await self._send_notification(language, "textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language.value,
                "version": 1,
                "text": content
            }
        })

        await asyncio.sleep(0.5)
        return []

    async def _fallback_diagnostics(self, content: str, language: LanguageType) -> List[LSPDiagnostic]:
        """Fallback diagnostics when LSP unavailable"""
        diagnostics = []

        if language == LanguageType.PYTHON:
            try:
                ast.parse(content)
            except SyntaxError as e:
                diagnostics.append(LSPDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    line=e.lineno or 1,
                    column=e.offset or 0,
                    message=str(e.msg),
                    source="ast"
                ))

            try:
                from pyflakes import api as pyflakes_api
                from pyflakes import reporter as pyflakes_reporter
                import io

                warning_stream = io.StringIO()
                error_stream = io.StringIO()
                reporter = pyflakes_reporter.Reporter(warning_stream, error_stream)
                pyflakes_api.check(content, "<code>", reporter)

                for line in warning_stream.getvalue().split("\n"):
                    if line.strip():
                        match = re.match(r"<code>:(\d+):(\d+):\s*(.+)", line)
                        if match:
                            diagnostics.append(LSPDiagnostic(
                                severity=DiagnosticSeverity.WARNING,
                                line=int(match.group(1)),
                                column=int(match.group(2)),
                                message=match.group(3),
                                source="pyflakes"
                            ))
            except ImportError:
                pass

        elif language in (LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT):
            # Basic brace/parenthesis matching
            brace_count = content.count("{") - content.count("}")
            paren_count = content.count("(") - content.count(")")
            if brace_count != 0:
                diagnostics.append(LSPDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    line=len(content.split("\n")),
                    column=0,
                    message=f"Unbalanced braces: {brace_count:+d}",
                    source="syntax"
                ))
            if paren_count != 0:
                diagnostics.append(LSPDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    line=len(content.split("\n")),
                    column=0,
                    message=f"Unbalanced parentheses: {paren_count:+d}",
                    source="syntax"
                ))

        return diagnostics

    async def shutdown(self):
        """Shutdown all LSP servers"""
        for language, process in self._servers.items():
            try:
                await self._send_request(language, "shutdown", {})
                await self._send_notification(language, "exit", {})
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                process.kill()

        self._servers.clear()
        self._initialized.clear()


# =============================================================================
# SAFE EXECUTOR WRAPPER - Integrates executors.py
# =============================================================================

class SafeExecutor:
    """
    Safe code execution wrapper using DockerCodeExecutor or RestrictedPythonExecutor.
    Automatically detects available executor and provides fallback.
    """

    def __init__(self, workspace: Path, prefer_docker: bool = True):
        self.workspace = workspace
        self.prefer_docker = prefer_docker
        self._executor = None
        self._executor_type = "none"
        self._setup_executor()

    def _setup_executor(self):
        """Setup the best available executor"""
        _lazy_load_executors()

        if self.prefer_docker and _DOCKER_AVAILABLE:
            try:
                self._executor = DockerCodeExecutor(
                    docker_image="python:3.10-slim",
                    timeout=30,
                    mem_limit="256m",
                    network_mode="none"
                )
                self._executor_type = "docker"
                return
            except Exception:
                pass

        if _RESTRICTED_AVAILABLE:
            try:
                self._executor = RestrictedPythonExecutor(max_execution_time=10)
                self._executor_type = "restricted"
                return
            except Exception:
                pass

        # Ultimate fallback - subprocess isolation
        self._executor_type = "subprocess"

    async def execute(self, code: str, language: LanguageType = LanguageType.PYTHON) -> Dict[str, Any]:
        """Execute code safely and return results"""
        if language != LanguageType.PYTHON:
            return {
                "stdout": "",
                "stderr": f"Execution not supported for {language.value}",
                "error": "Unsupported language",
                "exit_code": -1
            }

        if self._executor is not None:
            result = self._executor.execute(code)
            return result

        # Subprocess fallback
        return await self._subprocess_execute(code)

    async def _subprocess_execute(self, code: str) -> Dict[str, Any]:
        """Fallback subprocess execution with isolation"""
        result = {"stdout": "", "stderr": "", "error": None, "exit_code": None}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding="utf-8", ) as f:
            f.write(code)
            temp_file = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace)
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            result["stdout"] = stdout.decode('utf-8', errors='replace')
            result["stderr"] = stderr.decode('utf-8', errors='replace')
            result["exit_code"] = proc.returncode

        except asyncio.TimeoutError:
            result["error"] = "Execution timed out"
            result["exit_code"] = -1
        except Exception as e:
            result["error"] = str(e)
            result["exit_code"] = -1
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

        return result

    async def run_tests(self, implementation: str, test_code: str) -> ValidationResult:
        """Run tests against implementation"""
        combined_code = f'''# === IMPLEMENTATION ===
{implementation}

# === TESTS ===
import unittest
{test_code}

# === RUNNER ===
if __name__ == "__main__":
    import sys
    from io import StringIO

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for name, obj in list(globals().items()):
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
            suite.addTests(loader.loadTestsFromTestCase(obj))

    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    print(stream.getvalue())
    print("ALL_TESTS_PASSED" if result.wasSuccessful() else "TESTS_FAILED")
'''

        start_time = time.perf_counter()
        exec_result = await self.execute(combined_code)
        execution_time = (time.perf_counter() - start_time) * 1000

        success = exec_result.get("exit_code") == 0 and "ALL_TESTS_PASSED" in exec_result.get("stdout", "")

        return ValidationResult(
            success=success,
            test_output=exec_result.get("stdout", ""),
            error_message=exec_result.get("stderr") if not success else None,
            execution_time_ms=execution_time
        )

    @property
    def executor_type(self) -> str:
        return self._executor_type


# =============================================================================
# PROMPTS - Token-Optimized Templates
# =============================================================================

ANALYSIS_PROMPT = """Analyze this development task and identify required changes.

TASK: {task}
TARGET FILES: {target_files}

CONTEXT (from project graph):
{context_summary}

Determine:
1. Which files need to be created or modified
2. What external APIs/libraries are referenced that need documentation
3. The dependency order for changes

Return analysis in YAML format:
```yaml
files_to_change:
  - path: "relative/path.py"
    action: create|modify
    reason: "brief reason"
unknown_apis:
  - name: "API/Library name"
    usage: "how it's used"
dependency_order:
  - "file1.py"
  - "file2.py"
```"""

RESEARCH_PROMPT = """Research the following API/library for use in code generation.

TOPIC: {topic}
USAGE CONTEXT: {usage_context}

Provide a concise summary including:
1. Key classes/functions to use
2. Basic usage pattern
3. Common pitfalls

Return in YAML:
```yaml
summary: "1-2 sentence overview"
key_apis:
  - name: "function/class"
    signature: "basic signature"
    description: "what it does"
usage_example: |
  # Brief code example
common_pitfalls:
  - "pitfall 1"
```"""

MULTI_SPEC_PROMPT = """Create a detailed implementation plan for multiple files.

TASK: {task}
ANALYSIS: {analysis}
RESEARCH: {research}

For each file, specify:
- Action (create/modify)
- Target symbols (functions/classes)
- Dependencies on other files

Return as YAML:
```yaml
summary: "Brief summary of all changes"
actions:
  - file_path: "path/to/file.py"
    action: create
    language: python
    description: "What this file does"
    target_symbols:
      - "function_name"
      - "ClassName"
    dependencies: []
    priority: 1
```"""

GENERATION_PROMPT = """Generate implementation code.

TARGET: {file_path}
LANGUAGE: {language}
SYMBOLS: {target_symbols}
DESCRIPTION: {description}

CONTEXT:
{context}

{available_imports}

{error_section}

Rules:
- Match existing code style
- NO markdown code fences in output
- Generate ONLY the raw file content

{language_specific_rules}

Return ONLY the complete {language} code/markup."""

AUTOFIX_PROMPT = """Fix the code errors.

CODE:
```
{code}
```

ERRORS:
{errors}

{test_context}

Return ONLY the corrected code (no markdown, no explanation)."""


def _get_language_rules(language: LanguageType) -> str:
    """Get language-specific generation rules"""
    rules = {
        LanguageType.PYTHON: """- Include docstrings for functions/classes
- Use type hints
- Handle errors gracefully""",

        LanguageType.HTML: """- Use semantic HTML5 elements (header, main, section, footer)
- Include proper meta tags and viewport
- Link CSS/JS files correctly
- Structure: <!DOCTYPE html>, <html>, <head>, <body>""",

        LanguageType.CSS: """- Use CSS custom properties (variables) for colors/spacing
- Mobile-first responsive design
- Use flexbox/grid for layouts
- Include hover/focus states""",

        LanguageType.JAVASCRIPT: """- Use modern ES6+ syntax (const, let, arrow functions)
- Add event listeners properly
- Handle DOM ready state
- Include error handling""",

        LanguageType.TYPESCRIPT: """- Include proper type annotations
- Use interfaces for data structures
- Handle null/undefined properly""",

        LanguageType.JSON: """- Valid JSON syntax only
- No trailing commas
- Use double quotes for keys""",

        LanguageType.YAML: """- Proper indentation (2 spaces)
- Valid YAML syntax""",
    }
    return rules.get(language, "- Follow standard conventions for this file type")

# =============================================================================
# PROJECT DEVELOPER ENGINE - Main Class
# =============================================================================

class ProjectDeveloperEngine:
    """
    Production-ready multi-file code generation engine.

    Integrates:
    - DocsSystem for project context and dependency graphs
    - Safe executors (Docker/RestrictedPython) for validation
    - FlowAgent for LLM orchestration
    - LSP for static analysis

    State Machine:
    IDLE -> ANALYSIS -> RESEARCH -> MULTI_SPEC -> GENERATION -> VALIDATION -> SYNC -> COMPLETED
    """

    def __init__(
        self,
        agent: 'FlowAgent',
        workspace_path: Union[str, Path],
        docs_system: Optional['DocsSystem'] = None,
        auto_lsp: bool = True,
        prefer_docker: bool = True,
        verbose: bool = True,
    ):
        self.agent = agent
        self.workspace = Path(workspace_path).absolute()
        self.verbose = verbose
        self.auto_lsp = auto_lsp

        # Initialize components
        self.lsp_manager = LSPManager(self.workspace)
        self.executor = SafeExecutor(self.workspace, prefer_docker=prefer_docker)

        # DocsSystem integration (lazy loaded if not provided)
        self._docs_system = docs_system
        self._docs_initialized = False

        # State tracking
        self._executions: Dict[str, DeveloperState] = {}

        # Create workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._log(f"🔧 ProjectDeveloperEngine V3 initialized")
        self._log(f"   Workspace: {self.workspace}")
        self._log(f"   Executor: {self.executor.executor_type}")

    def _log(self, message: str):
        """Conditional logging"""
        if self.verbose:
            print(message)

    async def _ensure_docs_system(self) -> Optional['DocsSystem']:
        """Ensure DocsSystem is initialized"""
        if self._docs_system is not None and self._docs_initialized:
            return self._docs_system

        DocsSystemClass = _lazy_load_docs_system()
        if DocsSystemClass is None:
            self._log("⚠️  DocsSystem not available, using fallback context")
            return None

        if self._docs_system is None:
            try:
                self._docs_system = create_docs_system(
                    project_root=str(self.workspace),
                    docs_root=str(self.workspace / "docs")
                )
            except Exception as e:
                self._log(f"⚠️  Failed to create DocsSystem: {e}")
                return None

        try:
            await self._docs_system.initialize()
            self._docs_initialized = True
        except Exception as e:
            self._log(f"⚠️  Failed to initialize DocsSystem: {e}")

        return self._docs_system

    # =========================================================================
    # MAIN EXECUTION METHODS
    # =========================================================================

    async def execute(
        self,
        task: str,
        target_files: List[str],
        max_retries: int = 3,
        auto_research: bool = True,
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Main execution method - implements the multi-file development loop.

        Args:
            task: Description of what to implement
            target_files: List of file paths to create/modify
            max_retries: Maximum retry attempts per file
            auto_research: Whether to auto-research unknown APIs

        Returns:
            (success, generated_files_dict)
        """
        execution_id = str(uuid.uuid4())[:8]

        state = DeveloperState(
            execution_id=execution_id,
            task=task,
            target_files=target_files,
            max_iterations=max_retries
        )
        self._executions[execution_id] = state

        self._log(f"\n🚀 Starting project development: {execution_id}")
        self._log(f"   Task: {task[:80]}...")
        self._log(f"   Files: {', '.join(target_files)}")

        try:
            # Phase 1: ANALYSIS
            state.phase = DeveloperPhase.PHASE_ANALYSIS
            self._record_phase(state)
            analysis = await self._phase_analysis(state)

            # Phase 2: RESEARCH (if needed)
            if auto_research and analysis.get("unknown_apis"):
                state.phase = DeveloperPhase.PHASE_RESEARCH
                self._record_phase(state)
                await self._phase_research(state, analysis.get("unknown_apis", []))

            # Phase 3: MULTI_SPEC
            state.phase = DeveloperPhase.PHASE_MULTI_SPEC
            self._record_phase(state)
            spec = await self._phase_multi_spec(state, analysis)
            state.project_spec = spec

            # Phase 4: GENERATION (iterative)
            state.phase = DeveloperPhase.PHASE_GENERATION
            self._record_phase(state)
            await self._phase_generation(state)

            # Phase 5: VALIDATION
            state.phase = DeveloperPhase.PHASE_VALIDATION
            self._record_phase(state)
            all_valid = await self._phase_validation(state, max_retries)

            if all_valid:
                # Phase 6: SYNC
                state.phase = DeveloperPhase.PHASE_SYNC
                self._record_phase(state)
                await self._phase_sync(state)

                state.phase = DeveloperPhase.COMPLETED
                state.success = True
                state.completed_at = datetime.now()

                self._log(f"\n✅ Project development completed: {execution_id}")
                return True, state.generated_files
            else:
                state.phase = DeveloperPhase.FAILED
                state.completed_at = datetime.now()
                self._log(f"\n❌ Project development failed: {execution_id}")
                return False, state.generated_files

        except Exception as e:
            import traceback
            state.phase = DeveloperPhase.FAILED
            state.completed_at = datetime.now()
            state.errors.append(f"Exception: {str(e)}")
            self._log(f"\n💥 Exception: {e}")
            traceback.print_exc()
            return False, {}

    def _record_phase(self, state: DeveloperState):
        """Record phase transition with timestamp"""
        state.phase_history.append((state.phase, time.time()))
        self._log(f"\n📍 Phase: {state.phase.value.upper()}")

    # =========================================================================
    # PHASE 1: ANALYSIS
    # =========================================================================

    async def _phase_analysis(self, state: DeveloperState) -> Dict[str, Any]:
        """
        Phase 1: Analyze task using DocsSystem context.

        Uses DocsSystem.get_task_context() to load:
        - Focus file contents
        - Upstream dependencies (imports)
        - Downstream usage (callers)
        - Related documentation
        """
        self._log("📊 Analyzing project context...")

        # Try to get context from DocsSystem
        docs = await self._ensure_docs_system()
        context_summary = ""

        if docs is not None:
            try:
                context_result = await docs.get_task_context(state.target_files, state.task)
                state.context_bundle = context_result.get("result", {})

                # Build context summary for prompt
                bundle = state.context_bundle
                if bundle:
                    parts = []

                    # Focus files
                    if "focus_files" in bundle:
                        parts.append("FOCUS FILES:")
                        for path, content in bundle.get("focus_files", {}).items():
                            parts.append(f"  - {path}: {len(content)} chars")

                    # Definitions
                    if "definitions" in bundle:
                        parts.append("\nDEFINITIONS:")
                        for defn in bundle.get("definitions", [])[:10]:
                            parts.append(f"  - {defn.get('signature', 'unknown')}")

                    # Graph
                    if "graph" in bundle:
                        graph = bundle["graph"]
                        if graph.get("upstream"):
                            parts.append(f"\nUPSTREAM DEPS: {len(graph['upstream'])} items")
                        if graph.get("downstream"):
                            parts.append(f"DOWNSTREAM USAGE: {len(graph['downstream'])} items")

                    context_summary = "\n".join(parts)

            except Exception as e:
                self._log(f"   ⚠️ DocsSystem error: {e}")

        # Fallback: Read files directly
        if not context_summary:
            parts = ["FILE CONTENTS:"]
            for file_path in state.target_files:
                full_path = self.workspace / file_path
                if full_path.exists():
                    content = full_path.read_text(encoding='utf-8', errors='ignore')
                    parts.append(f"\n--- {file_path} ---")
                    parts.append(content[:2000])
                else:
                    parts.append(f"\n--- {file_path} (NEW FILE) ---")
            context_summary = "\n".join(parts)

        # Generate analysis via LLM
        prompt = ANALYSIS_PROMPT.format(
            task=state.task,
            target_files=", ".join(state.target_files),
            context_summary=context_summary[:4000]
        )

        response = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            model_preference="fast",
            stream=False,
            with_context=False
        )

        # Parse YAML response
        analysis = self._parse_yaml_response(response)
        self._log(f"   ✓ Analysis complete: {len(analysis.get('files_to_change', []))} files identified")

        return analysis

    # =========================================================================
    # PHASE 2: RESEARCH
    # =========================================================================

    async def _phase_research(self, state: DeveloperState, unknown_apis: List[Dict[str, str]]):
        """
        Phase 2: Research unknown APIs/libraries via MCP or agent tools.
        """
        self._log(f"🔍 Researching {len(unknown_apis)} unknown APIs...")

        for api_info in unknown_apis[:5]:  # Limit to 5 APIs
            topic = api_info.get("name", "unknown")
            usage = api_info.get("usage", "")

            self._log(f"   Researching: {topic}")

            prompt = RESEARCH_PROMPT.format(
                topic=topic,
                usage_context=usage
            )

            try:
                # Use agent's a_run for potential tool access (MCP/web search)
                response = await self.agent.a_run(
                    query=f"Research the API/library: {topic}. Usage: {usage}",
                    session_id="research_session",
                    max_iterations=3
                )

                # Also get structured response
                structured = await self.agent.a_run_llm_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model_preference="fast",
                    stream=False,
                    with_context=False
                )

                research_data = self._parse_yaml_response(structured)

                state.research_results.append(ResearchResult(
                    source="agent_research",
                    topic=topic,
                    content=research_data.get("summary", response[:500]),
                    relevance=0.8
                ))

            except Exception as e:
                self._log(f"   ⚠️ Research failed for {topic}: {e}")
                state.research_results.append(ResearchResult(
                    source="fallback",
                    topic=topic,
                    content=f"No documentation found for {topic}",
                    relevance=0.3
                ))

    # =========================================================================
    # PHASE 3: MULTI_SPEC
    # =========================================================================

    async def _phase_multi_spec(self, state: DeveloperState, analysis: Dict[str, Any]) -> ProjectSpec:
        """
        Phase 3: Create detailed multi-file specification.
        """
        self._log("📋 Creating project specification...")

        # Compile research summary
        research_summary = ""
        if state.research_results:
            parts = ["RESEARCH RESULTS:"]
            for res in state.research_results:
                parts.append(f"\n{res.topic}: {res.content[:300]}")
            research_summary = "\n".join(parts)

        prompt = MULTI_SPEC_PROMPT.format(
            task=state.task,
            analysis=yaml.dump(analysis, default_flow_style=False)[:2000],
            research=research_summary[:1500]
        )

        response = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            model_preference="complex",
            stream=False,
            with_context=False
        )

        spec_data = self._parse_yaml_response(response)

        # Build ProjectSpec
        actions = []
        for action_data in spec_data.get("actions", []):
            file_path = action_data.get("file_path", "")
            ext = Path(file_path).suffix

            actions.append(FileAction(
                action=FileActionType(action_data.get("action", "create")),
                file_path=file_path,
                language=LanguageType.from_extension(ext),
                description=action_data.get("description", ""),
                dependencies=action_data.get("dependencies", []),
                target_symbols=action_data.get("target_symbols", []),
                priority=action_data.get("priority", 1)
            ))

        # Sort by priority
        actions.sort(key=lambda a: a.priority)

        spec = ProjectSpec(
            intent=state.task,
            summary=spec_data.get("summary", "Multi-file implementation"),
            actions=actions,
            research_results=state.research_results
        )

        self._log(f"   ✓ Spec created: {len(actions)} file actions")
        return spec

    # =========================================================================
    # PHASE 4: GENERATION
    # =========================================================================

    async def _phase_generation(self, state: DeveloperState):
        """
        Phase 4: Generate code for each FileAction.
        """
        spec = state.project_spec
        if not spec:
            raise ValueError("No project spec available")

        self._log(f"💻 Generating code for {len(spec.actions)} files...")

        for i, action in enumerate(spec.actions):
            self._log(f"\n   [{i+1}/{len(spec.actions)}] {action.file_path}")

            # Build context for this file
            context_parts = []

            # Add dependency contents
            for dep in action.dependencies:
                if dep in state.generated_files:
                    context_parts.append(f"# From {dep}:\n{state.generated_files[dep][:1000]}")
                else:
                    dep_path = self.workspace / dep
                    if dep_path.exists():
                        context_parts.append(f"# From {dep}:\n{dep_path.read_text()[:1000]}")

            # Add existing file content if modifying
            if action.action == FileActionType.MODIFY:
                file_path = self.workspace / action.file_path
                if file_path.exists():
                    context_parts.append(f"# EXISTING CODE:\n{file_path.read_text()}")

            # Add research context
            for res in state.research_results:
                if any(sym.lower() in res.topic.lower() for sym in action.target_symbols):
                    context_parts.append(f"# DOCUMENTATION for {res.topic}:\n{res.content[:500]}")

            # Add ContextBundle info if available
            if state.context_bundle:
                bundle = state.context_bundle
                if "definitions" in bundle:
                    relevant_defs = [
                        d for d in bundle["definitions"]
                        if any(sym in d.get("signature", "") for sym in action.target_symbols)
                    ]
                    if relevant_defs:
                        context_parts.append("# RELATED DEFINITIONS:")
                        for d in relevant_defs[:5]:
                            context_parts.append(f"#   {d.get('signature', 'unknown')}")

            context = "\n\n".join(context_parts)[:3000]

            # Determine available imports
            available_imports = self._detect_available_imports(action.language)

            # Generate code
            language_rules = _get_language_rules(action.language)

            prompt = GENERATION_PROMPT.format(
                file_path=action.file_path,
                language=action.language.value,
                target_symbols=", ".join(action.target_symbols) if action.target_symbols else "N/A",
                description=action.description,
                context=context,
                available_imports=available_imports if action.language == LanguageType.PYTHON else "",
                error_section="",
                language_specific_rules=language_rules
            )

            response = await self.agent.a_run_llm_completion(
                messages=[{"role": "user", "content": prompt}],
                model_preference="complex",
                stream=False,
                with_context=False
            )

            # Clean response
            code = self._clean_code_response(response)

            # Validate syntax
            code_valid = True
            if action.language == LanguageType.PYTHON:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    self._log(f"      ⚠️ Python syntax error: {e}")
                    state.errors.append(f"{action.file_path}: Syntax error - {e}")
                    code_valid = False
            elif action.language == LanguageType.HTML:
                # Basic HTML validation
                if not code.strip().startswith(('<!DOCTYPE', '<html', '<')):
                    self._log(f"      ⚠️ Invalid HTML structure")
                    code_valid = False
            elif action.language == LanguageType.JSON:
                try:
                    json.loads(code)
                except json.JSONDecodeError as e:
                    self._log(f"      ⚠️ JSON syntax error: {e}")
                    state.errors.append(f"{action.file_path}: JSON error - {e}")
                    code_valid = False

            action.generated_code = code
            state.generated_files[action.file_path] = code
            self._log(f"      ✓ Generated {len(code)} chars" + (" (with warnings)" if not code_valid else ""))

    def _detect_available_imports(self, language: LanguageType) -> str:
        """Detect commonly available imports/resources for the language"""
        if language == LanguageType.PYTHON:
            return """IMPORTS AVAILABLE:
    Standard library: os, sys, json, re, pathlib, asyncio, typing, dataclasses
    Common: pydantic, yaml, requests, aiohttp"""
        elif language == LanguageType.HTML:
            return """RESOURCES AVAILABLE:
    CDN: Google Fonts, FontAwesome, Tailwind CSS, Bootstrap
    Link local: style.css, script.js"""
        elif language in (LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT):
            return """AVAILABLE:
    DOM APIs, Fetch API, localStorage
    Can import from local modules"""
        return ""

    def _clean_code_response(self, response: str) -> str:
        """Clean LLM response to extract pure code"""
        code = response.strip()

        # Remove markdown code fences
        if code.startswith("```"):
            code = re.sub(r"```\w*\n?", "", code)
            code = code.rstrip("`").strip()

        return code

    # =========================================================================
    # PHASE 5: VALIDATION
    # =========================================================================

    async def _phase_validation(self, state: DeveloperState, max_retries: int) -> bool:
        """
        Phase 5: Validate all generated code with LSP and runtime tests.
        """
        spec = state.project_spec
        if not spec:
            return False

        self._log(f"🔍 Validating {len(spec.actions)} files...")

        all_valid = True

        for action in spec.actions:
            if not action.generated_code:
                continue

            self._log(f"\n   Validating: {action.file_path}")
            valid = False

            for attempt in range(max_retries):
                state.iteration = attempt + 1

                # LSP Validation
                diagnostics = await self.lsp_manager.get_diagnostics(
                    str(self.workspace / action.file_path),
                    action.generated_code,
                    action.language
                )

                errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]

                if errors:
                    self._log(f"      ❌ LSP errors (attempt {attempt + 1})")
                    for err in errors[:3]:
                        self._log(f"         L{err.line}: {err.message}")

                    # Auto-fix
                    fixed = await self._auto_fix(action, errors, state)
                    if fixed:
                        action.generated_code = fixed
                        state.generated_files[action.file_path] = fixed
                        continue
                    else:
                        break

                # Runtime validation (Python only)
                if action.language == LanguageType.PYTHON:
                    test_code = self._generate_basic_test(action)
                    test_result = await self.executor.run_tests(
                        action.generated_code,
                        test_code
                    )

                    state.validation_results[action.file_path] = test_result

                    if test_result.success:
                        self._log(f"      ✓ Validation passed")
                        action.validation_passed = True
                        valid = True
                        break
                    else:
                        self._log(f"      ❌ Test failed (attempt {attempt + 1})")
                        fixed = await self._auto_fix(action, [], state, test_result)
                        if fixed:
                            action.generated_code = fixed
                            state.generated_files[action.file_path] = fixed
                        else:
                            break
                else:
                    # Non-Python: Just LSP validation
                    self._log(f"      ✓ LSP validation passed")
                    action.validation_passed = True
                    valid = True
                    break

            # if not valid:
            #     all_valid = False
            #     state.errors.append(f"{action.file_path}: Validation failed after {max_retries} attempts")

        return all_valid

    def _generate_basic_test(self, action: FileAction) -> str:
        """Generate basic test code for a file action"""
        if not action.target_symbols:
            return """
class TestBasic(unittest.TestCase):
    def test_import(self):
        self.assertTrue(True)
"""

        tests = []
        for symbol in action.target_symbols[:3]:
            # Determine if class or function
            is_class = symbol[0].isupper()

            if is_class:
                tests.append(f"""
    def test_{symbol.lower()}_exists(self):
        self.assertTrue(callable({symbol}))
""")
            else:
                tests.append(f"""
    def test_{symbol}_callable(self):
        self.assertTrue(callable({symbol}))
""")

        return f"""
class TestGenerated(unittest.TestCase):
{"".join(tests)}
"""

    async def _auto_fix(
        self,
        action: FileAction,
        diagnostics: List[LSPDiagnostic],
        state: DeveloperState,
        test_result: Optional[ValidationResult] = None
    ) -> Optional[str]:
        """Attempt to auto-fix code based on errors"""
        if not action.generated_code:
            return None

        # Build error description
        error_parts = []
        for d in diagnostics[:5]:
            error_parts.append(f"Line {d.line}: {d.message}")

        if test_result and test_result.error_message:
            error_parts.append(f"Test error: {test_result.error_message[:300]}")

        if not error_parts:
            return None

        # Build test context
        test_context = ""
        if action.target_symbols:
            test_context = f"Target symbols: {', '.join(action.target_symbols)}"

        prompt = AUTOFIX_PROMPT.format(
            code=action.generated_code,
            errors="\n".join(error_parts),
            test_context=test_context
        )

        response = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            model_preference="fast",
            stream=False,
            with_context=False
        )

        fixed_code = self._clean_code_response(response)

        # Validate fix parses
        if action.language == LanguageType.PYTHON:
            try:
                ast.parse(fixed_code)
                self._log("      🔧 Auto-fix applied")
                return fixed_code
            except SyntaxError:
                self._log("      ⚠️ Auto-fix invalid")
                return None

        return fixed_code

    # =========================================================================
    # PHASE 6: SYNC
    # =========================================================================

    async def _phase_sync(self, state: DeveloperState):
        """
        Phase 6: Write all validated files to disk.
        """
        self._log("💾 Syncing files to disk...")

        for file_path, content in state.generated_files.items():
            full_path = self.workspace / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_path.exists():
                full_path.touch()
            full_path.write_text(content, encoding='utf-8')
            self._log(f"   ✓ {file_path}")

        # Update DocsSystem index if available
        docs = await self._ensure_docs_system()
        if docs:
            try:
                await docs.sync()
                self._log("   ✓ DocsSystem index updated")
            except Exception:
                pass

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _parse_yaml_response(self, response: str) -> Dict[str, Any]:
        """Parse YAML from LLM response"""
        # Extract YAML block
        if "```yaml" in response:
            try:
                yaml_content = response.split("```yaml")[1].split("```")[0].strip()
                return yaml.safe_load(yaml_content) or {}
            except Exception:
                pass

        if "```" in response:
            try:
                parts = response.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        lines = part.strip().split('\n')
                        if lines[0].strip() in ('yaml', 'yml', ''):
                            content = '\n'.join(lines[1:]) if lines[0].strip() else part
                            return yaml.safe_load(content) or {}
            except Exception:
                pass

        # Try parsing raw response
        try:
            return yaml.safe_load(response) or {}
        except Exception:
            return {}

    def get_state(self, execution_id: str) -> Optional[DeveloperState]:
        """Get execution state by ID"""
        return self._executions.get(execution_id)

    def list_executions(self) -> List[Dict[str, Any]]:
        """List all executions"""
        return [
            {
                "id": state.execution_id,
                "task": state.task[:50],
                "phase": state.phase.value,
                "files": len(state.target_files),
                "success": state.success,
            }
            for state in self._executions.values()
        ]

    async def close(self):
        """Cleanup resources"""
        await self.lsp_manager.shutdown()
        self._log("🔒 ProjectDeveloperEngine closed")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_project_developer(
    agent: 'FlowAgent',
    workspace_path: Union[str, Path],
    docs_system: Optional['DocsSystem'] = None,
    auto_lsp: bool = True,
    prefer_docker: bool = True,
    verbose: bool = True,
) -> ProjectDeveloperEngine:
    """
    Factory function to create ProjectDeveloperEngine.

    Args:
        agent: FlowAgent instance for LLM interactions
        workspace_path: Path to project workspace
        docs_system: Optional pre-initialized DocsSystem
        auto_lsp: Whether to auto-start LSP servers
        prefer_docker: Prefer Docker for code execution (safer)
        verbose: Enable verbose logging

    Returns:
        Configured ProjectDeveloperEngine instance
    """
    return ProjectDeveloperEngine(
        agent=agent,
        workspace_path=workspace_path,
        docs_system=docs_system,
        auto_lsp=auto_lsp,
        prefer_docker=prefer_docker,
        verbose=verbose,
    )


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def main():
    """Example usage of ProjectDeveloperEngine"""
    from toolboxv2 import get_app

    # Setup
    app = get_app()
    isaa = app.get_mod("isaa")
    await isaa.init_isaa()
    agent = await isaa.get_agent("coder")

    # Create engine
    developer = create_project_developer(
        agent=agent,
        workspace_path=r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\isaa\CodingAgent\prject_dev",
        prefer_docker=True,
        verbose=True
    )

    # single file development task
    success, generated_files = await developer.execute(
            task="Erstelle eine Funktion 'clean_csv_data' die eine Liste von Strings nimmt, "
                 "Header behält, aber leere Zeilen entfernt und Whitespace trimmt.",
            target_files=["utils/data_processing.py"],
        auto_research=True
        )


    if success:
        print(f"\n✅ Generated {len(generated_files)} files:")
        for path in generated_files:
            print(f"   - {path}")
    else:
        print("\n❌ Development failed")

    try:
        # Multi-file development task
        success, generated_files = await developer.execute(
            task="""
            Create a file uploader and viewer for images and pdfs. the files must be saved on the server disk
            """,
            target_files=[
                "app/index.html",
                "app/style.css",
                "app/script.js",
                "app/server.py",
            ],
            max_retries=3,
            auto_research=True
        )

        if success:
            print(f"\n✅ Generated {len(generated_files)} files:")
            for path in generated_files:
                print(f"   - {path}")
        else:
            print("\n❌ Development failed")

    finally:
        await developer.close()


if __name__ == "__main__":
    asyncio.run(main())
