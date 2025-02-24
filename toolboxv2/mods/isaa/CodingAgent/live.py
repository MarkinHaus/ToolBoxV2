import fnmatch
import json
import pickle
import shutil
import sys
import tempfile
import textwrap
import time

import git
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

import toolboxv2
from toolboxv2 import Style, Spinner, get_app, get_logger
from toolboxv2.mods.isaa.CodingAgent.web import JSExecutionRecord, BrowserSession, Extractor, BrowserContext, js_prompt
from toolboxv2.mods.isaa.extras.modes import get_free_agent

from inspect import getdoc, signature, isfunction, ismethod, currentframe, Signature, isclass

from collections import Counter, defaultdict
from typing import Optional, Dict, Any, List, Union, Type, Tuple
from copy import deepcopy
import re

import importlib
import subprocess

from toolboxv2.mods.isaa.extras.session import ChatSession


### ---- Styles ------- ###

from enum import Enum, auto
from dataclasses import dataclass, asdict
import asyncio
import nest_asyncio
import ast
import io
import os
import traceback
from pathlib import Path
from typing import Tuple, Optional, Any, Union

class VerboseFormatter:
    def __init__(self,print_f, spinner_style: str = "d"):
        self.style = Style()
        self.current_spinner = None
        self.spinner_style = spinner_style
        self.print = print_f

    def print_header(self, text: str):
        """Print a formatted header with separator line"""
        width = 80
        self.print(f"\n{self.style.BLUE('=' * width)}")
        self.print(self.style.BLUE2(f"⚡ {text.center(width - 4)} ⚡"))
        self.print(f"{self.style.BLUE('=' * width)}\n")

    def print_section(self, title: str, content: str):
        """Print a formatted section with title and content"""
        self.print(f"{self.style.YELLOW('┌─')} {self.style.YELLOW2(title)}")
        for line in content.split('\n'):
            try:
                self.print(f"{self.style.YELLOW('│')} {line}")
            except Exception as e:
                try:
                    pos = int(str(e).split('position ')[1].split('-')[0])
                    line = line[:pos] + line[pos+1:]
                    self.print(f"{self.style.YELLOW('│')} {line}")
                except Exception as e:
                    self.print(f"{self.style.RED('│')} UNABLE TO PRINT {str(e)}")
        self.print(f"{self.style.YELLOW('└─')} {self.style.GREY('End of section')}\n")

    def print_iteration(self, current: int, maximum: int):
        """Print iteration progress with visual bar"""
        progress = int((current / maximum) * 20)
        bar = "█" * progress + "░" * (20 - progress)
        self.print(f"\r{self.style.CYAN(f'Iteration [{bar}] {current}/{maximum}')}  ", end='')

    def print_state(self, state: str, details: Optional[Dict[str, Any]] = None):
        """Print current state with optional details"""
        state_color = {
            'ACTION': self.style.GREEN2,
            'PROCESSING': self.style.YELLOW2,
            'BRAKE': self.style.RED2,
            'DONE': self.style.BLUE2
        }.get(state, self.style.WHITE2)

        self.print(f"\n{self.style.Bold(f'Current State:')} {state_color(state)}")

        if details:
            for key, value in details.items():
                self.print(f"  {self.style.GREY('├─')} {self.style.CYAN(key)}: {value}")

    def print_method_update(self, method_update: 'MethodUpdate'):
        """Print a formatted view of a MethodUpdate structure"""
        # Header with class and method name
        self.print(f"\n{self.style.BLUE('┏━')} {self.style.Bold('Method Update Details')}")

        # Class and method information
        self.print(f"{self.style.BLUE('┣━')} Class: {self.style.GREEN2(method_update.class_name)}")
        self.print(f"{self.style.BLUE('┣━')} Method: {self.style.YELLOW2(method_update.method_name)}")

        # Description if available
        if method_update.description:
            self.print(f"{self.style.BLUE('┣━')} Description:")
            for line in method_update.description.split('\n'):
                self.print(f"{self.style.BLUE('┃')}  {self.style.GREY(line)}")

        # Code section
        self.print(f"{self.style.BLUE('┣━')} Code:")
        code_lines = method_update.code.split('\n')
        for i, line in enumerate(code_lines):
            # Different styling for first and last lines
            if i == 0:
                self.print(f"{self.style.BLUE('┃')}  {self.style.CYAN('┌─')} {line}")
            elif i == len(code_lines) - 1:
                self.print(f"{self.style.BLUE('┃')}  {self.style.CYAN('└─')} {line}")
            else:
                self.print(f"{self.style.BLUE('┃')}  {self.style.CYAN('│')} {line}")

        # Footer
        self.print(f"{self.style.BLUE('┗━')} {self.style.GREY('End of method update')}\n")

    async def process_with_spinner(self, message: str, coroutine):
        """Execute a coroutine with a spinner indicator"""
        with Spinner(message, symbols=self.spinner_style) as spinner:
            result = await coroutine
            return result


class EnhancedVerboseOutput:
    def __init__(self, verbose: bool = True,print_f=None):
        self.verbose = verbose
        self.print = print_f or print
        self.formatter = VerboseFormatter(self.print)


    async def log_message(self, role: str, content: str):
        """Log chat messages with role-based formatting"""
        if not self.verbose:
            return

        role_formats = {
            'user': (self.formatter.style.GREEN, "👤"),
            'assistant': (self.formatter.style.BLUE, "🤖"),
            'system': (self.formatter.style.YELLOW, "⚙️")
        }

        color_func, icon = role_formats.get(role, (self.formatter.style.WHITE, "•"))
        self.print(f"\n{icon} {color_func(f'[{role}]')}")
        self.print(f"{self.formatter.style.GREY('└─')} {content}\n")

    async def log_think_result(self, result: Dict[str, Any]):
        """Log thinking results with structured formatting"""
        if not self.verbose:
            return

        self.formatter.print_section(
            "Action Result",
            f"Action: {result.get('action', 'N/A')}\n"
            f"context: {result.get('context', 'N/A')}\n"
            f"Content:\n{result.get('content', '')}"
        )

    async def log_process_result(self, result: Dict[str, Any]):
        """Log processing results with structured formatting"""
        if not self.verbose:
            return

        self.formatter.print_section(
            "Process Result",
            f"Completed: {result.get('is_completed', False)}\n"
            f"Effectiveness: {result.get('effectiveness', 'N/A')}\n"
            f"Recommendations: \n{result.get('recommendations', 'None')}\n"
            f"workflow: \n{result.get('workflow', 'None')}\n"
            f"errors: {result.get('errors', 'None')}\n"
            f"text: {result.get('text', 'None')}"
        )

    def log_header(self, text: str):
        """Log method update with structured formatting"""
        if not self.verbose:
            return

        self.formatter.print_header(text)

    def log_state(self, state: str, user_ns:dict, override=False):
        """Log method update with structured formatting"""
        if not self.verbose and not not override:
            return

        self.formatter.print_state(state, user_ns)

    async def process(self, message: str, coroutine):
        if not self.verbose:
            return await coroutine
        if message == "code":
            return await coroutine
        return await self.formatter.process_with_spinner(message, coroutine)

### -- TYPESs --- ###

class ThinkState(Enum):
    ACTION = auto()
    PROCESSING = auto()
    BRAKE = auto()
    DONE = auto()


class MethodUpdate(BaseModel):
    class_name: str = Field(..., description="Name of the class to update")
    method_name: str = Field(..., description="Name of the method to update")
    code: str = Field(..., description="Python code for the method implementation")
    description: Optional[str] = Field(None, description="Description of what the method does")


class ThinkResult(BaseModel):
    action: str = Field(..., description="Next action to take: 'code', 'update'', 'brake', 'done'")
    content: str = Field(..., description="Content related to the action")
    context: Optional[Dict[str, str | int | float | bool | Dict[str, str | int | float | bool ]]] = Field(default_factory=dict, description="Additional context for the action")

class ThinkResults(BaseModel):
    actions:List[ThinkResult]

@dataclass
class ExecutionRecord:
    code: str
    result: Any
    error: Optional[str] = None

    def __str__(self):
        return  '' if self.result is None and self.error is None else f"Output -> {self.result if self.result else ''}{'(error: '+self.error+')' if self.error else ''}"


@dataclass
class PipelineResult:
    variables: Dict[str, Any]
    result: Any
    execution_history: List[ExecutionRecord]
    message: List[Dict[str, str]]


import subprocess
import os
from pathlib import Path
import shutil
import json
import re
from typing import Dict, Any, Optional, Tuple
import io
from contextlib import redirect_stdout, redirect_stderr


class CargoRustInterface:
    '''Usage :
# Create interface
cargo_interface = CargoRustInterface()

# Set up new project
await cargo_interface.setup_project("hello_rust")

# Add a dependency
await cargo_interface.add_dependency("serde", "1.0")

# Write and run some code
code = """
fn main() {
    println!("Hello, Rust!");
}
"""
result = await cargo_interface.run_code(code)

# Modify code
new_function = """
fn main() {
    println!("Modified Hello, Rust!");
}
"""
await cargo_interface.modify_code(new_function, "main()")

# Build and test
await cargo_interface.build()
await cargo_interface.test()

    '''
    def __init__(self, session_dir=None, auto_remove=True):
        """Initialize the Rust/Cargo interface"""
        self.auto_remove = auto_remove
        self._session_dir = session_dir or Path.home() / '.cargo_sessions'
        self._session_dir.mkdir(exist_ok=True)
        self.vfs = VirtualFileSystem(self._session_dir / 'virtual_fs')
        self.output_history = {}
        self._execution_count = 0
        self.current_project = None

    def reset(self):
        """Reset the interface state"""
        if self.auto_remove and self.current_project:
            shutil.rmtree(self.current_project, ignore_errors=True)
        self.output_history.clear()
        self._execution_count = 0
        self.current_project = None

    async def setup_project(self, name: str) -> str:
        """Set up a new Cargo project"""
        try:
            project_path = self.vfs.base_dir / name
            if project_path.exists():
                shutil.rmtree(project_path)

            result = subprocess.run(
                ['cargo', 'new', str(project_path)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return f"Error creating project: {result.stderr}"

            self.current_project = project_path
            return f"Created new project at {project_path}"

        except Exception as e:
            return f"Failed to create project: {str(e)}"

    async def add_dependency(self, name: str, version: Optional[str] = None) -> str:
        """Add a dependency to Cargo.toml"""
        if not self.current_project:
            return "No active project"

        try:
            cargo_toml = self.current_project / "Cargo.toml"
            if not cargo_toml.exists():
                return "Cargo.toml not found"

            cmd = ['cargo', 'add', name]
            if version:
                cmd.extend(['--vers', version])

            result = subprocess.run(
                cmd,
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

        except Exception as e:
            return f"Failed to add dependency: {str(e)}"

    async def build(self, release: bool = False) -> str:
        """Build the project"""
        if not self.current_project:
            return "No active project"

        try:
            cmd = ['cargo', 'build']
            if release:
                cmd.append('--release')

            result = subprocess.run(
                cmd,
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            return result.stdout if result.returncode == 0 else f"Build error: {result.stderr}"

        except Exception as e:
            return f"Build failed: {str(e)}"

    async def test(self) -> str:
        """Run project tests"""
        if not self.current_project:
            return "No active project"

        try:
            result = subprocess.run(
                ['cargo', 'test'],
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            return result.stdout if result.returncode == 0 else f"Test error: {result.stderr}"

        except Exception as e:
            return f"Tests failed: {str(e)}"

    async def run_code(self, code: str) -> str:
        """Run Rust code"""
        if not self.current_project:
            return "No active project"

        try:
            # Write code to main.rs
            main_rs = self.current_project / "src" / "main.rs"
            with open(main_rs, 'w') as f:
                f.write(code)

            # Build and run
            build_result = subprocess.run(
                ['cargo', 'build'],
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            if build_result.returncode != 0:
                return f"Compilation error: {build_result.stderr}"

            run_result = subprocess.run(
                ['cargo', 'run'],
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            self._execution_count += 1
            output = {
                'code': code,
                'stdout': run_result.stdout,
                'stderr': run_result.stderr,
                'result': run_result.returncode == 0
            }
            self.output_history[self._execution_count] = output

            return run_result.stdout if run_result.returncode == 0 else f"Runtime error: {run_result.stderr}"

        except Exception as e:
            return f"Execution failed: {str(e)}"

    async def modify_code(self, code: str, object_name: str, file: str = "src/main.rs") -> str:
        """Modify existing Rust code"""
        if not self.current_project:
            return "No active project"

        try:
            file_path = self.current_project / file
            if not file_path.exists():
                return f"File {file} not found"

            with open(file_path, 'r') as f:
                content = f.read()

            # Handle function modification
            if object_name.endswith("()"):
                func_name = object_name[:-2]
                # Find and replace function definition
                pattern = f"fn {func_name}.*?}}(?=\n|$)"
                updated_content = re.sub(pattern, code.strip(), content, flags=re.DOTALL)
            else:
                # Handle other modifications (structs, constants, etc.)
                pattern = f"{object_name}.*?(?=\n|$)"
                updated_content = re.sub(pattern, code.strip(), content)

            with open(file_path, 'w') as f:
                f.write(updated_content)

            return f"Modified {object_name} in {file}"

        except Exception as e:
            return f"Modification failed: {str(e)}"

    def save_session(self, name: str):
        """Save current session state"""
        session_file = self._session_dir / f"{name}.json"
        state = {
            'output_history': self.output_history,
            'current_project': str(self.current_project) if self.current_project else None
        }

        with open(session_file, 'w') as f:
            json.dump(state, f)

    def load_session(self, name: str):
        """Load saved session state"""
        session_file = self._session_dir / f"{name}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                state = json.load(f)
                self.output_history = state['output_history']
                self.current_project = Path(state['current_project']) if state['current_project'] else None
### ---- logic ---- ###

class VirtualFileSystem:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.current_dir = base_dir
        self.virtual_files: Dict[str, str] = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_file(self, filepath: Union[str, Path], content: str) -> Path:
        """Write content to a virtual file and persist to disk using UTF-8"""
        abs_path = self._resolve_path(filepath)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Store in virtual filesystem
        rel_path = str(abs_path.relative_to(self.base_dir))
        self.virtual_files[rel_path] = content

        # Write to actual filesystem with UTF-8 encoding
        with open(abs_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(content)

        return abs_path

    def read_file(self, filepath: Union[str, Path]) -> str:
        """Read content from a virtual file using UTF-8"""
        abs_path = self._resolve_path(filepath)
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        rel_path = str(abs_path.relative_to(self.base_dir))

        # Check virtual filesystem first
        if rel_path in self.virtual_files:
            return self.virtual_files[rel_path]

        # Fall back to reading from disk with UTF-8 encoding
        with open(abs_path, encoding='utf-8', errors='replace') as f:
            content = f.read()
            self.virtual_files[rel_path] = content
            return content

    def delete_file(self, filepath: Union[str, Path]):
        """Delete a virtual file"""
        abs_path = self._resolve_path(filepath)
        rel_path = str(abs_path.relative_to(self.base_dir))

        if rel_path in self.virtual_files:
            del self.virtual_files[rel_path]

        if abs_path.exists():
            abs_path.unlink()

    def create_directory(self, dirpath: Union[str, Path]):
        """Create a new directory"""
        abs_path = self._resolve_path(dirpath)
        abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path


    def list_directory(self, dirpath: Union[str, Path] = '.') -> list:
        """List contents of a directory"""
        abs_path = self._resolve_path(dirpath)
        if not abs_path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        return [p.name for p in abs_path.iterdir()]

    def change_directory(self, dirpath: Union[str, Path]):
        """Change current working directory"""
        new_dir = self._resolve_path(dirpath)
        if not new_dir.exists() or not new_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {dirpath}")
        self.current_dir = new_dir

    def _resolve_path(self, filepath: Union[str, Path]) -> Path:
        """Convert relative path to absolute path"""
        filepath = Path(filepath)
        if filepath.is_absolute():
            if not str(filepath).startswith(str(self.base_dir)):
                raise ValueError("Path must be within base directory")
            return filepath
        return (self.current_dir / filepath).resolve()

    def save_state(self, state_file: Path):
        """Save virtual filesystem state to disk"""
        state = {
            'current_dir': str(self.current_dir.relative_to(self.base_dir)),
            'virtual_files': self.virtual_files
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

    def load_state(self, state_file: Path):
        """Load virtual filesystem state from disk"""
        if not state_file.exists():
            return

        with open(state_file) as f:
            state = json.load(f)
            self.current_dir = self.base_dir / state['current_dir']
            self.virtual_files = state['virtual_files']

    def print_file_structure(self, start_path: Union[str, Path] = '.', indent: str = ''):
        """Print the file structure starting from the given path"""
        start_path = self._resolve_path(start_path)
        if not start_path.exists():
            s = f"Path not found: {start_path}"
            return s

        s = f"{indent}{start_path.name}/"
        for item in sorted(start_path.iterdir()):
            if item.is_dir():
               s+= self.print_file_structure(item, indent + '  ')
            else:
                s = f"{indent}  {item.name}"
        return s


class EnhancedVirtualFileSystem:
    def __init__(self, base_path: Optional[str] = None):
        """
        Enhanced Virtual File System with Git integration

        Args:
            base_path (Optional[str]): Base directory for file operations
        """
        self.base_path = Path(base_path or os.getcwd()).absolute()
        self.repo = None
        self.logger = get_logger()
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """Ensure base path directory exists"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {self.base_path}")
        except Exception as e:
            self.logger.error(f"Failed to create directory: {e}")
            raise

    def init_git_repo(self, remote_url: Optional[str] = None) -> bool:
        """
        Initialize a git repository locally or clone from remote

        Args:
            remote_url (Optional[str]): Remote repository URL

        Returns:
            bool: True if repository is successfully initialized/cloned
        """
        try:
            if remote_url:
                # Clone remote repository
                self.repo = git.Repo.clone_from(remote_url, self.base_path)
                self.logger.info(f"Cloned repository from {remote_url}")
            else:
                # Initialize local repository
                self.repo = git.Repo.init(self.base_path)
                self.logger.info("Initialized local git repository")

            return True
        except Exception as e:
            self.logger.error(f"Git repository initialization failed: {e}")
            return False

    def commit_changes(self, message: str, force: bool = False) -> bool:
        """
        Commit changes to the repository

        Args:
            message (str): Commit message
            force (bool): Force commit even with unstaged changes

        Returns:
            bool: True if commit is successful
        """
        if not self.repo:
            self.logger.warning("Git repository not initialized")
            return False

        try:
            # Stage all changes
            self.repo.git.add(A=True)

            # Optional force commit
            if force:
                self.repo.git.add(update=True)

            # Commit changes
            self.repo.index.commit(message)
            self.logger.info(f"Changes committed: {message}")
            return True
        except Exception as e:
            self.logger.error(f"Commit failed: {e}")
            return False

    def pull_changes(self, remote: str = 'origin', branch: str = 'main') -> bool:
        """
        Pull changes from remote repository

        Args:
            remote (str): Remote name
            branch (str): Branch to pull

        Returns:
            bool: True if pull is successful
        """
        if not self.repo:
            self.logger.warning("Git repository not initialized")
            return False

        try:
            self.repo.remotes[remote].pull(branch)
            self.logger.info(f"Pulled changes from {remote}/{branch}")
            return True
        except Exception as e:
            self.logger.error(f"Pull failed: {e}")
            return False

    def push_changes(self, remote: str = 'origin', branch: str = 'main') -> bool:
        """
        Push changes to remote repository

        Args:
            remote (str): Remote name
            branch (str): Branch to push

        Returns:
            bool: True if push is successful
        """
        if not self.repo:
            self.logger.warning("Git repository not initialized")
            return False

        try:
            self.repo.remotes[remote].push(branch)
            self.logger.info(f"Pushed changes to {remote}/{branch}")
            return True
        except Exception as e:
            self.logger.error(f"Push failed: {e}")
            return False

    def get_repository_status(self) -> Dict[str, Any]:
        """
        Get comprehensive repository status

        Returns:
            Dict[str, Any]: Repository status details
        """
        if not self.repo:
            return {"error": "Repository not initialized"}

        status = {
            "is_dirty": self.repo.is_dirty(),
            "untracked_files": self.repo.untracked_files,
            "current_branch": self.repo.active_branch.name,
            "commits_ahead": len(list(self.repo.iter_commits())),
        }
        return status

    def create_snapshot(self, snapshot_name: str) -> bool:
        """
        Create a snapshot/checkpoint of current state

        Args:
            snapshot_name (str): Name of the snapshot

        Returns:
            bool: True if snapshot created successfully
        """
        try:
            self.commit_changes(f"Snapshot: {snapshot_name}", force=True)
            return True
        except Exception as e:
            self.logger.error(f"Snapshot creation failed: {e}")
            return False


from contextlib import redirect_stdout, redirect_stderr, contextmanager


class VirtualEnvContext:
    """Context manager for temporary virtual environment activation"""

    def __init__(self, venv_path: Path):
        self.venv_path = venv_path
        self._original_path = None
        self._original_sys_path = None
        self._original_prefix = None
        self._original_virtual_env = None

    def _get_venv_paths(self):
        """Get virtual environment paths based on platform"""
        if sys.platform == 'win32':
            site_packages = self.venv_path / 'Lib' / 'site-packages'
            scripts_dir = self.venv_path / 'Scripts'
            python_path = scripts_dir / 'python.exe'
        else:
            python_version = f'python{sys.version_info.major}.{sys.version_info.minor}'
            site_packages = self.venv_path / 'lib' / python_version / 'site-packages'
            scripts_dir = self.venv_path / 'bin'
            python_path = scripts_dir / 'python'

        return site_packages, scripts_dir, python_path

    def __enter__(self):
        # Save original state
        self._original_path = os.environ.get('PATH', '')
        self._original_sys_path = sys.path.copy()
        self._original_prefix = sys.prefix
        self._original_virtual_env = os.environ.get('VIRTUAL_ENV')

        # Get venv paths
        site_packages, scripts_dir, python_path = self._get_venv_paths()

        # Modify environment for venv
        if scripts_dir.exists():
            new_path = os.pathsep.join([str(scripts_dir), self._original_path])
            os.environ['PATH'] = new_path

        if site_packages.exists():
            sys.path.insert(0, str(site_packages))

        os.environ['VIRTUAL_ENV'] = str(self.venv_path)

        # Return the python executable path for potential subprocess calls
        return str(python_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        os.environ['PATH'] = self._original_path
        sys.path = self._original_sys_path

        if self._original_virtual_env is None:
            os.environ.pop('VIRTUAL_ENV', None)
        else:
            os.environ['VIRTUAL_ENV'] = self._original_virtual_env

class TeeStream:
    """Stream that writes to both console and buffer"""
    def __init__(self, console_stream, buffer_stream):
        self.console_stream = console_stream
        self.buffer_stream = buffer_stream

    def write(self, data):
        self.console_stream.write(data)
        self.buffer_stream.write(data)
        self.console_stream.flush()  # Ensure immediate console output

    def flush(self):
        self.console_stream.flush()
        self.buffer_stream.flush()


class ParentNodeTransformer(ast.NodeTransformer):
    """Add parent references to AST nodes"""
    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
        return super().visit(node)

class AsyncCodeDetector(ast.NodeVisitor):
    """Detect async code and top-level await"""
    def __init__(self):
        self.has_async = False
        self.has_top_level_await = False
        self.await_nodes = []

    def visit_AsyncFunctionDef(self, node):
        self.has_async = True
        self.generic_visit(node)

    def visit_Await(self, node):
        self.has_async = True
        # Track all await nodes
        self.await_nodes.append(node)
        # Check if this await is at top level
        parent = node
        while hasattr(parent, 'parent'):
            parent = parent.parent
            if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
                break
        else:
            self.has_top_level_await = True
        self.generic_visit(node)

def auto_install(package_name, install_method='pip', upgrade=False, quiet=False, version=None, extra_args=None):
    '''
    Enhanced auto-save import with version and extra arguments support
    '''
    try:
        # Attempt to import the package
        return importlib.import_module(package_name)
    except ImportError:
        # Package not found, prepare for installation
        print(f"Package '{package_name}' not found. Attempting to install...")
        try:
            # Determine Python executable based on virtual environment
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                venv_path = Path(venv_path)
                if sys.platform == 'win32':
                    python_exec = str(venv_path / 'Scripts' / 'python.exe')
                else:
                    python_exec = str(venv_path / 'bin' / 'python')
                # Check if the Python executable exists
                if not Path(python_exec).exists():
                    python_exec = sys.executable
            else:
                python_exec = sys.executable

            # Construct installation command with more flexibility
            install_cmd = [python_exec, "-m", install_method, "install"]
            if upgrade:
                install_cmd.append("--upgrade")
            # Support specific version installation
            if version:
                install_cmd.append(f"{package_name}=={version}")
            else:
                install_cmd.append(package_name)
            # Add extra arguments if provided
            if extra_args:
                install_cmd.extend(extra_args)
            # Run installation with appropriate verbosity
            installation_output = subprocess.run(
                install_cmd,
                capture_output=quiet,
                text=True
            )
            # Check installation status
            if installation_output.returncode == 0:
                print(f"Successfully installed {package_name}")
                return importlib.import_module(package_name)
            else:
                raise Exception(f"Installation failed: {installation_output.stderr}")
        except Exception as install_error:
            print(f"Error installing {package_name}: {install_error}")
            return None

class MockIPython:
    def __init__(self, _session_dir=None, auto_remove=True):
        self.auto_remove = auto_remove
        self.output_history = {}
        self._execution_count = 0
        self._session_dir = _session_dir or Path(get_app().appdata) / '.pipeline_sessions'
        self._session_dir.mkdir(exist_ok=True)
        self.vfs = VirtualFileSystem(self._session_dir / 'virtual_fs')
        self._venv_path = self._session_dir / 'venv'
        self.user_ns: Dict[str, Any] = {}
        nest_asyncio.apply()
        # Set up virtual environment if it doesn't exist
        with Spinner("Starting virtual environment"):
            self._setup_venv()
        self.reset()

    def _setup_venv(self):
        """Create virtual environment if it doesn't exist"""
        if not self._venv_path.exists():
            try:
                import venv
                builder = venv.EnvBuilder(with_pip=True)
                builder.create(self._venv_path)
            except Exception as e:
                raise RuntimeError(f"Failed to create virtual environment: {str(e)}")

    def _virtual_open(self, filepath, mode='r', *args, **kwargs):
        """Custom open function that uses virtual filesystem"""
        abs_path = self.vfs._resolve_path(filepath)

        if 'w' in mode or 'a' in mode:
            # Ensure parent directory exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Use actual filesystem but track in virtual fs
        real_file = open(abs_path, mode, *args, **kwargs)

        if 'r' in mode:
            # Track file content in virtual filesystem when reading
            rel_path = str(abs_path.relative_to(self.vfs.base_dir))
            if rel_path not in self.vfs.virtual_files:
                try:
                    self.vfs.virtual_files[rel_path] = real_file.read()
                    real_file.seek(0)
                except UnicodeDecodeError:
                    # Handle binary files
                    pass

        return real_file

    def reset(self):
        """Reset the interpreter state"""
        self.user_ns = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
            'toolboxv2': toolboxv2,
            # 'git': EnhancedVirtualFileSystem(str(self.vfs.base_dir / 'src')),
            # 'git_files': VirtualFileSystem(self.vfs.base_dir / 'src'),
            '__file__': None,
            '__path__': [str(self.vfs.current_dir)],
            'auto_install': auto_install,
            'modify_code': self.modify_code,
        }
        self.output_history.clear()
        self._execution_count = 0
        if self.auto_remove:
            shutil.rmtree(self.vfs.base_dir, ignore_errors=True)

    def get_namespace(self) -> Dict[str, Any]:
        """Get current namespace"""
        return self.user_ns.copy()

    def update_namespace(self, variables: Dict[str, Any]):
        """Update namespace with new variables"""
        self.user_ns.update(variables)

    @staticmethod
    def _parse_code(code: str) -> Tuple[Any, Optional[Any], bool, bool]:
        """Parse code and handle top-level await"""
        try:
            tree = ast.parse(code)
            # Add parent references
            ParentNodeTransformer().visit(tree)

            # Detect async features
            detector = AsyncCodeDetector()
            detector.visit(tree)

            if detector.has_top_level_await:
                # Wrap code in async function
                wrapped_code = "async def __wrapper():\n"
                wrapped_code += "    global result\n"  # Allow writing to global scope
                # add try:
                wrapped_code +="    try:"
                # Indent the original code
                wrapped_code += "\n".join(f"        {line}" for line in code.splitlines())
                # Add return statement for last expression
                wrapped_code +="    except Exception as e:"
                wrapped_code +="        import traceback"
                wrapped_code +="        print(traceback.format_exc())"
                wrapped_code +="        raise e"
                if isinstance(tree.body[-1], ast.Expr):
                    wrapped_code += "\n    return result"

                # Parse and compile wrapped code
                wrapped_tree = ast.parse(wrapped_code)
                return (
                    compile(wrapped_tree, '<exec>', 'exec'),
                    None,
                    True,
                    True
                )

            # Handle regular code
            if isinstance(tree.body[-1], ast.Expr):
                exec_code = ast.Module(
                    body=tree.body[:-1],
                    type_ignores=[]
                )
                eval_code = ast.Expression(
                    body=tree.body[-1].value
                )
                return (
                    compile(exec_code, '<exec>', 'exec'),
                    compile(eval_code, '<eval>', 'eval'),
                    detector.has_async,
                    False
                )

            return (
                compile(tree, '<exec>', 'exec'),
                None,
                detector.has_async,
                False
            )

        except SyntaxError as e:
            lines = code.splitlines()
            if e.lineno and e.lineno <= len(lines):
                line = lines[e.lineno - 1]
                arrow = ' ' * (e.offset - 1) + '^' if e.offset else ''
                error_msg = (
                    f"Syntax error at line {e.lineno}:\n"
                    f"{line}\n"
                    f"{arrow}\n"
                    f"{e.msg}"
                )
            else:
                error_msg = str(e)
            raise SyntaxError(error_msg) from e

    async def run_cell(self, code: str, live_output: bool = False) -> Any:
        """Async version of run_cell that handles both sync and async code"""
        result = None
        error = None
        tb = None
        original_dir = os.getcwd()

        if live_output:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            stdout = TeeStream(sys.__stdout__, stdout_buffer)
            stderr = TeeStream(sys.__stderr__, stderr_buffer)
        else:
            stdout = io.StringIO()
            stderr = io.StringIO()

        try:
            # Check if a file is already specified
            original_file = self.user_ns.get('__file__')
            if original_file is None:
                # Create temp file if no file specified
                temp_file = self.vfs.write_file(
                    f'src/temp/_temp_{self._execution_count}.py',
                    code
                )
                # work_ns = self.user_ns.copy()
                self.user_ns['__file__'] = str(temp_file)
            else:
                # Use existing file
                temp_file = Path(original_file)
                # Write code to the existing file
                self.vfs.write_file(temp_file, code)
                #work_ns = self.user_ns.copy()

            self.user_ns['__builtins__'] = __builtins__
            with VirtualEnvContext(self._venv_path) as python_exec:
                try:
                    exec_code, eval_code, is_async, has_top_level_await = self._parse_code(
                        code.encode('utf-8', errors='replace').decode('utf-8')
                    )
                    if exec_code is None:
                        return "No executable code"

                    os.chdir(str(temp_file.parent))
                    self.user_ns['PYTHON_EXEC'] = python_exec

                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        if has_top_level_await:
                            try:
                                # Execute wrapped code and await it
                                exec(exec_code, self.user_ns)
                                result = await self.user_ns['__wrapper']()
                            finally:
                                self.user_ns.pop('__wrapper', None)
                        elif is_async:
                            # Execute async code
                            exec(exec_code, self.user_ns)
                            if eval_code:
                                result = await eval(eval_code, self.user_ns)
                        else:
                            # Execute sync code
                            exec(exec_code, self.user_ns)
                            if eval_code:
                                result = eval(eval_code, self.user_ns)

                        if result is not None:
                            self.user_ns['_'] = result

                except Exception as e:
                    error = str(e)
                    tb = traceback.format_exc()
                    if live_output:
                        sys.__stderr__.write(f"{error}\n{tb}")
                    stderr.write(f"{error}\n{tb}")

                finally:
                    os.chdir(original_dir)
                    self._execution_count += 1
                    # self.user_ns = work_ns.copy()
                    if live_output:
                        stdout_value = stdout_buffer.getvalue()
                        stderr_value = stderr_buffer.getvalue()
                    else:
                        stdout_value = stdout.getvalue()
                        stderr_value = stderr.getvalue()

                    output = {
                        'code': code,
                        'stdout': stdout_value,
                        'stderr': stderr_value,
                        'result': result if result else "stdout"
                    }
                    self.output_history[self._execution_count] = output

                    if not result:
                        result = ""
                    if output['stdout']:
                        result = f"{result}\nstdout:{output['stdout']}"
                    if output['stderr']:
                        result = f"{result}\nstderr:{output['stderr']}"

                    if self.auto_remove and original_file is None:
                        # Only remove temp files, not user-specified files
                        self.vfs.delete_file(temp_file)

                    return result

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            if live_output:
                sys.__stderr__.write(error_msg)
            return error_msg

    async def modify_code(self, code: str = None, object_name: str = None, file: str = None) -> str:
        '''
        Modify existing code in memory (user namespace) and optionally in the corresponding file.

        This method updates variables, functions, or methods in the current Python session and can
        also update the corresponding source file if specified.

        Args:
            code: New value or implementation for the object
            object_name: Name of the object to modify (variable, function, or method)
            file: Path to the file to update (if None, only updates in memory)

        Returns:
            String describing the modification result

        Examples:

        # 1. Update a variable in memory
        await ipython.modify_code(code="5", object_name="x")

    # 2. Change a method implementation
    await ipython.modify_code(
        code='"""def sound(self):\n        return "Woof""""',
        object_name="Dog.sound"
    )

    # 3. Modify a function
    await ipython.modify_code(
        code='"""def calculate_age():\n    return 25"""',
        object_name="calculate_age"
    )

    # 4. Update variable in memory and file
    await ipython.modify_code(
        code="100",
        object_name="MAX_SIZE",
        file="config.py"
    )

    # 5. Modifying an attribute in __init__
    await ipython.modify_code(
        code='"""def __init__(self):\n        self.name = "Buddy""""',
        object_name="Dog.__init__"
    )
        '''
        try:
            if not object_name:
                raise ValueError("Object name must be specified")
            if code is None:
                raise ValueError("New code or value must be provided")

            # Process object name (handle methods with parentheses)
            clean_object_name = object_name.replace("()", "")

            # Step 1: Update in memory (user namespace)
            result_message = []

            # Handle different types of objects
            if "." in clean_object_name:
                # For methods or class attributes
                parts = clean_object_name.split(".")
                base_obj_name = parts[0]
                attr_name = parts[1]

                if base_obj_name not in self.user_ns:
                    raise ValueError(f"Object '{base_obj_name}' not found in namespace")

                base_obj = self.user_ns[base_obj_name]

                # Handle method definitions which are passed as docstrings
                if code.split('\n'):
                    method_code = code

                    # Parse the method code to extract its body
                    method_ast = ast.parse(method_code).body[0]
                    method_name = method_ast.name

                    # Create a new function object from the code
                    method_locals = {}
                    exec(
                        f"def _temp_func{signature(getattr(base_obj.__class__, attr_name, None))}: {method_ast.body[0].value.s}",
                        globals(), method_locals)
                    new_method = method_locals['_temp_func']

                    # Set the method on the class
                    setattr(base_obj.__class__, attr_name, new_method)
                    result_message.append(f"Updated method '{clean_object_name}' in memory")
                else:
                    # For simple attributes
                    setattr(base_obj, attr_name, eval(code, self.user_ns))
                    result_message.append(f"Updated attribute '{clean_object_name}' in memory")
            else:
                # For variables and functions
                if code.startswith('"""') and code.endswith('"""'):
                    # Handle function definitions
                    func_code = code.strip('"""')
                    func_ast = ast.parse(func_code).body[0]
                    func_name = func_ast.name

                    # Create a new function object from the code
                    func_locals = {}
                    exec(f"{func_code}", globals(), func_locals)
                    self.user_ns[clean_object_name] = func_locals[func_name]
                    result_message.append(f"Updated function '{clean_object_name}' in memory")
                else:
                    # Simple variable assignment
                    self.user_ns[clean_object_name] = eval(code, self.user_ns)
                    result_message.append(f"Updated variable '{clean_object_name}' in memory")

            # Step 2: Update in file if specified
            if file is not None:
                file_path = self.vfs._resolve_path(file)

                if not file_path.exists():
                    self.user_ns['__file__'] = str(file_path)
                    return await self.run_cell(code)

                # Read original content
                original_content = self.vfs.read_file(file_path)
                updated_content = original_content

                # Handle different object types for file updates
                if "." in clean_object_name:
                    # For methods
                    parts = clean_object_name.split(".")
                    class_name = parts[0]
                    method_name = parts[1]

                    if code.startswith('"""') and code.endswith('"""'):
                        method_code = code.strip('"""')

                        # Use ast to parse the file and find the method to replace
                        file_ast = ast.parse(original_content)
                        for node in ast.walk(file_ast):
                            if isinstance(node, ast.ClassDef) and node.name == class_name:
                                for method in node.body:
                                    if isinstance(method, ast.FunctionDef) and method.name == method_name:
                                        # Find the method in the source code
                                        method_pattern = f"def {method_name}.*?:(.*?)(?=\n    \w|\n\w|\Z)"
                                        method_match = re.search(method_pattern, original_content, re.DOTALL)

                                        if method_match:
                                            indentation = re.match(r"^(\s*)", method_match.group(0)).group(1)
                                            method_indented = textwrap.indent(method_code, indentation)
                                            updated_content = original_content.replace(
                                                method_match.group(0),
                                                method_indented
                                            )
                                            self.vfs.write_file(file_path, updated_content)
                                            result_message.append(
                                                f"Updated method '{clean_object_name}' in file '{file}'")
                else:
                    # For variables and functions
                    if code.startswith('"""') and code.endswith('"""'):
                        # Handle function updates
                        func_code = code.strip('"""')
                        func_pattern = f"def {clean_object_name}.*?:(.*?)(?=\n\w|\Z)"
                        func_match = re.search(func_pattern, original_content, re.DOTALL)

                        if func_match:
                            indentation = re.match(r"^(\s*)", func_match.group(0)).group(1)
                            func_indented = textwrap.indent(func_code, indentation)
                            updated_content = original_content.replace(
                                func_match.group(0),
                                func_indented
                            )
                            self.vfs.write_file(file_path, updated_content)
                            result_message.append(f"Updated function '{clean_object_name}' in file '{file}'")
                    else:
                        # Handle variable updates
                        var_pattern = f"{clean_object_name}\s*=.*"
                        var_replacement = f"{clean_object_name} = {code}"
                        updated_content = re.sub(var_pattern, var_replacement, original_content)

                        if updated_content != original_content:
                            self.vfs.write_file(file_path, updated_content)
                            result_message.append(f"Updated variable '{clean_object_name}' in file '{file}'")
                        else:
                            result_message.append(f"Could not find variable '{clean_object_name}' in file '{file}'")

            return "\n".join(result_message)

        except Exception as e:
            return f"Error during code modification: {str(e)}\n{traceback.format_exc()}"


    def save_session(self, name: str):
        """Save session with UTF-8 encoding"""
        session_file = self._session_dir / f"{name}.pkl"
        user_ns = self.user_ns.copy()
        output_history = self.output_history.copy()

        # Ensure all strings are properly encoded
        for key, value in user_ns.items():
            try:
                if isinstance(value, str):
                    value = value.encode('utf-8').decode('utf-8')
                pickle.dumps(value)
            except Exception as e:
                user_ns[key] = f"not serializable: {str(value)}"

        for key, value in output_history.items():
            try:
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str):
                            value[k] = v.encode('utf-8').decode('utf-8')
                pickle.dumps(value)
            except Exception as e:
                output_history[key] = f"not serializable: {str(value)}"


        session_data = {
            'user_ns': user_ns,
            'output_history': output_history,

        }

        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)

        # Save VFS state with UTF-8 encoding
        vfs_state_file = self._session_dir / f"{name}_vfs.json"
        with open(vfs_state_file, 'w', encoding='utf-8') as f:
            json.dump(self.vfs.virtual_files, f, ensure_ascii=False)

    def load_session(self, name: str):
        """Load session with UTF-8 encoding"""
        session_file = self._session_dir / f"{name}.pkl"
        if session_file.exists():
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
                self.user_ns.update(session_data['user_ns'])
                self.output_history.update(session_data['output_history'])

        # Load VFS state with UTF-8 encoding
        vfs_state_file = self._session_dir / f"{name}_vfs.json"
        if vfs_state_file.exists():
            with open(vfs_state_file, 'r', encoding='utf-8') as f:
                self.vfs.virtual_files = json.load(f)

    def __str__(self):
        """String representation of current session"""
        output = []
        for count, data in self.output_history.items():
            output.append(f"In [{count}]: {data['code']}")
            if data['stdout']:
                output.append(data['stdout'])
            if data['stderr']:
                output.append(f"Error: {data['stderr']}")
            if data['result'] is not None:
                output.append(f"Out[{count}]: {data['result']}")
        return "\n".join(output)


def super_strip(s: str) -> str:
    # Remove ANSI escape sequences (e.g. "\x1b[K", "\x1b[...m", etc.)
    s = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s)

    # Remove any header text before the first "Episode"
    episode_index = s.find("Episode")
    if episode_index != -1:
        s = s[episode_index:]

    # Split the string into lines (split on newline characters)
    lines = s.splitlines()
    processed_lines = []
    for line in lines:
        # If the line contains carriage returns,
        # only keep the text after the last one.
        if "\r" in line:
            line = line.split("\r")[-1]
        processed_lines.append(line)

    # Rejoin the processed lines with newline characters.
    return "\n".join(processed_lines)

async def default_python_execute_function(files):
    # Create a temporary directory to store the files
    temp_dir = Path("./temp_project")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write files to the temporary directory
        for file_path, content in files.items():
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Check if main.py exists
        main_file = temp_dir / "main.py"
        if main_file.exists():
            # Run main.py
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(main_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return f"Main execution result:\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"

        # If main.py doesn't exist, look for files with __main__ block
        main_files = []
        for file_path in temp_dir.glob("**/*.py"):
            if "__main__" in file_path.read_text():
                main_files.append(file_path)

        if main_files:
            results = []
            for file in main_files:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                results.append(f"Execution of {file.name}:\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}")
            return "\n\n".join(results)

        # If no main files found, run pytest
        pytest_output = subprocess.run(
            [sys.executable, "-m", "pytest", str(temp_dir)],
            capture_output=True,
            text=True
        )
        return f"Pytest execution result:\n{pytest_output.stdout}\n{pytest_output.stderr}"

    finally:
        # Clean up temporary directory
        for file in temp_dir.glob("**/*"):
            if file.is_file():
                file.unlink()
        for dir in reversed(list(temp_dir.glob("**/*"))):
            if dir.is_dir():
                dir.rmdir()
        temp_dir.rmdir()


async def default_rust_execute_function(files):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write files to the temporary directory
        for file_path, content in files.items():
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

        # Check if there's a Cargo.toml file
        if "Cargo.toml" in files:
            # Run cargo check for syntax and compiler errors
            process = await asyncio.create_subprocess_exec(
                "cargo", "check",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            check_stdout, check_stderr = await process.communicate()

            # Run cargo run for execution
            process = await asyncio.create_subprocess_exec(
                "cargo", "run",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            run_stdout, run_stderr = await process.communicate()

            return f"""Rust project execution result:

Cargo check (syntax and compiler hints):
{check_stdout.decode()}
{check_stderr.decode()}

Cargo run (execution result):
{run_stdout.decode()}
{run_stderr.decode()}
"""
        else:
            # Assume it's a single file project
            main_file = next((f for f in files.keys() if f.endswith('.rs')), None)
            if main_file:
                file_path = os.path.join(temp_dir, main_file)

                # Run rustc for compilation and syntax check
                process = await asyncio.create_subprocess_exec(
                    "rustc", file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                compile_stdout, compile_stderr = await process.communicate()

                if process.returncode == 0:
                    # Run the compiled executable
                    executable = os.path.join(temp_dir, os.path.splitext(main_file)[0])
                    process = await asyncio.create_subprocess_exec(
                        executable,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    run_stdout, run_stderr = await process.communicate()

                    return f"""Rust file execution result:

Compilation:
{compile_stdout.decode()}
{compile_stderr.decode()}

Execution:
{run_stdout.decode()}
{run_stderr.decode()}
"""
                else:
                    return f"""Rust file compilation failed:

{compile_stdout.decode()}
{compile_stderr.decode()}
"""
            else:
                return "No Rust files found in the project."

# Usage in run_project function:
# execute_function = default_rust_execute_function

class Pipeline:
    """
        A pipeline for executing AI agent-driven tasks with interactive code execution and variable management.

        The Pipeline class provides a structured environment for AI agents to:
        1. Execute code in a controlled environment
        2. Manage and track variables
        3. Update methods dynamically
        4. Save and load session states
        5. Generate detailed variable descriptions

        Attributes:
            agent: The AI agent instance used for task execution
            task (str): The task to be performed
            mas_iter (int): Maximum number of iterations allowed (default: 12)
            variables (Dict[str, Any]): Dictionary of variables available to the pipeline
            top_n (Optional[int]): Limit variable descriptions to top N most used
            execution_history (List[ExecutionRecord]): History of executed code and results
            session_name (Optional[str]): Name of the current session if saved
            ipython: IPython or MockIPython instance for code execution

        Example:
            >>> agent = get_free_agent("demo", "anthropic/claude-3-haiku-20240307")
            >>> pipeline = Pipeline(
            ...     agent=agent,
            ...     task="Calculate fibonacci sequence",
            ...     variables={"n": 10}
            ... )
            >>> result = pipeline.run()
            >>> print(result.result)

        Notes:
            - The pipeline uses either IPython if available or a MockIPython implementation
            - Variables can be provided as either a dictionary or list
            - Session state can be saved and loaded
            - Method updates are handled through a structured BaseModel approach
        """
    def __init__(
        self,
        agent: Any,
        verbose: bool=False,
        max_iter: int= 12,
        variables: Optional[Union[Dict[str, Any], List[Any]]] = None,
        top_n: Optional[bool] = None,
        restore: Optional[bool] = None,
        max_think_after_think = None,
        print_f=None,
        web_js=False,
        timeout_timer=25,
        v_agent=None,
    ):
        """
        Initialize the Pipeline.

        Args:
            agent: AI agent instance to use for task execution
            verbose: print internal results
            max_iter: Maximum number of iterations (default: 12)
            variables: Dictionary or list of variables to make available
            top_n: Limit variable descriptions to top N most used
            web_js: if the agent is allow to use the web
        """

        self.timeout_timer = timeout_timer
        self.top_n = top_n
        self.max_iter = max_iter
        self.max_think_after_think = max_think_after_think or max_iter // 2
        self.agent = agent
        self.v_agent = v_agent or agent
        self.agent.verbose = verbose
        self.task = None
        self.web_js = web_js
        self.verbose_output = EnhancedVerboseOutput(verbose=verbose, print_f=print_f)
        self.variables = self._process_variables(variables or {})
        self.variables['auto_install'] = auto_install
        self.execution_history = []
        self.session_name = None

        self.browser_session: Optional[BrowserSession] = BrowserSession(verbose)
        self.js_history: List[JSExecutionRecord] = []

        self._session_dir = Path(get_app().appdata) / 'ChatSession' / agent.amd.name
        self.ipython = MockIPython(self._session_dir, auto_remove=False)
        self.chat_session = ChatSession(agent.memory, space_name=f"ChatSession/{agent.amd.name}/Pipeline.session", max_length=max_iter)
        self.process_memory = ChatSession(agent.memory, space_name=f"ChatSession/{agent.amd.name}/Process.session", max_length=max_iter)

        # Initialize interpreter with variables
        self.init_keys = list(self.ipython.user_ns.keys()).copy()
        self.ipython.user_ns.update(self.variables)

        self.restore_var = restore

        if restore:
            self.restore()

    async def start_web(self):
        """Initialize pipeline including browser session"""
        await self.browser_session.initialize()
        # Set default timeout
        self.browser_session.page.set_default_timeout(30000)  # 30 seconds
        # Enable JavaScript
        await self.browser_session.context.add_init_script("""
            window.onError = function(message, source, lineno, colno, error) {
                console.error('JavaScript Error:', error);
                return false;
            };
        """)

    def on_exit(self):
        self.chat_session.on_exit()
        self.process_memory.on_exit()
        self.save_session(f"Pipeline_Session_{self.agent.amd.name}")

    def restore(self):
        self.load_session(f"Pipeline_Session_{self.agent.amd.name}")

    def save_session(self, name: str):
        """Save current session"""
        self.session_name = name
        self.ipython.save_session(name)

    async def save_web(self, name:str="main"):
        context = await self.browser_session.get_context_state()
        self.ipython.vfs.write_file(f"browser/context_{name}.json", json.dumps(asdict(context)))

    async def restore_web(self, name:str="main"):
        context = self.ipython.vfs.read_file(f"browser/context_{name}.json")
        await self.browser_session.restore_context(BrowserContext(**json.loads(context)))

    def load_session(self, name: str):
        """Load saved session"""
        self.ipython.load_session(name)
        self.variables.update(self.ipython.user_ns)


    def show_graph_html(self, output_file=None, get_output_html=False, get_output_net=False):

        if output_file is None:
            chat_graph = self.ipython._session_dir / 'chat_graph.html'
            process_graph = self.ipython._session_dir / 'process_graph.html'
            output_file = str(chat_graph.absolute())
            p_output_file = str(process_graph.absolute())
        else:
            output_file = output_file + '_chat_graph.html'
            p_output_file = output_file + '_process_graph.html'

        return (self.chat_session.mem.memories.get(
            self.chat_session.mem._sanitize_name(
                self.chat_session.space_name)).vis(output_file=output_file,
        get_output_html=get_output_html, get_output_net=get_output_net)  ,
                self.process_memory.mem.memories.get(
            self.process_memory.mem._sanitize_name(
                self.process_memory.space_name)).vis(output_file=p_output_file,
        get_output_html=get_output_html, get_output_net=get_output_net))

    @staticmethod
    def _process_variables(variables: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        """
        Process variables to generate meaningful names, using actual variable names where possible.
        Instances get lowercase names based on their class names.

        Args:
            variables: Dictionary of variables or list of variables to process

        Returns:
            Dict[str, Any]: Processed variables with meaningful names
        """
        if isinstance(variables, dict):
            return variables

        processed = {}
        name_counts = defaultdict(int)

        # Get caller's frame to find variable names
        caller_frame = currentframe().f_back
        caller_locals = {**caller_frame.f_locals, **caller_frame.f_globals}

        def find_var_name(obj: Any) -> str:
            # Find original variable name if exists
            var_names = [name for name, val in caller_locals.items()
                         if val is obj and not name.startswith('_')]
            if var_names:
                return var_names[0]

            # Special handling for functions
            if isfunction(obj) or isclass(obj):
                return obj.__name__
            # Handle instances
            elif hasattr(obj, '__class__'):
                base_name = obj.__class__.__name__.lower()  # Lowercase for instances
                count = name_counts[base_name]
                name_counts[base_name] += 1
                return f"{base_name}_{count + 1}" if count > 0 else base_name

            return type(obj).__name__

        # Process each variable
        for var in variables:
            name = find_var_name(var)
            while name in processed:
                if name.rpartition('_')[0]:
                    base, _, num = name.rpartition('_')
                    try:
                        num = int(num) + 1
                        name = f"{base}_{num}"
                    except ValueError:
                        name = f"{name}"
                else:
                    name = f"{name}"

            processed[name] = var

        return processed

    def _generate_variable_descriptions(
        self,
        top_n: Optional[int] = None
    ) -> str:
        """
        Generate detailed descriptions of variables, showing args, kwargs, docstrings, and return values.

        Args:
            top_n: Optional limit to show only top N variables

        Returns:
            str: Formatted variable descriptions in Markdown
        """
        if top_n is None:
            top_n = self.top_n

        def format_value_preview(var: Any) -> str:
            """Format preview of variable contents"""
            try:
                if isinstance(var, (int, float, bool, str)):
                    return f"`{repr(var)}`"
                elif isinstance(var, (list, tuple, set)):
                    preview = str(list(var)[:3])[:-1] + ", ...]"
                    return f"{len(var)} items: {preview}"
                elif isinstance(var, dict):
                    preview_items = [f"{repr(k)}: {repr(v)}" for k, v in list(var.items())[:3]]
                    return f"{len(var)} pairs: {{{', '.join(preview_items)}, ...}}"
                return f"<{type(var).__name__}>"
            except:
                return "<error getting value>"

        def get_instance_state(var: Any) -> Dict[str, Any]:
            """Get current instance state"""
            state = {}
            if hasattr(var, '__dict__'):
                for name, value in var.__dict__.items():
                    if not name.startswith('_') and not callable(value):
                        state[name] = format_value_preview(value)
            return state

        # Process variables
        variables = self.variables.items()
        if top_n:
            variables = list(variables)[:top_n]

        descriptions = []
        for name, var in variables:
            if name in ["PYTHON_EXEC", "__name__", "__builtins__", "__path__", "asyncio"]:
                continue

            desc_parts = [f"### {name}"]

            # Handle different types
            if isinstance(var, type):  # Class
                desc_parts.append(f"**Type:** `class '{var.__name__}'`")
                if var.__doc__:
                    desc_parts.append(f"**Documentation:**\n{var.__doc__.strip()}")

                # Show methods
                methods = []
                for attr_name, attr in var.__dict__.items():
                    if (not attr_name.startswith('_') or attr_name == "__init__") and (isfunction(attr) or ismethod(attr)):
                        try:
                            sig = signature(attr)
                            is_a = asyncio.iscoroutinefunction(var)
                            methods.append(f"- `{attr_name}{sig}` Async: `{is_a}")
                            if attr.__doc__:
                                r = attr.__doc__.split('\n')[0]
                                methods.append(f"  {r}")
                        except:
                            methods.append(f"- `{attr_name}()`")
                if methods:
                    desc_parts.append("**Methods:**\n" + "\n".join(methods))

            elif isfunction(var) or ismethod(var):  # Function
                try:
                    sig = signature(var)
                    desc_parts.append(f"**Signature:** `{var.__name__}{sig}`")
                    is_a = asyncio.iscoroutinefunction(var)
                    desc_parts.append(f"**IS Async:** `{is_a}`")
                    if var.__doc__:
                        desc_parts.append(f"**Documentation:**\n{var.__doc__.strip()}")
                    ret_anno = sig.return_annotation
                    if ret_anno != Signature.empty:
                        desc_parts.append(f"**Returns:** `{ret_anno}`")
                except:
                    desc_parts.append(f"**Function:** `{var.__name__}()`")

            elif isinstance(var, BaseModel):  # Pydantic model
                desc_parts.append(f"**Type:** Pydantic model '{var.__class__.__name__}'")
                fields = []
                for field_name, field in var.model_fields.items():
                    value = getattr(var, field_name, None)
                    fields.append(f"- `{field_name}: {field.annotation.__name__}` = {repr(value)}")
                if fields:
                    desc_parts.append("**Fields:**\n" + "\n".join(fields))

            else:  # Instance
                class_type = var.__class__
                desc_parts.append(f"**Type:** `{class_type.__module__}.{class_type.__name__}`")

                # Instance initialization details
                try:
                    init = class_type.__init__
                    sig = signature(init)
                    params = list(sig.parameters.items())[1:]  # Skip self
                    if params:
                        args = []
                        for name, param in params:
                            if param.default == param.empty:
                                args.append(name)
                            else:
                                args.append(f"{name}={param.default}")
                        desc_parts.append(f"**Init Args:** `{', '.join(args)}`")
                except:
                    pass

                # Instance state
                state = get_instance_state(var)
                if state:
                    desc_parts.append("**Current instance State:**")
                    for attr_name, attr_value in state.items():
                        desc_parts.append(f"- `{attr_name}` = {attr_value}")

                # Documentation
                doc = getdoc(var) or getdoc(class_type)
                if doc:
                    desc_parts.append(f"**Documentation:**\n{doc.strip()}")

            descriptions.append("\n".join(desc_parts))

        return "\n\n".join(descriptions)

    async def _execute_code(self, code: str, context:dict) -> ExecutionRecord:
        """Execute code and track results"""
        lang = context.get('lang', 'py')
        try:

            if'py' in lang:

                return await self._execute_py(code)

            elif self.web_js and 'js' in lang:
                return await self._execute_js(code, context)

        except Exception as e:
            record = ExecutionRecord(code=code, result=None, error=str(e))
            self.execution_history.append(record)
            return record
        record = ExecutionRecord(code=code, result=None, error=f"Invalid lang {lang} valid is, {'js' if self.web_js else 'py'}]")
        self.execution_history.append(record)
        return record

    async def _execute_py(self, code) -> ExecutionRecord:
        show = len(code) > 450 and code.count('while') > 1 and code.count('print') >= 1
        result = await self.ipython.run_cell(code, show)

        all_keys = list(self.ipython.user_ns.keys())

        new_keys = [key for key in all_keys if key not in self.init_keys]
        # Update pipeline variables from IPython namespace

        for var_name in new_keys:
            if var_name.startswith('_'):
                continue
            self.variables[var_name] = self.ipython.user_ns[var_name]

        record = ExecutionRecord(code=code, result=result, error=None)
        self.execution_history.append(record)
        return record

    async def _execute_js(self, code: str, context: dict) -> ExecutionRecord:
        """Execute JavaScript code in browser context"""

        if '<script>' in code:
            code = code.split('<script>')[1]
        if '</script>' in code:
            code = code.split('</script>')[0]
        def _format_error_markdown(error: str) -> str:
            """Format error as Markdown"""
            return f"""
# Execution Error
{error}
"""

        def _format_result_markdown(result_: dict) -> str:
            """Format execution result as Markdown"""

            def _clean_html_content(html: str) -> str:
                """Clean HTML content and convert to Markdown-like format"""
                soup = BeautifulSoup(html, 'html.parser')

                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                # Add Markdown formatting
                text = re.sub(r'^(.+)$', r'> \1', text, flags=re.MULTILINE)

                return text

            md_parts = []

            # Add title
            md_parts.append("# Page Analysis Results\n")

            # Format JavaScript result
            if result_.get('js_result'):
                md_parts.append("## JavaScript Execution Result")
                md_parts.append("```")
                md_parts.append(str(result_['js_result']))
                md_parts.append("```\n")

            # Format page state
            if 'page_state' in result_:
                md_parts.append("## Page Information")
                md_parts.append(f"- **URL**: {result_['page_state']['url']}")
                md_parts.append(f"- **Title**: {result_['page_state']['title']}\n")

                # Clean and format content
                if 'content' in result_['page_state']:
                    content = _clean_html_content(result_['page_state']['content'])
                    if content:
                        md_parts.append("### Page Content")
                        md_parts.append(content + "\n")

            # Format extracted data
            if result_.get('extracted_data'):
                md_parts.append("## Extracted Data")
                for key, value in result_['extracted_data'].items():
                    if value:
                        md_parts.append(f"### {key.replace('_', ' ').title()}")
                        if isinstance(value, list):
                            for item in value:
                                md_parts.append(f"- {item}")
                        else:
                            md_parts.append(str(value))
                        md_parts.append("")

            return "\n".join(md_parts)

        try:
            # Prepare execution context
            url = context.get('url')
            if url:
                await self.browser_session.page.goto(url, wait_until='networkidle')

            self.ipython.vfs.write_file(f'src/temp_js/_temp_{len(self.js_history)}.js', code)

            # Modified JavaScript execution to use DOM methods instead of Playwright API
            # Wrap the code in an async function for proper execution
            if 'async' not in code and 'await' in code:
                # If code contains await but isn't marked async, wrap it
                code = f"async () => {{\n{code}\n}}"

            # await self.browser_session.page.screenshot(path=self._session / "..")
            # Execute JavaScript
            result = await self.browser_session.page.evaluate("""
                    async () => {
                        try {
                            const result = await (async () => {
                                %s
                            })();
                            return { success: true, result };
                        } catch (error) {
                            return {
                                success: false,
                                error: error.toString(),
                                stack: error.stack
                            };
                        }
                    }
                    """ % code)

            if isinstance(result, dict) and 'success' in result:
                if not result['success']:
                    raise Exception(f"JavaScript Error: {result.get('error')}\nStack: {result.get('stack')}")
                result = result.get('result')

            # Capture page state after execution
            page_state = {
                'url': self.browser_session.page.url,
                'title': await self.browser_session.page.title(),
                'content': await self.browser_session.page.content(),
            }

            # Extract data using patterns if specified
            extracted_data = None
            if 'patterns' in context:
                extractor = Extractor(self.browser_session.page)
                extracted_data = await extractor.extract(context['patterns'])

            # Create execution record
            record = JSExecutionRecord(
                code=code,
                result=result,
                page_state=page_state,
                extracted_data=extracted_data
            )

            self.js_history.append(record)

            # Convert to standard ExecutionRecord for pipeline
            return ExecutionRecord(
                code=code,
                result=_format_result_markdown({
                    'js_result': result,
                    'page_state': page_state,
                    'extracted_data': extracted_data
                }),
                error=None
            )

        except Exception as e:
            error_md = _format_error_markdown(str(e))
            return ExecutionRecord(code=code, result=None, error=error_md)


    def __str__(self):
        """String representation of pipeline session"""
        return str(self.ipython)

    async def _process_think_result(self, think_result: ThinkResult, task:str) -> Tuple[ThinkState,  Optional[ExecutionRecord | str]]:
        """Process the result of agent thinking"""
        if think_result.action == 'brake':
            return ThinkState.BRAKE, think_result.content

        elif think_result.action == 'update':
            if think_result.context.get('object_name') is None:
                return ThinkState.ACTION, "no object_name specified in context!"
            if think_result.context.get('file') is not None:
                self.ipython.user_ns['__file__'] = think_result.context.get('file')
            result = await self.verbose_output.process(think_result.action,
                                                       self.ipython.modify_code(code=think_result.content,
                                                    object_name=think_result.context.get('object_name'),))
            return ThinkState.PROCESSING, result

        elif think_result.action == 'code':
            if think_result.context.get('file') is not None:
                self.ipython.user_ns['__file__'] = think_result.context.get('file')
            result = await self._execute_code(think_result.content, think_result.context)
            return ThinkState.PROCESSING, result

        elif think_result.action == 'done':
            return ThinkState.DONE, think_result.content

        elif think_result.action == 'infos':
            infos = await self.chat_session.get_reference(think_result.content, to_str=True)
            return ThinkState.ACTION, infos

        elif think_result.action == 'guide':
            details = await self.process_memory.get_reference(think_result.content, to_str=True)
            plan = await self.agent.a_mini_task(f"""You are an AI guidance system designed to help determine the next step in a task and provide instructions on how to proceed. Your role is to analyze the given information and offer clear, actionable guidance for the next steps.

First, carefully read and understand the main task:
<main_task>
{task}
</main_task>

Next, review the last thought of the agent, if available:
<last_thought>
{think_result.content}
{think_result.context}
</last_thought>

Then, examine the processing history, if provided:
<processing_history>
{details}
</processing_history>

To determine the next step and provide guidance, follow these instructions:

1. Analyze the main task, breaking it down into smaller, manageable steps if necessary.
2. Consider the last thought and processing history to understand the current progress and context.
3. Identify any gaps, challenges, or areas that need further attention.
4. Determine the most logical and efficient next step to move the task forward.
5. Provide clear, concise instructions on how to complete this next step.

When formulating your response, follow this structure:

1. Begin with a brief summary of the current situation, referencing the main task and any relevant information from the last thought or processing history.
2. Clearly state the next step that should be taken.
3. Provide detailed instructions on how to complete this step, including any specific techniques, methods, or considerations to keep in mind.
4. If applicable, mention any potential challenges or pitfalls to be aware of during this step.
5. Conclude with a brief statement on how this step contributes to the overall progress of the main task.

Format your response using the following sections:
<summary>
(Include your summary of the current situation here)
</summary>

<next_step>
(State the next step to be taken here)
</next_step>

<instructions>
(Provide detailed instructions for completing the next step here)
</instructions>

<challenges>
(If applicable, mention potential challenges or pitfalls here)
</challenges>

<conclusion>
(Briefly state how this step contributes to overall progress)
</conclusion>

Remember to be clear, concise, and specific in your guidance. Avoid vague or ambiguous instructions, and provide concrete examples or explanations where necessary.""")
            return ThinkState.ACTION, plan

        return ThinkState.ACTION, None

    async def execute(self, code:str):
        return str(await self._execute_code(code))

    def clear(self):
        self.chat_session.history = []
        self.process_memory.history = []
        self.execution_history = []
        self.variables = {}
        self.ipython.reset()
        self.js_history = []

    async def get_process_hint(self, task):
        return await self.process_memory.get_reference(task, to_str=True), await self.chat_session.get_reference(task, to_str=True)

    def show_vars(self):
        self.verbose_output.log_state("VARS", self.variables, override=True)

    def set_file(self, full_file_path_and_name):
        if not os.path.exists(full_file_path_and_name):
            print("Invalid file")
            return
        self.ipython.user_ns["__file__"] = full_file_path_and_name

    async def run(self, task, do_continue=False) -> PipelineResult:
        """Run the pipeline with separated thinking and processing phases"""
        state = ThinkState.ACTION
        result = None
        original_task = task
        if not do_continue:
            task = self.agent.mini_task(task, "user", f"""You are an AI assistant tasked with refactoring a user-provided task description into a more structured format with context learning and examples. Your goal is to create a comprehensive and well-organized task description that incorporates model flows and potential code fixes.

First, I will provide you with a task description and some example tasks. Please read them carefully:

<existing_globals>
{self._generate_variable_descriptions()}
</existing_globals>

<example_tasks>
Task: Create a simple analysis of a list of numbers
- Generate a list of 100 random numbers between 1-1000
- Calculate the mean, median, and standard deviation
- Create a histogram of the distribution
- Print all results and display the plot

Task: Create a reinforcement learning (RL) agent to play a simple game
- Set up an OpenAI Gym environment (e.g., CartPole)
- Implement a Q-learning or Deep Q-Network (DQN) agent
- Train the model and optimize hyperparameters
- Visualize learning progress with reward graphs
- Save and reload trained models for inference
- Provide an option to let the trained agent play in real time

Task: Perform edge detection on an image
- Load an image from a URL or local file
- Convert the image to grayscale
- Apply Gaussian blur to reduce noise
- Use Canny edge detection to extract edges
- Display the original and processed images side by side
- Save the output image

Task: Build a basic sentiment analysis system
- Load a dataset of movie reviews (you can use a small sample)
- Preprocess the text (remove punctuation, lowercase, etc.)
- Create a TF-IDF vectorizer
- Split data into training and testing sets
- Train a classifier (e.g., Naive Bayes or LogisticRegression)
- Evaluate performance with accuracy, precision, recall
- Create a confusion matrix visualization
- Make predictions on new sample texts
</example_tasks>

Now, please refactor the given task description using the following guidelines:

1. Analyze the task description and identify the main components and objectives.

2. Structure the refactored task in a similar format to the example tasks, including:
   - A clear title that summarizes the task
   - A difficulty level (Easy, Intermediate, Hard, or Super Hard)
   - A brief introduction to the task's context and purpose
   - A code block containing step-by-step instructions
   - A list of required skills, libraries, or technologies

3. Incorporate model flows by breaking down the task into logical steps and explaining the process flow.

4. Include potential code fixes or common pitfalls that users might encounter while working on the task.

5. Add context learning elements by providing brief explanations or resources for key concepts related to the task.

6. Ensure that the refactored task is comprehensive and can stand alone as a learning exercise.

Please provide your refactored task description within <refactored_task> tags. Use appropriate subheadings and formatting to make the description clear and easy to read.

Additional tips:
- Mention any prerequisites or assumed knowledge
- Suggest potential extensions or variations of the task for further learning

Remember to maintain the original intent and complexity of the task while improving its structure and clarity.""")
            if '<refactored_task>' in task:
                task = task.split('<refactored_task>')[1]
            if '</refactored_task>' in task:
                task = task.split('</refactored_task>')[0]
        code_follow_up_prompt = f"""
You are an AI assistant responsible for evaluating task completion and providing feedback on the execution process. Your goal is to determine if a given task has been completed based on the execution result, and to offer insights for future improvements.

You will be provided with two inputs:
<task_description>
{original_task}
{f'<refactored_task_description_from_ai>{task}</refactored_task_description_from_ai>' if not do_continue else ''}
</task_description>

<code>
#CODE#
</code>

<execution_result>
#EXECUTION_RESULT#
</execution_result>

First, carefully analyze the task description and the execution result. Determine whether the task has been completed successfully based on the information provided.

If the task is completed:
1. Prepare a brief statement indicating that the task is done.
2. Summarize the output for the user in a clear and concise manner.

If the task is not completed:
1. Prepare a brief statement indicating that the task is not done.
2. Identify the specific aspects of the task that remain incomplete.

Regardless of task completion status, evaluate the procedure and effectiveness of the execution:
1. Analyze the workflow: Describe the steps taken in the execution process.
2. Assess effectiveness: Determine how well the procedure achieved the desired outcome.
3. Identify errors: Pinpoint any mistakes or inefficiencies in the execution.
4. Provide recommendations: Suggest improvements for future task executions.

tip: Enclose mutil line strings property for python eval to function!
tip: Set is_completed True if all requirements are completed from <task_description>.
tip: Help the Agent with your analyses to finalize the <task_description>.
{f'tip: Prefer new informations from <execution_result> over <refactored_task_description_from_ai> based of <code>' if not do_continue else ''}
note : for the final result only toke information from the <execution_result>. if the relevant informations is not avalabel try string withe tips in the recommendations. else set is_completed True and return the teh Task failed!
Ensure that your evaluation is thorough, constructive, and provides actionable insights for improving future task executions.
Add guidance based on the the last execution result"""
        code_follow_up_prompt_ = [code_follow_up_prompt]
        py_chane_prompt = '''2. 'update':
    - For updating variables, functions, and methods in memory and optionally in files
    - Required: code in content
    - Required: object_name in context
    - Optional: file in context for file updates
    - Preserves method signatures and indentation
    - Required structure: {
         'context': {
             'object_name': "Name of variable, function, or method to update",
             'file': "Optional: Path to file to update (if none, updates only in memory)"
         },
         'content': 'New code, value, or implementation',
         'action': 'update'
    }

    - Examples:

        # 1. Update a variable in memory
        {
            'context': {
                'object_name': "x"
            },
            'content': "5",
            'action': 'update'
        }

        # 2. Change a method implementation
        {
            'context': {
                'object_name': "Dog.sound",
                'file': "animals.py"  # Optional: updates both in memory and file
            },
            'content': '"""def sound(self):\n        return "Woof""""',
            'action': 'update'
        }

        # 3. Modify a function with memory-only update
        {
            'context': {
                'object_name': "calculate_age"
            },
            'content': '"""def calculate_age():\n    return 25"""',
            'action': 'update'
        }

        # 4. Update class method with file persistence
        {
            'context': {
                'object_name': "User.__init__",
                'file': "models.py"
            },
            'content': '"""def __init__(self, name="Default"):\n        self.name = name\n        self.created_at = datetime.now()"""',
            'action': 'update'
        }

        # 5. Update global configuration variable
        {
            'context': {
                'object_name': "MAX_CONNECTIONS",
                'file': "config.py"
            },
            'content': "100",
            'action': 'update'
        }

        # 6. Modify class property
        {
            'context': {
                'object_name': "Product.price"
            },
            'content': "@property\ndef price(self):\n    return self._price * (1 + self.tax_rate)",
            'action': 'update'
        }

        # 7. Update function with complex logic
        {
            'context': {
                'object_name': "process_data",
                'file': "utils.py"
            },
            'content': '"""def process_data(data):\n    result = {}\n    for key, value in data.items():\n        result[key.lower()] = value.upper()\n    return result"""',
            'action': 'update'
        }'''
        initial_prompt = f"""
You are an AI {'js running (in playwright browser)' if self.web_js else 'py'} coding agent specializing in iterative development and code refinement, designed to perform tasks that involve thinking. Your goal is to complete the given task while demonstrating a clear thought process throughout the execution.
SYSTEM STATE:
<current_state>
Iteration: #ITER#
Status: #STATE#
Last EXECUTION: #EXECUTION#
</current_state>

ENVIRONMENT: {'current file :'+self.ipython.user_ns.get("__file__")  if self.ipython.user_ns.get("__file__") is not None else ''}

{'''<global_variables>
#LOCALS#
</global_variables>''' if not self.web_js else ''}

MEMORY:
<process_memory>
#PHINT#
</process_memory>

<chat_memory>
#CHINT#
</chat_memory>

VALIDATION CHECKLIST (Must verify before each action):
1. ✓ Check existing variables in ENVIRONMENT <global_variables>
2. ✓ Verify existing functions and classes
3. ✓ Review current imports
4. ✓ Confirm method signatures
5. ✓ Validate state preservation

WORKFLOW STEPS:
1. Analyze Current State:
   - Reason and use all avalabel context
   - Do not repeat the same errors
   - Review existing implementations
   - Check variable values
   - Verify import statements
   - Document dependencies

2. Plan Change:
   - NO example/simulation/simulate
   - Use existing code when possible
   - Prefer updates over rewrites

3. Execute Change:
   - Use appropriate action
   - Maintain existing state
   - Document modifications
   - Verify results

You will use a structure called ThinkResult to organize your thoughts and actions.
For each step of your task, follow this process:

ACTIONS:
1. 'code':
    - MUST check <global_variables> first
    - NEVER create demo functions
    - Include 'reason'
    - lang default {'js' if self.web_js else 'py'}
    - Required: code in content
    - code MUST call a function or display the row variabel / value at the end!
    - Required: {{'context':{{'lang':{'js' if self.web_js else 'py'},  'reason': ... }}...}}
    - Optional file key in context example  {{'context':{{'lang':{'js' if self.web_js else 'py'},  'file': 'main.py' ,  'reason': ... }}...}}
    - Tip: use comments to reason with in the code

    { js_prompt if self.web_js else py_chane_prompt}


3. 'infos': Request specific details
4. 'guide': Get step clarification
5. 'brake': Pause for assessment
6. 'done': Summarize changes

CODE CONSTRAINTS:
1. State Preservation:
   - ALL variables ar persist
   - ALL functions remain
   - ALL classes ar maintained

2. Import Management:
   - Check <global_variables> for modules
   - Use absolute imports
   - Document new dependencies

3. Function Handling:
   - NEVER overwrite existing
   - Use update for changes
   - Preserve signatures

4. Variable Scope:
   - Maintain existing scope
   - Check for conflicts
   - Document state changes

EXECUTION RULES:
1. VERIFY before create
2. UPDATE don't replace
3. TEST after each change

Next Action Required:
1. Review current state
2. Check existing code
3. Execute with state preservation

!!CRITICAL!!
- NO demo functions
- NO placeholder functions
- USE existing code
- FOR Implementations prefer writing large production redy code chunks.
- FOR reasoning and validation write small code blocks.
- THE CODE must call something or end the code with an value!
- NO INFINIT LOOPS! none breakable while loops ar not allowed, exception ui (closed by user)
- NO {'python' if not self.web_js else 'javascript'} top level return, only write the variabel or value itself!
- {'code is run using exec! do not use !pip ...' if not self.web_js else 'session is persistent'}
{'- instead use auto_install(package_name, install_method="pip", upgrade=False, quiet=False, version=None, extra_args=None)' if not self.web_js else '- The js code is NOT run as an module! no imports not script tags'}
# Example usage first time
│ auto_install('pandas', version='1.3.0')
│ import pandas
│ auto_install('pygame')
│ import pygame
│ auto_install('numpy')
│ import numpy as np
!TIPS!
- {'<global_variables> can contain instances and functions you can use in your python' if not self.web_js else 'write javascript'} code
- if the function is async you can use top level await
- if their is missing of informations try running code to get the infos
- if you got stuck or need assistance break with a question to the user.
{'- run functions from <global_variables> using name(*args, **kwargs) or await name(*args, **kwargs)' if not self.web_js else '- the js code is run async top level with in playwright'}
{'- <global_variables> ar global accessible!' if not self.web_js else '- include elements for retrival from a web page'}
{'- if an <global_variables> name is lower lists an redy to use instance' if not self.web_js else '- use {{"context" : {{"url": page }}}} to navigate to a web page '}
"""
        p_hint, c_hint = await self.get_process_hint(task)
        initial_prompt = initial_prompt.replace('#PHINT#', p_hint)
        initial_prompt = initial_prompt.replace('#CHINT#', c_hint)
        initial_prompt_ = initial_prompt
        iter_i = 0
        iter_p = 0
        iter_tat = 0
        next_infos = ""
        if not do_continue:
            await self.chat_session.add_message({'role': 'user', 'content': task})
        else:
            self.restore()
            await self.chat_session.add_message({'role': 'user', 'content': task})

        if self.web_js and self.browser_session.browser is None:
            await self.start_web()
            if self.restore_var:
                await self.restore_web()

        # await self.verbose_output.log_message('user', task)
        self.verbose_output.log_header(task)
        while state != ThinkState.DONE:
            iter_i += 1
            t0 = time.process_time()
            prompt = initial_prompt.replace('#ITER#', f'{iter_i} max {self.max_iter}')
            prompt = prompt.replace('#STATE#', f'{state.name}')
            prompt = prompt.replace('#EXECUTION#', f'{next_infos}')  if next_infos else prompt.replace('Last EXECUTION: #EXECUTION#', f'')
            prompt = prompt.replace('#LOCALS#', f'{self._generate_variable_descriptions()}')
            self.verbose_output.log_state(state.name, {})
            self.verbose_output.formatter.print_iteration(iter_i, self.max_iter)
            if state == ThinkState.ACTION:
                iter_tat +=1
                if iter_tat > self.max_think_after_think:
                    state = ThinkState.BRAKE
            else:
                iter_tat = 0

            if state == ThinkState.ACTION:
                # Get agent's thoughts
                think_dicts = await self.verbose_output.process(state.name, self.agent.a_format_class(
                    ThinkResults,
                    prompt,
                    message=self.chat_session.get_past_x(self.max_iter*2, last_u=not do_continue).copy()+([self.process_memory.history[-1]] if self.process_memory.history else []) ,
                ))
                think_dicts = think_dicts.get("actions")
                if think_dicts is None:
                    think_dict = await self.verbose_output.process(state.name, self.agent.a_format_class(
                        ThinkResult,
                        prompt,
                        message=self.chat_session.get_past_x(self.max_iter * 2, last_u=not do_continue).copy() + (
                            [self.process_memory.history[-1]] if self.process_memory.history else []),
                    ))
                if len(think_dicts) == 1:
                    think_dict = think_dicts[0]
                else:
                    for think_dict in think_dicts[:-1]:
                        if think_dict.get('context') is None:
                            think_dict['context'] = {'context': 'N/A'}
                        if not isinstance(think_dict.get('context'), dict):
                            think_dict['context'] = {'context': think_dict.get('context')}
                        think_result = ThinkResult(**think_dict)
                        await self.chat_session.add_message(
                            {'role': 'assistant', 'content': think_result.content + str(think_result.context)})
                        state, result = await self.verbose_output.process(think_dict.get("action"),
                                                                          self._process_think_result(think_result,
                                                                                                     task=task))
                        if result:
                            await self.chat_session.add_message(
                                {'role': 'system', 'content': 'Evaluation: ' + str(result)})
                            await self.verbose_output.log_message('system', str(result))
                    think_dict = think_dicts[-1]
                await self.verbose_output.log_think_result(think_dict)
                if think_dict.get('context') is None:
                    think_dict['context'] = {'context': 'N/A'}
                if not isinstance(think_dict.get('context'), dict):
                    think_dict['context'] = {'context': think_dict.get('context')}
                think_result = ThinkResult(**think_dict)
                state, result = await self.verbose_output.process(think_dict.get("action"), self._process_think_result(think_result, task=task))
                await self.chat_session.add_message({'role': 'assistant', 'content': think_result.content + str(think_result.context)})
                if result:
                    await self.chat_session.add_message({'role': 'system', 'content': 'Evaluation: '+str(result)})
                    await self.verbose_output.log_message('system', str(result))
                    code_follow_up_prompt_[0] = code_follow_up_prompt.replace("#EXECUTION_RESULT#", str(result))
                    if not self.web_js and isinstance(result ,ExecutionRecord):
                        code_follow_up_prompt_[0] = code_follow_up_prompt_[0].replace("#CODE#", result.code)
                    elif not self.web_js:
                        code_follow_up_prompt_[0] = code_follow_up_prompt_[0].replace("#CODE#", self._generate_variable_descriptions())
                    else:
                        code_follow_up_prompt_[0] = code_follow_up_prompt_[0].replace("#CODE#", "Focus on the result!")
                else:
                    code_follow_up_prompt_[0] = code_follow_up_prompt.replace("#EXECUTION_RESULT#", str(think_result))
                    code_follow_up_prompt_[0] = code_follow_up_prompt_[0].replace("#CODE#",
                                                                              self._generate_variable_descriptions())


            elif state == ThinkState.PROCESSING:
                # Get agent's thoughts
                class Next(BaseModel):
                    is_completed: bool
                    recommendations: str
                    errors: str
                    effectiveness: str
                    workflow: str
                    text: str
                # Format the agent's thoughts into a structured response
                if self.v_agent is not None:
                    _agent = self.v_agent
                else:
                    _agent = self.agent
                next_dict = await self.verbose_output.process(state.name, _agent.a_format_class(
                    Next,
                    code_follow_up_prompt_[0],
                    message=self.chat_session.get_past_x(self.max_iter*2, last_u=not do_continue).copy(),
                ))
                next_infos = json.dumps(next_dict)
                await self.verbose_output.log_process_result(next_dict)
                await self.process_memory.add_message({'role': 'assistant', 'content': next_infos.replace('workflow:', 'past-workflow:')})
                iter_p += 1
                code_follow_up_prompt_[0] = code_follow_up_prompt
                if not next_dict.get('is_completed', True):
                    state = ThinkState.ACTION
                    initial_prompt = initial_prompt_.replace('#ITER#',f'#ITER#\nReasoning assist result: {next_dict}')
                    continue
                elif next_dict.get('is_completed', False):
                    result = next_dict.get('text', '')
                    state = ThinkState.DONE
                    continue
                else:
                    result = next_dict.get('text', '')
                    break

            elif state == ThinkState.BRAKE:
                break

            if iter_i < self.max_iter:
                if time.process_time() -t0 < self.timeout_timer*2.5:
                    with Spinner(f"Prevent rate limit posing for {self.timeout_timer}s", symbols='+', time_in_s=self.timeout_timer, count_down=True):
                        await asyncio.sleep(self.timeout_timer)
            else:
                state = ThinkState.BRAKE
                if isinstance(result, ExecutionRecord):
                    result = result.result
                elif isinstance(result, str):
                    pass
                else:
                    result = "Max iterations"
                break

        self.verbose_output.log_state(state.name, {})

        return PipelineResult(
            variables=self.variables,
            result=result,
            execution_history=self.execution_history,
            message=self.chat_session.get_past_x(iter_i*2, last_u=not do_continue),
        )

    async def run_project(self, task, lang='py', execute_function=None):
        if execute_function is None:
            if lang == 'py':
                execute_function = default_python_execute_function
            elif lang == 'rust':
                execute_function = default_rust_execute_function
            else:
                raise ValueError(f"Unsupported language: {lang}")
        class FileAction(BaseModel):
            action: str
            path: str
            content: Optional[str] = None

        class ProjectThinkResult(BaseModel):
            action: str
            file_actions: List[FileAction]
            reasoning: str

        class ProjectPipelineResult(BaseModel):
            result: str
            execution_history: List[str]
            files: Dict[str, str]
        state = ThinkState.ACTION
        result = None
        original_task = task
        vfs = VirtualFileSystem(self._session_dir / f"project_{lang}")

        project_prompt = f"""
    You are an AI coding agent specializing in {lang} project development. Your task is to create, modify, and manage files within a project structure to complete the given task. Use the VirtualFileSystem to interact with files.

    TASK DESCRIPTION:
    {task}
    CURRENT FILES:
    #files#

    WORKFLOW STEPS:
    1. Analyze the current project state
    2. Plan necessary changes or additions
    3. Execute changes using file actions
    4. Evaluate the project's progress

    Use the ProjectThinkResult structure to organize your thoughts and actions:

    class ProjectThinkResult(BaseModel):
        action: str  # 'code', 'evaluate', 'done'
        file_actions: List[FileAction]
        reasoning: str

    class FileAction(BaseModel):
        action: str  # 'write', 'read', 'delete', 'list'
        path: str
        content: Optional[str] = None

    EXECUTION RULES:
    1. Use absolute paths for all file operations
    2. Maintain a clear project structure
    3. Document your code and reasoning
    4. Ensure all necessary files are created and properly linked
    5. Use the appropriate language syntax and best practices for {lang}

    Next Action Required:
    1. Review the current project state
    2. Plan the next step in project development
    3. Execute file actions to implement changes
    """

        execution_history = []
        files = {}

        iter_i = 0
        self.verbose_output.log_header(task)

        while state != ThinkState.DONE:
            iter_i += 1
            self.verbose_output.formatter.print_iteration(iter_i, self.max_iter)
            if iter_i>self.max_iter:
                break
            if state == ThinkState.ACTION:
                think_result = await self.agent.a_format_class(
                    ProjectThinkResult,
                    project_prompt.replace('#files#', vfs.print_file_structure()),
                    message=execution_history
                )
                self.verbose_output.log_state(state.name, think_result)
                think_result = ProjectThinkResult(**think_result)
                for file_action in think_result.file_actions:
                    path = str(Path(file_action.path).relative_to(vfs.base_dir))
                    if file_action.action == 'write':
                        vfs.write_file(path, file_action.content)
                        files[path] = file_action.content
                    elif file_action.action == 'read':
                        content = vfs.read_file(path)
                        files[path] = content
                    elif file_action.action == 'delete':
                        vfs.delete_file(path)
                        files.pop(path, None)
                    elif file_action.action == 'list':
                        dir_contents = vfs.list_directory(path)
                        files[path] = str(dir_contents)

                if think_result.action == 'evaluate':
                    state = ThinkState.PROCESSING
                elif think_result.action == 'done':
                    state = ThinkState.DONE

                execution_history.append(f"Action: {think_result.action}\nReasoning: {think_result.reasoning}")

            elif state == ThinkState.PROCESSING:
                if execute_function:
                    execution_result = await execute_function(files)
                    execution_history.append(f"Execution Result: {execution_result}")

                    evaluation_prompt = f"""
    Evaluate the current state of the project based on the execution result:

    {execution_result}

    Determine if the project is complete or if further modifications are needed.
    """
                    evaluation = await self.agent.a_format_class(
                        ProjectThinkResult,
                        evaluation_prompt,
                        message=execution_history
                    )
                    self.verbose_output.log_state(state.name, evaluation)
                    evaluation = ProjectThinkResult(**evaluation)
                    if evaluation.action == 'done':
                        state = ThinkState.DONE
                        result = execution_result
                    else:
                        state = ThinkState.ACTION
                else:
                    state = ThinkState.ACTION
            else:
                break

        return ProjectPipelineResult(
            result=result,
            execution_history=execution_history,
            files=files
        )

    async def __aenter__(self):
        self.clear()
        return self

    async def configure(self, verbose=None, print_function=None, with_js=False, agent=None, variables=None):
        if verbose is not None and (print_function is not None or verbose != self.verbose_output.verbose):
            if agent is None:
                agent = self.agent
            else:
                self.agent = agent
            agent.verbose = verbose
            self.verbose_output = EnhancedVerboseOutput(verbose=verbose, print_f=print_function)
            if with_js:
                self.browser_session: Optional[BrowserSession] = BrowserSession(verbose)
            if print_function is not None:
                agent.print_verbose = print_function
        if variables:
            self.variables = {**self.variables, **self._process_variables(variables)}
        if with_js:
            self.browser_session: Optional[BrowserSession] = BrowserSession(verbose)
            await self.start_web()
            if self.restore_var:
                await self.restore_web()
            self.web_js = with_js
        if self.restore_var:
            self.restore()

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.web_js:
            await self.browser_session.close()
            if self.restore_var:
                await self.save_web()
                self.save_session(f"Pipeline_Session_{self.agent.amd.name}")
        if exc_type is not None:
            print(f"Exception occurred: {exc_value}")
        else:
            print("Pipe Exit")

### -- extra -- ###

@dataclass
class SyncReport:
    """Report of variables synced from namespace to pipeline"""
    added: Dict[str, str]
    skipped: Dict[str, str]  # var_name -> reason
    errors: Dict[str, str]  # var_name -> error message

    def __str__(self) -> str:
        parts = []
        if self.added:
            parts.append("Added variables:")
            for name, type_ in self.added.items():
                parts.append(f"  - {name}: {type_}")
        if self.skipped:
            parts.append("\nSkipped variables:")
            for name, reason in self.skipped.items():
                parts.append(f"  - {name}: {reason}")
        if self.errors:
            parts.append("\nErrors:")
            for name, error in self.errors.items():
                parts.append(f"  - {name}: {error}")
        return "\n".join(parts)


def sync_globals_to_vars(
    pipeline: Any,
    namespace: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None,
    include_types: Optional[Union[Type, List[Type]]] = None,
    exclude_patterns: Optional[List[str]] = None,
    exclude_private: bool = True,
    deep_copy: bool = False,
    only_serializable: bool = False
) -> SyncReport:
    """
    Sync global variables or a specific namespace to pipeline variables.

    Args:
        pipeline: Pipeline instance to sync variables to
        namespace: Optional dictionary of variables (defaults to globals())
        prefix: Optional prefix for variable names (e.g., 'global_')
        include_types: Only include variables of these types
        exclude_patterns: List of regex patterns to exclude
        exclude_private: Exclude variables starting with underscore
        deep_copy: Create deep copies of variables instead of references
        only_serializable: Only include variables that can be serialized

    Returns:
        SyncReport with details about added, skipped and error variables

    Usage example:
# Basic usage - sync all globals
report = sync_globals_to_vars(pipeline)

# Sync only numeric types with prefix
report = sync_globals_to_vars(
    pipeline,
    include_types=[int, float],
    prefix="global_"
)

# Sync from specific namespace
import numpy as np
namespace = {"arr": np.array([1,2,3])}
report = sync_globals_to_vars(pipeline, namespace=namespace)

# Sync with deep copy and serialization check
report = sync_globals_to_vars(
    pipeline,
    deep_copy=True,
    only_serializable=True
)
    """
    # Initialize report
    report = SyncReport(
        added={},
        skipped={},
        errors={}
    )

    # Get namespace
    if namespace is None:
        # Get caller's globals
        namespace = currentframe().f_back.f_globals

    # Compile exclude patterns
    if exclude_patterns:
        patterns = [re.compile(pattern) for pattern in exclude_patterns]
    else:
        patterns = []

    # Normalize include_types
    if include_types and not isinstance(include_types, (list, tuple, set)):
        include_types = [include_types]
    def get_type_info(var: Any) -> str:
        """Helper to get detailed type information"""
        if isinstance(var, type):
            return f"class '{var.__name__}'"
        elif isinstance(var, BaseModel):
            return f"Pydantic model '{var.__class__.__name__}'"
        elif hasattr(var, '__class__'):
            type_name = var.__class__.__name__
            module_name = var.__class__.__module__
            if module_name != 'builtins':
                return f"{module_name}.{type_name}"
            return type_name
        return type(var).__name__
    # Process each variable
    for name, value in namespace.items():
        try:
            # Skip if matches exclude criteria
            if exclude_private and name.startswith('_'):
                report.skipped[name] = "private variable"
                continue

            if any(pattern.match(name) for pattern in patterns):
                report.skipped[name] = "matched exclude pattern"
                continue

            if include_types and not isinstance(value, tuple(include_types)):
                report.skipped[name] = f"type {type(value).__name__} not in include_types"
                continue

            # Test serialization if required
            if only_serializable:
                try:
                    import pickle
                    pickle.dumps(value)
                except Exception as e:
                    report.skipped[name] = f"not serializable: {str(e)}"
                    continue

            # Prepare variable
            var_value = deepcopy(value) if deep_copy else value
            var_name = f"{prefix}{name}" if prefix else name

            # Add to pipeline variables
            pipeline.variables[var_name] = var_value
            report.added[var_name] = get_type_info(value)

        except Exception as e:
            report.errors[name] = str(e)

    return report


if __name__ == '__main__':
    # agent = get_free_agent("demo", "anthropic/claude-3-haiku-20240307")
    async def run_code():
        mock_ipy = MockIPython()
        mock_ipy.user_ns['VirtualFileSystem'] = VirtualFileSystem
        # Run async code with top-level await
        result = await mock_ipy.run_cell("""
if __name__ == '__main__':
    x = 1
        """, live_output=True)
        result = await mock_ipy.run_cell('''
import git
import os
import pathlib
from typing import Union, Optional

class GitVirtualFileSystemAdapter:
    def __init__(self, virtual_fs):
        """
        Initialize GitVirtualFileSystemAdapter with a VirtualFileSystem instance

        Args:
            virtual_fs (VirtualFileSystem): The virtual file system to extend
        """
        self.virtual_fs = virtual_fs
        self.repo = None

    def clone_remote_repo(self, remote_url: str, local_path: Union[str, pathlib.Path] = None, branch: str = 'main'):
        """
        Clone a remote Git repository into the virtual file system

        Args:
            remote_url (str): URL of the remote Git repository
            local_path (Union[str, pathlib.Path], optional): Local path to clone into. Defaults to repo name.
            branch (str, optional): Branch to checkout. Defaults to 'main'.

        Returns:
            pathlib.Path: Path to the cloned repository
        """
        if local_path is None:
            local_path = os.path.basename(remote_url).replace('.git', '')

        # Ensure directory exists in virtual file system
        self.virtual_fs.create_directory(local_path)

        # Clone the repository
        self.repo = git.Repo.clone_from(remote_url, local_path)

        # Checkout specific branch if needed
        if branch != 'main':
            self.repo.git.checkout(branch)

        return pathlib.Path(local_path)

    def commit_changes(self, message: str, force: bool = False):
        """
        Commit changes to the local repository

        Args:
            message (str): Commit message
            force (bool, optional): Force commit even with untracked files. Defaults to False.
        """
        if not self.repo:
            raise ValueError("No repository initialized. Clone a repo first.")

        # Stage all changes
        if force:
            self.repo.git.add(A=True)
        else:
            self.repo.git.add(update=True)

        # Commit changes
        self.repo.index.commit(message)

    def push_changes(self, remote: str = 'origin', branch: str = None, force: bool = False):
        """
        Push local changes to remote repository

        Args:
            remote (str, optional): Remote name. Defaults to 'origin'.
            branch (str, optional): Branch to push. Uses current branch if not specified.
            force (bool, optional): Force push. Defaults to False.
        """
        if not self.repo:
            raise ValueError("No repository initialized. Clone a repo first.")

        if branch is None:
            branch = self.repo.active_branch.name

        if force:
            self.repo.git.push(remote, branch, force=True)
        else:
            self.repo.git.push(remote, branch)

    def pull_changes(self, remote: str = 'origin', branch: str = None, force: bool = False):
        """
        Pull changes from remote repository

        Args:
            remote (str, optional): Remote name. Defaults to 'origin'.
            branch (str, optional): Branch to pull. Uses current branch if not specified.
            force (bool, optional): Force pull, discarding local changes. Defaults to False.
        """
        if not self.repo:
            raise ValueError("No repository initialized. Clone a repo first.")

        if branch is None:
            branch = self.repo.active_branch.name

        if force:
            self.repo.git.fetch(remote)
            self.repo.git.reset(f'{remote}/{branch}', hard=True)
        else:
            self.repo.git.pull(remote, branch)

    def get_repo_overview(self) -> dict:
        """
        Provide an overview of the current repository for LLM context

        Returns:
            dict: Repository overview with key details
        """
        if not self.repo:
            return {"status": "No repository initialized"}

        return {
            "active_branch": self.repo.active_branch.name,
            "remote_urls": {rem.name: rem.url for rem in self.repo.remotes},
            "uncommitted_changes": len(self.repo.index.diff(self.repo.head.commit)) if self.repo.head.is_valid() else 0,
            "untracked_files": len(self.repo.untracked_files)
        }

# Optionally demonstrate basic usage
if __name__ == '__main__':
    vfs = VirtualFileSystem('.')
    git_adapter = GitVirtualFileSystemAdapter(vfs)

    # Example workflow (uncomment and modify as needed)
    # repo_path = git_adapter.clone_remote_repo('https://github.com/example/repo.git')
    # git_adapter.commit_changes('Initial commit')
    # git_adapter.push_changes()
''', live_output=False)

        print("Result:", result)


    # Run the async function
    asyncio.run(run_code())

    #asd = "Evaluation: Output -> \nstdout:Episode 0: Total Reward = 35.0, Epsilon = 0.98\n\r⣾ code | 0.04\x1b[K\r⣽ code | 0.15\x1b[K\r⣻ code | 0.26\x1b[K\r⢿ code | 0.37\x1b[KEpisode 50: Total Reward = 10.0, Epsilon = 0.06\n\r⡿ code | 0.48\x1b[K\r⣟ code | 0.59\x1b[K\r⣯ code | 0.70\x1b[K\r⣷ code | 0.81\x1b[K\r⣾ code | 0.92\x1b[K\r⣽ code | 1.03\x1b[K\r⣻ code | 1.14\x1b[K\r⢿ code | 1.25\x1b[K\r⡿ code | 1.37\x1b[K\r⣟ code | 1.47\x1b[K\r⣯ code | 1.58\x1b[K\r⣷ code | 1.70\x1b[K\r⣾ code | 1.81\x1b[K\r⣽ code | 1.92\x1b[K\r⣻ code | 2.03\x1b[KEpisode 100: Total Reward = 58.0, Epsilon = 0.01\n\r⢿ code | 2.14\x1b[K\r⡿ code | 2.25\x1b[K\r⣟ code | 2.36\x1b[K\r⣯ code | 2.47\x1b[K\r⣷ code | 2.58\x1b[K\r⣾ code | 2.69\x1b[K\r⣽ code | 2.80\x1b[K\r⣻ code | 2.91\x1b[K\r⢿ code | 3.02\x1b[K\r⡿ code | 3.13\x1b[K\r⣟ code | 3.24\x1b[K\r⣯ code | 3.35\x1b[K\r⣷ code | 3.46\x1b[K\r⣾ code | 3.57\x1b[K\r⣽ code | 3.68\x1b[K\r⣻ code | 3.79\x1b[K\r⢿ code | 3.90\x1b[K\r⡿ code | 4.01\x1b[K\r⣟ code | 4.12\x1b[K\r⣯ code | 4.23\x1b[K\r⣷ code | 4.34\x1b[K\r⣾ code | 4.45\x1b[K\r⣽ code | 4.56\x1b[K\r⣻ code | 4.67\x1b[K\r⢿ code | 4.79\x1b[K\r⡿ code | 4.90\x1b[K\r⣟ code | 5.01\x1b[K\r⣯ code | 5.12\x1b[K\r⣷ code | 5.23\x1b[K\r⣾ code | 5.34\x1b[K\r⣽ code | 5.45\x1b[K\r⣻ code | 5.56\x1b[K\r⢿ code | 5.67\x1b[K\r⡿ code | 5.78\x1b[K\r⣟ code | 5.89\x1b[K\r⣯ code | 6.00\x1b[K\r⣷ code | 6.11\x1b[K\r⣾ code | 6.22\x1b[K\r⣽ code | 6.32\x1b[K\r⣻ code | 6.42\x1b[K\r⢿ code | 6.53\x1b[K\r⡿ code | 6.64\x1b[K\r⣟ code | 6.75\x1b[K\r⣯ code | 6.86\x1b[K\r⣷ code | 6.97\x1b[K\r⣾ code | 7.08\x1b[K\r⣽ code | 7.19\x1b[K\r⣻ code | 7.30\x1b[K\r⢿ code | 7.41\x1b[K\r⡿ code | 7.52\x1b[K\r⣟ code | 7.63\x1b[K\r⣯ code | 7.74\x1b[K\r⣷ code | 7.85\x1b[KEpisode 150: Total Reward = 200.0, Epsilon = 0.01\n\r⣾ code | 7.96\x1b[K\r⣽ code | 8.07\x1b[K\r⣻ code | 8.18\x1b[K\r⢿ code | 8.29\x1b[K\r⡿ code | 8.40\x1b[K\r⣟ code | 8.51\x1b[K\r⣯ code | 8.62\x1b[K\r⣷ code | 8.73\x1b[K\r⣾ code | 8.84\x1b[K\r⣽ code | 8.95\x1b[K\r⣻ code | 9.07\x1b[K\r⢿ code | 9.18\x1b[K\r⡿ code | 9.29\x1b[K\r⣟ code | 9.40\x1b[K\r⣯ code | 9.51\x1b[K\r⣷ code | 9.61\x1b[K\r⣾ code | 9.72\x1b[K\r⣽ code | 9.84\x1b[K\r⣻ code | 9.94\x1b[K\r⢿ code | 10.04\x1b[K\r⡿ code | 10.15\x1b[K\r⣟ code | 10.26\x1b[K\r⣯ code | 10.37\x1b[K\r⣷ code | 10.48\x1b[K\r⣾ code | 10.59\x1b[K\r⣽ code | 10.71\x1b[K\r⣻ code | 10.81\x1b[K\r⢿ code | 10.92\x1b[K\r⡿ code | 11.03\x1b[K\r⣟ code | 11.15\x1b[K\r⣯ code | 11.26\x1b[K\r⣷ code | 11.37\x1b[K\r⣾ code | 11.48\x1b[K\r⣽ code | 11.59\x1b[K\r⣻ code | 11.70\x1b[K\r⢿ code | 11.81\x1b[K\r⡿ code | 11.92\x1b[K\r⣟ code | 12.03\x1b[K\r⣯ code | 12.14\x1b[K\r⣷ code | 12.25\x1b[K\r⣾ code | 12.36\x1b[K\r⣽ code | 12.47\x1b[K\r⣻ code | 12.57\x1b[K\r⢿ code | 12.67\x1b[K\r⡿ code | 12.77\x1b[K\r⣟ code | 12.87\x1b[K\r⣯ code | 12.98\x1b[K\r⣷ code | 13.08\x1b[K\r⣾ code | 13.19\x1b[K\r⣽ code | 13.29\x1b[K\r⣻ code | 13.39\x1b[K\r⢿ code | 13.51\x1b[K\r⡿ code | 13.61\x1b[K\r⣟ code | 13.71\x1b[K\r⣯ code | 13.83\x1b[K\r⣷ code | 13.93\x1b[K\r⣾ code | 14.04\x1b[K\r⣽ code | 14.14\x1b[K\r⣻ code | 14.24\x1b[K\r⢿ code | 14.36\x1b[K\r⡿ code | 14.46\x1b[K\r⣟ code | 14.56\x1b[K\r⣯ code | 14.66\x1b[K\r⣷ code | 14.78\x1b[K\r⣾ code | 14.88\x1b[K\r⣽ code | 14.98\x1b[K\r⣻ code | 15.08\x1b[K\r⢿ code | 15.18\x1b[K\r⡿ code | 15.28\x1b[K\r⣟ code | 15.39\x1b[K\r⣯ code | 15.50\x1b[K\r⣷ code | 15.60\x1b[K\r⣾ code | 15.70\x1b[K\r⣽ code | 15.80\x1b[K\r⣻ code | 15.90\x1b[K\r⢿ code | 16.01\x1b[K\r⡿ code | 16.11\x1b[K\r⣟ code | 16.21\x1b[K\r⣯ code | 16.33\x1b[K\r⣷ code | 16.43\x1b[K\r⣾ code | 16.53\x1b[K\r⣽ code | 16.65\x1b[K\r⣻ code | 16.75\x1b[KEpisode 200: Total Reward = 200.0, Epsilon = 0.01\n\r⢿ code | 16.86\x1b[K\r⡿ code | 16.96\x1b[K\r⣟ code | 17.06\x1b[K\r⣯ code | 17.16\x1b[K\r⣷ code | 17.28\x1b[K\r⣾ code | 17.38\x1b[K\r⣽ code | 17.50\x1b[K\r⣻ code | 17.60\x1b[K\r⢿ code | 17.70\x1b[K\r⡿ code | 17.80\x1b[K\r⣟ code | 17.90\x1b[K\r⣯ code | 18.01\x1b[K\r⣷ code | 18.11\x1b[K\r⣾ code | 18.21\x1b[K\r⣽ code | 18.33\x1b[K\r⣻ code | 18.43\x1b[K\r⢿ code | 18.53\x1b[K\r⡿ code | 18.63\x1b[K\r⣟ code | 18.73\x1b[K\r⣯ code | 18.85\x1b[K\r⣷ code | 18.95\x1b[K\r⣾ code | 19.06\x1b[K\r⣽ code | 19.16\x1b[K\r⣻ code | 19.27\x1b[K\r⢿ code | 19.38\x1b[K\r⡿ code | 19.48\x1b[K\r⣟ code | 19.58\x1b[K\r⣯ code | 19.70\x1b[K\r⣷ code | 19.80\x1b[K\r⣾ code | 19.90\x1b[K\r⣽ code | 20.00\x1b[K\r⣻ code | 20.12\x1b[K\r⢿ code | 20.22\x1b[K\r⡿ code | 20.32\x1b[K\r⣟ code | 20.42\x1b[K\r⣯ code | 20.53\x1b[K\r⣷ code | 20.63\x1b[K\r⣾ code | 20.75\x1b[K\r⣽ code | 20.85\x1b[K\r⣻ code | 20.95\x1b[K\r⢿ code | 21.07\x1b[K\r⡿ code | 21.17\x1b[K\r⣟ code | 21.27\x1b[K\r⣯ code | 21.38\x1b[K\r⣷ code | 21.48\x1b[K\r⣾ code | 21.59\x1b[K\r⣽ code | 21.70\x1b[K\r⣻ code | 21.80\x1b[K\r⢿ code | 21.90\x1b[K\r⡿ code | 22.00\x1b[K\r⣟ code | 22.12\x1b[K\r⣯ code | 22.22\x1b[K\r⣷ code | 22.32\x1b[K\r⣾ code | 22.43\x1b[K\r⣽ code | 22.53\x1b[K\r⣻ code | 22.64\x1b[K\r⢿ code | 22.74\x1b[K\r⡿ code | 22.84\x1b[K\r⣟ code | 22.95\x1b[K\r⣯ code | 23.05\x1b[K\r⣷ code | 23.15\x1b[K\r⣾ code | 23.25\x1b[K\r⣽ code | 23.37\x1b[K\r⣻ code | 23.47\x1b[K\r⢿ code | 23.57\x1b[K\r⡿ code | 23.67\x1b[K\r⣟ code | 23.79\x1b[K\r⣯ code | 23.89\x1b[K\r⣷ code | 24.00\x1b[K\r⣾ code | 24.10\x1b[K\r⣽ code | 24.20\x1b[K\r⣻ code | 24.32\x1b[K\r⢿ code | 24.42\x1b[K\r⡿ code | 24.54\x1b[K\r⣟ code | 24.64\x1b[K\r⣯ code | 24.74\x1b[K\r⣷ code | 24.85\x1b[K\r⣾ code | 24.97\x1b[K\r⣽ code | 25.07\x1b[K\r⣻ code | 25.17\x1b[K\r⢿ code | 25.29\x1b[K\r⡿ code | 25.39\x1b[K\r⣟ code | 25.49\x1b[K\r⣯ code | 25.60\x1b[K\r⣷ code | 25.71\x1b[K\r⣾ code | 25.81\x1b[K\r⣽ code | 25.91\x1b[K\r⣻ code | 26.02\x1b[K\r⢿ code | 26.12\x1b[K\r⡿ code | 26.24\x1b[K\r⣟ code | 26.34\x1b[K\r⣯ code | 26.44\x1b[K\r⣷ code | 26.54\x1b[K\r⣾ code | 26.66\x1b[KEpisode 250: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 26.76\x1b[K\r⣻ code | 26.87\x1b[K\r⢿ code | 26.97\x1b[K\r⡿ code | 27.07\x1b[K\r⣟ code | 27.17\x1b[K\r⣯ code | 27.27\x1b[K\r⣷ code | 27.37\x1b[K\r⣾ code | 27.49\x1b[K\r⣽ code | 27.59\x1b[K\r⣻ code | 27.71\x1b[K\r⢿ code | 27.81\x1b[K\r⡿ code | 27.92\x1b[K\r⣟ code | 28.02\x1b[K\r⣯ code | 28.12\x1b[K\r⣷ code | 28.22\x1b[K\r⣾ code | 28.32\x1b[K\r⣽ code | 28.42\x1b[K\r⣻ code | 28.53\x1b[K\r⢿ code | 28.64\x1b[K\r⡿ code | 28.74\x1b[K\r⣟ code | 28.86\x1b[K\r⣯ code | 28.96\x1b[K\r⣷ code | 29.06\x1b[K\r⣾ code | 29.17\x1b[K\r⣽ code | 29.28\x1b[K\r⣻ code | 29.38\x1b[K\r⢿ code | 29.48\x1b[K\r⡿ code | 29.58\x1b[K\r⣟ code | 29.69\x1b[K\r⣯ code | 29.79\x1b[K\r⣷ code | 29.91\x1b[K\r⣾ code | 30.01\x1b[K\r⣽ code | 30.11\x1b[K\r⣻ code | 30.22\x1b[K\r⢿ code | 30.33\x1b[K\r⡿ code | 30.44\x1b[K\r⣟ code | 30.54\x1b[K\r⣯ code | 30.64\x1b[K\r⣷ code | 30.76\x1b[K\r⣾ code | 30.86\x1b[K\r⣽ code | 30.96\x1b[K\r⣻ code | 31.06\x1b[K\r⢿ code | 31.16\x1b[K\r⡿ code | 31.28\x1b[K\r⣟ code | 31.39\x1b[K\r⣯ code | 31.49\x1b[K\r⣷ code | 31.60\x1b[K\r⣾ code | 31.70\x1b[K\r⣽ code | 31.80\x1b[K\r⣻ code | 31.90\x1b[K\r⢿ code | 32.01\x1b[K\r⡿ code | 32.13\x1b[K\r⣟ code | 32.24\x1b[K\r⣯ code | 32.34\x1b[K\r⣷ code | 32.44\x1b[K\r⣾ code | 32.55\x1b[K\r⣽ code | 32.66\x1b[K\r⣻ code | 32.77\x1b[K\r⢿ code | 32.88\x1b[K\r⡿ code | 32.99\x1b[K\r⣟ code | 33.10\x1b[K\r⣯ code | 33.21\x1b[K\r⣷ code | 33.32\x1b[K\r⣾ code | 33.43\x1b[K\r⣽ code | 33.54\x1b[K\r⣻ code | 33.65\x1b[K\r⢿ code | 33.75\x1b[K\r⡿ code | 33.86\x1b[K\r⣟ code | 33.96\x1b[K\r⣯ code | 34.06\x1b[K\r⣷ code | 34.18\x1b[K\r⣾ code | 34.28\x1b[K\r⣽ code | 34.40\x1b[K\r⣻ code | 34.51\x1b[K\r⢿ code | 34.61\x1b[K\r⡿ code | 34.71\x1b[K\r⣟ code | 34.82\x1b[K\r⣯ code | 34.92\x1b[K\r⣷ code | 35.03\x1b[K\r⣾ code | 35.14\x1b[K\r⣽ code | 35.25\x1b[K\r⣻ code | 35.36\x1b[K\r⢿ code | 35.47\x1b[K\r⡿ code | 35.58\x1b[K\r⣟ code | 35.69\x1b[K\r⣯ code | 35.80\x1b[K\r⣷ code | 35.90\x1b[K\r⣾ code | 36.01\x1b[K\r⣽ code | 36.12\x1b[K\r⣻ code | 36.22\x1b[K\r⢿ code | 36.33\x1b[K\r⡿ code | 36.43\x1b[K\r⣟ code | 36.53\x1b[K\r⣯ code | 36.65\x1b[K\r⣷ code | 36.75\x1b[K\r⣾ code | 36.85\x1b[K\r⣽ code | 36.95\x1b[K\r⣻ code | 37.07\x1b[K\r⢿ code | 37.18\x1b[K\r⡿ code | 37.28\x1b[K\r⣟ code | 37.38\x1b[K\r⣯ code | 37.50\x1b[K\r⣷ code | 37.60\x1b[K\r⣾ code | 37.72\x1b[KEpisode 300: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 37.82\x1b[K\r⣻ code | 37.93\x1b[K\r⢿ code | 38.05\x1b[K\r⡿ code | 38.15\x1b[K\r⣟ code | 38.27\x1b[K\r⣯ code | 38.37\x1b[K\r⣷ code | 38.47\x1b[K\r⣾ code | 38.57\x1b[K\r⣽ code | 38.67\x1b[K\r⣻ code | 38.77\x1b[K\r⢿ code | 38.87\x1b[K\r⡿ code | 38.98\x1b[K\r⣟ code | 39.09\x1b[K\r⣯ code | 39.20\x1b[K\r⣷ code | 39.32\x1b[K\r⣾ code | 39.42\x1b[K\r⣽ code | 39.52\x1b[K\r⣻ code | 39.64\x1b[K\r⢿ code | 39.74\x1b[K\r⡿ code | 39.84\x1b[K\r⣟ code | 39.94\x1b[K\r⣯ code | 40.04\x1b[K\r⣷ code | 40.14\x1b[K\r⣾ code | 40.24\x1b[K\r⣽ code | 40.35\x1b[K\r⣻ code | 40.45\x1b[K\r⢿ code | 40.55\x1b[K\r⡿ code | 40.65\x1b[K\r⣟ code | 40.77\x1b[K\r⣯ code | 40.87\x1b[K\r⣷ code | 40.99\x1b[K\r⣾ code | 41.09\x1b[K\r⣽ code | 41.19\x1b[K\r⣻ code | 41.30\x1b[K\r⢿ code | 41.41\x1b[K\r⡿ code | 41.52\x1b[K\r⣟ code | 41.62\x1b[K\r⣯ code | 41.72\x1b[K\r⣷ code | 41.84\x1b[K\r⣾ code | 41.94\x1b[K\r⣽ code | 42.04\x1b[K\r⣻ code | 42.15\x1b[K\r⢿ code | 42.26\x1b[K\r⡿ code | 42.36\x1b[K\r⣟ code | 42.47\x1b[K\r⣯ code | 42.57\x1b[K\r⣷ code | 42.67\x1b[K\r⣾ code | 42.77\x1b[K\r⣽ code | 42.89\x1b[K\r⣻ code | 42.99\x1b[K\r⢿ code | 43.09\x1b[K\r⡿ code | 43.21\x1b[K\r⣟ code | 43.31\x1b[K\r⣯ code | 43.41\x1b[K\r⣷ code | 43.52\x1b[K\r⣾ code | 43.62\x1b[K\r⣽ code | 43.74\x1b[K\r⣻ code | 43.84\x1b[K\r⢿ code | 43.95\x1b[K\r⡿ code | 44.06\x1b[K\r⣟ code | 44.16\x1b[K\r⣯ code | 44.27\x1b[K\r⣷ code | 44.37\x1b[K\r⣾ code | 44.49\x1b[K\r⣽ code | 44.59\x1b[K\r⣻ code | 44.69\x1b[K\r⢿ code | 44.81\x1b[K\r⡿ code | 44.91\x1b[K\r⣟ code | 45.02\x1b[K\r⣯ code | 45.13\x1b[K\r⣷ code | 45.24\x1b[K\r⣾ code | 45.36\x1b[K\r⣽ code | 45.47\x1b[K\r⣻ code | 45.58\x1b[K\r⢿ code | 45.69\x1b[K\r⡿ code | 45.80\x1b[K\r⣟ code | 45.91\x1b[K\r⣯ code | 46.01\x1b[K\r⣷ code | 46.11\x1b[K\r⣾ code | 46.23\x1b[K\r⣽ code | 46.33\x1b[K\r⣻ code | 46.44\x1b[K\r⢿ code | 46.56\x1b[K\r⡿ code | 46.66\x1b[K\r⣟ code | 46.76\x1b[K\r⣯ code | 46.88\x1b[K\r⣷ code | 46.98\x1b[K\r⣾ code | 47.09\x1b[K\r⣽ code | 47.21\x1b[K\r⣻ code | 47.31\x1b[K\r⢿ code | 47.41\x1b[K\r⡿ code | 47.53\x1b[K\r⣟ code | 47.63\x1b[K\r⣯ code | 47.74\x1b[K\r⣷ code | 47.85\x1b[K\r⣾ code | 47.96\x1b[K\r⣽ code | 48.06\x1b[K\r⣻ code | 48.16\x1b[K\r⢿ code | 48.26\x1b[K\r⡿ code | 48.37\x1b[K\r⣟ code | 48.48\x1b[K\r⣯ code | 48.59\x1b[K\r⣷ code | 48.70\x1b[K\r⣾ code | 48.81\x1b[K\r⣽ code | 48.92\x1b[K\r⣻ code | 49.03\x1b[K\r⢿ code | 49.13\x1b[K\r⡿ code | 49.24\x1b[K\r⣟ code | 49.35\x1b[K\r⣯ code | 49.45\x1b[K\r⣷ code | 49.56\x1b[K\r⣾ code | 49.66\x1b[K\r⣽ code | 49.76\x1b[K\r⣻ code | 49.88\x1b[K\r⢿ code | 49.99\x1b[K\r⡿ code | 50.09\x1b[K\r⣟ code | 50.21\x1b[K\r⣯ code | 50.32\x1b[KEpisode 350: Total Reward = 200.0, Epsilon = 0.01\n\r⣷ code | 50.43\x1b[K\r⣾ code | 50.54\x1b[K\r⣽ code | 50.65\x1b[K\r⣻ code | 50.75\x1b[K\r⢿ code | 50.86\x1b[K\r⡿ code | 50.96\x1b[K\r⣟ code | 51.08\x1b[K\r⣯ code | 51.18\x1b[K\r⣷ code | 51.28\x1b[K\r⣾ code | 51.38\x1b[K\r⣽ code | 51.50\x1b[K\r⣻ code | 51.60\x1b[K\r⢿ code | 51.71\x1b[K\r⡿ code | 51.81\x1b[K\r⣟ code | 51.92\x1b[K\r⣯ code | 52.02\x1b[K\r⣷ code | 52.13\x1b[K\r⣾ code | 52.23\x1b[K\r⣽ code | 52.33\x1b[K\r⣻ code | 52.43\x1b[K\r⢿ code | 52.53\x1b[K\r⡿ code | 52.65\x1b[K\r⣟ code | 52.75\x1b[K\r⣯ code | 52.85\x1b[K\r⣷ code | 52.97\x1b[K\r⣾ code | 53.07\x1b[K\r⣽ code | 53.18\x1b[K\r⣻ code | 53.29\x1b[K\r⢿ code | 53.40\x1b[K\r⡿ code | 53.51\x1b[K\r⣟ code | 53.62\x1b[K\r⣯ code | 53.73\x1b[K\r⣷ code | 53.84\x1b[K\r⣾ code | 53.95\x1b[K\r⣽ code | 54.05\x1b[K\r⣻ code | 54.15\x1b[K\r⢿ code | 54.27\x1b[K\r⡿ code | 54.37\x1b[K\r⣟ code | 54.48\x1b[K\r⣯ code | 54.58\x1b[K\r⣷ code | 54.69\x1b[K\r⣾ code | 54.80\x1b[K\r⣽ code | 54.90\x1b[K\r⣻ code | 55.02\x1b[K\r⢿ code | 55.12\x1b[K\r⡿ code | 55.22\x1b[K\r⣟ code | 55.33\x1b[K\r⣯ code | 55.44\x1b[K\r⣷ code | 55.54\x1b[K\r⣾ code | 55.64\x1b[K\r⣽ code | 55.74\x1b[K\r⣻ code | 55.84\x1b[K\r⢿ code | 55.94\x1b[K\r⡿ code | 56.04\x1b[K\r⣟ code | 56.15\x1b[K\r⣯ code | 56.25\x1b[K\r⣷ code | 56.35\x1b[K\r⣾ code | 56.47\x1b[K\r⣽ code | 56.57\x1b[K\r⣻ code | 56.67\x1b[K\r⢿ code | 56.79\x1b[K\r⡿ code | 56.89\x1b[K\r⣟ code | 56.99\x1b[K\r⣯ code | 57.09\x1b[K\r⣷ code | 57.20\x1b[K\r⣾ code | 57.30\x1b[K\r⣽ code | 57.42\x1b[K\r⣻ code | 57.52\x1b[K\r⢿ code | 57.64\x1b[K\r⡿ code | 57.74\x1b[K\r⣟ code | 57.84\x1b[K\r⣯ code | 57.95\x1b[K\r⣷ code | 58.05\x1b[K\r⣾ code | 58.16\x1b[K\r⣽ code | 58.27\x1b[K\r⣻ code | 58.37\x1b[K\r⢿ code | 58.47\x1b[K\r⡿ code | 58.57\x1b[K\r⣟ code | 58.67\x1b[K\r⣯ code | 58.79\x1b[K\r⣷ code | 58.89\x1b[K\r⣾ code | 59.01\x1b[K\r⣽ code | 59.11\x1b[K\r⣻ code | 59.21\x1b[K\r⢿ code | 59.36\x1b[K\r⡿ code | 59.48\x1b[K\r⣟ code | 59.59\x1b[K\r⣯ code | 59.69\x1b[K\r⣷ code | 59.81\x1b[K\r⣾ code | 59.91\x1b[K\r⣽ code | 60.02\x1b[K\r⣻ code | 60.14\x1b[K\r⢿ code | 60.24\x1b[K\r⡿ code | 60.36\x1b[K\r⣟ code | 60.46\x1b[K\r⣯ code | 60.56\x1b[K\r⣷ code | 60.67\x1b[K\r⣾ code | 60.78\x1b[K\r⣽ code | 60.89\x1b[K\r⣻ code | 60.99\x1b[K\r⢿ code | 61.11\x1b[K\r⡿ code | 61.22\x1b[K\r⣟ code | 61.33\x1b[K\r⣯ code | 61.44\x1b[K\r⣷ code | 61.54\x1b[K\r⣾ code | 61.66\x1b[K\r⣽ code | 61.76\x1b[K\r⣻ code | 61.86\x1b[K\r⢿ code | 61.98\x1b[K\r⡿ code | 62.08\x1b[K\r⣟ code | 62.19\x1b[K\r⣯ code | 62.31\x1b[K\r⣷ code | 62.41\x1b[K\r⣾ code | 62.51\x1b[K\r⣽ code | 62.63\x1b[K\r⣻ code | 62.74\x1b[K\r⢿ code | 62.86\x1b[K\r⡿ code | 62.96\x1b[K\r⣟ code | 63.07\x1b[K\r⣯ code | 63.18\x1b[K\r⣷ code | 63.29\x1b[K\r⣾ code | 63.41\x1b[K\r⣽ code | 63.51\x1b[K\r⣻ code | 63.63\x1b[K\r⢿ code | 63.73\x1b[K\r⡿ code | 63.83\x1b[K\r⣟ code | 63.93\x1b[K\r⣯ code | 64.04\x1b[K\r⣷ code | 64.15\x1b[K\r⣾ code | 64.26\x1b[K\r⣽ code | 64.36\x1b[K\r⣻ code | 64.48\x1b[K\r⢿ code | 64.58\x1b[K\r⡿ code | 64.68\x1b[K\r⣟ code | 64.80\x1b[K\r⣯ code | 64.91\x1b[K\r⣷ code | 65.01\x1b[K\r⣾ code | 65.11\x1b[KEpisode 400: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 65.21\x1b[K\r⣻ code | 65.34\x1b[K\r⢿ code | 65.45\x1b[K\r⡿ code | 65.57\x1b[K\r⣟ code | 65.68\x1b[K\r⣯ code | 65.80\x1b[K\r⣷ code | 65.90\x1b[K\r⣾ code | 66.00\x1b[K\r⣽ code | 66.10\x1b[K\r⣻ code | 66.23\x1b[K\r⢿ code | 66.33\x1b[K\r⡿ code | 66.45\x1b[K\r⣟ code | 66.55\x1b[K\r⣯ code | 66.65\x1b[K\r⣷ code | 66.76\x1b[K\r⣾ code | 66.88\x1b[K\r⣽ code | 66.98\x1b[K\r⣻ code | 67.08\x1b[K\r⢿ code | 67.20\x1b[K\r⡿ code | 67.30\x1b[K\r⣟ code | 67.41\x1b[K\r⣯ code | 67.52\x1b[K\r⣷ code | 67.63\x1b[K\r⣾ code | 67.73\x1b[K\r⣽ code | 67.83\x1b[K\r⣻ code | 67.95\x1b[K\r⢿ code | 68.05\x1b[K\r⡿ code | 68.16\x1b[K\r⣟ code | 68.27\x1b[K\r⣯ code | 68.38\x1b[K\r⣷ code | 68.48\x1b[K\r⣾ code | 68.60\x1b[K\r⣽ code | 68.70\x1b[K\r⣻ code | 68.82\x1b[K\r⢿ code | 68.92\x1b[K\r⡿ code | 69.03\x1b[K\r⣟ code | 69.13\x1b[K\r⣯ code | 69.23\x1b[K\r⣷ code | 69.34\x1b[K\r⣾ code | 69.46\x1b[K\r⣽ code | 69.60\x1b[K\r⣻ code | 69.70\x1b[K\r⢿ code | 69.82\x1b[K\r⡿ code | 69.93\x1b[K\r⣟ code | 70.03\x1b[K\r⣯ code | 70.13\x1b[K\r⣷ code | 70.23\x1b[K\r⣾ code | 70.35\x1b[K\r⣽ code | 70.45\x1b[K\r⣻ code | 70.57\x1b[K\r⢿ code | 70.67\x1b[K\r⡿ code | 70.77\x1b[K\r⣟ code | 70.87\x1b[K\r⣯ code | 70.98\x1b[K\r⣷ code | 71.09\x1b[K\r⣾ code | 71.23\x1b[K\r⣽ code | 71.33\x1b[K\r⣻ code | 71.44\x1b[K\r⢿ code | 71.55\x1b[K\r⡿ code | 71.65\x1b[K\r⣟ code | 71.77\x1b[K\r⣯ code | 71.89\x1b[K\r⣷ code | 72.05\x1b[K\r⣾ code | 72.15\x1b[K\r⣽ code | 72.27\x1b[K\r⣻ code | 72.37\x1b[K\r⢿ code | 72.49\x1b[K\r⡿ code | 72.59\x1b[K\r⣟ code | 72.70\x1b[K\r⣯ code | 72.80\x1b[K\r⣷ code | 72.92\x1b[K\r⣾ code | 73.02\x1b[K\r⣽ code | 73.14\x1b[K\r⣻ code | 73.24\x1b[K\r⢿ code | 73.34\x1b[K\r⡿ code | 73.45\x1b[K\r⣟ code | 73.55\x1b[K\r⣯ code | 73.66\x1b[K\r⣷ code | 73.77\x1b[K\r⣾ code | 73.87\x1b[K\r⣽ code | 73.99\x1b[K\r⣻ code | 74.09\x1b[K\r⢿ code | 74.19\x1b[K\r⡿ code | 74.30\x1b[K\r⣟ code | 74.40\x1b[K\r⣯ code | 74.51\x1b[K\r⣷ code | 74.62\x1b[K\r⣾ code | 74.72\x1b[K\r⣽ code | 74.86\x1b[K\r⣻ code | 74.98\x1b[K\r⢿ code | 75.09\x1b[K\r⡿ code | 75.19\x1b[K\r⣟ code | 75.29\x1b[K\r⣯ code | 75.40\x1b[K\r⣷ code | 75.50\x1b[K\r⣾ code | 75.61\x1b[K\r⣽ code | 75.72\x1b[K\r⣻ code | 75.84\x1b[K\r⢿ code | 75.95\x1b[K\r⡿ code | 76.08\x1b[K\r⣟ code | 76.19\x1b[K\r⣯ code | 76.30\x1b[K\r⣷ code | 76.41\x1b[K\r⣾ code | 76.52\x1b[K\r⣽ code | 76.64\x1b[K\r⣻ code | 76.75\x1b[K\r⢿ code | 76.86\x1b[K\r⡿ code | 76.97\x1b[K\r⣟ code | 77.08\x1b[K\r⣯ code | 77.19\x1b[K\r⣷ code | 77.31\x1b[K\r⣾ code | 77.43\x1b[K\r⣽ code | 77.54\x1b[K\r⣻ code | 77.65\x1b[K\r⢿ code | 77.76\x1b[K\r⡿ code | 77.87\x1b[K\r⣟ code | 77.98\x1b[K\r⣯ code | 78.09\x1b[K\r⣷ code | 78.20\x1b[K\r⣾ code | 78.31\x1b[K\r⣽ code | 78.42\x1b[K\r⣻ code | 78.53\x1b[K\r⢿ code | 78.64\x1b[K\r⡿ code | 78.75\x1b[K\r⣟ code | 78.86\x1b[K\r⣯ code | 78.97\x1b[K\r⣷ code | 79.08\x1b[K\r⣾ code | 79.19\x1b[K\r⣽ code | 79.30\x1b[K\r⣻ code | 79.41\x1b[K\r⢿ code | 79.52\x1b[K\r⡿ code | 79.63\x1b[K\r⣟ code | 79.75\x1b[K\r⣯ code | 79.88\x1b[K\r⣷ code | 79.98\x1b[K\r⣾ code | 80.09\x1b[K\r⣽ code | 80.20\x1b[K\r⣻ code | 80.31\x1b[K\r⢿ code | 80.44\x1b[K\r⡿ code | 80.56\x1b[K\r⣟ code | 80.66\x1b[K\r⣯ code | 80.77\x1b[K\r⣷ code | 80.88\x1b[K\r⣾ code | 80.99\x1b[K\r⣽ code | 81.10\x1b[K\r⣻ code | 81.21\x1b[K\r⢿ code | 81.32\x1b[K\r⡿ code | 81.43\x1b[K\r⣟ code | 81.54\x1b[K\r⣯ code | 81.65\x1b[K\r⣷ code | 81.76\x1b[K\r⣾ code | 81.87\x1b[K\r⣽ code | 81.98\x1b[K\r⣻ code | 82.09\x1b[K\r⢿ code | 82.20\x1b[K\r⡿ code | 82.31\x1b[K\r⣟ code | 82.42\x1b[K\r⣯ code | 82.53\x1b[K\r⣷ code | 82.65\x1b[KEpisode 450: Total Reward = 200.0, Epsilon = 0.01\n\r⣾ code | 82.78\x1b[K\r⣽ code | 82.90\x1b[K\r⣻ code | 83.01\x1b[K\r⢿ code | 83.12\x1b[K\r⡿ code | 83.23\x1b[K\r⣟ code | 83.34\x1b[K\r⣯ code | 83.45\x1b[K\r⣷ code | 83.56\x1b[K\r⣾ code | 83.72\x1b[K\r⣽ code | 83.83\x1b[K\r⣻ code | 83.94\x1b[K\r⢿ code | 84.05\x1b[K\r⡿ code | 84.16\x1b[K\r⣟ code | 84.27\x1b[K\r⣯ code | 84.38\x1b[K\r⣷ code | 84.49\x1b[K\r⣾ code | 84.60\x1b[K\r⣽ code | 84.71\x1b[K\r⣻ code | 84.82\x1b[K\r⢿ code | 84.93\x1b[K\r⡿ code | 85.08\x1b[K\r⣟ code | 85.19\x1b[K\r⣯ code | 85.34\x1b[K\r⣷ code | 85.45\x1b[K\r⣾ code | 85.56\x1b[K\r⣽ code | 85.67\x1b[K\r⣻ code | 85.78\x1b[K\r⢿ code | 85.89\x1b[K\r⡿ code | 86.00\x1b[K\r⣟ code | 86.10\x1b[K\r⣯ code | 86.20\x1b[K\r⣷ code | 86.30\x1b[K\r⣾ code | 86.40\x1b[K\r⣽ code | 86.52\x1b[K\r⣻ code | 86.62\x1b[K\r⢿ code | 86.72\x1b[K\r⡿ code | 86.83\x1b[K\r⣟ code | 86.93\x1b[K\r⣯ code | 87.04\x1b[K\r⣷ code | 87.15\x1b[K\r⣾ code | 87.25\x1b[K\r⣽ code | 87.35\x1b[K\r⣻ code | 87.47\x1b[K\r⢿ code | 87.57\x1b[K\r⡿ code | 87.68\x1b[K\r⣟ code | 87.79\x1b[K\r⣯ code | 87.90\x1b[K\r⣷ code | 88.00\x1b[K\r⣾ code | 88.12\x1b[K\r⣽ code | 88.22\x1b[K\r⣻ code | 88.34\x1b[K\r⢿ code | 88.44\x1b[K\r⡿ code | 88.55\x1b[K\r⣟ code | 88.65\x1b[K\r⣯ code | 88.77\x1b[K\r⣷ code | 88.87\x1b[K\r⣾ code | 88.99\x1b[K\r⣽ code | 89.09\x1b[K\r⣻ code | 89.19\x1b[K\r⢿ code | 89.29\x1b[K\r⡿ code | 89.40\x1b[K\r⣟ code | 89.52\x1b[K\r⣯ code | 89.62\x1b[K\r⣷ code | 89.74\x1b[K\r⣾ code | 89.84\x1b[K\r⣽ code | 89.95\x1b[K\r⣻ code | 90.06\x1b[K\r⢿ code | 90.16\x1b[K\r⡿ code | 90.27\x1b[K\r⣟ code | 90.37\x1b[K\r⣯ code | 90.48\x1b[K\r⣷ code | 90.60\x1b[K\r⣾ code | 90.71\x1b[K\r⣽ code | 90.82\x1b[K\r⣻ code | 90.92\x1b[K\r⢿ code | 91.04\x1b[K\r⡿ code | 91.14\x1b[K\r⣟ code | 91.24\x1b[K\r⣯ code | 91.36\x1b[K\r⣷ code | 91.46\x1b[K\r⣾ code | 91.56\x1b[K\r⣽ code | 91.67\x1b[K\r⣻ code | 91.77\x1b[K\r⢿ code | 91.87\x1b[K\r⡿ code | 91.97\x1b[K\r⣟ code | 92.09\x1b[K\r⣯ code | 92.19\x1b[K\r⣷ code | 92.29\x1b[K\r⣾ code | 92.41\x1b[K\r⣽ code | 92.51\x1b[K\r⣻ code | 92.61\x1b[K\r⢿ code | 92.71\x1b[K\r⡿ code | 92.82\x1b[K\r⣟ code | 92.92\x1b[K\r⣯ code | 93.05\x1b[K\r⣷ code | 93.16\x1b[K\r⣾ code | 93.28\x1b[K\r⣽ code | 93.39\x1b[K\r⣻ code | 93.51\x1b[K\r⢿ code | 93.61\x1b[K\r⡿ code | 93.71\x1b[K\r⣟ code | 93.82\x1b[K\r⣯ code | 93.93\x1b[K\r⣷ code | 94.03\x1b[K\r⣾ code | 94.13\x1b[K\r⣽ code | 94.24\x1b[K\r⣻ code | 94.34\x1b[K\r⢿ code | 94.46\x1b[K\r⡿ code | 94.56\x1b[K\r⣟ code | 94.68\x1b[K\r⣯ code | 94.78\x1b[K\r⣷ code | 94.89\x1b[K\r⣾ code | 94.99\x1b[K\r⣽ code | 95.11\x1b[K\r⣻ code | 95.21\x1b[K\r⢿ code | 95.31\x1b[K\r⡿ code | 95.43\x1b[K\r⣟ code | 95.53\x1b[K\r⣯ code | 95.63\x1b[K\r⣷ code | 95.74\x1b[K\r⣾ code | 95.84\x1b[K\r⣽ code | 95.94\x1b[K\r⣻ code | 96.04\x1b[K\r⢿ code | 96.16\x1b[K\r⡿ code | 96.26\x1b[K\r⣟ code | 96.36\x1b[K\r⣯ code | 96.46\x1b[K\r⣷ code | 96.58\x1b[K\r⣾ code | 96.68\x1b[K\r⣽ code | 96.78\x1b[K\r⣻ code | 96.88\x1b[K\r⢿ code | 96.99\x1b[K\r⡿ code | 97.10\x1b[K\r⣟ code | 97.20\x1b[K\r⣯ code | 97.31\x1b[K\r⣷ code | 97.41\x1b[K\r⣾ code | 97.51\x1b[K\r⣽ code | 97.63\x1b[K\r⣻ code | 97.73\x1b[K\r⢿ code | 97.85\x1b[K\r⡿ code | 97.95\x1b[K\r⣟ code | 98.06\x1b[K\r⣯ code | 98.16\x1b[K\r⣷ code | 98.28\x1b[K\r⣾ code | 98.38\x1b[K\r⣽ code | 98.48\x1b[K\r⣻ code | 98.60\x1b[K\r⢿ code | 98.70\x1b[K\r⡿ code | 98.80\x1b[K\r⣟ code | 98.90\x1b[K\r⣯ code | 99.01\x1b[K\r⣷ code | 99.11\x1b[K\r⣾ code | 99.23\x1b[K\r⣽ code | 99.33\x1b[K\r⣻ code | 99.45\x1b[K\r⢿ code | 99.55\x1b[K\r⡿ code | 99.65\x1b[K\r⣟ code | 99.76\x1b[K\r⣯ code | 99.87\x1b[K\r⣷ code | 99.98\x1b[K\r⣾ code | 100.08\x1b[K\r⣽ code | 100.18\x1b[K\r⣻ code | 100.30\x1b[K\r⢿ code | 100.40\x1b[K\r⡿ code | 100.50\x1b[K\r⣟ code | 100.60\x1b[K\r⣯ code | 100.72\x1b[K\r⣷ code | 100.82\x1b[K\r⣾ code | 100.92\x1b[K\r⣽ code | 101.03\x1b[K\r⣻ code | 101.13\x1b[K\r⢿ code | 101.23\x1b[K\r⡿ code | 101.33\x1b[K\r⣟ code | 101.45\x1b[K\r⣯ code | 101.55\x1b[K\r⣷ code | 101.65\x1b[K\r⣾ code | 101.76\x1b[K\r⣽ code | 101.87\x1b[K\r⣻ code | 101.98\x1b[K\r⢿ code | 102.08\x1b[K\r⡿ code | 102.18\x1b[K\r⣟ code | 102.30\x1b[K\r⣯ code | 102.40\x1b[K\r⣷ code | 102.50\x1b[K\r⣾ code | 102.62\x1b[K\r⣽ code | 102.72\x1b[K\r⣻ code | 102.83\x1b[K\r⢿ code | 102.93\x1b[K\r⡿ code | 103.04\x1b[K\r⣟ code | 103.15\x1b[K\r⣯ code | 103.25\x1b[K\r⣷ code | 103.35\x1b[K\r⣾ code | 103.47\x1b[K\r⣽ code | 103.57\x1b[K\r⣻ code | 103.67\x1b[K\r⢿ code | 103.78\x1b[K\r⡿ code | 103.89\x1b[K\r⣟ code | 103.99\x1b[K\r⣯ code | 104.10\x1b[K\r⣷ code | 104.20\x1b[K\r⣾ code | 104.30\x1b[K\r⣽ code | 104.40\x1b[K\r⣻ code | 104.50\x1b[K\r⢿ code | 104.62\x1b[K\r⡿ code | 104.72\x1b[K\r⣟ code | 104.82\x1b[K\r⣯ code | 104.92\x1b[K\r⣷ code | 105.02\x1b[K\r⣾ code | 105.13\x1b[K\r⣽ code | 105.24\x1b[K\r⣻ code | 105.35\x1b[K\r⢿ code | 105.45\x1b[K\r⡿ code | 105.55\x1b[K\r⣟ code | 105.65\x1b[K\r⣯ code | 105.75\x1b[K\r⣷ code | 105.85\x1b[K\r⣾ code | 105.95\x1b[K\r⣽ code | 106.05\x1b[K\r⣻ code | 106.17\x1b[K\r⢿ code | 106.27\x1b[K\r⡿ code | 106.39\x1b[K\r⣟ code | 106.49\x1b[K\r⣯ code | 106.59\x1b[K\r⣷ code | 106.69\x1b[K\r⣾ code | 106.80\x1b[K\r⣽ code | 106.90\x1b[K\r⣻ code | 107.01\x1b[K\r⢿ code | 107.11\x1b[K\r⡿ code | 107.22\x1b[K\r⣟ code | 107.32\x1b[K\r⣯ code | 107.42\x1b[K\r⣷ code | 107.52\x1b[K\r⣾ code | 107.64\x1b[K\r⣽ code | 107.74\x1b[K\r⣻ code | 107.85\x1b[K\r⢿ code | 107.96\x1b[K\r⡿ code | 108.06\x1b[K\r⣟ code | 108.17\x1b[K\r⣯ code | 108.27\x1b[K\r⣷ code | 108.37\x1b[K\r⣾ code | 108.49\x1b[K\r⣽ code | 108.59\x1b[K\r⣻ code | 108.69\x1b[K\r⢿ code | 108.79\x1b[K\r⡿ code | 108.91\x1b[K\r⣟ code | 109.01\x1b[K\r⣯ code | 109.11\x1b[K\r⣷ code | 109.22\x1b[K\r⣾ code | 109.33\x1b[K\r⣽ code | 109.44\x1b[K\r⣻ code | 109.54\x1b[K\r⢿ code | 109.64\x1b[K\r⡿ code | 109.74\x1b[K\r⣟ code | 109.84\x1b[K\r⣯ code | 109.94\x1b[K\r⣷ code | 110.06\x1b[K\r⣾ code | 110.16\x1b[K\r⣽ code | 110.27\x1b[K\r⣻ code | 110.38\x1b[K\r⢿ code | 110.48\x1b[K\r⡿ code | 110.58\x1b[K\r⣟ code | 110.69\x1b[K\nstderr:<string>:51: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\builder\\windows\\pytorch\\torch\\csrc\\utils\\tensor_new.cpp:281.)\n"
    #cleaned_result = super_strip(asd)
    #print(type(cleaned_result), len(asd), len(cleaned_result))
    #print(cleaned_result)

