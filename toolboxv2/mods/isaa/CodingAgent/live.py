import ast
import asyncio
import importlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass

### ---- Styles ------- ###
from inspect import (
    currentframe,
    signature,
)
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import toolboxv2
from toolboxv2 import Spinner, get_app


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
                text=True, check=True
            )

            if result.returncode != 0:
                return f"Error creating project: {result.stderr}"

            self.current_project = project_path
            return f"Created new project at {project_path}"

        except Exception as e:
            return f"Failed to create project: {str(e)}"

    async def add_dependency(self, name: str, version: str | None = None) -> str:
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
                text=True,check=True
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
                text=True, check=True
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

            with open(file_path) as f:
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
            with open(session_file) as f:
                state = json.load(f)
                self.output_history = state['output_history']
                self.current_project = Path(state['current_project']) if state['current_project'] else None
### ---- logic ---- ###

class VirtualFileSystem:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.current_dir = base_dir
        self.virtual_files: dict[str, str] = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def ifile_exists(self, filepath: str | Path) -> bool:
        """Check if a file exists"""
        abs_path = self._resolve_path(filepath)
        return abs_path.exists()

    def overwrite_lines(self, filepath: str | Path, new_content: str, lines: str = "") -> str:
        """Overwrite lines in a file"""
        try:
            content = self.read_file(filepath)
            content_lines = content.split('\n')
            start, end = map(int, lines.split('-'))
            # overwrite specif lines with new content keep the rest
            content_before = content_lines[:start-1]
            content_after = content_lines[end:]
            content_lines = content_before + new_content.split('\n') + content_after
            content = '\n'.join(content_lines)
            return self.write_file(filepath, content)
        except Exception as e:
            return f"Overwrite lines failed: {str(e)}"

    def write_file(self, filepath: str | Path, content: str) -> Path:
        """Write content to a virtual file and persist to disk using UTF-8"""
        try:
            abs_path = self._resolve_path(filepath)
        except ValueError:
            print("invalid :", filepath)
            filepath = "src/temp/_temp_fix.py"
            abs_path = self._resolve_path(filepath)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Store in virtual filesystem
        rel_path = str(abs_path.relative_to(self.base_dir))
        self.virtual_files[rel_path] = content

        # Write to actual filesystem with UTF-8 encoding
        with open(abs_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(content)

        parent_dir_str = str(abs_path.parent.absolute())
        if parent_dir_str not in sys.path and abs_path.suffix == '.py':
            sys.path.insert(0, parent_dir_str)

        return abs_path

    def read_file(self, filepath: str | Path) -> str:
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

    def delete_file(self, filepath: str | Path):
        """Delete a virtual file"""
        abs_path = self._resolve_path(filepath)
        rel_path = str(abs_path.relative_to(self.base_dir))

        if rel_path in self.virtual_files:
            del self.virtual_files[rel_path]

        if abs_path.exists():
            abs_path.unlink()

    def create_directory(self, dirpath: str | Path):
        """Create a new directory"""
        abs_path = self._resolve_path(dirpath)
        abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path

    def list_files(self, dirpath: str | Path = '.') -> list:
        """List files in a directory"""
        abs_path = self._resolve_path(dirpath)
        if not abs_path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        return [p.name for p in abs_path.iterdir() if p.is_file()]

    def list_directory(self, dirpath: str | Path = '.') -> list:
        """List contents of a directory"""
        abs_path = self._resolve_path(dirpath)
        if not abs_path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        return [p.name for p in abs_path.iterdir()]

    def change_directory(self, dirpath: str | Path):
        """Change current working directory"""
        new_dir = self._resolve_path(dirpath)
        if not new_dir.exists() or not new_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {dirpath}")
        self.current_dir = new_dir

    def _resolve_path(self, filepath: str | Path) -> Path:
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

    def print_file_structure(self, start_path: str | Path = '.', indent: str = ''):
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
            if isinstance(parent, ast.AsyncFunctionDef | ast.FunctionDef):
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
        self.user_ns: dict[str, Any] = {}
        # import nest_asyncio
        # nest_asyncio.apply()
        # Set up virtual environment if it doesn't exist
        with Spinner("Starting virtual environment"):
            self._setup_venv()
        self.reset()

    def _virtual_open(self, filepath, mode='r', *args, **kwargs):
        """Custom open function that uses virtual filesystem and makes files available for import"""
        try:
            abs_path = self.vfs._resolve_path(filepath)
        except ValueError:
            # If path resolution fails, try to resolve relative to current working directory
            abs_path = self.vfs.base_dir / filepath

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
                    content = real_file.read()
                    self.vfs.virtual_files[rel_path] = content
                    real_file.seek(0)
                except UnicodeDecodeError:
                    # Handle binary files
                    pass

        return real_file

    def _setup_venv(self):
        """Create virtual environment if it doesn't exist"""
        if not self._venv_path.exists():
            try:
                subprocess.run([sys.executable, "-m", "venv", str(self._venv_path)], check=True)
            except subprocess.CalledProcessError as e:
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
            '__file__': None,
            '__path__': [str(self.vfs.current_dir)],
            'auto_install': auto_install,
            'app': get_app(),
            'modify_code': self.modify_code,
            'open': self._virtual_open,
        }
        self.output_history.clear()
        self._execution_count = 0
        if self.auto_remove:
            shutil.rmtree(self.vfs.base_dir, ignore_errors=True)

    def get_namespace(self) -> dict[str, Any]:
        """Get current namespace"""
        return self.user_ns.copy()

    def update_namespace(self, variables: dict[str, Any]):
        """Update namespace with new variables"""
        self.user_ns.update(variables)

    @staticmethod
    def _parse_code(code: str) -> tuple[Any, Any | None, bool, bool]:
        """Parse code and handle top-level await"""
        code_ = ""
        for line in code.split('\n'):
            if line.strip().startswith('#'):
                continue
            if line.strip().startswith('asyncio.run('):
                line = (' ' *(len(line) - len(line.strip()))) + 'await ' + line.strip()[len('asyncio.run('):-1]
            code_ += line + '\n'
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
                wrapped_code += "    result = None\n"
                # add try:
                wrapped_code +="    try:\n"
                # Indent the original code
                wrapped_code += "\n".join(f"        {line}" for line in code.splitlines())
                # Add return statement for last expression
                wrapped_code +="\n    except Exception as e:\n"
                wrapped_code +="        import traceback\n"
                wrapped_code +="        print(traceback.format_exc())\n"
                wrapped_code +="        raise e\n"
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

            error_msg += traceback.format_exc()

            raise SyntaxError(error_msg) from e

    async def run_cell(self, code: str, live_output: bool = True) -> Any:
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
                    os.makedirs(str(temp_file.parent.absolute()), exist_ok=True)
                    os.chdir(str(temp_file.parent.absolute()))
                    self.user_ns['PYTHON_EXEC'] = python_exec

                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        if has_top_level_await:
                            try:
                                # Execute wrapped code and await it
                                exec(exec_code, self.user_ns)
                                result = self.user_ns['__wrapper']()
                                if asyncio.iscoroutine(result):
                                    result = await result
                            finally:
                                self.user_ns.pop('__wrapper', None)
                        elif is_async:
                            # Execute async code
                            exec(exec_code, self.user_ns)
                            if eval_code:
                                result = eval(eval_code, self.user_ns)
                                if asyncio.iscoroutine(result):
                                    result = await result
                        else:
                            # Execute sync code
                            exec(exec_code, self.user_ns)
                            if eval_code:
                                result = eval(eval_code, self.user_ns)

                        if result is not None:
                            self.user_ns['_'] = result
                except KeyboardInterrupt:
                    print("Stop execution manuel!")

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

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            if live_output:
                sys.__stderr__.write(error_msg)
            return error_msg

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
                                        method_pattern = fr"def {method_name}.*?:(.*?)(?=\n    \w|\n\w|\Z)"
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
                        func_pattern = fr"def {clean_object_name}.*?:(.*?)(?=\n\w|\Z)"
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
                        var_pattern = fr"{clean_object_name}\s*=.*"
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
            except Exception:
                user_ns[key] = f"not serializable: {str(value)}"

        for key, value in output_history.items():
            try:
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str):
                            value[k] = v.encode('utf-8').decode('utf-8')
                pickle.dumps(value)
            except Exception:
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
                # self.user_ns.update(session_data['user_ns'])
                self.output_history.update(session_data['output_history'])

        # Load VFS state with UTF-8 encoding
        vfs_state_file = self._session_dir / f"{name}_vfs.json"
        if vfs_state_file.exists():
            with open(vfs_state_file, encoding='utf-8') as f:
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

    def set_base_directory(self, path: str) -> str:
        """
        Set the base directory for the virtual file system and add it to sys.path for imports.

        Args:
            path: New base directory path

        Returns:
            Success message
        """
        try:
            new_path = Path(path) if isinstance(path, str) else path
            new_path.mkdir(parents=True, exist_ok=True)

            # Remove old base directory from sys.path if it exists
            old_base_str = str(self.vfs.base_dir)
            if old_base_str in sys.path:
                sys.path.remove(old_base_str)

            # Update VFS base directory
            self.vfs.base_dir = new_path
            self.vfs.current_dir = new_path

            # Add new base directory to sys.path for imports
            new_base_str = str(new_path)
            if new_base_str not in sys.path:
                sys.path.insert(0, new_base_str)

            # Update user namespace paths
            self.user_ns['__path__'] = [new_base_str]

            return f"Base directory set to: {new_path} (added to sys.path)"

        except Exception as e:
            return f"Set base directory error: {str(e)}"


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
            main_file = next((f for f in files if f.endswith('.rs')), None)
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


from pathlib import Path
from typing import Any


class ToolsInterface:
    """
    Minimalistic tools interface for LLMs providing code execution,
    virtual file system, and browser interaction capabilities.
    """

    def __init__(self,
                 session_dir: str | None = None,
                 auto_remove: bool = True,
                 variables: dict[str, Any] | None = None,
                 variable_manager: Any | None = None):
        """
        Initialize the tools interface.

        Args:
            session_dir: Directory for session storage
            auto_remove: Whether to auto-remove temporary files
            variables: Initial variables dictionary
            variable_manager: External variable manager instance
            web_llm: LLM model for web interactions
        """
        self._session_dir = Path(session_dir) if session_dir else Path(get_app().appdata) / '.tools_sessions'
        self._session_dir.mkdir(exist_ok=True)
        self.auto_remove = auto_remove
        self.variable_manager = variable_manager

        # Initialize Python execution environment
        self.ipython = MockIPython(self._session_dir, auto_remove=auto_remove)
        if variables:
            self.ipython.user_ns.update(variables)

        # Initialize virtual file system
        self.vfs = VirtualFileSystem(self._session_dir / 'virtual_fs')

        # Initialize Rust interface
        self.cargo = CargoRustInterface(self._session_dir, auto_remove=auto_remove)

        # Track execution state
        self._execution_history = []
        self._current_file = None

    async def execute_python(self, code: str) -> str:
        """
        Execute Python code in the virtual environment.

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        try:
            result = await self.ipython.run_cell(code, live_output=False)

            # Update variable manager if available
            if self.variable_manager:
                for key, value in self.ipython.user_ns.items():
                    if not key.startswith('_') and key not in ['__name__', '__builtins__']:
                        try:
                            self.variable_manager.set(f"python.{key}", value)
                        except:
                            pass  # Ignore non-serializable variables

            self._execution_history.append(('python', code, result))
            return str(result) if result else "Execution completed"

        except Exception as e:
            error_msg = f"Python execution error: {str(e)}\n{traceback.format_exc()}"
            self._execution_history.append(('python', code, error_msg))
            return error_msg

    async def execute_rust(self, code: str) -> str:
        """
        Execute Rust code using Cargo.

        Args:
            code: Rust code to execute

        Returns:
            Execution result as string
        """
        try:
            # Setup project if needed
            if not self.cargo.current_project:
                await self.cargo.setup_project("temp_rust_project")

            result = await self.cargo.run_code(code)
            self._execution_history.append(('rust', code, result))
            return result

        except Exception as e:
            error_msg = f"Rust execution error: {str(e)}"
            self._execution_history.append(('rust', code, error_msg))
            return error_msg

    async def write_file(self, filepath: str, content: str, lines: str = "") -> str:
        """
        Write content to a file in the virtual file system.

        Args:
            filepath: Path to the file
            content: Content to write
            lines: Optional line range to write (e.g., "1-3" for lines 1 to 3)

        Returns:
            Success message
        """

        try:
            if lines:
                abs_path = self.vfs.overwrite_lines(filepath, content, lines)
            else:
                abs_path = self.vfs.write_file(filepath, content)

            # Update variable manager if available
            if self.variable_manager:
                self.variable_manager.set(f"files.{filepath.replace('/', '.')}", {
                    'path': str(abs_path),
                    'size': len(content),
                    'content_preview': content[:100] + '...' if len(content) > 100 else content
                })

            return f"File written successfully: {abs_path}"

        except Exception as e:
            return f"File write error: {str(e)}"

    async def replace_in_file(self, filepath: str, old_content: str, new_content: str, precise: bool = True) -> str:
        """
        Replace exact content in file with new content.

        Args:
            filepath: Path to the file
            old_content: Exact content to replace (empty string for insertion at start)
            new_content: Content to replace with
            precise: If True, requires exact match; if False, allows single occurrence replacement

        Returns:
            Success message or error
        """
        try:
            # Read current file content
            try:
                current_content = self.vfs.read_file(filepath)
            except:
                return f"Error: File '{filepath}' not found or cannot be read"

            # Handle insertion at start (empty old_content)
            if not old_content:
                updated_content = new_content + current_content
                self.vfs.write_file(filepath, updated_content)
                return f"Content inserted at start of '{filepath}'"

            # Check if old_content exists
            if old_content not in current_content:
                return f"Error: Old content not found in '{filepath}' use read_file to check."

            # Count occurrences
            occurrences = current_content.count(old_content)

            if precise and occurrences > 1:
                return f"Error: Found {occurrences} occurrences of old content. Use precise=False to replace first occurrence."

            # Replace content (first occurrence if multiple)
            updated_content = current_content.replace(old_content, new_content, 1)

            # Write updated content
            self.vfs.write_file(filepath, updated_content)

            return f"Successfully replaced content in '{filepath}' ({occurrences} occurrence{'s' if occurrences > 1 else ''} found, 1 replaced)"

        except Exception as e:
            return f"Replace error: {str(e)}"

    async def read_file(self, filepath: str, lines: str="") -> str:
        """
        Read content from a file in the virtual file system.

        Args:
            filepath: Path to the file
            lines: Optional line range to read (e.g., "1-3" for lines 1 to 3)

        Returns:
            File content or error message
        """
        try:
            content = self.vfs.read_file(filepath)

            if lines:
                start, end = map(int, lines.split('-'))
                content = '\n'.join(content.split('\n')[start-1:end])
            # Update variable manager if available
            if self.variable_manager:
                self.variable_manager.set("files.last_read", {
                    'path': filepath,
                    'size': len(content),
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                })

            return content

        except Exception as e:
            return f"File read error: {str(e)}"

    async def list_files(self, dirpath: str = '.') -> str:
        """
        List files in a directory.

        Args:
            dirpath: Directory path to list

        Returns:
            File listing as string
        """
        try:
            files = self.vfs.list_files(dirpath)
            listing = "\n".join(f"- {file}" for file in files)
            return f"Files in '{dirpath}':\n{listing}"

        except Exception as e:
            return f"File listing error: {str(e)}"

    async def list_directory(self, dirpath: str = '.') -> str:
        """
        List contents of a directory.

        Args:
            dirpath: Directory path to list

        Returns:
            Directory listing as string
        """
        try:
            contents = self.vfs.list_directory(dirpath)
            listing = "\n".join(f"- {item}" for item in contents)

            # Update variable manager if available
            if self.variable_manager:
                self.variable_manager.set("files.last_listing", {
                    'directory': dirpath,
                    'items': contents,
                    'count': len(contents)
                })

            return f"Directory '{dirpath}' contents:\n{listing}"

        except Exception as e:
            return f"Directory listing error: {str(e)}"


    async def create_directory(self, dirpath: str) -> str:
        """
        Create a new directory.

        Args:
            dirpath: Path of directory to create

        Returns:
            Success message
        """
        try:
            abs_path = self.vfs.create_directory(dirpath)
            return f"Directory created successfully: {abs_path}"

        except Exception as e:
            return f"Directory creation error: {str(e)}"

    def set_base_directory(self, path: str) -> str:
        """
        Set the base directory for the virtual file system.

        Args:
            path: New base directory path

        Returns:
            Success message
        """
        try:
            new_path = Path(path) if isinstance(path, str) else path
            new_path = new_path.absolute()
            print(f"New path: {new_path}")
            new_path.mkdir(parents=True, exist_ok=True)
            self.vfs.base_dir = new_path
            self.vfs.current_dir = new_path

            # Update MockIPython base directory and sys.path
            result = self.ipython.set_base_directory(path)

            return result

        except Exception as e:
            return f"Set base directory error: {str(e)}"

    async def set_current_file(self, filepath: str) -> str:
        """
        Set the current file for Python execution context.

        Args:
            filepath: Path to set as current file

        Returns:
            Success message
        """
        try:
            abs_path = self.vfs._resolve_path(filepath)
            self.ipython.user_ns['__file__'] = str(abs_path)
            self._current_file = str(abs_path)

            return f"Current file set to: {abs_path}"

        except Exception as e:
            return f"Set current file error: {str(e)}"

    async def install_package(self, package_name: str, version: str | None = None) -> str:
        """
        Install a Python package in the virtual environment.

        Args:
            package_name: Name of the package to install
            version: Optional specific version to install

        Returns:
            Installation result
        """
        try:
            code = f"""
auto_install('{package_name}'{f", version='{version}'" if version else ""})
import {package_name.split('[')[0]}  # Import base package name
print(f"Successfully imported {package_name}")
"""
            result = await self.execute_python(code)
            return result

        except Exception as e:
            return f"Package installation error: {str(e)}"

    async def get_execution_history(self) -> str:
        """
        Get the execution history.

        Returns:
            Execution history as formatted string
        """
        if not self._execution_history:
            return "No execution history available."

        history_lines = []
        for i, (lang, code, result) in enumerate(self._execution_history[-10:], 1):
            history_lines.append(f"[{i}] {lang.upper()}:")
            history_lines.append(f"    Code: {code[:100]}..." if len(code) > 100 else f"    Code: {code}")
            history_lines.append(
                f"    Result: {str(result)[:200]}..." if len(str(result)) > 200 else f"    Result: {result}")
            history_lines.append("")

        return "\n".join(history_lines)

    async def clear_session(self) -> str:
        """
        Clear the current session (variables, history, files).

        Returns:
            Success message
        """
        try:
            # Reset Python environment
            self.ipython.reset()

            # Clear execution history
            self._execution_history.clear()

            # Clear VFS if auto_remove is enabled
            if self.auto_remove:
                shutil.rmtree(self.vfs.base_dir, ignore_errors=True)
                self.vfs.base_dir.mkdir(parents=True, exist_ok=True)
                self.vfs.virtual_files.clear()

            # Reset current file
            self._current_file = None

            return "Session cleared successfully"

        except Exception as e:
            return f"Clear session error: {str(e)}"

    async def get_variables(self) -> str:
        """
        Get current variables in JSON format.

        Returns:
            Variables as JSON string
        """
        try:
            # Get Python variables
            py_vars = {}
            for key, value in self.ipython.user_ns.items():
                if not key.startswith('_') and key not in ['__name__', '__builtins__']:
                    try:
                        # Try to serialize the value
                        json.dumps(value, default=str)
                        py_vars[key] = str(value)[:200] if len(str(value)) > 200 else value
                    except:
                        py_vars[key] = f"<{type(value).__name__}>"

            result = {
                'python_variables': py_vars,
                'current_file': self._current_file,
                'vfs_base': str(self.vfs.base_dir),
                'execution_count': len(self._execution_history)
            }

            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            return f"Get variables error: {str(e)}"

    def get_tools(self, name:str=None) -> list[tuple[Any, str, str]]:
        """
        Get all available tools as list of tuples (function, name, description).

        Returns:
            List of tool tuples
        """
        tools = [
            # Code execution tools
            (self.execute_python, "execute_python",
             "Execute Python code in virtual environment. all variables ar available under the python scope.\n"
             "The isaa_instance is available as isaa_instance in the python code."
             " Args: code (str) -> str"),

            # (self.execute_rust, "execute_rust",
            #  "Execute Rust code using Cargo. Args: code (str) -> str"),

            # File system tools
            (self.write_file, "write_file",
             "Write content to file in virtual filesystem. lines is a string with the line range to write (e.g., '1-3' for lines 1 to 3) Args: filepath (str), content (str), lines (str) = '' -> str"),

            (self.write_file, "create_file",
             "Write content to file in virtual filesystem.  Args: filepath (str), content (str) -> str"),

            (self.replace_in_file, "replace_in_file",
             "Replace exact content in file. Args: filepath (str), old_content (str), new_content (str), precise (bool) = True -> str"),

            (self.read_file, "read_file",
             "Read content from file in virtual filesystem. lines is a string with the line range to read (e.g., '1-3' for lines 1 to 3) Args: filepath (str), lines (str) = '' -> str"),

            (self.list_files, "list_files",
             "List files in directory. Args: dirpath (str) = '.' -> str"),

            (self.list_directory, "list_directory",
             "List directory contents. Args: dirpath (str) = '.' -> str"),

            (self.create_directory, "create_directory",
             "Create new directory. Args: dirpath (str) -> str"),

            # Configuration tools
            (self.set_base_directory, "set_base_directory",
             "Set base directory for virtual filesystem. Args: path (str) -> str"),

            (self.set_current_file, "set_current_file",
             "Set current file for Python execution context. Args: filepath (str) -> str"),

            (self.install_package, "install_package",
             "Install Python package. Args: package_name (str), version (Optional[str]) -> str"),

            # Session management tools
            (self.get_execution_history, "get_execution_history",
             "Get execution history. Args: None -> str"),

            (self.clear_session, "clear_session",
             "Clear current session. Args: None -> str"),

            (self.get_variables, "get_variables",
             "Get current variables as JSON. Args: None -> str"),
        ]
        if name is not None:
            tools = [t for t in tools if t[1] == name][0]
        return tools

    def __aenter__(self):
        return self

    async def __aexit__(self, *exe):
        await asyncio.sleep(0.01)

### -- extra -- ###

@dataclass
class SyncReport:
    """Report of variables synced from namespace to pipeline"""
    added: dict[str, str]
    skipped: dict[str, str]  # var_name -> reason
    errors: dict[str, str]  # var_name -> error message

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
    namespace: dict[str, Any] | None = None,
    prefix: str | None = None,
    include_types: type | list[type] | None = None,
    exclude_patterns: list[str] | None = None,
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
    if include_types and not isinstance(include_types, list | tuple | set):
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

        print("Result:", result)


    # Run the async function
    asyncio.run(run_code())

    #asd = "Evaluation: Output -> \nstdout:Episode 0: Total Reward = 35.0, Epsilon = 0.98\n\r⣾ code | 0.04\x1b[K\r⣽ code | 0.15\x1b[K\r⣻ code | 0.26\x1b[K\r⢿ code | 0.37\x1b[KEpisode 50: Total Reward = 10.0, Epsilon = 0.06\n\r⡿ code | 0.48\x1b[K\r⣟ code | 0.59\x1b[K\r⣯ code | 0.70\x1b[K\r⣷ code | 0.81\x1b[K\r⣾ code | 0.92\x1b[K\r⣽ code | 1.03\x1b[K\r⣻ code | 1.14\x1b[K\r⢿ code | 1.25\x1b[K\r⡿ code | 1.37\x1b[K\r⣟ code | 1.47\x1b[K\r⣯ code | 1.58\x1b[K\r⣷ code | 1.70\x1b[K\r⣾ code | 1.81\x1b[K\r⣽ code | 1.92\x1b[K\r⣻ code | 2.03\x1b[KEpisode 100: Total Reward = 58.0, Epsilon = 0.01\n\r⢿ code | 2.14\x1b[K\r⡿ code | 2.25\x1b[K\r⣟ code | 2.36\x1b[K\r⣯ code | 2.47\x1b[K\r⣷ code | 2.58\x1b[K\r⣾ code | 2.69\x1b[K\r⣽ code | 2.80\x1b[K\r⣻ code | 2.91\x1b[K\r⢿ code | 3.02\x1b[K\r⡿ code | 3.13\x1b[K\r⣟ code | 3.24\x1b[K\r⣯ code | 3.35\x1b[K\r⣷ code | 3.46\x1b[K\r⣾ code | 3.57\x1b[K\r⣽ code | 3.68\x1b[K\r⣻ code | 3.79\x1b[K\r⢿ code | 3.90\x1b[K\r⡿ code | 4.01\x1b[K\r⣟ code | 4.12\x1b[K\r⣯ code | 4.23\x1b[K\r⣷ code | 4.34\x1b[K\r⣾ code | 4.45\x1b[K\r⣽ code | 4.56\x1b[K\r⣻ code | 4.67\x1b[K\r⢿ code | 4.79\x1b[K\r⡿ code | 4.90\x1b[K\r⣟ code | 5.01\x1b[K\r⣯ code | 5.12\x1b[K\r⣷ code | 5.23\x1b[K\r⣾ code | 5.34\x1b[K\r⣽ code | 5.45\x1b[K\r⣻ code | 5.56\x1b[K\r⢿ code | 5.67\x1b[K\r⡿ code | 5.78\x1b[K\r⣟ code | 5.89\x1b[K\r⣯ code | 6.00\x1b[K\r⣷ code | 6.11\x1b[K\r⣾ code | 6.22\x1b[K\r⣽ code | 6.32\x1b[K\r⣻ code | 6.42\x1b[K\r⢿ code | 6.53\x1b[K\r⡿ code | 6.64\x1b[K\r⣟ code | 6.75\x1b[K\r⣯ code | 6.86\x1b[K\r⣷ code | 6.97\x1b[K\r⣾ code | 7.08\x1b[K\r⣽ code | 7.19\x1b[K\r⣻ code | 7.30\x1b[K\r⢿ code | 7.41\x1b[K\r⡿ code | 7.52\x1b[K\r⣟ code | 7.63\x1b[K\r⣯ code | 7.74\x1b[K\r⣷ code | 7.85\x1b[KEpisode 150: Total Reward = 200.0, Epsilon = 0.01\n\r⣾ code | 7.96\x1b[K\r⣽ code | 8.07\x1b[K\r⣻ code | 8.18\x1b[K\r⢿ code | 8.29\x1b[K\r⡿ code | 8.40\x1b[K\r⣟ code | 8.51\x1b[K\r⣯ code | 8.62\x1b[K\r⣷ code | 8.73\x1b[K\r⣾ code | 8.84\x1b[K\r⣽ code | 8.95\x1b[K\r⣻ code | 9.07\x1b[K\r⢿ code | 9.18\x1b[K\r⡿ code | 9.29\x1b[K\r⣟ code | 9.40\x1b[K\r⣯ code | 9.51\x1b[K\r⣷ code | 9.61\x1b[K\r⣾ code | 9.72\x1b[K\r⣽ code | 9.84\x1b[K\r⣻ code | 9.94\x1b[K\r⢿ code | 10.04\x1b[K\r⡿ code | 10.15\x1b[K\r⣟ code | 10.26\x1b[K\r⣯ code | 10.37\x1b[K\r⣷ code | 10.48\x1b[K\r⣾ code | 10.59\x1b[K\r⣽ code | 10.71\x1b[K\r⣻ code | 10.81\x1b[K\r⢿ code | 10.92\x1b[K\r⡿ code | 11.03\x1b[K\r⣟ code | 11.15\x1b[K\r⣯ code | 11.26\x1b[K\r⣷ code | 11.37\x1b[K\r⣾ code | 11.48\x1b[K\r⣽ code | 11.59\x1b[K\r⣻ code | 11.70\x1b[K\r⢿ code | 11.81\x1b[K\r⡿ code | 11.92\x1b[K\r⣟ code | 12.03\x1b[K\r⣯ code | 12.14\x1b[K\r⣷ code | 12.25\x1b[K\r⣾ code | 12.36\x1b[K\r⣽ code | 12.47\x1b[K\r⣻ code | 12.57\x1b[K\r⢿ code | 12.67\x1b[K\r⡿ code | 12.77\x1b[K\r⣟ code | 12.87\x1b[K\r⣯ code | 12.98\x1b[K\r⣷ code | 13.08\x1b[K\r⣾ code | 13.19\x1b[K\r⣽ code | 13.29\x1b[K\r⣻ code | 13.39\x1b[K\r⢿ code | 13.51\x1b[K\r⡿ code | 13.61\x1b[K\r⣟ code | 13.71\x1b[K\r⣯ code | 13.83\x1b[K\r⣷ code | 13.93\x1b[K\r⣾ code | 14.04\x1b[K\r⣽ code | 14.14\x1b[K\r⣻ code | 14.24\x1b[K\r⢿ code | 14.36\x1b[K\r⡿ code | 14.46\x1b[K\r⣟ code | 14.56\x1b[K\r⣯ code | 14.66\x1b[K\r⣷ code | 14.78\x1b[K\r⣾ code | 14.88\x1b[K\r⣽ code | 14.98\x1b[K\r⣻ code | 15.08\x1b[K\r⢿ code | 15.18\x1b[K\r⡿ code | 15.28\x1b[K\r⣟ code | 15.39\x1b[K\r⣯ code | 15.50\x1b[K\r⣷ code | 15.60\x1b[K\r⣾ code | 15.70\x1b[K\r⣽ code | 15.80\x1b[K\r⣻ code | 15.90\x1b[K\r⢿ code | 16.01\x1b[K\r⡿ code | 16.11\x1b[K\r⣟ code | 16.21\x1b[K\r⣯ code | 16.33\x1b[K\r⣷ code | 16.43\x1b[K\r⣾ code | 16.53\x1b[K\r⣽ code | 16.65\x1b[K\r⣻ code | 16.75\x1b[KEpisode 200: Total Reward = 200.0, Epsilon = 0.01\n\r⢿ code | 16.86\x1b[K\r⡿ code | 16.96\x1b[K\r⣟ code | 17.06\x1b[K\r⣯ code | 17.16\x1b[K\r⣷ code | 17.28\x1b[K\r⣾ code | 17.38\x1b[K\r⣽ code | 17.50\x1b[K\r⣻ code | 17.60\x1b[K\r⢿ code | 17.70\x1b[K\r⡿ code | 17.80\x1b[K\r⣟ code | 17.90\x1b[K\r⣯ code | 18.01\x1b[K\r⣷ code | 18.11\x1b[K\r⣾ code | 18.21\x1b[K\r⣽ code | 18.33\x1b[K\r⣻ code | 18.43\x1b[K\r⢿ code | 18.53\x1b[K\r⡿ code | 18.63\x1b[K\r⣟ code | 18.73\x1b[K\r⣯ code | 18.85\x1b[K\r⣷ code | 18.95\x1b[K\r⣾ code | 19.06\x1b[K\r⣽ code | 19.16\x1b[K\r⣻ code | 19.27\x1b[K\r⢿ code | 19.38\x1b[K\r⡿ code | 19.48\x1b[K\r⣟ code | 19.58\x1b[K\r⣯ code | 19.70\x1b[K\r⣷ code | 19.80\x1b[K\r⣾ code | 19.90\x1b[K\r⣽ code | 20.00\x1b[K\r⣻ code | 20.12\x1b[K\r⢿ code | 20.22\x1b[K\r⡿ code | 20.32\x1b[K\r⣟ code | 20.42\x1b[K\r⣯ code | 20.53\x1b[K\r⣷ code | 20.63\x1b[K\r⣾ code | 20.75\x1b[K\r⣽ code | 20.85\x1b[K\r⣻ code | 20.95\x1b[K\r⢿ code | 21.07\x1b[K\r⡿ code | 21.17\x1b[K\r⣟ code | 21.27\x1b[K\r⣯ code | 21.38\x1b[K\r⣷ code | 21.48\x1b[K\r⣾ code | 21.59\x1b[K\r⣽ code | 21.70\x1b[K\r⣻ code | 21.80\x1b[K\r⢿ code | 21.90\x1b[K\r⡿ code | 22.00\x1b[K\r⣟ code | 22.12\x1b[K\r⣯ code | 22.22\x1b[K\r⣷ code | 22.32\x1b[K\r⣾ code | 22.43\x1b[K\r⣽ code | 22.53\x1b[K\r⣻ code | 22.64\x1b[K\r⢿ code | 22.74\x1b[K\r⡿ code | 22.84\x1b[K\r⣟ code | 22.95\x1b[K\r⣯ code | 23.05\x1b[K\r⣷ code | 23.15\x1b[K\r⣾ code | 23.25\x1b[K\r⣽ code | 23.37\x1b[K\r⣻ code | 23.47\x1b[K\r⢿ code | 23.57\x1b[K\r⡿ code | 23.67\x1b[K\r⣟ code | 23.79\x1b[K\r⣯ code | 23.89\x1b[K\r⣷ code | 24.00\x1b[K\r⣾ code | 24.10\x1b[K\r⣽ code | 24.20\x1b[K\r⣻ code | 24.32\x1b[K\r⢿ code | 24.42\x1b[K\r⡿ code | 24.54\x1b[K\r⣟ code | 24.64\x1b[K\r⣯ code | 24.74\x1b[K\r⣷ code | 24.85\x1b[K\r⣾ code | 24.97\x1b[K\r⣽ code | 25.07\x1b[K\r⣻ code | 25.17\x1b[K\r⢿ code | 25.29\x1b[K\r⡿ code | 25.39\x1b[K\r⣟ code | 25.49\x1b[K\r⣯ code | 25.60\x1b[K\r⣷ code | 25.71\x1b[K\r⣾ code | 25.81\x1b[K\r⣽ code | 25.91\x1b[K\r⣻ code | 26.02\x1b[K\r⢿ code | 26.12\x1b[K\r⡿ code | 26.24\x1b[K\r⣟ code | 26.34\x1b[K\r⣯ code | 26.44\x1b[K\r⣷ code | 26.54\x1b[K\r⣾ code | 26.66\x1b[KEpisode 250: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 26.76\x1b[K\r⣻ code | 26.87\x1b[K\r⢿ code | 26.97\x1b[K\r⡿ code | 27.07\x1b[K\r⣟ code | 27.17\x1b[K\r⣯ code | 27.27\x1b[K\r⣷ code | 27.37\x1b[K\r⣾ code | 27.49\x1b[K\r⣽ code | 27.59\x1b[K\r⣻ code | 27.71\x1b[K\r⢿ code | 27.81\x1b[K\r⡿ code | 27.92\x1b[K\r⣟ code | 28.02\x1b[K\r⣯ code | 28.12\x1b[K\r⣷ code | 28.22\x1b[K\r⣾ code | 28.32\x1b[K\r⣽ code | 28.42\x1b[K\r⣻ code | 28.53\x1b[K\r⢿ code | 28.64\x1b[K\r⡿ code | 28.74\x1b[K\r⣟ code | 28.86\x1b[K\r⣯ code | 28.96\x1b[K\r⣷ code | 29.06\x1b[K\r⣾ code | 29.17\x1b[K\r⣽ code | 29.28\x1b[K\r⣻ code | 29.38\x1b[K\r⢿ code | 29.48\x1b[K\r⡿ code | 29.58\x1b[K\r⣟ code | 29.69\x1b[K\r⣯ code | 29.79\x1b[K\r⣷ code | 29.91\x1b[K\r⣾ code | 30.01\x1b[K\r⣽ code | 30.11\x1b[K\r⣻ code | 30.22\x1b[K\r⢿ code | 30.33\x1b[K\r⡿ code | 30.44\x1b[K\r⣟ code | 30.54\x1b[K\r⣯ code | 30.64\x1b[K\r⣷ code | 30.76\x1b[K\r⣾ code | 30.86\x1b[K\r⣽ code | 30.96\x1b[K\r⣻ code | 31.06\x1b[K\r⢿ code | 31.16\x1b[K\r⡿ code | 31.28\x1b[K\r⣟ code | 31.39\x1b[K\r⣯ code | 31.49\x1b[K\r⣷ code | 31.60\x1b[K\r⣾ code | 31.70\x1b[K\r⣽ code | 31.80\x1b[K\r⣻ code | 31.90\x1b[K\r⢿ code | 32.01\x1b[K\r⡿ code | 32.13\x1b[K\r⣟ code | 32.24\x1b[K\r⣯ code | 32.34\x1b[K\r⣷ code | 32.44\x1b[K\r⣾ code | 32.55\x1b[K\r⣽ code | 32.66\x1b[K\r⣻ code | 32.77\x1b[K\r⢿ code | 32.88\x1b[K\r⡿ code | 32.99\x1b[K\r⣟ code | 33.10\x1b[K\r⣯ code | 33.21\x1b[K\r⣷ code | 33.32\x1b[K\r⣾ code | 33.43\x1b[K\r⣽ code | 33.54\x1b[K\r⣻ code | 33.65\x1b[K\r⢿ code | 33.75\x1b[K\r⡿ code | 33.86\x1b[K\r⣟ code | 33.96\x1b[K\r⣯ code | 34.06\x1b[K\r⣷ code | 34.18\x1b[K\r⣾ code | 34.28\x1b[K\r⣽ code | 34.40\x1b[K\r⣻ code | 34.51\x1b[K\r⢿ code | 34.61\x1b[K\r⡿ code | 34.71\x1b[K\r⣟ code | 34.82\x1b[K\r⣯ code | 34.92\x1b[K\r⣷ code | 35.03\x1b[K\r⣾ code | 35.14\x1b[K\r⣽ code | 35.25\x1b[K\r⣻ code | 35.36\x1b[K\r⢿ code | 35.47\x1b[K\r⡿ code | 35.58\x1b[K\r⣟ code | 35.69\x1b[K\r⣯ code | 35.80\x1b[K\r⣷ code | 35.90\x1b[K\r⣾ code | 36.01\x1b[K\r⣽ code | 36.12\x1b[K\r⣻ code | 36.22\x1b[K\r⢿ code | 36.33\x1b[K\r⡿ code | 36.43\x1b[K\r⣟ code | 36.53\x1b[K\r⣯ code | 36.65\x1b[K\r⣷ code | 36.75\x1b[K\r⣾ code | 36.85\x1b[K\r⣽ code | 36.95\x1b[K\r⣻ code | 37.07\x1b[K\r⢿ code | 37.18\x1b[K\r⡿ code | 37.28\x1b[K\r⣟ code | 37.38\x1b[K\r⣯ code | 37.50\x1b[K\r⣷ code | 37.60\x1b[K\r⣾ code | 37.72\x1b[KEpisode 300: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 37.82\x1b[K\r⣻ code | 37.93\x1b[K\r⢿ code | 38.05\x1b[K\r⡿ code | 38.15\x1b[K\r⣟ code | 38.27\x1b[K\r⣯ code | 38.37\x1b[K\r⣷ code | 38.47\x1b[K\r⣾ code | 38.57\x1b[K\r⣽ code | 38.67\x1b[K\r⣻ code | 38.77\x1b[K\r⢿ code | 38.87\x1b[K\r⡿ code | 38.98\x1b[K\r⣟ code | 39.09\x1b[K\r⣯ code | 39.20\x1b[K\r⣷ code | 39.32\x1b[K\r⣾ code | 39.42\x1b[K\r⣽ code | 39.52\x1b[K\r⣻ code | 39.64\x1b[K\r⢿ code | 39.74\x1b[K\r⡿ code | 39.84\x1b[K\r⣟ code | 39.94\x1b[K\r⣯ code | 40.04\x1b[K\r⣷ code | 40.14\x1b[K\r⣾ code | 40.24\x1b[K\r⣽ code | 40.35\x1b[K\r⣻ code | 40.45\x1b[K\r⢿ code | 40.55\x1b[K\r⡿ code | 40.65\x1b[K\r⣟ code | 40.77\x1b[K\r⣯ code | 40.87\x1b[K\r⣷ code | 40.99\x1b[K\r⣾ code | 41.09\x1b[K\r⣽ code | 41.19\x1b[K\r⣻ code | 41.30\x1b[K\r⢿ code | 41.41\x1b[K\r⡿ code | 41.52\x1b[K\r⣟ code | 41.62\x1b[K\r⣯ code | 41.72\x1b[K\r⣷ code | 41.84\x1b[K\r⣾ code | 41.94\x1b[K\r⣽ code | 42.04\x1b[K\r⣻ code | 42.15\x1b[K\r⢿ code | 42.26\x1b[K\r⡿ code | 42.36\x1b[K\r⣟ code | 42.47\x1b[K\r⣯ code | 42.57\x1b[K\r⣷ code | 42.67\x1b[K\r⣾ code | 42.77\x1b[K\r⣽ code | 42.89\x1b[K\r⣻ code | 42.99\x1b[K\r⢿ code | 43.09\x1b[K\r⡿ code | 43.21\x1b[K\r⣟ code | 43.31\x1b[K\r⣯ code | 43.41\x1b[K\r⣷ code | 43.52\x1b[K\r⣾ code | 43.62\x1b[K\r⣽ code | 43.74\x1b[K\r⣻ code | 43.84\x1b[K\r⢿ code | 43.95\x1b[K\r⡿ code | 44.06\x1b[K\r⣟ code | 44.16\x1b[K\r⣯ code | 44.27\x1b[K\r⣷ code | 44.37\x1b[K\r⣾ code | 44.49\x1b[K\r⣽ code | 44.59\x1b[K\r⣻ code | 44.69\x1b[K\r⢿ code | 44.81\x1b[K\r⡿ code | 44.91\x1b[K\r⣟ code | 45.02\x1b[K\r⣯ code | 45.13\x1b[K\r⣷ code | 45.24\x1b[K\r⣾ code | 45.36\x1b[K\r⣽ code | 45.47\x1b[K\r⣻ code | 45.58\x1b[K\r⢿ code | 45.69\x1b[K\r⡿ code | 45.80\x1b[K\r⣟ code | 45.91\x1b[K\r⣯ code | 46.01\x1b[K\r⣷ code | 46.11\x1b[K\r⣾ code | 46.23\x1b[K\r⣽ code | 46.33\x1b[K\r⣻ code | 46.44\x1b[K\r⢿ code | 46.56\x1b[K\r⡿ code | 46.66\x1b[K\r⣟ code | 46.76\x1b[K\r⣯ code | 46.88\x1b[K\r⣷ code | 46.98\x1b[K\r⣾ code | 47.09\x1b[K\r⣽ code | 47.21\x1b[K\r⣻ code | 47.31\x1b[K\r⢿ code | 47.41\x1b[K\r⡿ code | 47.53\x1b[K\r⣟ code | 47.63\x1b[K\r⣯ code | 47.74\x1b[K\r⣷ code | 47.85\x1b[K\r⣾ code | 47.96\x1b[K\r⣽ code | 48.06\x1b[K\r⣻ code | 48.16\x1b[K\r⢿ code | 48.26\x1b[K\r⡿ code | 48.37\x1b[K\r⣟ code | 48.48\x1b[K\r⣯ code | 48.59\x1b[K\r⣷ code | 48.70\x1b[K\r⣾ code | 48.81\x1b[K\r⣽ code | 48.92\x1b[K\r⣻ code | 49.03\x1b[K\r⢿ code | 49.13\x1b[K\r⡿ code | 49.24\x1b[K\r⣟ code | 49.35\x1b[K\r⣯ code | 49.45\x1b[K\r⣷ code | 49.56\x1b[K\r⣾ code | 49.66\x1b[K\r⣽ code | 49.76\x1b[K\r⣻ code | 49.88\x1b[K\r⢿ code | 49.99\x1b[K\r⡿ code | 50.09\x1b[K\r⣟ code | 50.21\x1b[K\r⣯ code | 50.32\x1b[KEpisode 350: Total Reward = 200.0, Epsilon = 0.01\n\r⣷ code | 50.43\x1b[K\r⣾ code | 50.54\x1b[K\r⣽ code | 50.65\x1b[K\r⣻ code | 50.75\x1b[K\r⢿ code | 50.86\x1b[K\r⡿ code | 50.96\x1b[K\r⣟ code | 51.08\x1b[K\r⣯ code | 51.18\x1b[K\r⣷ code | 51.28\x1b[K\r⣾ code | 51.38\x1b[K\r⣽ code | 51.50\x1b[K\r⣻ code | 51.60\x1b[K\r⢿ code | 51.71\x1b[K\r⡿ code | 51.81\x1b[K\r⣟ code | 51.92\x1b[K\r⣯ code | 52.02\x1b[K\r⣷ code | 52.13\x1b[K\r⣾ code | 52.23\x1b[K\r⣽ code | 52.33\x1b[K\r⣻ code | 52.43\x1b[K\r⢿ code | 52.53\x1b[K\r⡿ code | 52.65\x1b[K\r⣟ code | 52.75\x1b[K\r⣯ code | 52.85\x1b[K\r⣷ code | 52.97\x1b[K\r⣾ code | 53.07\x1b[K\r⣽ code | 53.18\x1b[K\r⣻ code | 53.29\x1b[K\r⢿ code | 53.40\x1b[K\r⡿ code | 53.51\x1b[K\r⣟ code | 53.62\x1b[K\r⣯ code | 53.73\x1b[K\r⣷ code | 53.84\x1b[K\r⣾ code | 53.95\x1b[K\r⣽ code | 54.05\x1b[K\r⣻ code | 54.15\x1b[K\r⢿ code | 54.27\x1b[K\r⡿ code | 54.37\x1b[K\r⣟ code | 54.48\x1b[K\r⣯ code | 54.58\x1b[K\r⣷ code | 54.69\x1b[K\r⣾ code | 54.80\x1b[K\r⣽ code | 54.90\x1b[K\r⣻ code | 55.02\x1b[K\r⢿ code | 55.12\x1b[K\r⡿ code | 55.22\x1b[K\r⣟ code | 55.33\x1b[K\r⣯ code | 55.44\x1b[K\r⣷ code | 55.54\x1b[K\r⣾ code | 55.64\x1b[K\r⣽ code | 55.74\x1b[K\r⣻ code | 55.84\x1b[K\r⢿ code | 55.94\x1b[K\r⡿ code | 56.04\x1b[K\r⣟ code | 56.15\x1b[K\r⣯ code | 56.25\x1b[K\r⣷ code | 56.35\x1b[K\r⣾ code | 56.47\x1b[K\r⣽ code | 56.57\x1b[K\r⣻ code | 56.67\x1b[K\r⢿ code | 56.79\x1b[K\r⡿ code | 56.89\x1b[K\r⣟ code | 56.99\x1b[K\r⣯ code | 57.09\x1b[K\r⣷ code | 57.20\x1b[K\r⣾ code | 57.30\x1b[K\r⣽ code | 57.42\x1b[K\r⣻ code | 57.52\x1b[K\r⢿ code | 57.64\x1b[K\r⡿ code | 57.74\x1b[K\r⣟ code | 57.84\x1b[K\r⣯ code | 57.95\x1b[K\r⣷ code | 58.05\x1b[K\r⣾ code | 58.16\x1b[K\r⣽ code | 58.27\x1b[K\r⣻ code | 58.37\x1b[K\r⢿ code | 58.47\x1b[K\r⡿ code | 58.57\x1b[K\r⣟ code | 58.67\x1b[K\r⣯ code | 58.79\x1b[K\r⣷ code | 58.89\x1b[K\r⣾ code | 59.01\x1b[K\r⣽ code | 59.11\x1b[K\r⣻ code | 59.21\x1b[K\r⢿ code | 59.36\x1b[K\r⡿ code | 59.48\x1b[K\r⣟ code | 59.59\x1b[K\r⣯ code | 59.69\x1b[K\r⣷ code | 59.81\x1b[K\r⣾ code | 59.91\x1b[K\r⣽ code | 60.02\x1b[K\r⣻ code | 60.14\x1b[K\r⢿ code | 60.24\x1b[K\r⡿ code | 60.36\x1b[K\r⣟ code | 60.46\x1b[K\r⣯ code | 60.56\x1b[K\r⣷ code | 60.67\x1b[K\r⣾ code | 60.78\x1b[K\r⣽ code | 60.89\x1b[K\r⣻ code | 60.99\x1b[K\r⢿ code | 61.11\x1b[K\r⡿ code | 61.22\x1b[K\r⣟ code | 61.33\x1b[K\r⣯ code | 61.44\x1b[K\r⣷ code | 61.54\x1b[K\r⣾ code | 61.66\x1b[K\r⣽ code | 61.76\x1b[K\r⣻ code | 61.86\x1b[K\r⢿ code | 61.98\x1b[K\r⡿ code | 62.08\x1b[K\r⣟ code | 62.19\x1b[K\r⣯ code | 62.31\x1b[K\r⣷ code | 62.41\x1b[K\r⣾ code | 62.51\x1b[K\r⣽ code | 62.63\x1b[K\r⣻ code | 62.74\x1b[K\r⢿ code | 62.86\x1b[K\r⡿ code | 62.96\x1b[K\r⣟ code | 63.07\x1b[K\r⣯ code | 63.18\x1b[K\r⣷ code | 63.29\x1b[K\r⣾ code | 63.41\x1b[K\r⣽ code | 63.51\x1b[K\r⣻ code | 63.63\x1b[K\r⢿ code | 63.73\x1b[K\r⡿ code | 63.83\x1b[K\r⣟ code | 63.93\x1b[K\r⣯ code | 64.04\x1b[K\r⣷ code | 64.15\x1b[K\r⣾ code | 64.26\x1b[K\r⣽ code | 64.36\x1b[K\r⣻ code | 64.48\x1b[K\r⢿ code | 64.58\x1b[K\r⡿ code | 64.68\x1b[K\r⣟ code | 64.80\x1b[K\r⣯ code | 64.91\x1b[K\r⣷ code | 65.01\x1b[K\r⣾ code | 65.11\x1b[KEpisode 400: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 65.21\x1b[K\r⣻ code | 65.34\x1b[K\r⢿ code | 65.45\x1b[K\r⡿ code | 65.57\x1b[K\r⣟ code | 65.68\x1b[K\r⣯ code | 65.80\x1b[K\r⣷ code | 65.90\x1b[K\r⣾ code | 66.00\x1b[K\r⣽ code | 66.10\x1b[K\r⣻ code | 66.23\x1b[K\r⢿ code | 66.33\x1b[K\r⡿ code | 66.45\x1b[K\r⣟ code | 66.55\x1b[K\r⣯ code | 66.65\x1b[K\r⣷ code | 66.76\x1b[K\r⣾ code | 66.88\x1b[K\r⣽ code | 66.98\x1b[K\r⣻ code | 67.08\x1b[K\r⢿ code | 67.20\x1b[K\r⡿ code | 67.30\x1b[K\r⣟ code | 67.41\x1b[K\r⣯ code | 67.52\x1b[K\r⣷ code | 67.63\x1b[K\r⣾ code | 67.73\x1b[K\r⣽ code | 67.83\x1b[K\r⣻ code | 67.95\x1b[K\r⢿ code | 68.05\x1b[K\r⡿ code | 68.16\x1b[K\r⣟ code | 68.27\x1b[K\r⣯ code | 68.38\x1b[K\r⣷ code | 68.48\x1b[K\r⣾ code | 68.60\x1b[K\r⣽ code | 68.70\x1b[K\r⣻ code | 68.82\x1b[K\r⢿ code | 68.92\x1b[K\r⡿ code | 69.03\x1b[K\r⣟ code | 69.13\x1b[K\r⣯ code | 69.23\x1b[K\r⣷ code | 69.34\x1b[K\r⣾ code | 69.46\x1b[K\r⣽ code | 69.60\x1b[K\r⣻ code | 69.70\x1b[K\r⢿ code | 69.82\x1b[K\r⡿ code | 69.93\x1b[K\r⣟ code | 70.03\x1b[K\r⣯ code | 70.13\x1b[K\r⣷ code | 70.23\x1b[K\r⣾ code | 70.35\x1b[K\r⣽ code | 70.45\x1b[K\r⣻ code | 70.57\x1b[K\r⢿ code | 70.67\x1b[K\r⡿ code | 70.77\x1b[K\r⣟ code | 70.87\x1b[K\r⣯ code | 70.98\x1b[K\r⣷ code | 71.09\x1b[K\r⣾ code | 71.23\x1b[K\r⣽ code | 71.33\x1b[K\r⣻ code | 71.44\x1b[K\r⢿ code | 71.55\x1b[K\r⡿ code | 71.65\x1b[K\r⣟ code | 71.77\x1b[K\r⣯ code | 71.89\x1b[K\r⣷ code | 72.05\x1b[K\r⣾ code | 72.15\x1b[K\r⣽ code | 72.27\x1b[K\r⣻ code | 72.37\x1b[K\r⢿ code | 72.49\x1b[K\r⡿ code | 72.59\x1b[K\r⣟ code | 72.70\x1b[K\r⣯ code | 72.80\x1b[K\r⣷ code | 72.92\x1b[K\r⣾ code | 73.02\x1b[K\r⣽ code | 73.14\x1b[K\r⣻ code | 73.24\x1b[K\r⢿ code | 73.34\x1b[K\r⡿ code | 73.45\x1b[K\r⣟ code | 73.55\x1b[K\r⣯ code | 73.66\x1b[K\r⣷ code | 73.77\x1b[K\r⣾ code | 73.87\x1b[K\r⣽ code | 73.99\x1b[K\r⣻ code | 74.09\x1b[K\r⢿ code | 74.19\x1b[K\r⡿ code | 74.30\x1b[K\r⣟ code | 74.40\x1b[K\r⣯ code | 74.51\x1b[K\r⣷ code | 74.62\x1b[K\r⣾ code | 74.72\x1b[K\r⣽ code | 74.86\x1b[K\r⣻ code | 74.98\x1b[K\r⢿ code | 75.09\x1b[K\r⡿ code | 75.19\x1b[K\r⣟ code | 75.29\x1b[K\r⣯ code | 75.40\x1b[K\r⣷ code | 75.50\x1b[K\r⣾ code | 75.61\x1b[K\r⣽ code | 75.72\x1b[K\r⣻ code | 75.84\x1b[K\r⢿ code | 75.95\x1b[K\r⡿ code | 76.08\x1b[K\r⣟ code | 76.19\x1b[K\r⣯ code | 76.30\x1b[K\r⣷ code | 76.41\x1b[K\r⣾ code | 76.52\x1b[K\r⣽ code | 76.64\x1b[K\r⣻ code | 76.75\x1b[K\r⢿ code | 76.86\x1b[K\r⡿ code | 76.97\x1b[K\r⣟ code | 77.08\x1b[K\r⣯ code | 77.19\x1b[K\r⣷ code | 77.31\x1b[K\r⣾ code | 77.43\x1b[K\r⣽ code | 77.54\x1b[K\r⣻ code | 77.65\x1b[K\r⢿ code | 77.76\x1b[K\r⡿ code | 77.87\x1b[K\r⣟ code | 77.98\x1b[K\r⣯ code | 78.09\x1b[K\r⣷ code | 78.20\x1b[K\r⣾ code | 78.31\x1b[K\r⣽ code | 78.42\x1b[K\r⣻ code | 78.53\x1b[K\r⢿ code | 78.64\x1b[K\r⡿ code | 78.75\x1b[K\r⣟ code | 78.86\x1b[K\r⣯ code | 78.97\x1b[K\r⣷ code | 79.08\x1b[K\r⣾ code | 79.19\x1b[K\r⣽ code | 79.30\x1b[K\r⣻ code | 79.41\x1b[K\r⢿ code | 79.52\x1b[K\r⡿ code | 79.63\x1b[K\r⣟ code | 79.75\x1b[K\r⣯ code | 79.88\x1b[K\r⣷ code | 79.98\x1b[K\r⣾ code | 80.09\x1b[K\r⣽ code | 80.20\x1b[K\r⣻ code | 80.31\x1b[K\r⢿ code | 80.44\x1b[K\r⡿ code | 80.56\x1b[K\r⣟ code | 80.66\x1b[K\r⣯ code | 80.77\x1b[K\r⣷ code | 80.88\x1b[K\r⣾ code | 80.99\x1b[K\r⣽ code | 81.10\x1b[K\r⣻ code | 81.21\x1b[K\r⢿ code | 81.32\x1b[K\r⡿ code | 81.43\x1b[K\r⣟ code | 81.54\x1b[K\r⣯ code | 81.65\x1b[K\r⣷ code | 81.76\x1b[K\r⣾ code | 81.87\x1b[K\r⣽ code | 81.98\x1b[K\r⣻ code | 82.09\x1b[K\r⢿ code | 82.20\x1b[K\r⡿ code | 82.31\x1b[K\r⣟ code | 82.42\x1b[K\r⣯ code | 82.53\x1b[K\r⣷ code | 82.65\x1b[KEpisode 450: Total Reward = 200.0, Epsilon = 0.01\n\r⣾ code | 82.78\x1b[K\r⣽ code | 82.90\x1b[K\r⣻ code | 83.01\x1b[K\r⢿ code | 83.12\x1b[K\r⡿ code | 83.23\x1b[K\r⣟ code | 83.34\x1b[K\r⣯ code | 83.45\x1b[K\r⣷ code | 83.56\x1b[K\r⣾ code | 83.72\x1b[K\r⣽ code | 83.83\x1b[K\r⣻ code | 83.94\x1b[K\r⢿ code | 84.05\x1b[K\r⡿ code | 84.16\x1b[K\r⣟ code | 84.27\x1b[K\r⣯ code | 84.38\x1b[K\r⣷ code | 84.49\x1b[K\r⣾ code | 84.60\x1b[K\r⣽ code | 84.71\x1b[K\r⣻ code | 84.82\x1b[K\r⢿ code | 84.93\x1b[K\r⡿ code | 85.08\x1b[K\r⣟ code | 85.19\x1b[K\r⣯ code | 85.34\x1b[K\r⣷ code | 85.45\x1b[K\r⣾ code | 85.56\x1b[K\r⣽ code | 85.67\x1b[K\r⣻ code | 85.78\x1b[K\r⢿ code | 85.89\x1b[K\r⡿ code | 86.00\x1b[K\r⣟ code | 86.10\x1b[K\r⣯ code | 86.20\x1b[K\r⣷ code | 86.30\x1b[K\r⣾ code | 86.40\x1b[K\r⣽ code | 86.52\x1b[K\r⣻ code | 86.62\x1b[K\r⢿ code | 86.72\x1b[K\r⡿ code | 86.83\x1b[K\r⣟ code | 86.93\x1b[K\r⣯ code | 87.04\x1b[K\r⣷ code | 87.15\x1b[K\r⣾ code | 87.25\x1b[K\r⣽ code | 87.35\x1b[K\r⣻ code | 87.47\x1b[K\r⢿ code | 87.57\x1b[K\r⡿ code | 87.68\x1b[K\r⣟ code | 87.79\x1b[K\r⣯ code | 87.90\x1b[K\r⣷ code | 88.00\x1b[K\r⣾ code | 88.12\x1b[K\r⣽ code | 88.22\x1b[K\r⣻ code | 88.34\x1b[K\r⢿ code | 88.44\x1b[K\r⡿ code | 88.55\x1b[K\r⣟ code | 88.65\x1b[K\r⣯ code | 88.77\x1b[K\r⣷ code | 88.87\x1b[K\r⣾ code | 88.99\x1b[K\r⣽ code | 89.09\x1b[K\r⣻ code | 89.19\x1b[K\r⢿ code | 89.29\x1b[K\r⡿ code | 89.40\x1b[K\r⣟ code | 89.52\x1b[K\r⣯ code | 89.62\x1b[K\r⣷ code | 89.74\x1b[K\r⣾ code | 89.84\x1b[K\r⣽ code | 89.95\x1b[K\r⣻ code | 90.06\x1b[K\r⢿ code | 90.16\x1b[K\r⡿ code | 90.27\x1b[K\r⣟ code | 90.37\x1b[K\r⣯ code | 90.48\x1b[K\r⣷ code | 90.60\x1b[K\r⣾ code | 90.71\x1b[K\r⣽ code | 90.82\x1b[K\r⣻ code | 90.92\x1b[K\r⢿ code | 91.04\x1b[K\r⡿ code | 91.14\x1b[K\r⣟ code | 91.24\x1b[K\r⣯ code | 91.36\x1b[K\r⣷ code | 91.46\x1b[K\r⣾ code | 91.56\x1b[K\r⣽ code | 91.67\x1b[K\r⣻ code | 91.77\x1b[K\r⢿ code | 91.87\x1b[K\r⡿ code | 91.97\x1b[K\r⣟ code | 92.09\x1b[K\r⣯ code | 92.19\x1b[K\r⣷ code | 92.29\x1b[K\r⣾ code | 92.41\x1b[K\r⣽ code | 92.51\x1b[K\r⣻ code | 92.61\x1b[K\r⢿ code | 92.71\x1b[K\r⡿ code | 92.82\x1b[K\r⣟ code | 92.92\x1b[K\r⣯ code | 93.05\x1b[K\r⣷ code | 93.16\x1b[K\r⣾ code | 93.28\x1b[K\r⣽ code | 93.39\x1b[K\r⣻ code | 93.51\x1b[K\r⢿ code | 93.61\x1b[K\r⡿ code | 93.71\x1b[K\r⣟ code | 93.82\x1b[K\r⣯ code | 93.93\x1b[K\r⣷ code | 94.03\x1b[K\r⣾ code | 94.13\x1b[K\r⣽ code | 94.24\x1b[K\r⣻ code | 94.34\x1b[K\r⢿ code | 94.46\x1b[K\r⡿ code | 94.56\x1b[K\r⣟ code | 94.68\x1b[K\r⣯ code | 94.78\x1b[K\r⣷ code | 94.89\x1b[K\r⣾ code | 94.99\x1b[K\r⣽ code | 95.11\x1b[K\r⣻ code | 95.21\x1b[K\r⢿ code | 95.31\x1b[K\r⡿ code | 95.43\x1b[K\r⣟ code | 95.53\x1b[K\r⣯ code | 95.63\x1b[K\r⣷ code | 95.74\x1b[K\r⣾ code | 95.84\x1b[K\r⣽ code | 95.94\x1b[K\r⣻ code | 96.04\x1b[K\r⢿ code | 96.16\x1b[K\r⡿ code | 96.26\x1b[K\r⣟ code | 96.36\x1b[K\r⣯ code | 96.46\x1b[K\r⣷ code | 96.58\x1b[K\r⣾ code | 96.68\x1b[K\r⣽ code | 96.78\x1b[K\r⣻ code | 96.88\x1b[K\r⢿ code | 96.99\x1b[K\r⡿ code | 97.10\x1b[K\r⣟ code | 97.20\x1b[K\r⣯ code | 97.31\x1b[K\r⣷ code | 97.41\x1b[K\r⣾ code | 97.51\x1b[K\r⣽ code | 97.63\x1b[K\r⣻ code | 97.73\x1b[K\r⢿ code | 97.85\x1b[K\r⡿ code | 97.95\x1b[K\r⣟ code | 98.06\x1b[K\r⣯ code | 98.16\x1b[K\r⣷ code | 98.28\x1b[K\r⣾ code | 98.38\x1b[K\r⣽ code | 98.48\x1b[K\r⣻ code | 98.60\x1b[K\r⢿ code | 98.70\x1b[K\r⡿ code | 98.80\x1b[K\r⣟ code | 98.90\x1b[K\r⣯ code | 99.01\x1b[K\r⣷ code | 99.11\x1b[K\r⣾ code | 99.23\x1b[K\r⣽ code | 99.33\x1b[K\r⣻ code | 99.45\x1b[K\r⢿ code | 99.55\x1b[K\r⡿ code | 99.65\x1b[K\r⣟ code | 99.76\x1b[K\r⣯ code | 99.87\x1b[K\r⣷ code | 99.98\x1b[K\r⣾ code | 100.08\x1b[K\r⣽ code | 100.18\x1b[K\r⣻ code | 100.30\x1b[K\r⢿ code | 100.40\x1b[K\r⡿ code | 100.50\x1b[K\r⣟ code | 100.60\x1b[K\r⣯ code | 100.72\x1b[K\r⣷ code | 100.82\x1b[K\r⣾ code | 100.92\x1b[K\r⣽ code | 101.03\x1b[K\r⣻ code | 101.13\x1b[K\r⢿ code | 101.23\x1b[K\r⡿ code | 101.33\x1b[K\r⣟ code | 101.45\x1b[K\r⣯ code | 101.55\x1b[K\r⣷ code | 101.65\x1b[K\r⣾ code | 101.76\x1b[K\r⣽ code | 101.87\x1b[K\r⣻ code | 101.98\x1b[K\r⢿ code | 102.08\x1b[K\r⡿ code | 102.18\x1b[K\r⣟ code | 102.30\x1b[K\r⣯ code | 102.40\x1b[K\r⣷ code | 102.50\x1b[K\r⣾ code | 102.62\x1b[K\r⣽ code | 102.72\x1b[K\r⣻ code | 102.83\x1b[K\r⢿ code | 102.93\x1b[K\r⡿ code | 103.04\x1b[K\r⣟ code | 103.15\x1b[K\r⣯ code | 103.25\x1b[K\r⣷ code | 103.35\x1b[K\r⣾ code | 103.47\x1b[K\r⣽ code | 103.57\x1b[K\r⣻ code | 103.67\x1b[K\r⢿ code | 103.78\x1b[K\r⡿ code | 103.89\x1b[K\r⣟ code | 103.99\x1b[K\r⣯ code | 104.10\x1b[K\r⣷ code | 104.20\x1b[K\r⣾ code | 104.30\x1b[K\r⣽ code | 104.40\x1b[K\r⣻ code | 104.50\x1b[K\r⢿ code | 104.62\x1b[K\r⡿ code | 104.72\x1b[K\r⣟ code | 104.82\x1b[K\r⣯ code | 104.92\x1b[K\r⣷ code | 105.02\x1b[K\r⣾ code | 105.13\x1b[K\r⣽ code | 105.24\x1b[K\r⣻ code | 105.35\x1b[K\r⢿ code | 105.45\x1b[K\r⡿ code | 105.55\x1b[K\r⣟ code | 105.65\x1b[K\r⣯ code | 105.75\x1b[K\r⣷ code | 105.85\x1b[K\r⣾ code | 105.95\x1b[K\r⣽ code | 106.05\x1b[K\r⣻ code | 106.17\x1b[K\r⢿ code | 106.27\x1b[K\r⡿ code | 106.39\x1b[K\r⣟ code | 106.49\x1b[K\r⣯ code | 106.59\x1b[K\r⣷ code | 106.69\x1b[K\r⣾ code | 106.80\x1b[K\r⣽ code | 106.90\x1b[K\r⣻ code | 107.01\x1b[K\r⢿ code | 107.11\x1b[K\r⡿ code | 107.22\x1b[K\r⣟ code | 107.32\x1b[K\r⣯ code | 107.42\x1b[K\r⣷ code | 107.52\x1b[K\r⣾ code | 107.64\x1b[K\r⣽ code | 107.74\x1b[K\r⣻ code | 107.85\x1b[K\r⢿ code | 107.96\x1b[K\r⡿ code | 108.06\x1b[K\r⣟ code | 108.17\x1b[K\r⣯ code | 108.27\x1b[K\r⣷ code | 108.37\x1b[K\r⣾ code | 108.49\x1b[K\r⣽ code | 108.59\x1b[K\r⣻ code | 108.69\x1b[K\r⢿ code | 108.79\x1b[K\r⡿ code | 108.91\x1b[K\r⣟ code | 109.01\x1b[K\r⣯ code | 109.11\x1b[K\r⣷ code | 109.22\x1b[K\r⣾ code | 109.33\x1b[K\r⣽ code | 109.44\x1b[K\r⣻ code | 109.54\x1b[K\r⢿ code | 109.64\x1b[K\r⡿ code | 109.74\x1b[K\r⣟ code | 109.84\x1b[K\r⣯ code | 109.94\x1b[K\r⣷ code | 110.06\x1b[K\r⣾ code | 110.16\x1b[K\r⣽ code | 110.27\x1b[K\r⣻ code | 110.38\x1b[K\r⢿ code | 110.48\x1b[K\r⡿ code | 110.58\x1b[K\r⣟ code | 110.69\x1b[K\nstderr:<string>:51: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\builder\\windows\\pytorch\\torch\\csrc\\utils\\tensor_new.cpp:281.)\n"
    #cleaned_result = super_strip(asd)
    #print(type(cleaned_result), len(asd), len(cleaned_result))
    #print(cleaned_result)
