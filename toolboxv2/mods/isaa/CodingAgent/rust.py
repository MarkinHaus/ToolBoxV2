import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import toml
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
)

from toolboxv2.mods.isaa.CodingAgent.live import VirtualFileSystem
from toolboxv2.mods.isaa.extras.session import ChatSession


class CargoRustInterface:
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


import asyncio
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class ThinkState(Enum):
    ACTION = "action"
    PROCESSING = "processing"
    BRAKE = "brake"
    DONE = "done"


class ThinkResult(BaseModel):
    action: str
    content: str
    context: dict[str, Any] | None = None


@dataclass
class ExecutionRecord:
    code: str
    result: str | None
    error: str | None


@dataclass
class CrateInfo:
    name: str
    version: str
    description: str
    documentation: str | None
    repository: str | None
    downloads: int


class RustPipeline:
    """
    A pipeline specialized for Rust development with crates.io integration and build feedback.

    Features:
    - Crates.io documentation crawling and caching
    - Rust compiler feedback parsing
    - Project structure management
    - Auto-dependency resolution
    - Test execution and coverage analysis
    """

    def __init__(
        self,
        agent: Any,
        project_path: Path,
        verbose: bool = False,
        max_iter: int = 12,
        cache_dir: Path | None = None
    ):
        self.agent = agent
        self.project_path = Path(project_path)
        self.verbose = verbose
        self.max_iter = max_iter
        self.cache_dir = cache_dir or Path.home() / ".rust_pipeline_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize crawler for docs.rs
        self.crawler = AsyncWebCrawler(
            config=BrowserConfig(
                headless=True,
                extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
            )
        )

        # Track project state
        self.cargo_toml: dict | None = None
        self.current_crates: dict[str, CrateInfo] = {}
        self.execution_history: list[ExecutionRecord] = []

        # Initialize memory systems
        self.docs_memory = ChatSession(agent.memory, "RustDocs")
        self.build_memory = ChatSession(agent.memory, "BuildResults")

    async def __aenter__(self):
        await self.crawler.start()
        await self.load_project_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.crawler.close()

    async def load_project_state(self):
        """Load and parse current project state including Cargo.toml"""
        cargo_path = self.project_path / "Cargo.toml"
        if cargo_path.exists():
            with open(cargo_path) as f:
                self.cargo_toml = toml.load(f)

        # Load cached crate documentation
        for crate_file in self.cache_dir.glob("crate_*.json"):
            with open(crate_file) as f:
                crate_data = json.load(f)
                self.current_crates[crate_data["name"]] = CrateInfo(**crate_data)

    async def fetch_crate_docs(self, crate_name: str, version: str | None = None) -> str:
        """Fetch and cache documentation from docs.rs"""
        cache_file = self.cache_dir / f"crate_{crate_name}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                if version is None or cached["version"] == version:
                    return cached["documentation"]

        url = f"https://docs.rs/{crate_name}"
        if version:
            url += f"/{version}"

        result = await self.crawler.arun(
            url=url,
            config=CrawlerRunConfig(markdown_generator=DefaultMarkdownGenerator()),
            session_id="docs_session"
        )

        if result.success:
            crate_info = {
                "name": crate_name,
                "version": version or "latest",
                "documentation": result.markdown_v2.raw_markdown,
                "timestamp": datetime.now().isoformat()
            }
            with open(cache_file, "w") as f:
                json.dump(crate_info, f)
            return result.markdown_v2.raw_markdown
        else:
            return f"Error fetching docs: {result.error_message}"

    async def execute_rust(self, code: str, file_path: str | None = None) -> ExecutionRecord:
        """Execute Rust code, handling both file updates and compilation"""
        try:
            if file_path:
                # Update existing or create new file
                full_path = self.project_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(code)

                # Run cargo check for immediate feedback
                result = await self._run_cargo_command("check")
                return ExecutionRecord(code=code, result=result, error=None)
            else:
                # For single expression evaluation, use cargo eval
                result = await self._run_cargo_command("eval", input_code=code)
                return ExecutionRecord(code=code, result=result, error=None)
        except Exception as e:
            return ExecutionRecord(code=code, result=None, error=str(e))

    async def _run_cargo_command(self, cmd: str, input_code: str | None = None) -> str:
        """Execute cargo commands and capture output"""
        if cmd == "eval" and input_code:
            # Create temporary file for evaluation
            eval_path = self.project_path / "src" / "eval.rs"
            with open(eval_path, "w") as f:
                f.write(f"""
fn main() {{
    println!("{{:?}}", {{
        {input_code}
    }});
}}
""")

        process = await asyncio.create_subprocess_exec(
            "cargo", cmd,
            cwd=self.project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        return f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"

    async def run(self, task: str) -> dict[str, Any]:
        """Execute a Rust development task with documentation and build feedback"""
        state = ThinkState.ACTION
        result = None
        iteration = 0

        # Initialize task context
        context = {
            "task": task,
            "project_state": self.cargo_toml,
            "available_crates": list(self.current_crates.keys())
        }

        while state != ThinkState.DONE and iteration < self.max_iter:
            iteration += 1

            if state == ThinkState.ACTION:
                # Get agent's next action
                think_result = await self.agent.think(context)

                if think_result.action == "code":
                    # Execute Rust code
                    execution = await self.execute_rust(
                        think_result.content,
                        think_result.context.get("file_path")
                    )
                    self.execution_history.append(execution)
                    state = ThinkState.PROCESSING

                elif think_result.action == "docs":
                    # Fetch crate documentation
                    crate = think_result.context["crate"]
                    docs = await self.fetch_crate_docs(
                        crate,
                        think_result.context.get("version")
                    )
                    await self.docs_memory.add_message({
                        "role": "system",
                        "content": f"Documentation for {crate}:\n{docs}"
                    })
                    state = ThinkState.ACTION

                elif think_result.action == "done":
                    state = ThinkState.DONE
                    result = think_result.content

            elif state == ThinkState.PROCESSING:
                # Analyze execution results
                last_execution = self.execution_history[-1]
                if last_execution.error:
                    # Parse compiler errors
                    await self.build_memory.add_message({
                        "role": "system",
                        "content": f"Build error:\n{last_execution.error}"
                    })
                    state = ThinkState.ACTION
                else:
                    state = ThinkState.ACTION

            await asyncio.sleep(1)  # Prevent rate limiting

        return {
            "result": result,
            "execution_history": self.execution_history,
            "docs_memory": await self.docs_memory.get_messages(),
            "build_memory": await self.build_memory.get_messages()
        }
