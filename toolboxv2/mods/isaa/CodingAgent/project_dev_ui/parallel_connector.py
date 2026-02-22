"""
Parallel Manager Connector - Proxy Agent using ParallelManager

This module provides:
- Proxy agent that uses ParallelManager instead of ProjectDeveloperEngine
- Decomposes tasks into parallel subtasks
- Status callbacks for UI updates
- Optional pre-analysis phase
"""

import asyncio
import json
import logging
import re
import sys
import os
import time
import uuid
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ParallelConnector")

# Try to import real dependencies
try:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    HAS_FLOW_AGENT = True
except ImportError:
    HAS_FLOW_AGENT = False

try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from toolboxv2.mods.isaa.CodingAgent.manager import ParallelManager, ManagerResult
    HAS_MANAGER = True
except ImportError:
    HAS_MANAGER = False


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    PRE_ANALYZING = "pre_analyzing"
    DECOMPOSING = "decomposing"
    EXECUTING = "executing"
    VALIDATING = "validating"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class StatusUpdate:
    """Status update message for UI"""
    status: ExecutionStatus
    message: str
    phase: str
    progress: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExecutionResult:
    """Result of a parallel development execution"""
    success: bool
    applied_files: List[str]
    failed_tasks: List[str]
    execution_time: float
    total_tokens: int
    coder_count: int
    summary: str = ""


class ParallelProxyAgent:
    """
    Proxy Agent that uses ParallelManager for task execution.

    Responsibilities:
    - Parse user intent from natural language
    - Optional pre-analysis phase
    - Delegate to ParallelManager for decomposition and parallel execution
    - Provide status updates to UI
    """

    def __init__(
        self,
        workspace_path: str,
        status_callback: Optional[Callable[[StatusUpdate], None]] = None,
        use_mock: bool = False
    ):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.status_callback = status_callback
        self.use_mock = use_mock or not (HAS_FLOW_AGENT and HAS_MANAGER)
        self._agent: Optional['FlowAgent'] = None
        self._pre_analyze_enabled = True
        self._max_parallel = 4
        self._config = {}

    def set_config(self, pre_analyze: bool = True, max_parallel: int = 4, **kwargs):
        """Update configuration"""
        self._pre_analyze_enabled = pre_analyze
        self._max_parallel = max_parallel
        self._config = kwargs

    def _emit_status(self, status: ExecutionStatus, message: str,
                     phase: str = "", progress: float = 0.0, details: Dict = None):
        """Emit status update to callback"""
        update = StatusUpdate(
            status=status,
            message=message,
            phase=phase,
            progress=progress,
            details=details or {}
        )

        # Log to terminal
        log_msg = f"[{phase}] {message}"
        if details:
            if 'thinking' in details:
                log_msg += f" | 💭 {str(details['thinking'])[:60]}"
            elif 'tool' in details:
                log_msg += f" | 🔧 {details['tool']}"
            elif 'file' in details:
                log_msg += f" | 📄 {details['file']}"

        if status == ExecutionStatus.FAILED:
            logger.error(log_msg)
        elif status == ExecutionStatus.COMPLETED:
            logger.info(f"✅ {log_msg}")
        else:
            logger.info(log_msg)

        if self.status_callback:
            self.status_callback(update)
        return update

    async def initialize(self, agent: Optional['FlowAgent'] = None):
        """Initialize the agent connector"""
        if self.use_mock:
            self._emit_status(
                ExecutionStatus.PENDING,
                "🔧 Running in mock mode (ToolBoxV2 not available)",
                "init",
                1.0
            )
            return

        if agent is not None:
            self._agent = agent

        if self._agent is None and HAS_FLOW_AGENT:
            try:
                from toolboxv2 import get_app
                app = get_app()
                isaa = app.get_mod("isaa")
                await isaa.init_isaa()
                self._agent = await isaa.get_agent("coder")
            except Exception as e:
                self._emit_status(
                    ExecutionStatus.FAILED,
                    f"Failed to initialize FlowAgent: {e}",
                    "init",
                    0.0
                )
                self.use_mock = True
                return

        self._emit_status(
            ExecutionStatus.PENDING,
            "✅ ParallelProxyAgent initialized",
            "init",
            1.0
        )

    def parse_task(self, user_message: str) -> str:
        """
        Clean and prepare task for ParallelManager.
        Extracts the core task description.
        """
        # Remove common UI artifacts
        task = user_message.strip()
        task = re.sub(r'^\s*(create|make|build|write|generate|implement|add|update|modify|edit)\s+', '', task, flags=re.IGNORECASE)

        return task

    async def execute_task(
        self,
        user_message: str,
        pre_analyze: bool = None,
        max_parallel: int = None
    ) -> ExecutionResult:
        """
        Execute a development task using ParallelManager.

        This is the main entry point for the proxy agent.
        """
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Use config defaults if not specified
        if pre_analyze is None:
            pre_analyze = self._pre_analyze_enabled
        if max_parallel is None:
            max_parallel = self._max_parallel

        try:
            # Phase 1: Parse task
            self._emit_status(ExecutionStatus.PENDING, "🔍 Parsing your request...", "parse", 0.05)
            task = self.parse_task(user_message)

            self._emit_status(
                ExecutionStatus.PENDING,
                f"📝 Task: {task[:100]}",
                "parse",
                0.1,
                {"task": task}
            )

            # Phase 2: Pre-analysis (optional)
            if pre_analyze and not self.use_mock:
                self._emit_status(
                    ExecutionStatus.PRE_ANALYZING,
                    "🔬 Pre-analyzing project structure...",
                    "pre_analysis",
                    0.15,
                    {"thinking": "Scanning codebase for context..."}
                )
                # Pre-analysis happens inside ParallelManager

            # Phase 3: Decompose & Execute
            self._emit_status(
                ExecutionStatus.DECOMPOSING,
                "🧩 Decomposing task into parallel subtasks...",
                "decompose",
                0.2,
                {"thinking": "Identifying independent work units..."}
            )

            # Use mock or real execution
            if self.use_mock:
                result = await self._mock_execute(task)
            else:
                result = await self._real_execute(task, pre_analyze, max_parallel)

            # Phase 4: Done
            if result.success:
                self._emit_status(
                    ExecutionStatus.COMPLETED,
                    f"✅ Successfully completed: {result.summary}",
                    "complete",
                    1.0,
                    {
                        "files": result.applied_files,
                        "tokens": result.total_tokens,
                        "coders": result.coder_count,
                        "duration": result.execution_time
                    }
                )
            else:
                self._emit_status(
                    ExecutionStatus.FAILED,
                    f"❌ Execution failed: {result.summary}",
                    "failed",
                    0.0,
                    {"failed_tasks": result.failed_tasks}
                )

            return result

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self._emit_status(
                ExecutionStatus.FAILED,
                f"❌ Execution failed: {str(e)}",
                "error",
                0.0,
                {"error": error_msg}
            )
            return ExecutionResult(
                success=False,
                applied_files=[],
                failed_tasks=[str(e)],
                execution_time=time.time() - start_time,
                total_tokens=0,
                coder_count=0,
                summary=str(e)
            )

    async def _real_execute(
        self,
        task: str,
        pre_analyze: bool,
        max_parallel: int
    ) -> ExecutionResult:
        """Execute using real Manager or CoderAgent based on mode"""
        if not self._agent:
            raise RuntimeError("FlowAgent not initialized")

        mode = self._config.get("execution_mode", "parallel")

        # Callback wrapper für CoderAgent Logs
        def coder_log_bridge(section, content):
            # Mappt Coder-Logs auf StatusUpdates für die UI
            status_map = {
                "LLM": ExecutionStatus.GENERATING,
                "TOOL CALL": ExecutionStatus.EXECUTING,
                "TOOL RESULT": ExecutionStatus.EXECUTING,
                "IO": ExecutionStatus.SYNCING,
                "PARSER": ExecutionStatus.VALIDATING,
                "ERROR": ExecutionStatus.FAILED
            }
            status = status_map.get(section, ExecutionStatus.EXECUTING)

            # Details für das UI-Icon extrahieren
            details = {}
            if section == "LLM":
                details = {"thinking": content}
            elif section == "TOOL CALL":
                details = {"tool": content}
            elif section == "IO":
                details = {"file": content}

            self._emit_status(
                status=status,
                message=f"[{section}] {content[:100]}...",
                phase="direct_coder",
                progress=0.5,
                details=details
            )

        # --- MODE SELECTION ---
        if mode == "direct":
            # DIRECT CODER MODE
            from coder import CoderAgent, CoderResult

            self._emit_status(ExecutionStatus.EXECUTING, "🤖 Starting Direct Coder Agent...", "init", 0.1)

            coder = CoderAgent(
                self._agent,
                str(self.workspace_path),
                config={
                    "log_handler": coder_log_bridge,  # Inject Log Handler
                    "model": self._agent.amd.complex_llm_model,
                    "bash_timeout": 300
                }
            )

            res: CoderResult = await coder.execute(task)

            return ExecutionResult(
                success=res.success,
                applied_files=res.changed_files,
                failed_tasks=[res.message] if not res.success else [],
                execution_time=0.0,  # CoderResult hat keine Zeit, könnte man messen
                total_tokens=res.tokens_used,
                coder_count=1,
                summary=res.message
            )

        elif mode in ["sequential", "parallel", "swarm"]:
            # MANAGER MODES
            from toolboxv2.mods.isaa.CodingAgent.manager import SequentialManager, ParallelManager, ManagerResult

            # Config anpassen
            mgr_config = {
                **self._config,
                "max_parallel": max_parallel if mode != "sequential" else 1
            }

            if mode == "sequential":
                self._emit_status(ExecutionStatus.DECOMPOSING, "🔄 Starting Sequential Manager...", "init", 0.1)
                mgr = SequentialManager(self._agent, str(self.workspace_path), mgr_config)
            else:
                # Default to Parallel (covers 'parallel' and 'swarm')
                self._emit_status(ExecutionStatus.DECOMPOSING, f"⚡ Starting Parallel Manager ({mode})...", "init", 0.1)
                mgr = ParallelManager(self._agent, str(self.workspace_path), mgr_config)

            # Wrapper um Manager Logs abzufangen (falls Manager Logs unterstützt)
            # Hinweis: Der Manager nutzt intern CoderAgents, diese müssten die Config durchgereicht bekommen.
            # Hier gehen wir davon aus, dass der Manager Standard-Logging nutzt.

            result: ManagerResult = await mgr.run(task)

            return ExecutionResult(
                success=result.success,
                applied_files=result.applied_files,
                failed_tasks=result.failed_tasks,
                execution_time=result.duration_s,
                total_tokens=result.total_tokens,
                coder_count=result.coder_count,
                summary=result.summary
            )

        else:
            raise ValueError(f"Unknown execution mode: {mode}")

    async def _mock_execute(self, task: str) -> ExecutionResult:
        """Mock execution for testing without ToolBoxV2"""
        self._emit_status(
            ExecutionStatus.DECOMPOSING,
            "🧩 [Mock] Decomposing task...",
            "decompose",
            0.25,
            {"thinking": "Simulating task decomposition..."}
        )
        await asyncio.sleep(0.5)

        # Simulate subtasks
        subtasks = [
            {"id": "t1", "description": f"Analyze requirements for: {task[:50]}"},
            {"id": "t2", "description": f"Implement core functionality"},
            {"id": "t3", "description": f"Write tests and documentation"},
        ]

        self._emit_status(
            ExecutionStatus.EXECUTING,
            f"🚀 [Mock] Executing {len(subtasks)} parallel subtasks...",
            "execute",
            0.4,
            {"thinking": f"Simulating parallel execution with {len(subtasks)} workers..."}
        )
        await asyncio.sleep(1.0)

        self._emit_status(
            ExecutionStatus.VALIDATING,
            "🔍 [Mock] Validating work...",
            "validate",
            0.7,
            {"thinking": "Simulating validation checks..."}
        )
        await asyncio.sleep(0.5)

        self._emit_status(
            ExecutionStatus.MERGING,
            "🔀 [Mock] Merging results...",
            "merge",
            0.9,
            {"thinking": "Simulating result merge..."}
        )
        await asyncio.sleep(0.3)

        return ExecutionResult(
            success=True,
            applied_files=["mock_file1.py", "mock_file2.py"],
            failed_tasks=[],
            execution_time=2.3,
            total_tokens=1500,
            coder_count=3,
            summary=f"[Mock] Completed: {task[:80]}"
        )

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a file"""
        full_path = self.workspace_path / file_path
        if full_path.exists():
            return full_path.read_text(encoding='utf-8')
        return None

    def list_files(self) -> List[str]:
        """List all files in workspace"""
        files = []
        for path in self.workspace_path.rglob('*'):
            if path.is_file():
                files.append(str(path.relative_to(self.workspace_path)))
        return sorted(files)

    async def close(self):
        """Cleanup resources"""
        pass
