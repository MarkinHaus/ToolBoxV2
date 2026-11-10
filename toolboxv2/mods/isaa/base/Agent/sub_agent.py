"""
Sub-Agent System - Parallel Task Execution for FlowAgent

Features:
- spawn_sub_agent: Create isolated sub-agents for focused tasks
- wait_for: Await completion of one or more sub-agents
- Parallel execution with async support
- VFS write isolation (sub-agent can only write to output_dir)
- Fixed token budget per sub-agent
- Max depth = 1 (NO sub-sub-agents!)

Architecture:
- Sub-agents share tools with parent (no tool loading overhead)
- Sub-agents have isolated history (clean context)
- Sub-agents can READ entire VFS
- Sub-agents can ONLY WRITE to their output_dir
- Results are injected into parent's AutoFocus

Author: FlowAgent V3
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from toolboxv2 import get_logger
logger = get_logger()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SubAgentStatus(Enum):
    """Sub-agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"  # NEW: Hit max iterations
    PAUSED = "paused"  # NEW: Manually paused


@dataclass
class SubAgentConfig:
    """Configuration for sub-agent execution"""
    max_tokens: int = int(os.getenv("SUB_AGENT_MAX_TOKENS", "512000"))  # Token budget for this sub-agent
    max_iterations: int = int(os.getenv("SUB_AGENT_MAX_ITERATIONS", "120"))  # Max execution iterations
    timeout_seconds: int = int(os.getenv("SUB_AGENT_MAX_TIMEOUT", "1800"))  # Timeout in seconds

    # Inherited from parent (set at spawn time)
    model_preference: str = "fast"


@dataclass
class SubAgentState:
    """Runtime state of a sub-agent"""
    id: str
    task: str
    output_dir: str
    config: SubAgentConfig

    # Status
    status: SubAgentStatus = SubAgentStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    result: str | None = None
    error: str | None = None
    tokens_used: int = 0
    iterations_used: int = 0
    files_written: list[str] = field(default_factory=list)

    # Async handling
    _task: asyncio.Task | None = field(default=None, repr=False)
    _chunk_queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)

    # NEW: Resume support
    execution_context: Optional['ExecutionContext'] = None  # Preserved context
    resumable: bool = False  # Can this be resumed?
    original_budget: int = 5000  # Original token budget
    session_id: str = ""  # Session ID
    engine: Optional[Any] = None  # Sub-agent engine instance
    runtime: str = "v3"  # "v3" = ExecutionEngine, "page" = PageEngine (native)
    tools_called: int = 0  # page runtime: Anzahl Tool-Calls (-> resumable)
    page_path: str = ""  # page runtime: read-only Sicht auf Page+Offloads im VFS des Parents
    _page_refreshed: float = field(default=0.0, repr=False)


@dataclass
class PageResumeHandle:
    """Resume-Handle fuer Page-Sub-Agents (ersetzt den V3-ExecutionContext).

    Der eigentliche Zustand (page + tail) lebt in ``engine.sessions[session_id]`` und auf Disk."""
    engine: Any
    session_id: str
    steps_used: int = 0


@dataclass
class SubAgentResult:
    """Result returned from wait_for"""
    id: str
    success: bool
    status: SubAgentStatus
    result: str | None
    error: str | None
    output_dir: str
    files_written: list[str]
    tokens_used: int
    duration_seconds: float
    task: str = ""  # NEW: Original task
    token_budget: int = 0  # Fix R2 09.11.2026: Budget im Result (state wird nach completed geloescht)
    page_path: str = ""  # page runtime: read-only Sicht auf Page+Offloads im VFS des Parents

    # NEW: Resume support fields
    max_iterations_reached: bool = False  # NEW: Hit max iterations?
    resumable: bool = False  # NEW: Can be resumed?
    iterations_used: int = 0  # NEW: Total iterations used
    execution_context: Optional['ExecutionContext'] = None  # NEW: Preserved context


# =============================================================================
# SUB-AGENT MANAGER
# =============================================================================

class SubAgentManager:
    """
    Manages sub-agent lifecycle and execution.

    Responsibilities:
    - Spawn sub-agents with isolated contexts
    - Track running sub-agents
    - Handle wait_for logic (single or multiple)
    - Enforce token budgets
    - Collect and report results

    CRITICAL: Enforces max_depth=1 (no sub-sub-agents)
    """

    def __init__(
        self,
        parent_engine: Any,  # ExecutionEngine
        parent_session: Any,  # AgentSessionV2
        is_sub_agent: bool = False
    ):
        """
        Initialize SubAgentManager.

        Args:
            parent_engine: Parent ExecutionEngine instance
            parent_session: Parent session for VFS access
            is_sub_agent: If True, spawn is DISABLED (prevents sub-sub-agents)
        """
        self.parent_engine = parent_engine
        self.parent_session = parent_session
        self.is_sub_agent = is_sub_agent

        # Active sub-agents
        self._sub_agents: dict[str, SubAgentState] = {}

        # Completed results (kept for reference)
        self._completed: dict[str, SubAgentResult] = {}
        # Shared queue for forwarding sub-agent chunks to parent stream
        self._chunk_queue: asyncio.Queue = asyncio.Queue()

    def can_spawn(self) -> bool:
        """Check if spawning is allowed (False if already a sub-agent)"""
        return not self.is_sub_agent

    async def spawn(
        self,
        task: str,
        output_dir: str,
        wait: bool = True,
        budget: int = 64000,
        max_iterations: int = 120,
        timeout: int = 1800
    ) -> str | SubAgentResult:
        """
        Spawn a new sub-agent.

        Args:
            task: Task description for the sub-agent
            output_dir: VFS directory where sub-agent can write (e.g., /sub/research)
            wait: If True, wait for completion. If False, return immediately with ID.
            budget: Token budget for this sub-agent
            timeout: Timeout in seconds

        Returns:
            If wait=True: SubAgentResult
            If wait=False: sub_agent_id (str)

        Raises:
            RuntimeError: If called from a sub-agent (depth > 1)
        """
        if self.is_sub_agent:
            raise RuntimeError(
                "Sub-agents cannot spawn other sub-agents! "
                "Max depth is 1 to prevent infinite delegation."
            )

        # Generate unique ID
        sub_id = f"sub_{uuid.uuid4().hex[:8]}"

        # Normalize output_dir
        if not output_dir.startswith("/"):
            output_dir = f"/{output_dir}"
        if not output_dir.startswith("/global/sub/"):
            # Force sub-agent outputs under /global/sub/ (disk-backed via global mount)
            output_dir = f"/global/sub{output_dir}"

        # Create config
        config = SubAgentConfig(
            max_tokens=budget,
            timeout_seconds=timeout,
            max_iterations=max_iterations,
            model_preference=getattr(self.parent_engine, 'model_preference', 'fast')
        )


        # Create state
        state = SubAgentState(
            id=sub_id,
            task=task,
            output_dir=output_dir,
            config=config
        )

        self._sub_agents[sub_id] = state
        if wait:
            # Direkt awaiten – KEIN Task vorher erstellen
            try:
                await asyncio.wait_for(
                    self._run_sub_agent(state),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                state.status = SubAgentStatus.TIMEOUT
                state.error = f"Timeout after {timeout} seconds"
                state.completed_at = datetime.now()
            except Exception as e:
                if state.status == SubAgentStatus.RUNNING:
                    state.status = SubAgentStatus.FAILED
                    state.error = str(e)
                    state.completed_at = datetime.now()

            duration = 0.0
            if state.started_at and state.completed_at:
                duration = (state.completed_at - state.started_at).total_seconds()

            result = SubAgentResult(
                id=sub_id, success=state.status == SubAgentStatus.COMPLETED,
                status=state.status, result=state.result, error=state.error,
                output_dir=state.output_dir, files_written=state.files_written,
                tokens_used=state.tokens_used, duration_seconds=duration,
                task=state.task,
                token_budget=state.original_budget,  # Fix R2
                page_path=state.page_path,
                max_iterations_reached=(state.status == SubAgentStatus.MAX_ITERATIONS),
                resumable=state.resumable, iterations_used=state.iterations_used,
                execution_context=state.execution_context if state.resumable else None,
            )
            self._completed[sub_id] = result
            return result
        else:
            # Async: NUR hier Task erstellen
            state._task = asyncio.create_task(
                self._run_sub_agent(state)
            )
            return sub_id

    async def wait_for(
        self,
        sub_agent_ids: str | list[str],
        timeout: int = 300
    ) -> dict[str, SubAgentResult]:
        """
        Wait for one or more sub-agents to complete.

        Args:
            sub_agent_ids: Single ID or list of IDs
            timeout: Timeout in seconds (applies to all)

        Returns:
            Dict mapping sub_agent_id to SubAgentResult
        """
        if isinstance(sub_agent_ids, str):
            if '[' in sub_agent_ids:
                import json
                ids = json.loads(sub_agent_ids)
            else:
                ids = [sub_agent_ids]
        else:
            ids = list(sub_agent_ids)

        results = {}

        # Wait for all in parallel
        tasks = []
        for sub_id in ids:
            if sub_id in self._completed:
                # Already completed
                results[sub_id] = self._completed[sub_id]
            elif sub_id in self._sub_agents:
                tasks.append((sub_id, self._wait_single(sub_id, timeout)))
            else:
                # Unknown ID
                results[sub_id] = SubAgentResult(
                    id=sub_id,
                    success=False,
                    status=SubAgentStatus.FAILED,
                    result=None,
                    error=f"Unknown sub-agent ID: {sub_id}",
                    output_dir="",
                    files_written=[],
                    tokens_used=0,
                    duration_seconds=0
                )

        # Await all pending
        if tasks:
            gathered = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True
            )

            for (sub_id, _), result in zip(tasks, gathered):
                if isinstance(result, Exception):
                    results[sub_id] = SubAgentResult(
                        id=sub_id,
                        success=False,
                        status=SubAgentStatus.FAILED,
                        result=None,
                        error=str(result),
                        output_dir=self._sub_agents.get(sub_id,
                                                        SubAgentState(sub_id, "", "", SubAgentConfig())).output_dir,
                        files_written=[],
                        tokens_used=0,
                        duration_seconds=0
                    )
                else:
                    results[sub_id] = result

        return results

    async def _wait_single(self, sub_id: str, timeout: int) -> SubAgentResult:
        """Wait for a single sub-agent to complete"""
        state = self._sub_agents.get(sub_id)

        if not state:
            return SubAgentResult(
                id=sub_id,
                success=False,
                status=SubAgentStatus.FAILED,
                result=None,
                error=f"Unknown sub-agent: {sub_id}",
                output_dir="",
                files_written=[],
                tokens_used=0,
                duration_seconds=0
            )

        if state._task is None:
            return SubAgentResult(
                id=sub_id,
                success=False,
                status=SubAgentStatus.FAILED,
                result=None,
                error="Sub-agent task not started",
                output_dir=state.output_dir,
                files_written=[],
                tokens_used=0,
                duration_seconds=0
            )

        try:
            # Wait with timeout
            await asyncio.wait_for(state._task, timeout=timeout)
        except asyncio.TimeoutError:
            state.status = SubAgentStatus.TIMEOUT
            state.error = f"The wait command asyncio.TimeoutError sub agents still running."
            state.completed_at = datetime.now()

            # Cancel the task
            state._task.cancel()
            try:
                await state._task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            state.status = SubAgentStatus.FAILED
            state.error = str(e)
            state.completed_at = datetime.now()

        # Build result
        duration = 0.0
        if state.started_at and state.completed_at:
            duration = (state.completed_at - state.started_at).total_seconds()

        result = SubAgentResult(
            id=sub_id,
            success=state.status == SubAgentStatus.COMPLETED,
            status=state.status,
            result=state.result,
            error=state.error,
            output_dir=state.output_dir,
            files_written=state.files_written,
            tokens_used=state.tokens_used,
            duration_seconds=duration,
            task=state.task,  # NEW
            token_budget=state.original_budget,  # Fix R2
            page_path=state.page_path,
            max_iterations_reached=(state.status == SubAgentStatus.MAX_ITERATIONS),  # NEW
            resumable=state.resumable,  # NEW
            iterations_used=state.iterations_used,  # NEW
            execution_context=state.execution_context if state.resumable else None  # NEW
        )

        # Move to completed. Resumable states stay in _sub_agents as well:
        # resume_sub_agent needs the live SubAgentState (engine + preserved
        # ExecutionContext) — deleting it here made every resume after a
        # wait_for fail with "nicht gefunden".
        self._completed[sub_id] = result
        if not state.resumable:
            del self._sub_agents[sub_id]

        return result

    # ------------------------------------------------------------------
    # Engine selection: page (default) | v3
    # ------------------------------------------------------------------
    def _sub_engine_kind(self) -> str:
        """``ISAA_SUB_AGENT_ENGINE`` / ``amd.sub_agent_engine`` = "page" (default) | "v3".
        Faellt auf v3 zurueck, wenn die Page-Engine nicht importierbar ist."""
        agent = getattr(self.parent_engine, "agent", None)
        kind = (os.getenv("ISAA_SUB_AGENT_ENGINE")
                or getattr(getattr(agent, "amd", None), "sub_agent_engine", None)
                or "page").lower()
        if kind != "page":
            return "v3"
        try:
            from toolboxv2.mods.isaa.base.Agent.page.isaa import make_page_sub_engine  # noqa: F401
        except Exception as e:
            logger.warning("Sub-agent: page engine unavailable (%s) -> falling back to V3", e)
            return "v3"
        if not hasattr(agent, "session_manager"):
            logger.warning("Sub-agent: parent agent has no session_manager -> falling back to V3")
            return "v3"
        return "page"

    async def _run_sub_agent(self, state: SubAgentState):
        """Dispatch: native PageEngine (default) oder V3 ExecutionEngine."""
        if self._sub_engine_kind() == "page":
            await self._run_page_sub_agent(state)
        else:
            await self._run_v3_sub_agent(state)

    async def _run_page_sub_agent(self, state: SubAgentState):
        """Sub-Agent nativ auf der PageEngine.

        Gleiche Garantien wie V3: eigene Session ``<parent>__sub__<id>``, Schreibzugriff nur auf
        ``output_dir`` (RestrictedVFSWrapper), geteilte Tools, kein Sub-Sub-Agent (die Spawn-Tools
        existieren nur im V3-Main-Agent), Chunk-Forwarding mit ``_sub_agent_id``, Resume-Faehigkeit."""
        from toolboxv2.mods.isaa.base.Agent.page.isaa import make_page_sub_engine

        state.status = SubAgentStatus.RUNNING
        state.started_at = datetime.now()
        state.runtime = "page"
        agent = self.parent_engine.agent
        try:
            state.session_id = f"{self.parent_session.session_id}__sub__{state.id}"
            state.original_budget = state.config.max_tokens
            try:
                session = await agent.session_manager.get_or_create(state.session_id)
                agent.init_session_tools(session)
                state.engine = make_page_sub_engine(
                    agent, output_dir=state.output_dir,
                    max_steps=state.config.max_iterations, budget=state.config.max_tokens,
                )
                # Fix 09.11.2026: echten /global-Mount-Root an die Sub-Engine reichen -
                # _do_code_run stellt ihn Sub-Agent-Code als __VFS_ROOT__ bereit
                _mounts = getattr(self.parent_session.vfs, "mounts", None) or {}
                _gm = _mounts.get("/global") if hasattr(_mounts, "get") else None
                state.engine.vfs_global_root = str(getattr(_gm, "local_path", "") or "")
            except Exception as e:  # Setup-Fehler VOR dem Start -> sauberer V3-Fallback, nie still
                logger.warning("Sub-agent %s: page setup failed (%s) -> falling back to V3", state.id, e)
                state.runtime, state.engine = "v3", None
                return await self._run_v3_sub_agent(state)
            # Isolation VOR dem ersten Tool-Call; Page-Store darf nach /.page/<sid> persistieren
            session.vfs = RestrictedVFSWrapper(
                session.vfs, state.output_dir,
                extra_write_dirs=(f"/.page/{state.session_id}",),
            )
            self._mount_page_view(state)  # Parent kann Page + Offloads nativ lesen
            await self._consume_page_stream(state, state.task, state.config.max_iterations)
            self._finish_page_sub(state)
        except Exception as e:
            state.status = SubAgentStatus.FAILED
            state.error = str(e)
            try:
                self.parent_session.vfs.write(
                    f"{state.output_dir}/error.log",
                    f"# Sub-Agent Error\n\n**Task:** {state.task}\n\n**Error:** {str(e)}\n",
                )
            except Exception:
                pass
        finally:
            state.completed_at = datetime.now()
            if not state.resumable:
                await self._close_page_sub(state)

    async def _consume_page_stream(self, state: SubAgentState, query: str, max_steps: int):
        """Page-Stream konsumieren, State live halten, Chunks V3-kompatibel weiterreichen."""
        engine = state.engine
        base_iters = state.iterations_used
        steps, answer, success = 0, "", None
        try:
            async for chunk in engine.stream(query, session=state.session_id):
                t = chunk.get("type")
                if t == "iteration_start":
                    steps = chunk.get("iteration", steps)
                    state.iterations_used = base_iters + steps
                    self._refresh_page_view(state)
                elif t in ("content", "content_chunk"):
                    state.tokens_used += len(chunk.get("chunk", "")) // 4
                    if t == "content_chunk":
                        chunk = {**chunk, "type": "content"}  # V3-Name fuer den CLI-Renderer
                elif t == "tool_start":
                    state.tools_called += 1
                    engine.live.narrator_msg = f"[{steps}] {chunk.get('name', '')}"
                elif t == "tool_result":
                    # Fix 09.11.2026: Tool-Result-Tokens mitzaehlen - vorher zeigte die
                    # Live-Anzeige nur len(final_answer)//4 (~6% des Realverbrauchs)
                    state.tokens_used += (len(chunk.get("result", "")) + 3) // 4
                elif t == "final_answer":
                    answer, success = chunk.get("answer", ""), bool(chunk.get("success"))

                chunk["_sub_agent_id"] = state.id
                chunk["_sub_agent_task"] = state.task[:60]
                try:
                    self._chunk_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

            hit_max = success is False and steps >= max_steps and answer.startswith("Stopped after")
            if hit_max:
                state.status = SubAgentStatus.MAX_ITERATIONS
                state.result = answer
                state.resumable = state.tools_called > 0
                state.execution_context = (
                    PageResumeHandle(engine, state.session_id, state.iterations_used)
                    if state.resumable else None
                )
                self._chunk_queue.put_nowait({
                    "type": "max_iterations", "answer": answer,
                    "_sub_agent_id": state.id, "_sub_agent_task": state.task[:60],
                })
            else:
                state.resumable, state.execution_context = False, None
                state.result = answer
                if success:
                    state.status = SubAgentStatus.COMPLETED
                else:
                    state.status = SubAgentStatus.FAILED
                    state.error = "Execution finished without success flag"
            self._chunk_queue.put_nowait({"type": "_sub_done", "_sub_agent_id": state.id})
        except (RuntimeError, Exception) as e:
            if "Event loop is closed" in str(e):  # Windows httpx cleanup race (wie V3)
                if answer and state.status == SubAgentStatus.RUNNING:
                    state.result, state.status = answer, SubAgentStatus.COMPLETED
                elif state.status == SubAgentStatus.RUNNING:
                    state.status, state.error = SubAgentStatus.FAILED, f"Stream interrupted: {str(e)[:200]}"
            else:
                raise

    def _finish_page_sub(self, state: SubAgentState):
        """files_written einsammeln + result.md schreiben (identisch zum V3-Pfad)."""
        self._flush_sub_vfs(state)
        self._refresh_page_view(state, force=True)
        vfs = self.parent_session.vfs
        try:
            ls_result = vfs.ls(state.output_dir, recursive=True)
            if ls_result.get("success"):
                state.files_written = [f["path"] for f in ls_result.get("files", [])]
        except Exception:
            pass
        result_path = f"{state.output_dir}/result.md"
        try:
            existing = vfs.read(result_path)
            if existing.get("success") and existing.get("content", "").strip():
                result_path = f"{state.output_dir}/_result.md"
            if not vfs.read(result_path).get("success"):
                vfs.write(
                    result_path,
                    f"# Sub-Agent Result\n\n**Task:** {state.task}\n\n"
                    f"**Status:** {state.status.value}\n\n**Result:**\n{state.result}\n",
                )
                if result_path not in state.files_written:
                    state.files_written.append(result_path)
        except Exception:
            pass

    async def _close_page_sub(self, state: SubAgentState):
        """Sessions/Kernels/Locks der Sub-Engine freigeben (nur wenn nicht resumable)."""
        engine = state.engine
        if engine is None or state.runtime != "page":
            return
        try:
            from toolboxv2.mods.isaa.base.Agent.page.isaa import close_page_sub_engine
            await close_page_sub_engine(engine)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Parent-Sicht auf Page + Offloads des Sub-Agents (read-only Mount)
    # ------------------------------------------------------------------
    def _mount_page_view(self, state: SubAgentState) -> None:
        """Mountet ``<root>/<sub_sid>`` (state.json, versions.jsonl, Offload-Texte) read-only
        im VFS des Parents.  Pfad = ``/.page/<sub_sid>`` (gleiche Konvention wie der Page-Store:
        Ref ``<sid>/x`` == ``/.page/<sid>/x``); Fallback ``/_subpage/<sub_sid>``.
        Nie fatal: scheitert der Mount, laeuft der Sub-Agent normal weiter."""
        vfs = getattr(self.parent_session, "vfs", None)
        if vfs is None or not hasattr(vfs, "mount") or state.engine is None:
            return
        local = Path(state.engine.root) / state.session_id
        try:
            local.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Sub-agent %s: page dir %s not creatable: %s", state.id, local, e)
            return
        for vpath in (f"/.page/{state.session_id}", f"/_subpage/{state.session_id}"):
            try:
                res = vfs.mount(str(local), vfs_path=vpath, readonly=True, auto_sync=False)
            except Exception as e:
                logger.warning("Sub-agent %s: mount %s failed: %s", state.id, vpath, e)
                continue
            if isinstance(res, dict) and res.get("success") is False:
                logger.debug("Sub-agent %s: mount %s refused: %s", state.id, vpath, res.get("error"))
                continue
            state.page_path = vpath
            return

    def _refresh_page_view(self, state: SubAgentState, force: bool = False) -> None:
        """Mount neu scannen (neue Offloads / state.json).  Gedrosselt auf 2 s, ``force`` ignoriert das."""
        if not state.page_path:
            return
        now = time.monotonic()
        if not force and now - state._page_refreshed < 2.0:
            return
        state._page_refreshed = now
        try:
            self.parent_session.vfs.refresh_mount(state.page_path)
        except Exception as e:
            logger.debug("Sub-agent %s: refresh_mount failed: %s", state.id, e)

    def _page_path_for(self, sub_id: str, result: "SubAgentResult | None" = None) -> str:
        """Page-Pfad aus dem Result, sonst (z.B. nach resume) aus dem noch lebenden State."""
        st = self._sub_agents.get(sub_id)
        return (result.page_path if result is not None else "") or (st.page_path if st else "")

    def _flush_sub_vfs(self, state: SubAgentState) -> None:
        """Dirty Offloads der Sub-Session auf Disk schreiben, damit der Parent-Mount sie sieht."""
        try:
            agent = self.parent_engine.agent
            sess = agent.session_manager.sessions.get(state.session_id)
            if sess is not None:
                sess.vfs.sync_all()
        except Exception as e:
            logger.debug("Sub-agent %s: sync_all failed: %s", state.id, e)

    async def resume_page(
        self, state: SubAgentState, additional_iterations: int = 10,
        additional_budget: int = 3000, context: str | None = None,
    ) -> None:
        """Page-Sub-Agent fortsetzen: gleiche Session (page+tail intakt), neues Step-Limit."""
        engine = state.engine
        engine.cfg = replace(engine.cfg, max_steps=int(additional_iterations))
        state.original_budget += additional_budget
        state.status, state.resumable = SubAgentStatus.RUNNING, False
        query = "Continue the task where you stopped; your page and notes are intact."
        if context:
            query += f"\n\nADDITIONAL CONTEXT FROM MAIN AGENT:\n{context}"
        query += f"\n\nOriginal task: {state.task}"
        try:
            await self._consume_page_stream(state, query, int(additional_iterations))
            self._finish_page_sub(state)
        finally:
            state.completed_at = datetime.now()
            if not state.resumable:
                await self._close_page_sub(state)

    async def _run_v3_sub_agent(self, state: SubAgentState):
        """
        Execute sub-agent in isolated context (V3 ExecutionEngine).

        Creates a new ExecutionEngine instance with:
        - is_sub_agent=True (prevents further spawning)
        - VFS write restricted to output_dir
        - Shared tools from parent
        - Fresh history (isolated context)
        """
        state.status = SubAgentStatus.RUNNING
        state.started_at = datetime.now()

        try:
            # Import here to avoid circular imports
            from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

            # Create sub-agent engine
            # CRITICAL: is_sub_agent=True prevents spawn_sub_agent from being available
            sub_engine = ExecutionEngine(
                agent=self.parent_engine.agent,
                human_online=False,
                is_sub_agent=True,
                sub_agent_output_dir=state.output_dir,
                sub_agent_budget=state.config.max_tokens
            )

            # Store engine and session_id for resume
            state.engine = sub_engine
            state.session_id = f"{self.parent_session.session_id}__sub__{state.id}"
            state.original_budget = state.config.max_tokens

            # Execute with sub-agent context - STREAMING MODE
            # We consume the stream internally to update state live for the CLI
            stream_gen, ctx = await sub_engine.execute_stream(
                query=state.task,
                session_id=state.session_id,
                max_iterations=state.config.max_iterations
            )

            final_answer_acc = ""

            # Consume stream to keep state alive
            try:
                async for chunk in stream_gen(ctx):
                    c_type = chunk.get("type")

                    # Live State Updates for CLI
                    state.iterations_used = ctx.current_iteration

                    if c_type == "content":
                        txt = chunk.get("chunk", "")
                        state.tokens_used += len(txt) // 4
                        final_answer_acc += txt

                    elif c_type == "tool_start":
                        pass

                    elif c_type == "final_answer":
                        final_answer_acc = chunk.get("answer", "")

                    elif c_type == "max_iterations":
                        state.result = chunk.get("answer", "Max iterations reached")
                        state.status = SubAgentStatus.MAX_ITERATIONS
                        if len(ctx.tools_used) > 0:
                            state.resumable = True
                            state.execution_context = ctx
                        else:
                            state.resumable = False

                    elif c_type == "done":
                        state.result = chunk.get("final_answer", final_answer_acc)
                        if chunk.get("success", False):
                            state.status = SubAgentStatus.COMPLETED
                        elif state.status != SubAgentStatus.MAX_ITERATIONS:
                            state.status = SubAgentStatus.FAILED
                            state.error = "Execution finished without success flag"

                    # >>> Forward chunk to parent stream <
                    chunk["_sub_agent_id"] = state.id
                    chunk["_sub_agent_task"] = state.task[:60]
                    try:
                        self._chunk_queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        pass  # Drop chunk if queue full (shouldn't happen)

                # Signal: this sub-agent's stream is done
                self._chunk_queue.put_nowait({
                    "type": "_sub_done",
                    "_sub_agent_id": state.id,
                })
            except (RuntimeError, Exception) as e:
                err_str = str(e)
                if "Event loop is closed" in err_str:
                    # Windows httpx cleanup race condition –
                    # Stream war aktiv, Ergebnis evtl. teilweise da
                    print(f"[SubAgent {state.id}] Stream cleanup error (Windows httpx): {err_str[:100]}")
                    if final_answer_acc and state.status == SubAgentStatus.RUNNING:
                        state.result = final_answer_acc
                        state.status = SubAgentStatus.COMPLETED
                    elif state.status == SubAgentStatus.RUNNING:
                        state.status = SubAgentStatus.FAILED
                        state.error = f"Stream interrupted: {err_str[:200]}"
                else:
                    raise  # Andere Fehler normal propagieren
            # Ensure final state consistency
            if state.status == SubAgentStatus.RUNNING:
                # Fallback if loop ended without explicit done/max_iter
                state.status = SubAgentStatus.COMPLETED
                state.result = final_answer_acc

            # Final token sync from agent
            if sub_engine.agent:
                # Add overhead that might have been missed
                state.tokens_used = sub_engine.agent.total_tokens_out + sub_engine.agent.total_tokens_in

            # Collect files written
            # Check VFS for files in output_dir
            try:
                ls_result = self.parent_session.vfs.ls(state.output_dir, recursive=True)
                if ls_result.get("success"):
                    state.files_written = [
                        f["path"] for f in ls_result.get("files", [])
                    ]
            except Exception:
                pass

            result = final_answer_acc

            # Write result to output_dir/result.md if not already done by sub-agent
            # If sub-agent wrote real content → preserve it, use _result.md instead
            result_path = f"{state.output_dir}/result.md"
            try:
                existing = self.parent_session.vfs.read(result_path)
                if existing.get("success") and existing.get("content", "").strip():
                    # Sub-agent already wrote real content — don't overwrite
                    result_path = f"{state.output_dir}/_result.md"
                existing2 = self.parent_session.vfs.read(result_path)
                if not existing2.get("success"):
                    # Write framework result summary (only if no file exists at result_path)
                    self.parent_session.vfs.write(
                        result_path,
                        f"# Sub-Agent Result\n\n"
                        f"**Task:** {state.task}\n\n"
                        f"**Status:** {state.status.value}\n\n"
                        f"**Result:**\n{result}\n"
                    )
                    if result_path not in state.files_written:
                        state.files_written.append(result_path)
            except Exception:
                pass

        except Exception as e:
            state.status = SubAgentStatus.FAILED
            state.error = str(e)

            # Write error to output_dir
            try:
                error_path = f"{state.output_dir}/error.log"
                self.parent_session.vfs.write(
                    error_path,
                    f"# Sub-Agent Error\n\n"
                    f"**Task:** {state.task}\n\n"
                    f"**Error:** {str(e)}\n"
                )
            except Exception:
                pass

        finally:
            state.completed_at = datetime.now()

    def get_status(self, sub_id: str) -> dict | None:
        """Get status of a sub-agent"""
        if sub_id in self._sub_agents:
            state = self._sub_agents[sub_id]

            # Extract live narrator message if engine is running
            narrator_msg = ""
            if state.engine and hasattr(state.engine, "live"):
                narrator_msg = getattr(state.engine.live, "narrator_msg", "")

            info = {
                "id": sub_id,
                "status": state.status.value,
                "task": state.task,
                "output_dir": state.output_dir,
                "started_at": state.started_at.isoformat() if state.started_at else None,
                "tokens_used": state.tokens_used,
                "token_budget": state.config.max_tokens,
                "narrator_msg": narrator_msg
            }
            if state.page_path:  # page runtime: Page + Offloads per vfs_shell/vfs_view lesbar
                info["page_path"] = state.page_path
            return info
        elif sub_id in self._completed:
            result = self._completed[sub_id]
            info = {
                "id": sub_id,
                "status": result.status.value,
                "success": result.success,
                "task": result.task,
                "output_dir": result.output_dir,
                "files_written": result.files_written,
                "tokens_used": result.tokens_used,
                # Fix 09.11.2026 (R2): Budget aus Result - state ist hier bereits
                # aus _sub_agents entfernt, getattr(None,...) lieferte immer 0.
                "token_budget": result.token_budget,
                "duration_seconds": result.duration_seconds
            }
            page_path = self._page_path_for(sub_id, result)
            if page_path:
                info["page_path"] = page_path
            return info
        return None

    def get_all_status(self) -> dict:
        """Get status of all sub-agents"""
        return {
            "running": {
                sid: self.get_status(sid)
                for sid in self._sub_agents
            },
            "completed": {
                sid: self.get_status(sid)
                for sid in self._completed
            }
        }

    def format_results_for_auto_focus(self, results: dict[str, SubAgentResult]) -> str:
        """Format sub-agent results for AutoFocus injection"""
        lines = ["SUB-AGENT ERGEBNISSE:"]

        for sub_id, result in results.items():
            status_icon = "✅" if result.success else "❌"
            lines.append(
                f"• {status_icon} [{sub_id}]: {result.status.value} "
                f"→ {result.output_dir}"
            )

            # NEW: Show resume info for MAX_ITERATIONS
            if result.max_iterations_reached:
                lines.append(f"  ⏱️  Max Iterations erreicht!")
                if result.resumable:
                    lines.append(f"  ℹ️  Resumable: Ja (nutze resume_sub_agent('{sub_id}'))")
                else:
                    lines.append(f"  ℹ️  Resumable: Nein (keine Tools verwendet)")

            if result.files_written:
                files_str = ", ".join(result.files_written[:3])
                if len(result.files_written) > 3:
                    files_str += f" (+{len(result.files_written) - 3})"
                lines.append(f"  Files: {files_str}")

            page_path = self._page_path_for(sub_id, result)
            if page_path:
                lines.append(
                    f"  Page: {page_path}/ (read-only; state.json, versions.jsonl; "
                    f"Offload-Ref '<sid>/x' = {page_path}/x)"
                )

            if result.error:
                lines.append(f"  Error: {result.error[:60]}...")

        return "\n".join(lines)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

SUB_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spawn_sub_agent",
            "description": """Spawn a sub-agent to handle a focused task in parallel.

USE WHEN:
- Task can be split into independent parts
- You want to run multiple searches/analyses in parallel
- A sub-task needs focused attention without context pollution

SUB-AGENT LIMITATIONS:
- Can READ entire VFS
- Can ONLY WRITE to its output_dir (e.g., /sub/research_x/)
- CANNOT spawn further sub-agents (max depth = 1)
- Has fixed token budget
- Gets task in query, no access to current conversation

PROVIDE CLEAR TASKS: Sub-agents cannot ask clarifying questions!""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear, complete task description. Include ALL context needed."
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory for sub-agent output (e.g., 'research_topic1'). Will be created under /sub/"
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "True = wait for completion. False = run async, use wait_for later.",
                        "default": True
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Token budget for sub-agent (default: 64000)",
                        "default": 64000
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Max execution iterations for sub-agent (default: 65)",
                        "default": 120
                    }
                },
                "required": ["task", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for",
            "description": """Wait for one or more sub-agents to complete.

Use after spawning sub-agents with wait=False.
Returns results from all specified sub-agents.
Results are automatically added to your context.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agent_ids": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ],
                        "description": "Single sub-agent ID or list of IDs to wait for"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 300)",
                        "default": 300
                    }
                },
                "required": ["sub_agent_ids"]
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "sub_agents_status",
            "description": """Check the current status, token usage, and narrator progress of sub-agents.
Use this to monitor running sub-agents without blocking your execution. If no ID is provided, returns an overview of all.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agent_id": {
                        "type": "string",
                        "description": "Optional: The specific ID of a sub-agent. Omit to get an overview of all active and completed sub-agents."
                    }
                }
            }
        }
    },
]


# =============================================================================
# VFS WRITE RESTRICTION WRAPPER
# =============================================================================

class RestrictedVFSWrapper:
    """
    Wraps VFS to restrict write operations to a specific directory.

    Used by sub-agents to enforce output isolation.
    """

    def __init__(self, vfs: Any, allowed_write_dir: str, extra_write_dirs: tuple = ()):
        """
        Args:
            vfs: Original VFS instance
            allowed_write_dir: Only directory where writes are allowed
            extra_write_dirs: weitere Schreib-Praefixe fuer Engine-Interna (Page-Store: /.page/<sid>)
        """
        self._vfs = vfs
        self._allowed_dir = allowed_write_dir.rstrip("/")
        self._extra_dirs = tuple(d.rstrip("/") for d in extra_write_dirs)

    def _is_write_allowed(self, path: str) -> bool:
        """Check if write to path is allowed"""
        # Normalize path
        if not path.startswith("/"):
            path = f"/{path}"

        # Allow writes only under allowed_dir (+ engine-internal extra dirs)
        return any(
            path.startswith(d + "/") or path == d
            for d in (self._allowed_dir, *self._extra_dirs)
        )

    # Read operations - pass through
    def read(self, path: str) -> dict:
        return self._vfs.read(path)

    def list_files(self) -> dict:
        return self._vfs.list_files()

    def ls(self, path: str = "/", recursive: bool = False) -> dict:
        return self._vfs.ls(path, recursive)

    def open(self, path: str, line_start: int = 1, line_end: int = -1) -> dict:
        return self._vfs.open(path, line_start, line_end)

    # Write operations - restricted
    def write(self, path: str, content: str) -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only write to {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.write(path, content)

    def create(self, path: str, content: str = "") -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only create files in {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.create(path, content)

    def mkdir(self, path: str, parents: bool = False) -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only create directories in {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.mkdir(path, parents)

    def rmdir(self, path: str, force: bool = False) -> dict:
        if not self._is_write_allowed(path):
            return {
                "success": False,
                "error": f"Sub-agent can only remove directories in {self._allowed_dir}. "
                         f"Attempted: {path}"
            }
        return self._vfs.rmdir(path, force)

    def mv(self, source: str, destination: str) -> dict:
        # Both source and destination must be in allowed dir for move
        if not self._is_write_allowed(destination):
            return {
                "success": False,
                "error": f"Sub-agent can only move files within {self._allowed_dir}. "
                         f"Destination: {destination}"
            }
        return self._vfs.mv(source, destination)

    # Pass through other methods
    def __getattr__(self, name: str):
        return getattr(self._vfs, name)


# =============================================================================
# SKILL FOR SUB-AGENT USAGE
# =============================================================================

PARALLEL_SUBTASKS_SKILL = {
    "id": "parallel_subtasks",
    "name": "Parallel Sub-Task Execution",
    "triggers": [
        "parallel", "gleichzeitig", "recherchiere mehrere",
        "vergleiche", "sammle von verschiedenen", "und dann zusammen",
        "mehrere quellen", "verschiedene aspekte"
    ],
    "instruction": """Für parallelisierbare Aufgaben:
1. Identifiziere UNABHÄNGIGE Teilaufgaben (die nicht aufeinander warten)
2. Für jede Teilaufgabe:
   spawn_sub_agent(
     task="KLARE, VOLLSTÄNDIGE Beschreibung - Sub-Agent kann nicht nachfragen!",
     output_dir="aufgabe_name",
     wait=False
   )
3. Sammle alle sub_agent_ids
4. wait_for([alle_ids]) - wartet auf alle parallel
5. Lese Ergebnisse aus /sub/[name]/result.md
6. Kombiniere/Vergleiche die Ergebnisse

WICHTIG:
- Sub-Agents können NUR in ihren output_dir schreiben
- Sub-Agents können NICHT zurückfragen - gib ALLE Infos mit
- Sub-Agents können KEINE weiteren Sub-Agents spawnen
- Bei Fehlern: Prüfe /sub/[name]/error.log""",
    "tools_used": ["think", "spawn_sub_agent", "wait_for", "vfs_read", "final_answer"],
    "tool_groups": ["vfs"],
    "source": "predefined"
}
