"""Manager Agents for orchestrating multiple CoderAgents.

Three variants with increasing complexity:
  1. SequentialManager  — Pipeline: analyze → refine → execute → validate → accept
  2. ParallelManager    — Fan-out: decompose → spawn N coders → monitor → merge
  3. SwarmManager       — Adaptive: LLM-driven planning loop with dynamic re-planning

All variants use CoderPool from coder_toolset and share the same result type.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular deps
CoderAgent = None
CoderPool = None
MessageBuffer = None


def _ensure_imports():
    global CoderAgent, CoderPool, MessageBuffer
    if CoderAgent is None:
        from toolboxv2.mods.isaa.CodingAgent.coder import CoderAgent as _CA
        from toolboxv2.mods.isaa.CodingAgent.coder_toolset import CoderPool as _CP, MessageBuffer as _MB
        CoderAgent, CoderPool, MessageBuffer = _CA, _CP, _MB


# ═══════════════════════════════════════════════════════════════
#  SHARED TYPES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ManagerResult:
    success: bool
    summary: str
    applied_files: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    total_tokens: int = 0
    coder_count: int = 0
    duration_s: float = 0.0


@dataclass
class SubTask:
    """A decomposed piece of work for a single CoderAgent."""
    description: str
    scope: list[str] = field(default_factory=list)  # file hints
    depends_on: list[str] = field(default_factory=list)  # other subtask IDs
    priority: int = 0  # higher = more important
    id: str = ""


@dataclass
class PreAnalysisResult:
    """Result of pre-analysis phase before ParallelManager."""
    enhanced_task: str  # Refined task with context
    suggested_subtasks: list[SubTask] = field(default_factory=list)  # Optional pre-decomposition
    file_hints: list[str] = field(default_factory=list)  # Relevant files to consider
    risks: list[str] = field(default_factory=list)  # Potential issues
    confidence: float = 0.0  # How confident the analysis is (0-1)


class PreAnalyzer:
    """Optional pre-analysis phase for ParallelManager.

    Analyzes the task and project BEFORE spawning parallel coders.
    Can enhance the task description, suggest file scope, and identify risks.

    Benefits:
    - Better task decomposition
    - Focused file scoping
    - Early risk detection
    - Reduced wasted LLM calls

    Usage:
        analyzer = PreAnalyzer(agent, project_root)
        analysis = await analyzer.analyze(task)

        # Use enhanced task or suggested subtasks
        if analysis.suggested_subtasks:
            # Skip decomposition, use pre-computed subtasks
            subtasks = analysis.suggested_subtasks
        else:
            # Let ParallelManager decompose
            enhanced_task = analysis.enhanced_task
    """

    def __init__(self, agent, project_root: str, config: dict | None = None):
        _ensure_imports()
        self.agent = agent
        self.root = project_root
        self.config = config or {}

    async def analyze(self, task: str) -> PreAnalysisResult:
        """Analyze task before parallel execution.

        Returns enhanced task description and optional pre-decomposition.
        """
        # Step 1: Quick project scan
        context = await self._scan_project()

        # Step 2: LLM-based analysis
        analysis = await self._llm_analyze(task, context)

        return analysis

    async def _scan_project(self) -> dict:
        """Fast project structure scan."""
        from toolboxv2.mods.isaa.CodingAgent.coder_toolset import discover_files
        files = discover_files(self.root)

        # Group by file type for better context
        by_ext = {}
        for f in files:
            ext = Path(f).suffix.lower() or "none"
            by_ext.setdefault(ext, []).append(f)

        return {
            "tree": files[:500],
            "total_files": len(files),
            "by_extension": by_ext,
            "has_tests": any("test" in f.lower() for f in files),
            "main_languages": sorted(by_ext.keys(), key=lambda k: len(by_ext[k]), reverse=True)[:3],
        }

    async def _llm_analyze(self, task: str, context: dict) -> PreAnalysisResult:
        """LLM-based analysis to enhance task and suggest decomposition."""

        tree_str = "\n".join(context["tree"][:100])
        langs_str = ", ".join(context["main_languages"])

        prompt = f"""Analyze this development task and provide structured analysis.

TASK: {task}

PROJECT CONTEXT:
- Total files: {context['total_files']}
- Main languages: {langs_str}
- Has tests: {context['has_tests']}

FILE STRUCTURE (sample):
{tree_str}

Provide analysis in JSON format:
{{
    "enhanced_task": "More specific task with context and constraints",
    "file_hints": ["list of relevant files/patterns to focus on"],
    "risks": ["potential issues or dependencies to watch"],
    "can_parallelize": true/false,
    "suggested_subtasks": [
        {{"id": "t1", "description": "subtask 1", "scope": ["files"], "depends_on": [], "priority": 1}},
        ...
    ],
    "confidence": 0.8
}}

Rules:
- enhanced_task should be specific, with explicit file/module targets if possible
- file_hints should be specific file paths or patterns (e.g. ["auth/*.py", "utils/*"])
- Only suggest subtasks if the task is clearly decomposable into independent pieces
- Max 4 suggested subtasks
- confidence should be 0.5-1.0 based on how clear the requirements are
"""

        try:
            resp = await self.agent.a_run_llm_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            # Clean and parse JSON
            clean = resp.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)

            # Convert to PreAnalysisResult
            subtasks = []
            for st in data.get("suggested_subtasks", []):
                subtasks.append(SubTask(
                    id=st.get("id", ""),
                    description=st["description"],
                    scope=st.get("scope", []),
                    depends_on=st.get("depends_on", []),
                    priority=st.get("priority", 0),
                ))

            return PreAnalysisResult(
                enhanced_task=data.get("enhanced_task", task),
                suggested_subtasks=subtasks,
                file_hints=data.get("file_hints", []),
                risks=data.get("risks", []),
                confidence=data.get("confidence", 0.5),
            )
        except Exception as e:
            logger.warning(f"Pre-analysis LLM failed: {e}, using fallback")
            return PreAnalysisResult(
                enhanced_task=task,
                suggested_subtasks=[],
                file_hints=[],
                risks=[],
                confidence=0.0,
            )


# ═══════════════════════════════════════════════════════════════
#  VARIANT 1: SEQUENTIAL PIPELINE
# ═══════════════════════════════════════════════════════════════

class SequentialManager:
    """Simple pipeline manager. Processes tasks one at a time.

    Flow: analyze → refine → execute → validate → accept/reject

    Best for:
      - Single focused tasks
      - Tasks with strong dependencies
      - When you want predictable, reviewable steps
    """

    def __init__(self, agent, project_root: str, config: dict | None = None):
        _ensure_imports()
        self.agent = agent
        self.root = project_root
        self.config = config or {}
        self.pool = CoderPool()
        self.max_retries = self.config.get("max_retries", 2)

    async def run(self, task: str) -> ManagerResult:
        t0 = time.time()
        total_tokens = 0

        # Step 1: Analyze project
        logger.info(f"[Sequential] Analyzing project for: {task[:80]}")
        context = await self._analyze()

        # Step 2: Refine task with context
        refined = await self._refine(task, context)
        refined_task = refined.get("task", task)
        scope = refined.get("scope", [])
        logger.info(f"[Sequential] Refined: {refined_task[:120]}")

        # Step 3: Execute with retries
        for attempt in range(1 + self.max_retries):
            coder = CoderAgent(self.agent, self.root, self.config)
            result = await coder.execute(refined_task)
            total_tokens += result.tokens_used

            if not result.success:
                logger.warning(f"[Sequential] Attempt {attempt+1} failed: {result.message}")
                coder.worktree.cleanup()
                if attempt < self.max_retries:
                    # Feed error back into refined task for next attempt
                    refined_task = (
                        f"{refined_task}\n\n"
                        f"VORHERIGER VERSUCH FEHLGESCHLAGEN: {result.message}\n"
                        f"Versuche einen anderen Ansatz."
                    )
                    continue
                return ManagerResult(
                    success=False,
                    summary=f"Failed after {attempt+1} attempts: {result.message}",
                    total_tokens=total_tokens,
                    coder_count=attempt + 1,
                    duration_s=time.time() - t0,
                )

            # Step 4: Validate
            report = await self._validate(coder)

            if not report["passed"]:
                logger.warning(f"[Sequential] Validation failed: {report}")
                if attempt < self.max_retries:
                    # Inject validation errors and retry
                    errors = json.dumps(report.get("files", {}), indent=2)
                    refined_task = (
                        f"{refined_task}\n\n"
                        f"VALIDATION ERRORS:\n{errors}\n"
                        f"Korrigiere diese Fehler."
                    )
                    coder.worktree.cleanup()
                    continue

            # Step 5: Accept
            changed = await coder.worktree.apply_back()
            applied = result.changed_files
            coder.worktree.cleanup()

            return ManagerResult(
                success=True,
                summary=f"Completed: {len(applied)} files changed",
                applied_files=applied,
                total_tokens=total_tokens,
                coder_count=attempt + 1,
                duration_s=time.time() - t0,
            )

        # Should not reach here, but safety net
        return ManagerResult(
            success=False, summary="Exhausted retries",
            total_tokens=total_tokens, duration_s=time.time() - t0,
        )

    async def _analyze(self) -> dict:
        """Quick project structure scan."""
        from toolboxv2.mods.isaa.CodingAgent.coder_toolset import discover_files
        files = discover_files(self.root)
        return {
            "tree": files[:300],
            "total_files": len(files),
        }

    async def _refine(self, task: str, context: dict) -> dict:
        tree_str = "\n".join(context["tree"][:80])
        prompt = (
            f"Aufgabe: {task}\n\n"
            f"Projektstruktur ({context['total_files']} Dateien, Auszug):\n{tree_str}\n\n"
            "Verfeinere die Aufgabe: welche Dateien, welche Funktionen?\n"
            "JSON: {\"task\": \"...\", \"scope\": [\"files\"], \"risks\": [\"...\"]}"
        )
        resp = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}], stream=False)
        try:
            clean = resp.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(clean)
        except Exception:
            return {"task": task, "scope": [], "risks": []}

    async def _validate(self, coder) -> dict:
        """Run syntax + lint checks on worktree."""
        import shutil, subprocess
        wt = coder.worktree
        report = {"passed": True, "files": {}}
        if not wt.path:
            return report

        changed = await wt.changed_files()
        for f in changed:
            fp = wt.path / f
            if not fp.exists():
                continue
            if f.endswith(".py"):
                try:
                    compile(fp.read_text(errors="replace"), f, "exec")
                except SyntaxError as e:
                    report["passed"] = False
                    report["files"][f] = f"SyntaxError L{e.lineno}: {e.msg}"

        if shutil.which("ruff"):
            try:
                r = subprocess.run(
                    ["ruff", "check", "--select", "E,F", str(wt.path)],
                    capture_output=True, text=True, timeout=30)
                import re
                errs = [l for l in r.stdout.splitlines()
                        if re.match(r'.+:\d+:\d+: [EF]\d+', l)]
                if errs:
                    report["passed"] = False
                    report["lint_errors"] = errs[:20]
            except Exception:
                pass
        return report


# ═══════════════════════════════════════════════════════════════
#  VARIANT 2: PARALLEL FAN-OUT
# ═══════════════════════════════════════════════════════════════

class ParallelManager:
    """Fan-out manager. Decomposes task, spawns N coders in parallel.

    Flow: analyze → decompose → spawn all → monitor loop → validate each → merge

    Best for:
      - Large tasks touching independent modules
      - Refactoring across many files
      - When subtasks don't depend on each other
    """

    def __init__(self, agent, project_root: str, config: dict | None = None):
        _ensure_imports()
        self.agent = agent
        self.root = project_root
        self.config = config or {}
        self.max_parallel = self.config.get("max_parallel", 4)
        self.poll_interval = self.config.get("poll_interval", 5.0)
        self.timeout = self.config.get("manager_timeout", 600)
        self.pre_analyze = self.config.get("pre_analyze", True)  # NEW: optional pre-analysis

    async def run(self, task: str) -> ManagerResult:
        t0 = time.time()

        # Step 0: Optional pre-analysis (NEW)
        if self.pre_analyze:
            try:
                pre_analyzer = PreAnalyzer(self.agent, self.root, self.config)
                pre_result = await pre_analyzer.analyze(task)

                # Use enhanced task if confidence is high enough
                if pre_result.confidence >= 0.5:
                    task = pre_result.enhanced_task

                # If pre-analyzer suggested subtasks, use them directly
                if pre_result.suggested_subtasks and pre_result.confidence >= 0.7:
                    logger.info(f"[Parallel] Using pre-analyzed decomposition: {len(pre_result.suggested_subtasks)} subtasks")
                    subtasks = pre_result.suggested_subtasks
                else:
                    # Fall back to decomposition with enhanced context
                    context = await self._analyze()
                    # Merge pre-analysis context
                    context["file_hints"] = pre_result.file_hints
                    context["risks"] = pre_result.risks
                    subtasks = await self._decompose(task, context)
            except Exception as e:
                logger.warning(f"[Parallel] Pre-analysis failed: {e}, falling back to standard")
                context = await self._analyze()
                subtasks = await self._decompose(task, context)
        else:
            # Step 1: Analyze + Decompose (original path)
            context = await self._analyze()
            subtasks = await self._decompose(task, context)

        if not subtasks:
            # Single task, no decomposition needed → delegate to sequential
            seq = SequentialManager(self.agent, self.root, self.config)
            return await seq.run(task)

        logger.info(f"[Parallel] Decomposed into {len(subtasks)} subtasks")

        # Step 2: Spawn coders (respecting max_parallel)
        coders: dict[str, tuple[CoderAgent, SubTask]] = {}
        pending = list(subtasks)
        results_per_task: dict[str, Any] = {}
        total_tokens = 0

        async def _spawn_batch(batch: list[SubTask]):
            tasks = {}
            for st in batch:
                coder = CoderAgent(self.agent, self.root, self.config)
                coder.worktree.setup()
                tasks[st.id] = (coder, st, asyncio.create_task(coder.execute(st.description)))
            return tasks

        # Step 3: Process in waves
        wave = 0
        all_applied = []
        failed_tasks = []

        while pending or coders:
            # Fill up to max_parallel
            can_spawn = self.max_parallel - len(coders)
            batch = []
            remaining = []
            for st in pending:
                if can_spawn <= 0:
                    remaining.append(st)
                    continue
                # Check dependencies
                if st.depends_on and not all(d in results_per_task for d in st.depends_on):
                    remaining.append(st)
                    continue
                batch.append(st)
                can_spawn -= 1
            pending = remaining

            if batch:
                wave += 1
                logger.info(f"[Parallel] Wave {wave}: spawning {len(batch)} coders")
                new_tasks = await _spawn_batch(batch)
                for sid, (coder, st, future) in new_tasks.items():
                    coders[sid] = (coder, st, future)

            if not coders:
                break

            # Wait for any to complete
            futures = {sid: f for sid, (_, _, f) in coders.items()}
            done, _ = await asyncio.wait(
                futures.values(), timeout=self.poll_interval,
                return_when=asyncio.FIRST_COMPLETED)

            # Collect completed
            completed_ids = [sid for sid, f in futures.items() if f in done]
            for sid in completed_ids:
                coder, st, future = coders.pop(sid)
                try:
                    result = future.result()
                    total_tokens += result.tokens_used
                    results_per_task[sid] = result

                    if result.success:
                        # Validate before accepting
                        report = await self._validate_coder(coder)
                        if report["passed"]:
                            applied = await coder.worktree.apply_back()
                            all_applied.extend(result.changed_files)
                            logger.info(f"[Parallel] {sid} done: {len(result.changed_files)} files")
                        else:
                            failed_tasks.append(f"{sid}: validation failed")
                            logger.warning(f"[Parallel] {sid} validation failed")
                    else:
                        failed_tasks.append(f"{sid}: {result.message}")
                        logger.warning(f"[Parallel] {sid} failed: {result.message}")
                except Exception as e:
                    failed_tasks.append(f"{sid}: {e}")
                    logger.error(f"[Parallel] {sid} exception: {e}")
                finally:
                    coder.worktree.cleanup()

            # Timeout check
            if time.time() - t0 > self.timeout:
                for sid, (coder, st, future) in coders.items():
                    future.cancel()
                    coder.worktree.cleanup()
                    failed_tasks.append(f"{sid}: timeout")
                coders.clear()
                break

        success = len(failed_tasks) == 0
        return ManagerResult(
            success=success,
            summary=f"{len(all_applied)} files applied, {len(failed_tasks)} failed",
            applied_files=all_applied,
            failed_tasks=failed_tasks,
            total_tokens=total_tokens,
            coder_count=len(subtasks),
            duration_s=time.time() - t0,
        )

    async def _analyze(self) -> dict:
        from toolboxv2.mods.isaa.CodingAgent.coder_toolset import discover_files
        files = discover_files(self.root)
        return {"tree": files[:500], "total_files": len(files)}

    async def _decompose(self, task: str, context: dict) -> list[SubTask]:
        """LLM-driven task decomposition into independent subtasks."""
        tree_str = "\n".join(context["tree"][:120])
        prompt = (
            f"Aufgabe: {task}\n\n"
            f"Projektstruktur ({context['total_files']} Dateien):\n{tree_str}\n\n"
            "Zerlege in UNABHÄNGIGE Teilaufgaben die PARALLEL bearbeitet werden können.\n"
            "Wenn die Aufgabe nicht zerlegbar ist, antworte mit leerem Array.\n"
            "JSON Array: [{\"id\": \"t1\", \"description\": \"...\", \"scope\": [\"files\"], "
            "\"depends_on\": [], \"priority\": 1}]\n"
            "Regeln:\n"
            "- Jede Teilaufgabe MUSS eigenständig ausführbar sein\n"
            "- depends_on nur wenn WIRKLICH nötig (shared interfaces)\n"
            "- scope = Dateien die diese Teilaufgabe berührt\n"
            "- Max 6 Teilaufgaben"
        )
        resp = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}], stream=False)
        try:
            clean = resp.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            raw = json.loads(clean)
            if not isinstance(raw, list) or len(raw) == 0:
                return []
            return [SubTask(
                id=item.get("id", f"t{i}"),
                description=item["description"],
                scope=item.get("scope", []),
                depends_on=item.get("depends_on", []),
                priority=item.get("priority", 0),
            ) for i, item in enumerate(raw)]
        except Exception:
            return []

    async def _validate_coder(self, coder) -> dict:
        """Reuse SequentialManager's validation."""
        seq = SequentialManager(self.agent, self.root, self.config)
        return await seq._validate(coder)


# ═══════════════════════════════════════════════════════════════
#  VARIANT 3: ADAPTIVE SWARM
# ═══════════════════════════════════════════════════════════════

class SwarmManager:
    """LLM-driven adaptive manager with dynamic re-planning.

    The manager itself is an LLM agent that:
      - Plans and decomposes tasks
      - Spawns coders and monitors progress
      - Re-plans when coders report blockers
      - Can split, merge, or reassign work mid-flight
      - Makes accept/reject decisions based on validation

    Best for:
      - Complex multi-step tasks with unknowns
      - When requirements may shift during execution
      - When subtasks have complex interdependencies
    """

    MANAGER_SYSTEM = (
        "Du bist ein Manager-Agent der CoderAgents orchestriert.\n"
        "Du hast folgende Tools:\n"
        "- plan(task, context) → Erstelle/Update den Ausführungsplan\n"
        "- spawn(subtask) → Starte einen CoderAgent für eine Teilaufgabe\n"
        "- observe() → Status aller Coders + Nachrichten abfragen\n"
        "- respond(coder_id, message) → Nachricht an Coder senden\n"
        "- validate(coder_id) → Worktree prüfen\n"
        "- accept(coder_id) → Änderungen übernehmen\n"
        "- reject(coder_id, reason) → Änderungen ablehnen, Coder neu starten\n"
        "- finish(summary) → Gesamtaufgabe abschließen\n\n"
        "REGELN:\n"
        "1. Plane ZUERST, bevor du Coders startest\n"
        "2. Beobachte regelmäßig (observe) — Coders können Fragen haben\n"
        "3. Validiere IMMER bevor du akzeptierst\n"
        "4. Bei Fehlern: analysiere, passe Plan an, starte Coder neu\n"
        "5. Beende mit finish() wenn alle Teilaufgaben erledigt sind\n"
    )

    def __init__(self, agent, project_root: str, config: dict | None = None):
        _ensure_imports()
        self.agent = agent
        self.root = project_root
        self.config = config or {}
        self.pool = CoderPool()
        self.max_iterations = self.config.get("max_manager_iterations", 30)
        self.max_parallel = self.config.get("max_parallel", 4)
        self.timeout = self.config.get("manager_timeout", 900)

        # State
        self._plan: list[SubTask] = []
        self._active_coders: dict[str, tuple[CoderAgent, SubTask, asyncio.Task]] = {}
        self._completed: dict[str, Any] = {}  # coder_id → CoderResult
        self._all_applied: list[str] = []
        self._failed: list[str] = []
        self._total_tokens: int = 0
        self._context: dict = {}

    async def run(self, task: str) -> ManagerResult:
        t0 = time.time()

        # Initial context
        from toolboxv2.mods.isaa.CodingAgent.coder_toolset import discover_files
        files = discover_files(self.root)
        self._context = {"tree": files[:500], "total_files": len(files)}

        # Manager tools
        tools = self._manager_tools()
        messages = [
            {"role": "system", "content": self.MANAGER_SYSTEM},
            {"role": "user", "content": (
                f"AUFGABE: {task}\n\n"
                f"PROJEKT: {self.root} ({len(files)} Dateien)\n"
                f"Struktur (Auszug):\n" + "\n".join(files[:80])
            )},
        ]

        # Manager loop
        for iteration in range(self.max_iterations):
            if time.time() - t0 > self.timeout:
                logger.warning("[Swarm] Timeout reached")
                break

            # Check for completed coders before LLM call
            await self._collect_completed()

            resp = await self.agent.a_run_llm_completion(
                messages=messages, tools=tools,
                stream=False, get_response_message=True)

            content = resp.content or ""
            tool_calls = resp.tool_calls
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if not tool_calls:
                # No tool calls — check for natural finish
                if "finish" in content.lower() or not self._active_coders:
                    break
                continue

            # Execute manager tools
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    result = await self._dispatch_manager(tc.function.name, args)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str, ensure_ascii=False),
                    })

                    # Check for finish signal
                    if tc.function.name == "finish":
                        # Cleanup remaining coders
                        await self._cleanup_all()
                        return ManagerResult(
                            success=len(self._failed) == 0,
                            summary=args.get("summary", content),
                            applied_files=self._all_applied,
                            failed_tasks=self._failed,
                            total_tokens=self._total_tokens,
                            coder_count=len(self._completed),
                            duration_s=time.time() - t0,
                        )
                except Exception as e:
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps({"error": str(e)}),
                    })

        # Timeout/max iterations reached — collect what we have
        await self._cleanup_all()
        return ManagerResult(
            success=len(self._failed) == 0 and len(self._all_applied) > 0,
            summary=f"Ended after {iteration+1} iterations",
            applied_files=self._all_applied,
            failed_tasks=self._failed,
            total_tokens=self._total_tokens,
            coder_count=len(self._completed),
            duration_s=time.time() - t0,
        )

    async def _dispatch_manager(self, name: str, args: dict) -> dict:
        """Route manager tool calls."""
        if name == "plan":
            return await self._tool_plan(args)
        if name == "spawn":
            return await self._tool_spawn(args)
        if name == "observe":
            return await self._tool_observe()
        if name == "respond":
            return await self._tool_respond(args)
        if name == "validate":
            return await self._tool_validate(args)
        if name == "accept":
            return await self._tool_accept(args)
        if name == "reject":
            return await self._tool_reject(args)
        if name == "finish":
            return {"status": "finishing", "applied": self._all_applied, "failed": self._failed}
        return {"error": f"Unknown tool: {name}"}

    async def _tool_plan(self, args: dict) -> dict:
        """Create or update execution plan via LLM."""
        task = args.get("task", "")
        prompt = (
            f"Erstelle einen Ausführungsplan für: {task}\n"
            f"Kontext: {json.dumps(self._context, default=str)[:2000]}\n"
            f"Bereits erledigt: {list(self._completed.keys())}\n"
            f"Fehlgeschlagen: {self._failed}\n\n"
            "JSON: [{\"id\": \"t1\", \"description\": \"...\", \"scope\": [], "
            "\"depends_on\": [], \"priority\": 1}]"
        )
        resp = await self.agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}], stream=False)
        try:
            clean = resp.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            raw = json.loads(clean)
            self._plan = [SubTask(
                id=item.get("id", f"t{i}"),
                description=item["description"],
                scope=item.get("scope", []),
                depends_on=item.get("depends_on", []),
                priority=item.get("priority", 0),
            ) for i, item in enumerate(raw)]
            return {"plan": [asdict(st) for st in self._plan]}
        except Exception as e:
            return {"error": f"Plan parsing failed: {e}", "raw": resp[:500]}

    async def _tool_spawn(self, args: dict) -> dict:
        """Spawn a CoderAgent for a subtask."""
        if len(self._active_coders) >= self.max_parallel:
            return {"error": f"Max parallel ({self.max_parallel}) reached. Wait or kill a coder."}

        task_desc = args.get("task", args.get("subtask", ""))
        if not task_desc:
            return {"error": "No task description provided"}

        cid = self.pool.next_id()
        coder = CoderAgent(self.agent, self.root, self.config)

        async def _run():
            return await coder.execute(task_desc)

        future = asyncio.create_task(_run())
        st = SubTask(id=cid, description=task_desc, scope=args.get("scope", []))
        self._active_coders[cid] = (coder, st, future)
        return {"coder_id": cid, "task": task_desc[:200], "status": "spawned"}

    async def _tool_observe(self) -> dict:
        """Get status of all coders."""
        await self._collect_completed()

        status = {}
        for cid, (coder, st, future) in self._active_coders.items():
            wt = coder.worktree
            status[cid] = {
                "status": "done" if future.done() else "running",
                "task": st.description[:120],
                "tokens": coder.tracker.total_tokens,
                "changed_files": await wt.changed_files() if wt.path else [],
                "state": coder.state,
            }

        for cid, result in self._completed.items():
            status[cid] = {
                "status": "completed" if result.success else "failed",
                "task": "...",
                "files": result.changed_files,
                "message": result.message,
            }

        return {
            "active": len(self._active_coders),
            "completed": len(self._completed),
            "applied": len(self._all_applied),
            "failed": len(self._failed),
            "coders": status,
        }

    async def _tool_respond(self, args: dict) -> dict:
        """Send message to a running coder."""
        cid = args.get("coder_id", "")
        msg = args.get("message", "")
        if cid not in self._active_coders:
            return {"error": f"Coder {cid} not active"}
        coder, _, _ = self._active_coders[cid]
        if hasattr(coder, '_pending_messages'):
            coder._pending_messages.append(msg)
            return {"sent": True}
        return {"sent": False, "reason": "No message injection available"}

    async def _tool_validate(self, args: dict) -> dict:
        """Validate a completed coder's worktree."""
        cid = args.get("coder_id", "")
        # Check completed coders
        for active_cid, (coder, st, future) in self._active_coders.items():
            if active_cid == cid and future.done():
                seq = SequentialManager(self.agent, self.root, self.config)
                return await seq._validate(coder)
        return {"error": f"Coder {cid} not found or still running"}

    async def _tool_accept(self, args: dict) -> dict:
        """Accept a coder's changes into origin."""
        cid = args.get("coder_id", "")
        files = args.get("files")  # Optional cherry-pick

        if cid not in self._active_coders:
            return {"error": f"Coder {cid} not found"}

        coder, st, future = self._active_coders[cid]
        if not future.done():
            return {"error": f"Coder {cid} still running"}

        try:
            result = future.result()
            self._total_tokens += result.tokens_used

            if files:
                applied = await coder.worktree.apply_files(files)
            else:
                await coder.worktree.apply_back()
                applied = result.changed_files

            self._all_applied.extend(applied)
            self._completed[cid] = result
            coder.worktree.cleanup()
            del self._active_coders[cid]
            return {"accepted": applied, "coder_id": cid}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_reject(self, args: dict) -> dict:
        """Reject a coder's work and optionally respawn."""
        cid = args.get("coder_id", "")
        reason = args.get("reason", "")
        respawn = args.get("respawn", False)

        if cid not in self._active_coders:
            return {"error": f"Coder {cid} not found"}

        coder, st, future = self._active_coders[cid]
        if not future.done():
            future.cancel()

        coder.worktree.cleanup()
        self._failed.append(f"{cid}: {reason}")
        del self._active_coders[cid]

        if respawn:
            new_task = f"{st.description}\n\nVORHERIGER VERSUCH ABGELEHNT: {reason}"
            return await self._tool_spawn({"task": new_task, "scope": st.scope})

        return {"rejected": cid, "reason": reason}

    async def _collect_completed(self):
        """Move completed futures out of active set (internal bookkeeping)."""
        done_ids = [cid for cid, (_, _, f) in self._active_coders.items() if f.done()]
        # Don't auto-remove — let the manager decide via accept/reject
        pass

    async def _cleanup_all(self):
        """Cleanup all remaining active coders."""
        for cid, (coder, st, future) in list(self._active_coders.items()):
            if not future.done():
                future.cancel()
            try:
                if future.done():
                    result = future.result()
                    self._total_tokens += result.tokens_used
            except Exception:
                pass
            coder.worktree.cleanup()
        self._active_coders.clear()

    def _manager_tools(self) -> list:
        """Tool definitions for the manager LLM."""
        return [
            {"type": "function", "function": {
                "name": "plan",
                "description": "Erstelle oder aktualisiere den Ausführungsplan. Nutze nach Fehlern um neu zu planen.",
                "parameters": {"type": "object", "properties": {
                    "task": {"type": "string", "description": "Aufgabe oder aktualisierte Aufgabe"},
                }, "required": ["task"]},
            }},
            {"type": "function", "function": {
                "name": "spawn",
                "description": "Starte einen CoderAgent für eine Teilaufgabe. Max parallel begrenzt.",
                "parameters": {"type": "object", "properties": {
                    "task": {"type": "string", "description": "Konkrete Aufgabenbeschreibung für den Coder"},
                    "scope": {"type": "array", "items": {"type": "string"},
                              "description": "Relevante Dateipfade (optional)"},
                }, "required": ["task"]},
            }},
            {"type": "function", "function": {
                "name": "observe",
                "description": "Status aller Coders abrufen: Running/Done/Failed, Dateien, Tokens, Nachrichten.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "respond",
                "description": "Nachricht an einen laufenden Coder senden (Kurs-Korrektur, Antwort auf Frage).",
                "parameters": {"type": "object", "properties": {
                    "coder_id": {"type": "string"},
                    "message": {"type": "string"},
                }, "required": ["coder_id", "message"]},
            }},
            {"type": "function", "function": {
                "name": "validate",
                "description": "Prüfe Worktree eines fertigen Coders: Syntax, Lint, Diff.",
                "parameters": {"type": "object", "properties": {
                    "coder_id": {"type": "string"},
                }, "required": ["coder_id"]},
            }},
            {"type": "function", "function": {
                "name": "accept",
                "description": "Übernimm Änderungen eines Coders ins Origin-Repo. Optional Cherry-Pick.",
                "parameters": {"type": "object", "properties": {
                    "coder_id": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"},
                              "description": "Nur diese Dateien übernehmen (optional, default=alle)"},
                }, "required": ["coder_id"]},
            }},
            {"type": "function", "function": {
                "name": "reject",
                "description": "Lehne Arbeit eines Coders ab. Optional mit Respawn (neuer Versuch).",
                "parameters": {"type": "object", "properties": {
                    "coder_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Warum abgelehnt"},
                    "respawn": {"type": "boolean", "description": "Neuen Coder mit korrigierter Aufgabe starten?"},
                }, "required": ["coder_id", "reason"]},
            }},
            {"type": "function", "function": {
                "name": "finish",
                "description": "Gesamtaufgabe abschließen. Rufe auf wenn ALLE Teilaufgaben erledigt oder abgebrochen.",
                "parameters": {"type": "object", "properties": {
                    "summary": {"type": "string", "description": "Zusammenfassung was gemacht wurde"},
                }, "required": ["summary"]},
            }},
        ]
