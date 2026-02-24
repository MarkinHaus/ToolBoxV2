"""
Job Manager - Persistent Scheduled Agent Tasks
================================================

Core module for defining, persisting, and scheduling agent jobs.
Jobs survive CLI restarts and can auto-wake the CLI via OS schedulers.

Key design:
- TriggerRegistry: Extensible plugin system for custom triggers
- JobScheduler: Async tick loop that evaluates triggers and fires jobs
- JobEventBus: Enables job chaining (A completes -> B fires)
- All persistence via simple JSON file

Author: ISAA Team
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import signal
import socket
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA MODELS
# =============================================================================

class TriggerType(str, Enum):
    """Built-in trigger types. Custom triggers can use any string."""
    ON_TIME = "on_time"
    ON_INTERVAL = "on_interval"
    ON_CRON = "on_cron"
    ON_CLI_START = "on_cli_start"
    ON_CLI_EXIT = "on_cli_exit"
    ON_SYSTEM_BOOT = "on_system_boot"
    ON_SYSTEM_IDLE = "on_system_idle"
    ON_SYSTEM_SHUTDOWN = "on_system_shutdown"
    ON_JOB_COMPLETED = "on_job_completed"
    ON_JOB_FAILED = "on_job_failed"
    ON_JOB_TIMEOUT = "on_job_timeout"
    ON_NETWORK_AVAILABLE = "on_network_available"
    ON_FILE_CHANGED = "on_file_changed"
    ON_WEBHOOK_RECEIVED = "on_webhook_received"

    ON_DREAM_START = "on_dream_start"         # fires when a dream cycle begins
    ON_DREAM_END = "on_dream_end"             # fires when a dream cycle completes
    ON_DREAM_BUDGET_HIT = "on_dream_budget_hit"  # fires when dreamer hits max_budget
    ON_DREAM_SKILL_EVOLVED = "on_dream_skill_evolved"  # fires when a skill was evolved/created
    ON_AGENT_IDLE = "on_agent_idle"           # fires when agent has no runs for N seconds


@dataclass
class TriggerConfig:
    """Configuration for a job trigger."""
    trigger_type: str
    at_datetime: str | None = None          # on_time: ISO datetime
    interval_seconds: int | None = None     # on_interval
    cron_expression: str | None = None      # on_cron
    watch_job_id: str | None = None         # on_job_*
    watch_path: str | None = None           # on_file_changed
    watch_patterns: list[str] | None = None # on_file_changed globs
    webhook_path: str | None = None         # on_webhook_received
    idle_seconds: int | None = None         # on_system_idle threshold
    dream_config_json: str | None = None  # on_dream_*: serialized DreamConfig
    agent_idle_seconds: int | None = None  # on_agent_idle threshold
    extra: dict[str, Any] | None = None     # custom trigger data

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> TriggerConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class JobDefinition:
    """A persistent job that fires an agent query on trigger."""
    job_id: str
    name: str
    agent_name: str
    query: str
    trigger: TriggerConfig
    status: str = "active"         # active / paused / disabled / expired
    session_id: str = "default"
    timeout_seconds: int = 300
    max_retries: int = 0
    created_at: str = ""
    last_run_at: str | None = None
    last_result: str | None = None  # completed / failed / timeout
    run_count: int = 0
    fail_count: int = 0
    _last_fired_ts: float = 0.0    # internal: epoch of last fire

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trigger"] = self.trigger.to_dict()
        d.pop("_last_fired_ts", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> JobDefinition:
        trigger_data = d.pop("trigger", {})
        d.pop("_last_fired_ts", None)
        # Filter unknown fields
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        valid["trigger"] = TriggerConfig.from_dict(trigger_data)
        return cls(**valid)

    @staticmethod
    def generate_id() -> str:
        return f"job_{uuid.uuid4().hex[:8]}"


# =============================================================================
# TRIGGER REGISTRY (Extensible Plugin System)
# =============================================================================

@runtime_checkable
class TriggerEvaluator(Protocol):
    """Interface for trigger evaluators. Implement this for custom triggers."""

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        """Called when job is added or scheduler starts. Set up any watchers."""
        ...

    async def evaluate(self, job: JobDefinition) -> bool:
        """Return True if job should fire now."""
        ...

    async def teardown(self, job: JobDefinition) -> None:
        """Called when job is removed or scheduler stops. Clean up."""
        ...


class TriggerRegistry:
    """Registry for trigger evaluators. Extensible by users/plugins."""

    def __init__(self):
        self._evaluators: dict[str, TriggerEvaluator] = {}

    def register(self, trigger_type: str, evaluator: TriggerEvaluator) -> None:
        """Register a trigger evaluator for a trigger type."""
        self._evaluators[trigger_type] = evaluator
        _log.debug("Registered trigger evaluator: %s", trigger_type)

    def unregister(self, trigger_type: str) -> None:
        """Remove a trigger evaluator."""
        self._evaluators.pop(trigger_type, None)

    def get(self, trigger_type: str) -> TriggerEvaluator | None:
        """Get evaluator for a trigger type."""
        return self._evaluators.get(trigger_type)

    def available_types(self) -> list[str]:
        """List all registered trigger types."""
        return list(self._evaluators.keys())


# =============================================================================
# BUILT-IN TRIGGER EVALUATORS
# =============================================================================

class OnTimeEvaluator:
    """Fire once at a specific datetime, then expire."""

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        if not job.trigger.at_datetime:
            return False
        try:
            target = datetime.fromisoformat(job.trigger.at_datetime)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if now >= target:
                job.status = "expired"
                return True
        except (ValueError, TypeError):
            _log.warning("Invalid at_datetime for job %s", job.job_id)
        return False

    async def teardown(self, job: JobDefinition) -> None:
        pass


class OnIntervalEvaluator:
    """Fire every N seconds."""

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        if not job.trigger.interval_seconds:
            return False
        import time
        now = time.time()
        if now - job._last_fired_ts >= job.trigger.interval_seconds:
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        pass


class OnCronEvaluator:
    """Fire on cron schedule. Requires croniter (optional)."""

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        if not job.trigger.cron_expression:
            return False
        try:
            from croniter import croniter
        except ImportError:
            _log.debug("croniter not installed, skipping cron trigger for %s", job.job_id)
            return False
        try:
            now = datetime.now(timezone.utc)
            base = datetime.fromisoformat(job.last_run_at) if job.last_run_at else now.replace(hour=0, minute=0, second=0)
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            cron = croniter(job.trigger.cron_expression, base)
            next_fire = cron.get_next(datetime)
            if next_fire.tzinfo is None:
                next_fire = next_fire.replace(tzinfo=timezone.utc)
            return now >= next_fire
        except Exception as e:
            _log.warning("Cron eval error for %s: %s", job.job_id, e)
        return False

    async def teardown(self, job: JobDefinition) -> None:
        pass


class OnCliLifecycleEvaluator:
    """Fires on CLI start or exit. Triggered via fire_lifecycle(), not tick."""

    def __init__(self):
        self._pending: set[str] = set()  # job_ids that should fire

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        if job.job_id in self._pending:
            self._pending.discard(job.job_id)
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        self._pending.discard(job.job_id)

    def mark_pending(self, job_id: str):
        """Mark a job to fire on next evaluate."""
        self._pending.add(job_id)


class OnJobEventEvaluator:
    """Fires when a watched job completes/fails/times out."""

    def __init__(self):
        self._pending: set[str] = set()

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        if job.job_id in self._pending:
            self._pending.discard(job.job_id)
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        self._pending.discard(job.job_id)

    def notify(self, event_type: str, source_job_id: str, all_jobs: list[JobDefinition]):
        """Called when a job event happens. Checks if any watching jobs should fire."""
        for job in all_jobs:
            if job.status != "active":
                continue
            if job.trigger.trigger_type == event_type and job.trigger.watch_job_id == source_job_id:
                self._pending.add(job.job_id)


class OnNetworkEvaluator:
    """Fires on network availability transition (offline -> online)."""

    def __init__(self):
        self._was_online: bool | None = None
        self._last_check: float = 0.0
        self._check_interval: float = 30.0

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        import time
        now = time.time()
        if now - self._last_check < self._check_interval:
            return False
        self._last_check = now

        is_online = await asyncio.get_event_loop().run_in_executor(None, self._check_network)

        if self._was_online is not None and not self._was_online and is_online:
            self._was_online = is_online
            return True
        self._was_online = is_online
        return False

    async def teardown(self, job: JobDefinition) -> None:
        pass

    @staticmethod
    def _check_network() -> bool:
        try:
            s = socket.create_connection(("8.8.8.8", 53), timeout=3)
            s.close()
            return True
        except OSError:
            return False


class OnFileChangedEvaluator:
    """Fires when watched files/dirs change. Uses watchdog if available."""

    def __init__(self):
        self._pending: set[str] = set()
        self._observers: dict[str, Any] = {}

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        if not job.trigger.watch_path:
            return
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            _log.debug("watchdog not installed, file watching disabled for %s", job.job_id)
            return

        watch_path = Path(job.trigger.watch_path)
        if not watch_path.exists():
            _log.warning("Watch path does not exist: %s", watch_path)
            return

        evaluator_ref = self
        job_id = job.job_id
        patterns = job.trigger.watch_patterns

        class Handler(FileSystemEventHandler):
            _last_event: float = 0.0

            def on_any_event(self, event):
                import time
                now = time.time()
                if now - self._last_event < 2.0:  # debounce 2s
                    return
                if patterns:
                    import fnmatch
                    src = event.src_path
                    if not any(fnmatch.fnmatch(src, p) for p in patterns):
                        return
                self._last_event = now
                evaluator_ref._pending.add(job_id)

        observer = Observer()
        observer.schedule(Handler(), str(watch_path), recursive=True)
        observer.daemon = True
        observer.start()
        self._observers[job.job_id] = observer
        _log.debug("File watcher started for %s: %s", job.job_id, watch_path)

    async def evaluate(self, job: JobDefinition) -> bool:
        if job.job_id in self._pending:
            self._pending.discard(job.job_id)
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        observer = self._observers.pop(job.job_id, None)
        if observer:
            observer.stop()
            observer.join(timeout=2)


class OnSystemIdleEvaluator:
    """Fires when system has been idle for N seconds."""

    def __init__(self):
        self._last_check: float = 0.0
        self._check_interval: float = 60.0

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        import time
        now = time.time()
        if now - self._last_check < self._check_interval:
            return False
        self._last_check = now

        threshold = job.trigger.idle_seconds or 300
        idle_time = await asyncio.get_event_loop().run_in_executor(None, self._get_idle_time)
        return idle_time >= threshold

    async def teardown(self, job: JobDefinition) -> None:
        pass

    @staticmethod
    def _get_idle_time() -> float:
        """Get system idle time in seconds. Platform-specific."""
        import platform
        system = platform.system()

        if system == "Windows":
            try:
                import ctypes
                from ctypes import Structure, c_uint, sizeof, byref, windll

                class LASTINPUTINFO(Structure):
                    _fields_ = [("cbSize", c_uint), ("dwTime", c_uint)]

                lii = LASTINPUTINFO()
                lii.cbSize = sizeof(LASTINPUTINFO)
                windll.user32.GetLastInputInfo(byref(lii))
                millis = windll.kernel32.GetTickCount() - lii.dwTime
                return millis / 1000.0
            except Exception:
                return 0.0

        elif system == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["ioreg", "-c", "IOHIDSystem"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split("\n"):
                    if "HIDIdleTime" in line:
                        # Value is in nanoseconds
                        ns = int(line.split("=")[-1].strip())
                        return ns / 1_000_000_000.0
            except Exception:
                return 0.0

        elif system == "Linux":
            try:
                # xprintidle gives milliseconds
                import subprocess
                result = subprocess.run(
                    ["xprintidle"], capture_output=True, text=True, timeout=5
                )
                return int(result.stdout.strip()) / 1000.0
            except Exception:
                return 0.0

        return 0.0


class OnSystemShutdownEvaluator:
    """Fires on system/process shutdown. Registered via atexit + signals."""

    def __init__(self):
        self._pending: set[str] = set()
        self._registered = False

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        if not self._registered:
            self._registered = True
            evaluator_ref = self

            def _on_shutdown():
                for jid in list(evaluator_ref._job_ids):
                    evaluator_ref._pending.add(jid)

            atexit.register(_on_shutdown)

            # SIGTERM / SIGINT
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    original = signal.getsignal(sig)

                    def _handler(signum, frame, orig=original):
                        _on_shutdown()
                        if callable(orig) and orig not in (signal.SIG_DFL, signal.SIG_IGN):
                            orig(signum, frame)

                    signal.signal(sig, _handler)
                except (OSError, ValueError):
                    pass

        if not hasattr(self, '_job_ids'):
            self._job_ids: set[str] = set()
        self._job_ids.add(job.job_id)

    async def evaluate(self, job: JobDefinition) -> bool:
        if job.job_id in self._pending:
            self._pending.discard(job.job_id)
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        self._pending.discard(job.job_id)
        if hasattr(self, '_job_ids'):
            self._job_ids.discard(job.job_id)


class OnWebhookEvaluator:
    """Fires when a webhook is received. Requires external route registration."""

    def __init__(self):
        self._pending: set[str] = set()

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        if job.job_id in self._pending:
            self._pending.discard(job.job_id)
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        self._pending.discard(job.job_id)

    def trigger_webhook(self, job_id: str):
        """Called externally when a webhook is received."""
        self._pending.add(job_id)

class OnDreamEventEvaluator:
    """Fires on dream lifecycle events. Triggered programmatically by Dreamer."""

    def __init__(self):
        self._pending: dict[str, set[str]] = {  # event_type -> set of job_ids
            "on_dream_start": set(),
            "on_dream_end": set(),
            "on_dream_budget_hit": set(),
            "on_dream_skill_evolved": set(),
        }
        self._all_jobs: dict[str, str] = {}  # job_id -> trigger_type

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        self._all_jobs[job.job_id] = job.trigger.trigger_type

    async def evaluate(self, job: JobDefinition) -> bool:
        tt = job.trigger.trigger_type
        if tt in self._pending and job.job_id in self._pending[tt]:
            self._pending[tt].discard(job.job_id)
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        for s in self._pending.values():
            s.discard(job.job_id)
        self._all_jobs.pop(job.job_id, None)

    def notify(self, event_type: str, all_jobs: list[JobDefinition]):
        """Called by Dreamer when a dream event occurs."""
        for job in all_jobs:
            if job.status == "active" and job.trigger.trigger_type == event_type:
                self._pending.setdefault(event_type, set()).add(job.job_id)


class OnAgentIdleEvaluator:
    """Fires when agent has had no execution runs for N seconds.
    Perfect trigger for auto-dreaming."""

    def __init__(self):
        self._last_activity: dict[str, float] = {}  # agent_name -> epoch
        self._last_check: float = 0.0

    async def setup(self, job: JobDefinition, scheduler: JobScheduler) -> None:
        pass

    async def evaluate(self, job: JobDefinition) -> bool:
        import time
        now = time.time()
        # Throttle checks to every 30s
        if now - self._last_check < 30.0:
            return False
        self._last_check = now

        threshold = job.trigger.agent_idle_seconds or job.trigger.idle_seconds or 600
        last = self._last_activity.get(job.agent_name, 0.0)
        if last > 0 and (now - last) >= threshold:
            return True
        return False

    async def teardown(self, job: JobDefinition) -> None:
        pass

    def record_activity(self, agent_name: str):
        """Called by ExecutionEngine after every run to reset idle timer."""
        import time
        self._last_activity[agent_name] = time.time()

# =============================================================================
# JOB EVENT BUS (for job chaining)
# =============================================================================

class JobEventBus:
    """Simple event bus for job lifecycle events."""

    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, callback: Callable):
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable):
        if event in self._listeners:
            self._listeners[event] = [cb for cb in self._listeners[event] if cb != callback]

    def emit(self, event: str, data: dict[str, Any] | None = None):
        for cb in self._listeners.get(event, []):
            try:
                cb(event, data or {})
            except Exception as e:
                _log.warning("Event listener error for %s: %s", event, e)


# =============================================================================
# JOB SCHEDULER
# =============================================================================

class JobScheduler:
    """Async scheduler that manages jobs, evaluates triggers, fires callbacks."""

    def __init__(self, jobs_file: Path, fire_callback: Callable):
        self.jobs_file = jobs_file
        self._fire_callback = fire_callback
        self._jobs: dict[str, JobDefinition] = {}
        self._running = False
        self._tick_task: asyncio.Task | None = None
        self._firing: set[str] = set()  # job_ids currently executing

        # Public: extensible trigger registry
        self.trigger_registry = TriggerRegistry()
        self.event_bus = JobEventBus()

        # Register built-in evaluators
        self._cli_lifecycle_eval = OnCliLifecycleEvaluator()
        self._job_event_eval = OnJobEventEvaluator()
        self._network_eval = OnNetworkEvaluator()
        self._file_eval = OnFileChangedEvaluator()
        self._shutdown_eval = OnSystemShutdownEvaluator()
        self._webhook_eval = OnWebhookEvaluator()
        self._idle_eval = OnSystemIdleEvaluator()

        self.trigger_registry.register("on_time", OnTimeEvaluator())
        self.trigger_registry.register("on_interval", OnIntervalEvaluator())
        self.trigger_registry.register("on_cron", OnCronEvaluator())
        self.trigger_registry.register("on_cli_start", self._cli_lifecycle_eval)
        self.trigger_registry.register("on_cli_exit", self._cli_lifecycle_eval)
        self.trigger_registry.register("on_job_completed", self._job_event_eval)
        self.trigger_registry.register("on_job_failed", self._job_event_eval)
        self.trigger_registry.register("on_job_timeout", self._job_event_eval)
        self.trigger_registry.register("on_network_available", self._network_eval)
        self.trigger_registry.register("on_file_changed", self._file_eval)
        self.trigger_registry.register("on_system_idle", self._idle_eval)
        self.trigger_registry.register("on_system_shutdown", self._shutdown_eval)
        self.trigger_registry.register("on_system_boot", OnTimeEvaluator())  # handled by OS scheduler
        self.trigger_registry.register("on_webhook_received", self._webhook_eval)

        # Dream triggers
        self._dream_eval = OnDreamEventEvaluator()
        self._agent_idle_eval = OnAgentIdleEvaluator()

        self.trigger_registry.register("on_dream_start", self._dream_eval)
        self.trigger_registry.register("on_dream_end", self._dream_eval)
        self.trigger_registry.register("on_dream_budget_hit", self._dream_eval)
        self.trigger_registry.register("on_dream_skill_evolved", self._dream_eval)
        self.trigger_registry.register("on_agent_idle", self._agent_idle_eval)

        # Wire dream events to bus
        self.event_bus.on("dream_start", self._on_dream_event)
        self.event_bus.on("dream_end", self._on_dream_event)
        self.event_bus.on("dream_budget_hit", self._on_dream_event)
        self.event_bus.on("dream_skill_evolved", self._on_dream_event)

        # Wire up job event bus for chaining
        self.event_bus.on("job_completed", self._on_job_event)
        self.event_bus.on("job_failed", self._on_job_event)
        self.event_bus.on("job_timeout", self._on_job_event)

        # Load persisted jobs
        self._load_jobs()

    # --- Persistence ---

    def _load_jobs(self):
        """Load jobs from JSON file."""
        if not self.jobs_file.exists():
            return
        try:
            data = json.loads(self.jobs_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for jd in data:
                    try:
                        job = JobDefinition.from_dict(jd)
                        self._jobs[job.job_id] = job
                    except Exception as e:
                        _log.warning("Failed to load job: %s", e)
            _log.debug("Loaded %d jobs from %s", len(self._jobs), self.jobs_file)
        except Exception as e:
            _log.warning("Failed to load jobs file: %s", e)

    def _save_jobs(self):
        """Persist all jobs to JSON."""
        try:
            self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
            data = [job.to_dict() for job in self._jobs.values()]
            self.jobs_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            _log.warning("Failed to save jobs: %s", e)

    # --- CRUD ---

    def add_job(self, job: JobDefinition) -> str:
        """Add a job. Returns job_id."""
        if not job.job_id:
            job.job_id = JobDefinition.generate_id()
        if not job.created_at:
            job.created_at = datetime.now(timezone.utc).isoformat()
        self._jobs[job.job_id] = job
        self._save_jobs()
        # Setup trigger evaluator if scheduler is running
        if self._running:
            asyncio.ensure_future(self._setup_trigger(job))
        return job.job_id

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID. Returns True if found."""
        job = self._jobs.pop(job_id, None)
        if job:
            if self._running:
                asyncio.ensure_future(self._teardown_trigger(job))
            self._save_jobs()
            return True
        return False

    def pause_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == "active":
            job.status = "paused"
            self._save_jobs()
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == "paused":
            job.status = "active"
            self._save_jobs()
            return True
        return False

    def get_job(self, job_id: str) -> JobDefinition | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobDefinition]:
        return list(self._jobs.values())

    # --- Lifecycle ---

    async def start(self):
        """Start the scheduler tick loop."""
        if self._running:
            return
        self._running = True

        # Setup all trigger evaluators
        for job in self._jobs.values():
            await self._setup_trigger(job)

        # Start tick loop
        self._tick_task = asyncio.create_task(self._tick_loop())
        _log.debug("JobScheduler started with %d jobs", len(self._jobs))

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None

        # Teardown all evaluators
        for job in self._jobs.values():
            await self._teardown_trigger(job)

        self._save_jobs()
        _log.debug("JobScheduler stopped")

    async def _setup_trigger(self, job: JobDefinition):
        evaluator = self.trigger_registry.get(job.trigger.trigger_type)
        if evaluator:
            try:
                await evaluator.setup(job, self)
            except Exception as e:
                _log.warning("Trigger setup error for %s: %s", job.job_id, e)

    async def _teardown_trigger(self, job: JobDefinition):
        evaluator = self.trigger_registry.get(job.trigger.trigger_type)
        if evaluator:
            try:
                await evaluator.teardown(job)
            except Exception as e:
                _log.warning("Trigger teardown error for %s: %s", job.job_id, e)

    # --- Tick Loop ---

    async def _tick_loop(self):
        """Main scheduler loop, runs every 1 second."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.warning("Scheduler tick error: %s", e)
            await asyncio.sleep(1.0)

    async def _tick(self):
        """Evaluate all active jobs."""
        for job in list(self._jobs.values()):
            if job.status != "active":
                continue
            if job.job_id in self._firing:
                continue  # already executing

            should_fire = await self._evaluate_trigger(job)
            if should_fire:
                asyncio.ensure_future(self._fire_job(job))

    async def _evaluate_trigger(self, job: JobDefinition) -> bool:
        """Delegate trigger evaluation to the registry."""
        evaluator = self.trigger_registry.get(job.trigger.trigger_type)
        if not evaluator:
            _log.debug("No evaluator for trigger type: %s", job.trigger.trigger_type)
            return False
        try:
            return await evaluator.evaluate(job)
        except Exception as e:
            _log.warning("Trigger eval error for %s: %s", job.job_id, e)
            return False

    async def _fire_job(self, job: JobDefinition):
        """Fire a job: call the fire_callback, update stats, emit events."""
        import time
        self._firing.add(job.job_id)
        job._last_fired_ts = time.time()
        job.last_run_at = datetime.now(timezone.utc).isoformat()
        job.run_count += 1

        _log.debug("Firing job %s (%s)", job.job_id, job.name)

        try:
            result = await asyncio.wait_for(
                self._fire_callback(job),
                timeout=job.timeout_seconds
            )
            job.last_result = "completed"
            self.event_bus.emit("job_completed", {"job_id": job.job_id, "result": result})
        except asyncio.TimeoutError:
            job.last_result = "timeout"
            job.fail_count += 1
            self.event_bus.emit("job_timeout", {"job_id": job.job_id})
            _log.warning("Job %s timed out after %ds", job.job_id, job.timeout_seconds)
        except Exception as e:
            job.last_result = "failed"
            job.fail_count += 1
            self.event_bus.emit("job_failed", {"job_id": job.job_id, "error": str(e)})
            _log.warning("Job %s failed: %s", job.job_id, e)
        finally:
            self._firing.discard(job.job_id)
            self._save_jobs()

    # --- Lifecycle Hooks ---

    async def fire_lifecycle(self, event_name: str):
        """Fire all jobs matching a lifecycle event (on_cli_start / on_cli_exit)."""
        for job in self._jobs.values():
            if job.status != "active":
                continue
            if job.trigger.trigger_type == event_name:
                self._cli_lifecycle_eval.mark_pending(job.job_id)

    # --- Event Bus Wiring ---

    def _on_job_event(self, event: str, data: dict):
        """Handle job lifecycle events for chaining."""
        source_id = data.get("job_id", "")
        # Map event bus events to trigger types
        event_to_trigger = {
            "job_completed": "on_job_completed",
            "job_failed": "on_job_failed",
            "job_timeout": "on_job_timeout",
        }
        trigger_type = event_to_trigger.get(event)
        if trigger_type and source_id:
            self._job_event_eval.notify(trigger_type, source_id, list(self._jobs.values()))
    # --- Deramer Events ---

    def _on_dream_event(self, event: str, data: dict):
        """Handle dream lifecycle events."""
        event_to_trigger = {
            "dream_start": "on_dream_start",
            "dream_end": "on_dream_end",
            "dream_budget_hit": "on_dream_budget_hit",
            "dream_skill_evolved": "on_dream_skill_evolved",
        }
        trigger_type = event_to_trigger.get(event)
        if trigger_type:
            self._dream_eval.notify(trigger_type, list(self._jobs.values()))

    # --- Webhook Interface ---

    def trigger_webhook(self, job_id: str):
        """Externally trigger a webhook job."""
        self._webhook_eval.trigger_webhook(job_id)

    # --- Utility ---

    def find_jobs_by_name(self, name_fragment: str) -> list[JobDefinition]:
        """Find jobs by partial name match."""
        fragment = name_fragment.lower()
        return [j for j in self._jobs.values() if fragment in j.name.lower() or fragment in j.job_id.lower()]

    @property
    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "active")

    @property
    def total_count(self) -> int:
        return len(self._jobs)

    def add_dream_job(
        self,
        agent_name: str,
        trigger_type: str = "on_cron",
        cron_expression: str = "0 3 * * *",
        agent_idle_seconds: int | None = None,
        dream_config: dict | None = None,
        name: str = "auto-dream",
    ) -> str:
        """Convenience: create a dream job.

        Usage:
            # Nightly at 3am
            scheduler.add_dream_job("my_agent")

            # After 10min idle
            scheduler.add_dream_job("my_agent", trigger_type="on_agent_idle", agent_idle_seconds=600)

            # After every successful job
            scheduler.add_dream_job("my_agent", trigger_type="on_job_completed")
        """
        job = JobDefinition(
            job_id=JobDefinition.generate_id(),
            name=name,
            agent_name=agent_name,
            query="__dream__",  # magic query, intercepted by ExecutionEngine
            trigger=TriggerConfig(
                trigger_type=trigger_type,
                cron_expression=cron_expression if trigger_type == "on_cron" else None,
                agent_idle_seconds=agent_idle_seconds,
                extra={"dream_config": dream_config} if dream_config else None,
            ),
            timeout_seconds=600,
        )
        return self.add_job(job)
