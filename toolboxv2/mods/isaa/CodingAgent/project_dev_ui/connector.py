"""
connector.py - Production Bridge with Flow Control (Pause/Stop/Feedback)
FIXED: Robust stop during pause, context injection, file tracking after IO-COMMIT
"""

import asyncio
import threading
import queue
import time
import logging
import traceback
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Production Imports
from toolboxv2.mods.isaa.CodingAgent.manager import SequentialManager, ParallelManager, SwarmManager
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

@dataclass
class LogEvent:
    id: str
    timestamp: str
    section: str
    content: str
    agent_id: str = "Manager"

@dataclass
class TrackedFile:
    """File detected after IO-COMMIT"""
    path: str
    size: int
    modified: float
    is_new: bool = True

class AgentRunner:
    """
    Runs Agent Managers in a daemon thread with Pause/Resume/Stop capabilities.
    FIXED: Stop truly terminates, Resume injects context, Files tracked after IO-COMMIT.
    """
    def __init__(self, workspace_path: str, agent_instance: FlowAgent, mode: str = "parallel", config: Dict = None):
        self.workspace_path = workspace_path
        self.agent_instance = agent_instance
        self.mode = mode
        self.config = config or {}

        # Communication
        self.log_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Flow Control
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # Set = Paused
        self._pending_context: Optional[str] = None  # Context to inject on resume
        self._context_lock = threading.Lock()

        # File tracking
        self._file_snapshot: Dict[str, float] = {}  # path -> mtime before run
        self.tracked_files: List[TrackedFile] = []

        self._thread: Optional[threading.Thread] = None

    def _take_file_snapshot(self):
        """Snapshot workspace files before execution for diff detection"""
        self._file_snapshot.clear()
        ws = Path(self.workspace_path)
        if not ws.exists():
            return
        try:
            for p in ws.rglob('*'):
                if p.is_file() and not any(skip in str(p) for skip in ['node_modules', '.git', '__pycache__', '.venv']):
                    self._file_snapshot[str(p)] = p.stat().st_mtime
        except Exception:
            pass

    def _detect_changed_files(self):
        """Compare current workspace to snapshot, track new/changed files"""
        ws = Path(self.workspace_path)
        if not ws.exists():
            return

        try:
            for p in ws.rglob('*'):
                if not p.is_file():
                    continue
                if any(skip in str(p) for skip in ['node_modules', '.git', '__pycache__', '.venv']):
                    continue

                path_str = str(p)
                stat = p.stat()

                if path_str not in self._file_snapshot:
                    # New file
                    self.tracked_files.append(TrackedFile(
                        path=str(p.relative_to(ws)),
                        size=stat.st_size,
                        modified=stat.st_mtime,
                        is_new=True
                    ))
                elif stat.st_mtime > self._file_snapshot[path_str]:
                    # Modified file
                    self.tracked_files.append(TrackedFile(
                        path=str(p.relative_to(ws)),
                        size=stat.st_size,
                        modified=stat.st_mtime,
                        is_new=False
                    ))
        except Exception:
            pass

    def _log_handler(self, section: str, content: str):
        """
        Intercepts logs to control flow.
        FIXED: Checks for injected context after pause, detects files after IO-COMMIT.
        """
        # 1. Check for STOP
        if self._stop_event.is_set():
            raise InterruptedError("Execution stopped by user.")

        # 2. Check for PAUSE
        while self._pause_event.is_set():
            if self._stop_event.is_set():
                raise InterruptedError("Execution stopped by user during pause.")
            time.sleep(0.2)

        # 3. After resume: inject pending context into agent if available
        with self._context_lock:
            if self._pending_context:
                ctx = self._pending_context
                self._pending_context = None
                # Try to inject into the agent's message history
                self._inject_context_to_agent(ctx)

        # 4. Track files after IO-COMMIT
        if section == "IO-COMMIT":
            self._detect_changed_files()

        # 5. Process Log
        if not content:
            return
        ts = datetime.now().strftime("%H:%M:%S")

        self.log_queue.put(LogEvent(
            id=f"{time.time()}-{section}",
            timestamp=ts,
            section=section,
            content=content
        ))

    def _inject_context_to_agent(self, context: str):
        """Best-effort injection of user context into the running agent."""
        ts = datetime.now().strftime("%H:%M:%S")

        # Try multiple known agent interfaces
        agent = self.agent_instance
        injected = False

        # Method 1: FlowAgent with chat_history
        for attr in ("chat_history", "messages", "memory", "_messages", "history"):
            if hasattr(agent, attr):
                hist = getattr(agent, attr)
                if isinstance(hist, list):
                    hist.append({
                        "role": "user",
                        "content": f"[USER INSTRUCTION] {context}"
                    })
                    injected = True
                    break

        # Method 2: Agent with add_message / inject method
        if not injected:
            for method in ("add_message", "inject_message", "add_user_message"):
                if hasattr(agent, method) and callable(getattr(agent, method)):
                    try:
                        getattr(agent, method)("user", f"[USER INSTRUCTION] {context}")
                        injected = True
                    except Exception:
                        try:
                            getattr(agent, method)(f"[USER INSTRUCTION] {context}")
                            injected = True
                        except Exception:
                            pass
                    break

        status = "injected into agent" if injected else "logged (agent injection not supported)"
        self.log_queue.put(LogEvent(
            "ctx", ts, "USER_INJECT",
            f"Context {status}: {context}"
        ))

    def start(self, task: str):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self._pending_context = None
        self.tracked_files.clear()

        # Snapshot workspace before run
        self._take_file_snapshot()

        self._thread = threading.Thread(
            target=self._run_async_wrapper,
            args=(task,),
            name="AgentRunnerThread",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Signals the thread to stop. Works during pause too."""
        self._stop_event.set()
        # Unpause so the thread unblocks and hits the stop check
        self._pause_event.clear()

    def pause(self):
        """Pauses execution at the next log event."""
        self._pause_event.set()
        self.log_queue.put(LogEvent(
            "sys", datetime.now().strftime("%H:%M:%S"), "SYSTEM",
            "⏸️ PAUSED. Use Resume to continue or Stop to abort."
        ))

    def resume(self, new_context: str = None):
        """Resumes execution, optionally injecting context into the agent."""
        if new_context:
            with self._context_lock:
                self._pending_context = new_context

        self._pause_event.clear()
        self.log_queue.put(LogEvent(
            "sys", datetime.now().strftime("%H:%M:%S"), "SYSTEM",
            "▶️ RESUMED." + (f" (with context)" if new_context else "")
        ))

    def _run_async_wrapper(self, task: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            self._log_handler("SYSTEM", f"Initializing {self.mode.upper()} Engine...")

            run_config = self.config.copy()
            run_config["log_handler"] = self._log_handler

            result = None

            if self.mode == "direct":
                from toolboxv2.mods.isaa.CodingAgent.coder import CoderAgent
                self._log_handler("SYSTEM", "Using DIRECT mode (single CoderAgent, no manager)")

                coder = CoderAgent(
                    agent=self.agent_instance,
                    project_root=self.workspace_path,
                    config=run_config
                )
                result = loop.run_until_complete(coder.execute(task))

            else:
                manager = None
                if self.mode == "sequential":
                    manager = SequentialManager(self.agent_instance, self.workspace_path, run_config)
                elif self.mode == "parallel":
                    manager = ParallelManager(self.agent_instance, self.workspace_path, run_config)
                elif self.mode == "swarm":
                    manager = SwarmManager(self.agent_instance, self.workspace_path, run_config)

                if manager is None:
                    raise ValueError(f"Unknown mode: {self.mode}")

                result = loop.run_until_complete(manager.run(task))

            # Final file detection
            self._detect_changed_files()

            if result is not None:
                result_data = {"data": result, "tracked_files": self.tracked_files}
                if result.success:
                    self.result_queue.put({"status": "success", **result_data})
                else:
                    self.result_queue.put({"status": "failure", **result_data})
            else:
                self.result_queue.put({"status": "error", "message": "No result returned", "tracked_files": self.tracked_files})

        except InterruptedError:
            # Final file detection even on stop
            self._detect_changed_files()
            self.log_queue.put(LogEvent(
                "err", datetime.now().strftime("%H:%M:%S"), "SYSTEM",
                "🛑 PROCESS STOPPED BY USER"
            ))
            self.result_queue.put({"status": "stopped", "message": "User Interrupted", "tracked_files": self.tracked_files})
        except Exception as e:
            tb = traceback.format_exc()
            self._log_handler("ERROR", f"CRITICAL EXCEPTION:\n{tb}")
            self.result_queue.put({"status": "error", "message": str(e), "tracked_files": self.tracked_files})
        finally:
            loop.close()

    def get_logs(self) -> List[LogEvent]:
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def check_result(self) -> Optional[Dict]:
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def is_paused(self) -> bool:
        return self._pause_event.is_set()
