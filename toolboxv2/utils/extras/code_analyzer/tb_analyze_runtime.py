"""
tb_analyze_runtime — Runtime Analysis for ToolBoxV2

Wraps a target script with continuous monitoring.
All data is persisted immediately to JSONL files (crash-safe).
Report can be generated during or after the run.

Usage:
    # CLI: wrap any python command
    python tb_analyze_runtime.py run "python my_agent.py --flag" --interval 2 --outdir ./runtime_data

    # Generate report from collected data
    python tb_analyze_runtime.py report ./runtime_data --output report.html

    # API
    from tb_analyze_runtime import RuntimeMonitor, generate_runtime_report
    monitor = RuntimeMonitor(outdir="./runtime_data", interval=2)
    monitor.start(pid=12345)  # attach to running process
    ...
    monitor.stop()
    generate_runtime_report("./runtime_data", "report.html")
"""

from __future__ import annotations

import gc as gc_module
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from toolboxv2 import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Lazy Dependency Install (reuse from tb_analyze)
# ---------------------------------------------------------------------------

_RUNTIME_DEPS = {"psutil": "psutil", "objgraph": "objgraph", "memray": "memray"}


def _ensure_dep(name: str) -> None:
    """Install dependency lazily: uv → pip → error."""
    try:
        __import__(name)
        return
    except ImportError:
        pass

    pkg = _RUNTIME_DEPS.get(name, name)
    for cmd in [
        ["uv", "pip", "install", pkg],
        ["uvx", "pip", "install", pkg],
        [sys.executable, "-m", "pip", "install", pkg],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,  encoding="utf-8", errors="replace",timeout=120)
            if r.returncode == 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    raise RuntimeError(f"Cannot install {pkg}. Install manually: pip install {pkg}")


# ---------------------------------------------------------------------------
# Crash-safe JSONL Writer
# ---------------------------------------------------------------------------

class JsonlWriter:
    """Append-only JSONL writer with fsync on every write."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def write(self, data: dict) -> None:
        line = json.dumps(data, ensure_ascii=False, default=str) + "\n"
        os.write(self._fd, line.encode("utf-8"))
        os.fsync(self._fd)

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass

    @staticmethod
    def read(path: Path) -> list[dict]:
        """Read all records from a JSONL file."""
        records = []
        if not path.exists():
            return records
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
        return records


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def _collect_process(pid: int) -> dict:
    """Collect process metrics: memory, CPU, threads, FDs."""
    _ensure_dep("psutil")
    import psutil

    try:
        proc = psutil.Process(pid)
        mem = proc.memory_info()
        result = {
            "pid": pid,
            "rss_mb": round(mem.rss / (1024 * 1024), 2),
            "vms_mb": round(mem.vms / (1024 * 1024), 2),
            "cpu_percent": proc.cpu_percent(interval=0),
            "threads": proc.num_threads(),
            "status": proc.status(),
        }

        # FDs
        try:
            result["open_fds"] = proc.num_fds()
        except (AttributeError, psutil.AccessDenied):
            try:
                result["open_fds"] = proc.num_handles()
            except (AttributeError, psutil.AccessDenied):
                result["open_fds"] = -1

        return result
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        return {"pid": pid, "error": str(e)}


def _collect_children(pid: int) -> list[dict]:
    """Collect child process info."""
    _ensure_dep("psutil")
    import psutil

    try:
        proc = psutil.Process(pid)
        children = proc.children(recursive=True)
        return [
            {
                "pid": ch.pid,
                "name": ch.name(),
                "rss_mb": round(ch.memory_info().rss / (1024 * 1024), 1),
                "cpu_percent": ch.cpu_percent(interval=0),
            }
            for ch in children[:30]
        ]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _collect_network(pid: int) -> list[dict]:
    """Collect network connections for process and children."""
    _ensure_dep("psutil")
    import psutil

    try:
        proc = psutil.Process(pid)
        pids = {pid} | {ch.pid for ch in proc.children(recursive=True)}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []

    conns = []
    for p in pids:
        try:
            for c in psutil.Process(p).net_connections(kind="all"):
                conns.append({
                    "pid": p,
                    "fd": c.fd,
                    "family": str(c.family.name) if hasattr(c.family, "name") else str(c.family),
                    "type": str(c.type.name) if hasattr(c.type, "name") else str(c.type),
                    "laddr": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "",
                    "raddr": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "",
                    "status": c.status,
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return conns[:50]


def _collect_tracemalloc_top(n: int = 20) -> list[dict]:
    """Top N memory allocators from tracemalloc with full file:line detail."""
    if not tracemalloc.is_tracing():
        return []

    snap = tracemalloc.take_snapshot()
    # Filter out importlib and tracemalloc internals
    snap = snap.filter_traces([
        tracemalloc.Filter(False, "<frozen importlib.*>"),
        tracemalloc.Filter(False, "<frozen *>"),
        tracemalloc.Filter(False, tracemalloc.__file__),
    ])

    top = snap.statistics("lineno")[:n]
    return [
        {
            "file": str(stat.traceback[0].filename) if stat.traceback else "?",
            "line": stat.traceback[0].lineno if stat.traceback else 0,
            "size_kb": round(stat.size / 1024, 2),
            "count": stat.count,
        }
        for stat in top
    ]


def _collect_tracemalloc_diff(snap_before) -> list[dict]:
    """Compute allocation diff since snap_before. Shows what GREW."""
    if not tracemalloc.is_tracing() or snap_before is None:
        return []

    snap_after = tracemalloc.take_snapshot()
    snap_after = snap_after.filter_traces([
        tracemalloc.Filter(False, "<frozen importlib.*>"),
        tracemalloc.Filter(False, "<frozen *>"),
    ])

    diff = snap_after.compare_to(snap_before, "lineno")
    # Only show things that grew
    return [
        {
            "file": str(stat.traceback[0].filename) if stat.traceback else "?",
            "line": stat.traceback[0].lineno if stat.traceback else 0,
            "size_diff_kb": round(stat.size_diff / 1024, 2),
            "size_kb": round(stat.size / 1024, 2),
            "count_diff": stat.count_diff,
            "count": stat.count,
        }
        for stat in diff[:30]
        if stat.size_diff > 0
    ]


def _collect_gc_stats() -> dict:
    """GC statistics with uncollectable object tracing."""
    gc_module.collect()
    stats = gc_module.get_stats()

    result = {
        "generations": [
            {
                "collections": g["collections"],
                "collected": g["collected"],
                "uncollectable": g["uncollectable"],
            }
            for g in stats
        ],
        "objects_tracked": len(gc_module.get_objects()),
        "thresholds": gc_module.get_threshold(),
        "garbage_count": len(gc_module.garbage),
        "garbage_details": [],
    }

    # Trace uncollectable objects — these are definite leaks
    if gc_module.garbage:
        seen_types: dict[str, dict] = {}
        for obj in gc_module.garbage[:50]:
            obj_type = type(obj).__name__
            obj_module = getattr(type(obj), "__module__", "")
            key = f"{obj_module}.{obj_type}" if obj_module else obj_type

            if key not in seen_types:
                seen_types[key] = {
                    "type": key,
                    "count": 0,
                    "locations": [],
                }

            seen_types[key]["count"] += 1

            # Try to find where it's defined
            if len(seen_types[key]["locations"]) < 3:
                loc = _trace_object_location(obj)
                if loc and loc not in seen_types[key]["locations"]:
                    seen_types[key]["locations"].append(loc)

        result["garbage_details"] = list(seen_types.values())

    return result


def _trace_object_location(obj) -> str:
    """Try to find the source location of an object via its referrers."""
    try:
        referrers = gc_module.get_referrers(obj)
        for ref in referrers[:5]:
            # Skip frame objects (our own stack)
            if type(ref).__name__ == "frame":
                continue

            # Check if it's a module-level dict
            if isinstance(ref, dict):
                for mod_name, mod in sys.modules.items():
                    if mod is not None and getattr(mod, "__dict__", None) is ref:
                        # Find the variable name
                        for var_name, var_val in ref.items():
                            if var_val is obj:
                                mod_file = getattr(mod, "__file__", mod_name)
                                return f"{mod_file}:{var_name}"
                        return f"{getattr(mod, '__file__', mod_name)}"

            # Check if held by a list/set in a known module
            if isinstance(ref, (list, set)):
                parent_refs = gc_module.get_referrers(ref)
                for parent in parent_refs[:3]:
                    if type(parent).__name__ == "frame":
                        continue
                    if isinstance(parent, dict):
                        for mod_name, mod in sys.modules.items():
                            if mod is not None and getattr(mod, "__dict__", None) is parent:
                                for var_name, var_val in parent.items():
                                    if var_val is ref:
                                        mod_file = getattr(mod, "__file__", mod_name)
                                        return f"{mod_file}:{var_name}[{type(ref).__name__}(len={len(ref)})]"

            # Check if it's a class attribute
            if isinstance(ref, type):
                return f"class:{ref.__qualname__}"

    except Exception:
        pass
    return ""


def _find_variable_name(obj, referrers) -> str:
    """Find the variable name holding an object by tracing up the referrer chain.
    Handles nested containers: obj → list → module.variable"""
    for ref in referrers:
        if type(ref).__name__ == "frame":
            continue

        if isinstance(ref, dict):
            loc = _check_module_dict(ref, obj)
            if loc:
                return loc
            for other_ref in gc_module.get_referrers(ref):
                if isinstance(other_ref, type):
                    for attr_name in vars(other_ref):
                        try:
                            if getattr(other_ref, attr_name) is obj:
                                return f"class:{other_ref.__qualname__}.{attr_name}"
                        except Exception:
                            pass
                    break

        elif isinstance(ref, (list, set, tuple)):
            # obj is inside a container — trace the container's owner
            container_refs = gc_module.get_referrers(ref)
            for cref in container_refs[:5]:
                if type(cref).__name__ == "frame":
                    continue
                if isinstance(cref, dict):
                    loc = _check_module_dict_for_container(cref, ref)
                    if loc:
                        return f"{loc}[{type(ref).__name__}(len={len(ref)})]"
                elif isinstance(cref, type):
                    for attr_name, attr_val in vars(cref).items():
                        if attr_val is ref:
                            return f"class:{cref.__qualname__}.{attr_name}"

        elif isinstance(ref, type):
            for attr_name in vars(ref):
                try:
                    if getattr(ref, attr_name) is obj:
                        return f"class:{ref.__qualname__}.{attr_name}"
                except Exception:
                    pass

    return ""


def _check_module_dict(d: dict, obj) -> str:
    for mod_name, mod in sys.modules.items():
        if mod is not None and getattr(mod, "__dict__", None) is d:
            for var_name, var_val in d.items():
                if var_val is obj:
                    return f"{getattr(mod, '__file__', mod_name)}:{var_name}"
    return ""


def _check_module_dict_for_container(d: dict, container) -> str:
    for mod_name, mod in sys.modules.items():
        if mod is not None and getattr(mod, "__dict__", None) is d:
            for var_name, var_val in d.items():
                if var_val is container:
                    return f"{getattr(mod, '__file__', mod_name)}:{var_name}"
    return ""


def _collect_objgraph_growth() -> list[dict]:
    """Object type growth with deep referrer tracking — finds actual variable names and locations."""
    _ensure_dep("objgraph")
    import io
    import objgraph

    buf = io.StringIO()
    objgraph.show_growth(limit=15, file=buf)
    lines = buf.getvalue().strip().splitlines()
    result = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        type_name = parts[0]
        try:
            count = int(parts[1])
        except ValueError:
            count = 0
        growth = parts[2] if len(parts) > 2 else ""

        entry = {"type": type_name, "count": count, "growth": growth, "referrers": [], "locations": []}
        result.append(entry)

    # Separate pass: find the largest user-owned containers (lists, dicts, sets)
    # This is more reliable than sampling random objects by type
    _find_large_containers(result)

    return result


def _find_large_containers(growth_entries: list[dict]) -> None:
    """Scan user-module globals for large containers and attach to growth entries."""
    for mod_name, mod in sys.modules.items():
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            continue
        if "/usr/lib/" in mod_file or "site-packages" in mod_file:
            continue
        if mod_file.startswith("<"):
            continue

        fname = Path(mod_file).name

        try:
            mod_dict = vars(mod)
        except Exception:
            continue

        for var_name, var_val in mod_dict.items():
            if var_name.startswith("_"):
                continue

            if isinstance(var_val, list) and len(var_val) > 5:
                label = f"{fname}:{var_name}(list,len={len(var_val)})"
                loc = f"{mod_file}:{var_name}"
                _attach_to_growth(growth_entries, "list", label, loc)

                # Also check what the list contains
                if var_val:
                    item_type = type(var_val[0]).__name__
                    _attach_to_growth(growth_entries, item_type,
                                     f"in {fname}:{var_name}[{len(var_val)}×{item_type}]",
                                     f"{mod_file}:{var_name}→{item_type}")

            elif isinstance(var_val, dict) and len(var_val) > 5:
                label = f"{fname}:{var_name}(dict,len={len(var_val)})"
                loc = f"{mod_file}:{var_name}"
                _attach_to_growth(growth_entries, "dict", label, loc)

            elif isinstance(var_val, set) and len(var_val) > 5:
                label = f"{fname}:{var_name}(set,len={len(var_val)})"
                loc = f"{mod_file}:{var_name}"
                _attach_to_growth(growth_entries, "set", label, loc)


def _attach_to_growth(entries: list[dict], type_name: str, referrer: str, location: str) -> None:
    """Attach a referrer/location to the matching growth entry."""
    for entry in entries:
        if entry["type"] == type_name:
            if referrer not in entry["referrers"]:
                entry["referrers"].insert(0, referrer)
            loc_short = location.split("(")[0] if "(" in location else location
            if loc_short not in entry["locations"]:
                entry["locations"].insert(0, loc_short)
            return


def _describe_referrer(ref) -> str:
    """Create a human-readable description of what holds a reference."""
    if isinstance(ref, dict):
        # Check if it's a module dict
        for mod_name, mod in sys.modules.items():
            if mod is not None and getattr(mod, "__dict__", None) is ref:
                mod_file = getattr(mod, "__file__", "")
                if mod_file:
                    return f"{Path(mod_file).name}:{mod_name}"
                return f"module:{mod_name}"
        return f"dict(len={len(ref)})"
    elif isinstance(ref, list):
        return _trace_list_owner(ref)
    elif isinstance(ref, type):
        return f"class:{ref.__qualname__}"
    elif isinstance(ref, set):
        return f"set(len={len(ref)})"
    else:
        ref_type = type(ref).__name__
        ref_module = getattr(type(ref), "__module__", "")
        if ref_module and ref_module != "builtins":
            return f"{ref_module}.{ref_type}"
        return ref_type


def _trace_list_owner(lst) -> str:
    """Trace a list back to its owning variable in a module or class."""
    try:
        parents = gc_module.get_referrers(lst)
        for parent in parents[:5]:
            if type(parent).__name__ == "frame":
                continue
            if isinstance(parent, dict):
                for mod_name, mod in sys.modules.items():
                    if mod is not None and getattr(mod, "__dict__", None) is parent:
                        for var_name, var_val in parent.items():
                            if var_val is lst:
                                mod_file = getattr(mod, "__file__", mod_name)
                                return f"{Path(mod_file).name}:{var_name}(list,len={len(lst)})"
                # Class __dict__?
                for grandparent in gc_module.get_referrers(parent):
                    if isinstance(grandparent, type):
                        for attr_name, attr_val in vars(grandparent).items():
                            if attr_val is lst:
                                return f"class:{grandparent.__qualname__}.{attr_name}(list,len={len(lst)})"
            elif isinstance(parent, type):
                for attr_name, attr_val in vars(parent).items():
                    if attr_val is lst:
                        return f"class:{parent.__qualname__}.{attr_name}(list,len={len(lst)})"
    except Exception:
        pass
    return f"list(len={len(lst)})"


def _collect_loaded_modules() -> list[dict]:
    """Snapshot of all loaded Python modules with file paths.
    Shows which code files are actually used at runtime."""
    modules = []
    for name, mod in sorted(sys.modules.items()):
        fpath = getattr(mod, "__file__", None)
        if fpath is None:
            continue
        if fpath.startswith("<") or "site-packages" in fpath:
            continue
        # Skip frozen/builtin
        if "frozen" in fpath or "__pycache__" in fpath:
            continue
        try:
            size = os.path.getsize(fpath)
        except (OSError, TypeError):
            size = 0
        modules.append({
            "module": name,
            "file": fpath,
            "size_bytes": size,
        })
    return modules


def _collect_allocation_hotspots(n: int = 30) -> list[dict]:
    """Detailed allocation hotspots: file, line, function, size, count, frequency.
    Aggregated across tracemalloc traceback frames for deeper call context."""
    if not tracemalloc.is_tracing():
        return []

    snap = tracemalloc.take_snapshot()
    snap = snap.filter_traces([
        tracemalloc.Filter(False, "<frozen importlib.*>"),
        tracemalloc.Filter(False, "<frozen *>"),
        tracemalloc.Filter(False, tracemalloc.__file__),
    ])

    # Use traceback key for deeper info (file:line with call chain)
    stats = snap.statistics("traceback")[:n]
    result = []
    for stat in stats:
        frames = []
        for frame in stat.traceback[:5]:  # top 5 frames of call stack
            frames.append({
                "file": frame.filename,
                "line": frame.lineno,
            })
        result.append({
            "size_kb": round(stat.size / 1024, 2),
            "count": stat.count,
            "avg_bytes": round(stat.size / max(stat.count, 1)),
            "frames": frames,
            "top_file": frames[0]["file"] if frames else "?",
            "top_line": frames[0]["line"] if frames else 0,
        })
    return result


# ---------------------------------------------------------------------------
# Runtime Monitor — continuous, crash-safe
# ---------------------------------------------------------------------------

class RuntimeMonitor:
    """Continuous runtime monitor that persists all data immediately.

    Data files (JSONL, fsync'd every write):
        outdir/
            process.jsonl     — RSS, VMS, CPU, threads, FDs per interval
            memory.jsonl      — tracemalloc top allocators per interval
            memory_diff.jsonl — allocation diffs (what grew since last snapshot)
            network.jsonl     — network connections per interval
            children.jsonl    — child processes per interval
            gc.jsonl          — GC stats per interval
            objects.jsonl     — object type growth (objgraph)
            meta.json         — run metadata (command, start time, config)
    """

    def __init__(
        self,
        outdir: str | Path = "./runtime_data",
        interval: float = 2.0,
        tracemalloc_depth: int = 25,
        collect_objects: bool = True,
        collect_network: bool = True,
        collect_gc: bool = True,
    ):
        self.outdir = Path(outdir)
        self.interval = interval
        self.tracemalloc_depth = tracemalloc_depth
        self.collect_objects = collect_objects
        self.collect_network = collect_network
        self.collect_gc = collect_gc

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pid: int = 0
        self._t0: float = 0
        self._snap_before = None

        # Writers (opened on start)
        self._writers: dict[str, JsonlWriter] = {}

    def start(self, pid: int | None = None) -> None:
        """Start monitoring. pid=None monitors self."""
        self._pid = pid or os.getpid()
        self._t0 = time.monotonic()
        self._stop.clear()

        # Create output directory
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Open all writers
        for name in ["process", "memory", "memory_diff", "network", "children", "gc", "objects", "modules"]:
            self._writers[name] = JsonlWriter(self.outdir / f"{name}.jsonl")

        # Start tracemalloc in target process (only works if monitoring self)
        if pid is None or pid == os.getpid():
            tracemalloc.start(self.tracemalloc_depth)
            self._snap_before = tracemalloc.take_snapshot()

        # Write metadata
        meta = {
            "pid": self._pid,
            "start_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "interval": self.interval,
            "tracemalloc_depth": self.tracemalloc_depth,
            "config": {
                "collect_objects": self.collect_objects,
                "collect_network": self.collect_network,
                "collect_gc": self.collect_gc,
            },
        }
        (self.outdir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Start sampling thread
        self._thread = threading.Thread(target=self._loop, daemon=True, name="tb_runtime_monitor")
        self._thread.start()
        logger.info("Runtime monitor started: pid=%d interval=%.1fs outdir=%s", self._pid, self.interval, self.outdir)

    def stop(self) -> None:
        """Stop monitoring and close all writers."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        for w in self._writers.values():
            w.close()
        self._writers.clear()

        if tracemalloc.is_tracing():
            tracemalloc.stop()

        logger.info("Runtime monitor stopped. Data in %s", self.outdir)

    def _loop(self) -> None:
        """Main sampling loop."""
        cycle = 0
        while not self._stop.is_set():
            t = round(time.monotonic() - self._t0, 3)
            cycle += 1

            try:
                # Process metrics (every cycle)
                proc = _collect_process(self._pid)
                proc["t"] = t
                proc["cycle"] = cycle
                self._writers["process"].write(proc)

                # Memory top allocators + hotspots (every cycle)
                mem_top = _collect_tracemalloc_top(20)
                hotspots = _collect_allocation_hotspots(15)
                if mem_top or hotspots:
                    self._writers["memory"].write({
                        "t": t, "cycle": cycle,
                        "top": mem_top,
                        "hotspots": hotspots,
                    })

                # Memory diff (every 3rd cycle)
                if cycle % 3 == 0 and self._snap_before is not None:
                    diff = _collect_tracemalloc_diff(self._snap_before)
                    if diff:
                        self._writers["memory_diff"].write({"t": t, "cycle": cycle, "diff": diff})
                    # Update baseline snapshot
                    if tracemalloc.is_tracing():
                        self._snap_before = tracemalloc.take_snapshot()

                # Children (every 3rd cycle)
                if cycle % 3 == 0:
                    children = _collect_children(self._pid)
                    if children:
                        self._writers["children"].write({"t": t, "cycle": cycle, "children": children})

                # Network (every 5th cycle)
                if self.collect_network and cycle % 5 == 0:
                    conns = _collect_network(self._pid)
                    if conns:
                        self._writers["network"].write({"t": t, "cycle": cycle, "connections": conns})

                # GC stats (every 3rd cycle)
                if self.collect_gc and cycle % 3 == 0:
                    gc_stats = _collect_gc_stats()
                    gc_stats["t"] = t
                    gc_stats["cycle"] = cycle
                    self._writers["gc"].write(gc_stats)

                # Object growth (every 3rd cycle, offset from GC)
                if self.collect_objects and cycle % 3 == 1:
                    growth = _collect_objgraph_growth()
                    if growth:
                        self._writers["objects"].write({"t": t, "cycle": cycle, "growth": growth})

                # Loaded modules snapshot (first cycle + every 5th)
                if cycle == 1 or cycle % 5 == 0:
                    mods = _collect_loaded_modules()
                    if mods:
                        self._writers["modules"].write({"t": t, "cycle": cycle, "modules": mods})

            except Exception as e:
                logger.warning("Monitor cycle %d error: %s", cycle, e)

            self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# CLI Runner — wraps target command with monitoring
# ---------------------------------------------------------------------------

def run_with_monitoring(
    command: str,
    outdir: str = "./runtime_data",
    interval: float = 2.0,
    memray: bool = False,
    **monitor_kwargs,
) -> int:
    """Run a command with full runtime monitoring.

    Args:
        command: Shell command to run (e.g. "python my_agent.py")
        outdir: Output directory for data files
        interval: Sampling interval in seconds
        memray: Also run memray for deep allocation profiling
        **monitor_kwargs: Passed to RuntimeMonitor

    Returns:
        Exit code of the target process
    """
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Save command metadata
    meta_path = outdir_path / "meta.json"
    meta = {
        "command": command,
        "start_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": interval,
        "memray_enabled": memray,
    }

    # Optionally wrap with memray
    if memray:
        _ensure_dep("memray")
        memray_bin = str(outdir_path / "memray_capture.bin")
        # Inject memray into the command
        parts = command.split(None, 1)
        if parts[0] == "python" or parts[0] == "python3" or parts[0] == sys.executable:
            rest = parts[1] if len(parts) > 1 else ""
            command = f"{sys.executable} -m memray run -o {memray_bin} --force {rest}"
            meta["memray_bin"] = memray_bin

    meta_path.write_text(json.dumps(meta, indent=2))

    # Start target process
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Start monitor attached to the target PID
    monitor = RuntimeMonitor(
        outdir=outdir_path,
        interval=interval,
        **monitor_kwargs,
    )
    monitor.start(pid=proc.pid)

    # Wait for target to finish or be interrupted
    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            exit_code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = -9
    finally:
        monitor.stop()

        # Update meta with end info
        meta["end_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["exit_code"] = exit_code
        meta["duration_s"] = round(time.monotonic() - monitor._t0, 3)
        meta_path.write_text(json.dumps(meta, indent=2))

        # Generate memray stats if available
        if memray and "memray_bin" in meta:
            _generate_memray_artifacts(meta["memray_bin"], outdir_path)

    print(f"\nRuntime data saved to {outdir_path}/")
    print(f"Generate report: python tb_analyze_runtime.py report {outdir_path}")

    return exit_code


def _generate_memray_artifacts(bin_path: str, outdir: Path) -> None:
    """Generate memray stats and flamegraph from capture."""
    # Stats text
    try:
        r = subprocess.run(
            [sys.executable, "-m", "memray", "stats", bin_path],
            capture_output=True, text=True,  encoding="utf-8", errors="replace", timeout=60,
        )
        if r.stdout:
            (outdir / "memray_stats.txt").write_text(r.stdout)
    except Exception as e:
        logger.warning("memray stats failed: %s", e)

    # Flamegraph HTML
    try:
        flame_path = str(outdir / "memray_flamegraph.html")
        subprocess.run(
            [sys.executable, "-m", "memray", "flamegraph", bin_path, "-o", flame_path, "--force"],
            capture_output=True, timeout=60, encoding="utf-8", errors="replace",
        )
    except Exception as e:
        logger.warning("memray flamegraph failed: %s", e)


# ---------------------------------------------------------------------------
# Report Generator — reads JSONL data, produces TBJS Glass HTML with charts
# ---------------------------------------------------------------------------

def _load_data(outdir: Path) -> dict:
    """Load all JSONL data files from an analysis run."""
    data = {}
    for name in ["process", "memory", "memory_diff", "network", "children", "gc", "objects", "modules"]:
        path = outdir / f"{name}.jsonl"
        data[name] = JsonlWriter.read(path)

    meta_path = outdir / "meta.json"
    if meta_path.exists():
        data["meta"] = json.loads(meta_path.read_text())
    else:
        data["meta"] = {}

    # Memray stats
    stats_path = outdir / "memray_stats.txt"
    if stats_path.exists():
        data["memray_stats"] = stats_path.read_text()
    else:
        data["memray_stats"] = ""

    return data


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _led_bar(value: float, max_val: float, segments: int = 10, invert: bool = False) -> str:
    """LED score bar. invert=True means lower is better."""
    if max_val <= 0:
        filled = 0
    else:
        ratio = min(value / max_val, 1.0)
        if invert:
            ratio = 1.0 - ratio
        filled = int(ratio * segments)

    if filled >= 7:
        cls = "led-success"
    elif filled >= 4:
        cls = "led-warning"
    else:
        cls = "led-error"

    blocks = []
    for i in range(segments):
        active = f"led-active {cls}" if i < filled else ""
        blocks.append(f'<span class="led-block {active}"></span>')
    return f'<div class="led-bar">{"".join(blocks)}</div>'


def generate_runtime_report(outdir: str | Path, output: str | Path | None = None) -> str:
    """Generate a TBJS Glass HTML report from collected runtime data.

    Args:
        outdir: Directory with JSONL data files from RuntimeMonitor
        output: Optional path to write HTML file

    Returns:
        HTML string
    """
    outdir = Path(outdir)
    data = _load_data(outdir)
    meta = data["meta"]
    process_data = data["process"]
    memory_data = data["memory"]
    memory_diff_data = data["memory_diff"]
    gc_data = data["gc"]
    objects_data = data["objects"]
    children_data = data["children"]
    network_data = data["network"]

    # Compute summary
    if process_data:
        peak_rss = max(p.get("rss_mb", 0) for p in process_data)
        peak_vms = max(p.get("vms_mb", 0) for p in process_data)
        final_rss = process_data[-1].get("rss_mb", 0)
        start_rss = process_data[0].get("rss_mb", 0)
        rss_growth = final_rss - start_rss
        duration = process_data[-1].get("t", 0)
    else:
        peak_rss = peak_vms = final_rss = start_rss = rss_growth = duration = 0

    # Top leakers from last memory diff
    top_leakers = []
    if memory_diff_data:
        last_diff = memory_diff_data[-1].get("diff", [])
        top_leakers = sorted(last_diff, key=lambda x: x.get("size_diff_kb", 0), reverse=True)[:15]

    # Top allocators from last memory snapshot
    top_allocators = []
    if memory_data:
        top_allocators = memory_data[-1].get("top", [])[:15]

    # Allocation hotspots with call stacks
    hotspots = []
    if memory_data:
        hotspots = memory_data[-1].get("hotspots", [])[:15]

    # Aggregate allocation stats across all snapshots: file → total KB, total count, frequency
    file_alloc_stats: dict[str, dict] = {}
    for rec in memory_data:
        for a in rec.get("top", []):
            key = f"{a.get('file', '?')}:{a.get('line', 0)}"
            if key not in file_alloc_stats:
                file_alloc_stats[key] = {
                    "file": a.get("file", "?"),
                    "line": a.get("line", 0),
                    "max_kb": 0,
                    "appearances": 0,
                    "total_count": 0,
                }
            entry = file_alloc_stats[key]
            entry["max_kb"] = max(entry["max_kb"], a.get("size_kb", 0))
            entry["appearances"] += 1
            entry["total_count"] += a.get("count", 0)

    # Sort by max_kb descending
    alloc_aggregate = sorted(file_alloc_stats.values(), key=lambda x: x["max_kb"], reverse=True)[:20]
    total_snapshots = len(memory_data)

    # Loaded modules — which code was actually used
    modules_data = data["modules"]
    loaded_modules = []
    if modules_data:
        loaded_modules = modules_data[-1].get("modules", [])
    total_code_kb = sum(m.get("size_bytes", 0) for m in loaded_modules) / 1024

    # Unique network endpoints
    all_conns: dict[str, dict] = {}
    for rec in network_data:
        for c in rec.get("connections", []):
            key = f"{c.get('laddr', '')}→{c.get('raddr', '')}"
            if key not in all_conns and c.get("raddr"):
                all_conns[key] = c

    # Object growth trend
    obj_growth = []
    if objects_data:
        obj_growth = objects_data[-1].get("growth", [])[:10]

    # Build chart data as JSON for inline JS
    chart_times = [p.get("t", 0) for p in process_data]
    chart_rss = [p.get("rss_mb", 0) for p in process_data]
    chart_vms = [p.get("vms_mb", 0) for p in process_data]
    chart_cpu = [p.get("cpu_percent", 0) for p in process_data]
    chart_threads = [p.get("threads", 0) for p in process_data]
    chart_fds = [p.get("open_fds", 0) for p in process_data]

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>tb_analyze — Runtime Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{_runtime_css()}
</style>
</head>
<body>
<div class="report-container">

  <header class="report-header">
    <div>
      <h6 class="label">tb_analyze runtime</h6>
      <h1>Runtime Analysis Report</h1>
      <p class="meta">
        <span class="mono">{_esc(meta.get("command", "unknown"))}</span>
        &middot; {duration:.1f}s
        &middot; PID {meta.get("pid", "?")}
        &middot; {len(process_data)} samples
      </p>
    </div>
  </header>

  <!-- Summary Cards -->
  <section class="cards-grid">
    <div class="card metric-card">
      <h6 class="label has-tooltip" data-tooltip="Resident Set Size — actual physical memory used by the process. Peak value during the run.">Peak RSS</h6>
      <div class="metric-value mono {_mem_class(peak_rss)}">{peak_rss:.0f} MB</div>
      {_led_bar(peak_rss, max(peak_rss, 500), invert=True)}
    </div>
    <div class="card metric-card">
      <h6 class="label has-tooltip" data-tooltip="RSS growth from start to end of run. Positive = memory leak candidate. 0 or negative = stable.">RSS Growth</h6>
      <div class="metric-value mono {_growth_class(rss_growth)}">{"+" if rss_growth > 0 else ""}{rss_growth:.0f} MB</div>
      {_led_bar(max(rss_growth, 0), max(peak_rss, 100), invert=True)}
    </div>
    <div class="card metric-card">
      <h6 class="label has-tooltip" data-tooltip="Virtual Memory Size — total address space reserved. Includes mapped files, shared libraries, and reserved but unused pages.">Peak VMS</h6>
      <div class="metric-value mono">{peak_vms:.0f} MB</div>
      {_led_bar(peak_vms, max(peak_vms, 1000), invert=True)}
    </div>
    <div class="card metric-card">
      <h6 class="label has-tooltip" data-tooltip="Unique allocation locations that appeared in every single tracemalloc snapshot — persistent allocators that never get freed.">Alloc Hotspots</h6>
      <div class="metric-value mono">{len(alloc_aggregate)}</div>
      {_led_bar(len(alloc_aggregate), max(len(alloc_aggregate), 30), invert=True)}
    </div>
    <div class="card metric-card">
      <h6 class="label has-tooltip" data-tooltip="Python source files loaded into the interpreter during runtime. Shows which code was actually used vs available.">Loaded Modules</h6>
      <div class="metric-value mono">{len(loaded_modules)}</div>
      <div class="metric-sub">{total_code_kb:.0f} KB source</div>
    </div>
    <div class="card metric-card">
      <h6 class="label has-tooltip" data-tooltip="Run duration in seconds.">Duration</h6>
      <div class="metric-value mono">{duration:.1f}s</div>
      <div class="metric-sub">{len(process_data)} samples</div>
    </div>
  </section>

  <!-- Memory Timeline Chart -->
  <section class="card chart-section">
    <h6 class="label has-tooltip" data-tooltip="RSS (Resident Set Size) = actual physical RAM used by the process. VMS (Virtual Memory Size) = total address space reserved, includes shared libs and memory-mapped files. RSS is what matters for OOM — if RSS approaches your total RAM, other processes will be killed.">Memory Timeline (RSS / VMS)</h6>
    <canvas id="memChart" height="250"></canvas>
  </section>

  <!-- CPU Chart -->
  <section class="card chart-section">
    <h6 class="label has-tooltip" data-tooltip="CPU usage as percentage of one core. 100% = one core fully used. Values above 100% mean multiple cores are active.">CPU Usage (%)</h6>
    <canvas id="cpuChart" height="180"></canvas>
  </section>

  <!-- Threads Chart -->
  <section class="card chart-section">
    <h6 class="label has-tooltip" data-tooltip="Number of active threads in the process over time. Sudden spikes indicate parallel work (e.g. agents spawning). The final value shows how many threads were alive at the end.">Active Threads (count: {chart_threads[-1] if chart_threads else 0})</h6>
    <canvas id="threadChart" height="150"></canvas>
  </section>

  <!-- Top Memory Allocators -->
  {_allocators_section(top_allocators)}

  <!-- Allocation Hotspots (with call stacks) -->
  {_hotspots_section(hotspots)}

  <!-- Aggregate Allocation Stats (across all snapshots) -->
  {_alloc_aggregate_section(alloc_aggregate, total_snapshots)}

  <!-- Memory Leak Candidates (Diff) -->
  {_leakers_section(top_leakers)}

  <!-- Loaded Modules (runtime code coverage) -->
  {_modules_section(loaded_modules)}

  <!-- Object Type Growth -->
  {_objects_section(obj_growth)}

  <!-- Network Connections -->
  {_network_section(all_conns)}

  <!-- GC Stats -->
  {_gc_section(gc_data)}

  <!-- Memray Stats -->
  {_memray_section(data.get("memray_stats", ""))}

  <footer class="report-footer">
    <span class="label">Generated by tb_analyze_runtime</span>
    <span class="mono">{meta.get("start_time", "")}</span>
  </footer>

</div>

<script>
{_runtime_chart_js(chart_times, chart_rss, chart_vms, chart_cpu, chart_threads)}
</script>
</body>
</html>"""

    if output is not None:
        Path(output).write_text(html, encoding="utf-8")
        logger.info("Runtime report written to %s", output)

    return html


# ---------------------------------------------------------------------------
# Report Helpers
# ---------------------------------------------------------------------------

def _mem_class(mb: float) -> str:
    if mb > 1000:
        return "text-error"
    if mb > 500:
        return "text-warning"
    return ""


def _growth_class(mb: float) -> str:
    if mb > 100:
        return "text-error"
    if mb > 20:
        return "text-warning"
    if mb <= 0:
        return "text-success"
    return ""


def _count_unique_children(children_data: list[dict]) -> int:
    pids: set[int] = set()
    for rec in children_data:
        for ch in rec.get("children", []):
            pids.add(ch.get("pid", 0))
    return len(pids)


def _allocators_section(items: list[dict]) -> str:
    if not items:
        return ""
    rows = ""
    for a in items:
        fname = Path(a.get("file", "?")).name
        full = _esc(a.get("file", "?"))
        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono" title="{full}">{_esc(fname)}</div>'
            f'<div class="grid-cell mono">{a.get("line", 0)}</div>'
            f'<div class="grid-cell mono text-warning">{a.get("size_kb", 0):.0f} KB</div>'
            f'<div class="grid-cell mono">{a.get("count", 0)}</div>'
            f'</div>'
        )
    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Top memory allocators from tracemalloc — shows which file and line allocated the most memory. Snapshot from the last sampling cycle.">Top Memory Allocators</h6>
    <div class="grid-table" style="grid-template-columns: 1fr 60px 100px 80px;">
      <div class="grid-header">
        <div class="grid-cell">File</div>
        <div class="grid-cell">Line</div>
        <div class="grid-cell has-tooltip" data-tooltip="Current total allocation size from this location.">Size</div>
        <div class="grid-cell has-tooltip" data-tooltip="Number of live allocation objects from this location.">Count</div>
      </div>
      {rows}
    </div>
  </section>"""


def _leakers_section(items: list[dict]) -> str:
    if not items:
        return ""
    rows = ""
    for a in items:
        fname = Path(a.get("file", "?")).name
        full = _esc(a.get("file", "?"))
        diff_kb = a.get("size_diff_kb", 0)
        cls = "text-error" if diff_kb > 100 else "text-warning" if diff_kb > 10 else ""
        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono" title="{full}">{_esc(fname)}</div>'
            f'<div class="grid-cell mono">{a.get("line", 0)}</div>'
            f'<div class="grid-cell mono {cls}">+{diff_kb:.0f} KB</div>'
            f'<div class="grid-cell mono">{a.get("size_kb", 0):.0f} KB</div>'
            f'<div class="grid-cell mono">+{a.get("count_diff", 0)}</div>'
            f'</div>'
        )
    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Memory allocation growth between sampling intervals. Lines with large positive diffs are leak candidates — they allocate memory that is never freed.">Leak Candidates (Allocation Growth)</h6>
    <div class="grid-table" style="grid-template-columns: 1fr 60px 100px 100px 80px;">
      <div class="grid-header">
        <div class="grid-cell">File</div>
        <div class="grid-cell">Line</div>
        <div class="grid-cell has-tooltip" data-tooltip="Size increase since last diff snapshot.">Growth</div>
        <div class="grid-cell has-tooltip" data-tooltip="Current total size from this location.">Total</div>
        <div class="grid-cell has-tooltip" data-tooltip="New allocation count since last snapshot.">+Count</div>
      </div>
      {rows}
    </div>
  </section>"""


def _hotspots_section(items: list[dict]) -> str:
    """Allocation hotspots with call stack frames."""
    if not items:
        return ""
    rows = ""
    for h in items:
        frames = h.get("frames", [])
        top = frames[0] if frames else {}
        fname = Path(top.get("file", "?")).name
        full_path = _esc(top.get("file", "?"))
        # Build callstack tooltip
        stack = " ← ".join(f'{Path(f["file"]).name}:{f["line"]}' for f in frames[:4])
        size_cls = "text-error" if h.get("size_kb", 0) > 1000 else "text-warning" if h.get("size_kb", 0) > 100 else ""
        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono" title="{full_path}">{_esc(fname)}</div>'
            f'<div class="grid-cell mono">{top.get("line", 0)}</div>'
            f'<div class="grid-cell mono {size_cls}">{h.get("size_kb", 0):.0f} KB</div>'
            f'<div class="grid-cell mono">{h.get("count", 0)}</div>'
            f'<div class="grid-cell mono">{h.get("avg_bytes", 0):,} B</div>'
            f'<div class="grid-cell mono has-tooltip" data-tooltip="{_esc(stack)}" style="cursor:help">🔍</div>'
            f'</div>'
        )
    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Allocation hotspots with full call stack context from tracemalloc. Shows exactly which function allocated how much memory and how often. Hover 🔍 for call chain.">Allocation Hotspots (with Call Stack)</h6>
    <div class="grid-table" style="grid-template-columns: 1fr 60px 100px 80px 100px 40px;">
      <div class="grid-header">
        <div class="grid-cell">File</div>
        <div class="grid-cell">Line</div>
        <div class="grid-cell has-tooltip" data-tooltip="Total memory currently held by allocations from this location.">Size</div>
        <div class="grid-cell has-tooltip" data-tooltip="Number of live allocation objects from this location.">Count</div>
        <div class="grid-cell has-tooltip" data-tooltip="Average size per individual allocation.">Avg/Alloc</div>
        <div class="grid-cell">Stack</div>
      </div>
      {rows}
    </div>
  </section>"""


def _alloc_aggregate_section(items: list[dict], total_snapshots: int) -> str:
    """Aggregated allocation stats across all snapshots — shows frequency and persistence."""
    if not items:
        return ""
    rows = ""
    for a in items:
        fname = Path(a.get("file", "?")).name
        full = _esc(a.get("file", "?"))
        freq = a.get("appearances", 0)
        freq_pct = round(freq / max(total_snapshots, 1) * 100)
        freq_cls = "text-error" if freq_pct >= 80 else "text-warning" if freq_pct >= 40 else ""
        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono" title="{full}">{_esc(fname)}</div>'
            f'<div class="grid-cell mono">{a.get("line", 0)}</div>'
            f'<div class="grid-cell mono">{a.get("max_kb", 0):.0f} KB</div>'
            f'<div class="grid-cell mono">{a.get("total_count", 0):,}</div>'
            f'<div class="grid-cell mono {freq_cls}">{freq}/{total_snapshots} ({freq_pct}%)</div>'
            f'</div>'
        )
    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Aggregated allocation data across ALL snapshots. 'Frequency' shows in how many snapshots this location appeared in the top allocators — 100% means it was always there (persistent allocation, potential leak). 'Max Size' is the peak allocation seen.">Allocation Frequency (Aggregated)</h6>
    <div class="grid-table" style="grid-template-columns: 1fr 60px 100px 100px 120px;">
      <div class="grid-header">
        <div class="grid-cell">File</div>
        <div class="grid-cell">Line</div>
        <div class="grid-cell has-tooltip" data-tooltip="Peak allocation size seen across all snapshots.">Max Size</div>
        <div class="grid-cell has-tooltip" data-tooltip="Sum of allocation counts across all snapshots.">Total Allocs</div>
        <div class="grid-cell has-tooltip" data-tooltip="How often this location appeared in the top-20 allocators. High % = persistent, likely a leak.">Frequency</div>
      </div>
      {rows}
    </div>
  </section>"""


def _modules_section(modules: list[dict]) -> str:
    """Loaded Python modules — runtime code coverage."""
    if not modules:
        return ""
    # Sort by size descending
    modules = sorted(modules, key=lambda m: m.get("size_bytes", 0), reverse=True)
    rows = ""
    for m in modules[:30]:
        fname = Path(m.get("file", "?")).name
        full = _esc(m.get("file", "?"))
        size_kb = m.get("size_bytes", 0) / 1024
        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono">{_esc(m.get("module", "?"))}</div>'
            f'<div class="grid-cell mono" title="{full}">{_esc(fname)}</div>'
            f'<div class="grid-cell mono">{size_kb:.1f} KB</div>'
            f'</div>'
        )
    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Python source files loaded into the interpreter during runtime. Shows which code was actually imported and used. Files not listed here were never loaded — dead code at the module level.">Loaded Modules (Runtime Code Coverage)</h6>
    <div class="grid-table" style="grid-template-columns: 1fr 1fr 80px;">
      <div class="grid-header">
        <div class="grid-cell">Module</div>
        <div class="grid-cell">File</div>
        <div class="grid-cell">Size</div>
      </div>
      {rows}
    </div>
    <p class="meta" style="margin-top:var(--space-3)">{len(modules)} modules loaded &middot; {sum(m.get('size_bytes',0) for m in modules)/1024:.0f} KB total source</p>
  </section>"""


def _objects_section(items: list[dict]) -> str:
    if not items:
        return ""
    rows = ""
    for o in items:
        refs = o.get("referrers", [])
        locs = o.get("locations", [])
        growth_str = str(o.get("growth", ""))
        growth_cls = "text-error" if growth_str.startswith("+") and len(growth_str) > 4 else "text-warning" if growth_str.startswith("+") else ""

        # Location: show file:variable (most useful for finding the leak)
        if locs:
            loc_str = ", ".join(Path(l.split(":")[0]).name + ":" + ":".join(l.split(":")[1:]) if ":" in l else l for l in locs[:3])
            loc_tip = _esc(" | ".join(locs))
        else:
            loc_str = "—"
            loc_tip = "Could not trace source location"

        # Held by: show container type
        refs_str = ", ".join(refs[:3]) if refs else "—"
        refs_tip = _esc(" | ".join(refs)) if refs else "No referrer data"

        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono">{_esc(o.get("type", "?"))}</div>'
            f'<div class="grid-cell mono">{o.get("count", 0):,}</div>'
            f'<div class="grid-cell mono {growth_cls}">{_esc(growth_str)}</div>'
            f'<div class="grid-cell mono has-tooltip" data-tooltip="{loc_tip}" style="cursor:help;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(loc_str)}</div>'
            f'<div class="grid-cell mono has-tooltip" data-tooltip="{refs_tip}" style="cursor:help;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(refs_str)}</div>'
            f'</div>'
        )
    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Object types that grew in count. 'Source' shows the file and variable name where objects are stored. 'Held By' shows the container type (list, dict, class attribute) holding the references.">Object Type Growth</h6>
    <div class="grid-table" style="grid-template-columns: 100px 80px 70px 1fr 1fr;">
      <div class="grid-header">
        <div class="grid-cell">Type</div>
        <div class="grid-cell">Count</div>
        <div class="grid-cell">Growth</div>
        <div class="grid-cell has-tooltip" data-tooltip="Source file and variable name where these objects are stored. e.g. test_agent.py:conversation_history means the variable 'conversation_history' in test_agent.py holds these objects.">Source</div>
        <div class="grid-cell has-tooltip" data-tooltip="Container type holding references. e.g. test_agent.py:conversation_history(list,len=80) means a list with 80 items.">Held By</div>
      </div>
      {rows}
    </div>
  </section>"""


def _network_section(conns: dict[str, dict]) -> str:
    if not conns:
        return ""
    rows = "".join(
        f'<div class="grid-row">'
        f'<div class="grid-cell mono">{_esc(c.get("laddr", ""))}</div>'
        f'<div class="grid-cell mono">{_esc(c.get("raddr", ""))}</div>'
        f'<div class="grid-cell mono">{_esc(c.get("status", ""))}</div>'
        f'<div class="grid-cell mono">{c.get("pid", "")}</div>'
        f'</div>'
        for c in conns.values()
    )
    return f"""<section class="card">
    <h6 class="label">Network Connections</h6>
    <div class="grid-table" style="grid-template-columns: 1fr 1fr 100px 60px;">
      <div class="grid-header">
        <div class="grid-cell">Local</div>
        <div class="grid-cell">Remote</div>
        <div class="grid-cell">Status</div>
        <div class="grid-cell">PID</div>
      </div>
      {rows}
    </div>
  </section>"""


def _gc_section(gc_data: list[dict]) -> str:
    if not gc_data:
        return ""
    last = gc_data[-1]
    gens = last.get("generations", [])
    rows = ""
    for i, g in enumerate(gens):
        uncol = g.get("uncollectable", 0)
        uncol_cls = "text-error" if uncol > 0 else ""
        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono">Gen {i}</div>'
            f'<div class="grid-cell mono">{g.get("collections", 0)}</div>'
            f'<div class="grid-cell mono">{g.get("collected", 0)}</div>'
            f'<div class="grid-cell mono {uncol_cls}">{uncol}</div>'
            f'</div>'
        )

    # Garbage details — uncollectable objects with source locations
    garbage_html = ""
    garbage_details = last.get("garbage_details", [])
    if garbage_details:
        garbage_rows = ""
        for gd in garbage_details:
            locs = gd.get("locations", [])
            loc_str = ", ".join(
                Path(l.split(":")[0]).name + ":" + ":".join(l.split(":")[1:]) if ":" in l else l
                for l in locs[:3]
            ) if locs else "unknown"
            loc_tip = _esc(" | ".join(locs)) if locs else "Could not trace source"
            garbage_rows += (
                f'<div class="grid-row">'
                f'<div class="grid-cell mono text-error">{_esc(gd.get("type", "?"))}</div>'
                f'<div class="grid-cell mono text-error">{gd.get("count", 0)}</div>'
                f'<div class="grid-cell mono has-tooltip" data-tooltip="{loc_tip}" style="cursor:help">{_esc(loc_str)}</div>'
                f'</div>'
            )
        garbage_html = f"""
    <h6 class="label text-error" style="margin-top:var(--space-4)">⚠ Uncollectable Objects (Definite Leaks)</h6>
    <div class="grid-table" style="grid-template-columns: 1fr 80px 1fr;">
      <div class="grid-header">
        <div class="grid-cell">Type</div>
        <div class="grid-cell">Count</div>
        <div class="grid-cell has-tooltip" data-tooltip="Source file and variable where uncollectable objects originate. These objects have __del__ methods that prevent GC from breaking reference cycles.">Source Location</div>
      </div>
      {garbage_rows}
    </div>"""
    elif last.get("garbage_count", 0) == 0:
        garbage_html = '<p class="meta text-success" style="margin-top:var(--space-3)">✓ No uncollectable garbage — no __del__ reference cycle leaks detected.</p>'

    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Python garbage collector stats. 'Uncollectable' > 0 means reference cycles involving objects with __del__ methods — the GC cannot break these cycles, so memory leaks permanently.">GC Statistics</h6>
    <div class="grid-table" style="grid-template-columns: 80px 100px 100px 120px;">
      <div class="grid-header">
        <div class="grid-cell">Gen</div>
        <div class="grid-cell has-tooltip" data-tooltip="Number of times this generation was collected.">Collections</div>
        <div class="grid-cell has-tooltip" data-tooltip="Total objects freed by GC.">Collected</div>
        <div class="grid-cell has-tooltip" data-tooltip="Objects in reference cycles that cannot be freed. This is a definite leak.">Uncollectable</div>
      </div>
      {rows}
    </div>
    <p class="meta" style="margin-top:var(--space-3)">Objects tracked: {last.get("objects_tracked", "?")} &middot; Garbage queue: {last.get("garbage_count", 0)}</p>
    {garbage_html}
  </section>"""


def _memray_section(stats_text: str) -> str:
    if not stats_text.strip():
        return ""
    # Clean up emoji for HTML
    clean = _esc(stats_text)
    return f"""<section class="card">
    <h6 class="label has-tooltip" data-tooltip="Deep allocation profiling by memray. Shows every malloc/calloc/realloc with full callstack. The flamegraph (memray_flamegraph.html) is in the output directory.">Memray Allocation Stats</h6>
    <pre class="memray-pre">{clean}</pre>
  </section>"""


# ---------------------------------------------------------------------------
# Chart JS (vanilla Canvas — no dependencies)
# ---------------------------------------------------------------------------

def _runtime_chart_js(times, rss, vms, cpu, threads) -> str:
    return f"""
const T = {json.dumps(times)};
const RSS = {json.dumps(rss)};
const VMS = {json.dumps(vms)};
const CPU = {json.dumps(cpu)};
const THR = {json.dumps(threads)};

function drawLine(ctx, xs, ys, color, maxY, w, h, padL, padB) {{
  if (xs.length < 2) return;
  const maxX = xs[xs.length-1] || 1;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (let i = 0; i < xs.length; i++) {{
    const x = padL + (xs[i]/maxX) * (w - padL);
    const y = h - padB - (ys[i]/maxY) * (h - padB - 20);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();
}}

function drawAxes(ctx, w, h, padL, padB, maxX, maxY, unit) {{
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = '10px IBM Plex Mono, monospace';
  // Y axis ticks
  for (let i = 0; i <= 4; i++) {{
    const val = (maxY / 4 * i).toFixed(0);
    const y = h - padB - (i/4) * (h - padB - 20);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillText(val + unit, 2, y + 4);
  }}
  // X axis ticks
  for (let i = 0; i <= 4; i++) {{
    const val = (maxX / 4 * i).toFixed(0);
    const x = padL + (i/4) * (w - padL);
    ctx.fillText(val + 's', x, h - 2);
  }}
}}

function drawDots(ctx, xs, ys, color, maxY, w, h, padL, padB) {{
  if (xs.length < 1) return;
  const maxX = xs[xs.length-1] || 1;
  ctx.fillStyle = color;
  for (let i = 0; i < xs.length; i++) {{
    const x = padL + (xs[i]/maxX) * (w - padL);
    const y = h - padB - (ys[i]/maxY) * (h - padB - 20);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI*2);
    ctx.fill();
  }}
}}

function legend(ctx, items, x, y) {{
  ctx.font = '11px IBM Plex Mono, monospace';
  items.forEach((it, i) => {{
    ctx.fillStyle = it[1];
    ctx.fillRect(x + i*120, y, 12, 3);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText(it[0], x + i*120 + 16, y + 5);
  }});
}}

// Memory chart
(function() {{
  const c = document.getElementById('memChart');
  if (!c || T.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr; c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, padL = 55, padB = 25;
  const maxY = Math.max(...VMS, ...RSS) * 1.1 || 100;
  const maxX = T[T.length-1] || 1;
  drawAxes(ctx, w, h, padL, padB, maxX, maxY, 'MB');
  drawLine(ctx, T, VMS, 'rgba(120,160,255,0.4)', maxY, w, h, padL, padB);
  drawLine(ctx, T, RSS, 'oklch(65% 0.2 145)', maxY, w, h, padL, padB);
  drawDots(ctx, T, RSS, 'oklch(65% 0.2 145)', maxY, w, h, padL, padB);
  legend(ctx, [['RSS', 'oklch(65% 0.2 145)'], ['VMS', 'rgba(120,160,255,0.5)']], padL + 10, 10);
}})();

// CPU chart — separate, clean
(function() {{
  const c = document.getElementById('cpuChart');
  if (!c || T.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr; c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, padL = 55, padB = 25;
  const maxCPU = Math.max(...CPU, 10) * 1.2;
  const maxX = T[T.length-1] || 1;
  drawAxes(ctx, w, h, padL, padB, maxX, maxCPU, '%');
  drawLine(ctx, T, CPU, 'oklch(75% 0.18 85)', maxCPU, w, h, padL, padB);
  drawDots(ctx, T, CPU, 'oklch(75% 0.18 85)', maxCPU, w, h, padL, padB);
  legend(ctx, [['CPU %', 'oklch(75% 0.18 85)']], padL + 10, 10);
}})();

// Threads chart — separate with integer Y axis
(function() {{
  const c = document.getElementById('threadChart');
  if (!c || T.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr; c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, padL = 55, padB = 25;
  const maxT = Math.max(...THR) + 2;
  const maxX = T[T.length-1] || 1;
  drawAxes(ctx, w, h, padL, padB, maxX, maxT, '');
  drawLine(ctx, T, THR, 'oklch(55% 0.18 230)', maxT, w, h, padL, padB);
  drawDots(ctx, T, THR, 'oklch(55% 0.18 230)', maxT, w, h, padL, padB);
  legend(ctx, [['Threads', 'oklch(55% 0.18 230)']], padL + 10, 10);
}})();
"""


# ---------------------------------------------------------------------------
# CSS — TBJS Glass (shared with static report)
# ---------------------------------------------------------------------------

def _runtime_css() -> str:
    return """
:root, [data-theme="dark"] {
  --primary: oklch(55% 0.18 230);
  --success: oklch(65% 0.2 145);
  --warning: oklch(75% 0.18 85);
  --error: oklch(55% 0.22 25);
  --info: oklch(60% 0.15 230);
  --bg-base: #08080d;
  --bg-surface: rgba(10, 10, 18, 0.8);
  --bg-elevated: rgba(15, 15, 25, 0.9);
  --bg-sunken: rgba(0, 0, 0, 0.3);
  --glass-bg: rgba(255, 255, 255, 0.02);
  --glass-border: rgba(255, 255, 255, 0.05);
  --glass-blur: 12px;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --text-main: rgba(255, 255, 255, 0.85);
  --text-label: rgba(255, 255, 255, 0.4);
  --text-muted: rgba(255, 255, 255, 0.25);
  --surface-hover: color-mix(in oklch, var(--primary) 5%, transparent);
  --surface-badge: color-mix(in oklch, var(--primary) 15%, transparent);
  --font-sans: 'IBM Plex Sans', system-ui, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, monospace;
  --text-base: 13px; --text-sm: 11px; --text-xs: 9px;
  --text-h1: clamp(18px, 2vw, 22px);
  --space-1: 0.25rem; --space-2: 0.5rem; --space-3: 0.75rem;
  --space-4: 1rem; --space-5: 1.5rem; --space-6: 2rem; --space-8: 3rem;
  --radius-md: 6px; --radius-lg: 12px;
  --highlight-inset: inset 0 1px 0 rgba(255,255,255,0.05);
  --shadow-micro: 0 2px 4px rgba(0,0,0,0.5);
  --duration-fast: 150ms;
  --ease-default: cubic-bezier(0.4, 0, 0.2, 1);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--font-sans); font-size: var(--text-base);
  color: var(--text-main); background: var(--bg-base);
  line-height: 1.6; overflow-x: hidden; -webkit-font-smoothing: antialiased;
}
.mono { font-family: var(--font-mono); }
.label {
  font-family: var(--font-mono); font-size: var(--text-xs);
  text-transform: uppercase; letter-spacing: 2.5px;
  color: var(--text-label); user-select: none;
}
h1 { font-size: var(--text-h1); font-weight: 700; letter-spacing: -0.02em; margin-block-end: var(--space-3); }
h6 { font-family: var(--font-mono); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 2.5px; color: var(--text-label); margin-block-end: var(--space-2); }
.report-container { max-width: 1100px; margin: 0 auto; padding: var(--space-8) var(--space-5); }
.report-header { margin-block-end: var(--space-6); }
.meta { font-size: var(--text-sm); color: var(--text-label); margin-top: var(--space-2); }
.cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: var(--space-3); margin-block-end: var(--space-6); }
.card {
  background: var(--glass-bg); border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg); padding: var(--space-5);
  box-shadow: var(--highlight-inset), var(--shadow-micro);
  backdrop-filter: blur(var(--glass-blur));
  margin-block-end: var(--space-4);
}
.metric-card { text-align: center; }
.metric-value { font-size: 28px; font-weight: 700; line-height: 1; margin: var(--space-3) 0; }
.metric-sub { font-size: var(--text-sm); color: var(--text-muted); }
/* LED score bars */
.led-bar { display: flex; gap: 3px; justify-content: center; margin-top: var(--space-3); }
.led-block {
  width: 12px; height: 5px; border-radius: 1px;
  background: var(--border-subtle);
  transition: background var(--duration-fast), box-shadow var(--duration-fast);
}
.led-block.led-active.led-success { background: var(--success); box-shadow: 0 0 4px color-mix(in oklch, var(--success) 15%, transparent); }
.led-block.led-active.led-warning { background: var(--warning); box-shadow: 0 0 4px color-mix(in oklch, var(--warning) 15%, transparent); }
.led-block.led-active.led-error { background: var(--error); box-shadow: 0 0 4px color-mix(in oklch, var(--error) 15%, transparent); }
.chart-section canvas { width: 100%; border-radius: var(--radius-md); }
.grid-table { display: grid; }
.grid-header, .grid-row { display: contents; }
.grid-header .grid-cell {
  font-family: var(--font-mono); font-size: var(--text-xs); font-weight: 600;
  color: var(--text-label); text-transform: uppercase; letter-spacing: 1px;
  padding: var(--space-2) var(--space-3); border-bottom: 1px solid var(--border-subtle);
}
.grid-row .grid-cell {
  padding: var(--space-2) var(--space-3); border-bottom: 1px solid rgba(255,255,255,0.03);
  font-size: var(--text-sm);
}
.grid-row:hover .grid-cell { background: var(--surface-hover); }
.text-error { color: var(--error); }
.text-warning { color: var(--warning); }
.text-success { color: var(--success); }
.memray-pre {
  font-family: var(--font-mono); font-size: var(--text-sm);
  color: var(--text-main); white-space: pre-wrap; line-height: 1.5;
  padding: var(--space-3); background: var(--bg-sunken);
  border-radius: var(--radius-md); overflow-x: auto;
}
.report-footer {
  display: flex; justify-content: space-between;
  padding-top: var(--space-5); border-top: 1px solid var(--border-subtle);
  margin-top: var(--space-8);
}
/* Tooltips */
.has-tooltip { position: relative; cursor: help; border-bottom: 1px dotted var(--text-muted); }
.has-tooltip::after {
  content: attr(data-tooltip);
  position: absolute; bottom: calc(100% + 8px); left: 50%;
  transform: translateX(-50%) scale(0.96);
  width: max-content; max-width: min(320px, 90vw);
  padding: var(--space-3) var(--space-4);
  background: var(--bg-elevated); border: 1px solid var(--glass-border);
  border-radius: var(--radius-md); box-shadow: var(--highlight-inset), 0 4px 8px rgba(0,0,0,0.6);
  font-family: var(--font-sans); font-size: var(--text-sm); font-weight: 400;
  color: var(--text-main); text-transform: none; letter-spacing: 0;
  line-height: 1.5; white-space: normal; z-index: 100;
  opacity: 0; visibility: hidden; pointer-events: none;
  transition: opacity var(--duration-fast), visibility var(--duration-fast), transform var(--duration-fast);
}
.has-tooltip:hover::after { opacity: 1; visibility: visible; transform: translateX(-50%) scale(1); }
.metric-card .has-tooltip::after { bottom: auto; top: calc(100% + 6px); }
.grid-header .grid-cell.has-tooltip::after { left: auto; right: 0; transform: translateX(0) scale(0.96); }
.grid-header .grid-cell.has-tooltip:hover::after { transform: translateX(0) scale(1); }
.grid-header .grid-cell:first-child.has-tooltip::after { left: 0; right: auto; }
/* Data row tooltips — anchor right to prevent overflow */
.grid-row .grid-cell.has-tooltip::after {
  left: auto; right: 0; transform: translateX(0) scale(0.96);
  max-width: min(450px, 85vw); white-space: normal; word-break: break-all;
}
.grid-row .grid-cell.has-tooltip:hover::after { transform: translateX(0) scale(1); }
.grid-row .grid-cell:first-child.has-tooltip::after { left: 0; right: auto; }
section { margin-block-end: var(--space-4); }
@media (max-width: 767px) {
  .report-container { padding: var(--space-5) var(--space-3); }
  .cards-grid { grid-template-columns: repeat(2, 1fr); }
}
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="tb_analyze_runtime — Runtime analysis with continuous monitoring"
    )
    sub = parser.add_subparsers(dest="cmd")

    # run command
    run_p = sub.add_parser("run", help="Run a command with monitoring")
    run_p.add_argument("command", help='Command to run (e.g. "python my_agent.py")')
    run_p.add_argument("--outdir", default="./runtime_data", help="Output directory")
    run_p.add_argument("--interval", type=float, default=2.0, help="Sampling interval (seconds)")
    run_p.add_argument("--memray", action="store_true", help="Enable deep memray profiling")
    run_p.add_argument("--no-objects", action="store_true", help="Disable objgraph tracking")
    run_p.add_argument("--no-network", action="store_true", help="Disable network tracking")

    # report command
    rep_p = sub.add_parser("report", help="Generate HTML report from collected data")
    rep_p.add_argument("outdir", help="Directory with runtime data")
    rep_p.add_argument("--output", "-o", default=None, help="Output HTML path")

    args = parser.parse_args()

    if args.cmd == "run":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        exit_code = run_with_monitoring(
            command=args.command,
            outdir=args.outdir,
            interval=args.interval,
            memray=args.memray,
            collect_objects=not args.no_objects,
            collect_network=not args.no_network,
        )
        sys.exit(exit_code)

    elif args.cmd == "report":
        outdir = Path(args.outdir)
        output = args.output or str(outdir / "runtime_report.html")
        html = generate_runtime_report(outdir, output)
        print(f"Report: {output} ({len(html)} bytes)")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
