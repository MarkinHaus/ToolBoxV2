# debug_watchdog.py
import threading, sys, traceback, time, os, asyncio, linecache

# === ANPASSEN ===
PROJECT_ROOT = os.path.normpath(r"C:\Users\Markin\Workspace\ToolBoxV2")
TIMEOUT = 60
# ================

_loop_ref = None  # gespeicherte Loop-Referenz


def _is_project_frame(filename):
    norm = os.path.normpath(filename)
    return norm.startswith(PROJECT_ROOT) and ".venv" not in norm


def _dump_async_tasks():
    lines = []
    loop = _loop_ref

    if loop is None or loop.is_closed():
        lines.append("\n  ℹ Kein Event-Loop registriert (install(loop=...) nutzen)")
        return lines

    try:
        # all_tasks MIT expliziter loop-referenz
        tasks = asyncio.all_tasks(loop)
    except RuntimeError as e:
        lines.append(f"\n  ❌ all_tasks fehlgeschlagen: {e}")
        return lines

    if not tasks:
        lines.append("\n  ℹ Keine async Tasks aktiv.")
        return lines

    lines.append(f"\n{'─' * 70}")
    lines.append(f"🔄 ASYNC TASKS ({len(tasks)} aktiv)")
    lines.append(f"{'─' * 70}")

    for task in sorted(tasks, key=lambda t: t.get_name()):
        task_name = task.get_name()
        state = task._state if hasattr(task, '_state') else "?"

        # Coroutine-Info
        coro = task.get_coro()
        coro_info = ""
        cr_code = getattr(coro, 'cr_code', None)
        if cr_code:
            fname = cr_code.co_filename
            if _is_project_frame(fname):
                coro_info = f" → {os.path.relpath(fname, PROJECT_ROOT)}:{cr_code.co_firstlineno}"
            else:
                coro_info = f" → {os.path.basename(fname)}:{cr_code.co_firstlineno}"

        # Stack aus Task holen
        try:
            task_stack = task.get_stack(limit=50)
        except Exception:
            task_stack = []

        project_entries = []
        all_entries = []
        for frame in task_stack:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            name = frame.f_code.co_name
            try:
                line = linecache.getline(filename, lineno).strip()
            except Exception:
                line = ""
            entry = (filename, lineno, name, line)
            all_entries.append(entry)
            if _is_project_frame(filename):
                project_entries.append(entry)

        if project_entries:
            lines.append(f"\n  ▶ Task: {task_name} [{state}]{coro_info}")

            if all_entries:
                last = all_entries[-1]
                if not _is_project_frame(last[0]):
                    short = os.path.basename(last[0])
                    lines.append(f"    ⏸ wartet in: {short}:{last[1]} → {last[2]}()")

            lines.append(f"    📍 Dein Code:")
            for filename, lineno, name, line_text in project_entries:
                rel = os.path.relpath(filename, PROJECT_ROOT)
                lines.append(f"       {rel}:{lineno} → {name}()")
                if line_text:
                    lines.append(f"         | {line_text}")
        else:
            lines.append(f"  ◦ Task: {task_name} [{state}]{coro_info} (nur Library-Code)")

    return lines


def dump_project_threads(reason="manual"):
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"DEBUG STACK DUMP ({reason}) - {time.strftime('%H:%M:%S')}")
    lines.append(f"{'=' * 70}")

    frames = sys._current_frames()
    thread_map = {t.ident: t for t in threading.enumerate()}
    watchdog_tid = threading.current_thread().ident

    for tid, frame in frames.items():
        if tid == watchdog_tid:
            continue
        t = thread_map.get(tid)
        name = t.name if t else f"unknown-{tid}"
        stack = traceback.extract_stack(frame)
        project_frames = [f for f in stack if _is_project_frame(f.filename)]

        if project_frames:
            lines.append(f"\n--- Thread: {name} (0x{tid:04x}) ---")
            lib_frame = stack[-1] if stack else None
            if lib_frame and not _is_project_frame(lib_frame.filename):
                short = os.path.basename(lib_frame.filename)
                lines.append(f"  ⏸ blockiert in: {short}:{lib_frame.lineno} → {lib_frame.name}()")
            lines.append(f"  📍 Dein Code (aufrufreihenfolge):")
            for f in project_frames:
                rel = os.path.relpath(f.filename, PROJECT_ROOT)
                lines.append(f"     {rel}:{f.lineno} → {f.name}()")
                if f.line:
                    lines.append(f"       | {f.line.strip()}")

    # === ASYNC TASKS ===
    lines.extend(_dump_async_tasks())

    lines.append(f"\n{'=' * 70}\n")
    print("\n".join(lines), file=sys.stderr, flush=True)


class HangWatchdog:
    def __init__(self, timeout=TIMEOUT):
        self.timeout = timeout
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True, name="hang-watchdog")
        self._t.start()

    def _loop(self):
        while not self._stop.wait(self.timeout):
            dump_project_threads(reason=f"auto all {self.timeout}s")

    def stop(self):
        self._stop.set()


def install(timeout=TIMEOUT, loop=None):
    """
    Ganz oben im Entrypoint aufrufen.

    loop: asyncio event loop übergeben damit async tasks gedumpt werden.
          Wenn None, wird versucht den aktuellen Loop zu finden.
    """
    global _loop_ref
    if loop is not None:
        _loop_ref = loop
    else:
        try:
            _loop_ref = asyncio.get_event_loop()
        except RuntimeError:
            _loop_ref = None

    status = "mit async" if _loop_ref else "ohne async"
    print(f"[watchdog] active ({status}) – dumps every {timeout}s, PID={os.getpid()}", file=sys.stderr)
    return HangWatchdog(timeout=timeout)
