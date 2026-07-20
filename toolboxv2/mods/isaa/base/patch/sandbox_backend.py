"""
sandbox_backend.py
==================
Phase 1+2 of the VFS sandbox migration.

SandboxBackend         — protocol every backend must satisfy
AIOSandboxBackend      — agent-infra AIO Sandbox via `agent_sandbox` SDK
SandboxRegistry        — ONE container per AGENT (not per session);
                         sessions get isolated workdirs /work/<session_id>
DockerLifecycle        — auto pull/start of the AIO container, port scan
                         in the 7000–7099 range (e.g. 7053)
Remote 2-part key      — "<conn_key>.<folder_key>"
                         conn_key   -> resolved via ~/.tb_sandbox/remotes.json
                                       {conn_key: {"base_url": ..., "token": ...}}
                         folder_key -> namespace dir /work/<folder_key> on the
                                       remote sandbox

Obs contract (Phase 5): every backend operation calls
`self._emit(kind, payload)`. The session wires this to the
ObservabilityLayer (shell exec -> _audit, file transfer -> record_vfs_delta).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

AIO_IMAGE = os.getenv("TB_SANDBOX_IMAGE", "ghcr.io/agent-infra/sandbox:latest")
PORT_RANGE = (7000, 7099)
from toolboxv2 import get_app
STATE_DIR = Path(get_app().appdata) / ".tb_sandbox"
CONTAINER_PREFIX = "tb-sbx-"
KEY_RE = re.compile(r"^([A-Za-z0-9_-]{4,64})\.([A-Za-z0-9_-]{4,64})$")
GLOBAL_DIR = Path(os.getenv("TB_SANDBOX_GLOBAL_DIR", str(Path(get_app().data_dir) / "Agents"/ "VFS"/ "global")))
GLOBAL_MOUNT = "/work/global"


def configured_mounts() -> list[tuple[str, str]]:
    """[(host, container)] — global share first, then TB_SANDBOX_MOUNTS."""
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    mounts = [(str(GLOBAL_DIR), GLOBAL_MOUNT)]
    for spec in filter(None, os.getenv("TB_SANDBOX_MOUNTS", "").split(",")):
        if ":" in spec:
            host, cont = spec.split(":", 1)
            mounts.append((host.strip(), cont.strip()))
    return mounts

ObsEmitter = Callable[[str, dict], None]


def _ok(stdout: str = "", stderr: str = "", returncode: int = 0, **extra) -> dict:
    d = {"success": returncode == 0, "stdout": stdout, "stderr": stderr, "returncode": returncode}
    d.update(extra)
    return d


def _err(msg: Any, returncode: int = 1, **extra) -> dict:
    d = {"success": False, "stdout": "", "stderr": str(msg), "returncode": returncode}
    d.update(extra)
    return d


# =============================================================================
# PHASE 1 — BACKEND PROTOCOL
# =============================================================================

@runtime_checkable
class SandboxBackend(Protocol):
    """Contract for any sandbox backend (AIO today, microsandbox later)."""

    workdir: str

    def start(self) -> dict: ...
    def stop(self) -> dict: ...
    def health(self) -> dict: ...

    def exec(self, command: str, cwd: str | None = None, timeout: float | None = None) -> dict: ...
    def code(self, code: str, timeout: int | None = None) -> dict: ...

    def read(self, path: str, start_line: int | None = None, end_line: int | None = None) -> dict: ...
    def write(self, path: str, content: str, append: bool = False) -> dict: ...
    def ls(self, path: str, recursive: bool = False) -> dict: ...
    def grep(self, path: str, pattern: str, **kw) -> dict: ...

    def upload(self, host_src: str, sb_dst: str) -> dict: ...
    def download(self, sb_src: str, host_dst: str) -> dict: ...

    def browser_screenshot(self) -> dict: ...
    def browser_action(self, action: dict) -> dict: ...
    def browser_info(self) -> dict: ...


# =============================================================================
# PHASE 2a — DOCKER LIFECYCLE (local auto-setup)
# =============================================================================

def _docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


def _free_port(lo: int = PORT_RANGE[0], hi: int = PORT_RANGE[1]) -> int:
    for port in range(lo, hi + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port in {lo}-{hi}")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "-", name.lower())[:40]


class DockerLifecycle:
    """Pull/start/reuse the AIO container for one agent. Idempotent."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.container = CONTAINER_PREFIX + _slug(agent_name)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._reg = STATE_DIR / "containers.json"

    # -- registry ------------------------------------------------------------
    def _load(self) -> dict:
        try:
            return json.loads(self._reg.read_text())
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        self._reg.write_text(json.dumps(data, indent=2))

    # -- docker probes ---------------------------------------------------------
    def docker_available(self) -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            return _docker("info", timeout=10).returncode == 0
        except Exception:
            return False

    def _state(self) -> str:
        r = _docker("inspect", "-f", "{{.State.Status}}", self.container, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else "absent"

    def _image_present(self) -> bool:
        return _docker("image", "inspect", AIO_IMAGE, timeout=15).returncode == 0

    # -- main entry ------------------------------------------------------------
    def ensure(self) -> dict:
        """Container running -> base_url. Pulls image + starts if missing."""
        if not self.docker_available():
            return _err("docker not available — install Docker or use a remote sandbox key")

        reg = self._load()
        state = self._state()
        mounts = configured_mounts()

        if state == "running":
            if reg.get(self.container, {}).get("mounts") != mounts and \
               reg.get(self.container, {}).get("mounts") != [list(m) for m in mounts]:
                _docker("rm", "-f", self.container, timeout=30)   # mounts changed -> recreate
                state = "absent"
            else:
                port = reg.get(self.container, {}).get("port")
                if port is None:  # recover port from docker
                    r = _docker("port", self.container, "8080/tcp", timeout=15)
                    m = re.search(r":(\d+)\s*$", r.stdout.strip().splitlines()[0] if r.stdout.strip() else "")
                    port = int(m.group(1)) if m else None
                if port:
                    return _ok(base_url=f"http://127.0.0.1:{port}", port=port, reused=True)

        if state in ("exited", "created", "paused"):
            _docker("rm", "-f", self.container, timeout=30)

        if not self._image_present():
            pull = _docker("pull", AIO_IMAGE, timeout=900)
            if pull.returncode != 0:
                return _err(f"docker pull failed: {pull.stderr[-500:]}")

        port = _free_port()
        vol_args: list[str] = []
        for host, cont in mounts:
            Path(host).mkdir(parents=True, exist_ok=True)
            vol_args += ["-v", f"{host}:{cont}"]
        run = _docker(
            "run", "-d", "--name", self.container,
            "--security-opt", "seccomp=unconfined",
            "--restart", "unless-stopped",
            "--label", "tb.sandbox=1", "--label", f"tb.agent={self.agent_name}",
            "-p", f"{port}:8080",
            *vol_args,
            AIO_IMAGE,
            timeout=120,
        )
        if run.returncode != 0:
            return _err(f"docker run failed: {run.stderr[-500:]}")

        _docker("exec", "-u", "root", self.container, "sh", "-c",
                "mkdir -p /work && chmod 0777 /work", timeout=30)

        reg[self.container] = {"agent": self.agent_name, "port": port, "image": AIO_IMAGE,
                               "mounts": mounts, "started": time.time()}
        self._save(reg)
        return _ok(base_url=f"http://127.0.0.1:{port}", port=port, reused=False)

    def stop(self) -> dict:
        r = _docker("rm", "-f", self.container, timeout=60)
        reg = self._load()
        reg.pop(self.container, None)
        self._save(reg)
        return _ok(stdout=r.stdout) if r.returncode == 0 else _err(r.stderr)


# =============================================================================
# PHASE 2b — REMOTE 2-PART KEY
# =============================================================================

@dataclass
class RemoteTarget:
    base_url: str
    token: str | None
    folder_key: str

    @property
    def workdir(self) -> str:
        return f"/work/{self.folder_key}"


@dataclass
class SandboxPolicy:
    """Transparent permission set. Set from OUTSIDE (host code / env), shown
    to the agent via sandbox_permissions. Defaults are intentionally inclusive:
    the agent gets full control inside the sandbox; only the Schleuse (host
    boundary) is gated."""
    allowed_import_dirs: list[str] | None = None   # None = any host path may be imported
    allowed_export_dirs: list[str] | None = None   # None = any host path may be written
    export_prefix: str = "out/"                    # "" = whole workdir may leave
    allow_shell: bool = True
    allow_code: bool = True
    allow_browser: bool = True
    allow_network: bool = True                     # informational (enforced via container net)
    max_transfer_mb: int = 512

    @classmethod
    def from_env(cls) -> "SandboxPolicy":
        def _dirs(var: str) -> list[str] | None:
            v = os.getenv(var, "").strip()
            return [d.strip() for d in v.split(",") if d.strip()] or None
        return cls(
            allowed_import_dirs=[str(Path(get_app().data_dir) / "Agents"/ "VFS")].extend(_dirs("TB_SANDBOX_IMPORT_DIRS") if _dirs("TB_SANDBOX_IMPORT_DIRS") is not None else []),
            allowed_export_dirs=[str(Path(get_app().data_dir) / "Agents"/ "VFS")].extend(_dirs("TB_SANDBOX_EXPORT_DIRS") if _dirs("TB_SANDBOX_EXPORT_DIRS") is not None else []),
            export_prefix=os.getenv("TB_SANDBOX_EXPORT_PREFIX", "out/"),
            allow_shell=os.getenv("TB_SANDBOX_ALLOW_SHELL", "1") != "0",
            allow_code=os.getenv("TB_SANDBOX_ALLOW_CODE", "1") != "0",
            allow_browser=os.getenv("TB_SANDBOX_ALLOW_BROWSER", "1") != "0",
            allow_network=os.getenv("TB_SANDBOX_ALLOW_NETWORK", "1") != "0",
            max_transfer_mb=int(os.getenv("TB_SANDBOX_MAX_TRANSFER_MB", "512")),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def get_policy(session) -> SandboxPolicy:
    """session.sandbox_policy (set from outside) > env > inclusive defaults.
    Back-compat: legacy session.sandbox_allowed_dirs maps to both dir lists."""
    pol = getattr(session, "sandbox_policy", None)
    if isinstance(pol, SandboxPolicy):
        return pol
    pol = SandboxPolicy.from_env()
    legacy = getattr(session, "sandbox_allowed_dirs", None)
    if legacy:
        pol.allowed_import_dirs = pol.allowed_import_dirs or list(legacy)
        pol.allowed_export_dirs = pol.allowed_export_dirs or list(legacy)
    try:
        session.sandbox_policy = pol  # cache so the shown policy == enforced policy
    except Exception:
        pass
    return pol


def resolve_remote_key(key: str) -> RemoteTarget:
    """
    "<conn_key>.<folder_key>" -> RemoteTarget.
    conn_key is looked up in ~/.tb_sandbox/remotes.json:
        {"k7f2x9":{"base_url":"https://sbx.example.com:7053","token":"..."}}
    """
    m = KEY_RE.match(key.strip())
    if not m:
        raise ValueError("invalid sandbox key — expected '<conn_key>.<folder_key>'")
    conn_key, folder_key = m.group(1), m.group(2)
    remotes_file = STATE_DIR / "remotes.json"
    try:
        remotes = json.loads(remotes_file.read_text())
    except Exception:
        raise ValueError(f"no remotes registered ({remotes_file} missing/unreadable)")
    entry = remotes.get(conn_key)
    if not entry or "base_url" not in entry:
        raise ValueError(f"unknown connection key '{conn_key}'")
    return RemoteTarget(base_url=entry["base_url"].rstrip("/"),
                        token=entry.get("token"), folder_key=folder_key)


def register_remote(conn_key: str, base_url: str, token: str | None = None) -> dict:
    """Host-side helper (CLI/admin, not an agent tool)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    f = STATE_DIR / "remotes.json"
    try:
        remotes = json.loads(f.read_text())
    except Exception:
        remotes = {}
    remotes[conn_key] = {"base_url": base_url, "token": token}
    f.write_text(json.dumps(remotes, indent=2))
    return _ok(stdout=f"remote '{conn_key}' registered")


# =============================================================================
# PHASE 1 impl — AIO BACKEND
# =============================================================================

class AIOSandboxBackend:
    """
    agent-infra AIO Sandbox backend (full sandbox: shell, files, jupyter, browser).

    local : one docker container per AGENT, workdir /work/<session_id>
    remote: 2-part key, workdir /work/<folder_key>
    """

    def __init__(
        self,
        base_url: str,
        workdir: str,
        token: str | None = None,
        obs_emit: ObsEmitter | None = None,
        label: str = "local",
    ):
        from agent_sandbox import Sandbox  # lazy — pip install agent-sandbox
        headers = {"Authorization": f"Bearer {token}"} if token else None
        kw = {"base_url": base_url}
        if headers:
            kw["headers"] = headers
        self.client = Sandbox(**kw)
        self.base_url = base_url
        self.workdir = workdir
        self.label = label
        self._emit_fn = obs_emit
        self._shell_session = "tb-" + re.sub(r"[^A-Za-z0-9_-]", "-", workdir.strip("/").replace("/", "-"))[:48]
        self._shell_ready = False

    # -- obs (Phase 5 choke point) --------------------------------------------
    def _emit(self, kind: str, payload: dict) -> None:
        if self._emit_fn:
            try:
                self._emit_fn(kind, {"backend": self.label, "workdir": self.workdir, **payload})
            except Exception:
                pass  # obs must never break execution

    def set_obs(self, emit: ObsEmitter) -> None:
        self._emit_fn = emit

    # -- helpers ---------------------------------------------------------------
    def _abs(self, path: str) -> str:
        # Agent-facing namespace prefix: sbox:/x == /x, sbox:x == relative x.
        if path.startswith("sbox:"):
            path = path[5:]
        if path.startswith("/"):
            return path
        return f"{self.workdir.rstrip('/')}/{path}"

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> dict:
        r = self.exec(f"mkdir -p {self.workdir} && echo ready")
        return r if not r["success"] else _ok(stdout=f"workdir {self.workdir} ready @ {self.base_url}")

    def stop(self) -> dict:
        # remote/shared containers are NOT killed here — lifecycle owns that
        return _ok(stdout="detached")

    def health(self) -> dict:
        try:
            ctx = self.client.sandbox.get_context()
            home = getattr(getattr(ctx, "data", None), "home_dir", None) or "?"
            return _ok(stdout=f"healthy home={home}")
        except Exception as e:
            return _err(f"unhealthy: {e}")

    # -- shell / code ------------------------------------------------------------
    def _exec_raw(self, command: str, timeout: float = 30) -> str:
        """One-off command WITHOUT a shell session (bootstrap-safe)."""
        r = self.client.shell.exec_command(command=command, timeout=timeout)
        return getattr(getattr(r, "data", None), "output", "") or ""

    def _bootstrap_workdir(self) -> None:
        """Create + own the workdir before any session binds to it."""
        wd = self.workdir
        self._exec_raw(
            f"sudo mkdir -p '{wd}/out' && sudo chown -R $(whoami) '{wd}' "
            f"|| mkdir -p '{wd}/out'"
        )

    def _ensure_shell(self, force: bool = False) -> None:
        if self._shell_ready and not force:
            return
        try:
            self._bootstrap_workdir()
        except Exception:
            pass  # surfaced by the failing exec below if it actually matters
        try:
            self.client.shell.create_session(id=self._shell_session, exec_dir=self.workdir)
        except Exception:
            pass  # already exists -> fine; real errors surface in exec_command
        self._shell_ready = True

    def _exec_once(self, command: str, cwd: str | None, timeout: float | None):
        return self.client.shell.exec_command(
            command=command,
            id=self._shell_session,
            exec_dir=cwd or self.workdir,
            **({"timeout": timeout} if timeout else {}),
        )

    def exec(self, command: str, cwd: str | None = None, timeout: float | None = None) -> dict:
        t0 = time.time()
        try:
            self._ensure_shell()
            try:
                r = self._exec_once(command, cwd, timeout)
            except Exception as e:
                msg = str(e)
                if "404" in msg or "not found" in msg.lower():
                    self._ensure_shell(force=True)   # session died (container restart) -> recreate
                    r = self._exec_once(command, cwd, timeout)
                else:
                    raise
            d = getattr(r, "data", None)
            out = getattr(d, "output", "") or ""
            code = getattr(d, "exit_code", None)
            code = 0 if code is None else int(code)
            res = _ok(stdout=out, returncode=code) if code == 0 else _ok(stdout=out, stderr=out, returncode=code)
        except Exception as e:
            res = _err(e)
        self._emit("sandbox.exec", {"command": command[:300], "returncode": res["returncode"],
                                    "ms": round((time.time() - t0) * 1000, 1)})
        return res

    # -- labelled background shell sessions (A/B/C) -----------------------------
    #
    # Design: the agent-facing shell no longer blocks on the SDK's synchronous
    # exec_command timeout. Commands run DETACHED inside the sandbox, output is
    # appended to a per-session log file, and the host polls cheap size probes.
    # This gives:
    #   - early stop when output goes idle (instead of waiting out the timeout)
    #   - late output is captured in the log; the next call on the same session
    #     label returns everything new since the last read (cursor-based)
    #   - real multiline commands/heredocs: the command is written to a script
    #     file first, so no escaping ever passes through the exec channel
    SESSION_LABELS = ("A", "B", "C")

    def _job_paths(self, label: str) -> dict:
        base = f"{self.workdir.rstrip('/')}/.shell"
        return {"dir": base, "log": f"{base}/{label}.log", "pid": f"{base}/{label}.pid",
                "exit": f"{base}/{label}.exit", "script": f"{base}/{label}.cmd.sh"}

    def _job_state(self, label: str) -> tuple[bool, int, int | None]:
        """(running, log_size_bytes, exit_code|None) via one cheap probe."""
        p = self._job_paths(label)
        out = self._exec_raw(
            f"mkdir -p '{p['dir']}'; "
            f"if [ -f '{p['pid']}' ] && kill -0 $(cat '{p['pid']}') 2>/dev/null; "
            f"then echo RUN; else echo IDLE; fi; "
            f"([ -f '{p['log']}' ] && wc -c < '{p['log']}') || echo 0; "
            f"([ -f '{p['exit']}' ] && cat '{p['exit']}') || echo NA",
            timeout=20,
        )
        lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
        running = bool(lines) and lines[0] == "RUN"
        size = 0
        if len(lines) > 1:
            try:
                size = int(lines[1])
            except ValueError:
                pass
        exit_code = None
        if len(lines) > 2 and lines[2] != "NA":
            try:
                exit_code = int(lines[2])
            except ValueError:
                pass
        return running, size, exit_code

    def _job_read_delta(self, label: str, offset: int, limit: int = 30000) -> str:
        p = self._job_paths(label)
        return self._exec_raw(
            f"tail -c +{offset + 1} '{p['log']}' 2>/dev/null | head -c {limit}",
            timeout=30,
        )

    def _cursors(self) -> dict:
        cur = getattr(self, "_shell_cursors", None)
        if cur is None:
            cur = {}
            self._shell_cursors = cur
        return cur

    def exec_session(self, command: str = "", label: str = "A",
                     timeout: float | None = None, idle_timeout: float = 8.0,
                     reset: bool = False, poll_interval: float = 0.5) -> dict:
        """Run a command in labelled session A/B/C. Empty command = fetch new
        output of that session. reset=True kills the session's job and clears
        its log. Returns early once output has been idle for idle_timeout s."""
        label = (label or "A").strip().upper()
        if label not in self.SESSION_LABELS:
            return _err(f"unknown shell session '{label}' — max {len(self.SESSION_LABELS)} "
                        f"sessions, labels {list(self.SESSION_LABELS)}")
        p = self._job_paths(label)
        cursors = self._cursors()
        t0 = time.time()

        if reset:
            self._exec_raw(
                f"[ -f '{p['pid']}' ] && kill -9 $(cat '{p['pid']}') 2>/dev/null; "
                f"rm -f '{p['log']}' '{p['pid']}' '{p['exit']}' '{p['script']}'; echo reset",
                timeout=20,
            )
            cursors[label] = 0
            self._emit("sandbox.exec", {"command": f"<session {label} reset>", "returncode": 0,
                                        "ms": round((time.time() - t0) * 1000, 1)})
            return _ok(stdout=f"session {label} reset")

        running, size, exit_code = self._job_state(label)
        cursor = cursors.get(label, 0)
        if cursor > size:  # log was truncated/rotated externally
            cursor = 0

        # Poll mode: no command → return whatever is new since the last read.
        if not command.strip():
            delta = self._job_read_delta(label, cursor) if size > cursor else ""
            cursors[label] = size
            status = "running" if running else "idle"
            note = "" if running else (
                f"\n[exit code: {exit_code}]" if exit_code is not None else "")
            return _ok(stdout=(delta or f"(no new output, session {label} {status})") + note,
                       returncode=0 if (running or not exit_code) else exit_code)

        # Busy session: never queue silently — surface fresh output + options.
        if running:
            delta = self._job_read_delta(label, cursor) if size > cursor else ""
            cursors[label] = size
            return _err(
                f"session {label} is still running a command. New output since last read:\n"
                f"{delta or '(none)'}\n"
                f"→ poll with sandbox_shell(shell_session='{label}', command=''), use another "
                f"session label, or reset with reset=True.", returncode=2)

        # Start: leftover late output of the previous command is picked up first.
        late = self._job_read_delta(label, cursor) if size > cursor else ""

        # Multiline-safe: the command body goes into a script file (no escaping
        # through the exec channel), then runs detached with exit-code capture.
        w = self.write(p["script"], command if command.endswith("\n") else command + "\n")
        if not w.get("success"):
            return w
        # The braces-group form fully detaches the job even when the exec
        # channel captures stdout/stderr via pipes (child must not inherit them).
        start = self._exec_raw(
            f"rm -f '{p['exit']}'; cd '{self.workdir}' && "
            f"{{ nohup bash -c 'bash \"{p['script']}\"; echo $? > \"{p['exit']}\"' "
            f">> '{p['log']}' 2>&1 < /dev/null & }} 2>/dev/null; echo $!",
            timeout=20,
        )
        pid = start.strip().splitlines()[-1].strip() if start.strip() else ""
        self._exec_raw(f"echo '{pid}' > '{p['pid']}'", timeout=10)
        start_size = size

        # Host-side wait: early stop on process end OR idle output OR hard cap.
        hard_cap = min(float(timeout) if timeout else 120.0, 600.0)
        last_growth = time.time()
        last_size = start_size
        running_now, exit_code = True, None
        while True:
            time.sleep(poll_interval)
            running_now, cur_size, exit_code = self._job_state(label)
            if cur_size > last_size:
                last_size = cur_size
                last_growth = time.time()
            if not running_now:
                break
            now = time.time()
            if now - t0 >= hard_cap:
                break
            if now - last_growth >= max(idle_timeout, poll_interval):
                break

        _, final_size, exit_code2 = self._job_state(label)
        exit_code = exit_code2 if exit_code2 is not None else exit_code
        delta = self._job_read_delta(label, start_size) if final_size > start_size else ""
        cursors[label] = final_size

        out_parts = []
        if late:
            out_parts.append(f"[late output of previous command in session {label}]\n{late}\n---")
        out_parts.append(delta or "(no output)")
        if running_now:
            reason = "hard timeout" if (time.time() - t0 >= hard_cap) else \
                f"no new output for {idle_timeout:.0f}s"
            out_parts.append(
                f"\n⏳ command in session {label} is STILL RUNNING ({reason}). It keeps "
                f"running in the background; its output is captured. Fetch new output "
                f"later with sandbox_shell(shell_session='{label}', command='').")
            rc = 0
        else:
            rc = exit_code if exit_code is not None else 0
        res = _ok(stdout="\n".join(out_parts), returncode=rc) if rc == 0 else \
            {"success": False, "stdout": "\n".join(out_parts), "stderr": "", "returncode": rc}
        res["session"] = label
        res["running"] = running_now
        self._emit("sandbox.exec", {"command": command[:300], "session": label,
                                    "returncode": rc, "running": running_now,
                                    "ms": round((time.time() - t0) * 1000, 1)})
        return res

    def code(self, code: str, timeout: int | None = None) -> dict:
        t0 = time.time()
        try:
            r = self.client.jupyter.execute_code(
                code=code, cwd=self.workdir, **({"timeout": timeout} if timeout else {})
            )
            d = getattr(r, "data", None)
            outputs = getattr(d, "outputs", None) or []
            parts: list[str] = []
            for o in outputs:
                if isinstance(o, dict):
                    txt = o.get("text") or o.get("data", {}).get("text/plain") \
                          or "\n".join(o.get("traceback", []) or [])
                else:
                    txt = getattr(o, "text", None) \
                          or (getattr(o, "data", None) or {}).get("text/plain", "") \
                          or "\n".join(getattr(o, "traceback", None) or [])
                if txt:
                    parts.append(str(txt))
            res = _ok(stdout="\n".join(parts))
        except Exception as e:
            res = _err(e)
        self._emit("sandbox.code", {"chars": len(code), "returncode": res["returncode"],
                                    "ms": round((time.time() - t0) * 1000, 1)})
        return res

    # -- files ---------------------------------------------------------------------
    def read(self, path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
        try:
            kw: dict = {"file": self._abs(path)}
            if start_line is not None:
                kw["start_line"] = start_line
            if end_line is not None:
                kw["end_line"] = end_line
            r = self.client.file.read_file(**kw)
            content = getattr(getattr(r, "data", None), "content", "") or ""
            return _ok(stdout=content, content=content)
        except Exception as e:
            return _err(e)

    def write(self, path: str, content: str, append: bool = False) -> dict:
        try:
            self.client.file.write_file(file=self._abs(path), content=content, append=append)
            res = _ok(stdout=f"wrote {len(content)} chars -> {self._abs(path)}")
        except Exception as e:
            res = _err(e)
        self._emit("sandbox.write", {"path": self._abs(path), "chars": len(content),
                                     "append": append, "success": res["success"]})
        return res

    def ls(self, path: str = ".", recursive: bool = False) -> dict:
        try:
            r = self.client.file.list_path(path=self._abs(path), recursive=recursive, include_size=True)
            items = getattr(getattr(r, "data", None), "files", None) or []
            lines = []
            for it in items:
                name = getattr(it, "path", None) or getattr(it, "name", str(it))
                size = getattr(it, "size", "")
                lines.append(f"{name}\t{size}")
            return _ok(stdout="\n".join(lines))
        except Exception as e:
            return _err(e)

    def grep(self, path: str, pattern: str, **kw) -> dict:
        try:
            r = self.client.file.grep_files(path=self._abs(path), pattern=pattern, recursive=True, **kw)
            d = getattr(r, "data", None)
            matches = getattr(d, "matches", None) or getattr(d, "results", None) or []
            lines = [str(m) for m in matches]
            return _ok(stdout="\n".join(lines))
        except Exception as e:
            return _err(e)

    def edit(self, command: str, path: str, **kw) -> dict:
        """Precise file editing via the sandbox editor API.
        command: view | create | str_replace | insert | undo_edit
        kw: file_text, old_str, new_str, insert_line, view_range, replace_mode"""
        try:
            self._ensure_shell()
            r = self.client.file.str_replace_editor(command=command, path=self._abs(path), **kw)
            d = getattr(r, "data", None)
            out = getattr(d, "output", None) or getattr(d, "content", None) or str(d)
            res = _ok(stdout=str(out))
        except Exception as e:
            res = _err(e)
        self._emit("sandbox.edit", {"path": self._abs(path), "command": command,
                                    "success": res["success"]})
        return res

    # -- Schleuse (Phase 4 primitives) ---------------------------------------------
    def upload(self, host_src: str, sb_dst: str) -> dict:
        try:
            p = Path(host_src)
            if not p.is_file():
                return _err(f"not a file: {host_src}")
            data = p.read_bytes()
            dst = self._abs(sb_dst)
            self._ensure_shell()  # bootstrap workdir perms first
            try:
                self.client.file.upload_file(file=(p.name, data), path=dst)
                res = _ok(stdout=f"{host_src} -> sandbox:{dst} ({len(data)} bytes)", bytes=len(data))
            except Exception as e1:
                # fallback: base64 via write_file (robust against upload endpoint quirks)
                import base64
                try:
                    self.client.file.write_file(
                        file=dst, content=base64.b64encode(data).decode(), encoding="base64")
                    res = _ok(stdout=f"{host_src} -> sandbox:{dst} ({len(data)} bytes, b64 fallback)",
                              bytes=len(data))
                except Exception as e2:
                    hint = "PermissionError" if "Permission" in f"{e1}{e2}" else ""
                    res = _err(f"upload failed ({hint or 'both paths'}): {e2} | first: {e1}")
        except Exception as e:
            res = _err(e)
        self._emit("schleuse.in", {"host_src": host_src, "sb_dst": self._abs(sb_dst),
                                   "bytes": res.get("bytes", 0), "success": res["success"]})
        return res

    def download(self, sb_src: str, host_dst: str) -> dict:
        try:
            src = self._abs(sb_src)
            chunks = self.client.file.download_file(path=src)
            data = b"".join(chunks)
            Path(host_dst).parent.mkdir(parents=True, exist_ok=True)
            Path(host_dst).write_bytes(data)
            res = _ok(stdout=f"sandbox:{src} -> {host_dst} ({len(data)} bytes)", bytes=len(data))
        except Exception as e:
            res = _err(e)
        self._emit("schleuse.out", {"sb_src": self._abs(sb_src), "host_dst": host_dst,
                                    "bytes": res.get("bytes", 0), "success": res["success"]})
        return res

    def upload_dir(self, host_src: str, sb_dst: str) -> dict:
        """Whole directory IN: tar.gz on host -> upload -> extract in sandbox."""
        import tarfile
        import tempfile
        src = Path(host_src)
        if not src.is_dir():
            return _err(f"not a directory: {host_src}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tar_path = tmp.name
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(src, arcname=".")
            dst = self._abs(sb_dst)
            sb_tar = f"{dst.rstrip('/')}.__in.tar.gz"
            up = self.upload(tar_path, sb_tar)
            os.unlink(tar_path)
            if not up["success"]:
                return up
            r = self.exec(f"mkdir -p '{dst}' && tar -xzf '{sb_tar}' -C '{dst}' && rm -f '{sb_tar}' "
                          f"&& find '{dst}' -type f | wc -l")
            if r["success"]:
                r["stdout"] = f"{host_src} -> sandbox:{dst} ({r['stdout'].strip()} files)"
            return r
        except Exception as e:
            return _err(e)

    def download_dir(self, sb_src: str, host_dst_zip: str) -> dict:
        """Whole directory OUT: zip in sandbox -> download single .zip to host."""
        src = self._abs(sb_src)
        sb_zip = f"{src.rstrip('/')}.__out.zip"
        r = self.exec(f"cd '{src}' && (zip -qr '{sb_zip}' . || "
                      f"(tar -czf '{sb_zip}' . && echo tar-fallback))")
        if not r["success"]:
            return r
        if not host_dst_zip.endswith(".zip"):
            host_dst_zip += ".zip"
        dl = self.download(sb_zip, host_dst_zip)
        self.exec(f"rm -f '{sb_zip}'")
        return dl

    # -- browser ------------------------------------------------------------------
    def browser_screenshot(self) -> dict:
        try:
            data = b"".join(self.client.browser.screenshot())
            import base64 as _b64
            # P7: save to host for debugging AND return base64 for agent vision
            out = STATE_DIR / "screens" / f"{int(time.time())}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            b64 = _b64.b64encode(data).decode()
            return _ok(stdout=f"data:image/png;base64,{b64}",
                       path=str(out), bytes=len(data))
        except Exception as e:
            return _err(e)

    def browser_action(self, action: dict) -> dict:
        try:
            r = self.client.browser.execute_action(request=action)
            # P4: propagate SDK success=False
            if hasattr(r, "success") and r.success is False:
                return _err(getattr(r, "message", None) or "browser action failed")
            raw = getattr(r, "data", r)
            # P8: convert Pydantic models to dict
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()
            # P3: serialize as JSON for structured data
            if isinstance(raw, (dict, list)):
                stdout = json.dumps(raw, ensure_ascii=False, default=str)
            else:
                stdout = str(raw) if raw is not None else ""
            return _ok(stdout=stdout)
        except Exception as e:
            return _err(e)

    def browser_info(self) -> dict:
        try:
            r = self.client.browser.get_info()
            # P4: propagate SDK success=False
            if hasattr(r, "success") and r.success is False:
                return _err(getattr(r, "message", None) or "browser info failed")
            raw = getattr(r, "data", r)
            # P8: convert Pydantic models to dict
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()
            elif hasattr(raw, "dict") and callable(getattr(raw, "dict", None)):
                try:
                    raw = raw.dict()
                except Exception:
                    pass
            # P3: serialize as JSON for structured data
            if isinstance(raw, (dict, list)):
                stdout = json.dumps(raw, ensure_ascii=False, default=str)
            else:
                stdout = str(raw) if raw is not None else ""
            return _ok(stdout=stdout)
        except Exception as e:
            return _err(e)


# =============================================================================
# PHASE 2c — REGISTRY: one container per AGENT, workdir per SESSION
# =============================================================================

class SandboxRegistry:
    """
    agent_name -> one running local container (DockerLifecycle), shared by all
    sessions of that agent. Each session binds its own AIOSandboxBackend with
    workdir /work/<session_id>.

    connect_remote() overrides the local container for one session via 2-part key.
    """

    _instances: dict[str, "SandboxRegistry"] = {}

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.lifecycle = DockerLifecycle(agent_name)
        self._base_url: str | None = None

    @classmethod
    def for_agent(cls, agent_name: str) -> "SandboxRegistry":
        if agent_name not in cls._instances:
            cls._instances[agent_name] = cls(agent_name)
        return cls._instances[agent_name]

    def backend_for_session(self, session_id: str, obs_emit: ObsEmitter | None = None) -> AIOSandboxBackend | dict:
        if self._base_url is None:
            r = self.lifecycle.ensure()
            if not r["success"]:
                return r  # error dict — caller surfaces it to the agent
            self._base_url = r["base_url"]
            self._wait_ready(self._base_url)
        be = AIOSandboxBackend(
            base_url=self._base_url,
            workdir=f"/work/{session_id}",
            obs_emit=obs_emit,
            label=f"local:{self.agent_name}",
        )
        be.start()
        return be

    @staticmethod
    def connect_remote(key: str, obs_emit: ObsEmitter | None = None) -> AIOSandboxBackend:
        tgt = resolve_remote_key(key)
        be = AIOSandboxBackend(
            base_url=tgt.base_url, workdir=tgt.workdir, token=tgt.token,
            obs_emit=obs_emit, label=f"remote:{tgt.base_url}",
        )
        be.start()
        return be

    @staticmethod
    def _wait_ready(base_url: str, timeout: float = 60.0) -> None:
        import urllib.request
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                urllib.request.urlopen(base_url + "/v1/sandbox", timeout=2)
                return
            except Exception:
                time.sleep(1.0)

    def shutdown(self) -> dict:
        self._base_url = None
        return self.lifecycle.stop()
