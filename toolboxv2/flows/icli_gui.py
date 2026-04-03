# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "customtkinter==5.2.2",
#   "pyzmq>=25.0",
# ]
# ///

"""
icli_gui — Minimal launcher for ToolBoxV2 icli
Sessions / Agents / Projects + direct CLI spawn
Run: uv run icli_gui.py
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import customtkinter as ctk

# ─── Colors ───────────────────────────────────────────────────────────────────
C = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#21262d",
    "dim":     "#484f58",
    "text":    "#c9d1d9",
    "bright":  "#f0f6fc",
    "cyan":    "#39c5cf",
    "green":   "#3fb950",
    "amber":   "#d29922",
    "red":     "#f85149",
    "accent":  "#1f6feb",
}

POLL_MS = 80

# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentInfo:
    name: str
    persona: str = "assistant"
    sessions: list = field(default_factory=lambda: ["default"])


def _find_tb_appdata() -> Optional[Path]:
    try:
        from toolboxv2 import get_app
        p = Path(get_app("icli-gui-probe").appdata)
        return p if p.exists() else None
    except Exception:
        pass
    candidates = []
    if sys.platform == "win32":
        candidates = [Path(os.environ.get("APPDATA", "~")).expanduser() / "ToolBoxV2"]
    elif sys.platform == "darwin":
        candidates = [Path.home() / "Library" / "Application Support" / "ToolBoxV2"]
    else:
        candidates = [Path.home() / ".local" / "share" / "ToolBoxV2", Path.home() / ".ToolBoxV2"]
    for p in candidates:
        if (p / "icli").exists():
            return p
    return None


def load_state(appdata: Optional[Path]) -> dict:
    if not appdata:
        return {}
    sf = appdata / "icli" / "isaa_host_state.json"
    try:
        return json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
    except Exception:
        return {}


def load_agents(appdata: Optional[Path], state: dict) -> list[AgentInfo]:
    agents: list[AgentInfo] = []
    registry = state.get("agent_registry", {})
    for name, info in registry.items():
        agents.append(AgentInfo(name=name, persona=info.get("persona", "assistant"),
                                sessions=_scan_sessions(appdata, name)))
    # Scan Agents dir for any not in registry
    if appdata:
        agents_dir = appdata / "Agents"
        if agents_dir.exists():
            known = {a.name for a in agents}
            for d in sorted(agents_dir.iterdir()):
                if d.is_dir() and d.name not in known:
                    agents.append(AgentInfo(name=d.name, sessions=_scan_sessions(appdata, d.name)))
    if not agents:
        agents.append(AgentInfo(name="self", persona="Host Administrator"))
    return agents


def _scan_sessions(appdata: Optional[Path], agent_name: str) -> list[str]:
    sessions: set[str] = set()
    if appdata:
        agent_dir = appdata / "Agents" / agent_name
        for sub in ["sessions", "history"]:
            p = agent_dir / sub
            if p.exists():
                for f in p.iterdir():
                    if f.stem and not f.stem.startswith("."):
                        sessions.add(f.stem)
    result = sorted(sessions)
    if "default" not in result:
        result = ["default"] + result
    return result or ["default"]


GUI_CONFIG = Path.home() / ".icli_gui_config.json"


def load_projects() -> list[str]:
    try:
        return json.loads(GUI_CONFIG.read_text(encoding="utf-8")).get("projects", [])
    except Exception:
        return []


def save_projects(p: list[str]):
    try:
        GUI_CONFIG.write_text(json.dumps({"projects": p}, indent=2), encoding="utf-8")
    except Exception:
        pass


# ─── Terminal spawn ───────────────────────────────────────────────────────────

def _find_terminal() -> Optional[list[str]]:
    if sys.platform == "win32":
        for t, args in [("wt", ["new-tab", "--"]), ("powershell", ["/k"]), ("cmd", ["/k"])]:
            if shutil.which(t):
                return [t] + args
        return None
    if sys.platform == "darwin":
        return ["open", "-a", "Terminal", "--args"]
    for t, args in [
        ("kitty", ["-e"]), ("wezterm", ["start", "--"]), ("alacritty", ["-e"]),
        ("gnome-terminal", ["--"]), ("xfce4-terminal", ["-e"]),
        ("konsole", ["-e"]), ("xterm", ["-e"]),
    ]:
        if shutil.which(t):
            return [t] + args
    return None


def spawn_icli(agent: str, session: str, query: str = "",
               gui_mode: bool = False, cwd: Optional[str] = None) -> Optional[subprocess.Popen]:
    py = sys.executable

    cmd = [py, "-m", "toolboxv2.flows.icli", "--agent", agent, "--session", session]
    print(" ".join(cmd))
    if gui_mode:
        cmd += ["--gui", "--gui-session", session or str(uuid.uuid4())[:8]]
    if query.strip():
        cmd.append(query.strip())

    terminal = _find_terminal()
    kw: dict = {"start_new_session": True}
    if cwd and Path(cwd).is_dir():
        kw["cwd"] = cwd

    try:
        if terminal is None:
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            return subprocess.Popen(cmd, **kw)
        if sys.platform == "darwin":
            return subprocess.Popen(["open", "-a", "Terminal", "--args"] + cmd, **kw)
        return subprocess.Popen(terminal + cmd, **kw)
    except Exception:
        return None


# ─── Broker probe ─────────────────────────────────────────────────────────────

import socket

def probe_broker(host: str = "127.0.0.1", port: int = 5555, timeout: float = 0.3) -> bool:
    """TCP connect check — kein ZMQ Protokoll, keine EventType Validierung."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


# ─── Background worker ────────────────────────────────────────────────────────

class BgWorker(threading.Thread):
    def __init__(self, q: queue.Queue, appdata: Optional[Path]):
        super().__init__(daemon=True, name="icli-gui-bg")
        self.q = q
        self.appdata = appdata
        self._stop = threading.Event()

    def run(self):
        tick = 0
        while not self._stop.is_set():
            if tick % 5 == 0:
                self.q.put_nowait({"type": "broker", "online": probe_broker()})
            if tick % 30 == 0:
                state = load_state(self.appdata)
                self.q.put_nowait({"type": "state", "agents": load_agents(self.appdata, state), "state": state})
            tick += 1
            time.sleep(1)

    def stop(self):
        self._stop.set()


# ─── Widget helpers ───────────────────────────────────────────────────────────

FONT = "JetBrains Mono"


def lbl(parent, text, size=11, weight="normal", color=None, **kw):
    return ctk.CTkLabel(parent, text=text,
                        font=ctk.CTkFont(family=FONT, size=size, weight=weight),
                        text_color=color or C["text"], **kw)


def btn(parent, text, cmd, fg=None, hover=None, width=120, **kw):
    return ctk.CTkButton(parent, text=text, command=cmd,
                         fg_color=fg or C["accent"], hover_color=hover or C["cyan"],
                         text_color=C["bright"],
                         font=ctk.CTkFont(family=FONT, size=11, weight="bold"),
                         corner_radius=4, width=width, **kw)


def sep(parent):
    return ctk.CTkFrame(parent, height=1, fg_color=C["border"])


def frm(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=C["surface"], corner_radius=6, **kw)


# ─── Main app ─────────────────────────────────────────────────────────────────

class IcliLauncher(ctk.CTk):
    def __init__(self, appdata: Optional[Path]):
        super().__init__()
        self.appdata = appdata
        self._agents: list[AgentInfo] = []
        self._projects: list[str] = load_projects()
        self._broker_online = False
        self._q: queue.Queue = queue.Queue()
        self._sel_agent: Optional[str] = None
        self._sel_session: Optional[str] = None
        self._sel_project: Optional[str] = None

        ctk.set_appearance_mode("dark")
        self.title("icli launcher")
        self.geometry("900x640")
        self.minsize(740, 500)
        self.configure(fg_color=C["bg"])

        self._build()
        self._load()

        self._worker = BgWorker(self._q, appdata)
        self._worker.start()
        self.after(POLL_MS, self._poll)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Top bar
        top = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0, height=42)
        top.pack(fill="x")
        top.pack_propagate(False)
        lbl(top, "  ◯  icli", size=14, weight="bold", color=C["cyan"]).pack(side="left", pady=8)
        self._dot = lbl(top, "●", size=13, color=C["red"])
        self._dot.pack(side="right", padx=6)
        lbl(top, "zmq", size=9, color=C["dim"]).pack(side="right")
        if appdata:
            lbl(top, str(appdata)[-38:], size=9, color=C["dim"]).pack(side="right", padx=12)

        # Body: 3 columns
        body = ctk.CTkFrame(self, fg_color=C["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=6)
        body.columnconfigure(0, weight=1, minsize=155)
        body.columnconfigure(1, weight=1, minsize=155)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_agents(body)
        self._build_sessions(body)
        self._build_actions(body)
        self._build_projects()

    def _col_header(self, parent, text):
        h = ctk.CTkFrame(parent, fg_color=C["border"], height=26, corner_radius=3)
        h.pack(fill="x", pady=(0, 3))
        h.pack_propagate(False)
        lbl(h, f"  {text}", size=9, weight="bold", color=C["dim"]).pack(side="left", pady=3)

    # ── Agents ────────────────────────────────────────────────────────────────

    def _build_agents(self, parent):
        f = frm(parent)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._col_header(f, "AGENTS")
        self._agent_scroll = ctk.CTkScrollableFrame(f, fg_color=C["bg"], corner_radius=3)
        self._agent_scroll.pack(fill="both", expand=True, padx=3, pady=(0, 3))

    def _render_agents(self):
        for w in self._agent_scroll.winfo_children():
            w.destroy()
        for a in self._agents:
            sel = a.name == self._sel_agent
            bg = C["accent"] if sel else C["surface"]
            row = ctk.CTkFrame(self._agent_scroll, fg_color=bg, corner_radius=4, height=48)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)
            lbl(row, f"  {a.name}", size=11, weight="bold" if sel else "normal",
                color=C["bright"] if sel else C["text"]).pack(anchor="w", pady=(5, 0))
            lbl(row, f"  {a.persona[:30]}", size=9, color=C["cyan"] if sel else C["dim"]).pack(anchor="w")
            for w in [row] + list(row.winfo_children()):
                w.bind("<Button-1>", lambda e, n=a.name: self._sel_agent_cb(n))

    # ── Sessions ──────────────────────────────────────────────────────────────

    def _build_sessions(self, parent):
        f = frm(parent)
        f.grid(row=0, column=1, sticky="nsew", padx=(0, 5))
        self._col_header(f, "SESSIONS")
        self._session_scroll = ctk.CTkScrollableFrame(f, fg_color=C["bg"], corner_radius=3)
        self._session_scroll.pack(fill="both", expand=True, padx=3, pady=(0, 3))

    def _render_sessions(self):
        for w in self._session_scroll.winfo_children():
            w.destroy()
        agent = self._cur_agent()
        if not agent:
            lbl(self._session_scroll, "← select agent", size=10, color=C["dim"]).pack(pady=12)
            return
        for sid in agent.sessions:
            sel = sid == self._sel_session
            bg = C["accent"] if sel else C["surface"]
            row = ctk.CTkFrame(self._session_scroll, fg_color=bg, corner_radius=4, height=34)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)
            icon = "▸ " if sel else "  "
            lbl(row, f"{icon}{sid}", size=11, color=C["bright"] if sel else C["text"]).pack(anchor="w", padx=4, pady=5)
            for w in [row] + list(row.winfo_children()):
                w.bind("<Button-1>", lambda e, s=sid: self._sel_session_cb(s))
        # New session
        new_row = ctk.CTkFrame(self._session_scroll, fg_color=C["bg"], height=30)
        new_row.pack(fill="x", pady=(6, 0))
        new_row.pack_propagate(False)
        lbl(new_row, "  + new…", size=10, color=C["accent"]).pack(side="left", pady=4)
        for w in [new_row] + list(new_row.winfo_children()):
            w.bind("<Button-1>", lambda e: self._new_session())

    # ── Actions ───────────────────────────────────────────────────────────────

    def _build_actions(self, parent):
        f = frm(parent)
        f.grid(row=0, column=2, sticky="nsew")
        self._col_header(f, "LAUNCH")

        inner = ctk.CTkFrame(f, fg_color=C["bg"], corner_radius=3)
        inner.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        self._sel_lbl = lbl(inner, "no selection", size=10, color=C["dim"])
        self._sel_lbl.pack(anchor="w", padx=8, pady=(8, 2))

        sep(inner).pack(fill="x", padx=6, pady=5)

        # Prompt
        lbl(inner, "  initial prompt  (optional)", size=9, color=C["dim"]).pack(anchor="w", padx=8)
        self._prompt = ctk.CTkTextbox(
            inner, height=80, corner_radius=4,
            fg_color=C["surface"], text_color=C["text"],
            font=ctk.CTkFont(family=FONT, size=11),
            border_color=C["border"], border_width=1,
        )
        self._prompt.pack(fill="x", padx=8, pady=(3, 8))

        # Working dir
        lbl(inner, "  working dir  (optional)", size=9, color=C["dim"]).pack(anchor="w", padx=8)
        wd_row = ctk.CTkFrame(inner, fg_color=C["bg"])
        wd_row.pack(fill="x", padx=8, pady=(2, 8))
        self._wd_var = ctk.StringVar()
        self._wd_entry = ctk.CTkEntry(
            wd_row, textvariable=self._wd_var,
            fg_color=C["surface"], text_color=C["text"],
            font=ctk.CTkFont(family=FONT, size=10),
            border_color=C["border"], border_width=1,
            placeholder_text="~/projects/myapp",
        )
        self._wd_entry.pack(side="left", fill="x", expand=True)
        btn(wd_row, "← project", self._use_project,
            fg=C["surface"], hover=C["border"], width=80).pack(side="right", padx=(4, 0))

        sep(inner).pack(fill="x", padx=6, pady=3)

        # Buttons
        br = ctk.CTkFrame(inner, fg_color=C["bg"])
        br.pack(fill="x", padx=8, pady=6)
        btn(br, "▶  Start in Terminal", self._launch,
            fg=C["green"], hover=C["cyan"], width=175).pack(side="left")
        btn(br, "◎  GUI Mode", self._launch_gui,
            fg=C["accent"], hover=C["cyan"], width=105).pack(side="left", padx=(8, 0))

        sep(inner).pack(fill="x", padx=6, pady=(6, 3))

        # Log
        self._log = ctk.CTkTextbox(
            inner, height=110, corner_radius=3, state="disabled",
            fg_color=C["surface"], text_color=C["dim"],
            font=ctk.CTkFont(family=FONT, size=9),
        )
        self._log.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    # ── Projects bar ──────────────────────────────────────────────────────────

    def _build_projects(self):
        bar = ctk.CTkFrame(self, fg_color=C["surface"], height=36, corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        lbl(bar, "  projects:", size=9, color=C["dim"]).pack(side="left", pady=8)
        self._proj_bar = ctk.CTkFrame(bar, fg_color=C["surface"])
        self._proj_bar.pack(side="left", fill="x", expand=True)
        btn(bar, "+ add", self._add_project,
            fg=C["surface"], hover=C["border"], width=55).pack(side="right", padx=(0, 6), pady=5)
        self._render_projects()

    def _render_projects(self):
        for w in self._proj_bar.winfo_children():
            w.destroy()
        for p in self._projects[:8]:
            name = Path(p).name or p
            sel = p == self._sel_project
            ctk.CTkButton(
                self._proj_bar, text=name[:16], width=76,
                fg_color=C["accent"] if sel else C["border"],
                hover_color=C["cyan"],
                text_color=C["bright"],
                font=ctk.CTkFont(family=FONT, size=9),
                corner_radius=3, height=24,
                command=lambda path=p: self._pick_project(path),
            ).pack(side="left", padx=2, pady=6)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _sel_agent_cb(self, name: str):
        self._sel_agent = name
        self._sel_session = None
        self._render_agents()
        self._render_sessions()
        self._update_sel()

    def _sel_session_cb(self, sid: str):
        self._sel_session = sid
        self._render_sessions()
        self._update_sel()

    def _pick_project(self, path: str):
        self._sel_project = path
        self._wd_var.set(path)
        self._render_projects()

    def _use_project(self):
        if self._sel_project:
            self._wd_var.set(self._sel_project)

    def _new_session(self):
        d = ctk.CTkInputDialog(text="Session ID:", title="New Session")
        sid = (d.get_input() or "").strip()
        if sid:
            agent = self._cur_agent()
            if agent and sid not in agent.sessions:
                agent.sessions.insert(1, sid)
            self._sel_session = sid
            self._render_sessions()
            self._update_sel()

    def _add_project(self):
        d = ctk.CTkInputDialog(text="Directory path:", title="Add Project")
        p = (d.get_input() or "").strip()
        if p:
            path = str(Path(p).expanduser().resolve())
            if path not in self._projects:
                self._projects.insert(0, path)
                save_projects(self._projects)
                self._render_projects()
                self._log_write(f"+ {path}")

    def _update_sel(self):
        a = self._sel_agent or "—"
        s = self._sel_session or "—"
        self._sel_lbl.configure(
            text=f"  agent: {a}   session: {s}",
            text_color=C["cyan"] if self._sel_agent else C["dim"],
        )

    def _args(self) -> tuple[str, str, str, Optional[str]]:
        agent   = self._sel_agent or "self"
        session = self._sel_session or "default"
        prompt  = self._prompt.get("1.0", "end").strip()
        wd      = self._wd_var.get().strip() or None
        if wd:
            wd = str(Path(wd).expanduser())
        return agent, session, prompt, wd

    def _launch(self):
        a, s, q, wd = self._args()
        proc = spawn_icli(a, s, q, gui_mode=False, cwd=wd)
        self._log_write(f"▶ {a}@{s}  pid={proc.pid if proc else '?'}")

    def _launch_gui(self):
        a, s, q, wd = self._args()
        proc = spawn_icli(a, s, q, gui_mode=True, cwd=wd)
        self._log_write(f"◎ {a}@{s}  pid={proc.pid if proc else '?'}")

    def _log_write(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", f"{msg}\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _cur_agent(self) -> Optional[AgentInfo]:
        return next((a for a in self._agents if a.name == self._sel_agent), None)

    # ── Init ──────────────────────────────────────────────────────────────────

    def _load(self):
        state = load_state(self.appdata)
        self._agents = load_agents(self.appdata, state)
        active_agent   = state.get("active_agent", "self")
        active_session = state.get("active_session", "default")
        if any(a.name == active_agent for a in self._agents):
            self._sel_agent   = active_agent
            self._sel_session = active_session
        self._render_agents()
        self._render_sessions()
        self._update_sel()
        if not self.appdata:
            self._log_write("⚠ TB appdata not found — set working dir manually")

    # ── Poll ──────────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                item = self._q.get_nowait()
                t = item.get("type")
                if t == "broker":
                    online = item["online"]
                    if online != self._broker_online:
                        self._broker_online = online
                        self._dot.configure(text_color=C["green"] if online else C["red"])
                elif t == "state":
                    self._agents = item["agents"]
                    self._render_agents()
                    self._render_sessions()
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll)

    def on_close(self):
        self._worker.stop()
        self.destroy()


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    appdata = _find_tb_appdata()
    ctk.set_appearance_mode("dark")
    app = IcliLauncher(appdata)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
