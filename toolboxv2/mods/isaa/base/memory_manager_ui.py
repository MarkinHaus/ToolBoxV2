"""
Memory Manager UI — CustomTkinter 5.2.2 GUI for AISemanticMemory V2

Manages HybridMemoryStore instances via a dark-themed sidebar + panel layout.

Usage:
    from toolboxv2.mods.isaa.base.memory_manager_ui import launch
    launch()

Or directly:
    python -m toolboxv2.mods.isaa.base.memory_manager_ui
"""
from __future__ import annotations

import json
import logging
import queue
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── customtkinter 5.2.2 guard ─────────────────────────────────────────────────
try:
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    _CTK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CTK_AVAILABLE = False

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════
BG          = "#08080d"
BG_SIDEBAR  = "#0d0d14"
BG_CARD     = "#111119"
BG_HOVER    = "#1a1a28"
ACCENT      = "#6c63ff"
GREEN       = "#00d4a0"
RED         = "#ff4757"
AMBER       = "#ffa502"
FG          = "#e8e8f0"
FG_DIM      = "#6b6b80"
FG_HEAD     = "#ffffff"
BORDER      = "#1e1e2e"

SIDEBAR_W        = 280
SIDEBAR_W_MINI   = 38
MAX_ENTRIES_EST  = 5_000   # estimate for fill-bar 100 %

FONT_BODY   = ("IBM Plex Sans",  11)
FONT_MONO   = ("IBM Plex Mono",  10)
FONT_SMALL  = ("IBM Plex Sans",   9)
FONT_HEAD   = ("IBM Plex Sans",  13, "bold")
FONT_BIG    = ("IBM Plex Sans",  15, "bold")


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════

def _darken(hex_color: str, f: float = 0.72) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(int(r * f), int(g * f), int(b * f))


def _btn(
    parent: Any,
    text: str,
    command: Any,
    color: str = ACCENT,
    width: int = 120,
    **kw,
) -> "ctk.CTkButton":
    return ctk.CTkButton(
        parent, text=text, command=command,
        fg_color=color, hover_color=_darken(color),
        text_color=FG_HEAD, corner_radius=6,
        font=FONT_BODY, width=width, **kw,
    )


def _style_treeview() -> None:
    """Configure dark ttk.Treeview — must be called after Tk root exists."""
    s = ttk.Style()
    s.theme_use("default")
    s.configure(
        "Dark.Treeview",
        background=BG_CARD, foreground=FG, rowheight=26,
        fieldbackground=BG_CARD, borderwidth=0, font=FONT_MONO,
    )
    s.configure(
        "Dark.Treeview.Heading",
        background=BG_SIDEBAR, foreground=FG_DIM,
        borderwidth=0, relief="flat", font=FONT_SMALL,
    )
    s.map(
        "Dark.Treeview",
        background=[("selected", ACCENT)],
        foreground=[("selected", FG_HEAD)],
    )
    s.configure(
        "Dark.Vertical.TScrollbar",
        background=BG_CARD, troughcolor=BG_SIDEBAR,
        borderwidth=0, arrowsize=0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    semantic_memory: Any = None
    active_panel: str = "welcome"
    selected_space: Optional[str] = None
    actors: dict = field(default_factory=dict)       # name → ActorRunner
    sidebar_collapsed: bool = False
    sections_open: dict = field(default_factory=lambda: {
        "memories": True, "actors": False, "graph": False,
    })


class ActorRunner:
    """Thread-wrapper for MemoryKnowledgeActor.start_analysis_loop."""

    def __init__(
        self,
        name: str,
        spaces: list,
        task: str,
        max_iter: int = 10,
        agent_name: str = "summary",
    ) -> None:
        self.name       = name
        self.spaces     = spaces
        self.task       = task
        self.max_iter   = max_iter
        self.agent_name = agent_name
        self.status     = "pending"   # pending|running|done|error|stopped
        self.history: list = []
        self.out_q: queue.Queue = queue.Queue()

    def start(self, semantic_memory: Any) -> None:
        from toolboxv2 import get_app

        space = self.spaces[0] if self.spaces else "default"
        runner = self  # capture

        async def _run() -> None:
            from toolboxv2.mods.isaa.base.MemoryKnowledgeActor import MemoryKnowledgeActor
            runner.status = "running"
            try:
                mka = MemoryKnowledgeActor(
                    memory=semantic_memory, space_name=space
                )
                hist = await mka.start_analysis_loop(
                    user_task=runner.task,
                    max_iterations=runner.max_iter,
                    agent_name=runner.agent_name,
                )
                runner.history = hist
                runner.status  = "done"
                runner.out_q.put({"type": "done", "history": hist})
            except Exception as exc:
                runner.status = "error"
                runner.out_q.put({"type": "error", "msg": str(exc)})

        get_app().run_bg_task_advanced(_run)

    def stop(self) -> None:
        """Signal stop — loop can't be cancelled mid-await; status signals UI."""
        self.status = "stopped"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class StatRow(ctk.CTkFrame):
    """Label | CTkProgressBar (color by fill %) | count."""

    def __init__(self, parent: Any, label: str, value: int, total: int, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        ratio = min(value / max(total, 1), 1.0)
        color = GREEN if ratio < 0.7 else (AMBER if ratio < 0.9 else RED)

        ctk.CTkLabel(
            self, text=label, font=FONT_BODY, text_color=FG,
            width=150, anchor="w",
        ).pack(side="left")

        bar = ctk.CTkProgressBar(
            self, width=100, height=7, corner_radius=3,
            fg_color=BG_HOVER, progress_color=color,
        )
        bar.set(ratio)
        bar.pack(side="left", padx=(8, 8))

        ctk.CTkLabel(
            self, text=str(value), font=FONT_MONO,
            text_color=FG_DIM, width=60, anchor="e",
        ).pack(side="left")


# ══════════════════════════════════════════════════════════════════════════════
# PANELS
# ══════════════════════════════════════════════════════════════════════════════

class _PanelBase(ctk.CTkFrame):
    """Common base: dark background, standard header row."""

    def __init__(self, parent: Any, app: "MemoryManagerApp") -> None:
        super().__init__(parent, fg_color=BG, corner_radius=0)
        self.app = app

    def _header(self, title: str, right_widget: Optional[Any] = None) -> ctk.CTkFrame:
        bar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0, height=48)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkLabel(
            bar, text=title, font=FONT_HEAD, text_color=FG_HEAD, anchor="w",
        ).pack(side="left", padx=20, pady=12)
        if right_widget:
            right_widget(bar)
        return bar


# ─── Welcome ────────────────────────────────────────────────────────────────

class WelcomePanel(_PanelBase):

    def __init__(self, parent: Any, app: "MemoryManagerApp") -> None:
        super().__init__(parent, app)
        self._build()

    def _build(self) -> None:
        ctk.CTkLabel(self, text="Memory Manager", font=FONT_BIG,
                     text_color=FG_HEAD).pack(pady=(44, 4))
        ctk.CTkLabel(self, text="AISemanticMemory V2 — HybridMemoryStore",
                     font=FONT_SMALL, text_color=FG_DIM).pack(pady=(0, 32))

        mem = self.app.app_state.semantic_memory
        if mem is None:
            ctk.CTkLabel(
                self,
                text="⚠  Kein AISemanticMemory verbunden.\n"
                     "Stelle sicher, dass get_app() erreichbar ist.",
                text_color=AMBER, font=FONT_BODY, justify="center",
            ).pack(pady=16)
            return

        spaces = mem.memories
        total_active = 0
        for store in spaces.values():
            try:
                total_active += store.stats().get("active", 0)
            except Exception:
                pass

        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10)
        card.pack(padx=80, pady=8, fill="x")
        for label, val in [
            ("Memory Spaces", str(len(spaces))),
            ("Aktive Einträge (gesamt)", str(total_active)),
        ]:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=7)
            ctk.CTkLabel(row, text=label, font=FONT_BODY,
                         text_color=FG, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=val, font=FONT_MONO,
                         text_color=ACCENT, anchor="e").pack(side="right")

        if spaces:
            ctk.CTkLabel(self, text="Spaces öffnen", font=FONT_HEAD,
                         text_color=FG_HEAD).pack(pady=(24, 8))
            for name in list(spaces)[:10]:
                _btn(
                    self, f"→  {name}",
                    command=lambda n=name: self.app.show_panel("memory_detail", space=n),
                    color=BG_CARD, width=280,
                ).pack(pady=2)


# ─── Memory Detail  (F1 + F2) ───────────────────────────────────────────────

class MemoryDetailPanel(_PanelBase):
    """F1 — fill-stats + entry list.  F2 — chat-session delete."""

    def __init__(self, parent: Any, app: "MemoryManagerApp", space_name: str) -> None:
        super().__init__(parent, app)
        self.space_name = space_name
        self._entry_map: dict = {}    # treeview iid → sqlite entry_id
        self._build()

    # ── build ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        mem = self.app.app_state.semantic_memory
        if mem is None or self.space_name not in mem.memories:
            ctk.CTkLabel(
                self, text=f"Space '{self.space_name}' nicht gefunden.",
                text_color=RED, font=FONT_BODY,
            ).pack(pady=60)
            return

        store = mem.memories[self.space_name]
        try:
            stats = store.stats()
        except Exception as exc:
            ctk.CTkLabel(self, text=f"stats() Fehler: {exc}",
                         text_color=RED).pack(pady=60)
            return

        # Header
        self._header(
            self.space_name,
            right_widget=lambda bar: _btn(
                bar, "✕  Space entfernen",
                command=self._remove_space, color=RED, width=150,
            ).pack(side="right", padx=12, pady=8),
        )

        # Scrollable body
        scroll = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        scroll.pack(fill="both", expand=True)

        self._build_stats(scroll, stats)
        self._build_actions(scroll)
        self._build_entry_list(scroll, store)

    def _build_stats(self, parent: Any, stats: dict) -> None:
        ctk.CTkLabel(parent, text="Fill Level", font=FONT_HEAD,
                     text_color=FG_HEAD, anchor="w").pack(
            fill="x", padx=20, pady=(20, 6))

        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10)
        card.pack(fill="x", padx=16, pady=(0, 14))

        est_max = max(stats.get("total", 0), MAX_ENTRIES_EST)
        rows = [
            ("Aktive Einträge",   stats.get("active",     0), est_max),
            ("Gesamt Einträge",   stats.get("total",      0), est_max),
            ("Entities",          stats.get("entities",   0), max(stats.get("entities",   0) * 2, 50)),
            ("Relations",         stats.get("relations",  0), max(stats.get("relations",  0) * 2, 50)),
            ("Concepts",          stats.get("concepts",   0), max(stats.get("concepts",   0) * 2, 200)),
            ("FAISS Vektoren",    stats.get("faiss_size", 0), est_max),
        ]
        for label, val, mx in rows:
            StatRow(card, label, val, mx).pack(fill="x", padx=16, pady=4)

        meta = ctk.CTkFrame(card, fg_color="transparent")
        meta.pack(fill="x", padx=16, pady=(4, 12))
        for txt in [
            f"Space: {stats.get('space', self.space_name)}",
            f"Embedding dim: {stats.get('dim', '?')}",
            f"TTL entries: {stats.get('with_ttl', 0)}",
        ]:
            ctk.CTkLabel(meta, text=txt, font=FONT_SMALL,
                         text_color=FG_DIM, anchor="w").pack(fill="x", pady=1)

    def _build_actions(self, parent: Any) -> None:
        """F2 — Chat-Session delete buttons."""
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10)
        card.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(card, text="Chat Sessions löschen", font=FONT_HEAD,
                     text_color=FG_HEAD, anchor="w").pack(
            fill="x", padx=16, pady=(12, 8))

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(0, 12))

        _btn(row, "🗑  Soft Delete",
             command=self._clear_chat_soft, color=AMBER, width=180,
             ).pack(side="left", padx=(0, 10))
        _btn(row, "🗑  Hard Delete",
             command=self._clear_chat_hard, color=RED, width=180,
             ).pack(side="left")

    def _build_entry_list(self, parent: Any, store: Any) -> None:
        ctk.CTkLabel(parent, text="Einträge (letzte 50)", font=FONT_HEAD,
                     text_color=FG_HEAD, anchor="w").pack(
            fill="x", padx=20, pady=(4, 6))

        tree_card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=10)
        tree_card.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        cols = ("type", "role", "preview")
        self._tree = ttk.Treeview(
            tree_card, columns=cols, show="headings",
            style="Dark.Treeview", selectmode="extended", height=14,
        )
        self._tree.heading("type",    text="Type")
        self._tree.heading("role",    text="Role")
        self._tree.heading("preview", text="Content")
        self._tree.column("type",    width=70,  stretch=False)
        self._tree.column("role",    width=80,  stretch=False)
        self._tree.column("preview", width=600, stretch=True)

        vsb = ttk.Scrollbar(tree_card, orient="vertical",
                            command=self._tree.yview,
                            style="Dark.Vertical.TScrollbar")
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        vsb.pack(side="right", fill="y", pady=2)

        # Context menu
        self._ctx = tk.Menu(
            self._tree, tearoff=0,
            bg=BG_CARD, fg=FG, activebackground=ACCENT,
            activeforeground=FG_HEAD, bd=0,
        )
        self._ctx.add_command(label="Soft Delete",
                              command=lambda: self._delete_selected(hard=False))
        self._ctx.add_command(label="Hard Delete",
                              command=lambda: self._delete_selected(hard=True))
        self._ctx.add_separator()
        self._ctx.add_command(label="Inhalt kopieren", command=self._copy_content)
        self._tree.bind("<Button-3>", self._show_ctx)

        self._load_entries(store)

        # Buttons under list
        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(2, 12))
        _btn(btn_row, "Soft Delete (Auswahl)",
             command=lambda: self._delete_selected(hard=False),
             color=AMBER, width=200).pack(side="left", padx=(0, 10))
        _btn(btn_row, "Hard Delete (Auswahl)",
             command=lambda: self._delete_selected(hard=True),
             color=RED, width=200).pack(side="left")

    # ── data ─────────────────────────────────────────────────────────────────

    def _load_entries(self, store: Any) -> None:
        self._store = store
        self._tree.delete(*self._tree.get_children())
        self._entry_map.clear()
        try:
            rows = store._exec(
                """SELECT id, content_type, meta_role, content
                   FROM entries
                   WHERE space = ? AND is_active = 1
                   ORDER BY created_at DESC LIMIT 50""",
                (store.space,),
            ).fetchall()
        except Exception as exc:
            logger.error("Entry load: %s", exc)
            return
        for row in rows:
            preview = (row["content"] or "")[:140].replace("\n", " ")
            iid = self._tree.insert(
                "", "end",
                values=(row["content_type"], row["meta_role"] or "—", preview),
            )
            self._entry_map[iid] = row["id"]

    # ── actions ──────────────────────────────────────────────────────────────

    def _clear_chat_soft(self) -> None:
        if not messagebox.askyesno(
            "Chat Sessions soft-deleten",
            f"Alle Chat-Einträge (user/assistant/system/tool)\n"
            f"in '{self.space_name}' deaktivieren?",
        ):
            return
        self.app.status("Lösche Chat-Einträge …")
        store = self._store
        try:
            ids = [
                r["id"] for r in store._exec(
                    """SELECT id FROM entries
                       WHERE space=? AND meta_role IN ('user','assistant','system','tool')
                       AND is_active=1""",
                    (store.space,),
                ).fetchall()
            ]
            for eid in ids:
                store.delete(eid, hard=False)
            self.app.status(
                f"{len(ids)} Chat-Einträge in '{self.space_name}' soft-gelöscht."
            )
            self._load_entries(store)
        except Exception as exc:
            self.app.status(f"Fehler: {exc}", error=True)

    def _clear_chat_hard(self) -> None:
        if not messagebox.askyesno(
            "HARD DELETE",
            f"Alle Chat-Einträge in '{self.space_name}' PERMANENT löschen?\n"
            "Nicht umkehrbar!",
            icon="warning",
        ):
            return
        if not messagebox.askyesno("Bestätigen", "Wirklich permanent löschen?",
                                   icon="warning"):
            return
        self.app.status("Hard-Delete läuft …")
        store = self._store
        try:
            ids = [
                r["id"] for r in store._exec(
                    """SELECT id FROM entries
                       WHERE space=? AND meta_role IN ('user','assistant','system','tool')""",
                    (store.space,),
                ).fetchall()
            ]
            for eid in ids:
                store.delete(eid, hard=True)
            self.app.status(
                f"{len(ids)} Chat-Einträge in '{self.space_name}' permanent gelöscht."
            )
            self._load_entries(store)
        except Exception as exc:
            self.app.status(f"Fehler: {exc}", error=True)

    def _delete_selected(self, hard: bool = False) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        verb = "permanent löschen" if hard else "soft-deleten"
        if not messagebox.askyesno(f"{len(sel)} Einträge {verb}?",
                                   f"{len(sel)} Einträge {verb}?"):
            return
        store = self._store
        for iid in sel:
            eid = self._entry_map.get(iid)
            if eid:
                try:
                    store.delete(eid, hard=hard)
                except Exception as exc:
                    logger.error("Delete %s: %s", eid, exc)
        self._load_entries(store)
        self.app.status(
            f"{len(sel)} Einträge {'gelöscht' if hard else 'soft-gelöscht'}."
        )

    def _remove_space(self) -> None:
        if not messagebox.askyesno(
            "Space entfernen",
            f"Space '{self.space_name}' aus dem Memory-Manager entfernen?\n"
            "(Dateien auf Disk bleiben erhalten)",
        ):
            return
        mem = self.app.app_state.semantic_memory
        store = mem.memories.get(self.space_name)
        if store:
            try:
                store.close()
            except Exception:
                pass
            del mem.memories[self.space_name]
        self.app.status(f"Space '{self.space_name}' entfernt.")
        self.app.show_panel("welcome")
        self.app.refresh_sidebar()

    def _show_ctx(self, event: Any) -> None:
        row = self._tree.identify_row(event.y)
        if row:
            self._tree.selection_set(row)
            self._ctx.post(event.x_root, event.y_root)

    def _copy_content(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        eid = self._entry_map.get(sel[0])
        if not eid:
            return
        try:
            row = self._store._exec(
                "SELECT content FROM entries WHERE id=?", (eid,)
            ).fetchone()
            if row:
                self.clipboard_clear()
                self.clipboard_append(row["content"])
                self.app.status("Inhalt in Zwischenablage kopiert.")
        except Exception as exc:
            logger.error("Copy: %s", exc)


# ─── Actor Panel  (F3) ──────────────────────────────────────────────────────

class ActorPanel(_PanelBase):
    """F3 — Actor konfigurieren, starten, Live-Log."""

    def __init__(
        self,
        parent: Any,
        app: "MemoryManagerApp",
        runner: Optional[ActorRunner] = None,
    ) -> None:
        super().__init__(parent, app)
        self.runner     = runner
        self._poll_id   = None
        self._build()
        if runner and runner.status in ("running",):
            self._start_poll()
        elif runner and runner.history:
            self._replay_history(runner.history)

    # ── layout ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        self._header("Actor")

        body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=272)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_config(body)
        self._build_log(body)

    def _build_config(self, parent: Any) -> None:
        cfg = ctk.CTkScrollableFrame(
            parent, fg_color=BG_SIDEBAR, corner_radius=0,
            width=272,
        )
        cfg.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(cfg, text="Neuer Actor", font=FONT_HEAD,
                     text_color=FG_HEAD, anchor="w").pack(
            fill="x", padx=16, pady=(16, 10))

        # ── Space checkboxes ──
        ctk.CTkLabel(cfg, text="Memory Spaces", font=FONT_SMALL,
                     text_color=FG_DIM, anchor="w").pack(fill="x", padx=16)

        self._space_vars: dict = {}
        mem  = self.app.app_state.semantic_memory
        spaces = list(mem.memories.keys()) if mem else []

        box = ctk.CTkFrame(cfg, fg_color=BG_CARD, corner_radius=8)
        box.pack(fill="x", padx=16, pady=(4, 14))
        if spaces:
            for sp in spaces:
                pre = (sp == self.app.app_state.selected_space)
                var = tk.BooleanVar(value=pre)
                ctk.CTkCheckBox(
                    box, text=sp, variable=var,
                    font=FONT_BODY, text_color=FG,
                    fg_color=ACCENT, hover_color=_darken(ACCENT),
                    checkmark_color=FG_HEAD,
                ).pack(anchor="w", padx=10, pady=3)
                self._space_vars[sp] = var
        else:
            ctk.CTkLabel(box, text="  —  keine Spaces geladen",
                         font=FONT_SMALL, text_color=FG_DIM).pack(pady=6)

        # ── Task ──
        ctk.CTkLabel(cfg, text="Task / Frage", font=FONT_SMALL,
                     text_color=FG_DIM, anchor="w").pack(fill="x", padx=16)
        self._task = ctk.CTkTextbox(
            cfg, height=90, fg_color=BG_CARD, text_color=FG,
            font=FONT_BODY, corner_radius=8,
            border_width=1, border_color=BORDER,
        )
        self._task.pack(fill="x", padx=16, pady=(4, 14))

        # ── Max iterations ──
        r1 = ctk.CTkFrame(cfg, fg_color="transparent")
        r1.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(r1, text="Max Iterationen", font=FONT_BODY,
                     text_color=FG, anchor="w").pack(side="left")
        self._max_iter = ctk.CTkEntry(r1, width=58, font=FONT_MONO,
                                       fg_color=BG_CARD, text_color=FG,
                                       border_color=BORDER)
        self._max_iter.insert(0, "10")
        self._max_iter.pack(side="right")

        # ── Agent name ──
        r2 = ctk.CTkFrame(cfg, fg_color="transparent")
        r2.pack(fill="x", padx=16, pady=(0, 18))
        ctk.CTkLabel(r2, text="Agent Name", font=FONT_BODY,
                     text_color=FG, anchor="w").pack(side="left")
        self._agent = ctk.CTkEntry(r2, width=110, font=FONT_MONO,
                                    fg_color=BG_CARD, text_color=FG,
                                    border_color=BORDER)
        self._agent.insert(0, "summary")
        self._agent.pack(side="right")

        # ── Start button ──
        self._start_btn = _btn(
            cfg, "▶  Start Actor",
            command=self._start_actor,
            color=ACCENT, width=200,
        )
        self._start_btn.pack(pady=(0, 18))

        # Running actor status badge (if any)
        if self.runner:
            color_map = {
                "running": AMBER, "done": GREEN,
                "error": RED, "stopped": FG_DIM,
            }
            ctk.CTkLabel(
                cfg,
                text=f"● {self.runner.name}  [{self.runner.status}]",
                text_color=color_map.get(self.runner.status, FG_DIM),
                font=FONT_SMALL,
            ).pack(pady=4)

    def _build_log(self, parent: Any) -> None:
        log_frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=0)
        log_frame.grid(row=0, column=1, sticky="nsew")

        # Log topbar
        log_top = ctk.CTkFrame(log_frame, fg_color=BG_SIDEBAR,
                               corner_radius=0, height=40)
        log_top.pack(fill="x")
        log_top.pack_propagate(False)

        self._status_lbl = ctk.CTkLabel(
            log_top, text="Bereit", font=FONT_SMALL, text_color=FG_DIM,
        )
        self._status_lbl.pack(side="left", padx=14, pady=10)

        _btn(log_top, "Export JSON", command=self._export,
             color=BG_CARD, width=110).pack(side="right", padx=8, pady=5)

        # Log textbox
        self._log = ctk.CTkTextbox(
            log_frame, fg_color=BG_CARD, text_color=FG,
            font=FONT_MONO, corner_radius=0, state="disabled", wrap="word",
        )
        self._log.pack(fill="both", expand=True)

        # Color tags on the internal tk.Text
        tb = self._log._textbox
        tb.tag_configure("user",   foreground=ACCENT)
        tb.tag_configure("assist", foreground=FG_DIM)
        tb.tag_configure("tool",   foreground=GREEN)
        tb.tag_configure("err",    foreground=RED)
        tb.tag_configure("info",   foreground=FG_DIM)

    # ── actor control ────────────────────────────────────────────────────────

    def _start_actor(self) -> None:
        mem = self.app.app_state.semantic_memory
        if mem is None:
            self.app.status("Kein Memory verbunden.", error=True)
            return

        spaces = [s for s, v in self._space_vars.items() if v.get()]
        if not spaces:
            self.app.status("Mindestens einen Space auswählen.", error=True)
            return

        task = self._task.get("1.0", "end").strip()
        if not task:
            self.app.status("Task darf nicht leer sein.", error=True)
            return

        try:
            max_iter = int(self._max_iter.get())
        except ValueError:
            max_iter = 10

        agent_name = self._agent.get().strip() or "summary"
        name       = f"actor_{int(time.time())}"

        self.runner = ActorRunner(name, spaces, task, max_iter, agent_name)
        self.app.app_state.actors[name] = self.runner

        self._log_clear()
        self._log_write(f"[START]  Task: {task}\n", "user")
        self._log_write(f"[INFO]   Spaces: {', '.join(spaces)}\n", "info")
        self._log_write(f"[INFO]   Max iterations: {max_iter} | Agent: {agent_name}\n\n", "info")

        self._start_btn.configure(
            text="■  Stop", command=self._stop_actor,
            fg_color=RED, hover_color=_darken(RED),
        )
        self._status_lbl.configure(text=f"● {name}  [running]",
                                   text_color=AMBER)

        self.runner.start(mem)
        self.app.refresh_sidebar()
        self._start_poll()

    def _stop_actor(self) -> None:
        if self.runner:
            self.runner.stop()
            self._log_write("[STOPPED]  Actor gestoppt.\n", "err")
            self._status_lbl.configure(text="● stopped", text_color=FG_DIM)
        self._reset_btn()
        self._stop_poll()

    def _reset_btn(self) -> None:
        self._start_btn.configure(
            text="▶  Start Actor", command=self._start_actor,
            fg_color=ACCENT, hover_color=_darken(ACCENT),
        )

    # ── polling ──────────────────────────────────────────────────────────────

    def _start_poll(self) -> None:
        self._poll_id = self.after(400, self._poll)

    def _stop_poll(self) -> None:
        if self._poll_id:
            self.after_cancel(self._poll_id)
            self._poll_id = None

    def _poll(self) -> None:
        if not self.runner:
            return
        while not self.runner.out_q.empty():
            msg = self.runner.out_q.get_nowait()
            if msg["type"] == "done":
                self._replay_history(msg.get("history", []))
                self._status_lbl.configure(
                    text=f"✓ {self.runner.name}  [done]", text_color=GREEN,
                )
                self._reset_btn()
                self.app.refresh_sidebar()
                return
            elif msg["type"] == "error":
                self._log_write(f"[ERROR]  {msg['msg']}\n", "err")
                self._status_lbl.configure(text="● error", text_color=RED)
                self._reset_btn()
                return
        if self.runner.status == "running":
            self._poll_id = self.after(400, self._poll)

    def _replay_history(self, history: list) -> None:
        self._log_clear()
        tag_map = {"user": "user", "assistant": "assist", "tool": "tool"}
        for item in history:
            role    = item.get("role", "?")
            content = item.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content, indent=2, ensure_ascii=False)
            tag = tag_map.get(role, "info")
            self._log_write(f"[{role.upper()}]\n{content}\n\n", tag)

    # ── log helpers ──────────────────────────────────────────────────────────

    def _log_clear(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _log_write(self, text: str, tag: str = "") -> None:
        self._log.configure(state="normal")
        tb = self._log._textbox
        if tag:
            tb.insert("end", text, tag)
        else:
            self._log.insert("end", text)
        self._log.configure(state="disabled")
        tb.see("end")

    def _export(self) -> None:
        if not self.runner or not self.runner.history:
            self.app.status("Keine History vorhanden.", error=True)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{self.runner.name}_history.json",
        )
        if path:
            Path(path).write_text(
                json.dumps(self.runner.history, indent=2, default=str),
                encoding="utf-8",
            )
            self.app.status(f"History exportiert → {path}")

    def destroy(self) -> None:
        self._stop_poll()
        super().destroy()


# ─── Stubs  (F4 / F5 / F6) ──────────────────────────────────────────────────

class ActorStatePanel(_PanelBase):
    """F4 — alle Actor-States + History-Viewer."""

    def __init__(self, parent: Any, app: "MemoryManagerApp",
                 runner: Optional[ActorRunner] = None) -> None:
        super().__init__(parent, app)
        self._sel_runner: Optional[ActorRunner] = runner
        self._build()

    def _build(self) -> None:
        self._header("Actor States")

        body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=240)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── left: actor list ──────────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=BG_SIDEBAR, corner_radius=0, width=240)
        left.grid(row=0, column=0, sticky="nsew")
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="Actors", font=FONT_HEAD,
                     text_color=FG_HEAD, anchor="w").pack(
            fill="x", padx=14, pady=(14, 6))

        self._actor_list = ctk.CTkScrollableFrame(
            left, fg_color=BG_SIDEBAR, corner_radius=0)
        self._actor_list.pack(fill="both", expand=True)

        _btn(left, "✕  Alle beenden",
             command=self._stop_all, color=RED, width=210).pack(pady=8)

        # ── right: history detail ─────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")

        top = ctk.CTkFrame(right, fg_color=BG_SIDEBAR,
                           corner_radius=0, height=40)
        top.pack(fill="x")
        top.pack_propagate(False)
        self._detail_lbl = ctk.CTkLabel(
            top, text="— kein Actor ausgewählt —",
            font=FONT_SMALL, text_color=FG_DIM)
        self._detail_lbl.pack(side="left", padx=14, pady=10)
        _btn(top, "Export JSON", command=self._export,
             color=BG_CARD, width=110).pack(side="right", padx=8, pady=5)

        self._hist_log = ctk.CTkTextbox(
            right, fg_color=BG_CARD, text_color=FG,
            font=FONT_MONO, corner_radius=0, state="disabled", wrap="word")
        self._hist_log.pack(fill="both", expand=True)

        tb = self._hist_log._textbox
        tb.tag_configure("user",   foreground=ACCENT)
        tb.tag_configure("assist", foreground=FG_DIM)
        tb.tag_configure("tool",   foreground=GREEN)
        tb.tag_configure("err",    foreground=RED)
        tb.tag_configure("info",   foreground=FG_DIM)

        self._refresh_list()
        if self._sel_runner:
            self._show_runner(self._sel_runner)

    def _refresh_list(self) -> None:
        for w in self._actor_list.winfo_children():
            w.destroy()

        actors = self.app.app_state.actors
        if not actors:
            ctk.CTkLabel(self._actor_list, text="  — keine Actors —",
                         text_color=FG_DIM, font=FONT_SMALL).pack(pady=10)
            return

        col = {"running": AMBER, "done": GREEN, "error": RED,
               "stopped": FG_DIM, "pending": FG_DIM}
        for name, runner in actors.items():
            row = ctk.CTkFrame(self._actor_list, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=2)

            ctk.CTkButton(
                row, text=f"●  {name}",
                command=lambda r=runner: self._show_runner(r),
                fg_color="transparent", hover_color=BG_HOVER,
                text_color=col.get(runner.status, FG_DIM),
                anchor="w", font=FONT_SMALL, corner_radius=6, height=28,
            ).pack(side="left", fill="x", expand=True)

            ctk.CTkLabel(row, text=runner.status, font=("IBM Plex Mono", 8),
                         text_color=col.get(runner.status, FG_DIM),
                         width=56, anchor="e").pack(side="right")

    def _show_runner(self, runner: ActorRunner) -> None:
        self._sel_runner = runner
        self._detail_lbl.configure(
            text=f"{runner.name}  [{runner.status}]  "
                 f"spaces={', '.join(runner.spaces)}  "
                 f"iter={runner.max_iter}")

        self._hist_log.configure(state="normal")
        self._hist_log.delete("1.0", "end")
        tb = self._hist_log._textbox

        tag_map = {"user": "user", "assistant": "assist", "tool": "tool"}
        for item in runner.history:
            role    = item.get("role", "?")
            content = item.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content, indent=2, ensure_ascii=False)
            tag = tag_map.get(role, "info")
            tb.insert("end", f"[{role.upper()}]\n{content}\n\n", tag)

        self._hist_log.configure(state="disabled")
        tb.see("end")

    def _stop_all(self) -> None:
        for runner in self.app.app_state.actors.values():
            if runner.status == "running":
                runner.stop()
        self._refresh_list()
        self.app.status("Alle laufenden Actors gestoppt.")

    def _export(self) -> None:
        r = self._sel_runner
        if not r or not r.history:
            self.app.status("Keine History vorhanden.", error=True)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{r.name}_history.json",
        )
        if path:
            Path(path).write_text(
                json.dumps(r.history, indent=2, default=str), encoding="utf-8")
            self.app.status(f"History exportiert → {path}")


class GraphPanel(_PanelBase):
    """F5 — Graph generieren via MemoryGraphVisualizer."""

    def __init__(self, parent: Any, app: "MemoryManagerApp") -> None:
        super().__init__(parent, app)
        self._build()

    def _build(self) -> None:
        self._header("Graph Generator")

        scroll = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        scroll.pack(fill="both", expand=True)

        card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=10)
        card.pack(fill="x", padx=24, pady=20)

        # ── Space selector ────────────────────────────────────────────────────
        r0 = ctk.CTkFrame(card, fg_color="transparent")
        r0.pack(fill="x", padx=20, pady=(16, 8))
        ctk.CTkLabel(r0, text="Memory Space", font=FONT_BODY,
                     text_color=FG, anchor="w", width=140).pack(side="left")

        mem    = self.app.app_state.semantic_memory
        spaces = list(mem.memories.keys()) if mem else []
        self._space_var = tk.StringVar(
            value=self.app.app_state.selected_space or
                  (spaces[0] if spaces else ""))
        ctk.CTkOptionMenu(
            r0, values=spaces or ["—"],
            variable=self._space_var,
            fg_color=BG_SIDEBAR, button_color=ACCENT,
            button_hover_color=_darken(ACCENT),
            text_color=FG, font=FONT_BODY, width=220,
        ).pack(side="left", padx=(12, 0))

        # ── Mode ──────────────────────────────────────────────────────────────
        r1 = ctk.CTkFrame(card, fg_color="transparent")
        r1.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(r1, text="Modus", font=FONT_BODY,
                     text_color=FG, anchor="w", width=140).pack(side="left")

        self._mode_var = tk.StringVar(value="full")
        for val, lbl in [("full", "Full Graph"), ("sub", "Subgraph")]:
            ctk.CTkRadioButton(
                r1, text=lbl, value=val, variable=self._mode_var,
                command=self._on_mode_change,
                fg_color=ACCENT, hover_color=_darken(ACCENT),
                text_color=FG, font=FONT_BODY,
            ).pack(side="left", padx=(0, 20))

        # ── Entity ID (nur Subgraph) ──────────────────────────────────────────
        r2 = ctk.CTkFrame(card, fg_color="transparent")
        r2.pack(fill="x", padx=20, pady=8)
        self._entity_lbl = ctk.CTkLabel(
            r2, text="Entity ID", font=FONT_BODY,
            text_color=FG_DIM, anchor="w", width=140)
        self._entity_lbl.pack(side="left")
        self._entity_entry = ctk.CTkEntry(
            r2, width=280, font=FONT_MONO,
            fg_color=BG_SIDEBAR, text_color=FG_DIM,
            border_color=BORDER, state="disabled",
            placeholder_text="z.B. company:spacex",
        )
        self._entity_entry.pack(side="left", padx=(12, 0))

        # ── Depth ─────────────────────────────────────────────────────────────
        r3 = ctk.CTkFrame(card, fg_color="transparent")
        r3.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(r3, text="Max Depth", font=FONT_BODY,
                     text_color=FG, anchor="w", width=140).pack(side="left")
        self._depth = ctk.CTkEntry(
            r3, width=60, font=FONT_MONO,
            fg_color=BG_SIDEBAR, text_color=FG, border_color=BORDER)
        self._depth.insert(0, "3")
        self._depth.pack(side="left", padx=(12, 0))

        # ── Generate button ───────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(12, 18))
        self._gen_btn = _btn(
            btn_row, "⚙  Graph generieren",
            command=self._generate, color=ACCENT, width=200)
        self._gen_btn.pack(side="left")

        # ── Result card ───────────────────────────────────────────────────────
        self._result_card = ctk.CTkFrame(
            scroll, fg_color=BG_CARD, corner_radius=10)
        self._result_card.pack(fill="x", padx=24, pady=(0, 20))
        self._result_lbl = ctk.CTkLabel(
            self._result_card,
            text="Noch kein Graph generiert.",
            text_color=FG_DIM, font=FONT_BODY)
        self._result_lbl.pack(padx=20, pady=16)

    def _on_mode_change(self) -> None:
        sub = self._mode_var.get() == "sub"
        col = FG if sub else FG_DIM
        st  = "normal" if sub else "disabled"
        self._entity_lbl.configure(text_color=col)
        self._entity_entry.configure(state=st, text_color=col)

    def _generate(self) -> None:
        mem = self.app.app_state.semantic_memory
        if not mem:
            self.app.status("Kein Memory verbunden.", error=True)
            return

        space = self._space_var.get()
        if not space or space == "—" or space not in mem.memories:
            self.app.status("Ungültiger Space.", error=True)
            return

        store = mem.memories[space]
        mode  = self._mode_var.get()
        try:
            depth = int(self._depth.get())
        except ValueError:
            depth = 3

        self._gen_btn.configure(state="disabled", text="⏳ …")
        self.app.status("Graph wird generiert …")

        try:
            from toolboxv2.mods.isaa.base.memory_graph_visualizer import (
                MemoryGraphVisualizer)
            viz = MemoryGraphVisualizer(store, max_depth=depth)

            if mode == "full":
                html = viz.generate_full_graph_dashboard()
                fname = f"memory_graph_{space}_full.html"
            else:
                entity_id = self._entity_entry.get().strip()
                if not entity_id:
                    self.app.status("Entity ID fehlt.", error=True)
                    self._gen_btn.configure(state="normal",
                                            text="⚙  Graph generieren")
                    return
                html = viz.generate_entity_network_html(entity_id, depth)
                fname = f"memory_graph_{space}_{entity_id.replace(':', '_')}.html"

            path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML", "*.html")],
                initialfile=fname,
            )
            if not path:
                self._gen_btn.configure(state="normal",
                                        text="⚙  Graph generieren")
                return

            Path(path).write_text(html, encoding="utf-8")

            # Update result card
            for w in self._result_card.winfo_children():
                w.destroy()
            ctk.CTkLabel(
                self._result_card,
                text=f"✓  Graph gespeichert:",
                text_color=GREEN, font=FONT_HEAD,
            ).pack(anchor="w", padx=20, pady=(14, 2))
            ctk.CTkLabel(
                self._result_card,
                text=path, text_color=FG_DIM,
                font=FONT_MONO, wraplength=700, justify="left",
            ).pack(anchor="w", padx=20, pady=(0, 8))
            _btn(self._result_card, "📂  Im Browser öffnen",
                 command=lambda p=path: webbrowser.open(p),
                 color=ACCENT, width=200,
                 ).pack(anchor="w", padx=20, pady=(0, 16))

            self.app.status(f"Graph gespeichert → {path}")
            webbrowser.open(path)

        except Exception as exc:
            self.app.status(f"Fehler: {exc}", error=True)
            logger.exception("Graph generation failed")
        finally:
            self._gen_btn.configure(state="normal",
                                    text="⚙  Graph generieren")


class SearchPanel(_PanelBase):
    """F6 — Einträge suchen (FTS5 / Concept) & neue Einträge hinzufügen."""

    def __init__(self, parent: Any, app: "MemoryManagerApp") -> None:
        super().__init__(parent, app)
        self._entry_map: dict = {}   # treeview iid → entry_id
        self._store: Any = None
        self._build()

    def _build(self) -> None:
        self._header("Search & Add")

        body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)

        self._build_search_bar(body)
        self._build_results(body)
        self._build_add_form(body)

    # ── search bar ────────────────────────────────────────────────────────────

    def _build_search_bar(self, parent: Any) -> None:
        bar = ctk.CTkFrame(parent, fg_color=BG_SIDEBAR,
                           corner_radius=0, height=56)
        bar.grid(row=0, column=0, sticky="ew")
        bar.pack_propagate(False)

        mem    = self.app.app_state.semantic_memory
        spaces = list(mem.memories.keys()) if mem else []
        self._space_var = tk.StringVar(
            value=self.app.app_state.selected_space or
                  (spaces[0] if spaces else ""))

        ctk.CTkOptionMenu(
            bar, values=spaces or ["—"],
            variable=self._space_var,
            fg_color=BG_CARD, button_color=ACCENT,
            button_hover_color=_darken(ACCENT),
            text_color=FG, font=FONT_BODY, width=200,
        ).pack(side="left", padx=(12, 8), pady=10)

        self._query = ctk.CTkEntry(
            bar, placeholder_text="Suche …",
            font=FONT_BODY, fg_color=BG_CARD,
            text_color=FG, border_color=BORDER, height=34,
        )
        self._query.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=10)
        self._query.bind("<Return>", lambda _e: self._search())

        self._mode_var = tk.StringVar(value="fts5")
        for val, lbl in [("fts5", "FTS5"), ("concept", "Concept")]:
            ctk.CTkRadioButton(
                bar, text=lbl, value=val, variable=self._mode_var,
                fg_color=ACCENT, hover_color=_darken(ACCENT),
                text_color=FG, font=FONT_SMALL,
            ).pack(side="left", padx=6, pady=10)

        _btn(bar, "Suchen", command=self._search,
             color=ACCENT, width=90).pack(side="left", padx=(8, 12), pady=10)

    # ── results ───────────────────────────────────────────────────────────────

    def _build_results(self, parent: Any) -> None:
        tree_frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=0)
        tree_frame.grid(row=1, column=0, sticky="nsew")

        # count label
        self._count_lbl = ctk.CTkLabel(
            tree_frame, text="— keine Suche —",
            font=FONT_SMALL, text_color=FG_DIM, anchor="w")
        self._count_lbl.pack(fill="x", padx=12, pady=(6, 2))

        cols = ("score", "type", "role", "preview")
        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            style="Dark.Treeview", selectmode="extended", height=14)
        self._tree.heading("score",   text="Score")
        self._tree.heading("type",    text="Type")
        self._tree.heading("role",    text="Role")
        self._tree.heading("preview", text="Content")
        self._tree.column("score",   width=64,  stretch=False)
        self._tree.column("type",    width=70,  stretch=False)
        self._tree.column("role",    width=74,  stretch=False)
        self._tree.column("preview", width=600, stretch=True)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self._tree.yview,
                            style="Dark.Vertical.TScrollbar")
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        vsb.pack(side="right", fill="y", pady=2)

        # expand area (inline accordion)
        self._expand_frame = ctk.CTkFrame(
            parent, fg_color=BG_SIDEBAR, corner_radius=0)
        self._expand_frame.grid(row=2, column=0, sticky="ew")
        self._expand_box = ctk.CTkTextbox(
            self._expand_frame, height=90, fg_color=BG_SIDEBAR,
            text_color=FG, font=FONT_MONO, state="disabled", wrap="word")
        self._expand_box.pack(fill="x", padx=10, pady=(4, 4))
        self._expand_frame.grid_remove()   # hidden until selection

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # action row
        act = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0, height=40)
        act.grid(row=3, column=0, sticky="ew")
        act.pack_propagate(False)
        _btn(act, "Soft Delete (Auswahl)",
             command=lambda: self._delete_sel(hard=False),
             color=AMBER, width=200).pack(side="left", padx=(12, 8), pady=6)
        _btn(act, "Hard Delete (Auswahl)",
             command=lambda: self._delete_sel(hard=True),
             color=RED, width=200).pack(side="left", pady=6)

    # ── add form ──────────────────────────────────────────────────────────────

    def _build_add_form(self, parent: Any) -> None:
        add_card = ctk.CTkFrame(parent, fg_color=BG_CARD,
                                corner_radius=0)
        add_card.grid(row=4, column=0, sticky="ew")

        # toggle header
        self._add_open = tk.BooleanVar(value=False)
        ctk.CTkButton(
            add_card, text="▶  Neuen Eintrag hinzufügen",
            command=self._toggle_add,
            fg_color=BG_CARD, hover_color=BG_HOVER,
            text_color=FG_DIM, anchor="w", font=FONT_BODY,
            corner_radius=0, height=38,
        ).pack(fill="x")

        self._add_body = ctk.CTkFrame(add_card, fg_color=BG_CARD,
                                      corner_radius=0)
        # body is hidden until toggle

        grid = self._add_body
        grid.columnconfigure(1, weight=1)

        # content
        ctk.CTkLabel(grid, text="Content", font=FONT_BODY,
                     text_color=FG, anchor="w").grid(
            row=0, column=0, sticky="nw", padx=(16, 8), pady=(12, 4))
        self._add_content = ctk.CTkTextbox(
            grid, height=70, fg_color=BG_SIDEBAR,
            text_color=FG, font=FONT_BODY, corner_radius=6,
            border_width=1, border_color=BORDER)
        self._add_content.grid(row=0, column=1, columnspan=3,
                               sticky="ew", padx=(0, 16), pady=(12, 4))

        # type + role + concepts row
        ctk.CTkLabel(grid, text="Type", font=FONT_BODY,
                     text_color=FG, anchor="w").grid(
            row=1, column=0, sticky="w", padx=(16, 8), pady=4)
        self._add_type = ctk.CTkOptionMenu(
            grid, values=["text", "code", "fact", "entity"],
            fg_color=BG_SIDEBAR, button_color=ACCENT,
            button_hover_color=_darken(ACCENT), text_color=FG,
            font=FONT_BODY, width=120)
        self._add_type.grid(row=1, column=1, sticky="w",
                            padx=(0, 16), pady=4)

        ctk.CTkLabel(grid, text="Role (opt.)", font=FONT_BODY,
                     text_color=FG, anchor="w").grid(
            row=1, column=2, sticky="w", padx=(0, 8), pady=4)
        self._add_role = ctk.CTkOptionMenu(
            grid, values=["—", "user", "assistant", "system", "tool"],
            fg_color=BG_SIDEBAR, button_color=ACCENT,
            button_hover_color=_darken(ACCENT), text_color=FG,
            font=FONT_BODY, width=130)
        self._add_role.grid(row=1, column=3, sticky="w",
                            padx=(0, 16), pady=4)

        ctk.CTkLabel(grid, text="Concepts", font=FONT_BODY,
                     text_color=FG, anchor="w").grid(
            row=2, column=0, sticky="w", padx=(16, 8), pady=4)
        self._add_concepts = ctk.CTkEntry(
            grid, placeholder_text="auth, api, database  (kommasepariert)",
            font=FONT_BODY, fg_color=BG_SIDEBAR,
            text_color=FG, border_color=BORDER)
        self._add_concepts.grid(row=2, column=1, columnspan=3,
                                sticky="ew", padx=(0, 16), pady=4)

        # submit
        self._add_btn = _btn(
            grid, "＋  Hinzufügen (async → embedding wird generiert)",
            command=self._add_entry, color=GREEN, width=380)
        self._add_btn.grid(row=3, column=0, columnspan=4,
                           sticky="w", padx=16, pady=(4, 14))

    def _toggle_add(self) -> None:
        if self._add_body.winfo_ismapped():
            self._add_body.pack_forget()
        else:
            self._add_body.pack(fill="x")

    # ── logic ─────────────────────────────────────────────────────────────────

    def _get_store(self) -> Optional[Any]:
        mem   = self.app.app_state.semantic_memory
        space = self._space_var.get()
        if not mem or space == "—" or space not in mem.memories:
            self.app.status("Kein gültiger Space ausgewählt.", error=True)
            return None
        self._store = mem.memories[space]
        return self._store

    def _search(self) -> None:
        store = self._get_store()
        if not store:
            return
        q    = self._query.get().strip()
        mode = self._mode_var.get()
        if not q:
            self.app.status("Suchbegriff fehlt.", error=True)
            return

        self._tree.delete(*self._tree.get_children())
        self._entry_map.clear()
        self._hide_expand()

        try:
            if mode == "concept":
                rows = store._exec(
                    """SELECT ci.entry_id, e.content, e.content_type,
                              e.meta_role, 1.0 AS score
                       FROM concept_index ci
                       JOIN entries e ON e.id = ci.entry_id
                       WHERE ci.concept = ? AND e.is_active = 1 AND e.space = ?
                       LIMIT 40""",
                    (q.lower(), store.space),
                ).fetchall()
            else:  # fts5
                import re as _re
                safe = _re.sub(r'[\\/.:"\'(){}\[\]^~*!@#$&|<>=,;]', ' ', q).strip()
                safe = ' '.join(safe.split())
                if not safe:
                    self.app.status("Suchbegriff nach Bereinigung leer.", error=True)
                    return
                rows = store._exec(
                    """SELECT e.id AS entry_id, e.content, e.content_type,
                              e.meta_role,
                              bm25(entries_fts) AS score
                       FROM entries_fts
                       JOIN entries e ON e.id = entries_fts.entry_id
                       WHERE entries_fts MATCH ? AND e.space = ? AND e.is_active = 1
                       ORDER BY score
                       LIMIT 40""",
                    (safe, store.space),
                ).fetchall()
        except Exception as exc:
            self.app.status(f"Suche fehlgeschlagen: {exc}", error=True)
            return

        for row in rows:
            score   = row["score"] if mode != "concept" else "—"
            preview = (row["content"] or "")[:140].replace("\n", " ")
            iid = self._tree.insert(
                "", "end",
                values=(
                    f"{score:.3f}" if isinstance(score, float) else score,
                    row["content_type"],
                    row["meta_role"] or "—",
                    preview,
                ),
            )
            self._entry_map[iid] = row["entry_id"]

        self._count_lbl.configure(
            text=f"{len(rows)} Treffer für '{q}'  [{mode}]")
        self.app.status(f"Suche: {len(rows)} Treffer.")

    def _on_select(self, _event: Any) -> None:
        sel = self._tree.selection()
        if not sel or not self._store:
            self._hide_expand()
            return
        eid = self._entry_map.get(sel[0])
        if not eid:
            return
        try:
            row = self._store._exec(
                "SELECT content FROM entries WHERE id=?", (eid,)
            ).fetchone()
            if row:
                self._expand_box.configure(state="normal")
                self._expand_box.delete("1.0", "end")
                self._expand_box.insert("end", row["content"])
                self._expand_box.configure(state="disabled")
                self._expand_frame.grid()
        except Exception:
            pass

    def _hide_expand(self) -> None:
        self._expand_frame.grid_remove()

    def _delete_sel(self, hard: bool) -> None:
        sel = self._tree.selection()
        if not sel or not self._store:
            return
        verb = "permanent löschen" if hard else "soft-deleten"
        if not messagebox.askyesno(f"{len(sel)} Einträge {verb}?",
                                   f"{len(sel)} Einträge {verb}?"):
            return
        for iid in sel:
            eid = self._entry_map.get(iid)
            if eid:
                try:
                    self._store.delete(eid, hard=hard)
                except Exception as exc:
                    logger.error("Delete %s: %s", eid, exc)
        self._search()   # re-run search to refresh list
        self.app.status(
            f"{len(sel)} Einträge {'gelöscht' if hard else 'soft-gelöscht'}.")

    def _add_entry(self) -> None:
        mem = self.app.app_state.semantic_memory
        if not mem:
            self.app.status("Kein Memory verbunden.", error=True)
            return

        space   = self._space_var.get()
        content = self._add_content.get("1.0", "end").strip()
        if not content:
            self.app.status("Content darf nicht leer sein.", error=True)
            return

        ctype    = self._add_type.get()
        role_val = self._add_role.get()
        meta     = {"role": role_val} if role_val != "—" else {}
        concepts = [c.strip() for c in self._add_concepts.get().split(",")
                    if c.strip()]

        self._add_btn.configure(state="disabled", text="⏳ wird gespeichert …")
        self.app.status("Eintrag wird mit Embedding gespeichert …")

        from toolboxv2 import get_app

        async def _do_add() -> None:
            try:
                await mem.add_data(
                    space, content, metadata=meta,
                    concepts=concepts if concepts else None,
                    content_type=ctype,
                )
                self.after(0, self._add_done)
            except Exception as exc:
                self.after(0, lambda: self.app.status(
                    f"Add fehlgeschlagen: {exc}", error=True))
                self.after(0, lambda: self._add_btn.configure(
                    state="normal",
                    text="＋  Hinzufügen (async → embedding wird generiert)"))

        get_app().run_bg_task_advanced(_do_add)

    def _add_done(self) -> None:
        self._add_btn.configure(
            state="normal",
            text="＋  Hinzufügen (async → embedding wird generiert)")
        self._add_content.delete("1.0", "end")
        self._add_concepts.delete(0, "end")
        self.app.status("✓  Eintrag gespeichert.")
        self.app.refresh_sidebar()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

class AccordionSection(ctk.CTkFrame):
    """One collapsible sidebar section."""

    def __init__(
        self,
        parent: Any,
        title: str,
        icon: str = "",
        open_: bool = False,
    ) -> None:
        super().__init__(parent, fg_color=BG_SIDEBAR, corner_radius=0)
        self._open  = open_
        self._title = title
        self._icon  = icon

        self._hdr = ctk.CTkButton(
            self, text=self._hdr_text(),
            command=self._toggle,
            fg_color=BG_SIDEBAR, hover_color=BG_HOVER,
            text_color=FG, anchor="w",
            font=FONT_BODY, corner_radius=0, height=36,
        )
        self._hdr.pack(fill="x")

        self._body = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0)
        if self._open:
            self._body.pack(fill="x")

    def _hdr_text(self) -> str:
        arrow = "▼" if self._open else "▶"
        icon  = f" {self._icon} " if self._icon else "  "
        return f" {arrow}{icon}{self._title}"

    def _toggle(self) -> None:
        self._open = not self._open
        self._hdr.configure(text=self._hdr_text())
        if self._open:
            self._body.pack(fill="x")
        else:
            self._body.pack_forget()

    @property
    def content(self) -> ctk.CTkFrame:
        return self._body

    def clear(self) -> None:
        for w in self._body.winfo_children():
            w.destroy()


class Sidebar(ctk.CTkFrame):

    def __init__(self, parent: Any, app: "MemoryManagerApp") -> None:
        super().__init__(parent, fg_color=BG_SIDEBAR, corner_radius=0,
                         width=SIDEBAR_W)
        self.pack_propagate(False)
        self.app = app
        self._collapsed = False
        self._build()

    def _build(self) -> None:
        # Toggle
        self._toggle_btn = ctk.CTkButton(
            self, text="◀", command=self._toggle_collapse,
            fg_color=BG_SIDEBAR, hover_color=BG_HOVER,
            text_color=FG_DIM, width=SIDEBAR_W, height=34,
            corner_radius=0, font=FONT_SMALL, anchor="e",
        )
        self._toggle_btn.pack(fill="x")

        # Scrollable content area
        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color=BG_SIDEBAR, corner_radius=0,
        )
        self._scroll.pack(fill="both", expand=True)

        open_ = self.app.app_state.sections_open
        self._sec_mem   = AccordionSection(self._scroll, "Memories", "💾",
                                           open_=open_.get("memories", True))
        self._sec_mem.pack(fill="x")

        self._sec_actor = AccordionSection(self._scroll, "Actors",   "🤖",
                                           open_=open_.get("actors", False))
        self._sec_actor.pack(fill="x")

        self._sec_graph = AccordionSection(self._scroll, "Graph",    "🕸",
                                           open_=open_.get("graph", False))
        self._sec_graph.pack(fill="x")

        # Bottom actions
        sep = ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0)
        sep.pack(fill="x", pady=(4, 0))

        self._btn_new_actor = _btn(
            self, "＋  Neuer Actor",
            command=lambda: self.app.show_panel("actor"),
            color=ACCENT, width=SIDEBAR_W - 16,
        )
        self._btn_new_actor.pack(pady=(8, 2))

        self._btn_search = _btn(
            self, "🔍  Suchen",
            command=lambda: self.app.show_panel("search"),
            color=BG_CARD, width=SIDEBAR_W - 16,
        )
        self._btn_search.pack(pady=(0, 2))

        self._btn_clear_empty = _btn(
            self, "🧹  Clear All Empty",
            command=self._clear_empty_spaces,
            color=BG_CARD, width=SIDEBAR_W - 16,
        )
        self._btn_clear_empty.pack(pady=(0, 8))

        self.refresh()

    # ── refresh ──────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        self._refresh_memories()
        self._refresh_actors()
        self._refresh_graph()

    def _refresh_memories(self) -> None:
        self._sec_mem.clear()
        mem = self.app.app_state.semantic_memory
        if not mem or not mem.memories:
            ctk.CTkLabel(self._sec_mem.content, text="   —  leer",
                         text_color=FG_DIM, font=FONT_SMALL).pack(pady=4, anchor="w")
            return

        for name, store in mem.memories.items():
            try:
                st      = store.stats()
                active  = st.get("active", 0)
                total   = max(st.get("total", 1), 1)
                ratio   = min(active / total, 1.0)
                dot_col = GREEN if ratio < 0.7 else (AMBER if ratio < 0.9 else RED)
                badge   = "💬 " if self._is_chat_space(store) else ""
                label   = f"●  {badge}{name}   {active}"
            except Exception:
                dot_col = FG_DIM
                label   = f"●  {name}   ?"

            ctk.CTkButton(
                self._sec_mem.content,
                text=label,
                command=lambda n=name: self.app.show_panel("memory_detail", space=n),
                fg_color="transparent", hover_color=BG_HOVER,
                text_color=dot_col, anchor="w",
                font=FONT_SMALL, corner_radius=0, height=28,
            ).pack(fill="x", padx=6)

    def _refresh_actors(self) -> None:
        self._sec_actor.clear()
        actors = self.app.app_state.actors
        if not actors:
            ctk.CTkLabel(self._sec_actor.content, text="   —  keine",
                         text_color=FG_DIM, font=FONT_SMALL).pack(pady=4, anchor="w")
            return

        col = {"running": AMBER, "done": GREEN, "error": RED,
               "stopped": FG_DIM, "pending": FG_DIM}
        for name, runner in actors.items():
            ctk.CTkButton(
                self._sec_actor.content,
                text=f"●  {name}",
                command=lambda r=runner: self.app.show_panel("actor", runner=r),
                fg_color="transparent", hover_color=BG_HOVER,
                text_color=col.get(runner.status, FG_DIM), anchor="w",
                font=FONT_SMALL, corner_radius=0, height=28,
            ).pack(fill="x", padx=6)

    def _refresh_graph(self) -> None:
        self._sec_graph.clear()
        _btn(
            self._sec_graph.content, "⚙  Graph öffnen",
            command=lambda: self.app.show_panel("graph"),
            color=BG_CARD, width=SIDEBAR_W - 32,
        ).pack(pady=8, padx=8)

    @staticmethod
    def _is_chat_space(store: Any) -> bool:
        try:
            chat = store._exec(
                """SELECT COUNT(*) FROM entries
                   WHERE space=? AND meta_role IN ('user','assistant') AND is_active=1""",
                (store.space,),
            ).fetchone()[0]
            total = store._exec(
                "SELECT COUNT(*) FROM entries WHERE space=? AND is_active=1",
                (store.space,),
            ).fetchone()[0]
            return total > 0 and (chat / total) > 0.5
        except Exception:
            return False

    # ── collapse ─────────────────────────────────────────────────────────────

    def _clear_empty_spaces(self) -> None:
        mem = self.app.app_state.semantic_memory
        if not mem:
            return
        to_remove = []
        for name, store in list(mem.memories.items()):
            try:
                active = store.stats().get("active", 0)
                if active == 0:
                    to_remove.append(name)
            except Exception:
                pass
        if not to_remove:
            self.app.status("Keine leeren Spaces gefunden.")
            return
        if not messagebox.askyesno(
            "Clear Empty Spaces",
            f"{len(to_remove)} leere Space(s) entfernen?\n"
            f"{', '.join(to_remove)}\n(Dateien auf Disk bleiben erhalten)",
        ):
            return
        for name in to_remove:
            store = mem.memories.get(name)
            if store:
                try:
                    store.close()
                except Exception:
                    pass
                del mem.memories[name]
        self.refresh()
        self.app.status(
            f"{len(to_remove)} leere Space(s) entfernt: {', '.join(to_remove)}")
        if self.app.app_state.selected_space in to_remove:
            self.app.show_panel("welcome")

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.configure(width=SIDEBAR_W_MINI)
            self._scroll.pack_forget()
            self._btn_new_actor.pack_forget()
            self._btn_search.pack_forget()
            self._btn_clear_empty.pack_forget()
            self._toggle_btn.configure(text="▶", anchor="center",
                                       width=SIDEBAR_W_MINI)
        else:
            self.configure(width=SIDEBAR_W)
            self._toggle_btn.configure(text="◀", anchor="e",
                                       width=SIDEBAR_W)
            self._scroll.pack(fill="both", expand=True,
                              after=self._toggle_btn)
            self._btn_new_actor.pack(pady=(8, 2))
            self._btn_search.pack(pady=(0, 2))
            self._btn_clear_empty.pack(pady=(0, 8))
        self.app.app_state.sidebar_collapsed = self._collapsed


# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR & STATUSBAR
# ══════════════════════════════════════════════════════════════════════════════

class Topbar(ctk.CTkFrame):

    def __init__(self, parent: Any, app: "MemoryManagerApp") -> None:
        super().__init__(parent, fg_color=BG_SIDEBAR, corner_radius=0, height=46)
        self.pack_propagate(False)

        ctk.CTkLabel(self, text="●", text_color=ACCENT,
                     font=("IBM Plex Sans", 14, "bold")).pack(
            side="left", padx=(16, 4))
        ctk.CTkLabel(self, text="Memory Manager", font=FONT_HEAD,
                     text_color=FG_HEAD).pack(side="left")

        _btn(self, "⟳  Refresh", command=app.refresh_all,
             color=BG_CARD, width=100).pack(side="right", padx=10, pady=7)

        ctk.CTkLabel(self, text="ctk 5.2.2 | isaa v2",
                     font=FONT_SMALL, text_color=FG_DIM).pack(
            side="right", padx=10)


class Statusbar(ctk.CTkFrame):

    def __init__(self, parent: Any) -> None:
        super().__init__(parent, fg_color=BG_SIDEBAR, corner_radius=0, height=26)
        self.pack_propagate(False)
        self._lbl = ctk.CTkLabel(
            self, text="Bereit", font=FONT_SMALL,
            text_color=FG_DIM, anchor="w",
        )
        self._lbl.pack(side="left", padx=14)

    def set(self, msg: str, error: bool = False) -> None:
        self._lbl.configure(text=msg, text_color=RED if error else FG_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class MemoryManagerApp(ctk.CTk):

    def __init__(self) -> None:
        super().__init__(fg_color=BG)
        self.title("Memory Manager — ISAA V2")
        self.geometry("1300x820")
        self.minsize(960, 620)

        ctk.set_appearance_mode("dark")
        _style_treeview()

        self.app_state = AppState()
        self._connect_memory()
        self._current_panel: Optional[ctk.CTkFrame] = None
        self._build()

    # ── init ─────────────────────────────────────────────────────────────────

    def _connect_memory(self) -> None:
        try:
            from toolboxv2 import get_app          # noqa: PLC0415
            self.app_state.semantic_memory = get_app().get_mod("isaa").get_memory()
        except Exception as exc:
            logger.warning("AISemanticMemory: %s", exc)

    def _build(self) -> None:
        self._topbar = Topbar(self, self)
        self._topbar.pack(fill="x", side="top")

        self._statusbar = Statusbar(self)
        self._statusbar.pack(fill="x", side="bottom")

        mid = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        mid.pack(fill="both", expand=True)

        self._sidebar = Sidebar(mid, self)
        self._sidebar.pack(side="left", fill="y")

        ctk.CTkFrame(mid, fg_color=BORDER, width=1,
                     corner_radius=0).pack(side="left", fill="y")

        self._main = ctk.CTkFrame(mid, fg_color=BG, corner_radius=0)
        self._main.pack(side="left", fill="both", expand=True)

        self.show_panel("welcome")

    # ── panel router ─────────────────────────────────────────────────────────

    def show_panel(self, panel_id: str, **kw) -> None:
        if self._current_panel:
            self._current_panel.destroy()
            self._current_panel = None

        self.app_state.active_panel = panel_id

        panels: dict = {
            "welcome":       lambda: WelcomePanel(self._main, self),
            "memory_detail": lambda: MemoryDetailPanel(
                self._main, self, kw.get("space", ""),
            ),
            "actor":         lambda: ActorPanel(
                self._main, self, kw.get("runner"),
            ),
            "actor_state":   lambda: ActorStatePanel(
                self._main, self, kw.get("runner"),
            ),
            "graph":         lambda: GraphPanel(self._main, self),
            "search":        lambda: SearchPanel(self._main, self),
        }
        panel = panels.get(panel_id, lambda: WelcomePanel(self._main, self))()
        panel.pack(fill="both", expand=True)
        self._current_panel = panel

    # ── helpers ──────────────────────────────────────────────────────────────

    def refresh_all(self) -> None:
        self.refresh_sidebar()
        # Re-open the same panel (re-reads fresh data)
        cur = self.app_state.active_panel
        space = self.app_state.selected_space
        if cur == "memory_detail" and space:
            self.show_panel(cur, space=space)
        else:
            self.show_panel(cur)

    def refresh_sidebar(self) -> None:
        self._sidebar.refresh()

    def status(self, msg: str, error: bool = False) -> None:
        self._statusbar.set(msg, error=error)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def launch() -> None:
    if not _CTK_AVAILABLE:
        raise RuntimeError(
            "customtkinter nicht verfügbar.\n"
            "Install: pip install customtkinter==5.2.2"
        )
    app = MemoryManagerApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
