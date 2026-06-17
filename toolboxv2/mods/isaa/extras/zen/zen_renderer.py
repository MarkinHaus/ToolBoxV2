"""
ZenRendererV2 - Orchestration-aware TUI renderer for ISAA CLI v4.
================================================================

Reads engine.live (AgentLiveState) directly for real-time agent introspection.
Two modes:
  - Zen (default): minimal, grows with complexity, shows thought/focus
  - Debug (ENGINE_DEBUG=1): full dashboard with all sub-agents, tools, tokens

Key design:
  - prompt_toolkit compatible (HTML, print_formatted_text, patch_stdout safe)
  - Minimizable: user can collapse/expand without interrupting the agent
  - Progressive: starts quiet, reveals detail as the run gets complex
  - Multi-process: shows all background agents in a compact table

Step 2.5 integration: cli_v4.py replaces old ZenRenderer with this one.
Just swap `renderer = ZenRenderer()` -> `renderer = ZenRendererV2(engine)`.
"""

import json
import os
import random
import sys
import threading
import time
from collections import deque
from typing import Any

from prompt_toolkit import print_formatted_text, HTML

from toolboxv2.mods.isaa.base.AgentUtils import anything_from_str_to_dict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG = os.environ.get("AGENT_VERBOSE", "").lower() in ("1", "true", "yes")

# IBM-style symbols (no emoji decoration)
SYM = {
    "agent":    "◯",
    "sub":      "◈",
    "think":    "◎",
    "tool":     "◇",
    "ok":       "✓",
    "fail":     "✗",
    "done":     "●",
    "warn":     "⚠",
    "pause":    "⏸",
    "progress": "▸",
    "minimize": "▾",
    "expand":   "▴",
    "bar_fill": "━",
    "bar_empty":"─",
}

# Zen color palette (prompt_toolkit HTML fg)
C = {
    "dim":     "#6b7280",
    "cyan":    "#67e8f9",
    "green":   "#4ade80",
    "red":     "#f87171",
    "amber":   "#fbbf24",
    "white":   "#e5e7eb",
    "bright":  "#ffffff",
    "blue":    "#60a5fa",
}


def _esc(text: Any) -> str:
    """Escape text for prompt_toolkit HTML."""
    import html as _html
    return _html.escape(str(text))


def _bar(cur: int, total: int, width: int = 20) -> str:
    """ASCII progress bar."""
    if total <= 0:
        return SYM["bar_empty"] * width
    filled = int(width * cur / total)
    return SYM["bar_fill"] * filled + SYM["bar_empty"] * (width - filled)


def _short(s: str, n: int = 40) -> str:
    return s[:n] + ".." if len(s) > n + 2 else s


# ===========================================================================
# ZenRendererV2
# ===========================================================================

def _fmt_elapsed(secs: float) -> str:
    s = int(secs)

    weeks, s = divmod(s, 604800)   # 7 * 24 * 3600
    days, s = divmod(s, 86400)

    if weeks > 0:
        return f"{weeks}w{days}d"

    hours, s = divmod(s, 3600)

    if days > 0:
        return f"{days}d{hours:02d}h"

    minutes, seconds = divmod(s, 60)

    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"

class ZenRendererV2:
    """
    Reads engine.live for phase/thought/tool.
    Consumes stream chunks for content/reasoning display.
    Minimizable dashboard for background process overview.
    """

    def __init__(self, engine=None, agent=None):
        """
        Args:
            engine: ExecutionEngine instance (has .live: AgentLiveState).
                    If None, works in chunk-only mode (no live state).
        """
        self._footer_active = False
        self.wit_alim = False
        self._engine = engine
        self._agent = agent
        self._anim_thread = None
        self._anim_stop = threading.Event()
        self.minimized = False
        self._last_agent = None
        self._last_iter = 0
        self._in_think = False
        self._needs_newline = False
        self._think_buf = ""
        self._content_buf = ""
        self._chunk_count = 0
        self._zen_plus = None
        self._execution_id: str | None = None
        self._chunk_buffer: list = []

        self.thought_pool = deque(maxlen=20)  # Speichert die letzten 20 Wörter
        self._zen_symbols = ["◌", "◦", "•", "∙", "·"]
        self._current_word = "zen"

        self._print("🌌 Zen System")

    @property
    def engine(self):
        if self._engine is None:
            return self._engine
        if self._agent is None:
            raise ValueError("engin and agent ar None")
        return self._agent._get_execution_engine()

    # -- public API -----------------------------------------------------------

    def set_zen_plus(self, zp):
        self._zen_plus = zp

    def toggle_minimize(self):
        """Toggle between minimized (one-liner) and expanded view."""
        self.minimized = not self.minimized
        if self.minimized:
            self._print(f"<style fg='{C['dim']}'>{SYM['minimize']} minimized (press again to expand)</style>")
        else:
            self._print(f"<style fg='{C['dim']}'>{SYM['expand']} expanded</style>")

    def set_execution_id(self, eid: str):
        self._execution_id = eid

    def process_chunk(self, chunk: dict):
        """Main entry: render one stream chunk. prompt_toolkit safe."""
        self._chunk_count += 1
        if self._execution_id:
            chunk = dict(chunk)
            chunk["_execution_id"] = self._execution_id
        self._chunk_buffer.append(chunk)
        if self._zen_plus:
            self._zen_plus.feed_chunk(chunk)
        if self._zen_plus and self._zen_plus.active:
            return
        c_type = chunk.get("type", "")

        # In minimized mode: only show done/error, skip everything else
        if self.minimized and c_type not in ("done", "error", "final_answer"):
            return

        # Agent context header (on agent change)
        self._maybe_print_agent_header(chunk)

        if c_type == "_sub_done":
            return  # Internal sentinel, skip
        # Dispatch
        if c_type == "reasoning":
            self._on_reasoning(chunk)
        elif c_type == "content":
            self._on_content(chunk)
        elif c_type == "tool_start":
            self._on_tool_start(chunk)
        elif c_type == "tool_result":
            self._on_tool_result(chunk)
        elif c_type == "final_answer":
            self._on_final_answer(chunk)
        elif c_type == "done":
            self._on_done(chunk)
        elif c_type == "warning":
            self._print(f"  <style fg='{C['amber']}'>{SYM['warn']} {_esc(chunk.get('message', ''))}</style>")
        elif c_type == "error":
            self._print(f"  <style fg='{C['red']}'>{SYM['fail']} {_esc(chunk.get('error', ''))}</style>")

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        """'#67e8f9' -> '103;232;249' für ANSI 24-bit color."""
        h = hex_color.lstrip('#')
        return f"{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}"

    def print_processes(self, background_tasks: dict, agents_getter=None):
        """
        Print compact overview of all running background processes.
        Called by cli_v4 on /status or when user expands.

        Args:
            background_tasks: dict of BackgroundTask
            agents_getter: async callable(name) -> agent (for live state access)
        """
        running = [t for t in background_tasks.values() if t.status == "running"]
        if not running:
            self._print(f"<style fg='{C['dim']}'>  no background processes</style>")
            return

        self._print(f"\n<style fg='{C['cyan']}'>  {SYM['agent']} {len(running)} process(es) running</style>")

        for t in running:
            elapsed = time.time() - t.started_at.timestamp()
            elapsed_s = f"{elapsed:.0f}s"
            query_short = _short(t.query, 35)

            # Try to read engine.live if available
            live_info = ""
            if hasattr(t, '_agent_ref') and t._agent_ref:
                try:
                    eng = t._agent_ref._get_execution_engine()
                    live = eng.live
                    bar = _bar(live.iteration, live.max_iterations, 12)
                    phase = live.phase.value[:8]
                    tool = live.tool.name[:12] if live.tool.name else ""
                    live_info = f" {bar} {phase} {tool}"
                except Exception:
                    import traceback
                    traceback.print_exc()
                    pass
            else:
                live_info = "no agent ref!"

            self._print(
                f"  <style fg='{C['dim']}'>{SYM['progress']}</style> "
                f"<style fg='{C['cyan']}'>{t.agent_name[:12]:<12}</style> "
                f"<style fg='{C['dim']}'>{elapsed_s:>5} {_esc(query_short)}{_esc(live_info)}</style>"
            )

    def print_live_summary(self):
        """
        One-liner summary of current engine state.
        Called when minimized or for status bar.
        """
        if not self.engine:
            return
        live = self.engine.live
        if live.phase.value == "idle":
            return

        bar = _bar(live.iteration, live.max_iterations, 15)
        phase = live.phase.value
        thought = _short(live.thought, 50) if live.thought else ""
        tool = live.tool.name if live.tool.name else ""

        parts = [
            f"<style fg='{C['cyan']}'>{live.agent_name}</style>",
            f"<style fg='{C['dim']}'>{bar} {phase}</style>",
        ]
        if tool:
            parts.append(f"<style fg='{C['blue']}'>{SYM['tool']}{tool}</style>")
        if thought:
            parts.append(f"<style fg='{C['dim']}'>{SYM['think']}{_esc(thought[:40])}</style>")

        self._print("  ".join(parts))

    def print_final_summary(self, pane=None):
        """Druckt eine saubere Zusammenfassung nach dem Lauf (kein Datenmüll)."""
        self._print(f"\n<style fg='{C['cyan']}'>{'━' * 60}</style>")
        self._print(f"<style fg='{C['bright']} bold'>SESSION SUMMARY</style>")

        if pane:
            # Dateien auflisten
            if pane.files_touched:
                self._print(f"\n<style fg='{C['amber']}'>Modified Files:</style>")
                for f in sorted(pane.files_touched):
                    self._print(f"  <style fg='{C['green']}'>✓</style> <style fg='{C['white']}'>{f}</style>")

            # Statistiken
            dur = _fmt_elapsed(time.time() - pane.started_at)
            self._print(
                f"\n<style fg='{C['dim']}'>Duration: {dur} | Iterations: {pane.iteration} | Tools: {len(pane.tool_history)}</style>")

            # Finale Antwort (falls vorhanden) hervorheben
            final = pane.content_lines[-10:]  # Letzte Zeilen
            ans = "\n".join([line for line in final if line.strip()]).strip()
            if ans:
                self._print(f"\n<style fg='{C['cyan']}'>Final Answer:</style>")
                self._print(f"<style fg='{C['bright']}'>{_esc(ans)}</style>")

        self._print(f"<style fg='{C['cyan']}'>{'━' * 60}</style>\n")

    # -- Debug mode -----------------------------------------------------------

    def print_debug_panel(self):
        """
        Full debug panel: only shown when ENGINE_DEBUG=1.
        Shows raw live state, all fields.
        """
        if not DEBUG or not self.engine:
            return
        live = self.engine.live
        self._print(f"\n<style fg='{C['dim']}'>{'─' * 60}</style>")
        self._print(f"<style fg='{C['amber']}'>DEBUG</style> "
                     f"<style fg='{C['dim']}'>phase={live.phase.value} iter={live.iteration}/{live.max_iterations} "
                     f"elapsed={live.elapsed}s run={live.run_id}</style>")
        if live.thought:
            self._print(f"<style fg='{C['dim']}'>  thought: {_esc(_short(live.thought, 70))}</style>")
        if live.tool.name:
            self._print(f"<style fg='{C['dim']}'>  tool: {live.tool.name} args={_esc(_short(live.tool.args_summary, 50))}</style>")
        if live.status_msg:
            self._print(f"<style fg='{C['dim']}'>  status: {_esc(live.status_msg)}</style>")
        if live.error:
            self._print(f"<style fg='{C['red']}'>  error: {_esc(live.error)}</style>")
        if live.skills:
            self._print(f"<style fg='{C['dim']}'>  skills: {', '.join(live.skills)}</style>")
        if live.tools_loaded:
            self._print(f"<style fg='{C['dim']}'>  tools: {', '.join(live.tools_loaded[:8])}</style>")
        self._print(f"<style fg='{C['dim']}'>{'─' * 60}</style>")

    # -- internal rendering ---------------------------------------------------

    def _maybe_print_agent_header(self, chunk: dict):
        """Print agent context header on agent change, progress bar + token bar on iter change."""
        agent = chunk.get("agent", "")
        is_sub = chunk.get("is_sub", False)
        iter_n = chunk.get("iter", 0)
        max_n = chunk.get("max_iter", 0)

        # Sub-agent forwarded chunks carry an ID
        sub_id = chunk.get("_sub_agent_id", "")
        if sub_id:
            is_sub = True
            # Use sub_id as agent discriminator so header shows per sub-agent
            agent = f"{agent}:{sub_id}" if agent else sub_id

        if agent and agent != self._last_agent:
            prefix = SYM["sub"] if is_sub else SYM["agent"]
            color = C["dim"] if is_sub else C["cyan"]
            bar = _bar(iter_n, max_n, 12)
            self._print(
                f"\n<style fg='{color}'>{prefix} {_esc(agent)}</style>"
                f"  <style fg='{C['dim']}'>{bar} {iter_n}/{max_n}</style>"
            )
            self._print_token_bar(chunk)
            self._maybe_print_persona_skills(chunk)
            self._last_agent = agent
            self._last_iter = iter_n
            if DEBUG:
                self.print_debug_panel()

        elif iter_n > self._last_iter:
            bar = _bar(iter_n, max_n, 12)
            self._print(
                f"  <style fg='{C['dim']}'>{bar} {iter_n}/{max_n}</style>"
            )
            self._print_token_bar(chunk)
            self._last_iter = iter_n

    def _print_token_bar(self, chunk: dict):
        """Show working memory token usage bar after iteration line."""
        tokens_used = chunk.get("tokens_used", 0)
        tokens_max = chunk.get("tokens_max", 0)
        if tokens_max <= 0:
            return
        pct = min(100, int(100 * tokens_used / tokens_max))
        bar_w = 18
        filled = int(bar_w * tokens_used / tokens_max)
        bar_str = "\u2591" * filled + " " * (bar_w - filled)
        # Color: dim < 50%, amber 50-80%, red > 80%
        if pct < 50:
            col = C["dim"]
        elif pct < 80:
            col = C["amber"]
        else:
            col = C["red"]
        self._print(
            f"  <style fg='{col}'>Tokens: {tokens_used} [{bar_str}] / {tokens_max} ~ {pct}%</style>"
        )

    def _maybe_print_persona_skills(self, chunk: dict):
        """Show active persona and matched skills on agent header."""
        persona = chunk.get("persona", "default")
        persona_src = chunk.get("persona_source", "")
        persona_model = chunk.get("persona_model", "")
        persona_iterations_factor = chunk.get("persona_iterations_factor", 1)
        skills = chunk.get("skills", [])

        parts = []
        if persona and persona != "default":
            label = f"{persona}"
            if persona_src and persona_src != "default":
                label += f"\u00b7{persona_src}"
            if persona_model:
                label += f" [{persona_model}]"
            parts.append(f"<style fg='{C['blue']}'>\u2b21 {_esc(label)}</style>")

        if persona_iterations_factor != 1:
            text = f" [{'+' if persona_iterations_factor >= 1 else '-'}{persona_iterations_factor}% iterations]"
            parts.append(f"<style fg='{C['amber']}'>\u2b21 {_esc(text)}</style>")

        if skills:
            sk_str = ", ".join(skills[:3])
            if len(skills) > 3:
                sk_str += f" +{len(skills) - 3}"
            parts.append(f"<style fg='{C['dim']}'>\u2699 {_esc(sk_str)}</style>")

        if parts:
            self._print("  " + "  ".join(parts))

    def _on_reasoning(self, chunk: dict):
        if not self._in_think:
            # Show thought indicator, overwrite with \r
            print_formatted_text(
                HTML(f"  <style fg='{C['dim']}'>{SYM['think']} thinking...</style>"),
                end="\r",
            )
            self._in_think = True
        self._think_buf += chunk.get("chunk", "")
        words = [w for w in self._think_buf.split()]
        b = ""
        offset = 0
        for w in words:
            offset+=len(w)
            b += f"{w} "
            if offset > 12:
                self.thought_pool.append(b.strip())
                b = "" if random.randint(0, 10) < 9 else "..zen.. "
                offset = 0

    def _close_think(self):
        if self._in_think:
            # Update engine.live.thought if engine available
            if self.engine and self._think_buf:
                self.engine.live.thought = self._think_buf[:200]
            summary = _short(self._think_buf.strip().replace("\n", " "), 60)
            self._print(f"  <style fg='{C['dim']}'>{SYM['think']} {_esc(summary)}</style>          ")
            self._in_think = False
            self._think_buf = ""

    def _on_content(self, chunk: dict):
        self._close_think()
        text = chunk.get("chunk", "")
        print_formatted_text(HTML(f"<style fg='{C['white']}'>{_esc(text)}</style>"), end="", flush=True)
        self._needs_newline = True
        self._content_buf += chunk.get("chunk", "")
        words = [w for w in self._content_buf.split()]
        b = ""
        offset = 0
        for w in words:
            offset += len(w)
            b += f"{w} "
            if offset > 12:
                self.thought_pool.append(b.strip())
                b = "" if random.randint(0, 10) < 9 else "..zen.. "
                offset = 0

        if len(words) > 5:
            self._content_buf = ""

    def _on_tool_start(self, chunk: dict):
        self._close_think()
        name = chunk.get("name", "?")
        args = chunk.get("args", "")

        # think tool: show the actual thought, not just "..."
        if name == "think":
            thought_text = ""
            try:
                ad = json.loads(args) if isinstance(args, str) else args
                thought_text = ad.get("thought", "")
            except Exception:
                import traceback
                traceback.print_exc()
                pass
            if thought_text:
                summary = _short(thought_text.strip().replace("\n", " "), 70)
                self._print(
                    f"  <style fg='{C['dim']}'>{SYM['think']} {_esc(summary)}</style>",
                    end=""
                )
            else:
                self._print(
                    f"  <style fg='{C['dim']}'>{SYM['think']} thinking...</style>",
                    end=""
                )
            self._needs_newline = True
            return

        # final_answer: handled by _on_final_answer, just show start
        if name == "final_answer":
            return

        # Regular tools: extract key arg for compact display
        arg_str = ""
        if args:
            try:
                ad = json.loads(args) if isinstance(args, str) else args
                for k in ("path", "query", "command", "url", "filename", "task", "timeout", "tools", "category", "pattern"):
                    if k in ad:
                        arg_str += _short(str(ad[k]), 40)
                if not arg_str:
                    arg_str = ad[:15]+ "..." if isinstance(ad, str) else " - ".join(list(ad.keys())[:4])
            except Exception:
                import traceback
                traceback.print_exc()
                pass

        tool_color = C['blue'] if self._last_agent and ":" in self._last_agent else C['cyan']
        self._print(
            f"  <style fg='{tool_color}'>{SYM['tool']} {name:<14}</style>"
            f"  <style fg='{C['dim']}'>{_esc(arg_str)}</style>",
            end=""
        )

    def _on_tool_result(self, chunk: dict):
        name = chunk.get("name", "")
        if name == "final_answer":
            return

        # think tool: just close the line
        if name == "think":
            self._print(f"  <style fg='{C['green']}'>{SYM['ok']}</style>")
            return

        result = chunk.get("result", "")
        icon = SYM["ok"]
        color = C["green"]
        meta = ""

        try:
            try:
                rd = json.loads(result) if isinstance(result, str) else result

            except Exception as e:
                rd = result
            if not rd:
                rd = result
            if isinstance(rd, dict):
                meta = self._extract_tool_meta(name, rd)
                if not rd.get("success", True):
                    icon = SYM["fail"]
                    color = C["red"]
                    if not meta:
                        meta = _short(rd.get("error", "failed"), 40)
            elif isinstance(rd, list):
                meta = f"{len(rd)} items"
            elif isinstance(rd, str) and len(rd) > 0:
                meta = f"{len(rd)}ch"
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Plain text result
            if isinstance(result, str) and result:
                rlen = len(result)
                size = f" {rlen}b" if rlen < 1024 else f"{rlen / 1024:.1f}kb" if rlen < 1048576 else f"{rlen / 1048576:.1f}mb"
                meta = _short(result.strip().replace("\n", " "), 25) + size


        self._print(
            f"  <style fg='{C['dim']}'>{_esc(meta)}</style>"
            f"  <style fg='{color}'>{icon}</style>"
        )

    @staticmethod
    def _extract_tool_meta(name: str, rd: dict) -> str:
        """Extract compact, meaningful info from known tool result dicts."""
        n = name.lower()

        # --- Failure case (universal) ---
        if not rd.get("success", True):
            return _short(rd.get("error", "failed"), 40)

        # --- VFS file read ---
        if n in ("vfs_read", "vfs_open", "vfs_view"):
            content = rd.get("content", "")
            lines = rd.get("lines", rd.get("total_lines", 0))
            ft = rd.get("file_type", "")
            parts = []
            if lines:
                parts.append(f"{lines}L")
            elif content:
                parts.append(f"{len(content)}ch")
            if ft:
                parts.append(ft)
            return " ".join(parts) if parts else ""

        # --- VFS write/create ---
        if n in ("vfs_write", "vfs_create", "vfs_append"):

            s  = rd.get("size", rd.get("bytes", len(rd.get("content", ""))))

            lines = rd.get("lines", len(rd.get("content", "")))
            ft = rd.get("file_type", "")
            parts = []
            if s:
                size = f"{s}b" if s < 1024 else f"{s / 1024:.1f}kb" if s < 1048576 else f"{s / 1048576:.1f}mb"
                parts.append(size)
            if lines:
                parts.append(f"{lines}L")
            if ft:
                parts.append(ft)
            return " ".join(parts) if parts else "ok"

        # --- VFS edit ---
        if n == "vfs_edit":
            return f"lines {rd.get('line_start', '?')}-{rd.get('line_end', '?')}"

        # --- VFS list ---
        if n == "vfs_list":
            contents = rd.get("contents", rd.get("entries", []))
            if isinstance(contents, list):
                dirs = sum(1 for c in contents if isinstance(c, dict) and c.get("type") == "dir")
                files = len(contents) - dirs
                return f"{files}f {dirs}d"
            return ""

        # --- VFS delete/mkdir/rmdir/mv ---
        if n in ("vfs_delete", "vfs_mkdir", "vfs_rmdir", "vfs_mv"):
            return "ok"

        # --- VFS info ---
        if n == "vfs_info":
            size = rd.get("size", 0)
            ft = rd.get("file_type", rd.get("type", ""))
            lines = rd.get("lines", 0)
            parts = []
            if ft:
                parts.append(ft)
            if lines:
                parts.append(f"{lines}L")
            if size:
                parts.append(f"{size}b" if size < 1024 else f"{size / 1024:.1f}kb")
            return " ".join(parts) if parts else ""

        # --- VFS grep ---
        if n == "vfs_grep":
            matches = rd.get("matches", rd.get("count", ""))
            return f"{matches} matches" if matches else ""

        # --- VFS mount ---
        if n == "vfs_mount":
            indexed = rd.get("files_indexed", 0)
            mp = rd.get("mount_point", "")
            return f"{indexed} files \u2192 {_short(mp, 20)}" if indexed else _short(mp, 30)

        # --- VFS execute ---
        if n == "vfs_execute":
            rc = rd.get("return_code", rd.get("exit_code", "?"))
            stdout = rd.get("stdout", "")
            stderr = rd.get("stderr", "")
            out_lines = len(stdout.splitlines()) if stdout else 0
            err_lines = len(stderr.splitlines()) if stderr else 0
            parts = [f"rc={rc}"]
            if out_lines:
                parts.append(f"out:{out_lines}L")
            if err_lines:
                parts.append(f"err:{err_lines}L")
            return " ".join(parts)

        # --- Docker run ---
        if n == "docker_run":
            rc = rd.get("exit_code", rd.get("return_code", "?"))
            dur = rd.get("duration", "")
            parts = [f"exec={rc}"]
            if dur:
                parts.append(f"{dur:.1f}s" if isinstance(dur, float) else f"{dur}s")
            return " ".join(parts)

        # --- Docker status ---
        if n == "docker_status":
            running = rd.get("is_running", False)
            return "running" if running else "stopped"

        # --- fs_copy_to_vfs / fs_copy_from_vfs ---
        if n in ("fs_copy_to_vfs", "fs_copy_from_vfs", "fs_copy_dir_from_vfs"):
            size = rd.get("size", 0)
            vp = rd.get("vfs_path", rd.get("saved_path", ""))
            parts = []
            if size:
                parts.append(f"{size}b" if size < 1024 else f"{size / 1024:.1f}kb")
            if vp:
                parts.append(_short(vp, 25))
            return " ".join(parts) if parts else "ok"

        # --- VFS diagnostics ---
        if n == "vfs_diagnostics":
            errs = rd.get("errors", rd.get("error_count", 0))
            warns = rd.get("warnings", rd.get("warning_count", 0))
            hints = rd.get("hints", rd.get("hint_count", 0))
            return f"Er:{errs} Warn:{warns} Hits:{hints}"

        # --- list_tools ---
        if n == "list_tools":
            tools = rd.get("tools", rd.get("available", []))
            if isinstance(tools, list):
                return f"{len(tools)} tools"
            return ""

        # --- load_tools ---
        if n == "load_tools":
            loaded = rd.get("loaded", rd.get("tools", []))
            if isinstance(loaded, list):
                return ", ".join(loaded[:3]) + (f" +{len(loaded)-3}" if len(loaded) > 3 else "")
            return ""

        # --- history ---
        if n == "history":
            return ""

        # --- Sharing tools ---
        if "share" in n:
            sid = rd.get("id", rd.get("share_id", ""))
            return _short(str(sid), 20) if sid else "ok"

        # --- Discovery: list_tools ---
        if n == "list_tools":
            tools = rd.get("tools", rd.get("available", []))
            if isinstance(tools, list):
                return f"{len(tools)} tools"
            if isinstance(rd, str):
                return f"{rd.count(chr(10)) + 1} entries"
            return ""

        # --- Discovery: load_tools ---
        if n == "load_tools":
            loaded = rd.get("loaded", rd.get("tools", []))
            if isinstance(loaded, list):
                return ", ".join(loaded[:3]) + (f" +{len(loaded) - 3}" if len(loaded) > 3 else "")
            return _short(str(rd), 40) if isinstance(rd, str) else ""

        # --- Discovery: shift_focus ---
        if n == "shift_focus":
            nxt = rd.get("next_objective", "")
            return f"→ {_short(nxt, 40)}" if nxt else "ok"

        # --- Sub-Agent: spawn_sub_agent ---
        if n == "spawn_sub_agent":
            sid = rd.get("id", rd.get("sub_agent_id", ""))
            status = rd.get("status", "")
            out = rd.get("output_dir", "")
            if sid:
                return f"{_short(sid, 12)} {status}" if status else _short(sid, 20)
            return _short(out, 25) if out else ""

        # --- Sub-Agent: wait_for ---
        if n == "wait_for":
            results = rd.get("results", {})
            if isinstance(results, dict):
                ok = sum(1 for r in results.values() if isinstance(r, dict) and r.get("success"))
                fail = len(results) - ok
                return f"{ok}✓ {fail}✗" if fail else f"{ok}✓"
            return ""

        # --- Sub-Agent: resume_sub_agent ---
        if n == "resume_sub_agent":
            sid = rd.get("sub_agent_id", rd.get("id", ""))
            status = rd.get("status", "")
            parts = []
            if sid:
                parts.append(_short(sid, 12))
            if status:
                parts.append(status)
            return " ".join(parts) if parts else ""


        # =================================================================
        # DOCKER TOOLS
        # =================================================================

        if n == "docker_run":
            rc = rd.get("exit_code", rd.get("return_code", "?"))
            dur = rd.get("duration", "")
            parts = [f"rc={rc}"]
            if dur:
                parts.append(f"{dur:.1f}s" if isinstance(dur, (float, int)) else f"{dur}s")
            return " ".join(parts)

        if n == "docker_start_app":
            url = rd.get("url", "")
            port = rd.get("host_port", "")
            return f"{url}" if url else (f":{port}" if port else "started")

        if n in ("docker_stop_app",):
            return "stopped"

        if n == "docker_logs":
            logs = rd.get("logs", "")
            return f"{len(logs.splitlines())}L" if logs else "empty"

        if n == "docker_status":
            running = rd.get("is_running", False)
            return "running" if running else "stopped"

        # =================================================================
        # CHAIN TOOLS (chain_tools.py)
        # =================================================================

        if n == "create_validate_chain":
            cid = rd.get("id", rd.get("chain_id", ""))
            valid = rd.get("is_valid", rd.get("valid", None))
            steps = rd.get("step_count", rd.get("steps", ""))
            parts = []
            if cid:
                parts.append(_short(cid, 12))
            if valid is not None:
                parts.append("VALID" if valid else "INVALID")
            if steps:
                parts.append(f"{steps} steps")
            return " ".join(parts) if parts else ""

        if n == "run_chain":
            elapsed = rd.get("elapsed", rd.get("duration", ""))
            run_n = rd.get("run_count", rd.get("run", ""))
            parts = []
            if elapsed:
                parts.append(f"{elapsed:.1f}s" if isinstance(elapsed, (float, int)) else f"{elapsed}s")
            if run_n:
                parts.append(f"#{run_n}")
            return " ".join(parts) if parts else ""

        if n == "list_auto_get_fitting":
            count = rd.get("count", rd.get("total", ""))
            matched = rd.get("matched", rd.get("best", ""))
            if count:
                return f"{count} chains" + (f" best: {_short(str(matched), 15)}" if matched else "")
            return ""

        # =================================================================
        # GOOGLE CALENDAR TOOLS
        # =================================================================

        if n in ("calendar_login", "calendar_auth_callback"):
            return rd.get("status", "ok")

        if n == "calendar_auth_url":
            return "auth URL ready"

        if n == "calendar_list_events":
            events = rd.get("events", [])
            if isinstance(events, list):
                return f"{len(events)} events"
            return ""

        if n == "calendar_get_event":
            summary = rd.get("summary", rd.get("title", ""))
            start = rd.get("start", "")
            return _short(f"{summary} {start}", 40) if summary else ""

        if n == "calendar_create_event":
            summary = rd.get("summary", "")
            return _short(summary, 25) if summary else "created"

        if n == "calendar_update_event":
            return "updated"

        if n == "calendar_delete_event":
            return "deleted"

        if n == "calendar_find_free_slots":
            slots = rd.get("slots", rd.get("free_slots", []))
            if isinstance(slots, list):
                return f"{len(slots)} slots"
            return ""

        if n == "tasks_list_tasklists":
            lists = rd.get("tasklists", rd.get("lists", []))
            if isinstance(lists, list):
                return f"{len(lists)} lists"
            return ""

        if n == "tasks_list":
            tasks = rd.get("tasks", [])
            if isinstance(tasks, list):
                return f"{len(tasks)} tasks"
            return ""

        if n == "tasks_create":
            title = rd.get("title", "")
            return _short(title, 30) if title else "created"

        if n in ("tasks_complete", "tasks_update", "tasks_delete"):
            return "ok"

        # =================================================================
        # GOOGLE GMAIL TOOLS
        # =================================================================

        if n == "gmail_login":
            return rd.get("status", "ok")

        if n == "gmail_list":
            msgs = rd.get("messages", rd.get("emails", []))
            if isinstance(msgs, list):
                return f"{len(msgs)} emails"
            return ""

        if n == "gmail_read":
            subj = rd.get("subject", rd.get("Subject", ""))
            frm = rd.get("from", rd.get("From", ""))
            parts = []
            if subj:
                parts.append(_short(subj, 25))
            if frm:
                parts.append(_short(frm, 20))
            return " ".join(parts) if parts else ""

        if n == "gmail_send":
            to = rd.get("to", "")
            return f"\u2192 {_short(to, 25)}" if to else "sent"

        if n == "gmail_send_with_attachment":
            to = rd.get("to", "")
            att = rd.get("attachment", rd.get("filename", ""))
            parts = []
            if to:
                parts.append(f"\u2192 {_short(to, 20)}")
            if att:
                parts.append(f"\U0001f4ce {_short(att, 15)}")
            return " ".join(parts) if parts else "sent"

        if n == "gmail_search":
            results = rd.get("messages", rd.get("results", []))
            if isinstance(results, list):
                return f"{len(results)} found"
            return ""

        if n in ("gmail_mark_read", "gmail_mark_unread", "gmail_archive",
                  "gmail_trash", "gmail_modify_labels"):
            return "ok"

        if n == "gmail_reply":
            return "replied"

        # =================================================================
        # WEB BROWSER TOOLS (tooklit.py)
        # =================================================================

        if n == "tool_browser_start":
            headless = rd.get("headless", None)
            return f"headless={headless}" if headless is not None else "started"

        if n == "tool_browser_stop":
            return "stopped"

        if n == "tool_browser_status":
            running = rd.get("running", rd.get("is_running", False))
            url = rd.get("current_url", "")
            parts = ["running" if running else "stopped"]
            if url:
                parts.append(_short(url, 30))
            return " ".join(parts)

        if n == "tool_browser_set_headless":
            return f"headless={rd.get('headless', '?')}"

        if n in ("tool_web_search", "tool_search_site"):
            results = rd.get("results", [])
            if isinstance(results, list):
                return f"{len(results)} results"
            return ""

        if n == "tool_search_files":
            results = rd.get("results", [])
            if isinstance(results, list):
                return f"{len(results)} files"
            return ""

        if n == "tool_goto":
            title = rd.get("title", "")
            url = rd.get("url", rd.get("current_url", ""))
            return _short(title, 30) if title else _short(url, 35)

        if n in ("tool_back", "tool_refresh"):
            url = rd.get("url", rd.get("current_url", ""))
            return _short(url, 35) if url else "ok"

        if n == "tool_current_url":
            return _short(rd.get("url", rd.get("current_url", "")), 40)

        if n == "tool_click":
            return "clicked"

        if n == "tool_type":
            text = rd.get("text", rd.get("typed", ""))
            return f'"{_short(text, 25)}"' if text else "typed"

        if n in ("tool_select", "tool_scroll", "tool_scroll_to_bottom",
                  "tool_wait", "tool_hover"):
            return "ok"

        if n == "tool_extract":
            text = rd.get("text", rd.get("content", ""))
            links = rd.get("links", [])
            parts = []
            if text:
                parts.append(f"{len(text)}ch")
            if isinstance(links, list) and links:
                parts.append(f"{len(links)} links")
            return " ".join(parts) if parts else ""

        if n == "tool_extract_text":
            text = rd.get("text", rd.get("content", ""))
            return f"{len(text)}ch" if text else ""

        if n == "tool_extract_html":
            html = rd.get("html", rd.get("content", ""))
            return f"{len(html)}ch HTML" if html else ""

        if n == "tool_extract_links":
            links = rd.get("links", [])
            return f"{len(links)} links" if isinstance(links, list) else ""

        if n == "tool_extract_attribute":
            val = rd.get("value", rd.get("result", ""))
            return _short(str(val), 30) if val else ""

        if n == "tool_scrape_url":
            title = rd.get("title", "")
            content = rd.get("content", rd.get("text", ""))
            parts = []
            if title:
                parts.append(_short(title, 25))
            if content:
                parts.append(f"{len(content)}ch")
            return " ".join(parts) if parts else ""

        if n in ("tool_session_save", "tool_session_load"):
            return _short(rd.get("name", ""), 20) or "ok"

        # =================================================================
        # DESKTOP AUTOMATION TOOLS (destop_auto.py)
        # =================================================================

        if n == "scout_interface":
            apps = rd.get("open_applications", [])
            active = rd.get("active_application", {})
            active_name = active.get("name", "") if isinstance(active, dict) else ""
            interact = (rd.get("possible_actions", {}) or {}).get("interact", [])
            parts = []
            if isinstance(apps, list):
                parts.append(f"{len(apps)} apps")
            if active_name:
                parts.append(f"active: {_short(active_name, 15)}")
            if isinstance(interact, list) and interact:
                parts.append(f"{len(interact)} actions")
            return " ".join(parts) if parts else ""

        if n == "execute_action":
            status = rd.get("status", "")
            result = rd.get("result", rd.get("message", ""))
            if status == "success":
                return _short(str(result), 35) if result else "ok"
            return _short(str(result), 35) if result else status

        # =================================================================
        # SITUATION / BEHAVIOR TOOLS
        # =================================================================

        if n == "set_agent_situation":
            return _short(rd.get("intent", ""), 35) or "ok"

        if n == "check_permissions":
            allowed = rd.get("allowed", None)
            rule = rd.get("rule", "")
            if allowed is not None:
                icon = "\u2713" if allowed else "\u2717"
                return f"{icon} {_short(rule, 25)}" if rule else icon
            return ""

        if n == "history":
            return ""

        # --- Generic fallback: show keys ---
        if "message" in rd:
            return _short(str(rd["message"]), 45)
        if "info" in rd:
            return _short(str(rd["info"]), 45)
        if "content" in rd:
            s = len(rd["content"])
            return f"{s}b" if s < 1024 else f"{s / 1024:.1f}kb" if s < 1048576 else f"{s / 1048576:.1f}mb"
        if "size" in rd:
            s = rd["size"]
            return f"{s}b" if s < 1024 else f"{s / 1024:.1f}kb" if s < 1048576 else f"{s / 1048576:.1f}mb"
        if "result" in rd:
            return _short(str(rd["result"]), 45)
        return ""

    def _on_final_answer(self, chunk: dict):
        self._close_think()
        answer = chunk.get("answer", "")
        self._print(f"\n<style fg='{C['bright']}'>{_esc(answer)}</style>\n")

    def _on_done(self, chunk: dict):
        success = chunk.get("success", True)
        color = C["green"] if success else C["red"]
        iters = chunk.get("iter", 0)

        # Show elapsed if engine available
        elapsed = ""
        if self.engine:
            elapsed = f" {self.engine.live.elapsed}s"

        self._print(
            f"\n  <style fg='{color}'>{SYM['done']} complete</style>"
            f"  <style fg='{C['dim']}'>{iters} iter{elapsed}</style>\n"
        )

        # Debug: final panel
        if DEBUG:
            self.print_debug_panel()

    # -- output ---------------------------------------------------------------

    def _print(self, html_str: str, end="\n", **k):
        import re

        # Nicht in ZenPlus direkt printen
        if self._zen_plus and self._zen_plus.active:
            return

        try:
            print_formatted_text(HTML(html_str), end=end, **k)
        except Exception:
            plain = re.sub(r"<[^>]+>", "", html_str)
            print(plain, end=end, flush=True)




"""
Zen Terminal CLI Output Renderer
Pure Python, zero external dependencies.
ANSI-based dark terminal style inspired by TBJS Design System v3.0.
"""

from typing import AsyncGenerator, Any
import datetime
import textwrap
import re


# =============================================================================
# ANSI COLOR ENGINE -- Pure Python, no external libs
# =============================================================================

class _Ansi:
    """Low-level ANSI escape code factory."""
    ESC = "\x1b["

    @staticmethod
    def _code(*codes: int) -> str:
        return f"{_Ansi.ESC}{';'.join(map(str, codes))}m"

    @classmethod
    def reset(cls) -> str:
        return cls._code(0)

    @classmethod
    def bold(cls) -> str:
        return cls._code(1)

    @classmethod
    def dim(cls) -> str:
        return cls._code(2)

    @classmethod
    def italic(cls) -> str:
        return cls._code(3)

    @classmethod
    def underline(cls) -> str:
        return cls._code(4)

    @classmethod
    def fg(cls, r: int, g: int, b: int) -> str:
        return cls._code(38, 2, r, g, b)

    @classmethod
    def bg(cls, r: int, g: int, b: int) -> str:
        return cls._code(48, 2, r, g, b)

    @classmethod
    def fg_256(cls, n: int) -> str:
        return cls._code(38, 5, n)

    @classmethod
    def bg_256(cls, n: int) -> str:
        return cls._code(48, 5, n)


# =============================================================================
# TBJS TERMINAL DESIGN TOKENS -- Dark Terminal Variant
# =============================================================================

class T:
    """Design tokens -- colors, spacing, typography."""

    # -- Colors (RGB) ------------------------------------------------------
    BG          = (0, 0, 0)           # True black canvas
    BG_RAISED   = (10, 10, 15)        # Panels, cards
    BG_SUNKEN   = (3, 3, 6)          # Inputs, code blocks

    # Phosphor-bright accent -- deep tech blue (OKLCH 55% 0.18 230)
    PRIMARY     = (59, 130, 246)      # #3B82F6
    SUCCESS     = (34, 197, 94)       # #22C55E
    WARNING     = (234, 179, 8)       # #EAB308
    ERROR       = (239, 68, 68)       # #EF4444
    INFO        = (96, 165, 250)      # #60A5FA

    # Text hierarchy
    FG          = (235, 235, 235)     # Body -- 0.92 brightness
    FG_DIM      = (128, 128, 128)     # Labels, timestamps
    FG_MUTED    = (77, 77, 77)        # Hints, dividers

    # -- Typography --------------------------------------------------------
    H1          = 16
    H2          = 14
    H3          = 13
    BASE        = 12
    SM          = 11
    XS          = 10

    # -- Spacing (4px rhythm) ---------------------------------------------
    S1          = 2
    S2          = 4
    S3          = 8
    S4          = 12
    S5          = 16
    S6          = 24

    # -- Box-drawing characters -------------------------------------------
    HORIZ       = "\u2500"
    VERT        = "\u2502"
    CORNER_TL   = "\u250c"
    CORNER_TR   = "\u2510"
    CORNER_BL   = "\u2514"
    CORNER_BR   = "\u2518"
    T_DOWN      = "\u252c"
    T_UP        = "\u2534"
    T_RIGHT     = "\u251c"
    T_LEFT      = "\u2524"
    CROSS       = "\u253c"

    # -- Status glyphs ------------------------------------------------------
    DOT_ON      = "\u25cf"
    DOT_OFF     = "\u25cb"
    DOT_PEND    = "\u25d0"
    CROSS_G     = "\u2715"
    WARN_G      = "\u25b2"
    ARROW       = "\u2192"
    CARET       = "\u258c"
    BULLET      = "\u2022"
    CHECK       = "\u2713"
    CHEVRON     = "\u203a"
    PROMPT      = "$"
    PROMPT_ROOT = "#"

    # -- Width ------------------------------------------------------------
    TERM_WIDTH  = 80


# =============================================================================
# STYLE HELPER -- Fluent API for colored text
# =============================================================================

class Style:
    """Fluent style builder. Chainable: Style().bold().blue().text("hello")"""

    def __init__(self):
        self._codes: list[int] = []
        self._fg: tuple[int, int, int] | None = None
        self._bg: tuple[int, int, int] | None = None

    def _clone(self) -> "Style":
        s = Style()
        s._codes = self._codes.copy()
        s._fg = self._fg
        s._bg = self._bg
        return s

    def bold(self) -> "Style":
        s = self._clone()
        s._codes.append(1)
        return s

    def dim(self) -> "Style":
        s = self._clone()
        s._codes.append(2)
        return s

    def italic(self) -> "Style":
        s = self._clone()
        s._codes.append(3)
        return s

    def underline(self) -> "Style":
        s = self._clone()
        s._codes.append(4)
        return s

    def fg(self, r: int, g: int, b: int) -> "Style":
        s = self._clone()
        s._fg = (r, g, b)
        return s

    def bg(self, r: int, g: int, b: int) -> "Style":
        s = self._clone()
        s._bg = (r, g, b)
        return s

    def _build(self) -> str:
        parts = []
        if self._codes:
            parts.append(_Ansi._code(*self._codes))
        if self._fg:
            parts.append(_Ansi.fg(*self._fg))
        if self._bg:
            parts.append(_Ansi.bg(*self._bg))
        return "".join(parts)

    def text(self, s: str) -> str:
        return f"{self._build()}{s}{_Ansi.reset()}"

    def __call__(self, s: str) -> str:
        return self.text(s)

    # -- Preset colors (static methods for convenience) --------------------
    @staticmethod
    def primary(s: str = "") -> str:
        return Style().fg(*T.PRIMARY).text(s) if s else Style().fg(*T.PRIMARY)._build()

    @staticmethod
    def success(s: str = "") -> str:
        return Style().fg(*T.SUCCESS).text(s) if s else Style().fg(*T.SUCCESS)._build()

    @staticmethod
    def warning(s: str = "") -> str:
        return Style().fg(*T.WARNING).text(s) if s else Style().fg(*T.WARNING)._build()

    @staticmethod
    def error(s: str = "") -> str:
        return Style().fg(*T.ERROR).text(s) if s else Style().fg(*T.ERROR)._build()

    @staticmethod
    def info(s: str = "") -> str:
        return Style().fg(*T.INFO).text(s) if s else Style().fg(*T.INFO)._build()

    @staticmethod
    def fg_color(s: str = "") -> str:
        return Style().fg(*T.FG).text(s) if s else Style().fg(*T.FG)._build()

    @staticmethod
    def dim_color(s: str = "") -> str:
        return Style().fg(*T.FG_DIM).text(s) if s else Style().fg(*T.FG_DIM)._build()

    @staticmethod
    def muted(s: str = "") -> str:
        return Style().fg(*T.FG_MUTED).text(s) if s else Style().fg(*T.FG_MUTED)._build()

    @staticmethod
    def bold_primary(s: str) -> str:
        return Style().bold().fg(*T.PRIMARY).text(s)

    @staticmethod
    def bold_success(s: str) -> str:
        return Style().bold().fg(*T.SUCCESS).text(s)

    @staticmethod
    def bold_error(s: str) -> str:
        return Style().bold().fg(*T.ERROR).text(s)

    @staticmethod
    def bold_warning(s: str) -> str:
        return Style().bold().fg(*T.WARNING).text(s)

    @staticmethod
    def bg_primary(s: str) -> str:
        return f"{Style().bg(*T.PRIMARY).fg(*T.BG)._build()}{s}{_Ansi.reset()}"

    @staticmethod
    def bg_error(s: str) -> str:
        return f"{Style().bg(*T.ERROR).fg(*T.BG)._build()}{s}{_Ansi.reset()}"


# =============================================================================
# TERMINAL UI COMPONENTS -- Box drawing, panels, tables, status bars
# =============================================================================

class UI:
    """Terminal UI component factory -- pure ANSI, no external deps."""

    @staticmethod
    def _repeat(char: str, n: int) -> str:
        return char * max(0, n)

    @staticmethod
    def _strip_ansi(s: str) -> str:
        """Remove ANSI codes for width calculation."""
        return re.sub(r"\x1b\[[0-9;]*m", "", s)

    @staticmethod
    def _visible_width(s: str) -> int:
        """Width of string without ANSI codes."""
        return len(UI._strip_ansi(s))

    @staticmethod
    def _pad(s: str, width: int, align: str = "left") -> str:
        """Pad string to exact visible width, accounting for ANSI."""
        vis = UI._visible_width(s)
        if vis >= width:
            return s
        pad = " " * (width - vis)
        if align == "right":
            return pad + s
        elif align == "center":
            left = len(pad) // 2
            return " " * left + s + " " * (len(pad) - left)
        return s + pad

    # -- Horizontal rules --------------------------------------------------
    @staticmethod
    def hr(width: int = T.TERM_WIDTH, char: str = T.HORIZ,
           color: tuple[int, int, int] = T.FG_MUTED) -> str:
        line = UI._repeat(char, width)
        return Style().fg(*color).text(line)

    @staticmethod
    def hr_accent(width: int = T.TERM_WIDTH) -> str:
        return UI.hr(width, T.HORIZ, T.PRIMARY)

    @staticmethod
    def hr_double(width: int = T.TERM_WIDTH) -> str:
        return UI.hr(width, "\u2550", T.FG_MUTED)

    # -- Box frames --------------------------------------------------------
    @staticmethod
    def box_top(width: int = T.TERM_WIDTH,
                title: str = "",
                color: tuple[int, int, int] = T.PRIMARY) -> str:
        if title:
            title_vis = UI._visible_width(title)
            pad_left = 2
            pad_right = width - title_vis - pad_left - 2
            left = f"{T.CORNER_TL}{UI._repeat(T.HORIZ, pad_left)}"
            right = f"{UI._repeat(T.HORIZ, pad_right)}{T.CORNER_TR}"
            return f"{Style().fg(*color).text(left)}{title}{Style().fg(*color).text(right)}"
        return Style().fg(*color).text(f"{T.CORNER_TL}{UI._repeat(T.HORIZ, width - 2)}{T.CORNER_TR}")

    @staticmethod
    def box_bottom(width: int = T.TERM_WIDTH,
                   color: tuple[int, int, int] = T.PRIMARY) -> str:
        return Style().fg(*color).text(
            f"{T.CORNER_BL}{UI._repeat(T.HORIZ, width - 2)}{T.CORNER_BR}"
        )

    @staticmethod
    def box_line(content: str,
                 width: int = T.TERM_WIDTH,
                 color: tuple[int, int, int] = T.PRIMARY,
                 pad: int = 2) -> str:
        vis = UI._visible_width(content)
        avail = width - 2 - (pad * 2) - vis
        right_pad = " " * max(0, avail)
        return f"{Style().fg(*color).text(T.VERT)}{' ' * pad}{content}{right_pad}{' ' * pad}{Style().fg(*color).text(T.VERT)}"

    @staticmethod
    def box(content_lines: list[str],
            width: int = T.TERM_WIDTH,
            title: str = "",
            color: tuple[int, int, int] = T.PRIMARY) -> str:
        """Render a full box with content."""
        lines = [UI.box_top(width, title, color)]
        for line in content_lines:
            lines.append(UI.box_line(line, width, color))
        lines.append(UI.box_bottom(width, color))
        return "\n".join(lines)

    # -- Section headers ---------------------------------------------------
    @staticmethod
    def h1(text: str) -> str:
        prefix = Style.muted("# ")
        return f"\n{prefix}{Style().bold().fg(*T.FG).text(text)}"

    @staticmethod
    def h2(text: str) -> str:
        prefix = Style.muted("## ")
        return f"{prefix}{Style().fg(*T.FG).text(text)}"

    @staticmethod
    def h3(text: str) -> str:
        prefix = Style.muted("### ")
        return f"{prefix}{Style().fg(*T.FG_DIM).text(text)}"

    @staticmethod
    def prompt(text: str, is_root: bool = False) -> str:
        p = T.PROMPT_ROOT if is_root else T.PROMPT
        return f"{Style.success(p)} {Style.primary(text)}"

    # -- Labels & badges ---------------------------------------------------
    @staticmethod
    def label(text: str, color: tuple[int, int, int] = T.PRIMARY) -> str:
        """Bracketed label: [ LABEL ]"""
        brackets = Style.muted("[ ") + Style().fg(*color).text(text) + Style.muted(" ]")
        return brackets

    @staticmethod
    def badge(text: str, bg_color: tuple[int, int, int] = T.PRIMARY,
              fg_color: tuple[int, int, int] = T.BG) -> str:
        """Solid background badge."""
        return f"{Style().bg(*bg_color).fg(*fg_color)._build()} {text} {_Ansi.reset()}"

    @staticmethod
    def tag(text: str, color: tuple[int, int, int] = T.PRIMARY) -> str:
        """Inline tag: <tag>"""
        return f"{Style.muted('<')}{Style().fg(*color).text(text)}{Style.muted('>')}"

    # -- Status indicators -------------------------------------------------
    @staticmethod
    def status_online(label: str = "") -> str:
        dot = Style.success(T.DOT_ON)
        return f"{dot} {Style.fg_color(label)}" if label else dot

    @staticmethod
    def status_idle(label: str = "") -> str:
        dot = Style.muted(T.DOT_OFF)
        return f"{dot} {Style.dim_color(label)}" if label else dot

    @staticmethod
    def status_pending(label: str = "") -> str:
        dot = Style.warning(T.DOT_PEND)
        return f"{dot} {Style.warning(label)}" if label else dot

    @staticmethod
    def status_error(label: str = "") -> str:
        dot = Style.error(T.CROSS_G)
        return f"{dot} {Style.error(label)}" if label else dot

    @staticmethod
    def status_warn(label: str = "") -> str:
        dot = Style.warning(T.WARN_G)
        return f"{dot} {Style.warning(label)}" if label else dot

    # -- Timestamp ---------------------------------------------------------
    @staticmethod
    def timestamp(dt: datetime.datetime | None = None) -> str:
        if dt is None:
            dt = datetime.datetime.now(datetime.timezone.utc)
        ts = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        return Style.dim_color(ts)

    # -- Progress / meter --------------------------------------------------
    @staticmethod
    def meter(value: float, max_val: float = 100.0, width: int = 30,
              filled_color: tuple[int, int, int] = T.PRIMARY,
              empty_color: tuple[int, int, int] = T.FG_MUTED) -> str:
        ratio = min(1.0, max(0.0, value / max_val)) if max_val else 0
        filled = int(width * ratio)
        empty = width - filled
        bar = (Style().fg(*filled_color).text(UI._repeat("\u2588", filled)) +
               Style().fg(*empty_color).text(UI._repeat("\u2591", empty)))
        pct = f" {ratio * 100:.1f}%"
        return f"{bar}{Style.dim_color(pct)}"

    # -- Key-value pair ----------------------------------------------------
    @staticmethod
    def kv(key: str, value: str, key_width: int = 20) -> str:
        k = Style().fg(*T.FG_DIM).text(UI._pad(key + ":", key_width, "right"))
        v = Style().fg(*T.FG).text(value)
        return f"  {k} {v}"

    # -- Breadcrumb nav ----------------------------------------------------
    @staticmethod
    def breadcrumb(*parts: str) -> str:
        result = []
        for i, part in enumerate(parts):
            if i > 0:
                result.append(Style.muted(" / "))
            if i == len(parts) - 1:
                result.append(Style.fg_color(part))
            else:
                result.append(Style.primary(part))
        return "".join(result)

    # -- Table row ---------------------------------------------------------
    @staticmethod
    def table_row(*cells: str, widths: list[int] | None = None,
                  colors: list[tuple[int, int, int]] | None = None,
                  align: list[str] | None = None) -> str:
        if widths is None:
            widths = [20] * len(cells)
        if colors is None:
            colors = [T.FG] * len(cells)
        if align is None:
            align = ["left"] * len(cells)

        parts = []
        for i, (cell, w, c, a) in enumerate(zip(cells, widths, colors, align)):
            padded = UI._pad(cell, w, a)
            parts.append(Style().fg(*c).text(padded))

        sep = Style.muted(T.VERT)
        return f" {sep} ".join(parts)

    # -- Code block --------------------------------------------------------
    @staticmethod
    def code_block(lines: list[str], lang: str = "") -> str:
        header = f"{Style.muted('```')}{Style.primary(lang)}" if lang else Style.muted("```")
        footer = Style.muted("```")
        body = "\n".join(f"  {Style().fg(*T.FG_DIM).text(line)}" for line in lines)
        return f"{header}\n{body}\n{footer}"

    # -- Inline code -------------------------------------------------------
    @staticmethod
    def inline_code(text: str) -> str:
        return f"{Style.muted('`')}{Style().fg(*T.PRIMARY).text(text)}{Style.muted('`')}"

    # -- List item ---------------------------------------------------------
    @staticmethod
    def li(text: str, indent: int = 0, bullet: str = T.BULLET) -> str:
        indent_str = "  " * indent
        b = Style.muted(bullet)
        return f"{indent_str}{b} {Style.fg_color(text)}"

    # -- Quote / Note --------------------------------------------------------
    @staticmethod
    def note(text: str, color: tuple[int, int, int] = T.INFO) -> str:
        bar = Style().fg(*color).text(T.VERT)
        return f"{bar} {Style().fg(*T.FG_DIM).text(text)}"

    # -- Separator with label ----------------------------------------------
    @staticmethod
    def sep_label(label: str, width: int = T.TERM_WIDTH,
                  color: tuple[int, int, int] = T.FG_MUTED) -> str:
        vis = UI._visible_width(label)
        pad = (width - vis - 4) // 2
        left = UI._repeat(T.HORIZ, pad)
        right = UI._repeat(T.HORIZ, width - pad - vis - 4)
        return f"{Style().fg(*color).text(left)}  {Style().fg(*T.FG_DIM).text(label)}  {Style().fg(*color).text(right)}"

    # -- Compact info line -------------------------------------------------
    @staticmethod
    def info_line(*segments: tuple[str, tuple[int, int, int] | None]) -> str:
        """segments: [(text, color_or_None), ...] -- None = muted separator"""
        parts = []
        for text, color in segments:
            if color is None:
                parts.append(Style.muted(text))
            else:
                parts.append(Style().fg(*color).text(text))
        return " ".join(parts)


# =============================================================================
# ZEN STREAM RENDERER -- The main streaming output engine
# =============================================================================

class ZenRenderer:
    """
    Zen Terminal CLI Output Renderer.
    Converts agent streaming chunks into beautifully formatted terminal output.
    Pure Python, ANSI-only, zero external dependencies.
    """

    def __init__(self, term_width: int = T.TERM_WIDTH):
        self.width = term_width
        self._iteration = 0
        self._max_iter = 0
        self._tool_count = 0
        self._start_time = datetime.datetime.now(datetime.timezone.utc)
        self._last_narrator: str = ""

    def _header(self, title: str, color: tuple[int, int, int] = T.PRIMARY) -> str:
        """Agent response header with box frame."""
        agent_label = Style().bg(*color).fg(*T.BG)._build()
        agent_name = f" {T.PROMPT} ISAA "
        agent_reset = _Ansi.reset()

        top = UI.box_top(self.width, title=f"{agent_label}{agent_name}{agent_reset}", color=color)
        return top

    def _footer(self, color: tuple[int, int, int] = T.PRIMARY) -> str:
        return UI.box_bottom(self.width, color)

    def _elapsed(self) -> str:
        elapsed = datetime.datetime.now(datetime.timezone.utc) - self._start_time
        secs = elapsed.total_seconds()
        if secs < 60:
            return f"{secs:.1f}s"
        return f"{int(secs // 60)}m {secs % 60:.0f}s"

    # -- Chunk renderers -----------------------------------------------------

    def render_content(self, chunk: str) -> str:
        """Raw content stream (LLM tokens)."""
        return Style.fg_color(chunk)

    def render_reasoning(self, chunk: str) -> str:
        """Reasoning / thought process."""
        text = Style().fg(*T.SUCCESS).text(chunk)
        return f"{text}"

    def render_tool_start(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Tool invocation start."""
        self._tool_count += 1

        lines = [""]
        # Header line with tool number and name
        num = Style().fg(*T.FG_MUTED).text(f"#{self._tool_count:02d}")
        icon = Style().fg(*T.WARNING).text("\u2699")
        name_styled = Style.bold_primary(name)
        header = f"  {num} {icon} {Style.muted('TOOL')} {name_styled}"
        lines.append(header)

        # Arguments
        if args:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"data": args}
            lines.append(f"  {Style.muted('  ' + T.VERT)}")
            for k, v in args.items():
                val_str = str(v)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                key = Style().fg(*T.FG_DIM).text(f"    {T.VERT}  {k}:")
                val = Style().fg(*T.FG).text(val_str)
                lines.append(f"{key} {val}")
            lines.append(f"  {Style.muted('  ' + T.VERT)}")

        # Status line
        status = Style().fg(*T.WARNING).text(T.DOT_PEND)
        status_text = Style().fg(*T.WARNING).text("executing...")
        lines.append(f"  {Style.muted('  ' + T.CORNER_BL + T.HORIZ)} {status} {status_text}")

        return "\n".join(lines)

    def render_tool_result(self, result: Any, duration_ms: float | None = None) -> str:
        """Tool execution result."""
        result_str = str(result) if result is not None else ""

        # Truncate long results
        if len(result_str) > 300:
            result_str = result_str[:297] + "..."

        lines = []
        icon = Style().fg(*T.SUCCESS).text(T.CHECK)
        label = Style().fg(*T.SUCCESS).text("RESULT")

        if duration_ms is not None:
            dur = Style().fg(*T.FG_MUTED).text(f"({duration_ms:.0f}ms)")
            lines.append(f"  {Style.muted('     ' + T.CORNER_BL + T.HORIZ)} {icon} {label} {dur}")
        else:
            lines.append(f"  {Style.muted('     ' + T.CORNER_BL + T.HORIZ)} {icon} {label}")

        # Result content in indented block
        if result_str:
            wrapped = textwrap.wrap(result_str, width=self.width - 12)
            for wline in wrapped:
                lines.append(f"  {Style.muted('        ' + T.VERT)} {Style().fg(*T.FG_DIM).text(wline)}")

        return "\n".join(lines)

    def render_tool_error(self, error: str) -> str:
        """Tool execution error."""
        lines = []
        icon = Style().fg(*T.ERROR).text(T.CROSS_G)
        label = Style().fg(*T.ERROR).text("ERROR")
        lines.append(f"  {Style.muted('     ' + T.CORNER_BL + T.HORIZ)} {icon} {label}")

        wrapped = textwrap.wrap(error, width=self.width - 12)
        for wline in wrapped:
            lines.append(f"  {Style.muted('        ' + T.VERT)} {Style().fg(*T.ERROR).text(wline)}")

        return "\n".join(lines)

    def render_final_answer(self, answer: str) -> str:
        """Final agent response."""
        lines = [
            "",
            UI.hr_accent(self.width),
            "",
        ]

        # Header
        icon = Style().fg(*T.PRIMARY).text("\u270d")
        label = Style.bold_primary("ANTWORT")
        lines.append(f"  {icon} {label}")
        lines.append("")

        # Content in box
        content_lines = answer.split("\n")
        for line in content_lines:
            if line.strip():
                lines.append(f"  {Style.fg_color(line)}")
            else:
                lines.append("")

        lines.append("")
        lines.append(UI.hr_accent(self.width))

        return "\n".join(lines)

    def render_iteration_start(self, iteration: int, max_iter: int) -> str:
        """Iteration header."""
        self._iteration = iteration
        self._max_iter = max_iter

        lines = [""]

        # Separator with iteration info
        iter_label = f" ITERATION {iteration}/{max_iter} "
        lines.append(UI.sep_label(iter_label, self.width, T.PRIMARY))

        # Progress meter
        meter = UI.meter(iteration, max_iter, width=40)
        elapsed = Style().fg(*T.FG_MUTED).text(f"elapsed: {self._elapsed()}")
        lines.append(f"  {meter}  {elapsed}")

        return "\n".join(lines)

    def render_status(self, msg: str) -> str:
        """Status message (loading, processing, etc.)."""
        icon = Style().fg(*T.INFO).text(T.DOT_PEND)
        label = Style().fg(*T.INFO).text("STATUS")
        text = Style().fg(*T.FG_DIM).text(msg)
        return f"\n  {icon} {label} {text}"

    def render_post_processing(self, msg: str) -> str:
        """Post-processing status."""
        icon = Style().fg(*T.PRIMARY).text("\u270e")
        label = Style().fg(*T.PRIMARY).text("SAVE")
        text = Style().fg(*T.FG_DIM).text(msg)
        return f"\n  {icon} {label} {text}"

    def render_paused(self, run_id: str) -> str:
        """Execution paused."""
        lines = [""]
        lines.append(UI.hr(self.width, T.HORIZ, T.WARNING))

        icon = Style().fg(*T.WARNING).text("\u23f8")
        label = Style.bold_warning("PAUSIERT")
        lines.append(f"  {icon} {label}")

        lines.append(UI.kv("run_id", run_id, 12))
        lines.append(UI.kv("status", "waiting for human input", 12))

        lines.append(UI.hr(self.width, T.HORIZ, T.WARNING))
        return "\n".join(lines)

    def render_max_iterations(self, answer: str) -> str:
        """Max iterations reached."""
        lines = [""]
        lines.append(UI.hr(self.width, "\u2550", T.WARNING))

        icon = Style().fg(*T.WARNING).text(T.WARN_G)
        label = Style.bold_warning("MAX ITERATIONEN ERREICHT")
        lines.append(f"  {icon} {label}")

        lines.append("")
        lines.append(f"  {Style().fg(*T.FG_DIM).text('Letzte Antwort:')}")

        wrapped = textwrap.wrap(answer, width=self.width - 4)
        for wline in wrapped:
            lines.append(f"  {Style.fg_color(wline)}")

        lines.append(UI.hr(self.width, "\u2550", T.WARNING))
        return "\n".join(lines)

    def render_error(self, error: str) -> str:
        """Critical error."""
        lines = [""]
        lines.append(UI.hr(self.width, "\u2550", T.ERROR))

        icon = Style().fg(*T.ERROR).text("\u2718")
        label = Style.bold_error("FEHLER")
        lines.append(f"  {icon} {label}")

        lines.append("")
        wrapped = textwrap.wrap(error, width=self.width - 4)
        for wline in wrapped:
            lines.append(f"  {Style().fg(*T.ERROR).text(wline)}")

        lines.append(UI.hr(self.width, "\u2550", T.ERROR))
        return "\n".join(lines)

    def render_narrator(self, msg: str) -> str:
        """Narrator thought stream -- inline overwrite."""
        self._last_narrator = msg
        icon = Style().fg(*T.FG_MUTED).text("\u270c")
        text = Style().fg(*T.FG_DIM).text(msg[:80])
        # \r for inline overwrite in terminal
        return f"\r  {icon} {text}\r"

    def render_done(self, success: bool, meta: dict[str, Any] | None = None) -> str:
        """Execution complete."""
        lines = [""]

        if success:
            icon = Style().fg(*T.SUCCESS).text(T.DOT_ON)
            label = Style.bold_success("ABGESCHLOSSEN")
            color = T.SUCCESS
        else:
            icon = Style().fg(*T.WARNING).text(T.WARN_G)
            label = Style.bold_warning("ABGESCHLOSSEN (MIT WARNUNGEN)")
            color = T.WARNING

        lines.append(UI.hr(self.width, "\u2550", color))
        lines.append(f"  {icon} {label}")

        # Meta info
        if meta:
            lines.append("")
            for k, v in meta.items():
                lines.append(UI.kv(k, str(v), 16))

        # Summary stats
        lines.append("")
        stats = (
            Style.dim_color("iterations") + " " + Style.muted(T.VERT) + " " +
            Style.fg_color(f"{self._iteration}/{self._max_iter}") + "   " +
            Style.dim_color("tools") + " " + Style.muted(T.VERT) + " " +
            Style.fg_color(str(self._tool_count)) + "   " +
            Style.dim_color("elapsed") + " " + Style.muted(T.VERT) + " " +
            Style.fg_color(self._elapsed())
        )
        lines.append(f"  {stats}")

        lines.append(UI.hr(self.width, "\u2550", color))
        return "\n".join(lines)

    def render_user_input(self, query: str) -> str:
        """User query display."""
        lines = [""]
        lines.append(UI.hr(self.width, T.HORIZ, T.FG_MUTED))

        prompt = Style().fg(*T.SUCCESS).text(T.PROMPT)
        text = Style().fg(*T.FG).text(query)
        lines.append(f"{prompt} {text}")

        lines.append(UI.hr(self.width, T.HORIZ, T.FG_MUTED))
        return "\n".join(lines)

    def render_agent_start(self) -> str:
        """Agent response start frame."""
        return self._header("ISAA", T.PRIMARY)

    def render_agent_end(self) -> str:
        """Agent response end frame."""
        return self._footer(T.PRIMARY)



# =============================================================================
# DEMO / TEST -- Run this to see the output in action
# =============================================================================

if __name__ == "__main__":
    import asyncio

    # Mock a_stream for demonstration
    async def mock_a_stream(**kwargs):
        """Simulate agent streaming chunks."""
        chunks = [
            {"type": "status", "status_msg": "Initializing agent session..."},
            {"type": "iteration_start", "iteration": 1, "max_iter": 5},
            {"type": "content", "chunk": "I'll help you fetch your unread emails. "},
            {"type": "content", "chunk": "Let me use the Gmail tool."},
            {"type": "tool_start", "name": "gmail_search", "args": {"query": "is:unread", "max_results": 50, "days": 7}},
            {"type": "status", "status_msg": "Querying Gmail API"},
            {"type": "tool_result", "result": "Found 23 unread emails from the last 7 days in your Gmail (developer.hs2015@gmail.com).", "duration_ms": 1240},
            {"type": "content", "chunk": "\nHere are the most recent unread emails:\n"},
            {"type": "content", "chunk": "1. 2026-04-13 19:22 UTC\n   From: Corey Ganin <hello@returnmytime.com>\n   Subject: Build With AI - yes or no?\n\n"},
            {"type": "content", "chunk": "2. 2026-04-13 04:25 UTC\n   From: GitHub <noreply@github.com>\n   Subject: Security alert for your repository\n\n"},
            {"type": "content", "chunk": "3. 2026-04-12 22:10 UTC\n   From: AWS <notifications@aws.com>\n   Subject: Your monthly bill is ready\n\n"},
            {"type": "iteration_start", "iteration": 2, "max_iter": 5},
            {"type": "reasoning", "chunk": "The user asked for unread emails from last week. I found 23 total. I should present the most recent ones clearly."},
            {"type": "content", "chunk": "I've found 23 unread emails from the last 7 days. Above are the 3 most recent ones."},
            {"type": "post_processing", "status_msg": "Saving conversation to memory"},
            {"type": "done", "success": True, "meta": {"tokens_used": 2847, "model": "claude-sonnet-4"}},
        ]
        for c in chunks:
            yield c
            await asyncio.sleep(0.05)  # Simulate streaming delay

    # Mock self object
    class MockSelf:
        async def a_stream(self, **kwargs):
            async for chunk in mock_a_stream(**kwargs):
                yield chunk

    mock_self = MockSelf()
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

    async def demo():
        print("\n" + UI.hr(80, "\u2550", T.PRIMARY))
        print(Style().bold().fg(*T.PRIMARY).text("  ISAA TERMINAL CLI OUTPUT DEMO"))
        print(Style().fg(*T.FG_DIM).text("  Pure Python - ANSI-only - Zero dependencies"))
        print(UI.hr(80, "\u2550", T.PRIMARY))

        count = 0
        async for output in FlowAgent.a_stream_verbose(
            mock_self,
            query="fetch me my unread emails from last 1 week",
            max_iterations=5
        ):
            print(output, end="", flush=True)
            count += 1

        print(f"\n\n{Style().fg(*T.FG_MUTED).text(f'--- rendered {count} chunks ---')}")

    asyncio.run(demo())
