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
import time
from typing import Any

from prompt_toolkit import print_formatted_text, HTML

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG = os.environ.get("ENGINE_DEBUG", "").lower() in ("1", "true", "yes")

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


class ZenRendererV2:
    """
    Reads engine.live for phase/thought/tool.
    Consumes stream chunks for content/reasoning display.
    Minimizable dashboard for background process overview.
    """

    def __init__(self, engine=None):
        """
        Args:
            engine: ExecutionEngine instance (has .live: AgentLiveState).
                    If None, works in chunk-only mode (no live state).
        """
        self.engine = engine
        self.minimized = False
        self._last_agent = None
        self._last_iter = 0
        self._in_think = False
        self._think_buf = ""
        self._chunk_count = 0

    # -- public API -----------------------------------------------------------

    def toggle_minimize(self):
        """Toggle between minimized (one-liner) and expanded view."""
        self.minimized = not self.minimized
        if self.minimized:
            self._print(f"<style fg='{C['dim']}'>{SYM['minimize']} minimized (press again to expand)</style>")
        else:
            self._print(f"<style fg='{C['dim']}'>{SYM['expand']} expanded</style>")

    def process_chunk(self, chunk: dict):
        """Main entry: render one stream chunk. prompt_toolkit safe."""
        self._chunk_count += 1
        c_type = chunk.get("type", "")

        # In minimized mode: only show done/error, skip everything else
        if self.minimized and c_type not in ("done", "error", "final_answer"):
            return

        # Agent context header (on agent change)
        self._maybe_print_agent_header(chunk)

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
                    pass

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
        """Print agent context header on agent change, progress bar on iter change."""
        agent = chunk.get("agent", "")
        is_sub = chunk.get("is_sub", False)
        iter_n = chunk.get("iter", 0)
        max_n = chunk.get("max_iter", 0)

        if agent and agent != self._last_agent:
            prefix = SYM["sub"] if is_sub else SYM["agent"]
            color = C["dim"] if is_sub else C["cyan"]
            bar = _bar(iter_n, max_n, 12)
            self._print(
                f"\n<style fg='{color}'>{prefix} {_esc(agent)}</style>"
                f"  <style fg='{C['dim']}'>{bar} {iter_n}/{max_n}</style>"
            )
            self._last_agent = agent
            self._last_iter = iter_n
            if DEBUG:
                self.print_debug_panel()

        elif iter_n > self._last_iter:
            # Iteration changed: show compact progress update
            bar = _bar(iter_n, max_n, 12)
            self._print(
                f"  <style fg='{C['dim']}'>{bar} {iter_n}/{max_n}</style>"
            )
            self._last_iter = iter_n

    def _on_reasoning(self, chunk: dict):
        if not self._in_think:
            # Show thought indicator, overwrite with \r
            print_formatted_text(
                HTML(f"  <style fg='{C['dim']}'>{SYM['think']} thinking...</style>"),
                end="\r",
            )
            self._in_think = True
        self._think_buf += chunk.get("chunk", "")

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
        print_formatted_text(HTML(f"<style fg='{C['white']}'>{_esc(text)}</style>"), end="")

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
            return

        # final_answer: handled by _on_final_answer, just show start
        if name == "final_answer":
            return

        # Regular tools: extract key arg for compact display
        arg_str = ""
        if args:
            try:
                ad = json.loads(args) if isinstance(args, str) else args
                for k in ("path", "query", "command", "url", "filename"):
                    if k in ad:
                        arg_str = _short(str(ad[k]), 40)
                        break
                if not arg_str:
                    arg_str = "..."
            except Exception:
                pass

        self._print(
            f"  <style fg='{C['cyan']}'>{SYM['tool']} {name:<14}</style>"
            f"  <style fg='{C['dim']}'>{_esc(arg_str)}</style>",
            end=""
        )

    def _on_tool_result(self, chunk: dict):
        name = chunk.get("name", "")
        if name == "final_answer":
            return

        # think tool: just close the line with ✓
        if name == "think":
            self._print(f"  <style fg='{C['green']}'>{SYM['ok']}</style>")
            return

        result = chunk.get("result", "")
        icon = SYM["ok"]
        color = C["green"]
        meta = ""

        try:
            rd = json.loads(result)
            if isinstance(rd, dict):
                if not rd.get("success", True):
                    icon = SYM["fail"]
                    color = C["red"]
                    meta = _short(rd.get("error", ""), 30)
                elif "content" in rd:
                    meta = f"{len(rd['content'])}ch"
                elif "size" in rd:
                    s = rd["size"]
                    meta = f"{s}b" if s < 1024 else f"{s / 1024:.1f}kb"
        except Exception:
            pass

        self._print(
            f"  <style fg='{C['dim']}'>{_esc(meta)}</style>"
            f"  <style fg='{color}'>{icon}</style>"
        )

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

    @staticmethod
    def _print(html_str: str, end="\n"):
        """prompt_toolkit safe print."""
        try:
            print_formatted_text(HTML(html_str), end=end)
        except Exception:
            # Fallback: strip HTML, plain print (encoding safe)
            import re
            plain = re.sub(r"<[^>]+>", "", html_str)
            print(plain.encode("utf-8", errors="replace").decode("utf-8"), end=end, flush=True)
