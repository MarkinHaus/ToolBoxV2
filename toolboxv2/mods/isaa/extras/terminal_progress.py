# terminal_progress_production_v3.py

import json
import shutil
import sys
import time
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Annahme: Diese Klassen sind wie in Ihrem Code definiert.
from toolboxv2.utils.extras.Style import Style, remove_styles
from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent, NodeStatus, TaskPlan, ToolTask, LLMTask


class VerbosityMode(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"
    DEBUG = "debug"


def human_readable_time(seconds: float) -> str:
    """Konvertiert Sekunden in ein menschlich lesbares Format."""
    if seconds is None:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


class AgentExecutionState:
    """
    Verwaltet den gesamten Zustand des Agentenablaufs, um eine reichhaltige
    Visualisierung zu ermöglichen.
    """

    def __init__(self):
        self.agent_name = "Agent"
        self.execution_phase = 'initializing'
        self.start_time = time.time()
        self.error_count = 0
        self.outline = None
        self.outline_progress = {'current_step': 0, 'total_steps': 0}
        self.reasoning_notes = []
        self.current_reasoning_loop = 0
        self.active_delegation = None
        self.active_task_plan = None
        self.tool_history = []
        self.llm_interactions = {'total_calls': 0, 'total_cost': 0.0, 'total_tokens': 0}
        self.active_nodes = set()
        self.node_flow = []
        self.last_event_per_node = {}
        self.event_count = 0

class StateProcessor:
    """Verarbeitet ProgressEvents und aktualisiert den AgentExecutionState."""

    def __init__(self):
        self.state = AgentExecutionState()

    def process_event(self, event: ProgressEvent):
        self.state.event_count += 1
        if event.agent_name:
            self.state.agent_name = event.agent_name

        # System-Level Events
        if event.event_type == 'node_enter' and event.node_name:
            self.state.active_nodes.add(event.node_name)
            if event.node_name not in self.state.node_flow:
                self.state.node_flow.append(event.node_name)
        elif event.event_type == 'node_exit' and event.node_name:
            self.state.active_nodes.discard(event.node_name)
        elif event.event_type == 'error':
            self.state.error_count += 1

        if event.node_name:
            self.state.last_event_per_node[event.node_name] = event

        # Outline & Reasoning Events
        if event.event_type == 'outline_created' and isinstance(event.metadata.get('outline'), dict):
            self.state.execution_phase = 'planning'
            self.state.outline = event.metadata['outline']
            self.state.outline_progress['total_steps'] = len(self.state.outline.get('steps', []))

        elif event.event_type == 'reasoning_loop':
            self.state.execution_phase = 'reasoning'
            self.state.current_reasoning_loop = event.metadata.get('loop_number', 0)
            self.state.outline_progress['current_step'] = event.metadata.get('outline_step', 0) + 1
            self.state.active_delegation = None

        # Task Plan & Execution Events
        elif event.event_type == 'plan_created' and event.metadata.get('full_plan'):
            self.state.execution_phase = 'executing_plan'
            self.state.active_task_plan = event.metadata['full_plan']
            self.state.active_delegation = None

        elif event.event_type in ['task_start', 'task_complete', 'task_error']:
            self._update_task_plan_status(event)

        # Tool & LLM Events
        elif event.event_type == 'tool_call':
            if event.is_meta_tool:
                self._process_meta_tool_call(event)
            else:
                if event.status in [NodeStatus.COMPLETED, NodeStatus.FAILED]:
                    self.state.tool_history.append(event)
                    if len(self.state.tool_history) > 5:
                        self.state.tool_history.pop(0)

        elif event.event_type == 'llm_call' and event.success:
            llm = self.state.llm_interactions
            llm['total_calls'] += 1
            llm['total_cost'] += event.llm_cost or 0
            llm['total_tokens'] += event.llm_total_tokens or 0

        elif event.event_type == 'execution_complete':
            self.state.execution_phase = 'completed'

    def _process_meta_tool_call(self, event: ProgressEvent):
        args = event.tool_args or {}
        if event.status != NodeStatus.RUNNING:
            return

        if event.tool_name == 'internal_reasoning':
            note = {k: args.get(k) for k in ['thought', 'current_focus', 'key_insights', 'confidence_level']}
            self.state.reasoning_notes.append(note)
            if len(self.state.reasoning_notes) > 3:
                self.state.reasoning_notes.pop(0)

        elif event.tool_name == 'delegate_to_llm_tool_node':
            self.state.active_delegation = {
                'type': 'tool_delegation',
                'description': args.get('task_description', 'N/A'),
                'tools': args.get('tools_list', []),
                'status': 'running'
            }

        elif event.tool_name == 'create_and_execute_plan':
            self.state.active_delegation = {
                'type': 'plan_creation',
                'description': f"Erstelle Plan für {len(args.get('goals', []))} Ziele",
                'goals': args.get('goals', []),
                'status': 'planning'
            }

    def _update_task_plan_status(self, event: ProgressEvent):
        plan = self.state.active_task_plan
        if not plan or not hasattr(plan, 'tasks'):
            return

        for task in plan.tasks:
            if hasattr(task, 'id') and task.id == event.task_id:
                if event.event_type == 'task_start':
                    task.status = 'running'
                elif event.event_type == 'task_complete':
                    task.status = 'completed'
                    task.result = event.tool_result or (event.metadata or {}).get("result")
                elif event.event_type == 'task_error':
                    task.status = 'failed'
                    task.error = (event.error_details or {}).get('message', 'Unbekannter Fehler')
                break

class ProgressiveTreePrinter:
    """Eine moderne, produktionsreife Terminal-Visualisierung für den Agenten-Ablauf."""

    def __init__(self, **kwargs):
        self.processor = StateProcessor()
        self.style = Style()
        self.llm_stream_chunks = ""
        self._display_interval = 0.1
        self._last_update_time = time.time()
        self._current_output_lines = 0
        self._terminal_width = 80
        self._terminal_height = 24
        self._is_initialized = False

        # Terminal-Größe ermitteln
        self._update_terminal_size()

        # Original print sichern
        import builtins
        self._original_print = builtins.print
        builtins.print = self.new_print

    def _update_terminal_size(self):
        """Aktualisiert die Terminal-Dimensionen."""
        try:
            terminal_size = shutil.get_terminal_size()
            self._terminal_width = max(terminal_size.columns, 80)
            self._terminal_height = max(terminal_size.lines, 24)
        except:
            self._terminal_width = 80
            self._terminal_height = 24

    def _clear_screen(self):
        """Löscht den Bildschirm plattformübergreifend."""
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/MacOS
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()

    def _move_cursor_to_top(self):
        """Bewegt den Cursor an den Anfang des Terminals."""
        sys.stdout.write('\033[H')
        sys.stdout.flush()

    def _clear_current_output(self):
        """Löscht die aktuelle Ausgabe durch Überschreiben mit Leerzeichen."""
        if self._current_output_lines > 0:
            # Cursor nach oben bewegen
            sys.stdout.write(f'\033[{self._current_output_lines}A')
            # Jede Zeile löschen
            for _ in range(self._current_output_lines):
                sys.stdout.write('\033[2K')  # Zeile löschen
                sys.stdout.write('\033[1B')  # Eine Zeile nach unten
            # Cursor wieder nach oben
            sys.stdout.write(f'\033[{self._current_output_lines}A')
            sys.stdout.flush()

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Kürzt Text auf maximale Länge und fügt '...' hinzu."""
        if len(remove_styles(text)) <= max_length:
            return text

        # Berücksichtige Style-Codes beim Kürzen
        plain_text = remove_styles(text)
        if len(plain_text) > max_length - 3:
            truncated = plain_text[:max_length - 3] + "..."
            return truncated
        return text

    def _fit_content_to_terminal(self, lines: list) -> list:
        """Passt den Inhalt an die Terminal-Größe an."""
        fitted_lines = []
        available_width = self._terminal_width - 2  # Rand lassen

        for line in lines:
            if len(remove_styles(line)) > available_width:
                fitted_lines.append(self._truncate_text(line, available_width))
            else:
                fitted_lines.append(line)

        # Wenn zu viele Zeilen, die wichtigsten behalten
        max_lines = self._terminal_height - 3  # Platz für Header und Eingabezeile
        if len(fitted_lines) > max_lines:
            # Header behalten, dann die letzten Zeilen
            header_lines = fitted_lines[:5]  # Erste 5 Zeilen (Header)
            remaining_lines = fitted_lines[5:]

            if len(header_lines) < max_lines:
                content_space = max_lines - len(header_lines)
                fitted_lines = header_lines + remaining_lines[-content_space:]
            else:
                fitted_lines = fitted_lines[:max_lines]

        return fitted_lines

    async def progress_callback(self, event: ProgressEvent):
        """Haupteingangspunkt für Progress Events."""
        if event.event_type == 'execution_start' and event.node_name == 'FlowAgent':
            self.processor = StateProcessor()
            self._current_output_lines = 0
            self._is_initialized = True

        self.processor.process_event(event)

        # LLM Stream Handling
        if event.event_type == 'llm_stream_chunk':
            self.llm_stream_chunks += event.llm_output
            # Stream-Chunks auf vernünftige Größe begrenzen
            lines = self.llm_stream_chunks.split('\n')
            if len(lines) > 8:
                self.llm_stream_chunks = '\n'.join(lines[-8:])

        if event.event_type == 'llm_call':
            self.llm_stream_chunks = ""

        # Display nur bei wichtigen Events oder zeitbasiert aktualisieren
        should_update = (
            time.time() - self._last_update_time > self._display_interval or
            event.event_type in ['execution_complete', 'outline_created', 'plan_created', 'node_enter']
        )

        if should_update and self._is_initialized:
            self._update_display()
            self._last_update_time = time.time()

    def _update_display(self):
        """Aktualisiert die Anzeige im Terminal."""
        self._update_terminal_size()  # Terminal-Größe neu ermitteln

        output_lines = self._render_full_display()

        # Vorherige Ausgabe löschen
        self._clear_current_output()

        # Neue Ausgabe schreiben
        for line in output_lines:
            sys.stdout.write(line + '\n')

        sys.stdout.flush()

        # Neue Zeilenzahl speichern
        self._current_output_lines = len(output_lines)

    def _render_full_display(self) -> list:
        """Rendert die komplette Anzeige als Liste von Zeilen."""
        state = self.processor.state
        all_lines = []

        # Header
        header_lines = self._render_header(state).split('\n')
        all_lines.extend(header_lines)
        all_lines.append("")  # Leerzeile

        # Hauptinhalt basierend auf Ausführungsphase
        if state.outline:
            outline_content = self._render_outline_section(state)
            if outline_content:
                all_lines.extend(outline_content)
                all_lines.append("")

        reasoning_content = self._render_reasoning_section(state)
        if reasoning_content:
            all_lines.extend(reasoning_content)
            all_lines.append("")

        activity_content = self._render_activity_section(state)
        if activity_content:
            all_lines.extend(activity_content)
            all_lines.append("")

        if state.active_task_plan:
            plan_content = self._render_task_plan_section(state)
            if plan_content:
                all_lines.extend(plan_content)
                all_lines.append("")

        if state.tool_history:
            tool_content = self._render_tool_history_section(state)
            if tool_content:
                all_lines.extend(tool_content)
                all_lines.append("")

        system_content = self._render_system_flow_section(state)
        if system_content:
            all_lines.extend(system_content)

        # An Terminal-Größe anpassen
        return self._fit_content_to_terminal(all_lines)

    def _render_header(self, state: AgentExecutionState) -> str:
        """Rendert den Header."""
        runtime = human_readable_time(time.time() - state.start_time)
        title = self.style.Bold(f"🤖 {state.agent_name}")
        phase = self.style.CYAN(state.execution_phase.upper())
        health_color = self.style.GREEN if state.error_count == 0 else self.style.YELLOW
        health = health_color(f"Fehler: {state.error_count}")

        header_line = f"{title} [{phase}] | {health} | ⏱️ {runtime}"
        separator = self.style.GREY("═" * min(len(remove_styles(header_line)), self._terminal_width - 2))

        return f"{header_line}\n{separator}"

    def _render_outline_section(self, state: AgentExecutionState) -> list:
        """Rendert die Outline-Sektion."""
        outline = state.outline
        progress = state.outline_progress
        if not outline or not outline.get('steps'):
            return []

        lines = [self.style.Bold(self.style.YELLOW("📋 Agenten-Plan"))]

        for i, step in enumerate(outline['steps'][:5], 1):  # Nur erste 5 Schritte
            status_icon = "⏸️"
            line_style = self.style.GREY

            if i < progress['current_step']:
                status_icon = "✅"
                line_style = self.style.GREEN
            elif i == progress['current_step']:
                status_icon = "🔄"
                line_style = self.style.Bold

            desc = step.get('description', f'Schritt {i}')[:60]  # Beschreibung kürzen
            method = self.style.CYAN(f"({step.get('method', 'N/A')})")

            lines.append(line_style(f"  {status_icon} Schritt {i}: {desc} {method}"))

        if len(outline['steps']) > 5:
            lines.append(self.style.GREY(f"  ... und {len(outline['steps']) - 5} weitere Schritte"))

        return lines

    def _render_reasoning_section(self, state: AgentExecutionState) -> list:
        """Rendert die Reasoning-Sektion."""
        notes = state.reasoning_notes
        if not notes:
            return []

        lines = [self.style.Bold(self.style.YELLOW("🧠 Denkprozess"))]

        # Nur die neueste Notiz anzeigen
        note = notes[-1]
        thought = note.get('thought', '...')[:100]  # Gedanken kürzen
        lines.append(f"  💭 {thought}")

        if note.get('current_focus'):
            focus = note['current_focus'][:80]
            lines.append(f"  🎯 Fokus: {self.style.CYAN(focus)}")

        if note.get('confidence_level') is not None:
            confidence = note['confidence_level']
            lines.append(f"  📊 Zuversicht: {self.style.YELLOW(f'{confidence:.0%}')}")

        if note.get('key_insights'):
            lines.append(f"  💡 Erkenntnisse:")
            for insight in note['key_insights'][:2]:  # Nur erste 2 Erkenntnisse
                insight_text = insight[:70]
                lines.append(f"    • {self.style.GREY(insight_text)}")

        return lines

    def _render_activity_section(self, state: AgentExecutionState) -> list:
        """Rendert die aktuelle Aktivität."""
        lines = [self.style.Bold(self.style.YELLOW(f"🔄 Aktivität (Loop {state.current_reasoning_loop})"))]

        if state.active_delegation:
            delegation = state.active_delegation

            if delegation['type'] == 'plan_creation':
                desc = delegation['description'][:80]
                lines.append(f"  📝 {desc}")

                if delegation.get('goals'):
                    lines.append(f"  🎯 Ziele: {len(delegation['goals'])}")
                    for goal in delegation['goals'][:2]:  # Nur erste 2 Ziele
                        goal_text = goal[:60]
                        lines.append(f"    • {self.style.GREY(goal_text)}")

            elif delegation['type'] == 'tool_delegation':
                desc = delegation['description'][:80]
                lines.append(f"  🛠️ {desc}")
                status = delegation.get('status', 'unbekannt')
                lines.append(f"  📊 Status: {self.style.CYAN(status)}")

                if delegation.get('tools'):
                    tools_text = ', '.join(delegation['tools'][:3])  # Nur erste 3 Tools
                    lines.append(f"  🔧 Tools: {tools_text}")

        # LLM-Statistiken kompakt
        llm = state.llm_interactions
        if llm['total_calls'] > 0:
            cost = f"${llm['total_cost']:.3f}"
            lines.append(
                self.style.GREY(f"  🤖 LLM: {llm['total_calls']} Calls | {cost} | {llm['total_tokens']:,} Tokens"))

        # LLM Stream (gekürzt)
        if self.llm_stream_chunks:
            stream_lines = self.llm_stream_chunks.splitlines()[-8:]
            for stream_line in stream_lines:
                truncated = stream_line[:self._terminal_width - 6]
                lines.append(self.style.GREY(f"  💬 {truncated}"))

        return lines

    def _render_task_plan_section(self, state: AgentExecutionState) -> list:
        """Rendert den Task-Plan kompakt."""
        plan: TaskPlan = state.active_task_plan
        if not plan:
            return []

        lines = [self.style.Bold(self.style.YELLOW(f"⚙️ Plan: {plan.name}"))]

        # Nur aktive und wichtige Tasks anzeigen
        sorted_tasks = sorted(plan.tasks, key=lambda t: (
            0 if t.status == 'running' else
            1 if t.status == 'failed' else
            2 if t.status == 'pending' else 3,
            getattr(t, 'priority', 99),
            t.id
        ))

        displayed_count = 0
        max_display = 5

        for task in sorted_tasks:
            if displayed_count >= max_display:
                remaining = len(sorted_tasks) - displayed_count
                lines.append(self.style.GREY(f"  ... und {remaining} weitere Tasks"))
                break

            icon = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}.get(task.status, "❓")
            style_func = {"pending": self.style.GREY, "running": self.style.WHITE,
                          "completed": self.style.GREEN, "failed": self.style.RED}.get(task.status, self.style.WHITE)

            desc = task.description[:50]  # Beschreibung kürzen
            lines.append(style_func(f"  {icon} {task.id}: {desc}"))

            # Fehler anzeigen wenn vorhanden
            if hasattr(task, 'error') and task.error:
                error_text = task.error[:60]
                lines.append(self.style.RED(f"    🔥 {error_text}"))

            displayed_count += 1

        return lines

    def _render_tool_history_section(self, state: AgentExecutionState) -> list:
        """Rendert die Tool-Historie kompakt."""
        history = state.tool_history
        if not history:
            return []

        lines = [self.style.Bold(self.style.YELLOW("🛠️ Tool-Historie"))]

        # Nur die letzten 3 Tools
        for event in reversed(history[-3:]):
            icon = "✅" if event.success else "❌"
            style_func = self.style.GREEN if event.success else self.style.RED
            duration = f"({human_readable_time(event.node_duration)})" if event.node_duration else ""

            tool_line = f"  {icon} {event.tool_name} {duration}"
            lines.append(style_func(tool_line))

            # Fehler kurz anzeigen
            if not event.success and event.tool_error:
                error_text = event.tool_error[:50]
                lines.append(self.style.RED(f"    💥 {error_text}"))

        return lines

    def _render_system_flow_section(self, state: AgentExecutionState) -> list:
        """Rendert den System-Flow kompakt."""
        if not state.node_flow:
            return []

        lines = [self.style.Bold(self.style.YELLOW("🔧 System-Ablauf"))]

        # Nur aktive Nodes und die letzten paar
        recent_nodes = state.node_flow[-4:]  # Letzte 4 Nodes

        for i, node_name in enumerate(recent_nodes):
            is_last = (i == len(recent_nodes) - 1)
            prefix = "└─" if is_last else "├─"
            is_active = node_name in state.active_nodes
            icon = "🔄" if is_active else "✅"
            style_func = self.style.Bold if is_active else self.style.GREEN

            node_display = node_name[:30]  # Node-Namen kürzen
            lines.append(style_func(f"  {prefix} {icon} {node_display}"))

            # Aktive Node Details
            if is_active:
                last_event = state.last_event_per_node.get(node_name)
                if last_event and last_event.event_type == 'tool_call' and last_event.status == NodeStatus.RUNNING:
                    tool_name = last_event.tool_name[:25]
                    child_prefix = "     " if is_last else "  │  "
                    lines.append(self.style.GREY(f"{child_prefix}🔧 {tool_name}"))

        if len(state.node_flow) > 4:
            lines.append(self.style.GREY(f"  ... und {len(state.node_flow) - 4} weitere Nodes"))

        return lines

    def print_final_summary(self):
        """Zeigt die finale Zusammenfassung."""
        self._update_display()
        summary_lines = [
            "",
            self.style.GREEN2(self.style.Bold("🏁 Ausführung Abgeschlossen")),
            self.style.GREY(f"Events verarbeitet: {self.processor.state.event_count}"),
            self.style.GREY(f"Gesamtlaufzeit: {human_readable_time(time.time() - self.processor.state.start_time)}"),
            ""
        ]

        for line in summary_lines:
            print(line)

    def new_print(self, *args, **kwargs):
        """Ersetzt die Standard-print Funktion für saubere Integration."""
        self._update_display()
        # Normal printen
        self._original_print(*args, **kwargs)
        self._current_output_lines+=str(args).count('\n')

