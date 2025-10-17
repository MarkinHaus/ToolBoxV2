# terminal_progress_production_v3.py

import json
import shutil
import sys
import time
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from io import StringIO
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


from typing import Any, Dict, List, Union
import json


def arguments_summary(tool_args: dict[str, Any], max_length: int = 50) -> str:
    """
    Creates a summary of the tool arguments for display purposes.

    Args:
        tool_args: Dictionary containing tool arguments
        max_length: Maximum length for individual argument values in summary

    Returns:
        Formatted string summary of the arguments
    """

    if not tool_args:
        return "No arguments"

    return_str = ""

    # Handle different types of arguments
    for key, value in tool_args.items():
        # Format the key
        formatted_key = key.replace('_', ' ').title()

        # Handle different value types
        if value is None:
            formatted_value = "None"
        elif isinstance(value, bool):
            formatted_value = str(value)
        elif isinstance(value, (int, float)):
            formatted_value = str(value)
        elif isinstance(value, str):
            # Truncate long strings
            if len(value) > max_length:
                formatted_value = f'"{value[:max_length - 3]}..."'
            else:
                formatted_value = f'"{value}"'
        elif isinstance(value, list):
            if not value:
                formatted_value = "[]"
            elif len(value) == 1:
                item = value[0]
                if isinstance(item, str) and len(item) > max_length:
                    formatted_value = f'["{item[:max_length - 6]}..."]'
                else:
                    formatted_value = f'["{item}"]' if isinstance(item, str) else f'[{item}]'
            else:
                formatted_value = f"[{len(value)} items]"
        elif isinstance(value, dict):
            if not value:
                formatted_value = "{}"
            else:
                keys = list(value.keys())[:3]  # Show first 3 keys
                if len(value) <= 3:
                    formatted_value = f"{{{', '.join(keys)}}}"
                else:
                    formatted_value = f"{{{', '.join(keys)}, ...}} ({len(value)} keys)"
        else:
            # Fallback for other types
            str_value = str(value)
            if len(str_value) > max_length:
                formatted_value = f"{str_value[:max_length - 3]}..."
            else:
                formatted_value = str_value

        # Add to return string
        if return_str:
            return_str += ", "
        return_str += f"{formatted_key}: {formatted_value}"

    # Handle meta-tool specific summaries
    if "tool_name" in tool_args:
        tool_name = tool_args["tool_name"]

        if tool_name == "internal_reasoning":
            meta_summary = []
            if "thought_number" in tool_args and "total_thoughts" in tool_args:
                meta_summary.append(f"Thought {tool_args['thought_number']}/{tool_args['total_thoughts']}")
            if "current_focus" in tool_args and tool_args["current_focus"]:
                focus = tool_args["current_focus"]
                if len(focus) > 30:
                    focus = focus[:27] + "..."
                meta_summary.append(f"Focus: {focus}")
            if "confidence_level" in tool_args:
                meta_summary.append(f"Confidence: {tool_args['confidence_level']}")

            if meta_summary:
                return_str = f"Internal Reasoning - {', '.join(meta_summary)}"

        elif tool_name == "manage_internal_task_stack":
            action = tool_args.get("action", "unknown")
            task_desc = tool_args.get("task_description", "")
            if len(task_desc) > 40:
                task_desc = task_desc[:37] + "..."
            return_str = f"Task Stack - Action: {action.title()}, Task: {task_desc}"

        elif tool_name == "delegate_to_llm_tool_node":
            task_desc = tool_args.get("task_description", "")
            tools_count = len(tool_args.get("tools_list", []))
            if len(task_desc) > 40:
                task_desc = task_desc[:37] + "..."
            return_str = f"Delegate - Task: {task_desc}, Tools: {tools_count}"

        elif tool_name == "create_and_execute_plan":
            goals_count = len(tool_args.get("goals", []))
            return_str = f"Create Plan - Goals: {goals_count}"

        elif tool_name == "advance_outline_step":
            completed = tool_args.get("step_completed", False)
            next_focus = tool_args.get("next_step_focus", "")
            if len(next_focus) > 30:
                next_focus = next_focus[:27] + "..."
            return_str = f"Advance Step - Completed: {completed}, Next: {next_focus}"

        elif tool_name == "write_to_variables":
            scope = tool_args.get("scope", "unknown")
            key = tool_args.get("key", "")
            return_str = f"Write Variable - Scope: {scope}, Key: {key}"

        elif tool_name == "read_from_variables":
            scope = tool_args.get("scope", "unknown")
            key = tool_args.get("key", "")
            return_str = f"Read Variable - Scope: {scope}, Key: {key}"

        elif tool_name == "direct_response":
            final_answer = tool_args.get("final_answer", "")
            if len(final_answer) > 50:
                final_answer = final_answer[:47] + "..."
            return_str = f"Direct Response - Answer: {final_answer}"

    # Handle live tool specific summaries
    elif any(key in tool_args for key in ["code", "filepath", "package_name"]):
        if "code" in tool_args:
            code = tool_args["code"]
            code_preview = code.replace('\n', ' ').strip()
            if len(code_preview) > 40:
                code_preview = code_preview[:37] + "..."
            return_str = f"Execute Code - {code_preview}"

        elif "filepath" in tool_args:
            filepath = tool_args["filepath"]
            if "content" in tool_args:
                content_length = len(str(tool_args["content"]))
                return_str = f"File Operation - Path: {filepath}, Content: {content_length} chars"
            elif "old_content" in tool_args and "new_content" in tool_args:
                return_str = f"Replace in File - Path: {filepath}, Replace operation"
            else:
                return_str = f"File Operation - Path: {filepath}"

        elif "package_name" in tool_args:
            package = tool_args["package_name"]
            version = tool_args.get("version", "latest")
            return_str = f"Install Package - {package} ({version})"

    # Ensure we don't exceed reasonable length for the entire summary
    if len(return_str) > 200:
        return_str = return_str[:197] + "..."

    return return_str





class ProgressiveTreePrinter:
    """Eine moderne, produktionsreife Terminal-Visualisierung für den Agenten-Ablauf."""

    def __init__(self, **kwargs):
        self.processor = StateProcessor()
        self.style = Style()
        self.llm_stream_chunks = ""
        self.buffer = 0
        self._display_interval = 0.1
        self._last_update_time = time.time()
        self._terminal_width = 80
        self._terminal_height = 24
        self._is_initialized = False

        # Terminal-Größe ermitteln
        self._update_terminal_size()

        # Original print sichern
        import builtins
        self._original_print = builtins.print
        builtins.print = self.print
        self._terminal_content = []  # List für O(1) append


    def print(self, *args, **kwargs):
        """
        Überladene print Funktion die automatisch Content speichert
        """
        # Capture output in StringIO für Effizienz
        output = StringIO()
        if 'file' in kwargs:
            del kwargs['file']
        self._original_print(*args, file=output, **kwargs)
        content = output.getvalue()

        # Speichere nur wenn content nicht leer
        if content.strip():
            self._terminal_content.append(content.rstrip('\n'))

        # Normale Ausgabe
        self._original_print(*args, **kwargs)

    def live_print(self,*args, **kwargs):
        """
        Live print ohne Content-Speicherung für temporäre Ausgaben
        """
        self._original_print(*args, **kwargs)

    @staticmethod
    def clear():
        """
        Speichert aktuellen Terminal-Content und cleared das Terminal
        Systemagnostisch (Windows/Unix)
        """
        # Clear terminal - systemagnostisch
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/macOS
            os.system('clear')

    def restore_content(self):
        """
        Stellt den gespeicherten Terminal-Content in einer Aktion wieder her
        Effizient durch join operation
        """
        if self._terminal_content:
            # Effiziente Wiederherstellung mit join
            restored_output = '\n'.join(self._terminal_content)
            self._original_print(restored_output)

    def _update_terminal_size(self):
        """Aktualisiert die Terminal-Dimensionen."""
        try:
            terminal_size = shutil.get_terminal_size()
            self._terminal_width = max(terminal_size.columns, 80)
            self._terminal_height = max(terminal_size.lines, 24)
        except:
            self._terminal_width = 80
            self._terminal_height = 24

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
        if event.event_type == 'execution_start':
            self.processor = StateProcessor()
            self._is_initialized = True


        self.processor.process_event(event)

        # LLM Stream Handling
        if event.event_type == 'llm_stream_chunk':
            self.llm_stream_chunks += event.llm_output
            # Stream-Chunks auf vernünftige Größe begrenzen
            lines = self.llm_stream_chunks.replace('\\n', '\n').split('\n')
            if len(lines) > 8:
                self.llm_stream_chunks = '\n'.join(lines[-8:])
            self.buffer += 1
            if self.buffer > 5:
                self.buffer = 0
            else:
                return

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


        if event.event_type in ['execution_complete', 'error']:
            self.restore_content()
            self.print_final_summary()

    def _update_display(self):
        """Aktualisiert die Anzeige im Terminal."""
        self._update_terminal_size()  # Terminal-Größe neu ermitteln
        output_lines = self._render_full_display()

        self.clear()
        self.live_print('\n'.join(output_lines))


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
                error_text = task.error[:self._terminal_width - 5]
                lines.append(self.style.RED(f"    🔥 {error_text}"))

            displayed_count += 1

        return lines

    def _render_tool_history_section(self, state: AgentExecutionState) -> list:
        """Rendert die Tool-Historie kompakt."""
        history = state.tool_history
        if not history:
            return []

        lines = [self.style.Bold(self.style.YELLOW("🛠️ Tool-Historie"))]

        # Nur die letzten 5 Tools
        for event in reversed(history[-5:]):
            icon = "✅" if event.success else "❌"
            style_func = self.style.GREEN if event.success else self.style.RED
            duration = f"({human_readable_time(event.node_duration)})" if event.node_duration else ""

            tool_line = f"  {icon} {event.tool_name} {duration} {arguments_summary(event.tool_args, self._terminal_width)}"
            lines.append(style_func(tool_line))

            # Fehler kurz anzeigen
            if not event.success and event.tool_error:
                error_text = event.tool_error[:self._terminal_width - 5]
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
        self._update_terminal_size()  # Terminal-Größe neu ermitteln
        output_lines = self._render_full_display()
        print('\n'.join(output_lines))
        summary_lines = [
            "",
            self.style.GREEN2(self.style.Bold("🏁 Ausführung Abgeschlossen")),
            self.style.GREY(f"Events verarbeitet: {self.processor.state.event_count}"),
            self.style.GREY(f"Gesamtlaufzeit: {human_readable_time(time.time() - self.processor.state.start_time)}"),
            ""
        ]

        for line in summary_lines:
            print(line)


# Test cases to demonstrate functionality
if __name__ == "__main__":
    # Test with meta-tools
    test_cases = [
        # Internal reasoning
        {
            "tool_name": "internal_reasoning",
            "thought_number": 2,
            "total_thoughts": 5,
            "current_focus": "Analyzing the problem structure and identifying key components",
            "confidence_level": 0.8,
            "key_insights": ["insight1", "insight2"],
            "potential_issues": ["issue1"]
        },

        # Task stack management
        {
            "tool_name": "manage_internal_task_stack",
            "action": "push",
            "task_description": "Complete the data analysis for the quarterly report",
            "outline_step_ref": "step_3"
        },

        # Code execution
        {
            "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
        },

        # File operations
        {
            "filepath": "/path/to/document.txt",
            "content": "This is a long content string that should be truncated in the summary because it exceeds the maximum length limit"
        },

        # Simple arguments
        {
            "package_name": "numpy",
            "version": "1.24.0"
        },

        # Empty args
        {},

        # Complex nested structure
        {
            "complex_data": {
                "nested": {"deep": "value"},
                "list": [1, 2, 3, 4, 5],
                "boolean": True
            },
            "simple_string": "test"
        }
    ]

    print("Testing arguments_summary function:")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        result = arguments_summary(test_case)
        print(f"Test {i}:")
        print(f"Input: {test_case}")
        print(f"Summary: {result}")
        print("-" * 30)
