"""
FlowAgent UI v2 - Elegante Chat-UI mit Fokus auf Funktionalität
================================================================
Inspiriert von DeepSeek/Claude UI - Minimalistisch, elegant, funktional.

Kernprinzipien:
1. Sofortiges visuelles Feedback bei jeder Aktion
2. Nur Buttons die 100% funktionieren
3. Eleganter, übersichtlicher Chat-Bereich
4. Dark/Light Theme mit CSS-Variablen
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Minu Imports
from toolboxv2.mods.Minu import (
    MinuView,
    MinuSession,
    State,
    Component,
    ComponentType,
    ComponentStyle,
    Card,
    Text,
    Heading,
    Button,
    Input,
    Textarea,
    Select,
    Checkbox,
    Switch,
    Row,
    Column,
    Grid,
    Spacer,
    Divider,
    Alert,
    Progress,
    Spinner,
    Badge,
    Modal,
    Dynamic,
    Custom,
    register_view,
)

from toolboxv2 import get_app, App, Result


# ============================================================================
# DATA MODELS
# ============================================================================


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Enhanced chat message with internal agent state tracking"""

    id: str
    role: MessageRole
    content: str
    timestamp: str
    is_streaming: bool = False
    is_thinking: bool = False  # Legacy flag

    # Humanized Progress Data
    reasoning_steps: List[dict] = field(
        default_factory=list
    )  # Liste von 'internal_reasoning' Calls
    meta_tool_calls: List[dict] = field(
        default_factory=list
    )  # Liste aller anderen Meta-Tools
    regular_tool_calls: List[dict] = field(
        default_factory=list
    )  # Echte Tools (Suche etc.)

    current_phase: str = "idle"  # reasoning, planning, executing
    outline_progress: dict = field(
        default_factory=dict
    )  # {step: 1, total: 5, text: "..."}
    reasoning_loop: dict = field(
        default_factory=dict
    )  # Live reasoning loop data
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "is_streaming": self.is_streaming,
            "is_thinking": self.is_thinking,
            "reasoning_steps": self.reasoning_steps,
            "meta_tool_calls": self.meta_tool_calls,
            "regular_tool_calls": self.regular_tool_calls,
            "current_phase": self.current_phase,
            "outline_progress": self.outline_progress,
            "reasoning_loop": self.reasoning_loop,
            "error": self.error,
        }

# ============================================================================
# MAIN AGENT UI VIEW
# ============================================================================


class AgentChatView(MinuView):
    """
    Elegante Chat-UI für FlowAgent.
    Fokus auf Übersichtlichkeit und sofortiges User-Feedback.
    """

    # ===== STATE =====

    # Chat State
    messages = State([])  # List[ChatMessage.to_dict()]
    input_text = State("")

    # Status State - für sofortiges Feedback
    status = State("idle")  # idle, sending, thinking, streaming, error
    status_text = State("")  # Aktueller Status-Text

    # Agent Config
    agent_name = State("self")

    sessions = State([])  # List[{id: str, name: str, created: str}]
    session_manager_open = State(False)

    # Internal
    _agent = None
    _session_id = None

    def __init__(self, view_id: str = None):
        super().__init__(view_id)
        self._agent = None
        self._session_id = f"chat_{uuid.uuid4().hex[:8]}"

    # ===== RENDER =====

    def render(self) -> Component:
        """Main render - Clean, minimal layout"""
        return Column(
            # Chat Container
            self._render_chat_container(),
            className="h-screen flex flex-col",
            style="background: var(--bg-base); color: var(--text-primary);",
        )

    def _render_chat_container(self) -> Component:
        """Main chat container with messages and input"""

        return Column(
            # Messages Area
            self._dynamic_wrapper_messages(),
            # Input Area (fixed at bottom)
            self._render_input_area(),
            self._dynamic_wrapper_session_manager(),
            className="flex-1 flex flex-col max-w-4xl mx-auto w-full",
        )

    def _render_chat_messages_content(self)-> Component:
        return Column(
            # Empty State oder Messages
            self._render_empty_state()
            if not self.messages.value
            else self._render_messages(),
            className="flex-1 overflow-y-auto px-4 py-6",
            style="overflow-anchor: none;",
        )

    def _dynamic_wrapper_messages(self) -> Component:
        dyn = Dynamic(
            render_fn=self._render_chat_messages_content,
            bind=[self.status, self.messages],
        )
        # Registrieren damit die View Bescheid weiß (wichtig für Dependency Tracking)
        self.register_dynamic(dyn)
        return dyn

    def _dynamic_wrapper(self) -> Component:
        dyn = Dynamic(
            render_fn=self._render_buttons,
            bind=[self.status],
        )
        # Registrieren damit die View Bescheid weiß (wichtig für Dependency Tracking)
        self.register_dynamic(dyn)
        return dyn

    def _render_empty_state(self) -> Component:
        """Empty state when no messages"""
        user_name = self.user.name if self.user.is_authenticated else "??"
        return Column(
            # Logo/Icon
            Custom(
                html="""
                <div style="
                    width: 4rem;
                    height: 4rem;
                    border-radius: var(--radius-full);
                    background: color-mix(in oklch, var(--color-primary-500) 20%, transparent);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: var(--space-6);
                ">
                    <svg style="width: 2rem; height: 2rem; color: var(--color-primary-400);" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
                    </svg>
                </div>
                """
            ),
            Text(
                f"Wie kann ich Ihnen helfen {user_name}?",
                style="font-size: var(--text-2xl); font-weight: var(--weight-medium); color: var(--text-primary); margin-bottom: var(--space-2);",
            ),
            Text(
                "Stellen Sie eine Frage oder geben Sie eine Aufgabe ein.",
                style="color: var(--text-muted); text-align: center;",
            ),
            gap="0",
            align="center",
            className="flex-1 flex flex-col items-center justify-center",
        )

    def _render_messages(self) -> Component:
        """Render all messages"""
        messages = self.messages.value or []
        return Column(
            *[self._render_message(msg) for msg in messages],
            gap="6",
            className="pb-4",
        )

    def _render_message(self, msg: dict) -> Component:
        """Render single message - elegant and clean"""
        role = msg.get("role", "user")
        content = msg.get("content", "")

        is_user = role == "user"

        if is_user:
            return self._render_user_message(content)
        else:
            return self._render_assistant_message(msg)

    def _render_user_message(self, content: str) -> Component:
        """User message - subtle styling, readable"""
        return Row(
            Custom(
                html=f"""
                <div style="
                    border-radius: var(--radius-lg);
                    word-break: break-all;
                    background: var(--bg-elevated);
                    padding: var(--space-3) var(--space-4);
                    border: var(--border-width) solid var(--border-subtle);
                ">
                    <p style="color: var(--text-primary); white-space: pre-wrap; margin: 0;">{self._escape_html(content)}</p>
                </div>
                """
            ),
            justify="end",
        )

    def _render_assistant_message(self, msg: dict) -> Component:
        """
        Renders the assistant message in a professional, block-aligned structure.
        Stacks: ReasoningLoop -> Outline -> Phase -> Reasoning -> MetaTools -> Real Tools -> Content
        """
        content = msg.get("content", "")
        is_streaming = msg.get("is_streaming", False)
        reasoning_steps = msg.get("reasoning_steps", [])
        meta_tool_calls = msg.get("meta_tool_calls", [])
        regular_tools = msg.get("regular_tool_calls", [])
        outline = msg.get("outline_progress", {})
        reasoning_loop = msg.get("reasoning_loop", {})
        phase = msg.get("current_phase", "")

        blocks = []

        # 0. Live Reasoning Loop Indicator (NEW - Top priority when streaming)
        if is_streaming and reasoning_loop:
            blocks.append(self._render_reasoning_loop_indicator(reasoning_loop))

        # 1. Outline Progress (Top Bar)
        if outline and outline.get("total_steps", 0) > 0:
            blocks.append(self._render_outline_bar(outline))

        # 2. Phase Indicator (Animated Pulse)
        if is_streaming and phase != "idle":
            blocks.append(self._render_phase_indicator(phase))

        # 3. Internal Reasoning (Collapsible, showing latest thought)
        if reasoning_steps:
            blocks.append(self._render_reasoning_block(reasoning_steps))

        # 4. Meta Tools Log (Collapsible Task List)
        if meta_tool_calls:
            blocks.append(self._render_meta_tools_log(meta_tool_calls))

        # 5. Regular Tool Outputs (Code blocks / Results)
        if regular_tools:
            blocks.append(self._render_tool_badges(regular_tools))

        # 6. Main Text Content (Markdown)
        if content or is_streaming:
            blocks.append(self._render_content_block(content, is_streaming))


        return Row(
            # Avatar Icon
            Custom(
                html="""
                   <div style="
                       width: 2rem;
                       height: 2rem;
                       border-radius: var(--radius-lg);
                       background: linear-gradient(135deg, var(--color-primary-500), var(--color-accent));
                       display: flex;
                       align-items: center;
                       justify-content: center;
                       box-shadow: var(--shadow-md), 0 0 20px color-mix(in oklch, var(--color-primary-500) 30%, transparent);
                       flex-shrink: 0;
                       margin-top: var(--space-1);
                   ">
                       <svg style="width: 1.25rem; height: 1.25rem; color: var(--color-neutral-0);" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                   </div>
               """
            ),
            # Content Column
            Column(*blocks, gap="4", className="flex-1 min-w-0"),
            gap="4",
            align="start",
            style="width: 100%; padding: var(--space-2) 0; animation: fadeIn 0.3s ease-out;",
        )

    def _render_reasoning_loop_indicator(self, loop_data: dict) -> Component:
        """Render live reasoning loop progress indicator"""
        if not loop_data:
            return Spacer(size="0")

        loop_num = loop_data.get("loop_number", 0)
        outline_step = loop_data.get("outline_step", 0)
        outline_total = loop_data.get("outline_total", 0)
        context_size = loop_data.get("context_size", 0)
        task_stack_size = loop_data.get("task_stack_size", 0)
        auto_recovery = loop_data.get("auto_recovery_attempts", 0)
        metrics = loop_data.get("performance_metrics", {})

        # Performance metrics
        loop_times = metrics.get("loop_times", [])
        avg_time = sum(loop_times[-5:]) / len(loop_times[-5:]) if loop_times else 0
        progress_loops = metrics.get("progress_loops", 0)
        total_loops = metrics.get("total_loops", 0)
        efficiency = int((progress_loops / max(total_loops, 1)) * 100)

        # Progress percentage
        progress_pct = int((outline_step / max(outline_total, 1)) * 100) if outline_total else 0

        # Status color based on efficiency
        status_color = (
            "var(--color-success)" if efficiency >= 70
            else "var(--color-warning)" if efficiency >= 40
            else "var(--color-error)"
        )

        return Custom(
            html=f"""
            <div style="
                background: color-mix(in oklch, var(--color-accent) 8%, var(--bg-elevated));
                border: var(--border-width) solid color-mix(in oklch, var(--color-accent) 25%, transparent);
                border-radius: var(--radius-lg);
                padding: var(--space-3);
                margin-bottom: var(--space-3);
                animation: pulse-subtle 3s ease-in-out infinite;
            ">
                <style>
                    @keyframes pulse-subtle {{
                        0%, 100% {{ opacity: 1; }}
                        50% {{ opacity: 0.85; }}
                    }}
                    @keyframes spin {{
                        from {{ transform: rotate(0deg); }}
                        to {{ transform: rotate(360deg); }}
                    }}
                    @keyframes fadeIn {{
                        from {{ opacity: 0; transform: translateY(8px); }}
                        to {{ opacity: 1; transform: translateY(0); }}
                    }}
                </style>

                <!-- Header mit Loop Counter -->
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: var(--space-2);
                ">
                    <div style="display: flex; align-items: center; gap: var(--space-2);">
                        <div style="
                            width: 1.25rem;
                            height: 1.25rem;
                            border: 2px solid var(--color-accent);
                            border-top-color: transparent;
                            border-radius: 50%;
                            animation: spin 1s linear infinite;
                        "></div>
                        <span style="
                            color: var(--text-primary);
                            font-size: var(--text-sm);
                            font-weight: var(--weight-semibold);
                        ">
                            Reasoning Loop #{loop_num}
                        </span>
                    </div>

                    <div style="display: flex; align-items: center; gap: var(--space-3);">
                        <!-- Efficiency Badge -->
                        <span style="
                            font-size: var(--text-xs);
                            padding: var(--space-1) var(--space-2);
                            background: color-mix(in oklch, {status_color} 15%, transparent);
                            color: {status_color};
                            border-radius: var(--radius-sm);
                        ">
                            {efficiency}% Effizienz
                        </span>

                        <!-- Avg Time -->
                        {f'''
                        <span style="
                            font-size: var(--text-xs);
                            color: var(--text-muted);
                        ">
                            ~{avg_time:.1f}s/Loop
                        </span>
                        ''' if avg_time > 0 else ''}
                    </div>
                </div>

                <!-- Progress Bar -->
                {f'''
                <div style="margin-bottom: var(--space-2);">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        font-size: var(--text-xs);
                        color: var(--text-secondary);
                        margin-bottom: var(--space-1);
                    ">
                        <span>Outline Schritt {outline_step}/{outline_total}</span>
                        <span>{progress_pct}%</span>
                    </div>
                    <div style="
                        height: 4px;
                        background: var(--bg-sunken);
                        border-radius: var(--radius-full);
                        overflow: hidden;
                    ">
                        <div style="
                            height: 100%;
                            width: {progress_pct}%;
                            background: linear-gradient(90deg, var(--color-primary-500), var(--color-accent));
                            border-radius: var(--radius-full);
                            transition: width var(--duration-normal) var(--ease-out);
                        "></div>
                    </div>
                </div>
                ''' if outline_total > 0 else ''}

                <!-- Stats Row -->
                <div style="
                    display: flex;
                    gap: var(--space-4);
                    flex-wrap: wrap;
                ">
                    <div style="display: flex; align-items: center; gap: var(--space-1);">
                        <span style="font-size: var(--text-xs); color: var(--text-muted);">📚</span>
                        <span style="font-size: var(--text-xs); color: var(--text-secondary);">
                            Context: {context_size}
                        </span>
                    </div>

                    <div style="display: flex; align-items: center; gap: var(--space-1);">
                        <span style="font-size: var(--text-xs); color: var(--text-muted);">📋</span>
                        <span style="font-size: var(--text-xs); color: var(--text-secondary);">
                            Tasks: {task_stack_size}
                        </span>
                    </div>

                    {f'''
                    <div style="display: flex; align-items: center; gap: var(--space-1);">
                        <span style="font-size: var(--text-xs); color: var(--color-warning);">⚠️</span>
                        <span style="font-size: var(--text-xs); color: var(--color-warning);">
                            Recovery: {auto_recovery}
                        </span>
                    </div>
                    ''' if auto_recovery > 0 else ''}

                    <div style="display: flex; align-items: center; gap: var(--space-1);">
                        <span style="font-size: var(--text-xs); color: var(--text-muted);">🔄</span>
                        <span style="font-size: var(--text-xs); color: var(--text-secondary);">
                            {progress_loops}/{total_loops} produktiv
                        </span>
                    </div>
                </div>
            </div>
            """
        )

    def _render_reasoning_block(self, steps: list) -> Component:
        """Renders the reasoning steps as a professional Insight Card"""
        if not steps:
            return Spacer(size="0")

        cards_html = ""
        for i, step in enumerate(steps):
            thought_num = step.get("thought_number", i + 1)
            total = step.get("total_thoughts", len(steps))
            focus = step.get("current_focus", "")
            confidence = step.get("confidence_level", 0.5)
            insights = step.get("key_insights", [])
            next_needed = step.get("next_thought_needed", False)

            # Confidence color
            conf_color = (
                "color: var(--color-success)"
                if confidence >= 0.7
                else "color: var(--color-warning)"
                if confidence >= 0.4
                else "color: var(--color-error)"
            )

            conf_percent = int(confidence * 100)

            # Insights HTML
            insights_html = ""
            if insights:
                insights_items = "".join(
                    [
                        f'<li style="color: var(--text-secondary);">{self._escape_html(str(ins))}</li>'
                        for ins in insights[:3]
                    ]
                )
                insights_html = f"""
                    <ul style="
                        list-style-type: disc;
                        padding-left: 1.25rem;
                        font-size: var(--text-xs);
                        margin-top: var(--space-2);
                    ">
                        {insights_items}
                    </ul>
                """

            cards_html += f"""
            <div style="
                background: color-mix(in oklch, var(--color-primary-500) 5%, transparent);
                border: var(--border-width) solid color-mix(in oklch, var(--color-primary-500) 20%, transparent);
                border-radius: var(--radius-lg);
                padding: var(--space-3);
                margin-bottom: var(--space-2);
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: var(--space-2);
                ">
                    <div style="display: flex; align-items: center; gap: var(--space-2);">
                        <span style="
                            color: var(--color-primary-400);
                            font-size: var(--text-sm);
                            font-weight: var(--weight-medium);
                        ">
                            💭 Gedanke {thought_num}/{total}
                        </span>

                        {
                f'''
                        <span style="
                            font-size: var(--text-xs);
                            padding: 0.25rem 0.4rem;
                            background: color-mix(in oklch, var(--color-primary-500) 20%, transparent);
                            border-radius: var(--radius-sm);
                            color: var(--color-primary-300);
                        ">
                            weiter →
                        </span>
                        '''
                if next_needed
                else ""
            }
                    </div>

                    <span style="{conf_color}; font-size: var(--text-xs);">
                        {conf_percent}% sicher
                    </span>
                </div>

                {
                f'''
                <p style="
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                ">
                    {self._escape_html(focus)}
                </p>
                '''
                if focus
                else ""
            }

                {insights_html}
            </div>
            """

        return Custom(
            html=f'<div style="display:flex; flex-direction:column; gap:var(--space-2);">{cards_html}</div>'
        )

    def _render_meta_tools_log(self, tool_calls: list) -> Component:
        """Renders meta tools (Agent Actions) in a clean, collapsed log"""
        """Render collective meta-tool calls as collapsible card"""
        if not tool_calls:
            return Spacer(size="0")

        # Group by tool type
        tool_groups = {}
        for call in tool_calls:
            tool_name = call.get("tool_name", "unknown")
            if tool_name not in tool_groups:
                tool_groups[tool_name] = []
            tool_groups[tool_name].append(call)

        # Tool icons
        tool_icons = {
            "manage_internal_task_stack": "📚",
            "delegate_to_llm_tool_node": "🔄",
            "create_and_execute_plan": "📋",
            "advance_outline_step": "✅",
            "write_to_variables": "💾",
            "read_from_variables": "📖",
            "internal_reasoning": "💭",
        }

        # Summary badges
        badges_html = ""
        for tool_name, calls in tool_groups.items():
            icon = tool_icons.get(tool_name, "🔧")
            # Humanize tool name
            human_name = tool_name.replace("_", " ").title()
            if len(human_name) > 20:
                human_name = human_name[:18] + "..."
            count = len(calls)
            count_badge = (
                f'<span style="font-size: var(--text-xs);background: var(--bg-sunken);padding: 0 var(--space-1);border-radius: var(--radius-sm);color: var(--text-primary);">{count}</span>'
                if count > 1
                else ""
            )

            badges_html += f"""
                    <span style="
                        display: inline-flex;
                        align-items: center;
                        gap: var(--space-1);
                        padding: var(--space-1) var(--space-2);
                        background: color-mix(in oklch, var(--border-default) 50%, transparent);
                        border-radius: var(--radius-md);
                        font-size: var(--text-xs);
                        color: var(--text-secondary);
                    ">
                        {icon} {human_name} {count_badge}
                    </span>
                    """

        # Detailed list for expansion
        details_html = ""
        for call in tool_calls[-10:]:  # Show last 5
            tool_name = call.get("tool_name", "unknown")
            icon = tool_icons.get(tool_name, "🔧")
            success = call.get("success", True)
            duration = call.get("duration", 0)
            duration_str = f"{duration:.1f}s" if duration else ""

            status_icon = "✓" if success else "✗"
            status_color = "var(--color-success)" if success else "var(--color-error)"

            # Get key info from metadata
            metadata = call.get("metadata", {})
            info_parts = []
            if metadata.get("task_description"):
                info_parts.append(metadata["task_description"][:50])
            if metadata.get("args"):
                info_parts.append(f"Aktion: {metadata['args']}")
            if metadata.get("tools_count"):
                info_parts.append(f"{metadata['tools_count']} Tools")

            info_str = " · ".join(info_parts) if info_parts else ""

            details_html += f"""
                    <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding:var(--space-2) 0;
            border-bottom:var(--border-width) solid var(--border-subtle);
        ">
            <div style="display:flex; align-items:center; gap:var(--space-2);">
                <span>{icon}</span>

                <span style="
                    color:var(--text-secondary);
                    font-size:var(--text-sm);
                    font-weight:var(--weight-medium);
                ">
                    {tool_name.replace("_", " ").title()}
                </span>

                {
                f'''
                <span style="
                    color:var(--text-muted);
                    font-size:var(--text-xs);
                    white-space:nowrap;
                    overflow:hidden;
                    text-overflow:ellipsis;
                    max-width:200px;
                ">
                    {self._escape_html(info_str)}
                </span>
                '''
                if info_str
                else ""
            }
            </div>

            <div style="display:flex; align-items:center; gap:var(--space-2);">

                {
                f'''
                <span style="
                    color:var(--text-muted);
                    font-size:var(--text-xs);
                ">
                    {duration_str}
                </span>
                '''
                if duration_str
                else ""
            }

                <span style="color:{status_color};">{status_icon}</span>
            </div>
        </div>
                    """

        return Custom(
            html=f"""
                    <details style="
            background: color-mix(in oklch, var(--bg-elevated) 80%, transparent);
            border: var(--border-width) solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow: hidden;
            margin-bottom: var(--space-2);
        ">
            <summary style="
                cursor: pointer;
                padding: var(--space-2) var(--space-3);
                transition: background-color var(--duration-normal) var(--ease-default);
            "
                onmouseover="this.style.background='color-mix(in oklch, var(--border-default) 30%, transparent)'"
                onmouseout="this.style.background='transparent'"
            >
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                ">
                    <div style="display:flex; align-items:center; gap:var(--space-2);">

                        <svg style="
                            width: 1rem;
                            height: 1rem;
                            color: var(--text-muted);
                            transition: transform var(--duration-normal) var(--ease-default);
                        " fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M9 5l7 7-7 7"/>
                        </svg>

                        <span style="
                            font-size: var(--text-sm);
                            color: var(--text-secondary);
                        ">
                            Agent-Aktionen
                        </span>

                        <span style="
                            font-size: var(--text-xs);
                            color: var(--text-muted);
                        ">
                            {len(tool_calls)} ausgeführt
                        </span>
                    </div>
                </div>

                <div style="
                    display:flex;
                    flex-wrap:wrap;
                    gap:var(--space-1);
                    margin-top:var(--space-2);
                ">
                    {badges_html}
                </div>
            </summary>

            <div style="
                padding: var(--space-2) var(--space-3);
                border-top: var(--border-width) solid var(--border-subtle);
                font-size: var(--text-sm);
                color: var(--text-secondary);
            ">
                {details_html}
            </div>
        </details>

                    """
        )

    def _render_phase_indicator(self, phase: str) -> Component:
        icons = {
            "reasoning": "🧠",
            "planning": "📋",
            "executing": "⚡",
            "delegating": "🤝",
        }
        icon = icons.get(phase, "⏳")
        return Custom(
            html=f"""
               <div style="
                   display: inline-flex;
                   align-items: center;
                   gap: var(--space-2);
                   padding: var(--space-1) var(--space-3);
                   border-radius: var(--radius-full);
                   background: color-mix(in oklch, var(--color-primary-500) 10%, transparent);
                   border: var(--border-width) solid color-mix(in oklch, var(--color-primary-500) 20%, transparent);
                   color: var(--color-primary-400);
                   font-size: var(--text-xs);
                   font-weight: var(--weight-medium);
                   animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
               ">
                   <span>{icon}</span>
                   <span style="text-transform: uppercase; letter-spacing: var(--tracking-wide);">{phase}</span>
               </div>
           """
        )

    def _render_outline_bar(self, outline: dict) -> Component:
        # Simple progress bar based on step/total
        current = outline.get("current_step", 0)
        total = outline.get("total_steps", 1)
        step_name = outline.get("step_name", "")
        percentage = min(100, int((current / max(total, 1)) * 100))

        return Custom(
            html=f"""
                    <div style="margin-bottom: var(--space-3);">
                        <div style="
                            display: flex;
                            align-items: center;
                            justify-content: space-between;
                            font-size: var(--text-xs);
                            color: var(--text-muted);
                            margin-bottom: var(--space-1);
                        ">
                            <span style="display: flex; align-items: center; gap: var(--space-1);">
                                <svg style="width: 0.75rem; height: 0.75rem;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/>
                                </svg>
                                Schritt {current} von {total}
                            </span>
                            <span>{percentage}%</span>
                        </div>
                        <div style="
                            height: 4px;
                            background: var(--bg-sunken);
                            border-radius: var(--radius-full);
                            overflow: hidden;
                        ">
                            <div style="
                                height: 100%;
                                width: {percentage}%;
                                background: linear-gradient(90deg, var(--color-primary-500), var(--color-success));
                                transition: width var(--duration-slow) var(--ease-out);
                            "></div>
                        </div>
                        {f'<p style="font-size: var(--text-xs); color: var(--text-muted); margin-top: var(--space-1); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{self._escape_html(step_name)}</p>' if step_name else ""}
                    </div>
                    """
        )

    def _render_tool_badges(self, tool_names: List[dict[str, Any]]) -> Component:
        """Compact tool usage badges"""
        badges_html = " ".join(
            [
                f"""
            <span style="
                display: inline-flex;
                align-items: center;
                gap: var(--space-1);
                padding: var(--space-1) var(--space-2);
                font-size: var(--text-xs);
                background: color-mix(in oklch, var(--color-warning) 10%, transparent);
                color: var(--color-warning);
                border-radius: var(--radius-md);
            ">
                <svg style="width:0.75rem;height:0.75rem;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573
                         1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0
                         001.065 2.572c1.756.426 1.756 2.924 0
                         3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826
                         3.31-2.37 2.37a1.724 1.724 0 00-2.572
                         1.065c-.426 1.756-2.924 1.756-3.35
                         0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724
                         1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924
                         0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31
                         2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                </svg>
                {self._escape_html(data.get("tool", data.get("tool_name",  data.get("name", "Unknown"))))} <hr/>
            </span>
            """
                for data in tool_names[:5]
            ]
        )

        more = (
            f'<span style="color: var(--text-muted); font-size: var(--text-xs);">'
            f"+{len(tool_names) - 5} weitere</span>"
            if len(tool_names) > 5
            else ""
        )

        return Custom(
            html=f"""
        <div style="display:flex; flex-wrap:wrap; gap:var(--space-1); padding: var(--space-1) 0;">
            {badges_html}{more}
        </div>
        """
        )

    def _render_content_block(self, content: str, is_streaming: bool) -> Component:
        """Main content with markdown support"""
        cursor = '<span style="display:inline-block;width:8px;height:16px;background: var(--color-primary-400);animation: pulse-cursor 1s infinite;margin-left: 2px;"></span>' if is_streaming else ""

        # Simple markdown processing
        html_content = self._process_markdown(content)

        return Custom(
            html=f"""
            <style>
@keyframes pulse-cursor {{
    0%, 100% {{opacity:1 }}
    50% {{opacity:0.2 }}
}}
</style>
            <div style="max-width:none; color:var(--text-secondary); font-size: var(--text-sm); line-height: var(--leading-relaxed);">
    <div style="color:var(--text-primary); white-space:pre-wrap; line-height: var(--leading-relaxed);">
        {html_content}{cursor}
    </div>
</div>
            """
        )

    def _render_input_area(self) -> Component:
        """Input area with send button - FULLY FUNCTIONAL"""
        status = self.status.value
        is_busy = status not in ["idle", "error"]

        return Column(
            # Input Card
            Card(
                Column(
                    # Textarea for input
                    Textarea(
                        placeholder="Nachricht an FlowAgent...",
                        # value=self.input_text.value,
                        bind="input_text",
                        on_submit="send_message",
                        rows=2,
                        style="background: transparent; color: var(--text-primary); border: none; resize: none; width: 100%;",
                        className="placeholder-neutral-500",
                    ),
                    # Action Row
                    Row(
                        # Right side - buttons
                        self._dynamic_wrapper(),
                        justify="between",
                        align="center",
                        style="padding-top: var(--space-2); border-top: var(--border-width) solid var(--border-subtle);",
                    ),
                    gap="2",
                ),
                style="background: var(--bg-surface); border: var(--border-width) solid var(--border-default); border-radius: var(--radius-xl); margin-bottom: var(--space-6);",
            ),
            gap="2",
            className="px-4",
        )

    def _render_buttons(self) -> Component:

        status = self.status.value
        is_busy = status not in ["idle", "error"]
        return Row(
            # Clear Button - always functional
            Button(
                "Löschen",
                on_click="clear_chat",
                variant="ghost",
                icon="delete",
                style="color: var(--text-muted);",
            )
            if self.messages.value
            else None,
            # Stop Button - only when busy
            Button(
                "Stopp",
                on_click="stop_generation",
                variant="error",
                icon="stop",
            )
            if is_busy
            else None,
            # Send Button - only when not busy
            Button(
                "Senden",
                on_click="send_message",
                variant="primary",
                icon="send",
                disabled=is_busy #or not self.input_text.value.strip(),
            )
            if not is_busy
            else None,
            Button(
                f"Sessions ({len(self.sessions.value)})",
                on_click="toggle_session_manager",
                variant="secondary",
                icon="folder",
                style="width: min-content;"
            ) if not is_busy else None,
            gap="2",
        )


    # ===== HELPER METHODS =====

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _process_markdown(self, text: str) -> str:
        """Simple markdown to HTML conversion"""
        import re

        if not text:
            return ""

        # Escape HTML first
        text = self._escape_html(text)

        # Code blocks with language
        def code_block_replacer(match):
            lang = match.group(1) or ""
            code = match.group(2)
            return f"""
                <pre style="
                    background: var(--bg-sunken);
                    border-radius: var(--radius-lg);
                    padding: var(--space-4);
                    margin: var(--space-3) 0;
                    overflow-x: auto;
                    border: var(--border-width) solid var(--border-subtle);
                ">
                    <code style="
                        font-size: var(--text-sm);
                        color: var(--color-success);
                        font-family: var(--font-mono);
                    ">{code}</code>
                </pre>
                """

        text = re.sub(r'```(\w*)\n(.*?)```', code_block_replacer, text, flags=re.DOTALL)

        # Inline code
        text = re.sub(
            r'`([^`]+)`',
            r'<code style="background: var(--bg-sunken); padding: 0.125rem 0.375rem; border-radius: var(--radius-sm); font-size: var(--text-sm); color: var(--color-primary-400); font-family: var(--font-mono);">\1</code>',
            text
        )

        # Bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong style="font-weight: var(--weight-semibold);">\1</strong>', text)

        # Italic
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)

        # Links
        text = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            r'<a href="\2" style="color: var(--color-primary-400); text-decoration: none;" onmouseover="this.style.textDecoration=\'underline\'" onmouseout="this.style.textDecoration=\'none\'" target="_blank">\1</a>',
            text
        )

        return text

    # ===== EVENT HANDLERS - ALL 100% IMPLEMENTED =====

    async def send_message(self, event):
        """Orchestrates the chat interaction and parses agent events."""
        text = ""
        if isinstance(event, dict):
            text = event.get("value", "") or event.get("text", "")
        if not text:
            text = self.input_text.value

        text = text.strip()
        if not text:
            return

        # 1. UI Reset
        self.input_text.value = ""
        self.status.value = "thinking"

        # 2. Add User Message
        user_msg = ChatMessage(
            id=f"user_{uuid.uuid4().hex[:8]}",
            role=MessageRole.USER,
            content=text,
            timestamp=datetime.now().strftime("%H:%M"),
        )
        # Update list correctly (re-assign to trigger state)
        current_msgs = list(self.messages.value)
        current_msgs.append(user_msg.to_dict())
        self.messages.value = current_msgs

        if self._session:
            await self._session.force_flush()

        # 3. Create Assistant Placeholder
        ass_id = f"ass_{uuid.uuid4().hex[:8]}"
        ass_msg = ChatMessage(
            id=ass_id,
            role=MessageRole.ASSISTANT,
            content="",
            timestamp=datetime.now().strftime("%H:%M"),
            is_streaming=True,
            current_phase="starting",
        )
        current_msgs.append(ass_msg.to_dict())
        self.messages.value = current_msgs

        if self._session:
            await self._session.force_flush()

        # 4. Define the robust Progress Callback
        collected_content = []
        # Local containers to accumulate state before updating view
        local_reasoning = []
        local_meta = []
        local_regular = []
        local_outline = {}
        local_reasoning_loop = {}

        async def on_progress(event):
            nonlocal collected_content, local_reasoning, local_meta, local_outline, local_regular, local_reasoning_loop

            # Detect Event Type (Attribute or Dict access)
            e_type = getattr(event, "event_type", None) or event.get("event_type")

            # --- HANDLE STREAMING ---
            if e_type == "llm_stream_chunk" and hasattr(event, "llm_output") and event.llm_output:
                collected_content.append(event.llm_output.replace("META_TOOL_CALL:", '').replace("TOOL_CALL:", ''))
                # Update UI Content
                ac_content = "".join(collected_content)
                self._update_msg(ass_id, content=ac_content)
                # Flush often for streaming feel
                if self._session:
                    await self._session.force_flush()
                return

            # --- HANDLE TOOL CALLS ---
            # Helper to safely get attributes
            def get_attr(name, default=None):
                return getattr(event, name, None) or (
                    event.get(name, default) if isinstance(event, dict) else default
                )

            tool_name = get_attr("tool_name", "unknown")
            is_meta = get_attr("is_meta_tool")
            metadata = get_attr("metadata", {})

            # Case A: Internal Reasoning (Special Meta Tool)
            if e_type == "meta_tool_call" or "reasoning" in tool_name:
                tool_args = get_attr("tool_args", metadata) or metadata

                # Extract clean thought object
                thought = {
                    "thought_number": tool_args.get(
                        "thought_number", len(local_reasoning) + 1
                    ),
                    "total_thoughts": tool_args.get("total_thoughts", "?"),
                    "current_focus": tool_args.get("current_focus", "Reasoning..."),
                    "confidence_level": tool_args.get("confidence_level", 0.5),
                    "key_insights": tool_args.get("key_insights", []),
                }
                if thought not in local_reasoning and (tool_args.get("current_focus") or tool_args.get("key_insights")):
                    local_reasoning.append(thought)

                self._update_msg(
                    ass_id, reasoning_steps=local_reasoning, current_phase="reasoning"
                )
                if self._session:
                    await self._session.force_flush()

            # Case B: Other Meta Tools (Stack, Delegate, Plan)
            elif e_type == "meta_tool_call" or is_meta:
                # Add to meta log
                entry = {
                    "tool_name": tool_name,
                    "success": get_attr("success", True),
                    "duration": get_attr("duration"),
                    "args": get_attr("tool_args"),  # Optional: show args in tooltip?
                }
                local_meta.append(entry)

                # Update Outline if present in args
                tool_args = get_attr("tool_args", {})
                if "outline_step_progress" in tool_args:
                    # Parse simple string "1/5" or similar if needed, or use metadata
                    pass

                # Update Phase based on tool
                phase_map = {
                    "manage_internal_task_stack": "planning",
                    "delegate_to_llm_tool_node": "delegating",
                    "create_and_execute_plan": "planning",
                }
                new_phase = phase_map.get(tool_name, "executing")

                self._update_msg(
                    ass_id, meta_tool_calls=local_meta, current_phase=new_phase
                )
                if self._session:
                    await self._session.force_flush()

            # Case C: Regular Tools (Search, etc.)
            elif e_type == "tool_call" and not is_meta:
                # Add to regular tools list (Implement logic if needed)
                entry = {
                    "tool_name": tool_name,
                    "success": get_attr("success", True),
                    "duration": get_attr("duration"),
                    "args": get_attr("tool_args"),
                }
                local_regular.append(entry)
                self._update_msg(ass_id, current_phase="using_tool", regular_tool_calls=local_regular)
                if self._session:
                    await self._session.force_flush()

            # Case D: Meta Tool Batch Summary (Outline Update)
            elif e_type == "meta_tool_batch_complete":
                # Extract Outline Status
                if "outline_status" in metadata:
                    local_outline = metadata["outline_status"]
                    self._update_msg(ass_id, outline_progress=local_outline)
                    if self._session:
                        await self._session.force_flush()

            # Case E: reasoning_loop - LIVE PROGRESS UPDATE
            elif e_type == "reasoning_loop":
                # Update local reasoning loop data
                local_reasoning_loop = {
                    "loop_number": metadata.get("loop_number", 0),
                    "outline_step": metadata.get("outline_step", 0),
                    "outline_total": metadata.get("outline_total", 0),
                    "context_size": metadata.get("context_size", 0),
                    "task_stack_size": metadata.get("task_stack_size", 0),
                    "auto_recovery_attempts": metadata.get("auto_recovery_attempts", 0),
                    "performance_metrics": metadata.get("performance_metrics", {}),
                }
                self._update_msg(ass_id, reasoning_loop=local_reasoning_loop, current_phase="reasoning")
                if self._session:
                    await self._session.force_flush()

            else:
                print(f"Unhandled event type: {e_type}")

        # 5. Run Agent
        try:
            agent = await self._get_agent()  # Your existing getter
            if agent:
                if hasattr(agent, "set_progress_callback"):
                    agent.set_progress_callback(on_progress)

                result = await agent.a_run(query=text, session_id=self._session_id, fast_run=True)

                # Final content update
                final_text = result if isinstance(result, str) else str(result)
                # Fallback if streaming captured everything
                if not final_text and collected_content:
                    final_text = "".join(collected_content)

                self._update_msg(
                    ass_id,
                    content=final_text,
                    is_streaming=False,
                    current_phase="completed",
                    reasoning_loop={},  # Clear reasoning loop on completion
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_msg(
                ass_id, error=str(e), is_streaming=False, current_phase="error"
            )

        self.status.value = "idle"
        if self._session:
            await self._session.force_flush()

    def _update_msg(self, msg_id, **kwargs):
        """Helper to update a specific message in the state list efficiently"""
        # Note: We must create a NEW list to trigger ReactiveState detection
        # from copy import deepcopy
        current = list(self.messages.value)
        for i, m in enumerate(current):
            if m["id"] == msg_id:
                current[i].update(kwargs)  # Update dict in place
                break
        self.messages.value = current  # Trigger update

    async def stop_generation(self, event):
        """Stop current generation - 100% IMPLEMENTED"""
        self.status.value = "idle"
        self.status_text.value = ""

        agent = await self._get_agent()
        #if agent: # TODO
        #    await agent.stop()
        # Mark any streaming message as complete
        messages = list(self.messages.value)
        for i, msg in enumerate(messages):
            if msg.get("is_streaming"):
                messages[i]["is_streaming"] = False
                messages[i]["is_thinking"] = False
                messages[i]["reasoning_loop"] = {}  # Clear reasoning loop
                if not messages[i].get("content"):
                    messages[i]["content"] = "*[Generation gestoppt]*"
                else:
                    messages[i]["content"] += "\n\n*[Generation gestoppt]*"
                break

        self.messages.value = messages

        if self._session:
            await self._session.force_flush()

    async def clear_chat(self, event):
        """Clear all messages - 100% IMPLEMENTED"""
        self.messages.value = []
        self.status.value = "idle"
        self.status_text.value = ""
        self.input_text.value = ""

        if self._session:
            await self._session.force_flush()

    async def _get_agent(self):
        """Get or create agent instance"""
        if self._agent is not None:
            return self._agent

        try:
            app = get_app()
            isaa_mod = app.get_mod("isaa")
            if isaa_mod:
                self._agent = await isaa_mod.get_agent(self.agent_name.value)
                return self._agent
        except Exception as e:
            print(f"Failed to get agent: {e}")

        return None

    # ===== NEUE METHODEN =====

    def _render_session_manager(self) -> Component:
        """Kompakter Session Manager - Glassmorphism Style"""
        is_open = self.session_manager_open.value
        sessions = self.sessions.value or []
        current_id = self._session_id

        if not is_open:
            return Spacer(size="0")

        # Session Items
        session_items = []
        for sess in sessions:
            is_active = sess["id"] == current_id
            session_items.append(
                Row(
                    # Session Info
                    Column(
                        Text(
                            sess.get("name", sess["id"][:8]),
                            style="color: var(--text-secondary); font-size: var(--text-sm);",
                        ),
                        Text(
                            sess.get("created", ""),
                            style="color: var(--text-muted); font-size: var(--text-xs);",
                        ),
                        gap="0",
                    ),
                    # Actions
                    Row(
                        Button(
                            "✓" if is_active else "→",
                            on_click=f"switch_session:{sess['id']}",
                            variant="ghost",
                            disabled=is_active,
                            className="text-xs px-2",
                        ),
                        Button(
                            "×",
                            on_click=f"delete_session:{sess['id']}",
                            variant="ghost",
                            style="color: var(--text-muted); font-size: var(--text-xs); padding: 0 var(--space-1);",
                        ),
                        gap="1",
                    ),
                    justify="between",
                    align="center",
                    style=f"padding: var(--space-2) var(--space-3); border-radius: var(--radius-lg); {'background: color-mix(in oklch, var(--color-primary-500) 10%, transparent); border: var(--border-width) solid color-mix(in oklch, var(--color-primary-500) 20%, transparent);' if is_active else ''}",
                )
            )

        # Panel Content
        panel = Card(
            Column(
                # Liste oder Empty State
                Column(
                    *session_items,
                    gap="1",
                    className="max-h-32 overflow-y-auto",
                )
                if sessions
                else Text(
                    "Keine Sessions",
                    style="color: var(--text-muted); font-size: var(--text-sm); text-align: center; padding: var(--space-3) 0;",
                ),
                # Neue Session Button
                Divider(style="border-color: var(--border-subtle); margin: var(--space-2) 0;"),
                Button(
                    "Neue Session",
                    on_click="create_new_session",
                    variant="ghost",
                    icon="add",
                    className="w-full text-sm",
                ),
                gap="2",
            ),
            style="background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)); border: var(--border-width) solid var(--glass-border); border-radius: var(--radius-xl); padding: var(--space-3);",
        )

        return Column(
            panel,
            gap="2",
            className="pb-4",
        )

    def _dynamic_wrapper_session_manager(self) -> Component:
        """Dynamic wrapper für Session Manager"""
        dyn = Dynamic(
            render_fn=self._render_session_manager,
            bind=[self.session_manager_open, self.sessions],
        )
        self.register_dynamic(dyn)
        return dyn

    # ===== EVENT HANDLERS =====

    async def toggle_session_manager(self, event):
        """Toggle Session Manager"""
        self.session_manager_open.value = not self.session_manager_open.value
        if self._session:
            await self._session.force_flush()

    async def create_new_session(self, event):
        """Neue Session erstellen"""
        # Aktuelle speichern
        if self.messages.value:
            self._save_current_session()

        # Neue Session
        new_id = f"chat_{uuid.uuid4().hex[:8]}"
        new_session = {
            "id": new_id,
            "name": f"Session {len(self.sessions.value) + 1}",
            "created": datetime.now().strftime("%d.%m %H:%M"),
        }

        sessions = list(self.sessions.value)
        sessions.insert(0, new_session)
        self.sessions.value = sessions

        self._session_id = new_id
        self.messages.value = []

        if self._session:
            await self._session.force_flush()

    async def switch_session(self, event):
        """Session wechseln - event enthält session_id nach dem Doppelpunkt"""
        # Parse session_id aus event
        session_id = None
        if isinstance(event, dict):
            session_id = event.get("session_id") or event.get("value")
        elif isinstance(event, str) and ":" in event:
            session_id = event.split(":", 1)[1]

        if not session_id or session_id == self._session_id:
            return

        self._save_current_session()
        self._session_id = session_id
        self.messages.value = self._load_session_messages(session_id)

        if self._session:
            await self._session.force_flush()

    async def delete_session(self, event):
        """Session löschen"""
        session_id = None
        if isinstance(event, dict):
            session_id = event.get("session_id") or event.get("value")
        elif isinstance(event, str) and ":" in event:
            session_id = event.split(":", 1)[1]

        if not session_id:
            return

        self.sessions.value = [s for s in self.sessions.value if s["id"] != session_id]

        if session_id == self._session_id:
            self._session_id = f"chat_{uuid.uuid4().hex[:8]}"
            self.messages.value = []

        if self._session:
            await self._session.force_flush()

    def _save_current_session(self):
        """Aktuelle Session speichern"""
        sessions = list(self.sessions.value)
        exists = any(s["id"] == self._session_id for s in sessions)

        if not exists and self.messages.value:
            sessions.insert(
                0,
                {
                    "id": self._session_id,
                    "name": f"Session {len(sessions) + 1}",
                    "created": datetime.now().strftime("%d.%m %H:%M"),
                },
            )
            self.sessions.value = sessions

    def _load_session_messages(self, session_id: str) -> list:
        """Messages laden (Stub)"""
        return []




# ============================================================================
# REGISTRATION
# ============================================================================


def register_agent_chat_ui():
    """Register the Agent Chat UI view"""
    register_view("agent_ui", AgentChatView)  # Override old


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================


Name = "AgentChatUI"


def initialize(app: App, **kwargs) -> Result:
    """Initialize Agent Chat UI module"""
    register_agent_chat_ui()

    # Register UI route
    app.run_any(
        ("CloudM", "add_ui"),
        name="AgentChat",
        title="FlowAgent Chat",
        path="/api/Minu/render?view=agent_ui&ssr=true",
        description="Elegante Chat-Oberfläche für FlowAgent",
        icon="chat",
        auth=True,
    )

    return Result.ok(info="Agent Chat UI initialized")


__all__ = [
    "AgentChatView",
    "register_agent_chat_ui",
    "ChatMessage",
    "MessageRole",
]
