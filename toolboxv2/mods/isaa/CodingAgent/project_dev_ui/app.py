"""
app.py - Production UI for ProjectDeveloper.
Fixed: Better UI/UX, Chat-style logs, Status panel, 4 execution modes, scroll, minimize/maximize
"""

import streamlit as st
import asyncio
import time
import os
import sys
import uuid
import logging
import html
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add parent path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

# Production Imports
from db import get_db, ProjectStatus
from preview_server import PreviewServerManager
from connector import AgentRunner, LogEvent
from toolboxv2 import get_app

# --- CONFIGURATION & CONSTANTS ---

st.set_page_config(
    page_title="DEV.CONSOLE // PRODUCTION",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design System Palette
COLORS = {
    "bg": "#08080d",
    "text": "#e2e2e8",
    "textMuted": "rgba(255,255,255,0.45)",
    "accent": "#6366f1",
    "accentLight": "#a5b4fc",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "border": "rgba(255,255,255,0.06)",
    "surface": "rgba(255,255,255,0.02)",
    "surfaceHover": "rgba(255,255,255,0.04)",
}

# --- GLOBAL CSS INJECTION ---

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* RESET */
    .stApp {{
        background-color: {COLORS['bg']};
        font-family: 'IBM Plex Sans', sans-serif;
        color: {COLORS['text']};
    }}

    header[data-testid="stHeader"] {{
        background-color: #08080d !important;
    }}

    /* Collapse Button unabhängig vom Header sichtbar machen */
    div[data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        position: fixed !important;
        top: 10px;
        left: 10px;
        z-index: 9999 !important;
        background: rgba(0,0,0,0.6);
        border-radius: 8px;
        padding: 4px;
    }}

    /* REMOVE BLOAT */
    #MainMenu, footer {{visibility: hidden;}}

    .block-container {{padding-top: 1rem; padding-bottom: 1rem; max-width: 98%;}}

    /* TYPOGRAPHY */
    h1, h2, h3, h4 {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 300;
        letter-spacing: -0.5px;
        color: #fff;
        margin-bottom: 0.5rem;
    }}

    .tech-label {{
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.3);
        margin-bottom: 8px;
        display: block;
        font-weight: 600;
    }}

    /* INPUT ELEMENTS */
    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {{
        background-color: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 4px;
        color: {COLORS['text']};
        font-family: 'IBM Plex Mono', monospace;
    }}
    .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {{
        border-color: {COLORS['accent']} !important;
        box-shadow: 0 0 0 1px {COLORS['accent']}20;
    }}

    /* BUTTONS */
    .stButton > button {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text']};
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.15s ease;
        padding: 0.5rem 1rem;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['accent']}15;
        border-color: {COLORS['accent']}50;
        color: #fff;
    }}
    div[data-testid="stButton"] button[kind="primary"] {{
        background: {COLORS['accent']}20;
        border-color: {COLORS['accent']};
        color: #fff;
    }}
    div[data-testid="stButton"] button[kind="secondary"] {{
        border-color: {COLORS['error']};
        color: {COLORS['error']};
    }}

    /* SIDEBAR - Clean styling, no button interference */
    section[data-testid="stSidebar"] {{
        background-color: #050508 !important;
        border-right: 1px solid {COLORS['border']} !important;
    }}

    /* Sidebar Header Styling */
    .stSidebarHeader {{
        background-color: #0a0a10 !important;
        padding: 1rem !important;
        border-bottom: 1px solid {COLORS['border']} !important;
        margin-bottom: 1rem !important;
    }}
    .stSidebarHeader h1, .stSidebarHeader h2, .stSidebarHeader h3 {{
        color: {COLORS['text']} !important;
        margin: 0 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}

    /* Don't override button styles - let Streamlit handle them */

    /* STICKY HEADER - Fix header at top when scrolling */
    .sticky-header {{
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
        background: {COLORS['bg']} !important;
        padding: 1rem !important;
        border-bottom: 1px solid {COLORS['border']} !important;
        margin-bottom: 0.5rem !important;
        backdrop-filter: blur(10px) !important;
    }}

    /* MAIN PANELS */
    .main-panel {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}

    .panel-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {COLORS['border']};
    }}

    .panel-title {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {COLORS['textMuted']};
        font-weight: 600;
    }}

    .panel-actions {{
        display: flex;
        gap: 0.5rem;
    }}

    .panel-btn {{
        background: none;
        border: none;
        color: {COLORS['textMuted']};
        cursor: pointer;
        padding: 4px 8px;
        font-size: 12px;
        border-radius: 3px;
        transition: all 0.15s;
    }}
    .panel-btn:hover {{
        background: {COLORS['surfaceHover']};
        color: {COLORS['text']};
    }}

    .panel-content {{
        transition: all 0.3s ease;
        overflow: hidden;
    }}
    .panel-content.collapsed {{
        max-height: 0;
        padding: 0;
    }}
    .panel-content.expanded {{
        max-height: none;
    }}

    /* CHAT-STYLE LOG CONSOLE */
    .chat-container {{
        background: #00000040;
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        max-height: 700px;
        min-height: 300px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        padding: 1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        scroll-behavior: smooth;
    }}

    .chat-container::-webkit-scrollbar {{
        width: 6px;
    }}
    .chat-container::-webkit-scrollbar-track {{
        background: #000;
    }}
    .chat-container::-webkit-scrollbar-thumb {{
        background: #333;
        border-radius: 3px;
    }}

    .chat-message {{
        margin-bottom: 0.75rem;
        padding: 0.75rem;
        border-radius: 6px;
        max-width: 90%;
        animation: fadeIn 0.2s ease;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(5px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .chat-message.user {{
        background: {COLORS['accent']}20;
        align-self: flex-end;
        margin-left: auto;
        border: 1px solid {COLORS['accent']}30;
    }}

    .chat-message.assistant {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
    }}

    .chat-message.system {{
        background: transparent;
        border-left: 2px solid {COLORS['accent']};
        padding-left: 0.75rem;
    }}

    .chat-message.error {{
        background: {COLORS['error']}15;
        border: 1px solid {COLORS['error']}40;
    }}

    .chat-message.tool {{
        background: {COLORS['success']}10;
        border: 1px solid {COLORS['success']}30;
    }}

    .chat-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .chat-timestamp {{
        color: {COLORS['textMuted']};
    }}

    .chat-role {{
        font-weight: 600;
    }}
    .chat-role.user {{ color: {COLORS['accent']}; }}
    .chat-role.assistant {{ color: {COLORS['success']}; }}
    .chat-role.system {{ color: {COLORS['accentLight']}; }}
    .chat-role.error {{ color: {COLORS['error']}; }}
    .chat-role.tool {{ color: {COLORS['warning']}; }}

    .chat-content {{
        color: {COLORS['text']};
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.5;
    }}

    .chat-content code {{
        background: rgba(0,0,0,0.3);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
    }}

    .chat-content pre {{
        background: rgba(0,0,0,0.5);
        padding: 0.75rem;
        border-radius: 4px;
        overflow-x: auto;
        margin: 0.5rem 0;
        border: 1px solid {COLORS['border']};
    }}

    /* STATUS PANEL */
    .status-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }}

    .status-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }}

    .status-value {{
        font-size: 24px;
        font-weight: 600;
        color: {COLORS['accent']};
        font-family: 'IBM Plex Mono', monospace;
    }}

    .status-label {{
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {COLORS['textMuted']};
        margin-top: 0.25rem;
    }}

    /* CODER STATUS */
    .coder-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.5rem;
        margin-top: 0.5rem;
    }}

    .coder-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 0.5rem;
        font-size: 11px;
    }}
    .coder-card.active {{
        border-color: {COLORS['success']};
        background: {COLORS['success']}10;
    }}
    .coder-card.idle {{
        border-color: {COLORS['textMuted']};
        opacity: 0.6;
    }}

    .coder-id {{
        font-weight: 600;
        color: {COLORS['accent']};
        font-family: 'IBM Plex Mono', monospace;
    }}

    .coder-task {{
        color: {COLORS['textMuted']};
        font-size: 10px;
        margin-top: 0.25rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}

    /* PLAN DISPLAY */
    .plan-list {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    .plan-item {{
        padding: 0.5rem;
        margin-bottom: 0.25rem;
        background: {COLORS['surface']};
        border-radius: 4px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 12px;
    }}
    .plan-item.done {{
        opacity: 0.5;
        text-decoration: line-through;
    }}
    .plan-item.done .plan-icon {{
        color: {COLORS['success']};
    }}
    .plan-icon {{
        color: {COLORS['textMuted']};
        font-size: 14px;
    }}

    /* FILE LIST */
    .file-list {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    .file-item {{
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid {COLORS['border']};
        cursor: pointer;
        transition: background 0.15s;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 12px;
    }}
    .file-item:hover {{
        background: {COLORS['surfaceHover']};
    }}
    .file-icon {{
        color: {COLORS['accentLight']};
    }}
    .file-name {{
        flex: 1;
        font-family: 'IBM Plex Mono', monospace;
    }}
    .file-size {{
        color: {COLORS['textMuted']};
        font-size: 10px;
    }}

    /* CONTROL BAR */
    .control-bar {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['accent']}40;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }}

    .control-status {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 11px;
        font-weight: 600;
    }}
    .control-status.active {{
        color: {COLORS['success']};
    }}
    .control-status.idle {{
        color: {COLORS['textMuted']};
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {COLORS['textMuted']};
    }}
    .status-dot.active {{
        background: {COLORS['success']};
        animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    /* PREVIEW */
    .preview-container {{
        background: #fff;
        border-radius: 8px;
        overflow: hidden;
    }}
    .preview-placeholder {{
        padding: 3rem;
        text-align: center;
        color: {COLORS['textMuted']};
        border: 2px dashed {COLORS['border']};
        border-radius: 8px;
    }}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['accent']}20;
        border-color: {COLORS['accent']};
    }}

    /* CODE BLOCKS - Better syntax highlighting */
    .chat-content pre {{
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem;
        overflow-x: auto;
        margin: 0.75rem 0;
        font-size: 12px;
        line-height: 1.5;
    }}
    .chat-content pre code {{
        background: transparent;
        padding: 0;
        color: #e6edf3;
        font-family: 'IBM Plex Mono', 'Fira Code', monospace;
    }}
    .chat-content code {{
        background: rgba(110, 118, 129, 0.2);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        color: #ff7b72;
    }}

    /* CODE BLOCK WRAPPER - for copy button and header */
    .code-block-wrapper {{
        margin: 0.75rem 0;
        border-radius: 6px;
        overflow: hidden;
    }}
    .code-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #161b22;
        padding: 0.5rem 1rem;
        border-radius: 6px 6px 0 0;
        border: 1px solid #30363d;
        border-bottom: none;
    }}
    .code-lang {{
        font-size: 10px;
        text-transform: uppercase;
        color: #8b949e;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }}
    .copy-btn {{
        background: {COLORS['accent']}20;
        border: 1px solid {COLORS['accent']}40;
        color: {COLORS['accentLight']};
        border-radius: 3px;
        padding: 2px 8px;
        font-size: 10px;
        cursor: pointer;
        font-family: 'IBM Plex Mono', monospace;
        transition: all 0.2s;
    }}
    .copy-btn:hover {{
        background: {COLORS['accent']}30;
    }}

    /* SCROLL TO TOP BUTTON */
    .scroll-to-top {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: {COLORS['accent']};
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        cursor: pointer;
        display: none;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
    }}
    .scroll-to-top:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }}
    .scroll-to-top.visible {{
        display: flex;
    }}
</style>

<!-- Sidebar Collapse State Manager -->
<script>
(function() {{
    // Wait for DOM to be ready
    function initSidebarStateWatcher() {{
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) return;

        // MutationObserver to watch for sidebar class changes (Streamlit toggle)
        const observer = new MutationObserver(function(mutations) {{
            mutations.forEach(function(mutation) {{
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {{
                    // Check if sidebar is collapsed by Streamlit's toggle button
                    const isCollapsed = sidebar.classList.contains('collapsed') ||
                                      sidebar.style.display === 'none' ||
                                      sidebar.style.width === '0px';

                    if (isCollapsed) {{
                        localStorage.setItem('stSidebarCollapsed', 'true');
                    }} else {{
                        localStorage.setItem('stSidebarCollapsed', 'false');
                    }}
                }}
            }});
        }});

        // Start observing
        observer.observe(sidebar, {{
            attributes: true,
            attributeFilter: ['class', 'style']
        }});

        // Also watch for style changes (width/display)
        const styleObserver = new MutationObserver(function() {{
            const isCollapsed = sidebar.style.display === 'none' ||
                              sidebar.style.width === '0px' ||
                              parseInt(sidebar.style.width || '0') < 100;
            localStorage.setItem('stSidebarCollapsed', isCollapsed ? 'true' : 'false');
        }});

        styleObserver.observe(sidebar, {{
            attributes: true,
            attributeFilter: ['style']
        }});
    }}

    // Run on DOM ready
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initSidebarStateWatcher);
    }} else {{
        initSidebarStateWatcher();
    }}
}})();
</script>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---

def render_badge(text: str, color_key: str = "accent", icon: str = None) -> str:
    """Render a status badge with optional icon"""
    c = COLORS.get(color_key, COLORS['accent'])
    icon_html = f'<i class="{icon}"></i> ' if icon else ''
    return f"""
    <span style="
        display: inline-flex; align-items: center; gap: 4px;
        padding: 3px 8px; border-radius: 4px;
        font-size: 9px; font-weight: 600; font-family: 'IBM Plex Mono';
        background: {c}18; color: {c}; border: 1px solid {c}30;
        text-transform: uppercase; letter-spacing: 0.5px;
    ">{icon_html}{text}</span>
    """


def format_message_content(content: str) -> str:
    """
    Format message content with proper code block handling.
    - Detects code blocks (```, ~~~, edit blocks)
    - Detects inline code (`...`)
    - Properly escapes HTML outside code
    - Uses markdown for better rendering of non-code content
    """
    if not content:
        return ""

    import re

    # Check if content is entirely HTML (from agent output)
    # If it contains HTML-like structures but no code blocks, treat as code
    has_code_block = re.search(r'~~~edit:|```|~~~end~~~', content)

    # Check for traceback/error patterns that should be preformatted
    has_traceback = (
        'Traceback (most recent call last)' in content or
        'CRITICAL EXCEPTION' in content or
        content.count('File "') > 2 or
        content.strip().startswith('<div') and '</div>' in content
    )

    if has_traceback and not has_code_block:
        # Treat entire content as code (error/traceback)
        return f'<pre><code>{html.escape(content)}</code></pre>'

    result = []
    last_end = 0

    # Multiple code block patterns to detect
    patterns = [
        (r'~~~edit:([^\s~]+)~~~\n(.*?)\n~~~end~~~', 'edit', 3),  # Edit blocks
        (r'```(\w*)\n(.*?)\n```', 'code', 2),             # Standard markdown
        (r'~~~\n(.*?)\n~~~', 'code', 2),               # Tilde blocks
    ]

    # Try to find any code block pattern
    matched = False
    for pattern, block_type, min_lines in patterns:
        regex = re.compile(pattern, re.DOTALL)
        match = regex.search(content)
        if match and match.group(2).count('\n') >= min_lines - 1:
            # Found a code block
            before_text = content[:match.start()]
            if before_text:
                # Escape HTML but preserve newlines for non-code text
                escaped_before = html.escape(before_text).replace('\n', '<br>')
                result.append(escaped_before)

            lang = match.group(1) if block_type == 'edit' else match.group(1) if match.lastindex >= 1 else ""
            if block_type == 'edit':
                lang = lang or "diff"
            else:
                lang = lang or "text"

            code = match.group(2)
            escaped_code = html.escape(code)

            # Add copy button to code block
            copy_id = f"copy_{hash(code) % 10000}"
            result.append(f'''
<div class="code-block-wrapper">
    <div class="code-header">
        <span class="code-lang">{lang}</span>
        <button class="copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.parentElement.nextElementSibling.textContent); this.textContent='Copied!'; setTimeout(()=>this.textContent='Copy',2000);">Copy</button>
    </div>
    <pre><code class="language-{lang}">{escaped_code}</code></pre>
</div>''')

            last_end = match.end()
            matched = True
            break

    if not matched:
        # No code blocks found - check for inline code or HTML
        remaining = content[last_end:]

        # Check if content looks like code/HTML
        is_code_like = (
            '<!DOCTYPE' in remaining or '<html' in remaining.lower() or
            ('<div' in remaining.lower() and '</div>' in remaining.lower()) or
            '<span' in remaining.lower() or
            'def ' in remaining or 'function ' in remaining or
            remaining.count('{') > 2 or remaining.count(';') > 2
        )

        if is_code_like:
            # Wrap entire content in code block
            result.append(f'<pre><code>{html.escape(remaining)}</code></pre>')
        else:
            # Handle inline code and escape
            inline_pattern = re.compile(r'`([^`]+)`')
            parts = []
            code_last = 0

            for inline_match in inline_pattern.finditer(remaining):
                before = remaining[code_last:inline_match.start()]
                if before:
                    parts.append(html.escape(before).replace('\n', '<br>'))
                parts.append(f'<code>{html.escape(inline_match.group(1))}</code>')
                code_last = inline_match.end()

            if code_last < len(remaining):
                parts.append(html.escape(remaining[code_last:]).replace('\n', '<br>'))

            result.append(''.join(parts))
    else:
        # Add remaining text after code block
        remaining_text = content[last_end:]
        if remaining_text:
            result.append(html.escape(remaining_text).replace('\n', '<br>'))

    return ''.join(result)


def render_chat_logs(logs: List[LogEvent]) -> str:
    """
    Render logs as a self-contained HTML component for st.components.v1.html().
    Maps log sections to message roles for better UX.
    FIXED: Uses st.components.v1.html instead of st.markdown to support scripts.
    Includes all necessary CSS inline so it renders correctly in an iframe.
    """
    # Section to role mapping
    role_map = {
        "USER": "user",
        "INPUT": "user",
        "USER_INJECT": "user",
        "THOUGHT": "assistant",
        "LLM": "assistant",
        "MSG": "assistant",
        "STREAM": "assistant",
        "TOOL CALL": "tool",
        "TOOL RESULT": "tool",
        "IO": "tool",
        "IO-CALC": "tool",
        "IO-COMMIT": "tool",
        "PARSER": "tool",
        "SYSTEM": "system",
        "LOOP": "system",
        "PLAN UPDATE": "system",
        "DONE": "system",
        "EXIT": "system",
        "ACTION": "system",
        "SYNC": "system",
        "MEMORY": "system",
        "ERROR": "error",
        "FAIL": "error",
        "IO-ERR": "error",
        "IO-FAIL": "error",
        "IO-ROLLBACK": "error",
        "TOOL ERROR": "error",
        "LOOP-GUARD": "error",
        "SUCCESS": "system",
    }

    # Icon mapping
    icon_map = {
        "user": "fa-user",
        "assistant": "fa-robot",
        "tool": "fa-wrench",
        "system": "fa-info-circle",
        "error": "fa-exclamation-triangle",
    }

    # Color scheme (must be inline since this renders in an iframe)
    C = COLORS

    messages_html = ""
    if not logs:
        messages_html = f'<p style="color: rgba(255,255,255,0.45); text-align: center; padding: 2rem;">No messages yet. Start a task to see progress.</p>'
    else:
        for event in logs[-100:]:
            role = role_map.get(event.section, "system")
            icon = icon_map.get(role, "fa-circle")
            content = format_message_content(event.content)

            # Detect file operations
            if any(kw in event.content.upper() for kw in ["FILE:", "WRITTEN", "GENERATED", "EDITED"]):
                role = "tool"

            messages_html += f'''
            <div class="chat-message {role}">
                <div class="chat-header">
                    <span class="chat-role {role}"><i class="fa {icon}"></i> {event.section}</span>
                    <span class="chat-timestamp">{event.timestamp}</span>
                </div>
                <div class="chat-content">{content}</div>
            </div>
            '''

    return f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: transparent; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }}
.chat-container {{
    background: #00000040; border: 1px solid {C['border']}; border-radius: 8px;
    max-height: 680px; min-height: 200px; overflow-y: auto; display: flex;
    flex-direction: column; padding: 1rem; scroll-behavior: smooth;
}}
.chat-container::-webkit-scrollbar {{ width: 6px; }}
.chat-container::-webkit-scrollbar-track {{ background: #000; }}
.chat-container::-webkit-scrollbar-thumb {{ background: #333; border-radius: 3px; }}
.chat-message {{ margin-bottom: 0.75rem; padding: 0.75rem; border-radius: 6px; max-width: 90%; animation: fadeIn 0.2s ease; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(5px); }} to {{ opacity: 1; transform: translateY(0); }} }}
.chat-message.user {{ background: {C['accent']}20; align-self: flex-end; margin-left: auto; border: 1px solid {C['accent']}30; }}
.chat-message.assistant {{ background: {C['surface']}; border: 1px solid {C['border']}; }}
.chat-message.system {{ background: transparent; border-left: 2px solid {C['accent']}; padding-left: 0.75rem; }}
.chat-message.error {{ background: {C['error']}15; border: 1px solid {C['error']}40; }}
.chat-message.tool {{ background: {C['success']}10; border: 1px solid {C['success']}30; }}
.chat-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
.chat-timestamp {{ color: rgba(255,255,255,0.45); }}
.chat-role {{ font-weight: 600; }}
.chat-role.user {{ color: {C['accent']}; }}
.chat-role.assistant {{ color: {C['success']}; }}
.chat-role.system {{ color: {C['accentLight']}; }}
.chat-role.error {{ color: {C['error']}; }}
.chat-role.tool {{ color: {C['warning']}; }}
.chat-content {{ color: {C['text']}; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}
.chat-content code {{ background: rgba(110,118,129,0.2); padding: 2px 6px; border-radius: 3px; font-size: 11px; color: #ff7b72; }}
.chat-content pre {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; overflow-x: auto; margin: 0.75rem 0; font-size: 12px; line-height: 1.5; }}
.chat-content pre code {{ background: transparent; padding: 0; color: #e6edf3; }}
.code-block-wrapper {{ margin: 0.75rem 0; border-radius: 6px; overflow: hidden; }}
.code-header {{ display: flex; justify-content: space-between; align-items: center; background: #161b22; padding: 0.5rem 1rem; border: 1px solid #30363d; border-bottom: none; border-radius: 6px 6px 0 0; }}
.code-lang {{ font-size: 10px; text-transform: uppercase; color: #8b949e; font-weight: 600; }}
.copy-btn {{ background: {C['accent']}20; border: 1px solid {C['accent']}40; color: {C['accentLight']}; border-radius: 3px; padding: 2px 8px; font-size: 10px; cursor: pointer; transition: all 0.2s; }}
.copy-btn:hover {{ background: {C['accent']}30; }}
</style>
</head><body>
<div class="chat-container" id="chat-container">
{messages_html}
</div>
<script>
var c = document.getElementById('chat-container');
if (c) c.scrollTop = c.scrollHeight;
</script>
</body></html>'''


def render_status_panel(runner, logs: List[LogEvent]) -> str:
    """Render a status panel showing current execution state with Active Goal Tracking."""
    import re as _re

    is_running = runner and runner.is_running()
    is_paused = runner and runner.is_paused()

    # Extract progress from actual coder log patterns
    steps = []  # list of (status, title, [detail_lines])
    current_step_details = []
    current_task = "Idle"
    total_steps = 0
    completed_steps = 0
    iterations_done = 0

    # NEW: Track the semantic goal instead of just iteration numbers
    active_goal = "Initializing..."

    for log in logs[-80:]:
        section = log.section
        content = log.content

        # New iteration = new step boundary
        if section == "LOOP":
            iterations_done += 1
            current_task = content[:60]
            current_step_details = []
            # FIX: Use active_goal instead of technical iteration info
            steps.append(('active', f"Step {iterations_done}: {active_goal}", current_step_details))

        # Plan updates from coder
        elif section == "PLAN UPDATE":
            # FIX: Extract the first pending task to update the UI title
            found_goal = None
            for line in content.split('\n'):
                # Search for unchecked items (- [ ] or 1. ...) that are not marked done
                if ("- [ ]" in line or _re.match(r'^\d+\.\s', line)) and not any(
                    x in line for x in ['✔', '[x]', 'Done', 'Erledigt']):
                    clean = _re.sub(r'^(\d+\.|- \[ \]|\[ \]|[\*\-])\s*', '', line).strip()
                    if clean:
                        found_goal = clean[:55]
                        break

            if found_goal:
                active_goal = found_goal
                # Update the title of the currently active step immediately
                if steps and steps[-1][0] == 'active':
                    steps[-1] = ('active', f"Step {iterations_done}: {active_goal}", steps[-1][2])

            # Standard Plan Logic (Counts & Details)
            if "✔" in content or "Erledigt" in content:
                completed_steps += 1
                title = content.replace("✔ Erledigt:", "").replace("✔", "").strip()
                steps.append(('done', title[:60], [content]))
            elif "Neue Agenda" in content:
                m = _re.search(r'(\d+)\s+offene', content)
                if m:
                    total_steps = completed_steps + int(m.group(1))
                current_step_details.append(f"📋 {content}")

        # Completed actions
        elif section == "DONE":
            completed_steps += 1
            steps.append(('done', content[:60], [content]))

        elif section == "ACTION":
            if "edit blocks" in content.lower() or "stopping" in content.lower():
                completed_steps += 1
                steps.append(('done', content[:60], [content]))
            elif current_step_details is not None:
                current_step_details.append(f"⚡ {content[:80]}")

        elif section == "IO-COMMIT":
            if current_step_details is not None:
                current_step_details.append(f"✅ {content}")

        elif section == "IO-CALC":
            if current_step_details is not None:
                current_step_details.append(f"📁 {content[:80]}")

        elif section == "TOOL CALL":
            if current_step_details is not None:
                current_step_details.append(f"🔧 {content[:80]}")

        elif section == "TOOL ERROR" or section == "IO-ERR":
            if current_step_details is not None:
                current_step_details.append(f"❌ {content[:80]}")

        # Track current task footer
        if section in ("LLM", "STREAM", "TOOL CALL", "IO"):
            current_task = f"[{section}] {content[:50]}"

    # Ensure total >= completed
    total_steps = max(total_steps, completed_steps)
    if is_running and total_steps == 0:
        total_steps = iterations_done + 1

    if is_running:
        status_html = render_badge("ACTIVE", "success", "fa-spinner fa-spin")
    elif is_paused:
        status_html = render_badge("PAUSED", "warning", "fa-pause")
    else:
        status_html = render_badge("IDLE", "border", "fa-circle")

    progress_pct = (completed_steps / max(total_steps, 1)) * 100 if total_steps > 0 else 0

    # Build clickable steps HTML
    steps_html = ""
    if steps:
        steps_html = '<div style="margin-top: 0.75rem;">'
        for status, title, details in steps[-15:]:
            if status == "done":
                icon = "✔"
                color = COLORS['success']
                opacity = "0.7"
            elif status == "active":
                icon = "▶"
                color = COLORS['accent']
                opacity = "1.0"
            else:
                icon = "○"
                color = COLORS['textMuted']
                opacity = "0.9"

            escaped_title = html.escape(title)

            if details:
                detail_lines = "".join(
                    f'<div style="padding:2px 0; font-size:10px; color:rgba(255,255,255,0.5);">{html.escape(d)}</div>'
                    for d in details[-8:]
                )
                steps_html += f'''
                <details style="padding:3px 0; opacity:{opacity};">
                    <summary style="cursor:pointer; font-size:11px; color:{color}; list-style:none;">
                        <span>{icon} {escaped_title}</span>
                    </summary>
                    <div style="margin-left:1.2rem; padding:4px 0; border-left:1px solid rgba(255,255,255,0.1); padding-left:8px;">
                        {detail_lines}
                    </div>
                </details>'''
            else:
                steps_html += f'<div style="padding:3px 0; opacity:{opacity}; font-size:11px; color:{color};">{icon} {escaped_title}</div>'

        steps_html += '</div>'

    return f'''
    <div class="status-grid">
        <div class="status-card">
            <div class="status-value">{status_html}</div>
            <div class="status-label">Status</div>
        </div>
        <div class="status-card">
            <div class="status-value" style="font-size: 18px;">{completed_steps}/{total_steps}</div>
            <div class="status-label">Steps</div>
        </div>
        <div class="status-card">
            <div class="status-value">{progress_pct:.0f}%</div>
            <div class="status-label">Progress</div>
        </div>
        <div class="status-card">
            <div class="status-value" style="font-size: 14px;">{len([l for l in logs if l.section in ('TOOL CALL', 'IO', 'IO-CALC')])}</div>
            <div class="status-label">Tool Calls</div>
        </div>
        <div class="status-card">
            <div class="status-value" style="font-size: 14px;">{iterations_done}</div>
            <div class="status-label">Iterations</div>
        </div>
    </div>
    <div style="margin-bottom: 0.5rem; font-size: 11px; color: rgba(255,255,255,0.45);">
        <i class="fa fa-info-circle"></i> {html.escape(current_task)}
    </div>
    {steps_html}
    '''


def render_coder_status(logs: List[LogEvent]) -> str:
    """Render active coder status from logs"""
    # Extract coder info from logs (for parallel mode)
    coders = {}

    for log in logs:
        if "coder" in log.content.lower() or "worker" in log.content.lower():
            # Try to extract coder ID
            parts = log.content.split()
            for i, part in enumerate(parts):
                if "coder" in part.lower() or "worker" in part.lower():
                    coder_id = part.strip("[]:,").upper()
                    if coder_id not in coders:
                        coders[coder_id] = {"status": "active", "task": log.content[:60]}
                    break

    if not coders:
        return '<p style="color: var(--textMuted); font-size: 11px;">No parallel workers active.</p>'

    html = ['<div class="coder-grid">']
    for cid, info in coders.items():
        status_class = "active" if info["status"] == "active" else "idle"
        html.append(f'''
        <div class="coder-card {status_class}">
            <div class="coder-id">{cid}</div>
            <div class="coder-task">{info["task"]}</div>
        </div>
        ''')
    html.append('</div>')
    return "\n".join(html)


def render_file_list(files) -> str:
    """Render generated files list"""
    if not files:
        return '<p style="color: var(--textMuted); text-align: center; padding: 1rem;">No files generated yet.</p>'

    icon_map = {
        '.py': 'fa-python', '.js': 'fa-js', '.jsx': 'fa-react', '.ts': 'fa-code',
        '.tsx': 'fa-react', '.html': 'fa-html5', '.css': 'fa-css3', '.json': 'fa-file-code',
        '.md': 'fa-file-alt', '.vue': 'fa-vuejs'
    }

    html = ['<ul class="file-list">']
    for f in files[:50]:  # Limit to 50 files
        ext = Path(f.file_path).suffix.lower()
        icon = icon_map.get(ext, 'fa-file')
        size_kb = len(f.content) / 1024

        html.append(f'''
        <li class="file-item" onclick="selectFile('{f.file_path}')">
            <i class="fa {icon} file-icon"></i>
            <span class="file-name">{f.file_path}</span>
            <span class="file-size">{size_kb:.1f}KB</span>
        </li>
        ''')
    html.append('</ul>')

    if len(files) > 50:
        html.append(f'<p style="color: var(--textMuted); font-size: 10px; text-align: center;">... and {len(files) - 50} more files</p>')

    return "\n".join(html)


def render_collapsible_panel(title: str, content_id: str, content_html: str,
                             icon: str = "fa-chevron-right", default_expanded: bool = True) -> str:
    """Render a collapsible panel"""
    expanded_class = "expanded" if default_expanded else "collapsed"
    icon_class = "fa-chevron-down" if default_expanded else "fa-chevron-right"

    return f'''
    <div class="main-panel">
        <div class="panel-header">
            <span class="panel-title"><i class="fa {icon}"></i> {title}</span>
            <div class="panel-actions">
                <button class="panel-btn" onclick="togglePanel('{content_id}', this)">
                    <i class="fa {icon_class}" id="{content_id}-icon"></i>
                </button>
            </div>
        </div>
        <div class="panel-content {expanded_class}" id="{content_id}">
            {content_html}
        </div>
    </div>
    <script>
    function togglePanel(id, btn) {{
        const content = document.getElementById(id);
        const icon = document.getElementById(id + '-icon');
        if (content.classList.contains('expanded')) {{
            content.classList.remove('expanded');
            content.classList.add('collapsed');
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-right');
        }} else {{
            content.classList.remove('collapsed');
            content.classList.add('expanded');
            icon.classList.remove('fa-chevron-right');
            icon.classList.add('fa-chevron-down');
        }}
    }}
    </script>
    '''


# --- PRODUCTION AGENT INITIALIZATION ---

def get_production_agent():
    """Synchronous wrapper to retrieve the Real ISAA Coder Agent."""
    if 'agent' not in st.session_state or st.session_state.agent is None:
        try:
            with st.spinner("INITIALIZING PRODUCTION CORE..."):
                app = get_app()
                isaa = app.get_mod("isaa")

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                agent = loop.run_until_complete(isaa.get_agent("coder"))
                loop.close()

                if not agent:
                    st.error("FATAL: Could not retrieve ISAA Coder Agent.")
                    st.stop()

                st.session_state.agent = agent

        except Exception as e:
            st.error(f"SYSTEM INIT FAILURE: {e}")
            logging.error(f"Agent Init Failed: {e}")
            st.stop()

    return st.session_state.agent


# --- STATE MANAGEMENT ---

if 'runner' not in st.session_state:
    st.session_state.runner = None
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None
if 'panels_expanded' not in st.session_state:
    st.session_state.panels_expanded = {
        'chat': True,
        'status': True,
        'files': True,
        'preview': True
    }

# Initialize Core Dependencies
db = get_db()
agent = get_production_agent()


# --- SIDEBAR LAYOUT ---

# --- JAVASCRIPT FUNCTIONS (injected globally) ---
# These must be outside the sidebar to avoid rendering issues

st.markdown("""
<script>
(function() {
    // Sidebar toggle function - global scope
    window.toggleSidebarHeader = function() {
        const icon = document.getElementById('sidebar-header-icon');
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');

        if (icon && icon.classList.contains('fa-chevron-down')) {
            // Collapse
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-right');
            localStorage.setItem('stSidebarCollapsed', 'true');

            if (sidebar) {
                sidebar.style.width = '50px';
                sidebar.style.minWidth = '50px';
            }
        } else if (icon) {
            // Expand
            icon.classList.remove('fa-chevron-right');
            icon.classList.add('fa-chevron-down');
            localStorage.setItem('stSidebarCollapsed', 'false');

            if (sidebar) {
                sidebar.style.width = '';
                sidebar.style.minWidth = '';
            }
        }
    };

    // Watch for Streamlit's sidebar toggle button
    function setupSidebarObserver() {
        // Find sidebar section
        const findSidebar = () => document.querySelector('section[data-testid="stSidebar"]');

        let sidebar = findSidebar();

        // If sidebar doesn't exist yet, wait for it
        if (!sidebar) {
            const checkInterval = setInterval(() => {
                sidebar = findSidebar();
                if (sidebar) {
                    clearInterval(checkInterval);
                    observeSidebar(sidebar);
                }
            }, 100);
            return;
        }

        observeSidebar(sidebar);
    }

    function observeSidebar(sidebar) {
        // MutationObserver for class changes (Streamlit's toggle)
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const isCollapsed = sidebar.classList.contains('collapsed');
                    localStorage.setItem('stSidebarCollapsed', isCollapsed ? 'true' : 'false');

                    // Update icon if exists
                    const icon = document.getElementById('sidebar-header-icon');
                    if (icon) {
                        if (isCollapsed) {
                            icon.classList.remove('fa-chevron-down');
                            icon.classList.add('fa-chevron-right');
                        } else {
                            icon.classList.remove('fa-chevron-right');
                            icon.classList.add('fa-chevron-down');
                        }
                    }
                }
            });
        });

        observer.observe(sidebar, {
            attributes: true,
            attributeFilter: ['class']
        });

        // Also watch for style attribute changes
        const styleObserver = new MutationObserver(function() {
            const display = sidebar.style.display;
            const isHidden = display === 'none' || sidebar.offsetWidth < 50;
            localStorage.setItem('stSidebarCollapsed', isHidden ? 'true' : 'false');
        });

        styleObserver.observe(sidebar, {
            attributes: true,
            attributeFilter: ['style']
        });
    }

    // Initialize sidebar state on load
    function initSidebarState() {
        setupSidebarObserver();

        const collapsed = localStorage.getItem('stSidebarCollapsed');
        const icon = document.getElementById('sidebar-header-icon');
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');

        if (collapsed === 'true' && icon) {
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-right');

            if (sidebar) {
                sidebar.style.width = '50px';
                sidebar.style.minWidth = '50px';
            }
        }
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSidebarState);
    } else {
        initSidebarState();
    }
})();
</script>
""", unsafe_allow_html=True)


# --- SIDEBAR LAYOUT ---

with st.sidebar:
    # Use HTML container for the header with toggle button
    st.markdown("""
    <div id="sidebar-header" style="
        background: #0a0a10;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
        border-radius: 4px;
        cursor: pointer;
        user-select: none;
    " onclick="window.toggleSidebarHeader()">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <h3 style="
                color: #e2e2e8;
                margin: 0;
                font-size: 14px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">PROJECT.OS // PROD</h3>
            <i id="sidebar-header-icon" class="fa fa-chevron-down" style="color: #6b7280; font-size: 12px;"></i>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Selector
    projects = db.list_projects()
    proj_map = {p.id: p.name for p in projects}

    idx = 0
    if st.session_state.current_project_id in proj_map:
        idx = list(proj_map.keys()).index(st.session_state.current_project_id)

    selected_proj_id = st.selectbox(
        "ACTIVE WORKSPACE",
        options=list(proj_map.keys()),
        format_func=lambda x: proj_map[x],
        index=idx if projects else None
    )

    if selected_proj_id != st.session_state.current_project_id:
        st.session_state.current_project_id = selected_proj_id
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")

    # Engine Config - NOW WITH 4 MODES
    st.markdown('<span class="tech-label">EXECUTION STRATEGY</span>', unsafe_allow_html=True)
    engine_mode = st.selectbox("Mode", ["direct", "parallel", "sequential", "swarm"], index=1,
                               help="direct: Single Coder | parallel: Fan-out | sequential: Pipeline | swarm: Adaptive")

    if engine_mode in ["parallel", "swarm"]:
        max_workers = st.slider("Worker Threads", 2, 8, 4)
    else:
        max_workers = 1
        if engine_mode == "direct":
            st.caption("Direct mode uses 1 CoderAgent without manager.")
        else:
            st.caption("Sequential mode uses 1 worker.")

    st.markdown("---")

    # Status Panel
    st.markdown('<span class="tech-label">SYSTEM DIAGNOSTICS</span>', unsafe_allow_html=True)

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("ENGINE")
        if st.session_state.runner and st.session_state.runner.is_running():
             st.markdown(render_badge("ACTIVE", "success", "fa-bolt"), unsafe_allow_html=True)
        else:
             st.markdown(render_badge("IDLE", "border", "fa-circle"), unsafe_allow_html=True)
    with status_col2:
        st.markdown("AGENT")
        st.markdown(render_badge("ONLINE", "accent", "fa-check"), unsafe_allow_html=True)

    # Show active coders if parallel
    if st.session_state.runner and st.session_state.runner.is_running() and engine_mode in ["parallel", "swarm"]:
        st.markdown('<span class="tech-label">ACTIVE WORKERS</span>', unsafe_allow_html=True)
        coder_status = render_coder_status(st.session_state.logs)
        st.markdown(coder_status, unsafe_allow_html=True)

    st.markdown("---")

    # New Project
    with st.expander("NEW WORKSPACE"):
        with st.form("new_proj"):
            np_name = st.text_input("Project Name")
            np_path = st.text_input("Path", value=f"./projects/{uuid.uuid4().hex[:6]}")
            if st.form_submit_button("INITIALIZE"):
                if np_name:
                    pid = uuid.uuid4().hex[:8]
                    db.create_project(pid, np_name, "Production Project", np_path)
                    st.success("Workspace Created")
                    time.sleep(1)
                    st.rerun()


# --- MAIN CONTENT AREA ---

if not st.session_state.current_project_id:
    st.markdown("## NO WORKSPACE SELECTED")
    st.info("Create or select a workspace from the sidebar to begin.")
    st.stop()

p = db.get_project(st.session_state.current_project_id)

# STICKY HEADER - Always visible when scrolling
st.markdown(f'''
<div class="sticky-header">
    <h2 style="margin: 0;">{p.name.upper()}</h2>
    <span class="tech-label">ID: {p.id} // DIR: {p.workspace_path} // MODE: {engine_mode.upper()}</span>
</div>
''', unsafe_allow_html=True)

# Hidden refresh button (use keyboard shortcut R)
if st.button("🔄 REFRESH", key="main_refresh"):
    st.rerun()

# Layout Grid
col_console, col_files = st.columns([7, 5])

# --- LEFT COLUMN: EXECUTION & CHAT ---
with col_console:
    st.markdown('<span class="tech-label">COMMAND INTERFACE</span>', unsafe_allow_html=True)

    runner = st.session_state.runner
    is_running = runner and runner.is_running()
    is_paused = runner and runner.is_paused() if runner else False

    # --- CONTROL BAR ---
    if is_running:
        st.markdown(f"""
        <div class="control-bar">
            <div class="control-status active">
                <span class="status-dot active"></span>
                <span>EXECUTION IN PROGRESS</span>
            </div>
            <span style="flex-grow: 1;"></span>
            <span style="font-size: 10px; color: var(--textMuted);">LOG STREAMING ACTIVE</span>
        </div>
        """, unsafe_allow_html=True)

        # Action Buttons
        c_p, c_s, c_ctx = st.columns([1, 1, 3])

        with c_p:
            if is_paused:
                if st.button("▶ RESUME", type="primary", use_container_width=True):
                    runner.resume()
                    st.rerun()
            else:
                if st.button("⏸ PAUSE", use_container_width=True):
                    runner.pause()
                    st.rerun()

        with c_s:
            if st.button("⏹ STOP", type="secondary", use_container_width=True):
                runner.stop()
                st.rerun()

        with c_ctx:
            if is_paused:
                with st.form("ctx_inject"):
                    user_ctx = st.text_input("Inject Instructions", placeholder="e.g. 'Don't delete main.py', 'Use Flexbox'")
                    if st.form_submit_button("SEND & RESUME"):
                        runner.resume(user_ctx)
                        st.rerun()

    else:
        # Idle State: Task Input
        with st.form("execute_form"):
            task = st.text_area("Development Task", height=100,
                               placeholder="Describe features, refactoring or bugfixes...")

            col_btn1, col_btn2 = st.columns([1, 5])
            with col_btn1:
                run_pressed = st.form_submit_button("EXECUTE", type="primary")

        # Start Logic
        if run_pressed and task:
            if st.session_state.runner and st.session_state.runner.is_running():
                st.warning("Job already running.")
            else:
                runner = AgentRunner(
                    workspace_path=p.workspace_path,
                    agent_instance=st.session_state.agent,
                    mode=engine_mode,  # Now supports 4 modes
                    config={
                        "max_parallel": max_workers,
                        "model": st.session_state.agent.amd.complex_llm_model,
                        "execution_mode": engine_mode  # Pass to connector
                    }
                )
                st.session_state.runner = runner
                st.session_state.logs = []
                runner.start(task)
                st.rerun()

    # --- STATUS PANEL (using st.expander for reliable toggle) ---
    if st.session_state.logs:
        with st.expander("📊 EXECUTION STATUS", expanded=st.session_state.panels_expanded.get('status', True)):
            status_html = render_status_panel(st.session_state.runner, st.session_state.logs)
            st.markdown(status_html, unsafe_allow_html=True)

    # --- CHAT LOG CONSOLE ---
    # 1. Fetch new logs
    if st.session_state.runner:
        new_logs = st.session_state.runner.get_logs()
        if new_logs:
            st.session_state.logs.extend(new_logs)

        # Check completion
        res = st.session_state.runner.check_result()
        if res:
            # Log tracked files if any
            tracked = res.get('tracked_files', [])
            if tracked:
                file_summary = ", ".join(f.path for f in tracked[:10])
                extra = f" (+{len(tracked) - 10} more)" if len(tracked) > 10 else ""
                st.session_state.logs.append(LogEvent(
                    "files", "", "SYSTEM",
                    f"📁 {len(tracked)} file(s) written: {file_summary}{extra}"
                ))

            if res['status'] == 'success':
                st.session_state.logs.append(LogEvent("sys", "", "SYSTEM", "✅ Job Completed Successfully."))
                st.success("Execution Finished.")
            elif res['status'] == 'stopped':
                st.session_state.logs.append(LogEvent("sys", "", "SYSTEM", "🛑 Job Stopped by User."))
                st.warning("Execution Stopped.")
            else:
                msg = res.get('message', 'Unknown Error')
                st.session_state.logs.append(LogEvent("err", "", "ERROR", f"❌ Job Failed: {msg}"))
                st.error("Execution Failed.")

            st.session_state.runner = None
            st.rerun()

    # 2. Render Chat HTML (using st.components.v1.html for proper script/CSS support)
    st.markdown('<span class="tech-label">LIVE TELEMETRY</span>', unsafe_allow_html=True)
    chat_html = render_chat_logs(st.session_state.logs)
    log_count = len(st.session_state.logs)
    # Dynamic height: min 300, scale with log count, max 720
    chat_height = min(720, max(300, 120 + log_count * 8))
    st.components.v1.html(chat_html, height=chat_height, scrolling=False)

    # 3. Auto-refresh trigger
    if st.session_state.runner and st.session_state.runner.is_running():
        time.sleep(0.3)
        st.rerun()


# --- RIGHT COLUMN: FILES & PREVIEW ---
with col_files:
    st.markdown('<span class="tech-label">WORKSPACE DATA</span>', unsafe_allow_html=True)

    tab_files, tab_preview, tab_settings = st.tabs(["GENERATED FILES", "WEB PREVIEW", "SETTINGS"])

    with tab_files:
        files = db.get_generated_files(p.id)

        # ALSO scan workspace for files on disk (coder writes directly to workspace)
        ws_path = Path(p.workspace_path)
        disk_files = []
        if ws_path.exists():
            for fp in ws_path.rglob('*'):
                if not fp.is_file():
                    continue
                if any(skip in str(fp) for skip in ['node_modules', '.git', '__pycache__', '.venv', 'venv']):
                    continue
                rel = str(fp.relative_to(ws_path))
                disk_files.append((rel, fp))

        # Merge: DB files + disk-only files
        db_paths = {f.file_path for f in files} if files else set()
        extra_disk = [(rel, fp) for rel, fp in disk_files if rel not in db_paths]

        has_content = bool(files) or bool(extra_disk)

        if has_content:
            # Build unified file list
            all_file_paths = list(db_paths)
            for rel, _ in extra_disk:
                all_file_paths.append(rel)
            all_file_paths.sort()

            sel_file = st.selectbox("Select File", options=all_file_paths,
                                   label_visibility="collapsed")

            if sel_file:
                # Try DB first
                db_file = next((f for f in (files or []) if f.file_path == sel_file), None)
                if db_file:
                    st.caption(f"Language: {db_file.language} | Size: {len(db_file.content)}b | v{db_file.version} | 📦 DB")
                    lang = db_file.language if db_file.language != "unknown" else None
                    st.code(db_file.content, language=lang, line_numbers=True)
                else:
                    # Read from disk
                    disk_path = ws_path / sel_file
                    if disk_path.exists():
                        ext = disk_path.suffix.lower()
                        lang_map = {'.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
                                    '.ts': 'typescript', '.tsx': 'typescript', '.html': 'html',
                                    '.css': 'css', '.json': 'json', '.md': 'markdown',
                                    '.vue': 'html', '.svg': 'xml', '.yaml': 'yaml', '.yml': 'yaml',
                                    '.toml': 'toml', '.rs': 'rust', '.sh': 'bash'}
                        lang = lang_map.get(ext)
                        try:
                            content = disk_path.read_text(encoding='utf-8', errors='replace')
                            size = disk_path.stat().st_size
                            st.caption(f"Language: {lang or 'auto'} | Size: {size}b | 💾 Disk")
                            st.code(content, language=lang, line_numbers=True)
                        except Exception as e:
                            st.error(f"Cannot read file: {e}")
        else:
            st.info("No artifacts generated yet.")

    with tab_preview:
        try:
            server = PreviewServerManager.get_server(p.id, p.workspace_path)

            status_html = render_badge("ONLINE", "success", "fa-wifi") if server.is_running else render_badge("OFFLINE", "border", "fa-ban")

            col_p1, col_p2, col_p3 = st.columns([3, 1, 1])
            with col_p1:
                st.markdown(f"Server: {status_html}", unsafe_allow_html=True)
            with col_p2:
                if st.button("REFRESH"):
                    st.rerun()
            with col_p3:
                if st.button("OPEN"):
                    st.markdown(f'<a href="{server.url}" target="_blank"><button style="background:var(--accent);color:white;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;">OPEN IN BROWSER</button></a>',
                              unsafe_allow_html=True)

            if server.is_running:
                # Cache-bust: append timestamp so each rerun forces iframe reload
                preview_url = f"{server.url}?_t={int(time.time() * 1000)}"
                st.components.v1.iframe(preview_url, height=600, scrolling=True)
                st.caption(f"Local URL: {server.url}")
            else:
                st.markdown("""
                <div class="preview-placeholder">
                    <i class="fa fa-server" style="font-size: 32px; margin-bottom: 1rem;"></i><br>
                    Preview Server is inactive.<br>
                    It starts automatically when web files are detected.<br>
                    <small>Generate HTML/CSS/JS files to activate.</small>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Preview Error: {e}")

    with tab_settings:
        st.markdown("### Panel Controls")
        for panel_id, label in [('chat', 'Chat Logs'), ('status', 'Status Panel'),
                                ('files', 'File Panel'), ('preview', 'Preview Panel')]:
            is_exp = st.session_state.panels_expanded.get(panel_id, True)
            if st.checkbox(label, value=is_exp, key=f'panel_{panel_id}'):
                st.session_state.panels_expanded[panel_id] = True
            else:
                st.session_state.panels_expanded[panel_id] = False

        st.markdown("---")
        st.markdown("### Debug Info")
        st.json({
            "project_id": p.id,
            "workspace": p.workspace_path,
            "mode": engine_mode,
            "workers": max_workers,
            "log_count": len(st.session_state.logs),
            "is_running": is_running if runner else False
        })
