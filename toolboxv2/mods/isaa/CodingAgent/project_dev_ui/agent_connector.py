"""
Agent Connector - Proxy Agent for ProjectDeveloperEngine Integration

This module provides:
- Proxy agent that interprets user requests and delegates to ProjectDeveloperEngine
- Async execution management
- Status callbacks for UI updates
- Mock mode for standalone testing
"""

import asyncio
import json
import logging
import re
import sys
import os
import time
import uuid
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AgentConnector")

# Try to import real dependencies
try:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    HAS_FLOW_AGENT = True
except ImportError:
    HAS_FLOW_AGENT = False

try:
    # Import from the uploaded document context
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from toolboxv2.mods.isaa.CodingAgent.project_developer import ProjectDeveloperEngine, create_project_developer
    HAS_PROJECT_DEV = True
except ImportError:
    HAS_PROJECT_DEV = False


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    RESEARCHING = "researching"
    PLANNING = "planning"
    GENERATING = "generating"
    VALIDATING = "validating"
    FIXING = "fixing"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    EXECUTING = "executing"


@dataclass
class StatusUpdate:
    """Status update message for UI"""
    status: ExecutionStatus
    message: str
    phase: str
    progress: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExecutionResult:
    """Result of a development execution"""
    success: bool
    generated_files: Dict[str, str]
    errors: List[str]
    execution_time: float
    phases_completed: List[str]


class ProxyAgent:
    """
    Proxy Agent that interprets user requests and manages ProjectDeveloperEngine executions.

    Responsibilities:
    - Parse user intent from natural language
    - Determine target files and task structure
    - Manage execution lifecycle
    - Provide status updates to UI
    """

    def __init__(
        self,
        workspace_path: str,
        status_callback: Optional[Callable[[StatusUpdate], None]] = None,
        use_mock: bool = False
    ):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.status_callback = status_callback
        self.use_mock = use_mock or not (HAS_FLOW_AGENT and HAS_PROJECT_DEV)
        print(f"use_mock: {self.use_mock} HAS_FLOW_AGENT: {HAS_FLOW_AGENT} HAS_PROJECT_DEV: {HAS_PROJECT_DEV} use_mock:{use_mock}")
        self._developer: Optional['ProjectDeveloperEngine'] = None
        self._agent: Optional['FlowAgent'] = None
        self._execution_queue: Queue = Queue()
        self._current_execution: Optional[str] = None
        self._should_stop = False
        self._is_paused = False

        # Execution history for context continuity across follow-up tasks
        self._execution_history: List[Dict[str, Any]] = []

        # Mock state for testing
        self._mock_files: Dict[str, str] = {}

    def _emit_status(self, status: ExecutionStatus, message: str,
                     phase: str = "", progress: float = 0.0, details: Dict = None):
        """Emit status update to callback"""
        update = StatusUpdate(
            status=status,
            message=message,
            phase=phase,
            progress=progress,
            details=details or {}
        )

        # Log to terminal
        log_msg = f"[{phase}] {message}"
        if details:
            if 'thinking' in details:
                log_msg += f" | 💭 {str(details['thinking'])[:60]}"
            elif 'tool' in details:
                log_msg += f" | 🔧 {details['tool']}"
            elif 'file' in details:
                log_msg += f" | 📄 {details['file']}"

        if status == ExecutionStatus.FAILED:
            logger.error(log_msg)
        elif status == ExecutionStatus.COMPLETED:
            logger.info(f"✅ {log_msg}")
        else:
            logger.info(log_msg)

        if self.status_callback:
            self.status_callback(update)
        return update

    async def initialize(self, agent: Optional['FlowAgent'] = None):
        """Initialize the agent connector"""
        if self.use_mock:
            self._emit_status(
                ExecutionStatus.PENDING,
                "🔧 Running in mock mode (ToolBoxV2 not available)",
                "init",
                1.0
            )
            return

        if agent is not None:
            self._agent = agent

        if self._agent is None and HAS_FLOW_AGENT:
            try:
                from toolboxv2 import get_app
                app = get_app()
                isaa = app.get_mod("isaa")
                await isaa.init_isaa()
                self._agent = await isaa.get_agent("coder")
            except Exception as e:
                self._emit_status(
                    ExecutionStatus.FAILED,
                    f"Failed to initialize FlowAgent: {e}",
                    "init",
                    0.0
                )
                self.use_mock = True
                return

        if HAS_PROJECT_DEV and self._agent:
            self._developer = create_project_developer(
                agent=self._agent,
                workspace_path=str(self.workspace_path),
                prefer_docker=False,  # Use restricted for faster iteration
                verbose=True
            )
            self._emit_status(
                ExecutionStatus.PENDING,
                "✅ ProjectDeveloperEngine initialized",
                "init",
                1.0
            )

    def parse_task(self, user_message: str) -> Tuple[str, List[str]]:
        """
        Parse user message to extract task description and target files.

        Returns:
            (task_description, target_files)
        """
        # Look for explicit file mentions
        file_patterns = [
            r'(?:create|make|build|write|modify|update|edit)\s+(?:a\s+)?(?:file\s+)?["\']?([^\s"\']+\.[a-z]+)["\']?',
            r'(?:in|to)\s+["\']?([^\s"\']+\.[a-z]+)["\']?',
            r'([a-zA-Z_][a-zA-Z0-9_/]*\.(?:py|js|ts|html|css|json|yaml|yml|md))',
        ]

        files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            files.update(matches)

        # If no files found, try to infer from task type
        if not files:
            lower_msg = user_message.lower()

            if any(kw in lower_msg for kw in ['react', 'jsx', 'component']):
                files = {'index.html', 'App.jsx', 'style.css'}
            elif any(kw in lower_msg for kw in ['vue', 'vue.js', 'vuejs']):
                files = {'index.html', 'App.vue', 'style.css'}
            elif any(kw in lower_msg for kw in ['typescript', 'tsx']):
                files = {'index.html', 'App.tsx', 'style.css'}
            elif any(kw in lower_msg for kw in ['website', 'web app', 'frontend', 'page', 'landing']):
                files = {'index.html', 'style.css', 'script.js'}
            elif any(kw in lower_msg for kw in ['api', 'server', 'backend', 'endpoint', 'flask', 'fastapi']):
                files = {'server.py', 'routes.py'}
            elif any(kw in lower_msg for kw in ['function', 'utility', 'tool', 'helper']):
                files = {'utils.py'}
            elif any(kw in lower_msg for kw in ['dashboard', 'chart', 'visualization']):
                files = {'index.html', 'style.css', 'script.js', 'data.json'}
            else:
                files = {'main.py'}

        # Clean task description
        task = user_message.strip()

        return task, list(files)

    async def execute_task(
        self,
        user_message: str,
        max_retries: int = 3,
        auto_research: bool = False
    ) -> ExecutionResult:
        """
        Execute a development task from user message.

        This is the main entry point for the proxy agent.
        """
        execution_id = str(uuid.uuid4())[:8]
        self._current_execution = execution_id
        self._should_stop = False
        self._is_paused = False

        start_time = time.time()
        phases_completed = []
        errors = []
        generated_files = {}

        try:
            # Phase 1: Parse task
            self._emit_status(ExecutionStatus.ANALYZING, "🔍 Analyzing your request...", "parse", 0.1)
            task, target_files = self.parse_task(user_message)

            self._emit_status(
                ExecutionStatus.ANALYZING,
                f"📁 Identified {len(target_files)} target file(s): {', '.join(target_files)}",
                "parse",
                0.2,
                {"task": task, "files": target_files}
            )
            phases_completed.append("parse")

            if self._should_stop:
                return self._create_result(False, {}, ["Execution stopped by user"], time.time() - start_time, phases_completed)

            # Use mock or real execution
            if self.use_mock:
                generated_files = await self._mock_execute(task, target_files, max_retries)
            else:
                generated_files = await self._real_execute(task, target_files, max_retries, auto_research)

            phases_completed.append("execution")

            # Sync to disk
            self._emit_status(ExecutionStatus.SYNCING, "💾 Saving files to disk...", "sync", 0.95)
            for file_path, content in generated_files.items():
                full_path = self.workspace_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding='utf-8')
            phases_completed.append("sync")

            self._emit_status(
                ExecutionStatus.COMPLETED,
                f"✅ Successfully generated {len(generated_files)} file(s)!",
                "complete",
                1.0,
                {"files": list(generated_files.keys())}
            )

            return self._create_result(True, generated_files, errors, time.time() - start_time, phases_completed)

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            errors.append(error_msg)
            self._emit_status(
                ExecutionStatus.FAILED,
                f"❌ Execution failed: {str(e)}",
                "error",
                0.0,
                {"error": error_msg}
            )
            return self._create_result(False, generated_files, errors, time.time() - start_time, phases_completed)
        finally:
            self._current_execution = None

    async def _mock_execute(
        self,
        task: str,
        target_files: List[str],
        max_retries: int
    ) -> Dict[str, str]:
        """Mock execution for testing without ToolBoxV2 - with detailed status updates"""
        generated = {}

        # Phase 1: Analysis with thinking
        if self._should_stop:
            return generated
        while self._is_paused:
            await asyncio.sleep(0.5)

        self._emit_status(
            ExecutionStatus.ANALYZING,
            "📊 Analyzing project context...",
            "analysis",
            0.25,
            {"thinking": "Understanding the task requirements and identifying key components..."}
        )
        await asyncio.sleep(0.6)

        self._emit_status(
            ExecutionStatus.ANALYZING,
            "💭 Breaking down the task...",
            "analysis",
            0.3,
            {"thinking": f"Task: {task[:100]}...\nIdentified {len(target_files)} files to generate."}
        )
        await asyncio.sleep(0.4)

        # Phase 2: Research with tool usage
        if self._should_stop:
            return generated
        while self._is_paused:
            await asyncio.sleep(0.5)

        self._emit_status(
            ExecutionStatus.RESEARCHING,
            "🔧 Using: Documentation Search",
            "research",
            0.35,
            {"tool": "doc_search", "thinking": "Searching for relevant APIs and best practices..."}
        )
        await asyncio.sleep(0.5)

        self._emit_status(
            ExecutionStatus.RESEARCHING,
            "📚 Found relevant documentation",
            "research",
            0.4,
            {"thinking": "Identified patterns: Modern ES6+, CSS Grid/Flexbox, Python type hints..."}
        )
        await asyncio.sleep(0.4)

        # Phase 3: Planning with detailed thinking
        if self._should_stop:
            return generated
        while self._is_paused:
            await asyncio.sleep(0.5)

        self._emit_status(
            ExecutionStatus.PLANNING,
            "📋 Creating implementation plan...",
            "planning",
            0.45,
            {"thinking": "Determining file structure and dependencies..."}
        )
        await asyncio.sleep(0.4)

        file_plan = "\n".join([f"  • {f}" for f in target_files])
        self._emit_status(
            ExecutionStatus.PLANNING,
            "💭 Plan ready",
            "planning",
            0.5,
            {"thinking": f"File generation order:\n{file_plan}"}
        )
        await asyncio.sleep(0.3)

        # Phase 4: Generation with per-file updates
        total_files = len(target_files)
        for idx, file_path in enumerate(target_files):
            if self._should_stop:
                break
            while self._is_paused:
                await asyncio.sleep(0.5)

            progress = 0.5 + (0.35 * (idx / total_files))
            ext = Path(file_path).suffix.lower()

            # Show thinking for this file
            self._emit_status(
                ExecutionStatus.GENERATING,
                f"💭 Planning: {file_path}",
                "generation",
                progress,
                {"thinking": f"Generating {ext.lstrip('.')} file with appropriate structure..."}
            )
            await asyncio.sleep(0.3)

            # Show tool usage for code generation
            self._emit_status(
                ExecutionStatus.GENERATING,
                f"🔧 Using: Code Generator",
                "generation",
                progress + 0.05,
                {"tool": "code_gen", "file": file_path}
            )
            await asyncio.sleep(0.4)

            # Generate content
            content = self._generate_mock_content(file_path, ext, task)
            generated[file_path] = content

            # Show completion
            lines = len(content.split('\n'))
            self._emit_status(
                ExecutionStatus.GENERATING,
                f"📄 Generated: {file_path}",
                "generation",
                progress + 0.1,
                {"file": file_path, "lines": lines, "size": len(content)}
            )
            await asyncio.sleep(0.2)

        # Phase 5: Validation
        if self._should_stop:
            return generated
        while self._is_paused:
            await asyncio.sleep(0.5)

        self._emit_status(
            ExecutionStatus.VALIDATING,
            "🔧 Using: Syntax Validator",
            "validation",
            0.88,
            {"tool": "validator", "thinking": "Checking syntax and structure..."}
        )
        await asyncio.sleep(0.4)

        self._emit_status(
            ExecutionStatus.VALIDATING,
            "✅ Validation passed",
            "validation",
            0.92,
            {"thinking": f"All {len(generated)} files validated successfully."}
        )
        await asyncio.sleep(0.2)

        return generated

    def _generate_mock_content(self, file_path: str, ext: str, task: str) -> str:
        """Generate mock file content based on extension"""
        templates = {
            '.py': f'''"""
{file_path}
Generated for: {task[:100]}
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Result:
    """Result dataclass for function outputs"""
    success: bool
    data: Any
    message: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def main_function(input_data: Any) -> Result:
    """
    Main function implementation.

    Args:
        input_data: Input data to process

    Returns:
        Result object with processed data
    """
    try:
        # Process the input
        processed = process_data(input_data)
        return Result(success=True, data=processed, message="Processing complete")
    except Exception as e:
        return Result(success=False, data=None, message=str(e))


def process_data(data: Any) -> Dict[str, Any]:
    """Process input data"""
    return {{"input": data, "processed": True, "timestamp": datetime.now().isoformat()}}


def helper_function(value: str) -> str:
    """Helper utility function"""
    return value.strip().lower()


if __name__ == "__main__":
    # Example usage
    result = main_function("test input")
    print(f"Result: {{result}}")
''',
            '.js': f'''/**
 * {file_path}
 * Generated for: {task[:100]}
 */

// Check if React is available (loaded via importmap)
const isReact = typeof React !== 'undefined';

// Main Application
const App = {{
    state: {{
        initialized: false,
        data: null,
        count: 0,
        error: null
    }},

    async init() {{
        console.log('🚀 Initializing application...');
        try {{
            this.state.initialized = true;
            await this.loadData();
            this.render();
            this.setupEventListeners();
        }} catch (error) {{
            this.state.error = error.message;
            console.error('Init failed:', error);
        }}
    }},

    async loadData() {{
        return new Promise((resolve) => {{
            setTimeout(() => {{
                this.state.data = {{
                    loaded: true,
                    timestamp: new Date().toISOString(),
                    message: "Data loaded successfully!"
                }};
                resolve(this.state.data);
            }}, 100);
        }});
    }},

    setupEventListeners() {{
        const btn = document.getElementById('action-btn');
        if (btn) {{
            btn.addEventListener('click', () => this.handleAction());
        }}

        const incrementBtn = document.getElementById('increment-btn');
        if (incrementBtn) {{
            incrementBtn.addEventListener('click', () => this.increment());
        }}
    }},

    increment() {{
        this.state.count++;
        this.updateCounter();
    }},

    updateCounter() {{
        const counterEl = document.getElementById('counter');
        if (counterEl) {{
            counterEl.textContent = this.state.count;
        }}
    }},

    handleAction() {{
        const output = document.getElementById('output');
        if (output) {{
            const result = {{
                action: 'executed',
                timestamp: new Date().toISOString(),
                count: this.state.count,
                data: this.state.data
            }};
            output.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
        }}
    }},

    render() {{
        const container = document.getElementById('app');
        if (container) {{
            container.innerHTML = `
                <div class="app-content">
                    <header class="app-header">
                        <h1>🚀 Generated Application</h1>
                        <p class="subtitle">Built with ProjectDeveloper</p>
                    </header>

                    <main class="app-main">
                        <section class="card">
                            <h2>📊 Status</h2>
                            <p><strong>Initialized:</strong> ${{this.state.initialized ? '✅ Yes' : '❌ No'}}</p>
                            <p><strong>Data:</strong> ${{this.state.data ? '✅ Loaded' : '⏳ Loading...'}}</p>
                        </section>

                        <section class="card">
                            <h2>🔢 Counter Demo</h2>
                            <div class="counter-container">
                                <span class="counter-value" id="counter">${{this.state.count}}</span>
                                <button id="increment-btn" class="btn btn-primary">+ Increment</button>
                            </div>
                        </section>

                        <section class="card">
                            <h2>⚡ Actions</h2>
                            <button id="action-btn" class="btn btn-secondary">Execute Action</button>
                            <div id="output" class="output-area"></div>
                        </section>
                    </main>

                    <footer class="app-footer">
                        <p>Generated: ${{new Date().toLocaleString()}}</p>
                    </footer>
                </div>
            `;

            // Re-attach event listeners after render
            this.setupEventListeners();
        }}
    }}
}};

// Initialize on DOM ready
if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', () => App.init());
}} else {{
    App.init();
}}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = App;
}}
''',
            '.html': f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Application</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <!-- Main App Container - Used by JS/React/Vue -->
    <div id="app" class="app-container">
        <div class="loading-state">
            <div class="spinner"></div>
            <p>Loading application...</p>
        </div>
    </div>

    <!-- Root element for React apps -->
    <div id="root"></div>

    <!-- Load the application script -->
    <script src="script.js"></script>
</body>
</html>
''',
            '.css': f'''/*
 * {file_path}
 * Generated Styles for: {task[:80]}
 */

:root {{
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    --accent-color: #22d3ee;
    --bg-color: #0f172a;
    --bg-secondary: #1e293b;
    --card-bg: #1e293b;
    --text-color: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --success-color: #10b981;
    --error-color: #ef4444;
    --border-radius: 12px;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, var(--bg-color) 0%, var(--bg-secondary) 100%);
    color: var(--text-color);
    line-height: 1.6;
    min-height: 100vh;
}}

.app-container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
}}

/* Loading State */
.loading-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 50vh;
    color: var(--text-muted);
}}

.spinner {{
    width: 40px;
    height: 40px;
    border: 3px solid var(--bg-secondary);
    border-top-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
}}

@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}

/* Header */
.app-header {{
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
}}

.app-header h1 {{
    font-size: 2rem;
    margin-bottom: 0.5rem;
    color: white;
}}

.subtitle {{
    color: rgba(255, 255, 255, 0.8);
    font-size: 1rem;
}}

/* Cards */
.card {{
    background: var(--card-bg);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

.card h2 {{
    margin-bottom: 1rem;
    color: var(--accent-color);
    font-size: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.card p {{
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}}

/* Counter Demo */
.counter-container {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
}}

.counter-value {{
    font-size: 3rem;
    font-weight: 700;
    color: var(--primary-color);
    font-family: 'JetBrains Mono', monospace;
    min-width: 80px;
    text-align: center;
}}

/* Buttons */
.btn {{
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}}

.btn-primary {{
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}}

.btn-primary:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
}}

.btn-secondary {{
    background: var(--bg-secondary);
    color: var(--text-color);
    border: 1px solid var(--primary-color);
}}

.btn-secondary:hover {{
    background: var(--primary-color);
}}

/* Output Area */
.output-area {{
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.85rem;
    min-height: 80px;
    color: var(--accent-color);
    overflow-x: auto;
}}

.output-area pre {{
    margin: 0;
    white-space: pre-wrap;
}}

/* Footer */
.app-footer {{
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 2rem;
}}

/* Responsive */
@media (max-width: 640px) {{
    .app-container {{
        padding: 1rem;
    }}

    .app-header h1 {{
        font-size: 1.5rem;
    }}

    .counter-container {{
        flex-direction: column;
    }}
}}

/* Animation */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.app-content > * {{
    animation: fadeIn 0.5s ease forwards;
}}

.app-content > *:nth-child(1) {{ animation-delay: 0.1s; }}
.app-content > *:nth-child(2) {{ animation-delay: 0.2s; }}
.app-content > *:nth-child(3) {{ animation-delay: 0.3s; }}
.app-content > *:nth-child(4) {{ animation-delay: 0.4s; }}
''',
            '.json': f'''{{"name": "generated-project", "version": "1.0.0", "description": "{task[:100]}", "main": "main.py", "scripts": {{"start": "python main.py", "test": "python -m unittest"}}, "generated": "{datetime.now().isoformat()}"}}''',
            '.jsx': f'''/**
 * {file_path}
 * React Component - Generated for: {task[:80]}
 */

import React, {{ useState, useEffect }} from 'react';

// Main App Component
function App() {{
    const [count, setCount] = useState(0);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {{
        // Simulate data loading
        const timer = setTimeout(() => {{
            setData({{
                message: "Data loaded successfully!",
                timestamp: new Date().toISOString()
            }});
            setLoading(false);
        }}, 500);

        return () => clearTimeout(timer);
    }}, []);

    const handleAction = () => {{
        console.log('Action executed!', {{ count, data }});
        alert(`Action executed! Count: ${{count}}`);
    }};

    if (loading) {{
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <p>Loading...</p>
            </div>
        );
    }}

    return (
        <div className="app-container">
            <header className="app-header">
                <h1>⚛️ React Application</h1>
                <p className="subtitle">Built with ProjectDeveloper</p>
            </header>

            <main className="app-main">
                <section className="card">
                    <h2>📊 Status</h2>
                    <p><strong>Data:</strong> {{data?.message}}</p>
                    <p><strong>Time:</strong> {{data?.timestamp}}</p>
                </section>

                <section className="card">
                    <h2>🔢 Counter Demo</h2>
                    <div className="counter-container">
                        <span className="counter-value">{{count}}</span>
                        <div className="button-group">
                            <button
                                className="btn btn-primary"
                                onClick={{() => setCount(c => c + 1)}}
                            >
                                + Increment
                            </button>
                            <button
                                className="btn btn-secondary"
                                onClick={{() => setCount(0)}}
                            >
                                Reset
                            </button>
                        </div>
                    </div>
                </section>

                <section className="card">
                    <h2>⚡ Actions</h2>
                    <button className="btn btn-primary" onClick={{handleAction}}>
                        Execute Action
                    </button>
                </section>
            </main>

            <footer className="app-footer">
                <p>Generated with ❤️ by ProjectDeveloper</p>
            </footer>
        </div>
    );
}}

export default App;
''',
            '.tsx': f'''/**
 * {file_path}
 * TypeScript React Component - Generated for: {task[:80]}
 */

import React, {{ useState, useEffect, FC }} from 'react';

interface AppState {{
    count: number;
    data: {{ message: string; timestamp: string }} | null;
    loading: boolean;
}}

const App: FC = () => {{
    const [count, setCount] = useState<number>(0);
    const [data, setData] = useState<AppState['data']>(null);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {{
        const timer = setTimeout(() => {{
            setData({{
                message: "TypeScript data loaded!",
                timestamp: new Date().toISOString()
            }});
            setLoading(false);
        }}, 500);

        return () => clearTimeout(timer);
    }}, []);

    if (loading) {{
        return <div className="loading">Loading...</div>;
    }}

    return (
        <div className="app-container">
            <h1>⚛️ TypeScript React App</h1>
            <p>Count: {{count}}</p>
            <p>Data: {{data?.message}}</p>
            <button onClick={{() => setCount(c => c + 1)}}>
                Increment
            </button>
        </div>
    );
}};

export default App;
''',
            '.vue': f'''<!--
  {file_path}
  Vue Single File Component - Generated for: {task[:80]}
-->

<template>
    <div class="app-container">
        <header class="app-header">
            <h1>💚 Vue Application</h1>
            <p class="subtitle">Built with ProjectDeveloper</p>
        </header>

        <main class="app-main">
            <section class="card">
                <h2>📊 Status</h2>
                <p><strong>Message:</strong> {{{{ message }}}}</p>
                <p><strong>Loading:</strong> {{{{ loading ? 'Yes' : 'No' }}}}</p>
            </section>

            <section class="card">
                <h2>🔢 Counter Demo</h2>
                <div class="counter-container">
                    <span class="counter-value">{{{{ count }}}}</span>
                    <div class="button-group">
                        <button class="btn btn-primary" @click="increment">
                            + Increment
                        </button>
                        <button class="btn btn-secondary" @click="reset">
                            Reset
                        </button>
                    </div>
                </div>
            </section>

            <section class="card">
                <h2>📝 Input Demo</h2>
                <input
                    v-model="inputText"
                    placeholder="Type something..."
                    class="input-field"
                />
                <p v-if="inputText">You typed: {{{{ inputText }}}}</p>
            </section>
        </main>

        <footer class="app-footer">
            <p>Generated with 💚 by ProjectDeveloper</p>
        </footer>
    </div>
</template>

<script>
export default {{
    name: 'App',
    data() {{
        return {{
            count: 0,
            message: 'Vue app loaded successfully!',
            loading: false,
            inputText: ''
        }};
    }},
    methods: {{
        increment() {{
            this.count++;
        }},
        reset() {{
            this.count = 0;
        }}
    }},
    mounted() {{
        console.log('Vue app mounted!');
    }}
}};
</script>

<style scoped>
.input-field {{
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #334155;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.2);
    color: #f1f5f9;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}}

.input-field:focus {{
    outline: none;
    border-color: #6366f1;
}}
</style>
''',
        }

        return templates.get(ext, f"# {file_path}\n# Generated for: {task}\n")

    async def _real_execute(
        self,
        task: str,
        target_files: List[str],
        max_retries: int,
        auto_research: bool
    ) -> Dict[str, str]:
        """
        Execute using real ProjectDeveloperEngine with full context continuity.

        Features:
        - Maintains execution history for follow-up tasks
        - Provides detailed status updates with thinking/tool info
        - Auto-continues in same project with latest context
        """
        if not self._developer:
            raise RuntimeError("ProjectDeveloperEngine not initialized")

        # Track execution context for continuity
        if not hasattr(self, '_execution_history'):
            self._execution_history = []

        # Build context from previous executions in this session
        previous_context = ""
        if self._execution_history:
            ctx_parts = ["PREVIOUS TASKS IN THIS SESSION:"]
            for prev in self._execution_history[-3:]:  # Last 3 executions
                ctx_parts.append(f"- Task: {prev['task'][:100]}")
                ctx_parts.append(f"  Files: {', '.join(prev['files'])}")
            previous_context = "\n".join(ctx_parts)

        # Enhanced task with context
        enhanced_task = task
        if previous_context:
            enhanced_task = f"{task}\n\n{previous_context}"

        # Phase progress mapping
        phase_progress = {
            "ANALYSIS": (ExecutionStatus.ANALYZING, 0.15, "analysis"),
            "RESEARCH": (ExecutionStatus.RESEARCHING, 0.30, "research"),
            "MULTI_SPEC": (ExecutionStatus.PLANNING, 0.45, "planning"),
            "GENERATION": (ExecutionStatus.GENERATING, 0.60, "generation"),
            "VALIDATION": (ExecutionStatus.VALIDATING, 0.80, "validation"),
            "SYNC": (ExecutionStatus.SYNCING, 0.95, "sync"),
            "COMPLETED": (ExecutionStatus.COMPLETED, 1.0, "complete"),
            "FAILED": (ExecutionStatus.FAILED, 0.0, "failed"),
        }

        # Current file being processed
        current_file_idx = [0]
        total_files = len(target_files)

        # Wrap the developer's _log method for detailed status updates
        original_log = self._developer._log

        def enhanced_status_log(message: str):
            """Enhanced logging that extracts thinking and tool usage"""
            original_log(message)

            # Detect phase from message
            status = ExecutionStatus.GENERATING
            progress = 0.5
            phase = "execution"
            details = {}

            msg_lower = message.lower()

            # Phase detection with thinking details
            if "📍 phase:" in msg_lower:
                phase_name = message.split(":")[-1].strip().upper()
                if phase_name in phase_progress:
                    status, progress, phase = phase_progress[phase_name]
                    details["thinking"] = f"Entering {phase_name} phase..."

            elif "analyzing" in msg_lower or "📊" in message:
                status = ExecutionStatus.ANALYZING
                progress = 0.2
                phase = "analysis"
                details["thinking"] = "Analyzing project context and dependencies..."

            elif "research" in msg_lower or "🔍" in message:
                status = ExecutionStatus.RESEARCHING
                progress = 0.35
                phase = "research"
                if ":" in message:
                    topic = message.split(":")[-1].strip()
                    details["tool"] = "doc_search"
                    details["thinking"] = f"Researching: {topic}"

            elif "spec" in msg_lower or "📋" in message:
                status = ExecutionStatus.PLANNING
                progress = 0.45
                phase = "planning"
                details["thinking"] = "Creating detailed implementation specification..."

            elif "generating" in msg_lower or "generat" in msg_lower or "💻" in message:
                status = ExecutionStatus.GENERATING
                phase = "generation"

                # Track file progress
                if "[" in message and "/" in message:
                    try:
                        # Extract [1/3] style progress
                        bracket = message[message.find("["):message.find("]")+1]
                        parts = bracket.strip("[]").split("/")
                        current_file_idx[0] = int(parts[0])
                        progress = 0.5 + (0.25 * current_file_idx[0] / total_files)
                    except:
                        progress = 0.6
                else:
                    progress = 0.6

                # Extract file being generated
                for tf in target_files:
                    if tf in message:
                        details["file"] = tf
                        details["tool"] = "code_generator"
                        details["thinking"] = f"Generating {tf}..."
                        break

            elif "validat" in msg_lower or "🔍" in message:
                status = ExecutionStatus.VALIDATING
                progress = 0.82
                phase = "validation"
                details["tool"] = "lsp_validator"

                if "error" in msg_lower or "❌" in message:
                    details["thinking"] = "Found issues, attempting auto-fix..."
                elif "✓" in message or "pass" in msg_lower:
                    details["thinking"] = "Validation passed!"
                else:
                    details["thinking"] = "Running LSP and runtime validation..."

            elif "fix" in msg_lower or "🔧" in message:
                status = ExecutionStatus.FIXING
                progress = 0.85
                phase = "fixing"
                details["tool"] = "auto_fixer"
                details["thinking"] = "Applying automatic fixes..."

            elif "sync" in msg_lower or "💾" in message:
                status = ExecutionStatus.SYNCING
                progress = 0.95
                phase = "sync"
                details["thinking"] = "Writing files to disk..."

            elif "✅" in message and "complete" in msg_lower:
                status = ExecutionStatus.COMPLETED
                progress = 1.0
                phase = "complete"
                details["thinking"] = "All tasks completed successfully!"

            elif "❌" in message or "fail" in msg_lower or "error" in msg_lower:
                status = ExecutionStatus.FAILED
                progress = 0.0
                phase = "error"
                details["thinking"] = message

            # Emit the status update
            self._emit_status(status, message, phase, progress, details)

        # Replace log function
        self._developer._log = enhanced_status_log

        try:
            self._emit_status(
                ExecutionStatus.ANALYZING,
                "🚀 Starting ProjectDeveloperEngine execution...",
                "init",
                0.1,
                {"thinking": f"Task: {task[:100]}...", "files": target_files}
            )

            # Execute with the real engine
            success, generated_files = await self._developer.execute(
                task=enhanced_task,
                target_files=target_files,
                max_retries=max_retries,
                auto_research=auto_research
            )

            # Record execution for context continuity
            self._execution_history.append({
                "task": task,
                "files": list(generated_files.keys()),
                "success": success,
                "timestamp": time.time()
            })

            # Keep only last 10 executions
            if len(self._execution_history) > 10:
                self._execution_history = self._execution_history[-10:]

            if not success:
                # Get errors from developer state if available
                errors = []
                if hasattr(self._developer, '_executions') and self._developer._executions:
                    latest_state = list(self._developer._executions.values())[-1]
                    errors = latest_state.errors if hasattr(latest_state, 'errors') else []

                self._emit_status(
                    ExecutionStatus.FAILED,
                    f"❌ Development failed: {errors[0] if errors else 'Unknown error'}",
                    "failed",
                    0.0,
                    {"errors": errors}
                )
                raise RuntimeError(f"Development execution failed: {errors}")

            self._emit_status(
                ExecutionStatus.COMPLETED,
                f"✅ Successfully generated {len(generated_files)} file(s)",
                "complete",
                1.0,
                {"files": list(generated_files.keys())}
            )

            return generated_files

        finally:
            # Restore original log function
            self._developer._log = original_log

    def get_execution_context(self) -> Dict[str, Any]:
        """Get current execution context for UI display"""
        context = {
            "has_history": hasattr(self, '_execution_history') and bool(self._execution_history),
            "execution_count": len(getattr(self, '_execution_history', [])),
            "last_task": None,
            "last_files": [],
        }

        if context["has_history"]:
            last = self._execution_history[-1]
            context["last_task"] = last["task"]
            context["last_files"] = last["files"]

        return context

    def _create_result(
        self,
        success: bool,
        generated_files: Dict[str, str],
        errors: List[str],
        execution_time: float,
        phases_completed: List[str]
    ) -> ExecutionResult:
        """Create execution result object"""
        return ExecutionResult(
            success=success,
            generated_files=generated_files,
            errors=errors,
            execution_time=execution_time,
            phases_completed=phases_completed
        )

    def clear_execution_context(self):
        """Clear execution history for fresh context"""
        self._execution_history = []
        self._emit_status(
            ExecutionStatus.PENDING,
            "🧹 Execution context cleared",
            "clear",
            0.0
        )

    def stop_execution(self):
        """Stop current execution"""
        self._should_stop = True
        self._emit_status(
            ExecutionStatus.PAUSED,
            "⏹️ Stopping execution...",
            "stop",
            0.0
        )

    def pause_execution(self):
        """Pause current execution"""
        self._is_paused = True
        self._emit_status(
            ExecutionStatus.PAUSED,
            "⏸️ Execution paused",
            "pause",
            0.0
        )

    def resume_execution(self):
        """Resume paused execution"""
        self._is_paused = False
        self._emit_status(
            ExecutionStatus.GENERATING,
            "▶️ Execution resumed",
            "resume",
            0.0
        )

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a generated file"""
        full_path = self.workspace_path / file_path
        if full_path.exists():
            return full_path.read_text(encoding='utf-8')
        return self._mock_files.get(file_path)

    def list_files(self) -> List[str]:
        """List all files in workspace"""
        files = []
        for path in self.workspace_path.rglob('*'):
            if path.is_file():
                files.append(str(path.relative_to(self.workspace_path)))
        return sorted(files)

    async def run_tests(self, file_path: str) -> Dict[str, Any]:
        """Run tests for a specific file"""
        full_path = self.workspace_path / file_path
        if not full_path.exists():
            return {"success": False, "error": "File not found"}

        if file_path.endswith('.py'):
            return await self._run_python_tests(full_path)
        else:
            return {"success": True, "message": "No tests available for this file type"}

    async def _run_python_tests(self, file_path: Path) -> Dict[str, Any]:
        """Run Python unit tests"""
        import subprocess

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'unittest', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_path)
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Test execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute arbitrary code snippet"""
        if language != "python":
            return {"success": False, "error": f"Language {language} not supported"}

        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_path)
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    async def close(self):
        """Cleanup resources"""
        if self._developer:
            await self._developer.close()
