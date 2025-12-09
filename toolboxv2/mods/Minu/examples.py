"""
Minu UI Framework - Example Module (ÜBERARBEITET)
==================================
Demonstrates how to create reactive UIs with Minu in Toolbox modules.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

# Import Minu components
from .core import (
    Alert, Badge, Button, Card, Checkbox, Column, Divider, Form,
    Grid, Heading, Icon, Input, ListItem, MinuView, Modal, Progress,
    Row, Select, Spacer, Spinner, State, Switch, Table, Tabs,
    Text, Widget, List as MinuList
)

from toolboxv2 import App, RequestData, Result, get_app

# Module metadata
Name = 'Minu'
export = get_app(f"{Name}.Export").tb
version = '0.1.0'


# ============================================================================
# EXAMPLE 1: Simple Counter
# ============================================================================

class CounterView(MinuView):
    """A simple counter demonstrating reactive state."""
    count = State(0)

    def render(self):
        return Card(
            Heading("Counter Demo", level=2),
            Text(f"Current count: {self.count.value}", className="text-2xl font-bold", bind="count"),
            Spacer(),
            Row(
                Button("−", on_click="decrement", variant="secondary"),
                Button("+", on_click="increment", variant="primary"),
                gap="2"
            ),
            Row(
                Button("Reset", on_click="reset", variant="ghost"),
                gap="2"
            ),
            title="Reactive Counter",
            className="card animate-fade-in"
        )

    async def increment(self, event):
        self.count.value += 1

    async def decrement(self, event):
        self.count.value = max(0, self.count.value - 1)

    async def reset(self, event):
        self.count.value = 0


# ============================================================================
# EXAMPLE 2: User Profile Form
# ============================================================================

class ProfileFormView(MinuView):
    """A form demonstrating two-way data binding."""
    name = State("")
    email = State("")
    role = State("user")
    notifications = State(True)
    saved = State(False)

    def render(self):
        return Card(
            Heading("User Profile", level=2),
            Form(
                Input(
                    placeholder="Your name",
                    value=self.name.value,
                    bind="name",
                    label="Name"
                ),
                Input(
                    placeholder="your@email.com",
                    value=self.email.value,
                    bind="email",
                    input_type="email",
                    label="Email"
                ),
                Select(
                    options=[
                        {"value": "user", "label": "User"},
                        {"value": "admin", "label": "Administrator"},
                        {"value": "moderator", "label": "Moderator"}
                    ],
                    value=self.role.value,
                    bind="role",
                    label="Role"
                ),
                Spacer(),
                Switch(
                    label="Email notifications",
                    checked=self.notifications.value,
                    bind="notifications"
                ),
                Divider(),
                Row(
                    Button("Save Profile", on_click="save", variant="primary"),
                    Button("Cancel", on_click="cancel", variant="secondary"),
                    justify="end"
                ),
                on_submit="save"
            ),
            Alert(
                "Profile saved successfully!",
                variant="success",
                dismissible=True
            ) if self.saved.value else None,
            title="Edit Profile",
            className="card max-w-md"
        )

    async def save(self, event):
        if not self.name.value or not self.email.value:
            return
        self.saved.value = True

    async def cancel(self, event):
        self.name.value = ""
        self.email.value = ""
        self.role.value = "user"
        self.notifications.value = True
        self.saved.value = False


# ============================================================================
# EXAMPLE 3: Task List
# ============================================================================

# ============================================================================
# EXAMPLE 3: Task List
# ============================================================================


class TaskListView(MinuView):
    tasks = State(
        [
            {"id": 1, "text": "Learn Minu UI", "done": False},
            {"id": 2, "text": "Build awesome apps", "done": False},
        ]
    )
    new_task = State("")
    filter_mode = State("all")

    def render(self):
        tasks = self.tasks.value
        filter_mode = self.filter_mode.value

        if filter_mode == "active":
            filtered = [t for t in tasks if not t["done"]]
        elif filter_mode == "completed":
            filtered = [t for t in tasks if t["done"]]
        else:
            filtered = tasks

        completed_count = len([t for t in tasks if t["done"]])
        total = len(tasks)

        return Card(
            Heading("Task List", level=2),
            Progress(
                value=int((completed_count / total * 100) if total else 0),
                label=f"{completed_count}/{total} done",
            ),
            Spacer(),
            Row(
                Input(
                    placeholder="Add task...",
                    value=self.new_task.value,
                    bind="new_task",
                    on_submit="add_task",
                ),
                Button("+", on_click="add_task", variant="primary"),
                gap="2",
            ),
            Spacer(),
            Row(
                Button(
                    "All",
                    on_click="filter_all",
                    variant="primary" if filter_mode == "all" else "ghost",
                ),
                Button(
                    "Active",
                    on_click="filter_active",
                    variant="primary" if filter_mode == "active" else "ghost",
                ),
                Button(
                    "Done",
                    on_click="filter_completed",
                    variant="primary" if filter_mode == "completed" else "ghost",
                ),
                gap="1",
            ),
            Divider(),
            Column(
                *[
                    Row(
                        Checkbox(
                            label="", checked=t["done"], on_change=f"toggle_{t['id']}"
                        ),
                        Text(
                            t["text"],
                            className="flex-1"
                            + (" line-through text-gray-400" if t["done"] else ""),
                        ),
                        Button("×", on_click=f"del_{t['id']}", variant="ghost"),
                        className="p-2 hover:bg-gray-50 rounded",
                    )
                    for t in filtered
                ],
                gap="1",
            )
            if filtered
            else Text("No tasks", className="text-gray-400 text-center p-4"),
            title="Tasks",
            className="card max-w-md",
        )

    async def add_task(self, event):
        if self.new_task.value.strip():
            tasks = list(self.tasks.value)
            new_id = max([t["id"] for t in tasks], default=0) + 1
            tasks.append({"id": new_id, "text": self.new_task.value, "done": False})
            self.tasks.value = tasks
            self.new_task.value = ""

    async def filter_all(self, e):
        self.filter_mode.value = "all"

    async def filter_active(self, e):
        self.filter_mode.value = "active"

    async def filter_completed(self, e):
        self.filter_mode.value = "completed"

    def __getattr__(self, name):
        if name.startswith("toggle_"):
            tid = int(name.split("_")[1])
            return lambda e: self._toggle(tid)
        if name.startswith("del_"):
            tid = int(name.split("_")[1])
            return lambda e: self._del(tid)
        raise AttributeError(name)

    def _toggle(self, tid):
        tasks = [t.copy() for t in self.tasks.value]
        for t in tasks:
            if t["id"] == tid:
                t["done"] = not t["done"]
        self.tasks.value = tasks

    def _del(self, tid):
        self.tasks.value = [t for t in self.tasks.value if t["id"] != tid]


# ============================================================================
# EXAMPLE 4: Data Table
# ============================================================================

class DataTableView(MinuView):
    """A data table with sorting and filtering."""
    search = State("")
    data = State([
        {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "Admin"},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "User"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "role": "User"},
        {"id": 4, "name": "Diana", "email": "diana@example.com", "role": "Moderator"},
        {"id": 5, "name": "Eve", "email": "eve@example.com", "role": "User"},
    ])
    selected_row = State(None)

    def render(self):
        search = self.search.value.lower()
        data = self.data.value

        filtered = [
            row for row in data
            if not search or
               search in row["name"].lower() or
               search in row["email"].lower()
        ]

        return Card(
            Heading("User Management", level=2),
            Row(
                Input(
                    placeholder="Search users...",
                    value=self.search.value,
                    bind="search"
                ),
                Button("Add User", on_click="add_user", variant="primary"),
                Button("Export", on_click="export_data", variant="secondary"),
                justify="between"
            ),
            Spacer(),
            Text(f"Showing {len(filtered)} of {len(data)} users", className="text-sm text-secondary"),
            Spacer(),
            Table(
                columns=[
                    {"key": "id", "label": "#"},
                    {"key": "name", "label": "Name"},
                    {"key": "email", "label": "Email"},
                    {"key": "role", "label": "Role"}
                ],
                data=filtered,
                on_row_click="select_row"
            ),
            Card(
                Heading("Selected User", level=4),
                Text(f"Name: {self.selected_row.value['name']}"),
                Text(f"Email: {self.selected_row.value['email']}"),
                Badge(self.selected_row.value['role'], variant="primary"),
                Row(
                    Button("Edit", on_click="edit_user", variant="secondary"),
                    Button("Delete", on_click="delete_user", variant="ghost"),
                    gap="2"
                ),
                className="mt-4 p-4 bg-neutral-50"
            ) if self.selected_row.value else None,
            title="Data Table Demo",
            className="card"
        )

    async def select_row(self, event):
        self.selected_row.value = event

    async def add_user(self, event):
        pass

    async def edit_user(self, event):
        pass

    async def delete_user(self, event):
        if self.selected_row.value:
            data = [d for d in self.data.value if d["id"] != self.selected_row.value["id"]]
            self.data.value = data
            self.selected_row.value = None

    async def export_data(self, event):
        pass


# ============================================================================
# INITIALIZE & REGISTER
# ============================================================================

@export(mod_name=Name, name="initialize", initial=True)
def initialize(app: App, **kwargs) -> Result:
    """Initialize module and register all views"""
    from toolboxv2.mods.Minu import register_view

    # Register all example views
    register_view("counter", CounterView)
    register_view("profile_form", ProfileFormView)
    register_view("task_list", TaskListView)
    register_view("data_table", DataTableView)

    # Register UI route
    app.run_any(
        ("CloudM", "add_ui"),
        name="MinuExample",
        title="Minu UI Examples",
        path=f"/api/{Name}/demo",
        description="Minu UI Framework Examples",
        auth=False  # Kein Auth für Demo
    )

    return Result.ok(info="Minu UI Examples initialized")


# ============================================================================
# DEMO PAGE ENDPOINT
# ============================================================================

@export(mod_name=Name, name="demo", api=True, api_methods=["GET"], version=version)
async def get_demo_page(app: App) -> Result:
    """Serves the demo page with all examples"""

    html = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Minu UI Framework - Examples</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        :root {
            --color-primary-500: #3b82f6;
            --color-neutral-50: #f9fafb;
            --color-neutral-200: #e5e7eb;
            --color-neutral-800: #1f2937;
            --space-4: 1rem;
            --radius-md: 8px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background: var(--color-neutral-50);
            padding: var(--space-4);
            color: var(--color-neutral-800);
        }

        .page-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .page-header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .page-header p {
            color: #6b7280;
            font-size: 1.1rem;
        }

        .nav-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--color-neutral-200);
            justify-content: center;
            flex-wrap: wrap;
        }

        .nav-tabs button {
            padding: 0.75rem 1.5rem;
            background: none;
            border: none;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
            font-size: 1rem;
            transition: all 0.2s;
        }

        .nav-tabs button:hover {
            background: var(--color-neutral-50);
        }

        .nav-tabs button.active {
            border-bottom-color: var(--color-primary-500);
            color: var(--color-primary-500);
            font-weight: 600;
        }

        #view-container {
            max-width: 900px;
            margin: 0 auto;
            min-height: 400px;
        }

        .card {
            border-radius: var(--radius-md);
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }

        .btn {
            padding: 0.5rem 1rem;
            border-radius: var(--radius-md);
            border: none;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 1rem;
        }

        .btn-primary {
            background: var(--color-primary-500);
        }

        .btn-primary:hover {
            background: #2563eb;
        }

        .btn-secondary {
            background: var(--color-neutral-200);
        }

        .btn-secondary:hover {
            background: #d1d5db;
        }

        .btn-ghost {
            background: transparent;
        }

        .btn-ghost:hover {
            background: var(--color-neutral-50);
        }

        .flex { display: flex; }
        .flex-col { flex-direction: column; }
        .flex-1 { flex: 1; }
        .gap-1 { gap: 0.25rem; }
        .gap-2 { gap: 0.5rem; }
        .gap-4 { gap: 1rem; }
        .items-center { align-items: center; }
        .justify-between { justify-content: space-between; }
        .justify-end { justify-content: flex-end; }

        input, select, textarea {
            padding: 0.5rem;
            border: 1px solid var(--color-neutral-200);
            border-radius: var(--radius-md);
            width: 100%;
            font-size: 1rem;
        }

        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--color-primary-500);
        }

        h1, h2, h3, h4 {
            margin-bottom: 0.5rem;
        }

        .text-2xl { font-size: 1.5rem; }
        .font-bold { font-weight: 700; }
        .text-sm { font-size: 0.875rem; }
        .text-secondary { color: #6b7280; }

        .loading {
            text-align: center;
            padding: 2rem;
            color: #6b7280;
        }

        .error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 1rem;
            border-radius: var(--radius-md);
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <div class="page-header">
        <h1>🎨 Minu UI Framework</h1>
        <p>Interactive Examples & Component Showcase</p>
    </div>

    <div class="nav-tabs">
        <button onclick="loadView('counter')" class="active" data-view="counter">
            Counter
        </button>
        <button onclick="loadView('profile_form')" data-view="profile_form">
            Profile Form
        </button>
        <button onclick="loadView('task_list')" data-view="task_list">
            Task List
        </button>
        <button onclick="loadView('data_table')" data-view="data_table">
            Data Table
        </button>
    </div>

    <div id="view-container">
        <div class="card loading">
            <p>Loading Minu Framework...</p>
        </div>
    </div>

    <script type="module">
        let currentRenderer = null;

        // Load view function
        window.loadView = async function(viewName) {
            const container = document.getElementById('view-container');

            // Update active tab
            document.querySelectorAll('.nav-tabs button').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.view === viewName);
            });

            // Show loading
            container.innerHTML = '<div class="card loading"><p>Loading view...</p></div>';

            try {
                // Cleanup previous renderer
                if (currentRenderer) {
                    currentRenderer.unmount();
                }

                // Wait for TB to be ready
                if (!window.TB || !window.TB.ui) {
                    throw new Error('TBJS not loaded');
                }

                // Mount new view
                currentRenderer = await window.TB.ui.mountMinuView(
                    container,
                    viewName
                );

                console.log(`[Minu Demo] Loaded view: ${viewName}`);
            } catch (error) {
                console.error('[Minu Demo] Error loading view:', error);
                container.innerHTML = `
                    <div class="error">
                        <strong>Error loading view:</strong> ${error.message}
                        <br><br>
                        Make sure the Minu module is properly initialized.
                    </div>
                `;
            }
        };

        // Wait for TBJS to load, then load default view
        if (window.TB && window.TB.onLoaded) {
            window.TB.onLoaded(() => {
                loadView('counter');
            });
        } else {
            // Fallback: wait for window load
            window.addEventListener('load', () => {
                setTimeout(() => loadView('counter'), 100);
            });
        }
    </script>
</body>
</html>
    """

    return Result.html(data=html)
