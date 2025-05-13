
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from toolboxv2 import Code, Result, get_app
from toolboxv2.utils.extras.base_widget import get_user_from_request
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage
from toolboxv2.utils.system.session import RequestSession

Name = "DoNext"
version = "0.0.1"
export = get_app(f"{Name}.Export").tb

template = """<script defer="defer" src="/index.js" type=module></script><div><!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Action Manager</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        body {
            background: #f0f2f5;
            color: #1a1a1a;
            min-height: 100vh;
        }

        .app-container {
            max-width: 800px;
            margin: 0 auto;
            flex-direction: column;
            gap: 16px;
            padding: 20px;
        }

        .card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .badge {
            background: #e3e8ef;
            padding: 2px 4px;
            border-radius: 6px;
            font-size: 0.9em;
            margin-left: 6px;
            max-width: 100px;
            color: #000000;
        }

        .badge.priority-1 { background: #ff4d4f; color: white; }
        .badge.priority-2 { background: #ff7a45; color: white; }
        .badge.priority-3 { background: #ffa940; color: white; }
        .badge.priority-4 { background: #bae637; color: black; }
        .badge.priority-5 { background: #73d13d; color: white; }

        .badge.status-in_progress { background: #1890ff; color: white; }
        .badge.status-completed { background: #52c41a; color: white; }
        .badge.status-not_started { background: #d9d9d9; color: black; }

        .action-content {
            margin: 16px 0;
            font-size: 1.1em;
        }

        .button-group {
            display: flex;
            gap: 8px;
            margin-top: 16px;
        }

        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.2s;
        }

        .btn-primary {
            background: #1890ff;
            color: white;
        }

        .btn-primary:hover {
            background: #096dd9;
        }

        .btn-secondary {
            background: #f0f0f0;
            color: #1a1a1a;
        }

        .btn-secondary:hover {
            background: #d9d9d9;
        }

        .btn-remove {
            background: #a81103;
            color: #1a1a1a;
        }

        .btn-remove:hover {
            background: #75150c;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
        }

        .modal-content {
            background: white;
            border-radius: 16px;
            padding: 24px;
            width: 90%;
            max-width: 500px;
            margin: 50px auto;
            height: 90%;
            overflow-y: auto;
        }

        .input-group {
            margin-bottom: 16px;
        }

        .input-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .input-group input,
        .input-group select,
        .input-group textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            font-size: 1em;
            background-color:  #e3e8ef;
            color: #000000;
        }

        .history-section {
            transition: max-height 0.3s ease-out;
            overflow: hidden;
        }

        .history-header {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .history-list {
            margin-top: 16px;
        }

        .history-item {
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .history-item:last-child {
            border-bottom: none;
        }

        .action-hierarchy {
            margin-top: 16px;
        }

        .sub-action {
            margin-left: 24px;
            padding-left: 12px;
            border-left: 2px solid #f0f0f0;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }

        .tab {
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            background: #f0f0f0;
        }

        .tab.active {
            background: #1890ff;
            color: white;
        }
        .current-action {
            background: #2C3E50;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

    </style>
</head>
<body>
    <div class="app-container">
        <div class="tabs">
            <div class="tab active" onclick="switchTab('main')">Current Action</div>
            <div class="tab" onclick="switchTab('all')">All Actions</div>
            <div class="tab" onclick="switchTab('history')">History</div>
        </div>

        <div id="main-tab">
            <section class="card current-action">
                <div class="card-header">
                    <h2>Current Action</h2>
                    <div>
                        <span class="badge" id="elapsed-time" style="background: #2C3E50">00:00:00</span>
                    </div>
                </div>
                <div class="action-content" id="current-action-content">
                    No current action
                </div>
                <div class="action-hierarchy" id="current-sub-actions">
                    <!-- Sub-actions will be rendered here -->
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="openNewTaskModal()">New Action</button>
                    <button class="btn btn-secondary" onclick="completeCurrentAction()">Complete</button>
                </div>
            </section>

            <section class="card suggestion" id="suggestion1">
                <div class="card-header">
                    <h3>Next Suggested Action</h3>
                    <span class="badge" id="suggestion1-priority"></span>
                </div>
                <div class="action-content" id="suggestion1-content">
                    Loading...
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="startSuggestion(1)">Start</button>
                </div>
            </section>

            <section class="card suggestion" id="suggestion2">
                <div class="card-header">
                    <h3>Alternative Action</h3>
                    <span class="badge" id="suggestion2-priority"></span>
                </div>
                <div class="action-content" id="suggestion2-content">
                    Loading...
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="startSuggestion(2)">Start</button>
                </div>
            </section>
        </div>

        <div id="all-actions-tab" style="display: none;">
            <section class="card">
                <div class="card-header">
                    <h2>All Actions</h2>
                    <button class="btn btn-primary" onclick="openNewTaskModal()">New Action</button>
                </div>
                <div id="actions-hierarchy">
                    <!-- Actions hierarchy will be rendered here -->
                </div>
            </section>
        </div>

        <div id="history-tab" style="display: none;">
            <section class="card">
                <div class="card-header">
                    <h2>Action History</h2>
                </div>
                <div class="history-list" id="history-list">
                    <!-- History items will be rendered here -->
                </div>
            </section>
        </div>
    </div>

    <!-- New Action Modal -->
    <div class="modal" id="newTaskModal">
        <div class="modal-content">
            <div class="card-header">
                <h2>Create New Action</h2>
                <button class="btn btn-secondary" onclick="closeNewTaskModal()">×</button>
            </div>
            <form id="newTaskForm" onsubmit="createNewTask(event)">
                <div class="input-group">
                    <label for="taskTitle">Title</label>
                    <input type="text" id="taskTitle" required>
                </div>
                <div class="input-group">
                    <label for="taskDescription">Description</label>
                    <textarea id="taskDescription" rows="3"></textarea>
                </div>
                <div class="input-group">
                    <label for="taskParent">Parent Action</label>
                    <select id="taskParent">
                        <option value="">None (Top-level action)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="taskFrequency">Frequency</label>
                    <select id="taskFrequency" required>
                        <option value="one_time">One Time</option>
                        <option value="daily">Daily</option>
                        <option value="weekly">Weekly</option>
                        <option value="monthly">Monthly</option>
                        <option value="annually">Annually</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="taskPriority">Priority (1-5)</label>
                    <select id="taskPriority" required>
                        <option value="1">1 (Highest)</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5 (Lowest)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="taskFixedTime">Fixed Time (Optional)</label>
                    <input type="datetime-local" id="taskFixedTime">
                </div>
                <div class="button-group">
                    <button type="submit" class="btn btn-primary">Create</button>
                </div>
            </form>
        </div>
    </div>

<script unSave="true">
if (window.history.state){
// State management
let currentAction = null;
let suggestions = [];
let allActions = {};
let history = [];
let elapsedTimeInterval;

const API_BASE =  "/api/DoNext";

// Utility functions
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function formatDuration(ms) {
    const seconds = Math.floor((ms / 1000) % 60);
    const minutes = Math.floor((ms / (1000 * 60)) % 60);
    const hours = Math.floor(ms / (1000 * 60 * 60));
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// UI functions
function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');

    document.getElementById('main-tab').style.display = tab === 'main' ? 'block' : 'none';
    document.getElementById('all-actions-tab').style.display = tab === 'all' ? 'block' : 'none';
    document.getElementById('history-tab').style.display = tab === 'history' ? 'block' : 'none';

    if (tab === 'all') {
        refreshActionHierarchy();
    } else if (tab === 'history') {
        refreshHistory();
    }
}

function updateCurrentActionDisplay() {
    const content = document.getElementById('current-action-content');
    const subActions = document.getElementById('current-sub-actions');

    if (currentAction) {
        content.innerHTML = `
            <h3>${currentAction.title}</h3>
            ${currentAction.description ? `<p>${currentAction.description}</p>` : ''}
            <div class="badge priority-${currentAction.priority}">Priority ${currentAction.priority}</div>
            <div class="badge">${currentAction.frequency}</div>
        `;

        // Show sub-actions if any
        if (allActions[currentAction.id]) {
            subActions.innerHTML = allActions[currentAction.id].map(subAction => `
                <div class="sub-action">
                    <h4>${subAction.title}</h4>
                    <div class="badge priority-${subAction.priority}">Priority ${subAction.priority}</div>
                    <button class="btn btn-secondary" onclick="startAction('${subAction.id}')">Start</button>
                </div>
            `).join('');
        } else {
            subActions.innerHTML = '';
        }

        // Start elapsed time counter
        let startTime = new Date();
        if (elapsedTimeInterval) clearInterval(elapsedTimeInterval);
        elapsedTimeInterval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            document.getElementById('elapsed-time').textContent = formatDuration(elapsed);
        }, 1000);
    } else {
        content.innerHTML = 'No current action';
        subActions.innerHTML = '';
        if (elapsedTimeInterval) {
            clearInterval(elapsedTimeInterval);
            document.getElementById('elapsed-time').textContent = '00:00:00';
        }
    }
}

function updateSuggestions() {
    fetch(API_BASE+'/suggestions')
        .then(response => response.json())
        .then(data => {
            suggestions = data;
            data.forEach((suggestion, index) => {
                if (suggestion) {
                    const content = document.getElementById(`suggestion${index + 1}-content`);
                    const priority = document.getElementById(`suggestion${index + 1}-priority`);

                    content.innerHTML = `
                        <h3>${suggestion.title}</h3>
                        ${suggestion.description ? `<p>${suggestion.description}</p>` : ''}
                        <div class="badge">${suggestion.frequency}</div>
                    `;
                    priority.textContent = `Priority ${suggestion.priority}`;
                    priority.className = `badge priority-${suggestion.priority}`;
                } else {
                    document.getElementById(`suggestion${index + 1}-content`).innerHTML = 'No suggestion available';
                    document.getElementById(`suggestion${index + 1}-priority`).textContent = '';
                }
             });
    });
}

function refreshActionHierarchy() {
    fetch(API_BASE+'/all-actions')
        .then(response => response.json())
        .then(data => {
            allActions = data;
            const container = document.getElementById('actions-hierarchy');

            // Render root actions first
            const rootActions = data.root || [];
            container.innerHTML = rootActions.map(action => `
                <div class="card">
                    <div class="card-header">
                        <h3>${action.title}</h3>
                        <div>
                            <span class="badge priority-${action.priority}">Priority ${action.priority}</span>
                            <span class="badge status-${action.status}">${action.status}</span>
                        </div>
                    </div>
                    <div class="action-content">
                        ${action.description || ''}
                        ${action.fixed_time ? `<div class="badge">Due: ${new Date(action.fixed_time).toLocaleString()}</div>` : ''}
                    </div>
                    <div class="sub-actions" id="sub-${action.id}">
                        ${renderSubActions(action.id, data)}
                    </div>
                    <div class="button-group">
                        <button class="btn btn-primary" onclick="startAction('${action.id}')">Start</button>
                        <button class="btn btn-secondary" onclick="openNewTaskModal('${action.id}')">Add Sub-action</button>
                        <button class="btn btn-remove" onclick="removeAction('${action.id}')">Remove</button>
                    </div>
                </div>
            `).join('');
        });
}

function renderSubActions(parentId, data) {
    const subActions = data[parentId] || [];
    if (subActions.length === 0) return '';

    return subActions.map(action => `
        <div class="sub-action">
            <div class="card-header">
                <h4>${action.title}</h4>
                <div>
                    <span class="badge priority-${action.priority}">Priority ${action.priority}</span>
                    <span class="badge status-${action.status}">${action.status}</span>
                </div>
            </div>
            <div class="action-content">
                ${action.description || ''}
                ${action.fixed_time ? `<div class="badge">Due: ${new Date(action.fixed_time).toLocaleString()}</div>` : ''}
            </div>
            <div class="button-group">
                <button class="btn btn-primary" onclick="startAction('${action.id}')">Start</button>
                <button class="btn btn-remove" onclick="removeAction('${action.id}')">Remove</button>
            </div>
        </div>
    `).join('');
}

function refreshHistory() {
    fetch(API_BASE+'/history')
        .then(response => response.json())
        .then(data => {
            history = data;
            const container = document.getElementById('history-list');

            container.innerHTML = data.map(entry => `
                <div class="history-item">
                    <div>
                        <h4>${entry.action_title}</h4>
                        <span class="badge status-${entry.status}">${entry.status}</span>
                        ${entry.parent_id ? '<span class="badge">Sub-action</span>' : ''}
                    </div>
                    <div>
                        ${new Date(entry.timestamp).toLocaleString()}
                    </div>
                </div>
            `).join('');
        });
}

// Action Management Functions
function removeAction(actionId) {
    fetch(API_BASE+`/remove-action?actionId=${actionId}`, {
        method: 'POST'
    })
    .then(response => {
    console.log(response.json());
    refreshActionHierarchy();
    })
}
function startAction(actionId) {
    fetch(API_BASE+`/current-action?actionId=${actionId}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        currentAction = data;
        updateCurrentActionDisplay();
        updateSuggestions();
    });
}

function completeCurrentAction() {
    if (!currentAction) return;

    fetch(API_BASE+'/complete-current', {
        method: 'POST'
    })
    .then(() => {
        currentAction = null;
        updateCurrentActionDisplay();
        updateSuggestions();
        refreshHistory();
    });
}

function startSuggestion(index) {
    if (suggestions[index - 1]) {
        startAction(suggestions[index - 1].id);
    }
}

// Modal Management
function openNewTaskModal(parentId = '') {
    const modal = document.getElementById('newTaskModal');
    const parentSelect = document.getElementById('taskParent');

    // Update parent action options
    if (allActions.root) {
        parentSelect.innerHTML = '<option value="">None (Top-level action)</option>' +
            allActions.root.map(action =>
                `<option value="${action.id}" ${action.id === parentId ? 'selected' : ''}>${action.title}</option>`
            ).join('');
    }

    modal.style.display = 'block';
}

function closeNewTaskModal() {
    document.getElementById('newTaskModal').style.display = 'none';
    document.getElementById('newTaskForm').reset();
}

function init() {
    updateSuggestions();
    refreshActionHierarchy();
    refreshHistory();

    // Check for current action
    fetch(API_BASE+'/get-current-action')
        .then(response => response.json())
        .then(data => {
            if (data) {
                currentAction = data;

                updateCurrentActionDisplay();
                updateSuggestions();
            }
        });
}
function createNewTask(event) {
    event.preventDefault();

    const formData = {
        id: generateId(),
        title: document.getElementById('taskTitle').value,
        description: document.getElementById('taskDescription').value,
        parent_id: document.getElementById('taskParent').value || null,
        frequency: document.getElementById('taskFrequency').value,
        priority: parseInt(document.getElementById('taskPriority').value),
        fixed_time: document.getElementById('taskFixedTime').value || null
    };

    fetch(API_BASE+'/new-action', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(() => {
        closeNewTaskModal();
        updateSuggestions();
        refreshActionHierarchy();
    });
}

const init_timer = setTimeout(() => init(), 1500);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
   init();
   clearTimeout(init_timer);
});
}

</script>
</body>
</html></div>
"""

Managers = {}


class Frequency(str, Enum):
    ONE_TIME = "one_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUALLY = "annually"


class ActionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class Action(BaseModel):
    id: str
    title: str
    description: str | None = None
    parent_id: str | None = None
    frequency: Frequency
    priority: int = Field(ge=1, le=5)
    fixed_time: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    status: ActionStatus = Field(default=ActionStatus.NOT_STARTED)
    last_completed: datetime | None = None
    next_due: datetime | None = None

    def dict(self, *args, **kwargs):
        # Convert datetime objects to strings for serialization
        data = super().model_dump(*args, **kwargs)
        for field in ["fixed_time", "created_at", "last_completed", "next_due"]:
            if isinstance(data.get(field), datetime):
                data[field] = data[field].isoformat()
        return data

    @classmethod
    def from_dict(cls, **data):
        # Convert string representations of datetime back to datetime objects
        for field in ["fixed_time", "created_at", "last_completed", "next_due"]:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class Actions(BaseModel):
    actions: list[Action]


class HistoryEntry(BaseModel):
    action_id: str
    action_title: str
    timestamp: datetime
    status: ActionStatus
    parent_id: str | None = None

    def dict(self, *args, **kwargs):
        # Convert datetime objects to strings for serialization
        data = super().model_dump(*args, **kwargs)
        for field in ["timestamp"]:
            if isinstance(data.get(field), datetime):
                data[field] = data[field].isoformat()
        return data

    @classmethod
    def from_dict(cls, **data):
        # Convert string representations of datetime back to datetime objects
        for field in ["timestamp"]:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class ActionManager:
    def __init__(self, user_id, storage: BlobStorage | None = None):
        self.storage = storage
        self.user_id = user_id
        self.current_action: Action | None = None
        self._load_data()

    def _load_data(self):
        try:
            with BlobFile("DoNext/tasks", "r", self.storage) as f:
                data = f.read_json()
                self.actions = [Action.from_dict(**action) for action in data.get("tasks", [])]
                if data.get("current") is not None:
                    self.current_action = Action.from_dict(**data.get("current"))
                self.history = [HistoryEntry.from_dict(**entry) for entry in data.get("history", [])]
        except FileNotFoundError:
            self.actions = []
            self.history = []

    def _save_data(self):
        with BlobFile("DoNext/tasks", "w", self.storage) as f:
            f.write_json({
                "tasks": [action.dict() for action in self.actions],
                "history": [entry.dict() for entry in self.history],
                "current": self.current_action.dict() if self.current_action else None,
            })

    def new_action(self, action_data: dict[str, Any]) -> Action:
        action = Action.from_dict(**action_data)

        # Calculate next_due based on frequency
        if action.frequency != Frequency.ONE_TIME:
            action.next_due = action.fixed_time or datetime.now()

        self.actions.append(action)
        self._save_data()
        return action

    def add_actions(self, actions):
        self.actions.extend(actions)
        self._save_data()

    def get_current_action(self) -> dict | None:
        if self.current_action is None:
            return None
        return self.current_action.dict()

    def remove_action(self, action_id: str):
        action = next((a for a in self.actions if a.id == action_id), None)
        if not action:
            raise ValueError("Action not found")
        self.actions.remove(action)
        self._save_data()

    def set_current_action(self, action_id: str) -> dict:
        action = next((a for a in self.actions if a.id == action_id), None)
        if not action:
            raise ValueError("Action not found")

        self.current_action = action
        action.status = ActionStatus.IN_PROGRESS

        self.history.append(
            HistoryEntry(
                action_id=action.id,
                action_title=action.title,
                timestamp=datetime.now(),
                status=ActionStatus.IN_PROGRESS,
                parent_id=action.parent_id
            )
        )

        self._save_data()
        return action.dict()

    def complete_current_action(self) -> None:
        if not self.current_action:
            raise ValueError("No current action")

        self.current_action.status = ActionStatus.COMPLETED
        self.current_action.last_completed = datetime.now()

        # Update next_due based on frequency
        if self.current_action.frequency != Frequency.ONE_TIME:
            if self.current_action.frequency == Frequency.DAILY:
                self.current_action.next_due = datetime.now() + timedelta(days=1)
            elif self.current_action.frequency == Frequency.WEEKLY:
                self.current_action.next_due = datetime.now() + timedelta(weeks=1)
            elif self.current_action.frequency == Frequency.MONTHLY:
                self.current_action.next_due = datetime.now() + timedelta(days=30)
            elif self.current_action.frequency == Frequency.ANNUALLY:
                self.current_action.next_due = datetime.now() + timedelta(days=365)

        self.history.append(
            HistoryEntry(
                action_id=self.current_action.id,
                action_title=self.current_action.title,
                timestamp=datetime.now(),
                status=ActionStatus.COMPLETED,
                parent_id=self.current_action.parent_id
            )
        )

        self.current_action = None
        self._save_data()

    def get_suggestions(self) -> list[dict]:
        if self.current_action:
            # If there's a current action, only suggest its sub-actions
            return [
                       action.dict() for action in self.actions
                       if action.parent_id == self.current_action.id and action.status == ActionStatus.NOT_STARTED
                   ][:2]

        # Otherwise, suggest actions based on priority and due date
        available_actions = [
            action for action in self.actions
            if action.status == ActionStatus.NOT_STARTED and not action.parent_id  # Only top-level actions
        ]

        # Sort by priority and due date
        sorted_actions = sorted(
            available_actions,
            key=lambda x: (
                -x.priority,  # Higher priority first
                x.next_due or datetime.max  # Earlier due date first
            )
        )

        return list(map(lambda x: x.dict(), sorted_actions[:2]))

    def get_history(self) -> list[dict]:
        return list(map(lambda x: x.dict(), sorted(self.history, key=lambda x: x.timestamp, reverse=True)))

    def get_all_actions(self) -> dict[str, list[dict]]:
        # Group actions by parent_id
        result = {"root": []}

        for action in self.actions:
            if action.parent_id:
                if action.parent_id not in result:
                    result[action.parent_id] = []
                result[action.parent_id].append(action.dict())
            else:
                result["root"].append(action.dict())

        return result


def get_storage(app, user):
    if user is None or app is None:
        return None
    if user.name == "":
        storage = BlobStorage(app.data_dir + '/public', 0)
    else:
        storage = BlobStorage(app.data_dir + '/storages/' + user.uid)
    return storage


async def get_manager(app, request):
    if request is None:
        return None
    if app is None:
        app = get_app()
    user = await get_user_from_request(app, request)
    key = Code.one_way_hash(user.name, Name)
    if key in Managers:
        return Managers[key]
    else:
        Managers[key] = ActionManager(key, get_storage(app, user))
        return Managers[key]


@export(mod_name=Name, name="init", initial=True)
def init(app=None):
    if app is None:
        app = get_app()
    app.run_any(("CloudM","add_ui"),
                name=Name,
                title=Name,
                path=f"/api/{Name}/main_web_DoNext_entry",
                description="In Pre Demo"
                )


@export(mod_name=Name, name="new-action", api=True, row=True, request_as_kwarg=True, api_methods=['POST'])
async def api_new_action(app, request: RequestSession):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)

    json =  request.json()
    print("Request:", json)

    user_manager.new_action(json)

    return {'status': 'success'}


@export(mod_name=Name, name="current-action", api=True, row=True, request_as_kwarg=True)
async def set_current_action(app, request, actionId='0'):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)

    try:
        action = user_manager.set_current_action(actionId)
    except ValueError as e:
        return Result.default_user_error(str(e), 501).to_api_result()

    return action


@export(mod_name=Name, name="remove-action", api=True, row=True, request_as_kwarg=True)
async def remove_action(app, request, actionId='0'):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)

    try:
        user_manager.remove_action(actionId)
    except ValueError as e:
        return Result.default_user_error(str(e), 501).to_api_result()

    return {'status': 'success'}


@export(mod_name=Name, name="get-current-action", api=True, row=True, request_as_kwarg=True)
async def get_current_action(app, request):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)
    return user_manager.get_current_action()


@export(mod_name=Name, name="suggestions", api=True, row=True, request_as_kwarg=True)
async def get_suggestions(app, request):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)
    return user_manager.get_suggestions()


@export(mod_name=Name, name="history", api=True, row=True, request_as_kwarg=True)
async def get_history(app, request):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)
    return user_manager.get_history()


@export(mod_name=Name, name="all-actions", api=True, row=True, request_as_kwarg=True)
async def get_all_actions(app, request):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)
    return user_manager.get_all_actions()


@export(mod_name=Name, name="complete-current", api=True, row=True, request_as_kwarg=True, api_methods=['POST'])
async def complete_current_action(app, request):
    if request is None:
        return None

    if app is None:
        app = get_app()

    user_manager = await get_manager(app, request)
    try:
        user_manager.complete_current_action()
    except ValueError as e:
        return {'status': 'error', 'message': e}

    return {'status': 'success'}


@get_app().tb(mod_name=Name, version=version, level=0, api=True,
              name="main_web_DoNext_entry", row=True, state=False)
def DoNext(app=None):
    if app is None:
        app = get_app(Name)
    return Result.html(app.web_context()+template)


# Example usage
if __name__ == "__main__":
    manager = ActionManager("test")

    # Create a main action
    sport_action = manager.new_action({
        "id": "1",
        "title": "Do sports",
        "description": "Daily workout routine",
        "frequency": Frequency.DAILY,
        "priority": 5
    })

    # Create a sub-action
    pushups_action = manager.new_action({
        "id": "2",
        "title": "100 push-ups",
        "parent_id": "1",
        "frequency": Frequency.DAILY,
        "priority": 4
    })

    # Set current action
    manager.set_current_action("1")

    # Get suggestions (should only show sub-actions of current action)
    suggestions = manager.get_suggestions()

    # Complete current action
    manager.complete_current_action()

    # Get history
    history = manager.get_history()

    # Get all actions in hierarchical structure
    all_actions = manager.get_all_actions()
