# Add these to your existing module.py or a new one
# ... (keep existing imports and MOD_NAME, VERSION, export, etc.)
import json
import os  # Ensure os is imported
import uuid
from typing import Optional, List, Dict, Any

# Assuming Task and TaskChain are defined, e.g., in toolboxv2.mods.isaa.types
# from toolboxv2.mods.isaa.types import Task, TaskChain
# If they are not, here are placeholder Pydantic models:
from pydantic import BaseModel, Field as PydanticField

from toolboxv2 import get_app, App, RequestData, Result, ToolBoxError, ToolBoxResult, ToolBoxInfo, ToolBoxInterfaces
from toolboxv2.mods.isaa.types import TaskChain, Task

# Moduldefinition
MOD_NAME = "isaa.chainUi"
VERSION = "0.1.0"
export = get_app(f"{MOD_NAME}.API").tb
Name = MOD_NAME

# --- API Endpunkte für Aufgaben-Ketten (Task Chains) ---
CHAIN_DATA_PREFIX = "task_chains"
CHAIN_LIST_KEY_SUFFIX = "_list"

def get_current_username(request: Optional[RequestData] = None) -> str:
    if request:
        return request.session.user_name
    return "default_user"  # Fallback


async def get_user_data(app: App, username: str, data_key_prefix: str) -> List[Dict[str, Any]]:
    db = app.get_mod("DB")  # Standard DB-Instanz
    # db.edit_cli("RR") # Sicherstellen, dass der Read-Replica-Modus aktiv ist, falls nötig

    key = f"{data_key_prefix}_{username}"
    db_result = db.get(key)  # db.get ist jetzt async

    if not db_result.is_error() and db_result.is_data():
        try:
            data_json = db_result.get()
            if isinstance(data_json, bytes):
                data_json = data_json.decode()
            app.logger.info(f"ISAA Chain User data : {data_json}")
            if isinstance(data_json, list) and len(data_json) > 0:  # db.get kann eine Liste zurückgeben
                return json.loads(data_json[0])
            elif isinstance(data_json, str):
                return json.loads(data_json)
            return []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


async def save_user_data(app: App, username: str, data_key_prefix: str, data: List[Dict[str, Any]]):
    db = app.get_mod("DB")
    # db.edit_cli("RR") # oder einen spezifischen Schreibmodus, falls konfiguriert
    key = f"{data_key_prefix}_{username}"
    db.set(key, json.dumps(data))  # db.set ist jetzt async


async def get_task_chain_names(app: App, username: str) -> List[str]:
    # Chains are stored individually, but we maintain a list of names for discovery
    chain_list_data = await get_user_data(app, username, f"{CHAIN_DATA_PREFIX}{CHAIN_LIST_KEY_SUFFIX}")
    if chain_list_data and isinstance(chain_list_data, list):  # Should be a list of names
        return chain_list_data
    return []


async def save_task_chain_names(app: App, username: str, chain_names: List[str]):
    await save_user_data(app, username, f"{CHAIN_DATA_PREFIX}{CHAIN_LIST_KEY_SUFFIX}", chain_names)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def get_task_chain_list(app: App, request: Optional[RequestData] = None):
    username = get_current_username(request)
    chain_names = await get_task_chain_names(app, username)
    return Result.json(data=chain_names)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def get_task_chain_definition(app: App, request: Optional[RequestData] = None):
    username = get_current_username(request)
    chain_name = request.query_params.get("chain_name") if request and request.query_params else None

    if not chain_name:
        return Result.default_user_error(info="Chain name is required.", exec_code=400)

    # Chain definitions are stored under keys like "task_chains_MY_CHAIN_NAME"
    chain_data_list = await get_user_data(app, username, f"{CHAIN_DATA_PREFIX}_{chain_name}")
    if chain_data_list and isinstance(chain_data_list, dict):  # Expecting a single dict for the chain definition
        return Result.json(data=chain_data_list)
    # Compatibility for old format where it might be a list containing one dict
    elif chain_data_list and isinstance(chain_data_list, list) and len(chain_data_list) > 0 and isinstance(
        chain_data_list[0], dict):
        return Result.json(data=chain_data_list[0])
    return Result.default_user_error(info="Task chain not found.", exec_code=404)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def save_task_chain_definition(app: App, request: Optional[RequestData] = None,
                                     data: Optional[Dict[str, Any]] = None, **kwargs):
    username = get_current_username(request)
    if data and not getattr(request, 'body', None):  # Compatibility for direct data passthrough
        if not hasattr(request, 'body'):
            class DummyRequest:
                pass

            _request = DummyRequest()
            _request.body = data
            _request.session = request.session if hasattr(request, 'session') else type('DummySession', (),
                                                                                        {'user_name': username})()
            request = _request
        else:
            request.body = data

    if not request or not request.body or not isinstance(request.body, dict):
        return Result.default_user_error(info="Invalid chain data provided.", exec_code=400)

    chain_name = request.body.get("name")
    tasks_data = request.body.get("tasks")
    description = request.body.get("description", "")

    if not chain_name:
        return Result.default_user_error(info="Chain name cannot be empty.", exec_code=400)
    if not isinstance(tasks_data, list):
        return Result.default_user_error(info="Tasks must be a list.", exec_code=400)

    # Validate tasks using Pydantic model Task (optional but good practice)
    valid_tasks = []
    for i, task_dict in enumerate(tasks_data):
        try:
            # Ensure each task has a client-side ID or generate one
            if 'id' not in task_dict or not task_dict['id']:
                task_dict['id'] = str(uuid.uuid4())
            task_model = Task(**task_dict)
            valid_tasks.append(task_model.model_dump())
        except Exception as e:  # Pydantic ValidationError
            return Result.default_user_error(info=f"Invalid data for task at index {i}: {e}", exec_code=400)

    chain_definition = TaskChain(name=chain_name, description=description, tasks=valid_tasks).model_dump()

    # Save the chain definition
    await save_user_data(app, username, f"{CHAIN_DATA_PREFIX}_{chain_name}", chain_definition)

    # Update the list of chain names
    chain_names = await get_task_chain_names(app, username)
    if chain_name not in chain_names:
        chain_names.append(chain_name)
        await save_task_chain_names(app, username, chain_names)

    return Result.json(data=chain_definition, info="Task chain saved successfully.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['DELETE'])
async def delete_task_chain(app: App, request: Optional[RequestData] = None, **kwargs):
    username = get_current_username(request)
    chain_name = request.query_params.get("chain_name") if request and request.query_params else None

    if not chain_name:
        return Result.default_user_error(info="Chain name is required for deletion.", exec_code=400)

    # Delete the specific chain data
    db = app.get_mod("DB")
    key_to_delete = f"{CHAIN_DATA_PREFIX}_{chain_name}_{username}"  # Note: DB key format was prefix_username

    # Correct key to delete based on save_user_data structure
    # save_user_data creates key as: f"{data_key_prefix}_{username}"
    # So, data_key_prefix here is f"{CHAIN_DATA_PREFIX}_{chain_name}"
    key_to_delete_actual = f"{CHAIN_DATA_PREFIX}_{chain_name}_{username}"
    # Ah, the save_user_data appends username to the prefix. So, the prefix passed to it should be just CHAIN_DATA_PREFIX_{chain_name}
    # Let's fix the key for deletion:
    prefix_for_chain = f"{CHAIN_DATA_PREFIX}_{chain_name}"
    key_for_db_operation = f"{prefix_for_chain}_{username}"

    # In your DB mod, you'd have a delete method. Let's assume `db.delete(key)`
    # For now, to "delete", we can save an empty list or a specific marker,
    # or if your DB mod supports actual deletion, use that.
    # Let's assume db.set(key, None) or db.set(key, "DELETED_MARKER") effectively deletes.
    # Or, if your db.delete returns a status:
    delete_op_result = db.delete(
        key_for_db_operation)  # Assuming db.delete exists and is async or sync based on your DB mod
    if hasattr(delete_op_result, 'is_error') and delete_op_result.is_error():
        app.logger.warning(f"Failed to delete chain data for {chain_name} from DB or key not found.")
        # We can proceed to remove from list anyway if it exists there

    # Update the list of chain names
    chain_names = await get_task_chain_names(app, username)
    if chain_name in chain_names:
        chain_names.remove(chain_name)
        await save_task_chain_names(app, username, chain_names)
        return Result.ok(info=f"Task chain '{chain_name}' deleted successfully.")
    else:
        return Result.default_user_error(info=f"Task chain '{chain_name}' not found in the list.", exec_code=404)


# --- Endpoint for the Task Chain Editor UI ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, name="task_chain_editor", api_methods=['GET'])
async def get_task_chain_editor_page(app: App, request: Optional[RequestData] = None):
    if app is None:
        app = get_app()
    html_content = app.web_context() + """
    <div class="main-content frosted-glass">
    <title>Task Chain Editor</title>
    <style>
        .task-chain-editor-grid { display: grid; grid-template-columns: 300px 1fr; gap: 1.5rem; height: calc(100vh - 150px); }
        .chain-list-panel { border-right: 1px solid var(--tb-border-color, #e5e7eb); padding-right: 1rem; overflow-y: auto; }
        .task-editor-panel { overflow-y: auto; }
        .task-card {
            background-color: var(--tb-card-bg, #ffffff);
            border: 1px solid var(--tb-border-color, #e0e0e0);
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 0.75rem;
            cursor: grab;
            box-shadow: var(--tb-shadow-md, 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06));
        }
        .dark .task-card { background-color: var(--tb-card-bg-dark, #374151); border-color: var(--tb-border-color-dark, #4b5563); }
        .task-card:active { cursor: grabbing; background-color: var(--tb-primary-100, #ebf8ff); }
        .dark .task-card:active { background-color: var(--tb-primary-700, #2c5282); }
        .drag-over-placeholder { border: 2px dashed var(--tb-primary-500, #4299e1); background-color: var(--tb-primary-50, #ebf8ff); min-height: 50px; margin-bottom: 0.75rem; border-radius: 0.5rem; }
        .task-actions button { margin-left: 0.5rem; }
    </style>
    <div id="app-root" class="tb-container tb-mx-auto tb-p-4">
        <header class="tb-flex tb-justify-between tb-items-center tb-mb-6">
            <h1 class="tb-text-3xl tb-font-bold">Task Chain Editor</h1>
            <div>
                <span id="currentUserChainEditor" class="tb-mr-4"></span>
                 <div id="darkModeToggleContainerChainEditor" style="display: inline-block;"></div>
            </div>
        </header>

        <div class="task-chain-editor-grid">
            <!-- Linke Spalte: Kettenauswahl und -verwaltung -->
            <div class="chain-list-panel">
                <h2 class="tb-text-xl tb-font-semibold tb-mb-3">Chains</h2>
                <div class="tb-mb-3">
                    <select id="chainSelector" class="tb-input tb-w-full"></select>
                </div>
                <button id="newChainBtn" class="tb-btn tb-btn-success tb-w-full tb-mb-2">
                    <span class="material-symbols-outlined tb-mr-1">add</span> Neue Kette
                </button>
                <button id="deleteChainBtn" class="tb-btn tb-btn-danger tb-w-full tb-mb-2" disabled>
                    <span class="material-symbols-outlined tb-mr-1">delete</span> Kette Löschen
                </button>
                 <button id="saveChainBtn" class="tb-btn tb-btn-primary tb-w-full" disabled>
                    <span class="material-symbols-outlined tb-mr-1">save</span> Kette Speichern
                </button>
            </div>

            <!-- Rechte Spalte: Aufgaben-Editor für ausgewählte Kette -->
            <div class="task-editor-panel">
                <div class="tb-mb-4">
                    <label for="chainNameInput" class="tb-label">Name der Kette:</label>
                    <input type="text" id="chainNameInput" class="tb-input tb-w-full" placeholder="Name der Kette" disabled>
                </div>
                <div class="tb-mb-4">
                    <label for="chainDescriptionInput" class="tb-label">Beschreibung:</label>
                    <textarea id="chainDescriptionInput" class="tb-input tb-w-full" rows="2" placeholder="Optionale Beschreibung" disabled></textarea>
                </div>

                <div class="tb-flex tb-justify-between tb-items-center tb-mb-3">
                    <h2 class="tb-text-xl tb-font-semibold">Aufgaben</h2>
                    <button id="addTaskToChainBtn" class="tb-btn tb-btn-primary" disabled>
                         <span class="material-symbols-outlined tb-mr-1">playlist_add</span> Aufgabe Hinzufügen
                    </button>
                </div>
                <div id="taskListContainer" class="tb-min-h-[200px] tb-border tb-p-2 tb-rounded tb-bg-gray-50 dark:tb-bg-gray-800">
                    <!-- Aufgaben werden hier per Drag & Drop eingefügt und sortiert -->
                    <p id="noTasksMessage" class="tb-text-gray-500 dark:tb-text-gray-400">Wähle oder erstelle eine Kette.</p>
                </div>
            </div>
        </div>
    </div>
    <script defer src="https://unpkg.com/htmx.org@2.0.2/dist/htmx.min.js"></script>
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.153.0/three.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/marked-highlight/lib/index.umd.min.js"></script>
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script defer type="module">
        let currentChainName = null;
        let currentTasks = [];
        let currentChainDescription = "";
        let allChainNames = [];
        let draggedTaskElement = null;
        let placeholder = null;

        const TASK_TYPES = ["agent", "tool", "chain"]; // For 'use' field

        if (window.TB?.events) {
            if (window.TB.config?.get('appRootId')) {
                 initializeChainEditor();
            } else {
                window.TB.events.on('tbjs:initialized', initializeChainEditor, { once: true });
            }
        } else {
            document.addEventListener('tbjs:initialized', initializeChainEditor, { once: true });
        }

        function initializeChainEditor() {
            const username = TB.user.getUsername() || 'default_user';
            document.getElementById('currentUserChainEditor').textContent = `Benutzer: ${username}`;

            // Init DarkModeToggle specifically for this container if it's separate
            if (TB.ui && TB.ui.DarkModeToggle && document.getElementById('darkModeToggleContainerChainEditor')) {
                new TB.ui.DarkModeToggle({target: document.getElementById('darkModeToggleContainerChainEditor')});
            }


            document.getElementById('chainSelector').addEventListener('change', handleChainSelectionChange);
            document.getElementById('newChainBtn').addEventListener('click', handleNewChain);
            document.getElementById('saveChainBtn').addEventListener('click', handleSaveChain);
            document.getElementById('deleteChainBtn').addEventListener('click', handleDeleteChain);
            document.getElementById('addTaskToChainBtn').addEventListener('click', () => showTaskModal(null));

            document.getElementById('chainNameInput').addEventListener('input', (e) => {
                if (currentChainName !== null) { // Only allow editing name if a chain is "active"
                     // This change is temporary until saved.
                }
            });
            document.getElementById('chainDescriptionInput').addEventListener('input', (e) => {
                currentChainDescription = e.target.value;
            });


            loadChainList();
            updateEditorState(); // Initial UI state
        }

        async function loadChainList() {
            TB.ui.Loader.show('Lade Ketten...');
            try {
                const response = await TB.api.request('isaa.chainUi', 'get_task_chain_list', null, 'GET');
                if (response.error === TB.ToolBoxError.none && response.get()) {
                    allChainNames = response.get();
                    populateChainSelector();
                } else {
                    TB.ui.Toast.showError('Fehler beim Laden der Kettenliste: ' + response.info.help_text);
                    allChainNames = [];
                }
            } catch (e) {
                TB.ui.Toast.showError('Netzwerkfehler beim Laden der Kettenliste.');
                console.error(e);
                allChainNames = [];
            } finally {
                TB.ui.Loader.hide();
            }
        }

        function populateChainSelector() {
            const selector = document.getElementById('chainSelector');
            selector.innerHTML = '<option value="">-- Kette auswählen --</option>';
            allChainNames.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                selector.appendChild(option);
            });
        }

        async function handleChainSelectionChange() {
            const selector = document.getElementById('chainSelector');
            const selectedName = selector.value;
            if (selectedName) {
                await loadChainDefinition(selectedName);
            } else {
                currentChainName = null;
                currentTasks = [];
                currentChainDescription = "";
                updateEditorState();
                renderCurrentTasks();
            }
        }

        async function loadChainDefinition(chainName) {
            TB.ui.Loader.show('Lade Kettendetails...');
            try {
                const response = await TB.api.request('isaa.chainUi', `get_task_chain_definition?chain_name=${encodeURIComponent(chainName)}`, null, 'GET');
                if (response.error === TB.ToolBoxError.none && response.get()) {
                    const chainDef = response.get();
                    currentChainName = chainDef.name;
                    currentTasks = chainDef.tasks.map(task => ({ ...task, id: task.id || TB.utils.uuidv4() })); // Ensure client-side ID
                    currentChainDescription = chainDef.description || "";
                    renderCurrentTasks();
                } else {
                    TB.ui.Toast.showError('Kette nicht gefunden oder Fehler: ' + response.info.help_text);
                    currentChainName = null; // Or perhaps set to chainName to allow creating it if not found
                    currentTasks = [];
                    currentChainDescription = "";
                    renderCurrentTasks();
                }
            } catch (e) {
                TB.ui.Toast.showError('Netzwerkfehler beim Laden der Kettendetails.');
                console.error(e);
                currentChainName = null; currentTasks = []; currentChainDescription = ""; renderCurrentTasks();
            } finally {
                updateEditorState();
                TB.ui.Loader.hide();
            }
        }

        function updateEditorState() {
            const chainSelected = currentChainName !== null;
            document.getElementById('chainNameInput').value = currentChainName || '';
            document.getElementById('chainNameInput').disabled = !chainSelected; // Allow editing name if a chain is "active" for saving
            document.getElementById('chainDescriptionInput').value = currentChainDescription || '';
            document.getElementById('chainDescriptionInput').disabled = !chainSelected;
            document.getElementById('saveChainBtn').disabled = !chainSelected;
            document.getElementById('deleteChainBtn').disabled = !chainSelected;
            document.getElementById('addTaskToChainBtn').disabled = !chainSelected;
            document.getElementById('noTasksMessage').style.display = (chainSelected && currentTasks.length === 0) ? 'block' : 'none';
             if (!chainSelected) {
                 document.getElementById('taskListContainer').innerHTML = '<p id="noTasksMessage" class="tb-text-gray-500 dark:tb-text-gray-400">Wähle oder erstelle eine Kette.</p>';
             }
        }


        function renderCurrentTasks() {
            const container = document.getElementById('taskListContainer');
            container.innerHTML = ''; // Clear previous tasks
            if (currentChainName === null) {
                 container.innerHTML = '<p id="noTasksMessage" class="tb-text-gray-500 dark:tb-text-gray-400">Wähle oder erstelle eine Kette.</p>';
                 return;
            }
             if (currentTasks.length === 0) {
                container.innerHTML = '<p id="noTasksMessage" class="tb-text-gray-500 dark:tb-text-gray-400">Noch keine Aufgaben in dieser Kette. Füge eine hinzu!</p>';
                return;
            }


            currentTasks.forEach((task, index) => {
                const taskCard = document.createElement('div');
                taskCard.className = 'task-card';
                taskCard.setAttribute('draggable', 'true');
                taskCard.dataset.taskId = task.id; // Use client-side ID for dragging
                taskCard.dataset.index = index; // Keep original index for actions

                taskCard.innerHTML = `
                    <div class="tb-font-semibold tb-text-lg">${TB.utils.escapeHtml(task.name)} (${TB.utils.escapeHtml(task.use)})</div>
                    <p class="tb-text-sm tb-text-gray-600 dark:tb-text-gray-300 tb-mb-1">Args: <code class="tb-bg-gray-200 dark:tb-bg-gray-600 tb-px-1 tb-rounded">${TB.utils.escapeHtml(task.args)}</code></p>
                    <p class="tb-text-sm tb-text-gray-600 dark:tb-text-gray-300">Return Key: <code class="tb-bg-gray-200 dark:tb-bg-gray-600 tb-px-1 tb-rounded">${TB.utils.escapeHtml(task.return_key)}</code></p>
                    <div class="task-actions tb-text-right tb-mt-2">
                        <button class="tb-btn tb-btn-sm tb-btn-icon tb-text-blue-500" data-action="edit" title="Bearbeiten"><span class="material-symbols-outlined">edit</span></button>
                        <button class="tb-btn tb-btn-sm tb-btn-icon tb-text-red-500" data-action="delete" title="Löschen"><span class="material-symbols-outlined">delete</span></button>
                    </div>
                `;
                taskCard.addEventListener('dragstart', dragStart);
                taskCard.addEventListener('dragend', dragEnd);
                container.appendChild(taskCard);

                taskCard.querySelector('button[data-action="edit"]').addEventListener('click', () => showTaskModal(task, index));
                taskCard.querySelector('button[data-action="delete"]').addEventListener('click', () => handleRemoveTask(index));
            });
            // Add dragover and drop listeners to the container
            container.addEventListener('dragover', dragOver);
            container.addEventListener('drop', dropTask);
        }

        function handleNewChain() {
            TB.ui.Modal.show({
                title: 'Neue Kette erstellen',
                content: '<input type="text" id="newChainNameModal" class="tb-input tb-w-full" placeholder="Name der neuen Kette">',
                buttons: [
                    {text: 'Abbrechen', action: modal => modal.close()},
                    {
                        text: 'Erstellen',
                        variant: 'primary',
                        action: async modal => {
                            const newName = document.getElementById('newChainNameModal').value.trim();
                            if (!newName) {
                                TB.ui.Toast.showWarning('Kettenname darf nicht leer sein.');
                                return;
                            }
                            if (allChainNames.includes(newName)) {
                                TB.ui.Toast.showWarning('Eine Kette mit diesem Namen existiert bereits.');
                                return;
                            }
                            currentChainName = newName;
                            currentTasks = [];
                            currentChainDescription = ""; // Reset description for new chain

                            // Add to selector and select it
                            const selector = document.getElementById('chainSelector');
                            const option = document.createElement('option');
                            option.value = newName;
                            option.textContent = newName;
                            selector.appendChild(option);
                            selector.value = newName;
                            allChainNames.push(newName); // Add to client-side list

                            updateEditorState();
                            renderCurrentTasks();
                            modal.close();
                            TB.ui.Toast.showSuccess(`Neue Kette '${newName}' initialisiert. Speichern nicht vergessen!`);
                        }
                    }
                ],
                onOpen: () => document.getElementById('newChainNameModal')?.focus()
            });
        }

        async function handleSaveChain() {
            const chainNameFromInput = document.getElementById('chainNameInput').value.trim();
            if (!chainNameFromInput) {
                TB.ui.Toast.showWarning('Kettenname darf nicht leer sein.');
                return;
            }

            // If the name was changed from the input field and it's different from currentChainName,
            // it implies a rename or saving a new chain if currentChainName was from a newly created (unsaved) chain.
            let effectiveChainName = chainNameFromInput;
            let isRename = currentChainName && currentChainName !== chainNameFromInput && allChainNames.includes(currentChainName);

            if (isRename) {
                // Potentially handle rename logic: delete old, save new.
                // For simplicity now, we'll just save under the new name.
                // User might need to delete the old one manually if this is a "Save As" like behavior.
                // Or, prompt for confirmation of rename.
                 const confirmRename = await TB.ui.Modal.confirm({title: 'Kette umbenennen?', content: `Möchtest du die Kette von '${currentChainName}' zu '${effectiveChainName}' umbenennen? Die alte Kette wird dann gelöscht.`});
                 if (!confirmRename) return;
            }


            const payload = {
                name: effectiveChainName,
                description: currentChainDescription,
                tasks: currentTasks.map(({id, ...task}) => task) // Remove client-side ID before saving
            };

            TB.ui.Loader.show('Speichere Kette...');
            try {
                const response = await TB.api.request('isaa.chainUi', 'save_task_chain_definition', payload, 'POST');
                if (response.error === TB.ToolBoxError.none) {
                    TB.ui.Toast.showSuccess(`Kette '${effectiveChainName}' erfolgreich gespeichert.`);
                     if (isRename) {
                        // Delete the old chain definition from backend
                        await TB.api.request('isaa.chainUi', `delete_task_chain?chain_name=${encodeURIComponent(currentChainName)}`, null, 'DELETE');
                        // Update client-side list
                        allChainNames = allChainNames.filter(name => name !== currentChainName);
                    }
                    currentChainName = effectiveChainName; // Update current name to the saved one
                    if (!allChainNames.includes(effectiveChainName)) {
                        allChainNames.push(effectiveChainName);
                    }
                    populateChainSelector(); // Repopulate to reflect changes
                    document.getElementById('chainSelector').value = effectiveChainName; // Reselect
                    updateEditorState();
                } else {
                    TB.ui.Toast.showError('Fehler beim Speichern der Kette: ' + response.info.help_text);
                }
            } catch (e) {
                TB.ui.Toast.showError('Netzwerkfehler beim Speichern der Kette.');
                console.error(e);
            } finally {
                TB.ui.Loader.hide();
            }
        }

        async function handleDeleteChain() {
            if (!currentChainName) return;
            const confirmed = await TB.ui.Modal.confirm({
                title: 'Kette löschen?',
                content: `Möchtest du die Kette '${currentChainName}' wirklich löschen? Dies kann nicht rückgängig gemacht werden.`
            });
            if (!confirmed) return;

            TB.ui.Loader.show('Lösche Kette...');
            try {
                const response = await TB.api.request('isaa.chainUi', `delete_task_chain?chain_name=${encodeURIComponent(currentChainName)}`, null, 'DELETE');
                 if (response.error === TB.ToolBoxError.none) {
                    TB.ui.Toast.showSuccess(`Kette '${currentChainName}' gelöscht.`);
                    currentChainName = null;
                    currentTasks = [];
                    currentChainDescription = "";
                    await loadChainList(); // Reload list from server
                    updateEditorState();
                    renderCurrentTasks();
                } else {
                    TB.ui.Toast.showError('Fehler beim Löschen der Kette: ' + response.info.help_text);
                }
            } catch (e) {
                TB.ui.Toast.showError('Netzwerkfehler beim Löschen der Kette.');
                console.error(e);
            } finally {
                TB.ui.Loader.hide();
            }
        }


        function showTaskModal(task = null, index = -1) {
            const isEditing = task !== null;
            const modalTitle = isEditing ? 'Aufgabe bearbeiten' : 'Neue Aufgabe hinzufügen';
            const useOptions = TASK_TYPES.map(type => `<option value="${type}" ${task?.use === type ? 'selected' : ''}>${type}</option>`).join('');

            TB.ui.Modal.show({
                title: modalTitle,
                content: `
                    <form id="taskFormModal" class="tb-space-y-4">
                        <div>
                            <label for="taskUse" class="tb-label">Use (Typ):</label>
                            <select id="taskUseModal" class="tb-input tb-w-full">${useOptions}</select>
                        </div>
                        <div>
                            <label for="taskName" class="tb-label">Name (Agent/Tool/Chain):</label>
                            <input type="text" id="taskNameModal" class="tb-input tb-w-full" value="${TB.utils.escapeHtml(task?.name || '')}" required>
                        </div>
                        <div>
                            <label for="taskArgs" class="tb-label">Argumente:</label>
                            <input type="text" id="taskArgsModal" class="tb-input tb-w-full" value="${TB.utils.escapeHtml(task?.args || '$user-input')}">
                            <p class="tb-text-xs tb-text-gray-500 dark:tb-text-gray-400">Verwende $user-input für Benutzereingabe, $variablenName für vorherige Ergebnisse.</p>
                        </div>
                        <div>
                            <label for="taskReturnKey" class="tb-label">Return Key (Variable für Ergebnis):</label>
                            <input type="text" id="taskReturnKeyModal" class="tb-input tb-w-full" value="${TB.utils.escapeHtml(task?.return_key || 'result')}">
                        </div>
                    </form>
                `,
                buttons: [
                    { text: 'Abbrechen', action: modal => modal.close(), variant: 'secondary' },
                    {
                        text: isEditing ? 'Speichern' : 'Hinzufügen',
                        variant: 'primary',
                        action: modal => {
                            const newTask = {
                                id: task?.id || TB.utils.uuidv4(), // Retain ID if editing, else new
                                use: document.getElementById('taskUseModal').value,
                                name: document.getElementById('taskNameModal').value.trim(),
                                args: document.getElementById('taskArgsModal').value.trim(),
                                return_key: document.getElementById('taskReturnKeyModal').value.trim()
                            };
                            if (!newTask.name) {
                                TB.ui.Toast.showWarning('Aufgabenname darf nicht leer sein.');
                                return;
                            }
                            if (isEditing) {
                                currentTasks[index] = newTask;
                            } else {
                                currentTasks.push(newTask);
                            }
                            renderCurrentTasks();
                            modal.close();
                        }
                    }
                ],
                onOpen: () => document.getElementById('taskNameModal')?.focus()
            });
        }

        function handleRemoveTask(index) {
            currentTasks.splice(index, 1);
            renderCurrentTasks();
        }

        // --- Drag and Drop Logic ---
        function dragStart(e) {
            draggedTaskElement = e.target;
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', e.target.dataset.taskId); // Use client-side ID
            // Optional: add a class for styling the dragged item
            setTimeout(() => e.target.classList.add('tb-opacity-50'), 0);

            // Create placeholder
            placeholder = document.createElement('div');
            placeholder.className = 'drag-over-placeholder';
            placeholder.style.height = `${draggedTaskElement.offsetHeight}px`;
        }

        function dragEnd(e) {
            draggedTaskElement.classList.remove('tb-opacity-50');
            draggedTaskElement = null;
            if (placeholder && placeholder.parentNode) {
                placeholder.parentNode.removeChild(placeholder);
            }
            placeholder = null;
            // Remove any hover styles from all cards
             document.querySelectorAll('.task-card').forEach(card => card.style.borderTop = '');
        }

        function dragOver(e) {
            e.preventDefault(); // Necessary to allow dropping
            e.dataTransfer.dropEffect = 'move';
            const targetCard = e.target.closest('.task-card');
            const taskList = document.getElementById('taskListContainer');

            if (targetCard && targetCard !== draggedTaskElement) {
                const rect = targetCard.getBoundingClientRect();
                const isAfter = e.clientY > rect.top + rect.height / 2;

                // Remove existing placeholder before inserting new one
                if (placeholder && placeholder.parentNode) {
                    placeholder.parentNode.removeChild(placeholder);
                }

                if (isAfter) {
                    taskList.insertBefore(placeholder, targetCard.nextSibling);
                } else {
                    taskList.insertBefore(placeholder, targetCard);
                }
            } else if (!targetCard && taskList.children.length > 0 && placeholder && !placeholder.parentNode) {
                 // If dragging over empty space in container, append placeholder at the end
                 taskList.appendChild(placeholder);
            } else if (taskList.children.length === 0 && placeholder && !placeholder.parentNode) {
                taskList.appendChild(placeholder);
            }
        }

        function dropTask(e) {
            e.preventDefault();
            if (!draggedTaskElement || !placeholder || !placeholder.parentNode) return;

            const draggedTaskId = draggedTaskElement.dataset.taskId;
            const originalIndex = currentTasks.findIndex(t => t.id === draggedTaskId);
            if (originalIndex === -1) return;

            const taskToMove = currentTasks[originalIndex];

            // Find new index based on placeholder's position
            const children = Array.from(placeholder.parentNode.children);
            let newIndex = children.indexOf(placeholder);

            // Remove the task from its original position
            currentTasks.splice(originalIndex, 1);

            // Adjust newIndex if the original item was before the drop target
            if (originalIndex < newIndex && placeholder.previousSibling === draggedTaskElement) {
                // This case might not be perfectly accurate with placeholder logic, test thoroughly
            } else if (draggedTaskElement === placeholder.previousSibling) {
                // If dragging down, the placeholder index is effectively one less for the splice
                newIndex = Math.max(0, newIndex -1);
            }


            // Insert the task at the new position
            currentTasks.splice(newIndex, 0, taskToMove);

            renderCurrentTasks(); // Re-render the entire list to reflect new order
        }

    </script>
    </div>
    """
    return Result.html(data=html_content)


# Update initialize_module to register the new UI
@export(mod_name=MOD_NAME, version=VERSION)
def initialize_module(app: App):
    print(f"ISAA Chains Modul ({MOD_NAME} v{VERSION}) initialisiert.")
    if app is None:
        app = get_app()


    # Register new Task Chain Editor UI
    app.run_any(("CloudM", "add_ui"),
                name=f"{Name}_TaskChainEditor",  # Unique name for this UI
                title="Task Chain Editor",
                path=f"/api/{Name}/task_chain_editor",  # Unique path
                description="Visueller Editor für Aufgaben-Ketten (Task Chains)"
                )
    return Result.ok(info="Modul und Task Chain Editor UI bereit.")
