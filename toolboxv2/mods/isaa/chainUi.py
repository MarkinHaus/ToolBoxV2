# toolboxv2/mods/isaa/chainUi.py
import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field as PydanticField

from toolboxv2 import get_app, App, RequestData, Result
from toolboxv2.mods.isaa.types import TaskChain as ISAAPydanticTaskChain, Task as ISAAPydanticTask

# Module Definition
MOD_NAME = "isaa.chainUi"
VERSION = "0.2.0"  # Version bump for Drawflow integration
export = get_app(f"{MOD_NAME}.API").tb
Name = MOD_NAME


# --- Helper to get ISAA instance ---
def get_isaa_instance(app: App):
    isaa_mod = app.get_mod("isaa")
    if not isaa_mod:
        raise ValueError("ISAA module not found or loaded.")
    return isaa_mod


# --- Pydantic Models (alias for clarity if needed, but directly using ISAA's types is fine) ---
# class Task(ISAAPydanticTask): pass
# class TaskChain(ISAAPydanticTaskChain): pass


# --- API Endpoints for Task Chains using Drawflow & Global ISAA Chains ---

@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def get_task_chain_list(app: App, request: Optional[RequestData] = None):
    """Lists all available global task chains."""
    isaa = get_isaa_instance(app)
    try:
        chain_names = list(isaa.agent_chain.chains.keys() ) # This should return List[str]
        return Result.json(data=chain_names)
    except Exception as e:
        app.logger.error(f"Error listing task chains: {e}", exc_info=True)
        return Result.custom_error(info=f"Failed to list task chains: {str(e)}", exec_code=500)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def get_task_chain_definition(app: App, request: Optional[RequestData] = None):
    """Gets the definition of a specific task chain, including its Drawflow export if available."""
    chain_name = request.query_params.get("chain_name") if request and request.query_params else None
    if not chain_name:
        return Result.default_user_error(info="Chain name is required.", exec_code=400)

    isaa = get_isaa_instance(app)
    try:
        # Get logical task definition
        tasks_list_dicts = isaa.get_task(chain_name)
        if tasks_list_dicts is None:  # Check if chain exists
            return Result.default_user_error(info=f"Task chain '{chain_name}' not found.", exec_code=404)

        description = isaa.agent_chain.get_discr(chain_name) or ""

        # Attempt to load Drawflow specific data if it exists
        # This assumes Drawflow data is saved in a parallel file or embedded
        drawflow_data = None
        drawflow_file_path = isaa.agent_chain.directory / f"{chain_name}.drawflow.json"
        if drawflow_file_path.exists():
            try:
                with open(drawflow_file_path, 'r') as f:
                    drawflow_data = json.load(f)
            except Exception as e:
                app.logger.warning(f"Could not load Drawflow data for chain '{chain_name}': {e}")

        chain_pydantic_tasks = [ISAAPydanticTask(**task_dict) for task_dict in tasks_list_dicts]

        response_data = ISAAPydanticTaskChain(
            name=chain_name,
            description=description,
            tasks=chain_pydantic_tasks
        ).model_dump()

        if drawflow_data:
            response_data["drawflow_export"] = drawflow_data  # Embed Drawflow data

        return Result.json(data=response_data)

    except Exception as e:
        app.logger.error(f"Error getting task chain definition for '{chain_name}': {e}", exc_info=True)
        return Result.custom_error(info=f"Failed to get task chain definition: {str(e)}", exec_code=500)


class SaveTaskChainRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    tasks: List[ISAAPydanticTask]  # Logical tasks
    drawflow_export: Optional[Dict[str, Any]] = None  # Full Drawflow export data


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def save_task_chain_definition(app: App, request: Optional[RequestData] = None,
                                     data: SaveTaskChainRequest = None):
    """Saves a task chain definition. Expects logical tasks and optional Drawflow export."""
    if not data:  # Compatibility for direct data passthrough if decorator doesn't parse body for Pydantic model
        if request and request.body and isinstance(request.body, dict):
            try:
                data = SaveTaskChainRequest(**request.body)
            except Exception as e:
                return Result.default_user_error(info=f"Invalid chain data provided: {e}", exec_code=400)
        else:
            return Result.default_user_error(info="No chain data provided.", exec_code=400)
    elif isinstance(data, dict):
        data = SaveTaskChainRequest(**data)

    if not data.name:
        return Result.default_user_error(info="Chain name cannot be empty.", exec_code=400)

    isaa = get_isaa_instance(app)
    try:
        # Save logical tasks to ISAA's AgentChain
        task_dicts = [task.model_dump() for task in data.tasks]
        isaa.add_task(data.name, task_dicts)  # add_task replaces if exists
        if data.description is not None:  # Allow empty description
            isaa.agent_chain.add_discr(data.name, data.description)
        isaa.save_task(data.name)  # Persists the .chain.json file

        # Save Drawflow specific data if provided
        if data.drawflow_export:
            drawflow_file_path = Path(isaa.agent_chain.directory) / f"{data.name}.drawflow.json"
            try:
                with open(drawflow_file_path, 'w') as f:
                    json.dump(data.drawflow_export, f, indent=2)
                app.logger.info(f"Saved Drawflow data for chain '{data.name}' to {drawflow_file_path}")
            except Exception as e:
                app.logger.error(f"Failed to save Drawflow data for chain '{data.name}': {e}")
                # Optionally, inform client that logical save succeeded but visual save failed
                return Result.ok(
                    info=f"Task chain '{data.name}' saved (logical part), but Drawflow visual data failed to save.")

        return Result.ok(info=f"Task chain '{data.name}' saved successfully.")

    except Exception as e:
        app.logger.error(f"Error saving task chain '{data.name}': {e}", exc_info=True)
        return Result.custom_error(info=f"Failed to save task chain: {str(e)}", exec_code=500)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['DELETE'])
async def delete_task_chain(app: App, request: Optional[RequestData] = None):
    """Deletes a task chain."""
    chain_name = request.query_params.get("chain_name") if request and request.query_params else None
    if not chain_name:
        return Result.default_user_error(info="Chain name is required for deletion.", exec_code=400)

    isaa = get_isaa_instance(app)
    try:
        isaa.remove_task(chain_name)  # Removes from memory
        isaa.save_task()  # Saves all chains, effectively removing the deleted one from file
        # This also deletes the .chain.json file

        # Delete associated Drawflow file if it exists
        drawflow_file_path = isaa.agent_chain.directory / f"{chain_name}.drawflow.json"
        if drawflow_file_path.exists():
            try:
                drawflow_file_path.unlink()
                app.logger.info(f"Deleted Drawflow data for chain '{chain_name}'.")
            except Exception as e:
                app.logger.warning(f"Could not delete Drawflow data file for chain '{chain_name}': {e}")

        return Result.ok(info=f"Task chain '{chain_name}' deleted successfully.")
    except KeyError:  # If isaa.remove_task raises KeyError for non-existent chain
        return Result.default_user_error(info=f"Task chain '{chain_name}' not found.", exec_code=404)
    except Exception as e:
        app.logger.error(f"Error deleting task chain '{chain_name}': {e}", exc_info=True)
        return Result.custom_error(info=f"Failed to delete task chain: {str(e)}", exec_code=500)


class RunChainRequest(BaseModel):
    chain_name: str
    task_input: str
    # Potentially, allow passing a full chain definition for unsaved execution
    chain_definition: Optional[ISAAPydanticTaskChain] = None


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def run_chain_visualized(app: App, request: Optional[RequestData] = None, data: RunChainRequest = None):
    """Executes a specified task chain with the given input."""
    if not data:  # Compatibility
        if request and request.body and isinstance(request.body, dict):
            try:
                data = RunChainRequest(**request.body)
            except Exception as e:
                return Result.default_user_error(info=f"Invalid run chain data: {e}", exec_code=400)
        else:
            return Result.default_user_error(info="No run chain data provided.", exec_code=400)
    elif isinstance(data, dict):
        data = RunChainRequest(**data)

    if not data.chain_name:
        return Result.default_user_error(info="Chain name is required for execution.", exec_code=400)
    if data.task_input is None:  # Allow empty string as input
        return Result.default_user_error(info="Task input is required.", exec_code=400)

    isaa = get_isaa_instance(app)

    # TODO: Add SSE streaming for execution progress if desired in the future.
    # For now, simple blocking execution.

    try:
        # If chain_definition is provided, use it directly (for running unsaved chains)
        if data.chain_definition:
            app.logger.info(
                f"Executing unsaved chain definition for '{data.chain_name}' with input: {data.task_input[:50]}...")
            # Temporarily add this chain definition to isaa.agent_chain without saving to file
            # This requires isaa.agent_chain to support in-memory, non-persistent additions or direct execution
            # For simplicity, let's assume isaa.run_task can accept a task list directly if AgentChain is adapted.
            # If not, we'd save it temporarily or find another way.
            # For now, assuming run_task primarily uses named, saved chains.
            # A more robust solution would be to modify `isaa.run_task` or `ChainTreeExecutor`
            # to accept a raw list of task dictionaries.
            # This example will proceed assuming the chain must be saved first if not already.
            # Let's add a note that this feature (running unsaved chains from UI) needs more work on ISAA core.
            app.logger.warning(
                "Running unsaved chain definitions directly is not fully supported by this endpoint version. The chain should be saved first.")
            # Fallback to trying to run by name, assuming it was saved.

        app.logger.info(f"Executing chain '{data.chain_name}' with input: {data.task_input[:50]}...")
        # `isaa.run_task` is already async
        execution_result = await isaa.run_task(task_input=data.task_input, chain_name=data.chain_name)

        # `execution_result` structure depends on `ChainTreeExecutor.execute`
        # It's usually a dictionary of results.
        return Result.json(data={"output": execution_result, "final_message": "Chain execution completed."})

    except Exception as e:
        app.logger.error(f"Error executing task chain '{data.chain_name}': {e}", exc_info=True)
        return Result.custom_error(info=f"Chain execution failed: {str(e)}", exec_code=500)


# --- Endpoint for the Drawflow Task Chain Editor UI ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, name="task_chain_editor_drawflow", api_methods=['GET'])
async def get_task_chain_editor_page_drawflow(app: App, request: Optional[RequestData] = None):
    """Serves the HTML page for the Drawflow-based Task Chain Editor."""
    if app is None:  # Should not happen if called via export
        app = get_app()

    # The Drawflow HTML and JS will be substantial.
    # It's better to load it from a separate .html file for maintainability.
    # For this example, I'll provide a condensed version here.
    # In a real setup, use:
    #   ui_file_path = Path(__file__).parent / "task_chain_editor_drawflow.html"
    #   with open(ui_file_path, "r") as f:
    #       html_content = f.read()
    # And then inject app.web_context() if needed, or ensure tb.js handles it.

    html_content = DRAWFLOW_TASK_CHAIN_EDITOR_HTML_TEMPLATE  # Defined below
    return Result.html(data=app.web_context() + html_content)


# --- Initialization ---
@export(mod_name=MOD_NAME, version=VERSION)
def initialize_module(app: App):
    """Initializes the ISAA ChainUI module and registers its UI with CloudM."""
    print(f"ISAA Drawflow ChainUI Modul ({MOD_NAME} v{VERSION}) initialisiert.")
    if app is None:
        app = get_app()

    # Register the new Drawflow-based Task Chain Editor UI
    app.run_any(("CloudM", "add_ui"),
                name=f"{Name}_TaskChainEditorDrawflow",  # Unique name
                title="Task Chain Editor (Drawflow)",
                path=f"/api/{Name}/task_chain_editor_drawflow",  # Unique path
                description="Visual editor for ISAA Task Chains using Drawflow.",
                auth=True
                )
    return Result.ok(info="ISAA Drawflow ChainUI Modul und Editor UI bereit.")


# --- HTML Template for Drawflow Task Chain Editor ---
# This is a placeholder and would be a more complex HTML file.
DRAWFLOW_TASK_CHAIN_EDITOR_HTML_TEMPLATE = """
<title>Task Chain Editor (Drawflow)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jerosoler/Drawflow/dist/drawflow.min.css">
<script src="https://cdn.jsdelivr.net/gh/jerosoler/Drawflow/dist/drawflow.min.js"></script>
<style>
    /* Basic styling for the editor layout */
    .drawflow-editor-container { display: flex; height: calc(100vh - 180px); /* Adjust based on header/footer */ }
    .drawflow-sidebar { width: 250px; padding: 10px; border-right: 1px solid #ccc; overflow-y: auto; background: #f9f9f9; }
    .drawflow-sidebar h3 { margin-top: 0; font-size: 1.1em; }
    .drawflow-sidebar button { display: block; width: 100%; margin-bottom: 8px; }
    #drawflowCanvas { flex-grow: 1; position: relative; } /* Ensure drawflow div takes space */
    .drawflow-controls { padding: 10px; border-top: 1px solid #ccc; background: #f9f9f9; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .drawflow-controls .tb-input, .drawflow-controls .tb-btn { margin-bottom: 0; }
    /* Styles for node content */
    .drawflow-node .task-node-content { padding: 8px; font-size: 0.9em; }
    .drawflow-node .task-node-content strong { display: block; margin-bottom: 4px; }
    .drawflow-node .task-node-content p { margin: 2px 0; white-space: pre-wrap; word-break: break-all; }

    /* Dark mode considerations for Drawflow itself (may need more specific targeting) */
    .dark .drawflow-sidebar { background-color: var(--tb-bg-secondary-dark, #2d3748); border-right-color: var(--tb-border-color-dark, #4a5562); }
    .dark .drawflow-controls { background-color: var(--tb-bg-secondary-dark, #2d3748); border-top-color: var(--tb-border-color-dark, #4a5562); }
    /* Drawflow nodes are styled by its library, custom HTML within nodes should adapt */
    .dark .drawflow-node .task-node-content { /* Basic dark mode for custom content */ }
    .drawflow-node .inputs .input, .drawflow-node .outputs .output { background-color: var(--tb-primary-500) !important; } /* Example: Color connection points */
</style>

<div id="app-root" class="tb-container tb-mx-auto tb-p-4">
    <header class="tb-flex tb-justify-between tb-items-center tb-mb-4">
        <h1 class="tb-text-3xl tb-font-bold">Task Chain Editor (Drawflow)</h1>
        <div>
            <div id="darkModeToggleContainerDrawflow" style="display: inline-block;"></div>
        </div>
    </header>

    <div class="drawflow-controls">
        <select id="chainSelectorDrawflow" class="tb-input tb-input-sm" style="min-width: 150px;"></select>
        <input type="text" id="chainNameInputDrawflow" class="tb-input tb-input-sm" placeholder="Chain Name" style="flex-grow: 1; min-width: 150px;">
        <button id="newChainBtnDrawflow" class="tb-btn tb-btn-neutral tb-btn-sm"><span class="material-symbols-outlined tb-mr-1">add</span>New</button>
        <button id="saveChainBtnDrawflow" class="tb-btn tb-btn-primary tb-btn-sm"><span class="material-symbols-outlined tb-mr-1">save</span>Save</button>
        <button id="deleteChainBtnDrawflow" class="tb-btn tb-btn-danger tb-btn-sm"><span class="material-symbols-outlined tb-mr-1">delete</span>Delete</button>
        <hr class="tb-w-full tb-my-1 md:tb-w-auto md:tb-h-6 md:tb-border-l md:tb-mx-2">
        <input type="text" id="chainTaskInputDrawflow" class="tb-input tb-input-sm" placeholder="Input for chain execution" style="flex-grow: 2; min-width: 200px;">
        <button id="executeChainBtnDrawflow" class="tb-btn tb-btn-success tb-btn-sm"><span class="material-symbols-outlined tb-mr-1">play_arrow</span>Execute</button>
    </div>

    <div class="drawflow-editor-container tb-mt-2">
        <div class="drawflow-sidebar">
            <h3>Task Palette</h3>
            <button class="tb-btn tb-btn-secondary tb-btn-sm" data-task-type="agent">Agent Task</button>
            <button class="tb-btn tb-btn-secondary tb-btn-sm" data-task-type="tool">Tool Task</button>
            <button class="tb-btn tb-btn-secondary tb-btn-sm" data-task-type="chain">Sub-Chain Task</button>
            <hr class="tb-my-2">
            <div id="chainDescriptionContainer">
                 <label for="chainDescriptionInputDrawflow" class="tb-label tb-text-sm">Description:</label>
                 <textarea id="chainDescriptionInputDrawflow" class="tb-input tb-input-sm tb-w-full" rows="3" placeholder="Chain description..."></textarea>
            </div>
             <hr class="tb-my-2">
            <h3>Execution Log</h3>
            <div id="executionLogDrawflow" class="tb-text-xs tb-p-1 tb-bg-gray-100 dark:tb-bg-gray-800 tb-rounded tb-h-32 tb-overflow-y-auto">
                Ready.
            </div>
        </div>
        <div id="drawflowCanvas"></div> <!-- Drawflow will attach here -->
    </div>
</div>
# Last
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jerosoler/Drawflow/dist/drawflow.min.css">
<script src="https://cdn.jsdelivr.net/gh/jerosoler/Drawflow/dist/drawflow.min.js"></script>
# or version view releases https://github.com/jerosoler/Drawflow/releases
<link rel="stylesheet" href="https://unpkg.com/drawflow@0.0.60/dist/drawflow.min.css" />
<script src="https://unpkg.com/drawflow@0.0.60/dist/drawflow.min.js"></script>
<script type="module">
    // Ensure TB is loaded
    if (!window.TB) { console.error("TB.js not loaded!"); }

    let editor;
    let currentChain = { name: "", description: "", tasks: [], drawflow_export: null };
    let availableAgents = []; // To be fetched from ISAA UI or a dedicated endpoint
    let availableTools = [];  // Same as above
    // --- Initialization ---

    // Wait for tbjs to be initialized
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         initializeDrawflowEditor();
    } else {
        window.TB.events.on('tbjs:initialized', initializeDrawflowEditor, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', initializeDrawflowEditor, { once: true }); // Custom event dispatch from TB.init
}


    function initializeDrawflowEditor() {
        const drawflowContainer = document.getElementById('drawflowCanvas');
        if (!drawflowContainer) {
            console.error("Drawflow container not found!");
            return;
        }
        editor = new Drawflow(drawflowContainer, window.Vue, window.Vue // If using Vue, otherwise null or an empty object
        // { render: null, parent: null } // For vanilla JS
        );
        editor.reroute = true; // Enable rerouting connections
        editor.start();
        TB.logger.info("Drawflow editor initialized.");

        // Init DarkModeToggle
        if (TB.ui && TB.ui.DarkModeToggle && document.getElementById('darkModeToggleContainerDrawflow')) {
            new TB.ui.DarkModeToggle({ target: document.getElementById('darkModeToggleContainerDrawflow') });
        }

        // Event Listeners for UI controls
        document.getElementById('chainSelectorDrawflow').addEventListener('change', loadSelectedChain);
        document.getElementById('newChainBtnDrawflow').addEventListener('click', startNewChain);
        document.getElementById('saveChainBtnDrawflow').addEventListener('click', saveCurrentChain);
        document.getElementById('deleteChainBtnDrawflow').addEventListener('click', deleteCurrentChain);
        document.getElementById('executeChainBtnDrawflow').addEventListener('click', executeCurrentChain);

        document.querySelectorAll('.drawflow-sidebar button[data-task-type]').forEach(btn => {
            btn.addEventListener('click', () => addNodeToCanvas(btn.dataset.taskType));
        });

        // Listener for node selection to show properties (simplified)
        editor.on('nodeSelected', (nodeId) => {
            const nodeData = editor.getNodeFromId(nodeId);
            // For now, editing is via modal on double click (see node registration)
            TB.logger.info("Node selected:", nodeId, nodeData);
        });

        editor.on('connectionCreated', (connection) => {
            TB.logger.info("Connection created:", connection);
            // Auto-update internal chain structure if needed, or on save
        });

        loadChainList();
        // Potentially load available agents/tools for palette
        // loadAvailableAgentsAndTools();
    }

    // --- Chain Management API Calls ---
    async function loadChainList() {
        TB.ui.Loader.show("Loading chains...");
        try {
            const response = await TB.api.request('isaa.chainUi', 'get_task_chain_list', null, 'GET');
            if (response.error === TB.ToolBoxError.none && response.get()) {
                const chainNames = response.get();
                const selector = document.getElementById('chainSelectorDrawflow');
                selector.innerHTML = '<option value="">-- Select Chain --</option>';
                chainNames.forEach(name => {
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name;
                    selector.appendChild(option);
                });
            } else {
                TB.ui.Toast.showError("Error loading chain list: " + response.info?.help_text);
            }
        } catch(e) {
            TB.ui.Toast.showError("Network error loading chain list.");
            console.error(e);
        } finally {
            TB.ui.Loader.hide();
        }
    }

    async function loadSelectedChain() {
        const chainName = document.getElementById('chainSelectorDrawflow').value;
        if (!chainName) {
            startNewChain(); // Or clear editor
            return;
        }
        TB.ui.Loader.show(`Loading chain: ${chainName}...`);
        try {
            const response = await TB.api.request('isaa.chainUi', `get_task_chain_definition?chain_name=${encodeURIComponent(chainName)}`, null, 'GET');
            if (response.error === TB.ToolBoxError.none && response.get()) {
                const chainData = response.get();
                currentChain.name = chainData.name;
                currentChain.description = chainData.description || "";
                currentChain.tasks = chainData.tasks || []; // Logical tasks
                currentChain.drawflow_export = chainData.drawflow_export || null; // Drawflow visual data

                document.getElementById('chainNameInputDrawflow').value = currentChain.name;
                document.getElementById('chainDescriptionInputDrawflow').value = currentChain.description;

                editor.clearModuleSelected(); // Clear current Drawflow canvas
                if (currentChain.drawflow_export) {
                    editor.import(currentChain.drawflow_export);
                } else {
                    // Reconstruct Drawflow from logical tasks if no visual data
                    reconstructDrawflowFromTasks(currentChain.tasks);
                }
                TB.ui.Toast.showSuccess(`Chain '${chainName}' loaded.`);
            } else {
                TB.ui.Toast.showError("Error loading chain definition: " + response.info?.help_text);
            }
        } catch(e) {
            TB.ui.Toast.showError("Network error loading chain definition.");
            console.error(e);
        } finally {
            TB.ui.Loader.hide();
        }
    }

    function startNewChain() {
        const newName = "UntitledChain_" + TB.utils.uniqueId('').substring(0,4);
        currentChain = { name: newName, description: "", tasks: [], drawflow_export: null };
        document.getElementById('chainNameInputDrawflow').value = currentChain.name;
        document.getElementById('chainDescriptionInputDrawflow').value = "";
        document.getElementById('chainSelectorDrawflow').value = ""; // Deselect
        editor.clearModuleSelected();
        TB.ui.Toast.showInfo("New chain started. Remember to save.");
    }

    async function saveCurrentChain() {
        const chainName = document.getElementById('chainNameInputDrawflow').value.trim();
        const chainDescription = document.getElementById('chainDescriptionInputDrawflow').value.trim();

        if (!chainName) {
            TB.ui.Toast.showWarning("Chain name cannot be empty.");
            return;
        }

        // Update currentChain with latest from Drawflow export
        const drawflowExportData = editor.export();
        currentChain.name = chainName;
        currentChain.description = chainDescription;
        currentChain.drawflow_export = drawflowExportData;

        // Extract logical tasks from Drawflow data
        currentChain.tasks = extractLogicalTasksFromDrawflow(drawflowExportData);

        const payload = {
            name: currentChain.name,
            description: currentChain.description,
            tasks: currentChain.tasks,
            drawflow_export: currentChain.drawflow_export
        };

        TB.ui.Loader.show("Saving chain...");
        try {
            const response = await TB.api.request('isaa.chainUi', 'save_task_chain_definition', payload, 'POST');
            if (response.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(`Chain '${currentChain.name}' saved.`);
                await loadChainList(); // Refresh list
                document.getElementById('chainSelectorDrawflow').value = currentChain.name; // Reselect
            } else {
                TB.ui.Toast.showError("Error saving chain: " + response.info?.help_text);
            }
        } catch(e) {
            TB.ui.Toast.showError("Network error saving chain.");
            console.error(e);
        } finally {
            TB.ui.Loader.hide();
        }
    }

    async function deleteCurrentChain() {
        const chainName = document.getElementById('chainSelectorDrawflow').value;
        if (!chainName) {
            TB.ui.Toast.showWarning("No chain selected to delete.");
            return;
        }
        const confirmed = await TB.ui.Modal.confirm({
            title: "Delete Chain?",
            content: `Are you sure you want to delete the chain '${chainName}'?`
        });
        if (!confirmed) return;

        TB.ui.Loader.show("Deleting chain...");
        try {
            const response = await TB.api.request('isaa.chainUi', `delete_task_chain?chain_name=${encodeURIComponent(chainName)}`, null, 'DELETE');
            if (response.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess(`Chain '${chainName}' deleted.`);
                startNewChain(); // Reset to a new chain state
                await loadChainList(); // Refresh list
            } else {
                TB.ui.Toast.showError("Error deleting chain: " + response.info?.help_text);
            }
        } catch(e) {
            TB.ui.Toast.showError("Network error deleting chain.");
            console.error(e);
        } finally {
            TB.ui.Loader.hide();
        }
    }

    async function executeCurrentChain() {
        const chainName = document.getElementById('chainNameInputDrawflow').value.trim(); // Use current name in editor
        const taskInput = document.getElementById('chainTaskInputDrawflow').value;
        const logDiv = document.getElementById('executionLogDrawflow');

        if (!chainName) {
            TB.ui.Toast.showWarning("Cannot execute: Chain name is empty. Save the chain first or load an existing one.");
            return;
        }

        // Optional: Check if the chain is saved or has changes, prompt to save first.
        // For now, we assume execution uses the *saved version* of the chain by its name.
        // To execute the *current editor state* without saving, the backend would need to accept
        // the full chain definition (currentChain.tasks) along with the input.

        logDiv.innerHTML = `Executing chain '${chainName}' with input: "${TB.utils.escapeHtml(taskInput.substring(0,50))}..."\\n`;
        TB.ui.Loader.show(`Executing '${chainName}'...`);

        try {
            const payload = { chain_name: chainName, task_input: taskInput };
            // If you want to execute the current unsaved state:
            // payload.chain_definition = { name: chainName, description: currentChain.description, tasks: currentChain.tasks };

            const response = await TB.api.request('isaa.chainUi', 'run_chain_visualized', payload, 'POST');
            if (response.error === TB.ToolBoxError.none && response.get()) {
                const result = response.get();
                logDiv.innerHTML += "Execution Result:\\n" + JSON.stringify(result.output, null, 2) + "\\n";
                TB.ui.Toast.showSuccess(result.final_message || "Chain executed.");
            } else {
                logDiv.innerHTML += "Execution Error: " + response.info?.help_text + "\\n";
                TB.ui.Toast.showError("Error executing chain: " + response.info?.help_text);
            }
        } catch(e) {
            logDiv.innerHTML += "Network Error: " + e.message + "\\n";
            TB.ui.Toast.showError("Network error during execution.");
            console.error(e);
        } finally {
            TB.ui.Loader.hide();
            logDiv.scrollTop = logDiv.scrollHeight;
        }
    }

    // --- Drawflow Node and Data Handling ---
    function addNodeToCanvas(taskType) {
        // Default task data
        const defaultTaskData = {
            id: TB.utils.uniqueId('task_df_'), // Client-side Drawflow specific ID if needed, distinct from logical task ID
            logical_task_id: TB.utils.uniqueId('task_'), // The ISAAPydanticTask ID
            use: taskType,
            name: `New ${taskType.charAt(0).toUpperCase() + taskType.slice(1)} Task`,
            args: taskType === 'agent' ? '$user-input' : (taskType === 'tool' ? 'param=value' : 'sub_chain_name'),
            return_key: 'result'
        };

        // Customize HTML content for the node
        const nodeHTML = `
            <div class="task-node-content" ondblclick="showTaskEditModal(event, ${editor.id})">
                <strong>${defaultTaskData.name}</strong>
                <p>Use: ${defaultTaskData.use}</p>
                <p>Args: ${TB.utils.escapeHtml(defaultTaskData.args)}</p>
                <p>Returns: ${defaultTaskData.return_key}</p>
            </div>
        `;

        // Add node to Drawflow: addNode(name, inputs, outputs, posx, posy, class, data, html, typenode = false)
        // Let's make all tasks have 1 input and 1 output for sequential chaining.
        const newNodeId = editor.addNode(taskType, 1, 1, 150, 150, `task-node ${taskType}-node`, defaultTaskData, nodeHTML);
        TB.logger.info(`Added ${taskType} node with ID: ${newNodeId}`);
    }

    window.showTaskEditModal = (event, nodeId_numeric) => { // Make it global for ondblclick
        const nodeId = 'node-' + nodeId_numeric; // Drawflow IDs are usually node-X
        const node = editor.getNodeFromId(nodeId_numeric); // getNodeFromId expects numeric part
        if (!node || !node.data) {
            TB.ui.Toast.showError("Could not find node data to edit.");
            return;
        }

        const taskData = node.data; // This is our defaultTaskData structure

        const TASK_TYPES_OPTIONS = ["agent", "tool", "chain"].map(type =>
            `<option value="${type}" ${taskData.use === type ? 'selected' : ''}>${type}</option>`).join('');

        TB.ui.Modal.show({
            title: 'Edit Task Properties',
            content: `
                <form id="drawflowTaskFormModal" class="tb-space-y-3">
                    <div><label class="tb-label">Type (Use):</label><select id="dfTaskUse" class="tb-input tb-w-full">${TASK_TYPES_OPTIONS}</select></div>
                    <div><label class="tb-label">Name (Agent/Tool/Chain):</label><input type="text" id="dfTaskName" class="tb-input tb-w-full" value="${TB.utils.escapeHtml(taskData.name)}"></div>
                    <div><label class="tb-label">Arguments:</label><input type="text" id="dfTaskArgs" class="tb-input tb-w-full" value="${TB.utils.escapeHtml(taskData.args)}"></div>
                    <div><label class="tb-label">Return Key:</label><input type="text" id="dfTaskReturnKey" class="tb-input tb-w-full" value="${TB.utils.escapeHtml(taskData.return_key)}"></div>
                    <input type="hidden" id="dfTaskLogicalId" value="${taskData.logical_task_id}">
                </form>
            `,
            buttons: [
                { text: 'Cancel', action: modal => modal.close(), variant: 'secondary' },
                { text: 'Save Changes', variant: 'primary', action: modal => {
                    const updatedData = {
                        ...taskData, // Preserve existing fields like id
                        use: document.getElementById('dfTaskUse').value,
                        name: document.getElementById('dfTaskName').value.trim(),
                        args: document.getElementById('dfTaskArgs').value.trim(),
                        return_key: document.getElementById('dfTaskReturnKey').value.trim(),
                        logical_task_id: document.getElementById('dfTaskLogicalId').value // Ensure logical ID is preserved
                    };
                    // Update node's data and HTML content in Drawflow
                    editor.updateNodeDataFromId(nodeId_numeric, updatedData);

                    const newHTML = `
                        <div class="task-node-content" ondblclick="showTaskEditModal(event, ${nodeId_numeric})">
                            <strong>${updatedData.name}</strong>
                            <p>Use: ${updatedData.use}</p>
                            <p>Args: ${TB.utils.escapeHtml(updatedData.args)}</p>
                            <p>Returns: ${updatedData.return_key}</p>
                        </div>
                    `;
                    editor.updateNodeHTML(nodeId_numeric, newHTML); // Assumes such a method exists or find alternative
                                                              // Drawflow might re-render on data change if HTML is template-based,
                                                              // or need specific update like `editor.drawflow.drawflow.Home.data[nodeId_numeric].html = newHTML; editor.updateNodeValue(editor.drawflow.drawflow.Home.data[nodeId_numeric]);` (complex)
                                                              // Simpler: updateNodeDataFromId should be enough if HTML generation is dynamic in Drawflow core
                                                              // For now, we will assume updateNodeDataFromId re-renders if needed, or we handle HTML update on task extraction.

                    modal.close();
                    TB.ui.Toast.showSuccess("Task updated.");
                }}
            ]
        });
    };

    function reconstructDrawflowFromTasks(tasks) {
        editor.clearModuleSelected();
        let lastNodeId = null;
        let yPos = 50;

        tasks.forEach((task, index) => {
            const taskData = {
                id: TB.utils.uniqueId('task_df_'), // New Drawflow ID
                logical_task_id: task.id || TB.utils.uniqueId('task_'), // Use existing logical ID or generate
                use: task.use,
                name: task.name,
                args: task.args,
                return_key: task.return_key
            };
            const nodeHTML = `
                <div class="task-node-content" ondblclick="showTaskEditModal(event, ${editor.id})">
                     <strong>${taskData.name}</strong>
                     <p>Use: ${taskData.use}</p>
                     <p>Args: ${TB.utils.escapeHtml(taskData.args)}</p>
                     <p>Returns: ${taskData.return_key}</p>
                </div>
            `;
            const newNodeId = editor.addNode(task.use, 1, 1, 150, yPos, `task-node ${task.use}-node`, taskData, nodeHTML);
            yPos += 120; // Increment y position for next node

            if (lastNodeId !== null) {
                // Connect output_1 of lastNode to input_1 of newNode
                editor.addConnection(lastNodeId, newNodeId, 'output_1', 'input_1');
            }
            lastNodeId = newNodeId;
        });
    }

    function extractLogicalTasksFromDrawflow(drawflowExport) {
        // This is a critical and potentially complex function.
        // It needs to traverse the Drawflow graph (nodes and connections)
        // and reconstruct the ordered list of logical ISAAPydanticTask objects.

        if (!drawflowExport || !drawflowExport.drawflow || !drawflowExport.drawflow.Home || !drawflowExport.drawflow.Home.data) {
            TB.logger.warning("Invalid or empty Drawflow export data.");
            return [];
        }
        const dfNodes = drawflowExport.drawflow.Home.data;
        const nodesMap = new Map(Object.entries(dfNodes)); // Map of node_id_numeric -> node_object
        const tasks = [];

        // Find starting nodes (nodes with no inputs connected from other tasks in this export)
        const startNodeIds = [];
        nodesMap.forEach((node, nodeId) => {
            let isStartNode = true;
            if (node.inputs) {
                for (const inputClass in node.inputs) {
                    if (node.inputs[inputClass].connections && node.inputs[inputClass].connections.length > 0) {
                        // Check if any connection comes from a node within this export
                        node.inputs[inputClass].connections.forEach(conn => {
                            if (nodesMap.has(conn.node)) { // conn.node is the source node's numeric ID
                                isStartNode = false;
                            }
                        });
                    }
                }
            }
            // A simpler heuristic for sequential chains: if a node has inputs but no *incoming* connections from *other drawn nodes*.
            // For now, let's assume a simple linear chain and try to sort by Y position, then follow connections.
            // This needs a proper graph traversal for complex flows.
            if(isStartNode) startNodeIds.push(nodeId);
        });

        // Simplified: if multiple start nodes, pick one (e.g., lowest Y). For robust handling, support multiple parallel starts or error.
        // Or, assume the order in Object.values(dfNodes) might be somewhat indicative of creation order if not edited much.
        // For now, let's try a very basic traversal from the first start node found.

        let sortedTasks = [];
        let visited = new Set();

        function traverse(nodeIdNumeric) {
            if (visited.has(nodeIdNumeric)) return;
            visited.add(nodeIdNumeric);

            const node = nodesMap.get(String(nodeIdNumeric)); // Ensure key is string for Map
            if (node && node.data) {
                // Create logical task from node.data
                const logicalTask = {
                    id: node.data.logical_task_id || TB.utils.uniqueId('task_'),
                    use: node.data.use,
                    name: node.data.name,
                    args: node.data.args,
                    return_key: node.data.return_key
                };
                // Validate with Pydantic if desired: ISAAPydanticTask(**logicalTask)
                sortedTasks.push(logicalTask);

                // Follow outputs to next node
                if (node.outputs) {
                    for (const outputClass in node.outputs) {
                        if (node.outputs[outputClass].connections) {
                            node.outputs[outputClass].connections.forEach(conn => {
                                traverse(conn.node); // conn.node is the target node's numeric ID
                            });
                        }
                    }
                }
            }
        }

        // Fallback: iterate nodes as they appear in the export if no clear start or complex graph.
        // This might not preserve logical order perfectly if the graph is not simple.
        // A better approach is topological sort if it's a DAG.
        // For now, let's process in the order they appear if startNodeIds logic is too simple.
        if (startNodeIds.length > 0) {
             // Sort start nodes by Y position, then X, to get a more predictable entry point
            startNodeIds.sort((a, b) => {
                const nodeA = nodesMap.get(a);
                const nodeB = nodesMap.get(b);
                if (nodeA.pos_y === nodeB.pos_y) return nodeA.pos_x - nodeB.pos_x;
                return nodeA.pos_y - nodeB.pos_y;
            });
            traverse(startNodeIds[0]); // Traverse from the "top-most, left-most" start node
        } else {
             // If no clear start nodes (e.g., a cycle or all nodes have inputs), process in order of map
             TB.logger.warning("Could not determine clear start nodes for task extraction. Processing in exported order. Order may not be correct for complex graphs.");
             nodesMap.forEach((node, nodeId) => {
                 if (!visited.has(nodeId)) { // Process components if graph is disconnected
                     traverse(nodeId);
                 }
             });
        }

        if (sortedTasks.length !== nodesMap.size && nodesMap.size > 0) {
            TB.logger.warning(`Task extraction mismatch: extracted ${sortedTasks.length} tasks from ${nodesMap.size} Drawflow nodes. Graph might be complex or disconnected.`);
             // As a very naive fallback if traversal missed nodes, append them. THIS IS NOT ORDER-PRESERVING.
            nodesMap.forEach((node, nodeId) => {
                if (!sortedTasks.some(t => t.id === node.data.logical_task_id)) {
                    sortedTasks.push({
                        id: node.data.logical_task_id || TB.utils.uniqueId('task_'),
                        use: node.data.use, name: node.data.name,
                        args: node.data.args, return_key: node.data.return_key
                    });
                }
            });
        }

        return sortedTasks;
    }

</script>
"""
