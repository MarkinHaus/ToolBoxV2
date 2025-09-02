import contextlib

from toolboxv2 import RequestData, Result, get_app
from toolboxv2.mods.DB.types import DatabaseModes

# --- Unchanged Backend API Endpoints ---
# These functions are kept as provided, as the refactoring is focused on the UI.

Name = "DB"
export = get_app(from_="DB.EXPORT").tb


def _unwrap_data(data: any) -> any:
    """Helper to unwrap data if it's in a single-element list."""
    if isinstance(data, bytes):
        with contextlib.suppress(UnicodeDecodeError):
            data = data.decode('utf-8')
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


@export(mod_name=Name, name="api_get_status", api=True, request_as_kwarg=True)
async def api_get_status(self, request: RequestData):
    """Returns the current status of the DB manager."""
    return Result.json(data={"mode": self.mode})


@export(mod_name=Name, name="api_get_all_keys", api=True, request_as_kwarg=True)
async def api_get_all_keys(self, request: RequestData):
    """Returns a list of all keys in the database."""
    if self.data_base:
        keys_result = self.data_base.get('all-k')
        if keys_result.is_error():
            return keys_result

        unwrapped_keys = _unwrap_data(keys_result.get())
        if not isinstance(unwrapped_keys, list):
            self.app.logger.warning(f"get_all_keys did not return a list. Got: {type(unwrapped_keys)}")
            return Result.json(data=[])

        return Result.json(data=sorted(unwrapped_keys))
    return Result.default_internal_error("DB not initialized")


@export(mod_name=Name, name="api_get_value", api=True, request_as_kwarg=True)
async def api_get_value(self, request: RequestData, key: str):
    """Gets a value for a key and returns it as JSON-friendly text."""
    if not key:
        return Result.default_user_error("Key parameter is required.")
    value_res = self.get(key)
    if value_res.is_error():
        return value_res

    value_unwrapped = _unwrap_data(value_res.get())

    if isinstance(value_unwrapped, bytes):
        try:
            value_str = value_unwrapped.decode('utf-8')
        except UnicodeDecodeError:
            value_str = str(value_unwrapped)
    else:
        value_str = str(value_unwrapped)

    # Simplified for a JSON-focused UI. The client will handle formatting.
    return Result.json(data={"key": key, "value": value_str})


@export(mod_name=Name, name="api_set_value", api=True, api_methods=['POST'], request_as_kwarg=True)
async def api_set_value(self, request: RequestData):
    """Sets a key-value pair from a JSON POST body."""
    data = request.body
    if not data or 'key' not in data or 'value' not in data:
        return Result.default_user_error("Request body must contain 'key' and 'value'.")
    key = data['key']
    value = data['value']
    if not key:
        return Result.default_user_error("Key cannot be empty.")
    return self.set(key, value)


@export(mod_name=Name, name="api_delete_key", api=True, api_methods=['POST'], request_as_kwarg=True)
async def api_delete_key(self, request: RequestData):
    """Deletes a key from a JSON POST body."""
    data = request.body
    if not data or 'key' not in data:
        return Result.default_user_error("Request body must contain 'key'.")
    key = data['key']
    if not key:
        return Result.default_user_error("Key parameter is required.")
    return self.delete(key)


@export(mod_name=Name, name="api_change_mode", api=True, api_methods=['POST'], request_as_kwarg=True)
async def api_change_mode(self, request: RequestData):
    """Changes the database mode from a JSON POST body."""
    data = request.body
    if not data or "mode" not in data:
        return Result.default_user_error("Request body must contain 'mode'.")
    new_mode = data.get("mode", "LC")
    return self.edit_programmable(DatabaseModes.crate(new_mode))


# --- Refactored UI ---

@export(mod_name=Name, name="ui", api=True, state=False)
def db_manager_ui(**kwargs):
    """Serves the refactored, JSON-focused UI for the DB Manager."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DB Manager</title>
        <style>
            :root {
                --font-family-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                --font-family-mono: "SF Mono", "Menlo", "Monaco", "Courier New", Courier, monospace;
                --color-bg: #f8f9fa;
                --color-panel-bg: #ffffff;
                --color-border: #dee2e6;
                --color-text: #212529;
                --color-text-muted: #6c757d;
                --color-primary: #0d6efd;
                --color-primary-hover: #0b5ed7;
                --color-danger: #dc3545;
                --color-danger-hover: #bb2d3b;
                --color-key-folder-icon: #f7b731;
                --color-key-file-icon: #adb5bd;
                --color-key-hover-bg: #e9ecef;
                --color-key-selected-bg: #0d6efd;
                --color-key-selected-text: #ffffff;
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --radius: 0.375rem;
            }

            /* Basic styles */
            * { box-sizing: border-box; }
            html { font-size: 16px; }

            body {
                font-family: var(--font-family-sans);
                background-color: var(--color-bg);
                color: var(--color-text);
                margin: 0;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }

            /* Main layout */
            .db-manager-container { display: flex; flex-direction: column; height: 100%; gap: 1rem; }
            .db-header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 1rem; border-bottom: 1px solid var(--color-border); flex-shrink: 0; }
            .db-main-content { display: flex; gap: 1rem; flex: 1; min-height: 0; }

            /* Panels */
            .db-panel { background-color: var(--color-panel-bg); border: 1px solid var(--color-border); border-radius: var(--radius); box-shadow: var(--shadow-sm); display: flex; flex-direction: column; min-height: 0; }
            .key-panel { width: 350px; min-width: 250px; max-width: 450px; }
            .editor-panel, .placeholder-panel { flex-grow: 1; }
            .panel-header { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1rem; border-bottom: 1px solid var(--color-border); flex-shrink: 0; }
            .panel-header h2 { font-size: 1.1rem; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

            /* Controls */
            select, input[type="text"], textarea, button { font-size: 1rem; }
            select, input[type="text"] { background-color: var(--color-bg); color: var(--color-text); border: 1px solid var(--color-border); border-radius: var(--radius); padding: 0.5rem 0.75rem; }
            select:focus, input[type="text"]:focus, textarea:focus { outline: 2px solid var(--color-primary); outline-offset: -1px; }
            button { border: none; border-radius: var(--radius); padding: 0.5rem 1rem; font-weight: 500; cursor: pointer; transition: background-color 0.2s; }
            button.primary { background-color: var(--color-primary); color: white; }
            button.primary:hover { background-color: var(--color-primary-hover); }
            button.danger { background-color: var(--color-danger); color: white; }
            button.danger:hover { background-color: var(--color-danger-hover); }
            .header-actions { display: flex; gap: 0.5rem; }

            /* Key Tree View */
            #keySearchInput { width: calc(100% - 2rem); margin: 1rem; flex-shrink: 0; }
            .key-tree-container { font-family: var(--font-family-mono); font-size: 0.9rem; padding: 0 0.5rem 1rem; overflow-y: auto; flex: 1; min-height: 0; }
            .key-tree-container ul { list-style: none; padding-left: 0; margin: 0; }
            .key-tree-container li { padding-left: 20px; position: relative; }
            .node-label { display: flex; align-items: center; padding: 4px 8px; cursor: pointer; border-radius: 4px; word-break: break-all; user-select: none; }
            .node-label:hover { background-color: var(--color-key-hover-bg); }
            .node-label.selected { background-color: var(--color-key-selected-bg); color: var(--color-key-selected-text); }
            .node-label.selected .node-icon { color: var(--color-key-selected-text) !important; }
            .node-icon { width: 20px; text-align: center; margin-right: 5px; flex-shrink: 0; }
            .tree-folder > .node-label .node-icon { color: var(--color-key-folder-icon); font-style: normal; }
            .tree-folder > .node-label .node-icon::before { content: '▸'; display: inline-block; transition: transform 0.15s ease-in-out; }
            .tree-folder.open > .node-label .node-icon::before { transform: rotate(90deg); }
            .tree-leaf > .node-label .node-icon { color: var(--color-key-file-icon); }
            .tree-leaf > .node-label .node-icon::before { content: '•'; }
            .tree-children { display: none; }
            .tree-folder.open > .tree-children { display: block; }

            /* Editor Panel */
            .editor-toolbar { display: flex; gap: 1rem; align-items: center; padding: 0.75rem 1rem; border-bottom: 1px solid var(--color-border); flex-shrink: 0; }
            #valueEditor { flex: 1; width: 100%; min-height: 0; border: none; resize: none; font-family: var(--font-family-mono); font-size: 0.95rem; line-height: 1.5; padding: 1rem; background: transparent; color: var(--color-text); }
            #valueEditor:focus { outline: none; }

            /* Placeholder and Utility */
            .placeholder-panel { display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--color-text-muted); text-align: center; }
            .hidden { display: none !important; }
            .key-tree-container p.status-message { padding: 1rem; margin: 0; color: var(--color-text-muted); text-align: center; }

            /* Custom Scrollbars */
            .key-tree-container::-webkit-scrollbar, #valueEditor::-webkit-scrollbar { width: 8px; height: 8px; }
            .key-tree-container::-webkit-scrollbar-track, #valueEditor::-webkit-scrollbar-track { background: transparent; }
            .key-tree-container::-webkit-scrollbar-thumb, #valueEditor::-webkit-scrollbar-thumb { background-color: var(--color-border); border-radius: 4px; }
            .key-tree-container::-webkit-scrollbar-thumb:hover, #valueEditor::-webkit-scrollbar-thumb:hover { background-color: var(--color-text-muted); }
            #valueEditor::-webkit-scrollbar-corner { background: transparent; }

            /* Responsive */
            @media (max-width: 768px) {
                body { padding: 0.5rem; }
                .db-main-content { flex-direction: column; }
                .key-panel { width: 100%; max-height: 40vh; }
            }
        </style>
    </head>
    <body>
        <div id="dbManagerContainer" class="db-manager-container">
            <header class="db-header">
                <h1>DB Manager</h1>
                <div class="db-mode-selector">
                    <label for="modeSelect">Mode:</label>
                    <select id="modeSelect">
                        <option value="LC">Local Dict</option>
                        <option value="CB">Cloud Blob</option>
                        <option value="LR">Local Redis</option>
                        <option value="RR">Remote Redis</option>
                    </select>
                </div>
            </header>
            <main class="db-main-content">
                <aside id="keyPanel" class="db-panel key-panel">
                    <div class="panel-header">
                        <h2>Keys</h2>
                        <div class="header-actions">
                            <button id="addKeyBtn" title="Add New Key" style="font-size: 1.2rem;">+</button>
                            <button id="refreshKeysBtn" title="Refresh Keys">🔄</button>
                        </div>
                    </div>
                    <input type="text" id="keySearchInput" placeholder="Search keys...">
                    <div id="keyTreeContainer" class="key-tree-container"></div>
                </aside>
                <section id="editorPanel" class="db-panel editor-panel hidden">
                    <div class="panel-header">
                        <h2 id="selectedKey"></h2>
                        <div class="header-actions">
                            <button id="saveBtn" class="primary">Save</button>
                            <button id="deleteBtn" class="danger">Delete</button>
                        </div>
                    </div>
                    <div class="editor-toolbar">
                        <button id="formatBtn">Format JSON</button>
                    </div>
                    <textarea id="valueEditor" placeholder="Select a key to view its value..."></textarea>
                </section>
                <section id="placeholderPanel" class="db-panel editor-panel placeholder-panel">
                    <h3>Select a key to get started</h3>
                    <p>Or click the '+' button to add a new one.</p>
                </section>
            </main>
        </div>
        <script>
        (() => {
            "use strict";
            const API_NAME = "DB";

            class DBManager {
                constructor() {
                    this.cache = {
                        keys: [],
                        selectedKey: null
                    };
                    this.dom = {
                        modeSelect: document.getElementById('modeSelect'),
                        keySearchInput: document.getElementById('keySearchInput'),
                        keyTreeContainer: document.getElementById('keyTreeContainer'),
                        editorPanel: document.getElementById('editorPanel'),
                        placeholderPanel: document.getElementById('placeholderPanel'),
                        selectedKey: document.getElementById('selectedKey'),
                        valueEditor: document.getElementById('valueEditor'),
                        addKeyBtn: document.getElementById('addKeyBtn'),
                        refreshKeysBtn: document.getElementById('refreshKeysBtn'),
                        saveBtn: document.getElementById('saveBtn'),
                        deleteBtn: document.getElementById('deleteBtn'),
                        formatBtn: document.getElementById('formatBtn'),
                    };
                    this.init();
                }

                async init() {
                    this.setStatusMessage('Loading...');
                    this.addEventListeners();
                    await this.loadInitialStatus();
                    await this.loadKeys();
                }

                addEventListeners() {
                    this.dom.refreshKeysBtn.addEventListener('click', () => this.loadKeys());
                    this.dom.addKeyBtn.addEventListener('click', () => this.showAddKeyModal());
                    this.dom.saveBtn.addEventListener('click', () => this.saveValue());
                    this.dom.deleteBtn.addEventListener('click', () => this.confirmDeleteKey());
                    this.dom.formatBtn.addEventListener('click', () => this.formatJson());
                    this.dom.keySearchInput.addEventListener('input', (e) => this.renderKeyTree(e.target.value));
                    this.dom.modeSelect.addEventListener('change', (e) => this.changeMode(e.target.value));

                    this.dom.keyTreeContainer.addEventListener('click', (e) => {
                        const label = e.target.closest('.node-label');
                        if (!label) return;
                        const node = label.parentElement;
                        if (node.classList.contains('tree-folder')) {
                            node.classList.toggle('open');
                        } else if (node.dataset.key) {
                            this.selectKey(node.dataset.key);
                        }
                    });
                }

                async apiRequest(endpoint, payload = null, method = 'POST') {
                    if (!window.TB?.api?.request) {
                        console.error("TB.api not available!");
                        return { error: true, message: "TB.api not available" };
                    }
                    try {
                        const url = (method === 'GET' && payload) ? `${endpoint}?${new URLSearchParams(payload)}` : endpoint;
                        const body = (method !== 'GET') ? payload : null;
                        const response = await window.TB.api.request(API_NAME, url, body, method);

                        if (response.error && response.error !== 'none') {
                            const errorMsg = response.info?.help_text || response.error;
                            console.error(`API Error on ${endpoint}:`, errorMsg, response);
                            if (window.TB?.ui?.Toast) TB.ui.Toast.showError(errorMsg, { duration: 5000 });
                            return { error: true, message: errorMsg, data: response.get() };
                        }
                        return { error: false, data: response.get() };
                    } catch (err) {
                        console.error("Framework/Network Error:", err);
                        if (window.TB?.ui?.Toast) TB.ui.Toast.showError("Application or network error.", { duration: 5000 });
                        return { error: true, message: "Network error" };
                    }
                }

                async loadInitialStatus() {
                    const res = await this.apiRequest('api_get_status', null, 'GET');
                    if (!res.error) this.dom.modeSelect.value = res.data.mode;
                }

                async loadKeys() {
                    this.setStatusMessage('Loading keys...');
                    const res = await this.apiRequest('api_get_all_keys', null, 'GET');
                    if (!res.error) {
                        this.cache.keys = res.data || [];
                        this.renderKeyTree();
                    } else {
                        this.setStatusMessage('Failed to load keys.', true);
                    }
                }

                renderKeyTree(filter = '') {
                    const treeData = {};
                    const filteredKeys = this.cache.keys.filter(k => k.toLowerCase().includes(filter.toLowerCase().trim()));

                    for (const key of filteredKeys) {
                        let currentLevel = treeData;
                        const parts = key.split(':');
                        for (let i = 0; i < parts.length; i++) {
                            const part = parts[i];
                            if (!part) continue; // Skip empty parts from keys like "a::b"
                            const isLeaf = i === parts.length - 1;

                            if (!currentLevel[part]) {
                                currentLevel[part] = { _children: {} };
                            }
                            if (isLeaf) {
                                currentLevel[part]._fullKey = key;
                            }
                            currentLevel = currentLevel[part]._children;
                        }
                    }

                    const treeHtml = this.buildTreeHtml(treeData);
                    if (treeHtml) {
                        this.dom.keyTreeContainer.innerHTML = `<ul class="key-tree">${treeHtml}</ul>`;
                        // Re-select the key if it's still visible
                        if (this.cache.selectedKey) {
                             const nodeEl = this.dom.keyTreeContainer.querySelector(`[data-key="${this.cache.selectedKey}"] .node-label`);
                             if(nodeEl) nodeEl.classList.add('selected');
                        }
                    } else {
                         this.setStatusMessage(filter ? 'No keys match your search.' : 'No keys found.');
                    }
                }

                buildTreeHtml(node) {
                    return Object.keys(node).sort().map(key => {
                        const childNode = node[key];
                        const isFolder = Object.keys(childNode._children).length > 0;

                        if (isFolder) {
                            return `<li class="tree-folder" ${childNode._fullKey ? `data-key="${childNode._fullKey}"`: ''}>
                                        <div class="node-label"><i class="node-icon"></i>${key}</div>
                                        <ul class="tree-children">${this.buildTreeHtml(childNode._children)}</ul>
                                    </li>`;
                        } else {
                            return `<li class="tree-leaf" data-key="${childNode._fullKey}">
                                        <div class="node-label"><i class="node-icon"></i>${key}</div>
                                    </li>`;
                        }
                    }).join('');
                }

                async selectKey(key) {
                    if (!key) return;
                    this.showEditor(true);
                    this.cache.selectedKey = key;

                    document.querySelectorAll('.node-label.selected').forEach(el => el.classList.remove('selected'));
                    const nodeEl = this.dom.keyTreeContainer.querySelector(`[data-key="${key}"] > .node-label`);
                    if (nodeEl) nodeEl.classList.add('selected');

                    this.dom.selectedKey.textContent = key;
                    this.dom.selectedKey.title = key;
                    this.dom.valueEditor.value = "Loading...";

                    const res = await this.apiRequest('api_get_value', { key }, 'GET');
                    this.dom.valueEditor.value = res.error ? `Error: ${res.message}` : res.data.value;
                    if (!res.error) this.formatJson(false); // Auto-format if it's valid JSON, without showing an error
                }

                async saveValue() {
                    if (!this.cache.selectedKey) return;
                    if (window.TB?.ui?.Loader) TB.ui.Loader.show("Saving...");
                    const res = await this.apiRequest('api_set_value', {
                        key: this.cache.selectedKey,
                        value: this.dom.valueEditor.value
                    });
                    if (window.TB?.ui?.Loader) TB.ui.Loader.hide();
                    if (!res.error && window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Key saved successfully!");
                }

                async confirmDeleteKey() {
                    if (!this.cache.selectedKey) return;
                    if (!window.TB?.ui?.Modal) {
                        if(confirm(`Delete key "${this.cache.selectedKey}"?`)) this.deleteKey();
                        return;
                    }
                    TB.ui.Modal.confirm({
                        title: 'Delete Key?',
                        content: `Are you sure you want to delete the key "<strong>${this.cache.selectedKey}</strong>"?<br/>This action cannot be undone.`,
                        confirmButtonText: 'Delete',
                        confirmButtonVariant: 'danger',
                        onConfirm: () => this.deleteKey()
                    });
                }

                async deleteKey() {
                    const keyToDelete = this.cache.selectedKey;
                    if (!keyToDelete) return;
                    if (window.TB?.ui?.Loader) TB.ui.Loader.show("Deleting...");
                    const res = await this.apiRequest('api_delete_key', { key: keyToDelete });
                    if (window.TB?.ui?.Loader) TB.ui.Loader.hide();

                    if (!res.error) {
                        if (window.TB?.ui?.Toast) TB.ui.Toast.showSuccess(`Key "${keyToDelete}" deleted.`);
                        this.cache.selectedKey = null;
                        this.showEditor(false);
                        this.loadKeys(); // Refresh the key list
                    }
                }

                formatJson(showErrorToast = true) {
                    try {
                        const currentVal = this.dom.valueEditor.value.trim();
                        if (!currentVal) return;
                        const formatted = JSON.stringify(JSON.parse(currentVal), null, 2);
                        this.dom.valueEditor.value = formatted;
                    } catch (e) {
                        if (showErrorToast && window.TB?.ui?.Toast) {
                            TB.ui.Toast.showWarning("Value is not valid JSON.", { duration: 3000 });
                        }
                    }
                }

                showAddKeyModal() {
                     if (!window.TB?.ui?.Modal) { alert("Add Key modal not available."); return; }
                     TB.ui.Modal.show({
                        title: 'Add New Key',
                        content: `<input type="text" id="newKeyInput" placeholder="Enter new key name (e.g., app:settings:user)" style="width: 100%; margin-bottom: 1rem;"/>
                                  <textarea id="newValueInput" placeholder='Enter value (e.g., {"theme": "dark"})' style="width: 100%; height: 150px; font-family: var(--font-family-mono);"></textarea>`,
                        onOpen: (modal) => document.getElementById('newKeyInput').focus(),
                        buttons: [{
                            text: 'Save', variant: 'primary',
                            action: async (modal) => {
                                const newKey = document.getElementById('newKeyInput').value.trim();
                                const newValue = document.getElementById('newValueInput').value;
                                if (!newKey) { if (window.TB?.ui?.Toast) TB.ui.Toast.showError("Key name cannot be empty."); return; }
                                modal.close();
                                if (window.TB?.ui.Loader) TB.ui.Loader.show("Saving...");
                                const res = await this.apiRequest('api_set_value', { key: newKey, value: newValue });
                                if (window.TB?.ui.Loader) TB.ui.Loader.hide();
                                if (!res.error) {
                                    if (window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("New key created!");
                                    await this.loadKeys();
                                    this.selectKey(newKey);
                                }
                            }
                        }, { text: 'Cancel', action: (modal) => modal.close() }]
                    });
                }

                async changeMode(newMode) {
                    if (window.TB?.ui?.Loader) TB.ui.Loader.show(`Switching to ${newMode}...`);
                    const res = await this.apiRequest('api_change_mode', { mode: newMode });
                    if (!res.error) {
                       this.cache.selectedKey = null;
                       this.showEditor(false);
                       await this.loadKeys();
                       if (window.TB?.ui?.Toast) TB.ui.Toast.showSuccess(`Switched to ${newMode} mode.`);
                    } else {
                       if (window.TB?.ui?.Toast) TB.ui.Toast.showError(`Failed to switch mode.`);
                       await this.loadInitialStatus(); // Revert dropdown to actual status
                    }
                    if (window.TB?.ui?.Loader) TB.ui.Loader.hide();
                }

                showEditor(show) {
                    this.dom.editorPanel.classList.toggle('hidden', !show);
                    this.dom.placeholderPanel.classList.toggle('hidden', show);
                }

                setStatusMessage(message, isError = false) {
                    this.dom.keyTreeContainer.innerHTML = `<p class="status-message" style="${isError ? 'color: var(--color-danger);' : ''}">${message}</p>`;
                }
            }

            // Defer initialization until the ToolboxV2 framework is ready

             function onTbReady() { new DBManager(); }
             if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         onTbReady();
    } else {
        window.TB.events.on('tbjs:initialized', onTbReady, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', onTbReady, { once: true }); // Custom event dispatch from TB.init
}

        })();
        </script>
    </body>
    </html>
    """
    app = get_app(Name)
    try:
        # Prepend the web context to include necessary framework scripts (like TB.js)
        web_context = app.web_context()
        return Result.html(web_context + html_content)
    except Exception:
        # Fallback in case web_context is not available
        return Result.html(html_content)
