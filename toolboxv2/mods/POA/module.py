import json
import time
import uuid  # Für eindeutige IDs
from typing import Dict, Optional, List, Any

from toolboxv2 import get_app, App, RequestData, Result  # Annahme: RequestData und Result sind korrekt importiert

# Moduldefinition
MOD_NAME = "POA"  # Persönlicher Organisationsassistent
VERSION = "0.1.0"
export = get_app(f"{MOD_NAME}.API").tb  # API-Endpunkte werden hier registriert
Name = MOD_NAME

# --- Hilfsfunktionen für DB-Interaktion ---
# In einer echten Anwendung würde man den Benutzernamen dynamisch aus der Session/RequestData holen
# Für SPP1 verwenden wir einen festen Benutzernamen oder einen, der per Query-Parameter kommt.
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
            app.logger.info(f"POA User data : {data_json}")
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


# --- API Endpunkte für Aufgaben ---
@export(mod_name=MOD_NAME, version=VERSION)
async def version(app: App):
    return VERSION

# --- API Endpunkte für Aufgaben ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def get_tasks(app: App, request: Optional[RequestData] = None):
    username = get_current_username(request)
    tasks = await get_user_data(app, username, "poa_tasks")
    return Result.json(data=tasks)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def add_task(app: App, request, data: Optional[Dict[str, str]] = None, **kwargs):
    username = get_current_username(request)
    if not isinstance(data, dict):
        return Result.default_user_error(info="Invalid task data provided.", exec_code=400)

    task_text = data.get("text")
    if not task_text:
        return Result.default_user_error(info="Task text cannot be empty.", exec_code=400)

    tasks = await get_user_data(app, username, "poa_tasks")
    new_task = {
        "id": str(uuid.uuid4()),
        "text": task_text,
        "completed": False,
        "createdAt": time.time()
    }
    tasks.append(new_task)
    await save_user_data(app, username, "poa_tasks", tasks)
    return Result.json(data=new_task, info="Task added successfully.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['PUT'])
async def toggle_task(app: App,
                      request: Optional[RequestData] = None, **kwargs):  # Pfad könnte /api/POA/toggle_task?task_id=... sein
    username = get_current_username(request)
    task_id = request.query_params.get("task_id") if request and request.query_params else None

    if not task_id:
        return Result.default_user_error(info="Task ID is required.", exec_code=400)

    tasks = await get_user_data(app, username, "poa_tasks")
    task_found = False
    for task in tasks:
        if task["id"] == task_id:
            task["completed"] = not task["completed"]
            task_found = True
            break

    if not task_found:
        return Result.default_user_error(info="Task not found.", exec_code=404)

    await save_user_data(app, username, "poa_tasks", tasks)
    return Result.ok(info="Task status updated.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['DELETE'])
async def delete_task(app: App,
                      request: Optional[RequestData] = None, **kwargs):  # Pfad könnte /api/POA/delete_task?task_id=... sein
    username = get_current_username(request)
    task_id = request.query_params.get("task_id") if request and request.query_params else None

    if not task_id:
        return Result.default_user_error(info="Task ID is required.", exec_code=400)

    tasks = await get_user_data(app, username, "poa_tasks")
    tasks_updated = [task for task in tasks if task["id"] != task_id]

    if len(tasks) == len(tasks_updated):
        return Result.default_user_error(info="Task not found.", exec_code=404)

    await save_user_data(app, username, "poa_tasks", tasks_updated)
    return Result.ok(info="Task deleted successfully.")


# --- API Endpunkte für Notizen (ähnlich wie Aufgaben) ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def get_notes(app: App, request: Optional[RequestData] = None, **kwargs):
    username = get_current_username(request)
    notes = await get_user_data(app, username, "poa_notes")
    return Result.json(data=notes)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def add_note(app: App, request: Optional[RequestData] = None,data=None, **kwargs):
    username = get_current_username(request)
    if data and not request.body:
        request.body = data
    if not request or not request.body or not isinstance(request.body, dict):
        return Result.default_user_error(info="Invalid note data provided.", exec_code=400)

    title = request.body.get("title", "")
    content = request.body.get("content")

    if not content:
        return Result.default_user_error(info="Note content cannot be empty.", exec_code=400)

    notes = await get_user_data(app, username, "poa_notes")
    new_note = {
        "id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "createdAt": time.time(),
        "updatedAt": time.time()
    }
    notes.append(new_note)
    await save_user_data(app, username, "poa_notes", notes)
    return Result.json(data=new_note, info="Note added successfully.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['PUT'])
async def update_note(app: App, request: Optional[RequestData] = None,data=None, **kwargs):
    username = get_current_username(request)
    note_id = request.query_params.get("note_id") if request and request.query_params else None
    if data and not request.body:
        request.body = data
    if not note_id:
        return Result.default_user_error(info="Note ID is required.", exec_code=400)
    if not request or not request.body or not isinstance(request.body, dict):
        return Result.default_user_error(info="Invalid note data provided.", exec_code=400)

    notes = await get_user_data(app, username, "poa_notes")
    note_found = False
    for note in notes:
        if note["id"] == note_id:
            note["title"] = request.body.get("title", note["title"])
            note["content"] = request.body.get("content", note["content"])
            note["updatedAt"] = time.time()
            note_found = True
            break

    if not note_found:
        return Result.default_user_error(info="Note not found.", exec_code=404)

    await save_user_data(app, username, "poa_notes", notes)
    return Result.ok(info="Note updated successfully.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['DELETE'])
async def delete_note(app: App, request: Optional[RequestData] = None, **kwargs):
    username = get_current_username(request)
    note_id = request.query_params.get("note_id") if request and request.query_params else None

    if not note_id:
        return Result.default_user_error(info="Note ID is required.", exec_code=400)

    notes = await get_user_data(app, username, "poa_notes")
    notes_updated = [note for note in notes if note["id"] != note_id]

    if len(notes) == len(notes_updated):
        return Result.default_user_error(info="Note not found.", exec_code=404)

    await save_user_data(app, username, "poa_notes", notes_updated)
    return Result.ok(info="Note deleted successfully.")


# --- Hauptseite ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, name="main", api_methods=['GET'])  # Zugriff über /api/POA/
async def get_poa_page(app: App, request: Optional[RequestData] = None):
    # Diese Funktion liefert das Haupt-HTML für die POA-Anwendung
    # Das HTML wird im nächsten Abschnitt definiert.
    # Für dieses Beispiel geben wir einen Platzhalter zurück.
    # In einer echten Anwendung würdest du hier das HTML aus einer Datei laden.
    if app is None:
        app = get_app()
    html_content = app.web_context() + """
    <div class="main-content frosted-glass">
    <title>Persönlicher Organisationsassistent</title>
    <style>
        /* Eigene Basis-Styles, falls nötig */
        body { transition: background-color 0.3s, color 0.3s; }
        .task-item, .note-item {
            padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 0.375rem;
            /* Standard Tailwind Klassen werden später von tbjs oder direkt hinzugefügt */
        }
        .task-item.completed { text-decoration: line-through; opacity: 0.7; }
    </style>
    <div id="app-root" class="tb-container tb-mx-auto tb-p-4">
        <header class="tb-flex tb-justify-between tb-items-center tb-mb-6">
            <h1 class="tb-text-3xl tb-font-bold">Mein POA</h1>
            <div>
                <span id="currentUser" class="tb-mr-4"></span>
                <div id="darkModeToggleContainer" style="display: inline-block;">
                    <!-- DarkModeToggle wird von tbjs initialisiert -->
                </div>
            </div>
        </header>

        <main class="tb-grid tb-grid-cols-1 md:tb-grid-cols-2 tb-gap-6">
            <!-- Aufgaben Sektion -->
            <section id="tasks-section">
                <div class="tb-flex tb-justify-between tb-items-center tb-mb-3">
                    <h2 class="tb-text-2xl tb-font-semibold">Aufgaben</h2>
                    <button id="addTaskBtn" class="tb-btn tb-btn-primary">
                        <span class="material-symbols-outlined tb-mr-1">add_circle</span> Neue Aufgabe
                    </button>
                </div>
                <div id="taskList" class="tb-space-y-2">
                    <!-- Aufgaben werden hier dynamisch eingefügt -->
                </div>
            </section>

            <!-- Notizen Sektion -->
            <section id="notes-section">
                 <div class="tb-flex tb-justify-between tb-items-center tb-mb-3">
                    <h2 class="tb-text-2xl tb-font-semibold">Notizen</h2>
                    <button id="addNoteBtn" class="tb-btn tb-btn-primary">
                        <span class="material-symbols-outlined tb-mr-1">note_add</span> Neue Notiz
                    </button>
                </div>
                <div id="noteList" class="tb-grid tb-grid-cols-1 sm:tb-grid-cols-2 tb-gap-4">
                    <!-- Notizen werden hier dynamisch eingefügt -->
                </div>
            </section>
        </main>

    </div>

    <!-- tbjs JavaScript und Abhängigkeiten -->
    <script defer src="https://unpkg.com/htmx.org@2.0.2/dist/htmx.min.js"></script> <!-- Optional, falls genutzt -->
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.153.0/three.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/marked-highlight/lib/index.umd.min.js"></script>
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

    <!-- Eigene Anwendungslogik -->
    <script defer type="module">
    // Warten bis DOM geladen ist und tbjs initialisiert wurde

// Globale Variable für den aktuellen Benutzer (vereinfacht)
// In einer echten App würde dies durch TB.user.getUsername() nach Login kommen
let currentUsername = TB.user.getUsername() || 'default_user';
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         initializeApp();
    } else {
        window.TB.events.on('tbjs:initialized', initializeApp, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', initializeApp, { once: true }); // Custom event dispatch from TB.init
}

function initializeApp() {

    // Benutzername anzeigen
    const currentUserSpan = document.getElementById('currentUser');
    if (currentUserSpan) {
        currentUserSpan.textContent = `Benutzer: ${currentUsername}`;
    }
    TB.state.set('currentUser', { name: currentUsername }); // Für andere Module verfügbar machen

    // Event Listener für Buttons
    document.getElementById('addTaskBtn')?.addEventListener('click', showAddTaskModal);
    document.getElementById('addNoteBtn')?.addEventListener('click', showAddNoteModal);

    // Initiale Daten laden
    loadTasks();
    loadNotes();

    // Beispiel für eine Startanimation mit TB.graphics
    TB.graphics.playAnimationSequence("R1+32:P2-14");

}

// --- Aufgaben Funktionen ---
async function loadTasks() {
    try {
        const response = await TB.api.request('POA', `get_tasks`, null, 'GET');
        if (response.error === TB.ToolBoxError.none && response.get()) {
            renderTasks(response.get());
        } else {
            TB.ui.Toast.showError('Fehler beim Laden der Aufgaben: ' + response.info.help_text);
        }
    } catch (error) {
        TB.ui.Toast.showError('Netzwerkfehler beim Laden der Aufgaben.');
        console.error(error);
    }
}

function renderTasks(tasks) {
    const taskListDiv = document.getElementById('taskList');
    if (!taskListDiv) return;
    taskListDiv.innerHTML = ''; // Liste leeren

    if (tasks.length === 0) {
        taskListDiv.innerHTML = '<p class="tb-text-gray-500 dark:tb-text-gray-400">Keine Aufgaben vorhanden.</p>';
        return;
    }

    tasks.sort((a,b) => b.createdAt - a.createdAt).forEach(task => {
        const taskEl = document.createElement('div');
        taskEl.className = `task-item tb-bg-white dark:tb-bg-gray-700 tb-shadow tb-p-3 tb-rounded tb-flex tb-justify-between tb-items-center ${task.completed ? 'completed' : ''}`;
        taskEl.innerHTML = `
            <span class="tb-flex-grow ${task.completed ? 'tb-line-through tb-opacity-70' : ''}">${TB.utils.escapeHtml(task.text)}</span>
            <div class="tb-flex tb-items-center">
                <button class="tb-btn tb-btn-icon tb-text-green-500 hover:tb-text-green-700" data-task-id="${task.id}" data-action="toggle">
                    <span class="material-symbols-outlined">${task.completed ? 'check_box' : 'check_box_outline_blank'}</span>
                </button>
                <button class="tb-btn tb-btn-icon tb-text-red-500 hover:tb-text-red-700 tb-ml-2" data-task-id="${task.id}" data-action="delete">
                    <span class="material-symbols-outlined">delete</span>
                </button>
            </div>
        `;
        taskListDiv.appendChild(taskEl);
    });

    // Event Listener für Aktionen an Aufgaben-Buttons
    taskListDiv.querySelectorAll('button[data-task-id]').forEach(button => {
        button.addEventListener('click', handleTaskAction);
    });
}

async function handleTaskAction(event) {
    const button = event.currentTarget;
    const taskId = button.dataset.taskId;
    const action = button.dataset.action;

    TB.ui.Loader.show('Bearbeite...');
    try {
        if (action === 'toggle') {
            const response = await TB.api.request('POA', `toggle_task?task_id=${taskId}`, null, 'PUT');
            if (response.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess('Aufgabe aktualisiert!');
            } else {
                TB.ui.Toast.showError('Fehler: ' + response.info.help_text);
            }
        } else if (action === 'delete') {
            const response = await TB.api.request('POA', `delete_task?task_id=${taskId}`, null, 'DELETE');
             if (response.error === TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess('Aufgabe gelöscht!');
            } else {
                TB.ui.Toast.showError('Fehler: ' + response.info.help_text);
            }
        }
        await loadTasks(); // Liste neu laden
    } catch (error) {
        TB.ui.Toast.showError('Netzwerkfehler.');
        console.error(error);
    } finally {
        TB.ui.Loader.hide();
    }
}

function showAddTaskModal() {
    TB.ui.Modal.show({
        title: 'Neue Aufgabe hinzufügen',
        content: `
            <form id="addTaskFormModal">
                <div class="tb-form-group">
                    <label for="taskText" class="tb-label">Aufgabentext:</label>
                    <input type="text" id="taskTextModal" class="tb-input tb-w-full" required>
                </div>
            </form>
        `,
        buttons: [
            {
                text: 'Abbrechen',
                action: (modal) => modal.close(),
                variant: 'secondary'
            },
            {
                text: 'Hinzufügen',
                action: async (modal) => {
                    const taskText = document.getElementById('taskTextModal').value;
                    if (!taskText.trim()) {
                        TB.ui.Toast.showWarning('Aufgabentext darf nicht leer sein.');
                        return;
                    }
                    TB.ui.Loader.show('Speichere...');
                    try {
                        const response = await TB.api.request('POA', `add_task`, { text: taskText }, 'POST');
                        if (response.error === TB.ToolBoxError.none) {
                            TB.ui.Toast.showSuccess('Aufgabe hinzugefügt!');
                            await loadTasks();
                            modal.close();
                        } else {
                             TB.ui.Toast.showError('Fehler: ' + response.info.help_text);
                        }
                    } catch (error) {
                        TB.ui.Toast.showError('Netzwerkfehler.');
                         console.error(error);
                    } finally {
                        TB.ui.Loader.hide();
                    }
                },
                variant: 'primary'
            }
        ],
        onOpen: () => document.getElementById('taskTextModal')?.focus()
    });
}


// --- Notizen Funktionen (Struktur ähnlich zu Aufgaben) ---
async function loadNotes() {
    try {
        const response = await TB.api.request('POA', `get_notes`, null, 'GET');
        if (response.error === TB.ToolBoxError.none && response.get()) {
            renderNotes(response.get());
        } else {
            TB.ui.Toast.showError('Fehler beim Laden der Notizen: ' + response.info.help_text);
        }
    } catch (error) {
        TB.ui.Toast.showError('Netzwerkfehler beim Laden der Notizen.');
        console.error(error);
    }
}

function renderNotes(notes) {
    const noteListDiv = document.getElementById('noteList');
    if (!noteListDiv) return;
    noteListDiv.innerHTML = ''; // Liste leeren

    if (notes.length === 0) {
        noteListDiv.innerHTML = '<p class="tb-text-gray-500 dark:tb-text-gray-400">Keine Notizen vorhanden.</p>';
        return;
    }

    notes.sort((a,b) => b.updatedAt - a.updatedAt).forEach(note => {
        const noteEl = document.createElement('div');
        noteEl.className = 'note-item tb-bg-yellow-100 dark:tb-bg-yellow-800 tb-p-4 tb-rounded tb-shadow';
        // Gekürzter Inhalt für die Übersicht, Markdown für Details im Modal
        const shortContent = TB.utils.escapeHtml(note.content.length > 100 ? note.content.substring(0, 97) + '...' : note.content);

        noteEl.innerHTML = `
            <h3 class="tb-font-semibold tb-text-lg tb-mb-1">${TB.utils.escapeHtml(note.title) || 'Unbenannte Notiz'}</h3>
            <p class="tb-text-sm tb-mb-3">${shortContent}</p>
            <div class="tb-text-right">
                <button class="tb-btn tb-btn-sm tb-btn-secondary tb-mr-2" data-note-id="${note.id}" data-action="edit">
                    <span class="material-symbols-outlined tb-text-xs tb-mr-1">edit</span>Bearbeiten
                </button>
                <button class="tb-btn tb-btn-sm tb-btn-danger" data-note-id="${note.id}" data-action="delete">
                    <span class="material-symbols-outlined tb-text-xs tb-mr-1">delete</span>Löschen
                </button>
            </div>
        `;
        noteListDiv.appendChild(noteEl);
    });

    noteListDiv.querySelectorAll('button[data-note-id]').forEach(button => {
        button.addEventListener('click', handleNoteAction);
    });
}

async function handleNoteAction(event) {
    const button = event.currentTarget;
    const noteId = button.dataset.noteId;
    const action = button.dataset.action;

    if (action === 'edit') {
        // Finde die Notizdaten, um sie im Modal vorzufüllen
        const response = await TB.api.request('POA', `get_notes`, null, 'GET');
        if (response.error === TB.ToolBoxError.none && response.get()) {
            const note = response.get().find(n => n.id === noteId);
            if (note) showEditNoteModal(note);
        } else {
            TB.ui.Toast.showError('Notiz nicht gefunden.');
        }
    } else if (action === 'delete') {
        TB.ui.Modal.show({
            title: 'Notiz löschen?',
            content: '<p>Möchtest du diese Notiz wirklich unwiderruflich löschen?</p>',
            buttons: [
                { text: 'Abbrechen', action: modal => modal.close(), variant: 'secondary' },
                {
                    text: 'Löschen',
                    variant: 'danger',
                    action: async modal => {
                        TB.ui.Loader.show('Lösche...');
                        try {
                            const delResponse = await TB.api.request('POA', `delete_note?note_id=${noteId}`, null, 'DELETE');
                            if (delResponse.error === TB.ToolBoxError.none) {
                                TB.ui.Toast.showSuccess('Notiz gelöscht!');
                                await loadNotes();
                            } else {
                                TB.ui.Toast.showError('Fehler: ' + delResponse.info.help_text);
                            }
                        } catch (e) { TB.ui.Toast.showError('Netzwerkfehler.'); console.error(e); }
                        finally { TB.ui.Loader.hide(); modal.close(); }
                    }
                }
            ]
        });
    }
}

function showAddNoteModal() {
    showEditNoteModal(null); // Ruft die Edit-Modal Funktion ohne Daten auf
}

function showEditNoteModal(note) { // note kann null sein für eine neue Notiz
    const isEditing = note !== null;
    const title = isEditing ? note.title : '';
    const content = isEditing ? note.content : '';

    TB.ui.Modal.show({
        title: isEditing ? 'Notiz bearbeiten' : 'Neue Notiz erstellen',
        maxWidth: '600px',
        content: `
            <form id="noteFormModal">
                <div class="tb-form-group tb-mb-3">
                    <label for="noteTitleModal" class="tb-label">Titel:</label>
                    <input type="text" id="noteTitleModal" class="tb-input tb-w-full" value="${TB.utils.escapeHtml(title)}">
                </div>
                <div class="tb-form-group">
                    <label for="noteContentModal" class="tb-label">Inhalt (Markdown unterstützt):</label>
                    <textarea id="noteContentModal" class="tb-input tb-w-full" rows="8" style="min-height: 150px;">${TB.utils.escapeHtml(content)}</textarea>
                    <div id="notePreviewModal" class="tb-prose dark:tb-prose-invert tb-mt-2 tb-p-2 tb-border tb-rounded tb-bg-gray-50 dark:tb-bg-gray-800" style="min-height: 50px;"></div>
                </div>
            </form>
        `,
        buttons: [
            { text: 'Abbrechen', action: (modal) => modal.close(), variant: 'secondary' },
            {
                text: isEditing ? 'Speichern' : 'Erstellen',
                variant: 'primary',
                action: async (modal) => {
                    const newTitle = document.getElementById('noteTitleModal').value;
                    const newContent = document.getElementById('noteContentModal').value;

                    if (!newContent.trim()) {
                        TB.ui.Toast.showWarning('Notizinhalt darf nicht leer sein.');
                        return;
                    }
                    TB.ui.Loader.show(isEditing ? 'Aktualisiere...' : 'Speichere...');
                    try {
                        const payload = { title: newTitle, content: newContent };
                        const method = isEditing ? 'PUT' : 'POST';
                        const endpoint = isEditing ? `update_note?&note_id=${note.id}` : `add_note?`;

                        const response = await TB.api.request('POA', endpoint, payload, method);

                        if (response.error === TB.ToolBoxError.none) {
                            TB.ui.Toast.showSuccess(`Notiz ${isEditing ? 'aktualisiert' : 'erstellt'}!`);
                            await loadNotes();
                            modal.close();
                        } else {
                             TB.ui.Toast.showError('Fehler: ' + response.info.help_text);
                        }
                    } catch (error) {
                        TB.ui.Toast.showError('Netzwerkfehler.');
                         console.error(error);
                    } finally {
                        TB.ui.Loader.hide();
                    }
                }
            }
        ],
        onOpen: () => {
            const textarea = document.getElementById('noteContentModal');
            const previewDiv = document.getElementById('notePreviewModal');
            if (textarea && previewDiv && TB.ui.MarkdownRenderer) {
                 // Initial render
                previewDiv.innerHTML = TB.ui.MarkdownRenderer.render(textarea.value);
                // Live preview
                textarea.addEventListener('input', () => {
                    previewDiv.innerHTML = TB.ui.MarkdownRenderer.render(textarea.value);
                });
            }
            textarea?.focus();
        }
    });
}


    </script></div>
    """
    return Result.html(data=html_content)


# Initialisierungsfunktion für das Modul (optional)
@export(mod_name=MOD_NAME, version=VERSION, initial=True)
def initialize_poa_module(app: App):
    print(f"Persönlicher Organisationsassistent Modul ({MOD_NAME} v{VERSION}) initialisiert.")
    # Hier könnten Standard-DB-Strukturen geprüft oder erstellt werden, falls nötig.
    # Beispiel: Sicherstellen, dass der DB-Client korrekt konfiguriert ist.
    # db = app.get_mod("DB")

    if app is None:
        app = get_app()
    app.run_any(("CloudM","add_ui"),
                name=Name,
                title=Name,
                path=f"/api/{Name}/main",
                description="main"
                )
    return Result.ok(info="POA Modul bereit.")
