from fastapi.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from toolboxv2 import MainTool, get_app, App
from toolboxv2.utils.extras import BaseWidget
from toolboxv2.utils.extras.blobs import BlobStorage, BlobFile

Name = "DoNextWidget"
version = "0.0.1"


class DoNextWidget(MainTool, BaseWidget):
    def __init__(self, app=None):
        self.name = "DoNextWidget"
        self.color = "BLUE"
        self.version = "1.0.0"
        self.tasks_key = "tasks"
        self.history_key = "task_history"

        MainTool.__init__(self,
                          load=self.on_start,
                          v=self.version,
                          name=self.name,
                          color=self.color,
                          on_exit=self.on_exit)

        BaseWidget.__init__(self, name=self.name)

        # Registriere API-Endpunkte
        self.register(self.app, self.get_widget, self.version)
        self.register(self.app, self.fetch_tasks, self.version, name="tasks", api_methods=["GET"], row=True)
        self.register(self.app, self.save_task, self.version, name="save_task", api_methods=["POST"])
        self.register(self.app, self.save_task_history, self.version, name="history", api_methods=["POST"])
        self.register(self.app, self.get_task_history, self.version, name="task_history", api_methods=["GET"], row=True)

    async def get_blob_storage(self, request):

        user = await self.get_user_from_request(self.app, request)
        if user.name == "":
            return BlobStorage(self.app.data_dir + '/public', 0)
        if user.name == "root":
            return BlobStorage()
        return BlobStorage(self.app.data_dir + '/storages/' + user.uid)

    @staticmethod
    def get_template(request):
        return """
        <!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task Manager V2</title>
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
            padding: 20px;
        }

        .app-container {
            max-width: 600px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
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

        .current-action {
            background: #2C3E50;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .suggestion {
            background: white;
            cursor: pointer;
        }

        .suggestion:hover {
            background: #f8f9fa;
        }

        .history-section {
            background: white;
            margin-top: 20px;
            transition: max-height 0.3s ease-out;
            overflow: hidden;
        }

        .history-section.collapsed {
            max-height: 60px;
        }

        .history-section.expanded {
            max-height: 500px;
        }

        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            cursor: pointer;
        }

        .toggle-icon {
            transition: transform 0.3s ease;
        }

        .collapsed .toggle-icon {
            transform: rotate(180deg);
        }

        /* Neue Styles für Time Pin */
        .time-pin {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
            padding: 8px;
            background: rgba(44, 62, 80, 0.1);
            border-radius: 8px;
        }

        /* Rest des CSS bleibt unverändert */
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }

        .history-tabs {
            display: flex;
            gap: 10px;
        }

        .history-tab {
            padding: 5px 10px;
            cursor: pointer;
            border-radius: 4px;
        }

        .history-tab.active {
            background: #e9ecef;
        }

        .history-entry {
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }

        .time-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .time-block {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .action-content {
            margin: 16px 0;
            font-size: 1.1rem;
        }

        .button-group {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }

        .btn-primary {
            background: #ECF0F1;
            color: #2C3E50;
        }

        .btn-primary:hover {
            background: #D5DBDB;
        }

        .btn-secondary {
            background: rgba(236, 240, 241, 0.1);
            color: white;
        }

        .btn-secondary:hover {
            background: rgba(236, 240, 241, 0.2);
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .modal-content {
            background: white;
            padding: 24px;
            border-radius: 16px;
            width: 90%;
            max-width: 500px;
            max-height: 95%;
            overflow-y: auto;
        }

        .input-group {
            margin: 16px 0;
        }

        .input-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .input-group input,
        .input-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
        }

        .badge {
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
        }

        .badge-priority-high {
            background: rgba(231, 76, 60, 0.1);
            color: #E74C3C;
        }

        .badge-priority-medium {
            background: rgba(243, 156, 18, 0.1);
            color: #F39C12;
        }

        .badge-priority-low {
            background: rgba(46, 204, 113, 0.1);
            color: #2ECC71;
        }

        .badge-recurring {
            background: rgba(52, 152, 219, 0.1);
            color: #3498DB;
        }

        .history-list {
            max-height: 400px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <section class="card current-action">
            <div class="card-header">
                <h2>Aktuelle Aktion</h2>
                <div>
                    <span class="badge" id="elapsed-time">00:00:00</span>
                    <span class="badge" id="current-time">00:00:00</span>
                </div>
            </div>
            <div class="action-content" id="current-action-content">
                Keine aktuelle Aktion
            </div>
            <div class="button-group">
                <button class="btn btn-primary" onclick="openNewTaskModal()">Neue Aktion</button>
                <button class="btn btn-secondary" onclick="completeCurrentAction()">Abschließen</button>
            </div>
        </section>

        <section class="card suggestion" id="suggestion1" onclick="selectSuggestion(1)">
            <div class="card-header">
                <h3>Nächste priorisierte Aufgabe</h3>
                <span class="badge" id="suggestion1-priority"></span>
            </div>
            <div class="action-content" id="suggestion1-content">
                Wird geladen...
            </div>
        </section>

        <section class="card suggestion" id="suggestion2" onclick="selectSuggestion(2)">
            <div class="card-header">
                <h3>Alternative Aufgabe</h3>
                <span class="badge" id="suggestion2-priority"></span>
            </div>
            <div class="action-content" id="suggestion2-content">
                Wird geladen...
            </div>
        </section>

        <section class="card history-section collapsed" id="history-section">
            <div class="history-header" onclick="toggleHistory()">
                <h3>Letzte Aktivitäten (5)</h3>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="history-list" id="history-list">
                <!-- Wird dynamisch gefüllt -->
            </div>
        </section>
    </div>

    <!-- Modal für neue Aktion -->
    <div class="modal" id="newTaskModal">
        <div class="modal-content">
            <div class="card-header">
                <h2>Neue Aktion erstellen</h2>
                <button class="btn btn-secondary" onclick="closeNewTaskModal()">×</button>
            </div>
            <form id="newTaskForm" onsubmit="createNewTask(event)">
                <div class="input-group">
                    <label for="taskName">Aktionsname</label>
                    <input type="text" id="taskName" required>
                </div>
                <div class="input-group">
                    <label for="taskCategory">Kategorie</label>
                    <select id="taskCategory" required>
                        <option value="work">Arbeit</option>
                        <option value="personal">Persönlich</option>
                        <option value="household">Haushalt</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="taskPriority">Priorität</label>
                    <select id="taskPriority" required>
                        <option value="high">Hoch</option>
                        <option value="medium">Mittel</option>
                        <option value="low">Niedrig</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="taskRecurring">Wiederkehrend</label>
                    <select id="taskRecurring">
                        <option value="none">Nein</option>
                        <option value="daily">Täglich</option>
                        <option value="weekly">Wöchentlich</option>
                        <option value="monthly">Monatlich</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="taskTimePin">Zeitliche Planung</label>
                    <input type="datetime-local" id="taskTimePin">
                </div>
                <div class="input-group">
                    <label for="taskNotification">Benachrichtigung</label>
                    <select id="taskNotification">
                        <option value="none">Keine</option>
                        <option value="atTime">Zum Zeitpunkt</option>
                        <option value="5min">5 Minuten vorher</option>
                        <option value="15min">15 Minuten vorher</option>
                    </select>
                </div>
                <div class="button-group">
                    <button type="submit" class="btn btn-primary">Erstellen</button>
                </div>
            </form>
        </div>
    </div>

    <script unSave="true">
        // Datenverwaltung
        let currentTask = null;
        let currentTaskStartTime = null;
        let tasks = [];
        let taskHistory = [];

        // Zeit-Tracking
        function startTaskTimer() {
            currentTaskStartTime = new Date();
            updateElapsedTime();
        }

        function updateElapsedTime() {
            if (currentTask && currentTaskStartTime) {
                const elapsed = new Date() - currentTaskStartTime;
                const hours = Math.floor(elapsed / 3600000);
                const minutes = Math.floor((elapsed % 3600000) / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                document.getElementById('elapsed-time').textContent =
                    String(hours).padStart(2, '0') + ":" +
                    String(minutes).padStart(2, '0') + ":" +
                    String(seconds).padStart(2, '0');
            }
        }

        // UI Updates
        function updateCurrentAction() {
            const content = document.getElementById('current-action-content');
            if (currentTask) {
                content.innerHTML =
                    currentTask.name +
                    "<br>" +
                    "<small>" + currentTask.category + "</small>" +
                    '<span class="badge badge-priority-' + currentTask.priority + '">' + currentTask.priority + "</span>" +
                    (currentTask.recurring !== 'none'
                        ? '<span class="badge badge-recurring">' + currentTask.recurring + "</span>"
                        : '');
                startTaskTimer();
            } else {
                content.innerHTML = 'Keine aktuelle Aktion';
                document.getElementById('elapsed-time').textContent = '00:00:00';
            }
        }

        // Helper functions
        function formatDuration(ms) {
            const hours = Math.floor(ms / 3600000);
            const minutes = Math.floor((ms % 3600000) / 60000);
            const seconds = Math.floor((ms % 60000) / 1000);
            return String(hours).padStart(2, '0') + ":" +
                   String(minutes).padStart(2, '0') + ":" +
                   String(seconds).padStart(2, '0');
        }

        // Task-Management
        function getSuggestedTasks() {
            return tasks
                .filter(task => task !== currentTask)
                .sort((a, b) => {
                    // Priorisierung nach Wichtigkeit
                    const priorityOrder = { high: 3, medium: 2, low: 1 };
                    return priorityOrder[b.priority] - priorityOrder[a.priority];
                })
                .slice(0, 2);
        }

        function selectSuggestion(number) {
            const suggestions = getSuggestedTasks();
            if (suggestions[number - 1]) {
                if (currentTask) {
                    completeCurrentAction();
                }
                currentTask = suggestions[number - 1];
                updateCurrentAction();
                updateSuggestions();
            }
        }

        // Modal-Management
        function openNewTaskModal() {
            document.getElementById('newTaskModal').style.display = 'flex';
        }

        function closeNewTaskModal() {
            document.getElementById('newTaskModal').style.display = 'none';
        }

        // Zeit-Update
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent =
                now.toLocaleTimeString('de-DE');
            updateElapsedTime();
        }

        // Demo-Daten
        function loadDemoData() {
            tasks = [
                { id: 1, name: "E-Mails checken", category: "work", priority: "high", recurring: "daily", lastCompleted: null },
                { id: 2, name: "Wäsche waschen", category: "household", priority: "medium", recurring: "weekly", lastCompleted: null },
                { id: 3, name: "Sport machen", category: "personal", priority: "high", recurring: "daily", lastCompleted: null },
                { id: 4, name: "Projekt Review", category: "work", priority: "medium", recurring: "none", lastCompleted: null }
            ];

            // Demo-Verlaufsdaten
            const now = new Date();
            taskHistory = [
                {
                    task: { name: "Meeting vorbereiten", category: "work", priority: "high" },
                    startTime: new Date(now.getTime() - 7200000),
                    endTime: new Date(now.getTime() - 3600000)
                },
                {
                    task: { name: "Mittagspause", category: "personal", priority: "medium" },
                    startTime: new Date(now.getTime() - 3600000),
                    endTime: new Date(now.getTime() - 2700000)
                }
            ];
        }

        // API Funktionen
        const API_BASE_URL = '/api/DoNextWidget'; // Ändern Sie dies zur tatsächlichen API-URL

        async function fetchTasks() {
            try {
                const response = await fetch(API_BASE_URL+"/tasks");
                if (!response.ok) throw new Error('Fehler beim Laden der Tasks');
                const data = await response.json();
                tasks = data;
                updateSuggestions();
            } catch (error) {
                console.error('Fehler beim Laden der Tasks:', error);
            }
        }

        async function fetchTasksHistory() {
            try {
                const response = await fetch(API_BASE_URL+"/task_history");
                if (!response.ok) throw new Error('Fehler beim Laden der Tasks');
                const data = await response.json();
                taskHistory = data;
                updateSuggestions();
            } catch (error) {
                console.error('Fehler beim Laden der TasksHistory:', error);
            }
        }

        async function saveTask(task) {
            try {
                const response = await fetch(API_BASE_URL+"/save_task", {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(task)
                });
                if (!response.ok) throw new Error('Fehler beim Speichern des Tasks');
                const savedTask = await response.json();
                return savedTask;
            } catch (error) {
                console.error('Fehler beim Speichern des Tasks:', error);
                return null;
            }
        }

        async function saveTaskHistory(historyEntry) {
            try {
                const response = await fetch(API_BASE_URL+"/history", {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(historyEntry)
                });
                if (!response.ok) throw new Error('Fehler beim Speichern des Verlaufs');
            } catch (error) {
                console.error('Fehler beim Speichern des Verlaufs:', error);
            }
        }

        // Timer und Benachrichtigungen
        function scheduleTaskNotification(task) {
            if (!task.timePin || !task.notification) return;

            const timePin = new Date(task.timePin);
            let notificationTime = new Date(timePin);

            switch (task.notification) {
                case '5min':
                    notificationTime.setMinutes(timePin.getMinutes() - 5);
                    break;
                case '15min':
                    notificationTime.setMinutes(timePin.getMinutes() - 15);
                    break;
            }

            const timeUntilNotification = notificationTime.getTime() - Date.now();
            if (timeUntilNotification > 0) {
                setTimeout(() => {
                    if (Notification.permission === "granted") {
                        new Notification("Task Erinnerung: " + task.name, {
                            body: "Der Task " + task.name + " beginnt " + (task.notification === 'atTime' ? 'jetzt' : 'in Kürze') + "."
                        });
                    }
                }, timeUntilNotification);
            }
        }

        // UI Updates
        function toggleHistory() {
            const historySection = document.getElementById('history-section');
            historySection.classList.toggle('collapsed');
            historySection.classList.toggle('expanded');
        }

        function updateHistory() {
            const historyList = document.getElementById('history-list');
            const recentHistory = taskHistory
                .sort((a, b) => b.endTime - a.endTime)
                .slice(0, 5);

            historyList.innerHTML = recentHistory.map(function(entry) {
                return '<div class="history-entry">' +
                    "<strong>" + entry.task.name + "</strong>" +
                    "<br>" +
                    "<small>" +
                        new Date(entry.startTime).toLocaleString('de-DE') + " - " +
                        formatDuration(entry.endTime - entry.startTime) +
                    "</small>" +
                "</div>";
            }).join('');

        }

        function updateSuggestions() {
            const suggestions = getSuggestedTasks();

            ['suggestion1', 'suggestion2'].forEach((id, index) => {
                const content = document.getElementById(id + "-content");
                const priorityBadge = document.getElementById(id + "-priority");

                if (suggestions[index]) {
                    content.innerHTML =
                        suggestions[index].name +
                        "<br>" +
                        "<small>" + suggestions[index].category + "</small>" +
                        (suggestions[index].recurring !== 'none'
                            ? '<span class="badge badge-recurring">' + suggestions[index].recurring + "</span>"
                            : '') +
                        (suggestions[index].timePin
                            ? '<div class="time-pin">' +
                                "<span>📅</span> Geplant für: " + new Date(suggestions[index].timePin).toLocaleString('de-DE') +
                              "</div>"
                            : '');

                    priorityBadge.className = "badge badge-priority-" + suggestions[index].priority;
                    priorityBadge.textContent = suggestions[index].priority;
                } else {
                    content.innerHTML = 'Keine weiteren Vorschläge';
                    priorityBadge.className = '';
                    priorityBadge.textContent = '';
                }
            });
        }

        // Task Management
        async function createNewTask(event) {
            event.preventDefault();
            const newTask = {
                id: tasks.length + 1,
                name: document.getElementById('taskName').value,
                category: document.getElementById('taskCategory').value,
                priority: document.getElementById('taskPriority').value,
                recurring: document.getElementById('taskRecurring').value,
                timePin: document.getElementById('taskTimePin').value || null,
                notification: document.getElementById('taskNotification').value,
                lastCompleted: null
            };

            const savedTask = await saveTask(newTask);
            if (savedTask) {
                tasks.push(savedTask);
                if (currentTask) {
                    completeCurrentAction();
                }
                currentTask = savedTask;
                startTaskTimer();
                scheduleTaskNotification(savedTask);

                updateCurrentAction();
                updateSuggestions();
                closeNewTaskModal();
                event.target.reset();
            }
        }

        async function completeCurrentAction() {
            if (currentTask) {
                const historyEntry = {
                    task: { ...currentTask },
                    startTime: currentTaskStartTime,
                    endTime: new Date()
                };
                await saveTaskHistory(historyEntry);
                taskHistory.push(historyEntry);

                currentTask = null;
                currentTaskStartTime = null;
                updateCurrentAction();
                updateSuggestions();
                updateHistory();
            }
        }

        // Initialisierung
        async function init() {
            if (Notification.permission !== "granted") {
                await Notification.requestPermission();
            }
            await fetchTasks();
            await fetchTasksHistory();
            updateCurrentAction();
            updateSuggestions();
            updateHistory();
            setInterval(updateTime, 1000);
            updateTime();
        }
        setTimeout(async () => {await init();}, 0);

    </script>
</body>
</html>
"""

    async def fetch_tasks(self, request):
        """Holt alle Tasks für den aktuellen Benutzer"""
        storage = await self.get_blob_storage(request)
        try:
            with BlobFile("DoNextWidget/tasks", "r", storage=storage) as f:
                tasks = f.read_json().get("tasks", [])
            return JSONResponse(tasks)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def save_task(self, request):
        """Speichert einen neuen Task"""
        storage = await self.get_blob_storage(request)
        try:
            body = await request.json()
            with BlobFile("DoNextWidget/tasks", "r", storage=storage) as f:
                tasks = f.read_json().get("tasks", [])

            # Überprüfe, ob Task bereits existiert
            existing_task = next((task for task in tasks if task.get('id') == body.get('id')), None)

            if existing_task:
                # Update existierender Task
                index = tasks.index(existing_task)
                tasks[index] = body
            else:
                # Neuen Task hinzufügen
                tasks.append(body)

            with BlobFile("DoNextWidget/tasks", "w", storage=storage) as f:
                f.write_json({'tasks': tasks})
            return True
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def get_task_history(self, request):
        """Speichert den Task-Verlauf"""
        storage = await self.get_blob_storage(request)
        try:
            with BlobFile("DoNextWidget/history", "r", storage=storage) as f:
                history = f.read_json().get('history', [])
            return JSONResponse(history)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def save_task_history(self, request):
        """Speichert den Task-Verlauf"""
        storage = await self.get_blob_storage(request)
        try:
            body = await request.json()
            with BlobFile("DoNextWidget/history", "r", storage=storage) as f:
                history = f.read_json().get('history', [])

            history.append(body)
            # Begrenzen der Verlaufsgröße (z.B. letzte 5000 Einträge)
            history = history[-5000:]

            with BlobFile("DoNextWidget/history", "w", storage=storage) as f:
                f.write_json({'history': history})
            return JSONResponse({"status": "success"})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    def on_start(self):
        """Initialisierung beim Start"""
        self.register2reload(self.main)
        # Weitere Initialisierungslogik

    def on_exit(self):
        """Aufräumarbeiten beim Beenden"""
        pass

    async def get_widget(self, request, **kwargs):
        """Lädt das Widget"""
        w_id = self.get_s_id(request)
        if w_id.is_error():
            return w_id
        self.reload_guard(self.on_start)
        return self.load_widget(self.app, request, "main", self.hash_wrapper(w_id.get()))

    def main(self, request):
        """Hauptmethode zum Laden des Widgets"""
        w_id = self.get_s_id(request)
        if w_id.is_error():
            return w_id
        self.asset_loder(self.app, "main", self.hash_wrapper(w_id.get()), template=self.get_template())


Tools = DoNextWidget


@get_app().tb(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True,
              name="main_web_DoNextWidget_entry", row=True, state=False)
def entry(request: Request or None = None):
    return HTMLResponse(content=Tools.get_template(request))
