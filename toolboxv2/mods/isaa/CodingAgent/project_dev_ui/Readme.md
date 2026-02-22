# 🚀 ProjectDeveloper Studio

A sophisticated Streamlit-based web UI for the **ProjectDeveloperEngine V3** - an AI-powered multi-file code generation system.

![ProjectDeveloper Studio](https://img.shields.io/badge/Streamlit-1.40+-red?style=flat-square&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)

## ✨ Features

### 🎯 Two-Panel Interface
- **Left Panel**: Chat interface with formatted developer activity
- **Right Panel**: Live preview, code viewer, and function testing

### 📁 Project Management
- Multiple project sessions with SQLite persistence
- Switch between projects instantly
- Full chat history per project
- Generated file tracking with versioning

### 💻 Code Generation
- Natural language task descriptions
- Multi-file generation support
- Automatic file type detection
- Python, JavaScript, TypeScript, HTML, CSS, and more

### 👁️ Preview & Testing
- **Code View**: Syntax-highlighted code with line numbers
- **Live Preview**: **Real HTTP server** serving your apps with:
  - ⚛️ React (JSX/TSX) support via ESM imports
  - 💚 Vue SFC (Single File Component) support
  - 🔄 Hot-reload on file changes
  - 🔗 Open in new browser tab
- **Test Runner**: Execute Python code and run unit tests
- **File Browser**: Overview of all generated files with statistics

### 🔄 Execution Control
- Pause/Resume/Stop execution
- Real-time status updates
- Progress tracking per phase

## 🚀 Quick Start

### Installation

```bash
# Clone or copy the project files
cd project_dev_ui

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### With ToolBoxV2 (Full Features)

If you have ToolBoxV2 installed, the application will automatically use the real ProjectDeveloperEngine:

```bash
# Ensure ToolBoxV2 is in your Python path
pip install -e /path/to/toolboxv2

# The app will detect and use FlowAgent for LLM interactions
streamlit run app.py
```

## 📖 Usage

### Creating a Project

1. Open the sidebar (click the `>` arrow if collapsed)
2. Click "➕ New Project"
3. Enter a name and description
4. Click "Create Project"

### Generating Code

Simply describe what you want to build in natural language:

**Examples:**
- "Create a REST API with Flask that has user authentication"
- "Build a landing page with a hero section and contact form"
- "Make a utility function that processes CSV files"
- "Create a React component for a data table with sorting"

### Previewing Results

- **Code Tab**: View generated files with syntax highlighting
- **Preview Tab**: See HTML/CSS/JS rendered live
- **Test Tab**: Run Python code snippets against generated files
- **Files Tab**: Download individual files or everything as ZIP

## 🏗️ Architecture

```
project_dev_ui/
├── app.py              # Main Streamlit application
├── db.py               # SQLite database layer
├── agent_connector.py  # Proxy agent for ProjectDeveloperEngine
├── preview_server.py   # HTTP server for live app preview
├── requirements.txt    # Python dependencies
├── project_dev.db      # SQLite database (auto-created)
└── projects/           # Project workspaces (auto-created)
    └── {project_id}/   # Individual project files
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `DatabaseManager` | SQLite persistence for projects, chat, files |
| `ProxyAgent` | Interprets user requests, manages execution |
| `PreviewServer` | HTTP server for serving generated apps |
| `StatusUpdate` | Real-time progress callbacks |
| `ExecutionResult` | Structured execution output |

## 🖥️ Preview Server

The preview server enables real-time viewing of generated applications:

### Features
- **Real HTTP Server**: Each project gets its own server (ports 8600+)
- **React Support**: JSX/TSX files transformed via ESM imports from esm.sh
- **Vue Support**: SFC files parsed and transformed at runtime
- **Hot Reload**: Automatic reload on file changes
- **CORS Enabled**: Works in iframe embedding

### Framework Support

| Framework | Files | How it works |
|-----------|-------|--------------|
| React | `.jsx`, `.tsx` | ESM importmap + esm.sh CDN |
| Vue | `.vue` | Runtime SFC parsing |
| Static | `.html`, `.js`, `.css` | Direct serving |

### Usage

```python
from preview_server import create_preview_server

# Start server
server = create_preview_server("/path/to/workspace")
print(f"Preview at: {server.url}")

# Update files without disk write
server.update_file("app.js", "console.log('updated')")

# Stop server
server.stop()
```

## 🎨 UI Theme

The application features a custom dark theme with:
- **Colors**: Deep blue/purple gradients
- **Fonts**: JetBrains Mono (code), Outfit (UI)
- **Animations**: Smooth transitions and fade-ins

## ⚙️ Configuration

### Mock Mode vs Real Mode

The application supports two modes:

1. **Mock Mode** (Default): Generates sample code without LLM
2. **Real Mode**: Uses ToolBoxV2's FlowAgent for AI generation

To enable real mode, ensure ToolBoxV2 is installed and set `use_mock=False` in `agent_connector.py`.

### Database Location

By default, the SQLite database is stored at:
```
project_dev_ui/project_dev.db
```

To change this, modify the `get_db()` call in `app.py`.

## 🧪 Testing

The project uses unittest (per Markin's preference):

```bash
# Run tests
python -m unittest discover -s tests

# Test specific module
python -m unittest tests.test_db
```

## 📝 API Reference

### ProxyAgent

```python
agent = ProxyAgent(
    workspace_path="./workspace",
    status_callback=my_callback,
    use_mock=False
)

# Parse user intent
task, files = agent.parse_task("Create a function to process data")

# Execute task
result = await agent.execute_task("Create a REST API")

# Run tests
test_result = await agent.run_tests("utils.py")

# Execute arbitrary code
exec_result = await agent.execute_code("print('hello')")
```

### DatabaseManager

```python
db = get_db("my_database.db")

# Projects
project = db.create_project("id", "name", "desc", "/path")
projects = db.list_projects()
db.delete_project("id")

# Chat
db.add_chat_message("project_id", "user", "Hello")
messages = db.get_chat_history("project_id")

# Files
db.save_generated_file("project_id", "main.py", "code", "python")
files = db.get_generated_files("project_id")
```

## 🤝 Integration with ToolBoxV2

This UI is designed to work seamlessly with ToolBoxV2's ecosystem:

- **FlowAgent**: LLM orchestration with chain patterns
- **DocsSystem**: Project context and semantic search
- **DockerCodeExecutor**: Safe code execution
- **RestrictedPythonExecutor**: Sandboxed Python execution

## 📜 License

This project is part of the ToolBoxV2 ecosystem.

---

**Built with ❤️ using Streamlit and ProjectDeveloperEngine V3**
