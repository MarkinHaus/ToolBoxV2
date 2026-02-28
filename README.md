# ToolBoxV2 🧰

[![PyPI Version](https://img.shields.io/pypi/v/ToolBoxV2.svg)](https://pypi.python.org/pypi/ToolBoxV2)
[![Donate](https://img.shields.io/badge/Donate-Buy%20me%20a%20coffee-yellowgreen.svg)](https://www.buymeacoffee.com/markinhaus)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/MarkinHaus/ToolBoxV2)

A flexible modular framework for tools, functions, and complete applications – deployable locally, on the web, or as a desktop/mobile app.

---

## 🔍 Overview & Architecture

ToolBoxV2 is a Python-first framework with a high-performance worker infrastructure for web, desktop, and mobile applications.

### Architecture

```
                    ┌─────────────┐
                    │    Nginx    │
                    │ (Load Bal., │
                    │ Rate Limit) │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ HTTP Worker │   │ HTTP Worker │   │ WS Worker   │
  │  (WSGI)     │   │  (WSGI)     │   │ (asyncio)   │
  │  Port 8000  │   │  Port 8001  │   │  Port 8010  │
  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
         │                 │                 │
         └────────────┬────┴─────────────────┘
                      │
               ┌──────▼──────┐
               │ ZeroMQ      │
               │ Event Broker│
               │ (Pub/Sub)   │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │ ToolBoxV2   │
               │ App Instance│
               └─────────────┘
```

### Key Components

- **Python Backend**: Core library with modular architecture
- **Worker System**: High-performance WSGI/async workers (replacing the legacy Rust server)
- **ZeroMQ IPC**: Fast inter-worker communication
- **Tauri + tbjs**: Cross-platform desktop/web UI
- **Nginx**: Load balancing, rate limiting, SSL termination

For detailed worker documentation, see [toolboxv2/utils/workers/README.md](toolboxv2/utils/workers/README.md).

---

## 🎯 Goal

The primary goal of ToolBoxV2 is to provide a flexible platform that enables developers, end-users, and small to medium-sized businesses to efficiently create, customize, and use applications. It aims to:
*   Execute applications seamlessly.
*   Integrate diverse functionalities.
*   Ensure system-independence.

The underlying system, built on a monolithic modular architecture, combines the advantages of both approaches, enabling intuitive interaction with the digital world. It connects various components and provides utility functions accessible from anywhere. This platform promotes creative collaboration and eases access to digital resources.

---

## 🎯 Target Audiences & Use Cases

### 👩‍💻 For Developers

Utilize ToolBoxV2 as a framework to:
*   Create custom functions, widgets, or complete mini-applications.
*   Leverage existing modules (`mods`) or extend them with new components.
*   Build web, desktop, or mobile applications using a unified Python stack.
*   Customize the user interface via the web frontend (tbjs).

### 🙋 For End Users

Access and use a variety of pre-built applications and functions:
*   Directly in a web browser, or as a native desktop/mobile application (powered by Tauri).
*   No prior technical knowledge required.
*   Access flexible tools for tasks such as calendar management, note-taking, image diffusion, quote generation, etc.
*   Personalize the user interface to meet individual needs.

### 🏢 For Businesses / Operators

Deploy ToolBoxV2 as a customizable internal management system for:
*   Self-hosted and highly adaptable solutions.
*   Project, process, or employee management.
*   Integration of proprietary modules and functions.
*   Scalability suitable for small to medium-sized enterprises.
*   Modular, API-ready, and easily extensible.

---

## 🚀 Installation

We offer several ways to install ToolBoxV2, choose the one that best suits your needs!

### 🥇 Recommended: Zero the Hero Universal Installer (Easiest)

This is the recommended method for most users on **Linux, macOS, and Windows (via WSL or Git Bash)**. The "Zero the Hero" script intelligently handles Python installation (if needed), sets up a dedicated virtual environment, installs ToolBoxV2 Core, and makes the `tb` command available.

1.  **Download the installer:**
    ```bash
    # Using curl
    curl -sSL -o install_toolbox.sh https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh
    # Or using wget
    wget -qO install_toolbox.sh https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh
    ```
2.  **Make it executable:**
    ```bash
    chmod +x install_toolbox.sh
    ```

3.  **Run the installer:**
    ```bash
    ./install_toolbox.sh
    ```

4.  **Follow on-screen instructions.** The script will:
    *   ✅ Check for and offer to install required Python version (default: 3.11).
    *   ✅ Create an isolated environment for ToolBoxV2 (usually in `~/.local/share/ToolBoxV2` or `~/Applications/ToolBoxV2`).
    *   ✅ Install ToolBoxV2 Core using `pip` by default.
    *   ✅ Expose the `tb` command (usually via a symlink in `~/.local/bin/`).
    *   🎉 Run `tb -init main` to finalize setup.

    **Customization:**
    The script accepts optional arguments (e.g., `--version=0.5.0`, `--source=git`, `--manager=uv`, `--isaa`, `--dev`). If no arguments are given, it looks for an `init.config` file in the same directory. For details, run:
    ```bash

    ./install_toolbox.sh --help
    ```

or Windows onliner

```bash
    & ([scriptblock]::Create((irm "https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.ps1")))
```

---

### 💻 Advanced / Alternative Methods

#### 1. Python Package (Direct Installation)

For users who prefer to manage their Python environments manually.

**Using `pip`:**
```bash
    # Ensure you have Python 3.11 and pip installed
    # Recommended: Create and activate a virtual environment first!
    # python -m venv .venv && source .venv/bin/activate

    pip install ToolBoxV2
    # To install with optional extras (e.g., isaa, dev):
    # pip install "ToolBoxV2[isaa,dev]"
```

**Using `uv` (a fast Python package installer & resolver):**
```bash

    uv pip install ToolBoxV2
    # To install with optional extras:
    # uv pip install "ToolBoxV2[isaa,dev]"
```

After installation with pip or uv, you may need to initialize ToolBoxV2 manually:
```bash
    tb -init main
```
Ensure the directory containing the `tb` script (e.g., `~/.local/bin` for user installs, or your venv's `bin` directory) is in your system's `PATH`.

#### 2. From Source (For Developers / Bleeding Edge)

If you want to contribute or use the very latest (potentially unstable) code:

```bash
    git clone https://github.com/MarkinHaus/ToolBoxV2.git
    cd ToolBoxV2

    # IMPORTANT: Set up and activate a Python virtual environment
    # Example using Python's built-in venv:
    # python3 -m venv .venv
    # source .venv/bin/activate
    #
    # Example using uv:
    # uv venv .venv --python 3.11 # or your desired Python version
    # source .venv/bin/activate

    # Install in editable mode:
    echo "Choose your preferred installation method:"

    echo "  Option A: Using pip"
    echo "    pip install -e \".[dev,isaa]\"  # Install with dev and isaa extras"
    echo "    # or just: pip install -e ."

    echo "  Option B: Using uv"
    echo "    uv pip install -e \".[dev,isaa]\" # Install with dev and isaa extras"
    echo "    # or just: uv pip install -e ."

    # Your existing script for Python environment setup (if it offers more specific dev setup):
    # If you have specific Python dev environment needs beyond a simple venv, you can use:
    # chmod +x install_python_env.sh
    # ./install_python_env.sh

    # Initialize Git hooks (for contributors)
    bash .github/hooks/setup_hooks.sh

    # Initialize ToolBoxV2
    tb -init main
```

---

#### 📦 Installers via GitHub Releases (Recommended for GUI App)

Find platform-specific installers (e.g., `.dmg`, `.exe`, `.deb`, `.AppImage`, `.apk`) on our [**GitHub Releases Page**](https://github.com/MarkinHaus/ToolBoxV2/releases).
Look for releases tagged with "-App" (e.g., `simple-vX.Y.Z-App`).

1.  Go to the [**Latest App Release on GitHub**](https://github.com/MarkinHaus/ToolBoxV2/releases/latest). (Note: Manually filter/find the latest release with "App" in its name if the "latest" tag doesn't point to an App release).
2.  Download the appropriate installer for your operating system from the "Assets" section (e.g., `simple-core_X.Y.Z_aarch64.dmg` for macOS ARM, `simple-core_X.Y.Z_x64-setup.exe` for Windows).
3.  Run the installer and follow the on-screen instructions.

You can also use our [**Interactive Web Installer Page**](https://simplecore.app/web/core0/Installer.html) which attempts to auto-detect your OS and provide the correct download link from the latest GitHub App release.

---

### 🖥️ Worker System Deployment (Production)

ToolBoxV2 uses a Python-based worker infrastructure instead of a Rust server. For production deployments:

#### Start Worker System

```bash
# Start all workers (HTTP + WebSocket)
cd toolboxv2/utils/workers
python cli_worker_manager.py start

# Or use the tb command
tb workers start
```

#### Worker Components

- **HTTP Workers**: Handle API requests via raw WSGI (~5000 req/s per worker)
- **WebSocket Workers**: Handle real-time connections (~10,000 concurrent per instance)
- **ZeroMQ Event Broker**: High-performance inter-worker communication

For full documentation, see [toolboxv2/utils/workers/README.md](toolboxv2/utils/workers/README.md).

---

### 🖥️ Full Stack Desktop/Web Application (Tauri + Web)

This setup includes the Python backend, worker system, and Tauri/Web frontend.

**Prerequisites:**
*   Python 3.10 or higher
*   [Node.js](https://nodejs.org/) (which includes npm)
*   Tauri CLI (installed via npm)

for execution details use [package.json](toolboxv2/package.json)
or run tb --help

---

## 🧪 CI/CD & Deployment

Automated processes are managed using GitHub Actions for:
*   🔁 **Build & Test**: Validating Python components and worker system.
*   🚀 **Release**: Publishing to PyPI, building Tauri applications, and potentially Docker images.

---

## 🐳 Docker

### Build ToolBoxV2 Image

Build the main ToolBoxV2 Docker image for deployment:

```bash
# Build as toolboxv2:latest
tb docker-image

# Build with specific tag
tb docker-image --tag v1.0.0

# Build and push to Docker Hub
tb docker-image --tag v1.0.0 --push

# Force rebuild without cache
tb docker-image --no-cache
```

### Run Container

```bash
# Run with default CLI interface
docker run -it --rm toolboxv2:latest

# Run with persistent data volume
docker run -it --rm -v tb_data:/data toolboxv2:latest

# Run with exposed ports (for web interfaces)
docker run -it --rm -p 8080:8080 -p 8501:8501 toolboxv2:latest
```

### Container Images

| Image | Description | Use Case |
|-------|-------------|----------|
| `toolboxv2:latest` | Main ToolBoxV2 image | Production deployments |
| `toolboxv2:dev` | Development build | Testing and development |
| Docksh | SSH-accessible containers | Remote CLI access with tmux persistence |

---

## 🌱 Example Projects & Ideas

ToolBoxV2 can be used to build a wide range of applications, including:
*   🔗 Link shortener
*   🧠 Live notes with versioning
*   🎨 Diffusion system for generating visual assets
*   📅 Calendar and scheduling tools
*   📝 Quote/Offer generation system
*   🎮 Multiplayer TicTacToe
*   🤖 Chat/Voice bots with P2P communication capabilities

---

## 📚 Learn More / Further Information

*   [📦 Current Installer (Web Demo/Entry)](https://simplecore.app/web/core0/Installer.html)
*   [📚 Documentation (WIP)](https://markinhaus.github.io/ToolBoxV2/)
*   [🐍 PyPI Package](https://pypi.org/project/ToolBoxV2)
*   [🐙 GitHub Repository](https://github.com/MarkinHaus/ToolBoxV2)
*   [⚙️ Worker System Documentation](toolboxv2/utils/workers/README.md)

---
## 📄 License

This project is distributed under a custom license. Please refer to the [LICENSE](./LICENSE) file in the repository for detailed terms and conditions.
---

© 2022–2025 Markin Hausmanns – All rights reserved.
