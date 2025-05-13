# ToolBoxV2: Installation Guide

This guide provides instructions for installing ToolBoxV2, whether you need just the core Python library or the full-stack application including the Rust server and Tauri/Web frontend.

## 1. Installing the Core Python Library

This method is suitable if you primarily need to use ToolBoxV2 as a Python library within your own projects or want to develop Python-based modules for it.

### Option A: Stable Release from PyPI (Recommended)

This is the preferred method for installing the latest stable release of the ToolBoxV2 Python package.

1.  **Ensure you have Python and pip:**
    If you don't have Python and pip installed, this [Python installation guide](https://docs.python-guide.org/en/latest/starting/installation/) can help. We recommend Python 3.10 or newer.

2.  **Install ToolBoxV2:**
    Open your terminal or command prompt and run:
    ```bash
    pip install ToolBoxV2
    ```
    *Consider using a virtual environment to manage project dependencies:*
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv .venv
    # Activate it (Windows)
    # .venv\Scripts\activate
    # Activate it (macOS/Linux)
    # source .venv/bin/activate

    pip install ToolBoxV2
    ```

### Option B: From Source (Latest Development Version)

This method allows you to get the very latest code from the GitHub repository, which might include new features or changes not yet in a stable release.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MarkinHaus/ToolBoxV2.git
    cd ToolBoxV2
    ```

2.  **Install in Editable Mode:**
    This installs the package from your local clone, and any changes you make to the source code will be immediately reflected in your environment.
    *   **Using pip:**
        ```bash
        # Recommended: Activate a virtual environment first
        pip install -e .
        ```
    *   **Using `uv` (a fast Python package installer and resolver):**
        ```bash
        # Recommended: Activate a virtual environment first
        uv pip install -e .
        ```
    *   **Using the provided script (sets up environment):**
        This script creates a virtual environment and installs dependencies.
        ```bash
        chmod +x install_python_env.sh
        ./install_python_env.sh
        ```

### Option C: Directly from GitHub with pip

You can also install directly from the GitHub repository without cloning it first:
```bash
pip install git+https://github.com/MarkinHaus/ToolBoxV2.git
```

## 2. Installing the Full Stack Desktop/Web Application

This setup is for developers who want to run or develop the complete ToolBoxV2 application, including the Python backend, Rust server (Actix), and the Tauri-based desktop application or `tbjs` web frontend.

### Prerequisites

Ensure you have the following installed on your system:

*   **Python:** Version 3.10 or higher.
*   **Rust and Cargo:** Install from [rust-lang.org](https://www.rust-lang.org/tools/install).
*   **Node.js and npm/pnpm:** Install from [nodejs.org](https://nodejs.org/). We recommend `pnpm` for managing Node.js dependencies in this project.
    *   Install `pnpm` globally: `npm install -g pnpm`
*   **Tauri CLI:** Install using Cargo: `cargo install tauri-cli`

*Ensure the virtual environment created by the script (or one you created manually) is activated for the subsequent steps.*

3.  **Install Node.js Dependencies and Build Rust Components:**
    From the root of the `ToolBoxV2` directory:
    ```bash
    pnpm install  # Installs Node.js dependencies for tbjs and Tauri frontend
    ```
    The Rust backend (`src-core/`) and Tauri components are typically built as part of the `pnpm` scripts defined in `package.json`. If you need to build the Rust core manually:
    ```bash
    # (Usually not needed if using pnpm scripts)
    # cargo build --release --manifest-path src-core/Cargo.toml
    ```
    *the build step is Usually handled by the api flow*


### Running the Application in CLI
*   **Row python runner tb**
    ```bash
    tb -c {MOD_NAME} {FUCTION_NAME} {AGRGS} --kwargs name:value
    ```
*   **or run in ipython**
    ```bash
    tb --ipy
    ```
### Running the Application in Server mode for web and Desktop

Refer to the scripts in the `package.json` file for various ways to run and build the application. Common commands include:

*   **Web Development Mode (tbjs frontend with hot-reloading):**
    ```bash
    pnpm dev
    # or live
    ```
    This typically starts the Rust server and the web frontend development server.

*   **Tauri Desktop Application (Development Mode):**
    ```bash
    pnpm tauri dev
    ```
    This will build and run the Tauri desktop application with hot-reloading for the frontend.

*   **Build Tauri Desktop Application (Production):**
    ```bash
    pnpm tauri build # Or a custom script like `pnpm tauriB` if defined
    ```
    This creates a distributable binary of the desktop application.

For more specific build and run commands, please consult the `scripts` section in the `package.json` file located in the `ToolBoxV2` repository root or use the CLI help:
```bash
    tb --help
    # or
    python -m toolboxv2 --help
```

### developing tip use to activate all hooks
```bash
    bash .github/hooks/setup_hooks.sh
```

#### pre-commit hook

runs Ruff Bandit Safety versions and on <sum> in the commit msg auto summary of the changes
crates an report in local-reports

### ?????????

```bash
INSTALLER_URL="https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh"; (echo "Fetching installer script..." && curl -sSL -o installer.sh "$INSTALLER_URL" && echo "Creating default 'init.config'..." && cat <<EOL > init.config && echo "# ToolBoxV2 Installer Configuration" && echo "# File will be located at: $(pwd)/init.config" && echo "# Modify values below as needed before proceeding." && echo "# The installer (installer.sh) will use these if this file exists and no arguments are provided to it." && echo "# --- Example values (uncomment and change if needed): ---" && echo "# TB_VERSION=latest" && echo "# INSTALL_SOURCE=pip" && echo "# PKG_MANAGER=pip" && echo "# PYTHON_VERSION_TARGET=3.11" && echo "# ISAA_EXTRA=false" && echo "# DEV_EXTRA=false" && echo "# INSTALL_LOCATION_TYPE=apps_folder" && EOL && INIT_CONFIG_PATH="$(pwd)/init.config" && echo -e "\n\033[0;32m📄 Default 'init.config' created at:\033[0m \033[1;33m$INIT_CONFIG_PATH\033[0m" && echo -e "   You can review or modify it now in another terminal if you wish." && echo -e "   The main script (installer.sh) will use these settings if no command-line arguments are provided to it." && read -p "⏳ Press [Enter] to make the installer executable and run it..." REPLY && chmod +x installer.sh && echo "🚀 Running installer..." && ./installer.sh) || echo -e "\033[0;31m❌ An error occurred during the setup process. Please check messages above.\033[0m"
```
onliner installer
