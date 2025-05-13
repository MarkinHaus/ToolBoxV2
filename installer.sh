#!/usr/bin/env bash

echo "**************************************************************************"
echo "***████████╗*██████╗***██████╗**██╗*********██████╗***██████╗*██╗***██╗***"
echo "***╚══██╔══╝██╔═══██╗*██╔═══██╗*██║*********██╔══██╗*██╔═══██╗*╚██╗██╔╝***"
echo "******██║***██║***██║*██║***██║*██║*********██████╔╝*██║***██║**╚███╔╝****"
echo "******██║***██║***██║*██║***██║*██║*********██╔══██╗*██║***██║**██╔██╗****"
echo "******██║***╚██████╔╝*╚██████╔╝*███████╗****██████╔╝*╚██████╔╝*██╔╝*██╗***"
echo "******╚═╝****╚═════╝***╚═════╝**╚══════╝****╚═════╝***╚═════╝**╚═╝**╚═╝***"
echo "**************************************************************************"
echo "Zero the Hero - ToolBoxV2 Core Installer"

# -----------------------------------------------------------------------------
# METADATA
# -----------------------------------------------------------------------------
AUTHOR="Markin Hausmanns"
WEBPAGE="Simplecore.app"
SCRIPT_VERSION="1.0.0"
TOOLBOX_REPO="https://github.com/MarkinHaus/ToolBoxV2.git" # Placeholder
TOOLBOX_PYPI_NAME="ToolBoxV2" # Placeholder, ensure this is correct

# -----------------------------------------------------------------------------
# UTILITIES & CONFIGURATION
# -----------------------------------------------------------------------------
# Enable stricter error handling
set -eo pipefail

# Colors
C_RESET='\033[0m'
C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_BLUE='\033[0;34m'
C_MAGENTA='\033[0;35m'
C_CYAN='\033[0;36m'
C_BOLD='\033[1m'

# Emojis (ensure your terminal supports them)
E_ROCKET="🚀"
E_PYTHON="🐍"
E_GEAR="⚙️"
E_CHECK="✅"
E_CROSS="❌"
E_INFO="ℹ️"
E_WARN="⚠️"
E_BOX="📦"
E_LINK="🔗"
E_PARTY="🎉"

# Logging functions
log_info() { echo -e "${C_BLUE}${E_INFO} ${*}${C_RESET}"; }
log_success() { echo -e "${C_GREEN}${E_CHECK} ${*}${C_RESET}"; }
log_warning() { echo -e "${C_YELLOW}${E_WARN} ${*}${C_RESET}"; }
log_error() { echo -e "${C_RED}${E_CROSS} ERROR: ${*}${C_RESET}"; exit 1; }
log_title() { echo -e "\n${C_MAGENTA}${C_BOLD}--- ${*} ---${C_RESET}"; }

# Check if a command exists
command_exists() { command -v "$1" &>/dev/null; }

# Default values
DEFAULT_TB_VERSION="latest"
DEFAULT_INSTALL_SOURCE="pip" # git, pip
DEFAULT_PKG_MANAGER="uv"    # pip, uv, poetry
DEFAULT_ENV_MANAGER="venv"   # venv (native)
DEFAULT_PYTHON_VERSION_TARGET="3.11"
DEFAULT_ISAA_EXTRA="false"
DEFAULT_DEV_EXTRA="false"
DEFAULT_INSTALL_DIR_BASE="$HOME" # Base for storage, actual dir will be more specific
DEFAULT_INSTALL_LOCATION_TYPE="apps_folder" # pip_default, apps_folder

# Resolved configuration (will be populated)
TB_VERSION=""
INSTALL_SOURCE=""
PKG_MANAGER=""
ENV_MANAGER=""
PYTHON_VERSION_TARGET=""
ISAA_EXTRA=""
DEV_EXTRA=""
INSTALL_DIR="" # Actual full path for ToolBoxV2 installation
PYTHON_EXEC="" # Detected Python executable

# OS Detection
OS_TYPE=""
case "$(uname -s)" in
    Linux*)     OS_TYPE="Linux";;
    Darwin*)    OS_TYPE="Mac";;
    CYGWIN*|MINGW*|MSYS*) OS_TYPE="Windows_Bash";;
    *)          OS_TYPE="Unknown";;
esac

# -----------------------------------------------------------------------------
# CONFIGURATION LOADING AND ARGUMENT PARSING
# -----------------------------------------------------------------------------
load_config_from_file() {
    local config_file="init.config"
    if [[ -f "$config_file" ]]; then
        log_info "Loading configuration from ${C_CYAN}$config_file${C_RESET}..."
        # Sanitize and source the file
        # Ensure keys are valid and values are somewhat sane (basic protection)
        while IFS='=' read -r key value; do
            # Trim whitespace from key and value
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            # Remove potential quotes around value
            value="${value#\"}"
            value="${value%\"}"
            value="${value#\'}"
            value="${value%\'}"

            case "$key" in
                TB_VERSION) TB_VERSION="$value" ;;
                INSTALL_SOURCE) INSTALL_SOURCE="$value" ;;
                PKG_MANAGER) PKG_MANAGER="$value" ;;
                ENV_MANAGER) ENV_MANAGER="$value" ;; # Currently only venv
                PYTHON_VERSION_TARGET) PYTHON_VERSION_TARGET="$value" ;;
                ISAA_EXTRA) ISAA_EXTRA="$value" ;;
                DEV_EXTRA) DEV_EXTRA="$value" ;;
                INSTALL_LOCATION_TYPE) INSTALL_LOCATION_TYPE="$value" ;;
                *) log_warning "Unknown key in $config_file: $key" ;;
            esac
        done < <(grep -E '^[[:alnum:]_]+=' "$config_file") # Only process lines like KEY=VALUE
    else
        log_info "No ${C_CYAN}$config_file${C_RESET} found. Using defaults or arguments."
    fi
}

parse_arguments() {
    # If arguments are provided, they override everything
    if [[ $# -gt 0 ]]; then
        log_info "Parsing command-line arguments..."
        # Temporarily disable ISAA/DEV defaults if any arg is passed
        # This ensures that if user passes e.g. --version, ISAA isn't implicitly true
        # They have to explicitly ask for --isaa or --dev
        ISAA_EXTRA="false"
        DEV_EXTRA="false"

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --version=*) TB_VERSION="${1#*=}" ;;
                --version) TB_VERSION="$2"; shift ;;
                --source=*) INSTALL_SOURCE="${1#*=}" ;;
                --source) INSTALL_SOURCE="$2"; shift ;;
                --manager=*) PKG_MANAGER="${1#*=}" ;;
                --manager) PKG_MANAGER="$2"; shift ;;
                # --env is less critical as we only support venv for now
                # --env=*) ENV_MANAGER="${1#*=}" ;;
                # --env) ENV_MANAGER="$2"; shift ;;
                --python=*) PYTHON_VERSION_TARGET="${1#*=}" ;;
                --python) PYTHON_VERSION_TARGET="$2"; shift ;;
                --isaa) ISAA_EXTRA="true" ;;
                --dev) DEV_EXTRA="true" ;;
                --dir=*) INSTALL_LOCATION_TYPE="${1#*=}" ;; # apps_folder or pip_default
                --dir) INSTALL_LOCATION_TYPE="$2"; shift ;;
                --help|-h) show_help; exit 0 ;;
                *) log_error "Unknown argument: $1. Use --help for options." ;;
            esac
            shift
        done
        ARGS_PROVIDED="true"
    else
        ARGS_PROVIDED="false"
    fi
}

show_help() {
    echo -e "${C_BOLD}ToolBoxV2 Zero Installer${C_RESET}"
    echo -e "Version: $SCRIPT_VERSION"
    echo -e "Author: $AUTHOR"
    echo -e "Web: $WEBPAGE"
    echo -e "\nInstalls ToolBoxV2 Core and initializes it."
    echo -e "\n${C_YELLOW}USAGE:${C_RESET}"
    echo -e "  ./install_toolbox.sh [OPTIONS]"
    echo -e "\n${C_YELLOW}OPTIONS:${C_RESET}"
    echo -e "  ${C_CYAN}--version=<version>${C_RESET}   Specify ToolBoxV2 version (default: ${DEFAULT_TB_VERSION})."
    echo -e "  ${C_CYAN}--source=<src>${C_RESET}        Installation source: 'pip' or 'git' (default: ${DEFAULT_INSTALL_SOURCE})."
    echo -e "  ${C_CYAN}--manager=<mgr>${C_RESET}       Package manager: 'pip', 'uv', 'poetry' (default: ${DEFAULT_PKG_MANAGER})."
    echo -e "  ${C_CYAN}--python=<py_ver>${C_RESET}    Target Python version (e.g., 3.9, 3.10, 3.11) (default: ${DEFAULT_PYTHON_VERSION_TARGET})."
    echo -e "  ${C_CYAN}--isaa${C_RESET}                Install with 'isaa' extra (default: ${DEFAULT_ISAA_EXTRA})."
    echo -e "  ${C_CYAN}--dev${C_RESET}                 Install with 'dev' extra (default: ${DEFAULT_DEV_EXTRA})."
    echo -e "  ${C_CYAN}--dir=<type>${C_RESET}          Installation directory type: 'apps_folder' (e.g. ~/.local/share/ToolBoxV2) or 'pip_default' (uses pip's default behavior, may require sudo for global, not recommended for venv) (default: ${DEFAULT_INSTALL_LOCATION_TYPE})."
    echo -e "  ${C_CYAN}--help, -h${C_RESET}            Show this help message."
    echo -e "\nIf no arguments are provided, the script looks for an ${C_CYAN}init.config${C_RESET} file."
    echo -e "If the file doesn't exist, it uses hardcoded default values."
}

finalize_config() {
    # Apply defaults if values are still empty after arg parsing and file loading
    TB_VERSION="${TB_VERSION:-$DEFAULT_TB_VERSION}"
    INSTALL_SOURCE="${INSTALL_SOURCE:-$DEFAULT_INSTALL_SOURCE}"
    PKG_MANAGER="${PKG_MANAGER:-$DEFAULT_PKG_MANAGER}"
    ENV_MANAGER="${ENV_MANAGER:-$DEFAULT_ENV_MANAGER}" # Always venv for now
    PYTHON_VERSION_TARGET="${PYTHON_VERSION_TARGET:-$DEFAULT_PYTHON_VERSION_TARGET}"
    ISAA_EXTRA="${ISAA_EXTRA:-$DEFAULT_ISAA_EXTRA}"
    DEV_EXTRA="${DEV_EXTRA:-$DEFAULT_DEV_EXTRA}"
    INSTALL_LOCATION_TYPE="${INSTALL_LOCATION_TYPE:-$DEFAULT_INSTALL_LOCATION_TYPE}"

    # Validate choices
    case "$INSTALL_SOURCE" in
        pip|git) ;;
        *) log_error "Invalid install source: '$INSTALL_SOURCE'. Must be 'pip' or 'git'." ;;
    esac
    case "$PKG_MANAGER" in
        pip|uv|poetry) ;;
        *) log_error "Invalid package manager: '$PKG_MANAGER'. Must be 'pip', 'uv', or 'poetry'." ;;
    esac
    if [[ ! "$PYTHON_VERSION_TARGET" =~ ^3\.(9|[1-9][0-9])$ ]]; then # Allows 3.9, 3.10, 3.11 etc.
        log_error "Invalid Python version format: '$PYTHON_VERSION_TARGET'. Expected e.g., 3.9, 3.10, 3.11."
    fi

    # Determine installation directory
    if [[ "$INSTALL_LOCATION_TYPE" == "apps_folder" ]]; then
        if [[ "$OS_TYPE" == "Mac" ]]; then
            INSTALL_DIR="${DEFAULT_INSTALL_DIR_BASE}/Applications/ToolBoxV2"
        else # Linux, Windows_Bash
            INSTALL_DIR="${DEFAULT_INSTALL_DIR_BASE}/.local/share/ToolBoxV2"
        fi
    elif [[ "$INSTALL_LOCATION_TYPE" == "pip_default" ]]; then
        if [[ "$ENV_MANAGER" != "venv" ]]; then # Or if we were to allow non-venv global installs
            log_warning "Using 'pip_default' without a venv might install globally. This is generally not recommended."
            INSTALL_DIR="" # Let pip decide, or handle this scenario more explicitly if needed
        else
            # If venv is used, 'pip_default' means inside the venv. The venv itself needs a path.
            INSTALL_DIR="${DEFAULT_INSTALL_DIR_BASE}/.toolboxv2_env" # A default venv location
        fi
    else
        log_error "Invalid install location type: '$INSTALL_LOCATION_TYPE'."
    fi

    log_title "Installer Configuration"
    log_info "${C_CYAN}ToolBoxV2 Version:${C_RESET} $TB_VERSION"
    log_info "${C_CYAN}Install Source:${C_RESET}    $INSTALL_SOURCE"
    log_info "${C_CYAN}Package Manager:${C_RESET}   $PKG_MANAGER"
    log_info "${C_CYAN}Environment Manager:${C_RESET} $ENV_MANAGER"
    log_info "${C_CYAN}Target Python:${C_RESET}     $PYTHON_VERSION_TARGET"
    log_info "${C_CYAN}Install ISAA Extra:${C_RESET} $ISAA_EXTRA"
    log_info "${C_CYAN}Install DEV Extra:${C_RESET}  $DEV_EXTRA"
    if [[ -n "$INSTALL_DIR" ]]; then
        log_info "${C_CYAN}Install Directory:${C_RESET} $INSTALL_DIR"
    else
        log_info "${C_CYAN}Install Directory:${C_RESET} Pip default (system/user site-packages)"
    fi
    log_info "${C_CYAN}Operating System:${C_RESET}   $OS_TYPE"
}

# -----------------------------------------------------------------------------
# PREREQUISITE CHECKS & INSTALLATION
# -----------------------------------------------------------------------------
check_and_install_python() {
    log_title "${E_PYTHON} Python Setup"
    local py_major_minor="${PYTHON_VERSION_TARGET}" # e.g., 3.11
    local py_major="${py_major_minor%.*}" # e.g., 3
    local py_minor="${py_major_minor#*.}" # e.g., 11

    # Try specific version first, then generic python3
    if command_exists "python${py_major_minor}"; then
        PYTHON_EXEC="python${py_major_minor}"
    elif command_exists "python${py_major}.${py_minor}"; then # For versions like python3.11
         PYTHON_EXEC="python${py_major}.${py_minor}"
    elif command_exists "python${py_major}"; then
        PYTHON_EXEC="python${py_major}"
    elif command_exists "python"; then
        PYTHON_EXEC="python"
    fi

    if [[ -n "$PYTHON_EXEC" ]]; then
        current_version=$($PYTHON_EXEC -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_info "Found Python: $($PYTHON_EXEC --version 2>&1) (executable: $PYTHON_EXEC, version: $current_version)"
        if [[ "$current_version" == "$PYTHON_VERSION_TARGET" ]]; then
            log_success "Python $PYTHON_VERSION_TARGET is available."
            # Ensure pip for this python is available
            if ! $PYTHON_EXEC -m pip --version &>/dev/null; then
                log_warning "pip module not found for $PYTHON_EXEC. Attempting to ensure/install it."
                # This can be tricky. Python might be there but ensurepip might be needed or pythonX-pip package.
                # On debian/ubuntu: sudo apt install python3-pip python3-venv
                # For now, assume if Python is there, pip should be too, or user needs to fix this.
                # A more robust script might try `curl https://bootstrap.pypa.io/get-pip.py | $PYTHON_EXEC`
                log_error "Python $PYTHON_EXEC is present, but its 'pip' module is missing. Please install pip for $PYTHON_EXEC (e.g., 'sudo apt install python${PYTHON_VERSION_TARGET}-pip python${PYTHON_VERSION_TARGET}-venv' or use 'get-pip.py')."
            fi
            return
        else
            log_warning "Found Python $current_version, but target is $PYTHON_VERSION_TARGET."
            # Fall through to attempt installation or error out
        fi
    fi

    log_warning "Python $PYTHON_VERSION_TARGET not found or not the default."

    if [[ "$OS_TYPE" == "Linux" ]]; then
        if command_exists apt-get; then
            log_info "Attempting to install Python $PYTHON_VERSION_TARGET and venv using apt..."
            sudo apt-get update
            sudo apt-get install -y "python${PYTHON_VERSION_TARGET}" "python${PYTHON_VERSION_TARGET}-venv" "python${PYTHON_VERSION_TARGET}-pip" git # Also grab git here
        elif command_exists yum; then
            log_info "Attempting to install Python $PYTHON_VERSION_TARGET and venv using yum..."
            # Yum often has Python 3.x as python3, specific minors can be tricky (SCLs etc)
            # This is a simplified attempt; complex Python versioning on RHEL/CentOS might need manual setup
            sudo yum install -y "python3${py_minor}" "python3${py_minor}-devel" "python3${py_minor}-pip" git
        elif command_exists dnf; then
            log_info "Attempting to install Python $PYTHON_VERSION_TARGET and venv using dnf..."
            sudo dnf install -y "python${py_major_minor}" "python${py_major_minor}-devel" "python${py_major_minor}-pip" git
        elif command_exists pacman; then
            log_info "Attempting to install Python $PYTHON_VERSION_TARGET and venv using pacman..."
            sudo pacman -Syu --noconfirm "python${py_major_minor}" "python-pip" "python-virtualenv" git # Arch often has python for latest 3.x
        else
            log_error "Unsupported Linux package manager. Please install Python $PYTHON_VERSION_TARGET, its pip, venv, and git manually."
        fi
        # Re-check after install attempt
        if command_exists "python${PYTHON_VERSION_TARGET}"; then
             PYTHON_EXEC="python${PYTHON_VERSION_TARGET}"
             log_success "Python $PYTHON_VERSION_TARGET installed."
        else
             log_error "Failed to install Python $PYTHON_VERSION_TARGET. Please install it manually."
        fi
    elif [[ "$OS_TYPE" == "Mac" ]]; then
        if command_exists brew; then
            log_info "Attempting to install Python $PYTHON_VERSION_TARGET using Homebrew..."
            brew install "python@${PYTHON_VERSION_TARGET}" git
            # Homebrew Python might not be automatically linked or might be keg-only
            # Need to find its path e.g. /usr/local/opt/python@3.11/bin/python3.11
            # Or ensure it's in PATH. For simplicity, we'll assume user handles PATH or uses the direct path if needed.
            if command_exists "python${PYTHON_VERSION_TARGET}"; then
                PYTHON_EXEC="python${PYTHON_VERSION_TARGET}"
            else
                # Try Homebrew's opt path
                local brew_py_path
                brew_py_path=$(brew --prefix "python@${PYTHON_VERSION_TARGET}")/bin/"python${PYTHON_VERSION_TARGET}"
                if [[ -x "$brew_py_path" ]]; then
                    PYTHON_EXEC="$brew_py_path"
                    log_info "Using Python from $PYTHON_EXEC"
                else
                    log_error "Failed to find Homebrew Python $PYTHON_VERSION_TARGET executable after installation. Check your PATH or Homebrew setup."
                fi
            fi
            log_success "Python $PYTHON_VERSION_TARGET installed via Homebrew."
        else
            log_error "Homebrew not found. Please install Python $PYTHON_VERSION_TARGET and git manually (e.g., from python.org and Xcode Command Line Tools)."
        fi
    elif [[ "$OS_TYPE" == "Windows_Bash" ]]; then
        log_warning "Running in Git Bash / MSYS2 on Windows."
        log_warning "Please ensure Python $PYTHON_VERSION_TARGET (from python.org or Microsoft Store) and Git for Windows are installed and in your PATH."
        # Attempt to find python.exe
        if command_exists "python${PYTHON_VERSION_TARGET}.exe"; then
            PYTHON_EXEC="python${PYTHON_VERSION_TARGET}.exe"
        elif command_exists "python.exe"; then
             PYTHON_EXEC="python.exe" # Check version later
        else
            log_error "Python executable not found in PATH. Please install Python $PYTHON_VERSION_TARGET."
        fi
        current_version=$($PYTHON_EXEC -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ "$current_version" != "$PYTHON_VERSION_TARGET" ]]; then
            log_error "Found Python $current_version, but $PYTHON_VERSION_TARGET is required. Please install/configure the correct version."
        fi
        log_success "Assuming Python $PYTHON_VERSION_TARGET is configured for Windows."
    else
        log_error "Unsupported OS: $OS_TYPE. Please install Python $PYTHON_VERSION_TARGET, pip, venv, and git manually."
    fi

    # Final check for Python executable
    if ! command_exists "$PYTHON_EXEC"; then
        log_error "Python executable '$PYTHON_EXEC' for version $PYTHON_VERSION_TARGET could not be found or configured. Aborting."
    fi
    if ! $PYTHON_EXEC -m pip --version &>/dev/null; then
         log_error "Pip module for $PYTHON_EXEC is not available even after attempted installation. Aborting."
    fi
}

check_and_install_git() {
    if [[ "$INSTALL_SOURCE" == "git" ]]; then
        log_title "${E_GEAR} Git Setup"
        if command_exists git; then
            log_success "Git is already installed ($(git --version))."
        else
            log_warning "Git not found, required for '--source=git'."
            if [[ "$OS_TYPE" == "Linux" ]]; then
                if command_exists apt-get; then sudo apt-get install -y git
                elif command_exists yum; then sudo yum install -y git
                elif command_exists dnf; then sudo dnf install -y git
                elif command_exists pacman; then sudo pacman -Syu --noconfirm git
                else log_error "Cannot auto-install git. Please install it manually."; fi
            elif [[ "$OS_TYPE" == "Mac" ]]; then
                if command_exists brew; then brew install git
                else log_error "Cannot auto-install git via Homebrew. Install Xcode Command Line Tools or git manually."; fi
            elif [[ "$OS_TYPE" == "Windows_Bash" ]]; then
                log_error "Git for Windows is required. Please install it and ensure 'git' is in your PATH."
            else
                log_error "Cannot auto-install git for $OS_TYPE. Please install it manually."
            fi
            if command_exists git; then log_success "Git installed successfully."; else log_error "Git installation failed."; fi
        fi
    fi
}

check_and_install_pkg_managers() {
    # This assumes pip (from the Python we just configured) is the base installer for uv/poetry
    local VENV_PIP_EXEC="$INSTALL_DIR/.venv/bin/pip"
    if [[ ! -x "$VENV_PIP_EXEC" ]]; then
        # This case should ideally not happen if venv setup is correct
        # Use the global python's pip as a fallback to install into venv
        VENV_PIP_EXEC="$PYTHON_EXEC -m pip"
        log_warning "Venv pip not found directly, using $PYTHON_EXEC -m pip to install uv/poetry into venv if needed."
    fi


    if [[ "$PKG_MANAGER" == "uv" ]]; then
        log_title "${E_BOX} UV Setup"
        if ! "$INSTALL_DIR/.venv/bin/uv" --version &>/dev/null; then
            log_info "UV not found in venv. Installing UV using pip..."
            "$VENV_PIP_EXEC" install uv
            if ! "$INSTALL_DIR/.venv/bin/uv" --version &>/dev/null; then
                log_error "Failed to install UV."
            fi
            log_success "UV installed successfully in venv."
        else
            log_success "UV is already available in venv."
        fi
    elif [[ "$PKG_MANAGER" == "poetry" ]]; then
        log_title "${E_BOX} Poetry Setup"
        if ! "$INSTALL_DIR/.venv/bin/poetry" --version &>/dev/null; then
            log_info "Poetry not found in venv. Installing Poetry using pip..."
            # Note: Installing Poetry globally via pipx or get-poetry.py is usually recommended.
            # Installing into the venv like this makes it a local tool for this venv.
            "$VENV_PIP_EXEC" install poetry
            if ! "$INSTALL_DIR/.venv/bin/poetry" --version &>/dev/null; then
                log_error "Failed to install Poetry."
            fi
            log_success "Poetry installed successfully in venv."
        else
            log_success "Poetry is already available in venv."
        fi
    fi
}

# -----------------------------------------------------------------------------
# INSTALLATION PROCESS
# -----------------------------------------------------------------------------
setup_environment() {
    log_title "${E_GEAR} Environment Setup"
    if [[ -z "$INSTALL_DIR" ]]; then
        # This is for `INSTALL_LOCATION_TYPE="pip_default"` WITHOUT venv (e.g. global install)
        # This path is generally discouraged for user applications.
        log_info "Skipping dedicated environment setup (INSTALL_LOCATION_TYPE='pip_default' and no venv)."
        # In this mode, PKG_MANAGER_EXEC will be the global pip/uv/poetry
        # This needs more thought if we truly want to support non-venv global installs easily.
        # For now, the script is geared towards venv.
        return
    fi

    if [[ -d "$INSTALL_DIR/.venv" ]]; then
        log_info "Existing virtual environment found at $INSTALL_DIR/.venv."
        read -p "$(echo -e "${C_YELLOW}❓ Do you want to remove and recreate it? (y/N): ${C_RESET}")" -n 1 -r REPLY
        echo
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$INSTALL_DIR/.venv"
            # Also clear previous ToolBoxV2 install markers if any, to ensure fresh install logic
            rm -f "$INSTALL_DIR/.toolbox_version"
        else
            log_info "Using existing virtual environment."
            # We might want to check if it's compatible or just proceed.
            # For simplicity, we proceed. A more robust check might verify Python version in venv.
        fi
    fi

    if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
        log_info "Creating Python virtual environment in $INSTALL_DIR/.venv using $PYTHON_EXEC..."
        mkdir -p "$INSTALL_DIR"
        "$PYTHON_EXEC" -m venv "$INSTALL_DIR/.venv"
        log_success "Virtual environment created."
    fi

    # Activate (conceptually, for commands in this script)
    # For commands: $INSTALL_DIR/.venv/bin/python, $INSTALL_DIR/.venv/bin/pip
    log_info "Activating venv for subsequent commands."
    # No actual `source` needed if we call executables directly.

    # Upgrade pip in venv
    log_info "Upgrading pip, setuptools, and wheel in the virtual environment..."
    "$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
}

install_toolboxv2() {
    log_title "${E_ROCKET} Installing ToolBoxV2 Core"

    local venv_path="$INSTALL_DIR/.venv"
    local current_installed_version=""
    local version_file="$INSTALL_DIR/.toolbox_version" # For non-pip_default installs

    # Determine how to execute the package manager
    local PKG_EXEC_CMD=""
    case "$PKG_MANAGER" in
        pip) PKG_EXEC_CMD="$venv_path/bin/pip" ;;
        uv) PKG_EXEC_CMD="$venv_path/bin/uv pip" ;; # uv uses 'uv pip install'
        poetry) PKG_EXEC_CMD="$venv_path/bin/poetry add" ;; # poetry add for libraries
                # If ToolBoxV2 is an app, `poetry install` from a pyproject.toml would be different.
                # Assuming 'poetry add' behavior for adding it as a dependency in the venv.
    esac

    if [[ "$INSTALL_LOCATION_TYPE" == "pip_default" && -z "$INSTALL_DIR" ]]; then
        # Global/user site-packages install, check with global pip
        if $PYTHON_EXEC -m pip show "$TOOLBOX_PYPI_NAME" &>/dev/null; then
            current_installed_version=$($PYTHON_EXEC -m pip show "$TOOLBOX_PYPI_NAME" | grep Version | awk '{print $2}')
            log_info "ToolBoxV2 version ${C_CYAN}$current_installed_version${C_RESET} found in Python site-packages."
        fi
    elif [[ -f "$version_file" ]]; then # Check our marker file for venv installs
        current_installed_version=$(cat "$version_file")
        # Also verify with pip inside venv to be sure
        if "$venv_path/bin/pip" show "$TOOLBOX_PYPI_NAME" &>/dev/null; then
            actual_venv_version=$("$venv_path/bin/pip" show "$TOOLBOX_PYPI_NAME" | grep Version | awk '{print $2}')
            if [[ "$current_installed_version" == "$actual_venv_version" ]]; then
                 log_info "ToolBoxV2 version ${C_CYAN}$current_installed_version${C_RESET} found in $INSTALL_DIR."
            else
                log_warning "Version mismatch: marker file says $current_installed_version, venv pip says $actual_venv_version. Will proceed with specified version."
                current_installed_version="" # Force re-evaluation
            fi
        else
            log_warning "ToolBoxV2 marker file exists but not found by pip in venv. Re-installing."
            current_installed_version="" # Force re-evaluation
        fi
    fi

    if [[ -n "$current_installed_version" ]]; then
        if [[ "$TB_VERSION" == "latest" || "$TB_VERSION" == "$current_installed_version" ]]; then
            log_info "ToolBoxV2 version ${C_CYAN}$current_installed_version${C_RESET} is already installed and matches request."
            read -p "$(echo -e "${C_YELLOW}❓ Do you want to reinstall/update ToolBoxV2? (y/N): ${C_RESET}")" -n 1 -r REPLY
            echo
            if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
                log_info "Skipping ToolBoxV2 installation."
                # Ensure symlink exists even if skipping install
                if [[ "$INSTALL_LOCATION_TYPE" != "pip_default" || -n "$INSTALL_DIR" ]]; then
                    create_symlink "$venv_path/bin/tb"
                fi
                return # Skip to post-install
            fi
            log_info "Proceeding with reinstallation/update..."
        else # Different version requested
            log_info "Found version $current_installed_version, but $TB_VERSION is requested. Updating..."
        fi
    fi

    # Construct package string with extras and version
    local package_spec="$TOOLBOX_PYPI_NAME"
    local extras_string=""
    if [[ "$ISAA_EXTRA" == "true" ]]; then extras_string="isaa"; fi
    if [[ "$DEV_EXTRA" == "true" ]]; then
        if [[ -n "$extras_string" ]]; then extras_string+=","; fi
        extras_string+="dev"
    fi
    if [[ -n "$extras_string" ]]; then package_spec+="[${extras_string}]"; fi

    if [[ "$INSTALL_SOURCE" == "pip" ]]; then
        if [[ "$TB_VERSION" != "latest" ]]; then
            package_spec+="==$TB_VERSION"
        fi
    elif [[ "$INSTALL_SOURCE" == "git" ]]; then
        local git_url="$TOOLBOX_REPO"
        if [[ "$TB_VERSION" != "latest" ]]; then
            git_url+="@$TB_VERSION" # Assumes version is a tag or branch
        fi
        package_spec="git+$git_url#egg=$TOOLBOX_PYPI_NAME" # For extras, pip needs egg name
        if [[ -n "$extras_string" ]]; then
             package_spec="git+$git_url#egg=$TOOLBOX_PYPI_NAME[${extras_string}]"
        fi
         # For Poetry, git source is handled differently
        if [[ "$PKG_MANAGER" == "poetry" ]]; then
            # Poetry 'add' handles git URLs directly, but extras might be tricky.
            # This part may need specific adjustment if Poetry is the primary target for git installs.
            # For now, let's assume `pip` via Poetry for git sources if complexity arises.
            # `poetry add git+https...` is the command.
            package_spec="git+$git_url" # Poetry add can take extras with the package name, need to verify exact syntax
            # Poetry might require `poetry add --extras "isaa dev" <package_name_from_git>`
            # For simplicity, we'll assume ToolBoxV2's setup.py handles extras when installed from git.
            # This is the most common use case for pip:
            package_spec="git+${git_url}"
             if [[ -n "$extras_string" ]]; then
                package_spec="${TOOLBOX_PYPI_NAME}[${extras_string}] @ git+${git_url}" # PEP 508 URL
            fi
        fi

    fi

    log_info "Installing ${C_CYAN}$package_spec${C_RESET} using ${C_GREEN}$PKG_MANAGER${C_RESET}..."

    # For Poetry, if adding from git and need extras, the command is a bit more complex
    # e.g. poetry add --git <url> --branch <branch> <package_name_if_different_from_repo> --extras "extra1 extra2"
    # The current `package_spec` for git is more pip-centric.
    # If using Poetry to install from git with extras, this needs specific handling.
    # Let's assume for now poetry will install via pip for git sources if it is to run via PKG_EXEC_CMD.
    # Or, that ToolBoxV2's pyproject.toml (if used with Poetry) would define these.

    local install_cmd
    if [[ "$PKG_MANAGER" == "poetry" ]]; then
        # Poetry's 'add' command is different
        # It may not directly support the complex pip-style git+url[extras] string
        # If installing from git with poetry, it's typically:
        # poetry add <dependency_name> --git <git_url> [--branch <branch> | --tag <tag> | --rev <commit>] [--extras "extra1 extra2"]
        # This is too complex to map directly from the generic package_spec.
        # For now, if using Poetry with git source, we'll actually use pip within poetry's env for simplicity.
        if [[ "$INSTALL_SOURCE" == "git" ]]; then
            log_warning "Using pip via Poetry for Git source installation due to complexity with 'poetry add' for this scenario."
            install_cmd="$venv_path/bin/poetry run pip install --upgrade \"$package_spec\""
        else
            # For PyPI sources, 'poetry add' is fine
            # Construct extras for poetry
            local poetry_extras_arg=""
            if [[ "$ISAA_EXTRA" == "true" ]]; then poetry_extras_arg+=" --extras \"isaa\""; fi
            if [[ "$DEV_EXTRA" == "true" ]]; then poetry_extras_arg+=" --extras \"dev\""; fi # This will create two --extras flags which is fine for poetry 1.2+

            local poetry_pkg_name="$TOOLBOX_PYPI_NAME"
            if [[ "$TB_VERSION" != "latest" ]]; then
                 poetry_pkg_name+="==$TB_VERSION"
            fi
            install_cmd="$PKG_EXEC_CMD $poetry_extras_arg \"$poetry_pkg_name\""
        fi
    else
        install_cmd="$PKG_EXEC_CMD install --upgrade \"$package_spec\""
    fi


    log_info "Executing: ${C_YELLOW}$install_cmd${C_RESET}"
    if eval "$install_cmd"; then # Use eval carefully, ensure $install_cmd is safe
        log_success "ToolBoxV2 installed successfully."
        # Store version if installed in our managed dir
        if [[ -n "$INSTALL_DIR" && "$INSTALL_LOCATION_TYPE" != "pip_default" ]]; then
            local installed_tb_version
            installed_tb_version=$("$venv_path/bin/pip" show "$TOOLBOX_PYPI_NAME" | grep Version | awk '{print $2}')
            echo "$installed_tb_version" > "$version_file"
            log_info "Recorded ToolBoxV2 version ${C_CYAN}$installed_tb_version${C_RESET} to $version_file"
        fi

        if [[ "$INSTALL_LOCATION_TYPE" != "pip_default" || -n "$INSTALL_DIR" ]]; then
             create_symlink "$venv_path/bin/tb"
        else
            log_info "'tb' command should be available globally if Python scripts directory is in PATH."
        fi
    else
        log_error "ToolBoxV2 installation failed."
    fi
}

create_symlink() {
    local tb_executable_path="$1"
    local symlink_dir="$HOME/.local/bin"
    local symlink_path="$symlink_dir/tb"

    log_title "${E_LINK} Exposing 'tb' Command"

    if [[ ! -x "$tb_executable_path" ]]; then
        log_error "ToolBoxV2 executable 'tb' not found at $tb_executable_path. Cannot create symlink."
        return 1
    fi

    mkdir -p "$symlink_dir" # Ensure ~/.local/bin exists

    if [[ -L "$symlink_path" ]]; then # If it's already a symlink
        if [[ "$(readlink "$symlink_path")" == "$tb_executable_path" ]]; then
            log_success "'tb' command is already correctly symlinked: $symlink_path -> $tb_executable_path"
            ensure_path_configured "$symlink_dir"
            return 0
        else
            log_warning "Symlink $symlink_path exists but points elsewhere ($(readlink "$symlink_path")). Removing it."
            rm -f "$symlink_path"
        fi
    elif [[ -f "$symlink_path" ]]; then # If it's a file, not a symlink
        log_warning "$symlink_path exists and is not a symlink. Please remove it manually to proceed."
        return 1
    fi

    log_info "Creating symlink for 'tb' command: $symlink_path -> $tb_executable_path"
    ln -sfn "$tb_executable_path" "$symlink_path"
    log_success "'tb' command symlinked successfully."

    ensure_path_configured "$symlink_dir"
}

ensure_path_configured() {
    local dir_to_check="$1"
    if [[ ":$PATH:" != *":$dir_to_check:"* ]]; then
        log_warning "${C_YELLOW}Directory '$dir_to_check' is not in your PATH!${C_RESET}"
        echo -e "${C_YELLOW}To use the 'tb' command directly, add the following line to your shell configuration file (e.g., ~/.bashrc, ~/.zshrc, ~/.profile):${C_RESET}"
        echo -e "${C_CYAN}export PATH=\"\$HOME/.local/bin:\$PATH\"${C_RESET}"
        echo -e "${C_YELLOW}Then, source the file (e.g., 'source ~/.bashrc') or open a new terminal.${C_RESET}"
    else
        log_success "Directory '$dir_to_check' is in your PATH."
    fi
}

run_toolbox_init() {
    log_title "${E_PARTY} Finalizing Installation"
    local tb_cmd_path
    if [[ -n "$INSTALL_DIR" && "$INSTALL_LOCATION_TYPE" != "pip_default" ]]; then # Venv install
        tb_cmd_path="$INSTALL_DIR/.venv/bin/tb"
        if [[ -x "$HOME/.local/bin/tb" ]]; then # Prefer symlink if it exists and is configured
            if [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
                 tb_cmd_path="tb" # Use the symlinked command if PATH is set
            fi
        fi
    else # Global/user site-packages install
        tb_cmd_path="tb" # Assume it's in PATH
        # More robust: find it with `which tb` or `$PYTHON_EXEC -m site --user-base`/bin/tb
    fi

    if ! command_exists "$tb_cmd_path" && [[ "$tb_cmd_path" == "tb" ]]; then # if trying to use 'tb' directly and it's not found
        log_warning "Cannot find 'tb' in PATH. Attempting direct venv path."
        tb_cmd_path="$INSTALL_DIR/.venv/bin/tb" # Fallback for venv case
        if [[ ! -x "$tb_cmd_path" ]]; then
            log_error "ToolBoxV2 'tb' command not found. Initialization cannot proceed."
            return 1
        fi
    fi

    log_info "Running ToolBoxV2 initialization: ${C_CYAN}$tb_cmd_path -init${C_RESET}"
    # Use eval if tb_cmd_path might contain spaces or needs variable expansion, otherwise direct execution is safer.
    # For this case, direct execution should be fine.
    if "$tb_cmd_path" -init; then
        log_success "ToolBoxV2 initialized successfully!"
    else
        log_error "ToolBoxV2 initialization failed. Please run '$tb_cmd_path -init' manually."
    fi
}

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
main() {
    echo -e "${C_BOLD}${C_GREEN}Welcome to the ToolBoxV2 Zero Installer! ${E_ROCKET}${C_RESET}"
    echo -e "Let's get your ToolBoxV2 Core up and running."

    # Initial argument parsing (sets ARGS_PROVIDED)
    parse_arguments "$@"

    # Load from config file only if no args were provided
    if [[ "$ARGS_PROVIDED" == "false" ]]; then
        load_config_from_file
    fi

    # Finalize config (apply defaults if needed, validate)
    finalize_config

    # Prerequisite checks
    check_and_install_python
    check_and_install_git # Only if INSTALL_SOURCE=git

    # Setup Python environment (venv)
    # This is skipped if INSTALL_LOCATION_TYPE="pip_default" AND we decide not to force a venv for it
    if [[ "$INSTALL_LOCATION_TYPE" == "apps_folder" || ( "$INSTALL_LOCATION_TYPE" == "pip_default" && -n "$INSTALL_DIR" ) ]]; then
        setup_environment
        check_and_install_pkg_managers # Install uv/poetry inside the venv if selected
    else
        log_info "Skipping venv creation for 'pip_default' global/user install mode."
        # If uv/poetry selected for global, they must be pre-installed globally.
        if [[ "$PKG_MANAGER" == "uv" ]] && ! command_exists uv; then log_error "UV selected but not found globally."; fi
        if [[ "$PKG_MANAGER" == "poetry" ]] && ! command_exists poetry; then log_error "Poetry selected but not found globally."; fi
    fi


    # Install ToolBoxV2
    install_toolboxv2

    # Run post-installation
    run_toolbox_init

    log_title "${E_PARTY} Installation Complete! ${E_PARTY}"
    echo -e "${C_GREEN}ToolBoxV2 Core should now be ready.${C_RESET}"
    echo -e "You can typically run it using the ${C_CYAN}tb${C_RESET} command."
    echo -e "If you encounter issues, ensure ${C_CYAN}\$HOME/.local/bin${C_RESET} is in your PATH and you've opened a new terminal session."
    echo -e "Enjoy using ToolBoxV2!"
}

# --- Entry Point ---
# Handle SIGHUP, SIGINT, SIGQUIT, SIGTERM for cleanup if needed in future
# trap cleanup_function SIGHUP SIGINT SIGQUIT SIGTERM
main "$@"
