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

# Strict mode
set -eo pipefail

# -----------------------------------------------------------------------------
# METADATA & DEFAULTS
# -----------------------------------------------------------------------------
AUTHOR="Markin Hausmanns"
WEBPAGE="Simplecore.app"
SCRIPT_VERSION="2.1.0"
TOOLBOX_REPO="https://github.com/MarkinHaus/ToolBoxV2.git"
TOOLBOX_PYPI_NAME="ToolBoxV2"
TOOLBOX_REPO_NAME="ToolBoxV2"

# --- Configuration Defaults ---
DEFAULT_TB_VERSION="latest"
DEFAULT_INSTALL_SOURCE="pip"
DEFAULT_PKG_MANAGER="pip"
DEFAULT_PYTHON_VERSION_TARGET="3.11"
DEFAULT_ISAA_EXTRA="false"
DEFAULT_DEV_EXTRA="false"
DEFAULT_AUTO_INSTALL_DEPS="false" # New: Auto-install system dependencies
DEFAULT_INSTALL_DIR_BASE_LINUX_MAC="$HOME/.local/share"
DEFAULT_INSTALL_DIR_BASE_WINDOWS="$HOME/AppData/Local"
DEFAULT_TB_APP_NAME="ToolBoxV2"
DEFAULT_BIN_DIR_LINUX_MAC="$HOME/.local/bin"
DEFAULT_BIN_DIR_WINDOWS="$HOME/.local/bin"

# -----------------------------------------------------------------------------
# COLORS & EMOJIS
# -----------------------------------------------------------------------------
C_RESET='\033[0m'; C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[0;33m';
C_BLUE='\033[0;34m'; C_MAGENTA='\033[0;35m'; C_CYAN='\033[0;36m'; C_BOLD='\033[1m';
E_ROCKET="🚀"; E_PYTHON="🐍"; E_GEAR="⚙️"; E_CHECK="✅"; E_CROSS="❌"; E_INFO="ℹ️";
E_WARN="⚠️"; E_BOX="📦"; E_LINK="🔗"; E_PARTY="🎉"; E_GIT="🔧"; E_WINDOWS="🪟"; E_APPLE="🍎"; E_LINUX="🐧";
E_DOWNLOAD="📥";

# -----------------------------------------------------------------------------
# LOGGING FUNCTIONS
# -----------------------------------------------------------------------------
_log_base() { local color_code="$1"; shift; echo -e "${color_code}${*}${C_RESET}"; }
log_info() { _log_base "$C_BLUE" "${E_INFO} INFO: ${*}"; }
log_success() { _log_base "$C_GREEN" "${E_CHECK} SUCCESS: ${*}"; }
log_warning() { _log_base "$C_YELLOW" "${E_WARN} WARNING: ${*}"; }
log_error() { _log_base "$C_RED" "${E_CROSS} ERROR: ${*}"; exit 1; }
log_title() { echo -e "\n${C_MAGENTA}${C_BOLD}--- ${*} ---${C_RESET}"; }
log_debug() { [[ "$DEBUG" == "true" ]] && _log_base "$C_CYAN" "DEBUG: ${*}"; }

# -----------------------------------------------------------------------------
# OS DETECTION & PLATFORM SPECIFICS
# -----------------------------------------------------------------------------
OS_TYPE="Unknown"
PYTHON_EXEC_NAME="python3"
VENV_BIN_DIR_NAME="bin"
VENV_SCRIPTS_DIR_NAME="Scripts" # For Windows (unused if Python helper handles it)
USER_BIN_DIR=""
SYSTEM_PKG_INSTALLER_CMD="" # For auto-installing Python/Git
SUDO_CMD="sudo" # Assume sudo, can be empty if root or not needed

# Detect if running as root
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO_CMD="" # No sudo needed if already root
fi

case "$(uname -s)" in
    Linux*)
        OS_TYPE="Linux"
        USER_BIN_DIR="$DEFAULT_BIN_DIR_LINUX_MAC"
        if command_exists apt-get; then SYSTEM_PKG_INSTALLER_CMD="apt-get";
        elif command_exists yum; then SYSTEM_PKG_INSTALLER_CMD="yum";
        elif command_exists dnf; then SYSTEM_PKG_INSTALLER_CMD="dnf";
        elif command_exists pacman; then SYSTEM_PKG_INSTALLER_CMD="pacman";
        fi
        ;;
    Darwin*)
        OS_TYPE="Mac"
        USER_BIN_DIR="$DEFAULT_BIN_DIR_LINUX_MAC"
        if command_exists brew; then SYSTEM_PKG_INSTALLER_CMD="brew"; fi
        # On macOS, sudo might not be needed for brew if installed in user's homebrew dir
        # but system-wide changes might. For simplicity, keep sudo for now.
        ;;
    CYGWIN*|MINGW*|MSYS*|Windows_NT*)
        OS_TYPE="Windows"
        PYTHON_EXEC_NAME="python"
        USER_BIN_DIR="$DEFAULT_BIN_DIR_WINDOWS"
        if command_exists winget; then SYSTEM_PKG_INSTALLER_CMD="winget";
        elif command_exists choco; then SYSTEM_PKG_INSTALLER_CMD="choco";
        fi
        SUDO_CMD="" # Typically not used with winget/choco for user installs
        ;;
    *)
        log_warning "Unsupported OS: $(uname -s). Auto-installation of dependencies might not work."
        OS_TYPE="Linux" # Fallback for logic
        USER_BIN_DIR="$DEFAULT_BIN_DIR_LINUX_MAC"
        ;;
esac
log_debug "Detected OS_TYPE: $OS_TYPE, User bin dir: $USER_BIN_DIR, Python exec name hint: $PYTHON_EXEC_NAME"
log_debug "System Pkg Installer: $SYSTEM_PKG_INSTALLER_CMD, Sudo: $SUDO_CMD"

# -----------------------------------------------------------------------------
# PYTHON HELPER SCRIPT (Embedded)
# -----------------------------------------------------------------------------
# (Python helper script remains the same as in version 2.0.0)
# It's quite long, so I'll put a placeholder here.
# Ensure the PYTHON_HELPER_SCRIPT from the previous version is here.
read -r -d '' PYTHON_HELPER_SCRIPT <<EOF
import os
import sys
import shutil
import subprocess
import platform
import json
from pathlib import Path

def log_py(level, message):
    print(f"PY_{level.upper()}: {message}", file=sys.stderr if level == "error" else sys.stdout)

def get_platform_details():
    details = {
        "os_type": platform.system(), # Linux, Darwin, Windows
        "python_executable": sys.executable,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "venv_scripts_path_name": "Scripts" if platform.system() == "Windows" else "bin",
    }
    return details

def find_python_executable(target_version_str):
    target_major, target_minor = map(int, target_version_str.split('.'))

    if platform.system() == "Windows":
        # For py.exe, need to use -X.Y syntax
        try:
            res = subprocess.run(["py", f"-{target_version_str}", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, check=True, timeout=5)
            py_exec_path = res.stdout.strip()
            # Verify version again from the resolved path
            res_verify = subprocess.run([py_exec_path, "-c", f"import sys; assert sys.version_info >= ({target_major}, {target_minor}); print(f'{{sys.version_info.major}}.{{sys.version_info.minor}}')"], capture_output=True, text=True, check=True, timeout=5)
            if res_verify.stdout.strip().startswith(target_version_str):
                return py_exec_path
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass # py.exe method failed or timed out, try others

        # Fallback to checking common python names in PATH
        exec_names = [f"python{target_version_str}.exe", f"python{target_major}.exe", "python.exe"]
    else: # Linux/Mac
        exec_names = [f"python{target_version_str}", f"python{str(target_major)}.{str(target_minor)}", f"python{target_major}", "python3", "python"]

    for name in exec_names:
        py_exec = shutil.which(name)
        if py_exec:
            try:
                res = subprocess.run([py_exec, "-c", f"import sys; assert sys.version_info >= ({target_major}, {target_minor}); print(f'{{sys.version_info.major}}.{{sys.version_info.minor}}')"], capture_output=True, text=True, check=True, timeout=5)
                # We want at least the target version, but finding exact can be hard with auto-installs
                # For now, let's be strict on the *found* version matching the target string prefix.
                if res.stdout.strip().startswith(target_version_str):
                    return py_exec
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                log_py("debug", f"Python check for '{name}' (path: {py_exec}) failed or version mismatch: {e}. Output: {res.stdout.strip() if 'res' in locals() else 'N/A'}")
                continue
    return None


def create_venv(venv_dir, python_exec):
    try:
        subprocess.run([python_exec, "-m", "venv", venv_dir], check=True, capture_output=True, timeout=120)
        # Upgrade pip
        venv_python = Path(venv_dir) / get_platform_details()["venv_scripts_path_name"] / ("python.exe" if platform.system() == "Windows" else "python")
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, capture_output=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        log_py("error", f"Venv creation failed: {e.stderr.decode() if e.stderr else e.stdout.decode()}")
        return False
    except subprocess.TimeoutExpired:
        log_py("error", "Venv creation or pip upgrade timed out.")
        return False


def get_executable_path(venv_dir, exec_name):
    base_path = Path(venv_dir) / get_platform_details()["venv_scripts_path_name"]
    if platform.system() == "Windows":
        for suffix in [".exe", "", ".cmd", ".bat"]:
             p = base_path / f"{exec_name}{suffix}"
             if p.is_file() and os.access(p, os.X_OK): return str(p) # Added X_OK check
    else:
        p = base_path / exec_name
        if p.is_file() and os.access(p, os.X_OK): return str(p)
    return None

def create_user_executable_link(target_script_path, link_name, user_bin_dir_str):
    target = Path(target_script_path)
    user_bin_dir = Path(user_bin_dir_str)
    user_bin_dir.mkdir(parents=True, exist_ok=True)

    if platform.system() == "Windows":
        link_path_bat = user_bin_dir / f"{link_name}.bat"
        log_py("info", f"Creating Windows .bat wrapper: {link_path_bat} -> {target}")
        try:
            with open(link_path_bat, "w") as f:
                f.write(f'@echo off\n"{target}" %*\n')
            return str(link_path_bat)
        except IOError as e:
            log_py("error", f"Failed to create .bat wrapper: {e}")
    else: # Linux/Mac - create symlink
        link_path = user_bin_dir / link_name
        log_py("info", f"Creating symlink: {link_path} -> {target}")
        if link_path.exists() or link_path.is_symlink(): # Check if exists before unlinking
            try:
                link_path.unlink()
            except OSError as e:
                log_py("warning", f"Could not remove existing link/file at {link_path}: {e}. Attempting to overwrite.")
        try:
            os.symlink(target, link_path) # target must be absolute for robust symlinks generally
            os.chmod(link_path, 0o755)
            return str(link_path)
        except OSError as e:
            log_py("error", f"Failed to create symlink: {e}. You might need root/admin or fix permissions for {user_bin_dir}.")
    return None


if __name__ == "__main__":
    action = sys.argv[1]
    args = sys.argv[2:]
    result = {}
    try:
        if action == "get_platform_details":
            result = get_platform_details()
        elif action == "find_python":
            result["path"] = find_python_executable(args[0])
        elif action == "create_venv":
            result["success"] = create_venv(args[0], args[1])
        elif action == "get_executable_path":
            result["path"] = get_executable_path(args[0], args[1])
        elif action == "create_user_executable_link":
            result["path"] = create_user_executable_link(args[0], args[1], args[2])
        else:
            raise ValueError(f"Unknown Python helper action: {action}")
        print(json.dumps(result))
    except Exception as e:
        log_py("error", f"Python helper script failed (Action: {action}, Args: {args}): {type(e).__name__}: {e}")
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
EOF
# End of Python helper script placeholder

call_python_helper() {
    local action="$1"; shift
    local bootstrap_python_exec="$PYTHON_EXEC_NAME"
    if [[ -n "$PYTHON_EXEC_PATH" && -x "$PYTHON_EXEC_PATH" ]]; then # Prefer already validated Python
        bootstrap_python_exec="$PYTHON_EXEC_PATH"
    elif ! command -v "$bootstrap_python_exec" &>/dev/null; then
        if command -v python3 &>/dev/null; then bootstrap_python_exec="python3";
        elif command -v python &>/dev/null; then bootstrap_python_exec="python";
        else log_error "No suitable Python interpreter found to run the internal helper script."; fi
    fi
    log_debug "Calling Python helper: $bootstrap_python_exec -c \"\$PYTHON_HELPER_SCRIPT\" $action $*"
    local output
    output=$(echo "$PYTHON_HELPER_SCRIPT" | "$bootstrap_python_exec" - "$action" "$@" 2> >(tee /dev/stderr | grep -i "^PY_ERROR:" >&2) ) # Show PY_ERROR to stderr
    # Check for PY_ERROR in stderr might be too complex if python script also prints to stderr for debug
    # Rely on json error field for now.
    log_debug "Python helper raw output: $output"

    if echo "$output" | grep -q '"error":'; then
        # Error message already logged by python helper or the grep above
        log_warning "Python helper reported an error. See details above."
        return 1
    fi
    echo "$output"
}

get_json_value() {
    local key_to_extract="$1"
    # Using python for robust JSON parsing if available
    local bootstrap_python_exec="$PYTHON_EXEC_NAME"
    if [[ -n "$PYTHON_EXEC_PATH" && -x "$PYTHON_EXEC_PATH" ]]; then
        bootstrap_python_exec="$PYTHON_EXEC_PATH"
    elif ! command -v python3 &>/dev/null && command -v python &>/dev/null; then
        bootstrap_python_exec="python" # fallback if python3 not found
    fi

    if command_exists "$bootstrap_python_exec"; then
        echo "$JSON_STRING" | "$bootstrap_python_exec" -c "import sys, json; data = json.load(sys.stdin); print(data.get('$key_to_extract', ''))" 2>/dev/null
    else # Fallback to basic sed if no python for parsing
        echo "$JSON_STRING" | sed -n "s/.*\"${key_to_extract}\": \?\(\"[^\"]*\"\|[^,}]*\).*/\1/p" | tr -d '"'
    fi
}


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS (Bash side)
# -----------------------------------------------------------------------------
command_exists() { command -v "$1" &>/dev/null; }

prompt_yes_no() {
    local prompt_message="$1"; local default_answer="${2:-N}"; local REPLY
    while true; do
        read -p "$(echo -e "${C_YELLOW}❓ ${prompt_message} (${C_CYAN}y/N${C_RESET}${C_YELLOW}, default: $default_answer): ${C_RESET}")" -n 1 -r REPLY; echo
        REPLY=${REPLY:-$default_answer}
        case "$REPLY" in [Yy]) return 0 ;; [Nn]) return 1 ;; *) _log_base "$C_RED" "Please answer 'y' or 'n'." ;; esac
    done
}

# -----------------------------------------------------------------------------
# CONFIGURATION & ARGUMENT PARSING
# -----------------------------------------------------------------------------
TB_VERSION=""; INSTALL_SOURCE=""; PKG_MANAGER=""; PYTHON_VERSION_TARGET=""
ISAA_EXTRA=""; DEV_EXTRA=""; AUTO_INSTALL_DEPS=""
INSTALL_DIR_BASE=""; INSTALL_DIR=""; TOOLBOX_SRC_DIR=""
PYTHON_EXEC_PATH=""; VENV_PATH=""; ARGS_PROVIDED="false"

finalize_os_specific_defaults() {
    if [[ "$OS_TYPE" == "Windows" ]]; then INSTALL_DIR_BASE="$DEFAULT_INSTALL_DIR_BASE_WINDOWS"; else INSTALL_DIR_BASE="$DEFAULT_INSTALL_DIR_BASE_LINUX_MAC"; fi
    INSTALL_DIR="${INSTALL_DIR_BASE}/${DEFAULT_TB_APP_NAME}"
}

load_config_from_file() {
    local config_file="init.config"
    if [[ -f "$config_file" ]]; then
        log_info "Loading configuration from ${C_CYAN}$config_file${C_RESET}..."
        while IFS='=' read -r key value; do
            key=$(echo "$key" | xargs); value=$(echo "$value" | xargs); value="${value#\"}"; value="${value%\"}"; value="${value#\'}"; value="${value%\'}"
            case "$key" in
                TB_VERSION) TB_VERSION="$value" ;; INSTALL_SOURCE) INSTALL_SOURCE="$value" ;; PKG_MANAGER) PKG_MANAGER="$value" ;;
                PYTHON_VERSION_TARGET) PYTHON_VERSION_TARGET="$value" ;; ISAA_EXTRA) ISAA_EXTRA="$value" ;; DEV_EXTRA) DEV_EXTRA="$value" ;;
                AUTO_INSTALL_DEPS) AUTO_INSTALL_DEPS="$value" ;;
                *) log_warning "Unknown key in $config_file: $key" ;;
            esac
        done < <(grep -E '^[[:alnum:]_]+=' "$config_file")
    else log_info "No ${C_CYAN}$config_file${C_RESET} found. Using defaults or arguments."; fi
}

parse_arguments() {
    if [[ $# -gt 0 ]]; then
        log_info "Parsing command-line arguments..."; ARGS_PROVIDED="true"
        # Reset relevant defaults if any arg is passed, to ensure CLI overrides file/hardcoded
        ISAA_EXTRA="$DEFAULT_ISAA_EXTRA"; DEV_EXTRA="$DEFAULT_DEV_EXTRA"; AUTO_INSTALL_DEPS="$DEFAULT_AUTO_INSTALL_DEPS"

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --version=*) TB_VERSION="${1#*=}" ;; --version) TB_VERSION="$2"; shift ;;
                --source=*) INSTALL_SOURCE="${1#*=}" ;; --source) INSTALL_SOURCE="$2"; shift ;;
                --manager=*) PKG_MANAGER="${1#*=}" ;; --manager) PKG_MANAGER="$2"; shift ;;
                --python=*) PYTHON_VERSION_TARGET="${1#*=}" ;; --python) PYTHON_VERSION_TARGET="$2"; shift ;;
                --isaa) ISAA_EXTRA="true" ;; --dev) DEV_EXTRA="true" ;;
                --auto-install-deps) AUTO_INSTALL_DEPS="true" ;;
                --help|-h) show_help; exit 0 ;;
                *) log_error "Unknown argument: $1. Use --help for options." ;;
            esac; shift
        done
    fi
}

show_help() {
    # ... (show_help function as before, add --auto-install-deps) ...
    echo -e "  ${C_CYAN}--auto-install-deps${C_RESET} Attempt to automatically install missing system dependencies (Python, Git) using system package manager (e.g. winget, apt, brew). Default: ${DEFAULT_AUTO_INSTALL_DEPS}."
}

finalize_config() {
    finalize_os_specific_defaults
    TB_VERSION="${TB_VERSION:-$DEFAULT_TB_VERSION}"; INSTALL_SOURCE="${INSTALL_SOURCE:-$DEFAULT_INSTALL_SOURCE}"; PKG_MANAGER="${PKG_MANAGER:-$DEFAULT_PKG_MANAGER}"
    PYTHON_VERSION_TARGET="${PYTHON_VERSION_TARGET:-$DEFAULT_PYTHON_VERSION_TARGET}"; ISAA_EXTRA="${ISAA_EXTRA:-$DEFAULT_ISAA_EXTRA}"
    DEV_EXTRA="${DEV_EXTRA:-$DEFAULT_DEV_EXTRA}"; AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-$DEFAULT_AUTO_INSTALL_DEPS}"

    case "$INSTALL_SOURCE" in pip|git) ;; *) log_error "Invalid install source: '$INSTALL_SOURCE'."; esac
    case "$PKG_MANAGER" in pip|uv|poetry) ;; *) log_error "Invalid package manager: '$PKG_MANAGER'."; esac
    if [[ ! "$PYTHON_VERSION_TARGET" =~ ^3\.(9|[1-9][0-9])$ ]]; then log_error "Invalid Python version: '$PYTHON_VERSION_TARGET'."; fi

    VENV_PATH="$INSTALL_DIR/.venv"
    [[ "$INSTALL_SOURCE" == "git" ]] && TOOLBOX_SRC_DIR="$INSTALL_DIR/src/$TOOLBOX_REPO_NAME"

    log_title "${E_GEAR} Final Configuration"
    # ... (log configuration details as before, add AUTO_INSTALL_DEPS) ...
    log_info "${C_CYAN}Auto Install Deps:${C_RESET}  $AUTO_INSTALL_DEPS"
}

# -----------------------------------------------------------------------------
# AUTO-INSTALLER FUNCTIONS FOR SYSTEM DEPENDENCIES
# -----------------------------------------------------------------------------
attempt_install_python() {
    log_info "${E_DOWNLOAD} Attempting to install Python ${PYTHON_VERSION_TARGET} using ${SYSTEM_PKG_INSTALLER_CMD}..."
    local install_cmd_py=""
    local success=false
    # Define Python package names - these are best guesses and might need adjustment
    # For winget/choco, exact version might not be specifiable or syntax differs.
    # We aim for "a compatible Python 3".
    local python_pkg_name_generic="python3" # General
    local python_pkg_name_versioned="python${PYTHON_VERSION_TARGET}" # e.g. python3.11
    local python_pkg_name_venv_suffix="-venv" # e.g. python3.11-venv

    case "$SYSTEM_PKG_INSTALLER_CMD" in
        winget) install_cmd_py=("winget" "install" "--accept-source-agreements" "--accept-package-agreements" "-e" "--id" "Python.Python.3") ;; # Installs latest Python 3.x
        choco) install_cmd_py=("choco" "install" "python3" "-y") ;; # Installs latest Python 3.x
        apt-get) install_cmd_py=("$SUDO_CMD" "$SYSTEM_PKG_INSTALLER_CMD" "update" "-y" "&&" "$SUDO_CMD" "$SYSTEM_PKG_INSTALLER_CMD" "install" "-y" "$python_pkg_name_versioned" "${python_pkg_name_versioned}${python_pkg_name_venv_suffix}" "python3-pip") ;;
        yum|dnf) install_cmd_py=("$SUDO_CMD" "$SYSTEM_PKG_INSTALLER_CMD" "install" "-y" "$python_pkg_name_versioned" "python3-pip") ;; # python3-devel might be needed too
        pacman) install_cmd_py=("$SUDO_CMD" "$SYSTEM_PKG_INSTALLER_CMD" "-Syu" "--noconfirm" "$python_pkg_name_generic" "python-pip" "python-virtualenv") ;; # Arch usually has 'python' for latest 3.x
        brew) install_cmd_py=("$SYSTEM_PKG_INSTALLER_CMD" "install" "python@${PYTHON_VERSION_TARGET}") ;;
        *) log_warning "No known install command for Python with ${SYSTEM_PKG_INSTALLER_CMD}. Please install Python ${PYTHON_VERSION_TARGET} manually."; return 1 ;;
    esac

    log_info "Executing: ${C_YELLOW}${install_cmd_py[*]}${C_RESET}"
    if prompt_yes_no "Proceed with Python installation?" "Y"; then
        if [[ "${install_cmd_py[0]}" == "$SUDO_CMD" ]] || [[ "${install_cmd_py[0]}" == "cd" ]]; then # Handle complex commands with sudo or cd
             ( eval "${install_cmd_py[*]}" ) && success=true || success=false
        else
            "${install_cmd_py[@]}" && success=true || success=false
        fi

        if $success; then
            log_success "Python installation command executed. Verifying..."
            # Re-run Python check from helper
            local py_details_json; JSON_STRING=$(call_python_helper find_python "$PYTHON_VERSION_TARGET")
            PYTHON_EXEC_PATH=$(get_json_value path)
            if [[ -n "$PYTHON_EXEC_PATH" && "$PYTHON_EXEC_PATH" != "null" ]]; then
                log_success "Python ${PYTHON_VERSION_TARGET} (or compatible) successfully installed/found at: $PYTHON_EXEC_PATH"
                return 0
            else
                log_warning "Python installation command ran, but target version ${PYTHON_VERSION_TARGET} still not found. Manual installation might be required."
                return 1
            fi
        else
            log_warning "Python installation command failed. Please install Python ${PYTHON_VERSION_TARGET} manually."
            return 1
        fi
    else
        log_info "Python installation skipped by user."
        return 1
    fi
}

attempt_install_git() {
    log_info "${E_DOWNLOAD} Attempting to install Git using ${SYSTEM_PKG_INSTALLER_CMD}..."
    local install_cmd_git=""
    local success=false

    case "$SYSTEM_PKG_INSTALLER_CMD" in
        winget) install_cmd_git=("winget" "install" "--accept-source-agreements" "--accept-package-agreements" "-e" "--id" "Git.Git") ;;
        choco) install_cmd_git=("choco" "install" "git" "-y") ;;
        apt-get|yum|dnf) install_cmd_git=("$SUDO_CMD" "$SYSTEM_PKG_INSTALLER_CMD" "install" "-y" "git") ;;
        pacman) install_cmd_git=("$SUDO_CMD" "$SYSTEM_PKG_INSTALLER_CMD" "-Syu" "--noconfirm" "git") ;;
        brew) install_cmd_git=("$SYSTEM_PKG_INSTALLER_CMD" "install" "git") ;;
        *) log_warning "No known install command for Git with ${SYSTEM_PKG_INSTALLER_CMD}. Please install Git manually."; return 1 ;;
    esac

    log_info "Executing: ${C_YELLOW}${install_cmd_git[*]}${C_RESET}"
    if prompt_yes_no "Proceed with Git installation?" "Y"; then
        if [[ "${install_cmd_git[0]}" == "$SUDO_CMD" ]] || [[ "${install_cmd_git[0]}" == "cd" ]]; then
            ( eval "${install_cmd_git[*]}" ) && success=true || success=false
        else
            "${install_cmd_git[@]}" && success=true || success=false
        fi

        if $success; then
            log_success "Git installation command executed. Verifying..."
            if command_exists git; then
                log_success "Git successfully installed: $(git --version | head -n1)"
                return 0
            else
                log_warning "Git installation command ran, but 'git' command still not found. Manual installation might be required."
                return 1
            fi
        else
            log_warning "Git installation command failed. Please install Git manually."
            return 1
        fi
    else
        log_info "Git installation skipped by user."
        return 1
    fi
}

# -----------------------------------------------------------------------------
# CORE INSTALLATION STEPS (Modified for auto-install)
# -----------------------------------------------------------------------------
step_01_check_python() {
    log_title "${E_PYTHON} Step 1: Verifying Python ${PYTHON_VERSION_TARGET}"
    JSON_STRING=$(call_python_helper find_python "$PYTHON_VERSION_TARGET")
    PYTHON_EXEC_PATH=$(get_json_value path)

    if [[ -z "$PYTHON_EXEC_PATH" || "$PYTHON_EXEC_PATH" == "null" ]]; then
        log_warning "Python ${PYTHON_VERSION_TARGET} not found."
        if [[ "$AUTO_INSTALL_DEPS" == "true" && -n "$SYSTEM_PKG_INSTALLER_CMD" ]]; then
            if ! attempt_install_python; then
                 log_error "Automatic Python installation failed or was skipped. Please install Python ${PYTHON_VERSION_TARGET} and re-run."
            fi
            # PYTHON_EXEC_PATH should be updated by attempt_install_python if successful
        else
            log_error "Python ${PYTHON_VERSION_TARGET} not found. Auto-install is disabled or no system installer known. Please install Python and re-run."
            # ... (print manual install instructions as before) ...
        fi
    fi
    # One final check even after potential auto-install
    JSON_STRING=$(call_python_helper find_python "$PYTHON_VERSION_TARGET")
    PYTHON_EXEC_PATH=$(get_json_value path)
    if [[ -z "$PYTHON_EXEC_PATH" || "$PYTHON_EXEC_PATH" == "null" ]]; then
        log_error "Python ${PYTHON_VERSION_TARGET} still not usable after all checks/attempts. Aborting."
    fi
    log_success "Using Python: $PYTHON_EXEC_PATH (Version $PYTHON_VERSION_TARGET or compatible)"
}

step_02_check_git() {
    if [[ "$INSTALL_SOURCE" == "git" ]]; then
        log_title "${E_GIT} Step 2: Verifying Git"
        if ! command_exists git; then
            log_warning "Git not found (required for --source=git)."
            if [[ "$AUTO_INSTALL_DEPS" == "true" && -n "$SYSTEM_PKG_INSTALLER_CMD" ]]; then
                if ! attempt_install_git; then
                    log_error "Automatic Git installation failed or was skipped. Please install Git and re-run."
                fi
            else
                log_error "Git not found. Auto-install is disabled or no system installer known. Please install Git and re-run."
                 # ... (print manual install instructions as before) ...
            fi
        fi
        # Final check for Git
        if ! command_exists git; then
             log_error "Git still not usable after all checks/attempts. Aborting git source install."
        fi
        log_success "Git found: $(git --version | head -n1)"
    else
        log_info "Skipping Git check (not installing from git source)."
    fi
}

# --- Other steps (03_setup_environment to 07_finalize_installation) ---
# Remain largely the same as in version 2.0.0, as they depend on Python/Git being present.
# The Python helper handles venv creation and executable paths robustly.

step_03_setup_environment() {
    # ... (same as v2.0.0, ensure PYTHON_EXEC_PATH is used correctly for venv creation) ...
    log_title "${E_GEAR} Step 3: Setting up Installation Environment"
    if [[ -d "$INSTALL_DIR" ]]; then
        log_warning "Installation directory ${C_CYAN}$INSTALL_DIR${C_RESET} already exists."
        if [[ -d "$VENV_PATH" ]]; then
            if ! prompt_yes_no "A virtual environment already exists at ${C_CYAN}$VENV_PATH${C_RESET}. Remove and recreate it?" "Y"; then
                log_info "Using existing virtual environment. Make sure it's compatible."
                return
            fi
        fi
        if prompt_yes_no "Do you want to clean the existing directory ${C_CYAN}$INSTALL_DIR${C_RESET} before proceeding?" "Y"; then
             log_info "Removing existing directory: $INSTALL_DIR"; rm -rf "$INSTALL_DIR"
        elif [[ -d "$VENV_PATH" ]]; then
            log_info "Removing existing virtual environment: $VENV_PATH"; rm -rf "$VENV_PATH"
            [[ "$INSTALL_SOURCE" == "git" && -d "$TOOLBOX_SRC_DIR" ]] && rm -rf "$TOOLBOX_SRC_DIR"
        fi
    fi
    log_info "Creating application directory: $INSTALL_DIR"; mkdir -p "$INSTALL_DIR"
    log_info "Creating Python virtual environment in ${C_CYAN}$VENV_PATH${C_RESET} using $PYTHON_EXEC_PATH..."
    JSON_STRING=$(call_python_helper create_venv "$VENV_PATH" "$PYTHON_EXEC_PATH")
    if [[ $? -ne 0 ]] || [[ "$(get_json_value success)" != "True" ]]; then
        log_error "Virtual environment creation failed. Check logs from Python helper."
    fi
    log_success "Virtual environment created and pip upgraded."

    local venv_pip_exec_json; JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "pip")
    local venv_pip_exec=$(get_json_value path)
    if [[ -z "$venv_pip_exec" ]]; then log_error "Could not find pip in venv."; fi

    if [[ "$PKG_MANAGER" == "uv" ]]; then
        JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "uv"); local uv_path=$(get_json_value path)
        if [[ -z "$uv_path" ]]; then
            log_info "Installing UV into the virtual environment..."; "$venv_pip_exec" install uv
            JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "uv"); uv_path=$(get_json_value path)
            if [[ -z "$uv_path" ]]; then log_error "Failed to install UV into venv."; fi
            log_success "UV installed in venv."
        else log_success "UV already available in venv."; fi
    elif [[ "$PKG_MANAGER" == "poetry" ]]; then
        JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "poetry"); local poetry_path=$(get_json_value path)
        if [[ -z "$poetry_path" ]]; then
            log_info "Installing Poetry into the virtual environment..."; "$venv_pip_exec" install poetry
            JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "poetry"); poetry_path=$(get_json_value path)
            if [[ -z "$poetry_path" ]]; then log_error "Failed to install Poetry into venv."; fi
            log_success "Poetry installed in venv."
        else log_success "Poetry already available in venv."; fi
    fi
}

step_04_prepare_source_if_git() {
    # ... (same as v2.0.0) ...
    if [[ "$INSTALL_SOURCE" == "git" ]]; then
        log_title "${E_GIT} Step 4: Preparing Git Source ($TOOLBOX_REPO_NAME)"
        if [[ -d "$TOOLBOX_SRC_DIR" ]]; then
            log_info "Found existing source directory: $TOOLBOX_SRC_DIR"
            if ! (cd "$TOOLBOX_SRC_DIR" && git rev-parse --is-inside-work-tree &>/dev/null && git remote get-url origin | grep -qF "$TOOLBOX_REPO"); then
                log_warning "$TOOLBOX_SRC_DIR exists but is not a valid git repo for $TOOLBOX_REPO or remote URL mismatch."
                if prompt_yes_no "Remove and re-clone?" "Y"; then rm -rf "$TOOLBOX_SRC_DIR"; else
                    log_error "Cannot proceed with invalid source directory. Please fix or allow re-clone."; fi
            else
                 if prompt_yes_no "Update existing source (fetch & checkout ${C_CYAN}$TB_VERSION${C_RESET})?" "Y"; then
                    log_info "Updating existing source directory..."
                    ( cd "$TOOLBOX_SRC_DIR"; git fetch --all --prune --tags -q; log_info "Checking out version: ${C_CYAN}$TB_VERSION${C_RESET}..."; git checkout -q "$TB_VERSION"
                      if git symbolic-ref -q HEAD && [[ "$(git symbolic-ref -q --short HEAD)" == "$TB_VERSION" ]]; then
                        log_info "Pulling latest changes for branch ${C_CYAN}$TB_VERSION${C_RESET}..."; git pull -q origin "$TB_VERSION"; fi
                    ) || { log_warning "Failed to update existing source. Continuing with current state or will re-clone if empty."; }
                 else log_info "Skipping source update. Using existing local copy as is."; fi
            fi
        fi
        if [[ ! -d "$TOOLBOX_SRC_DIR" ]]; then
            log_info "Cloning ${C_CYAN}$TOOLBOX_REPO${C_RESET} into $TOOLBOX_SRC_DIR..."
            mkdir -p "$(dirname "$TOOLBOX_SRC_DIR")"
            local clone_branch_arg=""
            [[ "$TB_VERSION" != "latest" ]] && clone_branch_arg="--branch $TB_VERSION" # Git clone handles 'latest' by using default branch.
            if git clone $clone_branch_arg "$TOOLBOX_REPO" "$TOOLBOX_SRC_DIR"; then
                log_success "Repository cloned successfully (version: ${C_CYAN}$TB_VERSION${C_RESET})."
            else log_error "Failed to clone repository: $TOOLBOX_REPO (version: $TB_VERSION)"; fi
        fi
    fi
}
step_05_install_toolboxv2() {
    # ... (same as v2.0.0, ensure pkg_mgr_exec_path is correctly retrieved and used) ...
    log_title "${E_ROCKET} Step 5: Installing ToolBoxV2 ($TOOLBOX_PYPI_NAME)"
    local extras_list=(); [[ "$ISAA_EXTRA" == "true" ]] && extras_list+=("isaa"); [[ "$DEV_EXTRA" == "true" ]] && extras_list+=("dev")
    local extras_pip_format=""; local poetry_extras_args=()
    if (( ${#extras_list[@]} > 0 )); then
        extras_pip_format="[$(IFS=,; echo "${extras_list[*]}")]"
        for extra in "${extras_list[@]}"; do poetry_extras_args+=("--extras" "$extra"); done
    fi

    JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "$PKG_MANAGER")
    local pkg_mgr_exec_path=$(get_json_value path)
    if [[ -z "$pkg_mgr_exec_path" ]]; then log_error "Could not find $PKG_MANAGER executable in venv ${C_CYAN}$VENV_PATH${C_RESET}."; fi

    local install_cmd_array=(); local install_target_desc=""
    if [[ "$INSTALL_SOURCE" == "git" ]]; then
        install_target_desc="local source at $TOOLBOX_SRC_DIR"; local source_to_install_pip_uv="$TOOLBOX_SRC_DIR$extras_pip_format"
        case "$PKG_MANAGER" in
            pip) install_cmd_array=("$pkg_mgr_exec_path" "install" "--upgrade" "$source_to_install_pip_uv") ;;
            uv) install_cmd_array=("$pkg_mgr_exec_path" "pip" "install" "--system" "--upgrade" "$source_to_install_pip_uv") ;;
            poetry) log_info "Installing from git source using Poetry..."; install_cmd_array=("cd" "$TOOLBOX_SRC_DIR" "&&" "$pkg_mgr_exec_path" "install" "${poetry_extras_args[@]}") ;;
        esac
    else
        local package_spec="$TOOLBOX_PYPI_NAME"; [[ "$TB_VERSION" != "latest" ]] && package_spec+="==$TB_VERSION"
        local package_spec_with_extras="$package_spec$extras_pip_format"; install_target_desc="$package_spec_with_extras from PyPI"
        case "$PKG_MANAGER" in
            pip) install_cmd_array=("$pkg_mgr_exec_path" "install" "--upgrade" "$package_spec_with_extras") ;;
            uv) install_cmd_array=("$pkg_mgr_exec_path" "pip" "install" "--system" "--upgrade" "$package_spec_with_extras") ;;
            poetry) local poetry_context_dir="$INSTALL_DIR"
                if [[ ! -f "$poetry_context_dir/pyproject.toml" ]]; then
                    log_info "Creating dummy pyproject.toml for Poetry in ${C_CYAN}$poetry_context_dir${C_RESET}"
                    cat > "$poetry_context_dir/pyproject.toml" <<EOF
[tool.poetry]
name = "toolboxv2-runtime-env"; version = "0.1.0"; description = "Runtime environment for ToolBoxV2"; authors = ["ToolBoxV2 Installer"]; readme = "README.md"
[tool.poetry.dependencies]
python = "^${PYTHON_VERSION_TARGET}"
[build-system]
requires = ["poetry-core"]; build-backend = "poetry.core.masonry.api"
EOF
                    touch "$poetry_context_dir/README.md"
                fi
                install_cmd_array=("cd" "$poetry_context_dir" "&&" "$pkg_mgr_exec_path" "add" "$package_spec" "${poetry_extras_args[@]}") ;;
        esac
    fi
    log_info "Installing ${C_CYAN}$install_target_desc${C_RESET} using ${C_GREEN}$PKG_MANAGER${C_RESET}..."
    log_debug "Executing command array: ${install_cmd_array[*]}"
    if [[ "${install_cmd_array[0]}" == "cd" ]]; then ( eval "${install_cmd_array[*]}" ); else "${install_cmd_array[@]}"; fi
    if [[ $? -eq 0 ]]; then
        log_success "ToolBoxV2 ($TOOLBOX_PYPI_NAME) installed successfully."
        JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "pip"); local venv_pip_exec=$(get_json_value path)
        local installed_tb_version=$("$venv_pip_exec" show "$TOOLBOX_PYPI_NAME" 2>/dev/null | grep -i '^Version:' | awk '{print $2}')
        if [[ -n "$installed_tb_version" ]]; then echo "$installed_tb_version" > "$INSTALL_DIR/.toolbox_version"; log_info "Recorded ToolBoxV2 version ${C_CYAN}$installed_tb_version${C_RESET}.";
        else log_warning "Could not determine installed version of $TOOLBOX_PYPI_NAME."; fi
    else log_error "ToolBoxV2 installation failed. Check output above."; fi
}
step_06_create_tb_command() {
    # ... (same as v2.0.0, uses Python helper which is platform-aware) ...
    log_title "${E_LINK} Step 6: Making 'tb' Command Accessible"
    JSON_STRING=$(call_python_helper get_executable_path "$VENV_PATH" "tb")
    local tb_script_in_venv_path=$(get_json_value path)
    if [[ -z "$tb_script_in_venv_path" || "$tb_script_in_venv_path" == "null" ]]; then
        log_error "ToolBoxV2 'tb' executable not found in venv: ${C_CYAN}$VENV_PATH${C_RESET}. Installation might be incomplete."; return 1; fi
    log_info "'tb' script found at: ${C_CYAN}$tb_script_in_venv_path${C_RESET}"

    JSON_STRING=$(call_python_helper create_user_executable_link "$tb_script_in_venv_path" "tb" "$USER_BIN_DIR")
    local user_executable_path=$(get_json_value path)
    if [[ $? -ne 0 ]] || [[ -z "$user_executable_path" || "$user_executable_path" == "null" ]]; then
        log_warning "Python helper failed to create user executable link for 'tb', or path was empty."
        user_executable_path="" # Ensure it's empty if helper failed
    fi

    if [[ -n "$user_executable_path" ]]; then
        log_success "'tb' command link/wrapper created at: ${C_CYAN}$user_executable_path${C_RESET}"
        if [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
            log_warning "Directory ${C_CYAN}$USER_BIN_DIR${C_RESET} is not in your PATH."
            echo -e "  ${C_YELLOW}To use 'tb' command, add to PATH:${C_RESET}"
            if [[ "$OS_TYPE" == "Windows" ]]; then echo -e "  - Add ${C_CYAN}$USER_BIN_DIR${C_RESET} to User/System PATH variables (restart terminal)."; else
                echo -e "  - Add to shell config (~/.bashrc, ~/.zshrc): ${C_GREEN}export PATH=\"$USER_BIN_DIR:\$PATH\"${C_RESET}";
                echo -e "  - Then: ${C_GREEN}source ~/.your_shell_rc_file${C_RESET} or new terminal."; fi
        else log_success "Directory ${C_CYAN}$USER_BIN_DIR${C_RESET} is already in your PATH."; fi
    else
        log_warning "Could not automatically create a system-wide 'tb' command."
        log_info "Run ToolBoxV2: ${C_CYAN}$tb_script_in_venv_path${C_RESET}";
        log_info "Or activate venv: ${C_CYAN}source $VENV_PATH/${VENV_BIN_DIR_NAME}/activate${C_RESET} (then 'tb')"; fi
    TB_CMD_FOR_INIT="$tb_script_in_venv_path" # Safest direct path
    if command_exists tb && [[ -n "$user_executable_path" && ":$PATH:" == *":$USER_BIN_DIR:"* ]]; then TB_CMD_FOR_INIT="tb"; fi
}
step_07_finalize_installation() {
    # ... (same as v2.0.0) ...
    log_title "${E_PARTY} Step 7: Finalizing Installation"
    if [[ -z "$TB_CMD_FOR_INIT" ]]; then log_error "Path to 'tb' command for initialization is not set."; fi
    log_info "Running ToolBoxV2 initialization: ${C_CYAN}$TB_CMD_FOR_INIT -init main${C_RESET}"
    if "$TB_CMD_FOR_INIT" -init main; then log_success "ToolBoxV2 initialized successfully!"; else
        log_error "ToolBoxV2 initialization failed. Please try running '${C_CYAN}$TB_CMD_FOR_INIT -init main${C_RESET}' manually."; fi
}


# -----------------------------------------------------------------------------
# MAIN EXECUTION ORCHESTRATOR
# -----------------------------------------------------------------------------
main() {
    local os_icon="$E_GEAR"; [[ "$OS_TYPE" == "Windows" ]] && os_icon="$E_WINDOWS Bash"; [[ "$OS_TYPE" == "Linux" ]] && os_icon="$E_LINUX"; [[ "$OS_TYPE" == "Mac" ]] && os_icon="$E_APPLE"
    echo -e "${C_BOLD}${C_GREEN}Welcome to the ToolBoxV2 Zero Installer - v${SCRIPT_VERSION}! ${E_ROCKET}${C_RESET}"
    echo -e "Detected OS: ${C_CYAN}$OS_TYPE${C_RESET} ${os_icon}"

    finalize_os_specific_defaults
    parse_arguments "$@"
    [[ "$ARGS_PROVIDED" == "false" ]] && load_config_from_file
    finalize_config

    step_01_check_python
    step_02_check_git
    step_03_setup_environment
    step_04_prepare_source_if_git
    step_05_install_toolboxv2
    step_06_create_tb_command
    step_07_finalize_installation

    log_title "${E_PARTY} Installation Complete! ${E_PARTY}"
    echo -e "${C_GREEN}ToolBoxV2 Core should now be ready.${C_RESET}"
    echo -e "Try running ${C_CYAN}tb${C_RESET} in a new terminal."
    echo -e "If not found, ensure ${C_CYAN}$USER_BIN_DIR${C_RESET} is in PATH (see instructions above)."
    echo -e "Or activate: ${C_CYAN}source \"$VENV_PATH/$VENV_BIN_DIR_NAME/activate\"${C_RESET} then run ${C_CYAN}tb${C_RESET}."
    echo -e "Enjoy ToolBoxV2!"
}

# --- Script Entry Point ---
main "$@"
