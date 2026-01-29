#!/usr/bin/env pwsh

<#
    .SYNOPSIS
    ToolBoxV2 Zero Installer - PowerShell Version
    .DESCRIPTION
    Installs ToolBoxV2 Core, sets up the virtual environment, and configures the 'tb' command.
    .PARAMETER Version
    Specify ToolBoxV2 version (default: latest).
    .PARAMETER Source
    Installation source: 'pip' or 'git' (default: pip).
    .PARAMETER Manager
    Package manager: 'pip', 'uv', 'poetry' (default: pip).
    .PARAMETER Python
    Target Python version (default: 3.11).
    .PARAMETER Isaa
    Install with 'isaa' extra.
    .PARAMETER Dev
    Install with 'dev' extra.
    .PARAMETER AutoInstallDeps
    Attempt to auto-install system dependencies (Python, Git).
#>

param(
    [string]$Version = "",
    [string]$Source = "",
    [string]$Manager = "",
    [string]$Python = "",
    [switch]$Isaa,
    [switch]$Dev,
    [switch]$AutoInstallDeps
)

# -----------------------------------------------------------------------------
# METADATA & DEFAULTS
# -----------------------------------------------------------------------------
 $AUTHOR = "Markin Hausmanns"
 $WEBPAGE = "Simplecore.app"
 $SCRIPT_VERSION = "2.1.0"
 $TOOLBOX_REPO = "https://github.com/MarkinHaus/ToolBoxV2.git"
 $TOOLBOX_PYPI_NAME = "ToolBoxV2"
 $TOOLBOX_REPO_NAME = "ToolBoxV2"

# --- Configuration Defaults ---
 $DEFAULT_TB_VERSION = "latest"
 $DEFAULT_INSTALL_SOURCE = "pip"
 $DEFAULT_PKG_MANAGER = "pip"
 $DEFAULT_PYTHON_VERSION_TARGET = "3.11"
 $DEFAULT_ISAA_EXTRA = "false"
 $DEFAULT_DEV_EXTRA = "false"
 $DEFAULT_AUTO_INSTALL_DEPS = "false"
 $DEFAULT_INSTALL_DIR_BASE_LINUX_MAC = "$HOME/.local/share"
 $DEFAULT_INSTALL_DIR_BASE_WINDOWS = "$env:LOCALAPPDATA" # Uses AppData/Local on Windows
 $DEFAULT_TB_APP_NAME = "ToolBoxV2"
 $DEFAULT_BIN_DIR_LINUX_MAC = "$HOME/.local/bin"
# On Windows, we usually put a .bat wrapper in a folder that is in PATH, or add it.
# The script tries to detect a suitable bin dir. For Windows users, creating a folder in USERPROFILE and adding it to PATH is common.
 $DEFAULT_BIN_DIR_WINDOWS = "$HOME/.local/bin"

# -----------------------------------------------------------------------------
# DISPLAY HEADER
# -----------------------------------------------------------------------------
Write-Host "**************************************************************************" -ForegroundColor Magenta
Write-Host "***████████╗*██████╗***██████╗**██╗*********██████╗***██████╗*██╗***██╗***" -ForegroundColor Magenta
Write-Host "***╚══██╔══╝██╔═══██╗*██╔═══██╗*██║*********██╔══██╗*██╔═══██╗*╚██╗██╔╝***" -ForegroundColor Magenta
Write-Host "******██║***██║***██║*██║***██║*██║*********██████╔╝*██║***██║**╚███╔╝****" -ForegroundColor Magenta
Write-Host "******██║***██║***██║*██║***██║*██║*********██╔══██╗*██║***██║**██╔██╗****" -ForegroundColor Magenta
Write-Host "******██║***╚██████╔╝*╚██████╔╝*███████╗****██████╔╝*╚██████╔╝*██╔╝*██╗***" -ForegroundColor Magenta
Write-Host "******╚═╝****╚═════╝***╚═════╝**╚══════╝****╚═════╝***╚═════╝**╚═╝**╚═╝***" -ForegroundColor Magenta
Write-Host "**************************************************************************" -ForegroundColor Magenta
Write-Host "Zero the Hero - ToolBoxV2 Core Installer" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# LOGGING FUNCTIONS
# -----------------------------------------------------------------------------
function Log-Info {
    param([string]$Message)
    Write-Host "ℹ️  INFO: $Message" -ForegroundColor Cyan
}

function Log-Success {
    param([string]$Message)
    Write-Host "✅ SUCCESS: $Message" -ForegroundColor Green
}

function Log-Warning {
    param([string]$Message)
    Write-Host "⚠️  WARNING: $Message" -ForegroundColor Yellow
}

function Log-Error {
    param([string]$Message)
    Write-Host "❌ ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Log-Title {
    param([string]$Message)
    Write-Host "`n--- $Message ---" -ForegroundColor Magenta -BackgroundColor Black
}

function Log-Debug {
    param([string]$Message)
    if ($env:DEBUG -eq "true") {
        Write-Host "DEBUG: $Message" -ForegroundColor DarkCyan
    }
}

# -----------------------------------------------------------------------------
# OS DETECTION & PLATFORM SPECIFICS
# -----------------------------------------------------------------------------
 $OS_TYPE = "Unknown"
 $PYTHON_EXEC_NAME = "python3"
 $USER_BIN_DIR = ""
 $SYSTEM_PKG_INSTALLER_CMD = ""
 $IS_WINDOWS = $false
 $IS_LINUX = $false
 $IS_MAC = $false

if ($IsWindows) { $OS_TYPE = "Windows"; $IS_WINDOWS = $true }
elseif ($IsLinux) { $OS_TYPE = "Linux"; $IS_LINUX = $true }
elseif ($IsMacOS) { $OS_TYPE = "Mac"; $IS_MAC = $true }
else {
    # Fallback for older PowerShell versions or environments where $IsWindows isn't set
    if ($env:OS -eq "Windows_NT") { $OS_TYPE = "Windows"; $IS_WINDOWS = $true }
}

# Setup specific paths and commands based on OS
if ($IS_WINDOWS) {
    $PYTHON_EXEC_NAME = "python"
    $USER_BIN_DIR = $DEFAULT_BIN_DIR_WINDOWS
    if (Get-Command winget -ErrorAction SilentlyContinue) { $SYSTEM_PKG_INSTALLER_CMD = "winget" }
    elseif (Get-Command choco -ErrorAction SilentlyContinue) { $SYSTEM_PKG_INSTALLER_CMD = "choco" }
} else {
    $USER_BIN_DIR = $DEFAULT_BIN_DIR_LINUX_MAC
    if (Get-Command apt-get -ErrorAction SilentlyContinue) { $SYSTEM_PKG_INSTALLER_CMD = "apt-get" }
    elseif (Get-Command yum -ErrorAction SilentlyContinue) { $SYSTEM_PKG_INSTALLER_CMD = "yum" }
    elseif (Get-Command dnf -ErrorAction SilentlyContinue) { $SYSTEM_PKG_INSTALLER_CMD = "dnf" }
    elseif (Get-Command pacman -ErrorAction SilentlyContinue) { $SYSTEM_PKG_INSTALLER_CMD = "pacman" }
    elseif (Get-Command brew -ErrorAction SilentlyContinue) { $SYSTEM_PKG_INSTALLER_CMD = "brew" }
}

Log-Debug "Detected OS: $OS_TYPE, User Bin: $USER_BIN_DIR, Pkg Installer: $SYSTEM_PKG_INSTALLER_CMD"

# -----------------------------------------------------------------------------
# PYTHON HELPER SCRIPT (Embedded)
# -----------------------------------------------------------------------------
 $PYTHON_HELPER_SCRIPT = @"
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
        "os_type": platform.system(),
        "python_executable": sys.executable,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "venv_scripts_path_name": "Scripts" if platform.system() == "Windows" else "bin",
    }
    return details

def find_python_executable(target_version_str):
    target_major, target_minor = map(int, target_version_str.split('.'))
    if platform.system() == "Windows":
        try:
            res = subprocess.run(["py", f"-{target_version_str}", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, check=True, timeout=5)
            py_exec_path = res.stdout.strip()
            res_verify = subprocess.run([py_exec_path, "-c", f"import sys; assert sys.version_info >= ({target_major}, {target_minor}); print(f'{{sys.version_info.major}}.{{sys.version_info.minor}}')"], capture_output=True, text=True, check=True, timeout=5)
            if res_verify.stdout.strip().startswith(target_version_str):
                return py_exec_path
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        exec_names = [f"python{target_version_str}.exe", f"python{target_major}.exe", "python.exe"]
    else:
        exec_names = [f"python{target_version_str}", f"python{str(target_major)}.{str(target_minor)}", f"python{target_major}", "python3", "python"]

    for name in exec_names:
        py_exec = shutil.which(name)
        if py_exec:
            try:
                res = subprocess.run([py_exec, "-c", f"import sys; assert sys.version_info >= ({target_major}, {target_minor}); print(f'{{sys.version_info.major}}.{{sys.version_info.minor}}')"], capture_output=True, text=True, check=True, timeout=5)
                if res.stdout.strip().startswith(target_version_str):
                    return py_exec
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                continue
    return None

def create_venv(venv_dir, python_exec):
    try:
        subprocess.run([python_exec, "-m", "venv", venv_dir], check=True, capture_output=True, timeout=120)
        venv_python = Path(venv_dir) / get_platform_details()["venv_scripts_path_name"] / ("python.exe" if platform.system() == "Windows" else "python")
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, capture_output=True, timeout=120)
        return True
    except Exception as e:
        log_py("error", f"Venv creation failed: {e}")
        return False

def get_executable_path(venv_dir, exec_name):
    base_path = Path(venv_dir) / get_platform_details()["venv_scripts_path_name"]
    if platform.system() == "Windows":
        for suffix in [".exe", "", ".cmd", ".bat"]:
             p = base_path / f"{exec_name}{suffix}"
             if p.is_file() and os.access(p, os.X_OK): return str(p)
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
    else:
        link_path = user_bin_dir / link_name
        log_py("info", f"Creating symlink: {link_path} -> {target}")
        if link_path.exists() or link_path.is_symlink():
            try:
                link_path.unlink()
            except OSError: pass
        try:
            os.symlink(target, link_path)
            os.chmod(link_path, 0o755)
            return str(link_path)
        except OSError as e:
            log_py("error", f"Failed to create symlink: {e}.")
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
        log_py("error", f"Helper failed: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
"@

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

function Invoke-PythonHelper {
    param(
        [string]$Action,
        [string[]]$Arguments
    )

    $bootstrapPy = $PYTHON_EXEC_NAME
    # Try to find a valid python if the default name isn't found
    if (-not (Get-Command $bootstrapPy -ErrorAction SilentlyContinue)) {
        if (Get-Command python3 -ErrorAction SilentlyContinue) { $bootstrapPy = "python3" }
        elseif (Get-Command python -ErrorAction SilentlyContinue) { $bootstrapPy = "python" }
        else { Log-Error "No suitable Python interpreter found to run helper script." }
    }

    # Create temp file for the script
    $tempScriptPath = Join-Path $env:TEMP "toolbox_helper_py.py"
    Set-Content -Path $tempScriptPath -Value $PYTHON_HELPER_SCRIPT -Encoding UTF8

    try {
        $argList = @($tempScriptPath, $Action) + $Arguments
        $output = & $bootstrapPy @argList 2>&1
        # Filter out pure stderr lines if necessary, but helper prints JSON to stdout
        $jsonOutput = $output | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }

        # Convert JSON
        $result = $jsonOutput | ConvertFrom-Json

        if ($result.error) {
            Log-Warning "Python helper reported an error: $($result.error)"
            return $null
        }
        return $result
    }
    catch {
        Log-Error "Failed to execute Python helper: $_"
    }
    finally {
        if (Test-Path $tempScriptPath) { Remove-Item $tempScriptPath -Force }
    }
}

function Test-Command {
    param([string]$Cmd)
    return [bool](Get-Command $Cmd -ErrorAction SilentlyContinue)
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [string]$Default = "N"
    )
    $choice = ""
    while ($choice -notmatch "^[YN]$") {
        $promptStr = "❓ ${Prompt} (Y/N, default: $Default): "
        $inputRaw = Read-Host $promptStr
        if ([string]::IsNullOrWhiteSpace($inputRaw)) { $choice = $Default.ToUpper() }
        else { $choice = $inputRaw.ToUpper().Substring(0,1) }
    }
    return $choice -eq "Y"
}

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
 $TB_VERSION = $DEFAULT_TB_VERSION
 $INSTALL_SOURCE = $DEFAULT_INSTALL_SOURCE
 $PKG_MANAGER = $DEFAULT_PKG_MANAGER
 $PYTHON_VERSION_TARGET = $DEFAULT_PYTHON_VERSION_TARGET
 $ISAA_EXTRA = $DEFAULT_ISAA_EXTRA
 $DEV_EXTRA = $DEFAULT_DEV_EXTRA
 $AUTO_INSTALL_DEPS = $DEFAULT_AUTO_INSTALL_DEPS
 $INSTALL_DIR = ""
 $TOOLBOX_SRC_DIR = ""
 $PYTHON_EXEC_PATH = ""
 $VENV_PATH = ""

function Initialize-Defaults {
    if ($IS_WINDOWS) {
        $global:INSTALL_DIR_BASE = $DEFAULT_INSTALL_DIR_BASE_WINDOWS
    } else {
        $global:INSTALL_DIR_BASE = $DEFAULT_INSTALL_DIR_LINUX_MAC
    }
    $global:INSTALL_DIR = Join-Path $INSTALL_DIR_BASE $DEFAULT_TB_APP_NAME
    $global:VENV_PATH = Join-Path $INSTALL_DIR ".venv"
    if ($INSTALL_SOURCE -eq "git") {
        $global:TOOLBOX_SRC_DIR = Join-Path $INSTALL_DIR "src\$TOOLBOX_REPO_NAME"
    }
}

function Load-ConfigFile {
    $configPath = "init.config"
    if (Test-Path $configPath) {
        Log-Info "Loading configuration from $configPath..."
        Get-Content $configPath | ForEach-Object {
            if ($_ -match "^([^#].+?)=(.+)$") {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim().Trim('"').Trim("'")
                switch ($key) {
                    "TB_VERSION" { $global:TB_VERSION = $value }
                    "INSTALL_SOURCE" { $global:INSTALL_SOURCE = $value }
                    "PKG_MANAGER" { $global:PKG_MANAGER = $value }
                    "PYTHON_VERSION_TARGET" { $global:PYTHON_VERSION_TARGET = $value }
                    "ISAA_EXTRA" { $global:ISAA_EXTRA = $value }
                    "DEV_EXTRA" { $global:DEV_EXTRA = $value }
                    "AUTO_INSTALL_DEPS" { $global:AUTO_INSTALL_DEPS = $value }
                }
            }
        }
    } else {
        Log-Info "No init.config found. Using defaults or arguments."
    }
}

function Set-Configuration {
    # Override with CLI params if set
    if ($Version) { $TB_VERSION = $Version }
    if ($Source) { $INSTALL_SOURCE = $Source }
    if ($Manager) { $PKG_MANAGER = $Manager }
    if ($Python) { $PYTHON_VERSION_TARGET = $Python }
    if ($Isaa) { $ISAA_EXTRA = "true" }
    if ($Dev) { $DEV_EXTRA = "true" }
    if ($AutoInstallDeps) { $AUTO_INSTALL_DEPS = "true" }

    Initialize-Defaults

    Log-Title "Final Configuration"
    Log-Info "ToolBoxV2 Version: $TB_VERSION"
    Log-Info "Install Source:    $INSTALL_SOURCE"
    Log-Info "Package Manager:   $PKG_MANAGER"
    Log-Info "Target Python:     $PYTHON_VERSION_TARGET"
    Log-Info "Install ISAA:      $ISAA_EXTRA"
    Log-Info "Install DEV:       $DEV_EXTRA"
    Log-Info "Install Dir:       $INSTALL_DIR"
    Log-Info "OS:                $OS_TYPE"
    Log-Info "Auto Install Deps: $AUTO_INSTALL_DEPS"
}

# -----------------------------------------------------------------------------
# AUTO-INSTALLER FUNCTIONS
# -----------------------------------------------------------------------------
function Install-SystemPython {
    Log-Info "Attempting to install Python $PYTHON_VERSION_TARGET using $SYSTEM_PKG_INSTALLER_CMD..."
    $success = $false

    switch ($SYSTEM_PKG_INSTALLER_CMD) {
        "winget" {
            $cmd = @("winget", "install", "--accept-source-agreements", "--accept-package-agreements", "-e", "--id", "Python.Python.3")
        }
        "choco" {
            $cmd = @("choco", "install", "python3", "-y")
        }
        "apt-get" {
            $pkg = "python$($PYTHON_VERSION_TARGET.Split('.')[0]).$($PYTHON_VERSION_TARGET.Split('.')[1])"
            $cmd = @("sudo", "apt-get", "update", "-y"); & $cmd @cmd | Out-Null
            $cmd = @("sudo", "apt-get", "install", "-y", $pkg, "${pkg}-venv", "python3-pip")
        }
        "brew" {
            $cmd = @("brew", "install", "python@$PYTHON_VERSION_TARGET")
        }
        Default {
            Log-Warning "No known install command for Python with $SYSTEM_PKG_INSTALLER_CMD."
            return $false
        }
    }

    Log-Info "Executing: $($cmd -join ' ')"
    if (Read-YesNo "Proceed with Python installation?" "Y") {
        $process = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..$cmd.Length] -Wait -PassThru -NoNewWindow
        if ($process.ExitCode -eq 0) {
            $success = $true
            # Refresh environment variables (path) in current session if possible, mostly effective in new shells
            if ($IS_WINDOWS) { $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine") }
        }
    }

    if ($success) {
        Log-Success "Python installation command executed. Verifying..."
        $result = Invoke-PythonHelper -Action "find_python" -Arguments @($PYTHON_VERSION_TARGET)
        if ($result -and $result.path) {
            $global:PYTHON_EXEC_PATH = $result.path
            Log-Success "Python found at: $PYTHON_EXEC_PATH"
            return $true
        }
    }
    Log-Warning "Python installation/verification failed."
    return $false
}

function Install-SystemGit {
    Log-Info "Attempting to install Git using $SYSTEM_PKG_INSTALLER_CMD..."
    $success = $false

    switch ($SYSTEM_PKG_INSTALLER_CMD) {
        "winget" { $cmd = @("winget", "install", "--accept-source-agreements", "--accept-package-agreements", "-e", "--id", "Git.Git") }
        "choco" { $cmd = @("choco", "install", "git", "-y") }
        "apt-get" { $cmd = @("sudo", "apt-get", "install", "-y", "git") }
        "brew" { $cmd = @("brew", "install", "git") }
        Default { Log-Warning "No known install command for Git."; return $false }
    }

    Log-Info "Executing: $($cmd -join ' ')"
    if (Read-YesNo "Proceed with Git installation?" "Y") {
        $process = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..$cmd.Length] -Wait -PassThru -NoNewWindow
        if ($process.ExitCode -eq 0) { $success = $true }
    }

    if ($success -and (Test-Command "git")) {
        Log-Success "Git successfully installed."
        return $true
    }
    Log-Warning "Git installation/verification failed."
    return $false
}

# -----------------------------------------------------------------------------
# CORE INSTALLATION STEPS
# -----------------------------------------------------------------------------
function Step-01_CheckPython {
    Log-Title "Step 1: Verifying Python $PYTHON_VERSION_TARGET"

    $result = Invoke-PythonHelper -Action "find_python" -Arguments @($PYTHON_VERSION_TARGET)

    if (-not $result -or -not $result.path) {
        Log-Warning "Python $PYTHON_VERSION_TARGET not found."
        if ($AUTO_INSTALL_DEPS -eq "true" -and $SYSTEM_PKG_INSTALLER_CMD) {
            if (-not (Install-SystemPython)) {
                Log-Error "Automatic Python installation failed. Please install manually."
            }
        } else {
            Log-Error "Python not found and auto-install disabled."
        }
    }

    # Re-verify
    $result = Invoke-PythonHelper -Action "find_python" -Arguments @($PYTHON_VERSION_TARGET)
    if (-not $result -or -not $result.path) {
        Log-Error "Python still not usable."
    }

    $global:PYTHON_EXEC_PATH = $result.path
    Log-Success "Using Python: $PYTHON_EXEC_PATH"
}

function Step-02_CheckGit {
    if ($INSTALL_SOURCE -eq "git") {
        Log-Title "Step 2: Verifying Git"
        if (-not (Test-Command "git")) {
            Log-Warning "Git not found."
            if ($AUTO_INSTALL_DEPS -eq "true" -and $SYSTEM_PKG_INSTALLER_CMD) {
                if (-not (Install-SystemGit)) {
                    Log-Error "Automatic Git installation failed."
                }
            } else {
                Log-Error "Git not found and auto-install disabled."
            }
        }
        Log-Success "Git found: $(git --version)"
    } else {
        Log-Info "Skipping Git check."
    }
}

function Step-03_SetupEnvironment {
    Log-Title "Step 3: Setting up Environment"

    if (Test-Path $INSTALL_DIR) {
        Log-Warning "Directory exists: $INSTALL_DIR"
        if (Test-Path $VENV_PATH) {
            if (Read-YesNo "Remove existing venv?" "Y") {
                Remove-Item $VENV_PATH -Recurse -Force
            }
        }
    }

    if (-not (Test-Path $INSTALL_DIR)) { New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null }

    Log-Info "Creating venv at $VENV_PATH..."
    $result = Invoke-PythonHelper -Action "create_venv" -Arguments @($VENV_PATH, $PYTHON_EXEC_PATH)

    if (-not $result -or $result.success -ne $true) {
        Log-Error "Venv creation failed."
    }
    Log-Success "Venv created."

    # Install extra managers if needed
    $pipPathRes = Invoke-PythonHelper -Action "get_executable_path" -Arguments @($VENV_PATH, "pip")
    $pipExe = $pipPathRes.path

    if ($PKG_MANAGER -eq "uv") {
        $uvCheck = Invoke-PythonHelper -Action "get_executable_path" -Arguments @($VENV_PATH, "uv")
        if (-not $uvCheck.path) {
            Log-Info "Installing UV..."
            & $pipExe install uv
        }
    } elseif ($PKG_MANAGER -eq "poetry") {
        $poetryCheck = Invoke-PythonHelper -Action "get_executable_path" -Arguments @($VENV_PATH, "poetry")
        if (-not $poetryCheck.path) {
            Log-Info "Installing Poetry..."
            & $pipExe install poetry
        }
    }
}

function Step-04_PrepareSource {
    if ($INSTALL_SOURCE -ne "git") { return }

    Log-Title "Step 4: Preparing Git Source"

    if (Test-Path $TOOLBOX_SRC_DIR) {
        Log-Info "Updating existing repo..."
        Push-Location $TOOLBOX_SRC_DIR
        git fetch --all --prune --tags -q
        git checkout -q $TB_VERSION
        # If it's a branch, pull. If it's a tag/hash, checkout is enough.
        $ref = git symbolic-ref -q --short HEAD
        if ($ref -and $ref -eq $TB_VERSION) {
            git pull -q origin $TB_VERSION
        }
        Pop-Location
    } else {
        Log-Info "Cloning repo..."
        New-Item -ItemType Directory -Path (Split-Path $TOOLBOX_SRC_DIR) -Force | Out-Null
        $branchArg = if ($TB_VERSION -ne "latest") { "--branch", $TB_VERSION } else { @() }
        git clone $branchArg $TOOLBOX_REPO $TOOLBOX_SRC_DIR
    }
    Log-Success "Source ready."
}

function Step-05_InstallToolBox {
    Log-Title "Step 5: Installing ToolBoxV2"

    $extras = @()
    if ($ISAA_EXTRA -eq "true") { $extras += "isaa" }
    if ($DEV_EXTRA -eq "true") { $extras += "dev" }

    $extrasSuffix = ""
    if ($extras.Count -gt 0) { $extrasSuffix = "[$($extras -join ',')]" }

    $mgrRes = Invoke-PythonHelper -Action "get_executable_path" -Arguments @($VENV_PATH, $PKG_MANAGER)
    $mgrExe = $mgrRes.path

    if (-not $mgrExe) { Log-Error "Package manager $PKG_MANAGER not found in venv." }

    $installCmd = @()

    if ($INSTALL_SOURCE -eq "git") {
        $target = "$TOOLBOX_SRC_DIR$extrasSuffix"
        if ($PKG_MANAGER -eq "pip") { $installCmd = @($mgrExe, "install", "--upgrade", $target) }
        elseif ($PKG_MANAGER -eq "uv") { $installCmd = @($mgrExe, "pip", "install", "--system", "--upgrade", $target) }
        elseif ($PKG_MANAGER -eq "poetry") {
            Push-Location $TOOLBOX_SRC_DIR
            $installCmd = @($mgrExe, "install")
            if ($extras.Count -gt 0) { foreach ($e in $extras) { $installCmd += "--extras"; $installCmd += $e } }
        }
    } else {
        $pkg = $TOOLBOX_PYPI_NAME
        if ($TB_VERSION -ne "latest") { $pkg += "==$TB_VERSION" }
        $target = "$pkg$extrasSuffix"

        if ($PKG_MANAGER -eq "pip") { $installCmd = @($mgrExe, "install", "--upgrade", $target) }
        elseif ($PKG_MANAGER -eq "uv") { $installCmd = @($mgrExe, "pip", "install", "--system", "--upgrade", $target) }
        elseif ($PKG_MANAGER -eq "poetry") {
             # Dummy project for poetry to add dependencies
             $pyproject = @"
[tool.poetry]
name = "tb-runtime"
version = "0.1.0"
description = ""
authors = []

[tool.poetry.dependencies]
python = "^$PYTHON_VERSION_TARGET"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"@
             Set-Content (Join-Path $INSTALL_DIR "pyproject.toml") $pyproject
             Push-Location $INSTALL_DIR
             $installCmd = @($mgrExe, "add", $pkg)
             foreach ($e in $extras) { $installCmd += "--extras"; $installCmd += $e }
        }
    }

    Log-Info "Installing using $PKG_MANAGER..."
    & $installCmd[0] $installCmd[1..$installCmd.Length]
    if ($LASTEXITCODE -ne 0) { Log-Error "Installation failed." }

    if ($PKG_MANAGER -eq "poetry") { Pop-Location }
    Log-Success "ToolBoxV2 installed."
}

function Step-06_CreateCommand {
    Log-Title "Step 6: Creating 'tb' command"

    $tbRes = Invoke-PythonHelper -Action "get_executable_path" -Arguments @($VENV_PATH, "tb")
    $tbPath = $tbRes.path

    if (-not $tbPath) { Log-Warning "Could not find 'tb' executable in venv."; return }

    # Ensure user bin dir exists
    if (-not (Test-Path $USER_BIN_DIR)) { New-Item -ItemType Directory -Path $USER_BIN_DIR -Force | Out-Null }

    $linkRes = Invoke-PythonHelper -Action "create_user_executable_link" -Arguments @($tbPath, "tb", $USER_BIN_DIR)
    $linkPath = $linkRes.path

    if ($linkPath) {
        Log-Success "Command link created at: $linkPath"
        if ($env:PATH -notlike "*$USER_BIN_DIR*") {
            Log-Warning "$USER_BIN_DIR is not in your PATH."
            if ($IS_WINDOWS) {
                Write-Host "Add '$USER_BIN_DIR' to your User Path variables." -ForegroundColor Yellow
            } else {
                Write-Host "Add to shell config: export PATH=`"$USER_BIN_DIR:`$PATH`"" -ForegroundColor Yellow
            }
        }
    } else {
        Log-Warning "Could not create automatic link."
        Log-Info "Run directly: $tbPath"
    }
}

function Step-07_Finalize {
    Log-Title "Step 7: Finalizing"

    # Try to find 'tb' command
    $tbCmd = "tb"
    if (-not (Test-Command "tb")) {
        # Fallback to direct venv path
        $res = Invoke-PythonHelper -Action "get_executable_path" -Arguments @($VENV_PATH, "tb")
        $tbCmd = $res.path
    }

    if ($tbCmd) {
        Log-Info "Running init: $tbCmd -init main"
        & $tbCmd "-init" "main"
        if ($LASTEXITCODE -eq 0) { Log-Success "Initialization complete!" }
        else { Log-Warning "Initialization failed. Run manually: $tbCmd -init main" }
    }
}

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
Load-ConfigFile
Set-Configuration

Step-01_CheckPython
Step-02_CheckGit
Step-03_SetupEnvironment
Step-04_PrepareSource
Step-05_InstallToolBox
Step-06_CreateCommand
Step-07_Finalize

Log-Title "Installation Complete!"
Write-Host "ToolBoxV2 is ready. Try 'tb' in a new terminal." -ForegroundColor Green
