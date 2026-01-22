# === LLM Gateway Windows Setup Script ===
# For Ryzen CPU (12 cores, 48GB RAM, no GPU)
# Requires: Git, CMake, Visual Studio Build Tools, Python 3.10+

param(
    [switch]$SkipBuild,
    [switch]$SkipPython,
    [switch]$SkipWhisper
)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DATA_DIR = Join-Path $SCRIPT_DIR "data"
$MODELS_DIR = Join-Path $DATA_DIR "models"
$BUILD_DIR = Join-Path $SCRIPT_DIR "build"

Write-Host "=== LLM Gateway Windows Setup ===" -ForegroundColor Cyan
Write-Host "Script directory: $SCRIPT_DIR"

# Create directories
New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $MODELS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $SCRIPT_DIR "static") | Out-Null

# === 1. Check prerequisites ===
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

$missingTools = @()
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $missingTools += "git" }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { $missingTools += "cmake" }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { $missingTools += "python" }

if ($missingTools.Count -gt 0) {
    Write-Host "Missing tools: $($missingTools -join ', ')" -ForegroundColor Red
    Write-Host "Install with: winget install Git.Git; winget install Kitware.CMake; winget install Python.Python.3.12"
    exit 1
}

# Check for Visual Studio Build Tools
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) {
    Write-Host "Visual Studio Build Tools not found!" -ForegroundColor Red
    Write-Host "Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Host "Select 'Desktop development with C++'"
    exit 1
}

Write-Host "All prerequisites found!" -ForegroundColor Green

# === 2. Build llama.cpp ===
if (-not $SkipBuild) {
    Write-Host "[2/5] Building llama.cpp..." -ForegroundColor Yellow

    $LLAMA_DIR = Join-Path $BUILD_DIR "llama.cpp"

    if (-not (Test-Path $LLAMA_DIR)) {
        Write-Host "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp.git $LLAMA_DIR
    }

    Push-Location $LLAMA_DIR
    git pull

    # Build with CPU optimizations (AVX2 for Ryzen)
    $LLAMA_BUILD = Join-Path $LLAMA_DIR "build"

    cmake -B $LLAMA_BUILD `
        -DBUILD_SHARED_LIBS=OFF `
        -DGGML_NATIVE=ON `
        -DLLAMA_CURL=ON `
        -DCMAKE_BUILD_TYPE=Release

    cmake --build $LLAMA_BUILD --config Release --target llama-server llama-cli llama-quantize -j $env:NUMBER_OF_PROCESSORS

    # Copy binaries
    $binPath = Join-Path $LLAMA_BUILD "bin\Release"
    if (-not (Test-Path $binPath)) { $binPath = Join-Path $LLAMA_BUILD "bin" }

    Copy-Item (Join-Path $binPath "llama-server.exe") $SCRIPT_DIR -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $binPath "llama-cli.exe") $SCRIPT_DIR -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $binPath "llama-quantize.exe") $SCRIPT_DIR -Force -ErrorAction SilentlyContinue

    Pop-Location
    Write-Host "llama.cpp built successfully!" -ForegroundColor Green
}

# === 3. Build whisper.cpp (optional) ===
if (-not $SkipWhisper -and -not $SkipBuild) {
    Write-Host "[3/5] Building whisper.cpp..." -ForegroundColor Yellow

    $WHISPER_DIR = Join-Path $BUILD_DIR "whisper.cpp"

    if (-not (Test-Path $WHISPER_DIR)) {
        git clone https://github.com/ggerganov/whisper.cpp.git $WHISPER_DIR
    }

    Push-Location $WHISPER_DIR
    git pull

    $WHISPER_BUILD = Join-Path $WHISPER_DIR "build"

    cmake -B $WHISPER_BUILD `
        -DBUILD_SHARED_LIBS=OFF `
        -DGGML_NATIVE=ON `
        -DCMAKE_BUILD_TYPE=Release

    cmake --build $WHISPER_BUILD --config Release --target whisper-server -j $env:NUMBER_OF_PROCESSORS

    $binPath = Join-Path $WHISPER_BUILD "bin\Release"
    if (-not (Test-Path $binPath)) { $binPath = Join-Path $WHISPER_BUILD "bin" }

    Copy-Item (Join-Path $binPath "whisper-server.exe") $SCRIPT_DIR -Force -ErrorAction SilentlyContinue

    Pop-Location
    Write-Host "whisper.cpp built successfully!" -ForegroundColor Green
} else {
    Write-Host "[3/5] Skipping whisper.cpp..." -ForegroundColor Gray
}

# === 4. Setup Python environment ===
if (-not $SkipPython) {
    Write-Host "[4/5] Setting up Python environment..." -ForegroundColor Yellow

    $VENV_DIR = Join-Path $SCRIPT_DIR "venv"

    if (-not (Test-Path $VENV_DIR)) {
        python -m venv $VENV_DIR
    }

    # Activate and install
    $pipPath = Join-Path $VENV_DIR "Scripts\pip.exe"

    & $pipPath install --upgrade pip
    & $pipPath install fastapi uvicorn[standard] httpx aiosqlite psutil huggingface_hub pydantic python-multipart "passlib[bcrypt]"

    Write-Host "Python environment ready!" -ForegroundColor Green
} else {
    Write-Host "[4/5] Skipping Python setup..." -ForegroundColor Gray
}

# === 5. Create initial config ===
Write-Host "[5/5] Creating initial configuration..." -ForegroundColor Yellow

$CONFIG_FILE = Join-Path $DATA_DIR "config.json"

if (-not (Test-Path $CONFIG_FILE)) {
    # Generate secure admin key
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $ADMIN_KEY = "sk-admin-" + [BitConverter]::ToString($bytes).Replace("-", "").ToLower()

    $config = @{
        slots = @{
            "4801" = $null
            "4802" = $null
            "4803" = $null
            "4804" = $null
            "4805" = $null
            "4806" = $null
            "4807" = $null
        }
        hf_token = $null
        admin_key = $ADMIN_KEY
        default_threads = 10
        default_ctx_size = 8192
        pricing = @{
            input_per_1k = 0.0001
            output_per_1k = 0.0002
        }
    }

    $config | ConvertTo-Json -Depth 10 | Set-Content $CONFIG_FILE

    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  ADMIN API KEY (save this!):" -ForegroundColor Yellow
    Write-Host "  $ADMIN_KEY" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
}

# === Create startup script ===
$START_SCRIPT = Join-Path $SCRIPT_DIR "start_gateway.ps1"
$startContent = @"
# LLM Gateway Startup Script
`$SCRIPT_DIR = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$VENV_PYTHON = Join-Path `$SCRIPT_DIR "venv\Scripts\python.exe"
`$UVICORN = Join-Path `$SCRIPT_DIR "venv\Scripts\uvicorn.exe"

Push-Location `$SCRIPT_DIR
& `$UVICORN server:app --host 0.0.0.0 --port 4000 --workers 1
Pop-Location
"@
$startContent | Set-Content $START_SCRIPT

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Commands:" -ForegroundColor Cyan
Write-Host "  Start:   .\start_gateway.ps1"
Write-Host "  Dev:     .\venv\Scripts\activate; uvicorn server:app --host 0.0.0.0 --port 4000 --reload"
Write-Host ""
Write-Host "Access:" -ForegroundColor Cyan
Write-Host "  API:     http://localhost:4000/v1/"
Write-Host "  Admin:   http://localhost:4000/admin/"
Write-Host "  User:    http://localhost:4000/user/"
Write-Host ""
Write-Host "llama.cpp binaries location: $SCRIPT_DIR" -ForegroundColor Yellow
Write-Host "  - llama-server.exe"
Write-Host "  - llama-cli.exe"
Write-Host "  - llama-quantize.exe"
Write-Host ""
Write-Host "For ISAA RL integration, set LLAMA_CPP_PATH environment variable:" -ForegroundColor Yellow
Write-Host "  `$env:LLAMA_CPP_PATH = '$BUILD_DIR\llama.cpp'"

