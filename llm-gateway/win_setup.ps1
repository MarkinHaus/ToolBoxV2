# LLM Gateway Windows Setup Script
# PowerShell script for setting up the LLM Gateway on Windows

param(
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host @"
LLM Gateway Windows Setup Script

Usage:
    .\win_setup.ps1

This script will:
1. Check for Python 3.12 or later
2. Create a Python virtual environment
3. Install required Python dependencies
4. Check for Ollama installation
5. Generate initial configuration with admin API key

"@
    exit 0
}

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "LLM Gateway Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/5] Checking Python version..." -ForegroundColor Yellow

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed." -ForegroundColor Red
    Write-Host "Please install Python 3.12 or later from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Or use winget: winget install Python.Python.3.12" -ForegroundColor Yellow
    exit 1
}

$pythonVersionOutput = python --version 2>&1
$pythonVersion = $pythonVersionOutput -replace "Python ", ""
$versionParts = $pythonVersion.Split('.')
$majorVersion = [int]$versionParts[0]
$minorVersion = [int]$versionParts[1]

if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 12)) {
    Write-Host "ERROR: Python 3.12 or later is required." -ForegroundColor Red
    Write-Host "Current version: $pythonVersion" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python $pythonVersion detected" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "[2/5] Creating Python virtual environment..." -ForegroundColor Yellow
$VENV_DIR = Join-Path $SCRIPT_DIR "venv"

if (Test-Path $VENV_DIR) {
    Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Gray
} else {
    python -m venv $VENV_DIR
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}
Write-Host ""

# Install dependencies
Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Yellow
$pipPath = Join-Path $VENV_DIR "Scripts\pip.exe"

& $pipPath install --upgrade pip --quiet
& $pipPath install -r requirements.txt --quiet

Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Check for Ollama
Write-Host "[4/5] Checking for Ollama..." -ForegroundColor Yellow

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $ollamaVersion = ollama --version 2>&1
    Write-Host "✓ Ollama is already installed: $ollamaVersion" -ForegroundColor Green
} else {
    Write-Host "Ollama is not installed." -ForegroundColor Yellow
    $response = Read-Host "Would you like to install Ollama now? (y/n)"

    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Installing Ollama via winget..." -ForegroundColor Yellow

        # Try winget first
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            try {
                winget install --id=Ollama.Ollama -e --silent
                Write-Host "✓ Ollama installed" -ForegroundColor Green
            } catch {
                Write-Host "Failed to install via winget. Please download from https://ollama.com/download" -ForegroundColor Red
            }
        } else {
            Write-Host "winget not found. Please download Ollama from https://ollama.com/download" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Skipping Ollama installation." -ForegroundColor Gray
        Write-Host "You can install it later from https://ollama.com/download" -ForegroundColor Yellow
        Write-Host "Or use: winget install Ollama.Ollama" -ForegroundColor Yellow
    }
}
Write-Host ""

# Generate initial config
Write-Host "[5/5] Setting up configuration..." -ForegroundColor Yellow
$DATA_DIR = Join-Path $SCRIPT_DIR "data"
New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null

$CONFIG_FILE = Join-Path $DATA_DIR "config.json"

if (Test-Path $CONFIG_FILE) {
    Write-Host "Configuration file already exists. Skipping generation." -ForegroundColor Gray
} else {
    # Generate a random admin key
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $ADMIN_KEY = [BitConverter]::ToString($bytes).Replace("-", "").ToLower()

    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    $config = @{
        api_keys = @{
            $ADMIN_KEY = @{
                name = "admin"
                role = "admin"
                created_at = $timestamp
                rate_limit = @{
                    requests_per_minute = 100
                    tokens_per_minute = 100000
                }
            }
        }
        ollama = @{
            base_url = "http://localhost:11434"
            timeout = 300
        }
        server = @{
            host = "0.0.0.0"
            port = 4000
            log_level = "info"
        }
    }

    $config | ConvertTo-Json -Depth 10 | Set-Content $CONFIG_FILE -Encoding UTF8

    Write-Host "✓ Configuration file created with admin API key" -ForegroundColor Green
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "IMPORTANT: Save your admin API key!" -ForegroundColor Yellow
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host $ADMIN_KEY -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start Ollama (if not already running): ollama serve"
Write-Host "2. Pull a model: ollama pull llama3.2:latest"
Write-Host "3. Activate the virtual environment: .\venv\Scripts\Activate.ps1"
Write-Host "4. Run the gateway: python server.py"
Write-Host "   Or with uvicorn: uvicorn server:app --host 0.0.0.0 --port 4000"
Write-Host ""
Write-Host "The gateway will be available at: http://localhost:4000" -ForegroundColor Green
Write-Host "API documentation: http://localhost:4000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "For Docker deployment:" -ForegroundColor Cyan
Write-Host "  Bare mode (gateway only): docker compose up gateway"
Write-Host "  Docker mode (both): docker compose --profile ollama up"
Write-Host ""
