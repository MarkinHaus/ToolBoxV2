#!/usr/bin/env pwsh

# ============================================================
# ToolBoxV2 Installer v1.0.0 — Windows (PowerShell 5+)
# https://github.com/MarkinHaus/ToolBoxV2
#
# Usage:
#   irm https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.ps1 | iex
#   .\installer.ps1 [-Mode native|uv|docker|source] [-Path <dir>] [-Config <file>]
#   .\installer.ps1 -Uninstall
#   .\installer.ps1 -Update
# ============================================================

[CmdletBinding()]
param(
    [string]$Mode       = "",
    [string]$Path       = "",
    [string]$Config     = "",
    [string]$Branch     = "main",
    [switch]$Uninstall,
    [switch]$Update,
    [switch]$Help
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest


# -----------------------------------------------------------------------------
# DISPLAY HEADER
# -----------------------------------------------------------------------------
Write-Host "**************************************************************************" -ForegroundColor DarkCyan
Write-Host "***████████╗*██████╗***██████╗**██╗*********██████╗***██████╗*██╗***██╗***" -ForegroundColor DarkCyan
Write-Host "***╚══██╔══╝██╔═══██╗*██╔═══██╗*██║*********██╔══██╗*██╔═══██╗*╚██╗██╔╝***" -ForegroundColor DarkCyan
Write-Host "******██║***██║***██║*██║***██║*██║*********██████╔╝*██║***██║**╚███╔╝****" -ForegroundColor DarkCyan
Write-Host "******██║***██║***██║*██║***██║*██║*********██╔══██╗*██║***██║**██╔██╗****" -ForegroundColor DarkCyan
Write-Host "******██║***╚██████╔╝*╚██████╔╝*███████╗****██████╔╝*╚██████╔╝*██╔╝*██╗***" -ForegroundColor DarkCyan
Write-Host "******╚═╝****╚═════╝***╚═════╝**╚══════╝****╚═════╝***╚═════╝**╚═╝**╚═╝***" -ForegroundColor DarkCyan
Write-Host "**************************************************************************" -ForegroundColor DarkCyan
Write-Host "Zero the Hero - ToolBoxV2 Core Installer" -ForegroundColor Cyan

# ── Constants ────────────────────────────────────────────────
$INSTALLER_VERSION    = "1.0.0"
$TB_ARTIFACT_NAME     = "ToolBoxV2"
$REGISTRY_API         = "https://registry.simplecore.app/api/v1"
$GITHUB_REPO          = "MarkinHaus/ToolBoxV2"
$GITHUB_RAW           = "https://raw.githubusercontent.com/$GITHUB_REPO/main"
$MIN_DISK_MB          = 300
$FEATURES_IMMUTABLE   = "mini core"
$FEATURES_OPTIONAL    = @("cli","web","desktop","isaa","exotic")

# ── Colors ───────────────────────────────────────────────────
function Log   { param($m) Write-Host "[✓] $m" -ForegroundColor Green }
function Info  { param($m) Write-Host "[→] $m" -ForegroundColor Cyan }
function Warn  { param($m) Write-Host "[!] $m" -ForegroundColor Yellow }
function Fail  { param($m) Write-Host "[✗] $m" -ForegroundColor Red; exit 1 }
function Ask   { param($m) Write-Host "[?] $m" -ForegroundColor Magenta -NoNewline }
function Step  { param($m) Write-Host "`n── $m ──────────────────────────────────" -ForegroundColor Blue }

# ── Global State ─────────────────────────────────────────────
$ARCH             = if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq "Arm64") { "arm64" } else { "x86_64" }
$INSTALL_MODE     = $Mode
$SOURCE_FROM      = "git"
$SOURCE_BRANCH    = $Branch
$INSTALL_PATH     = $Path
$ENVIRONMENT      = "development"
$INSTANCE_ID      = "tbv2_main"
$FEATURES         = "core cli"
$OPT_NGINX        = $false
$OPT_DOCKER       = $false
$OPT_OLLAMA       = $false
$OPT_MINIO        = $false
$OPT_REGISTRY     = $false
$REGISTRY_URL     = $REGISTRY_API -replace "/api/v1", ""
$RUNTIME          = ""
$UV_BIN           = ""
$PYTHON_BIN       = ""
$REGISTRY_REACHABLE = $false
$ACTION           = "install"

if ($Uninstall) { $ACTION = "uninstall" }
if ($Update)    { $ACTION = "update" }
if ($Help) {
    Write-Host @"
Usage: .\installer.ps1 [options]
  -Mode <mode>      native | uv | docker | source
  -Path <dir>       Custom install directory
  -Config <file>    Load install config from YAML
  -Branch <branch>  Git branch (source mode only, default: main)
  -Update           Update existing installation
  -Uninstall        Remove ToolBoxV2
"@
    exit 0
}

# ============================================================
# HELPERS
# ============================================================
function Get-DefaultPath {
    return "$env:LOCALAPPDATA\toolboxv2"
}

function Get-BinDir {
    return "$env:LOCALAPPDATA\Microsoft\WindowsApps"
}

function Get-YamlField {
    param([string]$File, [string]$Key, [string]$Default = "")
    if (-not (Test-Path $File)) { return $Default }
    $line = Select-String -Path $File -Pattern "^${Key}:" | Select-Object -First 1
    if ($line) {
        $val = ($line.Line -split ":", 2)[1].Trim().Trim('"').Trim("'")
        return $val
    }
    return $Default
}

function Confirm-User {
    param([string]$Msg, [string]$Default = "y")
    $prompt = if ($Default -eq "y") { "[Y/n]" } else { "[y/N]" }
    Ask "$Msg $prompt : "
    $reply = Read-Host
    if ([string]::IsNullOrEmpty($reply)) { $reply = $Default }
    return $reply -match "^[Yy]"
}

function Prompt-Default {
    param([string]$Msg, [string]$Default)
    Ask "$Msg (default: $Default) : "
    $reply = Read-Host
    if ([string]::IsNullOrEmpty($reply)) { return $Default }
    return $reply
}

function Check-DiskSpace {
    param([string]$DirPath)
    $drive = [System.IO.Path]::GetPathRoot($DirPath)
    $disk = Get-PSDrive -Name ($drive.TrimEnd(":\")) -ErrorAction SilentlyContinue
    if ($disk -and ($disk.Free / 1MB) -lt $MIN_DISK_MB) {
        Fail "Not enough disk space (need ${MIN_DISK_MB}MB)"
    }
}

function Download-File {
    param([string]$Url, [string]$Dest)
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $wc = New-Object System.Net.WebClient
    $wc.DownloadFile($Url, $Dest)
}

function Invoke-RegistryGet {
    param([string]$Path)
    try {
        return Invoke-RestMethod -Uri "${REGISTRY_API}${Path}" -TimeoutSec 5 -ErrorAction Stop | ConvertTo-Json -Depth 5
    } catch { return "" }
}

# ============================================================
# PHASE 0 — DISCOVERY
# ============================================================
function Phase-Discovery {
    Step "Phase 0 — Discovery"
    Info "Platform: windows/$ARCH"

    $scanPaths = @(
        "$env:LOCALAPPDATA\toolboxv2",
        "$env:APPDATA\toolboxv2",
        "C:\toolboxv2",
        $env:TOOLBOX_HOME,
        $env:TB_INSTALL_DIR
    ) | Where-Object { $_ -and $_ -ne "" }

    $foundInstalls = @()
    foreach ($p in $scanPaths) {
        if (Test-Path "$p\install.manifest") { $foundInstalls += $p }
    }

    # Check PATH for tb binary
    $tbCmd = Get-Command tb -ErrorAction SilentlyContinue
    if ($tbCmd) {
        Info "Existing 'tb' binary found: $($tbCmd.Source)"
        $foundInstalls += [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetDirectoryName($tbCmd.Source))
    }

    # Check Docker
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        $imgs = docker images --format "{{.Repository}}:{{.Tag}}" 2>$null | Select-String "toolboxv2"
        if ($imgs) { Info "Docker TB image(s): $imgs" }
    }

    if ($foundInstalls.Count -gt 0) {
        Write-Host ""
        Warn "Existing ToolBoxV2 installation(s) found:"
        foreach ($p in $foundInstalls) {
            $ver  = Get-YamlField "$p\install.manifest" "tb_version" "unknown"
            $mode = Get-YamlField "$p\install.manifest" "install_mode" "unknown"
            Write-Host "  $p  ($mode, v$ver)"
        }
        Write-Host ""
        if ($ACTION -eq "install") {
            if (Confirm-User "Update existing installation?") {
                $script:ACTION       = "update"
                $script:INSTALL_PATH = $foundInstalls[0]
            } else {
                if (-not (Confirm-User "Install fresh alongside?")) { Fail "Aborted." }
            }
        }
    } else {
        Info "No existing installation found — fresh install"
    }
}

# ============================================================
# PHASE 1 — CONFIG
# ============================================================
function Phase-Config {
    Step "Phase 1 — Configuration"

    if (-not $Config -and (Test-Path "tb-install.yaml")) {
        $script:Config = "tb-install.yaml"
        Info "Found tb-install.yaml in current directory"
    }

    if ($Config) {
        if (-not (Test-Path $Config)) { Fail "Config file not found: $Config" }
        Info "Loading config: $Config"
        if (-not $INSTALL_MODE)  { $script:INSTALL_MODE  = Get-YamlField $Config "install_mode" "" }
        if (-not $INSTALL_PATH)  { $script:INSTALL_PATH  = Get-YamlField $Config "install_path" "" }
        $script:SOURCE_FROM   = Get-YamlField $Config "source_from"   "git"
        $script:SOURCE_BRANCH = Get-YamlField $Config "source_branch" "main"
        $script:ENVIRONMENT   = Get-YamlField $Config "environment"   "development"
        $script:INSTANCE_ID   = Get-YamlField $Config "instance_id"   "tbv2_main"
        $script:OPT_NGINX     = (Get-YamlField $Config "optional.nginx"           "false") -eq "true"
        $script:OPT_DOCKER    = (Get-YamlField $Config "optional.docker_runtime"  "false") -eq "true"
        $script:OPT_OLLAMA    = (Get-YamlField $Config "optional.ollama"          "false") -eq "true"
        $script:OPT_MINIO     = (Get-YamlField $Config "optional.minio"           "false") -eq "true"
        $script:OPT_REGISTRY  = (Get-YamlField $Config "optional.registry"        "false") -eq "true"
    }

    # Install mode
    if (-not $INSTALL_MODE) {
        Write-Host ""
        Write-Host "  Select install mode:"
        Write-Host "  1) native   — Single binary, no Python required (recommended)"
        Write-Host "  2) uv       — Python package via uv tool"
        Write-Host "  3) docker   — Containerized, isolated"
        Write-Host "  4) source   — Full source from Git or Registry"
        Ask "Mode [1-4] (default: 1) : "
        $choice = Read-Host
        $script:INSTALL_MODE = switch ($choice) {
            "2" { "uv" }; "3" { "docker" }; "4" { "source" }; default { "native" }
        }
    }

    if ($INSTALL_MODE -eq "source" -and -not $SOURCE_FROM) {
        Write-Host "  1) git      — Clone from GitHub"
        Write-Host "  2) registry — Download release tarball"
        Ask "Source [1-2] (default: 1) : "
        $sc = Read-Host
        $script:SOURCE_FROM = if ($sc -eq "2") { "registry" } else { "git" }
    }

    # Features
    Write-Host ""
    Write-Host "  Included (always): mini core"
    $selFeatures = $FEATURES
    foreach ($feat in $FEATURES_OPTIONAL) {
        $currently = if ($selFeatures -match "\b${feat}\b") { "yes" } else { "no" }
        $def = if ($currently -eq "yes") { "y" } else { "n" }
        if (Confirm-User "  Enable ${feat}? [currently: $currently]" $def) {
            if ($selFeatures -notmatch "\b${feat}\b") { $selFeatures += " $feat" }
        } else {
            $selFeatures = ($selFeatures -split " " | Where-Object { $_ -ne $feat }) -join " "
        }
    }
    $script:FEATURES = $selFeatures.Trim()

    $script:ENVIRONMENT = Prompt-Default "Environment (development|production|staging)" $ENVIRONMENT

    if (-not $INSTALL_PATH) {
        $default = Get-DefaultPath
        Ask "Custom install path? Leave empty for default ($default) : "
        $cp = Read-Host
        $script:INSTALL_PATH = if ($cp) { $cp } else { $default }
    }

    Log "Mode: $INSTALL_MODE$(if ($SOURCE_FROM) { " ($SOURCE_FROM)" })"
    Log "Path: $INSTALL_PATH"
    Log "Env:  $ENVIRONMENT"
    Log "Features: $FEATURES_IMMUTABLE $FEATURES"
}

# ============================================================
# PHASE 2 — PRE-FLIGHT
# ============================================================
function Phase-Preflight {
    Step "Phase 2 — Pre-flight Checks"

    # 1. Install path
    Info "Checking install path: $INSTALL_PATH"
    Check-DiskSpace $INSTALL_PATH
    try {
        New-Item -ItemType Directory -Path $INSTALL_PATH -Force | Out-Null
        $testFile = "$INSTALL_PATH\.tb_write_test"
        [System.IO.File]::WriteAllText($testFile, "test")
        Remove-Item $testFile -Force
    } catch {
        $script:INSTALL_PATH = Prompt-Default "Path not writable. Enter writable path" "$env:LOCALAPPDATA\toolboxv2"
        New-Item -ItemType Directory -Path $INSTALL_PATH -Force | Out-Null
    }
    Log "Install path OK: $INSTALL_PATH"

    # 2. Network
    Info "Checking network..."
    try {
        Invoke-WebRequest -Uri "https://1.1.1.1" -TimeoutSec 3 -UseBasicParsing | Out-Null
    } catch {
        try { Invoke-WebRequest -Uri "https://github.com" -TimeoutSec 3 -UseBasicParsing | Out-Null }
        catch { Fail "No network connectivity. Installer requires internet access." }
    }
    Log "Network OK"

    # 3. Registry
    Info "Checking registry..."
    try {
        $health = Invoke-RestMethod -Uri "$REGISTRY_API/health" -TimeoutSec 5 -ErrorAction Stop
        if ($health.status -eq "healthy") {
            $script:REGISTRY_REACHABLE = $true
            Log "Registry OK"
        }
    } catch {
        Warn "Registry unreachable — using GitHub Releases as fallback"
    }

    # 4. Runtime
    if ($INSTALL_MODE -eq "native" -or $INSTALL_MODE -eq "docker") {
        $script:RUNTIME = "none"
        Log "Runtime: none required for $INSTALL_MODE mode"
    } else {
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCmd) {
            $script:UV_BIN  = $uvCmd.Source
            $script:RUNTIME = "uv"
            Log "Runtime: uv found at $UV_BIN"
        } else {
            # Try Python 3.11+
            foreach ($py in @("python3.13","python3.12","python3.11","python3","python")) {
                $pyCmd = Get-Command $py -ErrorAction SilentlyContinue
                if ($pyCmd) {
                    $pyver = & $py -c "import sys; print(sys.version_info >= (3,11))" 2>$null
                    if ($pyver -eq "True") {
                        $script:PYTHON_BIN = $pyCmd.Source
                        $script:RUNTIME    = "venv"
                        Log "Runtime: Python found at $PYTHON_BIN (venv fallback)"
                        break
                    }
                }
            }

            if (-not $RUNTIME) {
                Info "No runtime found — bootstrapping uv..."
                Invoke-RestMethod "https://astral.sh/uv/install.ps1" | Invoke-Expression
                $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
                if (-not $uvCmd) { Fail "uv bootstrap failed. Install manually: https://docs.astral.sh/uv/" }
                $script:UV_BIN  = $uvCmd.Source
                $script:RUNTIME = "uv"
                Log "Runtime: uv bootstrapped"
            }
        }
    }

    # 5. Optional tools
    Check-Optional "docker"  $OPT_DOCKER  "Docker runtime"
    Check-Optional "nginx"   $OPT_NGINX   "nginx"
    Check-Optional "ollama"  $OPT_OLLAMA  "Ollama (LLM)"
}

function Check-Optional {
    param([string]$Name, [bool]$CfgVal, [string]$Label)
    $present = [bool](Get-Command $Name -ErrorAction SilentlyContinue)
    if ($CfgVal) {
        if ($present) { Log "$Label already installed" }
        else { Install-Optional $Name }
    } elseif (-not $present) {
        if (Confirm-User "$Label not found. Install it?") { Install-Optional $Name }
    } else {
        Log "$Label found"
    }
}

function Install-Optional {
    param([string]$Name)
    switch ($Name) {
        "docker"  { Start-Process "https://www.docker.com/products/docker-desktop" }
        "ollama"  { Invoke-RestMethod "https://ollama.ai/install.ps1" | Invoke-Expression }
        default   { Warn "Please install $Name manually" }
    }
}

# ============================================================
# PHASE 3 — INSTALL
# ============================================================
function Phase-Install {
    Step "Phase 3 — Install ($INSTALL_MODE)"
    foreach ($d in @("bin",".data",".config","logs")) {
        New-Item -ItemType Directory -Path "$INSTALL_PATH\$d" -Force | Out-Null
    }

    switch ($INSTALL_MODE) {
        "native"  { return Install-Native }
        "uv"      { return Install-Uv }
        "docker"  { return Install-Docker }
        "source"  { return Install-Source }
    }
}

function Get-VersionAndUrl {
    $platform = "windows"
    $archTag  = if ($ARCH -eq "x86_64") { "x64" } else { "arm64" }

    if ($REGISTRY_REACHABLE) {
        try {
            $resp = Invoke-RestMethod "$REGISTRY_API/artifacts/$TB_ARTIFACT_NAME/latest?platform=$platform&architecture=$ARCH" -ErrorAction Stop
            if ($resp.url) { return @{ Version=$resp.version; Url=$resp.url; Checksum=$resp.checksum } }
        } catch {}
    }

    # GitHub fallback
    Info "Using GitHub Releases..."
    $rel   = Invoke-RestMethod "https://api.github.com/repos/$GITHUB_REPO/releases/latest"
    $tag   = $rel.tag_name
    $fname = "toolbox-windows-$archTag.exe"
    $url   = "https://github.com/$GITHUB_REPO/releases/download/$tag/$fname"
    return @{ Version=($tag -replace "^v",""); Url=$url; Checksum="" }
}

function Install-Native {
    Info "Fetching latest binary..."
    $info = Get-VersionAndUrl
    if (-not $info.Url) { Fail "Could not determine download URL" }
    Info "Downloading ToolBoxV2 v$($info.Version)..."
    $dest = "$INSTALL_PATH\bin\tb.exe"
    Download-File $info.Url $dest

    if ($info.Checksum) {
        $actual = (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $info.Checksum.ToLower()) { Fail "Checksum mismatch — download may be corrupted" }
        Log "Checksum verified"
    }

    # Add to PATH via user environment
    $currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $binDir = "$INSTALL_PATH\bin"
    if ($currentPath -notlike "*$binDir*") {
        [System.Environment]::SetEnvironmentVariable("PATH", "$binDir;$currentPath", "User")
        $env:PATH = "$binDir;$env:PATH"
        Log "Added $binDir to user PATH"
    }

    Log "Binary installed: $dest"
    return $info.Version
}

function Install-Uv {
    Info "Installing via uv tool..."
    $extras = ($FEATURES -split " " | Where-Object { $_ -notmatch "^(core|mini)$" }) -join ","
    $pkg = if ($extras) { "ToolBoxV2[$extras]" } else { "ToolBoxV2" }
    & $UV_BIN tool install $pkg --force
    Log "Installed: $pkg"
    $ver = (& tb --version 2>$null) -replace "[^\d\.]","" | Select-Object -First 1
    return if ($ver) { $ver } else { "unknown" }
}

function Install-Venv {
    Info "Installing via pip/venv..."
    $venvPath = "$INSTALL_PATH\.venv"
    & $PYTHON_BIN -m venv $venvPath
    $pip = "$venvPath\Scripts\pip.exe"
    & $pip install --upgrade pip -q
    $extras = ($FEATURES -split " " | Where-Object { $_ -notmatch "^(core|mini)$" }) -join ","
    $pkg = if ($extras) { "ToolBoxV2[$extras]" } else { "ToolBoxV2" }
    & $pip install $pkg -q

    # Write wrapper
    $wrapper = "$INSTALL_PATH\bin\tb.cmd"
    "@echo off`n`"$venvPath\Scripts\tb.exe`" %*" | Set-Content $wrapper
    Log "Installed: $pkg via venv"
    return "unknown"
}

function Install-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "Docker not found" }
    $image = "ghcr.io/markinhaus/toolboxv2:latest"
    Info "Pulling Docker image: $image"
    docker pull $image
    $wrapper = "$INSTALL_PATH\bin\tb.cmd"
    "@echo off`ndocker run --rm -it -v `"$INSTALL_PATH\.data:/data`" -e TB_DATA_DIR=/data $image %*" | Set-Content $wrapper
    Log "Docker wrapper installed"
    return "latest"
}

function Install-Source {
    Info "Installing from source ($SOURCE_FROM)..."
    $srcDir = "$INSTALL_PATH\src"

    if ($SOURCE_FROM -eq "git") {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git not found" }
        if (Test-Path "$srcDir\.git") {
            git -C $srcDir fetch origin
            git -C $srcDir checkout $SOURCE_BRANCH
            git -C $srcDir pull origin $SOURCE_BRANCH
        } else {
            git clone --branch $SOURCE_BRANCH --depth 1 "https://github.com/$GITHUB_REPO.git" $srcDir
        }
    } else {
        $resp = Invoke-RestMethod "$REGISTRY_API/packages/ToolBoxV2/versions/latest/download" -ErrorAction Stop
        if (-not $resp.url) { Fail "Could not get tarball URL" }
        $tmp = "$env:TEMP\tbv2_src.zip"
        Download-File $resp.url $tmp
        Expand-Archive $tmp $srcDir -Force
        Remove-Item $tmp
    }

    Push-Location $srcDir
    if ($RUNTIME -eq "uv") {
        $extras = ($FEATURES -split " " | Where-Object { $_ -notmatch "^(core|mini)$" }) -join ","
        if ($extras) { & $UV_BIN sync --extra $extras } else { & $UV_BIN sync }
        "@echo off`ncd /d `"$srcDir`" && `"$UV_BIN`" run tb %*" | Set-Content "$INSTALL_PATH\bin\tb.cmd"
    } else {
        & $PYTHON_BIN -m venv "$srcDir\.venv"
        & "$srcDir\.venv\Scripts\pip.exe" install -e ".[$(($FEATURES -replace ' ',','))]" -q
        "@echo off`n`"$srcDir\.venv\Scripts\tb.exe`" %*" | Set-Content "$INSTALL_PATH\bin\tb.cmd"
    }
    Pop-Location

    $ver = (& "$INSTALL_PATH\bin\tb.cmd" --version 2>$null) -replace "[^\d\.]","" | Select-Object -First 1
    return if ($ver) { $ver } else { "dev" }
}

# ============================================================
# PHASE 4 — MANIFESTS & ENV
# ============================================================
function Phase-WriteManifests {
    param([string]$TbVersion)
    Step "Phase 4 — Writing Manifests & Env"

    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $optInstalled = @()
    if ($OPT_NGINX)    { $optInstalled += "nginx" }
    if ($OPT_DOCKER)   { $optInstalled += "docker" }
    if ($OPT_OLLAMA)   { $optInstalled += "ollama" }
    if ($OPT_MINIO)    { $optInstalled += "minio" }
    if ($OPT_REGISTRY) { $optInstalled += "registry" }

    @"
installer_version: "$INSTALLER_VERSION"
installed_at: "$timestamp"
install_mode: "$INSTALL_MODE"
source_from: "$SOURCE_FROM"
source_branch: "$SOURCE_BRANCH"
tb_version: "$TbVersion"
toolbox_home: "$INSTALL_PATH"
bin_path: "$INSTALL_PATH\bin\tb.exe"
src_path: "$INSTALL_PATH\src"
venv_path: "$INSTALL_PATH\.venv"
runtime: "$RUNTIME"
python_path: "$PYTHON_BIN"
uv_path: "$UV_BIN"
features: "$FEATURES_IMMUTABLE $FEATURES"
optional_installed: "$($optInstalled -join ' ')"
"@ | Set-Content "$INSTALL_PATH\install.manifest"
    Log "Written: $INSTALL_PATH\install.manifest"

    $manifestFile = "$INSTALL_PATH\tb-manifest.yaml"
    if (-not (Test-Path $manifestFile)) {
        @"
manifest_version: "1.0.0"
app:
  name: ToolBoxV2
  version: "$TbVersion"
  instance_id: "$INSTANCE_ID"
  environment: $ENVIRONMENT
  debug: false
  log_level: INFO
paths:
  data_dir: "$INSTALL_PATH\.data"
  config_dir: "$INSTALL_PATH\.config"
  logs_dir: "$INSTALL_PATH\logs"
  mods_dir: "$INSTALL_PATH\mods"
  dist_dir: "$INSTALL_PATH\dist"
registry:
  url: "$REGISTRY_URL"
  auto_update: false
"@ | Set-Content $manifestFile
        Log "Written: $manifestFile"
    } else {
        Info "tb-manifest.yaml exists — skipping (run 'tb manifest apply' to update)"
    }

    $envFile = "$INSTALL_PATH\.env"
    if (-not (Test-Path $envFile)) {
        @"
# ToolBoxV2 Environment — generated by installer
TOOLBOX_HOME=$INSTALL_PATH
TB_INSTALL_DIR=$INSTALL_PATH
TB_DATA_DIR=$INSTALL_PATH\.data
TB_DIST_DIR=$INSTALL_PATH\dist
TB_ENV=$ENVIRONMENT
TB_JWT_SECRET=                    # REQUIRED: set before production use
TB_COOKIE_SECRET=                 # REQUIRED: set before production use
TOOLBOXV2_BASE=localhost
TOOLBOXV2_BASE_PORT=8000
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
"@ | Set-Content $envFile
        Log "Written: $envFile"
        Warn "Edit $envFile to set TB_JWT_SECRET and TB_COOKIE_SECRET before production use"
    }

    # Write TOOLBOX_HOME to user environment
    [System.Environment]::SetEnvironmentVariable("TOOLBOX_HOME", $INSTALL_PATH, "User")
    $env:TOOLBOX_HOME = $INSTALL_PATH
    Log "TOOLBOX_HOME set in user environment"
}

# ============================================================
# UPDATE
# ============================================================
function Action-Update {
    Step "Update"
    $manifest = "$INSTALL_PATH\install.manifest"
    if (-not (Test-Path $manifest)) { Fail "No install.manifest found at $INSTALL_PATH" }
    $script:INSTALL_MODE  = Get-YamlField $manifest "install_mode"
    $script:RUNTIME       = Get-YamlField $manifest "runtime"
    $script:UV_BIN        = Get-YamlField $manifest "uv_path"
    $script:PYTHON_BIN    = Get-YamlField $manifest "python_path"

    switch ($INSTALL_MODE) {
        "native"  { $v = Install-Native; Phase-WriteManifests $v }
        "uv"      { & $UV_BIN tool upgrade ToolBoxV2; Log "Updated via uv" }
        "docker"  { docker pull ghcr.io/markinhaus/toolboxv2:latest; Log "Updated Docker image" }
        "source"  {
            $script:SOURCE_FROM   = Get-YamlField $manifest "source_from" "git"
            $script:SOURCE_BRANCH = Get-YamlField $manifest "source_branch" "main"
            $script:FEATURES      = Get-YamlField $manifest "features" "core cli"
            Install-Source
            Log "Updated from source"
        }
    }
}

# ============================================================
# UNINSTALL
# ============================================================
function Action-Uninstall {
    Step "Uninstall"
    $manifest = "$INSTALL_PATH\install.manifest"
    if (-not (Test-Path $manifest)) { Fail "No install.manifest found" }

    $mode     = Get-YamlField $manifest "install_mode"
    $homePath = Get-YamlField $manifest "toolbox_home"

    Warn "This will remove ToolBoxV2 from: $homePath"
    if (-not (Confirm-User "Continue with uninstall?")) { Fail "Aborted." }

    switch ($mode) {
        "docker" {
            docker rm -f toolboxv2 2>$null
            docker rmi ghcr.io/markinhaus/toolboxv2:latest 2>$null
        }
        "uv"     { & $UV_BIN tool uninstall ToolBoxV2 2>$null }
    }

    # Remove from PATH
    $currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $binDir = "$homePath\bin"
    $newPath = ($currentPath -split ";" | Where-Object { $_ -ne $binDir }) -join ";"
    [System.Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    [System.Environment]::SetEnvironmentVariable("TOOLBOX_HOME", $null, "User")

    if (Confirm-User "Remove all data and config in ${homePath}? (IRREVERSIBLE)" "n") {
        Remove-Item -Path $homePath -Recurse -Force
    }
    Log "Uninstall complete"
}

# ============================================================
# SUMMARY
# ============================================================
function Print-Summary {
    param([string]$TbVersion)
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║   ToolBoxV2 v$TbVersion installed successfully" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Home:     $INSTALL_PATH"
    Write-Host "  Mode:     $INSTALL_MODE"
    Write-Host "  Runtime:  $RUNTIME"
    Write-Host "  Features: $FEATURES_IMMUTABLE $FEATURES"
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Restart terminal (PATH updated)"
    Write-Host "  2. First run:    tb"
    Write-Host "  3. Check status: tb status"
    if (Select-String -Path "$INSTALL_PATH\.env" -Pattern "TB_JWT_SECRET=$" -ErrorAction SilentlyContinue) {
        Write-Host ""
        Write-Host "  [!] Set secrets before production:" -ForegroundColor Yellow
        Write-Host "      $INSTALL_PATH\.env"
    }
    Write-Host ""
}

# ============================================================
# MAIN
# ============================================================
Write-Host ""
Write-Host "  ╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║     ToolBoxV2 Installer v$INSTALLER_VERSION         ║" -ForegroundColor Cyan
Write-Host "  ╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

switch ($ACTION) {
    "install" {
        Phase-Discovery
        Phase-Config
        Phase-Preflight
        $tbVersion = if ($INSTALL_MODE -eq "uv" -and $RUNTIME -eq "venv") {
            Install-Venv
        } else {
            Phase-Install
        }
        Phase-WriteManifests $tbVersion
        Print-Summary $tbVersion
    }
    "update"    { Phase-Discovery; Action-Update }
    "uninstall" { Phase-Discovery; Action-Uninstall }
}
