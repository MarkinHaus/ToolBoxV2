# TB Language - Mobile Build Script (PowerShell)
# Builds tb-runtime for Android and iOS platforms

param(
    [Parameter(Position=0)]
    [ValidateSet('android', 'ios', 'ios-universal', 'all')]
    [string]$Platform,
    
    [switch]$Debug,
    [string]$Features = "",
    [string]$Package = "tb-runtime",
    [switch]$Help
)

# Configuration
$BuildType = if ($Debug) { "debug" } else { "release" }

# Function to print colored messages
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if a target is installed
function Test-Target {
    param([string]$Target)
    $installedTargets = rustup target list --installed
    return $installedTargets -contains $Target
}

# Function to install a target
function Install-Target {
    param([string]$Target)
    Write-Info "Installing target: $Target"
    rustup target add $Target
}

# Function to build for a specific target
function Build-Target {
    param(
        [string]$Target,
        [string]$PlatformName
    )
    
    Write-Info "Building for $PlatformName ($Target)..."
    
    $buildArgs = @(
        "build",
        "--$BuildType",
        "--target", $Target,
        "-p", $Package
    )
    
    if ($Features) {
        $buildArgs += @("--features", $Features)
    }
    
    $process = Start-Process -FilePath "cargo" -ArgumentList $buildArgs -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -eq 0) {
        Write-Info "✓ Successfully built for $PlatformName"
        return $true
    } else {
        Write-Error-Custom "✗ Failed to build for $PlatformName"
        return $false
    }
}

# Function to build all Android targets
function Build-Android {
    Write-Info "=== Building for Android ==="
    
    $androidTargets = @(
        @{Target="aarch64-linux-android"; Name="ARM64"},
        @{Target="armv7-linux-androideabi"; Name="ARMv7"},
        @{Target="i686-linux-android"; Name="x86"},
        @{Target="x86_64-linux-android"; Name="x86_64"}
    )
    
    foreach ($targetInfo in $androidTargets) {
        $target = $targetInfo.Target
        $name = $targetInfo.Name
        
        if (-not (Test-Target $target)) {
            Write-Warn "Target $target not installed"
            Install-Target $target
        }
        
        Build-Target $target "Android $name"
    }
    
    Write-Info "=== Android build complete ==="
    Write-Info "Libraries located in:"
    foreach ($targetInfo in $androidTargets) {
        $target = $targetInfo.Target
        $libName = $Package -replace '-', '_'
        Write-Host "  target\$target\$BuildType\lib$libName.so"
    }
}

# Function to build all iOS targets
function Build-iOS {
    Write-Info "=== Building for iOS ==="
    
    # Check if running on macOS (not applicable for Windows, but kept for completeness)
    if ($env:OS -ne "Darwin") {
        Write-Warn "iOS builds are typically done on macOS, but cross-compilation may be possible"
    }
    
    $iosTargets = @(
        @{Target="aarch64-apple-ios"; Name="iOS Device (ARM64)"},
        @{Target="x86_64-apple-ios"; Name="iOS Simulator (Intel)"},
        @{Target="aarch64-apple-ios-sim"; Name="iOS Simulator (Apple Silicon)"}
    )
    
    foreach ($targetInfo in $iosTargets) {
        $target = $targetInfo.Target
        $name = $targetInfo.Name
        
        if (-not (Test-Target $target)) {
            Write-Warn "Target $target not installed"
            Install-Target $target
        }
        
        Build-Target $target $name
    }
    
    Write-Info "=== iOS build complete ==="
    Write-Info "Libraries located in:"
    foreach ($targetInfo in $iosTargets) {
        $target = $targetInfo.Target
        $libName = $Package -replace '-', '_'
        Write-Host "  target\$target\$BuildType\lib$libName.a"
    }
}

# Function to show usage
function Show-Usage {
    Write-Host @"
TB Language - Mobile Build Script (PowerShell)

Usage: .\build-mobile.ps1 [PLATFORM] [OPTIONS]

PLATFORMS:
    android         Build for all Android targets
    ios             Build for all iOS targets
    ios-universal   Build universal iOS libraries (macOS only)
    all             Build for all platforms (Android + iOS)

OPTIONS:
    -Debug          Build in debug mode (default: release)
    -Features       Enable specific features (e.g., "mobile")
    -Package        Package to build (default: tb-runtime)
    -Help           Show this help message

EXAMPLES:
    .\build-mobile.ps1 android                      # Build for Android (release)
    .\build-mobile.ps1 ios                          # Build for iOS (release)
    .\build-mobile.ps1 all                          # Build for all platforms
    .\build-mobile.ps1 android -Debug               # Build for Android (debug)
    .\build-mobile.ps1 android -Features mobile     # Build with mobile features
    .\build-mobile.ps1 android -Package tb-cli      # Build tb-cli for Android

REQUIREMENTS:
    - Rust toolchain (rustup)
    - Android NDK (for Android builds)
    - Xcode (for iOS builds, macOS only)
    - Configured .cargo/config.toml with NDK paths

For more information, see MOBILE_BUILD.md
"@
}

# Main execution
if ($Help) {
    Show-Usage
    exit 0
}

if (-not $Platform) {
    Write-Error-Custom "No platform specified"
    Show-Usage
    exit 1
}

Write-Info "TB Language Mobile Build Script"
Write-Info "Package: $Package"
Write-Info "Build type: $BuildType"
if ($Features) {
    Write-Info "Features: $Features"
}
Write-Host ""

switch ($Platform) {
    'android' {
        Build-Android
    }
    'ios' {
        Build-iOS
    }
    'ios-universal' {
        Write-Error-Custom "Universal iOS builds require macOS and the lipo tool"
        Write-Info "Please use build-mobile.sh on macOS for universal iOS builds"
        exit 1
    }
    'all' {
        Build-Android
        Write-Host ""
        Build-iOS
    }
}

Write-Host ""
Write-Info "Build complete! 🚀"

