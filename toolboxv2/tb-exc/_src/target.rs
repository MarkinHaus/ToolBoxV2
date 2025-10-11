// File: tb_lang/src/target.rs

use std::env;
use std::fmt;
use std::str::FromStr;

/// Supported compilation targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPlatform {
    /// Linux x86_64
    LinuxX64,
    /// Linux ARM64
    LinuxArm64,
    /// Windows x86_64
    WindowsX64,
    /// Windows ARM64
    WindowsArm64,
    /// macOS x86_64 (Intel)
    MacOSX64,
    /// macOS ARM64 (Apple Silicon)
    MacOSArm64,
    /// WebAssembly
    Wasm,
    /// Android ARM64
    AndroidArm64,
    /// Android x86_64
    AndroidX64,
    /// iOS ARM64
    IOSArm64,
    /// iOS Simulator x86_64
    IOSSimulatorX64,
}

impl TargetPlatform {
    /// Detect current platform
    pub fn current() -> Self {
        let os = env::consts::OS;
        let arch = env::consts::ARCH;

        match (os, arch) {
            ("linux", "x86_64") => TargetPlatform::LinuxX64,
            ("linux", "aarch64") => TargetPlatform::LinuxArm64,
            ("windows", "x86_64") => TargetPlatform::WindowsX64,
            ("windows", "aarch64") => TargetPlatform::WindowsArm64,
            ("macos", "x86_64") => TargetPlatform::MacOSX64,
            ("macos", "aarch64") => TargetPlatform::MacOSArm64,
            _ => {
                eprintln!("⚠️  Unknown platform {}/{}, defaulting to Linux x64", os, arch);
                TargetPlatform::LinuxX64
            }
        }
    }

    /// Get Rust target triple
    pub fn rust_target(&self) -> &'static str {
        match self {
            TargetPlatform::LinuxX64 => "x86_64-unknown-linux-gnu",
            TargetPlatform::LinuxArm64 => "aarch64-unknown-linux-gnu",
            TargetPlatform::WindowsX64 => "x86_64-pc-windows-msvc",
            TargetPlatform::WindowsArm64 => "aarch64-pc-windows-msvc",
            TargetPlatform::MacOSX64 => "x86_64-apple-darwin",
            TargetPlatform::MacOSArm64 => "aarch64-apple-darwin",
            TargetPlatform::Wasm => "wasm32-unknown-unknown",
            TargetPlatform::AndroidArm64 => "aarch64-linux-android",
            TargetPlatform::AndroidX64 => "x86_64-linux-android",
            TargetPlatform::IOSArm64 => "aarch64-apple-ios",
            TargetPlatform::IOSSimulatorX64 => "x86_64-apple-ios",
        }
    }

    /// Get executable extension
    pub fn exe_extension(&self) -> &'static str {
        match self {
            TargetPlatform::WindowsX64 | TargetPlatform::WindowsArm64 => ".exe",
            TargetPlatform::Wasm => ".wasm",
            _ => "",
        }
    }

    /// Get shared library extension
    pub fn lib_extension(&self) -> &'static str {
        match self {
            TargetPlatform::WindowsX64 | TargetPlatform::WindowsArm64 => ".dll",
            TargetPlatform::MacOSX64 | TargetPlatform::MacOSArm64
            | TargetPlatform::IOSArm64 | TargetPlatform::IOSSimulatorX64 => ".dylib",
            _ => ".so",
        }
    }

    /// Check if target is cross-compilation
    pub fn is_cross_compile(&self) -> bool {
        *self != Self::current()
    }

    /// Get platform name for output directory
    pub fn platform_name(&self) -> &'static str {
        match self {
            TargetPlatform::LinuxX64 => "linux-x64",
            TargetPlatform::LinuxArm64 => "linux-arm64",
            TargetPlatform::WindowsX64 => "windows-x64",
            TargetPlatform::WindowsArm64 => "windows-arm64",
            TargetPlatform::MacOSX64 => "macos-x64",
            TargetPlatform::MacOSArm64 => "macos-arm64",
            TargetPlatform::Wasm => "wasm",
            TargetPlatform::AndroidArm64 => "android-arm64",
            TargetPlatform::AndroidX64 => "android-x64",
            TargetPlatform::IOSArm64 => "ios-arm64",
            TargetPlatform::IOSSimulatorX64 => "ios-simulator-x64",
        }
    }

    /// Check if target supports multi-threading
    pub fn supports_threading(&self) -> bool {
        !matches!(self, TargetPlatform::Wasm)
    }

    /// Check if target supports filesystem
    pub fn supports_filesystem(&self) -> bool {
        !matches!(self, TargetPlatform::Wasm)
    }

    /// Get compiler flags for optimization
    pub fn optimization_flags(&self) -> Vec<String> {
        let mut flags = vec![
            "-C".to_string(),
            "opt-level=3".to_string(),
            "-C".to_string(),
            "lto=fat".to_string(),
            "-C".to_string(),
            "codegen-units=1".to_string(),
            "-C".to_string(),
            "panic=abort".to_string(),
        ];

        // Target-specific optimizations
        if !self.is_cross_compile() {
            flags.extend([
                "-C".to_string(),
                "target-cpu=native".to_string(),
            ]);
        }

        // Strip symbols for release
        flags.extend([
            "-C".to_string(),
            "strip=symbols".to_string(),
        ]);

        flags
    }
}

impl fmt::Display for TargetPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.platform_name())
    }
}

impl FromStr for TargetPlatform {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "linux-x64" | "linux" | "linux-x86_64" => Ok(TargetPlatform::LinuxX64),
            "linux-arm64" | "linux-aarch64" => Ok(TargetPlatform::LinuxArm64),
            "windows-x64" | "windows" | "win64" => Ok(TargetPlatform::WindowsX64),
            "windows-arm64" => Ok(TargetPlatform::WindowsArm64),
            "macos-x64" | "macos" | "darwin" => Ok(TargetPlatform::MacOSX64),
            "macos-arm64" | "macos-m1" | "apple-silicon" => Ok(TargetPlatform::MacOSArm64),
            "wasm" | "wasm32" | "web" => Ok(TargetPlatform::Wasm),
            "android-arm64" | "android" => Ok(TargetPlatform::AndroidArm64),
            "android-x64" => Ok(TargetPlatform::AndroidX64),
            "ios-arm64" | "ios" => Ok(TargetPlatform::IOSArm64),
            "ios-simulator" => Ok(TargetPlatform::IOSSimulatorX64),
            _ => Err(format!("Unknown target platform: {}", s)),
        }
    }
}

/// Compilation configuration
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    pub target: TargetPlatform,
    pub optimize: bool,
    pub strip: bool,
    pub output_name: String,
    pub include_runtime: bool,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            target: TargetPlatform::current(),
            optimize: true,
            strip: true,
            output_name: "app".to_string(),
            include_runtime: true,
        }
    }
}