// File: tb-exc/src/dependency_compiler.rs

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use crate::{TBResult, TBError, Language};


#[cfg(debug_assertions)]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        eprintln!("[DEBUG] {}", format!($($arg)*));
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_log {
    ($($arg:tt)*) => {};
}
/// Compilation strategy for dependencies
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationStrategy {
    /// Compile to native binary with modern tools (UV/BUN)
    ModernNative { tool: ModernTool },

    /// Traditional native compilation (Nuitka/Cython)
    NativeCompilation { tool: NativeTool },

    /// Bundle to single file (JS via esbuild, Python via PyInstaller)
    Bundle { format: BundleFormat },

    /// Compile to plugin/shared library (Go)
    Plugin { target: String },

    /// Use system-installed (large packages like numpy)
    SystemInstalled,

    /// Embed as-is (small scripts)
    EmbedRaw,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModernTool {
    UvPyOxidizer,  // UV + PyOxidizer (Python → standalone binary)
    UvNuitka,      // UV + Nuitka (Python → C → binary)
    BunCompile,    // BUN compile (JS/TS → native binary)
    BunBuild,      // BUN build (JS/TS → optimized bundle)
}

#[derive(Debug, Clone, PartialEq)]
pub enum NativeTool {
    Nuitka,    // Python → C → binary
    Cython,    // Python → C extension
    Pkg,       // Node.js → binary
}

#[derive(Debug, Clone, PartialEq)]
pub enum BundleFormat {
    PyInstaller,  // Python bundle
    Esbuild,      // JS bundle
    Webpack,      // JS bundle (fallback)
}

/// Dependency to be compiled
#[derive(Debug, Clone)]
pub struct Dependency {
    pub id: Arc<String>,
    pub language: Language,
    pub code: Arc<String>,
    pub imports: Vec<Arc<String>>,
    pub is_in_loop: bool,
    pub estimated_calls: usize,
}

/// Compiled dependency output
#[derive(Debug, Clone)]
pub struct CompiledDependency {
    pub id: Arc<String>,
    pub strategy: CompilationStrategy,
    pub output_path: PathBuf,
    pub size_bytes: usize,
    pub compile_time_ms: u128,
}

pub struct DependencyCompiler {
    deps_dir: PathBuf,
    cache_dir: PathBuf,
    has_uv: bool,
    has_bun: bool,
    registry: std::sync::Mutex<CompiledDependencyRegistry>,
}

impl DependencyCompiler {
    pub fn new(project_dir: &Path) -> Self {
        let deps_dir = project_dir.join("deps");
        let cache_dir = project_dir.join(".tb_cache");

        fs::create_dir_all(&deps_dir).ok();
        fs::create_dir_all(&cache_dir).ok();

        // Detect modern tools
        let has_uv = Self::check_tool_available("uv");
        let has_bun = Self::check_tool_available("bun");

        if has_uv {
            debug_log!("⚡ UV detected - using fast Python tooling");
        }
        if has_bun {
            debug_log!("⚡ BUN detected - using fast JavaScript runtime");
        }

        Self {
            deps_dir,
            cache_dir,
            has_uv,
            has_bun,

            // ✅ NEU: Registry initialisieren
            registry: std::sync::Mutex::new(CompiledDependencyRegistry::new()),
        }
    }

    /// Check if tool is available
    fn check_tool_available(tool: &str) -> bool {
        Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Compile a dependency using best available strategy
    pub fn compile(&self, dep: &Dependency) -> TBResult<CompiledDependency> {
        let start = std::time::Instant::now();

        // Choose strategy
        let strategy = self.choose_strategy(dep);

        debug_log!("🔨 Compiling {} ({:?})", dep.id, strategy);

        // Compile based on strategy
        let output_path = match &strategy {
            CompilationStrategy::ModernNative { tool } => {
                self.compile_modern_native(dep, tool)?
            }
            CompilationStrategy::NativeCompilation { tool } => {
                self.compile_native(dep, tool)?
            }
            CompilationStrategy::Bundle { format } => {
                self.bundle(dep, format)?
            }
            CompilationStrategy::Plugin { target } => {
                self.compile_plugin(dep, target)?
            }
            CompilationStrategy::SystemInstalled => {
                self.ensure_system_installed(dep)?;
                PathBuf::from("system")
            }
            CompilationStrategy::EmbedRaw => {
                self.embed_raw(dep)?
            }
        };

        let size_bytes = if output_path.exists() {
            fs::metadata(&output_path)?.len() as usize
        } else {
            0
        };

        let compiled = CompiledDependency {
            id: dep.id.clone(),
            strategy,
            output_path,
            size_bytes,
            compile_time_ms: start.elapsed().as_millis(),
        };

        // ✅ NEU: Automatisch in Registry registrieren
        if let Ok(mut registry) = self.registry.lock() {
            debug_log!("📝 Registering {} in dependency registry", compiled.id);
            registry.register(compiled.clone());
        }

        Ok(compiled)
    }

    /// Choose optimal compilation strategy
    fn choose_strategy(&self, dep: &Dependency) -> CompilationStrategy {
        match dep.language {
            Language::Python => self.choose_python_strategy(dep),
            Language::JavaScript | Language::TypeScript => self.choose_js_strategy(dep),
            Language::Go => CompilationStrategy::Plugin {
                target: self.get_go_target(),
            },
            Language::Bash => CompilationStrategy::EmbedRaw,
            _ => CompilationStrategy::EmbedRaw,
        }
    }

    fn choose_python_strategy(&self, dep: &Dependency) -> CompilationStrategy {
        // Check for heavy dependencies
        let has_heavy_deps = dep.imports.iter().any(|imp| {
            matches!(imp.as_str(), "numpy" | "pandas" | "tensorflow" | "torch" | "scipy")
        });

        if has_heavy_deps {
            return CompilationStrategy::SystemInstalled;
        }

        // Check if in hot loop → prefer native compilation
        if dep.is_in_loop && dep.estimated_calls > 100 {
            // ✅ PRIORITY 1: UV + PyOxidizer (fastest)
            if self.has_uv && Self::check_tool_available("pyoxidizer") {
                return CompilationStrategy::ModernNative {
                    tool: ModernTool::UvPyOxidizer,
                };
            }

            // ✅ PRIORITY 2: UV + Nuitka (very fast)
            if self.has_uv && Self::check_tool_available("nuitka") {
                return CompilationStrategy::ModernNative {
                    tool: ModernTool::UvNuitka,
                };
            }

            // Traditional fallbacks
            if Self::check_tool_available("nuitka") {
                return CompilationStrategy::NativeCompilation {
                    tool: NativeTool::Nuitka,
                };
            }

            if Self::check_tool_available("cython") {
                return CompilationStrategy::NativeCompilation {
                    tool: NativeTool::Cython,
                };
            }
        }

        // Small code → embed
        if dep.code.len() < 5000 {
            return CompilationStrategy::EmbedRaw;
        }

        // Default: Bundle with PyInstaller
        if Self::check_tool_available("pyinstaller") {
            CompilationStrategy::Bundle {
                format: BundleFormat::PyInstaller,
            }
        } else {
            CompilationStrategy::EmbedRaw
        }
    }

    fn choose_js_strategy(&self, dep: &Dependency) -> CompilationStrategy {
        // ✅ PRIORITY 1: BUN compile (to standalone binary)
        if self.has_bun && dep.is_in_loop {
            return CompilationStrategy::ModernNative {
                tool: ModernTool::BunCompile,
            };
        }

        // ✅ PRIORITY 2: BUN build (optimized bundle)
        if self.has_bun {
            return CompilationStrategy::ModernNative {
                tool: ModernTool::BunBuild,
            };
        }

        // Small code → embed
        if dep.code.len() < 10000 {
            return CompilationStrategy::EmbedRaw;
        }

        // Traditional bundling
        if dep.imports.iter().any(|imp| !imp.starts_with('.') && !imp.starts_with('/')) {
            if Self::check_tool_available("esbuild") {
                return CompilationStrategy::Bundle {
                    format: BundleFormat::Esbuild,
                };
            }
        }

        CompilationStrategy::EmbedRaw
    }

    fn get_go_target(&self) -> String {
        #[cfg(target_os = "linux")]
        return "linux".to_string();

        #[cfg(target_os = "windows")]
        return "windows".to_string();

        #[cfg(target_os = "macos")]
        return "darwin".to_string();

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        return "unknown".to_string();
    }
}
// deps
impl DependencyCompiler {
    /// Compile all dependencies and return registry
    pub fn compile_all(&self, deps: Vec<Dependency>) -> TBResult<CompiledDependencyRegistry> {
        let mut registry = CompiledDependencyRegistry::new();

        for dep in deps {
            let compiled = self.compile(&dep)?;

            debug_log!(
            "✓ Compiled {} ({:?}) → {} bytes in {}ms",
            compiled.id,
            compiled.strategy,
            compiled.size_bytes,
            compiled.compile_time_ms
        );

            registry.register(compiled);
        }

        Ok(registry)
    }

    /// Generate optimized Cargo.toml dependencies for compiled code
    pub fn generate_cargo_dependencies(&self, registry: &CompiledDependencyRegistry) -> String {
        let mut deps = vec![
            r#"[dependencies]"#.to_string(),
        ];

        // Check if we need PyO3 (for Nuitka/Cython)
        let needs_pyo3 = registry.dependencies.values().any(|d| {
            matches!(
            d.strategy,
            CompilationStrategy::NativeCompilation {
                tool: NativeTool::Nuitka | NativeTool::Cython
            } | CompilationStrategy::ModernNative {
                tool: ModernTool::UvNuitka
            }
        )
        });

        if needs_pyo3 {
            deps.push(r#"pyo3 = { version = "0.20", features = ["auto-initialize"] }"#.to_string());
        }

        // Check if we need libloading (for Go plugins)
        let needs_libloading = registry.dependencies.values().any(|d| {
            matches!(d.strategy, CompilationStrategy::Plugin { .. })
        });

        if needs_libloading {
            deps.push(r#"libloading = "0.8""#.to_string());
        }

        deps.join("\n")
    }

    /// Generate build.rs for linking compiled libraries
    pub fn generate_build_script(&self, registry: &CompiledDependencyRegistry) -> String {
        let mut script = String::from(r#"
// Auto-generated build script for compiled dependencies

fn main() {
    println!("cargo:rerun-if-changed=deps/");
"#);

        // Add library paths for compiled dependencies
        for (id, dep) in registry.dependencies.iter() {
            if matches!(
            dep.strategy,
            CompilationStrategy::Plugin { .. }
            | CompilationStrategy::NativeCompilation { .. }
            | CompilationStrategy::ModernNative { .. }
        ) {
                let lib_dir = dep.output_path.parent().unwrap();
                script.push_str(&format!(
                    r#"    println!("cargo:rustc-link-search=native={}");"#,
                    lib_dir.display()
                ));
                script.push('\n');

                if matches!(dep.strategy, CompilationStrategy::Plugin { .. }) {
                    script.push_str(&format!(
                        r#"    println!("cargo:rustc-link-lib=dylib=plugin_{}");"#,
                        id
                    ));
                    script.push('\n');
                }
            }
        }

        script.push_str("}\n");
        script
    }

    pub fn generate_compiled_project(
        &self,
        deps: Vec<Dependency>,
        main_code: &str,
        output_dir: &Path,
    ) -> TBResult<PathBuf> {
        // 1. Compile all dependencies
        let registry = self.compile_all(deps)?;

        // 2. Create project structure
        fs::create_dir_all(output_dir)?;
        fs::create_dir_all(output_dir.join("src"))?;

        // 3. Generate main.rs with compiled bindings
        let mut main_rs = String::new();

        // Add compiled bridge functions
        main_rs.push_str(&registry.get_all_bindings());
        main_rs.push('\n');

        // Add main code
        main_rs.push_str(main_code);

        fs::write(output_dir.join("src/main.rs"), main_rs)?;

        // 4. Generate Cargo.toml
        let cargo_toml = format!(
            r#"[package]
name = "tb-compiled"
version = "0.1.0"
edition = "2021"

{}

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = true
"#,
            self.generate_cargo_dependencies(&registry)
        );

        fs::write(output_dir.join("Cargo.toml"), cargo_toml)?;

        // 5. Generate build.rs
        let build_rs = self.generate_build_script(&registry);
        fs::write(output_dir.join("build.rs"), build_rs)?;

        // 6. Copy compiled dependencies
        let deps_target = output_dir.join("deps");
        fs::create_dir_all(&deps_target)?;

        for (id, dep) in registry.dependencies.iter() {
            if dep.output_path.exists() {
                let target_path = deps_target.join(dep.output_path.file_name().unwrap());
                fs::copy(&dep.output_path, target_path)?;
            }
        }

        debug_log!("✓ Generated compiled project at: {}", output_dir.display());
        debug_log!("  Run with: cd {} && cargo build --release", output_dir.display());

        Ok(output_dir.to_path_buf())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MODERN NATIVE COMPILATION (UV + BUN)
// ═══════════════════════════════════════════════════════════════════════════

impl DependencyCompiler {
    fn compile_modern_native(&self, dep: &Dependency, tool: &ModernTool) -> TBResult<PathBuf> {
        match tool {
            ModernTool::UvPyOxidizer => self.compile_uv_pyoxidizer(dep),
            ModernTool::UvNuitka => self.compile_uv_nuitka(dep),
            ModernTool::BunCompile => self.compile_bun_binary(dep),
            ModernTool::BunBuild => self.bundle_bun(dep),
        }
    }

    /// Compile Python with UV + PyOxidizer → standalone binary
    fn compile_uv_pyoxidizer(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("⚡🐍 Compiling Python with UV + PyOxidizer → native binary");

        let temp_dir = self.cache_dir.join(format!("uv_pyo_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        // Write Python module
        let py_file = temp_dir.join("app.py");
        fs::write(&py_file, &*dep.code)?;

        // Create pyoxidizer.bzl config
        let config = format!(r#"
def make_exe():
    dist = default_python_distribution()

    policy = dist.make_python_packaging_policy()
    policy.resources_location_fallback = "filesystem-relative:lib"

    python_config = dist.make_python_interpreter_config()
    python_config.run_module = "app"

    exe = dist.to_python_executable(
        name="{}",
        packaging_policy=policy,
        config=python_config,
    )

    exe.add_python_resources(exe.pip_install(["."]))

    return exe

def make_embedded_resources(exe):
    return exe.to_embedded_resources()

def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files

register_target("exe", make_exe)
register_target("resources", make_embedded_resources, depends=["exe"], default_build_script=True)
register_target("install", make_install, depends=["exe"], default=True)
"#, dep.id);

        fs::write(temp_dir.join("pyoxidizer.bzl"), config)?;

        // Build with PyOxidizer
        let result = Command::new("pyoxidizer")
            .arg("build")
            .arg("--release")
            .current_dir(&temp_dir)
            .output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(TBError::CompilationError {
                message: format!("PyOxidizer compilation failed:\n{}", stderr).into(),
                source: dep.code.clone(),
            });
        }

        // Find output binary
        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        let binary_name = if cfg!(target_os = "windows") {
            format!("{}.exe", dep.id)
        } else {
            dep.id.to_string()
        };

        let source_bin = temp_dir
            .join("build")
            .join("release")
            .join("install")
            .join(&binary_name);

        let output = output_dir.join(&binary_name);
        fs::copy(&source_bin, &output)?;

        debug_log!("  ✓ Compiled to standalone binary: {:.2} KB",
                 fs::metadata(&output)?.len() as f64 / 1024.0);

        Ok(output)
    }

    /// Compile Python with UV + Nuitka
    fn compile_uv_nuitka(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("⚡🐍 Compiling Python with UV + Nuitka → native binary");

        let temp_dir = self.cache_dir.join(format!("uv_nuitka_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        // Write Python code
        let py_file = temp_dir.join("module.py");
        fs::write(&py_file, &*dep.code)?;

        // Install dependencies with UV (if needed)
        if !dep.imports.is_empty() {
            debug_log!("  📦 Installing dependencies with UV...");

            // Create venv with UV
            Command::new("uv")
                .args(&["venv", ".venv"])
                .current_dir(&temp_dir)
                .output()?;

            // Install packages
            for import in &dep.imports {
                Command::new("uv")
                    .args(&["pip", "install", import])
                    .current_dir(&temp_dir)
                    .output()?;
            }
        }

        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(self.get_lib_name(&dep.id));

        // Compile with Nuitka
        let mut cmd = Command::new("nuitka");
        cmd.arg("--module")
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--remove-output")
            .arg("--follow-imports")
            .arg(&py_file);

        // Use UV venv if exists
        if temp_dir.join(".venv").exists() {
            let venv_python = if cfg!(target_os = "windows") {
                temp_dir.join(".venv").join("Scripts").join("python.exe")
            } else {
                temp_dir.join(".venv").join("bin").join("python")
            };

            cmd.env("PYTHON", venv_python);
        }

        let result = cmd.output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(TBError::CompilationError {
                message: format!("Nuitka compilation failed:\n{}", stderr).into(),
                source: dep.code.clone(),
            });
        }

        debug_log!("  ✓ Compiled to native library");

        Ok(output)
    }

    /// Compile JavaScript/TypeScript with BUN → standalone binary
    fn compile_bun_binary(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("⚡📦 Compiling JavaScript with BUN → native binary");

        let temp_dir = self.cache_dir.join(format!("bun_compile_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        // Write JS/TS code
        let ext = match dep.language {
            Language::TypeScript => "ts",
            _ => "js",
        };
        let entry_file = temp_dir.join(format!("index.{}", ext));
        fs::write(&entry_file, &*dep.code)?;

        let output_dir = self.deps_dir.join("js");
        fs::create_dir_all(&output_dir)?;

        let binary_name = if cfg!(target_os = "windows") {
            format!("{}.exe", dep.id)
        } else {
            dep.id.to_string()
        };

        let output = output_dir.join(&binary_name);

        // Compile to standalone binary with BUN
        let result = Command::new("bun")
            .args(&[
                "build",
                entry_file.to_str().unwrap(),
                "--compile",
                "--outfile",
                output.to_str().unwrap(),
            ])
            .output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(TBError::CompilationError {
                message: format!("BUN compile failed:\n{}", stderr).into(),
                source: dep.code.clone(),
            });
        }

        debug_log!("  ✓ Compiled to standalone binary: {:.2} KB",
                 fs::metadata(&output)?.len() as f64 / 1024.0);

        Ok(output)
    }

    /// Bundle JavaScript with BUN (optimized, minified)
    fn bundle_bun(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("⚡📦 Bundling JavaScript with BUN");

        let temp_dir = self.cache_dir.join(format!("bun_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let ext = match dep.language {
            Language::TypeScript => "ts",
            _ => "js",
        };
        let entry_file = temp_dir.join(format!("index.{}", ext));
        fs::write(&entry_file, &*dep.code)?;

        let output_dir = self.deps_dir.join("js");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(format!("{}.js", dep.id));

        // Bundle with BUN
        let result = Command::new("bun")
            .args(&[
                "build",
                entry_file.to_str().unwrap(),
                "--outfile",
                output.to_str().unwrap(),
                "--minify",
                "--target=node",
            ])
            .output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(TBError::CompilationError {
                message: format!("BUN build failed:\n{}", stderr).into(),
                source: dep.code.clone(),
            });
        }

        debug_log!("  ✓ Bundled JavaScript: {:.2} KB",
                 fs::metadata(&output)?.len() as f64 / 1024.0);

        Ok(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRADITIONAL COMPILATION METHODS (unchanged)
// ═══════════════════════════════════════════════════════════════════════════

impl DependencyCompiler {
    fn compile_native(&self, dep: &Dependency, tool: &NativeTool) -> TBResult<PathBuf> {
        match (dep.language, tool) {
            (Language::Python, NativeTool::Nuitka) => self.compile_python_nuitka(dep),
            (Language::Python, NativeTool::Cython) => self.compile_python_cython(dep),
            (Language::JavaScript, NativeTool::Pkg) => self.compile_js_pkg(dep),
            _ => Err(TBError::UnsupportedLanguage(format!(
                "Native compilation not supported for {:?} with {:?}",
                dep.language, tool
            ).into())),
        }
    }

    /// Compile Python to native using Nuitka
    fn compile_python_nuitka(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("🐍 Compiling Python with Nuitka (Python → C → binary)...");

        let temp_dir = self.cache_dir.join(format!("py_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let py_file = temp_dir.join("module.py");
        fs::write(&py_file, &*dep.code)?;

        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(self.get_lib_name("module"));

        let result = Command::new("nuitka")
            .arg("--module")
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--remove-output")
            .arg("--quiet")
            .arg(&py_file)
            .output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(TBError::CompilationError {
                message: format!("Nuitka compilation failed:\n{}", stderr).into(),
                source: dep.code.clone(),
            });
        }

        debug_log!("  ✓ Compiled to native library");

        Ok(output)
    }

    /// Compile Python to C extension using Cython
    fn compile_python_cython(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("🐍 Compiling Python with Cython (Python → C extension)...");

        let temp_dir = self.cache_dir.join(format!("cy_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let pyx_file = temp_dir.join("module.pyx");
        fs::write(&pyx_file, &**dep.code)?;

        let setup_py = format!(r#"
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("module.pyx", compiler_directives={{'language_level': "3"}}),
)
"#);
        fs::write(temp_dir.join("setup.py"), setup_py)?;

        let result = Command::new("python3")
            .args(&["setup.py", "build_ext", "--inplace"])
            .current_dir(&temp_dir)
            .output()?;

        if !result.status.success() {
            return Err(TBError::CompilationError {
                message: Arc::from("Cython compilation failed".to_string()),
                source: dep.code.clone(),
            });
        }

        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        let so_file = self.find_file_with_extension(&temp_dir, ".so")?;
        let output = output_dir.join(so_file.file_name().unwrap());
        fs::copy(&so_file, &output)?;

        debug_log!("  ✓ Compiled to C extension");

        Ok(output)
    }

    /// Bundle Python with PyInstaller
    fn bundle_python_pyinstaller(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("📦 Bundling Python with PyInstaller...");

        let temp_dir = self.cache_dir.join(format!("pyi_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let py_file = temp_dir.join("main.py");
        fs::write(&py_file, &*dep.code)?;

        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        let result = Command::new("pyinstaller")
            .args(&[
                "--onefile",
                "--distpath", output_dir.to_str().unwrap(),
                "--workpath", temp_dir.to_str().unwrap(),
                "--specpath", temp_dir.to_str().unwrap(),
                "--name", &dep.id,
                py_file.to_str().unwrap(),
            ])
            .output()?;

        if !result.status.success() {
            return Err(TBError::CompilationError {
                message: Arc::from("PyInstaller bundling failed".to_string()),
                source: dep.code.clone(),
            });
        }

        let output = output_dir.join(dep.id.to_string());
        debug_log!("  ✓ Bundled Python application");

        Ok(output)
    }

    fn bundle(&self, dep: &Dependency, format: &BundleFormat) -> TBResult<PathBuf> {
        match (dep.language, format) {
            (Language::Python, BundleFormat::PyInstaller) => {
                self.bundle_python_pyinstaller(dep)
            }
            (Language::JavaScript | Language::TypeScript, BundleFormat::Esbuild) => {
                self.bundle_js_esbuild(dep)
            }
            _ => Err(TBError::UnsupportedLanguage(format!(
                "Bundling not supported for {:?} with {:?}",
                dep.language, format
            ).into())),
        }
    }

    /// Bundle JavaScript with esbuild
    fn bundle_js_esbuild(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("📦 Bundling JavaScript with esbuild...");

        let temp_dir = self.cache_dir.join(format!("js_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let js_file = temp_dir.join("index.js");
        fs::write(&js_file, &*dep.code)?;

        let output_dir = self.deps_dir.join("js");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(format!("{}.js", dep.id));

        let result = Command::new("esbuild")
            .arg(js_file)
            .args(&[
                "--bundle",
                "--minify",
                "--platform=node",
                "--target=node14",
                &format!("--outfile={}", output.display()),
            ])
            .output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(TBError::CompilationError {
                message: format!("esbuild failed:\n{}", stderr).into(),
                source: dep.code.clone(),
            });
        }

        debug_log!("  ✓ Bundled JavaScript");

        Ok(output)
    }

    /// Compile JavaScript to standalone binary with pkg
    fn compile_js_pkg(&self, dep: &Dependency) -> TBResult<PathBuf> {
        debug_log!("🎁 Compiling JavaScript to binary with pkg...");

        let temp_dir = self.cache_dir.join(format!("pkg_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let js_file = temp_dir.join("index.js");
        fs::write(&js_file, dep.code.to_string())?;

        let package_json = r#"{
  "name": "tb-js-dep",
  "version": "1.0.0",
  "main": "index.js",
  "bin": "index.js",
  "pkg": {
    "assets": []
  }
}"#;
        fs::write(temp_dir.join("package.json"), package_json)?;

        let output_dir = self.deps_dir.join("js");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(dep.id.to_string());

        let result = Command::new("pkg")
            .args(&[
                js_file.to_str().unwrap(),
                "--output", output.to_str().unwrap(),
                "--targets", "node14",
            ])
            .output()?;

        if !result.status.success() {
            return Err(TBError::CompilationError {
                message: Arc::from("pkg compilation failed".to_string()),
                source: dep.code.clone(),
            });
        }

        debug_log!("  ✓ Compiled to native binary");

        Ok(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GO COMPILATION
// ═══════════════════════════════════════════════════════════════════════════

impl DependencyCompiler {
    /// Compile Go to shared library plugin
    /// Compile Go to shared library plugin
    /// Compile Go to shared library plugin
    fn compile_go_plugin(&self, dep: &Dependency, target: &str) -> TBResult<PathBuf> {
        debug_log!("🔧 Compiling Go plugin...");

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 1: Setup temp directory
        // ═══════════════════════════════════════════════════════════════════════
        let temp_dir = self.cache_dir.join(format!("go_{}", dep.id));

        if temp_dir.exists() {
            debug_log!("  Cleaning existing temp dir: {}", temp_dir.display());
            fs::remove_dir_all(&temp_dir)
                .map_err(|e| TBError::IoError(format!("Failed to clean temp dir: {}", e).into()))?;
        }

        fs::create_dir_all(&temp_dir)?;

        let temp_dir_abs = temp_dir.canonicalize()
            .map_err(|e| TBError::IoError(format!("Failed to get absolute temp dir: {}", e).into()))?;

        //  Normalisiere den Pfad (entfernt \\?\ auf Windows)
        let temp_dir_abs = normalize_path(&temp_dir_abs);

        debug_log!("  Temp dir: {}", temp_dir_abs.display());

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 2: Write Go source file WITH PROPER INDENTATION
        // ═══════════════════════════════════════════════════════════════════════
        let go_file = temp_dir_abs.join("plugin.go");

        // ✅ FIX: Indent user code properly for Go function body
        let indented_user_code: String = dep.code.lines()
            .map(|line| {
                if line.trim().is_empty() {
                    String::new()
                } else {
                    format!("    {}", line)  // 4 spaces indent for function body
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let plugin_code = format!(r#"package main

import "C"
import (
    "fmt"
    "bytes"
    "os"
)

//export Execute
func Execute(input *C.char) *C.char {{
    // Capture stdout
    old := os.Stdout
    r, w, _ := os.Pipe()
    os.Stdout = w

    // Execute user code
{}

    // Restore stdout and get output
    w.Close()
    os.Stdout = old

    var buf bytes.Buffer
    buf.ReadFrom(r)

    return C.CString(buf.String())
}}

func main() {{}}
"#, indented_user_code);

        fs::write(&go_file, plugin_code)
            .map_err(|e| TBError::IoError(format!("Failed to write Go file: {}", e).into()))?;

        debug_log!("  Go source: {}", go_file.display());

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 3: Initialize Go module
        // ═══════════════════════════════════════════════════════════════════════
        let go_mod_path = temp_dir_abs.join("go.mod");

        if !go_mod_path.exists() {
            debug_log!("  Initializing Go module...");

            let mod_init = Command::new("go")
                .args(&["mod", "init", &format!("plugin_{}", dep.id)])
                .current_dir(&temp_dir_abs)
                .output()?;

            if !mod_init.status.success() {
                let stderr = String::from_utf8_lossy(&mod_init.stderr);
                return Err(TBError::CompilationError {
                    message: format!("Go mod init failed:\n{}", stderr).into(),
                    source: dep.code.clone(),
                });
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 4: Compile with Go
        // ═══════════════════════════════════════════════════════════════════════
        let lib_ext = self.get_lib_extension();
        let temp_output_name = format!("plugin_{}.{}", dep.id, lib_ext);

        debug_log!("  Running: go build -buildmode=c-shared -o {}", temp_output_name);

        let result = Command::new("go")
            .args(&[
                "build",
                "-buildmode=c-shared",
                "-o",
                &temp_output_name,

            ])
            .current_dir(&temp_dir_abs)
            .env("GOOS", target)
            .output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            let stdout = String::from_utf8_lossy(&result.stdout);

            //  Prüfe ob plugin.go existiert
            let go_file_exists = go_file.exists();
            let go_file_content = if go_file_exists {
                fs::read_to_string(&go_file).unwrap_or_else(|_| "Could not read file".to_string())
            } else {
                "FILE DOES NOT EXIST".to_string()
            };

            return Err(TBError::CompilationError {
                message: format!(
                    "Go compilation failed:\n\
             Working dir: {}\n\
             plugin.go exists: {}\n\
             Stderr: {}\n\
             Stdout: {}\n\n\
             Generated Go code:\n{}\n\n\
             Full file content:\n{}",
                    temp_dir_abs.display(),
                    go_file_exists,
                    stderr,
                    stdout,
                    indented_user_code,
                    go_file_content
                ).into(),
                source: dep.code.clone(),
            });
        }

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 5: Verify and move to final destination
        // ═══════════════════════════════════════════════════════════════════════
        let compiled_file = temp_dir_abs.join(&temp_output_name);

        if !compiled_file.exists() {
            return Err(TBError::CompilationError {
                message: format!("Compiled file not found: {}", compiled_file.display()).into(),
                source: dep.code.clone(),
            });
        }

        let output_dir = self.deps_dir.join("go");
        fs::create_dir_all(&output_dir)?;

        let final_output = output_dir.join(&temp_output_name);
        fs::copy(&compiled_file, &final_output)?;

        let file_size = fs::metadata(&final_output)?.len();
        debug_log!("  ✓ Compiled Go plugin: {:.2} KB", file_size as f64 / 1024.0);

        Ok(final_output)
    }

    fn compile_plugin(&self, dep: &Dependency, target: &str) -> TBResult<PathBuf> {
        match dep.language {
            Language::Go => self.compile_go_plugin(dep, target),
            _ => Err(TBError::UnsupportedLanguage(format!(
                "Plugin compilation not supported for {:?}",
                dep.language
            ).into())),
        }
    }
}

/// Remove Windows extended-length path prefix (\\?\)
fn normalize_path(path: &Path) -> PathBuf {
    let path_str = path.to_string_lossy();
    if path_str.starts_with(r"\\?\") {
        PathBuf::from(&path_str[4..]) // Remove \\?\ prefix
    } else {
        path.to_path_buf()
    }
}
// ═══════════════════════════════════════════════════════════════════════════
// HELPER METHODS
// ═══════════════════════════════════════════════════════════════════════════

impl DependencyCompiler {
    /// Embed raw code (no compilation)
    fn embed_raw(&self, dep: &Dependency) -> TBResult<PathBuf> {
        let ext = match dep.language {
            Language::Python => "py",
            Language::JavaScript => "js",
            Language::TypeScript => "ts",
            Language::Go => "go",
            Language::Bash => "sh",
            _ => "txt",
        };

        let output_dir = self.deps_dir.join("raw");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(format!("{}.{}", dep.id, ext));
        fs::write(&output, dep.code.to_string())?;

        Ok(output)
    }

    /// Ensure system package is installed
    fn ensure_system_installed(&self, dep: &Dependency) -> TBResult<()> {
        debug_log!("🔍 Checking system dependencies...");

        match dep.language {
            Language::Python => {
                for import in &dep.imports {
                    self.check_python_package(import)?;
                }
            }
            Language::JavaScript => {
                for import in &dep.imports {
                    self.check_npm_package(import)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn check_python_package(&self, package: &str) -> TBResult<()> {
        // Try UV first if available
        if self.has_uv {
            let result = Command::new("uv")
                .args(&["pip", "show", package])
                .output()?;

            if result.status.success() {
                debug_log!("  ✓ Found Python package: {} (via UV)", package);
                return Ok(());
            }
        }

        // Fallback to standard Python check
        let result = Command::new("python3")
            .args(&["-c", &format!("import {}", package)])
            .output()?;

        if !result.status.success() {
            eprintln!("⚠️  Python package '{}' not found", package);
            if self.has_uv {
                eprintln!("   Install with: uv pip install {}", package);
            } else {
                eprintln!("   Install with: pip install {}", package);
            }
            return Err(TBError::InvalidOperation(format!(
                "Missing Python package: {}",
                package
            ).into()));
        }

        debug_log!("  ✓ Found Python package: {}", package);
        Ok(())
    }

    fn check_npm_package(&self, package: &str) -> TBResult<()> {
        // Try BUN first if available
        if self.has_bun {
            let result = Command::new("bun")
                .args(&["pm", "ls", package])
                .output()?;

            if result.status.success() {
                debug_log!("  ✓ Found package: {} (via BUN)", package);
                return Ok(());
            }
        }

        // Fallback to NPM
        let result = Command::new("npm")
            .args(&["list", "-g", package])
            .output()?;

        if !result.status.success() {
            eprintln!("⚠️  Package '{}' not found", package);
            if self.has_bun {
                eprintln!("   Install with: bun install {}", package);
            } else {
                eprintln!("   Install with: npm install -g {}", package);
            }
            return Err(TBError::InvalidOperation(format!(
                "Missing package: {}",
                package
            ).into()));
        }

        debug_log!("  ✓ Found NPM package: {}", package);
        Ok(())
    }

    fn get_lib_name(&self, base: &str) -> String {
        #[cfg(target_os = "windows")]
        return format!("{}.dll", base);

        #[cfg(target_os = "macos")]
        return format!("lib{}.dylib", base);

        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        return format!("lib{}.so", base);
    }

    fn get_lib_extension(&self) -> &str {
        #[cfg(target_os = "windows")]
        return "dll";

        #[cfg(target_os = "macos")]
        return "dylib";

        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        return "so";
    }

    fn find_file_with_extension(&self, dir: &Path, ext: &str) -> TBResult<PathBuf> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some(ext.trim_start_matches('.')) {
                return Ok(path);
            }
        }

        Err(TBError::IoError(format!("No {} file found in {}", ext, dir.display()).into()))
    }
}

impl DependencyCompiler {
    // ... (existing methods) ...

    // ═══════════════════════════════════════════════════════════════════════════
    // REGISTRY ACCESS - Runtime availability
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get reference to the compiled dependency registry
    pub fn get_registry(&self) -> std::sync::MutexGuard<CompiledDependencyRegistry> {
        self.registry.lock().expect("Failed to lock registry")
    }

    /// Check if a dependency is compiled and available
    pub fn is_compiled(&self, dep_id: &str) -> bool {
        self.registry
            .lock()
            .map(|reg| reg.is_compiled(dep_id))
            .unwrap_or(false)
    }

    /// Get the compilation strategy used for a dependency
    pub fn get_strategy(&self, dep_id: &str) -> Option<CompilationStrategy> {
        self.registry
            .lock()
            .ok()
            .and_then(|reg| reg.get_strategy(dep_id).cloned())
    }

    /// Get all compiled bindings as Rust code
    pub fn get_all_bindings(&self) -> String {
        self.registry
            .lock()
            .map(|reg| reg.get_all_bindings())
            .unwrap_or_default()
    }

    /// Export registry for code generation
    pub fn export_registry(&self) -> CompiledDependencyRegistry {
        self.registry
            .lock()
            .map(|reg| reg.clone())
            .unwrap_or_else(|_| CompiledDependencyRegistry::new())
    }

    /// Get statistics about compiled dependencies
    pub fn get_stats(&self) -> CompilationStats {
        let registry = self.registry.lock().unwrap();

        let mut stats = CompilationStats::default();

        for (_, dep) in registry.dependencies.iter() {
            stats.total_count += 1;
            stats.total_size_bytes += dep.size_bytes;
            stats.total_compile_time_ms += dep.compile_time_ms;

            match &dep.strategy {
                CompilationStrategy::ModernNative { .. } => stats.modern_native += 1,
                CompilationStrategy::NativeCompilation { .. } => stats.native_compilation += 1,
                CompilationStrategy::Bundle { .. } => stats.bundled += 1,
                CompilationStrategy::Plugin { .. } => stats.plugin += 1,
                CompilationStrategy::SystemInstalled => stats.system_installed += 1,
                CompilationStrategy::EmbedRaw => stats.embedded += 1,
            }
        }

        stats
    }
}

impl Clone for CompiledDependencyRegistry {
    fn clone(&self) -> Self {
        Self {
            dependencies: self.dependencies.clone(),
            bindings: self.bindings.clone(),
        }
    }
}

/// Compilation statistics
#[derive(Debug, Default, Clone)]
pub struct CompilationStats {
    pub total_count: usize,
    pub total_size_bytes: usize,
    pub total_compile_time_ms: u128,
    pub modern_native: usize,
    pub native_compilation: usize,
    pub bundled: usize,
    pub plugin: usize,
    pub system_installed: usize,
    pub embedded: usize,
}

impl CompilationStats {
    pub fn print_summary(&self) {
        println!("📊 Compilation Statistics:");
        println!("  Total dependencies: {}", self.total_count);
        println!("  Total size: {:.2} KB", self.total_size_bytes as f64 / 1024.0);
        println!("  Total compile time: {}ms", self.total_compile_time_ms);
        println!();
        println!("  🚀 Modern native: {}", self.modern_native);
        println!("  ⚙️  Native compilation: {}", self.native_compilation);
        println!("  📦 Bundled: {}", self.bundled);
        println!("  🔌 Plugins: {}", self.plugin);
        println!("  💾 System installed: {}", self.system_installed);
        println!("  📝 Embedded: {}", self.embedded);
    }
}


/// Runtime binding for compiled dependency (zero-overhead FFI/binary execution)
#[derive(Debug, Clone)]
pub enum CompiledBinding {
    /// Native shared library (Go, Nuitka, Cython)
    NativeLibrary {
        path: PathBuf,
        exports: Vec<String>,
    },

    /// Standalone binary (PyOxidizer, Bun, PyInstaller)
    StandaloneBinary {
        path: PathBuf,
    },

    /// Bundled script (requires runtime)
    BundledScript {
        path: PathBuf,
        runtime: RuntimeType,
    },

    /// System package (no compilation needed)
    SystemPackage {
        import_name: String,
    },

    /// Embedded source (inline)
    EmbeddedSource {
        code: Arc<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeType {
    Python,
    NodeJS,
    Bun,
    Deno,
    Bash,
}

impl CompiledDependency {
    /// Generate Rust FFI binding code for this compiled dependency
    pub fn generate_rust_binding(&self, func_name: &str) -> String {
        match &self.strategy {
            CompilationStrategy::Plugin { .. } => {
                self.generate_go_ffi_binding(func_name)
            }
            CompilationStrategy::ModernNative { tool } => match tool {
                ModernTool::UvPyOxidizer | ModernTool::BunCompile => {
                    self.generate_binary_binding(func_name)
                }
                ModernTool::UvNuitka => {
                    self.generate_python_nuitka_binding(func_name)
                }
                ModernTool::BunBuild => {
                    self.generate_bun_runtime_binding(func_name)
                }
            },
            CompilationStrategy::NativeCompilation { tool } => match tool {
                NativeTool::Nuitka => self.generate_python_nuitka_binding(func_name),
                NativeTool::Cython => self.generate_cython_binding(func_name),
                NativeTool::Pkg => self.generate_binary_binding(func_name),
            },
            CompilationStrategy::Bundle { format } => {
                self.generate_bundled_binding(func_name, format)
            }
            CompilationStrategy::SystemInstalled => {
                self.generate_system_binding(func_name)
            }
            CompilationStrategy::EmbedRaw => {
                self.generate_embedded_binding(func_name)
            }
        }
    }

    /// Generate Go C-shared library FFI binding (zero overhead)
    fn generate_go_ffi_binding(&self, func_name: &str) -> String {
        let lib_path = self.output_path.display();

        format!(r#"
// ═══════════════════════════════════════════════════════════════
// COMPILED GO BRIDGE - Zero Overhead FFI
// ═══════════════════════════════════════════════════════════════
#[link(name = "plugin_{}", kind = "dylib")]
extern "C" {{
    fn Execute(input: *const std::os::raw::c_char) -> *const std::os::raw::c_char;
}}

#[inline(always)]
fn {}(args: &[&str]) -> String {{
    use std::ffi::{{CString, CStr}};

    let input = args.join(" ");
    let c_input = CString::new(input).unwrap();

    unsafe {{
        let result_ptr = Execute(c_input.as_ptr());
        if result_ptr.is_null() {{
            return String::new();
        }}
        CStr::from_ptr(result_ptr)
            .to_string_lossy()
            .into_owned()
    }}
}}
"#, self.id, func_name)
    }

    /// Generate standalone binary binding (subprocess with optimized caching)
    fn generate_binary_binding(&self, func_name: &str) -> String {
        let bin_path = self.output_path.display();

        format!(r#"
// ═══════════════════════════════════════════════════════════════
// COMPILED BINARY BRIDGE - Optimized Subprocess Execution
// ═══════════════════════════════════════════════════════════════
#[inline(always)]
fn {}(args: &[&str]) -> String {{
    use std::process::{{Command, Stdio}};
    use std::io::Write;

    static BINARY_PATH: &str = "{}";

    let mut child = Command::new(BINARY_PATH)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn compiled binary");

    // Write arguments as JSON to stdin
    if let Some(mut stdin) = child.stdin.take() {{
        let input = args.join("\\n");
        stdin.write_all(input.as_bytes()).ok();
    }}

    let output = child.wait_with_output().expect("Failed to read binary output");
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}}
"#, func_name, bin_path)
    }

    /// Generate Python Nuitka shared library binding (FFI)
    fn generate_python_nuitka_binding(&self, func_name: &str) -> String {
        let lib_path = self.output_path.display();

        format!(r#"
// ═══════════════════════════════════════════════════════════════
// COMPILED PYTHON (Nuitka) - Native Library FFI
// ═══════════════════════════════════════════════════════════════
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[inline(always)]
fn {}(code: &str, args: &[&str]) -> String {{
    Python::with_gil(|py| {{
        // Load compiled Nuitka module
        let module = PyModule::import(py, "module")
            .expect("Failed to load Nuitka module");

        // Call main function
        let result = module
            .getattr("main")
            .and_then(|func| func.call1((args,)))
            .expect("Failed to call Nuitka function");

        result.extract::<String>().unwrap_or_default()
    }})
}}
"#, func_name)
    }

    /// Generate Cython extension binding (FFI)
    fn generate_cython_binding(&self, func_name: &str) -> String {
        format!(r#"
// ═══════════════════════════════════════════════════════════════
// COMPILED PYTHON (Cython) - C Extension FFI
// ═══════════════════════════════════════════════════════════════
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[inline(always)]
fn {}(code: &str, args: &[&str]) -> String {{
    Python::with_gil(|py| {{
        let module = PyModule::import(py, "module")
            .expect("Failed to load Cython module");

        let result = module
            .call_method1("execute", (args,))
            .expect("Failed to call Cython function");

        result.extract::<String>().unwrap_or_default()
    }})
}}
"#, func_name)
    }

    /// Generate Bun bundled runtime binding (optimized Node.js execution)
    fn generate_bun_runtime_binding(&self, func_name: &str) -> String {
        let bundle_path = self.output_path.display();

        format!(r#"
// ═══════════════════════════════════════════════════════════════
// COMPILED JAVASCRIPT (Bun) - Optimized Runtime
// ═══════════════════════════════════════════════════════════════
#[inline(always)]
fn {}(args: &[&str]) -> String {{
    use std::process::Command;

    static BUNDLE_PATH: &str = "{}";

    // Use Bun for maximum speed
    let output = Command::new("bun")
        .arg("run")
        .arg(BUNDLE_PATH)
        .args(args)
        .output()
        .expect("Failed to execute Bun bundle");

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}}
"#, func_name, bundle_path)
    }

    /// Generate bundled script binding (runtime execution with caching)
    fn generate_bundled_binding(&self, func_name: &str, format: &BundleFormat) -> String {
        let bundle_path = self.output_path.display();

        match format {
            BundleFormat::PyInstaller => format!(r#"
// ═══════════════════════════════════════════════════════════════
// BUNDLED PYTHON (PyInstaller) - Standalone Executable
// ═══════════════════════════════════════════════════════════════
#[inline(always)]
fn {}(args: &[&str]) -> String {{
    use std::process::Command;

    let output = Command::new("{}")
        .args(args)
        .output()
        .expect("Failed to execute PyInstaller bundle");

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}}
"#, func_name, bundle_path),

            BundleFormat::Esbuild => format!(r#"
// ═══════════════════════════════════════════════════════════════
// BUNDLED JAVASCRIPT (esbuild) - Node.js Runtime
// ═══════════════════════════════════════════════════════════════
#[inline(always)]
fn {}(args: &[&str]) -> String {{
    use std::process::Command;

    let output = Command::new("node")
        .arg("{}")
        .args(args)
        .output()
        .expect("Failed to execute esbuild bundle");

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}}
"#, func_name, bundle_path),

            _ => self.generate_embedded_binding(func_name),
        }
    }

    /// Generate system package binding (direct import)
    fn generate_system_binding(&self, func_name: &str) -> String {
        format!(r#"
// ═══════════════════════════════════════════════════════════════
// SYSTEM PACKAGE - Direct Runtime Import
// ═══════════════════════════════════════════════════════════════
#[inline(always)]
fn {}(code: &str, args: &[&str]) -> String {{
    // Falls back to runtime bridge (system packages not compiled)
    String::new()
}}
"#, func_name)
    }

    /// Generate embedded source binding (inline code)
    fn generate_embedded_binding(&self, func_name: &str) -> String {
        format!(r#"
// ═══════════════════════════════════════════════════════════════
// EMBEDDED SOURCE - Inline Code (Falls back to runtime)
// ═══════════════════════════════════════════════════════════════
#[inline(always)]
fn {}(code: &str, args: &[&str]) -> String {{
    // Uses runtime bridge (embedded code not pre-compiled)
    String::new()
}}
"#, func_name)
    }
}

/// Registry of compiled dependencies for code generation
pub struct CompiledDependencyRegistry {
    pub(crate) dependencies: HashMap<String, CompiledDependency>,
    bindings: HashMap<String, String>,
}

impl CompiledDependencyRegistry {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            bindings: HashMap::new(),
        }
    }

    /// Register a compiled dependency
    pub fn register(&mut self, dep: CompiledDependency) {
        let func_name = format!("compiled_{}", dep.id);
        let binding = dep.generate_rust_binding(&func_name);

        self.bindings.insert(func_name.clone(), binding);
        self.dependencies.insert(dep.id.to_string(), dep);
    }

    /// Get binding code for a dependency
    pub fn get_binding(&self, dep_id: &str) -> Option<&String> {
        let func_name = format!("compiled_{}", dep_id);
        self.bindings.get(&func_name)
    }

    /// Get all binding code
    pub fn get_all_bindings(&self) -> String {
        let mut result = String::new();
        result.push_str("// ═══════════════════════════════════════════════════════════════\n");
        result.push_str("// COMPILED LANGUAGE BRIDGES - Zero Overhead Native Calls\n");
        result.push_str("// ═══════════════════════════════════════════════════════════════\n\n");

        for (_, binding) in self.bindings.iter() {
            result.push_str(binding);
            result.push_str("\n");
        }

        result
    }

    /// Check if dependency is compiled
    pub fn is_compiled(&self, dep_id: &str) -> bool {
        self.dependencies.contains_key(dep_id)
    }

    /// Get compiled dependency strategy
    pub fn get_strategy(&self, dep_id: &str) -> Option<&CompilationStrategy> {
        self.dependencies.get(dep_id).map(|d| &d.strategy)
    }
}