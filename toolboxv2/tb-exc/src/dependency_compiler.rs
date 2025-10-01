// File: tb_lang/src/dependency_compiler.rs

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use crate::{TBResult, TBError, Language};

/// Compilation strategy for dependencies
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationStrategy {
    /// Compile to native C library (Python via Nuitka/Cython)
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
    pub id: String,
    pub language: Language,
    pub code: String,
    pub imports: Vec<String>,
    pub is_in_loop: bool,
    pub estimated_calls: usize,
}

/// Compiled dependency output
#[derive(Debug, Clone)]
pub struct CompiledDependency {
    pub id: String,
    pub strategy: CompilationStrategy,
    pub output_path: PathBuf,
    pub size_bytes: usize,
    pub compile_time_ms: u128,
}

pub struct DependencyCompiler {
    deps_dir: PathBuf,
    cache_dir: PathBuf,
}

impl DependencyCompiler {
    pub fn new(project_dir: &Path) -> Self {
        let deps_dir = project_dir.join("deps");
        let cache_dir = project_dir.join(".tb_cache");

        fs::create_dir_all(&deps_dir).ok();
        fs::create_dir_all(&cache_dir).ok();

        Self {
            deps_dir,
            cache_dir,
        }
    }

    /// Compile a dependency using best available strategy
    pub fn compile(&self, dep: &Dependency) -> TBResult<CompiledDependency> {
        let start = std::time::Instant::now();

        // Choose strategy
        let strategy = self.choose_strategy(dep);

        // Compile based on strategy
        let output_path = match &strategy {
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

        Ok(CompiledDependency {
            id: dep.id.clone(),
            strategy,
            output_path,
            size_bytes,
            compile_time_ms: start.elapsed().as_millis(),
        })
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
            // Try Nuitka first (best performance)
            if self.has_tool("nuitka") {
                return CompilationStrategy::NativeCompilation {
                    tool: NativeTool::Nuitka,
                };
            }

            // Fallback to Cython
            if self.has_tool("cython") {
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
        if self.has_tool("pyinstaller") {
            CompilationStrategy::Bundle {
                format: BundleFormat::PyInstaller,
            }
        } else {
            CompilationStrategy::EmbedRaw
        }
    }

    fn choose_js_strategy(&self, dep: &Dependency) -> CompilationStrategy {
        // Small code → embed
        if dep.code.len() < 10000 {
            return CompilationStrategy::EmbedRaw;
        }

        // Has node_modules → bundle
        if dep.imports.iter().any(|imp| !imp.starts_with('.') && !imp.starts_with('/')) {
            if self.has_tool("esbuild") {
                return CompilationStrategy::Bundle {
                    format: BundleFormat::Esbuild,
                };
            }
        }

        CompilationStrategy::EmbedRaw
    }

    /// Check if external tool is available
    fn has_tool(&self, tool: &str) -> bool {
        Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
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


// Python compilation methods
impl DependencyCompiler {
    /// Compile Python to native using Nuitka
    fn compile_python_nuitka(&self, dep: &Dependency) -> TBResult<PathBuf> {
        println!("🐍 Compiling Python with Nuitka (Python → C → binary)...");

        // Write Python code to temp file
        let temp_dir = self.cache_dir.join(format!("py_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let py_file = temp_dir.join("module.py");
        fs::write(&py_file, &dep.code)?;

        // Output path
        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(self.get_lib_name("module"));

        // Compile with Nuitka
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
                message: format!("Nuitka compilation failed:\n{}", stderr),
                source: dep.code.clone(),
            });
        }

        println!("  ✓ Compiled to native library");

        Ok(output)
    }

    /// Compile Python to C extension using Cython
    fn compile_python_cython(&self, dep: &Dependency) -> TBResult<PathBuf> {
        println!("🐍 Compiling Python with Cython (Python → C extension)...");

        let temp_dir = self.cache_dir.join(format!("cy_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        // Write .pyx file
        let pyx_file = temp_dir.join("module.pyx");
        fs::write(&pyx_file, &dep.code)?;

        // Write setup.py
        let setup_py = format!(r#"
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("module.pyx", compiler_directives={{'language_level': "3"}}),
)
"#);
        fs::write(temp_dir.join("setup.py"), setup_py)?;

        // Compile
        let result = Command::new("python3")
            .args(&["setup.py", "build_ext", "--inplace"])
            .current_dir(&temp_dir)
            .output()?;

        if !result.status.success() {
            return Err(TBError::CompilationError {
                message: "Cython compilation failed".to_string(),
                source: dep.code.clone(),
            });
        }

        // Find output .so file
        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        let so_file = self.find_file_with_extension(&temp_dir, ".so")?;
        let output = output_dir.join(so_file.file_name().unwrap());
        fs::copy(&so_file, &output)?;

        println!("  ✓ Compiled to C extension");

        Ok(output)
    }

    /// Bundle Python with PyInstaller
    fn bundle_python_pyinstaller(&self, dep: &Dependency) -> TBResult<PathBuf> {
        println!("📦 Bundling Python with PyInstaller...");

        let temp_dir = self.cache_dir.join(format!("pyi_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let py_file = temp_dir.join("main.py");
        fs::write(&py_file, &dep.code)?;

        let output_dir = self.deps_dir.join("python");
        fs::create_dir_all(&output_dir)?;

        // Bundle with PyInstaller
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
                message: "PyInstaller bundling failed".to_string(),
                source: dep.code.clone(),
            });
        }

        let output = output_dir.join(&dep.id);
        println!("  ✓ Bundled Python application");

        Ok(output)
    }

    fn compile_native(&self, dep: &Dependency, tool: &NativeTool) -> TBResult<PathBuf> {
        match (dep.language, tool) {
            (Language::Python, NativeTool::Nuitka) => self.compile_python_nuitka(dep),
            (Language::Python, NativeTool::Cython) => self.compile_python_cython(dep),
            (Language::JavaScript, NativeTool::Pkg) => self.compile_js_pkg(dep),
            _ => Err(TBError::UnsupportedLanguage(format!(
                "Native compilation not supported for {:?} with {:?}",
                dep.language, tool
            ))),
        }
    }
}


// JavaScript compilation methods
impl DependencyCompiler {
    /// Bundle JavaScript with esbuild
    fn bundle_js_esbuild(&self, dep: &Dependency) -> TBResult<PathBuf> {
        println!("📦 Bundling JavaScript with esbuild...");

        let temp_dir = self.cache_dir.join(format!("js_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let js_file = temp_dir.join("index.js");
        fs::write(&js_file, &dep.code)?;

        let output_dir = self.deps_dir.join("js");
        fs::create_dir_all(&output_dir)?;

        let output = output_dir.join(format!("{}.js", dep.id));

        // Bundle with esbuild
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
                message: format!("esbuild failed:\n{}", stderr),
                source: dep.code.clone(),
            });
        }

        println!("  ✓ Bundled JavaScript");

        Ok(output)
    }

    /// Compile JavaScript to standalone binary with pkg
    fn compile_js_pkg(&self, dep: &Dependency) -> TBResult<PathBuf> {
        println!("🎁 Compiling JavaScript to binary with pkg...");

        let temp_dir = self.cache_dir.join(format!("pkg_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        let js_file = temp_dir.join("index.js");
        fs::write(&js_file, &dep.code)?;

        // Create package.json
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

        let output = output_dir.join(&dep.id);

        // Compile with pkg
        let result = Command::new("pkg")
            .args(&[
                js_file.to_str().unwrap(),
                "--output", output.to_str().unwrap(),
                "--targets", "node14",
            ])
            .output()?;

        if !result.status.success() {
            return Err(TBError::CompilationError {
                message: "pkg compilation failed".to_string(),
                source: dep.code.clone(),
            });
        }

        println!("  ✓ Compiled to native binary");

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
            ))),
        }
    }
}

// Go compilation methods
impl DependencyCompiler {
    /// Compile Go to shared library plugin
    fn compile_go_plugin(&self, dep: &Dependency, target: &str) -> TBResult<PathBuf> {
        println!("🔧 Compiling Go plugin...");

        let temp_dir = self.cache_dir.join(format!("go_{}", dep.id));
        fs::create_dir_all(&temp_dir)?;

        // Write Go code
        let go_file = temp_dir.join("plugin.go");

        // Wrap code in plugin structure
        let plugin_code = format!(r#"
package main

import "C"

{}

//export Execute
func Execute(input *C.char) *C.char {{
    // Call user's main function
    main()
    return C.CString("ok")
}}

func main() {{}}
"#, dep.code);

        fs::write(&go_file, plugin_code)?;

        let output_dir = self.deps_dir.join("go");
        fs::create_dir_all(&output_dir)?;

        let lib_ext = self.get_lib_extension();
        let output = output_dir.join(format!("plugin_{}.{}", dep.id, lib_ext));

        // Compile to shared library
        let result = Command::new("go")
            .args(&[
                "build",
                "-buildmode=c-shared",
                "-o", output.to_str().unwrap(),
                go_file.to_str().unwrap(),
            ])
            .current_dir(&temp_dir)
            .env("GOOS", target)
            .output()?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(TBError::CompilationError {
                message: format!("Go compilation failed:\n{}", stderr),
                source: dep.code.clone(),
            });
        }

        println!("  ✓ Compiled Go plugin");

        Ok(output)
    }

    fn compile_plugin(&self, dep: &Dependency, target: &str) -> TBResult<PathBuf> {
        match dep.language {
            Language::Go => self.compile_go_plugin(dep, target),
            _ => Err(TBError::UnsupportedLanguage(format!(
                "Plugin compilation not supported for {:?}",
                dep.language
            ))),
        }
    }
}

// Helper methods
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
        fs::write(&output, &dep.code)?;

        Ok(output)
    }

    /// Ensure system package is installed
    fn ensure_system_installed(&self, dep: &Dependency) -> TBResult<()> {
        println!("🔍 Checking system dependencies...");

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
        let result = Command::new("python3")
            .args(&["-c", &format!("import {}", package)])
            .output()?;

        if !result.status.success() {
            eprintln!("⚠️  Python package '{}' not found", package);
            eprintln!("   Install with: pip install {}", package);
            return Err(TBError::InvalidOperation(format!(
                "Missing Python package: {}",
                package
            )));
        }

        println!("  ✓ Found Python package: {}", package);
        Ok(())
    }

    fn check_npm_package(&self, package: &str) -> TBResult<()> {
        let result = Command::new("npm")
            .args(&["list", "-g", package])
            .output()?;

        if !result.status.success() {
            eprintln!("⚠️  NPM package '{}' not found globally", package);
            eprintln!("   Install with: npm install -g {}", package);
            return Err(TBError::InvalidOperation(format!(
                "Missing NPM package: {}",
                package
            )));
        }

        println!("  ✓ Found NPM package: {}", package);
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

        Err(TBError::IoError(format!("No {} file found in {}", ext, dir.display())))
    }
}