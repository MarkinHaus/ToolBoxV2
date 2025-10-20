use std::path::PathBuf;
use std::process::Command;
use std::fs;
use tb_core::{PluginLanguage, PluginMode, Result, TBError};

/// Plugin compiler for different languages with dependency management
pub struct PluginCompiler {
    temp_dir: PathBuf,
    output_dir: PathBuf,
}

impl PluginCompiler {
    pub fn new(temp_dir: PathBuf, output_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&temp_dir)?;
        fs::create_dir_all(&output_dir)?;

        Ok(Self {
            temp_dir,
            output_dir,
        })
    }

    /// Compile plugin with dependency support
    pub fn compile(
        &self,
        language: &PluginLanguage,
        mode: &PluginMode,
        source: &str,
        name: &str,
        requires: &[String],  // NEW: Dependencies
    ) -> Result<PathBuf> {
        match language {
            PluginLanguage::Python => self.compile_python(source, name, mode, requires),
            PluginLanguage::JavaScript => self.compile_javascript(source, name, mode, requires),
            PluginLanguage::Go => self.compile_go(source, name, requires),
            PluginLanguage::Rust => self.compile_rust(source, name),
        }
    }

    fn compile_rust(&self, source: &str, name: &str) -> Result<PathBuf> {
        // Write source to temp file
        let source_file = self.temp_dir.join(format!("{}.rs", name));
        std::fs::write(&source_file, source)?;

        // Compile with rustc
        let output_file = self.output_dir.join(self.library_name(name));

        let status = Command::new("rustc")
            .args(&[
                "--crate-type", "cdylib",
                "-C", "opt-level=3",
                "-C", "lto=fat",
                "-o", output_file.to_str().unwrap(),
                source_file.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| TBError::CompilationError {
                message: format!("Failed to run rustc: {}", e),
            })?;

        if !status.success() {
            return Err(TBError::CompilationError {
                message: "Rust compilation failed".to_string(),
            });
        }

        Ok(output_file)
    }

    /// Python compilation with dependency installation
    fn compile_python(
        &self,
        source: &str,
        name: &str,
        mode: &PluginMode,
        requires: &[String],
    ) -> Result<PathBuf> {
        let source_file = self.temp_dir.join(format!("{}.py", name));

        // Install dependencies if needed
        if !requires.is_empty() {
            self.install_python_deps(requires)?;
        }

        // Write source (native Python code)
        fs::write(&source_file, source)?;

        match mode {
            PluginMode::Jit => {
                // JIT: Return Python file path for PyO3
                Ok(source_file)
            }
            PluginMode::Compile => {
                // Compile: Use Nuitka
                self.compile_python_nuitka(&source_file, name)
            }
        }
    }

    fn install_python_deps(&self, requires: &[String]) -> Result<()> {
        for dep in requires {
            println!("Installing Python dependency: {}", dep);

            let status = Command::new("pip")
                .args(&["install", "--quiet", dep])
                .status()
                .map_err(|e| TBError::PluginError {
                    message: format!("Failed to install {}: {}", dep, e),
                })?;

            if !status.success() {
                return Err(TBError::PluginError {
                    message: format!("Failed to install dependency: {}", dep),
                });
            }
        }
        Ok(())
    }

    fn compile_python_nuitka(&self, source_file: &PathBuf, name: &str) -> Result<PathBuf> {
        let output_file = self.output_dir.join(self.library_name(name));

        let status = Command::new("python3")
            .args(&[
                "-m",
                "nuitka",
                "--standalone",
                "--module",
                "--output-dir",
                self.output_dir.to_str().unwrap(),
                source_file.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| TBError::CompilationError {
                message: format!("Nuitka compilation failed: {}", e),
            })?;

        if !status.success() {
            return Err(TBError::CompilationError {
                message: "Python compilation failed".to_string(),
            });
        }

        Ok(output_file)
    }

    /// JavaScript compilation with npm dependencies
    fn compile_javascript(
        &self,
        source: &str,
        name: &str,
        mode: &PluginMode,
        requires: &[String],
    ) -> Result<PathBuf> {
        let source_file = self.temp_dir.join(format!("{}.js", name));

        // Create package.json if dependencies exist
        if !requires.is_empty() {
            self.create_package_json(name, requires)?;
            self.install_npm_deps()?;
        }

        // Write source (native JavaScript code)
        fs::write(&source_file, source)?;

        match mode {
            PluginMode::Jit => {
                // JIT: Return JS file for Node.js
                Ok(source_file)
            }
            PluginMode::Compile => {
                // Compile: Use esbuild for bundling
                self.compile_javascript_esbuild(&source_file, name)
            }
        }
    }

    fn create_package_json(&self, name: &str, requires: &[String]) -> Result<()> {
        let package_json = format!(
            r#"{{
  "name": "{}",
  "version": "1.0.0",
  "dependencies": {{
    {}
  }}
}}"#,
            name,
            requires
                .iter()
                .map(|dep| format!("\"{}\":\"latest\"", dep))
                .collect::<Vec<_>>()
                .join(",\n    ")
        );

        let package_file = self.temp_dir.join("package.json");
        fs::write(package_file, package_json)?;
        Ok(())
    }

    fn install_npm_deps(&self) -> Result<()> {
        println!("Installing npm dependencies...");

        let status = Command::new("npm")
            .arg("install")
            .current_dir(&self.temp_dir)
            .status()
            .map_err(|e| TBError::PluginError {
                message: format!("npm install failed: {}", e),
            })?;

        if !status.success() {
            return Err(TBError::PluginError {
                message: "npm install failed".to_string(),
            });
        }
        Ok(())
    }

    fn compile_javascript_esbuild(&self, source_file: &PathBuf, name: &str) -> Result<PathBuf> {
        let output_file = self.output_dir.join(format!("{}.bundle.js", name));

        let status = Command::new("esbuild")
            .args(&[
                source_file.to_str().unwrap(),
                "--bundle",
                "--platform=node",
                "--format=cjs",
                &format!("--outfile={}", output_file.display()),
            ])
            .status()
            .map_err(|e| TBError::CompilationError {
                message: format!("esbuild failed: {}", e),
            })?;

        if !status.success() {
            return Err(TBError::CompilationError {
                message: "JavaScript bundling failed".to_string(),
            });
        }

        Ok(output_file)
    }

    /// Go compilation with module support
    fn compile_go(&self, source: &str, name: &str, requires: &[String]) -> Result<PathBuf> {
        let source_file = self.temp_dir.join(format!("{}.go", name));

        // Create go.mod if dependencies exist
        if !requires.is_empty() {
            self.create_go_mod(name, requires)?;
        }

        // Write source (native Go code)
        fs::write(&source_file, source)?;

        let output_file = self.output_dir.join(self.library_name(name));

        let status = Command::new("go")
            .args(&[
                "build",
                "-buildmode=plugin",
                "-o",
                output_file.to_str().unwrap(),
                source_file.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| TBError::CompilationError {
                message: format!("Go compilation failed: {}", e),
            })?;

        if !status.success() {
            return Err(TBError::CompilationError {
                message: "Go compilation failed".to_string(),
            });
        }

        Ok(output_file)
    }

    fn create_go_mod(&self, name: &str, requires: &[String]) -> Result<()> {
        let go_mod = format!(
            "module {}\n\ngo 1.21\n\nrequire (\n{}\n)\n",
            name,
            requires
                .iter()
                .map(|dep| format!("\t{} v0.0.0", dep))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let mod_file = self.temp_dir.join("go.mod");
        fs::write(mod_file, go_mod)?;

        // Run go mod tidy
        Command::new("go")
            .args(&["mod", "tidy"])
            .current_dir(&self.temp_dir)
            .status()?;

        Ok(())
    }

    fn library_name(&self, name: &str) -> String {
        #[cfg(target_os = "windows")]
        return format!("{}.dll", name);

        #[cfg(target_os = "macos")]
        return format!("lib{}.dylib", name);

        #[cfg(target_os = "linux")]
        return format!("lib{}.so", name);
    }
}

