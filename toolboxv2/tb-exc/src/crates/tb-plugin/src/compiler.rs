use std::path::PathBuf;
use std::process::Command;
use std::fs;
use tb_core::{PluginLanguage, PluginMode, Result, TBError};
use serde::{Serialize, Deserialize};

/// Plugin metadata for YAML storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadataYaml {
    pub name: String,
    pub language: String,
    pub functions: Vec<FunctionMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    pub name: String,
    pub params: Vec<ParamMetadata>,
    pub return_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamMetadata {
    pub name: String,
    pub param_type: String,
}

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
            .map_err(|e| TBError::compilation_error(format!("Failed to run rustc: {}", e)))?;

        if !status.success() {
            return Err(TBError::compilation_error("Rust compilation failed"));
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

    pub fn install_python_deps(&self, requires: &[String]) -> Result<()> {
        for dep in requires {
            // Silent installation - no output

            // Try uv pip first (faster), then fall back to python -m pip
            let mut cmd = Command::new("uv");
            cmd.args(&["pip", "install", "--quiet", dep]);

            let status = cmd.status();

            let success = if let Ok(status) = status {
                status.success()
            } else {
                // Fall back to python -m pip
                let status = Command::new("python")
                    .args(&["-m", "pip", "install", "--quiet", dep])
                    .status()
                    .map_err(|e| TBError::plugin_error(format!("Failed to install {}: {}", dep, e)))?;
                status.success()
            };

            if !success {
                return Err(TBError::plugin_error(format!("Failed to install dependency: {}", dep)));
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
            .map_err(|e| TBError::compilation_error(format!("Nuitka compilation failed: {}", e)))?;

        if !status.success() {
            return Err(TBError::compilation_error("Python compilation failed"));
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
            .map_err(|e| TBError::plugin_error(format!("npm install failed: {}", e)))?;

        if !status.success() {
            return Err(TBError::plugin_error("npm install failed"));
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
            .map_err(|e| TBError::compilation_error(format!("esbuild failed: {}", e)))?;

        if !status.success() {
            return Err(TBError::compilation_error("JavaScript bundling failed"));
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
            .map_err(|e| TBError::compilation_error(format!("Go compilation failed: {}", e)))?;

        if !status.success() {
            return Err(TBError::compilation_error("Go compilation failed"));
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

    /// Generate metadata YAML file for a plugin
    pub fn generate_metadata(
        &self,
        plugin_path: &PathBuf,
        language: &PluginLanguage,
    ) -> Result<()> {
        let source = fs::read_to_string(plugin_path)?;
        let functions = self.extract_functions(language, &source)?;

        let metadata = PluginMetadataYaml {
            name: plugin_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            language: format!("{:?}", language),
            functions,
        };

        let yaml_path = plugin_path.with_extension("meta.yaml");
        let yaml_content = serde_yaml::to_string(&metadata)
            .map_err(|e| TBError::plugin_error(format!("Failed to serialize metadata: {}", e)))?;

        fs::write(yaml_path, yaml_content)?;
        Ok(())
    }

    /// Load metadata from YAML file if it exists
    pub fn load_metadata(&self, plugin_path: &PathBuf) -> Result<Option<PluginMetadataYaml>> {
        let yaml_path = plugin_path.with_extension("meta.yaml");

        if !yaml_path.exists() {
            return Ok(None);
        }

        let yaml_content = fs::read_to_string(yaml_path)?;
        let metadata: PluginMetadataYaml = serde_yaml::from_str(&yaml_content)
            .map_err(|e| TBError::plugin_error(format!("Failed to parse metadata: {}", e)))?;

        Ok(Some(metadata))
    }

    /// Extract function signatures from source code (heuristic)
    fn extract_functions(
        &self,
        language: &PluginLanguage,
        source: &str,
    ) -> Result<Vec<FunctionMetadata>> {
        let mut functions = Vec::new();

        match language {
            PluginLanguage::Rust => {
                // Look for "pub extern "C" fn" or "#[no_mangle]"
                for line in source.lines() {
                    if line.contains("pub extern \"C\" fn") || line.contains("#[no_mangle]") {
                        if let Some(func) = self.parse_rust_function(line) {
                            functions.push(func);
                        }
                    }
                }
            }
            PluginLanguage::Python => {
                // Look for "def function_name("
                for line in source.lines() {
                    if line.trim().starts_with("def ") {
                        if let Some(func) = self.parse_python_function(line) {
                            functions.push(func);
                        }
                    }
                }
            }
            _ => {
                // For other languages, return empty for now
            }
        }

        Ok(functions)
    }

    fn parse_rust_function(&self, line: &str) -> Option<FunctionMetadata> {
        // Simple heuristic: extract function name
        // Format: "pub extern "C" fn function_name(args) -> ReturnType"
        let parts: Vec<&str> = line.split("fn ").collect();
        if parts.len() < 2 {
            return None;
        }

        let rest = parts[1];
        let name_end = rest.find('(')?;
        let name = rest[..name_end].trim().to_string();

        Some(FunctionMetadata {
            name,
            params: vec![ParamMetadata {
                name: "args".to_string(),
                param_type: "Any".to_string(),
            }],
            return_type: "Any".to_string(),
        })
    }

    fn parse_python_function(&self, line: &str) -> Option<FunctionMetadata> {
        // Format: "def function_name(args):"
        let parts: Vec<&str> = line.split("def ").collect();
        if parts.len() < 2 {
            return None;
        }

        let rest = parts[1];
        let name_end = rest.find('(')?;
        let name = rest[..name_end].trim().to_string();

        Some(FunctionMetadata {
            name,
            params: vec![ParamMetadata {
                name: "args".to_string(),
                param_type: "Any".to_string(),
            }],
            return_type: "Any".to_string(),
        })
    }
}

