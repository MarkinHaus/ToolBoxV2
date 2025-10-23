use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::collections::HashSet;
use std::env;
use tb_cache::CacheManager;
use tb_codegen::RustCodeGenerator;
use tb_core::{Result, StringInterner, Value, Statement, Program, ImportItem};
use tb_jit::JitExecutor;
use tb_optimizer::{Optimizer, OptimizerConfig};
use tb_parser::{Lexer, Parser};
use tb_types::TypeChecker;

pub fn run_file(
    file: &Path,
    mode: &str,
    opt_level: u8,
    interner: Arc<StringInterner>,
    cache_manager: Arc<CacheManager>,
) -> Result<Value> {
    // Check cache first
    if mode == "compile" {
        if let Some(_cached) = cache_manager.import_cache().get(file)? {
            // Execute cached binary
            return execute_cached_binary(file);
        }
    }

    // Read source
    let source = fs::read_to_string(file)?;

    // Lex
    let mut lexer = Lexer::new(&source, Arc::clone(&interner));
    let tokens = lexer.tokenize();

    // Parse (with source and file path for better error messages)
    let mut parser = Parser::new_with_source_and_path(tokens, source.clone(), file.to_path_buf());
    let mut program = parser.parse()?;

    // Load imports before type checking
    let file_dir = file.parent().unwrap_or_else(|| Path::new("."));
    let mut loaded_modules = HashSet::new();
    load_imports(&mut program, file_dir, Arc::clone(&interner), &mut loaded_modules)?;

    // Type check (with source context for better error messages)
    let mut type_checker = TypeChecker::new_with_source(source.clone(), Some(file.to_path_buf()));
    type_checker.check_program(&program)?;

    // Optimize
    let mut optimizer_config = OptimizerConfig::default();
    optimizer_config.optimization_level = opt_level;
    let mut optimizer = Optimizer::new(optimizer_config);
    optimizer.optimize(&mut program)?;

    match mode {
        "jit" => {
            // JIT execution (with source context for better error messages)
            let mut executor = JitExecutor::new_with_source(source.clone(), Some(file.to_path_buf()));
            executor.execute(&program)
        }
        "compile" => {
            // Generate and compile
            let mut codegen = RustCodeGenerator::new();
            let rust_code = codegen.generate(&program)?;

            // Write to temp file
            let temp_dir = tempfile::tempdir()?;
            let rust_file = temp_dir.path().join("program.rs");
            fs::write(&rust_file, rust_code)?;

            // Compile with rustc
            let output = temp_dir.path().join("program");
            let compile_output = Command::new("rustc")
                .args(&[
                    "-C", "opt-level=3",
                    "-o", output.to_str().unwrap(),
                    rust_file.to_str().unwrap(),
                ])
                .output()?;

            if !compile_output.status.success() {
                let stderr = String::from_utf8_lossy(&compile_output.stderr);
                return Err(tb_core::TBError::CompilationError {
                    message: "Rust compilation failed".to_string(),
                    compiler_output: Some(stderr.to_string()),
                });
            }

            // Cache the result
            cache_manager.import_cache().put(file, &program)?;

            // Execute
            let output = Command::new(&output).output()?;
            println!("{}", String::from_utf8_lossy(&output.stdout));

            Ok(Value::None)
        }
        _ => Err(tb_core::TBError::runtime_error(format!("Unknown mode: {}", mode))),
    }
}

pub fn compile_file(
    file: &Path,
    output: &Path,
    opt_level: u8,
    interner: Arc<StringInterner>,
    _cache_manager: Arc<CacheManager>,
) -> Result<()> {
    let source = fs::read_to_string(file)?;

    let mut lexer = Lexer::new(&source, Arc::clone(&interner));
    let tokens = lexer.tokenize();

    let mut parser = Parser::new_with_source_and_path(tokens, source.clone(), file.to_path_buf());
    let mut program = parser.parse()?;

    // ✅ FIX: Load imports before type checking (same as run_file)
    let file_dir = file.parent().unwrap_or_else(|| Path::new("."));
    let mut loaded_modules = HashSet::new();
    load_imports(&mut program, file_dir, Arc::clone(&interner), &mut loaded_modules)?;

    // Type check (with source context for better error messages)
    let mut type_checker = TypeChecker::new_with_source(source.clone(), Some(file.to_path_buf()));
    type_checker.check_program(&program)?;

    let mut optimizer_config = OptimizerConfig::default();
    optimizer_config.optimization_level = opt_level;
    let mut optimizer = Optimizer::new(optimizer_config);
    optimizer.optimize(&mut program)?;

    let mut codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate(&program)?;

    // Get required crates from code generator
    let required_crates = codegen.get_required_crates();

    // DEBUG: Save generated code for inspection
    if std::env::var("TB_DEBUG_CODEGEN").is_ok() {
        let debug_file = output.with_extension("rs");
        fs::write(&debug_file, &rust_code)?;
        eprintln!("Generated Rust code saved to: {}", debug_file.display());
        eprintln!("Required crates: {:?}", required_crates);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // NEW FAST COMPILATION STRATEGY: Use rustc directly instead of Cargo
    // ═══════════════════════════════════════════════════════════════════════════

    // Check if we should use the fast rustc path (default: yes)
    let use_fast_compile = std::env::var("TB_USE_CARGO").is_err();

    if use_fast_compile {
        return compile_with_rustc(file, output, &rust_code, &required_crates);
    }

    // FALLBACK: Old Cargo-based compilation (slow but more compatible)
    let temp_dir = tempfile::tempdir()?;
    let project_dir = temp_dir.path();

    // Build dependencies section dynamically based on required crates
    let mut dependencies = String::new();

    // ALWAYS add tb_runtime as it contains DictValue and built-in functions
    // Use path dependency to the local tb-runtime crate

    // First, try environment variable TB_RUNTIME_PATH
    let tb_runtime_path = if let Ok(env_path) = std::env::var("TB_RUNTIME_PATH") {
        std::path::PathBuf::from(env_path)
    } else {
        // Try to find tb-runtime relative to the binary
        let exe_path = std::env::current_exe().ok();

        let mut candidates = Vec::new();

        // PRIORITY 1: Relative to binary (most reliable)
        if let Some(exe) = exe_path {
            if let Some(parent) = exe.parent() {
                // From target/release/tb.exe -> ../../crates/tb-runtime
                candidates.push(parent.join("../../crates/tb-runtime"));
                candidates.push(parent.join("../../../crates/tb-runtime"));
                candidates.push(parent.join("../crates/tb-runtime"));
            }
        }

        // PRIORITY 2: Relative to current directory
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("tb-exc/src/crates/tb-runtime"));
            candidates.push(cwd.join("toolboxv2/tb-exc/src/crates/tb-runtime"));
            candidates.push(cwd.join("crates/tb-runtime"));
        }

        // Find first existing path and canonicalize it
        candidates.into_iter()
            .find(|path| path.exists())
            .and_then(|p| p.canonicalize().ok())
            .unwrap_or_else(|| {
                // Last resort: use a default path
                std::path::PathBuf::from("crates/tb-runtime")
            })
    };

    // Convert Windows backslashes to forward slashes for TOML compatibility
    let mut tb_runtime_path_str = tb_runtime_path.display().to_string().replace('\\', "/");

    // ✅ FIX: Remove Windows UNC prefix //?/ which breaks Cargo.toml
    if tb_runtime_path_str.starts_with("//?/") {
        tb_runtime_path_str = tb_runtime_path_str[4..].to_string();
    }

    dependencies.push_str(&format!("tb-runtime = {{ path = \"{}\" }}\n", tb_runtime_path_str));

    // If serde_json or serde_yaml is needed, add serde first
    if required_crates.contains(&"serde_json") || required_crates.contains(&"serde_yaml") {
        dependencies.push_str("serde = { version = \"1.0\", features = [\"derive\"] }\n");
    }

    for crate_name in &required_crates {
        let version = match *crate_name {
            "serde_json" => "1.0",
            "serde_yaml" => "0.9",
            "chrono" => "0.4",
            _ => continue,
        };
        dependencies.push_str(&format!("{} = \"{}\"\n", crate_name, version));
    }

    // Create Cargo.toml with only required dependencies
    let cargo_toml = format!(
        r#"[package]
name = "tb_compiled"
version = "0.1.0"
edition = "2021"

[dependencies]
{}
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
"#,
        dependencies
    );
    fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

    // Create src directory and main.rs
    let src_dir = project_dir.join("src");
    fs::create_dir(&src_dir)?;
    fs::write(src_dir.join("main.rs"), rust_code)?;

    // Compile with cargo
    let compile_output = Command::new("cargo")
        .args(&["build", "--release"])
        .current_dir(project_dir)
        .output()?;

    if !compile_output.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output.stderr);
        return Err(tb_core::TBError::CompilationError {
            message: "Rust compilation failed".to_string(),
            compiler_output: Some(stderr.to_string()),
        });
    }

    // Copy the compiled binary to the output location
    #[cfg(target_os = "windows")]
    let binary_name = "tb_compiled.exe";
    #[cfg(not(target_os = "windows"))]
    let binary_name = "tb_compiled";

    let compiled_binary = project_dir.join("target").join("release").join(binary_name);

    // ✅ FIX: Ensure output has .exe extension on Windows
    #[cfg(target_os = "windows")]
    let final_output = if output.extension().and_then(|s| s.to_str()) != Some("exe") {
        output.with_extension("exe")
    } else {
        output.to_path_buf()
    };
    #[cfg(not(target_os = "windows"))]
    let final_output = output.to_path_buf();

    fs::copy(&compiled_binary, &final_output)?;

    Ok(())
}

fn execute_cached_binary(_file: &Path) -> Result<Value> {
    // TODO: Implement cached binary execution
    Ok(Value::None)
}

/// Load imports recursively and merge them into the program
fn load_imports(
    program: &mut Program,
    file_dir: &Path,
    interner: Arc<StringInterner>,
    loaded: &mut HashSet<PathBuf>,
) -> Result<()> {
    // Extract import statements
    let imports: Vec<ImportItem> = program
        .statements
        .iter()
        .filter_map(|stmt| {
            if let Statement::Import { items, .. } = stmt {
                Some(items.clone())
            } else {
                None
            }
        })
        .flatten()
        .collect();

    // Load each import
    for item in imports {
        // FIX: Properly construct platform-specific path
        let import_path = file_dir.join(item.path.as_ref());

        // Try to canonicalize if file exists, otherwise use as-is
        let canonical_path = if import_path.exists() {
            import_path.canonicalize().unwrap_or(import_path.clone())
        } else {
            import_path.clone()
        };

        // Skip if already loaded (prevent circular imports)
        if loaded.contains(&canonical_path) {
            continue;
        }

        // Read and parse import file - use canonical path
        let source = fs::read_to_string(&canonical_path).map_err(|e| {
            // On Windows, provide more detailed error with proper path formatting
            #[cfg(target_os = "windows")]
            {
                tb_core::TBError::runtime_error(format!(
                        "Failed to load import '{}': {} (normalized: {})",
                        canonical_path.display(),
                        e,
                        canonical_path.to_string_lossy().replace("\\\\?\\", "")
                    ))
            }
            #[cfg(not(target_os = "windows"))]
            {
                tb_core::TBError::runtime_error(format!("Failed to load import '{}': {}", canonical_path.display(), e))
            }
        })?;

        let mut lexer = Lexer::new(&source, Arc::clone(&interner));
        let tokens = lexer.tokenize();
        let mut parser = Parser::new_with_source_and_path(tokens, source.clone(), canonical_path.clone());
        let mut import_program = parser.parse()?;

        // Mark as loaded - use canonical path
        loaded.insert(canonical_path.clone());

        // Recursively load imports from the imported file
        if let Some(import_dir) = canonical_path.parent() {
            load_imports(&mut import_program, import_dir, Arc::clone(&interner), loaded)?;
        }

        // Merge imported statements into main program (before import statements)
        // Filter out import statements from the imported module
        let imported_statements: Vec<Statement> = import_program
            .statements
            .into_iter()
            .filter(|stmt| !matches!(stmt, Statement::Import { .. }))
            .collect();

        // Insert imported statements at the beginning
        program.statements.splice(0..0, imported_statements);
    }

    Ok(())
}

/// Fast compilation using persistent Cargo project (2-5s instead of 25-33s)
/// Uses workspace dependencies that are already compiled
fn compile_with_rustc(
    _file: &Path,
    output: &Path,
    rust_code: &str,
    _required_crates: &[&str],
) -> Result<()> {
    use std::io::Write;

    // Find workspace root
    let exe_path = env::current_exe()?;
    let workspace_root = exe_path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or_else(|| tb_core::TBError::runtime_error("Cannot find workspace root"))?;

    // Create persistent compile cache directory
    let compile_cache = workspace_root.join("target").join("tb-compile-cache");
    fs::create_dir_all(&compile_cache)?;

    // Create Cargo.toml if it doesn't exist
    let cargo_toml = compile_cache.join("Cargo.toml");
    if !cargo_toml.exists() {
        let toml_content = format!(
r#"[package]
name = "tb-compiled"
version = "0.1.0"
edition = "2021"

[workspace]

[dependencies]
tb-runtime = {{ path = "{}/crates/tb-runtime", features = ["full"] }}
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
serde_yaml = "0.9"

[profile.release]
opt-level = 3
lto = false
codegen-units = 16
"#,
            workspace_root.display().to_string().replace("\\", "/")
        );
        fs::write(&cargo_toml, toml_content)?;
    }

    // Create src directory
    let src_dir = compile_cache.join("src");
    fs::create_dir_all(&src_dir)?;

    // Write main.rs
    let main_rs = src_dir.join("main.rs");
    fs::write(&main_rs, rust_code)?;

    // Compile with Cargo (uses cached dependencies)
    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg(&cargo_toml)
        .env("CARGO_TARGET_DIR", compile_cache.join("target"));

    let output_result = cmd.output()?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(tb_core::TBError::runtime_error(format!(
            "Cargo compilation failed:\n{}",
            stderr
        )));
    }

    // Copy binary to output location
    #[cfg(target_os = "windows")]
    let binary_name = "tb-compiled.exe";
    #[cfg(not(target_os = "windows"))]
    let binary_name = "tb-compiled";

    let compiled_binary = compile_cache.join("target").join("release").join(binary_name);

    // Ensure output has .exe extension on Windows
    #[cfg(target_os = "windows")]
    let final_output = if output.extension().and_then(|s| s.to_str()) != Some("exe") {
        output.with_extension("exe")
    } else {
        output.to_path_buf()
    };
    #[cfg(not(target_os = "windows"))]
    let final_output = output.to_path_buf();

    fs::copy(&compiled_binary, &final_output)?;

    Ok(())
}

