// file: tb-exc/src/main.rs

use std::path::{Path, PathBuf};
use std::{fs, io, process};
use std::str::FromStr;
use tb_lang::{TB, Config, ExecutionMode, CompilationTarget, Compiler, TargetPlatform, Parser, Lexer, TBCore, DependencyCompiler, Value, streaming, STRING_INTERNER};
use std::env;
use std::io::Write;
use bumpalo::Bump;
use tb_lang::streaming::StreamingExecutor;

fn main() {
    let default_stack = 3 * 1024 * 1024; // Fallback: 3 MB
    let stack_size = env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(default_stack);

    std::thread::Builder::new()
        .stack_size(stack_size)
        .spawn(|| {
            actual_main();
        })
        .unwrap()
        .join()
        .unwrap();
}
fn actual_main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    let result = match args[1].as_str() {
        "run" => handle_run(&args[2..]),
        "compile" => handle_compile(&args[2..]),
        "deps" => handle_deps(&args[2..]),
        "build" => handle_build(&args[2..]),
        "repl" => handle_repl(&args[2..]),
        "check" => handle_check(&args[2..]),
        "version" | "-v" | "--version" => {
            println!("TB Language v1.0.0");
            Ok(())
        }
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            eprintln!("Run 'tb help' for usage information");
            Err(())
        }
    };

    if result.is_err() {
        process::exit(1);
    }
}

fn handle_run(args: &[String]) -> Result<(), ()> {
    if args.is_empty() {
        eprintln!("Error: No file specified");
        eprintln!("Usage: tb run <file.tb> [--mode <compiled|jit|streaming>]");
        return Err(());
    }

    let file_path = Path::new(&args[0]);

    eprintln!("[DEBUG] Starting execution of: {}", file_path.display());

    // Parse options
    let mut mode = ExecutionMode::Jit { cache_enabled: true };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                if i + 1 < args.len() {
                    mode = match args[i + 1].as_str() {
                        "compiled" => ExecutionMode::Compiled { optimize: true },
                        "jit" => ExecutionMode::Jit { cache_enabled: true },
                        "streaming" => ExecutionMode::Streaming {
                            auto_complete: true,
                            suggestions: true
                        },
                        m => {
                            eprintln!("Unknown mode: {}", m);
                            return Err(());
                        }
                    };
                    i += 2;
                } else {
                    eprintln!("--mode requires an argument");
                    return Err(());
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                return Err(());
            }
        }
    }

    // println!("╔════════════════════════════════════════════════════════════════╗");
    // println!("║                    TB Language Execution                       ║");
    // println!("╚════════════════════════════════════════════════════════════════╝");
    // println!();
    // println!("📄 File: {}", file_path.display());
    // println!("⚙️  Mode: {}", mode_to_string(&mode));
    // println!();

    let mut config = Config::default();
    config.mode = mode;

    let mut tb = TB::with_config(config);

    match tb.execute_file(file_path) {
        Ok(value) => {
            print!("{}", value);
            Ok(())
        }
        Err(e) => {
            eprintln!("[DEBUG] Execution failed: {}", e); // ADD THIS
            eprintln!("\n✗ Execution failed:");
            eprintln!("{}", e);
            Err(())
        }
    }
}

fn handle_deps(args: &[String]) -> Result<(), ()> {
    if args.is_empty() {
        eprintln!("Usage: tb deps <command>");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  compile <file>    Compile all dependencies");
        eprintln!("  list <file>       List dependencies");
        eprintln!("  clean             Clean dependency cache");
        return Err(());
    }

    match args[0].as_str() {
        "compile" => {
            if args.len() < 2 {
                eprintln!("Usage: tb deps compile <file>");
                return Err(());
            }

            let input = Path::new(&args[1]);
            let source = fs::read_to_string(input).map_err(|e| {
                eprintln!("Failed to read file: {}", e);
            })?;

            // Parse and analyze
            let clean_source = TBCore::strip_directives(&source);
            let mut lexer = Lexer::new(&clean_source);
            let tokens = lexer.tokenize().map_err(|e| {
                eprintln!("Tokenization failed: {}", e);
            })?;

            let arena = Bump::new();
            let mut parser = Parser::new(tokens, &arena);
            let statements = parser.parse().map_err(|e| {
                eprintln!("Parse failed: {}", e);
            })?;

            let core = TBCore::new(Config::default());
            let dependencies = core.analyze_dependencies(&statements).map_err(|e| {
                eprintln!("Analysis failed: {}", e);
            })?;

            println!("Found {} dependencies", dependencies.len());

            let compiler = DependencyCompiler::new(Path::new("."));

            for dep in dependencies {
                println!("\n📦 {}: {:?}", dep.id, dep.language);
                match compiler.compile(&dep) {
                    Ok(compiled) => {
                        println!("  ✓ Success");
                        println!("    Strategy: {:?}", compiled.strategy);
                        println!("    Output: {}", compiled.output_path.display());
                        println!("    Size: {} bytes", compiled.size_bytes);
                        println!("    Time: {}ms", compiled.compile_time_ms);
                    }
                    Err(e) => {
                        eprintln!("  ✗ Failed: {}", e);
                    }
                }
            }

            Ok(())
        }

        "list" => {
            if args.len() < 2 {
                eprintln!("Usage: tb deps list <file>");
                return Err(());
            }

            // Similar to compile but just list
            println!("Dependency listing not yet fully implemented");
            Ok(())
        }

        "clean" => {
            let cache_dir = Path::new(".tb_cache");
            if cache_dir.exists() {
                fs::remove_dir_all(cache_dir).map_err(|e| {
                    eprintln!("Failed to clean cache: {}", e);
                })?;
                println!("✓ Cache cleaned");
            } else {
                println!("No cache to clean");
            }
            Ok(())
        }

        _ => {
            eprintln!("Unknown deps command: {}", args[0]);
            Err(())
        }
    }
}


fn handle_compile(args: &[String]) -> Result<(), ()> {
    if args.len() < 2 {
        eprintln!("Error: Not enough arguments");
        eprintln!("Usage: tb compile <input.tb> <output> [--target <platform>] [--optimize <level>]");
        return Err(());
    }

    let input_path = Path::new(&args[0]);
    let output_path = Path::new(&args[1]);

    let mut target = TargetPlatform::current();
    let mut optimize = true;

    // Parse options
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                if i + 1 < args.len() {
                    target = TargetPlatform::from_str(&args[i + 1])
                        .map_err(|e| {
                            eprintln!("Error: {}", e);
                        })?;
                    i += 2;
                } else {
                    eprintln!("--target requires an argument");
                    return Err(());
                }
            }
            "--no-optimize" => {
                optimize = false;
                i += 1;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                return Err(());
            }
        }
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    TB Language Compiler                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("📄 Input:     {}", input_path.display());
    println!("📦 Output:    {}", output_path.display());
    println!("🎯 Target:    {}", target);
    println!("⚡ Optimize:  {}", if optimize { "Yes (O3)" } else { "No" });

    if target.is_cross_compile() {
        println!("🔄 Cross-compile mode");
    }

    println!();
    println!("⚙️  Compiling...");

    // Read source
    let source = fs::read_to_string(input_path).map_err(|e| {
        eprintln!("Failed to read file: {}", e);
    })?;

    // ✅ FIX: Use TBCore to handle imports properly
    let mut config = Config::parse(&source).map_err(|e| {
        eprintln!("Config parse failed: {}", e);
    })?;

    // Override with CLI options
    config.mode = ExecutionMode::Compiled { optimize };
    config.target = match target {
        TargetPlatform::Wasm => CompilationTarget::Wasm,
        _ => CompilationTarget::Native,
    };

    // Create TBCore with config
    let mut tb_core = TBCore::new(config.clone());

    // Use compile_to_file which will load imports
    tb_core.compile_to_file(&source, output_path).map_err(|e| {
        eprintln!("Compilation failed: {}", e);
    })?;

    println!("✓ Compilation successful!");

    // Show binary info
    if let Ok(metadata) = fs::metadata(output_path) {
        let size_kb = metadata.len() as f64 / 1024.0;
        println!("📊 Binary size: {:.2} KB", size_kb);
    }

    println!();
    println!("🚀 Run with: {}", output_path.display());

    Ok(())
}

// ADD handle_build():
fn handle_build(args: &[String]) -> Result<(), ()> {
    if args.is_empty() {
        eprintln!("Usage: tb build <input.tb> [OPTIONS]");
        eprintln!();
        eprintln!("OPTIONS:");
        eprintln!("  --targets <platforms>    Comma-separated list (e.g., linux-x64,windows-x64,macos-arm64)");
        eprintln!("  --output <dir>           Output directory (default: dist/)");
        eprintln!("  --name <name>            Binary name (default: app)");
        eprintln!();
        eprintln!("EXAMPLES:");
        eprintln!("  tb build app.tb");
        eprintln!("  tb build app.tb --targets linux-x64,windows-x64,wasm");
        eprintln!("  tb build app.tb --output bin/ --name myapp");
        return Err(());
    }

    let input = Path::new(&args[0]);

    let mut targets = vec![TargetPlatform::current()];
    let mut output_dir = PathBuf::from("dist");
    let mut app_name = "app".to_string();

    // Parse args
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--targets" => {
                if i + 1 < args.len() {
                    targets = args[i + 1]
                        .split(',')
                        .filter_map(|t| TargetPlatform::from_str(t.trim()).ok())
                        .collect();

                    if targets.is_empty() {
                        eprintln!("Error: No valid targets specified");
                        return Err(());
                    }
                    i += 2;
                } else {
                    eprintln!("--targets requires argument");
                    return Err(());
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("--output requires argument");
                    return Err(());
                }
            }
            "--name" => {
                if i + 1 < args.len() {
                    app_name = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("--name requires argument");
                    return Err(());
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                return Err(());
            }
        }
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    TB Multi-Platform Build                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("📄 Source:  {}", input.display());
    println!("📂 Output:  {}", output_dir.display());
    println!("📦 Name:    {}", app_name);
    println!("🎯 Targets: {}", targets.len());
    for target in &targets {
        println!("   • {}", target);
    }
    println!();

    // Load and parse source
    let source = fs::read_to_string(input).map_err(|e| {
        eprintln!("Failed to read source: {}", e);
    })?;

    let clean_source = TBCore::strip_directives(&source);
    let mut lexer = Lexer::new(&clean_source);
    let tokens = lexer.tokenize().map_err(|e| {
        eprintln!("Tokenization failed: {}", e);
    })?;

    let arena = Bump::new();
    let mut parser = Parser::new(tokens, &arena);
    let statements = parser.parse().map_err(|e| {
        eprintln!("Parse failed: {}", e);
    })?;

    // Build for each target
    let mut success_count = 0;

    for target in &targets {
        print!("🔨 Building for {}...", target);
        use std::io::Write;
        std::io::stdout().flush().ok();

        let config = Config {
            mode: ExecutionMode::Compiled { optimize: true },
            target: if matches!(target, TargetPlatform::Wasm) {
                CompilationTarget::Wasm
            } else {
                CompilationTarget::Native
            },
            ..Config::default()
        };

        let compiler = Compiler::new(config)
            .with_target(*target)
            .with_optimization(3);

        // Create output path
        let target_dir = output_dir.join(target.platform_name());
        fs::create_dir_all(&target_dir).ok();

        let output_file = target_dir.join(format!(
            "{}{}",
            app_name,
            target.exe_extension()
        ));

        match compiler.compile_to_file(&statements, &output_file) {
            Ok(_) => {
                let size_kb = fs::metadata(&output_file)
                    .map(|m| m.len() as f64 / 1024.0)
                    .unwrap_or(0.0);

                println!(" ✓ ({:.2} KB)", size_kb);
                success_count += 1;
            }
            Err(e) => {
                println!(" ✗");
                eprintln!("   Error: {}", e);
            }
        }
    }

    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    if success_count == targets.len() {
        println!("║  ✓ Build successful for all {} targets!                      ║", success_count);
    } else {
        println!("║  ⚠ Build completed: {}/{} targets successful                 ║",
                 success_count, targets.len());
    }
    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// REPL
pub fn handle_repl(_args: &[String]) -> Result<(), ()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              TB Interactive Shell v2.0 - Enhanced              ║");
    println!("║  Type ':help' for commands, 'exit' to quit                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let arena: &'static Bump = Box::leak(Box::new(Bump::new()));

    let mut executor = StreamingExecutor::new(arena);
    let mut line_number = 1;

    loop {
        // Dynamic prompt showing mode
        let mode_char = match executor.config.mode {
            ExecutionMode::Jit { .. } => 'J',
            ExecutionMode::Compiled { .. } => 'C',
            ExecutionMode::Streaming { .. } => 'S',
        };

        print!("tb[{}:{}]> ", line_number, mode_char);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            eprintln!("Error reading input. Exiting.");
            break;
        }

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle exit
        if matches!(input, "exit" | "quit" | ":quit" | ":exit") {
            println!("Goodbye!");
            break;
        }

        // Handle clear screen
        if input == "clear" || input == ":clear" {
            print!("\x1B[2J\x1B[1;1H");
            continue;
        }

        if input == "help" || input == ":help" || input == ":h" || input == "h" || input == ":?" || input == "?" {
            print_repl_help();
            continue;
        }

        // Execute line
        match executor.execute_line(input) {
            Ok(response) => {
                match response {
                    streaming::ReplResponse::Value { value, execution_time } => {
                        // Don't print Unit values
                        if !matches!(value, Value::Unit) {
                            println!("✓ {} ({}ms)", value, execution_time.as_millis());
                        }
                    }

                    streaming::ReplResponse::Message(msg) => {
                        println!("{}", msg);
                    }

                    streaming::ReplResponse::Incomplete => {
                        // Multi-line input continues
                        print!("  ... ");
                        io::stdout().flush().unwrap();
                        continue;
                    }
                }
            }

            Err(e) => {
                eprintln!("✗ Error: {}", e.detailed_message());
            }
        }

        line_number += 1;
    }

    // Print final stats before exit
    println!("\n{}", executor.format_metrics());
    println!("{}", STRING_INTERNER.health_report());

    Ok(())
}

/// Helper print functions
fn print_repl_help() {
    println!("{}", streaming::StreamingExecutor::help_text());
}

fn handle_check(args: &[String]) -> Result<(), ()> {
    if args.is_empty() {
        eprintln!("Error: No file specified");
        eprintln!("Usage: tb check <file.tb>");
        return Err(());
    }

    let file_path = Path::new(&args[0]);

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    TB Language Checker                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("📄 Checking: {}", file_path.display());
    println!();

    let source = std::fs::read_to_string(file_path).map_err(|e| {
        eprintln!("Failed to read file: {}", e);
    })?;

    // Try to parse and type-check without executing
    use tb_lang::{Lexer, Parser, TypeChecker, TypeMode};

    let mut lexer = Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("✗ Lexer error: {}", e);
            return Err(());
        }
    };

    println!("✓ Tokenization successful ({} tokens)", tokens.len());

    let arena = Bump::new();
    let mut parser = Parser::new(tokens, &arena);
    let statements = match parser.parse() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("✗ Parse error: {}", e);
            return Err(());
        }
    };

    println!("✓ Parsing successful ({} statements)", statements.len());

    let mut type_checker = TypeChecker::new(TypeMode::Static);
    match type_checker.check_statements(&statements) {
        Ok(_) => {
            println!("✓ Type checking passed");
        }
        Err(e) => {
            eprintln!("✗ Type error: {}", e);
            return Err(());
        }
    }

    println!();
    println!("✓ All checks passed!");

    Ok(())
}

fn print_help() {
    println!(r#"
╔════════════════════════════════════════════════════════════════╗
║                    TB Language v1.0.0                          ║
║          Unified Multi-Language Programming Environment        ║
╚════════════════════════════════════════════════════════════════╝

USAGE:
    tb <COMMAND> [OPTIONS]

COMMANDS:
    run <file>              Execute a TB program (JIT mode)
    compile <in> <out>      Compile TB to native binary
    build <file>            Build for multiple platforms
    repl                    Start interactive shell
    check <file>            Check syntax and types
    version                 Show version information
    help                    Show this help message

COMPILATION:
    compile <input.tb> <output> [OPTIONS]
        --target <platform>     Target platform (default: current)
        --no-optimize          Disable optimizations

    Supported targets:
        linux-x64              Linux x86_64
        linux-arm64            Linux ARM64
        windows-x64            Windows x86_64
        windows-arm64          Windows ARM64
        macos-x64              macOS Intel
        macos-arm64            macOS Apple Silicon
        wasm                   WebAssembly
        android-arm64          Android ARM64
        android-x64            Android x86_64
        ios-arm64              iOS ARM64
        ios-simulator          iOS Simulator

MULTI-PLATFORM BUILD:
    build <input.tb> [OPTIONS]
        --targets <list>       Comma-separated platforms
        --output <dir>         Output directory (default: dist/)
        --name <name>          Binary name (default: app)

    Examples:
        tb build app.tb
        tb build app.tb --targets linux-x64,windows-x64,macos-arm64
        tb build app.tb --targets wasm --name webapp

EXECUTION MODES:
    --mode <MODE>           Execution mode
        compiled            Native compilation (fastest)
        jit                 Just-in-time (fast startup)
        streaming           Interactive with live feedback

EXAMPLES:
    # Run program
    tb run examples/hello.tb

    # Compile for current platform
    tb compile app.tb myapp

    # Cross-compile for Windows
    tb compile app.tb myapp.exe --target windows-x64

    # Build for all major platforms
    tb build app.tb --targets linux-x64,windows-x64,macos-arm64,wasm

    # Compile WASM for web
    tb compile app.tb app.wasm --target wasm

    # Android build
    tb compile app.tb app --target android-arm64

PERFORMANCE:
    Compiled binaries achieve near-native Rust performance:
    • 0-5% overhead vs handwritten Rust
    • Full LLVM optimizations (LTO, target-cpu)
    • Strip symbols for minimal size
    • Memory footprint: 10MB - 500MB

For more information, visit: https://github.com/yourusername/tb-lang
"#);
}


