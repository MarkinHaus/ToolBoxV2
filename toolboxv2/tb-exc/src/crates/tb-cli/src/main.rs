use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;
use std::sync::Arc;
use tb_cache::CacheManager;
use tb_core::StringInterner;

mod repl;
mod runner;

#[derive(Parser)]
#[command(name = "tb")]
#[command(about = "TB Language Compiler & Interpreter", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a TB script
    Run {
        /// Path to the script file
        file: PathBuf,

        /// Execution mode
        #[arg(short = 'm', long, default_value = "jit")]
        mode: String,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "2")]
        opt_level: u8,
    },

    /// Compile a TB script to native binary
    Compile {
        /// Path to the script file
        file: PathBuf,

        /// Output file
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "3")]
        opt_level: u8,
    },

    /// Start interactive REPL
    Repl {
        /// Initial mode
        #[arg(short, long, default_value = "jit")]
        mode: String,
    },

    /// Cache management
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Show version and build info
    Version,
}

#[derive(Subcommand)]
enum CacheAction {
    /// Clear all caches
    Clear,

    /// Show cache statistics
    Stats,
}

fn main() -> anyhow::Result<()> {
    // Enable Rust backtraces in debug mode
    #[cfg(debug_assertions)]
    {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    let cli = Cli::parse();

    // Initialize global resources
    let interner = Arc::new(StringInterner::new(Default::default()));
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tb-lang");
    let cache_manager = Arc::new(CacheManager::new(cache_dir, 100)?);

    match cli.command {
        Commands::Run { file, mode, opt_level } => {
            runner::run_file(
                &file,
                &mode,
                opt_level,
                Arc::clone(&interner),
                Arc::clone(&cache_manager),
            )?;
        }

        Commands::Compile { file, output, opt_level } => {
            let output_file = output.unwrap_or_else(|| {
                let mut path = file.clone();
                path.set_extension("");
                path
            });

            println!("{} {}", "Compiling:".bright_green().bold(), file.display());

            let start = std::time::Instant::now();

            let result = runner::compile_file(
                &file,
                &output_file,
                opt_level,
                Arc::clone(&interner),
                Arc::clone(&cache_manager),
            );

            // Handle errors with detailed messages
            if let Err(e) = result {
                eprintln!("{}", e.detailed_message());
                std::process::exit(1);
            }

            let elapsed = start.elapsed();

            println!(
                "{} {} ({:?})",
                "Success:".bright_green().bold(),
                output_file.display(),
                elapsed
            );
        }

        Commands::Repl { mode } => {
            println!("{}", "TB Language REPL".bright_cyan().bold());
            println!("{} {}", "Mode:".bright_blue(), mode);
            println!("{}", "Type :help for commands".bright_black());
            println!();

            repl::start_repl(&mode, interner, cache_manager)?;
        }

        Commands::Cache { action } => match action {
            CacheAction::Clear => {
                cache_manager.import_cache().clear()?;
                cache_manager.clear_hot_cache();
                println!("{}", "Cache cleared".bright_green());
            }
            CacheAction::Stats => {
                let import_stats = cache_manager.import_cache().stats();
                let cache_stats = cache_manager.stats();
                let interner_stats = interner.stats();

                println!("{}", "Cache Statistics".bright_cyan().bold());
                println!();
                println!("{}", "Import Cache:".bright_yellow());
                println!("  Entries: {}", import_stats.entries);
                println!("  Size: {} MB", import_stats.total_bytes / (1024 * 1024));
                println!();
                println!("{}", "Hot Cache:".bright_yellow());
                println!("  Entries: {}", cache_stats.hot_cache_entries);
                println!("  Size: {} MB", cache_stats.hot_cache_bytes / (1024 * 1024));
                println!();
                println!("{}", "String Interner:".bright_yellow());
                println!("  Entries: {}", interner_stats.total_entries);
                println!("  Hit Rate: {:.1}%", interner_stats.hit_rate * 100.0);
                println!("  Memory Saved: {} KB", interner_stats.total_saved_bytes / 1024);
            }
        },

        Commands::Version => {
            println!("{} {}", "TB Language".bright_cyan().bold(), env!("CARGO_PKG_VERSION"));
            println!("Built with Rust {}", rustc_version());
        }
    }

    Ok(())
}

fn rustc_version() -> String {
    option_env!("RUSTC_VERSION")
        .unwrap_or("unknown")
        .to_string()
}




