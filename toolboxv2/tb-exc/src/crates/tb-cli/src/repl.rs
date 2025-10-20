use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;
use tb_cache::CacheManager;
use tb_core::{Result, StringInterner, Value};
use tb_jit::JitExecutor;
use tb_parser::{Lexer, Parser};

pub fn start_repl(
    mode: &str,
    interner: Arc<StringInterner>,
    _cache_manager: Arc<CacheManager>,
) -> anyhow::Result<()> {
    let mut rl = DefaultEditor::new()?;
    let mut executor = JitExecutor::new();
    let mut line_number = 1;

    loop {
        let prompt = format!("tb[{}:{}]> ", line_number, mode.chars().next().unwrap().to_uppercase());

        match rl.readline(&prompt) {
            Ok(line) => {
                if line.trim().is_empty() {
                    continue;
                }

                rl.add_history_entry(&line)?;

                // Handle REPL commands
                if line.starts_with(':') {
                    handle_command(&line, &mut executor)?;
                    continue;
                }

                // Execute code
                match execute_line(&line, &mut executor, &interner) {
                    Ok(value) => {
                        println!("{} {} ({})", "✓".bright_green(), value, value.type_name().bright_black());
                    }
                    Err(e) => {
                        println!("{} {}", "✗".bright_red(), e);
                    }
                }

                line_number += 1;
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("exit");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

fn execute_line(
    line: &str,
    executor: &mut JitExecutor,
    interner: &Arc<StringInterner>,
) -> Result<Value> {
    let mut lexer = Lexer::new(line, Arc::clone(interner));
    let tokens = lexer.tokenize();

    let mut parser = Parser::new_with_source(tokens, line.to_string());
    let program = parser.parse()?;

    executor.execute(&program)
}

fn handle_command(command: &str, _executor: &mut JitExecutor) -> anyhow::Result<()> {
    let parts: Vec<&str> = command[1..].split_whitespace().collect();

    match parts.get(0).copied() {
        Some("help") => {
            println!("{}", "REPL Commands:".bright_cyan().bold());
            println!("  :help           Show this help");
            println!("  :clear          Clear screen");
            println!("  :stats          Show statistics");
            println!("  :exit           Exit REPL");
        }
        Some("clear") => {
            print!("\x1B[2J\x1B[1;1H");
        }
        Some("stats") => {
            println!("{}", "Statistics:".bright_cyan().bold());
            println!("  TODO: Implement stats");
        }
        Some("exit") => {
            std::process::exit(0);
        }
        _ => {
            println!("{} Unknown command: {}", "Error:".bright_red(), command);
        }
    }

    Ok(())
}

