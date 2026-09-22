# TASK: Ticket 8 – 2-Ebenen TUI: Zen Mode & Zen+ Ratatui Inspector

Du arbeitest im isolierten Git-Worktree auf dem Branch `feat/tui`. Nur das
Crate `crates/tui` (Paketname: `isaa-tui`) wird implementiert. `crates/core`
und `crates/harness` NICHT anfassen.

## ZIEL VON TICKET 8

1. **Ebene 1 (Zen Mode, Default)**:
   - Normaler Terminal-Scrollback (KEIN Alternate Screen).
   - Flackerfreies Rendering über `crossterm` (differenzielle Updates, kein Full-Repaint-Spam).
   - Einzeilige Tool-Accordions: `◇ fs_read {src/lib.rs} ── [✓ 120L] (8ms)`;
     bei Fehler (✗) klappen die letzten Zeilen auf.
   - Live Token-Meter: `ctx 14.2k/128k [██░░░░░░░░] 11%`.
   - Nicht-destruktives Interrupt-HUD bei `Ctrl+C`:
     `[i] Info [b] Background [s] Stop [r] Context [n] Next`.
2. **Ebene 2 (Zen+ Inspector, F2 / Tab)**:
   - Umschalten auf Alternate Screen (`EnterAlternateScreen`).
   - Split-Screen: links Trajectory-DAG / Task-Baum (`/tree`) + System-1-Metriken;
     rechts Raw JSON I/O & Diffs.
   - `[Esc]` schließt sofort, nahtlose Rückkehr in den Zen Mode.
   - **RAII-Guard**: `TerminalGuard` garantiert via `Drop`:
     `LeaveAlternateScreen` + `cursor::Show` werden IMMER ausgeführt (auch bei Panic).

## ARCHITEKTUR & SCHNITTSTELLEN

- Zustand/Updates ausschließlich über:
  - `tokio::sync::watch::Receiver<std::sync::Arc<isaa_core::State>>`
  - `tokio::sync::mpsc::Receiver<TuiUpdate>` (`TuiUpdate` existiert bereits in `isaa_core::domain`)
- `crates/tui` blockiert zu keiner Zeit den Tokio-Executor des Kerns
  (Rendering in eigenen Tasks; Event-Loop via `crossterm::event::EventStream`).
- Reine Logik (Projektion, Formatierung) in frei testbaren Funktionen halten,
  getrennt vom Terminal-I/O:
  - `project_dag_tree(&State) -> String` (ASCII-Baum der DAG-Knoten inkl. Status)
  - `render_token_meter(used: u32, budget: u32) -> String` (exaktes Format wie oben)
  - `TerminalGuard::new()/Drop` für den RAII-Test

## INTEGRATIONSTESTS (`crates/tui/tests/tui_tests.rs`)

Headless (kein Terminal nötig):
1. State-Projektion: `State` mit einigen Knoten/Kanten → DAG-ASCII-Tree-String
   enthält Knoten-IDs/Status in stabiler Reihenfolge.
2. Token-Meter-Renderer: Format-Genauigkeit (z. B. 14200/128000 →
   `ctx 14.2k/128k [██░░░░░░░░] 11%`), Rundung, Randfälle (0, budget 0).
3. RAII: `TerminalGuard` anlegen und droppen → Restore-Aufrufe erfolgen
   (z. B. über injizierbares Fake-Terminal/Callback oder garantierte Drop-Semantik
   ohne Panic).

## Regeln & DoD

1. Keine neuen Dependencies: crates/tui hat bereits isaa-core, tokio, ratatui,
   crossterm, serde_json. Falls crossterm `event-stream` im Workspace fehlt:
   Workspace-Cargo.toml darf NICHT geändert werden — nutze die vorhandene
   Feature-Kombination (Poll-basiertes Event-Reading in `spawn_blocking`/Task geht notfalls).
2. Keine OS-`#[cfg]`-Blöcke (`#[cfg(test)]` ok).
3. Kein globales `cargo fmt` — nur neue/bearbeitete Dateien.
4. Gates (IM Workflow-Root des Worktrees ausführen):
   - `cargo test -p isaa-tui` grün
   - `cargo clippy -p isaa-tui --all-targets -- -D warnings` = 0
5. Abschluss: alles auf `feat/tui` committen (git add NUR crates/tui-Dateien).

Berichte am Ende: Dateien + Kurzfassung Testausgabe.
