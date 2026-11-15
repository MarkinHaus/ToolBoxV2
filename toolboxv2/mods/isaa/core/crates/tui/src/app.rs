//! Ebene 1 (Zen Mode) + Ebene 2 (Zen+ Inspector).
//!
//! Zen: kein Alternate Screen, normaler Scrollback, differenzielle Updates
//! über ratatui Inline-Viewport. Inspector: Alternate Screen via RAII-Guard.
//! Events werden in einem `spawn_blocking`-Poller gelesen (crossterm poll/read),
//! damit der Tokio-Executor nie blockiert wird.

use std::io;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, poll, read};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};
use tokio::sync::{mpsc, watch};
use tracing::warn;

use isaa_core::{State, TuiUpdate, Version};

use crate::guard::TerminalGuard;
use crate::logic::{project_dag_tree, render_token_meter};

const HUD_HEIGHT: u16 = 3;

type ZenTerminal = Terminal<CrosstermBackend<io::Stdout>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Zen,
    /// Ctrl+C HUD, nicht-destruktiv.
    Interrupt,
    /// Zen+ Inspector (Alternate Screen).
    Inspector,
}

#[derive(Clone, Debug)]
pub struct ToolLineView {
    pub label: String,
    pub ok: bool,
    pub ms: u32,
    /// Letzte Zeilen bei Fehler (aufgeklapptes Accordion).
    pub tail: Vec<String>,
}

/// Reiner UI-Zustand, akkumuliert aus `TuiUpdate` + State-Snapshots.
#[derive(Default)]
pub struct UiState {
    pub tool_lines: Vec<ToolLineView>,
    /// Wie viele Zeilen schon in den Scrollback geflusht wurden.
    pub printed: usize,
    pub token_used: u32,
    pub token_budget: u32,
    pub dag_version: Version,
    pub node_count: usize,
    pub state: Option<Arc<State>>,
    /// Raw-JSON-Protokoll für den Inspector (rechte Spalte).
    pub inspector_log: Vec<String>,
}

impl UiState {
    pub fn apply(&mut self, u: &TuiUpdate) {
        match u {
            TuiUpdate::ToolLine { label, ok, ms, .. } => {
                self.tool_lines.push(ToolLineView {
                    label: label.clone(),
                    ok: *ok,
                    ms: *ms,
                    tail: Vec::new(),
                });
            }
            TuiUpdate::TokenMeter { used, budget } => {
                self.token_used = *used;
                self.token_budget = *budget;
            }
            TuiUpdate::DagSnapshot { version, nodes } => {
                self.dag_version = *version;
                self.node_count = *nodes;
            }
        }
        if let Ok(json) = serde_json::to_string(u) {
            self.inspector_log.push(json);
            let cap = 512;
            if self.inspector_log.len() > cap {
                let drop = self.inspector_log.len() - cap;
                self.inspector_log.drain(0..drop);
            }
        }
    }
}

/// `◇ fs_read {src/lib.rs} ── [✓] (8ms)` — Fehler: ✗.
/// ponytail: Kein Zeilenzahl-Suffix (`120L`) und kein Error-Tail-Text, weil
/// `TuiUpdate::ToolLine` weder Zeilenzahl noch Fehlertext trägt — Feld in
/// isaa-core ergänzen, wenn gewünscht.
pub fn format_tool_line(v: &ToolLineView) -> String {
    let sym = if v.ok { "✓" } else { "✗" };
    format!("◇ {} ── [{sym}] ({}ms)", v.label, v.ms)
}

fn ui_lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Haupt-Loop: konsumiert State-Watches + TuiUpdates, rendert Zen Mode und
/// Zen+ Inspector. Blockiert den Tokio-Executor nicht.
pub async fn run(
    mut state_rx: watch::Receiver<Arc<State>>,
    mut upd_rx: mpsc::Receiver<TuiUpdate>,
) -> io::Result<()> {
    let ui = Arc::new(Mutex::new(UiState::default()));

    // Event-Quelle: eigener Blocking-Thread (Poll), nie der Executor.
    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<Event>();
    tokio::task::spawn_blocking(move || loop {
        match poll(Duration::from_millis(100)) {
            Ok(true) => match read() {
                Ok(ev) => {
                    if ev_tx.send(ev).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("crossterm read failed: {e}");
                    break;
                }
            },
            Ok(false) => {}
            Err(e) => {
                warn!("crossterm poll failed: {e}");
                break;
            }
        }
    });

    let mut zen = Terminal::with_options(
        CrosstermBackend::new(io::stdout()),
        TerminalOptions {
            viewport: Viewport::Inline(HUD_HEIGHT),
        },
    )?;
    let mut insp: Option<(ZenTerminal, TerminalGuard)> = None;
    let mut mode = Mode::Zen;

    loop {
        tokio::select! {
            r = state_rx.changed() => {
                if r.is_err() {
                    break;
                }
                let st = state_rx.borrow_and_update().clone();
                ui_lock(&ui).state = Some(st);
            }
            u = upd_rx.recv() => {
                let Some(u) = u else { break };
                let unprinted = {
                    let mut g = ui_lock(&ui);
                    g.apply(&u);
                    g.tool_lines.len() - g.printed
                };
                if unprinted > 0 && insp.is_none() {
                    let mut g = ui_lock(&ui);
                    for v in &g.tool_lines[g.printed..] {
                        let line = format_tool_line(v);
                        zen.insert_before(1, |buf| {
                            buf.set_line(0, 0, &Line::from(line.clone()), buf.area.width);
                        })?;
                    }
                    g.printed = g.tool_lines.len();
                }
            }
            ev = ev_rx.recv() => {
                let Some(ev) = ev else { break };
                handle_event(&ev, &mut mode, &mut insp, &ui, &mut zen)?;
            }
        }

        // Einheitlicher Redraw-Punkt: differenziell, kein Full-Repaint-Spam.
        match insp.as_mut() {
            Some((t, _)) => {
                let g = ui_lock(&ui);
                t.draw(|f| render_inspector(f, &g))?;
            }
            None => {
                zen.draw(|f| {
                    let g = ui_lock(&ui);
                    render_zen(f, &g, mode);
                })?;
            }
        }
    }

    // Restore garantiert (auch über Drop des Guards bei ?-Fehlern oben).
    if let Some((_, guard)) = insp.take() {
        drop(guard);
    }
    Ok(())
}

fn handle_event(
    ev: &Event,
    mode: &mut Mode,
    insp: &mut Option<(ZenTerminal, TerminalGuard)>,
    _ui: &Mutex<UiState>,
    zen: &mut ZenTerminal,
) -> io::Result<()> {
    let Event::Key(k) = ev else {
        return Ok(());
    };
    if k.kind != KeyEventKind::Press {
        return Ok(());
    }
    let ctrl_c = k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL);
    match mode {
        Mode::Zen => match k.code {
            KeyCode::F(2) | KeyCode::Tab => match TerminalGuard::new() {
                Ok(guard) => match Terminal::new(CrosstermBackend::new(io::stdout())) {
                    Ok(t) => {
                        insp.replace((t, guard));
                        *mode = Mode::Inspector;
                    }
                    Err(e) => {
                        drop(guard);
                        warn!("inspector terminal failed: {e}");
                    }
                },
                Err(e) => warn!("cannot enter inspector: {e}"),
            },
            _ if ctrl_c => *mode = Mode::Interrupt,
            _ => {}
        },
        Mode::Interrupt => match k.code {
            KeyCode::Esc => *mode = Mode::Zen,
            _ if ctrl_c => *mode = Mode::Zen,
            KeyCode::Char(c) => {
                if let Some(note) = interrupt_action(c) {
                    let line = format!("→ {note}");
                    zen.insert_before(1, |buf| {
                        buf.set_line(0, 0, &Line::from(line), buf.area.width);
                    })?;
                    *mode = Mode::Zen;
                }
            }
            _ => {}
        },
        Mode::Inspector => {
            if matches!(k.code, KeyCode::Esc | KeyCode::F(2) | KeyCode::Tab) || ctrl_c {
                // Nahtlose Rückkehr: Guard-Drop restauriert Alt-Screen/Cursor.
                if let Some((_, guard)) = insp.take() {
                    drop(guard);
                }
                zen.clear()?;
                *mode = Mode::Zen;
            }
        }
    }
    Ok(())
}

/// ponytail: HUD-Auswahl nur Display + Scrollback-Notiz — es existiert noch
/// kein Rückkanal zur Shell; sobald ein Command-Kanal existiert, hier anbinden.
fn interrupt_action(c: char) -> Option<&'static str> {
    match c {
        'i' => Some("info snapshot in scrollback"),
        'b' => Some("background jobs listed"),
        's' => Some("stop requested (non-destructive)"),
        'r' => Some("context report queued"),
        'n' => Some("next step queued"),
        _ => None,
    }
}

fn render_zen(f: &mut Frame<'_>, ui: &UiState, mode: Mode) {
    let meter = render_token_meter(ui.token_used, ui.token_budget);
    let meta = format!("dag v{} · {} nodes", ui.dag_version, ui.node_count);
    let mut lines = vec![
        Line::from(vec![
            Span::raw(meter),
            Span::raw("  "),
            Span::raw(meta),
        ]),
        Line::from("[F2/Tab] Inspector   [Ctrl+C] Interrupt"),
    ];
    if mode == Mode::Interrupt {
        lines.push(Line::from(
            "[i] Info [b] Background [s] Stop [r] Context [n] Next   [Esc] Dismiss",
        ));
    }
    f.render_widget(Paragraph::new(lines), f.area());
}

fn s1_metrics(s: &State) -> String {
    let n = s.decision_cache.len();
    let avg_bps = if n == 0 {
        0
    } else {
        s.decision_cache
            .values()
            .map(|d| d.confidence_bps as u64)
            .sum::<u64>()
            / n as u64
    };
    format!(
        "version {}\nactive_adapter {}\ndecisions {}\navg_confidence {avg_bps}bps ({}%)",
        s.version,
        s.active_adapter.as_deref().unwrap_or("<none>"),
        n,
        avg_bps / 100
    )
}

fn render_inspector(f: &mut Frame<'_>, ui: &UiState) {
    let [left, right] = Layout::horizontal([
        Constraint::Percentage(50),
        Constraint::Percentage(50),
    ])
    .areas(f.area());

    let mut left_lines: Vec<Line> = vec![Line::from("── Trajectory DAG ──")];
    if let Some(st) = &ui.state {
        for l in project_dag_tree(st).lines() {
            left_lines.push(Line::from(l.to_string()));
        }
        left_lines.push(Line::from(""));
        left_lines.push(Line::from("── System 1 ──"));
        for l in s1_metrics(st).lines() {
            left_lines.push(Line::from(l.to_string()));
        }
    }
    f.render_widget(
        Paragraph::new(left_lines)
            .block(Block::default().borders(Borders::ALL).title(" Tree / S1 ")),
        left,
    );

    // Rechte Spalte: neueste Raw-JSON-Zeilen unten.
    let h = right.height.saturating_sub(2) as usize;
    let skip = ui.inspector_log.len().saturating_sub(h);
    let right_lines: Vec<Line> = ui.inspector_log[skip..]
        .iter()
        .map(|s| Line::from(s.clone()))
        .collect();
    f.render_widget(
        Paragraph::new(right_lines)
            .block(Block::default().borders(Borders::ALL).title(" Raw JSON I/O ")),
        right,
    );
}
