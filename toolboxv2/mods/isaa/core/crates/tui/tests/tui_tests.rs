use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use isaa_core::{Effect, Event, NodeId, NodeKind, State, TuiUpdate, WorkspacePolicy};
use isaa_core::reduce;
use isaa_tui::{format_tool_line, project_dag_tree, render_token_meter, TerminalGuard, UiState};

// ---------- Helpers ----------

/// State via öffentlicher Reducer-Kette (keine direkten im/slotmap-Zugriffe nötig).
fn build_tree_state() -> State {
    let empty = State::new(WorkspacePolicy::default());
    let (st, fx) = reduce(
        &empty,
        &Event::TaskSubmitted {
            goal: "root goal".into(),
        },
    );
    let parent = fx
        .iter()
        .find_map(|e| match e {
            Effect::InvokeS1 { node, .. } => Some(*node),
            _ => None,
        })
        .expect("TaskSubmitted emits InvokeS1");
    let (st, _) = reduce(
        &st,
        &Event::SubDagExpanded {
            parent,
            nodes: vec![
                NodeKind::Gate {
                    check: "coverage>=0.8".into(),
                },
                NodeKind::Composite {
                    goal: "child goal".into(),
                },
            ],
            deps: vec![(0, 1)],
        },
    );
    st
}

// ---------- 1. State-Projektion ----------

#[test]
fn dag_tree_contains_nodes_status_in_stable_order() {
    let tree = project_dag_tree(&build_tree_state());

    assert!(tree.contains("n0 [Running] composite \"root goal\""), "{tree}");
    assert!(tree.contains("n1 [Pending] gate \"coverage>=0.8\""), "{tree}");
    assert!(
        tree.contains("└─ n2 [Pending] composite \"child goal\""),
        "{tree}"
    );

    // Stabile Reihenfolge.
    let i_root = tree.find("root goal").unwrap();
    let i_gate = tree.find("coverage>=0.8").unwrap();
    let i_child = tree.find("child goal").unwrap();
    assert!(i_root < i_gate && i_gate < i_child);

    // Deterministisch über Rekonstruktion.
    assert_eq!(tree, project_dag_tree(&build_tree_state()));
}

#[test]
fn dag_tree_empty_state_is_empty() {
    assert_eq!(project_dag_tree(&State::new(WorkspacePolicy::default())), "");
}

// ---------- 2. Token-Meter ----------

#[test]
fn token_meter_exact_format() {
    assert_eq!(
        render_token_meter(14200, 128000),
        "ctx 14.2k/128k [██░░░░░░░░] 11%"
    );
}

#[test]
fn token_meter_edges() {
    assert_eq!(render_token_meter(0, 128000), "ctx 0/128k [░░░░░░░░░░] 0%");
    assert_eq!(
        render_token_meter(128000, 128000),
        "ctx 128k/128k [██████████] 100%"
    );
    // Budget 0: kein Divide-by-zero, leere Anzeige.
    assert_eq!(render_token_meter(100, 0), "ctx 100/0 [░░░░░░░░░░] 0%");
}

#[test]
fn token_meter_rounding() {
    assert_eq!(
        render_token_meter(64000, 128000),
        "ctx 64k/128k [█████░░░░░] 50%"
    );
    // 997/128000 = 0.78% -> 0.78*10 = 7.8 -> rundet auf 1 Block.
    assert_eq!(
        render_token_meter(997, 128000),
        "ctx 997/128k [█░░░░░░░░░] 1%"
    );
    // Over-budget wird geklemmt (100%, voller Balken).
    assert_eq!(
        render_token_meter(200000, 128000),
        "ctx 200k/128k [██████████] 100%"
    );
}

// ---------- 3. RAII TerminalGuard ----------

#[test]
fn guard_drop_runs_restore_hooks_in_order() {
    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let l = log.clone();
    let guard = TerminalGuard::with_restore(
        || Ok(()),
        move || {
            l.lock().unwrap().push("leave_alternate_screen");
            l.lock().unwrap().push("cursor_show");
        },
    );
    drop(guard);
    assert_eq!(
        *log.lock().unwrap(),
        vec!["leave_alternate_screen", "cursor_show"]
    );
}

#[test]
fn guard_restores_on_panic_unwind() {
    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let l = log.clone();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _guard = TerminalGuard::with_restore(
            || Ok(()),
            move || l.lock().unwrap().push("restored"),
        );
        panic!("boom");
    }));
    assert!(result.is_err());
    assert_eq!(*log.lock().unwrap(), vec!["restored"]);
}

// ---------- Bonus: UI-State-Akkumulation (pure Logik) ----------

#[test]
fn ui_state_applies_updates_and_logs_json() {
    let mut ui = UiState::default();
    ui.apply(&TuiUpdate::TokenMeter {
        used: 14200,
        budget: 128000,
    });
    assert_eq!(
        render_token_meter(ui.token_used, ui.token_budget),
        "ctx 14.2k/128k [██░░░░░░░░] 11%"
    );

    ui.apply(&TuiUpdate::ToolLine {
        node: NodeId::default(),
        label: "fs_read".into(),
        ok: true,
        ms: 8,
    });
    assert_eq!(format_tool_line(&ui.tool_lines[0]), "◇ fs_read ── [✓] (8ms)");

    ui.apply(&TuiUpdate::DagSnapshot {
        version: 3,
        nodes: 5,
    });
    assert_eq!(ui.dag_version, 3);
    assert_eq!(ui.node_count, 5);

    assert_eq!(ui.inspector_log.len(), 3);
    assert!(ui.inspector_log[0].starts_with('{'));
}
