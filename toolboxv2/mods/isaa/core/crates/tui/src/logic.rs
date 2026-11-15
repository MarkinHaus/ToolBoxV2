//! Pure projection & formatting logic. No terminal I/O — headless testable.

use std::collections::{HashMap, HashSet};

use isaa_core::{NodeKind, NodeStatus, State};

/// `ctx 14.2k/128k [██░░░░░░░░] 11%`
pub fn render_token_meter(used: u32, budget: u32) -> String {
    let ratio = if budget == 0 {
        0.0
    } else {
        used as f64 / budget as f64
    };
    let pct = ((ratio * 100.0).round() as i64).clamp(0, 100);
    let filled = ((pct + 9) / 10).clamp(0, 10) as usize;
    let bar = "█".repeat(filled) + &"░".repeat(10 - filled);
    format!("ctx {}/{} [{bar}] {pct}%", fmt_k(used), fmt_k(budget))
}

/// 999 -> "999", 14_200 -> "14.2k", 128_000 -> "128k"
fn fmt_k(v: u32) -> String {
    if v < 1000 {
        return v.to_string();
    }
    let s = format!("{:.1}k", v as f64 / 1000.0);
    match s.strip_suffix(".0k") {
        Some(head) => format!("{head}k"),
        None => s,
    }
}

fn status_str(s: NodeStatus) -> String {
    format!("{s:?}")
}

fn kind_label(k: &NodeKind) -> String {
    match k {
        NodeKind::Atomic { tool, .. } => format!("atomic {tool}"),
        NodeKind::Composite { goal } => format!("composite \"{goal}\""),
        NodeKind::Gate { check } => format!("gate \"{check}\""),
    }
}

/// ASCII-Baum der DAG-Knoten inkl. Status, stabile Reihenfolge
/// (SlotMap-Insertionsreihenfolge; Kinder in OrdMap-Key-Sortierung).
/// Knoten mit mehreren Eltern erscheinen genau einmal (erster Pfad gewinnt).
pub fn project_dag_tree(state: &State) -> String {
    let has_parent: HashSet<isaa_core::NodeId> = state
        .edges
        .values()
        .flatten()
        .copied()
        .collect();
    let index: HashMap<isaa_core::NodeId, usize> = state
        .nodes
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (id, i))
        .collect();
    let mut out = String::new();
    let mut visited = HashSet::new();
    for (id, _) in state.nodes.iter() {
        if has_parent.contains(&id) {
            continue;
        }
        walk(state, id, "", &index, &mut out, &mut visited);
    }
    out
}

fn walk(
    state: &State,
    id: isaa_core::NodeId,
    prefix: &str,
    index: &HashMap<isaa_core::NodeId, usize>,
    out: &mut String,
    visited: &mut HashSet<isaa_core::NodeId>,
) {
    let node = &state.nodes[id];
    let line = format!(
        "{prefix}n{} [{}] {}",
        index[&id],
        status_str(node.status),
        kind_label(&node.kind)
    );
    out.push_str(&line);
    out.push('\n');
    if !visited.insert(id) {
        return;
    }
    if let Some(children) = state.edges.get(&id) {
        let n = children.len();
        for (j, child) in children.iter().enumerate() {
            let last = j + 1 == n;
            if last {
                walk(state, *child, &format!("{prefix}└─ "), index, out, visited);
            } else {
                walk(state, *child, &format!("{prefix}├─ "), index, out, visited);
            }
        }
    }
}
