use crate::domain::*;
use bytes::Bytes;
use smallvec::SmallVec;

/// Pure reducer: no I/O, no clock, no RNG. Deterministic by construction.
pub fn reduce(state: &State, event: &Event) -> (State, SmallVec<[Effect; 4]>) {
    let mut next = state.clone();
    let mut fx = SmallVec::new();
    next.version += 1;

    match event {
        Event::TaskSubmitted { goal } => {
            let sig = blake3::hash(goal.as_bytes()).into();
            let node_id = next.nodes.insert(DagNode {
                id: NodeId::default(),
                kind: NodeKind::Composite { goal: goal.clone() },
                status: NodeStatus::Ready,
                sig,
                attempts: 0,
                result: None,
            });
            next.nodes[node_id].id = node_id;

            // Ask System 1 immediately.
            fx.push(Effect::InvokeS1 {
                node: node_id,
                state_sig: sig,
                goal: goal.clone(),
            });
            fx.push(Effect::EmitTui {
                update: TuiUpdate::DagSnapshot {
                    version: next.version,
                    nodes: next.nodes.len(),
                },
            });
        }

        Event::DecisionArrived { node, decision } => {
            let confident = decision.confidence_bps >= 7500;
            let familiar = decision.coverage >= 0.35;
            let simple = decision.complexity < 1.5;

            if confident && familiar && simple {
                if let Some(tool) = &decision.chosen_action {
                    if let Some(n) = next.nodes.get_mut(*node) {
                        n.status = NodeStatus::Running;
                        fx.push(Effect::ExecuteTool {
                            node: *node,
                            tool: tool.clone(),
                            args: Bytes::new(),
                            idem_key: n.sig,
                        });
                    }
                }
            } else {
                // Escalate to System 2 / 2.5.
                if let Some(n) = next.nodes.get(*node) {
                    if let NodeKind::Composite { goal } = &n.kind {
                        fx.push(Effect::InvokeS2Plan {
                            node: *node,
                            goal: goal.clone(),
                        });
                    }
                }
            }
        }

        Event::SubDagExpanded {
            parent,
            nodes: kinds,
            deps,
        } => {
            let mut ids: Vec<NodeId> = Vec::with_capacity(kinds.len());
            for kind in kinds {
                let bytes = postcard::to_stdvec(kind).unwrap_or_default();
                let sig: Sig = blake3::hash(&bytes).into();
                let id = next.nodes.insert(DagNode {
                    id: NodeId::default(),
                    kind: kind.clone(),
                    status: NodeStatus::Pending,
                    sig,
                    attempts: 0,
                    result: None,
                });
                next.nodes[id].id = id;
                ids.push(id);
            }
            for (from, to) in deps {
                if let (Some(&a), Some(&b)) = (ids.get(*from), ids.get(*to)) {
                    match next.edges.get_mut(&a) {
                        Some(v) => v.push_back(b),
                        None => {
                            next.edges.insert(a, im::vector![b]);
                        }
                    }
                }
            }
            // Parent is now decomposed: children carry the work.
            if let Some(p) = next.nodes.get_mut(*parent) {
                if p.status == NodeStatus::Ready {
                    p.status = NodeStatus::Running;
                }
            }
            fx.push(Effect::EmitTui {
                update: TuiUpdate::DagSnapshot {
                    version: next.version,
                    nodes: next.nodes.len(),
                },
            });
        }

        Event::ToolStarted { node } => {
            if let Some(n) = next.nodes.get_mut(*node) {
                if matches!(n.status, NodeStatus::Pending | NodeStatus::Ready) {
                    n.status = NodeStatus::Running;
                }
                n.attempts = n.attempts.saturating_add(1);
            }
        }

        Event::ToolCompleted { node, result, ok } => {
            if let Some(n) = next.nodes.get_mut(*node) {
                n.status = if *ok {
                    NodeStatus::Done
                } else {
                    NodeStatus::Failed
                };
                n.result = Some(result.clone());
            }

            let (outcome, action) = if *ok {
                (Outcome::VerifiedSuccess, "tool_exec")
            } else {
                (Outcome::VerifiedFailure, "tool_exec")
            };
            fx.push(Effect::PersistTrace {
                row: Box::new(TraceRow {
                    trace_id: state.nodes.get(*node).map(|n| n.sig).unwrap_or([0; 32]),
                    task_family: "general".into(),
                    active_adapter: next.active_adapter.clone(),
                    s1_confidence_bps: 9000,
                    input_projection: String::new(),
                    action_taken: action.into(),
                    outcome,
                    latency_ms: 10,
                }),
            });
            fx.push(Effect::EmitTui {
                update: TuiUpdate::ToolLine {
                    node: *node,
                    label: "tool".into(),
                    ok: *ok,
                    ms: 10,
                },
            });
        }

        Event::ArtifactInstalled {
            playbook_name,
            content,
        } => {
            let target_path = next.workspace.root.join("playbooks").join(playbook_name);
            fx.push(Effect::WriteVfs {
                path: target_path,
                content: content.clone(),
            });
        }

        Event::RollbackRequested { target_version: _ } => {
            // Rollback is signalled only; the O(1) pointer swap happens in the shell.
        }
    }

    (next, fx)
}
