use isaa_core::*;
use bytes::Bytes;
use proptest::prelude::*;
use slotmap::KeyData;

fn arb_kind() -> impl Strategy<Value = NodeKind> {
    prop_oneof![
        ("[a-z_]{1,8}", proptest::collection::vec(any::<u8>(), 0..16))
            .prop_map(|(tool, args)| NodeKind::Atomic { tool, args: Bytes::from(args) }),
        "[a-z ]{1,24}".prop_map(|goal| NodeKind::Composite { goal }),
        "[a-z ]{1,24}".prop_map(|check| NodeKind::Gate { check }),
    ]
}

fn arb_node_id() -> impl Strategy<Value = NodeId> {
    (0u64..8).prop_map(|i| NodeId::from(KeyData::from_ffi(i)))
}

fn arb_decision() -> impl Strategy<Value = Decision> {
    (
        "[a-z]{1,8}",
        0u16..10_000,
        0.0f32..3.0,
        0.0f32..1.0,
        proptest::option::of("[a-z_]{1,8}"),
    )
        .prop_map(|(task_category, confidence_bps, complexity, coverage, chosen_action)| Decision {
            task_category,
            confidence_bps,
            complexity,
            coverage,
            chosen_action,
        })
}

fn arb_event() -> impl Strategy<Value = Event> {
    prop_oneof![
        "[a-z ]{1,24}".prop_map(|goal| Event::TaskSubmitted { goal }),
        (arb_node_id(), arb_decision())
            .prop_map(|(node, decision)| Event::DecisionArrived { node, decision }),
        (
            arb_node_id(),
            proptest::collection::vec(arb_kind(), 0..4),
            proptest::collection::vec((0usize..4, 0usize..4), 0..4),
        )
            .prop_map(|(parent, nodes, deps)| Event::SubDagExpanded { parent, nodes, deps }),
        arb_node_id().prop_map(|node| Event::ToolStarted { node }),
        (arb_node_id(), proptest::collection::vec(any::<u8>(), 0..16), any::<bool>())
            .prop_map(|(node, result, ok)| Event::ToolCompleted { node, result: Bytes::from(result), ok }),
        (0u64..1000).prop_map(|target_version| Event::RollbackRequested { target_version }),
        ("[a-z_]{1,8}", "[a-z ]{1,24}")
            .prop_map(|(playbook_name, content)| Event::ArtifactInstalled { playbook_name, content }),
    ]
}

fn arb_events() -> impl Strategy<Value = Vec<Event>> {
    proptest::collection::vec(arb_event(), 0..32)
}

fn initial_state() -> State {
    State::new(WorkspacePolicy::default())
}

proptest! {
    /// Identical event sequence applied to identical initial state
    /// must yield bit-identical states (BLAKE3 over postcard bytes).
    #[test]
    fn fold_is_deterministic(events in arb_events()) {
        let run = || events.iter().fold(initial_state(), |s, e| reduce(&s, e).0);
        let a = run();
        let b = run();
        prop_assert_eq!(a.hash(), b.hash());
        prop_assert_eq!(a.version, events.len() as u64);
    }

    /// Reducing the same event against clones of the same state
    /// must produce bit-identical successors.
    #[test]
    fn single_step_is_deterministic(events in arb_events()) {
        let s0 = events.iter().fold(initial_state(), |s, e| reduce(&s, e).0);
        for e in &events {
            let (a, fa) = reduce(&s0, e);
            let (b, fb) = reduce(&s0, e);
            prop_assert_eq!(a.hash(), b.hash());
            prop_assert_eq!(format!("{fa:?}"), format!("{fb:?}"));
        }
    }

    /// Version strictly increases by one per event; reducer emits no clock/RNG state.
    #[test]
    fn version_monotonic(events in arb_events()) {
        let mut s = initial_state();
        let mut prev = s.version;
        for e in &events {
            let (next, _) = reduce(&s, e);
            prop_assert_eq!(next.version, prev + 1);
            s = next;
            prev = s.version;
        }
    }
}
