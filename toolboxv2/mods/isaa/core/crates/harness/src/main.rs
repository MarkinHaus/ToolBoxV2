use isaa_core::{reduce, Event, State, WorkspacePolicy};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut state = State::new(WorkspacePolicy::default());
    let (next, fx) = reduce(&state, &Event::TaskSubmitted { goal: "demo task".into() });
    state = next;

    println!("state v{} ({} nodes), {} effects", state.version, state.nodes.len(), fx.len());
}
