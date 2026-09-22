use bytes::Bytes;
use im::{OrdMap as PMap, Vector as PVec};
use serde::{Deserialize, Serialize};
use slotmap::{new_key_type, SlotMap};
use std::path::PathBuf;

pub type Version = u64;
pub type Sig = [u8; 32]; // BLAKE3

new_key_type! { pub struct NodeId; }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspacePolicy {
    pub root: PathBuf,
    pub allowed_subprocesses: Vec<String>,
    pub max_fuel_per_wasm: u64,
}

impl Default for WorkspacePolicy {
    fn default() -> Self {
        Self {
            root: PathBuf::from("workspace"),
            allowed_subprocesses: Vec::new(),
            max_fuel_per_wasm: 1 << 30,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    Atomic { tool: String, args: Bytes },
    Composite { goal: String },
    Gate { check: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus { Pending, Ready, Running, Done, Failed }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub status: NodeStatus,
    pub sig: Sig,
    pub attempts: u8,
    pub result: Option<Bytes>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    pub task_category: String,
    pub confidence_bps: u16, // 0..10_000
    pub complexity: f32,     // 0.0..3.0
    pub coverage: f32,       // 0.0..1.0 (BM25++ Signal)
    pub chosen_action: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct State {
    pub version: Version,
    pub workspace: WorkspacePolicy,
    pub nodes: SlotMap<NodeId, DagNode>,
    /// OrdMap (not HashMap): iteration order is key-sorted, hence postcard
    /// serialization is bit-stable across runs. im::HashMap seeds RandomState
    /// per process and would break state-hash determinism.
    pub edges: PMap<NodeId, PVec<NodeId>>, // Deps -> Dependents
    pub decision_cache: PMap<Sig, Decision>,
    pub active_adapter: Option<String>,    // System 2.5 LoRA Adapter ID
}

impl State {
    pub fn new(workspace: WorkspacePolicy) -> Self {
        Self {
            version: 0,
            workspace,
            nodes: SlotMap::with_key(),
            edges: PMap::new(),
            decision_cache: PMap::new(),
            active_adapter: None,
        }
    }

    /// Bit-identical fingerprint of a state (postcard + BLAKE3).
    pub fn hash(&self) -> Sig {
        let bytes = postcard::to_stdvec(self).expect("state is postcard-serializable");
        blake3::hash(&bytes).into()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Event {
    TaskSubmitted { goal: String },
    DecisionArrived { node: NodeId, decision: Decision },
    SubDagExpanded { parent: NodeId, nodes: Vec<NodeKind>, deps: Vec<(usize, usize)> },
    ToolStarted { node: NodeId },
    ToolCompleted { node: NodeId, result: Bytes, ok: bool },
    RollbackRequested { target_version: Version },
    ArtifactInstalled { playbook_name: String, content: String },
}

#[derive(Clone, Debug)]
pub enum Effect {
    InvokeS1 { node: NodeId, state_sig: Sig, goal: String },
    InvokeS2Plan { node: NodeId, goal: String },
    ExecuteTool { node: NodeId, tool: String, args: Bytes, idem_key: Sig },
    PersistTrace { row: Box<TraceRow> },
    WriteVfs { path: PathBuf, content: String },
    EmitTui { update: TuiUpdate },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TuiUpdate {
    ToolLine { node: NodeId, label: String, ok: bool, ms: u32 },
    TokenMeter { used: u32, budget: u32 },
    DagSnapshot { version: Version, nodes: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRow {
    pub trace_id: Sig,
    pub task_family: String,
    pub active_adapter: Option<String>,
    pub s1_confidence_bps: u16,
    pub input_projection: String,
    pub action_taken: String,
    pub outcome: Outcome, // VerifiedSuccess | VerifiedFailure
    pub latency_ms: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    VerifiedSuccess,
    VerifiedFailure,
    Unknown,
}
