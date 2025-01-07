from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import time
import json
from copy import deepcopy


@dataclass
class NetworkBranch:
    branch_id: str
    parent_branch_id: Optional[str]
    timestamp: float
    ring_states: Dict[str, Dict]
    connection_states: Dict[str, Dict[str, float]]
    active_concepts: Dict[str, Set[str]]
    metrics: Dict[str, any]


@dataclass
class NetworkStateSnapshot:
    timestamp: float
    active_ring: str
    active_concepts: Dict[str, List[Dict]]
    connection_strengths: Dict[str, Dict[str, float]]
    metrics: Dict[str, any]


@dataclass
class NetworkDataHolder:
    """Data holder for managing network state and visualization data"""
    network_id: str
    history: List[NetworkStateSnapshot] = field(default_factory=list)
    branches: Dict[str, NetworkBranch] = field(default_factory=dict)
    current_branch_id: str = "main"
    max_history_length: int = 100

    def capture_state(self, network) -> NetworkStateSnapshot:
        """Capture current network state"""
        active_concepts = {}
        for ring_id, ring in network.rings.items():
            active_concepts[ring_id] = [
                {
                    "id": concept.id,
                    "name": concept.name,
                    "stage": concept.stage,
                    "metadata": concept.metadata
                }
                for concept in ring.concept_graph.concepts.values()
                if concept.stage > 0
            ]

        snapshot = NetworkStateSnapshot(
            timestamp=time.time(),
            active_ring=network.metrics.last_activated,
            active_concepts=active_concepts,
            connection_strengths=deepcopy(network.connections),
            metrics=network.get_metrics()
        )

        self._add_to_history(snapshot)
        return snapshot

    def create_branch(self, network, branch_name: str) -> str:
        """Create a new branch from current state"""
        branch_id = f"{branch_name}-{int(time.time())}"

        branch = NetworkBranch(
            branch_id=branch_id,
            parent_branch_id=self.current_branch_id,
            timestamp=time.time(),
            ring_states={
                ring_id: {
                    "concepts": [c.to_dict() for c in ring.concept_graph.concepts.values()],
                    "stage": ring.stage
                }
                for ring_id, ring in network.rings.items()
            },
            connection_states=deepcopy(network.connections),
            active_concepts={
                ring_id: {c.id for c in ring.concept_graph.concepts.values() if c.stage > 0}
                for ring_id, ring in network.rings.items()
            },
            metrics=network.get_metrics()
        )

        self.branches[branch_id] = branch
        return branch_id

    def switch_branch(self, network, branch_id: str) -> bool:
        """Switch to a different branch"""
        if branch_id not in self.branches:
            return False

        branch = self.branches[branch_id]

        # Restore network state from branch
        for ring_id, state in branch.ring_states.items():
            if ring_id in network.rings:
                network.rings[ring_id].concept_graph.concepts.clear()
                for concept_data in state["concepts"]:
                    network.rings[ring_id].concept_graph.add_concept_from_dict(concept_data)
                network.rings[ring_id].stage = state["stage"]

        network.connections = deepcopy(branch.connection_states)
        network.metrics.base_concepts = branch.active_concepts

        self.current_branch_id = branch_id
        return True

    def _add_to_history(self, snapshot: NetworkStateSnapshot):
        """Add snapshot to history with length limit"""
        self.history.append(snapshot)
        if len(self.history) > self.max_history_length:
            self.history.pop(0)

    def get_visualization_data(self) -> Dict:
        """Get data formatted for visualization"""
        if not self.history:
            return {}

        latest = self.history[-1]
        return {
            "network_id": self.network_id,
            "active_ring": latest.active_ring,
            "rings": {
                ring_id: {
                    "concepts": concepts,
                    "connections": latest.connection_strengths.get(ring_id, {})
                }
                for ring_id, concepts in latest.active_concepts.items()
            },
            "metrics": latest.metrics,
            "branches": [
                {
                    "id": b_id,
                    "parent": branch.parent_branch_id,
                    "timestamp": branch.timestamp
                }
                for b_id, branch in self.branches.items()
            ]
        }

    def step_back(self, network, steps: int = 1) -> bool:
        """Step back in history by specified number of steps"""
        if len(self.history) < steps or steps < 1:
            return False

        target_snapshot = self.history[-steps]

        # Restore network state
        for ring_id, concepts in target_snapshot.active_concepts.items():
            if ring_id in network.rings:
                network.rings[ring_id].concept_graph.concepts.clear()
                for concept_data in concepts:
                    network.rings[ring_id].concept_graph.add_concept(
                        concept_data["id"],
                        concept_data["name"],
                        concept_data["stage"],
                        concept_data["metadata"]
                    )

        network.connections = deepcopy(target_snapshot.connection_strengths)
        network.metrics.last_activated = target_snapshot.active_ring

        # Remove stepped back snapshots
        self.history = self.history[:-steps]
        return True

    def export_state(self, filepath: str):
        """Export current state to file"""
        state = {
            "network_id": self.network_id,
            "current_branch": self.current_branch_id,
            "branches": {
                b_id: {
                    "parent": branch.parent_branch_id,
                    "timestamp": branch.timestamp,
                    "ring_states": branch.ring_states,
                    "connection_states": branch.connection_states,
                    "active_concepts": {k: list(v) for k, v in branch.active_concepts.items()},
                    "metrics": branch.metrics
                }
                for b_id, branch in self.branches.items()
            },
            "history": [
                {
                    "timestamp": snap.timestamp,
                    "active_ring": snap.active_ring,
                    "active_concepts": snap.active_concepts,
                    "connection_strengths": snap.connection_strengths,
                    "metrics": snap.metrics
                }
                for snap in self.history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def import_state(cls, filepath: str) -> 'NetworkDataHolder':
        """Import state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        holder = cls(network_id=state["network_id"])
        holder.current_branch_id = state["current_branch"]

        # Restore branches
        for b_id, b_data in state["branches"].items():
            holder.branches[b_id] = NetworkBranch(
                branch_id=b_id,
                parent_branch_id=b_data["parent"],
                timestamp=b_data["timestamp"],
                ring_states=b_data["ring_states"],
                connection_states=b_data["connection_states"],
                active_concepts={k: set(v) for k, v in b_data["active_concepts"].items()},
                metrics=b_data["metrics"]
            )

        # Restore history
        holder.history = [
            NetworkStateSnapshot(
                timestamp=snap["timestamp"],
                active_ring=snap["active_ring"],
                active_concepts=snap["active_concepts"],
                connection_strengths=snap["connection_strengths"],
                metrics=snap["metrics"]
            )
            for snap in state["history"]
        ]

        return holder
