import os
import random
import threading
import base64
import hashlib
import json
import zlib
from collections import defaultdict
from datetime import datetime

import networkx as nx
from tqdm import tqdm

from toolboxv2.utils.Irings.NetworkDataHolder import NetworkDataHolder
from toolboxv2.utils.Irings.tk_live import NetworkVisualizer, process_with_visualization
from toolboxv2.utils.Irings.utils import SemanticConceptSplitter, SubConcept
from toolboxv2.utils.Irings.Optimizer import RingRestructurer, TopologyOptimizer
from toolboxv2.utils.Irings.one import IntelligenceRing, Concept
from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np
from dataclasses import dataclass

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import math
from tqdm import tqdm


class ParallelProcessor:
    def __init__(self, original_processor):
        self.processor = original_processor
        # Locks for thread safety
        self.metrics_lock = Lock()
        self.rings_lock = Lock()

    def process_batch(self, batch):
        """Process a batch of subconcepts"""
        batch_results = []
        for subconcept in batch:
            try:
                ring_id = self.processor._route_information(subconcept.vector)

                # Handle visualization if present
                if hasattr(self.processor, 'v') and self.processor.v is not None:
                    with self.rings_lock:
                        self.processor.v.mark_active(ring_id)

                # Process in ring with lock for thread safety
                with self.rings_lock:
                    self.processor._process_in_ring(
                        ring_id,
                        subconcept.metadata["text"],
                        subconcept.vector
                    )

                # Update metrics with lock
                with self.metrics_lock:
                    self.processor._update_metrics(ring_id)

                # Handle restructuring if needed
                if subconcept.importance > 0.65:
                    with self.rings_lock:
                        RingRestructurer(self.processor.rings[ring_id]).restructure()

                batch_results.append(ring_id)
            except Exception as e:
                print(f"Error processing subconcept: {e}")
                batch_results.append(None)

        return batch_results


def parallel_process(self, subconcepts, max_workers=None):
    """
    Process subconcepts in parallel using ThreadPool
    """
    # Calculate batch size (min of 100 or 10% of input size)
    total_size = len(subconcepts)
    batch_size = min(10, max(1, total_size // 10))

    # Create batches
    batches = [
        subconcepts[i:i + batch_size]
        for i in range(0, total_size, batch_size)
    ]

    # Initialize parallel processor
    processor = ParallelProcessor(self)
    results = []

    # Default to number of CPUs if max_workers not specified
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    # Process batches in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(processor.process_batch, batch) for batch in batches]

        # Show progress with tqdm
        for future in tqdm(
            futures,
            desc="Processing batches",
            total=len(batches)
        ):
            batch_results = future.result()
            results.extend(batch_results)

    return results

class RingSerializer:

    def _concept_to_dict(self, concept: Concept) -> Dict[str, Any]:
        return {
            'id': concept.id,
            'name': concept.name,
            'ttl': int(concept.ttl),
            'created_at': concept.created_at.isoformat(),
            'vector': base64.b64encode(concept.vector.tobytes()).decode('utf-8'),
            'vector_shape': concept.vector.shape,
            'contradictions': list(concept.contradictions),
            'similar_concepts': list(concept.similar_concepts),
            'relations': {k: float(v) for k, v in concept.relations.items()},
            'stage': concept.stage,
            'metadata': concept.metadata

        }

    def _dict_to_concept(self, data: Dict[str, Any]) -> Concept:
        vector_bytes = base64.b64decode(data['vector'].encode('utf-8'))
        vector = np.frombuffer(vector_bytes, dtype=np.float32).reshape(data['vector_shape'])

        return Concept(
            id=data['id'],
            name=data['name'],
            ttl=data['ttl'],
            created_at=datetime.fromisoformat(data['created_at']),
            vector=vector,
            contradictions=set(data['contradictions']),
            similar_concepts=set(data['similar_concepts']),
            relations=data['relations'],
            stage=data['stage'],
            metadata=data['metadata']
        )

    def ring_to_hex(self, ring: IntelligenceRing) -> str:
        data = {
            'ring_id': ring.ring_id,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_threads': ring.num_threads,
                'max_concepts': ring.concept_graph.max_concepts,
                'max_relations': ring.concept_graph.max_relations
            }
        }
        json_str = json.dumps(data)
        compressed = zlib.compress(json_str.encode('utf-8'))
        hex_code = compressed.hex()
        checksum = hashlib.sha256(hex_code.encode()).hexdigest()[:8]

        return f"{checksum}:{hex_code}"

    def hex_to_ring(self, hex_string: str) -> IntelligenceRing:
        try:
            checksum, hex_code = hex_string.split(':')
            if checksum != hashlib.sha256(hex_code.encode()).hexdigest()[:8]:
                raise ValueError("Checksum verification failed")

            compressed = bytes.fromhex(hex_code)
            json_str = zlib.decompress(compressed).decode('utf-8')
            data = json.loads(json_str)

            ring = IntelligenceRing(
                ring_id=data['ring_id'],
                num_threads=data['metadata']['num_threads']
            )

            ring.concept_graph.max_concepts = data['metadata']['max_concepts']
            ring.concept_graph.max_relations = data['metadata']['max_relations']

            return ring

        except Exception as e:
            raise ValueError(f"Failed to deserialize ring: {str(e)}")


class RingFormatter:
    @staticmethod
    def format_ring_data(ring_data: dict, mode: str) -> str:
        formatters = {
            'explain': RingFormatter._format_explanation,
            'graph': RingFormatter._format_graph,
            'summary': RingFormatter._format_summary,
            'technical': RingFormatter._format_technical,
            'llm': RingFormatter._format_llm,
            'row': lambda x: x
        }
        return formatters.get(mode, RingFormatter._format_explanation)(ring_data)

    @staticmethod
    def _format_explanation(data: dict) -> str:
        concepts = sorted(data['concepts'], key=lambda x: x['stage'], reverse=True)

        output = [
            f"Ring ID: {data['ring_id']}",
            "\nKey Concepts:",
            *[f"- {c['name']} (Stage {c['stage']})" for c in concepts if c['stage'] > 2],
            "\nEmerging Concepts:",
            *[f"- {c['name']} (Stage {c['stage']})" for c in concepts if c['stage'] <= 2],
            f"\nTotal Concepts: {len(concepts)}",
            f"Created: {data['metadata']['timestamp']}"
        ]
        return '\n'.join(output)

    @staticmethod
    def _format_graph(data: dict) -> str:
        output = ["Concept Graph:", ""]
        for concept in data['concepts']:
            output.extend([
                f"{concept['name']}",
                f"├── Stage: {concept['stage']}",
                f"└── Metadata: {concept['metadata']}"
            ])
        return '\n'.join(output)

    @staticmethod
    def _format_summary(data: dict) -> str:
        mature_concepts = [c for c in data['concepts'] if c['stage'] > 2]
        return (f"Ring {data['ring_id']} contains {len(data['concepts'])} concepts "
                f"({len(mature_concepts)} mature)")

    @staticmethod
    def _format_technical(data: dict) -> str:
        return json.dumps(data, indent=2)

    @staticmethod
    def _format_llm(data: dict) -> str:
        concepts = sorted(data['concepts'], key=lambda x: x['stage'], reverse=True)

        output = [
            f"# Ring Analysis ({data['ring_id']})",
            f"## Maturity Analysis",
            f"- Total Concepts: {len(concepts)}",
            f"- Mature Concepts: {len([c for c in concepts if c['stage'] > 2])}",
            f"- Developing Concepts: {len([c for c in concepts if c['stage'] <= 2])}",
            "",
            "## Key Concepts",
            *[f"- {c['name']} [Stage {c['stage']}]" for c in concepts[:5]],
            "",
            "## Metadata",
            f"- Created: {data['metadata']['timestamp']}"
        ]
        return '\n'.join(output)


@dataclass
class NetworkMetrics:
    new_ideas: int = 0
    latest_concepts: List[str] = None
    base_concepts: Dict[str, Set[str]] = None
    last_activated: str = ""
    interactions: Dict[str, int] = None

    def __post_init__(self):
        if self.latest_concepts is None:
            self.latest_concepts = []
        if self.base_concepts is None:
            self.base_concepts = {}
        if self.interactions is None:
            self.interactions = defaultdict(int)


class NetworkState:
    ACTIVE = "active"
    FROZEN = "frozen"
    DISABLED = "disabled"


class NetworkManager:
    def __init__(self, num_empty_rings: int, preset_rings: List[IntelligenceRing] or None, name="MainN",
                 max_connections=None, max_new=15):
        self.max_connections = (num_empty_rings + (
            len(preset_rings) if preset_rings is not None else 0)) // 2 if max_connections is None else max_connections
        self.max_new = max_new
        self.name = name
        self.rings: Dict[str, IntelligenceRing] = {}
        self.connections: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.state = NetworkState.ACTIVE
        self.metrics = NetworkMetrics()
        self.history: List[Dict] = []
        self.lock = threading.Lock()
        self.data_holder = NetworkDataHolder(self.name, [])

        self.show = NetworkDisplayer(self)

        self._initialize_network(num_empty_rings, preset_rings)

        self.splitter = SemanticConceptSplitter(
            next(iter(self.rings.values())).input_processor
        )
        self.topology_optimizer = TopologyOptimizer(self)

    def _initialize_network(self, num_empty: int, presets: List[IntelligenceRing]):
        # Add preset rings
        if presets is None:
            presets = []
        for ring in presets:
            self.rings[ring.ring_id] = ring
            self.metrics.base_concepts[ring.ring_id] = {
                c.id for c in ring.concept_graph.concepts.values()
                if c.stage > 1
            }

        # Create empty rings
        for i in range(num_empty):
            ring_id = f"ring-{i}-" + self.name
            self.rings[ring_id] = IntelligenceRing(ring_id)

        # Initialize full connectivity
        self._establish_initial_connections()

    def add_ring(self, ring_id):
        self.rings[ring_id + '-' + self.name] = IntelligenceRing(ring_id + '-' + self.name)
        return ring_id + '-' + self.name

    def _establish_initial_connections(self):
        ring_ids = list(self.rings.keys())
        for i, ring_id in enumerate(ring_ids):
            for other_id in ring_ids[i + 1:]:
                weight = 1.0
                self.connections[ring_id][other_id] = weight
                self.connections[other_id][ring_id] = weight

                self.rings[ring_id].adapter.connect_ring(
                    self.rings[other_id].adapter
                )
                self.rings[other_id].adapter.connect_ring(
                    self.rings[ring_id].adapter
                )

    def process_input(self, text: str, v=None, get_subconcepts=False) -> List[str] or Tuple[
        List[str], List[SubConcept]]:
        if self.state != NetworkState.ACTIVE:
            raise ValueError("Network is not active")
        vector = self.rings[next(iter(self.rings))].input_processor.process_text(text)
        subconcepts = self.splitter.split(text, vector)
        self.metrics.latest_concepts = []
        # Process each subconcept
        results = []
        if len(subconcepts) < 10:
            for subconcept in tqdm(iterable=subconcepts, desc="Process", total=len(subconcepts)):
                ring_id = self._route_information(subconcept.vector)
                if v is not None:
                    v.mark_active(ring_id)
                self._process_in_ring(
                    ring_id,
                    subconcept.metadata["text"],
                    subconcept.vector
                )
                self._update_metrics(ring_id)
                results.append(ring_id)
                if subconcept.importance > 0.65:
                    RingRestructurer(self.rings[ring_id]).restructure()
        else:
            results = parallel_process(self, subconcepts)
        # Optimize network topology
        if get_subconcepts:
            return results, subconcepts
        return results

    def update_connections(self):
        self.topology_optimizer.optimize()

    def _route_information(self, vector: np.ndarray) -> str:
        best_ring_id = None
        best_similarity = -1

        for ring_id, ring in self.rings.items():
            if not ring.concept_graph.concepts:
                best_similarity = 0.5
                best_ring_id = ring_id if best_ring_id is None else best_ring_id
                continue
            c = list(ring.concept_graph.concepts.values())
            random.shuffle(c)
            for concept in c[:25]:
                if not concept.vector.shape and 'text' in concept.metadata:
                    concept.vector = ring.input_processor.process_text(concept.metadata.get('text', concept.name))
            # Get similarity to ring's key concepts
            c = list(ring.concept_graph.concepts.values())
            random.shuffle(c)
            similarities = [
                np.dot(vector, concept.vector)
                for concept in c[:25]
                if concept.stage > 0
            ]
            if similarities:
                max_sim = max(similarities)
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_ring_id = ring_id

        if best_similarity < 0.2:
            return self._get_empty_ring()

        if best_ring_id is None:
            return self.add_ring("new-" + str(len(self.rings.keys())))

        return best_ring_id

    def _process_in_ring(self, ring_id: str, text: str, vector: np.ndarray):
        ring = self.rings[ring_id]

        # If empty ring, establish key concept
        if not ring.concept_graph.concepts:
            concept_id = ring.process(text, metadata={"type": "key_concept"})
            return

        # Process in existing ring

        subconcepts = self.splitter.split(text, vector)

        # Process each subconcept
        for subconcept in subconcepts[:self.max_new]:
            parrent_id = ring.process(subconcept.metadata["text"],
                                      vector=subconcept.vector,
                                      importance=subconcept.importance,
                                      metadata={"type": "derived"})
            self.metrics.latest_concepts.append(f"{ring_id}:{parrent_id}")

        # Update connections based on concept relations
        self._update_connections(ring_id)

    def _is_connected(self, source_ring_id):
        if source_ring_id not in self.rings:
            source_ring_id = self.add_ring(source_ring_id.replace('-' + self.name, ''))
        if self.connections.keys() == self.rings.keys():
            return
        for ring_keys in self.rings.keys():
            if ring_keys not in self.connections:
                self.connections[ring_keys] = {}
        ring_ids = list(self.rings.keys())
        for i, ring_id in enumerate(ring_ids):
            weight = 1.0
            self.connections[ring_id][source_ring_id] = weight
            self.connections[source_ring_id][ring_id] = weight

            self.rings[ring_id].adapter.connect_ring(
                self.rings[source_ring_id].adapter
            )
            self.rings[source_ring_id].adapter.connect_ring(
                self.rings[ring_id].adapter
            )

    def _update_connections(self, source_ring_id: str):
        with self.lock:
            self._is_connected(source_ring_id)
            connections = self.connections.get(source_ring_id)

            # Remove lowest weight connection if too many
            if len(connections) > self.max_connections:
                min_ring = min(connections.items(), key=lambda x: x[1])[0]
                del connections[min_ring]
                if min_ring in self.connections and source_ring_id in self.connections[min_ring]:
                    del self.connections[min_ring][source_ring_id]
                if min_ring in self.rings[source_ring_id].adapter.connected_rings:
                    self.rings[source_ring_id].adapter.connected_rings.pop(min_ring)
                if source_ring_id in self.rings[min_ring].adapter.connected_rings:
                    self.rings[min_ring].adapter.connected_rings.pop(source_ring_id)

            # Update remaining weights based on concept similarity
            source_ring = self.rings[source_ring_id]
            for target_id in connections:
                target_ring = self.rings[target_id]

                sim = self._calculate_ring_similarity(
                    source_ring.concept_graph,
                    target_ring.concept_graph
                )
                connections[target_id] = sim
                self.connections[target_id][source_ring_id] = sim

    def _calculate_ring_similarity(self, graph1, graph2) -> float:
        similarities = []
        for c1 in list(graph1.concepts.values()).copy():
            for c2 in list(graph2.concepts.values()).copy():
                similarities.append(np.dot(c1.vector, c2.vector))
        return max(similarities) if similarities else 0.0

    def _get_empty_ring(self) -> str:
        for ring_id, ring in self.rings.items():
            if not ring.concept_graph.concepts:
                return ring_id
        return next(iter(self.rings))  # Fallback to first ring

    def _update_metrics(self, ring_id: str):
        self.metrics.last_activated = ring_id
        self.metrics.interactions[ring_id] += 1

        # Track concepts
        ring = self.rings[ring_id]
        current_concepts = {
            c.id for c in ring.concept_graph.concepts.values()
            if c.stage > 1
        }

        if ring_id in self.metrics.base_concepts:
            new = len(current_concepts - self.metrics.base_concepts[ring_id])
            self.metrics.new_ideas += new

        self.metrics.base_concepts[ring_id] = current_concepts

    def freeze(self):
        self.state = NetworkState.FROZEN
        self._save_state()

    def resume(self):
        self.state = NetworkState.ACTIVE

    def disable(self):
        self.state = NetworkState.DISABLED

    def step_back(self):
        if not self.history:
            return

        prev_state = self.history.pop()
        self.rings = prev_state["rings"]
        self.connections = prev_state["connections"]
        self.metrics = prev_state["metrics"]

    def _save_state(self):
        state = {
            "rings": self.rings.copy(),
            "connections": self.connections.copy(),
            "metrics": self.metrics
        }
        self.history.append(state)

    def get_metrics(self) -> Dict:
        return {
            "new_ideas": self.metrics.new_ideas,
            "base_concepts": {
                k: len(v) for k, v in self.metrics.base_concepts.items()
            },
            "last_activated": self.metrics.last_activated,
            "interactions": dict(self.metrics.interactions),
            "network_state": self.state
        }

    def get_references(self, text: str, top_k: int = 5, concept_elem=None, ring_id=None, all_=None) -> List[
        Tuple[str, float]]:
        v = self.rings[next(iter(self.rings))].input_processor.process_text(text)
        if v is None:
            v = self.rings[next(iter(self.rings))].input_processor.process_text(text[:50])
        sub_set = self.splitter.split(text, v)
        all_similarities = []

        for s in sub_set:
            rings = self.rings.items() if all_ is True else self._route_information(
                s.vector) if ring_id not in self.rings else ring_id
            rings = rings if all_ is True else [(rings, self.rings[rings])]
            for ring_id_, ring in rings:
                try:
                    similarities = ring.retrieval_system.find_similar(s.vector, top_k)
                except RuntimeError:
                    continue
                all_similarities.extend([
                    (
                        f"{ring_id_}:{concept_id}" if concept_elem is None else getattr(
                            self.get_concept_from_code(f"{ring_id_}:{concept_id}"), concept_elem), score)
                    for concept_id, score in similarities
                ])

        return sorted(all_similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def get_concept_from_code(self, code: str) -> Optional[Concept]:
        if ':' not in code:
            return None
        ring_id, concept_id = code.split(':')[0], code.split(':')[-1]
        if ring_id not in self.rings:
            return None
        return self.rings[ring_id].get_concept_by_id(concept_id)

    def get_best_ring(self, text, is_vector=False):
        if not is_vector:
            text = self.rings[next(iter(self.rings))].input_processor.process_text(text)
        return self._route_information(text)

    def save_to_file(self, filepath: str):
        serializer = NetworkSerializer()
        hex_data = serializer.network_to_hex(self)
        with open(filepath, 'w') as f:
            f.write(hex_data)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'NetworkManager':
        serializer = NetworkSerializer()
        with open(filepath, 'r') as f:
            hex_data = f.read()
        return serializer.hex_to_network(hex_data)


class NetworkSerializer:
    def __init__(self):
        self.ring_serializer = RingSerializer()

    def network_to_hex(self, network: NetworkManager) -> str:
        data = {
            'rings': {
                ring_id: self.ring_serializer.ring_to_hex(ring)
                for ring_id, ring in network.rings.items()
            },
            'connections': {k: {_k: float(_v) for _k, _v in v.items()} for k, v in network.connections.items()},
            'state': network.state,
            'name': network.name,
            'metrics': {
                'new_ideas': network.metrics.new_ideas,
                'base_concepts': {
                    k: list(v) for k, v in network.metrics.base_concepts.items()
                },
                'last_activated': network.metrics.last_activated,
                'interactions': dict(network.metrics.interactions)
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        json_str = json.dumps(data)
        compressed = zlib.compress(json_str.encode('utf-8'))
        hex_code = compressed.hex()
        checksum = hashlib.sha256(hex_code.encode()).hexdigest()[:8]

        return f"{checksum}:{hex_code}"

    def hex_to_network(self, hex_string: str) -> NetworkManager:
        # try:
        checksum, hex_code = hex_string.split(':')
        if checksum != hashlib.sha256(hex_code.encode()).hexdigest()[:8]:
            raise ValueError("Checksum verification failed")

        compressed = bytes.fromhex(hex_code)
        json_str = zlib.decompress(compressed).decode('utf-8')
        data = json.loads(json_str)

        # Restore rings
        rings = {
            ring_id: self.ring_serializer.hex_to_ring(ring_hex)
            for ring_id, ring_hex in data['rings'].items()
        }

        # Create network
        network = NetworkManager(0, list(rings.values()), name=data['name'])
        # network.rings = rings
        network.connections = data['connections']
        network.state = data['state']

        # Restore metrics
        network.metrics.new_ideas = data['metrics']['new_ideas']
        network.metrics.base_concepts = {
            k: set(v) for k, v in data['metrics']['base_concepts'].items()
        }
        network.metrics.last_activated = data['metrics']['last_activated']
        network.metrics.interactions = defaultdict(int, data['metrics']['interactions'])

        # Restore connections in ring adapters
        for ring_id, ring in network.rings.items():
            if ring_id not in network.connections:
                for r in network.rings.values():
                    if r == rings:
                        continue
                    ring.adapter.connect_ring(r)
                continue
            ring.adapter.connected_rings = {
                k: network.rings[k].adapter
                for k in network.connections[ring_id]
            }

        return network

    #except Exception as e:
    #    raise ValueError(f"Failed to deserialize network: {str(e)}")


class NetworkDisplayer:
    def __init__(self, network: NetworkManager):
        self.network = network
        self.ring_formatter = RingFormatter()

    def display_state(self, mode: str = 'explain') -> str:
        network_data = self._gather_network_data()

        if mode == 'graph':
            return self._display_network_graph(network_data)
        elif mode == 'metrics':
            return self._display_metrics(network_data)
        else:
            return self._display_rings(network_data, mode)

    def _gather_network_data(self) -> Dict:
        return {
            'rings': [{
                'ring_id': ring_id,
                'concepts': [
                    {
                        'name': concept.name,
                        'stage': concept.stage,
                        'metadata': concept.metadata,
                        'relations': concept.relations
                    }
                    for concept in ring.concept_graph.concepts.values()
                ],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'connections': len(self.network.connections[ring_id])
                }
            } for ring_id, ring in self.network.rings.items()],
            'connections': dict(self.network.connections),
            'metrics': self.network.get_metrics(),
            'name': self.network.name
        }

    def _display_network_graph(self, data: Dict) -> str:
        G = nx.Graph()

        # Add ring nodes
        for ring in data['rings']:
            G.add_node(ring['ring_id'],
                       type='ring',
                       concepts=len(ring['concepts']))

        # Add connections
        for source, targets in data['connections'].items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)

        # Generate ASCII representation
        output = ["Network Topology:", ""]

        for ring_id in G.nodes():
            connections = sorted(G.edges(ring_id),
                                 key=lambda x: G.edges[x]['weight'],
                                 reverse=True)

            output.append(f"{ring_id} ({G.nodes[ring_id]['concepts']} concepts)")
            for source, target in connections:
                other = target if source == ring_id else source
                weight = G.edges[source, target]['weight']
                output.append(f"├── {other} (weight: {weight:.2f})")
            output.append("")

        return "\n".join(output)

    def _display_metrics(self, data: Dict) -> str:
        metrics = data['metrics']
        return (
            f"Network Metrics\n"
            f"---------------\n"
            f"New Ideas: {metrics['new_ideas']}\n"
            f"Active Rings: {len([r for r in metrics['interactions'].values() if r > 0])}\n"
            f"Total Concepts: {sum(len(r['concepts']) for r in data['rings'])}\n"
            f"Last Active: {metrics['last_activated']}\n"
            f"State: {metrics['network_state']}\n\n"
            f"Ring Activity:\n" +
            "\n".join(f"- {ring}: {count} interactions"
                      for ring, count in metrics['interactions'].items())
        )

    def _display_rings(self, data: Dict, mode: str) -> str:
        outputs = []
        for ring_data in data['rings']:
            formatted = self.ring_formatter.format_ring_data(ring_data, mode)
            outputs.append(formatted)
            outputs.append("\n" + "-" * 40 + "\n")
        return "\n".join(outputs)

    def export_state(self, mode: str = 'technical') -> Dict:
        data = self._gather_network_data()
        return {
            'network_state': self._display_network_graph(data),
            'metrics': self._display_metrics(data),
            'rings': [
                self.ring_formatter.format_ring_data(ring, mode)
                for ring in data['rings']
            ]
        }


if __name__ == "__main__":
    # preset_rings = [IntelligenceRing("preset-1"), IntelligenceRing("preset-2")]
    network = NetworkManager(num_empty_rings=3, preset_rings=None, name="test")

    visualizer = NetworkVisualizer(network)

    # Process input
    print(network.metrics)
    # Get references
    references = network.get_references("what is the NetworkManager")
    print(references)
    displayer = NetworkDisplayer(network)

    # Display full network state
    print(displayer.display_state('explain'))

    # Show network topology
    print(displayer.display_state('graph'))

    # Show metrics
    print(displayer.display_state('metrics'))

    # Export detailed state
    state = displayer.export_state('technical')

    process_with_visualization(network, visualizer, """Weiter zum Inhalt

Weiter zur Fußzeile
Bewölkt
‎-1‎
‎°C‎



Entdecken
Folgen
Nachrichten
Unterhaltung
Lifestyle
Autos

Personalisieren
headphone stoped
share
more

SZ.de
147.6K Follower
Bundestagswahl 2025: Musk bezeichnet Steinmeier als „antidemokratischen Tyrannen“
Artikel von Alle Entwicklungen im Liveblog • 3 Std. • 1 Minuten Lesezeit


Elon Musk mischt sich seit Wochen in den deutschen Wahlkampf ein.
Elon Musk mischt sich seit Wochen in den deutschen Wahlkampf ein.
© David Swanson/REUTERS
Der Multimilliardär verbreitet das Video einer rechten Influencerin, die dem Bundespräsidenten fälschlicherweise unterstellt, dieser erwäge, die Bundestagswahl zu annullieren. Die Bundesregierung wertet Musks jüngste Äußerungen als Versuch, sich in die Wahl einzumischen.

Musk bezeichnet Steinmeier als „antidemokratischen Tyrannen“
Für unseren Liveblog verwenden wir neben eigenen Recherchen Material der Nachrichtenagenturen dpa, Reuters, epd, KNA und Bloomberg.

Verwandtes Video: SPD zu Abstrichen bei Bürgergeld bereit (ProSieben)

SZ.de
Besuchen Sie SZ.de
Reaktionen auf Vergewaltigungs-Urteil: „Die Schuld muss endlich die Seiten wechseln“
Krieg in der Ukraine: Heusgen fordert Debatte über nukleare Abschreckung in Europa
Mehr für Sie
AfD-Wahlaufruf: Zeitung löst mit Pro-AfD-Text von Elon Musk Empörung aus
Handelsblatt
Handelsblatt
AfD-Wahlaufruf: Zeitung löst mit Pro-AfD-Text von Elon Musk Empörung aus

more
Jimmy Carter starb im Alter von 100 Jahren: Diese zwei Dinge halfen ihm, so lange zu leben, sagt sein Enkel
Business Insider Deutschland
Business Insider Deutschland
Jimmy Carter starb im Alter von 100 Jahren: Diese zwei Dinge halfen ihm, so lange zu leben, sagt sein Enkel

more
Friedrich Merz: Umfrage-Schock für die Union - CDU-Chef nicht beliebtester Kanzlerkandidat
news.de
news.de
Friedrich Merz: Umfrage-Schock für die Union - CDU-Chef nicht beliebtester Kanzlerkandidat

more
Transit: Was hinter dem Gas-Streit zwischen der Slowakei und Kiew steckt
Handelsblatt
Handelsblatt
Transit: Was hinter dem Gas-Streit zwischen der Slowakei und Kiew steckt

more
„Tatort“-Star muss Masturbationsszene drehen – „Sehr intim“
Der Westen
Der Westen
„Tatort“-Star muss Masturbationsszene drehen – „Sehr intim“

more
Trump-Vertrauter Musk macht deutsches Staatsoberhaupt verächtlich
Frankfurter Allgemeine Zeitung
Frankfurter Allgemeine Zeitung
Trump-Vertrauter Musk macht deutsches Staatsoberhaupt verächtlich

more
Habeck-Kennerin über Kanzlerkandidaten - „Die einzige Konkurrentin, vor der Habeck Angst haben muss, ist Alice Weidel“
FOCUS online
FOCUS online
Habeck-Kennerin über Kanzlerkandidaten - „Die einzige Konkurrentin, vor der Habeck Angst haben muss, ist Alice Weidel“

more
Trotz Krebserkrankung – Asma al-Assad darf nicht nach Großbritannien reisen
WELT
WELT
·
2 Std.
Trotz Krebserkrankung – Asma al-Assad darf nicht nach Großbritannien reisen


Erstaunlich einfach: So zahlt die Pflegekasse Ihren nagelneuen Treppenlift
Checkfox | Treppenlifte
Erstaunlich einfach: So zahlt die Pflegekasse Ihren nagelneuen Treppenlift
Anzeige
more
: Gericht bestätigt Trump-Verurteilung wegen sexuellen Missbrauchs
Handelsblatt
Handelsblatt
·
15 Std.
: Gericht bestätigt Trump-Verurteilung wegen sexuellen Missbrauchs


„Leider haben wir von Russland in den ersten drei Tagen nichts als idiotische Versionen gehört“
WELT
WELT
·
1T
„Leider haben wir von Russland in den ersten drei Tagen nichts als idiotische Versionen gehört“


Stefan Mross blamiert sich bei Schag den Star: Volksmusiker kann keine Trompete spielen – dann kritisiert er Moderator
moviepilot
moviepilot
·
23 Std.
Stefan Mross blamiert sich bei Schag den Star: Volksmusiker kann keine Trompete spielen – dann kritisiert er Moderator


Bundestagswahl: Söder garantiert bei Wahlsieg Regierungsbildung ohne Beteiligung der Grünen
Handelsblatt
Handelsblatt
·
1T
Bundestagswahl: Söder garantiert bei Wahlsieg Regierungsbildung ohne Beteiligung der Grünen


Peter Maffay: Fans schockiert über Wagenknecht-Talk – „Völlig durch!“
Schlager.de
Schlager.de
·
1T
Peter Maffay: Fans schockiert über Wagenknecht-Talk – „Völlig durch!“


headphone stoped
share
more

DER SPIEGEL
409.8K Follower
Joe Biden glaubt offenbar, er hätte Trump geschlagen
2Tage • 2 Minuten Lesezeit

Joe Biden bereut sein Ausscheiden aus dem US-Wahlkampf: Weil er glaubt, dass er gegen Trump gewonnen hätte. Das berichtet die »Washington Post«. Und noch etwas bedauere der Noch-Präsident.


Joe Biden glaubt offenbar, er hätte Trump geschlagen
Joe Biden glaubt offenbar, er hätte Trump geschlagen
© Brendan Smialowski / AFP
Joe Biden bedauert, dass er sich aus dem diesjährigen Präsidentschaftsrennen zurückgezogen hat und glaubt, dass er Donald Trump in der Wahl letzten Monat besiegt hätte – das berichtet die »Washington Post«, die sich auf Quellen aus dem Weißen Haus bezieht.

Demnach habe der US-Präsident auch gesagt, dass er mit der Entscheidung für Merrick Garland als Generalstaatsanwalt einen Fehler begangen habe. Garland hatte die Aufarbeitung von Donald Trumps Rolle im Aufstand vom 6. Januar 2021 nur langsam vorangetrieben. Gleichzeitig hatte sein Justizministerium Bidens Sohn Hunter aggressiv verfolgt.

Hallo günstiger Strom
Solaranlagen Magazin
Hallo günstiger Strom
Anzeige
DER SPIEGEL fasst die wichtigsten News des Tages für Sie zusammen: Was heute wirklich wichtig war - und was es bedeutet. Ihr tägliches Newsletter-Update um 18 Uhr. Jetzt kostenfrei abonnieren.

Biden hat sich bisher kaum öffentlich zu seiner Entscheidung geäußert und vor allem keine Kritik an der Kampagne von Kamala Harris geübt. Laut der »Washington Post« haben er und seine Berater in den vergangenen Tagen aber oft gesagt, dass es ein Fehler gewesen wäre, seine Kandidatur im Juli zurückzuziehen.

Das geschah auf massiven Druck aus seiner eigenen Partei, nachdem er im Monat zuvor in einer TV-Debatte gegen Donald Trump eine schlechte Figur gemacht hatte. Die Demokraten beriefen sich auf Umfragen, die Biden eine fast sichere Niederlage vorraussagten.

Statt seiner setzte sich Vizepräsidentin Kamala Harris an die Spitze der Demokraten-Kampagne. Das führte zunächst zu einer Welle der Begeisterung und verbesserten Umfragewerte, endete aber letztlich in einer entscheidenden Niederlage bei den Wahlmännern und der Volksabstimmung.

Bis heute steht die Frage im Raum, ob es nicht doch die bessere Strategie gewesen wäre, an Biden festzuhalten. Er selbst scheint das immer noch zu glauben.

Verwandtes Video: Trump kritisiert Bidens Duldung von US-Waffen in der Ukraine (glomex)


Leadgen logo
DER SPIEGEL
Entdecken Sie alle exklusiven Inhalte auf SPIEGEL.de mit SPIEGEL+.
4 Wochen SPIEGEL+ für € 0,-, danach € 4,99 pro Woche
Mehr erfahren
DER SPIEGEL
DER SPIEGEL
DER SPIEGEL
Mehr für Sie
Bundestagswahl: „Schande über ihn“ – Musk attackiert Steinmeier
Handelsblatt
Handelsblatt
Bundestagswahl: „Schande über ihn“ – Musk attackiert Steinmeier

more
Reaktionen auf Musk: Wer kriegt hier den Hintern nicht hoch?
Frankfurter Allgemeine Zeitung
Frankfurter Allgemeine Zeitung
Reaktionen auf Musk: Wer kriegt hier den Hintern nicht hoch?

more
Burnout, TV-Eklat und Geert Wilders als heimlicher Minister: Die niederländische Regierung wankt von Krise zu Krise
Neue Zürcher Zeitung Deutschland
Neue Zürcher Zeitung Deutschland
Burnout, TV-Eklat und Geert Wilders als heimlicher Minister: Die niederländische Regierung wankt von Krise zu Krise

more
Kaja Kallas: Neue europäische Außenbeauftragte verärgert EU-Staaten
Handelsblatt
Handelsblatt
Kaja Kallas: Neue europäische Außenbeauftragte verärgert EU-Staaten

more
USA: Trump für Wiederwahl Johnsons auf Chefposten in US-Parlament
Handelsblatt
Handelsblatt
USA: Trump für Wiederwahl Johnsons auf Chefposten in US-Parlament

more
Günther Jauch: Welche politische Ausrichtung hat der „Wer wird Millionär“-Star?
tvmovie.de
tvmovie.de
Günther Jauch: Welche politische Ausrichtung hat der „Wer wird Millionär“-Star?

more
Alijew führt Putin vor – der aserbaidschanische Machthaber kann sich das leisten
Neue Zürcher Zeitung Deutschland
Neue Zürcher Zeitung Deutschland
Alijew führt Putin vor – der aserbaidschanische Machthaber kann sich das leisten

more
Donald Trump: "Was ist hier los?" Designierter US-Präsident rastet aus und schlägt um sich
news.de
news.de
·
2T
Donald Trump: "Was ist hier los?" Designierter US-Präsident rastet aus und schlägt um sich


Erstaunlich einfach: So zahlt die Pflegekasse Ihren nagelneuen Treppenlift
Checkfox | Treppenlifte
Erstaunlich einfach: So zahlt die Pflegekasse Ihren nagelneuen Treppenlift
Anzeige
more
Wladimir Putin: Zitter-Anfälle und Parkinson-Symptome: So steht es wirklich um seine Gesundheit
news.de
news.de
·
7 Std.
Wladimir Putin: Zitter-Anfälle und Parkinson-Symptome: So steht es wirklich um seine Gesundheit


Brigitte Macron: Der Tag, an dem ihr erster Ehemann von ihrer Beziehung mit Emmanuel Macron erfuhr
Oh!mymag
Oh!mymag
·
19 Std.
Brigitte Macron: Der Tag, an dem ihr erster Ehemann von ihrer Beziehung mit Emmanuel Macron erfuhr


US-Regierung: Trump scheitert mit Berufung in Prozess wegen sexueller Nötigung
SZ.de
SZ.de
·
15 Std.
US-Regierung: Trump scheitert mit Berufung in Prozess wegen sexueller Nötigung


Thomas Gottschalk gibt Abschied bekannt: „Ich bin es leid“
Schlager.de
Schlager.de
·
9 Std.
Thomas Gottschalk gibt Abschied bekannt: „Ich bin es leid“


Experte sicher: Wenn Putin stirbt, droht Chaos
News in Five
News in Five
·
1T
Experte sicher: Wenn Putin stirbt, droht Chaos


headphone stoped
share
more

Der Westen
399.6K Follower
Formel 1: Alles vorbei für Max Verstappen? Düstere Prognose macht die Runde
Artikel von Marco Hintermüller • 17 Std. • 2 Minuten Lesezeit
Ausgewählte Turniere
Finale · 10. Nov.

NASCAR Cup Series Championship
Finale · 3. Nov.

XFINITY 500
Finale · 27. Okt.

Straight Talk Wireless 400

Vier Titel, zahlreiche Rekorde und noch mehr Rennsiege – seit 2021 dominiert Max Verstappen die Formel 1 nach Belieben. Auch in der abgelaufenen Saison konnte ihm letztlich niemand das Wasser reichen. Allerdings: Das hing besonders mit seinem Raketenstart ins Jahr zusammen.

Zum Ende der Saison schwächelte Red Bull gewaltig, die Konkurrenz holte in der Formel 1 immer weiter auf. Insbesondere McLaren und Ferrari machten ordentlich Druck. Das veranlasst einen ehemaligen Fahrer zu einer düsteren Prognose.

Verkehrswert-Rechner 2025
Hausverkauf
Verkehrswert-Rechner 2025
Anzeige
Formel 1: Wie gut ist Red Bull noch?
Wie gut ist Red Bull noch? Diese Frage wird sich das Paddock stellen, wenn es im März wieder raus auf die Rennstrecke geht. Sieben der ersten elf Rennen hatte Verstappen für sich entscheiden können, war damit in der WM-Wertung früh und schnell davon gezogen. Das war auch bitter nötig. Denn in der zweiten Saisonhälfte konnte er nur noch zwei weitere Grands Prix gewinnen.

Expand article logo  Weiterlesen


Kann Max Verstappen in der Formel 1 weiterhin gewinnen?
Kann Max Verstappen in der Formel 1 weiterhin gewinnen?
© IMAGO/Jay Hirano







Der Westen
Besuchen Sie Der Westen
Strompreis-Explosion in NRW! So viel musst du 2025 draufzahlen
Rewe reagiert nach Payback-Aus – die Meinung der Kunden ist eindeutig
Silvester-Schlagerbooom 2025: Noch vor dem Start herrscht bittere Gewissheit
Mehr für Sie
Zweifel an "krankem" Antonelli: "Muss wirklich schlimm sein, wenn er nicht fahren will
GPblog DE
GPblog DE
Zweifel an "krankem" Antonelli: "Muss wirklich schlimm sein, wenn er nicht fahren will

more
Analyse von Ulrich Reitz - Ein Musk-Satz von Robert Habeck ist furchteinflößend und entlarvend
FOCUS online
FOCUS online
Analyse von Ulrich Reitz - Ein Musk-Satz von Robert Habeck ist furchteinflößend und entlarvend

more
Meteorologe verblüfft: „Sowas noch nie gesehen“ – Wettermodell kippt binnen 48 Stunden komplett
FR
FR
Meteorologe verblüfft: „Sowas noch nie gesehen“ – Wettermodell kippt binnen 48 Stunden komplett

more
Die größten F1-Reinfälle 2025: Scheitert Hamilton bei Ferrari?
playIndicator
glomex
glomex
Die größten F1-Reinfälle 2025: Scheitert Hamilton bei Ferrari?

more
Hochzeitsgesellschaft verunglückt – mindestens 71 Tote
Berliner Morgenpost
Berliner Morgenpost
Hochzeitsgesellschaft verunglückt – mindestens 71 Tote

more
Neuville legt sich fest: Startnummer 1 kehrt 2025 in die WRC zurück
motorsport.com
motorsport.com
Neuville legt sich fest: Startnummer 1 kehrt 2025 in die WRC zurück

more
Ich hätte Magnussen gerne neben Verstappen bei Red Bull gesehen.
GPblog DE
GPblog DE
Ich hätte Magnussen gerne neben Verstappen bei Red Bull gesehen.

more
Berufungsgericht bestätigt Urteil gegen Trump
WELT
WELT
·
20 Std.
Berufungsgericht bestätigt Urteil gegen Trump


Was kostet ein Treppenlift?
Aroundhome | Treppenlift
Was kostet ein Treppenlift?
Anzeige
more
Deutscher Ex-Formel-1-Fahrer packt aus: Michael Schumacher hat ihm große Chance verbaut
TAG24
TAG24
·
1T
Deutscher Ex-Formel-1-Fahrer packt aus: Michael Schumacher hat ihm große Chance verbaut


Sophia Flörsch wagt neuen Schritt
sport1.de
sport1.de
·
14 Std.
Sophia Flörsch wagt neuen Schritt


Sabotage vermutet! Hunderttausende ohne TV und Handy
TAG24
TAG24
·
18 Std.
Sabotage vermutet! Hunderttausende ohne TV und Handy


Formel 1: Verstappen muss höllisch aufpassen – es droht eine Sperre
Thüringen 24
Thüringen 24
·
1T
Formel 1: Verstappen muss höllisch aufpassen – es droht eine Sperre


Niedergeschlagener Ski-Star gesteht: „Skifahren ist mir derzeit egal“
Merkur
Merkur
·
20 Std.
Niedergeschlagener Ski-Star gesteht: „Skifahren ist mir derzeit egal“


headphone stoped
share
more

Der Westen
399.6K Follower
Video mit Annalena Baerbock aufgetaucht: So hast du sie noch nie gesehen
Artikel von Marcel Görmann • 2 Std. • 2 Minuten Lesezeit

Annalena Baerbock war mal kurz nicht Außenministerin, sondern eine ganz normale Frau, wie auf einer ü30-Party, irgendwo in Deutschland an einem Samstagabend…

+++ Interessant: „Baerbock lebt in eigener Welt“ – TV-Moderator schießt gegen Grüne live auf Sendung +++


Parteitags-Party mit Baerbock
Parteitags-Party mit Baerbock
© IMAGO/Jörg Halisch
Zu einem Parteitag gehört auch eine ordentliche Feier – so auch am Samstagabend bei den Grünen in Wiesbaden. Friedrich Merz hat vorgelegt – Annalena Baerbock zieht nach! Jetzt ist ein Video aufgetaucht, auf dem man die Außenministerin ganz ausgelassen auf der Tanzfläche sieht.

Omid Nouripour legt auf – Baerbock bewegt die Hüften
Aufgelegt hatte der bisherige Parteichef Omid Nouripour, wie schon beim Parteitag 2022 in Bonn. Damals gab es noch einen kleinen Skandal um fehlende Corona-Masken bei vielen Delegierten auf der Party.

Diese Zeiten sind längst vorbei. Und trotz mäßiger Umfragewerte für die Grünen (aktuell 10-12 Prozent deutschlandweit), war Annalena Baerbock und den Delegierten zum Tanzen zumute. Ein Clip zeigt sie fröhlich vor dem DJ Pult.

Wie sollte es anders sein, lief natürlich ein Song von Taylor Swift. „Shake it off“. In dem Lied geht es um die Hater, die man einfach abschütteln soll. Wie passend, angesichts des Grünen-Bashings seit Jahren. Auf Instagram ist ein Ausschnitt veröffentlicht worden (einmal nach rechts swipen und anklicken, damit das Video startet).


View this post on Instagram

Außenministerin stresst den DJ
Die Außenministerin konnte mit Parteifreundinnen wie Katharina Dröge und Britta Haßelmann um sich herum ein bisschen Anpassung rauslassen. Nebenbei aber soll sie sich auch immer wieder in die Playlist von Nouripour eingemischt haben, verrät Fraktionschefin Dröge auf Instagram.

Mehr Themen für dich:

Maite Kelly mit neuer TV-Herausforderung – so haben Fans sie noch nie gesehen
Annalena Baerbock privat: Ex-Mann, Töchter und Hobby – das ist bisher kaum bekannt
Robert Habeck nach „Schwachkopf“-Affäre: So rigoros wehrt er sich mit Anzeigen
Mittlerweile aber hat Baerbock längst wieder die harte Realität eingeholt. Am Montag drohte sie China am Rande eines EU-Treffens in Brüssel mit Konsequenzen angesichts der mutmaßlichen Drohenproduktion für Russlands Krieg in der Ukraine.

Verwandtes Video: Falschmeldungen über Baerbock und Habeck (glomex)

Der Westen
Besuchen Sie Der Westen
Strompreis-Explosion in NRW! So viel musst du 2025 draufzahlen
Rewe reagiert nach Payback-Aus – die Meinung der Kunden ist eindeutig
Silvester-Schlagerbooom 2025: Noch vor dem Start herrscht bittere Gewissheit
Mehr für Sie

Feedback

Profilbild
""")

    # Display full network state
    print(displayer.display_state('explain'))

    # Show network topology
    print(displayer.display_state('graph'))

    # Show metrics
    print(displayer.display_state('metrics'))

    # Export detailed state
    state = displayer.export_state('technical')

    # Control network
    #network.freeze()
    #network.resume()
    while t := input("Q:"):
        print(network.get_concept_from_code(network.get_references(t)[0][0]).metadata)
    network.step_back()
    while t := input("bQ:"):
        print(network.get_references(t))
