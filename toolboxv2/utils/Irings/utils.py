import base64
import hashlib
import json
import zlib
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from .zero import IntelligenceRing, Concept
from typing import List, Dict, Set, Any

from dataclasses import dataclass


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
            'concepts': [self._concept_to_dict(c) for c in ring.concept_graph.concepts.values()],
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

            for concept_data in data['concepts']:
                concept = self._dict_to_concept(concept_data)
                ring.concept_graph.concepts[concept.id] = concept

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
class SplitConfig:
    min_chunk_size: int = 50
    max_chunk_size: int = 500
    similarity_threshold: float = 0.7
    min_importance: float = 0.3


@dataclass
class SubConcept:
    vector: np.ndarray
    importance: float
    relations: Set[str]
    metadata: Dict


class ConceptSplitter(ABC):
    @abstractmethod
    def split(self, text: str, vector: np.ndarray) -> List[SubConcept]:
        pass

