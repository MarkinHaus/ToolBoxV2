import os
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
import threading
import numpy as np
import multiprocessing as mp
from queue import PriorityQueue
import uuid

import torch
from transformers import AutoTokenizer, AutoModel

from toolboxv2 import get_logger
from toolboxv2.utils.system import FileCache


logger = get_logger()


class InputProcessor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.vector_size = 512
        cache_dir = os.getenv('APPDATA') if os.name == 'nt' else os.getenv('XDG_CONFIG_HOME') or os.path.expanduser(
            '~/.config') if os.name == 'posix' else "."
        cache = FileCache(folder=cache_dir + f'\\ToolBoxV2\\cache\\InputProcessor\\',
                          filename=cache_dir + f'\\ToolBoxV2\\cache\\InputProcessor\\cache.db')

        def helper(t, lang):
            result = cache.get(str((t, lang)))
            if result is not None:
                return result

            result = self.get_embedding(t)
            cache.set(str((t, lang)), result)
            return result

        self.get_concept_vector = helper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    @staticmethod
    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _cached_embedding(self, text: str) -> np.ndarray:
        # Perform text processing and return the embedding vector.
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.vector_size,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy()[0]

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        try:
            return self._cached_embedding(text)
        except Exception as e:
            print(text, type(text))
            print(f"Error generating embedding: {str(e)}")
            return None

    def batch_get_embeddings(self, texts: list[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Get embeddings for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            numpy.ndarray or None: Matrix of embedding vectors if successful
        """
        try:
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.vector_size,
                    return_tensors='pt'
                ).to(self.device)

                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # Perform pooling and normalize
                batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

            return np.vstack(embeddings)

        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return None

    def process_text(self, text: str) -> np.ndarray:
        # Simplified vector creation for example
        ptext = self.get_embedding(text)
        if ptext is None:
            ptext = self.batch_get_embeddings([text], 1)
        if ptext is None:
            return np.random.randn(self.vector_size)
        return ptext


@dataclass
class Concept:
    id: str
    name: str
    ttl: int
    created_at: datetime
    vector: np.ndarray
    contradictions: Set[str]
    similar_concepts: Set[str]
    relations: Dict[str, float]
    stage: int
    metadata: Dict

    def is_expired(self) -> bool:
        if self.ttl == -1:
            return False
        return ((datetime.now() - self.created_at).total_seconds() / (60*60)) > self.ttl


# Module 2: Retrieval System
class RetrievalSystem:
    def __init__(self, concept_graph: 'ConceptGraph'):
        self.concept_graph = concept_graph
        self.similarity_threshold = 0.8

    def find_similar(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        similarities = []
        for concept_id, concept in self.concept_graph.concepts.items():
            sim = np.dot(vector, concept.vector)
            similarities.append((concept_id, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def search_by_metadata(self, metadata_query: Dict) -> List[str]:
        results = []
        for concept_id, concept in self.concept_graph.concepts.items():
            if all(concept.metadata.get(k) == v for k, v in metadata_query.items()):
                results.append(concept_id)
        return results


# Module 3: Internal Processing
class ConceptGraph:
    def __init__(self, max_concepts: int = 1000, max_relations_per_concept: int = 50):
        self.concepts: Dict[str, Concept] = {}
        self.max_concepts = max_concepts
        self.max_relations = max_relations_per_concept
        self.lock = threading.Lock()

    def _cleanup_oldest(self):
        ks = []
        for k, concept in self.concepts.items():
            if concept.is_expired():
                ks.append(k)
        for k in ks:
            del self.concepts[k]

    def add_concept(self, name: str, vector: np.ndarray, ttl: int = 3600,
                    metadata: Dict = None) -> str:
        with self.lock:
            if len(self.concepts) >= self.max_concepts:
                self._cleanup_oldest()

            concept_id = str(uuid.uuid4())
            concept = Concept(
                id=concept_id,
                name=name,
                ttl=ttl,
                created_at=datetime.now(),
                vector=vector,
                contradictions=set(),
                similar_concepts=set(),
                relations={},
                stage=1,
                metadata=metadata or {}
            )
            self.concepts[concept_id] = concept
            self._update_similarities(concept_id)
            return concept_id

    def _update_similarities(self, concept_id: str):
        concept = self.concepts[concept_id]
        for other_id, other in self.concepts.items():
            if other_id != concept_id:
                similarity = np.dot(concept.vector, other.vector)
                if similarity > 0.8:
                    concept.similar_concepts.add(other_id)
                    other.similar_concepts.add(concept_id)
                elif similarity < -0.2:
                    concept.contradictions.add(other_id)
                    other.contradictions.add(concept_id)


# Module 4: Adapter System
class RingAdapter:
    def __init__(self, ring_id: str):
        self.ring_id = ring_id
        self.connected_rings: Dict[str, 'RingAdapter'] = {}
        self.message_queue = PriorityQueue()

    def connect_ring(self, other_ring: 'RingAdapter'):
        self.connected_rings[other_ring.ring_id] = other_ring

    def broadcast_concept(self, concept: Concept):
        for ring_id, ring in self.connected_rings.items():
            ring.receive_concept(concept)

    def receive_concept(self, concept: Concept):
        self.message_queue.put((1, concept))


# Main Intelligence Ring Class
class IntelligenceRing:
    def __init__(self, ring_id: str, num_threads: int = mp.cpu_count()):
        self.ring_id = ring_id
        self.concept_graph = ConceptGraph()
        self.input_processor = InputProcessor()
        self.retrieval_system = RetrievalSystem(self.concept_graph)
        self.adapter = RingAdapter(ring_id)
        self.processing_queue = PriorityQueue()
        self.num_threads = num_threads
        self.workers = []
        self._initialize_workers()

    def _initialize_workers(self):
        for _ in range(self.num_threads):
            worker = threading.Thread(target=self._process_queue)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _process_queue(self):
        while True:
            try:
                priority, concept_id = self.processing_queue.get()
                if concept_id in self.concept_graph.concepts:
                    self._advance_concept_stage(concept_id)
                self.processing_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing concept: {e}")

    def get_concept_by_id(self, concept_id) -> Concept:
        return self.concept_graph.concepts.get(concept_id)


# Example Usage
def main():
    # Create an intelligence ring
    ring = IntelligenceRing("ring-1")

    # Create a key concept with 10 relations
    key_concept_vector = np.random.randn(256)
    key_concept_id = ring.concept_graph.add_concept(
        name="AI Ethics",
        vector=key_concept_vector,
        ttl=-1,  # Immortal concept
        metadata={"domain": "technology", "importance": "high"}
    )

    # Create 10 related concepts
    related_concepts = [
        ("Data Privacy", {"domain": "security"}),
        ("Machine Learning", {"domain": "technology"}),
        ("Human Rights", {"domain": "society"}),
        ("Algorithm Bias", {"domain": "technology"}),
        ("Transparency", {"domain": "governance"}),
        ("Accountability", {"domain": "governance"}),
        ("Social Impact", {"domain": "society"}),
        ("Innovation", {"domain": "technology"}),
        ("Regulation", {"domain": "legal"}),
        ("User Trust", {"domain": "business"})
    ]

    # Add related concepts and create relationships
    for name, metadata in related_concepts:
        vector = np.random.randn(256)
        concept_id = ring.concept_graph.add_concept(
            name=name,
            vector=vector,
            ttl=3600,
            metadata=metadata
        )

        # Create relationship with key concept
        similarity = np.dot(vector, key_concept_vector)
        ring.concept_graph.concepts[key_concept_id].relations[concept_id] = similarity

    # Example: Retrieve similar concepts
    key_concept_vector = ring.concept_graph.concepts[key_concept_id].vector
    similar_concepts = ring.retrieval_system.find_similar(key_concept_vector)

    print("Similar concepts to 'AI Ethics':")
    for concept_id, similarity in similar_concepts:
        concept = ring.concept_graph.concepts[concept_id]
        print(f"- {concept.name}: {similarity:.3f}")

    # Example: Search by metadata
    tech_concepts = ring.retrieval_system.search_by_metadata({"domain": "technology"})
    print("\nTechnology-related concepts:")
    for concept_id in tech_concepts:
        print(f"- {ring.concept_graph.concepts[concept_id].name}")


if __name__ == "__main__":
    main()
