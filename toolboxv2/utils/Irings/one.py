import os
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from datetime import datetime
import threading
import numpy as np
import multiprocessing as mp
from queue import PriorityQueue
import uuid
import json
from pathlib import Path
import io
from PIL import Image

import base64

import torch
from lancedb.embeddings.registry import register

from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import lancedb
from lancedb.pydantic import LanceModel, Vector

from lancedb.embeddings import EmbeddingFunctionRegistry

from toolboxv2 import get_logger, Singleton
from toolboxv2.utils.Irings.zero import RingAdapter
from toolboxv2.utils.Irings.utils import TransformerSplitter
from toolboxv2.utils.system import FileCache

logger = get_logger()


class InputData:
    def __init__(
        self,
        content: Union[str, bytes, np.ndarray],
        modality: str,
        metadata: Optional[Dict] = None
    ):
        self.content = content
        self.modality = modality  # 'text', 'image', or 'audio'
        self.metadata = metadata or {}


@register("intelligence-ring-embeddings")
class IntelligenceRingEmbeddings(lancedb.embeddings.EmbeddingFunction):
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_name: str = "openai/clip-vit-base-patch32"
    wav2vec_name: str = "facebook/wav2vec2-base-960h"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    vector_size: int = 2036
    tokenizer: Optional[Any] = None
    text_model: Optional[Any] = None

    clip_processor: Optional[Any] = None
    clip_model: Optional[Any] = None

    audio_processor: Optional[Any] = None
    audio_model: Optional[Any] = None

    text_projection: Optional[Any] = None
    image_projection: Optional[Any] = None
    audio_projection: Optional[Any] = None

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._ndims = self.vector_size

        # Text embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.text_model = AutoModel.from_pretrained(self.name).to(self.device)

        # Image embedding model (CLIP)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_name).to(self.device)

        # Audio embedding model (Wav2Vec2)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec_name)
        self.audio_model = Wav2Vec2Model.from_pretrained(self.wav2vec_name).to(self.device)

        # Projection layers to align dimensions
        self.text_projection = torch.nn.Linear(
            self.text_model.config.hidden_size,
            self.vector_size
        ).to(self.device)
        self.image_projection = torch.nn.Linear(
            self.clip_model.config.vision_config.hidden_size,
            self.vector_size
        ).to(self.device)
        self.audio_projection = torch.nn.Linear(
            self.audio_model.config.hidden_size,
            self.vector_size
        ).to(self.device)

    def _process_text(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.vector_size,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**encoded_input)
            embeddings = self._mean_pooling(outputs, encoded_input['attention_mask'])
            projected = self.text_projection(embeddings)
            return torch.nn.functional.normalize(projected, p=2, dim=1)

    def _process_image(self, image_data: Union[bytes, str]) -> torch.Tensor:
        # Handle different image input types
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                # Handle base64 encoded images
                image_data = base64.b64decode(image_data.split(',')[1])
            else:
                # Handle file paths
                with open(image_data, 'rb') as f:
                    image_data = f.read()

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Process image with CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            projected = self.image_projection(outputs)
            return torch.nn.functional.normalize(projected, p=2, dim=1)

    def _process_audio(self, audio_data: Union[bytes, str, np.ndarray]) -> torch.Tensor:
        try:
            import torchaudio
        except ImportError:
            raise ValueError("Couldn't load audio install torchaudio'")
        # Handle different audio input types
        if isinstance(audio_data, str):
            if audio_data.startswith('data:audio'):
                # Handle base64 encoded audio
                audio_data = base64.b64decode(audio_data.split(',')[1])
                waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
            else:
                # Handle file paths
                waveform, sample_rate = torchaudio.load(audio_data)
        elif isinstance(audio_data, bytes):
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
        else:
            # Assume numpy array with sample rate in metadata
            waveform = torch.from_numpy(audio_data)
            sample_rate = 16000  # Default sample rate

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Process audio with Wav2Vec2
        inputs = self.audio_processor(waveform, sampling_rate=16000, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            # Mean pooling over time dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)
            projected = self.audio_projection(embeddings)
            return torch.nn.functional.normalize(projected, p=2, dim=1)

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def process_input(self, input_data: InputData) -> np.ndarray:
        if input_data.modality == "text":
            embeddings = self._process_text(input_data.content)
        elif input_data.modality == "image":
            embeddings = self._process_image(input_data.content)
        elif input_data.modality == "audio":
            embeddings = self._process_audio(input_data.content)
        else:
            raise ValueError(f"Unsupported modality: {input_data.modality}")

        return embeddings.cpu().numpy()

    def compute_query_embeddings(self, query: Union[str, bytes, np.ndarray], modality: str = "text") -> List[
        np.ndarray]:
        """Compute embeddings for query input"""
        input_data = InputData(query, modality)
        embedding = self.process_input(input_data)
        return [embedding.squeeze()]

    def compute_source_embeddings(self, sources: List[Union[str, bytes, np.ndarray]], modalities: List[str]) -> List[
        np.ndarray]:
        """Compute embeddings for source inputs"""
        embeddings = []
        for source, modality in zip(sources, modalities):
            input_data = InputData(source, modality)
            embedding = self.process_input(input_data)
            embeddings.append(embedding.squeeze())
        return embeddings

    def ndims(self) -> int:
        return self._ndims


# LanceDB Schema for Concepts
class ConceptSchema(LanceModel):
    vector: Vector(2036)
    id: str
    name: str
    ttl: int
    created_at: datetime
    contradictions: List[str]
    similar_concepts: List[str]
    relations: str
    stage: int
    metadata: str
    modality: str  # New field


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
    modality: str = "text"

    def is_expired(self) -> bool:
        if self.ttl == -1:
            return False
        return ((datetime.now() - self.created_at).total_seconds() / (60 * 60)) > self.ttl

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "vector": self.vector.tolist(),
            "contradictions": list(self.contradictions),
            "similar_concepts": list(self.similar_concepts),
            "relations": self.relations,
            "stage": self.stage,
            "metadata": self.metadata,
            "modality": self.modality
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Concept':
        return cls(
            id=data["id"],
            name=data["name"],
            ttl=data["ttl"],
            created_at=datetime.fromisoformat(data["created_at"]),
            vector=np.array(data["vector"]),
            contradictions=set(data["contradictions"]),
            similar_concepts=set(data["similar_concepts"]),
            relations=data["relations"],
            stage=data["stage"],
            metadata=data["metadata"],
            modality=data["modality"]
        )


class InputProcessor(metaclass=Singleton):
    def __init__(self):
        self.embedding_function = IntelligenceRingEmbeddings()
        self.vector_size = self.embedding_function.vector_size

        cache_dir = os.getenv('APPDATA') if os.name == 'nt' else os.getenv('XDG_CONFIG_HOME') or os.path.expanduser(
            '~/.config') if os.name == 'posix' else "."
        self.cache = FileCache(
            folder=cache_dir + f'\\ToolBoxV2\\cache\\InputProcessor\\',
            filename=cache_dir + f'\\ToolBoxV2\\cache\\InputProcessor\\cache.db'
        )

    def get_embedding(self, content: Union[str, bytes, np.ndarray], modality: str = "text") -> Optional[np.ndarray]:
        # Only cache text embeddings
        if modality == "text" and isinstance(content, str):
            cache_key = str((content, "en"))
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return np.array(cached_result)

        # Split content into 2700-character chunks
        chunks = [content]
        if modality == "text" and isinstance(content, str):
            chunks = [content[i:i + 2700] for i in range(0, len(content), 2700)]

        # Initialize a list to store embeddings for all chunks
        chunk_embeddings = []

        for chunk in chunks:
            # Check cache for each chunk
            cache_key = str((chunk, "en"))
            cached_result = self.cache.get(cache_key)

            if cached_result is not None:
                chunk_embeddings.append(np.array(cached_result))
            else:
                input_data = InputData(chunk, modality)
                try:
                    chunk_embedding = self.embedding_function.process_input(input_data).flatten().tolist()
                except RuntimeError as e:
                    try:
                        input_data = InputData(chunk[:len(chunk)//2], modality)
                        chunk_embedding = self.embedding_function.process_input(input_data).flatten().tolist()
                        if chunk_embedding:
                            chunk_embeddings.append(chunk_embedding)
                    except Exception:
                        continue
                    try:
                        input_data = InputData(chunk[len(chunk)//2:], modality)
                        chunk_embedding = self.embedding_function.process_input(input_data).flatten().tolist()
                        if chunk_embedding:
                            chunk_embeddings.append(chunk_embedding)
                    except Exception:
                        if len(chunk_embeddings) == 0 and chunks.index(chunk) == len(chunks)-1:
                            raise e


                if modality == "text" and isinstance(content, str):
                    # Cache the chunk embedding
                    self.cache.set(cache_key, chunk_embedding)

                chunk_embeddings.append(chunk_embedding)

        # Combine all chunk embeddings into a single vector of the same shape
        combined_embedding = np.mean(chunk_embeddings, axis=0) if chunk_embeddings else None

        return combined_embedding


    def batch_get_embeddings(self, contents: List[Union[str, bytes, np.ndarray]], modalities: List[str]) -> Optional[
        np.ndarray]:
        try:
            results = []
            for content, modality in zip(contents, modalities):
                embedding = self.get_embedding(content, modality)
                if embedding is not None:
                    results.append(embedding)
            return np.vstack(results) if results else None
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return None

    def process_text(self, content: str):
        if not content:
            return np.zeros(self.vector_size)
        emb = self.get_embedding(content, modality="text")
        return emb

    def pcs(self, x,y):
        ex, ey = self.process_text(x), self.process_text(y)
        if ex is not None and ey is not None:
            return self.compute_similarity(ex,ey)
        return -1

    @staticmethod
    def compute_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        return float(np.dot(x1, x2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0



class RetrievalSystem:
    def __init__(self, concept_graph: 'ConceptGraph'):
        self.concept_graph = concept_graph
        self.similarity_threshold = 0.8

    def find_similar(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        # query_vector = vector.tolist()
        results = self.concept_graph.table.search(vector).limit(top_k).to_list()
        return sorted([(result["id"], 1 / result["_distance"] if result["_distance"] > 0 else 1) for result in results],
                      key=lambda x: x[1], reverse=True)


class ConceptGraph:
    def __init__(self, ring_id: str, max_concepts: int = 1000, max_relations_per_concept: int = 50):
        self.ring_id = ring_id
        self.max_concepts = max_concepts
        self.max_relations = max_relations_per_concept
        self.lock = threading.Lock()

        # Initialize LanceDB
        db_path = Path(f"./.data/intelligence_rings/{ring_id}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))

        # Create or load the table
        if f"concepts_{ring_id}" in self.db.table_names():
            self.table = self.db.open_table(f"concepts_{ring_id}")
        else:
            self.table = self.db.create_table(
                f"concepts_{ring_id}",
                schema=ConceptSchema.to_arrow_schema(),
                # mode="create"
            )

        # Load concepts into memory
        self.concepts: Dict[str, Concept] = self._load_concepts()

    def _load_concepts(self) -> Dict[str, Concept]:
        concepts = {}
        for row in self.table.to_pandas().to_dict('records'):
            concept = Concept(
                id=row["id"],
                name=row["name"],
                ttl=row["ttl"],
                created_at=row["created_at"],
                vector=np.array(row["vector"]),
                contradictions=set(row["contradictions"]),
                similar_concepts=set(row["similar_concepts"]),
                relations=json.loads(row["relations"]),
                stage=row["stage"],
                metadata=json.loads(row["metadata"]),
                modality=row["modality"]
            )
            concepts[concept.id] = concept
        return concepts

    def _cleanup_oldest(self):
        with self.lock:
            expired_ids = []
            expired_vec = []
            for concept_id, concept in self.concepts.items():
                if concept.is_expired():
                    expired_ids.append(concept_id)
                    expired_vec.append(concept.vector)

            # Remove from memory
            for concept_id in expired_ids:
                del self.concepts[concept_id]

            # Remove from LanceDB
            if expired_ids:
                try:
                    self.table.delete(f"id IN {tuple(expired_ids)}")
                except Exception:
                    print("Could not delete using ID try v")
                    for vec in expired_vec:
                        self.table.delete(f"vector = {vec}")

    def add_concept(self, name: str, vector: np.ndarray, ttl: int = 2,
                    metadata: Dict = None, modality: str = "text") -> str:
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
                metadata=metadata or {},
                modality=modality
            )

            # Add to memory
            self.concepts[concept_id] = concept
            self.table.add([{
                "vector": vector,
                "id": concept_id,
                "name": concept.name,
                "ttl": concept.ttl,
                "created_at": concept.created_at,
                "contradictions": list(concept.contradictions),
                "similar_concepts": list(concept.similar_concepts),
                "relations": json.dumps(concept.relations),
                "stage": concept.stage,
                "metadata": json.dumps(concept.metadata),
                "modality": concept.modality
            }])
            self._update_similarities(concept_id)
            return concept_id

    def _update_similarities(self, concept_id: str):
        concept = self.concepts[concept_id]
        vector = concept.vector.tolist()

        self.table.delete(f"vector = {vector}")

        # Use LanceDB for similarity search
        similar_concepts = self.table.search(vector).limit(self.max_relations).to_list()
        v = {}  #{"stage": min(max(concept.stage+1, 3), concept.stage)}
        for similar in similar_concepts:
            if similar["id"] != concept_id:
                similarity = 1 / similar["_distance"] if similar[
                                                             "_distance"] > 0 else 1  # Convert distance to similarity
                if similarity > 0.8:
                    concept.similar_concepts.add(similar["id"])
                elif similarity < 0.1:
                    concept.contradictions.add(similar["id"])

                # Update relations
                concept.relations[similar["id"]] = similarity
        # self.table.delete(f"id = {concept_id}")
        self.table.add([{
            "vector": vector,
            "id": concept_id,
            "name": concept.name,
            "ttl": concept.ttl,
            "created_at": concept.created_at,
            "contradictions": list(concept.contradictions),
            "similar_concepts": list(concept.similar_concepts),
            "relations": json.dumps(concept.relations),
            "stage": concept.stage + 1,
            "metadata": json.dumps(concept.metadata),
            "modality": concept.modality
        }])


class IntelligenceRing:
    def __init__(self, ring_id: str, num_threads: int = mp.cpu_count()):
        self.ring_id = ring_id
        self.concept_graph = ConceptGraph(ring_id)
        self.input_processor = InputProcessor()
        self.splitter = TransformerSplitter(self.input_processor)
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

    def _advance_concept_stage(self, concept_id: str):
        concept = self.concept_graph.concepts.get(concept_id)
        if concept:
            concept.stage += 1
            # Update stage in LanceDB
            self.concept_graph.table.update().where(f"id = '{concept_id}'").set({
                "stage": concept.stage
            }).execute()

    def get_concept_by_id(self, concept_id: str) -> Optional[Concept]:
        return self.concept_graph.concepts.get(concept_id)

    def process(self, text, metadata=None, name=None, ttl=None):
        concepts = []
        sub_concepts = self.splitter.split(text, self.input_processor.process_text(text))
        for concept in sub_concepts:
            concepts.append(self.concept_graph.add_concept(
                name=name if name else concept.metadata['text'][:50],
                vector=concept.vector,
                ttl=(int(concept.importance * 10) if concept.importance > 0 else 1) if ttl is None else ttl,
                metadata={**concept.metadata, **metadata} if metadata else concept.metadata
            ))
        return concepts


def main():
    # Create an intelligence ring
    ring = IntelligenceRing("ring-1")

    # Create a key concept
    text = " AI Ethics and its implications on society." * 63
    vector = ring.input_processor.get_embedding(text)
    for i in range(100):
        print(i,  ring.input_processor.get_embedding(text* i).shape, len(text* i))

    key_concept_id = ring.process(text,
                                  name="AI Ethics",
                                  ttl=-1,
                                  metadata={"domain": "technology", "importance": "high"}
                                  )

    # Create related concepts
    related_concepts = [
        ("Data Privacy", "The importance of protecting personal information in AI systems"),
        ("Machine Learning", "Core principles and applications of machine learning"),
        ("Human Rights", "Intersection of AI and human rights considerations"),
    ]

    for name, description in related_concepts:
        ring.process(description,
                     name=name,
                     ttl=3600,
                     metadata={"domain": "technology"}
                     )

    # Retrieve similar concepts
    similar_concepts = ring.retrieval_system.find_similar(vector)
    print("\nSimilar concepts to 'AI Ethics':")
    for concept_id, similarity in similar_concepts:
        concept = ring.get_concept_by_id(concept_id)
        if concept:
            print(f"- {concept.name}: {similarity:.3f}")

        # Search by metadata
    # Example: Print concept relationships
    print("\nConcept relationships for AI Ethics:")
    key_concept = ring.get_concept_by_id(key_concept_id)
    if key_concept:
        for related_id, similarity in key_concept.relations.items():
            related = ring.get_concept_by_id(related_id)
            if related:
                print(f"- {related.name}: {similarity:.3f}")


if __name__ == "__main__":
    main()
