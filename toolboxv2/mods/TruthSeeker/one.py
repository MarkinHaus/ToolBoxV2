import base64
import io
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)

from toolboxv2 import Singleton, get_logger
from toolboxv2.utils.system import FileCache

logger = get_logger()


class InputData:
    def __init__(
        self,
        content: str | bytes | np.ndarray,
        modality: str,
        metadata: dict | None = None
    ):
        self.content = content
        self.modality = modality  # 'text', 'image', or 'audio'
        self.metadata = metadata or {}


class IntelligenceRingEmbeddings:
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_name: str = "openai/clip-vit-base-patch32"
    wav2vec_name: str = "facebook/wav2vec2-base-960h"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    vector_size: int = 768
    tokenizer: Any | None = None
    text_model: Any | None = None

    clip_processor: Any | None = None
    clip_model: Any | None = None

    audio_processor: Any | None = None
    audio_model: Any | None = None

    text_projection: Any | None = None
    image_projection: Any | None = None
    audio_projection: Any | None = None

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

    def _process_image(self, image_data: bytes | str) -> torch.Tensor:
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

    def _process_audio(self, audio_data: bytes | str | np.ndarray) -> torch.Tensor:
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

    def compute_query_embeddings(self, query: str | bytes | np.ndarray, modality: str = "text") -> list[
        np.ndarray]:
        """Compute embeddings for query input"""
        input_data = InputData(query, modality)
        embedding = self.process_input(input_data)
        return [embedding.squeeze()]

    def compute_source_embeddings(self, sources: list[str | bytes | np.ndarray], modalities: list[str]) -> list[
        np.ndarray]:
        """Compute embeddings for source inputs"""
        embeddings = []
        for source, modality in zip(sources, modalities, strict=False):
            input_data = InputData(source, modality)
            embedding = self.process_input(input_data)
            embeddings.append(embedding.squeeze())
        return embeddings

    def ndims(self) -> int:
        return self._ndims


@dataclass
class Concept:
    id: str
    name: str
    ttl: int
    created_at: datetime
    vector: np.ndarray
    contradictions: set[str]
    similar_concepts: set[str]
    relations: dict[str, float]
    stage: int
    metadata: dict
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
        self.started = threading.Event()
        def helper():
            print("InputProcessor starting ...")
            self.started.clear()
            self.embedding_function = IntelligenceRingEmbeddings()
            self.vector_size = self.embedding_function.vector_size

            cache_dir = os.getenv('APPDATA') if os.name == 'nt' else os.getenv('XDG_CONFIG_HOME') or os.path.expanduser(
                '~/.config') if os.name == 'posix' else "."
            self.cache = FileCache(
                folder=cache_dir + '\\ToolBoxV2\\cache\\InputProcessor\\',
                filename=cache_dir + '\\ToolBoxV2\\cache\\InputProcessor\\cache.db'
            )
            print("InputProcessor online ...")
            self.started.set()
        threading.Thread(target=helper, daemon=True).start()

    def get_embedding(self, content: str | bytes | np.ndarray, modality: str = "text") -> np.ndarray | None:
        while not self.started.is_set():
            time.sleep(0.2)
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


                if chunk_embedding is not None and modality == "text" and isinstance(content, str):
                    # Cache the chunk embedding
                    self.cache.set(cache_key, chunk_embedding)

                chunk_embeddings.append(chunk_embedding)

        # Combine all chunk embeddings into a single vector of the same shape
        combined_embedding = np.mean([e for e in chunk_embeddings if len(e) == self.vector_size], axis=0) if chunk_embeddings else None
        return combined_embedding


    def batch_get_embeddings(self, contents: list[str | bytes | np.ndarray], modalities: list[str]) -> np.ndarray | None:
        try:
            results = []
            for content, modality in zip(contents, modalities, strict=False):
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

