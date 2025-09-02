from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Chunk:
    """Represents a chunk of text with its embedding and metadata"""
    text: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    content_hash: str
    cluster_id: int | None = None


class AbstractVectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embeddings and their corresponding chunks to the store"""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def save(self) -> bytes:
        """Save the vector store to disk"""
        pass

    @abstractmethod
    def load(self, data: bytes) -> 'AbstractVectorStore':
        """Load the vector store from disk"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store"""
        pass

    @abstractmethod
    def rebuild_index(self) -> None:
        """Optional for faster searches"""
        pass

