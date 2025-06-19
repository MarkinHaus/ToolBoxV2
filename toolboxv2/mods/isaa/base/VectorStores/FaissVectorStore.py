import pickle

import numpy as np

from toolboxv2.mods.isaa.base.VectorStores.types import AbstractVectorStore, Chunk


class FaissVectorStore(AbstractVectorStore):
    def __init__(self, dimension: int):
        import faiss

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        if len(self.chunks) == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, k)

        results = []
        for i, score in zip(indices[0], distances[0], strict=False):
            if score >= min_similarity and i < len(self.chunks):
                results.append(self.chunks[i])
        return results

    def save(self) -> bytes:
        import faiss

        index_bytes = faiss.serialize_index(self.index)
        data = {
            'index_bytes': index_bytes,
            'chunks': self.chunks,
            'dimension': self.dimension
        }
        return pickle.dumps(data)

    def load(self, data: bytes) -> 'FaissVectorStore':
        import faiss

        loaded = pickle.loads(data)
        self.dimension = loaded['dimension']
        self.index = faiss.deserialize_index(loaded['index_bytes'])
        self.chunks = loaded['chunks']
        return self

    def clear(self) -> None:
        import faiss

        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def rebuild_index(self) -> None:
        pass  # FAISS manages its own index

