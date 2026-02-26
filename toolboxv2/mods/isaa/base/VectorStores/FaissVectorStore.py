import os
import pickle
import logging

import numpy as np

from toolboxv2.mods.isaa.base.VectorStores.types import AbstractVectorStore, Chunk


class FaissVectorStore(AbstractVectorStore):
    def __init__(self, dimension: int):
        # 1. Retrieve environment variables
        agent_verbose = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
        toolbox_level = os.getenv("TOOLBOX_LOGGING_LEVEL", "INFO").upper()

        # 2. Configure faiss.loader suppression
        # Logic: Suppress if AGENT_VERBOSE is false
        # OR if AGENT_VERBOSE is true but we aren't in 'DEBUG' (min debug) or 'NOTSET' (lower)
        faiss_logger = logging.getLogger("faiss.loader")

        if not agent_verbose or toolbox_level not in ["DEBUG", "NOTSET"]:
            # Set to WARNING to hide the INFO "Loading faiss..." messages
            faiss_logger.setLevel(logging.WARNING)
        else:
            # Allow INFO/DEBUG messages if verbose is on and we are in debug mode
            faiss_logger.setLevel(logging.INFO)
        import faiss
        self.faiss = faiss
        self.dimension = dimension
        self.index = self.faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
            )
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(
        self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7
    ) -> list[Chunk]:
        if len(self.chunks) == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, k)

        from dataclasses import replace

        results = []
        for i, score in zip(indices[0], distances[0], strict=False):
            if score >= min_similarity and i < len(self.chunks):
                # Create a copy of the chunk with the score attached
                chunk_with_score = replace(self.chunks[i], score=float(score))
                results.append(chunk_with_score)
        return results

    def save(self) -> bytes:
        index_bytes = self.faiss.serialize_index(self.index)
        data = {
            "index_bytes": index_bytes,
            "chunks": self.chunks,
            "dimension": self.dimension,
        }
        return pickle.dumps(data)

    def load(self, data: bytes) -> "FaissVectorStore":

        loaded = pickle.loads(data)
        self.dimension = loaded["dimension"]
        self.index = self.faiss.deserialize_index(loaded["index_bytes"])
        self.chunks = loaded["chunks"]
        return self

    def clear(self) -> None:

        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def rebuild_index(self) -> None:
        pass  # FAISS manages its own index
