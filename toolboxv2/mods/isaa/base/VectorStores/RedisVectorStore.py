import contextlib
import json
import pickle

import redis

try:
    from redis.commands.search.field import TextField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition
    from redis.commands.search.query import Query
except ImportError:
    def VectorField(*a, **k):
        return None
    def TextField(*a, **k):
        return None
    def Query(*a, **k):
        return None
    def IndexDefinition(*a, **k):
        return None

import numpy as np

from toolboxv2.mods.isaa.base.VectorStores.types import AbstractVectorStore, Chunk


class RedisVectorStore(AbstractVectorStore):
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 index_name: str = "chunks_index",
                 vector_dim: int = 768):
        self.redis_client = redis.from_url(redis_url)
        self.index_name = index_name
        self.vector_dim = vector_dim
        self.prefix = "chunk:"
        self._create_index_if_not_exists()

    def _create_index_if_not_exists(self):
        try:
            self.redis_client.ft(self.index_name).info()
        except:
            schema = (
                TextField("text"),
                TextField("metadata"),
                TextField("content_hash"),
                TextField("cluster_id"),
                VectorField("embedding", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            definition = IndexDefinition(prefix=[self.prefix])
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition
            )

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        pipe = self.redis_client.pipeline()
        for emb, chunk in zip(embeddings, chunks, strict=False):
            key = f"{self.prefix}{chunk.content_hash}"
            pipe.hset(key, mapping={
                "text": chunk.text,
                "metadata": json.dumps(chunk.metadata),
                "content_hash": chunk.content_hash,
                "cluster_id": str(chunk.cluster_id) if chunk.cluster_id is not None else "",
                "embedding": emb.astype(np.float32).tobytes()
            })
        pipe.execute()

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        query_bytes = query_embedding.astype(np.float32).tobytes()
        query = (
            Query(f"*=>[KNN {k} @embedding $vec AS score]")
            .return_fields("text", "metadata", "content_hash", "cluster_id", "score")
            .dialect(2)
        )
        params = {"vec": query_bytes}
        results = self.redis_client.ft(self.index_name).search(query, query_params=params)

        chunks = []
        for doc in results.docs:
            similarity = 1 - float(doc.score)  # Convert distance to similarity
            if similarity >= min_similarity:
                embedding_bytes = self.redis_client.hget(doc.id, "embedding")
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                metadata = json.loads(doc.metadata)
                cluster_id = int(doc.cluster_id) if doc.cluster_id else None

                chunk = Chunk(
                    text=doc.text,
                    embedding=embedding,
                    metadata=metadata,
                    content_hash=doc.content_hash,
                    cluster_id=cluster_id
                )
                chunks.append(chunk)

        return chunks[:k]

    def save(self) -> bytes:
        # Export all data from Redis
        keys = self.redis_client.keys(f"{self.prefix}*")
        data = []
        for key in keys:
            hash_data = self.redis_client.hgetall(key)
            data.append(hash_data)
        return pickle.dumps(data)

    def load(self, data: bytes) -> 'RedisVectorStore':
        self.clear()
        loaded_data = pickle.loads(data)
        pipe = self.redis_client.pipeline()
        for hash_data in loaded_data:
            key = f"{self.prefix}{hash_data[b'content_hash'].decode()}"
            pipe.hset(key, mapping={
                k.decode(): v for k, v in hash_data.items()
            })
        pipe.execute()
        return self

    def clear(self) -> None:
        keys = self.redis_client.keys(f"{self.prefix}*")
        if keys:
            self.redis_client.delete(*keys)
        with contextlib.suppress(Exception):
            self.redis_client.ft(self.index_name).dropindex(delete_documents=False)
        self._create_index_if_not_exists()

    def rebuild_index(self) -> None:
        pass  # Redis manages its own index
