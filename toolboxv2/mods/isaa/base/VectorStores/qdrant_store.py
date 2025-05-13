"""
QdrantVectorStore implementation for the toolboxv2 system.
This vector store uses the Qdrant vector database (https://github.com/qdrant/qdrant-client)
for storing and searching embeddings.
"""

import pickle
import uuid

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        Range,
        VectorParams,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from toolboxv2.mods.isaa.base.VectorStores.defaults import AbstractVectorStore, Chunk


class QdrantVectorStore(AbstractVectorStore):
    """Vector store implementation using Qdrant vector database.

    This implementation uses the Qdrant client to store and search embeddings.
    It supports both local and remote Qdrant instances.
    """

    def __init__(
        self,
        collection_name: str = "default_collection",
        location: str | None = ":memory:",  # Use in-memory Qdrant by default
        url: str | None = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: bool | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        embedding_size: int = 768,
        distance: str = "Cosine",
        **kwargs,
    ):
        """Initialize the QdrantVectorStore.

        Args:
            collection_name: Name of the collection to use in Qdrant
            location: If ":memory:" - use in-memory Qdrant instance.
                      If None - use parameters from url or host/port.
            url: URL of the Qdrant server (e.g., "http://localhost:6333")
            port: Port of the REST API interface
            grpc_port: Port of the gRPC interface
            prefer_grpc: If true - use gPRC interface whenever possible
            https: If true - use HTTPS(SSL) protocol
            api_key: API key for authentication in Qdrant Cloud
            timeout: Timeout for REST and gRPC API requests
            host: Host name of Qdrant service
            path: Persistence path for local Qdrant
            embedding_size: Size of the embedding vectors
            distance: Distance function to use ("Cosine", "Euclid", or "Dot")
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client is not available. Please install it with: pip install qdrant-client"
            )

        self.collection_name = collection_name
        self.embedding_size = embedding_size

        # Map distance string to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        self.distance = distance_map.get(distance, Distance.COSINE)

        # Initialize Qdrant client
        self.client = QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            timeout=timeout,
            host=host,
            path=path,
        )

        # Create collection if it doesn't exist
        self._ensure_collection_exists()

        # Keep a local cache of chunks for faster access
        self.chunks = []

    def _ensure_collection_exists(self) -> None:
        """Ensure that the collection exists in Qdrant."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=self.distance,
                ),
            )

            # Create payload index for content_hash for faster lookups
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content_hash",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    # on_disk=True
                ),
            )

            # Create payload index for cluster_id for faster filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="cluster_id",
                field_schema=models.IntegerIndexParams(
                    type=models.IntegerIndexType.INTEGER,
                ),
            )

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embeddings and their corresponding chunks to the store.

        Args:
            embeddings: Numpy array of embeddings
            chunks: List of Chunk objects corresponding to the embeddings
        """
        if len(embeddings) == 0 or len(chunks) == 0:
            return

        # Prepare points for Qdrant
        points = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks, strict=False)):
            # Generate a UUID for the point if not already present
            point_id = str(uuid.uuid4())

            # Create payload from chunk
            payload = {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "content_hash": chunk.content_hash,
            }

            # Add cluster_id if it exists
            if chunk.cluster_id is not None:
                payload["cluster_id"] = chunk.cluster_id

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload,
            )

            points.append(point)

            # Add to local cache
            self.chunks.append(chunk)

        # Upsert points to Qdrant
        if len(points) < 1000:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        else:
            for i in range(0, len(points), 1000):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points[i:i+1000],
                    wait=False,
                )


    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        """Search for similar vectors.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of Chunk objects for the most similar vectors
        """
        # Convert similarity threshold to distance threshold based on distance metric
        if self.distance == Distance.COSINE:
            # For cosine, similarity of 0.7 means distance of 0.3
            score_threshold = 1 - min_similarity
        elif self.distance == Distance.DOT:
            # For dot product, higher is more similar
            score_threshold = min_similarity
        else:  # Euclidean
            # For Euclidean, lower is more similar, but there's no direct conversion
            # Using a heuristic: similarity of 0.7 means distance of ~0.5
            score_threshold = (1 - min_similarity) * 2

        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            score_threshold=score_threshold,
        )

        # Convert results to Chunks
        chunks = []
        for result in search_result:
            # Extract data from the result
            payload = result.payload
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            content_hash = payload.get("content_hash", "")
            cluster_id = payload.get("cluster_id")

            # Create embedding from the stored vector
            # Note: Qdrant doesn't return the vector by default, so we need to retrieve it separately
            # This is a limitation of this implementation
            embedding = np.array([])

            # Create and add the chunk
            chunk = Chunk(
                text=text,
                embedding=embedding,
                metadata=metadata,
                content_hash=content_hash,
                cluster_id=cluster_id,
            )
            chunks.append(chunk)

        return chunks

    def save(self) -> bytes:
        """Save the vector store to disk.

        Returns:
            Serialized state of the vector store
        """
        # We only need to save the configuration, as the data is stored in Qdrant
        state = {
            "collection_name": self.collection_name,
            "embedding_size": self.embedding_size,
            "distance": self.distance.name,
            "chunks": self.chunks,
        }
        return pickle.dumps(state)

    def load(self, data: bytes) -> 'QdrantVectorStore':
        """Load the vector store from disk.

        Args:
            data: Serialized state of the vector store

        Returns:
            Loaded vector store
        """
        state = pickle.loads(data)

        # Update configuration
        self.collection_name = state.get("collection_name", self.collection_name)
        self.embedding_size = state.get("embedding_size", self.embedding_size)

        # Convert distance string back to enum if needed
        distance_name = state.get("distance")
        if distance_name:
            self.distance = getattr(Distance, distance_name, self.distance)

        # Load chunks
        self.chunks = state.get("chunks", [])

        # Ensure collection exists with correct settings
        self._ensure_collection_exists()

        return self

    def clear(self) -> None:
        """Clear all data from the store."""
        # Delete the collection if it exists
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        # Recreate the collection
        self._ensure_collection_exists()

        # Clear local cache
        self.chunks = []

    def rebuild_index(self) -> None:
        """Rebuild the index if needed.

        For Qdrant, this is a no-op as the index is maintained automatically.
        """
        pass

    def get_by_content_hash(self, content_hash: str) -> list[Chunk]:
        """Retrieve chunks by content hash.

        Args:
            content_hash: Content hash to search for

        Returns:
            List of Chunk objects with matching content hash
        """
        # Create filter for content hash
        content_filter = Filter(
            must=[
                FieldCondition(
                    key="content_hash",
                    match=MatchValue(value=content_hash),
                ),
            ],
        )

        # Search in Qdrant
        search_result = self.client.scroll(
            collection_name=self.collection_name,
            filter=content_filter,
            limit=100,  # Adjust as needed
            with_vectors=True,  # Request vectors to be returned
        )

        # Convert results to Chunks
        chunks = []
        for point in search_result[0]:
            # Extract data from the result
            payload = point.payload
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            content_hash = payload.get("content_hash", "")
            cluster_id = payload.get("cluster_id")

            # Get the embedding if available
            embedding = np.array(point.vector) if hasattr(point, "vector") and point.vector else np.array([])

            # Create and add the chunk
            chunk = Chunk(
                text=text,
                embedding=embedding,
                metadata=metadata,
                content_hash=content_hash,
                cluster_id=cluster_id,
            )
            chunks.append(chunk)

        return chunks

    def get_by_cluster_id(self, cluster_id: int) -> list[Chunk]:
        """Retrieve chunks by cluster ID.

        Args:
            cluster_id: Cluster ID to search for

        Returns:
            List of Chunk objects with matching cluster ID
        """
        # Create filter for cluster ID
        cluster_filter = Filter(
            must=[
                FieldCondition(
                    key="cluster_id",
                    match=MatchValue(value=cluster_id),
                ),
            ],
        )

        # Search in Qdrant
        search_result = self.client.scroll(
            collection_name=self.collection_name,
            filter=cluster_filter,
            limit=100,  # Adjust as needed
            with_vectors=True,  # Request vectors to be returned
        )

        # Convert results to Chunks
        chunks = []
        for point in search_result[0]:
            # Extract data from the result
            payload = point.payload
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            content_hash = payload.get("content_hash", "")
            cluster_id = payload.get("cluster_id")

            # Get the embedding if available
            embedding = np.array(point.vector) if hasattr(point, "vector") and point.vector else np.array([])

            # Create and add the chunk
            chunk = Chunk(
                text=text,
                embedding=embedding,
                metadata=metadata,
                content_hash=content_hash,
                cluster_id=cluster_id,
            )
            chunks.append(chunk)

        return chunks

    def count(self) -> int:
        """Get the number of vectors in the store.

        Returns:
            Number of vectors in the store
        """
        # Get collection info
        collection_info = self.client.get_collection(self.collection_name)

        # Return vector count
        return collection_info.vectors_count

    def __len__(self) -> int:
        """Get the number of vectors in the store.

        Returns:
            Number of vectors in the store
        """
        return self.count()
