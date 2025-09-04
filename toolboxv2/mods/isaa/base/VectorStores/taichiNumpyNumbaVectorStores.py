from toolboxv2.mods.isaa.base.VectorStores.types import AbstractVectorStore, Chunk

try:
    numba = __import__("numba")
except ImportError:
    def numba():
        return None
    numba.jit =lambda **_:lambda x:x
    numba.njit =lambda **_:lambda x:x
    print("numba not found, using fallback")
import contextlib
import os
import pickle
import threading
from dataclasses import dataclass

import numpy as np


try:
    ti = __import__("taichi")

    use_gpu = False
    ti.init(arch=ti.gpu if use_gpu else ti.cpu)
except ImportError:
    ti = lambda: None
    print("taichi not found, using fallback")

try:

    @ti.kernel
    def batch_normalize(
        vectors: ti.types.ndarray(dtype=ti.f32),
        output: ti.types.ndarray(dtype=ti.f32),
        n: ti.i32,
        dim: ti.i32
    ):
        ti.loop_config(block_dim=256)
        for i in range(n):
            # Calculate norm
            norm_sq = 0.0
            for j in range(dim):
                val = vectors[i, j]
                norm_sq += val * val

            norm = ti.sqrt(norm_sq)
            inv_norm = 1.0 / (norm + 1e-8)

            # Normalize
            for j in range(dim):
                output[i, j] = vectors[i, j] * inv_norm
except AttributeError:

    import math
    def batch_normalize(
        vectors,
        output,
        n,
        dim
    ):
        for i in range(n):
            # Calculate norm
            norm_sq = 0.0
            for j in range(dim):
                val = vectors[i, j]
                norm_sq += val * val

            norm = math.sqrt(norm_sq)
            inv_norm = 1.0 / (norm + 1e-8)

            # Normalize
            for j in range(dim):
                output[i, j] = vectors[i, j] * inv_norm
class NumpyVectorStore(AbstractVectorStore):
    def __init__(self, use_gpu=False):
        self.embeddings = np.empty((0, 0))
        self.chunks = []
        # Initialize Taich


        self.normalized_embeddings = None

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be 2D array")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Mismatch between embeddings and chunks count")

        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            if embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Embedding dimensions must match")
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)
        # Reset normalized embeddings cache
        self.normalized_embeddings = None

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        if self.embeddings.size == 0:
            return []

        # Pre-compute normalized embeddings if not cached
        if self.normalized_embeddings is None:
            self._precompute_normalized_embeddings()

        # Normalize query
        query_norm = self._normalize_vector(query_embedding)

        # Enhanced Taichi kernel for similarity computation
        n = len(self.chunks)
        similarities = np.zeros(n, dtype=np.float32)

        @ti.kernel
        def compute_similarities_optimized(
            query: ti.types.ndarray(dtype=ti.f32),
            embeddings: ti.types.ndarray(dtype=ti.f32),
            similarities: ti.types.ndarray(dtype=ti.f32),
            n: ti.i32,
            dim: ti.i32
        ):
            ti.loop_config(block_dim=256)
            for i in range(n):
                dot_product = 0.0
                # Vectorized dot product computation
                for j in range(dim):
                    dot_product += embeddings[i, j] * query[j]
                similarities[i] = dot_product

        # Alternative optimized kernel using tile-based computation
        @ti.kernel
        def compute_similarities_tiled(
            query: ti.types.ndarray(dtype=ti.f32),
            embeddings: ti.types.ndarray(dtype=ti.f32),
            similarities: ti.types.ndarray(dtype=ti.f32),
            n: ti.i32,
            dim: ti.i32
        ):
            tile_size = 16  # Adjust based on hardware
            for i in range(n):
                dot_product = 0.0
                # Process in tiles for better cache utilization
                for jt in range(0, dim):
                    if jt % tile_size != 0:
                        continue
                    tile_sum = 0.0
                    for j in range(jt, ti.min(jt + tile_size, dim)):
                        tile_sum += embeddings[i, j] * query[j]
                    dot_product += tile_sum
                similarities[i] = dot_product

        # Choose the appropriate kernel based on dimension size
        if query_embedding.shape[0] >= 256:
            compute_similarities_tiled(
                query_norm.astype(np.float32),
                self.normalized_embeddings,
                similarities,
                n,
                query_embedding.shape[0]
            )
        else:
            compute_similarities_optimized(
                query_norm.astype(np.float32),
                self.normalized_embeddings,
                similarities,
                n,
                query_embedding.shape[0]
            )

        # Optimize top-k selection
        if k >= n:
            indices = np.argsort(-similarities)
        else:
            # Use partial sort for better performance when k < n
            indices = np.argpartition(-similarities, k)[:k]
            indices = indices[np.argsort(-similarities[indices])]

        # Filter results efficiently using vectorized operations
        mask = similarities[indices] >= min_similarity
        filtered_indices = indices[mask]
        return [self.chunks[idx] for idx in filtered_indices[:k]]

    def save(self) -> bytes:
        return pickle.dumps({
            'embeddings': self.embeddings,
            'chunks': self.chunks
        })

    def load(self, data: bytes) -> 'NumpyVectorStore':
        loaded = pickle.loads(data)
        self.embeddings = loaded['embeddings']
        self.chunks = loaded['chunks']
        return self

    def clear(self) -> None:
        self.embeddings = np.empty((0, 0))
        self.chunks = []
        self.normalized_embeddings = None

    def rebuild_index(self) -> None:
        pass  # No index to rebuild for numpy implementation

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a single vector efficiently."""
        return vector / (np.linalg.norm(vector) + 1e-8)

    def _precompute_normalized_embeddings(self) -> None:
        """Pre-compute and cache normalized embeddings."""
        # Allocate output array
        self.normalized_embeddings = np.empty_like(self.embeddings, dtype=np.float32)

        # Normalize embeddings using Taichi
        batch_normalize(
            self.embeddings.astype(np.float32),
            self.normalized_embeddings,
            self.embeddings.shape[0],
            self.embeddings.shape[1]
        )



@dataclass
class VectorStoreConfig:
    max_memory_mb: int = 1024  # Maximum memory in MB
    dimension: int = 768  # Embedding dimension
    ef_construction: int = 200  # HNSW index parameter
    M: int = 16  # HNSW index parameter
    batch_size: int = 1000  # Batch processing size
    use_mmap: bool = True  # Use memory mapping for large datasets
    use_gpu: bool = False

try:

    @ti.kernel
    def batch_normalize(
        vectors: ti.types.ndarray(dtype=ti.f32),
        output: ti.types.ndarray(dtype=ti.f32),
        n: ti.i32,
        dim: ti.i32
    ):
        for i in range(n):
            # Use block-wise processing for better cache utilization
            norm_sq = 0.0
            for j in range(0, dim):  # Process in blocks of 16
                if j % 16 != 0:
                    continue
                block_sum = 0.0
                for k in range(j, ti.min(j + 16, dim)):
                    val = vectors[i, k]
                    block_sum += val * val
                norm_sq += block_sum

            norm = ti.sqrt(norm_sq)
            inv_norm = 1.0 / (norm + 1e-8)

            # Normalize with block processing
            for j in range(0, dim):
                if j % 16 != 0:
                    continue
                for k in range(j, ti.min(j + 16, dim)):
                    output[i, k] = vectors[i, k] * inv_norm
except AttributeError:
    import math

    def batch_normalize(
        vectors,
        output,
        n,
        dim
    ):
        for i in range(n):
            # Use block-wise processing for better cache utilization
            norm_sq = 0.0
            for j in range(0, dim):  # Process in blocks of 16
                if j % 16 != 0:
                    continue
                block_sum = 0.0
                for k in range(j, min(j + 16, dim)):
                    val = vectors[i, k]
                    block_sum += val * val
                norm_sq += block_sum

            norm = math.sqrt(norm_sq)
            inv_norm = 1.0 / (norm + 1e-8)

            # Normalize with block processing
            for j in range(0, dim):
                if j % 16 != 0:
                    continue
                for k in range(j, min(j + 16, dim)):
                    output[i, k] = vectors[i, k] * inv_norm

class EnhancedVectorStore(AbstractVectorStore):
    def __init__(self, config: VectorStoreConfig):
        self.hard = False
        self.config = config or VectorStoreConfig(max_memory_mb=4096)
        self.lock = threading.Lock()

        # Initialize Taichi with appropriate backend
        ti.init(arch=ti.cuda if self.config.use_gpu else ti.cpu)
        import hnswlib
        # Initialize HNSW index
        self.index = hnswlib.Index(space='cosine', dim=self.config.dimension)
        self.index.init_index(
            max_elements=self._calculate_max_elements(),
            ef_construction=self.config.ef_construction,
            M=self.config.M
        )

        # Initialize storage
        self.chunks = []
        self.embeddings = np.empty((0, self.config.dimension), dtype=np.float32)
        self.mmap_file = None
        self.mmap_array = None

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        with self.lock:
            if len(embeddings.shape) != 2:
                raise ValueError("Embeddings must be 2D array")
            if len(chunks) != embeddings.shape[0]:
                raise ValueError("Mismatch between embeddings and chunks count")
            if embeddings.shape[1] != self.config.dimension:
                raise ValueError(f"Expected dimension {self.config.dimension}, got {embeddings.shape[1]}")

            total_elements = len(embeddings) + len(self.chunks)
            if total_elements > self._calculate_max_elements():
                print("Would exceed maximum memory limit")
                if self.hard:
                    raise MemoryError("Would exceed maximum memory limit")
                return

            normalized = np.empty_like(embeddings, dtype=np.float32)

            batch_normalize(
                embeddings.astype(np.float32),
                normalized,
                embeddings.shape[0],
                embeddings.shape[1]
            )

            # Generate IDs efficiently
            start_id = len(self.chunks)
            all_ids = np.arange(start_id, start_id + len(embeddings), dtype=np.int32)

            # Add to HNSW index in one batch for better performance
            self.index.add_items(normalized, all_ids)

            # Update storage efficiently
            if self.config.use_mmap:
                if self.mmap_array is None:
                    self._initialize_mmap(embeddings.shape[1])
                # Use direct numpy assignment for better performance
                self.mmap_array[start_id:start_id + len(embeddings)] = embeddings
            else:
                # Pre-allocate if needed
                if not hasattr(self, 'embeddings_buffer'):
                    initial_size = max(total_elements * 2, 1000)
                    self.embeddings_buffer = np.empty((initial_size, self.config.dimension), dtype=np.float32)
                    if len(self.chunks) > 0:
                        self.embeddings_buffer[:len(self.chunks)] = self.embeddings
                elif self.embeddings_buffer.shape[0] < total_elements:
                    new_size = max(total_elements * 2, self.embeddings_buffer.shape[0] * 2)
                    new_buffer = np.empty((new_size, self.config.dimension), dtype=np.float32)
                    new_buffer[:len(self.chunks)] = self.embeddings_buffer[:len(self.chunks)]
                    self.embeddings_buffer = new_buffer

                self.embeddings_buffer[start_id:start_id + len(embeddings)] = embeddings
                self.embeddings = self.embeddings_buffer[:total_elements]

            # Extend chunks list
            self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        if len(self.chunks) == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Search using HNSW index
        labels, distances = self.index.knn_query(query_norm.reshape(1, -1), k=k)

        # Convert distances to similarities and filter
        similarities = 1 - distances[0]
        return [self.chunks[idx] for idx, sim in zip(labels[0], similarities, strict=False) if sim >= min_similarity]

    def _calculate_max_elements(self) -> int:
        bytes_per_vector = self.config.dimension * 4  # 4 bytes per float32
        return (self.config.max_memory_mb * 1024 * 1024) // bytes_per_vector

    def _initialize_mmap(self, dim: int):
        mmap_path = f"vector_store_{id(self)}.mmap"
        initial_size = self._calculate_max_elements() * dim * 4

        with open(mmap_path, 'wb') as f:
            f.write(b'\0' * initial_size)

        self.mmap_file = open(mmap_path, 'r+b')
        self.mmap_array = np.memmap(
            self.mmap_file,
            dtype=np.float32,
            mode='r+',
            shape=(self._calculate_max_elements(), dim)
        )

    def save(self) -> bytes:
        with self.lock:
            self.index.save_index("index.temp")
            index = open("index.s.temp", 'rb').read()
            os.remove("index.s.temp")
            state = {
                'chunks': self.chunks,
                'index': index,
                'config': self.config
            }
            return pickle.dumps(state)

    def load(self, data: bytes) -> 'EnhancedVectorStore':
        with self.lock:
            state = pickle.loads(data)
            self.chunks = state['chunks']
            self.config = state['config']
            open("index.l.temp", "wb").write(state['index'])
            self.index.load_index("index.l.temp")
            os.remove("index.f.temp")
            return self

    def clear(self) -> None:
        import hnswlib
        with self.lock:
            self.index = hnswlib.Index(space='cosine', dim=self.config.dimension)
            self.chunks = []
            self.embeddings = np.empty((0, self.config.dimension))
            if self.mmap_file:
                self.mmap_file.close()
                self.mmap_array = None
                with contextlib.suppress(Exception):
                    os.remove(self.mmap_file.name)

    def rebuild_index(self) -> None:
        pass  # HNSW index is built incrementally


@numba.jit(nopython=True, fastmath=True, cache=True)
def _partition_vectors_helper(vectors, centroids):
    # Pre-allocate arrays for results
    assignments = np.empty(vectors.shape[0], dtype=np.int32)
    similarities = np.empty(vectors.shape[0], dtype=np.float32)

    # Assign each vector to nearest centroid
    for i in range(vectors.shape[0]):
        best_centroid = 0
        best_similarity = -1.0
        for j in range(centroids.shape[0]):
            similarity = 0.0
            for k in range(vectors.shape[1]):
                similarity += vectors[i, k] * centroids[j, k]
            if similarity > best_similarity:
                best_similarity = similarity
                best_centroid = j
        assignments[i] = best_centroid
        similarities[i] = best_similarity
    return assignments, similarities


class FastVectorStore(AbstractVectorStore):
    def __init__(self):
        torch = __import__('torch')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunks = []
        self.embeddings = None
        self.current_size = 0
        self.initial_buffer_size = 1000000

        # Index structures
        self.index_built = False
        self.index_dirty = False
        self.partition_size = 10000
        self.partition_assignments = None
        self.partition_centroids = None

        # Compile JIT functions
        self._compile_jit_functions()

    def _compile_jit_functions(self):
        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _normalize_batch_FastVectorStore(vectors):
            output = np.empty_like(vectors)
            for i in numba.prange(vectors.shape[0]):
                norm = 0.0
                for j in range(vectors.shape[1]):
                    norm += vectors[i, j] * vectors[i, j]
                norm = 1.0 / (np.sqrt(norm) + 1e-8)
                for j in range(vectors.shape[1]):
                    output[i, j] = vectors[i, j] * norm
            return output

        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _compute_centroids_FastVectorStore(vectors, n_centroids):
            chunk_size = vectors.shape[0] // n_centroids
            centroids = np.empty((n_centroids, vectors.shape[1]), dtype=np.float32)
            for i in numba.prange(n_centroids):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < n_centroids - 1 else vectors.shape[0]
                centroid = np.zeros(vectors.shape[1], dtype=np.float32)
                count = end_idx - start_idx
                for j in range(start_idx, end_idx):
                    for k in range(vectors.shape[1]):
                        centroid[k] += vectors[j, k]
                for k in range(vectors.shape[1]):
                    centroids[i, k] = centroid[k] / count
            return centroids

        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _batch_similarity_FastVectorStore(query, vectors):
            similarities = np.empty(vectors.shape[0], dtype=np.float32)
            for i in numba.prange(vectors.shape[0]):
                dot = 0.0
                for j in range(vectors.shape[1]):
                    dot += query[j] * vectors[i, j]
                similarities[i] = dot
            return similarities

        self._normalize_batch = _normalize_batch_FastVectorStore
        self._compute_centroids = _compute_centroids_FastVectorStore
        self._batch_similarity = _batch_similarity_FastVectorStore

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) == 0:
            return

        # Initialize storage if needed
        if self.embeddings is None:
            self.embeddings = np.empty((self.initial_buffer_size, embeddings.shape[1]), dtype=np.float32)

        # Ensure capacity
        if self.current_size + len(embeddings) > self.embeddings.shape[0]:
            new_size = max(self.current_size + len(embeddings), self.embeddings.shape[0] * 2)
            new_storage = np.empty((new_size, embeddings.shape[1]), dtype=np.float32)
            if self.current_size > 0:
                new_storage[:self.current_size] = self.embeddings[:self.current_size]
            self.embeddings = new_storage

        # Normalize and add embeddings
        normalized = self._normalize_batch(embeddings.astype(np.float32))
        self.embeddings[self.current_size:self.current_size + len(embeddings)] = normalized
        self.chunks.extend(chunks)
        self.current_size += len(embeddings)
        if len(embeddings) > 1000:
            self.rebuild_index()

    def _build_index_if_needed(self):
        if not self.index_dirty:
            return

        if self.current_size < self.partition_size:
            self.index_built = True
            self.index_dirty = False
            return

        # Compute number of partitions
        n_partitions = max(1, self.current_size // self.partition_size)

        # Compute centroids
        vectors = self.embeddings[:self.current_size]
        self.partition_centroids = self._compute_centroids(vectors, n_partitions)

        # Partition vectors
        self.partition_assignments, _ = _partition_vectors_helper(vectors, self.partition_centroids)

        self.index_built = True
        self.index_dirty = False

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        if self.current_size == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Build index if needed
        self._build_index_if_needed()

        if not self.index_built or self.current_size < self.partition_size:
            # Direct search for small datasets
            similarities = self._batch_similarity(
                query_norm,
                self.embeddings[:self.current_size]
            )

            # Process results
            top_k = min(k, len(similarities))
            if top_k <= 0:
                return []
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

            # Filter by similarity threshold
            mask = similarities[top_indices] >= min_similarity
            filtered_indices = top_indices[mask]

        else:
            # Use index for large datasets
            partition_similarities = self._batch_similarity(query_norm, self.partition_centroids)
            top_partitions = np.argsort(-partition_similarities)[:2]

            # Get candidates from top partitions
            candidate_mask = np.isin(self.partition_assignments, top_partitions)
            candidate_indices = np.where(candidate_mask)[0]

            if len(candidate_indices) == 0:
                return []

            # Compute similarities for candidates
            candidates = self.embeddings[candidate_indices]
            similarities = self._batch_similarity(query_norm, candidates)

            # Get top-k results
            top_k = min(k, len(similarities))
            if top_k <= 0:
                return []
            top_local_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_local_indices = top_local_indices[np.argsort(-similarities[top_local_indices])]

            # Map to original indices and filter
            filtered_indices = candidate_indices[top_local_indices[similarities[top_local_indices] >= min_similarity]]

        return [self.chunks[idx] for idx in filtered_indices]

    def save(self) -> bytes:
        state = {
            'embeddings': self.embeddings[:self.current_size],
            'chunks': self.chunks,
            'current_size': self.current_size,
            'partition_centroids': self.partition_centroids,
            'index_partitions': self.index_partitions,
            'index_built': self.index_built,
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, data: bytes) -> 'FastVectorStore':
        state = pickle.loads(data)
        self.current_size = state['current_size']
        self.chunks = state['chunks']
        self.embeddings = state['embeddings']
        self.partition_centroids = state['partition_centroids']
        self.index_partitions = state['index_partitions']
        self.index_built = state['index_built']
        return self

    def clear(self) -> None:
        self.embeddings = None
        self.chunks = []
        self.current_size = 0
        self.index_built = False
        self.index_dirty = False
        self.partition_centroids = None
        self.index_partitions = []

    def rebuild_index(self) -> None:
        self.index_dirty = True
        self._build_index_if_needed()

class FastVectorStoreO(AbstractVectorStore):
    def __init__(self, embedding_size=768, initial_buffer_size=1000):
        torch = __import__('torch')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_size = embedding_size
        self.initial_buffer_size = initial_buffer_size

        # Pre-allocate embeddings buffer with known size
        self.embeddings = np.empty((initial_buffer_size, embedding_size), dtype=np.float32)
        self.chunks = []
        self.current_size = 0

        # Index structures
        self.index_built = False
        self.index_dirty = False
        self.partition_size = 10000
        self.partition_assignments = None
        self.partition_centroids = None

        # Compile JIT functions
        self._compile_jit_functions()

    def _compile_jit_functions(self):
        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _normalize_batch(vectors):
            output = np.empty_like(vectors)
            for i in numba.prange(vectors.shape[0]):
                row = vectors[i]
                norm = np.sqrt(np.sum(row ** 2)) + 1e-8
                inv_norm = 1.0 / norm
                output[i] = row * inv_norm
            return output

        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _compute_centroids(vectors, n_centroids):
            chunk_size = vectors.shape[0] // n_centroids
            centroids = np.empty((n_centroids, vectors.shape[1]), dtype=vectors.dtype)
            for i in numba.prange(n_centroids):
                start = i * chunk_size
                end = start + chunk_size
                if i == n_centroids - 1:
                    end = vectors.shape[0]
                chunk = vectors[start:end]
                centroids[i] = np.sum(chunk, axis=0) / chunk.shape[0]
            return centroids

        @numba.jit(nopython=True, fastmath=True, cache=True)
        def _batch_similarity(query, vectors):
            return vectors @ query

        self._normalize_batch = _normalize_batch
        self._compute_centroids = _compute_centroids
        self._batch_similarity = _batch_similarity

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) == 0:
            return

        # Initialize storage if needed
        if self.embeddings is None:
            self.embeddings = np.empty((self.initial_buffer_size, embeddings.shape[1]), dtype=np.float32)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)

            # Calculate required space
        required_space = self.current_size + len(embeddings)
        if required_space > self.embeddings.shape[0]:
            # Increase by max between doubling and required_space
            new_size = max(required_space, self.embeddings.shape[0] * 2)
            new_storage = np.empty((new_size, self.embedding_size), dtype=np.float32)
            if self.current_size > 0:
                new_storage[:self.current_size] = self.embeddings[:self.current_size]
            self.embeddings = new_storage

        # Normalize and add embeddings
        normalized = self._normalize_batch(embeddings)
        end = self.current_size + len(embeddings)
        self.embeddings[self.current_size:end] = normalized
        self.chunks.extend(chunks)
        self.current_size = end

        # Rebuild index only when crossing partition boundaries
        if self.current_size // self.partition_size > (self.current_size - len(embeddings)) // self.partition_size:
            self.rebuild_index()

    def _build_index_if_needed(self):
        if not self.index_dirty:
            return

        if self.current_size < self.partition_size:
            self.index_built = True
            self.index_dirty = False
            return

        # Compute number of partitions
        n_partitions = max(1, self.current_size // self.partition_size)

        # Compute centroids
        vectors = self.embeddings[:self.current_size]
        self.partition_centroids = self._compute_centroids(vectors, n_partitions)

        # Partition vectors
        self.partition_assignments, _ = _partition_vectors_helper(vectors, self.partition_centroids)

        self.index_built = True
        self.index_dirty = False

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        if self.current_size == 0:
            return []

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32, copy=False)
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Build index if needed
        self._build_index_if_needed()

        if not self.index_built or self.current_size < self.partition_size:
            # Direct search for small datasets
            similarities = self._batch_similarity(
                query_norm,
                self.embeddings[:self.current_size]
            )

            # Process results
            top_k = min(k, len(similarities))
            if top_k <= 0:
                return []
            top_indices = np.argpartition(-similarities, top_k-1)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

            # Filter by similarity threshold
            mask = similarities[top_indices] >= min_similarity
            filtered_indices = top_indices[mask]

        else:
            # Use index for large datasets
            partition_similarities = self._batch_similarity(query_norm, self.partition_centroids)
            top_partitions = np.argsort(-partition_similarities)[:2]

            # Get candidates from top partitions
            candidate_mask = np.isin(self.partition_assignments, top_partitions)
            candidate_indices = np.where(candidate_mask)[0]

            if len(candidate_indices) == 0:
                return []

            # Compute similarities for candidates
            candidates = self.embeddings[candidate_indices]
            similarities = self._batch_similarity(query_norm, candidates)

            # Get top-k results
            top_k = min(k, len(similarities))
            if top_k <= 0:
                return []
            top_local_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_local_indices = top_local_indices[np.argsort(-similarities[top_local_indices])]

            # Map to original indices and filter
            filtered_indices = candidate_indices[top_local_indices[similarities[top_local_indices] >= min_similarity]]

        return [self.chunks[idx] for idx in filtered_indices if idx < len(self.chunks)]

    def save(self) -> bytes:
        state = {
            'embeddings': self.embeddings[:self.current_size],
            'chunks': self.chunks,
            'current_size': self.current_size,
            'partition_centroids': self.partition_centroids,
            'index_built': self.index_built,
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, data: bytes) -> 'FastVectorStore':
        state = pickle.loads(data)
        self.current_size = state['current_size']
        self.chunks = state['chunks']
        self.embeddings = state['embeddings']
        self.partition_centroids = state['partition_centroids']
        self.index_built = state['index_built']
        return self

    def clear(self) -> None:
        self.embeddings = None
        self.chunks = []
        self.current_size = 0
        self.index_built = False
        self.index_dirty = False
        self.partition_centroids = None

    def rebuild_index(self) -> None:
        self.index_dirty = True
        self._build_index_if_needed()

class FastVectorStore1(AbstractVectorStore):
    def __init__(self):
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.chunks = []
        self.normalized_embeddings = None

    @staticmethod
    @numba.jit(parallel=True)
    def _batch_normalize(embeddings, normalized):
        n, dim = embeddings.shape
        for i in numba.prange(n):
            norm = 0.0
            for j in range(dim):
                norm += embeddings[i, j] * embeddings[i, j]
            norm = np.sqrt(norm) + 1e-8
            for j in range(dim):
                normalized[i, j] = embeddings[i, j] / norm

    @staticmethod
    @numba.njit(parallel=True)
    def _compute_similarities(query, embeddings, similarities, n, dim):
        for i in numba.prange(n):
            dot_product = 0.0
            for j in range(dim):
                dot_product += embeddings[i, j] * query[j]
            similarities[i] = dot_product

    @staticmethod
    @numba.njit()
    def _normalize_vector(vector):
        norm = np.sqrt(np.sum(vector * vector)) + 1e-8
        return vector / norm

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be 2D array")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Mismatch between embeddings and chunks count")

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)

        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            if embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Embedding dimensions must match")
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.chunks.extend(chunks)
        self.normalized_embeddings = None  # Reset cache

    def _precompute_normalized_embeddings(self) -> None:
        if self.normalized_embeddings is None:
            self.normalized_embeddings = np.empty_like(self.embeddings, dtype=np.float32)
            self._batch_normalize(self.embeddings, self.normalized_embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        if self.embeddings.size == 0:
            return []

        # Ensure query is float32
        query_embedding = query_embedding.astype(np.float32)

        # Pre-compute normalized embeddings if not cached
        if self.normalized_embeddings is None:
            self._precompute_normalized_embeddings()

        # Normalize query
        query_norm = self._normalize_vector(query_embedding)

        # Compute similarities
        n = len(self.chunks)
        similarities = np.zeros(n, dtype=np.float32)
        self._compute_similarities(
            query_norm,
            self.normalized_embeddings,
            similarities,
            n,
            query_embedding.shape[0]
        )

        # Optimize top-k selection
        if k >= n:
            indices = np.argsort(-similarities)
        else:
            indices = np.argpartition(-similarities, k)[:k]
            indices = indices[np.argsort(-similarities[indices])]

        # Filter results
        mask = similarities[indices] >= min_similarity
        filtered_indices = indices[mask]
        return [self.chunks[idx] for idx in filtered_indices[:k]]

    def save(self) -> bytes:
        return pickle.dumps({
            'embeddings': self.embeddings,
            'chunks': self.chunks
        })

    def load(self, data: bytes) -> 'FastVectorStoreO':
        loaded = pickle.loads(data)
        self.embeddings = loaded['embeddings'].astype(np.float32)
        self.chunks = loaded['chunks']
        self.normalized_embeddings = None
        return self

    def clear(self) -> None:
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.chunks = []
        self.normalized_embeddings = None

    def rebuild_index(self) -> None:
        pass  # No index to rebuild for this implementation

class FastVectorStore2(AbstractVectorStore):
    def __init__(self, embedding_size=768, initial_buffer_size=1000000, use_gpu=False):
        # Initialize Taichi with GPU if available



        self.embedding_size = embedding_size
        self.initial_buffer_size = initial_buffer_size

        # Pre-allocate embeddings buffer
        self.embeddings = np.empty((initial_buffer_size, embedding_size), dtype=np.float32)
        self.chunks = []
        self.current_size = 0

        # Index structures
        self.index_built = False
        self.index_dirty = False
        self.partition_size = 10000
        self.partition_assignments = None
        self.partition_centroids = None

        # Compile kernels
        self._compile_kernels()

    def _compile_kernels(self):

        # Taichi kernels for GPU-accelerated operations
        @ti.kernel
        def normalize_batch_gpu_FastVectorStore2(
            vectors: ti.types.ndarray(),
            output: ti.types.ndarray(),
            n: ti.i32,
            dim: ti.i32
        ):
            for i in range(n):
                norm_sq = 0.0
                for j in range(dim):
                    norm_sq += vectors[i, j] * vectors[i, j]
                norm = ti.sqrt(norm_sq) + 1e-8
                for j in range(dim):
                    output[i, j] = vectors[i, j] / norm

        @ti.kernel
        def batch_similarity_gpu_FastVectorStore2(
            query: ti.types.ndarray(),
            vectors: ti.types.ndarray(),
            output: ti.types.ndarray(),
            n: ti.i32,
            dim: ti.i32
        ):
            for i in range(n):
                dot_product = 0.0
                for j in range(dim):
                    dot_product += vectors[i, j] * query[j]
                output[i] = dot_product

        # Numba functions for CPU operations
        @numba.jit(nopython=True, parallel=True, fastmath=True)
        def compute_centroids_cpu_FastVectorStore2(vectors, n_centroids):
            chunk_size = vectors.shape[0] // n_centroids
            centroids = np.empty((n_centroids, vectors.shape[1]), dtype=vectors.dtype)
            for i in numba.prange(n_centroids):
                start = i * chunk_size
                end = start + chunk_size if i < n_centroids - 1 else vectors.shape[0]
                chunk = vectors[start:end]
                centroids[i] = np.mean(chunk, axis=0)
            return centroids

        self._normalize_batch_gpu = normalize_batch_gpu_FastVectorStore2
        self._batch_similarity_gpu = batch_similarity_gpu_FastVectorStore2
        self._compute_centroids_cpu = compute_centroids_cpu_FastVectorStore2

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) == 0:
            return

        # Ensure correct data type
        embeddings = embeddings.astype(np.float32, copy=False)

        # Resize if needed
        required_space = self.current_size + len(embeddings)
        if required_space > self.embeddings.shape[0]:
            new_size = max(required_space, self.embeddings.shape[0] * 2)
            new_storage = np.empty((new_size, self.embedding_size), dtype=np.float32)
            if self.current_size > 0:
                new_storage[:self.current_size] = self.embeddings[:self.current_size]
            self.embeddings = new_storage

        # Normalize embeddings using GPU if available
        normalized = np.empty_like(embeddings)
        self._normalize_batch_gpu(
            embeddings,
            normalized,
            embeddings.shape[0],
            embeddings.shape[1]
        )

        # Add to storage
        end = self.current_size + len(embeddings)
        self.embeddings[self.current_size:end] = normalized
        self.chunks.extend(chunks)
        self.current_size = end

        # Mark index as dirty
        self.index_dirty = True

    def search(self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.7) -> list[Chunk]:
        if self.current_size == 0:
            return []

        # Normalize query
        query_norm = query_embedding.astype(np.float32)
        query_norm /= (np.linalg.norm(query_norm) + 1e-8)

        # Build index if needed
        self._build_index_if_needed()

        if not self.index_built or self.current_size < self.partition_size:
            # Direct search using GPU
            similarities = np.zeros(self.current_size, dtype=np.float32)
            self._batch_similarity_gpu(
                query_norm,
                self.embeddings[:self.current_size],
                similarities,
                self.current_size,
                self.embedding_size
            )
        else:
            # Use indexed search
            partition_similarities = np.zeros(len(self.partition_centroids), dtype=np.float32)
            self._batch_similarity_gpu(
                query_norm,
                self.partition_centroids,
                partition_similarities,
                len(self.partition_centroids),
                self.embedding_size
            )

            # Get candidates from top partitions
            top_partitions = np.argsort(-partition_similarities)[:2]
            candidate_mask = np.isin(self.partition_assignments, top_partitions)
            candidate_indices = np.where(candidate_mask)[0]

            if len(candidate_indices) == 0:
                return []

            # Compute similarities for candidates
            similarities = np.zeros(len(candidate_indices), dtype=np.float32)
            self._batch_similarity_gpu(
                query_norm,
                self.embeddings[candidate_indices],
                similarities,
                len(candidate_indices),
                self.embedding_size
            )

        # Process results
        top_k = min(k, len(similarities))
        if top_k <= 0:
            return []
        indices = np.argpartition(-similarities, top_k)[:top_k]
        indices = indices[np.argsort(-similarities[indices])]

        # Filter by similarity threshold
        mask = similarities[indices] >= min_similarity
        filtered_indices = indices[mask]

        return [self.chunks[idx] for idx in filtered_indices]

    def _build_index_if_needed(self):
        if not self.index_dirty:
            return

        if self.current_size < self.partition_size:
            self.index_built = True
            self.index_dirty = False
            return

        # Compute centroids using CPU (Numba)
        n_partitions = max(1, self.current_size // self.partition_size)
        vectors = self.embeddings[:self.current_size]
        self.partition_centroids = self._compute_centroids_cpu(vectors, n_partitions)

        # Compute assignments using GPU
        self.partition_assignments = np.zeros(self.current_size, dtype=np.int32)
        similarities = np.zeros((self.current_size, n_partitions), dtype=np.float32)

        # Compute similarities to centroids
        for i in range(n_partitions):
            centroid = self.partition_centroids[i:i + 1]
            self._batch_similarity_gpu(
                centroid[0],
                vectors,
                similarities[:, i],
                self.current_size,
                self.embedding_size
            )

        self.partition_assignments = np.argmax(similarities, axis=1)
        self.index_built = True
        self.index_dirty = False

    @staticmethod
    @numba.jit(parallel=True)
    def _batch_normalize(embeddings, normalized):
        n, dim = embeddings.shape
        for i in numba.prange(n):
            norm = 0.0
            for j in range(dim):
                norm += embeddings[i, j] * embeddings[i, j]
            norm = np.sqrt(norm) + 1e-8
            for j in range(dim):
                normalized[i, j] = embeddings[i, j] / norm

    @staticmethod
    @numba.njit(parallel=True)
    def _compute_similarities(query, embeddings, similarities, n, dim):
        for i in numba.prange(n):
            dot_product = 0.0
            for j in range(dim):
                dot_product += embeddings[i, j] * query[j]
            similarities[i] = dot_product

    @staticmethod
    @numba.njit()
    def _normalize_vector(vector):
        norm = np.sqrt(np.sum(vector * vector)) + 1e-8
        return vector / norm


    def _precompute_normalized_embeddings(self) -> None:
        if self.normalized_embeddings is None:
            self.normalized_embeddings = np.empty_like(self.embeddings, dtype=np.float32)
            self._batch_normalize(self.embeddings, self.normalized_embeddings)

    def save(self) -> bytes:
        return pickle.dumps({
            'embeddings': self.embeddings,
            'chunks': self.chunks
        })

    def load(self, data: bytes) -> 'FastVectorStoreO':
        loaded = pickle.loads(data)
        self.embeddings = loaded['embeddings'].astype(np.float32)
        self.chunks = loaded['chunks']
        self.normalized_embeddings = None
        return self

    def clear(self) -> None:
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.chunks = []
        self.normalized_embeddings = None

    def rebuild_index(self) -> None:
        pass  # No index to rebuild for this implementation
