"""
Vector store implementations for the toolboxv2 system.
"""

from toolboxv2.mods.isaa.base.VectorStores.defaults import (
    AbstractVectorStore,
    EnhancedVectorStore,
    FaissVectorStore,
    FastVectorStore,
    FastVectorStore1,
    FastVectorStore2,
    FastVectorStoreO,
    NumpyVectorStore,
    RedisVectorStore,
    VectorStoreConfig,
)

try:
    from toolboxv2.mods.isaa.base.VectorStores.qdrant_store import QdrantVectorStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

__all__ = [
    "AbstractVectorStore",
    "NumpyVectorStore",
    "FastVectorStore",
    "FastVectorStoreO",
    "FastVectorStore1",
    "FastVectorStore2",
    "EnhancedVectorStore",
    "VectorStoreConfig",
    "RedisVectorStore",
    "FaissVectorStore",
]

if QDRANT_AVAILABLE:
    __all__.append("QdrantVectorStore")
