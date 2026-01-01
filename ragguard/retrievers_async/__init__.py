"""
Async retrievers for RAGGuard.

Provides async/await support for permission-aware vector search.
Compatible with FastAPI, async frameworks, and concurrent workloads.

This module uses lazy imports to avoid circular dependencies and
improve startup performance.
"""

from typing import TYPE_CHECKING

# Import base class eagerly
from .base import AsyncSecureRetrieverBase

# Import utility functions eagerly (no dependencies)
from .utils import (
    BatchSearchResult,
    MultiUserSearchResult,
    batch_search_async,
    multi_user_search_async,
    run_sync_retriever_async,
)

# Type checking imports (not evaluated at runtime)
if TYPE_CHECKING:
    from .arangodb import AsyncArangoDBSecureRetriever
    from .azure_search import AsyncAzureSearchSecureRetriever
    from .chromadb import AsyncChromaDBSecureRetriever
    from .elasticsearch import AsyncElasticsearchSecureRetriever, AsyncOpenSearchSecureRetriever
    from .faiss import AsyncFAISSSecureRetriever
    from .graph_base import AsyncGraphSecureRetrieverBase
    from .milvus import AsyncMilvusSecureRetriever
    from .neo4j import AsyncNeo4jSecureRetriever
    from .neptune import AsyncNeptuneSecureRetriever
    from .pgvector import AsyncPgvectorSecureRetriever
    from .pinecone import AsyncPineconeSecureRetriever
    from .qdrant import AsyncQdrantSecureRetriever
    from .tigergraph import AsyncTigerGraphSecureRetriever
    from .weaviate import AsyncWeaviateSecureRetriever

# Lazy import support for retriever classes
__all__ = [
    # Base classes
    "AsyncSecureRetrieverBase",
    "AsyncGraphSecureRetrieverBase",
    # Vector database retrievers
    "AsyncQdrantSecureRetriever",
    "AsyncChromaDBSecureRetriever",
    "AsyncPineconeSecureRetriever",
    "AsyncWeaviateSecureRetriever",
    "AsyncPgvectorSecureRetriever",
    "AsyncFAISSSecureRetriever",
    "AsyncMilvusSecureRetriever",
    "AsyncElasticsearchSecureRetriever",
    "AsyncOpenSearchSecureRetriever",
    "AsyncAzureSearchSecureRetriever",
    # Graph database retrievers
    "AsyncNeo4jSecureRetriever",
    "AsyncNeptuneSecureRetriever",
    "AsyncTigerGraphSecureRetriever",
    "AsyncArangoDBSecureRetriever",
    # Result types
    "BatchSearchResult",
    "MultiUserSearchResult",
    # Utility functions
    "run_sync_retriever_async",
    "batch_search_async",
    "multi_user_search_async",
]


def __getattr__(name: str):
    """
    Lazy import for retriever classes.

    This prevents circular imports and improves startup time by only
    importing retriever classes when they're actually used.
    """
    # Map of class names to module names
    retriever_map = {
        # Vector database retrievers
        "AsyncQdrantSecureRetriever": "qdrant",
        "AsyncChromaDBSecureRetriever": "chromadb",
        "AsyncPineconeSecureRetriever": "pinecone",
        "AsyncWeaviateSecureRetriever": "weaviate",
        "AsyncPgvectorSecureRetriever": "pgvector",
        "AsyncFAISSSecureRetriever": "faiss",
        "AsyncMilvusSecureRetriever": "milvus",
        "AsyncElasticsearchSecureRetriever": "elasticsearch",
        "AsyncOpenSearchSecureRetriever": "elasticsearch",
        "AsyncAzureSearchSecureRetriever": "azure_search",
        # Graph database retrievers
        "AsyncNeo4jSecureRetriever": "neo4j",
        "AsyncNeptuneSecureRetriever": "neptune",
        "AsyncTigerGraphSecureRetriever": "tigergraph",
        "AsyncArangoDBSecureRetriever": "arangodb",
        # Base classes
        "AsyncGraphSecureRetrieverBase": "graph_base",
    }

    if name in retriever_map:
        # Import from the specific module
        module_name = retriever_map[name]
        from importlib import import_module
        module = import_module(f".{module_name}", package=__name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
