"""
Async retrievers for RAGGuard.

Provides async/await support for permission-aware vector search.
Compatible with FastAPI, async frameworks, and concurrent workloads.

This module re-exports from the retrievers_async package for backwards compatibility.

For direct imports:
    from ragguard.retrievers_async import AsyncQdrantSecureRetriever
    from ragguard.retrievers_async.qdrant import AsyncQdrantSecureRetriever
"""

# Re-export all async retrievers from package
from .retrievers_async import (
    AsyncChromaDBSecureRetriever,
    AsyncFAISSSecureRetriever,
    AsyncPgvectorSecureRetriever,
    AsyncPineconeSecureRetriever,
    AsyncQdrantSecureRetriever,
    AsyncSecureRetrieverBase,
    AsyncWeaviateSecureRetriever,
    batch_search_async,
    multi_user_search_async,
    run_sync_retriever_async,
)

__all__ = [
    "AsyncChromaDBSecureRetriever",
    "AsyncFAISSSecureRetriever",
    "AsyncPgvectorSecureRetriever",
    "AsyncPineconeSecureRetriever",
    "AsyncQdrantSecureRetriever",
    "AsyncSecureRetrieverBase",
    "AsyncWeaviateSecureRetriever",
    "batch_search_async",
    "multi_user_search_async",
    "run_sync_retriever_async",
]
