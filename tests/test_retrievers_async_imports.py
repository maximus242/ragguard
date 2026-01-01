"""
Tests for async retrievers module imports.

Tests that all async retriever exports are available.
"""

import pytest


class TestAsyncRetrieverImports:
    """Tests for async retriever module imports."""

    def test_import_from_retrievers_async_module(self):
        """Test importing from ragguard.retrievers_async module."""
        from ragguard.retrievers_async import (
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

        # Verify all classes/functions are importable
        assert AsyncChromaDBSecureRetriever is not None
        assert AsyncFAISSSecureRetriever is not None
        assert AsyncPgvectorSecureRetriever is not None
        assert AsyncPineconeSecureRetriever is not None
        assert AsyncQdrantSecureRetriever is not None
        assert AsyncSecureRetrieverBase is not None
        assert AsyncWeaviateSecureRetriever is not None
        assert batch_search_async is not None
        assert multi_user_search_async is not None
        assert run_sync_retriever_async is not None

    def test_import_from_compat_module(self):
        """Test importing from ragguard.retrievers_async (compat)."""
        from ragguard import retrievers_async

        # Check __all__ contains expected exports
        expected = [
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

        for name in expected:
            assert name in retrievers_async.__all__
            assert hasattr(retrievers_async, name)

    def test_direct_submodule_imports(self):
        """Test direct imports from async retriever submodules."""
        from ragguard.retrievers_async.base import AsyncSecureRetrieverBase
        from ragguard.retrievers_async.chromadb import AsyncChromaDBSecureRetriever
        from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever
        from ragguard.retrievers_async.pgvector import AsyncPgvectorSecureRetriever
        from ragguard.retrievers_async.pinecone import AsyncPineconeSecureRetriever
        from ragguard.retrievers_async.qdrant import AsyncQdrantSecureRetriever
        from ragguard.retrievers_async.weaviate import AsyncWeaviateSecureRetriever

        assert AsyncSecureRetrieverBase is not None
        assert AsyncQdrantSecureRetriever is not None
        assert AsyncChromaDBSecureRetriever is not None
        assert AsyncPineconeSecureRetriever is not None
        assert AsyncFAISSSecureRetriever is not None
        assert AsyncWeaviateSecureRetriever is not None
        assert AsyncPgvectorSecureRetriever is not None

    def test_utils_imports(self):
        """Test importing utility functions."""
        from ragguard.retrievers_async.utils import (
            batch_search_async,
            multi_user_search_async,
            run_sync_retriever_async,
        )

        assert callable(batch_search_async)
        assert callable(multi_user_search_async)
        assert callable(run_sync_retriever_async)
