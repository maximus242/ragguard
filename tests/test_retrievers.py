"""
Extended tests for retriever imports and base functionality.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestRetrieverImports:
    """Tests for retriever module imports."""

    def test_import_all_retrievers(self):
        """Test all retrievers can be imported."""
        from ragguard.retrievers import (
            BaseSecureRetriever,
            ChromaDBSecureRetriever,
            FAISSSecureRetriever,
            PgvectorSecureRetriever,
            PineconeSecureRetriever,
            QdrantSecureRetriever,
            WeaviateSecureRetriever,
        )

        assert BaseSecureRetriever is not None
        assert QdrantSecureRetriever is not None
        assert ChromaDBSecureRetriever is not None
        assert PineconeSecureRetriever is not None
        assert FAISSSecureRetriever is not None
        assert PgvectorSecureRetriever is not None
        assert WeaviateSecureRetriever is not None

    def test_retriever_module_all(self):
        """Test __all__ exports."""
        from ragguard import retrievers

        expected = [
            "BaseSecureRetriever",
            "QdrantSecureRetriever",
            "ChromaDBSecureRetriever",
        ]

        for name in expected:
            assert hasattr(retrievers, name)

    def test_import_specialized_retrievers(self):
        """Test specialized retrievers can be imported."""
        # These may fail if dependencies aren't installed, which is OK
        try:
            from ragguard.retrievers.arangodb import ArangoDBSecureRetriever
            assert ArangoDBSecureRetriever is not None
        except ImportError:
            pass

        try:
            from ragguard.retrievers.azure_search import AzureSearchSecureRetriever
            assert AzureSearchSecureRetriever is not None
        except ImportError:
            pass

        try:
            from ragguard.retrievers.elasticsearch import ElasticsearchSecureRetriever
            assert ElasticsearchSecureRetriever is not None
        except ImportError:
            pass

        try:
            from ragguard.retrievers.milvus import MilvusSecureRetriever
            assert MilvusSecureRetriever is not None
        except ImportError:
            pass

        try:
            from ragguard.retrievers.neo4j import Neo4jSecureRetriever
            assert Neo4jSecureRetriever is not None
        except ImportError:
            pass

        try:
            from ragguard.retrievers.neptune import NeptuneSecureRetriever
            assert NeptuneSecureRetriever is not None
        except ImportError:
            pass

        try:
            from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever
            assert TigerGraphSecureRetriever is not None
        except ImportError:
            pass
