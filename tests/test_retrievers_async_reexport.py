"""
Tests for retrievers_async re-export module.
"""

import pytest


class TestRetrieversAsyncReexport:
    """Tests for retrievers_async.py module."""

    def test_import_from_reexport(self):
        """Test importing from retrievers_async.py."""
        from ragguard import retrievers_async

        # Should have all expected exports
        assert hasattr(retrievers_async, '__all__')

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
