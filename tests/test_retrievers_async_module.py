"""
Tests for retrievers_async module re-exports.
"""

import pytest


class TestRetrieversAsyncTopLevelModule:
    """Tests for the top-level ragguard.retrievers_async module (re-export)."""

    def test_import_top_level_retrievers_async(self):
        """Test importing the top-level retrievers_async module directly."""
        # This covers ragguard/retrievers_async.py (the 2-line re-export file)
        import ragguard.retrievers_async as top_level_module
        assert top_level_module is not None
        assert hasattr(top_level_module, '__all__')
        assert 'AsyncQdrantSecureRetriever' in top_level_module.__all__

    def test_all_top_level_exports_accessible(self):
        """Test all exports from top-level module are accessible."""
        import ragguard.retrievers_async as top_level_module

        for name in top_level_module.__all__:
            assert hasattr(top_level_module, name), f"Missing export: {name}"


class TestRetrieversAsyncModuleExports:
    """Tests for retrievers_async module exports."""

    def test_import_module(self):
        """Test module imports successfully."""
        from ragguard import retrievers_async
        assert retrievers_async is not None

    def test_import_top_level_module(self):
        """Test top-level retrievers_async module."""
        # Import the top-level module directly to cover its imports
        import ragguard.retrievers_async as compat_module
        assert compat_module is not None

        # Verify exports via the compatibility module
        assert hasattr(compat_module, 'AsyncQdrantSecureRetriever')
        assert hasattr(compat_module, 'AsyncChromaDBSecureRetriever')
        assert hasattr(compat_module, 'batch_search_async')

    def test_async_retriever_base_export(self):
        """Test AsyncSecureRetrieverBase is exported."""
        from ragguard.retrievers_async import AsyncSecureRetrieverBase
        assert AsyncSecureRetrieverBase is not None

    def test_all_async_retrievers_exported(self):
        """Test all async retrievers are exported."""
        from ragguard import retrievers_async

        expected_exports = [
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

        for name in expected_exports:
            assert hasattr(retrievers_async, name), f"{name} not exported"

    def test_all_exports_list(self):
        """Test __all__ contains expected exports."""
        from ragguard import retrievers_async

        assert "__all__" in dir(retrievers_async)
        assert "AsyncSecureRetrieverBase" in retrievers_async.__all__
        assert "AsyncQdrantSecureRetriever" in retrievers_async.__all__

    def test_direct_import_from_package(self):
        """Test direct import from retrievers_async package."""
        from ragguard.retrievers_async.qdrant import AsyncQdrantSecureRetriever
        assert AsyncQdrantSecureRetriever is not None

    def test_batch_search_function_export(self):
        """Test batch_search_async is exported and callable."""
        from ragguard.retrievers_async import batch_search_async
        assert callable(batch_search_async)

    def test_multi_user_search_export(self):
        """Test multi_user_search_async is exported."""
        from ragguard.retrievers_async import multi_user_search_async
        assert callable(multi_user_search_async)

    def test_run_sync_retriever_export(self):
        """Test run_sync_retriever_async is exported."""
        from ragguard.retrievers_async import run_sync_retriever_async
        assert callable(run_sync_retriever_async)
