"""
Tests to improve coverage for ragguard/retrievers/milvus.py to 95%+.

Focuses on:
- Legacy Collection API paths
- _verify_collection edge cases
- _execute_search error handling
- _post_filter_results with different hit formats
- _check_backend_health variations
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


class TestMilvusSecureRetrieverInit:
    """Tests for MilvusSecureRetriever initialization."""

    def test_import_error_when_pymilvus_missing(self):
        """Test ImportError when pymilvus is not installed."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        # Mock pymilvus not being available
        with patch.dict('sys.modules', {'pymilvus': None}):
            with patch('ragguard.retrievers.milvus.MilvusClient', None):
                from ragguard.retrievers import milvus
                original_client = milvus.MilvusClient
                milvus.MilvusClient = None

                try:
                    with pytest.raises(ImportError) as exc:
                        milvus.MilvusSecureRetriever(
                            client=MagicMock(),
                            collection_name="test",
                            policy=policy
                        )
                    assert "pymilvus" in str(exc.value)
                finally:
                    milvus.MilvusClient = original_client

    def test_init_with_milvus_client_api(self):
        """Test initialization with MilvusClient API."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        client = MagicMock()
        client.describe_collection.return_value = {"name": "test_collection"}

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import MilvusSecureRetriever

            retriever = MilvusSecureRetriever(
                client=client,
                collection_name="test_collection",
                policy=policy
            )

            assert retriever.collection_name == "test_collection"
            assert retriever.vector_field == "vector"
            assert retriever.output_fields == ["*"]

    def test_init_custom_output_fields(self):
        """Test initialization with custom output fields."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import MilvusSecureRetriever

            retriever = MilvusSecureRetriever(
                client=client,
                collection_name="test",
                policy=policy,
                output_fields=["id", "text", "metadata"]
            )

            assert retriever.output_fields == ["id", "text", "metadata"]


class TestMilvusVerifyCollection:
    """Tests for _verify_collection method."""

    def test_verify_collection_not_found(self):
        """Test _verify_collection when collection doesn't exist."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        client = MagicMock()
        client.describe_collection.return_value = None

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import MilvusSecureRetriever

            # Should log warning but not raise
            retriever = MilvusSecureRetriever(
                client=client,
                collection_name="missing",
                policy=policy
            )

            assert retriever is not None

    def test_verify_collection_exception(self):
        """Test _verify_collection when exception occurs."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        client = MagicMock()
        client.describe_collection.side_effect = RuntimeError("Connection failed")

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import MilvusSecureRetriever

            # Should catch exception and log warning
            retriever = MilvusSecureRetriever(
                client=client,
                collection_name="test",
                policy=policy
            )

            assert retriever is not None


class TestMilvusExecuteSearch:
    """Tests for _execute_search method."""

    def create_retriever(self, client=None):
        """Helper to create a retriever for testing."""
        from ragguard import Policy
        from ragguard.retrievers.milvus import MilvusSecureRetriever

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        if client is None:
            client = MagicMock()
            client.describe_collection.return_value = {"name": "test"}

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            return MilvusSecureRetriever(
                client=client,
                collection_name="test",
                policy=policy
            )

    def test_execute_search_milvus_client_api(self):
        """Test _execute_search with MilvusClient API."""
        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.search.return_value = [[
            {"id": "doc1", "distance": 0.1, "text": "Hello"},
            {"id": "doc2", "distance": 0.2, "text": "World"}
        ]]

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            results = retriever._execute_search(
                query=[0.1, 0.2, 0.3],
                filter=None,
                limit=10,
                _user={"id": "alice"},
                _search_params={"metric_type": "L2", "params": {"nprobe": 10}}
            )

            assert len(results) == 2
            assert results[0]["id"] == "doc1"

    def test_execute_search_empty_results(self):
        """Test _execute_search with empty results."""
        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.search.return_value = []

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            results = retriever._execute_search(
                query=[0.1, 0.2],
                filter=None,
                limit=5,
                _user={"id": "alice"},
                _search_params={}
            )

            assert results == []

    def test_execute_search_connection_error(self):
        """Test _execute_search with connection error."""
        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.search.side_effect = ConnectionError("Network error")

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            with pytest.raises(ConnectionError):
                retriever._execute_search(
                    query=[0.1],
                    filter=None,
                    limit=10,
                    _user={},
                    _search_params={}
                )

    def test_execute_search_generic_error(self):
        """Test _execute_search with generic error."""
        from ragguard.exceptions import RetrieverError

        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.search.side_effect = ValueError("Invalid query")

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            with pytest.raises(RetrieverError) as exc:
                retriever._execute_search(
                    query=[0.1],
                    filter=None,
                    limit=10,
                    _user={},
                    _search_params={}
                )

            assert "Milvus search failed" in str(exc.value)


class TestMilvusPostFilterResults:
    """Tests for _post_filter_results method."""

    def create_retriever(self):
        """Helper to create a retriever."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import MilvusSecureRetriever
            return MilvusSecureRetriever(
                client=client,
                collection_name="test",
                policy=policy
            )

    def test_post_filter_dict_format(self):
        """Test _post_filter_results with dict format hits."""
        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever()

            hits = [
                {"id": "doc1", "distance": 0.1, "text": "Hello", "category": "greeting"},
                {"id": "doc2", "distance": 0.2, "text": "World", "category": "noun"}
            ]

            results = retriever._post_filter_results(hits, {"id": "alice"})

            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[0]["metadata"]["text"] == "Hello"
            assert results[0]["distance"] == 0.1
            assert results[0]["score"] > 0

    def test_post_filter_legacy_entity_format(self):
        """Test _post_filter_results with legacy entity format."""
        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever()

            # Mock legacy hit format
            hit = MagicMock()
            hit.entity = MagicMock()
            hit.entity.fields = {"text": "Legacy doc", "owner": "alice"}
            hit.entity.id = "legacy1"
            hit.id = "legacy1"
            hit.distance = 0.15

            results = retriever._post_filter_results([hit], {"id": "alice"})

            assert len(results) == 1
            assert results[0]["id"] == "legacy1"
            assert results[0]["metadata"]["text"] == "Legacy doc"

    def test_post_filter_fallback_format(self):
        """Test _post_filter_results with fallback attribute access."""
        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever()
            retriever.output_fields = ["text", "category"]

            # Mock object without entity or dict interface
            hit = MagicMock(spec=[])  # Empty spec to avoid hasattr returning True
            hit.id = "fallback1"
            hit.distance = 0.3
            hit.text = "Fallback text"
            hit.category = "test"

            # Need to remove entity attribute
            del hit.entity

            results = retriever._post_filter_results([hit], {"id": "alice"})

            assert len(results) == 1
            assert results[0]["id"] == "fallback1"


class TestMilvusCheckBackendHealth:
    """Tests for _check_backend_health method."""

    def create_retriever(self, client=None):
        """Helper to create a retriever."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        if client is None:
            client = MagicMock()
            client.describe_collection.return_value = {"name": "test"}

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import MilvusSecureRetriever
            return MilvusSecureRetriever(
                client=client,
                collection_name="test",
                policy=policy
            )

    def test_health_check_milvus_client_api(self):
        """Test health check with MilvusClient API."""
        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.get_collection_stats.return_value = {"row_count": 1000}
        client.list_collections.return_value = ["test"]

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)
            health = retriever._check_backend_health()

            assert health["collection_exists"] is True
            assert health["connection_alive"] is True
            assert health["collection_info"]["row_count"] == 1000

    def test_health_check_collection_not_found(self):
        """Test health check when collection not found."""
        from ragguard.exceptions import HealthCheckError

        client = MagicMock()
        client.describe_collection.return_value = None

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            with pytest.raises(HealthCheckError) as exc:
                retriever._check_backend_health()

            assert "not found" in str(exc.value)

    def test_health_check_stats_error(self):
        """Test health check when stats retrieval fails."""
        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.get_collection_stats.side_effect = RuntimeError("Stats error")
        client.list_collections.return_value = ["test"]

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)
            health = retriever._check_backend_health()

            assert health["collection_exists"] is True
            assert "stats_error" in health["collection_info"]

    def test_health_check_connection_error(self):
        """Test health check when connection fails."""
        from ragguard.exceptions import RetrieverConnectionError

        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.get_collection_stats.return_value = {"row_count": 100}
        client.list_collections.side_effect = ConnectionError("Connection lost")

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            with pytest.raises(RetrieverConnectionError):
                retriever._check_backend_health()


class TestMilvusSearch:
    """Tests for search method."""

    def create_retriever(self, client=None):
        """Helper to create a retriever."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        if client is None:
            client = MagicMock()
            client.describe_collection.return_value = {"name": "test"}
            client.search.return_value = [[
                {"id": "doc1", "distance": 0.1, "text": "Test"}
            ]]

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import MilvusSecureRetriever
            return MilvusSecureRetriever(
                client=client,
                collection_name="test",
                policy=policy
            )

    def test_search_with_metric_type(self):
        """Test search with custom metric type."""
        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.search.return_value = [[{"id": "doc1", "distance": 0.1}]]

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            results = retriever.search(
                query=[0.1, 0.2, 0.3],
                user={"id": "alice"},
                limit=5,
                metric_type="COSINE"
            )

            # User context is passed via kwargs now (no instance state to clean up)

    def test_search_with_search_params(self):
        """Test search with custom search params."""
        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}
        client.search.return_value = [[{"id": "doc1", "distance": 0.1}]]

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            retriever = self.create_retriever(client)

            results = retriever.search(
                query=[0.1, 0.2],
                user={"id": "alice"},
                limit=10,
                search_params={"nprobe": 20, "ef": 100}
            )

            # Verify search params were used
            call_kwargs = client.search.call_args.kwargs
            assert "search_params" in call_kwargs


class TestZillizSecureRetriever:
    """Tests for ZillizSecureRetriever alias."""

    def test_zilliz_is_alias(self):
        """Test that ZillizSecureRetriever is an alias for MilvusSecureRetriever."""
        from ragguard.retrievers.milvus import MilvusSecureRetriever, ZillizSecureRetriever

        assert issubclass(ZillizSecureRetriever, MilvusSecureRetriever)

    def test_zilliz_instantiation(self):
        """Test instantiating ZillizSecureRetriever."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        client = MagicMock()
        client.describe_collection.return_value = {"name": "test"}

        with patch('ragguard.retrievers.milvus.MilvusClient', MagicMock()):
            from ragguard.retrievers.milvus import ZillizSecureRetriever

            retriever = ZillizSecureRetriever(
                client=client,
                collection_name="zilliz_collection",
                policy=policy
            )

            assert retriever.backend_name == "milvus"
