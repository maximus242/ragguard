"""
Additional tests for retrievers/base.py to maximize coverage.
"""

from unittest.mock import MagicMock, patch

import pytest


def create_test_retriever():
    """Create a test retriever for testing."""
    from ragguard.policy.models import Policy
    from ragguard.retrievers.base import BaseSecureRetriever

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "allow"
    })

    # Create a concrete implementation for testing
    class TestRetriever(BaseSecureRetriever):
        @property
        def backend_name(self):
            return "chromadb"  # Use a valid backend

        def _execute_search(self, query, filter, limit, **kwargs):
            return [{"id": "doc1", "score": 0.9}]

        def _check_backend_health(self):
            return {"status": "healthy"}

    retriever = TestRetriever(
        policy=policy,
        client=MagicMock(),
        collection="test_collection",
        embed_fn=lambda x: [0.1] * 128
    )

    return retriever


class TestBaseSecureRetrieverBatchSearch:
    """Tests for batch_search method edge cases."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever for testing."""
        return create_test_retriever()

    def test_batch_search_empty_queries(self, mock_retriever):
        """Test batch_search with empty queries list."""
        user = {"id": "alice"}
        results = mock_retriever.batch_search([], user, limit=10)

        assert results == []

    def test_batch_search_with_embeddings(self, mock_retriever):
        """Test batch_search with pre-computed embeddings."""
        user = {"id": "alice"}
        queries = [[0.1] * 128, [0.2] * 128]

        results = mock_retriever.batch_search(queries, user, limit=10)

        assert len(results) == 2

    def test_batch_search_with_text_queries(self, mock_retriever):
        """Test batch_search with text queries (uses embed_fn)."""
        user = {"id": "alice"}
        queries = ["query 1", "query 2"]

        results = mock_retriever.batch_search(queries, user, limit=10)

        assert len(results) == 2

    def test_batch_search_text_without_embed_fn(self):
        """Test batch_search raises error for text query without embed_fn."""
        from ragguard.exceptions import RetrieverError
        from ragguard.policy.models import Policy
        from ragguard.retrievers.base import BaseSecureRetriever

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "allow"
        })

        class TestRetriever(BaseSecureRetriever):
            @property
            def backend_name(self):
                return "chromadb"  # Use a valid backend

            def _execute_search(self, query, filter, limit, **kwargs):
                return []

            def _check_backend_health(self):
                return {}

        retriever = TestRetriever(
            policy=policy,
            client=MagicMock(),
            collection="test_collection",
            embed_fn=None  # No embed function
        )

        user = {"id": "alice"}
        queries = ["text query"]

        with pytest.raises(RetrieverError, match="embed_fn"):
            retriever.batch_search(queries, user, limit=10)

    def test_batch_search_validation_error(self):
        """Test batch_search with validation error."""
        retriever = create_test_retriever()
        retriever._enable_validation = True

        # Mock the validator to raise an error
        retriever._validator = MagicMock()
        retriever._validator.validate_user_context.side_effect = ValueError("Invalid user")

        user = {"id": "alice"}
        queries = ["query 1"]

        with pytest.raises(ValueError, match="Invalid user"):
            retriever.batch_search(queries, user, limit=10)

    def test_batch_search_filter_build_error(self):
        """Test batch_search with filter build error."""
        from ragguard.exceptions import RetrieverError

        retriever = create_test_retriever()

        # Mock the policy engine to raise an error
        retriever.policy_engine.to_filter = MagicMock(side_effect=Exception("Filter error"))

        user = {"id": "alice"}
        queries = [[0.1] * 128]

        with pytest.raises(RetrieverError, match="Failed to build permission filter"):
            retriever.batch_search(queries, user, limit=10)

    def test_batch_search_execute_error(self):
        """Test batch_search with execute search error."""
        from ragguard.policy.models import Policy
        from ragguard.retrievers.base import BaseSecureRetriever

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "allow"
        })

        class TestRetriever(BaseSecureRetriever):
            @property
            def backend_name(self):
                return "chromadb"  # Use a valid backend

            def _execute_search(self, query, filter, limit, **kwargs):
                raise Exception("Search failed")

            def _check_backend_health(self):
                return {}

        retriever = TestRetriever(
            policy=policy,
            client=MagicMock(),
            collection="test_collection",
            embed_fn=lambda x: [0.1] * 128
        )

        user = {"id": "alice"}
        queries = [[0.1] * 128]

        with pytest.raises(Exception, match="Search failed"):
            retriever.batch_search(queries, user, limit=10)


class TestCustomFilterValidation:
    """Tests for _validate_custom_filter_result method."""

    @pytest.fixture
    def pgvector_retriever(self):
        """Create a pgvector test retriever."""
        from ragguard.policy.models import Policy
        from ragguard.retrievers.base import BaseSecureRetriever

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "allow"
        })

        class TestRetriever(BaseSecureRetriever):
            @property
            def backend_name(self):
                return "pgvector"

            def _execute_search(self, query, filter, limit, **kwargs):
                return []

            def _check_backend_health(self):
                return {}

        return TestRetriever(
            policy=policy,
            client=MagicMock(),
            collection="test_collection",
            embed_fn=lambda x: [0.1] * 128
        )

    def test_validate_empty_dict(self, pgvector_retriever):
        """Test validation with empty dict filter."""
        pgvector_retriever._validate_custom_filter_result({}, {"id": "alice"}, "TestBuilder")
        # Should log warning but not raise

    def test_validate_empty_list(self, pgvector_retriever):
        """Test validation with empty list filter."""
        pgvector_retriever._validate_custom_filter_result([], {"id": "alice"}, "TestBuilder")
        # Should log warning but not raise

    def test_validate_pgvector_wrong_type(self, pgvector_retriever):
        """Test validation for pgvector with wrong type."""
        pgvector_retriever._validate_custom_filter_result(
            "wrong type",
            {"id": "alice"},
            "TestBuilder"
        )
        # Should log warning but not raise

    def test_validate_pgvector_invalid_tuple_length(self, pgvector_retriever):
        """Test validation for pgvector with invalid tuple length."""
        pgvector_retriever._validate_custom_filter_result(
            ("clause",),  # Only 1 element, should be 2
            {"id": "alice"},
            "TestBuilder"
        )
        # Should log warning but not raise

    def test_validate_pgvector_invalid_clause_type(self, pgvector_retriever):
        """Test validation for pgvector with invalid clause type."""
        pgvector_retriever._validate_custom_filter_result(
            (123, []),  # First element should be string
            {"id": "alice"},
            "TestBuilder"
        )
        # Should log warning but not raise

    def test_validate_pgvector_valid_tuple(self, pgvector_retriever):
        """Test validation for pgvector with valid tuple."""
        pgvector_retriever._validate_custom_filter_result(
            ("WHERE x = $1", [1]),
            {"id": "alice"},
            "TestBuilder"
        )
        # Should not log warning


class TestChromaDBFilterValidation:
    """Tests for _validate_custom_filter_result with chromadb."""

    @pytest.fixture
    def chromadb_retriever(self):
        """Create a chromadb test retriever."""
        from ragguard.policy.models import Policy
        from ragguard.retrievers.base import BaseSecureRetriever

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "allow"
        })

        class TestRetriever(BaseSecureRetriever):
            @property
            def backend_name(self):
                return "chromadb"

            def _execute_search(self, query, filter, limit, **kwargs):
                return []

            def _check_backend_health(self):
                return {}

        return TestRetriever(
            policy=policy,
            client=MagicMock(),
            collection="test_collection",
            embed_fn=lambda x: [0.1] * 128
        )

    def test_validate_chromadb_wrong_type(self, chromadb_retriever):
        """Test validation for chromadb with wrong type."""
        chromadb_retriever._validate_custom_filter_result(
            "wrong type",
            {"id": "alice"},
            "TestBuilder"
        )
        # Should log warning but not raise

    def test_validate_chromadb_valid_dict(self, chromadb_retriever):
        """Test validation for chromadb with valid dict."""
        chromadb_retriever._validate_custom_filter_result(
            {"$and": [{"field": "value"}]},
            {"id": "alice"},
            "TestBuilder"
        )
        # Should not log warning


class TestRetrieverCacheOperations:
    """Tests for cache-related operations."""

    def test_invalidate_filter_cache(self):
        """Test invalidate_filter_cache method."""
        retriever = create_test_retriever()

        # Should not raise
        retriever.invalidate_filter_cache()

    def test_get_cache_stats(self):
        """Test get_cache_stats method."""
        retriever = create_test_retriever()

        stats = retriever.get_cache_stats()
        assert stats is None or isinstance(stats, dict)


class TestRetrieverWithMetrics:
    """Tests for metrics integration."""

    def test_search_records_metrics(self):
        """Test that search records metrics when available."""
        retriever = create_test_retriever()

        # Mock the metrics collector
        mock_collector = MagicMock()
        mock_collector.is_enabled.return_value = True

        with patch('ragguard.retrievers.base._metrics_available', True):
            with patch('ragguard.retrievers.base.get_metrics_collector', return_value=mock_collector):
                user = {"id": "alice"}
                results = retriever.search([0.1] * 128, user, limit=10)

                assert len(results) == 1


class TestSearchValidation:
    """Tests for search validation errors."""

    def test_search_validation_error(self):
        """Test search with validation error."""
        retriever = create_test_retriever()
        retriever._enable_validation = True

        # Mock the validator to raise an error
        retriever._validator = MagicMock()
        retriever._validator.validate_user_context.side_effect = ValueError("Bad user")

        user = {"id": "alice"}

        with pytest.raises(ValueError, match="Bad user"):
            retriever.search([0.1] * 128, user, limit=10)


class TestRetrieverHealthCheck:
    """Tests for health check method."""

    def test_health_check(self):
        """Test health_check method."""
        retriever = create_test_retriever()

        health = retriever.health_check()

        assert isinstance(health, dict)
        assert "backend" in health
        assert "healthy" in health


class TestRetrieverPolicyProperty:
    """Tests for policy property."""

    def test_policy_setter(self):
        """Test setting policy property."""
        from ragguard.policy.models import Policy

        retriever = create_test_retriever()

        new_policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "new_rule", "allow": {"everyone": True}}],
            "default": "deny"
        })

        retriever.policy = new_policy

        assert retriever.policy == new_policy
