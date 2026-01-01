"""
Tests for the preview_filter method.

Tests the ability to preview what filter would be generated without executing a search.
"""

from unittest.mock import MagicMock, Mock

import pytest

# Skip all tests if chromadb is not installed
pytest.importorskip("chromadb", exc_type=ImportError)

from ragguard.policy import Policy
from ragguard.retrievers import ChromaDBSecureRetriever


@pytest.fixture
def sample_policy():
    """Create a sample policy for testing."""
    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "public-docs",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            }
        ],
        "default": "deny"
    })


@pytest.fixture
def deny_all_policy():
    """Create a policy that denies all access."""
    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-only",
                "allow": {
                    "roles": ["admin"]
                }
            }
        ],
        "default": "deny"
    })


@pytest.fixture
def mock_collection():
    """Create a mock ChromaDB collection."""
    mock = Mock()
    mock.query = Mock(return_value={
        "ids": [["doc1"]],
        "distances": [[0.1]],
        "metadatas": [[{"visibility": "public"}]],
        "documents": [["text"]]
    })
    return mock


class TestPreviewFilter:
    """Test the preview_filter method."""

    def test_preview_returns_filter(self, sample_policy, mock_collection):
        """Test that preview returns a filter."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy
        )

        user = {"id": "alice", "department": "engineering"}
        preview = retriever.preview_filter(user)

        assert "filter" in preview
        assert preview["backend"] == "chromadb"
        assert preview["user_id"] == "alice"
        assert preview["policy_version"] == "1"
        assert "cache_hit" in preview
        assert "would_deny_all" in preview

    def test_preview_with_debug_format(self, sample_policy, mock_collection):
        """Test preview with debug format includes extra info."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy
        )

        user = {"id": "alice", "department": "engineering"}
        preview = retriever.preview_filter(user, format="debug")

        assert "filter_str" in preview
        assert "user_context" in preview
        assert preview["user_context"] == user

    def test_preview_detects_deny_all(self, deny_all_policy, mock_collection):
        """Test that preview detects deny-all scenarios."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=deny_all_policy
        )

        # User without admin role should be denied
        user = {"id": "bob", "roles": ["user"]}
        preview = retriever.preview_filter(user)

        # Should generate some kind of deny filter
        assert preview["filter"] is not None

    def test_preview_shows_cache_hit(self, sample_policy, mock_collection):
        """Test that preview shows cache hit status."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            enable_filter_cache=True
        )

        user = {"id": "alice", "department": "engineering"}

        # First call - cache miss
        preview1 = retriever.preview_filter(user)
        assert preview1["cache_hit"] is False

        # Second call - cache hit
        preview2 = retriever.preview_filter(user)
        assert preview2["cache_hit"] is True

    def test_preview_does_not_execute_search(self, sample_policy, mock_collection):
        """Test that preview doesn't execute a search."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy
        )

        user = {"id": "alice", "department": "engineering"}
        retriever.preview_filter(user)

        # Query should not be called
        mock_collection.query.assert_not_called()

    def test_preview_thread_safe(self, sample_policy, mock_collection):
        """Test that preview is thread-safe during policy updates."""
        import threading
        import time

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy
        )

        results = []
        errors = []

        def preview_many_times():
            try:
                for _ in range(100):
                    preview = retriever.preview_filter({"id": "alice", "department": "eng"})
                    results.append(preview)
            except Exception as e:
                errors.append(e)

        def update_policy():
            try:
                for i in range(10):
                    new_policy = Policy.from_dict({
                        "version": "1",
                        "rules": [{"name": f"test_{i}", "allow": {"everyone": True}}],
                        "default": "deny"
                    })
                    retriever.policy = new_policy
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=preview_many_times),
            threading.Thread(target=preview_many_times),
            threading.Thread(target=update_policy)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0
        # Should have many successful results
        assert len(results) == 200


class TestPreviewFilterDenyAllDetection:
    """Test deny-all detection in preview_filter."""

    def test_detects_pgvector_deny_all(self, sample_policy, mock_collection):
        """Test detection of pgvector WHERE FALSE pattern."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy
        )

        # Test pgvector-style tuple
        assert retriever._is_deny_all_filter(("WHERE FALSE", [])) is True
        assert retriever._is_deny_all_filter(("WHERE status = %s", ["active"])) is False

    def test_detects_string_deny_all(self, sample_policy, mock_collection):
        """Test detection of string-based deny patterns."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy
        )

        assert retriever._is_deny_all_filter("1=0") is True
        assert retriever._is_deny_all_filter("FALSE") is True
        assert retriever._is_deny_all_filter("status = 'active'") is False

    def test_none_filter_not_deny_all(self, sample_policy, mock_collection):
        """Test that None filter is not deny-all."""
        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy
        )

        assert retriever._is_deny_all_filter(None) is False


class TestPreviewFilterWithCustomBuilder:
    """Test preview_filter with custom filter builder."""

    def test_preview_uses_custom_builder(self, sample_policy, mock_collection):
        """Test that preview uses custom filter builder when provided."""
        # Create custom filter builder
        custom_builder = Mock()
        custom_builder.build_filter = Mock(return_value={"custom": "filter"})

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=custom_builder
        )

        user = {"id": "alice"}
        preview = retriever.preview_filter(user)

        assert preview["filter"] == {"custom": "filter"}
        assert preview["cache_hit"] is False  # Custom builders don't use cache
        custom_builder.build_filter.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
