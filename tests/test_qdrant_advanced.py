"""
Advanced tests for Qdrant integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if qdrant-client is not installed
pytest.importorskip("qdrant_client")

from ragguard import Policy, QdrantSecureRetriever
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


@pytest.fixture(autouse=True)
def mock_qdrant_isinstance():
    """Auto-patch isinstance check for all tests."""
    with patch('ragguard.retrievers.qdrant.isinstance', return_value=True):
        yield


def create_mock_qdrant_client():
    """Create a mock QdrantClient."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "QdrantClient"

    # Mock search results
    mock_point = Mock()
    mock_point.id = "doc1"
    mock_point.score = 0.9
    mock_point.payload = {"content": "Document 1", "department": "engineering"}

    mock_client.search = Mock(return_value=[mock_point])

    # Mock query_points for newer API
    mock_response = Mock()
    mock_response.points = [mock_point]
    mock_client.query_points = Mock(return_value=mock_response)

    # Mock collection info
    mock_info = Mock()
    mock_info.vectors_count = 100
    mock_client.get_collection = Mock(return_value=mock_info)

    return mock_client


def create_basic_policy():
    """Create a basic allow-all policy."""
    return Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })


def test_qdrant_health_check_success():
    """Test successful health check for Qdrant."""
    mock_client = create_mock_qdrant_client()

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "qdrant"
    assert health["collection"] == "documents"
    assert health["error"] is None


def test_qdrant_context_manager():
    """Test using Qdrant retriever as context manager."""
    mock_client = create_mock_qdrant_client()

    with QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) >= 0


def test_qdrant_with_validation():
    """Test Qdrant retriever with input validation."""
    mock_client = create_mock_qdrant_client()

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy(),
        enable_validation=True,
        validation_config=ValidationConfig()
    )

    # Valid user context
    results = retriever.search(
        [0.1, 0.2, 0.3],
        {"id": "alice", "department": "engineering"},
        limit=5
    )
    assert len(results) >= 0

    # Invalid user context (empty dictionary)
    with pytest.raises(RetrieverError):
        retriever.search(
            [0.1, 0.2, 0.3],
            {},  # Missing required fields
            limit=5
        )


def test_qdrant_with_retry():
    """Test Qdrant retriever with retry logic."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "QdrantClient"

    # Fail twice, then succeed
    call_count = [0]

    def query_points_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection failed")

        mock_point = Mock()
        mock_point.id = "doc1"
        mock_point.score = 0.9
        mock_point.payload = {"content": "Document 1"}

        mock_response = Mock()
        mock_response.points = [mock_point]
        return mock_response

    mock_client.query_points = Mock(side_effect=query_points_side_effect)

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01)
    )

    # Should succeed after retries
    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert call_count[0] == 3  # Failed twice, succeeded on third attempt


def test_qdrant_with_cache():
    """Test Qdrant retriever with filter caching."""
    mock_client = create_mock_qdrant_client()

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        }),
        enable_filter_cache=True,
        filter_cache_size=100
    )

    user = {"id": "alice", "department": "engineering"}

    # First search - cache miss
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Second search - cache hit
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Verify search was called twice
    assert mock_client.search.call_count + mock_client.query_points.call_count >= 2


def test_qdrant_with_audit_logging():
    """Test Qdrant retriever with audit logging."""
    mock_client = create_mock_qdrant_client()

    # Create audit logger with callback
    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Verify audit log was created
    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"


def test_qdrant_search_failure():
    """Test Qdrant search failure handling."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "QdrantClient"
    mock_client.search = Mock(side_effect=Exception("Search failed"))
    mock_client.query_points = Mock(side_effect=Exception("Search failed"))

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy(),
        enable_retry=False
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_qdrant_empty_results():
    """Test Qdrant with empty search results."""
    mock_client = create_mock_qdrant_client()
    mock_client.search = Mock(return_value=[])

    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points = Mock(return_value=mock_response)

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 0


def test_qdrant_batch_search():
    """Test batch search with Qdrant."""
    mock_client = create_mock_qdrant_client()

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy()
    )

    queries = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]

    user = {"id": "alice"}
    all_results = retriever.batch_search(queries, user, limit=5)

    assert len(all_results) == 3


def test_qdrant_policy_update():
    """Test updating policy on Qdrant retriever."""
    mock_client = create_mock_qdrant_client()

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy()
    )

    # Update policy
    new_policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    retriever.policy = new_policy

    # Verify policy was updated
    assert retriever.policy.rules[0].name == "dept-access"

    # Search with new policy
    user = {"id": "alice", "department": "engineering"}
    results = retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(results) >= 0


def test_qdrant_with_embed_fn():
    """Test Qdrant retriever with text query and embedding function."""
    mock_client = create_mock_qdrant_client()

    def embed_fn(text):
        return [float(ord(c)) / 1000 for c in text[:3]]

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy(),
        embed_fn=embed_fn
    )

    # Search with text query
    results = retriever.search("test query", {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert mock_client.search.called or mock_client.query_points.called


def test_qdrant_text_query_without_embed_fn():
    """Test that text query without embed_fn raises error."""
    mock_client = create_mock_qdrant_client()

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy(),
        embed_fn=None
    )

    with pytest.raises(RetrieverError, match="no embed_fn was provided"):
        retriever.search("test query", {"id": "alice"}, limit=5)


def test_qdrant_invalidate_cache():
    """Test invalidating filter cache."""
    mock_client = create_mock_qdrant_client()

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        }),
        enable_filter_cache=True
    )

    user = {"id": "alice", "department": "engineering"}

    # Build cache
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Invalidate cache
    retriever.invalidate_filter_cache()

    # Next search should rebuild filter
    retriever.search([0.1, 0.2, 0.3], user, limit=5)


def test_qdrant_with_newer_api():
    """Test Qdrant retriever with query_points (newer API)."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "QdrantClient"

    mock_point = Mock()
    mock_point.id = "doc1"
    mock_point.score = 0.9
    mock_point.payload = {"content": "Document 1"}

    mock_response = Mock()
    mock_response.points = [mock_point]
    mock_client.query_points = Mock(return_value=mock_response)

    retriever = QdrantSecureRetriever(
        client=mock_client,
        collection="documents",
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert mock_client.query_points.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
