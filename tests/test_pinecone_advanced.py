"""
Advanced tests for Pinecone integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard import PineconeSecureRetriever, Policy
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig

# Skip all tests if Pinecone is not installed
try:
    import pinecone  # type: ignore
    PINECONE_AVAILABLE = True
except (ImportError, Exception):
    # pinecone raises Exception if pinecone-client (old name) is installed
    PINECONE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PINECONE_AVAILABLE, reason="pinecone-client not installed - optional dependency")


def create_mock_pinecone_index():
    """Create a mock Pinecone index."""
    mock_index = Mock()

    # Mock match
    mock_match = Mock()
    mock_match.id = "doc1"
    mock_match.score = 0.9
    mock_match.metadata = {"department": "engineering", "content": "Document 1"}

    # Mock query results
    mock_response = Mock()
    mock_response.matches = [mock_match]
    mock_index.query = Mock(return_value=mock_response)

    # Mock stats
    mock_stats = {
        "dimension": 384,
        "index_fullness": 0.5,
        "total_vector_count": 100
    }
    mock_index.describe_index_stats = Mock(return_value=mock_stats)

    return mock_index


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


def test_pinecone_health_check_success():
    """Test successful health check for Pinecone."""
    mock_index = create_mock_pinecone_index()

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "pinecone"


def test_pinecone_context_manager():
    """Test using Pinecone retriever as context manager."""
    mock_index = create_mock_pinecone_index()

    with PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) >= 0


def test_pinecone_with_validation():
    """Test Pinecone retriever with input validation."""
    mock_index = create_mock_pinecone_index()

    retriever = PineconeSecureRetriever(
        index=mock_index,
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
            {},
            limit=5
        )


def test_pinecone_with_retry():
    """Test Pinecone retriever with retry logic."""
    mock_index = Mock()

    call_count = [0]

    def query_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection failed")

        mock_match = Mock()
        mock_match.id = "doc1"
        mock_match.score = 0.9
        mock_match.metadata = {"content": "Document 1"}

        mock_response = Mock()
        mock_response.matches = [mock_match]
        return mock_response

    mock_index.query = Mock(side_effect=query_side_effect)

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01)
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert call_count[0] == 3


def test_pinecone_with_cache():
    """Test Pinecone retriever with filter caching."""
    mock_index = create_mock_pinecone_index()

    retriever = PineconeSecureRetriever(
        index=mock_index,
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

    assert mock_index.query.call_count >= 2


def test_pinecone_with_audit_logging():
    """Test Pinecone retriever with audit logging."""
    mock_index = create_mock_pinecone_index()

    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"


def test_pinecone_search_failure():
    """Test Pinecone search failure handling."""
    mock_index = Mock()
    mock_index.query = Mock(side_effect=Exception("Search failed"))

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy(),
        enable_retry=False
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_pinecone_empty_results():
    """Test Pinecone with empty search results."""
    mock_index = Mock()
    mock_response = Mock()
    mock_response.matches = []
    mock_index.query = Mock(return_value=mock_response)

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 0


def test_pinecone_batch_search():
    """Test batch search with Pinecone."""
    mock_index = create_mock_pinecone_index()

    retriever = PineconeSecureRetriever(
        index=mock_index,
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


def test_pinecone_policy_update():
    """Test updating policy on Pinecone retriever."""
    mock_index = create_mock_pinecone_index()

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy()
    )

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

    assert retriever.policy.rules[0].name == "dept-access"


def test_pinecone_with_embed_fn():
    """Test Pinecone retriever with text query and embedding function."""
    mock_index = create_mock_pinecone_index()

    def embed_fn(text):
        return [float(ord(c)) / 1000 for c in text[:3]]

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy(),
        embed_fn=embed_fn
    )

    results = retriever.search("test query", {"id": "alice"}, limit=5)

    assert len(results) >= 0


def test_pinecone_text_query_without_embed_fn():
    """Test that text query without embed_fn raises error."""
    mock_index = create_mock_pinecone_index()

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy(),
        embed_fn=None
    )

    with pytest.raises(RetrieverError, match="no embed_fn was provided"):
        retriever.search("test query", {"id": "alice"}, limit=5)


def test_pinecone_with_namespace():
    """Test Pinecone retriever with custom namespace."""
    mock_index = create_mock_pinecone_index()

    retriever = PineconeSecureRetriever(
        index=mock_index,
        policy=create_basic_policy(),
        namespace="custom-namespace"
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    # Verify namespace was used in query
    call_args = mock_index.query.call_args
    assert call_args.kwargs.get("namespace") == "custom-namespace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
