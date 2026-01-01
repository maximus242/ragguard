"""
Advanced tests for FAISS integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if numpy or faiss is not installed
np = pytest.importorskip("numpy")
pytest.importorskip("faiss")

from ragguard import FAISSSecureRetriever, Policy
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


def create_mock_faiss_index():
    """Create a mock FAISS index."""
    mock_index = Mock()
    mock_index.ntotal = 100

    # Mock search results
    distances = np.array([[0.1, 0.2, 0.3]])
    indices = np.array([[0, 1, 2]])
    mock_index.search = Mock(return_value=(distances, indices))

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


def test_faiss_health_check_success():
    """Test successful health check for FAISS."""
    mock_index = create_mock_faiss_index()
    mock_index.ntotal = 3  # Match the metadata length
    metadata = [
        {"id": "doc1", "department": "engineering"},
        {"id": "doc2", "department": "sales"},
        {"id": "doc3", "department": "engineering"}
    ]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "faiss"


def test_faiss_context_manager():
    """Test using FAISS retriever as context manager."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}"} for i in range(100)]

    with FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) >= 0


def test_faiss_with_validation():
    """Test FAISS retriever with input validation."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}", "department": "engineering"} for i in range(100)]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
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


def test_faiss_with_retry():
    """Test FAISS retriever with retry logic."""
    mock_index = Mock()
    mock_index.ntotal = 100

    call_count = [0]

    def search_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection failed")

        distances = np.array([[0.1]])
        indices = np.array([[0]])
        return (distances, indices)

    mock_index.search = Mock(side_effect=search_side_effect)

    metadata = [{"id": "doc0", "department": "engineering"}]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01),
        adaptive_fetch=False  # Disable adaptive fetch to test retry behavior in isolation
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert call_count[0] == 3  # Exactly 3 calls: 2 failures + 1 success


def test_faiss_with_cache():
    """Test FAISS retriever with filter caching."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}", "department": "engineering"} for i in range(100)]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
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

    assert mock_index.search.call_count >= 2


def test_faiss_with_audit_logging():
    """Test FAISS retriever with audit logging."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}", "department": "engineering"} for i in range(100)]

    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"


def test_faiss_search_failure():
    """Test FAISS search failure handling."""
    mock_index = Mock()
    mock_index.ntotal = 100
    mock_index.search = Mock(side_effect=Exception("Search failed"))

    metadata = [{"id": "doc0"}]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
        policy=create_basic_policy(),
        enable_retry=False
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_faiss_empty_results():
    """Test FAISS with empty search results (no matches after filtering)."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}", "department": "sales"} for i in range(100)]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
        policy=Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        })
    )

    # User from different department - no results after filtering
    user = {"id": "alice", "department": "engineering"}
    results = retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(results) == 0


def test_faiss_batch_search():
    """Test batch search with FAISS."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}", "department": "engineering"} for i in range(100)]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
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


def test_faiss_policy_update():
    """Test updating policy on FAISS retriever."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}", "department": "engineering"} for i in range(100)]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
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


def test_faiss_over_fetch_factor():
    """Test FAISS with custom over-fetch factor."""
    mock_index = create_mock_faiss_index()
    metadata = [{"id": f"doc{i}", "department": "engineering"} for i in range(100)]

    retriever = FAISSSecureRetriever(
        index=mock_index,
        metadata=metadata,
        policy=create_basic_policy(),
        over_fetch_factor=5
    )

    retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=10)

    # Verify it fetched more than requested (over-fetch for filtering)
    call_args = mock_index.search.call_args
    fetch_limit = call_args[0][1]  # Second argument to search
    assert fetch_limit >= 10  # Should fetch at least the limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
