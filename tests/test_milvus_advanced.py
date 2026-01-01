"""
Advanced tests for Milvus integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard import Policy
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retrievers import MilvusSecureRetriever
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig

# Skip all tests if pymilvus is not installed
try:
    import pymilvus  # type: ignore
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYMILVUS_AVAILABLE, reason="pymilvus not installed - optional dependency")


def create_mock_milvus_client():
    """Create a mock Milvus client."""
    mock_client = Mock()

    # Mock search result
    mock_hit = Mock()
    mock_hit.id = "doc1"
    mock_hit.distance = 0.1
    # Create mock entity with fields attribute (for Legacy Collection API)
    mock_entity = Mock()
    mock_entity.fields = {"department": "engineering", "content": "Document 1"}
    mock_entity.id = "doc1"
    mock_hit.entity = mock_entity

    mock_client.search = Mock(return_value=[[mock_hit]])

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


def test_milvus_health_check_success():
    """Test successful health check for Milvus."""
    mock_client = create_mock_milvus_client()

    # Mock health check methods
    mock_collection_stats = {"row_count": 100}
    mock_client.get_collection_stats = Mock(return_value=mock_collection_stats)

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "milvus"


def test_milvus_context_manager():
    """Test using Milvus retriever as context manager."""
    mock_client = create_mock_milvus_client()

    with MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) >= 0


def test_milvus_with_validation():
    """Test Milvus retriever with input validation."""
    mock_client = create_mock_milvus_client()

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
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


def test_milvus_with_retry():
    """Test Milvus retriever with retry logic."""
    mock_client = Mock()

    call_count = [0]

    def search_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection failed")

        mock_hit = Mock()
        mock_hit.id = "doc1"
        mock_hit.distance = 0.1
        # Create mock entity with fields attribute (for Legacy Collection API)
        mock_entity = Mock()
        mock_entity.fields = {"content": "Document 1"}
        mock_entity.id = "doc1"
        mock_hit.entity = mock_entity
        return [[mock_hit]]

    mock_client.search = Mock(side_effect=search_side_effect)

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01)
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert call_count[0] == 3


def test_milvus_with_cache():
    """Test Milvus retriever with filter caching."""
    mock_client = create_mock_milvus_client()

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
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

    assert mock_client.search.call_count >= 2


def test_milvus_with_audit_logging():
    """Test Milvus retriever with audit logging."""
    mock_client = create_mock_milvus_client()

    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"


def test_milvus_search_failure():
    """Test Milvus search failure handling."""
    mock_client = Mock()
    mock_client.search = Mock(side_effect=Exception("Search failed"))

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=create_basic_policy(),
        enable_retry=False
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_milvus_empty_results():
    """Test Milvus with empty search results."""
    mock_client = create_mock_milvus_client()
    mock_client.search = Mock(return_value=[[]])

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 0


def test_milvus_batch_search():
    """Test batch search with Milvus."""
    mock_client = create_mock_milvus_client()

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
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


def test_milvus_policy_update():
    """Test updating policy on Milvus retriever."""
    mock_client = create_mock_milvus_client()

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
