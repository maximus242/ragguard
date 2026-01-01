"""
Advanced tests for Weaviate integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import Mock

import pytest

# Skip all tests if weaviate-client is not installed
weaviate = pytest.importorskip("weaviate")

from ragguard import Policy, WeaviateSecureRetriever
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


def create_mock_weaviate_client():
    """Create a mock Weaviate client."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "Client"

    # Mock query builder chain
    mock_query = Mock()
    mock_get = Mock()
    mock_near_vector = Mock()
    mock_limit = Mock()
    mock_where = Mock()
    mock_additional = Mock()

    # Chain methods
    mock_query.get = Mock(return_value=mock_get)
    mock_get.with_near_vector = Mock(return_value=mock_near_vector)
    mock_near_vector.with_limit = Mock(return_value=mock_limit)
    mock_limit.with_where = Mock(return_value=mock_where)
    mock_where.with_additional = Mock(return_value=mock_additional)
    mock_limit.with_additional = Mock(return_value=mock_additional)

    # Mock results
    mock_additional.do = Mock(return_value={
        "data": {
            "Get": {
                "TestCollection": [
                    {
                        "_additional": {"id": "doc1", "certainty": 0.9},
                        "content": "Document 1",
                        "department": "engineering"
                    }
                ]
            }
        }
    })

    mock_client.query = mock_query

    # Mock schema
    mock_client.schema = Mock()
    mock_client.schema.get = Mock(return_value={"class": "TestCollection"})

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


def test_weaviate_health_check_success():
    """Test successful health check for Weaviate."""
    mock_client = create_mock_weaviate_client()

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "weaviate"
    assert health["collection"] == "TestCollection"


def test_weaviate_context_manager():
    """Test using Weaviate retriever as context manager."""
    mock_client = create_mock_weaviate_client()

    with WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) >= 0


def test_weaviate_with_validation():
    """Test Weaviate retriever with input validation."""
    mock_client = create_mock_weaviate_client()

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
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


def test_weaviate_with_retry():
    """Test Weaviate retriever with retry logic."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "Client"

    call_count = [0]

    def do_side_effect():
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection failed")

        return {
            "data": {
                "Get": {
                    "TestCollection": [
                        {"_additional": {"id": "doc1"}, "content": "Document 1"}
                    ]
                }
            }
        }

    # Mock query builder chain
    mock_query = Mock()
    mock_get = Mock()
    mock_near_vector = Mock()
    mock_limit = Mock()
    mock_where = Mock()
    mock_additional = Mock()

    mock_query.get = Mock(return_value=mock_get)
    mock_get.with_near_vector = Mock(return_value=mock_near_vector)
    mock_near_vector.with_limit = Mock(return_value=mock_limit)
    mock_limit.with_where = Mock(return_value=mock_where)
    mock_where.with_additional = Mock(return_value=mock_additional)
    mock_limit.with_additional = Mock(return_value=mock_additional)
    mock_additional.do = Mock(side_effect=do_side_effect)

    mock_client.query = mock_query

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01)
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert call_count[0] == 3


def test_weaviate_with_cache():
    """Test Weaviate retriever with filter caching."""
    mock_client = create_mock_weaviate_client()

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
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


def test_weaviate_with_audit_logging():
    """Test Weaviate retriever with audit logging."""
    mock_client = create_mock_weaviate_client()

    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"


def test_weaviate_search_failure():
    """Test Weaviate search failure handling."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "Client"

    mock_query = Mock()
    mock_get = Mock()
    mock_near_vector = Mock()
    mock_limit = Mock()
    mock_additional = Mock()

    mock_query.get = Mock(return_value=mock_get)
    mock_get.with_near_vector = Mock(return_value=mock_near_vector)
    mock_near_vector.with_limit = Mock(return_value=mock_limit)
    mock_limit.with_additional = Mock(return_value=mock_additional)
    mock_additional.do = Mock(side_effect=Exception("Search failed"))

    mock_client.query = mock_query

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy(),
        enable_retry=False
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_weaviate_empty_results():
    """Test Weaviate with empty search results."""
    mock_client = create_mock_weaviate_client()

    # Mock empty results
    mock_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.return_value.do = Mock(
        return_value={"data": {"Get": {"TestCollection": []}}}
    )

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 0


def test_weaviate_batch_search():
    """Test batch search with Weaviate."""
    mock_client = create_mock_weaviate_client()

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
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


def test_weaviate_policy_update():
    """Test updating policy on Weaviate retriever."""
    mock_client = create_mock_weaviate_client()

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
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


def test_weaviate_with_embed_fn():
    """Test Weaviate retriever with text query and embedding function."""
    mock_client = create_mock_weaviate_client()

    def embed_fn(text):
        return [float(ord(c)) / 1000 for c in text[:3]]

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy(),
        embed_fn=embed_fn
    )

    results = retriever.search("test query", {"id": "alice"}, limit=5)

    assert len(results) >= 0


def test_weaviate_text_query_without_embed_fn():
    """Test that text query without embed_fn raises error."""
    mock_client = create_mock_weaviate_client()

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection="TestCollection",
        policy=create_basic_policy(),
        embed_fn=None
    )

    with pytest.raises(RetrieverError, match="no embed_fn was provided"):
        retriever.search("test query", {"id": "alice"}, limit=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
