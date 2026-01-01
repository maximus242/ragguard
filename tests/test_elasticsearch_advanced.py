"""
Advanced tests for Elasticsearch/OpenSearch integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard import ElasticsearchSecureRetriever, OpenSearchSecureRetriever, Policy
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


def create_mock_es_client():
    """Create a mock Elasticsearch client."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "Elasticsearch"
    mock_client.search = Mock(return_value={
        "hits": {
            "hits": [
                {
                    "_id": "doc1",
                    "_score": 0.9,
                    "_source": {
                        "text": "Document 1",
                        "department": "engineering"
                    }
                }
            ]
        }
    })
    mock_client.ping = Mock(return_value=True)
    mock_client.indices = Mock()
    mock_client.indices.exists = Mock(return_value=True)
    mock_client.count = Mock(return_value={"count": 100})
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


def test_elasticsearch_health_check_success():
    """Test successful health check for Elasticsearch."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "elasticsearch"
    assert health["collection"] == "documents"
    assert health["details"]["connection_alive"] is True
    assert health["details"]["index_exists"] is True
    assert health["details"]["index_info"]["document_count"] == 100
    assert health["error"] is None


def test_elasticsearch_health_check_connection_failed():
    """Test health check when connection fails."""
    mock_client = create_mock_es_client()
    mock_client.ping = Mock(return_value=False)

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is False
    assert "Ping failed" in health["error"]


def test_elasticsearch_health_check_index_missing():
    """Test health check when index doesn't exist."""
    mock_client = create_mock_es_client()
    mock_client.indices.exists = Mock(return_value=False)

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="missing_index",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is False
    assert "does not exist" in health["error"]


def test_elasticsearch_context_manager():
    """Test using Elasticsearch retriever as context manager."""
    mock_client = create_mock_es_client()

    with ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) == 1


def test_elasticsearch_with_validation():
    """Test Elasticsearch retriever with input validation."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
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
    assert len(results) == 1

    # Invalid user context (empty dictionary)
    with pytest.raises(RetrieverError):  # Validation should fail
        retriever.search(
            [0.1, 0.2, 0.3],
            {},  # Missing required fields
            limit=5
        )


def test_elasticsearch_with_retry():
    """Test Elasticsearch retriever with retry logic."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "Elasticsearch"

    # Fail twice, then succeed
    call_count = [0]

    def search_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            # Raise a retryable exception (not wrapped in RetrieverError)
            raise OSError("Connection failed")
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 0.9,
                        "_source": {"text": "Document 1"}
                    }
                ]
            }
        }

    mock_client.search = Mock(side_effect=search_side_effect)

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01)
    )

    # Should succeed after retries
    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 1
    assert call_count[0] == 3  # Failed twice, succeeded on third attempt


def test_elasticsearch_with_cache():
    """Test Elasticsearch retriever with filter caching."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
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

    # Second search - cache hit (same user context)
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Verify search was called twice (cache doesn't affect execution, just filter building)
    assert mock_client.search.call_count == 2


def test_elasticsearch_with_audit_logging():
    """Test Elasticsearch retriever with audit logging."""
    mock_client = create_mock_es_client()

    # Create audit logger with callback
    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Verify audit log was created
    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"
    assert audit_entries[0]["results_returned"] == 1


def test_elasticsearch_search_failure():
    """Test Elasticsearch search failure handling."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "Elasticsearch"
    mock_client.search = Mock(side_effect=Exception("Search failed"))

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy(),
        enable_retry=False  # Disable retry for immediate failure
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_elasticsearch_empty_results():
    """Test Elasticsearch with empty search results."""
    mock_client = create_mock_es_client()
    mock_client.search = Mock(return_value={
        "hits": {"hits": []}
    })

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 0


def test_elasticsearch_with_custom_num_candidates():
    """Test Elasticsearch search with custom num_candidates parameter."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    )

    retriever.search(
        [0.1, 0.2, 0.3],
        {"id": "alice"},
        limit=5,
        num_candidates=100
    )

    call_args = mock_client.search.call_args
    assert call_args.kwargs["knn"]["num_candidates"] == 100


def test_elasticsearch_with_source_fields():
    """Test Elasticsearch search with custom source fields."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    )

    retriever.search(
        [0.1, 0.2, 0.3],
        {"id": "alice"},
        limit=5,
        _source=["title", "content"]
    )

    call_args = mock_client.search.call_args
    assert call_args.kwargs["_source"] == ["title", "content"]


def test_opensearch_health_check():
    """Test health check for OpenSearch retriever."""
    mock_client = create_mock_es_client()
    mock_client.__class__.__name__ = "OpenSearch"

    retriever = OpenSearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "elasticsearch"  # Uses same backend name


def test_elasticsearch_batch_search():
    """Test batch search with Elasticsearch."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
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
    assert mock_client.search.call_count == 3


def test_elasticsearch_policy_update():
    """Test updating policy on Elasticsearch retriever."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy()
    )

    # Update policy
    new_policy = Policy.from_dict({
        "version": "1",  # Version must be "1"
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    retriever.policy = new_policy

    # Verify policy was updated (check rule name instead of version)
    assert retriever.policy.rules[0].name == "dept-access"

    # Search with new policy
    user = {"id": "alice", "department": "engineering"}
    results = retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Verify filter was applied based on new policy
    call_args = mock_client.search.call_args
    assert "filter" in call_args.kwargs["knn"]


def test_elasticsearch_invalidate_cache():
    """Test invalidating filter cache."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
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

    # Next search should rebuild filter (can't easily verify, but ensures no error)
    retriever.search([0.1, 0.2, 0.3], user, limit=5)


def test_elasticsearch_with_embed_fn():
    """Test Elasticsearch retriever with text query and embedding function."""
    mock_client = create_mock_es_client()

    def embed_fn(text):
        return [float(ord(c)) / 1000 for c in text[:3]]

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy(),
        embed_fn=embed_fn
    )

    # Search with text query
    results = retriever.search("test query", {"id": "alice"}, limit=5)

    assert len(results) == 1
    assert mock_client.search.called


def test_elasticsearch_text_query_without_embed_fn():
    """Test that text query without embed_fn raises error."""
    mock_client = create_mock_es_client()

    retriever = ElasticsearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=create_basic_policy(),
        embed_fn=None  # No embedding function
    )

    with pytest.raises(RetrieverError, match="no embed_fn was provided"):
        retriever.search("test query", {"id": "alice"}, limit=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
