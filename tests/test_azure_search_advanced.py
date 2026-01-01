"""
Advanced tests for Azure AI Search integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard import AzureCognitiveSearchSecureRetriever, AzureSearchSecureRetriever, Policy
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


def create_mock_azure_client():
    """Create a mock Azure SearchClient."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "SearchClient"

    # Mock search results
    mock_result = {
        "id": "doc1",
        "@search.score": 0.9,
        "content": "Document 1",
        "department": "engineering"
    }
    mock_client.search = Mock(return_value=[mock_result])

    # Mock service client for health checks
    mock_service_client = Mock()
    mock_service_client.get_index = Mock(return_value=Mock())

    mock_index_stats = Mock()
    mock_index_stats.document_count = 100
    mock_service_client.get_index_statistics = Mock(return_value=mock_index_stats)

    # Store service client reference
    mock_client._service_client = mock_service_client

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


def test_azure_search_health_check_success():
    """Test successful health check for Azure Search."""
    mock_client = create_mock_azure_client()

    # Mock the azure.search.documents.models module
    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': MagicMock()
    }):
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy()
        )

        # Mock health check dependencies
        with patch.object(retriever, '_check_backend_health') as mock_health:
            mock_health.return_value = {
                "connection_alive": True,
                "index_exists": True,
                "index_info": {"document_count": 100}
            }

            health = retriever.health_check()

            assert health["healthy"] is True
            assert health["backend"] == "azure_search"
            assert health["collection"] == "documents"
            assert health["details"]["connection_alive"] is True


def test_azure_search_context_manager():
    """Test using Azure Search retriever as context manager."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        with AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy()
        ) as retriever:
            assert retriever is not None
            results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
            assert len(results) == 1


def test_azure_search_with_validation():
    """Test Azure Search retriever with input validation."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
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


def test_azure_search_with_retry():
    """Test Azure Search retriever with retry logic."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "SearchClient"

    # Fail twice, then succeed
    call_count = [0]
    mock_result = {
        "id": "doc1",
        "@search.score": 0.9,
        "content": "Document 1"
    }

    def search_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            # Raise a retryable exception (not wrapped in RetrieverError)
            raise OSError("Connection failed")
        return [mock_result]

    mock_client.search = Mock(side_effect=search_side_effect)

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
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


def test_azure_search_with_cache():
    """Test Azure Search retriever with filter caching."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
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

        # Second search - cache hit
        retriever.search([0.1, 0.2, 0.3], user, limit=5)

        # Verify search was called twice
        assert mock_client.search.call_count == 2


def test_azure_search_with_audit_logging():
    """Test Azure Search retriever with audit logging."""
    mock_client = create_mock_azure_client()

    # Create audit logger with callback
    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
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


def test_azure_search_search_failure():
    """Test Azure Search search failure handling."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "SearchClient"
    mock_client.search = Mock(side_effect=Exception("Search failed"))

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy(),
            enable_retry=False  # Disable retry for immediate failure
        )

        with pytest.raises(RetrieverError, match="Azure AI Search failed"):
            retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_azure_search_empty_results():
    """Test Azure Search with empty search results."""
    mock_client = create_mock_azure_client()
    mock_client.search = Mock(return_value=[])

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy()
        )

        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

        assert len(results) == 0


def test_azure_search_batch_search():
    """Test batch search with Azure Search."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
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


def test_azure_search_policy_update():
    """Test updating policy on Azure Search retriever."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
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

        # Verify filter was applied
        call_args = mock_client.search.call_args
        assert "filter" in call_args.kwargs


def test_azure_search_with_embed_fn():
    """Test Azure Search retriever with text query and embedding function."""
    mock_client = create_mock_azure_client()

    def embed_fn(text):
        return [float(ord(c)) / 1000 for c in text[:3]]

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy(),
            embed_fn=embed_fn
        )

        # Search with text query
        results = retriever.search("test query", {"id": "alice"}, limit=5)

        assert len(results) == 1
        assert mock_client.search.called


def test_azure_search_text_query_without_embed_fn():
    """Test that text query without embed_fn raises error."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy(),
            embed_fn=None  # No embedding function
        )

        with pytest.raises(RetrieverError, match="no embed_fn was provided"):
            retriever.search("test query", {"id": "alice"}, limit=5)


def test_azure_cognitive_search_alias():
    """Test that AzureCognitiveSearchSecureRetriever works correctly."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        # Should work the same as AzureSearchSecureRetriever
        retriever = AzureCognitiveSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy()
        )

        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) == 1


def test_azure_search_with_custom_vector_field():
    """Test Azure Search with custom vector field name."""
    mock_client = create_mock_azure_client()

    mock_azure_module = MagicMock()
    mock_vectorized_query_class = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query_class

    with patch.dict('sys.modules', {
        'azure': MagicMock(),
        'azure.search': MagicMock(),
        'azure.search.documents': MagicMock(),
        'azure.search.documents.models': mock_azure_module
    }):
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=create_basic_policy(),
            vector_field="custom_embedding"
        )

        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

        # Verify VectorizedQuery was called with custom field
        assert mock_vectorized_query_class.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
