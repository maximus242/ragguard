"""
Tests for Azure AI Search integration.

Tests the AzureSearchSecureRetriever and filter builders.
"""

from unittest.mock import MagicMock, Mock

import pytest


def test_azure_search_filter_builder():
    """Test Azure AI Search filter builder with basic policy."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice", "department": "engineering"}
    filter_str = to_azure_search_filter(policy, user)

    # Should create OData eq expression
    assert filter_str is not None
    assert "department eq 'engineering'" in filter_str


def test_azure_search_filter_with_role():
    """Test Azure AI Search filter with role-based access."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "admin-access",
            "match": {"doc_type": "report"},  # Add match condition
            "allow": {
                "roles": ["admin"]
            }
        }],
        "default": "deny"
    })

    # Admin user - should get filter for doc_type (because rule has match)
    user = {"id": "alice", "roles": ["admin"]}
    filter_str = to_azure_search_filter(policy, user)
    assert "doc_type eq 'report'" in filter_str

    # Non-admin user - should get deny filter
    user2 = {"id": "bob", "roles": ["user"]}
    filter_str2 = to_azure_search_filter(policy, user2)
    assert "search.score() eq -1" in filter_str2


def test_azure_search_filter_with_match():
    """Test Azure AI Search filter with match conditions."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-docs",
            "match": {"visibility": "public"},
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should have OData eq for visibility
    assert "visibility eq 'public'" in filter_str


def test_azure_search_filter_with_list_match():
    """Test Azure AI Search filter with list match (in operator)."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "multi-dept",
            "match": {"department": ["engineering", "product"]},
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should use 'in' operator for multiple values
    assert "department in (" in filter_str
    assert "'engineering'" in filter_str
    assert "'product'" in filter_str


def test_azure_search_filter_with_range():
    """Test Azure AI Search filter with range operators."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "clearance-check",
            "allow": {
                "conditions": ["document.clearance_level <= user.clearance"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice", "clearance": 3}
    filter_str = to_azure_search_filter(policy, user)

    # Should create OData le (less than or equal) expression
    assert filter_str is not None
    assert "clearance_level le 3" in filter_str


def test_azure_search_filter_with_or():
    """Test Azure AI Search filter with OR logic."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-access",
                "match": {"visibility": "internal"},
                "allow": {"roles": ["admin"]}
            },
            {
                "name": "dept-access",
                "allow": {"conditions": ["user.department == document.department"]}
            }
        ],
        "default": "deny"
    })

    # User with department and admin role - both rules match
    user = {"id": "alice", "department": "engineering", "roles": ["admin"]}
    filter_str = to_azure_search_filter(policy, user)

    # Should have 'or' between rules (one for admin with match, one for dept)
    assert " or " in filter_str
    assert "visibility eq 'internal'" in filter_str
    assert "department eq 'engineering'" in filter_str


def test_azure_search_filter_exists():
    """Test Azure AI Search filter with exists operator."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "has-owner",
            "allow": {
                "conditions": ["document.owner exists"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should have 'ne null' for exists
    assert "owner ne null" in filter_str


def test_azure_search_filter_not_exists():
    """Test Azure AI Search filter with not exists operator."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "no-owner",
            "allow": {
                "conditions": ["document.owner not exists"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should have 'eq null' for not exists
    assert "owner eq null" in filter_str


def test_azure_search_filter_nested_field():
    """Test Azure AI Search filter with nested field access."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "org-access",
            "allow": {
                "conditions": ["user.org.id == document.owner.org_id"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice", "org": {"id": "org123"}}
    filter_str = to_azure_search_filter(policy, user)

    # Azure uses '/' for nested fields
    assert "owner/org_id eq 'org123'" in filter_str


def test_azure_search_filter_value_in_array():
    """Test Azure AI Search filter with 'user.id in document.array' pattern."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "shared-with",
            "allow": {
                "conditions": ["user.id in document.shared_with"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should use collection/any lambda
    assert "shared_with/any(" in filter_str
    assert "x eq 'alice'" in filter_str


def test_azure_search_filter_not_in_array():
    """Test Azure AI Search filter with 'user.id not in document.array' pattern."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-blocked",
            "allow": {
                "conditions": ["user.id not in document.blocked_users"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should use collection/all with ne
    assert "blocked_users/all(" in filter_str
    assert "x ne 'alice'" in filter_str


def test_azure_search_filter_with_in_operator():
    """Test Azure AI Search filter with 'in' operator on literal list."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "allowed-depts",
            "allow": {
                "conditions": ["document.department in ['engineering', 'product', 'design']"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should create 'in' operator
    assert "department in (" in filter_str
    assert "'engineering'" in filter_str
    assert "'product'" in filter_str


def test_azure_search_filter_not_equals():
    """Test Azure AI Search filter with != operator."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-internal",
            "allow": {
                "conditions": ["document.classification != 'internal'"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    filter_str = to_azure_search_filter(policy, user)

    # Should create 'ne' expression
    assert "classification ne 'internal'" in filter_str


def test_azure_search_filter_string_escaping():
    """Test that single quotes are escaped in OData strings."""
    from ragguard import Policy
    from ragguard.filters.builder import to_azure_search_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    # Department with single quote
    user = {"id": "alice", "department": "Bob's Team"}
    filter_str = to_azure_search_filter(policy, user)

    # Single quotes should be doubled
    assert "department eq 'Bob''s Team'" in filter_str


def test_azure_search_retriever_basic():
    """Test basic Azure AI Search retriever initialization and search."""
    import sys
    from unittest.mock import patch

    from ragguard import AzureSearchSecureRetriever, Policy

    # Create mock SearchClient
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

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    # Mock the entire azure.search.documents.models module
    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {'azure': MagicMock(),
                                     'azure.search': MagicMock(),
                                     'azure.search.documents': MagicMock(),
                                     'azure.search.documents.models': mock_azure_module}):
        # Create retriever
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=policy,
            vector_field="embedding"
        )

        # Search
        query_vector = [0.1, 0.2, 0.3]
        user = {"id": "alice"}
        results = retriever.search(query_vector, user, limit=5)

        # Verify search was called
        assert mock_client.search.called

        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.9


def test_azure_search_retriever_with_filter():
    """Test Azure AI Search retriever with permission filter."""
    import sys
    from unittest.mock import patch

    from ragguard import AzureSearchSecureRetriever, Policy

    # Create mock client
    mock_client = Mock()
    mock_client.__class__.__name__ = "SearchClient"

    mock_result = {
        "id": "doc1",
        "@search.score": 0.9,
        "content": "Document 1",
        "department": "engineering"
    }
    mock_client.search = Mock(return_value=[mock_result])

    # Create policy with department filter
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    # Mock the entire azure.search.documents.models module
    mock_azure_module = MagicMock()
    mock_vectorized_query = MagicMock()
    mock_azure_module.VectorizedQuery = mock_vectorized_query

    with patch.dict('sys.modules', {'azure': MagicMock(),
                                     'azure.search': MagicMock(),
                                     'azure.search.documents': MagicMock(),
                                     'azure.search.documents.models': mock_azure_module}):
        # Create retriever
        retriever = AzureSearchSecureRetriever(
            client=mock_client,
            index="documents",
            policy=policy
        )

        # Search with user from engineering
        query_vector = [0.1, 0.2, 0.3]
        user = {"id": "alice", "department": "engineering"}
        results = retriever.search(query_vector, user, limit=5)

        # Verify filter was applied
        call_args = mock_client.search.call_args
        assert "filter" in call_args.kwargs
        filter_str = call_args.kwargs["filter"]

        # Should have OData eq for department
        assert "department eq 'engineering'" in filter_str


def test_azure_cognitive_search_alias():
    """Test that AzureCognitiveSearchSecureRetriever is an alias."""
    from ragguard import AzureCognitiveSearchSecureRetriever, AzureSearchSecureRetriever

    # Should be a subclass
    assert issubclass(AzureCognitiveSearchSecureRetriever, AzureSearchSecureRetriever)


def test_azure_search_retriever_invalid_client():
    """Test that invalid client type raises error."""
    from ragguard import AzureSearchSecureRetriever, Policy
    from ragguard.exceptions import RetrieverError

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    # Try to create with invalid client
    with pytest.raises(RetrieverError, match="Expected SearchClient"):
        AzureSearchSecureRetriever(
            client={"invalid": "client"},
            index="documents",
            policy=policy
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
