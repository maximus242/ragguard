"""
Tests for Elasticsearch/OpenSearch integration.

Tests the ElasticsearchSecureRetriever and filter builders.
"""

from unittest.mock import MagicMock, Mock

import pytest


def test_elasticsearch_filter_builder():
    """Test Elasticsearch filter builder with basic policy."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should create a term query
    assert filter_query is not None
    assert "term" in filter_query
    assert filter_query["term"]["department"] == "engineering"


def test_elasticsearch_filter_with_role():
    """Test Elasticsearch filter with role-based access."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "admin-access",
            "allow": {
                "roles": ["admin"]
            }
        }],
        "default": "deny"
    })

    # Admin user - should get match_all
    user = {"id": "alice", "roles": ["admin"]}
    filter_query = to_elasticsearch_filter(policy, user)
    assert filter_query is not None
    assert "match_all" in filter_query

    # Non-admin user - should get deny filter
    user2 = {"id": "bob", "roles": ["user"]}
    filter_query2 = to_elasticsearch_filter(policy, user2)
    assert "bool" in filter_query2
    assert "must" in filter_query2["bool"]


def test_elasticsearch_filter_with_match():
    """Test Elasticsearch filter with match conditions."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should have term query for visibility
    assert "term" in filter_query
    assert filter_query["term"]["visibility"] == "public"


def test_elasticsearch_filter_with_list_match():
    """Test Elasticsearch filter with list match (terms query)."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should use terms query for multiple values
    assert "terms" in filter_query
    assert filter_query["terms"]["department"] == ["engineering", "product"]


def test_elasticsearch_filter_with_range():
    """Test Elasticsearch filter with range operators."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should create range query
    assert filter_query is not None
    assert "range" in filter_query
    assert filter_query["range"]["clearance_level"]["lte"] == 3


def test_elasticsearch_filter_with_or():
    """Test Elasticsearch filter with OR logic."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-access",
                "allow": {"roles": ["admin"]}
            },
            {
                "name": "dept-access",
                "allow": {"conditions": ["user.department == document.department"]}
            }
        ],
        "default": "deny"
    })

    # User with department but no admin role - only one rule matches
    user = {"id": "alice", "department": "engineering", "roles": ["user"]}
    filter_query = to_elasticsearch_filter(policy, user)

    # Only dept-access matches, so should return single filter
    assert "term" in filter_query
    assert filter_query["term"]["department"] == "engineering"

    # Admin user with department - both rules match
    user2 = {"id": "bob", "roles": ["admin"], "department": "finance"}
    filter_query2 = to_elasticsearch_filter(policy, user2)

    # Should have bool should query (OR between rules)
    assert "bool" in filter_query2
    assert "should" in filter_query2["bool"]
    assert len(filter_query2["bool"]["should"]) == 2  # Both rules match

    # Verify both filters are present
    should_filters = filter_query2["bool"]["should"]
    # One should be match_all (admin rule), one should be term (dept rule)
    has_match_all = any("match_all" in f for f in should_filters)
    has_term = any("term" in f for f in should_filters)
    assert has_match_all and has_term


def test_elasticsearch_filter_exists():
    """Test Elasticsearch filter with exists operator."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should have exists query
    assert "exists" in filter_query
    assert filter_query["exists"]["field"] == "owner"


def test_elasticsearch_retriever_basic():
    """Test basic Elasticsearch retriever initialization and search."""
    from ragguard import ElasticsearchSecureRetriever, Policy

    # Create mock Elasticsearch client
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

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    # Create retriever
    retriever = ElasticsearchSecureRetriever(
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
    call_args = mock_client.search.call_args

    # Verify knn query structure
    assert "knn" in call_args.kwargs
    assert call_args.kwargs["knn"]["field"] == "embedding"
    assert call_args.kwargs["knn"]["query_vector"] == query_vector
    assert call_args.kwargs["knn"]["k"] == 5

    # Verify results
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] == 0.9
    assert results[0]["metadata"]["department"] == "engineering"


def test_elasticsearch_retriever_with_filter():
    """Test Elasticsearch retriever with permission filter."""
    from ragguard import ElasticsearchSecureRetriever, Policy

    # Create mock client
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

    # Create retriever
    retriever = ElasticsearchSecureRetriever(
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
    assert "filter" in call_args.kwargs["knn"]
    filter_query = call_args.kwargs["knn"]["filter"]

    # Should have term query for department
    assert "term" in filter_query
    assert filter_query["term"]["department"] == "engineering"


def test_opensearch_retriever_alias():
    """Test that OpenSearchSecureRetriever works like ElasticsearchSecureRetriever."""
    from ragguard import ElasticsearchSecureRetriever, OpenSearchSecureRetriever, Policy

    # OpenSearchSecureRetriever should be a subclass
    assert issubclass(OpenSearchSecureRetriever, ElasticsearchSecureRetriever)

    # Should work with OpenSearch client mock
    mock_client = Mock()
    mock_client.__class__.__name__ = "OpenSearch"
    mock_client.search = Mock(return_value={
        "hits": {"hits": []}
    })

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Should initialize without error
    retriever = OpenSearchSecureRetriever(
        client=mock_client,
        index="documents",
        policy=policy
    )

    assert retriever.backend_name == "elasticsearch"  # Uses same backend name


def test_elasticsearch_retriever_invalid_client():
    """Test that invalid client type raises error."""
    from ragguard import ElasticsearchSecureRetriever, Policy
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
    with pytest.raises(RetrieverError, match="Expected Elasticsearch or OpenSearch client"):
        ElasticsearchSecureRetriever(
            client={"invalid": "client"},
            index="documents",
            policy=policy
        )


def test_elasticsearch_filter_with_nested_field():
    """Test Elasticsearch filter with nested field access."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should create term query with nested field
    assert "term" in filter_query
    assert filter_query["term"]["owner.org_id"] == "org123"


def test_elasticsearch_filter_not_equals():
    """Test Elasticsearch filter with != operator."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should create bool must_not query
    assert "bool" in filter_query
    assert "must_not" in filter_query["bool"]
    assert filter_query["bool"]["must_not"][0]["term"]["classification"] == "internal"


def test_elasticsearch_filter_with_in_operator():
    """Test Elasticsearch filter with 'in' operator."""
    from ragguard import Policy
    from ragguard.filters.builder import to_elasticsearch_filter

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
    filter_query = to_elasticsearch_filter(policy, user)

    # Should create terms query
    assert "terms" in filter_query
    assert filter_query["terms"]["department"] == ["engineering", "product", "design"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
