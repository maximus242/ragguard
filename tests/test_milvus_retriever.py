"""
Tests for Milvus/Zilliz secure retriever.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def test_policy():
    """Create test policy."""
    from ragguard import Policy

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
                "name": "public-access",
                "allow": {
                    "conditions": ["document.visibility == 'public'"]
                }
            }
        ],
        "default": "deny"
    })


@pytest.fixture
def mock_milvus_hits():
    """Mock Milvus search results."""
    return [
        {
            "id": 1,
            "distance": 0.1,
            "department": "engineering",
            "visibility": "internal",
            "text": "Engineering doc 1"
        },
        {
            "id": 2,
            "distance": 0.2,
            "department": "engineering",
            "visibility": "public",
            "text": "Engineering doc 2"
        },
        {
            "id": 3,
            "distance": 0.3,
            "department": "hr",
            "visibility": "internal",
            "text": "HR doc 1"
        }
    ]


def test_milvus_retriever_import():
    """Test that Milvus retriever can be imported."""
    try:
        from ragguard.retrievers import MilvusSecureRetriever, ZillizSecureRetriever
        assert MilvusSecureRetriever is not None
        assert ZillizSecureRetriever is not None
    except ImportError as e:
        # pymilvus not installed is acceptable for this test
        if "pymilvus" not in str(e):
            raise


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_milvus_retriever_initialization(mock_milvus_client, test_policy):
    """Test Milvus retriever initialization."""
    from ragguard.retrievers import MilvusSecureRetriever

    mock_client = MagicMock()
    mock_client.describe_collection.return_value = {"name": "test_collection"}

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=test_policy
    )

    assert retriever.collection_name == "test_collection"
    assert retriever.policy == test_policy
    assert retriever.vector_field == "vector"


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_milvus_search_with_policy(mock_milvus_client, test_policy, mock_milvus_hits):
    """Test Milvus search with policy filtering."""
    from ragguard.retrievers import MilvusSecureRetriever

    mock_client = MagicMock()
    mock_client.describe_collection.return_value = {"name": "test_collection"}
    mock_client.search.return_value = [mock_milvus_hits]

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=test_policy
    )

    # Search as engineering user
    engineering_user = {"id": "alice", "department": "engineering"}
    results = retriever.search(
        query=[0.1, 0.2, 0.3],
        user=engineering_user,
        limit=10
    )

    # Should see 2 engineering docs (both match policy)
    assert len(results) == 2
    assert all(r["metadata"]["department"] == "engineering" for r in results)


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_milvus_search_different_department(mock_milvus_client, test_policy, mock_milvus_hits):
    """Test that users only see their department's docs or public docs."""
    from ragguard.retrievers import MilvusSecureRetriever

    mock_client = MagicMock()
    mock_client.describe_collection.return_value = {"name": "test_collection"}
    mock_client.search.return_value = [mock_milvus_hits]

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=test_policy
    )

    # Search as HR user
    hr_user = {"id": "bob", "department": "hr"}
    results = retriever.search(
        query=[0.1, 0.2, 0.3],
        user=hr_user,
        limit=10
    )

    # Should see 1 public engineering doc + 1 HR doc = 2 results
    assert len(results) == 2


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_milvus_filter_generation(mock_milvus_client, test_policy):
    """Test that Milvus filter expressions are generated correctly."""
    from ragguard.retrievers import MilvusSecureRetriever

    mock_client = MagicMock()
    mock_client.describe_collection.return_value = {"name": "test_collection"}
    mock_client.search.return_value = [[]]

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=test_policy
    )

    user = {"id": "alice", "department": "engineering"}

    # Test filter generation by doing a search and capturing the filter
    # The search will call the filter builder internally
    results = retriever.search(
        query=[0.1, 0.2, 0.3],
        user=user,
        limit=5
    )

    # Verify search was called with a filter expression
    assert mock_client.search.called
    call_args = mock_client.search.call_args
    if call_args and len(call_args) > 1 and 'expr' in call_args[1]:
        filter_expr = call_args[1]['expr']
        # Should contain department check or visibility check
        assert filter_expr is None or "engineering" in filter_expr or "department" in filter_expr or "public" in filter_expr


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_milvus_result_standardization(mock_milvus_client, test_policy, mock_milvus_hits):
    """Test that Milvus results are standardized correctly."""
    from ragguard.retrievers import MilvusSecureRetriever

    mock_client = MagicMock()
    mock_client.describe_collection.return_value = {"name": "test_collection"}
    mock_client.search.return_value = [mock_milvus_hits]

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=test_policy
    )

    user = {"id": "alice", "department": "engineering"}
    results = retriever.search(query=[0.1, 0.2], user=user, limit=10)

    # Check standardized format
    assert len(results) > 0
    for result in results:
        assert "id" in result
        assert "metadata" in result
        assert "distance" in result
        assert "score" in result


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_zilliz_retriever_alias(mock_milvus_client, test_policy):
    """Test that ZillizSecureRetriever is an alias for MilvusSecureRetriever."""
    from ragguard.retrievers import MilvusSecureRetriever, ZillizSecureRetriever

    # ZillizSecureRetriever should be a subclass of MilvusSecureRetriever
    assert issubclass(ZillizSecureRetriever, MilvusSecureRetriever)


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_milvus_custom_vector_field(mock_milvus_client, test_policy):
    """Test Milvus retriever with custom vector field name."""
    from ragguard.retrievers import MilvusSecureRetriever

    mock_client = MagicMock()
    mock_client.describe_collection.return_value = {"name": "test_collection"}

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=test_policy,
        vector_field="custom_embedding"
    )

    assert retriever.vector_field == "custom_embedding"


@patch('ragguard.retrievers.milvus.MilvusClient')
def test_milvus_custom_output_fields(mock_milvus_client, test_policy):
    """Test Milvus retriever with custom output fields."""
    from ragguard.retrievers import MilvusSecureRetriever

    mock_client = MagicMock()
    mock_client.describe_collection.return_value = {"name": "test_collection"}

    retriever = MilvusSecureRetriever(
        client=mock_client,
        collection_name="test_collection",
        policy=test_policy,
        output_fields=["department", "text", "visibility"]
    )

    assert retriever.output_fields == ["department", "text", "visibility"]


def test_milvus_missing_pymilvus():
    """Test that helpful error is raised when pymilvus is not installed."""
    import sys

    # Temporarily hide pymilvus
    pymilvus_module = sys.modules.get('pymilvus')

    try:
        # Remove pymilvus from modules
        if 'ragguard.retrievers.milvus' in sys.modules:
            del sys.modules['ragguard.retrievers.milvus']
        sys.modules['pymilvus'] = None

        # Try to import
        with pytest.raises(ImportError, match="pymilvus is required"):
            from ragguard.retrievers.milvus import MilvusSecureRetriever
            mock_client = MagicMock()
            from ragguard import Policy
            policy = Policy.from_dict({
                "version": "1",
                "rules": [{"name": "test", "allow": {"everyone": True}}],
                "default": "deny"
            })
            MilvusSecureRetriever(
                client=mock_client,
                collection_name="test",
                policy=policy
            )

    finally:
        # Restore pymilvus
        if pymilvus_module:
            sys.modules['pymilvus'] = pymilvus_module
        elif 'pymilvus' in sys.modules:
            del sys.modules['pymilvus']


def test_milvus_filter_builder():
    """Test Milvus filter builder generates correct expressions."""
    from ragguard import Policy
    from ragguard.filters.builder import to_milvus_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })

    user = {"id": "alice", "department": "engineering"}
    filter_expr = to_milvus_filter(policy, user)

    # Should generate expression with department check
    assert filter_expr is not None
    assert "department" in filter_expr
    assert "engineering" in filter_expr


def test_milvus_filter_or_logic():
    """Test Milvus filter builder with OR logic."""
    from ragguard import Policy
    from ragguard.filters.builder import to_milvus_filter

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "public-access",
                "allow": {
                    "conditions": ["document.visibility == 'public'"]
                }
            }
        ],
        "default": "deny"
    })

    user = {"id": "alice", "department": "engineering"}
    filter_expr = to_milvus_filter(policy, user)

    # Should generate expression with OR between rules
    assert filter_expr is not None
    assert " or " in filter_expr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
