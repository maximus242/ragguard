"""
Tests for Weaviate secure retriever.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard import WeaviateSecureRetriever
from ragguard.exceptions import RetrieverError


@pytest.fixture
def sample_policy():
    """Create a sample policy for testing."""
    policy_dict = {
        "version": "1",
        "rules": [
            {
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    }
    from ragguard.policy.models import Policy
    return Policy.from_dict(policy_dict)


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client."""
    client = Mock()
    client.__class__.__name__ = "Client"  # Mimic Weaviate Client class name
    client.query = Mock()
    return client


def test_weaviate_retriever_initialization(sample_policy, mock_weaviate_client):
    """Test Weaviate retriever initialization."""
    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy
        )

        assert retriever.backend_name == "weaviate"
        assert retriever.client == mock_weaviate_client
        assert retriever.collection == "TestCollection"


def test_weaviate_retriever_missing_dependency(sample_policy):
    """Test error when weaviate is not installed."""
    with patch.dict('sys.modules', {'weaviate': None}):
        mock_client = Mock()

        with pytest.raises(RetrieverError, match="weaviate-client not installed"):
            WeaviateSecureRetriever(
                client=mock_client,
                collection="TestCollection",
                policy=sample_policy
            )


def test_weaviate_retriever_invalid_client_type(sample_policy):
    """Test error when client is not a valid Weaviate client."""
    with patch.dict('sys.modules', {'weaviate': Mock()}):
        mock_client = Mock()
        mock_client.__class__.__name__ = "InvalidClient"  # Wrong type

        with pytest.raises(RetrieverError, match="Expected weaviate.Client or weaviate.WeaviateClient"):
            WeaviateSecureRetriever(
                client=mock_client,
                collection="TestCollection",
                policy=sample_policy
            )


def test_weaviate_retriever_search_with_filter(sample_policy, mock_weaviate_client):
    """Test search with permission filter."""
    # Mock the query builder chain
    mock_builder = Mock()
    mock_builder.do.return_value = {
        "data": {
            "Get": {
                "TestCollection": [
                    {"department": "engineering", "content": "Doc 1"},
                    {"department": "engineering", "content": "Doc 2"}
                ]
            }
        }
    }

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test query",
            user={"department": "engineering"},
            limit=10
        )

        # Verify query builder was called correctly
        assert mock_get.called
        assert mock_builder.with_near_vector.called
        assert mock_builder.with_limit.called
        assert mock_builder.with_where.called  # Filter should be applied
        assert mock_builder.do.called

        # Verify results
        assert len(results) == 2


def test_weaviate_retriever_search_admin_access(sample_policy, mock_weaviate_client):
    """Test search with admin access (broader permissions)."""
    # Create policy with role-based access
    policy_dict = {
        "version": "1",
        "rules": [
            {
                "name": "admin-access",
                "allow": {
                    "roles": ["admin"]
                }
            }
        ],
        "default": "deny"
    }
    from ragguard.policy.models import Policy
    admin_policy = Policy.from_dict(policy_dict)

    mock_builder = Mock()
    mock_builder.do.return_value = {
        "data": {
            "Get": {
                "TestCollection": [
                    {"content": "Doc 1"},
                    {"content": "Doc 2"},
                    {"content": "Doc 3"}
                ]
            }
        }
    }

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=admin_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test query",
            user={"roles": ["admin"]},
            limit=10
        )

        # Should get results
        assert len(results) == 3


def test_weaviate_retriever_no_results(sample_policy, mock_weaviate_client):
    """Test search with no results."""
    mock_builder = Mock()
    mock_builder.do.return_value = {
        "data": {
            "Get": {
                "TestCollection": []
            }
        }
    }

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test query",
            user={"department": "engineering"},
            limit=10
        )

        assert len(results) == 0


def test_weaviate_retriever_with_vector_query(sample_policy, mock_weaviate_client):
    """Test search with pre-computed vector."""
    mock_builder = Mock()
    mock_builder.do.return_value = {
        "data": {
            "Get": {
                "TestCollection": [
                    {"department": "engineering", "content": "Doc 1"}
                ]
            }
        }
    }

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy
        )

        query_vector = [0.1] * 768
        results = retriever.search(
            query=query_vector,
            user={"department": "engineering"},
            limit=5
        )

        # Verify near_vector was called with the query vector
        call_args = mock_builder.with_near_vector.call_args
        assert call_args[0][0]["vector"] == query_vector
        assert len(results) == 1


def test_weaviate_retriever_with_custom_properties(sample_policy, mock_weaviate_client):
    """Test search with custom properties."""
    mock_builder = Mock()
    mock_builder.do.return_value = {
        "data": {
            "Get": {
                "TestCollection": [
                    {"title": "Title 1", "content": "Doc 1"}
                ]
            }
        }
    }

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5,
            properties=["title", "content"]
        )

        # Verify get was called with custom properties
        call_args = mock_get.call_args
        assert call_args[0][1] == ["title", "content"]


def test_weaviate_retriever_with_additional_metadata(sample_policy, mock_weaviate_client):
    """Test search with custom additional metadata."""
    mock_builder = Mock()
    mock_builder.do.return_value = {
        "data": {
            "Get": {
                "TestCollection": [
                    {"content": "Doc 1"}
                ]
            }
        }
    }

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5,
            with_additional=["score", "explainScore"]
        )

        # Verify with_additional was called with custom fields
        call_args = mock_builder.with_additional.call_args
        assert call_args[0][0] == ["score", "explainScore"]


def test_weaviate_retriever_search_error(sample_policy, mock_weaviate_client):
    """Test search error handling."""
    mock_builder = Mock()
    mock_builder.do.side_effect = Exception("Weaviate error")

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        with pytest.raises(RetrieverError, match="Weaviate search failed"):
            retriever.search(
                query="test",
                user={"department": "engineering"},
                limit=10
            )


def test_weaviate_retriever_empty_response(sample_policy, mock_weaviate_client):
    """Test handling of empty/malformed response."""
    mock_builder = Mock()
    mock_builder.do.return_value = {}  # Empty response

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=10
        )

        # Should handle gracefully and return empty list
        assert len(results) == 0


def test_weaviate_retriever_v4_client(sample_policy):
    """Test initialization with Weaviate v4 client."""
    mock_client = Mock()
    mock_client.__class__.__name__ = "WeaviateClient"  # v4 client type

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_client,
            collection="TestCollection",
            policy=sample_policy
        )

        assert retriever.client == mock_client


def test_weaviate_retriever_with_certainty(sample_policy, mock_weaviate_client):
    """Test search with custom certainty threshold."""
    mock_builder = Mock()
    mock_builder.do.return_value = {
        "data": {
            "Get": {
                "TestCollection": []
            }
        }
    }

    mock_builder.with_additional = Mock(return_value=mock_builder)
    mock_builder.with_where = Mock(return_value=mock_builder)
    mock_builder.with_limit = Mock(return_value=mock_builder)
    mock_builder.with_near_vector = Mock(return_value=mock_builder)

    mock_get = Mock(return_value=mock_builder)
    mock_weaviate_client.query.get = mock_get

    with patch.dict('sys.modules', {'weaviate': Mock()}):
        retriever = WeaviateSecureRetriever(
            client=mock_weaviate_client,
            collection="TestCollection",
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5,
            certainty=0.85
        )

        # Verify with_near_vector was called with custom certainty
        call_args = mock_builder.with_near_vector.call_args
        assert call_args[0][0]["certainty"] == 0.85
