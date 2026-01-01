"""
Tests for Pinecone secure retriever.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard import PineconeSecureRetriever
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
def mock_index():
    """Create a mock Pinecone index."""
    index = Mock()
    index.query = Mock()
    return index


def test_pinecone_retriever_initialization(sample_policy, mock_index):
    """Test Pinecone retriever initialization."""
    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            namespace="test-namespace"
        )

        assert retriever.backend_name == "pinecone"
        assert retriever.index == mock_index
        assert retriever.namespace == "test-namespace"


def test_pinecone_retriever_missing_dependency(sample_policy):
    """Test error when pinecone is not installed."""
    with patch.dict('sys.modules', {'pinecone': None}):
        mock_index = Mock()

        with pytest.raises(RetrieverError, match="pinecone-client not installed"):
            PineconeSecureRetriever(
                index=mock_index,
                policy=sample_policy
            )


def test_pinecone_retriever_search_with_filter(sample_policy, mock_index):
    """Test search with permission filter."""
    # Mock query response
    mock_response = Mock()
    mock_response.matches = [
        Mock(id="doc1", score=0.95, metadata={"department": "engineering"}),
        Mock(id="doc2", score=0.90, metadata={"department": "engineering"})
    ]
    mock_index.query.return_value = mock_response

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test query",
            user={"department": "engineering"},
            limit=10
        )

        # Verify query was called with correct parameters
        assert mock_index.query.called
        call_args = mock_index.query.call_args
        assert call_args[1]["top_k"] == 10
        assert "filter" in call_args[1]
        assert call_args[1]["filter"] == {"department": {"$eq": "engineering"}}

        # Verify results
        assert len(results) == 2


def test_pinecone_retriever_dict_response(sample_policy, mock_index):
    """Test search with dict response format."""
    # Mock dict-style response (some Pinecone versions)
    mock_index.query.return_value = {
        "matches": [
            {"id": "doc1", "score": 0.95, "metadata": {"department": "eng"}},
            {"id": "doc2", "score": 0.90, "metadata": {"department": "eng"}}
        ]
    }

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "eng"},
            limit=10
        )

        assert len(results) == 2


def test_pinecone_retriever_no_results(sample_policy, mock_index):
    """Test search with no results."""
    mock_response = Mock()
    mock_response.matches = []
    mock_index.query.return_value = mock_response

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=10
        )

        assert len(results) == 0


def test_pinecone_retriever_with_vector_query(sample_policy, mock_index):
    """Test search with pre-computed vector."""
    mock_response = Mock()
    mock_response.matches = [
        Mock(id="doc1", score=0.95, metadata={"department": "engineering"})
    ]
    mock_index.query.return_value = mock_response

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy
        )

        query_vector = [0.1] * 768
        results = retriever.search(
            query=query_vector,
            user={"department": "engineering"},
            limit=5
        )

        assert len(results) == 1
        call_args = mock_index.query.call_args
        assert call_args[1]["vector"] == query_vector


def test_pinecone_retriever_with_namespace_override(sample_policy, mock_index):
    """Test search with namespace override."""
    mock_response = Mock()
    mock_response.matches = []
    mock_index.query.return_value = mock_response

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            namespace="default-ns",
            embed_fn=lambda x: [0.1] * 768
        )

        retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5,
            namespace="custom-ns"
        )

        call_args = mock_index.query.call_args
        assert call_args[1]["namespace"] == "custom-ns"


def test_pinecone_retriever_with_sparse_vector(sample_policy, mock_index):
    """Test search with sparse vector."""
    mock_response = Mock()
    mock_response.matches = []
    mock_index.query.return_value = mock_response

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        sparse_vec = {"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]}
        retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5,
            sparse_vector=sparse_vec
        )

        call_args = mock_index.query.call_args
        assert "sparse_vector" in call_args[1]
        assert call_args[1]["sparse_vector"] == sparse_vec


def test_pinecone_retriever_search_error(sample_policy, mock_index):
    """Test search error handling."""
    mock_index.query.side_effect = Exception("Pinecone error")

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        with pytest.raises(RetrieverError, match="Pinecone search failed"):
            retriever.search(
                query="test",
                user={"department": "engineering"},
                limit=10
            )


def test_pinecone_retriever_include_values(sample_policy, mock_index):
    """Test search with include_values parameter."""
    mock_response = Mock()
    mock_response.matches = []
    mock_index.query.return_value = mock_response

    with patch.dict('sys.modules', {'pinecone': Mock()}):
        retriever = PineconeSecureRetriever(
            index=mock_index,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5,
            include_values=True
        )

        call_args = mock_index.query.call_args
        assert "include_values" in call_args[1]
        assert call_args[1]["include_values"] is True
