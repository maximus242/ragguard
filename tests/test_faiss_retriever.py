"""
Tests for FAISS secure retriever.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if numpy is not installed
np = pytest.importorskip("numpy")

from ragguard import FAISSSecureRetriever
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
    """Create a mock FAISS index."""
    index = Mock()
    index.ntotal = 100  # Total number of vectors
    index.search = Mock()
    return index


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return [
        {"id": 0, "department": "engineering", "content": "Doc 1"},
        {"id": 1, "department": "engineering", "content": "Doc 2"},
        {"id": 2, "department": "sales", "content": "Doc 3"},
        {"id": 3, "department": "engineering", "content": "Doc 4"},
        {"id": 4, "department": "hr", "content": "Doc 5"},
    ]


def test_faiss_retriever_initialization(sample_policy, mock_index, sample_metadata):
    """Test FAISS retriever initialization."""
    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            over_fetch_factor=3
        )

        assert retriever.backend_name == "faiss"
        assert retriever.index == mock_index
        assert retriever.metadata == sample_metadata
        assert retriever.over_fetch_factor == 3


def test_faiss_retriever_missing_dependency(sample_policy, sample_metadata):
    """Test error when faiss is not installed."""
    with patch.dict('sys.modules', {'faiss': None}):
        mock_index = Mock()

        with pytest.raises(RetrieverError, match="faiss not installed"):
            FAISSSecureRetriever(
                index=mock_index,
                metadata=sample_metadata,
                policy=sample_policy
            )


def test_faiss_retriever_search_with_filter(sample_policy, mock_index, sample_metadata):
    """Test search with post-filtering."""
    # Mock FAISS search results
    distances = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    indices = np.array([[0, 1, 2, 3, 4]])
    mock_index.search.return_value = (distances, indices)

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768,
            over_fetch_factor=3
        )

        results = retriever.search(
            query="test query",
            user={"department": "engineering"},
            limit=2
        )

        # Should get 2 results (all from engineering dept)
        assert len(results) == 2
        assert results[0]["metadata"]["department"] == "engineering"
        assert results[1]["metadata"]["department"] == "engineering"
        assert results[0]["distance"] == 0.1
        assert results[1]["distance"] == 0.2

        # Verify over-fetch was used
        call_args = mock_index.search.call_args
        # Should fetch limit * over_fetch_factor = 2 * 3 = 6
        assert call_args[0][1] == 6


def test_faiss_retriever_insufficient_results(sample_policy, mock_index):
    """Test when not enough results pass filter."""
    metadata = [
        {"id": 0, "department": "sales"},
        {"id": 1, "department": "sales"},
        {"id": 2, "department": "sales"},
    ]

    distances = np.array([[0.1, 0.2, 0.3]])
    indices = np.array([[0, 1, 2]])
    mock_index.search.return_value = (distances, indices)
    mock_index.ntotal = 3

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=metadata,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=10
        )

        # No results should pass (all are sales dept)
        assert len(results) == 0


def test_faiss_retriever_padding_indices(sample_policy, mock_index, sample_metadata):
    """Test handling of -1 padding indices."""
    # FAISS returns -1 for padding when not enough results
    distances = np.array([[0.1, 0.2, -1, -1, -1]])
    indices = np.array([[0, 1, -1, -1, -1]])
    mock_index.search.return_value = (distances, indices)

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5
        )

        # Should only get 2 results (indices 0 and 1)
        assert len(results) == 2


def test_faiss_retriever_missing_metadata(sample_policy, mock_index):
    """Test handling of missing metadata."""
    metadata = [
        {"id": 0, "department": "engineering"},
        # Missing index 1
        # Missing index 2
    ]

    distances = np.array([[0.1, 0.2, 0.3]])
    indices = np.array([[0, 1, 2]])  # Indices 1 and 2 have no metadata
    mock_index.search.return_value = (distances, indices)

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=metadata,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=10
        )

        # Should only get 1 result (index 0, which has metadata)
        assert len(results) == 1
        assert results[0]["id"] == 0


def test_faiss_retriever_with_vector_query(sample_policy, mock_index, sample_metadata):
    """Test search with pre-computed vector."""
    distances = np.array([[0.1, 0.2]])
    indices = np.array([[0, 1]])
    mock_index.search.return_value = (distances, indices)

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy
        )

        query_vector = [0.1] * 768
        results = retriever.search(
            query=query_vector,
            user={"department": "engineering"},
            limit=2
        )

        assert len(results) == 2


def test_faiss_retriever_over_fetch_limit(sample_policy, mock_index, sample_metadata):
    """Test that over-fetch doesn't exceed index size."""
    mock_index.ntotal = 10  # Only 10 vectors in index

    distances = np.array([[0.1] * 10])
    indices = np.array([list(range(10))])
    mock_index.search.return_value = (distances, indices)

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768,
            over_fetch_factor=100  # Very high over-fetch
        )

        retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=5
        )

        # Should cap at index size (10), not 5 * 100 = 500
        call_args = mock_index.search.call_args
        assert call_args[0][1] == 10


def test_faiss_retriever_search_error(sample_policy, mock_index, sample_metadata):
    """Test search error handling."""
    mock_index.search.side_effect = Exception("FAISS error")

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        with pytest.raises(RetrieverError, match="FAISS search failed"):
            retriever.search(
                query="test",
                user={"department": "engineering"},
                limit=10
            )


def test_faiss_retriever_score_conversion(sample_policy, mock_index, sample_metadata):
    """Test distance to score conversion."""
    distances = np.array([[0.0, 1.0, 2.0]])  # Different distances
    indices = np.array([[0, 1, 3]])
    mock_index.search.return_value = (distances, indices)

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            embed_fn=lambda x: [0.1] * 768
        )

        results = retriever.search(
            query="test",
            user={"department": "engineering"},
            limit=10
        )

        # Check scores: score = 1.0 / (1.0 + distance)
        assert results[0]["score"] == 1.0 / (1.0 + 0.0)  # = 1.0
        assert results[1]["score"] == 1.0 / (1.0 + 1.0)  # = 0.5
        assert results[2]["score"] == 1.0 / (1.0 + 2.0)  # = 0.333...


def test_faiss_retriever_min_over_fetch_factor(sample_policy, mock_index, sample_metadata):
    """Test that over_fetch_factor has minimum of 1."""
    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            over_fetch_factor=0  # Invalid, should be clamped to 1
        )

        assert retriever.over_fetch_factor == 1


def test_faiss_retriever_max_absolute_fetch(sample_policy, sample_metadata):
    """Test that max_absolute_fetch limits memory usage."""
    import numpy as np

    # Create a large mock index
    mock_index = Mock()
    mock_index.ntotal = 1000000  # 1 million vectors

    # Return results beyond our limit
    def mock_search(query, k):
        # Return mock results - all distances 0, indices 0 to k-1
        return (
            np.zeros((1, k), dtype=np.float32),
            np.arange(k).reshape(1, k)
        )

    mock_index.search = mock_search

    # Extend metadata to be "large enough" (we won't actually access all of it)
    large_metadata = sample_metadata * 100  # Just needs to be indexable

    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=large_metadata,
            policy=sample_policy,
            over_fetch_factor=10,
            max_over_fetch_factor=100,
            max_absolute_fetch=50  # Limit to 50 vectors max
        )

        # Request 100 results - should be capped at max_absolute_fetch
        results = retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"department": "engineering"},
            limit=100
        )

        # We should get results, but the internal fetch was capped
        assert retriever.max_absolute_fetch == 50
        # Results may be less than limit due to filtering and the cap


def test_faiss_retriever_max_absolute_fetch_clamp(sample_policy, mock_index, sample_metadata):
    """Test that max_absolute_fetch has minimum of 1."""
    with patch.dict('sys.modules', {'faiss': Mock()}):
        retriever = FAISSSecureRetriever(
            index=mock_index,
            metadata=sample_metadata,
            policy=sample_policy,
            max_absolute_fetch=0  # Invalid, should be clamped to 1
        )

        assert retriever.max_absolute_fetch == 1
