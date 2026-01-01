"""
Tests for ChromaDB secure retriever.
"""

from unittest.mock import Mock, patch

import pytest

# Skip all tests if chromadb is not installed
pytest.importorskip("chromadb", exc_type=ImportError)

from ragguard import ChromaDBSecureRetriever, load_policy
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
def mock_collection():
    """Create a mock ChromaDB collection."""
    collection = Mock()
    collection.name = "test_collection"
    collection.query = Mock()
    return collection


def test_chromadb_retriever_initialization(sample_policy, mock_collection):
    """Test ChromaDB retriever initialization."""
    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=sample_policy
    )

    assert retriever.backend_name == "chromadb"
    assert retriever.chroma_collection == mock_collection
    assert retriever.collection == "test_collection"


def test_chromadb_retriever_missing_dependency(sample_policy):
    """Test error when chromadb is not installed."""
    with patch.dict('sys.modules', {'chromadb': None}):
        mock_collection = Mock()
        mock_collection.name = "test"

        with pytest.raises(RetrieverError, match="chromadb not installed"):
            ChromaDBSecureRetriever(
                collection=mock_collection,
                policy=sample_policy
            )


def test_chromadb_retriever_search_with_filter(sample_policy, mock_collection):
    """Test search with permission filter."""
    # Mock query response
    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "metadatas": [[
            {"department": "engineering", "content": "Doc 1"},
            {"department": "engineering", "content": "Doc 2"}
        ]],
        "documents": [["Document 1 text", "Document 2 text"]],
        "distances": [[0.5, 0.7]]
    }

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=sample_policy,
        embed_fn=lambda x: [0.1] * 768
    )

    results = retriever.search(
        query="test query",
        user={"department": "engineering"},
        limit=10
    )

    # Verify query was called with correct parameters
    assert mock_collection.query.called
    call_args = mock_collection.query.call_args
    assert call_args[1]["n_results"] == 10
    assert "where" in call_args[1]
    assert call_args[1]["where"] == {"department": {"$eq": "engineering"}}

    # Verify results
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["metadata"]["department"] == "engineering"
    assert results[0]["document"] == "Document 1 text"
    assert results[0]["distance"] == 0.5


def test_chromadb_retriever_search_no_results(sample_policy, mock_collection):
    """Test search with no results."""
    mock_collection.query.return_value = {
        "ids": [[]],
        "metadatas": [[]],
        "documents": [[]],
        "distances": [[]]
    }

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=sample_policy,
        embed_fn=lambda x: [0.1] * 768
    )

    results = retriever.search(
        query="test query",
        user={"department": "engineering"},
        limit=10
    )

    assert len(results) == 0


def test_chromadb_retriever_search_error(sample_policy, mock_collection):
    """Test search error handling."""
    mock_collection.query.side_effect = Exception("ChromaDB error")

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=sample_policy,
        embed_fn=lambda x: [0.1] * 768
    )

    with pytest.raises(RetrieverError, match="ChromaDB search failed"):
        retriever.search(
            query="test query",
            user={"department": "engineering"},
            limit=10
        )


def test_chromadb_retriever_with_vector_query(sample_policy, mock_collection):
    """Test search with pre-computed vector."""
    mock_collection.query.return_value = {
        "ids": [["doc1"]],
        "metadatas": [[{"department": "engineering"}]],
        "documents": [["Document text"]],
        "distances": [[0.3]]
    }

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=sample_policy
    )

    query_vector = [0.1] * 768
    results = retriever.search(
        query=query_vector,
        user={"department": "engineering"},
        limit=5
    )

    assert len(results) == 1
    assert mock_collection.query.called
    call_args = mock_collection.query.call_args
    assert call_args[1]["query_embeddings"] == [query_vector]


def test_chromadb_retriever_with_where_document(sample_policy, mock_collection):
    """Test search with where_document filter."""
    mock_collection.query.return_value = {
        "ids": [["doc1"]],
        "metadatas": [[{"department": "engineering"}]],
        "documents": [["Important document"]],
        "distances": [[0.2]]
    }

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=sample_policy,
        embed_fn=lambda x: [0.1] * 768
    )

    results = retriever.search(
        query="test",
        user={"department": "engineering"},
        limit=5,
        where_document={"$contains": "important"}
    )

    call_args = mock_collection.query.call_args
    assert "where_document" in call_args[1]
    assert call_args[1]["where_document"] == {"$contains": "important"}


def test_chromadb_retriever_admin_full_access(mock_collection):
    """Test admin getting full access (no filter)."""
    policy_dict = {
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    }
    from ragguard.policy.models import Policy
    policy = Policy.from_dict(policy_dict)

    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2", "doc3"]],
        "metadatas": [[{}, {}, {}]],
        "documents": [["Doc 1", "Doc 2", "Doc 3"]],
        "distances": [[0.1, 0.2, 0.3]]
    }

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1] * 768
    )

    results = retriever.search(
        query="test",
        user={"roles": ["admin"]},
        limit=10
    )

    # Admin should have no filter (None or no 'where' key)
    call_args = mock_collection.query.call_args
    assert "where" not in call_args[1] or call_args[1].get("where") is None
    assert len(results) == 3
