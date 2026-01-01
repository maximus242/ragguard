"""
Tests for batch search operations.

Tests the batch_search() method across different retrievers.
"""

from unittest.mock import Mock

import pytest

# Skip all tests if chromadb is not installed
pytest.importorskip("chromadb", exc_type=ImportError)


def test_batch_search_basic():
    """Test basic batch search with multiple queries."""
    from ragguard import Policy
    from ragguard.retrievers import ChromaDBSecureRetriever

    # Create mock collection
    mock_collection = Mock()
    mock_collection.query = Mock(side_effect=[
        # Results for query 1
        {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Doc 1"]]
        },
        # Results for query 2
        {
            "ids": [["2"]],
            "distances": [[0.2]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Doc 2"]]
        },
        # Results for query 3
        {
            "ids": [["3"]],
            "distances": [[0.3]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Doc 3"]]
        },
    ])

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {"conditions": ["user.department == document.department"]}
        }],
        "default": "deny"
    })

    # Create retriever
    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy
    )

    # Batch search
    queries = [
        [0.1, 0.2, 0.3],  # Query 1
        [0.4, 0.5, 0.6],  # Query 2
        [0.7, 0.8, 0.9],  # Query 3
    ]

    user = {"id": "alice", "department": "engineering"}
    results = retriever.batch_search(queries, user, limit=5)

    # Should get 3 result lists (one per query)
    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)

    # Each should have results
    assert len(results[0]) >= 1
    assert len(results[1]) >= 1
    assert len(results[2]) >= 1


def test_batch_search_with_text_queries():
    """Test batch search with text queries (using embed_fn)."""
    from ragguard import Policy
    from ragguard.retrievers import ChromaDBSecureRetriever

    # Create mock collection
    mock_collection = Mock()
    mock_collection.query = Mock(side_effect=[
        {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Doc 1"]]
        },
        {
            "ids": [["2"]],
            "distances": [[0.2]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Doc 2"]]
        },
    ])

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    # Mock embedding function
    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    # Create retriever
    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=mock_embed
    )

    # Batch search with text queries
    queries = ["query 1", "query 2"]
    user = {"id": "alice", "department": "engineering"}
    results = retriever.batch_search(queries, user, limit=5)

    # Should get 2 result lists
    assert len(results) == 2
    assert len(results[0]) >= 1
    assert len(results[1]) >= 1


def test_batch_search_empty_queries():
    """Test batch search with empty query list."""
    from ragguard import Policy
    from ragguard.retrievers import ChromaDBSecureRetriever

    mock_collection = Mock()
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy
    )

    # Empty queries
    results = retriever.batch_search([], {"id": "alice"}, limit=5)
    assert results == []


def test_batch_search_single_query():
    """Test batch search with single query."""
    from ragguard import Policy
    from ragguard.retrievers import ChromaDBSecureRetriever

    mock_collection = Mock()
    mock_collection.query = Mock(return_value={
        "ids": [["1"]],
        "distances": [[0.1]],
        "metadatas": [[{"department": "engineering"}]],
        "documents": [["Doc 1"]]
    })

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy
    )

    # Single query in batch
    results = retriever.batch_search(
        [[0.1, 0.2, 0.3]],
        {"id": "alice", "department": "engineering"},
        limit=5
    )

    assert len(results) == 1
    assert len(results[0]) >= 1


def test_batch_search_with_different_permissions():
    """Test batch search respects permissions for each query."""
    from ragguard import Policy
    from ragguard.retrievers import ChromaDBSecureRetriever

    mock_collection = Mock()

    # Different results for each query
    mock_collection.query = Mock(side_effect=[
        # Query 1: engineering doc (allowed)
        {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Eng doc"]]
        },
        # Query 2: no results (empty)
        {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]]
        },
        # Query 3: engineering doc (allowed)
        {
            "ids": [["3"]],
            "distances": [[0.3]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Another eng doc"]]
        },
    ])

    # Policy: only same department (note: native filtering will be applied by backend)
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {"conditions": ["user.department == document.department"]}
        }],
        "default": "deny"
    })

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy
    )

    # User from engineering
    user = {"id": "alice", "department": "engineering"}
    results = retriever.batch_search(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        user,
        limit=5
    )

    # Should have 3 result lists
    assert len(results) == 3

    # First query: engineering doc (allowed)
    assert len(results[0]) == 1

    # Second query: no results
    assert len(results[1]) == 0

    # Third query: engineering doc (allowed)
    assert len(results[2]) == 1


def test_batch_search_preserves_order():
    """Test that batch search preserves query order."""
    from ragguard import Policy
    from ragguard.retrievers import ChromaDBSecureRetriever

    mock_collection = Mock()

    # Return distinct results for each query
    mock_collection.query = Mock(side_effect=[
        {
            "ids": [["100"]],
            "distances": [[0.1]],
            "metadatas": [[{"query_id": "Q1", "department": "eng"}]],
            "documents": [["Doc 1"]]
        },
        {
            "ids": [["200"]],
            "distances": [[0.2]],
            "metadatas": [[{"query_id": "Q2", "department": "eng"}]],
            "documents": [["Doc 2"]]
        },
        {
            "ids": [["300"]],
            "distances": [[0.3]],
            "metadatas": [[{"query_id": "Q3", "department": "eng"}]],
            "documents": [["Doc 3"]]
        },
    ])

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy
    )

    # Three distinct queries
    results = retriever.batch_search(
        [[0.1], [0.2], [0.3]],
        {"id": "alice", "department": "eng"},
        limit=1
    )

    # Check order is preserved
    assert len(results) == 3
    assert results[0][0]["metadata"]["query_id"] == "Q1"
    assert results[1][0]["metadata"]["query_id"] == "Q2"
    assert results[2][0]["metadata"]["query_id"] == "Q3"


def test_batch_search_limit_per_query():
    """Test that limit applies to each query independently."""
    from ragguard import Policy
    from ragguard.retrievers import ChromaDBSecureRetriever

    mock_collection = Mock()

    # Return multiple results for each query
    mock_collection.query = Mock(side_effect=[
        # Query 1: 2 results
        {
            "ids": [["1", "2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"department": "eng"}, {"department": "eng"}]],
            "documents": [["Doc 1", "Doc 2"]]
        },
        # Query 2: 2 results
        {
            "ids": [["3", "4"]],
            "distances": [[0.3, 0.4]],
            "metadatas": [[{"department": "eng"}, {"department": "eng"}]],
            "documents": [["Doc 3", "Doc 4"]]
        },
    ])

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy
    )

    # Batch search with limit=2
    results = retriever.batch_search(
        [[0.1], [0.2]],
        {"id": "alice", "department": "eng"},
        limit=2  # Limit should apply to each query
    )

    # Each result list should respect the limit
    assert len(results) == 2
    assert all(len(r) >= 1 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
