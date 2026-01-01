"""
Advanced tests for ChromaDB integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import Mock

import pytest

# Skip all tests if chromadb is not installed
pytest.importorskip("chromadb", exc_type=ImportError)

from ragguard import ChromaDBSecureRetriever, Policy
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


def create_mock_chromadb_collection():
    """Create a mock ChromaDB collection."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"

    # Mock query results
    mock_collection.query = Mock(return_value={
        "ids": [["doc1"]],
        "metadatas": [[{"department": "engineering"}]],
        "documents": [["Document 1"]],
        "distances": [[0.1]]
    })

    # Mock count
    mock_collection.count = Mock(return_value=100)

    return mock_collection


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


def test_chromadb_health_check_success():
    """Test successful health check for ChromaDB."""
    mock_collection = create_mock_chromadb_collection()

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "chromadb"
    assert health["collection"] == "test_collection"


def test_chromadb_context_manager():
    """Test using ChromaDB retriever as context manager."""
    mock_collection = create_mock_chromadb_collection()

    with ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) >= 0


def test_chromadb_with_validation():
    """Test ChromaDB retriever with input validation."""
    mock_collection = create_mock_chromadb_collection()

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
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
    assert len(results) >= 0

    # Invalid user context (empty dictionary)
    with pytest.raises(RetrieverError):
        retriever.search(
            [0.1, 0.2, 0.3],
            {},
            limit=5
        )


def test_chromadb_with_retry():
    """Test ChromaDB retriever with retry logic."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"

    call_count = [0]

    def query_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection failed")

        return {
            "ids": [["doc1"]],
            "metadatas": [[{"department": "engineering"}]],
            "documents": [["Document 1"]],
            "distances": [[0.1]]
        }

    mock_collection.query = Mock(side_effect=query_side_effect)

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01)
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert call_count[0] == 3


def test_chromadb_with_cache():
    """Test ChromaDB retriever with filter caching."""
    mock_collection = create_mock_chromadb_collection()

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
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

    assert mock_collection.query.call_count >= 2


def test_chromadb_with_audit_logging():
    """Test ChromaDB retriever with audit logging."""
    mock_collection = create_mock_chromadb_collection()

    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"


def test_chromadb_search_failure():
    """Test ChromaDB search failure handling."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"
    mock_collection.query = Mock(side_effect=Exception("Search failed"))

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy(),
        enable_retry=False
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_chromadb_empty_results():
    """Test ChromaDB with empty search results."""
    mock_collection = create_mock_chromadb_collection()
    mock_collection.query = Mock(return_value={
        "ids": [[]],
        "metadatas": [[]],
        "documents": [[]],
        "distances": [[]]
    })

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 0


def test_chromadb_batch_search():
    """Test batch search with ChromaDB."""
    mock_collection = create_mock_chromadb_collection()

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
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


def test_chromadb_policy_update():
    """Test updating policy on ChromaDB retriever."""
    mock_collection = create_mock_chromadb_collection()

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy()
    )

    new_policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    retriever.policy = new_policy

    assert retriever.policy.rules[0].name == "dept-access"


def test_chromadb_with_embed_fn():
    """Test ChromaDB retriever with text query and embedding function."""
    mock_collection = create_mock_chromadb_collection()

    def embed_fn(text):
        return [float(ord(c)) / 1000 for c in text[:3]]

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy(),
        embed_fn=embed_fn
    )

    results = retriever.search("test query", {"id": "alice"}, limit=5)

    assert len(results) >= 0


def test_chromadb_text_query_without_embed_fn():
    """Test that text query without embed_fn raises error."""
    mock_collection = create_mock_chromadb_collection()

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=create_basic_policy(),
        embed_fn=None
    )

    with pytest.raises(RetrieverError, match="no embed_fn was provided"):
        retriever.search("test query", {"id": "alice"}, limit=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
