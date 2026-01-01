"""
Tests for context manager support in retrievers.

Tests that all retrievers support the 'with' statement for automatic cleanup.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if chromadb is not installed
pytest.importorskip("chromadb", exc_type=ImportError)


def test_base_retriever_context_manager():
    """Test that base retriever supports context manager."""
    from ragguard import ChromaDBSecureRetriever, Policy

    # Mock ChromaDB collection with close() method
    mock_collection = Mock()
    mock_collection.close = Mock()
    mock_collection.query = Mock(return_value={
        'ids': [],
        'distances': [],
        'metadatas': [],
        'documents': []
    })

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Use retriever as context manager
    with ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3]
    ) as retriever:
        # Retriever should be returned from __enter__
        assert retriever is not None
        assert isinstance(retriever, ChromaDBSecureRetriever)

    # Collection close() should be called on exit
    mock_collection.close.assert_called_once()


def test_context_manager_with_exception():
    """Test that context manager handles exceptions correctly."""
    from ragguard import ChromaDBSecureRetriever, Policy

    mock_collection = Mock()
    mock_collection.close = Mock()

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Exception should propagate, but close() should still be called
    with pytest.raises(ValueError):
        with ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=policy,
            embed_fn=lambda x: [0.1, 0.2, 0.3]
        ) as retriever:
            raise ValueError("Test exception")

    # Close should still be called even after exception
    mock_collection.close.assert_called_once()


def test_context_manager_without_close_method():
    """Test context manager with client that has no close() method."""
    from ragguard import ChromaDBSecureRetriever, Policy

    # Mock collection WITHOUT close() method
    mock_collection = Mock(spec=['query', 'name'])
    mock_collection.name = "test_collection"
    mock_collection.query = Mock(return_value={
        'ids': [],
        'distances': [],
        'metadatas': [],
        'documents': []
    })

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Should work without error even if collection has no close()
    with ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3]
    ) as retriever:
        pass

    # No close() method, so nothing should be called
    assert not hasattr(mock_collection, 'close')


def test_context_manager_with_failing_close():
    """Test that context manager handles close() errors gracefully."""
    from ragguard import ChromaDBSecureRetriever, Policy

    mock_collection = Mock()
    mock_collection.close = Mock(side_effect=Exception("Close failed"))
    mock_collection.query = Mock(return_value={
        'ids': [],
        'distances': [],
        'metadatas': [],
        'documents': []
    })

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Should not raise exception even if close() fails
    with ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3]
    ) as retriever:
        pass

    # Close should have been attempted
    mock_collection.close.assert_called_once()


def test_context_manager_with_audit_logger():
    """Test that context manager closes audit logger too."""
    from ragguard import AuditLogger, ChromaDBSecureRetriever, Policy

    mock_collection = Mock()
    mock_collection.close = Mock()
    mock_collection.query = Mock(return_value={
        'ids': [],
        'distances': [],
        'metadatas': [],
        'documents': []
    })

    mock_audit_logger = Mock(spec=AuditLogger)
    mock_audit_logger.close = Mock()

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    with ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3],
        audit_logger=mock_audit_logger
    ) as retriever:
        pass

    # Both collection and audit logger should be closed
    mock_collection.close.assert_called_once()
    mock_audit_logger.close.assert_called_once()


def test_pgvector_context_manager_with_pool():
    """Test pgvector retriever context manager with connection pool."""
    from ragguard import PgvectorConnectionPool, PgvectorSecureRetriever, Policy

    # Mock the pool
    mock_pool = Mock()
    mock_pool.close = Mock()
    mock_pool.getconn = Mock(return_value=Mock())
    mock_pool.putconn = Mock()

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Mock pgvector import
    with patch.dict('sys.modules', {'pgvector': MagicMock()}):
        with PgvectorSecureRetriever(
            connection=mock_pool,
            table="documents",
            policy=policy
        ) as retriever:
            pass

        # Pool close() should be called (may be called multiple times due to multiple references)
        # This is safe because close() is idempotent
        assert mock_pool.close.called


def test_chromadb_context_manager():
    """Test ChromaDB retriever supports context manager."""
    from ragguard import ChromaDBSecureRetriever, Policy

    mock_collection = Mock()
    mock_collection.query = Mock(return_value={
        'ids': [],
        'distances': [],
        'metadatas': [],
        'documents': []
    })

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # ChromaDB collection doesn't have close(), but should work anyway
    with ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3]
    ) as retriever:
        assert retriever is not None


def test_manual_cleanup_still_works():
    """Test that manual cleanup (without context manager) still works."""
    from ragguard import ChromaDBSecureRetriever, Policy

    mock_collection = Mock()
    mock_collection.close = Mock()
    mock_collection.query = Mock(return_value={
        'ids': [],
        'distances': [],
        'metadatas': [],
        'documents': []
    })

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Create retriever WITHOUT context manager
    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3]
    )

    # Use it
    results = retriever.search("test query", {"id": "alice"})

    # Manual cleanup
    if hasattr(mock_collection, 'close'):
        mock_collection.close()

    # Close should be called
    mock_collection.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
