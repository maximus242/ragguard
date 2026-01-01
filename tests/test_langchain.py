"""
Tests for LangChain integration.

These tests verify that RAGGuard integrates correctly with LangChain.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


def test_langchain_retriever_initialization():
    """Test LangChainSecureRetriever initialization."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainSecureRetriever

    # Use in-memory Qdrant client
    client = QdrantClient(":memory:")

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_public',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='documents',
        policy=policy,
        embedding_function=mock_embed
    )

    assert retriever.qdrant_client == client
    assert retriever.collection == 'documents'
    assert retriever.policy == policy
    assert retriever.embedding_function == mock_embed
    assert retriever.current_user is None


def test_langchain_retriever_set_user():
    """Test LangChainSecureRetriever.set_user method."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainSecureRetriever

    client = QdrantClient(":memory:")

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_public',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='documents',
        policy=policy,
        embedding_function=mock_embed
    )

    user = {'id': 'alice', 'department': 'engineering'}
    result = retriever.set_user(user)

    assert retriever.current_user == user
    assert result == retriever  # Should return self for chaining


def test_langchain_retriever_no_user_raises_error():
    """Test LangChainSecureRetriever raises error without user context."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient

    from ragguard import Policy
    from ragguard.exceptions import RetrieverError
    from ragguard.integrations.langchain import LangChainSecureRetriever

    client = QdrantClient(":memory:")

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='documents',
        policy=policy,
        embedding_function=mock_embed
    )

    # Should raise error if no user context
    with pytest.raises(RetrieverError, match="User context required"):
        retriever.invoke("test query")


def test_langchain_retriever_with_audit_logger():
    """Test LangChainSecureRetriever with audit logging."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient

    from ragguard import AuditLogger, Policy
    from ragguard.integrations.langchain import LangChainSecureRetriever

    client = QdrantClient(":memory:")

    audit_logger = AuditLogger()

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='documents',
        policy=policy,
        embedding_function=mock_embed,
        audit_logger=audit_logger
    )

    assert retriever.audit_logger == audit_logger


def test_langchain_pgvector_retriever_initialization():
    """Test LangChainPgvectorSecureRetriever initialization."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("pgvector")

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainPgvectorSecureRetriever

    mock_conn = Mock()

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_public',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainPgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embedding_function=mock_embed
    )

    assert retriever.connection == mock_conn
    assert retriever.table == 'documents'
    assert retriever.policy == policy
    assert retriever.current_user is None


def test_langchain_pgvector_retriever_set_user():
    """Test LangChainPgvectorSecureRetriever.set_user method."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("pgvector")

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainPgvectorSecureRetriever

    mock_conn = Mock()

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_public',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainPgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embedding_function=mock_embed
    )

    user = {'id': 'alice'}
    result = retriever.set_user(user)

    assert retriever.current_user == user
    assert result == retriever


def test_langchain_pgvector_retriever_no_user_raises_error():
    """Test LangChainPgvectorSecureRetriever raises error without user context."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("pgvector")

    from ragguard import Policy
    from ragguard.exceptions import RetrieverError
    from ragguard.integrations.langchain import LangChainPgvectorSecureRetriever

    mock_conn = Mock()

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainPgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embedding_function=mock_embed
    )

    # Should raise error if no user context
    with pytest.raises(RetrieverError, match="User context required"):
        retriever.invoke("test query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
