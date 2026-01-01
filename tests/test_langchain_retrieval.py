"""
Tests for LangChain integration - actual retrieval tests.

These tests cover the document retrieval and conversion logic.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


def test_langchain_retriever_actual_retrieval():
    """Test LangChainSecureRetriever actually retrieves and converts documents."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainSecureRetriever

    # Create in-memory Qdrant client with collection
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_docs",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    # Add test documents
    client.upsert(
        collection_name="test_docs",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3],
                payload={"text": "Document about engineering", "department": "engineering"}
            ),
            PointStruct(
                id=2,
                vector=[0.2, 0.3, 0.4],
                payload={"text": "Document about finance", "department": "finance"}
            ),
        ],
    )

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='test_docs',
        policy=policy,
        embedding_function=embed
    )

    user = {'id': 'alice', 'department': 'engineering'}

    # Test retrieval
    docs = retriever._get_relevant_documents("test query", user=user)

    # Should get both documents (policy allows all)
    assert len(docs) == 2
    assert any('engineering' in doc.page_content.lower() for doc in docs)
    assert any('finance' in doc.page_content.lower() for doc in docs)
    assert all('score' in doc.metadata for doc in docs)
    assert all('department' in doc.metadata for doc in docs)


def test_langchain_retriever_with_content_field():
    """Test LangChainSecureRetriever handles 'content' field instead of 'text'."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainSecureRetriever

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_docs",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    # Add document with 'content' instead of 'text'
    client.upsert(
        collection_name="test_docs",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Document with content field", "type": "article"}
            ),
        ],
    )

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='test_docs',
        policy=policy,
        embedding_function=embed
    )

    user = {'id': 'alice'}
    docs = retriever._get_relevant_documents("test", user=user)

    assert len(docs) == 1
    assert docs[0].page_content == "Document with content field"
    assert docs[0].metadata['type'] == 'article'


def test_langchain_retriever_with_k_parameter():
    """Test LangChainSecureRetriever handles k parameter for limit."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainSecureRetriever

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_docs",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    # Add multiple documents
    points = [
        PointStruct(
            id=i,
            vector=[0.1, 0.2, 0.3],
            payload={"text": f"Document {i}", "visibility": "public"}
        )
        for i in range(10)
    ]
    client.upsert(collection_name="test_docs", points=points)

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='test_docs',
        policy=policy,
        embedding_function=embed
    )

    user = {'id': 'alice'}

    # Test with k=3
    docs = retriever._get_relevant_documents("test", user=user, k=3)
    assert len(docs) == 3


def test_langchain_retriever_with_search_kwargs():
    """Test LangChainSecureRetriever merges search_kwargs."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainSecureRetriever

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_docs",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    client.upsert(
        collection_name="test_docs",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3],
                payload={"text": "Test document"}
            ),
        ],
    )

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainSecureRetriever(
        qdrant_client=client,
        collection='test_docs',
        policy=policy,
        embedding_function=embed
    )

    # Set default search kwargs
    retriever.search_kwargs = {'score_threshold': 0.5}

    user = {'id': 'alice'}
    docs = retriever._get_relevant_documents("test", user=user, k=5)

    assert len(docs) <= 5


def test_langchain_pgvector_retriever_actual_retrieval():
    """Test LangChainPgvectorSecureRetriever retrieves and converts documents."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("pgvector")

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainPgvectorSecureRetriever

    mock_conn = Mock(spec=["cursor"])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    # Mock query results with different field names
    mock_cursor.fetchall.return_value = [
        {'id': 1, 'text': 'Engineering document', 'department': 'engineering', 'distance': 0.1},
        {'id': 2, 'content': 'Finance document', 'department': 'finance', 'distance': 0.2}
    ]
    mock_cursor.description = [('id',), ('text',), ('department',), ('distance',)]

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainPgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embedding_function=embed
    )

    user = {'id': 'alice'}
    docs = retriever._get_relevant_documents("test query", user=user)

    # Should convert both documents
    assert len(docs) == 2
    assert docs[0].page_content == 'Engineering document'
    assert docs[0].metadata['department'] == 'engineering'


def test_langchain_pgvector_retriever_with_k_parameter():
    """Test LangChainPgvectorSecureRetriever handles k parameter."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("pgvector")

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainPgvectorSecureRetriever

    mock_conn = Mock(spec=["cursor"])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = [
        {'id': 1, 'text': 'Doc 1', 'distance': 0.1},
        {'id': 2, 'text': 'Doc 2', 'distance': 0.2},
        {'id': 3, 'text': 'Doc 3', 'distance': 0.3}
    ]
    mock_cursor.description = [('id',), ('text',), ('distance',)]

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainPgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embedding_function=embed
    )

    user = {'id': 'alice'}
    docs = retriever._get_relevant_documents("test", user=user, k=2)

    # Should still get all results from mock (limit is passed to SQL, not here)
    assert len(docs) == 3


def test_langchain_pgvector_retriever_content_field():
    """Test LangChainPgvectorSecureRetriever handles content field."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("pgvector")

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainPgvectorSecureRetriever

    mock_conn = Mock(spec=["cursor"])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    # Use 'content' field instead of 'text'
    mock_cursor.fetchall.return_value = [
        {'id': 1, 'content': 'Document content', 'distance': 0.1}
    ]
    mock_cursor.description = [('id',), ('content',), ('distance',)]

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainPgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embedding_function=embed,
        text_column='content'
    )

    user = {'id': 'alice'}
    docs = retriever._get_relevant_documents("test", user=user)

    assert len(docs) == 1
    assert docs[0].page_content == 'Document content'


def test_langchain_pgvector_retriever_custom_text_column():
    """Test LangChainPgvectorSecureRetriever with custom text column."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("pgvector")

    from ragguard import Policy
    from ragguard.integrations.langchain import LangChainPgvectorSecureRetriever

    mock_conn = Mock(spec=["cursor"])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = [
        {'id': 1, 'body': 'Custom text column', 'distance': 0.1}
    ]
    mock_cursor.description = [('id',), ('body',), ('distance',)]

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

    def embed(text):
        return [0.1, 0.2, 0.3]

    retriever = LangChainPgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embedding_function=embed,
        text_column='body'
    )

    user = {'id': 'alice'}
    docs = retriever._get_relevant_documents("test", user=user)

    assert len(docs) == 1
    assert docs[0].page_content == 'Custom text column'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
