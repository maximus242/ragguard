"""
Tests for pgvector retriever.

These tests use mocking to avoid requiring an actual PostgreSQL database.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


def test_pgvector_retriever_initialization():
    """Test PgvectorSecureRetriever initialization."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

    # Mock PostgreSQL connection
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

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy
    )

    assert retriever.connection == mock_conn
    assert retriever.table == 'documents'
    assert retriever.embedding_column == 'embedding'
    assert retriever.backend_name == 'pgvector'


def test_pgvector_retriever_custom_embedding_column():
    """Test PgvectorSecureRetriever with custom embedding column."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

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

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='my_docs',
        policy=policy,
        embedding_column='vector_column'
    )

    assert retriever.embedding_column == 'vector_column'
    assert retriever.table == 'my_docs'


def test_pgvector_retriever_search_with_filter():
    """Test PgvectorSecureRetriever search with permission filter."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

    mock_conn = Mock(spec=['cursor'])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    # Mock query results
    mock_cursor.fetchall.return_value = [
        {'id': 1, 'content': 'doc1', 'department': 'engineering', 'distance': 0.1},
        {'id': 2, 'content': 'doc2', 'department': 'engineering', 'distance': 0.2}
    ]
    mock_cursor.description = [
        ('id',), ('content',), ('department',), ('distance',)
    ]

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'department_access',
                'allow': {
                    'conditions': ['user.department == document.department']
                }
            }
        ],
        'default': 'deny'
    })

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy
    )

    user = {'id': 'alice', 'department': 'engineering'}
    query_vector = [0.1, 0.2, 0.3]

    results = retriever.search(
        query=query_vector,
        user=user,
        limit=5
    )

    # Verify search was executed
    assert mock_cursor.execute.called
    assert mock_cursor.fetchall.called
    assert len(results) == 2


def test_pgvector_retriever_search_with_text():
    """Test PgvectorSecureRetriever search with text query."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

    mock_conn = Mock(spec=['cursor'])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = []
    mock_cursor.description = []

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

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy,
        embed_fn=mock_embed
    )

    user = {'id': 'alice'}

    results = retriever.search(
        query="test query",
        user=user,
        limit=5
    )

    # Verify search was executed
    assert mock_cursor.execute.called


def test_pgvector_retriever_sql_generation():
    """Test PgvectorSecureRetriever generates correct SQL."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

    mock_conn = Mock(spec=['cursor'])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = []
    mock_cursor.description = []

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

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='my_table',
        policy=policy,
        embedding_column='my_embedding'
    )

    user = {'id': 'alice'}
    query_vector = [0.1, 0.2, 0.3]

    retriever.search(
        query=query_vector,
        user=user,
        limit=10
    )

    # Verify SQL contains expected components
    call_args = mock_cursor.execute.call_args
    sql = call_args[0][0]

    assert 'my_table' in sql
    assert 'my_embedding' in sql
    assert 'WHERE' in sql
    assert 'ORDER BY distance' in sql
    assert 'LIMIT' in sql


def test_pgvector_retriever_different_distance_operators():
    """Test PgvectorSecureRetriever with different distance operators."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

    mock_conn = Mock(spec=['cursor'])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = []
    mock_cursor.description = []

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

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy
    )

    user = {'id': 'alice'}
    query_vector = [0.1, 0.2, 0.3]

    # Test with cosine distance
    retriever.search(
        query=query_vector,
        user=user,
        limit=5,
        distance_op='<=>'
    )

    call_args = mock_cursor.execute.call_args
    sql = call_args[0][0]

    assert '<=>' in sql


def test_pgvector_retriever_dict_cursor():
    """Test PgvectorSecureRetriever with DictCursor."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

    mock_conn = Mock(spec=['cursor'])  # Specify spec to avoid pool detection
    mock_dict_cursor = Mock()

    # Mock DictCursor
    with patch('psycopg2.extras.DictCursor') as mock_dict_cursor_class:
        mock_conn.cursor.return_value = mock_dict_cursor

        mock_dict_cursor.fetchall.return_value = [
            {'id': 1, 'content': 'doc1', 'distance': 0.1}
        ]
        mock_dict_cursor.description = [('id',), ('content',), ('distance',)]

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

        retriever = PgvectorSecureRetriever(
            connection=mock_conn,
            table='documents',
            policy=policy
        )

        user = {'id': 'alice'}
        query_vector = [0.1, 0.2, 0.3]

        results = retriever.search(
            query=query_vector,
            user=user,
            limit=5
        )

        assert mock_dict_cursor.execute.called


def test_pgvector_retriever_regular_cursor_conversion():
    """Test PgvectorSecureRetriever converts regular cursor results to dicts."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy

    mock_conn = Mock(spec=['cursor'])  # Specify spec to avoid pool detection
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    # Return tuple results (regular cursor)
    mock_cursor.fetchall.return_value = [
        (1, 'doc1', 'engineering', 0.1),
        (2, 'doc2', 'engineering', 0.2)
    ]
    mock_cursor.description = [
        ('id',), ('content',), ('department',), ('distance',)
    ]

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

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy
    )

    user = {'id': 'alice'}
    query_vector = [0.1, 0.2, 0.3]

    results = retriever.search(
        query=query_vector,
        user=user,
        limit=5
    )

    # Results should be converted to dicts
    assert len(results) == 2
    assert results[0]['id'] == 1
    assert results[0]['content'] == 'doc1'
    assert results[1]['id'] == 2


def test_pgvector_retriever_error_handling():
    """Test PgvectorSecureRetriever handles database errors."""
    pytest.importorskip("pgvector")

    from ragguard import PgvectorSecureRetriever, Policy
    from ragguard.exceptions import RetrieverError

    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor

    # Simulate database error
    mock_cursor.execute.side_effect = Exception("Database connection failed")

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

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table='documents',
        policy=policy
    )

    user = {'id': 'alice'}
    query_vector = [0.1, 0.2, 0.3]

    with pytest.raises(RetrieverError):
        retriever.search(
            query=query_vector,
            user=user,
            limit=5
        )


def test_pgvector_retriever_missing_dependency():
    """Test PgvectorSecureRetriever raises error when pgvector not installed."""
    from ragguard import Policy
    from ragguard.exceptions import RetrieverError

    # Mock pgvector not being installed
    with patch.dict('sys.modules', {'pgvector': None}):
        with pytest.raises(RetrieverError, match="pgvector not installed"):
            # This will fail during import check
            from ragguard.retrievers.pgvector import PgvectorSecureRetriever

            mock_conn = Mock()
            policy = Policy.from_dict({
                'version': '1',
                'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
                'default': 'deny'
            })

            PgvectorSecureRetriever(
                connection=mock_conn,
                table='documents',
                policy=policy
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
