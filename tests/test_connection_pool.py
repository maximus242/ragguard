"""
Tests for connection pooling.

Tests PgvectorConnectionPool and ManagedConnection for PostgreSQL/pgvector.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Skip all tests if psycopg2 is not installed
pytest.importorskip("psycopg2")


def test_managed_connection_with_direct_connection():
    """Test ManagedConnection with a direct database connection."""
    from ragguard.connection_pool import ManagedConnection

    # Mock connection (specifically NOT a pool - no getconn/putconn callable methods)
    mock_conn = Mock(spec=['cursor', 'commit', 'rollback', 'close'])

    # Create managed connection
    managed = ManagedConnection(mock_conn)

    # Should not be identified as pool
    assert not managed._is_pool

    # Get should return the connection directly
    conn = managed.get()
    assert conn is mock_conn

    # Release should do nothing (user manages lifecycle)
    managed.release()


def test_managed_connection_with_pool():
    """Test ManagedConnection with a connection pool."""
    from ragguard.connection_pool import ManagedConnection

    # Mock pool with getconn/putconn methods
    mock_pool = Mock()
    mock_pool.getconn = Mock(return_value=Mock())
    mock_pool.putconn = Mock()

    # Create managed connection
    managed = ManagedConnection(mock_pool)

    # Should be identified as pool
    assert managed._is_pool

    # Get should call getconn
    conn = managed.get()
    assert mock_pool.getconn.called
    assert conn is not None

    # Second get should return same connection (cached)
    conn2 = managed.get()
    assert conn is conn2
    assert mock_pool.getconn.call_count == 1

    # Release should call putconn
    managed.release()
    assert mock_pool.putconn.called
    mock_pool.putconn.assert_called_once_with(conn)

    # After release, _current_conn should be None
    assert managed._current_conn is None


def test_managed_connection_context_manager():
    """Test ManagedConnection context manager."""
    from ragguard.connection_pool import ManagedConnection

    # Mock pool
    mock_pool = Mock()
    mock_conn = Mock()
    mock_pool.getconn = Mock(return_value=mock_conn)
    mock_pool.putconn = Mock()

    managed = ManagedConnection(mock_pool)

    # Use context manager
    with managed.use() as conn:
        assert conn is mock_conn
        assert mock_pool.getconn.called

    # Should release after exiting context
    assert mock_pool.putconn.called


def test_pgvector_pool_psycopg2_creation():
    """Test creating a PgvectorConnectionPool with psycopg2."""
    from ragguard import PgvectorConnectionPool

    with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
        mock_pool_instance = Mock()
        mock_pool_class.return_value = mock_pool_instance

        pool = PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            min_size=2,
            max_size=10,
            driver="psycopg2"
        )

        # Should create SimpleConnectionPool
        mock_pool_class.assert_called_once_with(
            2, 10, "postgresql://localhost/test"
        )

        assert pool._pool is mock_pool_instance


def test_pgvector_pool_psycopg3_creation():
    """Test creating a PgvectorConnectionPool with psycopg3."""
    from ragguard import PgvectorConnectionPool

    # Mock the psycopg_pool module since it may not be installed
    mock_pool_class = Mock()
    mock_pool_instance = Mock()
    mock_pool_class.return_value = mock_pool_instance

    mock_psycopg_pool = MagicMock()
    mock_psycopg_pool.ConnectionPool = mock_pool_class

    with patch.dict('sys.modules', {'psycopg_pool': mock_psycopg_pool}):
        pool = PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            min_size=2,
            max_size=10,
            driver="psycopg3"
        )

        # Should create ConnectionPool
        mock_pool_class.assert_called_once_with(
            "postgresql://localhost/test",
            min_size=2,
            max_size=10
        )

        assert pool._pool is mock_pool_instance


def test_pgvector_pool_invalid_driver():
    """Test that invalid driver raises error."""
    from ragguard import PgvectorConnectionPool

    with pytest.raises(ValueError, match="Unsupported driver"):
        PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            driver="invalid"
        )


def test_pgvector_pool_getconn_psycopg2():
    """Test getting connection from psycopg2 pool."""
    from ragguard import PgvectorConnectionPool

    with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
        mock_pool_instance = Mock()
        mock_conn = Mock()
        mock_pool_instance.getconn = Mock(return_value=mock_conn)
        mock_pool_class.return_value = mock_pool_instance

        pool = PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            driver="psycopg2"
        )

        conn = pool.getconn()

        assert conn is mock_conn
        mock_pool_instance.getconn.assert_called_once_with(None)


def test_pgvector_pool_putconn_psycopg2():
    """Test returning connection to psycopg2 pool."""
    from ragguard import PgvectorConnectionPool

    with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
        mock_pool_instance = Mock()
        mock_conn = Mock()
        mock_pool_instance.putconn = Mock()
        mock_pool_class.return_value = mock_pool_instance

        pool = PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            driver="psycopg2"
        )

        pool.putconn(mock_conn)

        mock_pool_instance.putconn.assert_called_once_with(mock_conn, None, False)


def test_pgvector_pool_context_manager():
    """Test PgvectorConnectionPool context manager."""
    from ragguard import PgvectorConnectionPool

    with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
        mock_pool_instance = Mock()
        mock_conn = Mock()
        mock_pool_instance.getconn = Mock(return_value=mock_conn)
        mock_pool_instance.putconn = Mock()
        mock_pool_instance.closeall = Mock()
        mock_pool_class.return_value = mock_pool_instance

        with PgvectorConnectionPool(dsn="postgresql://localhost/test", driver="psycopg2") as pool:
            conn = pool.getconn()
            assert conn is mock_conn

        # Should close pool on exit
        mock_pool_instance.closeall.assert_called_once()


def test_pgvector_pool_get_connection_context():
    """Test get_connection() context manager."""
    from ragguard import PgvectorConnectionPool

    with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
        mock_pool_instance = Mock()
        mock_conn = Mock()
        mock_pool_instance.getconn = Mock(return_value=mock_conn)
        mock_pool_instance.putconn = Mock()
        mock_pool_class.return_value = mock_pool_instance

        pool = PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            driver="psycopg2"
        )

        with pool.get_connection() as conn:
            assert conn is mock_conn
            mock_pool_instance.getconn.assert_called_once()

        # Should return connection to pool
        mock_pool_instance.putconn.assert_called_once_with(mock_conn, None, False)


def test_pgvector_pool_close():
    """Test closing the pool."""
    from ragguard import PgvectorConnectionPool

    with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
        mock_pool_instance = Mock()
        mock_pool_instance.closeall = Mock()
        mock_pool_class.return_value = mock_pool_instance

        pool = PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            driver="psycopg2"
        )

        pool.close()

        mock_pool_instance.closeall.assert_called_once()
        assert pool._pool is None


def test_pgvector_pool_getconn_after_close():
    """Test that getting connection after close raises error."""
    from ragguard import PgvectorConnectionPool

    with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
        mock_pool_instance = Mock()
        mock_pool_class.return_value = mock_pool_instance

        pool = PgvectorConnectionPool(
            dsn="postgresql://localhost/test",
            driver="psycopg2"
        )

        pool.close()

        with pytest.raises(RuntimeError, match="Connection pool is closed"):
            pool.getconn()


def test_pgvector_retriever_with_pool():
    """Test PgvectorSecureRetriever with connection pool."""
    from unittest.mock import MagicMock

    from ragguard import PgvectorConnectionPool, PgvectorSecureRetriever, Policy

    # Mock the pool
    mock_pool = Mock()
    mock_conn = Mock()
    mock_cursor = Mock()

    # Setup pool behavior
    mock_pool.getconn = Mock(return_value=mock_conn)
    mock_pool.putconn = Mock()

    # Setup connection behavior
    mock_conn.cursor = Mock(return_value=mock_cursor)
    mock_cursor.description = [('id',), ('content',), ('embedding',), ('distance',)]
    mock_cursor.fetchall = Mock(return_value=[
        ('doc1', 'Content 1', [0.1, 0.2], 0.5)
    ])
    mock_cursor.close = Mock()

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Mock pgvector import
    with patch.dict('sys.modules', {'pgvector': MagicMock()}):
        # Create retriever with pool
        retriever = PgvectorSecureRetriever(
            connection=mock_pool,
            table="documents",
            policy=policy
        )

        # Search
        query_vector = [0.1, 0.2, 0.3]
        user = {"id": "alice"}
        results = retriever.search(query_vector, user, limit=10)

        # Should get connection from pool
        mock_pool.getconn.assert_called()

        # Should return connection to pool
        mock_pool.putconn.assert_called()

        # Should have results
        assert len(results) == 1


def test_pgvector_retriever_with_direct_connection():
    """Test PgvectorSecureRetriever still works with direct connection."""
    from unittest.mock import MagicMock

    from ragguard import PgvectorSecureRetriever, Policy

    # Mock direct connection (not a pool)
    mock_conn = Mock()
    mock_cursor = Mock()

    # No getconn/putconn methods (not a pool)
    del mock_conn.getconn
    del mock_conn.putconn

    # Setup connection behavior
    mock_conn.cursor = Mock(return_value=mock_cursor)
    mock_cursor.description = [('id',), ('content',), ('embedding',), ('distance',)]
    mock_cursor.fetchall = Mock(return_value=[
        ('doc1', 'Content 1', [0.1, 0.2], 0.5)
    ])
    mock_cursor.close = Mock()

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Mock pgvector import
    with patch.dict('sys.modules', {'pgvector': MagicMock()}):
        # Create retriever with direct connection
        retriever = PgvectorSecureRetriever(
            connection=mock_conn,
            table="documents",
            policy=policy
        )

        # Search
        query_vector = [0.1, 0.2, 0.3]
        user = {"id": "alice"}
        results = retriever.search(query_vector, user, limit=10)

        # Should use connection directly
        mock_conn.cursor.assert_called()

        # Should have results
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
