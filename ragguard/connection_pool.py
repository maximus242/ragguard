"""
Connection pooling utilities for RAGGuard retrievers.

Provides connection pool wrappers for backends that benefit from pooling,
especially PostgreSQL/pgvector which uses stateful database connections.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Optional, Protocol, Union

logger = logging.getLogger(__name__)


class ConnectionPoolProtocol(Protocol):
    """Protocol for connection pool implementations."""

    def getconn(self) -> Any:
        """Get a connection from the pool."""
        ...

    def putconn(self, conn: Any) -> None:
        """Return a connection to the pool."""
        ...

    def closeall(self) -> None:
        """Close all connections in the pool."""
        ...


class PgvectorConnectionPool:
    """
    Connection pool wrapper for PostgreSQL/pgvector.

    Supports both psycopg2 and psycopg3 connection pools.
    Provides thread-safe connection management with automatic cleanup.

    Example:
        >>> # Using psycopg2
        >>> from ragguard import PgvectorConnectionPool
        >>>
        >>> pool = PgvectorConnectionPool(
        ...     dsn="postgresql://localhost/mydb",
        ...     min_size=2,
        ...     max_size=10,
        ...     driver="psycopg2"
        ... )
        >>>
        >>> # Use with retriever
        >>> retriever = PgvectorSecureRetriever(
        ...     connection=pool,
        ...     table="documents",
        ...     policy=policy
        ... )
        >>>
        >>> # Or use context manager
        >>> with pool.get_connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT 1")
        >>>
        >>> # Clean up when done
        >>> pool.close()
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_size: int = 2,
        max_size: int = 10,
        driver: str = "psycopg2",
        **kwargs
    ):
        """
        Initialize connection pool.

        Args:
            dsn: PostgreSQL connection string (e.g., "postgresql://localhost/mydb")
            min_size: Minimum number of connections to maintain
            max_size: Maximum number of connections to create
            driver: Database driver ("psycopg2" or "psycopg3")
            **kwargs: Additional connection parameters passed to the pool
        """
        # Validate connection pool size parameters
        if not isinstance(min_size, int) or min_size < 0:
            raise ValueError(f"min_size must be a non-negative integer, got {min_size}")
        if not isinstance(max_size, int) or max_size < 1:
            raise ValueError(f"max_size must be a positive integer, got {max_size}")
        if min_size > max_size:
            raise ValueError(
                f"min_size ({min_size}) cannot be greater than max_size ({max_size})"
            )

        # Hard limit to prevent DoS via resource exhaustion
        MAX_POOL_SIZE_LIMIT = 5000
        if max_size > MAX_POOL_SIZE_LIMIT:
            raise ValueError(
                f"Connection pool max_size ({max_size}) exceeds hard limit ({MAX_POOL_SIZE_LIMIT}). "
                f"This would cause severe resource exhaustion. Use a smaller value."
            )

        if max_size > 1000:
            logger.warning(
                "Connection pool max_size is very large (%d). "
                "This could cause resource exhaustion. Consider using a smaller value.",
                max_size
            )

        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.driver = driver
        self._pool = None
        self._lock = threading.Lock()
        self._closed_explicitly = False  # Track if close() was called

        # Create the pool
        self._create_pool(**kwargs)

    def _create_pool(self, **kwargs) -> None:
        """Create the underlying connection pool."""
        if self.driver == "psycopg2":
            try:
                from psycopg2.pool import SimpleConnectionPool
                self._pool = SimpleConnectionPool(
                    self.min_size,
                    self.max_size,
                    self.dsn,
                    **kwargs
                )
            except ImportError:
                raise ImportError(
                    "psycopg2 not installed. Install with: pip install psycopg2-binary"
                )

        elif self.driver == "psycopg3":
            try:
                from psycopg_pool import ConnectionPool
                self._pool = ConnectionPool(
                    self.dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    **kwargs
                )
            except ImportError:
                raise ImportError(
                    "psycopg3 pool not installed. Install with: pip install psycopg[pool]"
                )

        else:
            raise ValueError(f"Unsupported driver: {self.driver}. Use 'psycopg2' or 'psycopg3'")

    def getconn(self, key: Optional[Any] = None) -> Any:
        """
        Get a connection from the pool.

        Args:
            key: Optional key for connection tracking (psycopg2 only)

        Returns:
            Database connection
        """
        with self._lock:
            if self._pool is None:
                raise RuntimeError("Connection pool is closed")

            if self.driver == "psycopg2":
                return self._pool.getconn(key)
            else:  # psycopg3
                return self._pool.getconn()

    def putconn(self, conn: Any, key: Optional[Any] = None, close: bool = False) -> None:
        """
        Return a connection to the pool.

        Args:
            conn: Connection to return
            key: Optional key for connection tracking (psycopg2 only)
            close: Whether to close the connection instead of returning it
        """
        with self._lock:
            if self._pool is None:
                return  # Pool already closed

            if self.driver == "psycopg2":
                self._pool.putconn(conn, key, close)
            else:  # psycopg3
                if close:
                    conn.close()
                else:
                    self._pool.putconn(conn)

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.

        Automatically returns the connection to the pool when done.

        Example:
            >>> with pool.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT 1")
        """
        conn = self.getconn()
        try:
            yield conn
        finally:
            self.putconn(conn)

    def close(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            self._closed_explicitly = True
            if self._pool is not None:
                if self.driver == "psycopg2":
                    self._pool.closeall()
                else:  # psycopg3
                    self._pool.close()
                self._pool = None

    def closeall(self) -> None:
        """Alias for close() to match pool protocol."""
        self.close()

    def __enter__(self):
        """Support context manager for pool lifecycle."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close pool when exiting context."""
        self.close()
        return False

    def __del__(self):
        """
        Clean up pool on deletion.

        Note: Relying on __del__ for cleanup is unreliable in Python.
        Always call close() explicitly or use the context manager:

            # Preferred: context manager
            with PgvectorConnectionPool(...) as pool:
                retriever = PgvectorSecureRetriever(connection=pool, ...)
                # pool is automatically closed when exiting the block

            # Alternative: explicit close
            pool = PgvectorConnectionPool(...)
            try:
                # use pool
            finally:
                pool.close()
        """
        try:
            # Warn if pool wasn't closed explicitly (potential resource leak)
            if self._pool is not None and not self._closed_explicitly:
                logger.warning(
                    "PgvectorConnectionPool was garbage collected without being closed. "
                    "This may cause connection leaks. Always call pool.close() explicitly "
                    "or use the context manager: 'with PgvectorConnectionPool(...) as pool:'"
                )
            self.close()
        except Exception as e:
            # Log cleanup failures to help diagnose connection pool leaks
            logger.warning(
                "Failed to close connection pool during cleanup: %s",
                str(e), exc_info=False
            )


class ManagedConnection:
    """
    Wrapper for managing a single connection from a pool.

    Used by retrievers to get connections from a pool automatically.
    This allows the retriever to be initialized with either a connection
    or a pool, and handle them uniformly.
    """

    def __init__(self, source: Union[Any, ConnectionPoolProtocol]):
        """
        Initialize managed connection.

        Args:
            source: Either a connection or a connection pool
        """
        self.source = source
        self._is_pool = hasattr(source, 'getconn') and hasattr(source, 'putconn')
        self._current_conn = None

    def get(self) -> Any:
        """
        Get a connection.

        If source is a pool, gets a connection from the pool.
        If source is a connection, returns it directly.
        """
        if self._is_pool:
            if self._current_conn is None:
                self._current_conn = self.source.getconn()
            return self._current_conn
        else:
            return self.source

    def release(self) -> None:
        """
        Release the connection.

        If source is a pool, returns the connection to the pool.
        If source is a connection, does nothing (user manages lifecycle).
        """
        if self._is_pool and self._current_conn is not None:
            self.source.putconn(self._current_conn)
            self._current_conn = None

    @contextmanager
    def use(self):
        """
        Context manager for using a connection.

        Automatically releases connection back to pool when done.

        Example:
            >>> managed = ManagedConnection(pool)
            >>> with managed.use() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT 1")
        """
        conn = self.get()
        try:
            yield conn
        finally:
            self.release()


__all__ = [
    "ConnectionPoolProtocol",
    "ManagedConnection",
    "PgvectorConnectionPool",
]
