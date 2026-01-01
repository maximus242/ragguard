"""
pgvector implementation of secure retriever.

Wraps PostgreSQL connection with pgvector extension to provide
permission-aware vector search.

Supports both direct connections and connection pools for production use.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..connection_pool import ConnectionPoolProtocol, ManagedConnection
from ..exceptions import RetrieverError
from ..policy.models import Policy
from ..retry import RetryConfig
from .base import BaseSecureRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig


class PgvectorSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for PostgreSQL with pgvector.

    Wraps a PostgreSQL connection and injects permission filters
    into vector similarity queries.
    """

    def __init__(
        self,
        connection: Union[Any, ConnectionPoolProtocol],  # Connection or pool
        table: str,
        policy: Policy,
        embedding_column: str = "embedding",
        audit_logger: Optional[AuditLogger] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        custom_filter_builder: Optional[Any] = None,
        enable_filter_cache: bool = True,
        filter_cache_size: int = 1000,
        retry_config: Optional['RetryConfig'] = None,
        enable_retry: bool = True,
        validation_config: Optional[Any] = None,
        enable_validation: bool = True,
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fail_on_audit_error: bool = False,
        *,
        config: Optional["SecureRetrieverConfig"] = None
    ):
        """
        Initialize pgvector secure retriever.

        Args:
            connection: PostgreSQL connection or connection pool.
                - Single connection: psycopg2.connect() or psycopg3.connect()
                - Connection pool: PgvectorConnectionPool or any pool with
                  getconn()/putconn() methods
            table: Table name containing documents and embeddings
            policy: Access control policy
            embedding_column: Name of the column containing embeddings (default: "embedding")
            audit_logger: Optional audit logger
            embed_fn: Optional function to convert text to embeddings
            custom_filter_builder: Optional custom filter builder for complex permissions
            enable_filter_cache: Whether to enable filter caching (default: True)
            filter_cache_size: Maximum number of cached filters (default: 1000)
            retry_config: Optional retry configuration (defaults to 3 retries with exponential backoff)
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration (uses defaults if not provided)
            enable_validation: Whether to enable input validation (default: True)
            enable_circuit_breaker: Whether to enable circuit breaker (default: True)
            circuit_breaker_config: Optional circuit breaker configuration
            fail_on_audit_error: If True, raise an error when audit logging fails.
                                 If False (default), log a warning and continue.

        Example with connection pool:
            >>> from ragguard import PgvectorConnectionPool, PgvectorSecureRetriever
            >>>
            >>> pool = PgvectorConnectionPool(
            ...     dsn="postgresql://localhost/mydb",
            ...     min_size=2,
            ...     max_size=10
            ... )
            >>>
            >>> retriever = PgvectorSecureRetriever(
            ...     connection=pool,
            ...     table="documents",
            ...     policy=policy
            ... )
        """
        try:
            # Check if pgvector is available
            import pgvector
        except ImportError:
            raise RetrieverError(
                "pgvector not installed. Install with: pip install ragguard[pgvector]"
            )

        # Validate SQL identifiers to prevent injection
        self._validate_sql_identifier(table, "table")
        self._validate_sql_identifier(embedding_column, "embedding_column")

        super().__init__(
            client=connection,
            collection=table,  # Use collection for table name
            policy=policy,
            audit_logger=audit_logger,
            embed_fn=embed_fn,
            custom_filter_builder=custom_filter_builder,
            enable_filter_cache=enable_filter_cache,
            filter_cache_size=filter_cache_size,
            retry_config=retry_config,
            enable_retry=enable_retry,
            validation_config=validation_config,
            enable_validation=enable_validation,
            enable_circuit_breaker=enable_circuit_breaker,
            circuit_breaker_config=circuit_breaker_config,
            fail_on_audit_error=fail_on_audit_error,
            config=config
        )

        # Wrap connection in ManagedConnection for unified handling
        self._managed_conn = ManagedConnection(connection)
        self.table = table
        self.embedding_column = embedding_column

    @staticmethod
    def _validate_sql_identifier(value: str, param_name: str) -> None:
        """
        Validate SQL identifier to prevent SQL injection.

        Args:
            value: The identifier to validate
            param_name: Name of the parameter (for error messages)

        Raises:
            RetrieverError: If the identifier contains invalid characters
        """
        if not value or not isinstance(value, str):
            raise RetrieverError(f"{param_name} must be a non-empty string")

        if len(value) > 63:  # PostgreSQL identifier length limit
            raise RetrieverError(f"{param_name} too long: {len(value)} chars (max 63)")

        # Allow alphanumeric, underscore, and dollar sign (PostgreSQL standard)
        # Must start with letter or underscore
        if not value[0].isalpha() and value[0] != '_':
            raise RetrieverError(
                f"Invalid {param_name}: must start with letter or underscore, got '{value[0]}'"
            )

        if not all(c.isalnum() or c in ('_', '$') for c in value):
            raise RetrieverError(
                f"Invalid {param_name}: contains invalid characters. "
                f"Only alphanumeric, underscore, and $ are allowed. Got: '{value}'"
            )

        # Check for PostgreSQL reserved keywords to prevent confusion/injection
        # This is a subset of the most dangerous reserved words
        reserved_keywords = {
            'select', 'insert', 'update', 'delete', 'drop', 'create', 'alter',
            'truncate', 'grant', 'revoke', 'execute', 'call', 'do', 'copy',
            'table', 'database', 'schema', 'index', 'view', 'function',
            'trigger', 'procedure', 'role', 'user', 'all', 'and', 'or', 'not',
            'null', 'true', 'false', 'from', 'where', 'join', 'union', 'group',
            'order', 'having', 'limit', 'offset', 'as', 'on', 'in', 'exists',
        }
        if value.lower() in reserved_keywords:
            raise RetrieverError(
                f"Invalid {param_name}: '{value}' is a PostgreSQL reserved keyword. "
                f"Choose a different name to avoid potential SQL issues."
            )

    @property
    def connection(self) -> Any:
        """
        Get the underlying connection or pool.

        For backward compatibility, this returns the original connection/pool
        that was passed to __init__.

        Returns:
            The connection or connection pool
        """
        return self._managed_conn.source

    @property
    def backend_name(self) -> str:
        """Return backend name for filter generation."""
        return "pgvector"

    def _execute_search(
        self,
        query: list[float],
        filter: tuple[str, list[Any]],
        limit: int,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Execute pgvector search with permission filter.

        Automatically gets a connection from the pool (if using pooling)
        or uses the direct connection.

        Args:
            query: Query embedding vector
            filter: Tuple of (WHERE clause, parameters)
            limit: Maximum results
            **kwargs: Additional search arguments

        Returns:
            List of result dictionaries with 'distance' and all columns
        """
        # Pop internal kwargs that shouldn't be passed to client
        kwargs.pop('_user', None)
        kwargs.pop('_search_stats', None)

        where_clause, params = filter

        # Get distance operator from kwargs (default to L2 distance <->)
        distance_op = kwargs.get("distance_op", "<->")

        # Validate distance operator against allowlist to prevent SQL injection
        valid_distance_ops = {"<->", "<=>", "<#>"}  # L2, cosine, inner product
        if distance_op not in valid_distance_ops:
            raise RetrieverError(
                f"Invalid distance_op: '{distance_op}'. "
                f"Must be one of: {', '.join(sorted(valid_distance_ops))}"
            )

        # Build the SQL query
        # The query vector is the first parameter
        # Table/column names are developer config, distance_op validated above, user values use %s params
        sql = (
            f"SELECT *, {self.embedding_column} {distance_op} %s as distance "  # nosec B608
            f"FROM {self.table} "
            f"{where_clause} "
            f"ORDER BY distance "
            f"LIMIT %s"
        )

        # Combine query vector + filter params + limit
        all_params = [query] + params + [limit]

        try:
            # Use managed connection (handles both pools and direct connections)
            with self._managed_conn.use() as connection:
                # Check if we should return dict-like results
                try:
                    # Try to use DictCursor if available (psycopg2)
                    import psycopg2.extras
                    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
                except (ImportError, AttributeError):
                    # Fall back to regular cursor
                    cursor = connection.cursor()

                try:
                    cursor.execute(sql, all_params)
                    results = cursor.fetchall()

                    # Convert results to dictionaries if using regular cursor
                    if results and not isinstance(results[0], dict):
                        columns = [desc[0] for desc in cursor.description]
                        results = [dict(zip(columns, row)) for row in results]

                    return results
                finally:
                    cursor.close()

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"pgvector search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check pgvector backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - table_exists: bool
            - table_info: dict (row count, columns, etc.)
            - extension_installed: bool
        """
        health_info = {}

        from ..exceptions import ConfigurationError, HealthCheckError

        try:
            with self._managed_conn.use() as connection:
                cursor = connection.cursor()
                try:
                    # Check if pgvector extension is installed
                    cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                    extension_result = cursor.fetchone()
                    health_info["extension_installed"] = extension_result is not None

                    if not health_info["extension_installed"]:
                        raise ConfigurationError("pgvector extension not installed", parameter="extension")

                    # Check if table exists and get info
                    cursor.execute("""
                        SELECT COUNT(*) as row_count
                        FROM information_schema.tables
                        WHERE table_name = %s
                    """, [self.table])
                    row = cursor.fetchone()
                    table_exists = row[0] > 0 if row else False
                    health_info["table_exists"] = table_exists

                    if not table_exists:
                        raise HealthCheckError("pgvector", f"Table '{self.table}' does not exist")

                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {self.table}")  # nosec B608
                    count_row = cursor.fetchone()
                    row_count = count_row[0] if count_row else 0
                    health_info["table_info"] = {"row_count": row_count}

                    # Check if embedding column exists
                    cursor.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = %s AND column_name = %s
                    """, [self.table, self.embedding_column])
                    column_info = cursor.fetchone()
                    if column_info:
                        health_info["embedding_column_exists"] = True
                        health_info["table_info"]["embedding_column_type"] = column_info[1]
                    else:
                        raise HealthCheckError("pgvector", f"Embedding column '{self.embedding_column}' not found")

                    health_info["connection_alive"] = True
                finally:
                    cursor.close()

        except (HealthCheckError, ConfigurationError):
            raise
        except Exception as e:
            raise HealthCheckError("pgvector", cause=e)

        return health_info
