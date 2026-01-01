"""
Async pgvector retriever with permission-aware search.

Uses run_in_executor to make PostgreSQL non-blocking.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..policy.models import Policy
from ..retry import RetryConfig, get_shared_executor
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase


class AsyncPgvectorSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async pgvector retriever with permission-aware search.

    Uses run_in_executor to make PostgreSQL non-blocking.

    Example:
        ```python
        import psycopg2
        from ragguard.retrievers_async import AsyncPgvectorSecureRetriever
        from ragguard import PgvectorConnectionPool

        # Create connection pool
        pool = PgvectorConnectionPool(dsn="postgresql://...", max_size=10)

        # Create async retriever
        retriever = AsyncPgvectorSecureRetriever(
            connection=pool,
            table="documents",
            policy=policy
        )

        # Use in async context
        async def search():
            results = await retriever.search(
                query=[0.1, 0.2, ...],
                user={"department": "eng"},
                limit=10
            )
            return results
        ```
    """

    def __init__(
        self,
        connection: Any,
        table: str,
        policy: Policy,
        embedding_column: str = "embedding",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async pgvector secure retriever.

        Args:
            connection: PostgreSQL connection or pool
            table: Table name
            policy: Access control policy
            embedding_column: Column name for embeddings
            embed_fn: Optional embedding function
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration (uses defaults if not provided)
            enable_validation: Whether to enable input validation (default: True)
        """
        super().__init__(
            policy=policy,
            embed_fn=embed_fn,
            audit_logger=audit_logger,
            retry_config=retry_config,
            enable_retry=enable_retry,
            validation_config=validation_config,
            enable_validation=enable_validation
        )

        from ..retrievers.pgvector import PgvectorSecureRetriever

        # Create sync retriever to leverage existing logic
        self._sync_retriever = PgvectorSecureRetriever(
            connection=connection,
            table=table,
            policy=policy,
            embedding_column=embedding_column,
            embed_fn=embed_fn,
            audit_logger=audit_logger,
            retry_config=retry_config,
            enable_retry=enable_retry
        )
        self.table = table

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "pgvector"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Async permission-aware search.

        Args:
            query: Query text or vector
            user: User context
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of result dictionaries
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            get_shared_executor(),
            lambda: self._sync_retriever.search(query=query, user=user, limit=limit, **kwargs)
        )

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute search - delegates to sync retriever."""
        # Not used directly since search() delegates to sync retriever
        raise NotImplementedError("Use search() instead")
