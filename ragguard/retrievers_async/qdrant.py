"""
Async Qdrant retriever with permission-aware search.

Compatible with FastAPI and other async frameworks.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase


class AsyncQdrantSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async Qdrant retriever with permission-aware search.

    Compatible with FastAPI and other async frameworks.

    Example:
        ```python
        from qdrant_client import AsyncQdrantClient
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncQdrantSecureRetriever

        # Create async client
        client = AsyncQdrantClient(url="http://localhost:6333")

        # Create async retriever
        retriever = AsyncQdrantSecureRetriever(
            client=client,
            collection="documents",
            policy=load_policy("policy.yaml")
        )

        # Use in async context
        async def search_docs(query_text, user_context):
            results = await retriever.search(
                query=query_text,
                user=user_context,
                limit=10
            )
            return results

        # In FastAPI
        @app.post("/search")
        async def search_endpoint(query: str, user: dict):
            return await retriever.search(query=query, user=user, limit=10)
        ```
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async Qdrant secure retriever.

        Args:
            client: AsyncQdrantClient instance
            collection: Collection name
            policy: Access control policy
            embed_fn: Optional embedding function (if not provided, query must be a vector)
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration (defaults to 3 retries with exponential backoff)
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration (uses defaults if not provided)
            enable_validation: Whether to enable input validation (default: True)
        """
        try:
            from qdrant_client import AsyncQdrantClient
            if not isinstance(client, AsyncQdrantClient):
                raise TypeError(
                    f"client must be AsyncQdrantClient, got {type(client)}. "
                    "Use QdrantSecureRetriever for sync QdrantClient."
                )
        except ImportError:
            raise ImportError(
                "qdrant-client not installed. "
                "Install with: pip install ragguard[qdrant]"
            )

        super().__init__(
            policy=policy,
            embed_fn=embed_fn,
            audit_logger=audit_logger,
            retry_config=retry_config,
            enable_retry=enable_retry,
            validation_config=validation_config,
            enable_validation=enable_validation
        )

        self.client = client
        self.collection = collection

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "qdrant"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Async permission-aware vector search.

        Args:
            query: Query text (requires embed_fn) or query vector
            user: User context for permission filtering
            limit: Maximum number of results
            **kwargs: Additional arguments passed to Qdrant

        Returns:
            List of results (Qdrant ScoredPoint objects)
        """
        # Convert query to vector using base class method
        query_vector = await self._get_query_vector(query)

        # Generate filter from policy
        from ..filters.builder import to_qdrant_filter
        qdrant_filter = to_qdrant_filter(self.policy, user)

        # Execute search with optional retry
        results = await self._execute_search(query_vector, qdrant_filter, limit, **kwargs)

        # Log audit event using base class method
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute the actual Qdrant search."""
        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    query_filter=native_filter,
                    limit=limit,
                    **kwargs
                )
            return await _search_with_retry()
        else:
            return await self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=native_filter,
                limit=limit,
                **kwargs
            )
