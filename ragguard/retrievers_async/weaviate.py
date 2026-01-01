"""
Async Weaviate retriever with permission-aware search.

Uses run_in_executor for Weaviate v3, or native async for v4.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure, get_shared_executor
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase


class AsyncWeaviateSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async Weaviate retriever with permission-aware search.

    Uses run_in_executor for Weaviate v3, or native async for v4.

    Example:
        ```python
        import weaviate
        from ragguard.retrievers_async import AsyncWeaviateSecureRetriever

        # Create Weaviate client
        client = weaviate.Client("http://localhost:8080")

        # Create async retriever
        retriever = AsyncWeaviateSecureRetriever(
            client=client,
            collection="Documents",
            policy=policy
        )

        # Use in async context
        async def search():
            results = await retriever.search(
                query=[0.1, 0.2, ...],
                user={"role": "engineer"},
                limit=10
            )
            return results
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
        Initialize async Weaviate secure retriever.

        Args:
            client: Weaviate client instance
            collection: Collection/class name
            policy: Access control policy
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

        self.client = client
        self.collection = collection

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "weaviate"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Async permission-aware search.

        Args:
            query: Query text or vector
            user: User context
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of Weaviate objects
        """
        # Convert query to vector using base class method
        query_vector = await self._get_query_vector(query)

        # Generate filter
        from ..filters.builder import to_weaviate_filter
        weaviate_filter = to_weaviate_filter(self.policy, user)

        # Execute search
        results = await self._execute_search(query_vector, weaviate_filter, limit, **kwargs)

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute the actual Weaviate search in thread pool."""
        loop = asyncio.get_running_loop()

        def _sync_search():
            result = (
                self.client.query
                .get(self.collection, ["*"])
                .with_near_vector({"vector": query_vector})
                .with_where(native_filter) if native_filter else
                self.client.query
                .get(self.collection, ["*"])
                .with_near_vector({"vector": query_vector})
            )
            return result.with_limit(limit).do()

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await loop.run_in_executor(get_shared_executor(), _sync_search)
            result = await _search_with_retry()
        else:
            result = await loop.run_in_executor(get_shared_executor(), _sync_search)

        # Extract results with proper null checks
        if result and 'data' in result:
            data = result['data']
            if data and isinstance(data, dict) and 'Get' in data:
                get_data = data['Get']
                if get_data and isinstance(get_data, dict):
                    objects = get_data.get(self.collection, [])
                    return objects if isinstance(objects, list) else []
        return []
