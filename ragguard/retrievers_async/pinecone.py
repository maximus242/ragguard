"""
Async Pinecone retriever with permission-aware search.

Pinecone doesn't have native async support, so this uses
run_in_executor to make it non-blocking for async contexts.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure, get_shared_executor
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase


class AsyncPineconeSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async Pinecone retriever with permission-aware search.

    Pinecone doesn't have native async support, so this uses
    run_in_executor to make it non-blocking for async contexts.

    Example:
        ```python
        from pinecone import Pinecone
        from ragguard.retrievers_async import AsyncPineconeSecureRetriever

        # Create Pinecone client
        pc = Pinecone(api_key="your-api-key")
        index = pc.Index("your-index")

        # Create async retriever
        retriever = AsyncPineconeSecureRetriever(
            index=index,
            policy=policy
        )

        # Use in async context
        async def search():
            results = await retriever.search(
                query=[0.1, 0.2, 0.3, ...],
                user={"department": "engineering"},
                limit=10
            )
            return results
        ```
    """

    def __init__(
        self,
        index: Any,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        namespace: str = "",
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async Pinecone secure retriever.

        Args:
            index: Pinecone Index instance
            policy: Access control policy
            embed_fn: Optional embedding function
            namespace: Pinecone namespace (default: "")
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

        self.index = index
        self.namespace = namespace

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "pinecone"

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
            List of Pinecone matches
        """
        # Convert query to vector using base class method
        query_vector = await self._get_query_vector(query)

        # Generate filter
        from ..filters.builder import to_pinecone_filter
        pinecone_filter = to_pinecone_filter(self.policy, user)

        # Execute search
        results = await self._execute_search(query_vector, pinecone_filter, limit, **kwargs)

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
        """Execute the actual Pinecone search in thread pool."""
        loop = asyncio.get_running_loop()

        def _sync_search():
            return self.index.query(
                vector=query_vector,
                filter=native_filter,
                top_k=limit,
                namespace=self.namespace,
                include_metadata=True,
                **kwargs
            )

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await loop.run_in_executor(get_shared_executor(), _sync_search)
            result = await _search_with_retry()
        else:
            result = await loop.run_in_executor(get_shared_executor(), _sync_search)

        return result.matches if hasattr(result, 'matches') else []
