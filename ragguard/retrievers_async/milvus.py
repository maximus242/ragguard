"""
Async Milvus/Zilliz retriever with permission-aware search.

Compatible with FastAPI and other async frameworks.

Note: Milvus doesn't have a native async client, so this uses
run_in_executor to run synchronous operations in a thread pool.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase

logger = get_logger(__name__)


class AsyncMilvusSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async Milvus/Zilliz retriever with permission-aware search.

    Compatible with FastAPI and other async frameworks. Uses thread pool
    for non-blocking operation since Milvus doesn't have a native async client.

    Example:
        ```python
        from pymilvus import MilvusClient
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncMilvusSecureRetriever

        # Create client (sync - will be wrapped)
        client = MilvusClient(uri="http://localhost:19530")

        # Create async retriever
        retriever = AsyncMilvusSecureRetriever(
            client=client,
            collection_name="documents",
            policy=load_policy("policy.yaml"),
            vector_field="embedding"
        )

        # Use in async context
        async def search_docs(query_vector, user_context):
            results = await retriever.search(
                query=query_vector,
                user=user_context,
                limit=10
            )
            return results
        ```
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        policy: Policy,
        vector_field: str = "vector",
        output_fields: Optional[List[str]] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async Milvus secure retriever.

        Args:
            client: MilvusClient instance (sync client - will be wrapped)
            collection_name: Name of the Milvus collection
            policy: Access control policy
            vector_field: Name of the vector field (default: "vector")
            output_fields: List of fields to return (default: all)
            embed_fn: Optional embedding function (if not provided, query must be a vector)
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration
            enable_validation: Whether to enable input validation (default: True)
        """
        import importlib.util
        if importlib.util.find_spec("pymilvus") is None:
            raise ImportError(
                "pymilvus not installed. "
                "Install with: pip install ragguard[milvus]"
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
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.output_fields = output_fields or ["*"]

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "milvus"

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
            **kwargs: Additional arguments passed to Milvus

        Returns:
            List of results
        """
        # Validate user context
        if self._enable_validation:
            self._validator.validate_user_context(user)

        # Convert query to vector
        query_vector = await self._get_query_vector(query)

        # Generate filter from policy
        from ..filters.builder import to_milvus_filter
        milvus_filter = to_milvus_filter(self.policy, user)

        # Execute search
        results = await self._execute_search(query_vector, milvus_filter, limit, **kwargs)

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Optional[str],
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute the actual Milvus search in thread pool."""
        loop = asyncio.get_running_loop()

        def _sync_search():
            search_params = {
                "collection_name": self.collection_name,
                "data": [query_vector],
                "anns_field": self.vector_field,
                "limit": limit,
                "output_fields": self.output_fields,
            }

            # Add filter if provided
            if native_filter:
                search_params["filter"] = native_filter

            # Add search params if provided
            if "search_params" in kwargs:
                search_params["search_params"] = kwargs["search_params"]

            # Execute search
            results = self.client.search(**search_params)

            # Flatten results (Milvus returns list of lists)
            if results and len(results) > 0:
                return results[0]
            return []

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await loop.run_in_executor(None, _sync_search)
            return await _search_with_retry()
        else:
            return await loop.run_in_executor(None, _sync_search)
