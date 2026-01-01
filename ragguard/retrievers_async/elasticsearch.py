"""
Async Elasticsearch/OpenSearch retriever with permission-aware search.

Compatible with FastAPI and other async frameworks.
Supports both Elasticsearch and OpenSearch (which use the same Query DSL).
"""

from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase

logger = get_logger(__name__)


class AsyncElasticsearchSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async Elasticsearch/OpenSearch retriever with permission-aware search.

    Compatible with FastAPI and other async frameworks. Uses AsyncElasticsearch
    client for non-blocking operations.

    Example:
        ```python
        from elasticsearch import AsyncElasticsearch
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncElasticsearchSecureRetriever

        # Create async client
        client = AsyncElasticsearch(hosts=["http://localhost:9200"])

        # Create async retriever
        retriever = AsyncElasticsearchSecureRetriever(
            client=client,
            index="documents",
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
        index: str,
        policy: Policy,
        vector_field: str = "embedding",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async Elasticsearch secure retriever.

        Args:
            client: AsyncElasticsearch instance
            index: Index name
            policy: Access control policy
            vector_field: Name of the vector field (default: "embedding")
            embed_fn: Optional embedding function (if not provided, query must be a vector)
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration
            enable_validation: Whether to enable input validation (default: True)
        """
        # Validate client type
        client_type = type(client).__name__
        if "AsyncElasticsearch" not in client_type and "AsyncOpenSearch" not in client_type:
            raise RetrieverError(
                f"Expected AsyncElasticsearch or AsyncOpenSearch client, got {client_type}. "
                f"Use ElasticsearchSecureRetriever for sync clients."
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
        self.index = index
        self.vector_field = vector_field

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "elasticsearch"

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
            **kwargs: Additional arguments passed to Elasticsearch

        Returns:
            List of results (Elasticsearch hits)
        """
        # Validate user context
        if self._enable_validation:
            self._validator.validate_user_context(user)

        # Convert query to vector
        query_vector = await self._get_query_vector(query)

        # Generate filter from policy
        from ..filters.builder import to_elasticsearch_filter
        es_filter = to_elasticsearch_filter(self.policy, user)

        # Execute search
        results = await self._execute_search(query_vector, es_filter, limit, **kwargs)

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Optional[Dict[str, Any]],
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute the actual Elasticsearch search."""
        # Build kNN query
        knn_query = {
            "field": self.vector_field,
            "query_vector": query_vector,
            "k": limit,
            "num_candidates": kwargs.pop("num_candidates", limit * 10)
        }

        # Add filter to kNN if present
        if native_filter:
            knn_query["filter"] = native_filter

        async def _do_search():
            response = await self.client.search(
                index=self.index,
                knn=knn_query,
                size=limit,
                **kwargs
            )
            return response.get("hits", {}).get("hits", [])

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await _do_search()
            return await _search_with_retry()
        else:
            return await _do_search()

    async def close(self):
        """Close the async client connection."""
        if hasattr(self.client, 'close'):
            await self.client.close()


# Alias for OpenSearch compatibility
AsyncOpenSearchSecureRetriever = AsyncElasticsearchSecureRetriever
