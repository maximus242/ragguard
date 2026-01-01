"""
Qdrant implementation of secure retriever.

Wraps QdrantClient to provide permission-aware vector search.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..exceptions import RetrieverError
from ..policy.models import Policy
from ..retry import RetryConfig
from .base import BaseSecureRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig


class QdrantSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for Qdrant vector database.

    Wraps a QdrantClient and injects permission filters into search queries.

    Example with config object (recommended):
        >>> from ragguard import SecureRetrieverConfig, QdrantSecureRetriever
        >>> config = SecureRetrieverConfig.production()
        >>> retriever = QdrantSecureRetriever(client, "docs", policy, config=config)

    Example with individual parameters (backward compatible):
        >>> retriever = QdrantSecureRetriever(
        ...     client=client,
        ...     collection="docs",
        ...     policy=policy,
        ...     enable_retry=True,
        ...     enable_validation=True
        ... )
    """

    def __init__(
        self,
        client: Any,  # QdrantClient
        collection: str,
        policy: Policy,
        audit_logger: Optional[AuditLogger] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        custom_filter_builder: Optional[Any] = None,
        enable_filter_cache: bool = True,
        filter_cache_size: int = 1000,
        retry_config: Optional[RetryConfig] = None,
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
        Initialize Qdrant secure retriever.

        Args:
            client: QdrantClient instance
            collection: Collection name
            policy: Access control policy
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
            config: Optional SecureRetrieverConfig for consolidated settings.
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise RetrieverError(
                "qdrant-client not installed. Install with: pip install ragguard[qdrant]"
            )

        if not isinstance(client, QdrantClient):
            raise RetrieverError(
                f"Expected QdrantClient, got {type(client).__name__}"
            )

        super().__init__(
            client=client,
            collection=collection,
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

    @property
    def backend_name(self) -> str:
        """Return backend name for filter generation."""
        return "qdrant"

    def _execute_search(
        self,
        query: list[float],
        filter: Any,
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute Qdrant search with permission filter.

        Args:
            query: Query embedding vector
            filter: Qdrant Filter object
            limit: Maximum results
            **kwargs: Additional Qdrant search arguments

        Returns:
            List of Qdrant ScoredPoint objects
        """
        # Pop internal kwargs that shouldn't be passed to client
        kwargs.pop('_user', None)
        kwargs.pop('_search_stats', None)

        try:
            # Use query_points (newer API) with fallback to search (older API)
            if hasattr(self.client, 'query_points'):
                results = self.client.query_points(
                    collection_name=self.collection,
                    query=query,
                    query_filter=filter,
                    limit=limit,
                    **kwargs
                ).points
            else:
                # Fallback for older qdrant-client versions
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=query,
                    query_filter=filter,
                    limit=limit,
                    **kwargs
                )
            return results
        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"Qdrant search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Qdrant backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - collection_exists: bool
            - collection_info: dict (vectors count, config, etc.)
        """
        health_info = {}

        from ..exceptions import HealthCheckError, RetrieverConnectionError

        # Check if collection exists
        try:
            collection_info = self.client.get_collection(self.collection)
            health_info["collection_exists"] = True
            health_info["collection_info"] = {
                "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else None,
                "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else None,
                "status": str(collection_info.status) if hasattr(collection_info, 'status') else None,
            }
        except Exception as e:
            raise HealthCheckError("Qdrant", f"Collection '{self.collection}' not accessible", cause=e)

        # Check connection by getting cluster info
        try:
            self.client.get_collections()
            health_info["connection_alive"] = True
        except Exception as e:
            raise RetrieverConnectionError("Qdrant", cause=e)

        return health_info
