"""
Weaviate implementation of secure retriever.

Wraps Weaviate Client to provide permission-aware vector search.
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


class WeaviateSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for Weaviate vector database.

    Wraps a Weaviate Client and injects permission filters into search queries.
    """

    def __init__(
        self,
        client: Any,  # weaviate.Client or weaviate.WeaviateClient
        collection: str,
        policy: Policy,
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
        Initialize Weaviate secure retriever.

        Args:
            client: Weaviate Client instance
            collection: Collection/class name
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
        """
        try:
            import weaviate
        except ImportError:
            raise RetrieverError(
                "weaviate-client not installed. Install with: pip install ragguard[weaviate]"
            )

        # Check if client is a Weaviate client (support both v3 and v4)
        valid_types = ['Client', 'WeaviateClient']
        client_type_name = type(client).__name__

        if client_type_name not in valid_types:
            raise RetrieverError(
                f"Expected weaviate.Client or weaviate.WeaviateClient, got {client_type_name}"
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
        return "weaviate"

    def _execute_search(
        self,
        query: list[float],
        filter: Optional[dict[str, Any]],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute Weaviate search with permission filter.

        Args:
            query: Query embedding vector
            filter: Weaviate where filter (dict format)
            limit: Maximum results
            **kwargs: Additional Weaviate search arguments

        Returns:
            List of Weaviate objects with their properties and metadata
        """
        # Pop internal kwargs that shouldn't be passed to client
        kwargs.pop('_user', None)
        kwargs.pop('_search_stats', None)

        try:
            # Build the query
            query_builder = (
                self.client.query
                .get(self.collection, kwargs.get('properties', ['*']))
                .with_near_vector({
                    "vector": query,
                    "certainty": kwargs.get('certainty', 0.7)
                })
                .with_limit(limit)
            )

            # Add permission filter if present
            if filter:
                query_builder = query_builder.with_where(filter)

            # Add additional metadata
            if kwargs.get('with_additional'):
                query_builder = query_builder.with_additional(kwargs['with_additional'])
            else:
                # By default, include id, certainty, and vector
                query_builder = query_builder.with_additional(['id', 'certainty', 'vector'])

            # Execute query
            result = query_builder.do()

            # Extract results from response with proper null checks
            if result and 'data' in result:
                data = result['data']
                if data and isinstance(data, dict) and 'Get' in data:
                    get_data = data['Get']
                    if get_data and isinstance(get_data, dict):
                        objects = get_data.get(self.collection, [])
                        return objects if isinstance(objects, list) else []

            return []

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"Weaviate search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Weaviate backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - collection_exists: bool
            - collection_info: dict (object count, etc.)
        """
        health_info = {}

        # Check if collection/class exists
        try:
            # Get schema to check if collection exists
            schema = self.client.schema.get(self.collection)
            if schema:
                health_info["collection_exists"] = True

                # Get object count using aggregate query
                try:
                    result = (
                        self.client.query
                        .aggregate(self.collection)
                        .with_meta_count()
                        .do()
                    )

                    object_count = 0
                    if result and 'data' in result and 'Aggregate' in result['data']:
                        agg_data = result['data']['Aggregate'].get(self.collection, [])
                        if agg_data and len(agg_data) > 0:
                            object_count = agg_data[0].get('meta', {}).get('count', 0)

                    health_info["collection_info"] = {
                        "object_count": object_count,
                        "class_name": self.collection
                    }
                except Exception as count_error:
                    # If count fails, still report collection exists
                    health_info["collection_info"] = {
                        "object_count": None,
                        "class_name": self.collection,
                        "count_error": str(count_error)
                    }
            else:
                from ..exceptions import HealthCheckError
                raise HealthCheckError("Weaviate", f"Collection/class '{self.collection}' schema not found")
        except Exception as e:
            from ..exceptions import HealthCheckError
            if isinstance(e, HealthCheckError):
                raise
            raise HealthCheckError("Weaviate", f"Collection/class '{self.collection}' not accessible", cause=e)

        # Verify connection
        try:
            from ..exceptions import RetrieverConnectionError
            # Check if client is ready
            if hasattr(self.client, 'is_ready'):
                is_ready = self.client.is_ready()
                if not is_ready:
                    raise RetrieverConnectionError("Weaviate", "Client not ready")
            health_info["connection_alive"] = True
        except RetrieverConnectionError:
            raise
        except Exception as e:
            from ..exceptions import RetrieverConnectionError
            raise RetrieverConnectionError("Weaviate", cause=e)

        return health_info
