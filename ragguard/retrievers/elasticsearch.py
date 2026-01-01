"""
Elasticsearch / OpenSearch implementation of secure retriever.

Wraps Elasticsearch/OpenSearch clients to provide permission-aware vector search.
Works with both elasticsearch-py and opensearch-py libraries.
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


class ElasticsearchSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for Elasticsearch and OpenSearch.

    Wraps an Elasticsearch or OpenSearch client and injects permission filters
    into vector search queries using the kNN search API.

    Supports both:
    - Elasticsearch 8.0+ (elasticsearch-py)
    - OpenSearch 2.0+ (opensearch-py)

    Both use the same Query DSL format, so this retriever works for both.
    """

    def __init__(
        self,
        client: Any,  # Elasticsearch or OpenSearch client
        index: str,
        policy: Policy,
        vector_field: str = "embedding",
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
        Initialize Elasticsearch/OpenSearch secure retriever.

        Args:
            client: Elasticsearch or OpenSearch client instance
            index: Index name
            policy: Access control policy
            vector_field: Name of the field containing embeddings (default: "embedding")
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
        # Validate client type (either Elasticsearch or OpenSearch)
        client_type = type(client).__name__
        if "Elasticsearch" not in client_type and "OpenSearch" not in client_type:
            raise RetrieverError(
                f"Expected Elasticsearch or OpenSearch client, got {client_type}. "
                f"Install with: pip install ragguard[elasticsearch] or pip install ragguard[opensearch]"
            )

        super().__init__(
            client=client,
            collection=index,  # Use 'collection' field for index name
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

        self.vector_field = vector_field

    @property
    def backend_name(self) -> str:
        """Return backend name for filter generation."""
        return "elasticsearch"

    def _execute_search(
        self,
        query: list[float],
        filter: Optional[dict[str, Any]],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute Elasticsearch/OpenSearch kNN search with permission filter.

        Args:
            query: Query embedding vector
            filter: Elasticsearch query dict (bool query)
            limit: Maximum results
            **kwargs: Additional search arguments

        Returns:
            List of search results with score, id, and source
        """
        try:
            # Build kNN query
            knn_query = {
                "field": self.vector_field,
                "query_vector": query,
                "k": limit,
                "num_candidates": kwargs.get("num_candidates", limit * 2)
            }

            # Add filter if provided
            if filter:
                knn_query["filter"] = filter

            # Execute kNN search
            response = self.client.search(
                index=self.collection,
                knn=knn_query,
                size=limit,
                _source=kwargs.get("_source", True)
            )

            # Convert response to standard format
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "metadata": hit["_source"],
                    "document": hit.get("_source", {}).get("text", "")
                })

            return results

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"Elasticsearch/OpenSearch search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Elasticsearch/OpenSearch backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - index_exists: bool
            - index_info: dict (document count, etc.)
        """
        from ..exceptions import ConfigurationError, HealthCheckError, RetrieverConnectionError

        health_info = {}

        # Check if client is alive
        try:
            if self.client is None:
                raise ConfigurationError("Elasticsearch/OpenSearch client is None", parameter="client")

            # Ping the cluster to verify connection
            if not self.client.ping():
                raise RetrieverConnectionError("Elasticsearch", "Ping failed - cluster not responding")

            health_info["connection_alive"] = True
        except (ConfigurationError, RetrieverConnectionError):
            raise
        except Exception as e:
            raise RetrieverConnectionError("Elasticsearch", cause=e)

        # Check if index exists
        try:
            index_exists = self.client.indices.exists(index=self.collection)
            health_info["index_exists"] = index_exists

            if not index_exists:
                raise HealthCheckError("Elasticsearch", f"Index '{self.collection}' does not exist")
        except HealthCheckError:
            raise
        except Exception as e:
            raise HealthCheckError("Elasticsearch", "Failed to check index existence", cause=e)

        # Get index stats (document count)
        try:
            count_response = self.client.count(index=self.collection)
            document_count = count_response.get("count", 0) if isinstance(count_response, dict) else count_response.body.get("count", 0)

            health_info["index_info"] = {
                "document_count": document_count,
                "index_name": self.collection
            }
        except Exception as e:
            raise HealthCheckError("Elasticsearch", "Failed to get index stats", cause=e)

        return health_info


class OpenSearchSecureRetriever(ElasticsearchSecureRetriever):
    """
    Alias for ElasticsearchSecureRetriever.

    OpenSearch uses the same Query DSL as Elasticsearch, so this is
    just a convenience alias for clarity when using OpenSearch.
    """
    pass
