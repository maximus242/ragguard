"""
Pinecone secure retriever with permission-aware search.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..exceptions import RetrieverError
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .base import BaseSecureRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig


class PineconeSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for Pinecone.

    Enforces document-level permissions by injecting permission filters
    into Pinecone queries using metadata filtering.

    Example:
        ```python
        from pinecone import Pinecone
        from ragguard import PineconeSecureRetriever, load_policy

        pc = Pinecone(api_key="your-api-key")
        index = pc.Index("your-index-name")

        policy = load_policy("policy.yaml")

        retriever = PineconeSecureRetriever(
            index=index,
            policy=policy,
            embed_fn=embeddings.embed_query  # Optional, for text queries
        )

        # Search with permission filtering
        results = retriever.search(
            query="What is our policy?",
            user={"id": "alice", "department": "engineering"},
            limit=10
        )
        ```
    """

    def __init__(
        self,
        index: Any,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        custom_filter_builder: Optional[Any] = None,
        namespace: str = "",
        enable_filter_cache: bool = True,
        filter_cache_size: int = 1000,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True,
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fail_on_audit_error: bool = False,
        *,
        config: Optional["SecureRetrieverConfig"] = None
    ):
        """
        Initialize Pinecone secure retriever.

        Args:
            index: Pinecone Index instance
            policy: Access control policy
            embed_fn: Optional function to convert text to embeddings
            audit_logger: Optional audit logger
            custom_filter_builder: Optional custom filter builder
            namespace: Pinecone namespace (default: "")
            enable_filter_cache: Whether to enable filter caching (default: True)
            filter_cache_size: Maximum size of filter cache (default: 1000)
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
            import pinecone
        except ImportError:
            raise RetrieverError(
                "pinecone-client not installed. Install with: pip install ragguard[pinecone]"
            )

        self.index = index
        self.namespace = namespace

        super().__init__(
            client=None,  # Pinecone uses index directly
            collection="",  # Pinecone uses index name
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
        return "pinecone"

    def _execute_search(
        self,
        query: List[float],
        filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """
        Execute Pinecone query with permission filter.

        Args:
            query: Query embedding
            filter: Pinecone metadata filter (dict format)
            limit: Maximum number of results
            **kwargs: Additional Pinecone query parameters

        Returns:
            List of Pinecone matches with metadata
        """
        # Pop internal kwargs that shouldn't be passed to client
        kwargs.pop('_user', None)
        kwargs.pop('_search_stats', None)

        try:
            # Build query parameters
            query_params = {
                "vector": query,
                "top_k": limit,
                "namespace": kwargs.get("namespace", self.namespace),
                "include_metadata": kwargs.get("include_metadata", True),
            }

            # Add permission filter if present
            if filter is not None:
                query_params["filter"] = filter

            # Add additional parameters
            if "include_values" in kwargs:
                query_params["include_values"] = kwargs["include_values"]

            if "sparse_vector" in kwargs:
                query_params["sparse_vector"] = kwargs["sparse_vector"]

            # Execute query
            response = self.index.query(**query_params)

            # Extract matches from response
            if hasattr(response, "matches"):
                return response.matches
            elif isinstance(response, dict) and "matches" in response:
                return response["matches"]
            else:
                return []

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"Pinecone search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Pinecone backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - index_exists: bool
            - index_info: dict (dimension, count, etc.)
        """
        health_info = {}

        from ..exceptions import ConfigurationError, HealthCheckError

        # Check if index exists and is accessible
        try:
            if self.index is None:
                raise ConfigurationError("Pinecone index is None", parameter="index")

            # Get index stats
            stats = self.index.describe_index_stats()
            health_info["index_exists"] = True

            # Extract stats information
            index_info = {
                "dimension": stats.dimension if hasattr(stats, 'dimension') else stats.get('dimension'),
                "total_vector_count": stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats.get('total_vector_count', 0)
            }

            # Add namespace info if available
            if hasattr(stats, 'namespaces'):
                index_info["namespaces"] = stats.namespaces
            elif isinstance(stats, dict) and 'namespaces' in stats:
                index_info["namespaces"] = stats['namespaces']

            health_info["index_info"] = index_info
        except ConfigurationError:
            raise
        except Exception as e:
            raise HealthCheckError("Pinecone", "Index not accessible", cause=e)

        # Successfully getting stats means connection is alive
        health_info["connection_alive"] = True

        return health_info
