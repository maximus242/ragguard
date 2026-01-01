"""
Azure AI Search implementation of secure retriever.

Wraps Azure SearchClient to provide permission-aware vector search.
Works with Azure AI Search (formerly Azure Cognitive Search).
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


class AzureSearchSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for Azure AI Search.

    Wraps an Azure SearchClient and injects permission filters
    into vector search queries using Azure's vector search API.

    Supports Azure AI Search (formerly Azure Cognitive Search) with:
    - Hybrid search (vector + keyword)
    - OData filter syntax for metadata filtering
    - Vector similarity search
    """

    def __init__(
        self,
        client: Any,  # SearchClient from azure-search-documents
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
        Initialize Azure AI Search secure retriever.

        Args:
            client: SearchClient instance from azure-search-documents
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
        # Validate client type
        client_type = type(client).__name__
        if "SearchClient" not in client_type:
            raise RetrieverError(
                f"Expected SearchClient from azure-search-documents, got {client_type}. "
                f"Install with: pip install ragguard[azure]"
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
        return "azure_search"

    def _execute_search(
        self,
        query: list[float],
        filter: Optional[str],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute Azure AI Search vector search with permission filter.

        Args:
            query: Query embedding vector
            filter: OData filter string
            limit: Maximum results
            **kwargs: Additional search arguments

        Returns:
            List of search results with score, id, and document
        """
        try:
            # Build vector query using Azure's VectorizedQuery
            try:
                from azure.search.documents.models import VectorizedQuery
            except ImportError:
                raise RetrieverError(
                    "azure-search-documents not installed. "
                    "Install with: pip install ragguard[azure]"
                )

            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query,
                k_nearest_neighbors=limit,
                fields=self.vector_field
            )

            # Execute search
            search_params = {
                "vector_queries": [vector_query],
                "top": limit,
                "select": kwargs.get("select"),
            }

            # Add filter if provided
            if filter:
                search_params["filter"] = filter

            # Execute search
            results = self.client.search(
                search_text=kwargs.get("search_text"),  # For hybrid search
                **search_params
            )

            # Convert results to standard format
            output = []
            for result in results:
                output.append({
                    "id": result.get("id") or result.get("document_id"),
                    "score": result.get("@search.score", 0.0),
                    "metadata": dict(result),
                    "document": result.get("content") or result.get("text", "")
                })

            return output

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"Azure AI Search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Azure AI Search backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - index_exists: bool
            - index_info: dict (document count, etc.)
        """
        from ..exceptions import (
            ConfigurationError,
            HealthCheckError,
            RetrieverPermissionError,
        )

        health_info = {}

        # Check if client is accessible and index exists
        try:
            if self.client is None:
                raise ConfigurationError("Azure SearchClient is None", parameter="client")

            # Get document count - this verifies both connection and index existence
            document_count = self.client.get_document_count()

            health_info["connection_alive"] = True
            health_info["index_exists"] = True
            health_info["index_info"] = {
                "document_count": document_count,
                "index_name": self.collection
            }
        except ConfigurationError:
            raise
        except Exception as e:
            # Try to determine if it's a connection issue or index issue
            error_message = str(e).lower()
            if "not found" in error_message or "does not exist" in error_message:
                raise HealthCheckError("Azure Search", f"Index '{self.collection}' does not exist", cause=e)
            elif "unauthorized" in error_message or "authentication" in error_message:
                raise RetrieverPermissionError("Azure Search", resource=self.collection, message="Authentication failed")
            else:
                raise HealthCheckError("Azure Search", cause=e)

        return health_info


class AzureCognitiveSearchSecureRetriever(AzureSearchSecureRetriever):
    """
    Alias for AzureSearchSecureRetriever.

    Azure Cognitive Search was renamed to Azure AI Search, but this
    alias is provided for backwards compatibility.
    """
    pass
