"""
ChromaDB secure retriever with permission-aware search.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .base import BaseSecureRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig


class ChromaDBSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for ChromaDB.

    Enforces document-level permissions by injecting permission filters
    into ChromaDB queries using metadata filtering.

    Example:
        ```python
        import chromadb
        from ragguard import ChromaDBSecureRetriever, load_policy

        client = chromadb.Client()
        collection = client.get_or_create_collection("my_collection")

        policy = load_policy("policy.yaml")

        retriever = ChromaDBSecureRetriever(
            collection=collection,
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
        collection: Any,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        custom_filter_builder: Optional[Any] = None,
        enable_filter_cache: bool = True,
        filter_cache_size: int = 1000,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True,
        fail_on_audit_error: bool = False,
        *,
        config: Optional["SecureRetrieverConfig"] = None
    ):
        """
        Initialize ChromaDB secure retriever.

        Args:
            collection: ChromaDB Collection instance
            policy: Access control policy
            embed_fn: Optional function to convert text to embeddings
            audit_logger: Optional audit logger
            custom_filter_builder: Optional custom filter builder
            enable_filter_cache: Whether to enable filter caching (default: True)
            filter_cache_size: Maximum size of filter cache (default: 1000)
            retry_config: Optional retry configuration (defaults to 3 retries with exponential backoff)
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration (uses defaults if not provided)
            enable_validation: Whether to enable input validation (default: True)
            fail_on_audit_error: If True, raise an error when audit logging fails.
                                 If False (default), log a warning and continue.
            config: Optional SecureRetrieverConfig for consolidated settings.
        """
        try:
            import chromadb
        except ImportError:
            raise RetrieverError(
                "chromadb not installed. Install with: pip install ragguard[chromadb]"
            )

        self.chroma_collection = collection

        # Call parent with dummy values for client/collection since ChromaDB uses different API
        super().__init__(
            client=None,  # ChromaDB doesn't use a separate client
            collection=collection.name,
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
            fail_on_audit_error=fail_on_audit_error,
            config=config
        )

    @property
    def backend_name(self) -> str:
        """Return backend name for filter generation."""
        return "chromadb"

    def _execute_search(
        self,
        query: List[float],
        filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """
        Execute ChromaDB query with permission filter.

        Args:
            query: Query embedding
            filter: ChromaDB where filter (dict format)
            limit: Maximum number of results
            **kwargs: Additional ChromaDB query parameters

        Returns:
            List of ChromaDB results with metadata
        """
        # Pop internal kwargs that shouldn't be passed to client
        kwargs.pop('_user', None)
        kwargs.pop('_search_stats', None)

        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [query],
                "n_results": limit,
                "include": kwargs.get("include", ["metadatas", "documents", "distances"])
            }

            # Add permission filter if present
            if filter is not None:
                query_params["where"] = filter

            # Add additional where_document filter if provided
            if "where_document" in kwargs:
                query_params["where_document"] = kwargs["where_document"]

            # Execute query
            response = self.chroma_collection.query(**query_params)

            # ChromaDB returns results in dict format with lists
            # Convert to list of result objects for consistency
            results = []
            if response and "ids" in response and response["ids"]:
                ids_list = response["ids"]
                ids = ids_list[0] if ids_list else []

                # Safely extract optional fields with proper bounds checking
                metadatas_list = response.get("metadatas") or [[]]
                metadatas = metadatas_list[0] if metadatas_list else []

                documents_list = response.get("documents") or [[]]
                documents = documents_list[0] if documents_list else []

                distances_list = response.get("distances") or [[]]
                distances = distances_list[0] if distances_list else []

                for i, doc_id in enumerate(ids):
                    result = {
                        "id": doc_id,
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "document": documents[i] if i < len(documents) else None,
                        "distance": distances[i] if i < len(distances) else None
                    }
                    results.append(result)

            return results

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"ChromaDB search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check ChromaDB backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - collection_exists: bool
            - collection_info: dict (count, metadata, etc.)
        """
        health_info = {}

        from ..exceptions import ConfigurationError, HealthCheckError, RetrieverConnectionError

        # Check if collection exists and is accessible
        try:
            if self.chroma_collection is None:
                raise ConfigurationError("ChromaDB collection is None", parameter="chroma_collection")

            # Get collection count
            count = self.chroma_collection.count()
            health_info["collection_exists"] = True
            health_info["collection_info"] = {
                "count": count,
                "name": self.chroma_collection.name,
                "metadata": self.chroma_collection.metadata if hasattr(self.chroma_collection, 'metadata') else None
            }
        except ConfigurationError:
            raise
        except Exception as e:
            raise HealthCheckError("ChromaDB", f"Collection '{self.collection}' not accessible", cause=e)

        # Verify connection by attempting to peek at collection
        try:
            # Try to peek at collection to verify it's accessible
            self.chroma_collection.peek(limit=1)
            health_info["connection_alive"] = True
        except Exception as e:
            raise RetrieverConnectionError("ChromaDB", cause=e)

        return health_info
