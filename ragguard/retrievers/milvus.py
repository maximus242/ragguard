"""
Milvus/Zilliz Secure Retriever for RAGGuard.

This module provides a secure retriever for Milvus (open-source) and Zilliz Cloud
(managed service) with policy-based access control.

Milvus is a high-performance vector database designed for AI applications.
Zilliz Cloud is the fully-managed version of Milvus.

Example:
    ```python
    from pymilvus import MilvusClient
    from ragguard import Policy
    from ragguard.retrievers import MilvusSecureRetriever

    # Connect to Milvus
    client = MilvusClient(uri="http://localhost:19530")

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {"conditions": ["user.department == document.department"]}
        }],
        "default": "deny"
    })

    # Create secure retriever
    retriever = MilvusSecureRetriever(
        client=client,
        collection_name="my_collection",
        policy=policy
    )

    # Search with user context
    user = {"id": "alice", "department": "engineering"}
    results = retriever.search(
        query=[0.1, 0.2, ...],  # embedding vector
        user=user,
        limit=10
    )
    ```

Requirements:
    - pymilvus>=2.3.0 (Milvus Python SDK)
"""

from typing import Any, Dict, List, Optional, Union

try:
    from pymilvus import Collection, MilvusClient, connections
    from pymilvus.client.types import LoadState
except ImportError:
    MilvusClient = None
    Collection = None
    connections = None

from typing import TYPE_CHECKING

from ..audit import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..logging import get_logger
from ..policy import Policy
from ..retry import RetryConfig
from .base import BaseSecureRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig

logger = get_logger(__name__)


class MilvusSecureRetriever(BaseSecureRetriever):
    """
    Secure retriever for Milvus/Zilliz with policy-based access control.

    This retriever supports both:
    - Milvus (open-source, self-hosted)
    - Zilliz Cloud (managed service)

    Attributes:
        client: Milvus client instance
        collection_name: Name of the Milvus collection
        policy: RAGGuard policy for access control
        vector_field: Name of the vector field (default: "vector")
        output_fields: List of metadata fields to return
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        policy: Policy,
        vector_field: str = "vector",
        output_fields: Optional[List[str]] = None,
        audit_logger: Optional[AuditLogger] = None,
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
        Initialize Milvus secure retriever.

        Args:
            client: Milvus client (MilvusClient or legacy connection)
            collection_name: Name of the Milvus collection
            policy: RAGGuard policy for access control
            vector_field: Name of the vector field in collection (default: "vector")
            output_fields: List of metadata fields to retrieve (default: all)
            audit_logger: Optional audit logger
            enable_filter_cache: Whether to enable filter caching
            filter_cache_size: Maximum size of filter cache
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic
            validation_config: Optional validation configuration (uses defaults if not provided)
            enable_validation: Whether to enable input validation (default: True)
            enable_circuit_breaker: Whether to enable circuit breaker (default: True)
            circuit_breaker_config: Optional circuit breaker configuration
            fail_on_audit_error: If True, raise an error when audit logging fails.
                                 If False (default), log a warning and continue.

        Raises:
            ImportError: If pymilvus is not installed

        Note:
            custom_filter_builder is not supported for Milvus because Milvus uses
            a string-based filter expression syntax that requires post-filtering
            for complex permission scenarios. Use the standard policy-based filtering.
        """
        if MilvusClient is None:
            raise ImportError(
                "pymilvus is required for Milvus integration. "
                "Install it with: pip install pymilvus"
            )

        super().__init__(
            client=client,
            collection=collection_name,
            policy=policy,
            audit_logger=audit_logger,
            custom_filter_builder=None,  # Not supported for Milvus (see docstring)
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

        # Store Milvus-specific attributes
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.output_fields = output_fields or ["*"]

        # Check if collection exists and is loaded
        self._verify_collection()

    def _verify_collection(self):
        """Verify that the collection exists and is loaded."""
        try:
            # MilvusClient API
            if hasattr(self.client, 'describe_collection'):
                collection_info = self.client.describe_collection(self.collection_name)
                if not collection_info:
                    raise ValueError(f"Collection '{self.collection_name}' not found")
            # Legacy API
            elif Collection is not None:
                collection = Collection(self.collection_name)
                # Load collection if not loaded
                if hasattr(collection, 'load_state'):
                    if collection.load_state != LoadState.Loaded:
                        logger.info(
                            "Loading collection",
                            extra={
                                "extra_fields": {
                                    "collection_name": self.collection_name
                                }
                            }
                        )
                        collection.load()
        except Exception as e:
            logger.warning(
                "Could not verify collection",
                extra={
                    "extra_fields": {
                        "collection_name": self.collection_name,
                        "error": str(e)
                    }
                }
            )

    @property
    def backend_name(self) -> str:
        """Return backend name for filter generation."""
        return "milvus"

    def _execute_search(
        self,
        query: List[float],
        filter: Optional[str],
        limit: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute Milvus search with permission filter and post-filtering.

        Since Milvus native filtering may not support all policy expressions,
        we fetch extra results (limit * 3) and post-filter them.

        Args:
            query: Query embedding vector
            filter: Milvus filter expression string
            limit: Maximum results requested
            **kwargs: Additional Milvus search arguments

        Returns:
            List of filtered result dicts with id, metadata, distance, score
        """
        # Get search params and user from kwargs (passed by search() and base class)
        # This is thread-safe as each search call has its own kwargs
        search_params = kwargs.pop('_search_params', {"metric_type": "L2", "params": {"nprobe": 10}})
        user = kwargs.pop('_user', {})
        kwargs.pop('_search_stats', None)  # Milvus doesn't use search stats

        # Fetch extra results for post-filtering
        fetch_limit = limit * 3

        try:
            # MilvusClient API (new)
            if hasattr(self.client, 'search'):
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query],
                    filter=filter if filter else None,
                    limit=fetch_limit,
                    output_fields=self.output_fields,
                    search_params=search_params,
                    **kwargs
                )

                # Extract results from MilvusClient response
                if results and len(results) > 0:
                    hits = results[0]  # First query results
                else:
                    hits = []

            # Legacy Collection API
            elif Collection is not None:
                collection = Collection(self.collection_name)

                search_results = collection.search(
                    data=[query],
                    anns_field=self.vector_field,
                    param=search_params,
                    limit=fetch_limit,
                    expr=filter if filter else None,
                    output_fields=self.output_fields,
                    **kwargs
                )

                hits = search_results[0] if search_results else []

            else:
                raise RuntimeError("No valid Milvus client available")

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            from ..exceptions import RetrieverError
            raise RetrieverError(f"Milvus search failed: {e}")

        # Post-filter results with policy evaluation
        filtered_results = self._post_filter_results(hits, user)

        # Return up to the requested limit
        return filtered_results[:limit]

    def search(
        self,
        query: Union[List[float], str],
        user: Dict[str, Any],
        limit: int = 10,
        metric_type: str = "L2",
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search Milvus collection with policy enforcement.

        Note: This method stores user context temporarily for post-filtering,
        then calls the parent search() method which handles validation,
        audit logging, and retry logic.

        Args:
            query: Query vector (list of floats) or text (if using text search)
            user: User context dictionary
            limit: Maximum number of results to return
            metric_type: Distance metric ("L2", "IP", "COSINE", etc.)
            search_params: Milvus search parameters (e.g., {"nprobe": 10})
            **kwargs: Additional arguments passed to Milvus search

        Returns:
            List of authorized documents with metadata

        Example:
            >>> user = {"id": "alice", "department": "engineering"}
            >>> results = retriever.search(
            ...     query=[0.1, 0.2, 0.3, ...],
            ...     user=user,
            ...     limit=10,
            ...     metric_type="COSINE"
            ... )
        """
        # Pass search params through kwargs - base class will pass _user automatically
        # This is thread-safe as each search call has its own kwargs
        actual_search_params = search_params or {"metric_type": metric_type, "params": {"nprobe": 10}}
        return super().search(
            query=query,
            user=user,
            limit=limit,
            _search_params=actual_search_params,
            **kwargs
        )

    def _post_filter_results(
        self,
        hits: List[Any],
        user: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Post-filter results based on policy evaluation.

        This is needed because Milvus native filtering may not support
        all policy expression features. We evaluate each result against
        the policy to ensure only authorized documents are returned.

        Args:
            hits: Milvus search hits
            user: User context

        Returns:
            Filtered results as normalized dicts
        """
        filtered = []

        for hit in hits:
            # Extract metadata from hit
            if hasattr(hit, 'entity'):
                # Legacy Collection API
                metadata = hit.entity.fields
                doc_id = hit.entity.id if hasattr(hit.entity, 'id') else hit.id
                distance = hit.distance
            elif isinstance(hit, dict):
                # MilvusClient API
                metadata = {k: v for k, v in hit.items() if k not in ['id', 'distance']}
                doc_id = hit.get('id')
                distance = hit.get('distance', 0.0)
            else:
                # Fallback: try to extract fields
                metadata = {k: getattr(hit, k, None) for k in self.output_fields if k != "*"}
                doc_id = getattr(hit, 'id', None)
                distance = getattr(hit, 'distance', 0.0)

            # Evaluate policy using the shared policy engine
            if self.policy_engine.evaluate(user, metadata):
                filtered.append({
                    "id": doc_id,
                    "metadata": metadata,
                    "distance": distance,
                    "score": 1.0 / (1.0 + distance) if distance is not None else 0.0
                })

        return filtered

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Milvus backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool
            - collection_exists: bool
            - collection_info: dict (entity count, load state, etc.)
        """
        from ..exceptions import ConfigurationError, HealthCheckError, RetrieverConnectionError

        health_info = {}

        # Check if collection exists and get stats
        try:
            # MilvusClient API
            if hasattr(self.client, 'describe_collection'):
                collection_info = self.client.describe_collection(self.collection_name)
                if not collection_info:
                    raise HealthCheckError("Milvus", f"Collection '{self.collection_name}' not found")

                health_info["collection_exists"] = True

                # Get collection stats
                try:
                    stats = self.client.get_collection_stats(self.collection_name)
                    row_count = stats.get('row_count', 0) if isinstance(stats, dict) else 0

                    health_info["collection_info"] = {
                        "row_count": row_count,
                        "collection_name": self.collection_name
                    }
                except Exception as stats_error:
                    # If stats fail, still report collection exists
                    health_info["collection_info"] = {
                        "collection_name": self.collection_name,
                        "stats_error": str(stats_error)
                    }

            # Legacy Collection API
            elif Collection is not None:
                collection = Collection(self.collection_name)
                health_info["collection_exists"] = True

                # Get entity count
                try:
                    num_entities = collection.num_entities
                    load_state = collection.load_state if hasattr(collection, 'load_state') else None

                    health_info["collection_info"] = {
                        "num_entities": num_entities,
                        "load_state": str(load_state) if load_state else None,
                        "collection_name": self.collection_name
                    }
                except Exception as stats_error:
                    health_info["collection_info"] = {
                        "collection_name": self.collection_name,
                        "stats_error": str(stats_error)
                    }
            else:
                raise ConfigurationError("No valid Milvus client available", parameter="client")

        except (HealthCheckError, ConfigurationError):
            raise
        except Exception as e:
            raise HealthCheckError("Milvus", f"Collection '{self.collection_name}' not accessible", cause=e)

        # Verify connection
        try:
            # Try to list collections to verify connection
            if hasattr(self.client, 'list_collections'):
                self.client.list_collections()
            elif connections is not None:
                # Legacy API - connections should be available if client is connected
                pass
            health_info["connection_alive"] = True
        except Exception as e:
            raise RetrieverConnectionError("Milvus", cause=e)

        return health_info


class ZillizSecureRetriever(MilvusSecureRetriever):
    """
    Secure retriever for Zilliz Cloud (managed Milvus).

    This is an alias for MilvusSecureRetriever since Zilliz uses the same API.

    Example:
        ```python
        from pymilvus import MilvusClient
        from ragguard.retrievers import ZillizSecureRetriever

        # Connect to Zilliz Cloud
        client = MilvusClient(
            uri="https://your-cluster.zillizcloud.com",
            token="your-api-key"
        )

        retriever = ZillizSecureRetriever(
            client=client,
            collection_name="my_collection",
            policy=policy
        )
        ```
    """
    pass


__all__ = ["MilvusSecureRetriever", "ZillizSecureRetriever"]
