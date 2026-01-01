"""
FAISS secure retriever with permission-aware search.

FAISS doesn't support native metadata filtering, so this retriever:
1. Performs an over-fetch (retrieves more results than requested)
2. Post-filters results based on permissions
3. Returns the top N permitted results

This is less efficient than native filtering but provides compatibility with FAISS.
"""

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..exceptions import RetrieverError
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .base import BaseSecureRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig

logger = get_logger(__name__)


class FAISSSecureRetriever(BaseSecureRetriever):
    """
    Permission-aware retriever for FAISS.

    Since FAISS doesn't support metadata filtering, this retriever
    over-fetches results and filters them in memory based on permissions.

    Example:
        ```python
        import faiss
        import numpy as np
        from ragguard import FAISSSecureRetriever, load_policy

        # Create FAISS index
        dimension = 768
        index = faiss.IndexFlatL2(dimension)

        # Add vectors with metadata tracking
        vectors = np.random.random((1000, dimension)).astype('float32')
        index.add(vectors)

        # Metadata stored separately (FAISS doesn't store metadata)
        metadata = [
            {"id": i, "department": "engineering", "public": False}
            for i in range(1000)
        ]

        policy = load_policy("policy.yaml")

        retriever = FAISSSecureRetriever(
            index=index,
            metadata=metadata,
            policy=policy,
            embed_fn=embeddings.embed_query,
            over_fetch_factor=3  # Fetch 3x results for filtering
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
        metadata: List[Dict[str, Any]],
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        over_fetch_factor: int = 3,
        adaptive_fetch: bool = True,
        max_over_fetch_factor: int = 10,
        max_absolute_fetch: int = 100000,
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
        Initialize FAISS secure retriever.

        Args:
            index: FAISS Index instance
            metadata: List of metadata dicts (one per vector in index)
            policy: Access control policy
            embed_fn: Optional function to convert text to embeddings
            audit_logger: Optional audit logger
            over_fetch_factor: Initial over-fetch multiplier for filtering (default: 3)
                              Higher values increase chance of getting enough results but are slower
            adaptive_fetch: Whether to automatically increase over-fetch if initial results are
                           insufficient (default: True). This helps with restrictive policies.
            max_over_fetch_factor: Maximum over-fetch multiplier when adaptive_fetch is enabled
                                  (default: 10). Prevents excessive fetching.
            max_absolute_fetch: Absolute maximum number of vectors to fetch per search
                               (default: 100000). Provides memory safety regardless of
                               index size and limit values.
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

        Note:
            custom_filter_builder is not supported for FAISS because FAISS doesn't support
            native metadata filtering. This retriever uses post-filtering via policy engine.
        """
        try:
            import faiss
        except ImportError:
            raise RetrieverError(
                "faiss not installed. Install with: pip install ragguard[faiss]"
            )

        self.index = index
        self.metadata = metadata
        # Ensure max_over_fetch_factor is at least 1
        self.max_over_fetch_factor = max(1, max_over_fetch_factor)
        # Clamp over_fetch_factor to be between 1 and max_over_fetch_factor
        self.over_fetch_factor = max(1, min(over_fetch_factor, self.max_over_fetch_factor))
        self.adaptive_fetch = adaptive_fetch
        self.max_absolute_fetch = max(1, max_absolute_fetch)

        # Validate metadata length matches index size
        if len(metadata) != index.ntotal:
            logger.warning(
                "Metadata length does not match index size",
                extra={
                    "extra_fields": {
                        "metadata_length": len(metadata),
                        "index_size": index.ntotal
                    }
                }
            )

        super().__init__(
            client=None,  # FAISS doesn't use a client
            collection="",  # FAISS doesn't have collections
            policy=policy,
            audit_logger=audit_logger,
            embed_fn=embed_fn,
            custom_filter_builder=None,  # Not supported for FAISS (see docstring)
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
        # Note: policy_engine is created by parent __init__, no need to create it here

        # Warn users about post-filtering limitations
        warnings.warn(
            "FAISSSecureRetriever uses post-filtering (not native database filtering). "
            "This provides security but may return fewer results than requested for "
            "restrictive policies. For production at scale, consider Qdrant, pgvector, "
            "Pinecone, Weaviate, or ChromaDB which support native filtering.",
            UserWarning,
            stacklevel=2
        )
        logger.warning(
            "FAISS uses post-filtering - see documentation for limitations",
            extra={"extra_fields": {"backend": "faiss", "over_fetch_factor": self.over_fetch_factor}}
        )

    @property
    def backend_name(self) -> str:
        """Return backend name for filter generation."""
        return "faiss"

    def _execute_search(
        self,
        query: List[float],
        filter: Any,  # Unused for FAISS
        limit: int,
        **kwargs
    ) -> List[Any]:
        """
        Execute FAISS search with post-filtering and adaptive over-fetching.

        Args:
            query: Query embedding
            filter: Unused (FAISS doesn't support native filtering)
            limit: Maximum number of results to return
            **kwargs: Additional parameters including 'user' for permission checks

        Returns:
            List of filtered results with metadata
        """
        try:
            import numpy as np

            # Get user context and stats container from kwargs (passed by base class search method)
            # This is thread-safe as each search call has its own kwargs
            user = kwargs.pop('_user', {})
            search_stats = kwargs.pop('_search_stats', {})

            # Handle empty index case
            if self.index.ntotal == 0:
                return []

            # Convert query vector to numpy array
            query_array = np.array([query], dtype=np.float32)

            # Start with initial over-fetch factor
            current_factor = self.over_fetch_factor
            results: List[Dict[str, Any]] = []
            seen_indices = set()
            total_documents_checked = 0
            total_documents_filtered = 0

            while len(results) < limit:
                # Calculate fetch limit for this iteration with bounds protection
                fetch_limit = limit * current_factor
                # Apply bounds: index size and absolute maximum
                fetch_limit = min(fetch_limit, self.index.ntotal, self.max_absolute_fetch)

                # Search FAISS index
                distances, indices = self.index.search(query_array, fetch_limit)

                # Process results - check for empty results
                if indices is None or len(indices) == 0 or len(indices[0]) == 0:
                    break  # No more results possible

                for i, idx in enumerate(indices[0]):
                    if idx == -1:  # FAISS returns -1 for padding
                        continue

                    # Skip already-seen indices (from previous iterations)
                    if idx in seen_indices:
                        continue
                    seen_indices.add(idx)

                    if idx >= len(self.metadata):
                        continue  # Skip if metadata missing

                    doc_metadata = self.metadata[idx]
                    total_documents_checked += 1

                    # Check permissions using policy engine
                    if self.policy_engine.evaluate(user, doc_metadata):
                        result = {
                            "id": doc_metadata.get("id", idx),
                            "metadata": doc_metadata,
                            "distance": float(distances[0][i]),
                            "score": 1.0 / (1.0 + float(distances[0][i]))  # Convert distance to similarity score
                        }
                        results.append(result)

                        # Stop when we have enough results
                        if len(results) >= limit:
                            break
                    else:
                        total_documents_filtered += 1

                # Check if we should try again with larger fetch
                if len(results) < limit and self.adaptive_fetch:
                    # Double the factor for next iteration
                    next_factor = current_factor * 2

                    # Stop if we've hit any limit: max factor, entire index, or absolute max
                    if (next_factor > self.max_over_fetch_factor or
                        fetch_limit >= self.index.ntotal or
                        fetch_limit >= self.max_absolute_fetch):
                        break

                    current_factor = next_factor
                    logger.debug(
                        f"Adaptive fetch: increasing factor to {current_factor} "
                        f"(have {len(results)}/{limit} results after checking {total_documents_checked} docs)"
                    )
                else:
                    break

            # Store filtering stats in the thread-safe container for audit logging
            search_stats.update({
                'documents_checked': total_documents_checked,
                'documents_filtered': total_documents_filtered,
                'documents_returned': len(results),
                'final_over_fetch_factor': current_factor,
                'adaptive_fetch_used': current_factor > self.over_fetch_factor
            })

            return results

        except (ConnectionError, TimeoutError, OSError):
            # Let retryable exceptions pass through for retry decorator
            raise
        except Exception as e:
            raise RetrieverError(f"FAISS search failed: {e}")

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check FAISS backend health.

        Returns:
            Dictionary with health information:
            - connection_alive: bool (always True for FAISS as it's local)
            - index_valid: bool
            - index_info: dict (size, dimension, metadata length, etc.)
        """
        from ..exceptions import ConfigurationError, HealthCheckError

        health_info = {}

        # Check if index is valid
        try:
            if self.index is None:
                raise ConfigurationError("FAISS index is None", parameter="index")

            # Get index size
            ntotal = self.index.ntotal
            health_info["index_valid"] = True

            # Get index dimension
            dimension = self.index.d if hasattr(self.index, 'd') else None

            # Check if index is trained (for indexes that require training)
            is_trained = self.index.is_trained if hasattr(self.index, 'is_trained') else True

            health_info["index_info"] = {
                "ntotal": ntotal,
                "dimension": dimension,
                "is_trained": is_trained
            }

            # Verify metadata length matches index size
            metadata_length = len(self.metadata)
            if metadata_length != ntotal:
                health_info["index_info"]["metadata_mismatch"] = True
                health_info["index_info"]["metadata_length"] = metadata_length
                raise HealthCheckError(
                    "FAISS",
                    f"Metadata length ({metadata_length}) does not match index size ({ntotal})"
                )
            else:
                health_info["index_info"]["metadata_length"] = metadata_length
                health_info["index_info"]["metadata_mismatch"] = False

        except (HealthCheckError, ConfigurationError):
            raise
        except Exception as e:
            raise HealthCheckError("FAISS", "Index validation failed", cause=e)

        # FAISS is a local in-memory index, so connection is always alive if index is valid
        health_info["connection_alive"] = True

        return health_info

    def search(
        self,
        query: Any,
        user: dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Override search to pass user context to _execute_search.

        FAISS doesn't support native filtering, so we need to pass
        the user context through to _execute_search for post-filtering.
        """
        # User context is passed through to _execute_search via kwargs by the base class
        # No need for thread-local storage - the base class passes _user in kwargs
        return super().search(query=query, user=user, limit=limit, **kwargs)
