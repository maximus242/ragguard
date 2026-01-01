"""
Abstract base class for secure retrievers.

Defines the interface for permission-aware vector search across different
database backends.
"""

import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ..audit.logger import AuditLogger, NullAuditLogger
from ..circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    get_circuit_breaker,
)
from ..config import SecureRetrieverConfig
from ..exceptions import AuditLogError, RetrieverError, RetrieverTimeoutError
from ..logging import get_logger
from ..policy.engine import PolicyEngine
from ..policy.models import Policy
from ..retry import RetryConfig, retry_on_failure
from ..types import (
    EmbeddingVector,
    FilterResult,
    FilterResultType,
    FilterType,
    VectorDatabaseClient,
)
from ..validation import InputValidator, ValidationConfig

if TYPE_CHECKING:
    from ..filters.custom import CustomFilterBuilder

# Create logger for this module
logger = get_logger(__name__)

# Import metrics collector (optional - gracefully handle if not available)
try:
    from ..metrics import get_metrics_collector
    _metrics_available = True
except ImportError:
    _metrics_available = False
    get_metrics_collector = None  # type: ignore


class BaseSecureRetriever(ABC):
    """
    Base class for permission-aware retrievers.

    Wraps a vector database client and enforces document-level permissions
    by injecting filters into the vector search.
    """

    def __init__(
        self,
        client: VectorDatabaseClient,
        collection: str,
        policy: Policy,
        audit_logger: Optional[AuditLogger] = None,
        embed_fn: Optional[Callable[[str], EmbeddingVector]] = None,
        custom_filter_builder: Optional['CustomFilterBuilder'] = None,
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
        config: Optional[SecureRetrieverConfig] = None
    ):
        """
        Initialize the secure retriever.

        Args:
            client: Vector database client (e.g., QdrantClient)
            collection: Collection/table name
            policy: Access control policy
            audit_logger: Optional audit logger (defaults to NullAuditLogger)
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
            fail_on_audit_error: If True, raise an error when audit logging fails (fail-closed).
                                 If False (default), log a warning and continue (fail-open).
                                 Set to True for compliance-critical applications where
                                 missing audit logs are unacceptable.
            config: Optional SecureRetrieverConfig that consolidates all settings.
                   When provided, config values override individual parameters.
                   Use this for cleaner configuration:

                   Example:
                       config = SecureRetrieverConfig.production()
                       retriever = QdrantSecureRetriever(client, "docs", policy, config=config)
        """
        # Apply config values if provided (config overrides individual params)
        if config is not None:
            enable_filter_cache = config.enable_filter_cache
            filter_cache_size = config.filter_cache_size
            retry_config = config.retry_config
            enable_retry = config.enable_retry
            validation_config = config.validation_config
            enable_validation = config.enable_validation
        self.client = client
        self.collection = collection

        # Thread lock for policy updates (ensures atomic policy/engine updates)
        self._policy_lock = threading.RLock()

        # Store cache configuration for policy updates
        self._enable_filter_cache = enable_filter_cache
        self._filter_cache_size = filter_cache_size

        # Store retry configuration
        self._enable_retry = enable_retry
        self._retry_config = retry_config or RetryConfig(
            max_retries=3,
            initial_delay=0.1,
            max_delay=10.0,
            exponential_base=2,
            jitter=True
        )

        # Store validation configuration
        self._enable_validation = enable_validation
        self._validator = InputValidator(validation_config or ValidationConfig())

        # Store circuit breaker configuration
        self._enable_circuit_breaker = enable_circuit_breaker
        self._circuit_breaker_config = circuit_breaker_config

        # Use property setter to initialize policy and engine
        self._policy = policy
        self.policy_engine = PolicyEngine(
            policy,
            enable_filter_cache=enable_filter_cache,
            filter_cache_size=filter_cache_size
        )

        self.audit_logger = audit_logger or NullAuditLogger()
        self._fail_on_audit_error = fail_on_audit_error
        self.embed_fn = embed_fn
        self.custom_filter_builder = custom_filter_builder

        # Initialize circuit breaker (uses shared instance per backend)
        if self._enable_circuit_breaker:
            self._circuit_breaker = get_circuit_breaker(
                self.backend_name,
                self._circuit_breaker_config
            )
        else:
            self._circuit_breaker = None

        # Lazy-initialized executor for timeout handling
        # Using a class-level executor prevents memory leaks from creating
        # new executors for each search, and allows proper cleanup on close()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()

        # Log initialization
        logger.info(
            "Retriever initialized",
            extra={
                "extra_fields": {
                    "backend": self.backend_name,
                    "collection": collection,
                    "filter_cache_enabled": enable_filter_cache,
                    "retry_enabled": enable_retry,
                    "validation_enabled": enable_validation,
                    "circuit_breaker_enabled": enable_circuit_breaker
                }
            }
        )

    @property
    def policy(self) -> Policy:
        """Get the current access control policy (thread-safe)."""
        with self._policy_lock:
            return self._policy

    def _get_policy_engine_snapshot(self) -> PolicyEngine:
        """
        Get a thread-safe snapshot of the current policy engine.

        This ensures that the policy engine used throughout a single search
        operation remains consistent, even if the policy is updated mid-search.

        Returns:
            The current PolicyEngine instance
        """
        with self._policy_lock:
            return self.policy_engine

    @policy.setter
    def policy(self, new_policy: Policy) -> None:
        """
        Update the access control policy.

        This safely updates the policy by recreating the PolicyEngine with the
        new policy while preserving cache configuration. The cache is automatically
        invalidated since the new policy will have a different hash.

        Args:
            new_policy: The new access control policy

        Example:
            >>> retriever = QdrantSecureRetriever(...)
            >>> # Update policy (cache is automatically handled)
            >>> retriever.policy = load_policy("new_policy.yaml")
        """
        logger.info(
            "Updating policy",
            extra={
                "extra_fields": {
                    "backend": self.backend_name,
                    "collection": self.collection,
                    "old_policy_version": self._policy.version if hasattr(self._policy, 'version') else None,
                    "new_policy_version": new_policy.version if hasattr(new_policy, 'version') else None
                }
            }
        )

        # Recreate the policy engine with the new policy FIRST
        # This ensures the engine's policy hash matches the new policy
        # Creating the engine first avoids race condition where _policy is updated
        # but policy_engine is still old
        new_engine = PolicyEngine(
            new_policy,
            enable_filter_cache=self._enable_filter_cache,
            filter_cache_size=self._filter_cache_size
        )

        # Use lock to ensure atomic update of both policy and engine
        # This prevents race conditions where threads see mismatched policy/engine
        with self._policy_lock:
            self._policy = new_policy
            self.policy_engine = new_engine

        logger.info(
            "Policy updated successfully",
            extra={
                "extra_fields": {
                    "backend": self.backend_name,
                    "filter_cache_enabled": self._enable_filter_cache
                }
            }
        )

    def search(
        self,
        query: Union[str, list[float]],
        user: dict[str, Any],
        limit: int = 10,
        timeout: Optional[float] = None,
        **kwargs
    ) -> list[Any]:
        """
        Execute permission-aware vector search.

        This is the main entry point. It:
        1. Validates user context
        2. Converts query to embedding if needed
        3. Builds permission filter from policy + user context
        4. Executes search with filter injected
        5. Logs the query for audit
        6. Returns results

        Args:
            query: Query text (if embed_fn provided) or embedding vector
            user: User context (id, roles, department, etc.)
            limit: Maximum number of results
            timeout: Optional timeout in seconds for the search operation.
                    If None, uses the default timeout from retry_config.
                    If 0 or negative, no timeout is applied.
            **kwargs: Additional backend-specific arguments

        Returns:
            List of results (format depends on backend)

        Raises:
            RetrieverError: If search fails or validation fails
            RetrieverTimeoutError: If search exceeds the specified timeout
        """
        # Track query start time and metrics
        query_start_time = time.time()
        error_type = None
        cache_hit = False

        # Increment active queries counter
        if _metrics_available:
            collector = get_metrics_collector()
            if collector.is_enabled():
                collector.query_started()

        try:
            # Validate user context
            if self._enable_validation:
                try:
                    self._validator.validate_user_context(user)
                except Exception as e:
                    error_type = "ValidationError"
                    logger.warning(
                        "User context validation failed",
                        extra={
                            "extra_fields": {
                                "backend": self.backend_name,
                                "user_id": user.get("id") if isinstance(user, dict) else None,
                                "error": str(e)
                            }
                        }
                    )
                    raise

            # Log search request
            logger.debug(
                "Starting search",
                extra={
                    "extra_fields": {
                        "backend": self.backend_name,
                        "collection": self.collection,
                        "user_id": user.get("id") if isinstance(user, dict) else None,
                        "limit": limit,
                        "query_type": "text" if isinstance(query, str) else "vector"
                    }
                }
            )

            # Convert text query to embedding if needed
            if isinstance(query, str):
                if self.embed_fn is None:
                    error_type = "ConfigurationError"
                    raise RetrieverError(
                        "Query is a string but no embed_fn was provided. "
                        "Either provide embed_fn or pass embeddings directly."
                    )
                query_vector = self.embed_fn(query)
                query_for_log = query
            else:
                query_vector = query
                query_for_log = "[vector]"

            # Build the permission filter (with timing)
            filter_build_start = time.time()
            try:
                # Get thread-safe snapshot of policy engine for this search
                # This prevents race conditions if policy is updated mid-search
                policy_engine_snapshot = self._get_policy_engine_snapshot()

                if self.custom_filter_builder:
                    # Use custom filter builder for complex permission scenarios
                    native_filter = self.custom_filter_builder.build_filter(
                        self.policy,
                        user,
                        self.backend_name
                    )

                    # Validate the custom filter result for common issues
                    builder_name = type(self.custom_filter_builder).__name__
                    self._validate_custom_filter_result(native_filter, user, builder_name)

                    logger.debug(
                        "Built custom permission filter",
                        extra={
                            "extra_fields": {
                                "backend": self.backend_name,
                                "user_id": user.get("id") if isinstance(user, dict) else None
                            }
                        }
                    )
                else:
                    # Use standard policy engine (thread-safe snapshot)
                    native_filter = policy_engine_snapshot.to_filter(user, self.backend_name)

                    # Check if this specific call was a cache hit (thread-safe)
                    cache_hit = policy_engine_snapshot.was_last_call_cache_hit()

                    logger.debug(
                        "Built permission filter from policy",
                        extra={
                            "extra_fields": {
                                "backend": self.backend_name,
                                "user_id": user.get("id") if isinstance(user, dict) else None,
                                "filter_cached": cache_hit
                            }
                        }
                    )

                # Record filter build time
                filter_build_duration = time.time() - filter_build_start
                if _metrics_available:
                    collector = get_metrics_collector()
                    if collector.is_enabled():
                        collector.record_filter_build(filter_build_duration)

                        # Update cache size gauge
                        cache_stats = policy_engine_snapshot.get_cache_stats()
                        if cache_stats:
                            collector.update_cache_size(cache_stats['size'])

            except Exception as e:
                error_type = "FilterBuildError"
                logger.error(
                    "Failed to build permission filter",
                    extra={
                        "extra_fields": {
                            "backend": self.backend_name,
                            "user_id": user.get("id") if isinstance(user, dict) else None,
                            "error": str(e)
                        }
                    }
                )
                raise RetrieverError(f"Failed to build permission filter: {e}")

            # Execute search with permission filter (with timeout, circuit breaker and retry if enabled)
            try:
                # Check circuit breaker BEFORE executing search
                if self._circuit_breaker is not None:
                    self._circuit_breaker.check()  # Raises CircuitBreakerOpen if open

                # Determine effective timeout
                effective_timeout = timeout
                if effective_timeout is None:
                    # Use default from retry config if available
                    effective_timeout = getattr(self._retry_config, 'request_timeout', None)

                # Define the search function
                # Pass user context to _execute_search via kwargs
                # This is needed for FAISS/Milvus which need user for post-filtering
                # Also pass a stats container for thread-safe stats collection (used by FAISS)
                search_stats: dict[str, Any] = {}
                search_kwargs = {**kwargs, '_user': user, '_search_stats': search_stats}

                def _do_search():
                    if self._enable_retry:
                        @retry_on_failure(config=self._retry_config)
                        def _search_with_retry():
                            return self._execute_search(
                                query_vector,
                                native_filter,
                                limit,
                                **search_kwargs
                            )
                        return _search_with_retry()
                    else:
                        return self._execute_search(
                            query_vector,
                            native_filter,
                            limit,
                            **search_kwargs
                        )

                # Execute with timeout if specified
                if effective_timeout is not None and effective_timeout > 0:
                    results = self._run_with_timeout(_do_search, effective_timeout)
                else:
                    results = _do_search()

                # Record success with circuit breaker
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()

            except CircuitBreakerOpen:
                # Don't record circuit breaker exceptions as failures
                error_type = "CircuitBreakerOpen"
                raise
            except RetrieverTimeoutError:
                # Timeout is a retryable failure
                error_type = "TimeoutError"
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                raise
            except Exception as e:
                # Record failure with circuit breaker (for non-circuit-breaker exceptions)
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                error_type = "SearchExecutionError"
                logger.error(
                    "Search execution failed",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "backend": self.backend_name,
                            "collection": self.collection,
                            "user_id": user.get("id") if isinstance(user, dict) else None,
                            "error": str(e)
                        }
                    }
                )

                # Audit log the failed search attempt
                try:
                    self.audit_logger.log(
                        user=user,
                        query=query_for_log,
                        results_count=0,
                        filter_applied=native_filter,
                        additional_info={"error": str(e), "error_type": error_type}
                    )
                except Exception as audit_err:
                    if self._fail_on_audit_error:
                        raise AuditLogError(
                            message="Failed to log search error event",
                            cause=audit_err
                        )
                    logger.warning(
                        "Audit logging failed for error case",
                        extra={"extra_fields": {"error": str(audit_err)}}
                    )

                raise RetrieverError(f"Search failed: {e}")

            # Audit log
            try:
                # Collect additional info for audit log
                additional_info = {}

                # Include filtering stats if available (from FAISS post-filtering)
                # Stats are passed through the thread-safe search_stats container
                if search_stats:
                    additional_info.update(search_stats)

                self.audit_logger.log(
                    user=user,
                    query=query_for_log,
                    results_count=len(results),
                    filter_applied=native_filter,
                    additional_info=additional_info if additional_info else None
                )
            except Exception as audit_err:
                if self._fail_on_audit_error:
                    raise AuditLogError(
                        message="Failed to log search event",
                        cause=audit_err
                    )
                # Don't fail the search if logging fails (fail-open mode)
                logger.warning(
                    "Audit logging failed",
                    extra={"extra_fields": {"error": str(audit_err)}}
                )

            # Log successful search
            logger.info(
                "Search completed",
                extra={
                    "extra_fields": {
                        "backend": self.backend_name,
                        "collection": self.collection,
                        "user_id": user.get("id") if isinstance(user, dict) else None,
                        "results_count": len(results),
                        "requested_limit": limit
                    }
                }
            )

            return results

        except Exception as e:
            # Set error type if not already set
            if not error_type:
                error_type = type(e).__name__
            raise

        finally:
            # Record metrics (always runs, even if exception occurred)
            if _metrics_available:
                collector = get_metrics_collector()
                if collector.is_enabled():
                    query_duration = time.time() - query_start_time

                    # Decrement active queries counter
                    collector.query_finished()

                    # Record query metrics
                    result_count = len(results) if 'results' in locals() else 0
                    collector.record_query(
                        duration=query_duration,
                        backend=self.backend_name,
                        result_count=result_count,
                        cache_hit=cache_hit,
                        user_id=user.get("id") if isinstance(user, dict) else None,
                        error=error_type
                    )

    def batch_search(
        self,
        queries: list[Union[str, list[float]]],
        user: dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> list[list[Any]]:
        """
        Execute permission-aware batch vector search for multiple queries.

        This is optimized to build the permission filter ONCE for all queries,
        avoiding redundant filter generation overhead.

        This is useful for RAG applications that need to retrieve documents for
        multiple queries at once (e.g., sub-questions, multi-step reasoning).

        Args:
            queries: List of query texts or embedding vectors
            user: User context (id, roles, department, etc.)
            limit: Maximum number of results per query
            **kwargs: Additional backend-specific arguments

        Returns:
            List of result lists - one list per query

        Example:
            >>> queries = [
            ...     "What is our revenue?",
            ...     "Who are our customers?",
            ...     "What are our products?"
            ... ]
            >>> results = retriever.batch_search(
            ...     queries=queries,
            ...     user={"department": "finance"},
            ...     limit=5
            ... )
            >>> len(results)  # 3 result lists
            3
            >>> len(results[0])  # Up to 5 results for first query
            5

        Note:
            Subclasses can override this for more efficient batch operations
            (e.g., using backend's native batch API).
        """
        if not queries:
            return []

        # Validate user context ONCE for all queries
        if self._enable_validation:
            try:
                self._validator.validate_user_context(user)
            except Exception as e:
                logger.warning(
                    "User context validation failed in batch_search",
                    extra={
                        "extra_fields": {
                            "backend": self.backend_name,
                            "user_id": user.get("id") if isinstance(user, dict) else None,
                            "error": str(e),
                            "query_count": len(queries)
                        }
                    }
                )
                raise

        # Build the permission filter ONCE for all queries
        # This is the key optimization - avoid rebuilding filter per query
        filter_build_start = time.time()
        try:
            policy_engine_snapshot = self._get_policy_engine_snapshot()

            if self.custom_filter_builder:
                native_filter = self.custom_filter_builder.build_filter(
                    self.policy,
                    user,
                    self.backend_name
                )
                builder_name = type(self.custom_filter_builder).__name__
                self._validate_custom_filter_result(native_filter, user, builder_name)
            else:
                native_filter = policy_engine_snapshot.to_filter(user, self.backend_name)

            filter_build_duration = time.time() - filter_build_start
            logger.debug(
                "Built permission filter for batch search",
                extra={
                    "extra_fields": {
                        "backend": self.backend_name,
                        "user_id": user.get("id") if isinstance(user, dict) else None,
                        "query_count": len(queries),
                        "filter_build_ms": filter_build_duration * 1000
                    }
                }
            )

        except Exception as e:
            logger.error(
                "Failed to build permission filter for batch search",
                extra={
                    "extra_fields": {
                        "backend": self.backend_name,
                        "user_id": user.get("id") if isinstance(user, dict) else None,
                        "error": str(e)
                    }
                }
            )
            raise RetrieverError(f"Failed to build permission filter: {e}")

        # Execute searches with the pre-built filter
        all_results = []
        for i, query in enumerate(queries):
            try:
                # Convert text query to embedding if needed
                if isinstance(query, str):
                    if self.embed_fn is None:
                        raise RetrieverError(
                            "Query is a string but no embed_fn was provided. "
                            "Either provide embed_fn or pass embeddings directly."
                        )
                    query_vector = self.embed_fn(query)
                else:
                    query_vector = query

                # Execute the underlying search with pre-built filter
                # This bypasses the full search() overhead
                results = self._execute_search(
                    query=query_vector,
                    filter=native_filter,
                    limit=limit,
                    **kwargs
                )
                all_results.append(results)

            except Exception as e:
                logger.warning(
                    f"Batch search query {i} failed",
                    extra={
                        "extra_fields": {
                            "backend": self.backend_name,
                            "query_index": i,
                            "error": str(e)
                        }
                    }
                )
                # Re-raise to fail the batch (consistent with single search behavior)
                raise

        return all_results

    def invalidate_filter_cache(self) -> None:
        """
        Clear the filter cache.

        Note: You typically don't need to call this manually. The cache is
        automatically invalidated when you update the policy via the property
        setter (retriever.policy = new_policy).

        This method is useful for:
        - Testing: Force cache misses to verify behavior
        - Memory management: Clear cache to free memory
        - Manual cache control: Force filter regeneration

        Example:
            >>> retriever = QdrantSecureRetriever(...)
            >>> # Policy updates are automatic (no need to invalidate):
            >>> retriever.policy = new_policy  # Cache is auto-cleared
            >>>
            >>> # Manual invalidation for testing:
            >>> retriever.invalidate_filter_cache()
        """
        self.policy_engine.invalidate_cache()

    @property
    def cache_stats(self) -> Optional[dict[str, Any]]:
        """
        Get filter cache statistics.

        Returns:
            Dictionary with cache hit rate, size, etc., or None if caching is disabled.
            Contains:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Percentage of requests that hit the cache (0.0 to 1.0)
            - size: Current number of entries in cache
            - max_size: Maximum cache size

        Example:
            >>> stats = retriever.cache_stats
            >>> print(f"Cache hit rate: {stats['hit_rate']:.1%}")
            Cache hit rate: 95.3%
        """
        return self.policy_engine.get_cache_stats()

    # Backwards compatibility alias
    def get_cache_stats(self) -> Optional[dict[str, Any]]:
        """Deprecated: Use cache_stats property instead."""
        return self.cache_stats

    def preview_filter(
        self,
        user: dict[str, Any],
        format: str = "native"
    ) -> dict[str, Any]:
        """
        Preview the filter that would be generated for a user without executing a search.

        This is useful for:
        - Debugging policy configurations
        - Validating filter generation before deployment
        - Testing what access a user would have
        - Dry-run validation of policy changes

        Args:
            user: User context dictionary (same format as search())
            format: Output format - "native" (backend-specific) or "debug" (human-readable)

        Returns:
            Dictionary containing:
            - filter: The generated filter (format depends on backend)
            - backend: The backend name
            - user_id: User ID from context (if present)
            - policy_version: Current policy version
            - cache_hit: Whether the filter was retrieved from cache
            - would_deny_all: True if the filter would deny all access

        Example:
            >>> preview = retriever.preview_filter({"id": "alice", "department": "engineering"})
            >>> print(f"Filter for alice: {preview['filter']}")
            >>> if preview['would_deny_all']:
            ...     print("Warning: User has no access to any documents!")
        """
        # Get thread-safe snapshot of policy engine
        policy_engine_snapshot = self._get_policy_engine_snapshot()

        # Build the filter
        if self.custom_filter_builder:
            native_filter = self.custom_filter_builder.build_filter(
                self.policy,
                user,
                self.backend_name
            )
            # Validate the custom filter result
            builder_name = type(self.custom_filter_builder).__name__
            self._validate_custom_filter_result(native_filter, user, builder_name)
            cache_hit = False
        else:
            native_filter = policy_engine_snapshot.to_filter(user, self.backend_name)
            cache_hit = policy_engine_snapshot.was_last_call_cache_hit()

        # Determine if this is a deny-all filter
        would_deny_all = self._is_deny_all_filter(native_filter)

        result = {
            "filter": native_filter,
            "backend": self.backend_name,
            "user_id": user.get("id") if isinstance(user, dict) else None,
            "policy_version": self.policy.version if hasattr(self.policy, 'version') else None,
            "cache_hit": cache_hit,
            "would_deny_all": would_deny_all
        }

        if format == "debug":
            result["filter_str"] = str(native_filter)
            result["user_context"] = user

        return result

    def _is_deny_all_filter(self, native_filter: FilterType) -> bool:
        """
        Check if a filter would deny all access.

        This is backend-specific, so we check for common deny-all patterns.
        """
        if native_filter is None:
            return False

        # Check for common deny-all patterns
        if isinstance(native_filter, dict):
            # Some backends use empty dict to deny
            if native_filter == {}:
                return False  # Empty dict might mean no filter (allow all)
            # Check for explicit false conditions
            if native_filter.get("$and") == [{"__deny_all__": True}]:
                return True

        # Check for pgvector-style "WHERE FALSE"
        if isinstance(native_filter, tuple):
            clause, _ = native_filter
            if isinstance(clause, str) and "FALSE" in clause.upper():
                return True

        # Check for string-based deny patterns
        if isinstance(native_filter, str):
            if "FALSE" in native_filter.upper() or "1=0" in native_filter:
                return True

        return False

    def _validate_custom_filter_result(
        self,
        filter_obj: Any,
        user: dict[str, Any],
        builder_name: str = "custom"
    ) -> None:
        """
        Validate the result from a custom filter builder.

        SECURITY: This validation helps catch common mistakes in custom filter
        builders that could lead to security issues.

        Args:
            filter_obj: The filter returned by the custom builder
            user: User context (for logging)
            builder_name: Name of the builder for error messages

        Raises:
            RetrieverError: If filter is critically invalid (e.g., wrong type for backend)

        Logs warnings for ambiguous but not necessarily invalid values.
        """
        user_id = user.get("id", "unknown") if isinstance(user, dict) else "unknown"

        # Check for empty dict (ambiguous semantics)
        if isinstance(filter_obj, dict) and len(filter_obj) == 0:
            logger.warning(
                "Custom filter builder returned empty dict",
                extra={
                    "extra_fields": {
                        "builder": builder_name,
                        "user_id": user_id,
                        "backend": self.backend_name,
                        "issue": "Empty dict has ambiguous semantics. "
                                 "Use None for allow-all or a deny-all pattern for deny-all."
                    }
                }
            )

        # Check for empty list (ambiguous semantics)
        elif isinstance(filter_obj, list) and len(filter_obj) == 0:
            logger.warning(
                "Custom filter builder returned empty list",
                extra={
                    "extra_fields": {
                        "builder": builder_name,
                        "user_id": user_id,
                        "backend": self.backend_name,
                        "issue": "Empty list has ambiguous semantics."
                    }
                }
            )

        # Backend-specific type checks
        if self.backend_name == "pgvector":
            if filter_obj is not None and not isinstance(filter_obj, tuple):
                logger.warning(
                    "Custom filter builder returned wrong type for pgvector",
                    extra={
                        "extra_fields": {
                            "builder": builder_name,
                            "user_id": user_id,
                            "expected": "tuple[str, list]",
                            "got": type(filter_obj).__name__
                        }
                    }
                )
            elif isinstance(filter_obj, tuple):
                if len(filter_obj) != 2:
                    logger.warning(
                        "Custom filter builder returned invalid pgvector tuple",
                        extra={
                            "extra_fields": {
                                "builder": builder_name,
                                "user_id": user_id,
                                "expected": "tuple of length 2 (clause, params)",
                                "got": f"tuple of length {len(filter_obj)}"
                            }
                        }
                    )
                elif not isinstance(filter_obj[0], str):
                    logger.warning(
                        "Custom filter builder returned invalid pgvector clause",
                        extra={
                            "extra_fields": {
                                "builder": builder_name,
                                "user_id": user_id,
                                "expected": "str for WHERE clause",
                                "got": type(filter_obj[0]).__name__
                            }
                        }
                    )

        elif self.backend_name in ("chromadb", "pinecone", "weaviate", "elasticsearch"):
            if filter_obj is not None and not isinstance(filter_obj, dict):
                logger.warning(
                    f"Custom filter builder returned wrong type for {self.backend_name}",
                    extra={
                        "extra_fields": {
                            "builder": builder_name,
                            "user_id": user_id,
                            "expected": "dict",
                            "got": type(filter_obj).__name__
                        }
                    }
                )

        elif self.backend_name == "milvus":
            if filter_obj is not None and not isinstance(filter_obj, str):
                logger.warning(
                    "Custom filter builder returned wrong type for milvus",
                    extra={
                        "extra_fields": {
                            "builder": builder_name,
                            "user_id": user_id,
                            "expected": "str (expression)",
                            "got": type(filter_obj).__name__
                        }
                    }
                )

    @property
    def circuit_breaker_stats(self) -> Optional[dict[str, Any]]:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker state and stats, or None if disabled.
            Contains:
            - state: Current state (closed, open, half_open)
            - failure_count: Current consecutive failure count
            - success_count: Current consecutive success count (in half_open)
            - total_failures: Total failures since creation
            - total_successes: Total successes since creation
            - total_rejected: Total requests rejected while open

        Example:
            >>> stats = retriever.circuit_breaker_stats
            >>> if stats and stats['state'] == 'open':
            ...     print(f"Circuit open! Retry in {stats.get('time_until_retry', 'unknown')}s")
        """
        if self._circuit_breaker is None:
            return None

        stats = self._circuit_breaker.stats
        return {
            "state": stats.state.value,
            "failure_count": stats.failure_count,
            "success_count": stats.success_count,
            "total_failures": stats.total_failures,
            "total_successes": stats.total_successes,
            "total_rejected": stats.total_rejected,
            "last_failure_time": stats.last_failure_time,
            "last_state_change": stats.last_state_change
        }

    # Backwards compatibility alias
    def get_circuit_breaker_stats(self) -> Optional[dict[str, Any]]:
        """Deprecated: Use circuit_breaker_stats property instead."""
        return self.circuit_breaker_stats

    def reset_circuit_breaker(self) -> None:
        """
        Manually reset the circuit breaker to closed state.

        Use this to force the circuit closed after addressing backend issues.
        Only effective if circuit breaker is enabled.

        Example:
            >>> # After fixing the backend issue
            >>> retriever.reset_circuit_breaker()
        """
        if self._circuit_breaker is not None:
            self._circuit_breaker.reset()

    def _get_executor(self) -> ThreadPoolExecutor:
        """
        Get the shared executor, creating it lazily if needed.

        Uses double-checked locking for thread-safe lazy initialization.
        """
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    def _run_with_timeout(self, func: Callable[[], Any], timeout: float) -> Any:
        """
        Run a function with a timeout.

        Uses a shared ThreadPoolExecutor to run the function in a separate thread
        with a timeout. If the function doesn't complete within the timeout,
        raises RetrieverTimeoutError immediately without waiting for the thread.

        Note: On timeout, the background thread continues running but the caller
        gets control back immediately. The thread will complete naturally and
        the executor cleans up the thread when done.

        Args:
            func: Zero-argument callable to execute
            timeout: Timeout in seconds

        Returns:
            Result from the function

        Raises:
            RetrieverTimeoutError: If function doesn't complete within timeout
        """
        executor = self._get_executor()
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            # Attempt to cancel the future (won't stop running task, but marks it)
            future.cancel()

            logger.warning(
                "Search operation timed out",
                extra={
                    "extra_fields": {
                        "backend": self.backend_name,
                        "timeout": timeout
                    }
                }
            )
            raise RetrieverTimeoutError(
                backend=self.backend_name,
                operation="search",
                timeout=timeout
            )

    def close(self) -> None:
        """
        Close the retriever and release resources.

        This shuts down the internal thread pool executor. After calling close(),
        the retriever should not be used for searches.
        """
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None

    def __del__(self) -> None:
        """Clean up executor on garbage collection."""
        try:
            self.close()
        except Exception as e:
            # Log but don't raise during garbage collection
            # Raising in __del__ can cause issues during interpreter shutdown
            logger.debug(
                "Error during retriever cleanup",
                extra={"extra_fields": {"error": str(e)}}
            )

    @abstractmethod
    def _execute_search(
        self,
        query: EmbeddingVector,
        filter: FilterType,
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute the actual search with the native filter.

        This must be implemented by subclasses for each database backend.

        Args:
            query: Embedding vector
            filter: Native filter for this backend
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of results in backend-specific format
        """
        pass

    def __enter__(self):
        """
        Enter context manager.

        Allows using the retriever with the 'with' statement for automatic cleanup.

        Example:
            >>> with QdrantSecureRetriever(client, collection, policy) as retriever:
            ...     results = retriever.search(query, user)
            >>> # Auto-cleanup happens here
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager.

        Performs cleanup when exiting the 'with' block. This includes:
        - Closing database clients (if they have a close() method)
        - Cleaning up resources
        - Releasing connections back to pools

        Args:
            exc_type: Exception type (if an exception occurred)
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            False to propagate exceptions, True to suppress them
        """
        # Try to close various client/connection attributes
        # Different retrievers store their connections in different attributes
        for attr_name in ['client', 'connection', 'chroma_collection', 'index', '_managed_conn']:
            if hasattr(self, attr_name):
                resource = getattr(self, attr_name, None)
                if resource is not None and hasattr(resource, 'close'):
                    try:
                        close_method = resource.close
                        if callable(close_method):
                            close_method()
                    except Exception as e:
                        # Log cleanup failures to help diagnose connection leaks
                        logger.warning(
                            "Failed to close %s during cleanup: %s",
                            attr_name, str(e), exc_info=False
                        )

        # Close audit logger if it has a close() method
        if hasattr(self, 'audit_logger') and self.audit_logger is not None:
            if hasattr(self.audit_logger, 'close'):
                try:
                    close_method = self.audit_logger.close
                    if callable(close_method):
                        close_method()
                except Exception as e:
                    # Log cleanup failures to help diagnose issues
                    logger.warning(
                        "Failed to close audit_logger during cleanup: %s",
                        str(e), exc_info=False
                    )

        # Don't suppress exceptions
        return False

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on the retriever and backend.

        Verifies that:
        1. Backend connection is alive
        2. Collection/table/index is accessible
        3. Basic operations work

        Returns:
            Dictionary with health status information:
            {
                "healthy": bool,
                "backend": str,
                "collection": str,
                "details": dict,
                "error": str (if unhealthy)
            }

        Example:
            >>> health = retriever.health_check()
            >>> if health["healthy"]:
            ...     print("Retriever is healthy")
            ... else:
            ...     print(f"Health check failed: {health['error']}")
        """
        logger.debug(
            "Starting health check",
            extra={
                "extra_fields": {
                    "backend": self.backend_name,
                    "collection": self.collection
                }
            }
        )

        result = {
            "healthy": False,
            "backend": self.backend_name,
            "collection": self.collection,
            "details": {},
            "error": None
        }

        try:
            # Call backend-specific health check
            backend_details = self._check_backend_health()
            result["details"] = backend_details
            result["healthy"] = True

            logger.info(
                "Health check passed",
                extra={
                    "extra_fields": {
                        "backend": self.backend_name,
                        "collection": self.collection,
                        "details": backend_details
                    }
                }
            )
        except Exception as e:
            result["error"] = str(e)
            result["healthy"] = False

            logger.warning(
                "Health check failed",
                extra={
                    "extra_fields": {
                        "backend": self.backend_name,
                        "collection": self.collection,
                        "error": str(e)
                    }
                }
            )

        return result

    @abstractmethod
    def _check_backend_health(self) -> dict[str, Any]:
        """
        Perform backend-specific health checks.

        This must be implemented by subclasses to check:
        - Connection status
        - Collection/table/index existence
        - Any backend-specific diagnostics

        Returns:
            Dictionary with backend-specific health information

        Raises:
            Exception: If health check fails
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        Return the backend name ("qdrant", "pgvector", etc.).

        Used by policy engine to generate the correct filter format.
        """
        pass
