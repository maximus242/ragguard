"""
Abstract base class for async secure retrievers.

Defines the interface for permission-aware vector search across different
database backends in async contexts.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..audit.logger import AuditLogger, NullAuditLogger
from ..exceptions import RetrieverTimeoutError
from ..logging import get_logger
from ..policy.engine import PolicyEngine
from ..policy.models import Policy
from ..retry import RetryConfig, run_in_executor_with_backpressure
from ..validation import InputValidator, ValidationConfig

logger = get_logger(__name__)


class AsyncSecureRetrieverBase(ABC):
    """
    Abstract base class for async permission-aware retrievers.

    Provides common functionality for async retrievers including:
    - Query vector conversion (text -> embeddings)
    - Audit logging in async contexts
    - Retry configuration
    - Policy engine integration

    Subclasses must implement:
    - _execute_search(): Backend-specific search logic
    - backend_name: Property identifying the backend type
    """

    def __init__(
        self,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async secure retriever base.

        Args:
            policy: Access control policy
            embed_fn: Optional embedding function (if not provided, query must be a vector)
            audit_logger: Optional audit logger (defaults to NullAuditLogger)
            retry_config: Optional retry configuration (defaults to 3 retries with exponential backoff)
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration (uses defaults if not provided)
            enable_validation: Whether to enable input validation (default: True)
        """
        self.policy = policy
        self.embed_fn = embed_fn
        self.audit_logger = audit_logger or NullAuditLogger()
        self.engine = PolicyEngine(policy)

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

    @abstractmethod
    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Execute async permission-aware vector search.

        This is the main entry point for async search. Subclasses must implement this
        method to:
        1. Convert query to vector if needed (using _get_query_vector)
        2. Generate permission filter from policy
        3. Execute search with retry logic if enabled
        4. Log audit event (using _log_audit)
        5. Return results

        Args:
            query: Query text (requires embed_fn) or query vector
            user: User context for permission filtering
            limit: Maximum number of results
            **kwargs: Additional backend-specific arguments

        Returns:
            List of results (format depends on backend)

        Raises:
            RetrieverError: If search fails
            ValueError: If query is string but embed_fn not provided
        """
        pass

    async def _get_query_vector(self, query: Union[str, List[float]]) -> List[float]:
        """
        Convert query to vector if needed.

        Handles both string queries (converts using embed_fn) and vector queries
        (passes through). Runs embedding function in thread pool to avoid blocking
        the event loop.

        Args:
            query: Query text or vector

        Returns:
            Query vector (embedding)

        Raises:
            ValueError: If query is string but embed_fn not provided
        """
        if isinstance(query, str):
            if self.embed_fn is None:
                raise ValueError(
                    "embed_fn required when query is string. "
                    "Provide embed_fn or pass query as vector."
                )
            query_vector = await run_in_executor_with_backpressure(
                self.embed_fn, query
            )
            return query_vector
        else:
            return query

    async def _log_audit(
        self,
        user: Dict[str, Any],
        query: Any,
        results: List[Any],
        backend: Optional[str] = None
    ) -> None:
        """
        Log audit event in async context.

        Runs audit logging in thread pool to avoid blocking the event loop.
        This is a shared implementation that can be used by all async retrievers.

        Args:
            user: User context
            query: Original query (string or vector)
            results: Search results
            backend: Backend name (defaults to self.backend_name)
        """
        if not self.audit_logger:
            return

        try:
            await run_in_executor_with_backpressure(
                self.audit_logger.log,
                {
                    "user": user,
                    "query": str(query)[:100],
                    "results_count": len(results),
                    "backend": backend or self.backend_name
                }
            )
        except Exception as e:
            # Log warning but don't fail the search - audit logging is best-effort
            # in async context to avoid blocking the event loop
            logger.warning(
                "Async audit logging failed",
                extra={
                    "extra_fields": {
                        "backend": backend or self.backend_name,
                        "user_id": user.get("id") if isinstance(user, dict) else None,
                        "error": str(e)
                    }
                }
            )

    @abstractmethod
    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """
        Execute the actual search with the native filter.

        This must be implemented by subclasses for each database backend.
        This method should contain the backend-specific search logic.

        Args:
            query_vector: Embedding vector
            native_filter: Native filter for this backend
            limit: Maximum results
            **kwargs: Additional backend-specific arguments

        Returns:
            List of results in backend-specific format

        Raises:
            Exception: If search fails
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        Return the backend name ("qdrant", "chromadb", "pinecone", etc.).

        Used by policy engine to generate the correct filter format.
        """
        pass

    async def batch_search(
        self,
        queries: List[Union[str, List[float]]],
        user: Dict[str, Any],
        limit: int = 10,
        max_concurrent: int = 10,
        timeout: Optional[float] = None,
        **kwargs
    ) -> List[List[Any]]:
        """
        Execute multiple searches concurrently for the same user.

        Args:
            queries: List of query texts or vectors
            user: User context for permission filtering
            limit: Maximum number of results per query
            max_concurrent: Maximum concurrent searches (default: 10)
            timeout: Optional timeout in seconds for entire batch (default: uses retry_config.total_timeout)
            **kwargs: Additional backend-specific arguments

        Returns:
            List of result lists, one per query

        Raises:
            RetrieverTimeoutError: If batch doesn't complete within timeout
        """
        if not queries:
            return []

        # Validate user context ONCE upfront for all queries
        if self._enable_validation:
            try:
                self._validator.validate_user_context(user)
            except Exception as e:
                logger.warning(
                    "User context validation failed in async batch_search",
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

        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_search(q):
            async with semaphore:
                return await self.search(query=q, user=user, limit=limit, **kwargs)

        tasks = [bounded_search(q) for q in queries]

        # Use timeout to prevent indefinite hangs
        batch_timeout = timeout if timeout is not None else self._retry_config.total_timeout
        try:
            return await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=batch_timeout
            )
        except asyncio.TimeoutError:
            raise RetrieverTimeoutError(
                backend=self.backend_name,
                operation="batch_search",
                timeout=batch_timeout
            )

    async def multi_user_search(
        self,
        query: Union[str, List[float]],
        users: List[Dict[str, Any]],
        limit: int = 10,
        max_concurrent: int = 10,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, List[Any]]:
        """
        Execute the same query for multiple users concurrently.

        Args:
            query: Query text or vector
            users: List of user contexts (each must have an 'id' field)
            limit: Maximum number of results per user
            max_concurrent: Maximum concurrent searches (default: 10)
            timeout: Optional timeout in seconds for entire batch (default: uses retry_config.total_timeout)
            **kwargs: Additional backend-specific arguments

        Returns:
            Dictionary mapping user_id -> results

        Raises:
            RetrieverTimeoutError: If batch doesn't complete within timeout
        """
        if not users:
            return {}

        # Validate all user contexts upfront for fail-fast behavior
        if self._enable_validation:
            for i, user in enumerate(users):
                try:
                    self._validator.validate_user_context(user)
                except Exception as e:
                    logger.warning(
                        "User context validation failed in async multi_user_search",
                        extra={
                            "extra_fields": {
                                "backend": self.backend_name,
                                "user_index": i,
                                "user_id": user.get("id") if isinstance(user, dict) else None,
                                "error": str(e),
                                "total_users": len(users)
                            }
                        }
                    )
                    raise

        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_for_user(user: Dict[str, Any]) -> Tuple[str, List[Any]]:
            async with semaphore:
                user_id = user.get("id", str(id(user)))
                results = await self.search(query=query, user=user, limit=limit, **kwargs)
                return (user_id, results)

        tasks = [search_for_user(u) for u in users]

        # Use timeout to prevent indefinite hangs
        batch_timeout = timeout if timeout is not None else self._retry_config.total_timeout
        try:
            results_list = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=batch_timeout
            )
            return dict(results_list)
        except asyncio.TimeoutError:
            raise RetrieverTimeoutError(
                backend=self.backend_name,
                operation="multi_user_search",
                timeout=batch_timeout
            )

    async def _run_with_timeout(
        self,
        coro,
        timeout: Optional[float] = None,
        operation: str = "operation"
    ) -> Any:
        """
        Run a coroutine with a timeout.

        Uses the request_timeout from retry_config if no timeout specified.

        Args:
            coro: Coroutine to run
            timeout: Timeout in seconds (defaults to retry_config.request_timeout)
            operation: Operation name for error messages

        Returns:
            Result of the coroutine

        Raises:
            RetrieverTimeoutError: If operation times out
        """
        if timeout is None:
            timeout = self._retry_config.request_timeout

        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise RetrieverTimeoutError(
                backend=self.backend_name,
                operation=operation,
                timeout=timeout
            )
