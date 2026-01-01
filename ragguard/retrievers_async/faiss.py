"""
Async FAISS retriever with permission-aware search.

Uses run_in_executor to make FAISS non-blocking.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..policy.models import Policy
from ..retry import RetryConfig, get_shared_executor
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase


class AsyncFAISSSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async FAISS retriever with permission-aware search.

    Uses run_in_executor to make FAISS non-blocking.

    Example:
        ```python
        import faiss
        import numpy as np
        from ragguard.retrievers_async import AsyncFAISSSecureRetriever

        # Create FAISS index
        dimension = 768
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # Create async retriever
        retriever = AsyncFAISSSecureRetriever(
            index=index,
            metadata=metadata,
            policy=policy
        )

        # Use in async context
        async def search():
            results = await retriever.search(
                query=[0.1, 0.2, ...],
                user={"role": "user"},
                limit=10
            )
            return results
        ```
    """

    def __init__(
        self,
        index: Any,
        metadata: List[Dict[str, Any]],
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        over_fetch_factor: int = 3,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async FAISS secure retriever.

        Args:
            index: FAISS index
            metadata: List of metadata dicts
            policy: Access control policy
            embed_fn: Optional embedding function
            over_fetch_factor: Over-fetch multiplier for post-filtering
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration (uses defaults if not provided)
            enable_validation: Whether to enable input validation (default: True)
        """
        super().__init__(
            policy=policy,
            embed_fn=embed_fn,
            audit_logger=audit_logger,
            retry_config=retry_config,
            enable_retry=enable_retry,
            validation_config=validation_config,
            enable_validation=enable_validation
        )

        from ..retrievers.faiss import FAISSSecureRetriever

        # Create sync retriever
        self._sync_retriever = FAISSSecureRetriever(
            index=index,
            metadata=metadata,
            policy=policy,
            embed_fn=embed_fn,
            over_fetch_factor=over_fetch_factor,
            audit_logger=audit_logger,
            retry_config=retry_config,
            enable_retry=enable_retry,
            validation_config=validation_config,
            enable_validation=enable_validation
        )
        self.index = index
        self.metadata = metadata

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "faiss"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Async permission-aware search.

        Args:
            query: Query text or vector
            user: User context
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of result dictionaries
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            get_shared_executor(),
            lambda: self._sync_retriever.search(query=query, user=user, limit=limit, **kwargs)
        )

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute search - delegates to sync retriever."""
        # Not used directly since search() delegates to sync retriever
        raise NotImplementedError("Use search() instead")
