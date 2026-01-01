"""
Async ChromaDB retriever with permission-aware search.

Note: ChromaDB doesn't have native async support, so this uses
run_in_executor to make it non-blocking for async contexts.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure, get_shared_executor
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase


class AsyncChromaDBSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async ChromaDB retriever with permission-aware search.

    Note: ChromaDB doesn't have native async support, so this uses
    run_in_executor to make it non-blocking for async contexts.

    Example:
        ```python
        import chromadb
        from ragguard.retrievers_async import AsyncChromaDBSecureRetriever

        # Create client (sync, but we'll use it async)
        client = chromadb.HttpClient(host="localhost", port=8000)
        collection = client.get_collection("documents")

        # Create async retriever
        retriever = AsyncChromaDBSecureRetriever(
            collection=collection,
            policy=policy
        )

        # Use in async context
        async def search():
            results = await retriever.search(
                query="search text",
                user={"role": "employee"},
                limit=10
            )
            return results
        ```
    """

    def __init__(
        self,
        collection: Any,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async ChromaDB secure retriever.

        Args:
            collection: ChromaDB collection
            policy: Access control policy
            embed_fn: Optional embedding function
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration (defaults to 3 retries with exponential backoff)
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

        self.collection = collection

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "chromadb"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Async permission-aware search.

        Uses run_in_executor to make ChromaDB non-blocking.

        Args:
            query: Query text or vector
            user: User context
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of result dictionaries
        """
        # Generate filter
        from ..filters.builder import to_chromadb_filter
        chromadb_filter = to_chromadb_filter(self.policy, user)

        # Execute search
        results = await self._execute_search(query, chromadb_filter, limit, **kwargs)

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query: Union[str, List[float]],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute the actual ChromaDB search in thread pool."""
        loop = asyncio.get_running_loop()

        def _sync_search():
            # Handle query type
            if isinstance(query, str):
                return self.collection.query(
                    query_texts=[query],
                    where=native_filter,
                    n_results=limit,
                    **kwargs
                )
            else:
                return self.collection.query(
                    query_embeddings=[query],
                    where=native_filter,
                    n_results=limit,
                    **kwargs
                )

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await loop.run_in_executor(get_shared_executor(), _sync_search)
            result = await _search_with_retry()
        else:
            result = await loop.run_in_executor(get_shared_executor(), _sync_search)

        # Convert to list of dicts with proper bounds checking
        results = []
        if result and 'ids' in result and result['ids']:
            ids_list = result['ids']
            ids = ids_list[0] if ids_list else []

            # Safely extract optional fields
            metadatas_list = result.get('metadatas') or [[]]
            metadatas = metadatas_list[0] if metadatas_list else []

            documents_list = result.get('documents') or [[]]
            documents = documents_list[0] if documents_list else []

            distances_list = result.get('distances') or [[]]
            distances = distances_list[0] if distances_list else []

            for i, doc_id in enumerate(ids):
                results.append({
                    'id': doc_id,
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'document': documents[i] if i < len(documents) else None,
                    'distance': distances[i] if i < len(distances) else None
                })

        return results
