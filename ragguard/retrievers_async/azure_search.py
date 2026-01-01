"""
Async Azure AI Search retriever with permission-aware search.

Compatible with FastAPI and other async frameworks.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase

logger = get_logger(__name__)


class AsyncAzureSearchSecureRetriever(AsyncSecureRetrieverBase):
    """
    Async Azure AI Search retriever with permission-aware search.

    Compatible with FastAPI and other async frameworks.

    Note: Azure Search SDK doesn't have native async support, so this uses
    run_in_executor to run synchronous operations in a thread pool.

    Example:
        ```python
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncAzureSearchSecureRetriever

        # Create client (sync - will be wrapped)
        client = SearchClient(
            endpoint="https://your-search.search.windows.net",
            index_name="documents",
            credential=AzureKeyCredential("your-api-key")
        )

        # Create async retriever
        retriever = AsyncAzureSearchSecureRetriever(
            client=client,
            index="documents",
            policy=load_policy("policy.yaml"),
            vector_field="embedding"
        )

        # Use in async context
        async def search_docs(query_vector, user_context):
            results = await retriever.search(
                query=query_vector,
                user=user_context,
                limit=10
            )
            return results
        ```
    """

    def __init__(
        self,
        client: Any,
        index: str,
        policy: Policy,
        vector_field: str = "embedding",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async Azure AI Search secure retriever.

        Args:
            client: SearchClient instance from azure-search-documents (sync)
            index: Index name
            policy: Access control policy
            vector_field: Name of the vector field (default: "embedding")
            embed_fn: Optional embedding function (if not provided, query must be a vector)
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration
            enable_validation: Whether to enable input validation (default: True)
        """
        # Validate client type
        client_type = type(client).__name__
        if "SearchClient" not in client_type:
            raise RetrieverError(
                f"Expected SearchClient from azure-search-documents, got {client_type}. "
                f"Install with: pip install ragguard[azure]"
            )

        super().__init__(
            policy=policy,
            embed_fn=embed_fn,
            audit_logger=audit_logger,
            retry_config=retry_config,
            enable_retry=enable_retry,
            validation_config=validation_config,
            enable_validation=enable_validation
        )

        self.client = client
        self.index = index
        self.vector_field = vector_field

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "azure_search"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Async permission-aware vector search.

        Args:
            query: Query text (requires embed_fn) or query vector
            user: User context for permission filtering
            limit: Maximum number of results
            **kwargs: Additional arguments passed to Azure Search

        Returns:
            List of results (Azure Search documents)
        """
        # Validate user context
        if self._enable_validation:
            self._validator.validate_user_context(user)

        # Convert query to vector
        query_vector = await self._get_query_vector(query)

        # Generate filter from policy
        from ..filters.builder import to_azure_search_filter
        azure_filter = to_azure_search_filter(self.policy, user)

        # Execute search
        results = await self._execute_search(query_vector, azure_filter, limit, **kwargs)

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Optional[str],
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute the actual Azure Search in thread pool."""
        loop = asyncio.get_running_loop()

        def _sync_search():
            try:
                from azure.search.documents.models import VectorizedQuery
            except ImportError:
                raise RetrieverError(
                    "azure-search-documents not installed. "
                    "Install with: pip install ragguard[azure]"
                )

            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=limit,
                fields=self.vector_field
            )

            search_args = {
                "vector_queries": [vector_query],
                "top": limit,
            }

            # Add filter if present
            if native_filter:
                search_args["filter"] = native_filter

            # Add any additional kwargs
            search_args.update(kwargs)

            # Execute search
            results = list(self.client.search(**search_args))
            return results

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await loop.run_in_executor(None, _sync_search)
            return await _search_with_retry()
        else:
            return await loop.run_in_executor(None, _sync_search)
