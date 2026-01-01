"""
Utility functions for async retrievers.

Provides helper functions for batch searching, multi-user search,
and running sync retrievers in async contexts.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class BatchSearchResult:
    """
    Result of a batch search operation with error handling.

    Attributes:
        results: List of successful results (None for failed queries)
        errors: List of exceptions (None for successful queries)
        success_count: Number of successful queries
        failure_count: Number of failed queries

    Example:
        ```python
        batch_result = await batch_search_async(retriever, queries, user)

        if batch_result.has_failures:
            print(f"Warning: {batch_result.failure_count} queries failed")
            for i, error in enumerate(batch_result.errors):
                if error:
                    print(f"Query {i} failed: {error}")

        # Process successful results
        for i, results in enumerate(batch_result.results):
            if results is not None:
                process_results(results)
        ```
    """
    results: List[Optional[List[Any]]] = field(default_factory=list)
    errors: List[Optional[Exception]] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Number of successful queries."""
        return sum(1 for r in self.results if r is not None)

    @property
    def failure_count(self) -> int:
        """Number of failed queries."""
        return sum(1 for e in self.errors if e is not None)

    @property
    def has_failures(self) -> bool:
        """True if any query failed."""
        return self.failure_count > 0

    @property
    def all_succeeded(self) -> bool:
        """True if all queries succeeded."""
        return self.failure_count == 0

    def successful_results(self) -> List[List[Any]]:
        """Return only successful results (filters out None values)."""
        return [r for r in self.results if r is not None]

    def raise_on_any_failure(self) -> None:
        """Raise the first error if any query failed."""
        for error in self.errors:
            if error is not None:
                raise error


@dataclass
class MultiUserSearchResult:
    """
    Result of a multi-user search operation with error handling.

    Attributes:
        results: Dict mapping user ID to results (None for failed searches)
        errors: Dict mapping user ID to exceptions (None for successful searches)
    """
    results: Dict[str, Optional[List[Any]]] = field(default_factory=dict)
    errors: Dict[str, Optional[Exception]] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        """Number of successful user searches."""
        return sum(1 for r in self.results.values() if r is not None)

    @property
    def failure_count(self) -> int:
        """Number of failed user searches."""
        return sum(1 for e in self.errors.values() if e is not None)

    @property
    def has_failures(self) -> bool:
        """True if any user search failed."""
        return self.failure_count > 0

    def successful_results(self) -> Dict[str, List[Any]]:
        """Return only successful results."""
        return {k: v for k, v in self.results.items() if v is not None}

    def failed_users(self) -> List[str]:
        """Return list of user IDs that had failures."""
        return [k for k, v in self.errors.items() if v is not None]


async def run_sync_retriever_async(
    retriever: Any,
    query: Union[str, List[float]],
    user: Dict[str, Any],
    limit: int = 10,
    **kwargs
) -> List[Any]:
    """
    Run a sync retriever in async context using thread pool.

    This is a utility for using sync retrievers (Pinecone, Weaviate, etc.)
    in async frameworks without blocking the event loop.

    Example:
        ```python
        from ragguard import QdrantSecureRetriever
        from ragguard.retrievers_async import run_sync_retriever_async

        # Sync retriever
        retriever = QdrantSecureRetriever(...)

        # Use in async context
        async def search():
            results = await run_sync_retriever_async(
                retriever=retriever,
                query="search text",
                user={"role": "employee"},
                limit=10
            )
            return results
        ```

    Args:
        retriever: Sync retriever instance
        query: Query text or vector
        user: User context
        limit: Maximum results
        **kwargs: Additional arguments

    Returns:
        Search results
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: retriever.search(query=query, user=user, limit=limit, **kwargs)
    )


async def batch_search_async(
    retriever: Any,
    queries: List[Union[str, List[float]]],
    user: Dict[str, Any],
    limit: int = 10,
    return_exceptions: bool = True,
    **kwargs
) -> BatchSearchResult:
    """
    Execute multiple searches concurrently with error handling.

    By default, captures exceptions per-query so partial results are preserved.
    Set return_exceptions=False to fail fast on first error (legacy behavior).

    Example:
        ```python
        queries = ["query 1", "query 2", "query 3"]
        batch_result = await batch_search_async(
            retriever=retriever,
            queries=queries,
            user=user,
            limit=10
        )

        # Check for failures
        if batch_result.has_failures:
            print(f"{batch_result.failure_count} queries failed")

        # Process successful results
        for results in batch_result.successful_results():
            process(results)

        # Or raise on any failure
        batch_result.raise_on_any_failure()
        ```

    Args:
        retriever: Async retriever
        queries: List of queries
        user: User context (same for all queries)
        limit: Maximum results per query
        return_exceptions: If True (default), capture exceptions per-query.
                          If False, raise immediately on first failure.
        **kwargs: Additional arguments

    Returns:
        BatchSearchResult with results and errors for each query
    """
    tasks = [
        retriever.search(query=q, user=user, limit=limit, **kwargs)
        for q in queries
    ]

    if return_exceptions:
        # Capture exceptions - allows partial results
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Optional[List[Any]]] = []
        errors: List[Optional[Exception]] = []

        for raw in raw_results:
            if isinstance(raw, Exception):
                results.append(None)
                errors.append(raw)
            else:
                results.append(raw)
                errors.append(None)

        return BatchSearchResult(results=results, errors=errors)
    else:
        # Fail fast - raise on first error (legacy behavior)
        raw_results = await asyncio.gather(*tasks)
        return BatchSearchResult(
            results=raw_results,
            errors=[None] * len(raw_results)
        )


async def multi_user_search_async(
    retriever: Any,
    query: Union[str, List[float]],
    users: List[Dict[str, Any]],
    limit: int = 10,
    return_exceptions: bool = True,
    **kwargs
) -> MultiUserSearchResult:
    """
    Execute same search for multiple users concurrently with error handling.

    Useful for cache warming or batch processing. By default, captures
    exceptions per-user so partial results are preserved.

    Example:
        ```python
        users = [
            {"id": "alice", "role": "admin"},
            {"id": "bob", "role": "user"},
            {"id": "charlie", "role": "guest"}
        ]

        result = await multi_user_search_async(
            retriever=retriever,
            query="company policies",
            users=users,
            limit=10
        )

        # Check for failures
        if result.has_failures:
            print(f"Failed users: {result.failed_users()}")

        # Get only successful results
        for user_id, docs in result.successful_results().items():
            print(f"{user_id}: {len(docs)} documents")
        ```

    Args:
        retriever: Async retriever
        query: Single query
        users: List of user contexts
        limit: Maximum results per user
        return_exceptions: If True (default), capture exceptions per-user.
                          If False, raise immediately on first failure.
        **kwargs: Additional arguments

    Returns:
        MultiUserSearchResult with results and errors for each user
    """
    user_ids = [user.get('id', f'user_{i}') for i, user in enumerate(users)]
    tasks = [
        retriever.search(query=query, user=user, limit=limit, **kwargs)
        for user in users
    ]

    if return_exceptions:
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[str, Optional[List[Any]]] = {}
        errors: Dict[str, Optional[Exception]] = {}

        for user_id, raw in zip(user_ids, raw_results):
            if isinstance(raw, Exception):
                results[user_id] = None
                errors[user_id] = raw
            else:
                results[user_id] = raw
                errors[user_id] = None

        return MultiUserSearchResult(results=results, errors=errors)
    else:
        raw_results = await asyncio.gather(*tasks)
        return MultiUserSearchResult(
            results=dict(zip(user_ids, raw_results)),
            errors=dict.fromkeys(user_ids)
        )
