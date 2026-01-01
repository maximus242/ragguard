"""
Async Amazon Neptune retriever with permission-aware graph queries.

Compatible with FastAPI and other async frameworks.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..filters.backends.neptune import to_neptune_filter
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .graph_base import AsyncGraphSecureRetrieverBase

logger = get_logger(__name__)


class AsyncNeptuneSecureRetriever(AsyncGraphSecureRetrieverBase):
    """
    Async Neptune retriever with permission-aware graph queries.

    Compatible with FastAPI and other async frameworks. Uses thread pool
    execution since Gremlin Python client doesn't have native async support.

    Example:
        ```python
        from gremlin_python.driver import client
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncNeptuneSecureRetriever

        # Create Gremlin client
        gremlin_client = client.Client(
            'wss://your-neptune-endpoint:8182/gremlin',
            'g'
        )

        # Create async retriever
        retriever = AsyncNeptuneSecureRetriever(
            client=gremlin_client,
            node_label="Document",
            policy=load_policy("policy.yaml")
        )

        # Use in async context
        async def search_graph(user_context):
            results = await retriever.property_search(
                properties={"category": "engineering"},
                user=user_context,
                limit=10
            )
            return results
        ```
    """

    def __init__(
        self,
        client: Any,
        node_label: str,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async Neptune secure retriever.

        Args:
            client: Gremlin client connection
            node_label: Vertex label for documents (e.g., "Document")
            policy: Access control policy
            embed_fn: Optional embedding function
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration
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

        self.client = client
        self.node_label = node_label

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "neptune"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Async permission-aware search.

        For Neptune, this performs a property-based search if query is a dict,
        or a vector search if embed_fn is available.

        Args:
            query: Query (dict for properties, string for text, list for vector)
            user: User context for permission filtering
            limit: Maximum number of results
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        if isinstance(query, dict):
            return await self.property_search(query, user, limit=limit, **kwargs)

        # For text/vector queries, use the base implementation
        query_vector = await self._get_query_vector(query)
        permission_filter = await self._build_permission_clause(user)

        # Neptune doesn't have native vector search, so we use post-filtering
        results = await self._execute_search(query_vector, permission_filter, limit, **kwargs)

        await self._log_audit(user, query, results)
        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute vector search (Neptune doesn't have native vector search)."""
        raise NotImplementedError(
            "Neptune doesn't support native vector search. "
            "Use property_search() or graph_query() instead."
        )

    async def graph_query(
        self,
        query: str,
        user: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Execute a Gremlin query with permission filtering.

        Args:
            query: Gremlin query string
            user: User context for permission filtering
            params: Query parameters (not used for Gremlin strings)
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        permission_filter = to_neptune_filter(self.policy, user)
        limit = kwargs.get("limit", 100)

        results = await self._execute_gremlin(query, permission_filter, limit)

        await self._log_audit(user, query, results)
        return results

    async def property_search(
        self,
        properties: Dict[str, Any],
        user: Dict[str, Any],
        node_label: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Search for vertices by property values.

        Args:
            properties: Property name -> value to match
            user: User context for permission filtering
            node_label: Optional node label (uses default if not specified)
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching vertices
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        label = node_label or self.node_label
        permission_filter = to_neptune_filter(self.policy, user)

        loop = asyncio.get_running_loop()

        def _sync_search():
            # Build Gremlin query
            query_parts = [f"g.V().hasLabel('{label}')"]

            # Add property filters
            for key, value in properties.items():
                if isinstance(value, str):
                    query_parts.append(f".has('{key}', '{value}')")
                elif isinstance(value, (int, float)):
                    query_parts.append(f".has('{key}', {value})")
                elif isinstance(value, bool):
                    query_parts.append(f".has('{key}', {str(value).lower()})")
                elif isinstance(value, list):
                    values_str = ", ".join(
                        f"'{v}'" if isinstance(v, str) else str(v)
                        for v in value
                    )
                    query_parts.append(f".has('{key}', within({values_str}))")

            # Add permission filters
            for f in permission_filter:
                filter_str = self._filter_to_gremlin_string(f)
                if filter_str:
                    query_parts.append(filter_str)

            query_parts.append(f".limit({limit})")
            query_parts.append(".valueMap(true)")

            query = "".join(query_parts)

            result_set = self.client.submit(query)
            return result_set.all().result()

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await loop.run_in_executor(None, _sync_search)
            results = await _search_with_retry()
        else:
            results = await loop.run_in_executor(None, _sync_search)

        await self._log_audit(user, str(properties), results)
        return results

    async def traverse(
        self,
        start_node_id: str,
        user: Dict[str, Any],
        relationship_types: Optional[List[str]] = None,
        direction: str = "outgoing",
        max_depth: int = 1,
        limit: int = 100,
        **kwargs
    ) -> List[Any]:
        """
        Traverse the graph from a starting node.

        Args:
            start_node_id: ID of the starting node
            user: User context for permission filtering
            relationship_types: Types of relationships to follow
            direction: "outgoing", "incoming", or "both"
            max_depth: Maximum traversal depth
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of nodes reached
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        permission_filter = to_neptune_filter(self.policy, user)

        loop = asyncio.get_running_loop()

        def _sync_traverse():
            # Build relationship pattern
            rel_pattern = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_pattern = f":{rel_types}"

            # Build direction pattern
            if direction == "outgoing":
                edge_step = f".out('{relationship_types[0]}')" if relationship_types else ".out()"
            elif direction == "incoming":
                edge_step = f".in('{relationship_types[0]}')" if relationship_types else ".in()"
            else:
                edge_step = f".both('{relationship_types[0]}')" if relationship_types else ".both()"

            # Build repeat for depth
            if max_depth > 1:
                edge_step = f".repeat({edge_step.lstrip('.')}).times({max_depth})"

            # Build query
            query_parts = [
                f"g.V('{start_node_id}')",
                edge_step,
                f".hasLabel('{self.node_label}')",
            ]

            # Add permission filters
            for f in permission_filter:
                filter_str = self._filter_to_gremlin_string(f)
                if filter_str:
                    query_parts.append(filter_str)

            query_parts.extend([
                ".dedup()",
                f".limit({limit})",
                ".valueMap(true)"
            ])

            query = "".join(query_parts)
            result_set = self.client.submit(query)
            return result_set.all().result()

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _traverse_with_retry():
                return await loop.run_in_executor(None, _sync_traverse)
            results = await _traverse_with_retry()
        else:
            results = await loop.run_in_executor(None, _sync_traverse)

        await self._log_audit(user, f"traverse:{start_node_id}", results)
        return results

    async def _build_permission_clause(
        self,
        user: Dict[str, Any],
        node_alias: str = "n"
    ) -> str:
        """Build a Gremlin permission clause."""
        filters = to_neptune_filter(self.policy, user)
        if not filters:
            return "true"
        # Return the filter list for use in post-filtering
        return filters

    def _filter_to_gremlin_string(self, f: dict) -> str:
        """Convert a filter spec to Gremlin query string fragment."""
        if f["type"] == "hasNot":
            return f".hasNot('{f['property']}')"

        elif f["type"] == "has":
            prop = f["property"]
            pred = f["predicate"]
            value = f["value"]

            if pred == "eq":
                if isinstance(value, str):
                    return f".has('{prop}', '{value}')"
                return f".has('{prop}', {value})"

            elif pred == "neq":
                if isinstance(value, str):
                    return f".has('{prop}', neq('{value}'))"
                return f".has('{prop}', neq({value}))"

            elif pred == "within":
                values_str = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v)
                    for v in value
                )
                return f".has('{prop}', within({values_str}))"

            elif pred == "without":
                values_str = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v)
                    for v in value
                )
                return f".has('{prop}', without({values_str}))"

            elif pred == "gt":
                return f".has('{prop}', gt({value}))"
            elif pred == "gte":
                return f".has('{prop}', gte({value}))"
            elif pred == "lt":
                return f".has('{prop}', lt({value}))"
            elif pred == "lte":
                return f".has('{prop}', lte({value}))"
            elif pred == "exists":
                return f".has('{prop}')"

        elif f["type"] == "or":
            or_parts = []
            for child_filters in f["children"]:
                child_parts = []
                for cf in child_filters:
                    child_str = self._filter_to_gremlin_string(cf)
                    if child_str:
                        child_parts.append(child_str.lstrip("."))
                if child_parts:
                    or_parts.append("__." + ".".join(child_parts))

            if or_parts:
                return ".or(" + ", ".join(or_parts) + ")"

        return ""

    async def _execute_gremlin(
        self,
        query: str,
        permission_filter: List[dict],
        limit: int
    ) -> List[Any]:
        """Execute a Gremlin query with post-filtering."""
        loop = asyncio.get_running_loop()

        def _sync_execute():
            result_set = self.client.submit(query)
            results = result_set.all().result()

            # Post-filter if needed
            if permission_filter:
                filtered = []
                for result in results:
                    if self._matches_filters(result, permission_filter):
                        filtered.append(result)
                        if len(filtered) >= limit:
                            break
                return filtered
            return results[:limit]

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _query_with_retry():
                return await loop.run_in_executor(None, _sync_execute)
            return await _query_with_retry()
        else:
            return await loop.run_in_executor(None, _sync_execute)

    def _matches_filters(self, vertex: dict, filters: List[dict]) -> bool:
        """Check if vertex matches permission filters."""
        for f in filters:
            if f["type"] == "or":
                if not any(
                    all(self._matches_single_filter(vertex, cf) for cf in child)
                    for child in f["children"]
                ):
                    return False
            elif not self._matches_single_filter(vertex, f):
                return False
        return True

    def _matches_single_filter(self, vertex: dict, f: dict) -> bool:
        """Check if vertex matches a single filter."""
        if f["type"] == "hasNot":
            return f["property"] not in vertex

        elif f["type"] == "has":
            prop = f["property"]
            pred = f["predicate"]
            value = f["value"]

            if prop not in vertex:
                return pred == "exists" and value is False

            vertex_value = vertex[prop]

            if pred == "eq":
                return vertex_value == value
            elif pred == "neq":
                return vertex_value != value
            elif pred == "within":
                return vertex_value in value
            elif pred == "without":
                return vertex_value not in value
            elif pred == "gt":
                return vertex_value > value
            elif pred == "gte":
                return vertex_value >= value
            elif pred == "lt":
                return vertex_value < value
            elif pred == "lte":
                return vertex_value <= value
            elif pred == "containing":
                if isinstance(vertex_value, list):
                    return value in vertex_value
                return False
            elif pred == "notContaining":
                if isinstance(vertex_value, list):
                    return value not in vertex_value
                return True
            elif pred == "exists":
                return True

        return True

    async def close(self):
        """Close the client connection."""
        if hasattr(self.client, 'close'):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.client.close)
