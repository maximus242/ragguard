"""
Async TigerGraph retriever with permission-aware GSQL queries.

Compatible with FastAPI and other async frameworks.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..filters.backends.tigergraph import to_tigergraph_filter
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .graph_base import AsyncGraphSecureRetrieverBase

logger = get_logger(__name__)


class AsyncTigerGraphSecureRetriever(AsyncGraphSecureRetrieverBase):
    """
    Async TigerGraph retriever with permission-aware GSQL queries.

    Compatible with FastAPI and other async frameworks. Uses thread pool
    execution since pyTigerGraph doesn't have native async support.

    Example:
        ```python
        import pyTigerGraph as tg
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncTigerGraphSecureRetriever

        # Create connection
        conn = tg.TigerGraphConnection(
            host="https://your-instance.i.tgcloud.io",
            graphname="MyGraph",
            username="user",
            password="pass"
        )

        # Create async retriever
        retriever = AsyncTigerGraphSecureRetriever(
            connection=conn,
            vertex_type="Document",
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
        connection: Any,
        vertex_type: str,
        policy: Policy,
        graph_name: Optional[str] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True,
        vertex_alias: str = "v"
    ):
        """
        Initialize async TigerGraph secure retriever.

        Args:
            connection: pyTigerGraph connection
            vertex_type: Vertex type for documents (e.g., "Document")
            policy: Access control policy
            graph_name: Optional graph name (uses connection default if not specified)
            embed_fn: Optional embedding function
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration
            enable_validation: Whether to enable input validation (default: True)
            vertex_alias: Alias for vertex in queries (default: "v")
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

        self.connection = connection
        self.vertex_type = vertex_type
        self.graph_name = graph_name
        self.vertex_alias = vertex_alias

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "tigergraph"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Async permission-aware search.

        Args:
            query: Query (dict for properties, string for GSQL)
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

        # For GSQL queries
        return await self.graph_query(query, user, limit=limit, **kwargs)

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute vector search (TigerGraph has limited native vector support)."""
        raise NotImplementedError(
            "TigerGraph vector search not implemented. "
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
        Execute a GSQL query with permission filtering.

        Args:
            query: GSQL query string
            user: User context for permission filtering
            params: Query parameters
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        permission_filter = to_tigergraph_filter(self.policy, user, self.vertex_alias)
        limit = kwargs.get("limit", 100)

        loop = asyncio.get_running_loop()

        def _sync_query():
            where_clause, filter_params = permission_filter

            # Check if this is an installed query call
            if query.strip().startswith("RUN"):
                result = self.connection.runInstalledQuery(query, params=params or {})
                if where_clause and where_clause != "TRUE":
                    return self._post_filter_results(result, permission_filter)
                return result[:limit]

            # For interpreted queries, inject WHERE clause
            modified_query = query
            if where_clause and where_clause != "TRUE":
                query_upper = query.upper()
                if "WHERE" in query_upper:
                    where_idx = query_upper.index("WHERE")
                    where_end = where_idx + 5
                    modified_query = (
                        query[:where_end] +
                        f" ({where_clause}) AND" +
                        query[where_end:]
                    )
                elif "SELECT" in query_upper:
                    for marker in ["ORDER", "LIMIT", "GROUP", "HAVING"]:
                        if marker in query_upper:
                            marker_idx = query_upper.index(marker)
                            modified_query = (
                                query[:marker_idx] +
                                f"WHERE {where_clause} " +
                                query[marker_idx:]
                            )
                            break
                    else:
                        modified_query = query + f" WHERE {where_clause}"

            if "LIMIT" not in query.upper():
                modified_query += f" LIMIT {limit}"

            return self.connection.gsql(modified_query)

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _query_with_retry():
                return await loop.run_in_executor(None, _sync_query)
            results = await _query_with_retry()
        else:
            results = await loop.run_in_executor(None, _sync_query)

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
            node_label: Optional vertex type (uses default if not specified)
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching vertices
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        vertex_type = node_label or self.vertex_type
        permission_filter = to_tigergraph_filter(self.policy, user, self.vertex_alias)

        loop = asyncio.get_running_loop()

        def _sync_search():
            where_clause, filter_params = permission_filter

            # Build property conditions
            prop_conditions = []
            for key, value in properties.items():
                if not key.isidentifier():
                    raise ValueError(f"Invalid property key: '{key}'")
                if isinstance(value, str):
                    escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
                    prop_conditions.append(f'{self.vertex_alias}.{key} == "{escaped_value}"')
                elif isinstance(value, bool):
                    prop_conditions.append(
                        f'{self.vertex_alias}.{key} == {"TRUE" if value else "FALSE"}'
                    )
                elif isinstance(value, (int, float)):
                    prop_conditions.append(f'{self.vertex_alias}.{key} == {value}')
                elif isinstance(value, list):
                    escaped_values = []
                    for v in value:
                        if isinstance(v, str):
                            escaped_v = v.replace('\\', '\\\\').replace('"', '\\"')
                            escaped_values.append(f'"{escaped_v}"')
                        else:
                            escaped_values.append(str(v))
                    values_str = ", ".join(escaped_values)
                    prop_conditions.append(f'{self.vertex_alias}.{key} IN ({values_str})')

            # Combine with permission filter
            all_conditions = prop_conditions.copy()
            if where_clause and where_clause != "TRUE":
                all_conditions.append(f"({where_clause})")

            where = " AND ".join(all_conditions) if all_conditions else "TRUE"

            # Try REST API first
            try:
                vertices = self.connection.getVertices(vertex_type, limit=limit * 2)

                filtered = []
                for v in vertices:
                    if self._matches_properties(v, properties):
                        if self._matches_permission_filter(v, permission_filter):
                            filtered.append(v)
                            if len(filtered) >= limit:
                                break
                return filtered

            except Exception:
                # Fall back to GSQL
                # vertex_type/alias are developer config; property values are escaped above
                query = (
                    f"SELECT {self.vertex_alias} FROM {vertex_type}:{self.vertex_alias} "  # nosec B608
                    f"WHERE {where} LIMIT {limit}"
                )
                return self.connection.gsql(query)

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

        permission_filter = to_tigergraph_filter(self.policy, user, self.vertex_alias)
        relationship_type = relationship_types[0] if relationship_types else "edge"

        loop = asyncio.get_running_loop()

        def _sync_traverse():
            where_clause, filter_params = permission_filter

            # Build direction pattern
            if direction == "outgoing":
                edge_pattern = f"-({relationship_type})->"
            elif direction == "incoming":
                edge_pattern = f"<-({relationship_type})-"
            else:
                edge_pattern = f"-({relationship_type})-"

            if max_depth == 1:
                depth_pattern = edge_pattern
            else:
                depth_pattern = f"({edge_pattern})*1..{max_depth}"

            where_part = f"WHERE {where_clause}" if where_clause and where_clause != "TRUE" else ""

            # vertex_type is developer config; start_node_id validated by caller
            query = (
                f"SELECT t FROM {self.vertex_type}:s{depth_pattern}{self.vertex_type}:t "  # nosec B608
                f'WHERE s.id == "{start_node_id}" '
                f"{where_part} "
                f"LIMIT {limit}"
            )

            return self.connection.gsql(query)

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
        node_alias: str = "v"
    ) -> str:
        """Build a GSQL permission clause."""
        where_clause, _ = to_tigergraph_filter(self.policy, user, node_alias)
        return where_clause or "TRUE"

    def _matches_properties(self, vertex: dict, properties: dict) -> bool:
        """Check if vertex matches property filters."""
        attrs = vertex.get("attributes", vertex)
        for key, value in properties.items():
            vertex_value = attrs.get(key)
            if isinstance(value, list):
                if vertex_value not in value:
                    return False
            elif vertex_value != value:
                return False
        return True

    def _matches_permission_filter(
        self,
        vertex: dict,
        permission_filter: tuple
    ) -> bool:
        """Check if vertex matches permission filter."""
        where_clause, _ = permission_filter

        if not where_clause or where_clause == "TRUE":
            return True
        if where_clause == "FALSE":
            return False

        # For complex filters, deny by default (fail-secure)
        logger.warning(
            "TigerGraph async post-filter cannot evaluate complex filter, denying access."
        )
        return False

    def _post_filter_results(self, results: list, permission_filter: tuple) -> list:
        """Post-filter results based on permissions."""
        return [r for r in results if self._matches_permission_filter(r, permission_filter)]

    async def close(self):
        """Close the connection (no-op for TigerGraph)."""
        pass
