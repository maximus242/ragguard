"""
TigerGraph secure retriever.

Provides permission-aware querying for TigerGraph using GSQL.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..filters.backends.tigergraph import to_tigergraph_filter
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .graph_base import BaseGraphRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig

logger = get_logger(__name__)


class TigerGraphSecureRetriever(BaseGraphRetriever):
    """
    Permission-aware retriever for TigerGraph.

    Wraps a pyTigerGraph connection and enforces document-level
    permissions by injecting WHERE conditions into GSQL queries.

    Example:
        >>> import pyTigerGraph as tg
        >>> from ragguard import TigerGraphSecureRetriever, load_policy
        >>>
        >>> conn = tg.TigerGraphConnection(
        ...     host="https://your-instance.i.tgcloud.io",
        ...     graphname="MyGraph",
        ...     username="user",
        ...     password="pass"
        ... )
        >>> policy = load_policy("policy.yaml")
        >>> retriever = TigerGraphSecureRetriever(
        ...     connection=conn,
        ...     vertex_type="Document",
        ...     policy=policy
        ... )
        >>>
        >>> # Property-based search
        >>> results = retriever.search(
        ...     query={"category": "engineering"},
        ...     user={"id": "alice", "department": "engineering"}
        ... )
    """

    def __init__(
        self,
        connection: Any,
        vertex_type: str,
        policy: Policy,
        graph_name: Optional[str] = None,
        audit_logger: Optional[AuditLogger] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        enable_filter_cache: bool = True,
        filter_cache_size: int = 1000,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True,
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        vertex_alias: str = "v",
        *,
        config: Optional["SecureRetrieverConfig"] = None
    ):
        """
        Initialize the TigerGraph secure retriever.

        Args:
            connection: pyTigerGraph connection
            vertex_type: Vertex type for documents (e.g., "Document")
            policy: Access control policy
            graph_name: Optional graph name (uses connection default if not specified)
            audit_logger: Optional audit logger
            embed_fn: Optional embedding function
            enable_filter_cache: Whether to enable filter caching
            filter_cache_size: Maximum cached filters
            retry_config: Retry configuration
            enable_retry: Whether to enable retries
            validation_config: Validation configuration
            enable_validation: Whether to enable validation
            enable_circuit_breaker: Whether to enable circuit breaker
            circuit_breaker_config: Circuit breaker configuration
            vertex_alias: Alias for vertex in queries (default: "v")
        """
        self.connection = connection
        self.vertex_type = vertex_type
        self.graph_name = graph_name
        self.vertex_alias = vertex_alias

        super().__init__(
            client=connection,
            node_label=vertex_type,
            policy=policy,
            database=graph_name,
            audit_logger=audit_logger,
            embed_fn=embed_fn,
            enable_filter_cache=enable_filter_cache,
            filter_cache_size=filter_cache_size,
            retry_config=retry_config,
            enable_retry=enable_retry,
            validation_config=validation_config,
            enable_validation=enable_validation,
            enable_circuit_breaker=enable_circuit_breaker,
            circuit_breaker_config=circuit_breaker_config,
            config=config
        )

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "tigergraph"

    def _execute_graph_query(
        self,
        query: str,
        permission_filter: tuple[str, dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute a GSQL query with permission constraints.

        For interpreted queries, permissions are injected into the WHERE clause.
        For installed queries, post-filtering is applied.
        """
        where_clause, filter_params = permission_filter

        # Check if this is an installed query call
        if query.strip().startswith("RUN"):
            # Can't inject into installed queries, run and post-filter
            result = self.connection.runInstalledQuery(query, params=kwargs)
            if where_clause and where_clause != "TRUE":
                return self._post_filter_results(result, permission_filter)
            return result[:limit]

        # For interpreted queries, inject WHERE clause
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
                # Find place to insert WHERE (after FROM clause)
                select_idx = query_upper.index("SELECT")
                # Look for common clause markers
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
            else:
                modified_query = query

            if "LIMIT" not in query.upper():
                modified_query += f" LIMIT {limit}"
        else:
            modified_query = query
            if "LIMIT" not in query.upper():
                modified_query += f" LIMIT {limit}"

        logger.debug(
            "Executing TigerGraph query",
            extra={
                "extra_fields": {
                    "query_length": len(modified_query),
                    "vertex_type": self.vertex_type
                }
            }
        )

        # Run as interpreted query
        result = self.connection.gsql(modified_query)
        return result

    def _execute_property_search(
        self,
        properties: dict[str, Any],
        permission_filter: tuple[str, dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute property-based vertex search.
        """
        where_clause, filter_params = permission_filter

        # Build property conditions
        prop_conditions = []
        for key, value in properties.items():
            # Validate property key to prevent injection
            if not key.isidentifier():
                raise ValueError(f"Invalid property key: '{key}'. Must be a valid identifier.")
            if isinstance(value, str):
                # Escape embedded quotes to prevent GSQL injection
                escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
                prop_conditions.append(f'{self.vertex_alias}.{key} == "{escaped_value}"')
            elif isinstance(value, bool):
                prop_conditions.append(f'{self.vertex_alias}.{key} == {"TRUE" if value else "FALSE"}')
            elif isinstance(value, (int, float)):
                prop_conditions.append(f'{self.vertex_alias}.{key} == {value}')
            elif isinstance(value, list):
                escaped_values = []
                for v in value:
                    if isinstance(v, str):
                        # Escape embedded quotes in list values
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

        # Build GSQL query using getVertices REST endpoint is more efficient
        # but for complex filters we use interpreted GSQL
        # vertex_type/alias are developer config; property values are escaped above
        query = (
            f"SELECT {self.vertex_alias} FROM {self.vertex_type}:{self.vertex_alias} "  # nosec B608
            f"WHERE {where} "
            f"LIMIT {limit}"
        )

        logger.debug(
            "Executing TigerGraph property search",
            extra={
                "extra_fields": {
                    "vertex_type": self.vertex_type,
                    "properties": list(properties.keys()),
                    "limit": limit
                }
            }
        )

        # Use REST API for better performance
        try:
            # Try using getVertices API with filters
            vertices = self.connection.getVertices(
                self.vertex_type,
                limit=limit * 2  # Get more to account for permission filtering
            )

            # Apply property and permission filters
            filtered = []
            for v in vertices:
                if self._matches_properties(v, properties):
                    if self._matches_permission_filter(v, permission_filter):
                        filtered.append(v)
                        if len(filtered) >= limit:
                            break

            return filtered

        except Exception as e:
            logger.debug(f"REST API failed, falling back to GSQL: {e}")
            result = self.connection.gsql(query)
            return result

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
        permission_filter: tuple[str, dict]
    ) -> bool:
        """
        Check if vertex matches permission filter.

        This is a simplified evaluator for post-filtering.
        """
        where_clause, _ = permission_filter

        if not where_clause or where_clause == "TRUE":
            return True
        if where_clause == "FALSE":
            return False

        # For complex filters, we'd need a full GSQL parser
        # This is a basic implementation that handles simple cases
        attrs = vertex.get("attributes", vertex)

        # Try to evaluate simple conditions
        # This is limited - for production, use the native query approach with
        # INSTALLED queries that include permission filters in the query itself
        #
        # SECURITY: Default to DENY if we can't parse the filter
        # This follows the principle of fail-secure: if we can't determine
        # permissions, we deny access rather than granting it
        logger.warning(
            "TigerGraph post-filter cannot evaluate complex filter, denying access. "
            "For better performance and security, use INSTALLED queries with embedded permission filters.",
            extra={"extra_fields": {"where_clause": where_clause[:100] if where_clause else None}}
        )
        return False  # SECURITY: Deny by default if we can't parse

    def _post_filter_results(
        self,
        results: list,
        permission_filter: tuple[str, dict]
    ) -> list:
        """Post-filter results based on permissions."""
        filtered = []
        for result in results:
            if self._matches_permission_filter(result, permission_filter):
                filtered.append(result)
        return filtered

    def _build_permission_clause(
        self,
        user: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Build permission WHERE clause for the user.
        """
        return to_tigergraph_filter(self.policy, user, self.vertex_alias)

    def _execute_traversal(
        self,
        start_node_id: str,
        relationship_type: str,
        user: dict[str, Any],
        direction: str,
        depth: int,
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute relationship traversal with permissions.
        """
        where_clause, filter_params = self._build_permission_clause(user)

        # Build direction pattern
        if direction == "outgoing":
            edge_pattern = f"-({relationship_type})->"
        elif direction == "incoming":
            edge_pattern = f"<-({relationship_type})-"
        else:  # both
            edge_pattern = f"-({relationship_type})-"

        # Build traversal query
        if depth == 1:
            depth_pattern = edge_pattern
        else:
            depth_pattern = f"({edge_pattern})*1..{depth}"

        # Build GSQL traversal query
        where_part = f"WHERE {where_clause}" if where_clause and where_clause != "TRUE" else ""

        # vertex_type is developer config; start_node_id validated by BaseGraphRetriever.traverse()
        query = (
            f"SELECT t FROM {self.vertex_type}:s{depth_pattern}{self.vertex_type}:t "  # nosec B608
            f'WHERE s.id == "{start_node_id}" '
            f"{where_part} "
            f"LIMIT {limit}"
        )

        logger.debug(
            "Executing TigerGraph traversal",
            extra={
                "extra_fields": {
                    "start_node": start_node_id,
                    "relationship": relationship_type,
                    "direction": direction,
                    "depth": depth
                }
            }
        )

        result = self.connection.gsql(query)
        return result

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check TigerGraph connection health.
        """
        # Check graph metadata
        try:
            # Get graph statistics
            stats = self.connection.getStatistics()

            return {
                "connection_alive": True,
                "graph_name": self.graph_name or self.connection.graphname,
                "vertex_type": self.vertex_type,
                "statistics": stats
            }
        except Exception:
            # Try simpler health check
            echo_result = self.connection.echo()
            return {
                "connection_alive": echo_result.get("message") == "Hello GSQL",
                "graph_name": self.graph_name or self.connection.graphname,
                "vertex_type": self.vertex_type
            }
