"""
Amazon Neptune secure retriever.

Provides permission-aware querying for Amazon Neptune using Gremlin.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..exceptions import RetrieverError
from ..filters.backends.neptune import to_neptune_filter
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .graph_base import BaseGraphRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig

logger = get_logger(__name__)


class NeptuneSecureRetriever(BaseGraphRetriever):
    """
    Permission-aware retriever for Amazon Neptune graph database.

    Wraps a Gremlin client connection and enforces document-level
    permissions by injecting filter predicates into traversals.

    Example:
        >>> from gremlin_python.driver import client
        >>> from ragguard import NeptuneSecureRetriever, load_policy
        >>>
        >>> gremlin_client = client.Client(
        ...     'wss://your-neptune-endpoint:8182/gremlin',
        ...     'g'
        ... )
        >>> policy = load_policy("policy.yaml")
        >>> retriever = NeptuneSecureRetriever(
        ...     client=gremlin_client,
        ...     node_label="Document",
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
        client: Any,
        node_label: str,
        policy: Policy,
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
        *,
        config: Optional["SecureRetrieverConfig"] = None
    ):
        """
        Initialize the Neptune secure retriever.

        Args:
            client: Gremlin client connection
            node_label: Vertex label for documents (e.g., "Document")
            policy: Access control policy
            audit_logger: Optional audit logger
            embed_fn: Optional embedding function for vector search
            enable_filter_cache: Whether to enable filter caching
            filter_cache_size: Maximum cached filters
            retry_config: Retry configuration
            enable_retry: Whether to enable retries
            validation_config: Validation configuration
            enable_validation: Whether to enable validation
            enable_circuit_breaker: Whether to enable circuit breaker
            circuit_breaker_config: Circuit breaker configuration
        """
        self.gremlin_client = client

        super().__init__(
            client=client,
            node_label=node_label,
            policy=policy,
            database=None,  # Neptune doesn't have separate databases
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
        return "neptune"

    def _execute_graph_query(
        self,
        query: str,
        permission_filter: List[dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute a Gremlin query string with permission constraints.

        Note: For string queries, permissions are applied as a post-filter
        since we can't easily inject into arbitrary Gremlin strings.
        For better performance, use property-based search.
        """
        # Execute the query
        result_set = self.gremlin_client.submit(query)
        results = result_set.all().result()

        # Post-filter based on permissions if needed
        if permission_filter:
            filtered = []
            for result in results:
                if self._matches_filters(result, permission_filter):
                    filtered.append(result)
                    if len(filtered) >= limit:
                        break
            return filtered

        return results[:limit]

    def _matches_filters(self, vertex: dict, filters: List[dict]) -> bool:
        """
        Check if a vertex matches the permission filters.

        Used for post-filtering when we can't inject into the query.
        """
        for f in filters:
            if f["type"] == "or":
                # At least one child group must match
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

    def _execute_property_search(
        self,
        properties: dict[str, Any],
        permission_filter: List[dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute property-based vertex search with Gremlin.
        """
        try:
            from gremlin_python.process.traversal import P
        except ImportError:
            raise RetrieverError(
                "gremlin_python not installed. Install with: pip install gremlinpython"
            )

        # Build Gremlin query string
        # Start with label filter
        query_parts = [f"g.V().hasLabel('{self.node_label}')"]

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

        # Add permission filters as Gremlin predicates
        for f in permission_filter:
            filter_str = self._filter_to_gremlin_string(f)
            if filter_str:
                query_parts.append(filter_str)

        # Add limit
        query_parts.append(f".limit({limit})")
        query_parts.append(".valueMap(true)")

        query = "".join(query_parts)

        logger.debug(
            "Executing Neptune property search",
            extra={
                "extra_fields": {
                    "node_label": self.node_label,
                    "properties": list(properties.keys()),
                    "query_length": len(query)
                }
            }
        )

        result_set = self.gremlin_client.submit(query)
        return result_set.all().result()

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
            # Build or() traversal
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

    def _build_permission_clause(
        self,
        user: dict[str, Any]
    ) -> List[dict]:
        """
        Build permission filter specs for the user.
        """
        return to_neptune_filter(self.policy, user)

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
        permission_filter = self._build_permission_clause(user)

        # Build direction-specific traversal
        if direction == "outgoing":
            edge_step = f".out('{relationship_type}')"
        elif direction == "incoming":
            edge_step = f".in('{relationship_type}')"
        else:  # both
            edge_step = f".both('{relationship_type}')"

        # Build repeat for depth traversal
        if depth > 1:
            edge_step = f".repeat({edge_step.lstrip('.')}).times({depth})"

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

        logger.debug(
            "Executing Neptune traversal",
            extra={
                "extra_fields": {
                    "start_node": start_node_id,
                    "relationship": relationship_type,
                    "direction": direction,
                    "depth": depth
                }
            }
        )

        result_set = self.gremlin_client.submit(query)
        return result_set.all().result()

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Neptune connection health.
        """
        # Run a simple query to verify connectivity
        result_set = self.gremlin_client.submit("g.V().limit(1).count()")
        count = result_set.all().result()

        return {
            "connection_alive": True,
            "node_label": self.node_label,
            "can_query": True
        }

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the client on context exit."""
        if hasattr(self, 'gremlin_client') and self.gremlin_client:
            try:
                self.gremlin_client.close()
            except Exception as e:
                logger.warning(f"Error closing Gremlin client: {e}")
        return super().__exit__(exc_type, exc_val, exc_tb)
