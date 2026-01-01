"""
Base class for permission-aware graph database retrievers.

Extends BaseSecureRetriever with graph-specific capabilities for databases
like Neo4j, Amazon Neptune, TigerGraph, and ArangoDB.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .base import BaseSecureRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig

logger = get_logger(__name__)


class BaseGraphRetriever(BaseSecureRetriever):
    """
    Base class for permission-aware graph database retrievers.

    Extends BaseSecureRetriever with graph-specific capabilities:
    - Graph query execution (Cypher, Gremlin, GSQL, AQL)
    - Relationship-based permission models
    - Node type/label handling

    Graph retrievers can handle multiple query types:
    - String: Native graph query (e.g., Cypher for Neo4j)
    - Dict: Node property matching
    - List[float]: Vector similarity search (if graph has embeddings)
    """

    def __init__(
        self,
        client: Any,
        node_label: str,
        policy: Policy,
        database: Optional[str] = None,
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
        Initialize the graph retriever.

        Args:
            client: Graph database driver/client
            node_label: Primary node label/type to query (e.g., "Document")
            policy: Access control policy
            database: Optional database/graph name (for multi-database setups)
            audit_logger: Optional audit logger
            embed_fn: Optional function to convert text to embeddings (for vector search)
            enable_filter_cache: Whether to enable filter caching
            filter_cache_size: Maximum cached filters
            retry_config: Retry configuration
            enable_retry: Whether to enable retries
            validation_config: Validation configuration
            enable_validation: Whether to enable validation
            enable_circuit_breaker: Whether to enable circuit breaker
            circuit_breaker_config: Circuit breaker configuration
        """
        self.node_label = node_label
        self.database = database

        # Call parent with collection set to node_label for compatibility
        super().__init__(
            client=client,
            collection=node_label,
            policy=policy,
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

        logger.info(
            "Graph retriever initialized",
            extra={
                "extra_fields": {
                    "backend": self.backend_name,
                    "node_label": node_label,
                    "database": database
                }
            }
        )

    def _execute_search(
        self,
        query: Union[str, list[float], dict],
        filter: Any,
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute the search with permission filter.

        For graph databases, this routes to the appropriate method based on query type:
        - String: Graph query (Cypher, Gremlin, etc.)
        - Dict: Property matching
        - List[float]: Vector similarity (if supported)

        Args:
            query: Graph query, property dict, or embedding vector
            filter: Permission filter (Cypher WHERE clause, Gremlin predicates, etc.)
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching nodes/documents
        """
        if isinstance(query, str):
            # Native graph query with permission injection
            return self._execute_graph_query(query, filter, limit, **kwargs)
        elif isinstance(query, dict):
            # Property-based search
            return self._execute_property_search(query, filter, limit, **kwargs)
        elif isinstance(query, list):
            # Vector similarity (if graph supports embeddings)
            return self._execute_vector_search(query, filter, limit, **kwargs)
        else:
            raise ValueError(
                f"Unsupported query type: {type(query).__name__}. "
                f"Expected str (graph query), dict (property match), or list (vector)."
            )

    @abstractmethod
    def _execute_graph_query(
        self,
        query: str,
        permission_filter: Any,
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute a native graph query with permission constraints.

        Args:
            query: Native graph query (Cypher, Gremlin, GSQL, AQL)
            permission_filter: Permission filter to inject into query
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching nodes/documents
        """
        pass

    @abstractmethod
    def _execute_property_search(
        self,
        properties: dict[str, Any],
        permission_filter: Any,
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute a property-based node search with permission constraints.

        Args:
            properties: Node properties to match
            permission_filter: Permission filter
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching nodes
        """
        pass

    def _execute_vector_search(
        self,
        vector: list[float],
        permission_filter: Any,
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute vector similarity search with permission constraints.

        Default implementation raises NotImplementedError. Override in subclasses
        that support vector search (e.g., Neo4j with vector index, Weaviate).

        Args:
            vector: Embedding vector for similarity search
            permission_filter: Permission filter
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching nodes ordered by similarity

        Raises:
            NotImplementedError: If vector search is not supported
        """
        raise NotImplementedError(
            f"{self.backend_name} does not support vector similarity search. "
            f"Use a graph query or property search instead."
        )

    @abstractmethod
    def _build_permission_clause(
        self,
        user: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Build permission constraint clause for the graph query language.

        Args:
            user: User context

        Returns:
            Tuple of (clause_string, parameters)
            - clause_string: Query fragment (e.g., Cypher WHERE clause)
            - parameters: Query parameters to bind
        """
        pass

    @staticmethod
    def _validate_graph_identifier(value: str, param_name: str) -> None:
        """
        Validate graph identifier to prevent query injection.

        Node IDs and relationship types must be safe for interpolation into
        graph queries (Cypher, Gremlin, GSQL, AQL).

        Args:
            value: The identifier to validate
            param_name: Name of the parameter (for error messages)

        Raises:
            ValueError: If the identifier contains dangerous characters
        """
        if not value or not isinstance(value, str):
            raise ValueError(f"{param_name} must be a non-empty string")

        if len(value) > 256:
            raise ValueError(f"{param_name} too long: {len(value)} chars (max 256)")

        # Reject characters that could enable injection across graph query languages
        # These characters are dangerous in Cypher, Gremlin, GSQL, and/or AQL
        dangerous_chars = {
            '"', "'",  # String delimiters - enable breakout
            ';', '\n',  # Statement terminators
            '{', '}',  # Object/block delimiters
            '(', ')',  # Function calls, grouping
            '[', ']',  # Array/list access
            '`',  # Backtick escaping
            '\\',  # Escape sequences
            '/', '*',  # Comments
            '|', '&',  # Logical operators in some contexts
            '<', '>',  # Comparison/injection
            '=',  # Assignment/comparison
            '\r', '\t', '\0',  # Control characters
        }

        found_dangerous = [c for c in value if c in dangerous_chars]
        if found_dangerous:
            raise ValueError(
                f"Invalid {param_name}: contains dangerous characters {found_dangerous!r}. "
                f"Only alphanumeric characters, hyphens, underscores, colons, and dots are allowed."
            )

    def traverse(
        self,
        start_node_id: str,
        relationship_type: str,
        user: dict[str, Any],
        direction: str = "outgoing",
        depth: int = 1,
        limit: int = 100,
        **kwargs
    ) -> list[Any]:
        """
        Traverse relationships from a starting node with permission filtering.

        This is a graph-specific operation that follows relationships while
        respecting document-level permissions.

        Args:
            start_node_id: ID of the starting node
            relationship_type: Type of relationship to traverse
            user: User context for permission filtering
            direction: Traversal direction ("outgoing", "incoming", "both")
            depth: Maximum traversal depth (1 = immediate neighbors)
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of reachable nodes that the user has access to
        """
        # Validate inputs to prevent query injection
        self._validate_graph_identifier(start_node_id, "start_node_id")
        self._validate_graph_identifier(relationship_type, "relationship_type")

        # Validate direction
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(
                f"Invalid direction: {direction}. "
                f"Must be 'outgoing', 'incoming', or 'both'."
            )

        # Build traversal query with permissions
        return self._execute_traversal(
            start_node_id=start_node_id,
            relationship_type=relationship_type,
            user=user,
            direction=direction,
            depth=depth,
            limit=limit,
            **kwargs
        )

    @abstractmethod
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
        Execute a relationship traversal with permission constraints.

        Must be implemented by each graph backend.

        Args:
            start_node_id: Starting node ID
            relationship_type: Relationship type to follow
            user: User context
            direction: Traversal direction
            depth: Maximum depth
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of reachable, accessible nodes
        """
        pass
