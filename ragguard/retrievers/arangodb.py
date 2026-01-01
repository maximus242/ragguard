"""
ArangoDB secure retriever.

Provides permission-aware querying for ArangoDB using AQL.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..exceptions import RetrieverError
from ..filters.backends.arangodb import to_arangodb_filter
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .graph_base import BaseGraphRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig

logger = get_logger(__name__)


class ArangoDBSecureRetriever(BaseGraphRetriever):
    """
    Permission-aware retriever for ArangoDB.

    Wraps an ArangoDB database connection and enforces document-level
    permissions by injecting FILTER conditions into AQL queries.

    Example:
        >>> from arango import ArangoClient
        >>> from ragguard import ArangoDBSecureRetriever, load_policy
        >>>
        >>> client = ArangoClient(hosts='http://localhost:8529')
        >>> db = client.db('mydb', username='root', password='password')
        >>> policy = load_policy("policy.yaml")
        >>> retriever = ArangoDBSecureRetriever(
        ...     database=db,
        ...     collection_name="documents",
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
        database: Any,
        collection_name: str,
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
        doc_alias: str = "doc",
        edge_collection: Optional[str] = None,
        *,
        config: Optional["SecureRetrieverConfig"] = None
    ):
        """
        Initialize the ArangoDB secure retriever.

        Args:
            database: ArangoDB database object (from python-arango)
            collection_name: Document collection name
            policy: Access control policy
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
            doc_alias: Alias for documents in queries (default: "doc")
            edge_collection: Optional edge collection for graph traversals
        """
        self.database = database
        self.collection_name = collection_name
        self.doc_alias = doc_alias
        self.edge_collection = edge_collection

        super().__init__(
            client=database,
            node_label=collection_name,
            policy=policy,
            database=database.name if hasattr(database, 'name') else None,
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
        return "arangodb"

    def _execute_graph_query(
        self,
        query: str,
        permission_filter: tuple[str, dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute an AQL query with permission constraints.
        """
        filter_clause, bind_vars = permission_filter

        # Inject permission filter into the query
        if filter_clause and filter_clause != "true":
            query_upper = query.upper()
            if "FILTER" in query_upper:
                # Find first FILTER and prepend our condition
                filter_idx = query_upper.index("FILTER")
                filter_end = filter_idx + 6
                modified_query = (
                    query[:filter_end] +
                    f" ({filter_clause}) &&" +
                    query[filter_end:]
                )
            elif "RETURN" in query_upper:
                return_idx = query_upper.index("RETURN")
                modified_query = (
                    query[:return_idx] +
                    f"FILTER {filter_clause}\n" +
                    query[return_idx:]
                )
            else:
                modified_query = query + f"\nFILTER {filter_clause}"
        else:
            modified_query = query

        # Add LIMIT if not present
        if limit and "LIMIT" not in query.upper():
            if "RETURN" in modified_query.upper():
                return_idx = modified_query.upper().index("RETURN")
                modified_query = (
                    modified_query[:return_idx] +
                    f"LIMIT {limit}\n" +
                    modified_query[return_idx:]
                )
            else:
                modified_query += f"\nLIMIT {limit}"

        # Merge bind vars
        all_bind_vars = {**kwargs}
        all_bind_vars.update(bind_vars)

        logger.debug(
            "Executing ArangoDB query",
            extra={
                "extra_fields": {
                    "query_length": len(modified_query),
                    "collection": self.collection_name
                }
            }
        )

        cursor = self.database.aql.execute(
            modified_query,
            bind_vars=all_bind_vars
        )
        return list(cursor)

    def _execute_property_search(
        self,
        properties: dict[str, Any],
        permission_filter: tuple[str, dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute property-based document search.
        """
        filter_clause, bind_vars = permission_filter

        # Build property filter conditions
        prop_conditions = []
        for i, (key, value) in enumerate(properties.items()):
            param_name = f"prop_{i}"
            if isinstance(value, list):
                prop_conditions.append(f"{self.doc_alias}.{key} IN @{param_name}")
            else:
                prop_conditions.append(f"{self.doc_alias}.{key} == @{param_name}")
            bind_vars[param_name] = value

        # Combine with permission filter
        all_conditions = prop_conditions.copy()
        if filter_clause and filter_clause != "true":
            all_conditions.append(f"({filter_clause})")

        filter_expr = " && ".join(all_conditions) if all_conditions else "true"

        query = f"""
        FOR {self.doc_alias} IN {self.collection_name}
        FILTER {filter_expr}
        LIMIT {limit}
        RETURN {self.doc_alias}
        """

        logger.debug(
            "Executing ArangoDB property search",
            extra={
                "extra_fields": {
                    "collection": self.collection_name,
                    "properties": list(properties.keys()),
                    "limit": limit
                }
            }
        )

        cursor = self.database.aql.execute(query, bind_vars=bind_vars)
        return list(cursor)

    def _execute_vector_search(
        self,
        vector: list[float],
        permission_filter: tuple[str, dict],
        limit: int,
        vector_field: str = "embedding",
        **kwargs
    ) -> list[Any]:
        """
        Execute vector similarity search.

        Requires ArangoDB 3.10+ with vector indexes.

        Args:
            vector: Embedding vector
            permission_filter: Permission constraints
            limit: Maximum results
            vector_field: Field containing embeddings (default: "embedding")
        """
        filter_clause, bind_vars = permission_filter
        bind_vars["query_vector"] = vector

        # Build the vector search query
        # ArangoDB uses APPROX_NEAR for approximate nearest neighbor search
        if filter_clause and filter_clause != "true":
            query = f"""
            FOR {self.doc_alias} IN APPROX_NEAR(
                {self.collection_name},
                @query_vector,
                {limit * 2}
            )
            FILTER {filter_clause}
            LIMIT {limit}
            RETURN {{
                doc: {self.doc_alias},
                score: {self.doc_alias}._score
            }}
            """
        else:
            query = f"""
            FOR {self.doc_alias} IN APPROX_NEAR(
                {self.collection_name},
                @query_vector,
                {limit}
            )
            RETURN {{
                doc: {self.doc_alias},
                score: {self.doc_alias}._score
            }}
            """

        logger.debug(
            "Executing ArangoDB vector search",
            extra={
                "extra_fields": {
                    "collection": self.collection_name,
                    "vector_dim": len(vector),
                    "limit": limit
                }
            }
        )

        cursor = self.database.aql.execute(query, bind_vars=bind_vars)
        return list(cursor)

    def _build_permission_clause(
        self,
        user: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Build permission FILTER clause for the user.
        """
        return to_arangodb_filter(self.policy, user, self.doc_alias)

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
        Execute graph traversal with permissions.

        Requires edge_collection to be specified at initialization.
        """
        if not self.edge_collection:
            raise RetrieverError(
                "Graph traversal requires edge_collection to be specified. "
                "Initialize retriever with edge_collection='your_edges'"
            )

        filter_clause, bind_vars = self._build_permission_clause(user)
        bind_vars["start_id"] = start_node_id

        # Build direction
        if direction == "outgoing":
            dir_str = "OUTBOUND"
        elif direction == "incoming":
            dir_str = "INBOUND"
        else:
            dir_str = "ANY"

        # Build traversal query
        filter_part = f"FILTER {filter_clause}" if filter_clause and filter_clause != "true" else ""

        query = f"""
        FOR v, e, p IN 1..{depth} {dir_str}
            @start_id
            {self.edge_collection}
            OPTIONS {{bfs: true}}
        FILTER e.type == @edge_type
        {filter_part}
        LIMIT {limit}
        RETURN DISTINCT v
        """

        bind_vars["edge_type"] = relationship_type

        logger.debug(
            "Executing ArangoDB traversal",
            extra={
                "extra_fields": {
                    "start_node": start_node_id,
                    "relationship": relationship_type,
                    "direction": direction,
                    "depth": depth
                }
            }
        )

        cursor = self.database.aql.execute(query, bind_vars=bind_vars)
        return list(cursor)

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check ArangoDB connection health.
        """
        # Check database properties
        properties = self.database.properties()

        # Check if collection exists
        collection = self.database.collection(self.collection_name)
        collection_exists = collection.properties() if collection else None

        return {
            "connection_alive": True,
            "database_name": properties.get("name"),
            "collection_name": self.collection_name,
            "collection_exists": collection_exists is not None,
            "is_system": properties.get("isSystem", False)
        }
