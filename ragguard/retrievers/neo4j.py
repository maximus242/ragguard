"""
Neo4j secure retriever.

Provides permission-aware querying for Neo4j graph databases using Cypher.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from ..audit.logger import AuditLogger
from ..circuit_breaker import CircuitBreakerConfig
from ..exceptions import RetrieverError
from ..filters.backends.neo4j import to_neo4j_filter
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .graph_base import BaseGraphRetriever

if TYPE_CHECKING:
    from ..config import SecureRetrieverConfig

logger = get_logger(__name__)


class Neo4jSecureRetriever(BaseGraphRetriever):
    """
    Permission-aware retriever for Neo4j graph databases.

    Wraps a Neo4j driver and enforces document-level permissions by
    injecting WHERE clauses into Cypher queries.

    Example:
        >>> from neo4j import GraphDatabase
        >>> from ragguard import Neo4jSecureRetriever, load_policy
        >>>
        >>> driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        >>> policy = load_policy("policy.yaml")
        >>> retriever = Neo4jSecureRetriever(
        ...     driver=driver,
        ...     node_label="Document",
        ...     policy=policy
        ... )
        >>>
        >>> # Property-based search with permissions
        >>> results = retriever.search(
        ...     query={"category": "engineering"},
        ...     user={"id": "alice", "department": "engineering"}
        ... )
        >>>
        >>> # Custom Cypher query with permission injection
        >>> results = retriever.search(
        ...     query="MATCH (d:Document)-[:AUTHORED_BY]->(a:Author) WHERE a.name = 'Alice'",
        ...     user={"id": "alice", "roles": ["engineer"]}
        ... )
    """

    def __init__(
        self,
        driver: Any,
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
        node_alias: str = "doc",
        *,
        config: Optional["SecureRetrieverConfig"] = None
    ):
        """
        Initialize the Neo4j secure retriever.

        Args:
            driver: Neo4j driver instance
            node_label: Label of document nodes (e.g., "Document")
            policy: Access control policy
            database: Optional database name (for multi-database Neo4j)
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
            node_alias: Alias for document node in queries (default: "doc")
        """
        # Validate driver type
        try:
            from neo4j import Driver
            if not isinstance(driver, Driver):
                raise RetrieverError(
                    f"Expected neo4j.Driver, got {type(driver).__name__}. "
                    f"Install neo4j with: pip install neo4j"
                )
        except ImportError:
            # Allow duck typing if neo4j not installed during type check
            logger.warning(
                "neo4j package not installed, skipping driver type check"
            )

        self.driver = driver
        self.node_alias = node_alias

        super().__init__(
            client=driver,
            node_label=node_label,
            policy=policy,
            database=database,
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
        return "neo4j"

    def _get_session(self):
        """Get a Neo4j session, optionally for a specific database."""
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def _execute_graph_query(
        self,
        query: str,
        permission_filter: tuple[str, dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute a Cypher query with permission constraints.

        The permission filter is injected into the query's WHERE clause.
        """
        where_clause, filter_params = permission_filter

        # Inject permission filter into the query
        if where_clause:
            # Find WHERE clause in query or add one
            query_upper = query.upper()
            if "WHERE" in query_upper:
                # Append to existing WHERE with AND
                where_idx = query_upper.index("WHERE")
                where_end = where_idx + 5  # Length of "WHERE"
                modified_query = (
                    query[:where_end] +
                    f" ({where_clause}) AND" +
                    query[where_end:]
                )
            else:
                # Find appropriate place to insert WHERE
                # Typically after MATCH clause and before RETURN
                if "RETURN" in query_upper:
                    return_idx = query_upper.index("RETURN")
                    modified_query = (
                        query[:return_idx] +
                        f"WHERE {where_clause} " +
                        query[return_idx:]
                    )
                else:
                    # Just append WHERE clause
                    modified_query = query + f" WHERE {where_clause}"
        else:
            modified_query = query

        # Add LIMIT if not present
        if limit and "LIMIT" not in query.upper():
            modified_query += f" LIMIT {limit}"

        # Merge parameters
        all_params = {**kwargs}
        all_params.update(filter_params)

        logger.debug(
            "Executing Neo4j query",
            extra={
                "extra_fields": {
                    "query": modified_query[:200],
                    "param_count": len(all_params)
                }
            }
        )

        with self._get_session() as session:
            result = session.run(modified_query, **all_params)
            return [record.data() for record in result]

    def _execute_property_search(
        self,
        properties: dict[str, Any],
        permission_filter: tuple[str, dict],
        limit: int,
        **kwargs
    ) -> list[Any]:
        """
        Execute property-based node search.
        """
        where_clause, filter_params = permission_filter

        # Build property match conditions
        prop_conditions = []
        prop_params = {}
        for i, (key, value) in enumerate(properties.items()):
            param_name = f"prop_{i}"
            prop_conditions.append(f"{self.node_alias}.{key} = ${param_name}")
            prop_params[param_name] = value

        # Combine with permission filter
        all_conditions = prop_conditions.copy()
        if where_clause:
            all_conditions.append(f"({where_clause})")

        where = " AND ".join(all_conditions) if all_conditions else "true"

        query = f"""
        MATCH ({self.node_alias}:{self.node_label})
        WHERE {where}
        RETURN {self.node_alias}
        LIMIT {limit}
        """

        all_params = {**prop_params, **filter_params}

        logger.debug(
            "Executing Neo4j property search",
            extra={
                "extra_fields": {
                    "node_label": self.node_label,
                    "properties": list(properties.keys()),
                    "limit": limit
                }
            }
        )

        with self._get_session() as session:
            result = session.run(query, **all_params)
            return [record.data() for record in result]

    def _execute_vector_search(
        self,
        vector: list[float],
        permission_filter: tuple[str, dict],
        limit: int,
        index_name: str = "document_embeddings",
        **kwargs
    ) -> list[Any]:
        """
        Execute vector similarity search using Neo4j's vector index.

        Requires Neo4j 5.11+ with vector search capability.

        Args:
            vector: Embedding vector
            permission_filter: Permission constraints
            limit: Maximum results
            index_name: Name of the vector index (default: "document_embeddings")
        """
        where_clause, filter_params = permission_filter

        # Build the vector search query
        if where_clause:
            query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
            YIELD node AS {self.node_alias}, score
            WHERE {where_clause}
            RETURN {self.node_alias}, score
            ORDER BY score DESC
            LIMIT $limit
            """
        else:
            query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
            YIELD node AS {self.node_alias}, score
            RETURN {self.node_alias}, score
            ORDER BY score DESC
            LIMIT $limit
            """

        params = {
            "index_name": index_name,
            "top_k": limit * 2,  # Fetch more to account for permission filtering
            "query_vector": vector,
            "limit": limit,
            **filter_params
        }

        logger.debug(
            "Executing Neo4j vector search",
            extra={
                "extra_fields": {
                    "index_name": index_name,
                    "vector_dim": len(vector),
                    "limit": limit
                }
            }
        )

        with self._get_session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def _build_permission_clause(
        self,
        user: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Build permission WHERE clause for the user.
        """
        return to_neo4j_filter(self.policy, user, self.node_alias)

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
            pattern = f"-[:{relationship_type}*1..{depth}]->"
        elif direction == "incoming":
            pattern = f"<-[:{relationship_type}*1..{depth}]-"
        else:  # both
            pattern = f"-[:{relationship_type}*1..{depth}]-"

        # Build query
        if where_clause:
            query = f"""
            MATCH (start:{self.node_label} {{id: $start_id}})
            MATCH (start){pattern}({self.node_alias}:{self.node_label})
            WHERE {where_clause}
            RETURN DISTINCT {self.node_alias}
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (start:{self.node_label} {{id: $start_id}})
            MATCH (start){pattern}({self.node_alias}:{self.node_label})
            RETURN DISTINCT {self.node_alias}
            LIMIT {limit}
            """

        params = {"start_id": start_node_id, **filter_params}

        logger.debug(
            "Executing Neo4j traversal",
            extra={
                "extra_fields": {
                    "start_node": start_node_id,
                    "relationship": relationship_type,
                    "direction": direction,
                    "depth": depth
                }
            }
        )

        with self._get_session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def _check_backend_health(self) -> dict[str, Any]:
        """
        Check Neo4j connection health.
        """
        with self._get_session() as session:
            # Run a simple query to verify connectivity
            result = session.run("CALL dbms.components() YIELD name, versions")
            components = list(result)

            return {
                "connection_alive": True,
                "database": self.database or "default",
                "node_label": self.node_label,
                "components": [
                    {"name": c["name"], "versions": c["versions"]}
                    for c in components
                ]
            }

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the driver on context exit."""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except Exception as e:
                logger.warning(f"Error closing Neo4j driver: {e}")
        return super().__exit__(exc_type, exc_val, exc_tb)
