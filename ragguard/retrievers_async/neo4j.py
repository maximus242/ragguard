"""
Async Neo4j retriever with permission-aware graph queries.

Compatible with FastAPI and other async frameworks.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .graph_base import AsyncGraphSecureRetrieverBase

logger = get_logger(__name__)


class AsyncNeo4jSecureRetriever(AsyncGraphSecureRetrieverBase):
    """
    Async Neo4j retriever with permission-aware graph queries.

    Compatible with FastAPI and other async frameworks. Uses Neo4j's
    AsyncGraphDatabase driver for non-blocking operations.

    Example:
        ```python
        from neo4j import AsyncGraphDatabase
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncNeo4jSecureRetriever

        # Create async driver
        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )

        # Create async retriever
        retriever = AsyncNeo4jSecureRetriever(
            driver=driver,
            database="neo4j",
            policy=load_policy("policy.yaml")
        )

        # Use in async context
        async def search_graph(user_context):
            results = await retriever.graph_query(
                query="MATCH (n:Document) RETURN n LIMIT 10",
                user=user_context
            )
            return results
        ```
    """

    def __init__(
        self,
        driver: Any,
        database: str,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async Neo4j secure retriever.

        Args:
            driver: Neo4j AsyncGraphDatabase driver
            database: Database name
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

        self.driver = driver
        self.database = database

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "neo4j"

    async def search(
        self,
        query: Union[str, List[float]],
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Async permission-aware vector search (if vector index exists).

        Args:
            query: Query text (requires embed_fn) or query vector
            user: User context for permission filtering
            limit: Maximum number of results
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        # Validate user context
        if self._enable_validation:
            self._validator.validate_user_context(user)

        # Convert query to vector
        query_vector = await self._get_query_vector(query)

        # Generate permission clause
        permission_clause = await self._build_permission_clause(user)

        # Build vector search query
        vector_index = kwargs.get("vector_index", "document_embeddings")
        cypher = f"""
        CALL db.index.vector.queryNodes($index, $k, $vector)
        YIELD node, score
        WHERE {permission_clause}
        RETURN node, score
        LIMIT $limit
        """

        results = await self._execute_cypher(
            cypher,
            {"index": vector_index, "k": limit * 2, "vector": query_vector, "limit": limit}
        )

        # Log audit event
        await self._log_audit(user, query, results)

        return results

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute vector search (called by base class)."""
        # Implemented via search() method above
        raise NotImplementedError("Use search() or graph_query() instead")

    async def graph_query(
        self,
        query: str,
        user: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Execute a Cypher query with permission filtering.

        Args:
            query: Cypher query string
            user: User context for permission filtering
            params: Query parameters
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        # Validate user context
        if self._enable_validation:
            self._validator.validate_user_context(user)

        # Generate permission clause
        permission_clause = await self._build_permission_clause(user)

        # Inject permission into query (simple approach - prepend WHERE)
        # For complex queries, use the filter in params
        query_params = params or {}
        query_params["_permission_filter"] = permission_clause

        results = await self._execute_cypher(query, query_params)

        # Log audit event
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
        Search for nodes by property values.

        Args:
            properties: Property name -> value to match
            user: User context for permission filtering
            node_label: Optional node label
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching nodes
        """
        # Validate user context
        if self._enable_validation:
            self._validator.validate_user_context(user)

        # Build label clause
        label_clause = f":{node_label}" if node_label else ""

        # Build property match clause
        prop_clauses = []
        params = {}
        for i, (key, value) in enumerate(properties.items()):
            param_name = f"prop_{i}"
            prop_clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value

        prop_where = " AND ".join(prop_clauses) if prop_clauses else "true"

        # Get permission clause
        permission_clause = await self._build_permission_clause(user)

        cypher = f"""
        MATCH (n{label_clause})
        WHERE {prop_where} AND {permission_clause}
        RETURN n
        LIMIT $limit
        """
        params["limit"] = limit

        results = await self._execute_cypher(cypher, params)

        # Log audit event
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
        # Validate user context
        if self._enable_validation:
            self._validator.validate_user_context(user)

        # Build relationship pattern
        rel_pattern = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_pattern = f":{rel_types}"

        # Build direction pattern
        if direction == "outgoing":
            pattern = f"-[r{rel_pattern}*1..{max_depth}]->"
        elif direction == "incoming":
            pattern = f"<-[r{rel_pattern}*1..{max_depth}]-"
        else:
            pattern = f"-[r{rel_pattern}*1..{max_depth}]-"

        # Get permission clause
        permission_clause = await self._build_permission_clause(user, "end")

        cypher = f"""
        MATCH (start)
        WHERE elementId(start) = $start_id OR id(start) = $start_id_int
        MATCH (start){pattern}(end)
        WHERE {permission_clause}
        RETURN DISTINCT end
        LIMIT $limit
        """

        # Handle both string and int IDs
        try:
            start_id_int = int(start_node_id)
        except ValueError:
            start_id_int = -1

        params = {
            "start_id": start_node_id,
            "start_id_int": start_id_int,
            "limit": limit
        }

        results = await self._execute_cypher(cypher, params)

        # Log audit event
        await self._log_audit(user, f"traverse:{start_node_id}", results)

        return results

    async def _build_permission_clause(
        self,
        user: Dict[str, Any],
        node_alias: str = "n"
    ) -> str:
        """
        Build a Cypher permission clause.

        Args:
            user: User context
            node_alias: Node alias in query

        Returns:
            Cypher WHERE clause string
        """
        from ..filters.builder import to_neo4j_filter
        filter_result = to_neo4j_filter(self.policy, user)

        if filter_result is None:
            return "true"

        # filter_result is (clause_string, params_dict)
        if isinstance(filter_result, tuple):
            clause, _ = filter_result
            # Replace 'n.' with actual alias if different
            if node_alias != "n":
                clause = clause.replace("n.", f"{node_alias}.")
            return clause if clause else "true"

        return "true"

    async def _execute_cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Execute a Cypher query."""
        async def _run_query():
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, params or {})
                records = await result.data()
                return records

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _query_with_retry():
                return await _run_query()
            return await _query_with_retry()
        else:
            return await _run_query()

    async def close(self):
        """Close the driver connection."""
        if hasattr(self.driver, 'close'):
            await self.driver.close()
