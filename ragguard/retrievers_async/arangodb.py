"""
Async ArangoDB retriever with permission-aware AQL queries.

Compatible with FastAPI and other async frameworks.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..filters.backends.arangodb import to_arangodb_filter
from ..logging import get_logger
from ..policy.models import Policy
from ..retry import RetryConfig, async_retry_on_failure
from ..validation import ValidationConfig
from .graph_base import AsyncGraphSecureRetrieverBase

logger = get_logger(__name__)


class AsyncArangoDBSecureRetriever(AsyncGraphSecureRetrieverBase):
    """
    Async ArangoDB retriever with permission-aware AQL queries.

    Compatible with FastAPI and other async frameworks. Uses thread pool
    execution since python-arango doesn't have native async support.

    Example:
        ```python
        from arango import ArangoClient
        from ragguard import load_policy
        from ragguard.retrievers_async import AsyncArangoDBSecureRetriever

        # Create client
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db('mydb', username='root', password='password')

        # Create async retriever
        retriever = AsyncArangoDBSecureRetriever(
            database=db,
            collection_name="documents",
            policy=load_policy("policy.yaml")
        )

        # Use in async context
        async def search_docs(user_context):
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
        database: Any,
        collection_name: str,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True,
        doc_alias: str = "doc",
        edge_collection: Optional[str] = None
    ):
        """
        Initialize async ArangoDB secure retriever.

        Args:
            database: ArangoDB database object (from python-arango)
            collection_name: Document collection name
            policy: Access control policy
            embed_fn: Optional embedding function
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic (default: True)
            validation_config: Optional validation configuration
            enable_validation: Whether to enable input validation (default: True)
            doc_alias: Alias for documents in queries (default: "doc")
            edge_collection: Optional edge collection for graph traversals
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

        self.database = database
        self.collection_name = collection_name
        self.doc_alias = doc_alias
        self.edge_collection = edge_collection

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "arangodb"

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
            query: Query (dict for properties, string for AQL, list for vector)
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

        if isinstance(query, list):
            # Vector search
            return await self._vector_search(query, user, limit=limit, **kwargs)

        # AQL query
        return await self.graph_query(query, user, limit=limit, **kwargs)

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        """Execute vector search."""
        # Delegate to _vector_search
        raise NotImplementedError("Use search() with vector query instead")

    async def _vector_search(
        self,
        vector: List[float],
        user: Dict[str, Any],
        limit: int = 10,
        vector_field: str = "embedding",
        **kwargs
    ) -> List[Any]:
        """
        Execute vector similarity search.

        Requires ArangoDB 3.10+ with vector indexes.

        Args:
            vector: Embedding vector
            user: User context
            limit: Maximum results
            vector_field: Field containing embeddings (default: "embedding")
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        permission_filter = to_arangodb_filter(self.policy, user, self.doc_alias)

        loop = asyncio.get_running_loop()

        def _sync_search():
            filter_clause, bind_vars = permission_filter
            bind_vars["query_vector"] = vector

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

            cursor = self.database.aql.execute(query, bind_vars=bind_vars)
            return list(cursor)

        if self._enable_retry:
            @async_retry_on_failure(config=self._retry_config)
            async def _search_with_retry():
                return await loop.run_in_executor(None, _sync_search)
            results = await _search_with_retry()
        else:
            results = await loop.run_in_executor(None, _sync_search)

        await self._log_audit(user, f"vector_search:{len(vector)}d", results)
        return results

    async def graph_query(
        self,
        query: str,
        user: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Execute an AQL query with permission filtering.

        Args:
            query: AQL query string
            user: User context for permission filtering
            params: Query parameters (bind variables)
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        permission_filter = to_arangodb_filter(self.policy, user, self.doc_alias)
        limit = kwargs.get("limit", 100)

        loop = asyncio.get_running_loop()

        def _sync_query():
            filter_clause, bind_vars = permission_filter

            # Inject permission filter
            modified_query = query
            if filter_clause and filter_clause != "true":
                query_upper = query.upper()
                if "FILTER" in query_upper:
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
            all_bind_vars = {**(params or {})}
            all_bind_vars.update(bind_vars)

            cursor = self.database.aql.execute(modified_query, bind_vars=all_bind_vars)
            return list(cursor)

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
        Search for documents by property values.

        Args:
            properties: Property name -> value to match
            user: User context for permission filtering
            node_label: Optional collection name (uses default if not specified)
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of matching documents
        """
        if self._enable_validation:
            self._validator.validate_user_context(user)

        collection = node_label or self.collection_name
        permission_filter = to_arangodb_filter(self.policy, user, self.doc_alias)

        loop = asyncio.get_running_loop()

        def _sync_search():
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
            FOR {self.doc_alias} IN {collection}
            FILTER {filter_expr}
            LIMIT {limit}
            RETURN {self.doc_alias}
            """

            cursor = self.database.aql.execute(query, bind_vars=bind_vars)
            return list(cursor)

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

        Requires edge_collection to be specified at initialization.

        Args:
            start_node_id: ID of the starting node
            user: User context for permission filtering
            relationship_types: Types of relationships to follow (edge types)
            direction: "outgoing", "incoming", or "both"
            max_depth: Maximum traversal depth
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of nodes reached
        """
        if not self.edge_collection:
            raise RetrieverError(
                "Graph traversal requires edge_collection to be specified. "
                "Initialize retriever with edge_collection='your_edges'"
            )

        if self._enable_validation:
            self._validator.validate_user_context(user)

        permission_filter = to_arangodb_filter(self.policy, user, self.doc_alias)
        relationship_type = relationship_types[0] if relationship_types else None

        loop = asyncio.get_running_loop()

        def _sync_traverse():
            filter_clause, bind_vars = permission_filter
            bind_vars["start_id"] = start_node_id

            # Build direction
            if direction == "outgoing":
                dir_str = "OUTBOUND"
            elif direction == "incoming":
                dir_str = "INBOUND"
            else:
                dir_str = "ANY"

            # Build filter part
            filter_parts = []
            if filter_clause and filter_clause != "true":
                filter_parts.append(filter_clause)
            if relationship_type:
                bind_vars["edge_type"] = relationship_type
                filter_parts.append("e.type == @edge_type")

            filter_part = "FILTER " + " && ".join(filter_parts) if filter_parts else ""

            query = f"""
            FOR v, e, p IN 1..{max_depth} {dir_str}
                @start_id
                {self.edge_collection}
                OPTIONS {{bfs: true}}
            {filter_part}
            LIMIT {limit}
            RETURN DISTINCT v
            """

            cursor = self.database.aql.execute(query, bind_vars=bind_vars)
            return list(cursor)

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
        node_alias: str = "doc"
    ) -> str:
        """Build an AQL permission clause."""
        filter_clause, _ = to_arangodb_filter(self.policy, user, node_alias)
        return filter_clause or "true"

    async def close(self):
        """Close the database connection (no-op for python-arango)."""
        pass
