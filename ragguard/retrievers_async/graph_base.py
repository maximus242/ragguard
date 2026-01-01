"""
Abstract base class for async graph database retrievers.

Provides common functionality for async graph retrievers including:
- Graph query execution
- Permission filter building
- Traversal operations
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

from ..audit.logger import AuditLogger
from ..policy.models import Policy
from ..retry import RetryConfig
from ..validation import ValidationConfig
from .base import AsyncSecureRetrieverBase


class AsyncGraphSecureRetrieverBase(AsyncSecureRetrieverBase):
    """
    Abstract base class for async graph database retrievers.

    Extends AsyncSecureRetrieverBase with graph-specific operations:
    - Graph queries (Cypher, Gremlin, GSQL, AQL)
    - Property-based search
    - Graph traversals

    Subclasses must implement backend-specific methods.
    """

    def __init__(
        self,
        policy: Policy,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retry: bool = True,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize async graph secure retriever base.

        Args:
            policy: Access control policy
            embed_fn: Optional embedding function
            audit_logger: Optional audit logger
            retry_config: Optional retry configuration
            enable_retry: Whether to enable retry logic
            validation_config: Optional validation configuration
            enable_validation: Whether to enable input validation
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

    @abstractmethod
    async def graph_query(
        self,
        query: str,
        user: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Execute a graph query with permission filtering.

        Args:
            query: Graph query string (Cypher, Gremlin, GSQL, AQL)
            user: User context for permission filtering
            params: Query parameters
            **kwargs: Additional backend-specific arguments

        Returns:
            List of results
        """
        pass

    @abstractmethod
    async def property_search(
        self,
        properties: Dict[str, Any],
        user: Dict[str, Any],
        node_label: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        """
        Search for nodes by property values with permission filtering.

        Args:
            properties: Dictionary of property name -> value to match
            user: User context for permission filtering
            node_label: Optional node label to filter by
            limit: Maximum number of results
            **kwargs: Additional backend-specific arguments

        Returns:
            List of matching nodes
        """
        pass

    @abstractmethod
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
        Traverse the graph from a starting node with permission filtering.

        Args:
            start_node_id: ID of the starting node
            user: User context for permission filtering
            relationship_types: Types of relationships to follow
            direction: Traversal direction ("outgoing", "incoming", "both")
            max_depth: Maximum traversal depth
            limit: Maximum number of results
            **kwargs: Additional backend-specific arguments

        Returns:
            List of nodes reached by traversal
        """
        pass

    @abstractmethod
    async def _build_permission_clause(
        self,
        user: Dict[str, Any],
        node_alias: str = "n"
    ) -> str:
        """
        Build a permission clause for the query language.

        Args:
            user: User context
            node_alias: Alias for the node in the query

        Returns:
            Permission clause string
        """
        pass
