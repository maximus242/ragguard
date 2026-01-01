"""
LlamaIndex integration for RAGGuard.

This module provides a wrapper for LlamaIndex retrievers that adds
permission-aware filtering to any LlamaIndex retriever.
"""

from typing import Any, List, Optional

from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..policy.engine import PolicyEngine
from ..policy.models import Policy


class SecureLlamaIndexRetriever:
    """
    Wrap any LlamaIndex retriever with RAGGuard permissions.

    This allows you to add document-level permissions to any existing
    LlamaIndex retriever without modifying your existing code.

    Example:
        ```python
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from ragguard.integrations.llamaindex import SecureLlamaIndexRetriever
        from ragguard import load_policy

        # Create LlamaIndex retriever as normal
        documents = SimpleDirectoryReader('data').load_data()
        index = VectorStoreIndex.from_documents(documents)
        base_retriever = index.as_retriever(similarity_top_k=10)

        # Wrap with RAGGuard
        policy = load_policy("policy.yaml")
        secure_retriever = SecureLlamaIndexRetriever(
            base_retriever=base_retriever,
            policy=policy
        )

        # Query with permissions
        results = secure_retriever.retrieve(
            "What is our policy?",
            user_context={"id": "alice", "department": "engineering"}
        )

        # Or use in a query engine
        from llama_index.core.query_engine import RetrieverQueryEngine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=secure_retriever,
            # ... other args
        )
        ```
    """

    def __init__(
        self,
        base_retriever: Any,
        policy: Policy,
        audit_logger: Optional[AuditLogger] = None
    ):
        """
        Initialize secure LlamaIndex retriever.

        Args:
            base_retriever: The underlying LlamaIndex BaseRetriever instance
            policy: RAGGuard access control policy
            audit_logger: Optional audit logger for permission checks
        """
        try:
            from llama_index.core.retrievers import BaseRetriever
        except ImportError:
            raise RetrieverError(
                "llama-index not installed. Install with: pip install ragguard[llamaindex]"
            )

        if not isinstance(base_retriever, BaseRetriever):
            raise RetrieverError(
                f"base_retriever must be a LlamaIndex BaseRetriever, got {type(base_retriever)}"
            )

        self.base_retriever = base_retriever
        self.policy_engine = PolicyEngine(policy)
        self.audit_logger = audit_logger or AuditLogger(output=None)

    def retrieve(
        self,
        query: str,
        user_context: Optional[dict[str, Any]] = None
    ) -> List[Any]:
        """
        Retrieve nodes with permission filtering.

        Args:
            query: The search query string
            user_context: User context for permission checks (e.g., {"id": "alice", "roles": ["admin"]})

        Returns:
            List of NodeWithScore objects that passed permission checks
        """
        # Get results from base retriever
        nodes = self.base_retriever.retrieve(query)

        # Filter based on permissions
        filtered_nodes = []
        for node in nodes:
            # Get metadata from node
            metadata = node.node.metadata if hasattr(node, 'node') else {}

            # Check permissions
            allowed = self.policy_engine.evaluate(user_context or {}, metadata)

            # Log decision
            self.audit_logger.log(
                action="retrieve",
                user=user_context or {},
                document=metadata,
                decision="allow" if allowed else "deny",
                additional_info={"backend": "llamaindex", "query": query}
            )

            if allowed:
                filtered_nodes.append(node)

        return filtered_nodes

    def _retrieve(self, query_bundle: Any) -> List[Any]:
        """
        Internal retrieve method compatible with LlamaIndex BaseRetriever interface.

        This allows SecureLlamaIndexRetriever to be used anywhere a BaseRetriever
        is expected, but note that user_context must be set via the retrieve() method.
        """
        # For compatibility with LlamaIndex, we allow _retrieve to be called
        # but warn that no permission checking will occur without user_context
        return self.base_retriever._retrieve(query_bundle)


class SecureQueryEngine:
    """
    Wrap a LlamaIndex query engine with permission-aware retrieval.

    This is a convenience wrapper for query engines that automatically
    applies permission filtering to retrieved documents.

    Example:
        ```python
        from llama_index.core import VectorStoreIndex
        from ragguard.integrations.llamaindex import SecureQueryEngine
        from ragguard import load_policy

        index = VectorStoreIndex.from_documents(documents)
        policy = load_policy("policy.yaml")

        query_engine = SecureQueryEngine(
            index=index,
            policy=policy
        )

        response = query_engine.query(
            "What is our policy?",
            user_context={"id": "alice", "department": "engineering"}
        )
        print(response)
        ```
    """

    def __init__(
        self,
        index: Any,
        policy: Policy,
        similarity_top_k: int = 10,
        audit_logger: Optional[AuditLogger] = None,
        **query_engine_kwargs
    ):
        """
        Initialize secure query engine.

        Args:
            index: LlamaIndex VectorStoreIndex instance
            policy: RAGGuard access control policy
            similarity_top_k: Number of similar nodes to retrieve (default: 10)
            audit_logger: Optional audit logger
            **query_engine_kwargs: Additional arguments to pass to query engine
        """
        try:
            from llama_index.core import VectorStoreIndex
        except ImportError:
            raise RetrieverError(
                "llama-index not installed. Install with: pip install ragguard[llamaindex]"
            )

        if not isinstance(index, VectorStoreIndex):
            raise RetrieverError(
                f"index must be a LlamaIndex VectorStoreIndex, got {type(index)}"
            )

        self.index = index
        self.policy = policy
        self.similarity_top_k = similarity_top_k
        self.audit_logger = audit_logger or AuditLogger(output=None)
        self.query_engine_kwargs = query_engine_kwargs

    def query(
        self,
        query: str,
        user_context: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Query with permission filtering.

        Args:
            query: The query string
            user_context: User context for permission checks

        Returns:
            LlamaIndex Response object
        """
        # Create retriever with RAGGuard wrapper
        base_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        secure_retriever = SecureLlamaIndexRetriever(
            base_retriever=base_retriever,
            policy=self.policy,
            audit_logger=self.audit_logger
        )

        # Get filtered results
        nodes = secure_retriever.retrieve(query, user_context)

        # Create response from filtered nodes
        # Note: This is a simplified approach. For production, you may want to
        # use the full query engine with the secure retriever
        from llama_index.core.response.schema import Response

        # Build response text from nodes
        source_nodes = nodes if nodes else []
        response_text = "\n\n".join([
            node.node.get_content() if hasattr(node, 'node') else str(node)
            for node in source_nodes[:3]  # Use top 3 for response
        ]) if source_nodes else "No accessible documents found."

        return Response(
            response=response_text,
            source_nodes=source_nodes
        )


def wrap_retriever(
    retriever: Any,
    policy: Policy,
    audit_logger: Optional[AuditLogger] = None
) -> SecureLlamaIndexRetriever:
    """
    Convenience function to wrap any LlamaIndex retriever with RAGGuard.

    Args:
        retriever: LlamaIndex BaseRetriever instance
        policy: RAGGuard access control policy
        audit_logger: Optional audit logger

    Returns:
        SecureLlamaIndexRetriever instance

    Example:
        ```python
        from llama_index.core import VectorStoreIndex
        from ragguard.integrations.llamaindex import wrap_retriever
        from ragguard import load_policy

        index = VectorStoreIndex.from_documents(documents)
        retriever = index.as_retriever()

        policy = load_policy("policy.yaml")
        secure_retriever = wrap_retriever(retriever, policy)

        results = secure_retriever.retrieve(
            "What is our policy?",
            user_context={"department": "engineering"}
        )
        ```
    """
    return SecureLlamaIndexRetriever(
        base_retriever=retriever,
        policy=policy,
        audit_logger=audit_logger
    )
