"""
LangGraph integration for RAGGuard.

Provides secure retriever nodes for LangGraph workflows with permission-aware search.
"""

from typing import Any, Callable, Dict, List, Optional, TypedDict

# Check if LangGraph is available
try:
    from langchain_core.documents import Document
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Create dummy classes for type hints
    StateGraph = object  # type: ignore
    Document = object  # type: ignore


def _check_langgraph_available():
    """Check if LangGraph is installed."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph integration requires langgraph and langchain. "
            "Install with: pip install ragguard[langchain] langgraph"
        )


from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..policy.models import Policy
from ..retrievers import QdrantSecureRetriever


class RetrieverState(TypedDict, total=False):
    """State schema for retriever node.

    Attributes:
        query: The search query string
        user: User context for permission filtering
        documents: Retrieved documents (output)
        limit: Maximum number of results to return
        metadata: Additional search metadata
    """
    query: str
    user: Dict[str, Any]
    documents: List[Document]
    limit: int
    metadata: Dict[str, Any]


class SecureRetrieverNode:
    """
    LangGraph-compatible secure retriever node.

    This node can be added to a LangGraph StateGraph to perform
    permission-aware retrieval within an agent workflow.

    Example:
        ```python
        from langgraph.graph import StateGraph
        from ragguard.integrations.langgraph import SecureRetrieverNode, RetrieverState
        from ragguard import load_policy

        # Create retriever node
        retriever_node = SecureRetrieverNode(
            qdrant_client=client,
            collection="documents",
            policy=load_policy("policy.yaml"),
            embedding_function=embeddings.embed_query
        )

        # Create workflow
        workflow = StateGraph(RetrieverState)
        workflow.add_node("retrieve", retriever_node)

        # Set up graph
        workflow.set_entry_point("retrieve")
        workflow.set_finish_point("retrieve")

        # Compile and run
        app = workflow.compile()
        result = app.invoke({
            "query": "What is our company policy?",
            "user": {"id": "alice", "roles": ["employee"]},
            "limit": 10
        })

        print(f"Found {len(result['documents'])} documents")
        ```
    """

    def __init__(
        self,
        qdrant_client: Any = None,
        collection: str = None,
        policy: Policy = None,
        embedding_function: Callable[[str], List[float]] = None,
        audit_logger: Optional[AuditLogger] = None,
        retriever: Any = None,
        **kwargs
    ):
        """
        Initialize secure retriever node.

        You can either provide Qdrant-specific parameters (qdrant_client,
        collection, etc.) or pass an existing retriever.

        Args:
            qdrant_client: QdrantClient instance (optional if retriever provided)
            collection: Collection name (optional if retriever provided)
            policy: Access control policy (optional if retriever provided)
            embedding_function: Function to convert text to embeddings (optional if retriever provided)
            audit_logger: Optional audit logger
            retriever: Existing RAGGuard retriever (alternative to qdrant_client)
            **kwargs: Additional arguments
        """
        _check_langgraph_available()

        if retriever is not None:
            # Use provided retriever
            self._retriever = retriever
        elif qdrant_client and collection and policy and embedding_function:
            # Create Qdrant retriever
            self._retriever = QdrantSecureRetriever(
                client=qdrant_client,
                collection=collection,
                policy=policy,
                audit_logger=audit_logger,
                embed_fn=embedding_function
            )
        else:
            raise ValueError(
                "Must provide either 'retriever' or (qdrant_client, collection, policy, embedding_function)"
            )

    def __call__(self, state: RetrieverState) -> RetrieverState:
        """
        Execute retrieval with permission filtering.

        This is the node function that LangGraph will call.

        Args:
            state: Current workflow state containing query and user

        Returns:
            Updated state with documents populated
        """
        query = state.get("query")
        user = state.get("user")
        limit = state.get("limit", 10)

        if not query:
            raise RetrieverError("State must contain 'query' field")

        if not user:
            raise RetrieverError(
                "State must contain 'user' field with user context for permission filtering"
            )

        # Execute permission-aware search
        results = self._retriever.search(
            query=query,
            user=user,
            limit=limit
        )

        # Convert to LangChain Documents
        documents = []
        for result in results:
            # Handle different result formats
            if hasattr(result, 'payload'):
                # Qdrant ScoredPoint
                payload = result.payload
                score = result.score
            elif hasattr(result, 'metadata'):
                # Generic format
                payload = result.metadata or {}
                score = getattr(result, 'score', 0.0)
            else:
                # Fallback
                payload = {}
                score = 0.0

            # Extract text content
            page_content = (
                payload.pop("text", None)
                or payload.pop("content", None)
                or str(payload)
            )

            # Create Document
            doc = Document(
                page_content=page_content,
                metadata={
                    **payload,
                    "score": score,
                }
            )
            documents.append(doc)

        # Update state
        return {
            **state,
            "documents": documents,
            "metadata": {
                **state.get("metadata", {}),
                "retrieved_count": len(documents)
            }
        }


class SecureRetrieverTool:
    """
    LangGraph-compatible tool for secure retrieval in agent workflows.

    This can be used as a tool that agents can call during execution.

    Example:
        ```python
        from langchain.agents import create_react_agent
        from ragguard.integrations.langgraph import SecureRetrieverTool

        # Create tool
        retriever_tool = SecureRetrieverTool(
            qdrant_client=client,
            collection="documents",
            policy=policy,
            embedding_function=embeddings.embed_query,
            name="search_documents",
            description="Search for company documents with permission filtering"
        )

        # Use in agent
        tools = [retriever_tool.as_tool()]
        agent = create_react_agent(llm, tools)
        ```
    """

    def __init__(
        self,
        qdrant_client: Any = None,
        collection: str = None,
        policy: Policy = None,
        embedding_function: Callable[[str], List[float]] = None,
        name: str = "secure_retriever",
        description: str = "Retrieve documents with permission filtering",
        retriever: Any = None,
        **kwargs
    ):
        """
        Initialize secure retriever tool.

        Args:
            qdrant_client: QdrantClient instance
            collection: Collection name
            policy: Access control policy
            embedding_function: Function to convert text to embeddings
            name: Tool name
            description: Tool description
            retriever: Existing retriever (alternative)
            **kwargs: Additional arguments
        """
        _check_langgraph_available()

        self.name = name
        self.description = description

        # Create underlying node
        self._node = SecureRetrieverNode(
            qdrant_client=qdrant_client,
            collection=collection,
            policy=policy,
            embedding_function=embedding_function,
            retriever=retriever,
            **kwargs
        )

    def run(self, query: str, user: Dict[str, Any], limit: int = 10) -> List[Document]:
        """
        Execute retrieval.

        Args:
            query: Search query
            user: User context
            limit: Maximum results

        Returns:
            List of documents
        """
        state: RetrieverState = {
            "query": query,
            "user": user,
            "limit": limit,
            "documents": [],
            "metadata": {}
        }

        result = self._node(state)
        return result["documents"]

    def as_tool(self):
        """
        Convert to LangChain Tool.

        Returns:
            LangChain Tool instance
        """
        try:
            from langchain.tools import Tool
        except ImportError:
            raise ImportError(
                "LangChain Tool requires langchain. "
                "Install with: pip install langchain"
            )

        return Tool(
            name=self.name,
            description=self.description,
            func=lambda query, user={}, limit=10: self.run(query, user, limit)
        )


# Convenience functions for creating retrievers
def create_secure_retriever_node(
    qdrant_client: Any,
    collection: str,
    policy: Policy,
    embedding_function: Callable[[str], List[float]],
    **kwargs
) -> SecureRetrieverNode:
    """
    Create a secure retriever node for LangGraph workflows.

    This is a convenience function for creating a SecureRetrieverNode.

    Args:
        qdrant_client: QdrantClient instance
        collection: Collection name
        policy: Access control policy
        embedding_function: Embedding function
        **kwargs: Additional arguments

    Returns:
        SecureRetrieverNode instance
    """
    return SecureRetrieverNode(
        qdrant_client=qdrant_client,
        collection=collection,
        policy=policy,
        embedding_function=embedding_function,
        **kwargs
    )


def create_secure_retriever_tool(
    qdrant_client: Any,
    collection: str,
    policy: Policy,
    embedding_function: Callable[[str], List[float]],
    name: str = "search_documents",
    description: str = "Search for documents with permission filtering",
    **kwargs
) -> SecureRetrieverTool:
    """
    Create a secure retriever tool for LangGraph agents.

    Args:
        qdrant_client: QdrantClient instance
        collection: Collection name
        policy: Access control policy
        embedding_function: Embedding function
        name: Tool name
        description: Tool description
        **kwargs: Additional arguments

    Returns:
        SecureRetrieverTool instance
    """
    return SecureRetrieverTool(
        qdrant_client=qdrant_client,
        collection=collection,
        policy=policy,
        embedding_function=embedding_function,
        name=name,
        description=description,
        **kwargs
    )
