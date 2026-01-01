"""
CrewAI integration for RAGGuard.

Provides secure retriever tools for CrewAI agent workflows with permission-aware search.

CrewAI is a popular framework for orchestrating role-playing, autonomous AI agents.
This integration allows CrewAI agents to search documents while respecting
access control policies.

Example:
    ```python
    from crewai import Agent, Task, Crew
    from ragguard.integrations.crewai import SecureRetrieverTool
    from ragguard import Policy

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "dept-access", "allow": {"conditions": ["user.department == document.department"]}}],
        "default": "deny"
    })

    # Create secure retriever tool
    search_tool = SecureRetrieverTool(
        retriever=qdrant_retriever,  # Any RAGGuard retriever
        name="search_documents",
        description="Search company documents with permission filtering"
    )

    # Create agent with the tool
    researcher = Agent(
        role="Research Assistant",
        goal="Find relevant information from company documents",
        tools=[search_tool],
        verbose=True
    )

    # Set user context for the search
    search_tool.set_user({"id": "alice", "department": "engineering"})

    # Create and run crew
    task = Task(
        description="Find all documents about the new product launch",
        agent=researcher
    )

    crew = Crew(agents=[researcher], tasks=[task])
    result = crew.kickoff()
    ```

Requirements:
    - crewai>=0.28.0
"""

from typing import Any, Callable, Dict, List, Optional, Type

# Check if CrewAI is available
try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    BaseTool = object  # type: ignore
    BaseModel = object  # type: ignore

    def Field(*args, **kwargs):  # type: ignore
        """Stub for pydantic Field when crewai not installed."""
        return None


def _check_crewai_available():
    """Check if CrewAI is installed."""
    if not CREWAI_AVAILABLE:
        raise ImportError(
            "CrewAI integration requires crewai. "
            "Install with: pip install crewai"
        )


from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..policy.models import Policy


class SecureSearchInput(BaseModel):
    """Input schema for secure search tool."""
    query: str = Field(description="The search query to find relevant documents")
    limit: int = Field(default=5, description="Maximum number of results to return")


class SecureRetrieverTool(BaseTool):
    """
    CrewAI tool for permission-aware document retrieval.

    This tool wraps any RAGGuard retriever and enforces access control
    policies when CrewAI agents search for documents.

    The tool requires a user context to be set before use. The user context
    determines what documents the agent can access based on the policy.

    Attributes:
        name: Tool name shown to agents
        description: Tool description for agent reasoning
        retriever: RAGGuard secure retriever instance
        current_user: Current user context for permission filtering

    Example:
        ```python
        from ragguard.integrations.crewai import SecureRetrieverTool
        from ragguard.retrievers import QdrantSecureRetriever

        # Create RAGGuard retriever
        retriever = QdrantSecureRetriever(
            client=qdrant_client,
            collection="documents",
            policy=policy
        )

        # Create CrewAI tool
        tool = SecureRetrieverTool(
            retriever=retriever,
            name="search_docs",
            description="Search internal documents"
        )

        # Set user context (required before use)
        tool.set_user({"id": "alice", "roles": ["engineer"]})

        # Use in CrewAI agent
        agent = Agent(
            role="Researcher",
            tools=[tool]
        )
        ```
    """
    name: str = "secure_document_search"
    description: str = (
        "Search for documents with permission filtering. "
        "Returns only documents the current user is authorized to access. "
        "Input should be a search query string."
    )
    args_schema: Type[BaseModel] = SecureSearchInput

    # RAGGuard retriever - stored as private attribute
    _retriever: Any = None
    _current_user: Optional[Dict[str, Any]] = None
    _embed_fn: Optional[Callable[[str], List[float]]] = None

    def __init__(
        self,
        retriever: Any = None,
        name: str = "secure_document_search",
        description: str = None,
        qdrant_client: Any = None,
        collection: str = None,
        policy: Policy = None,
        embedding_function: Callable[[str], List[float]] = None,
        audit_logger: Optional[AuditLogger] = None,
        **kwargs
    ):
        """
        Initialize secure retriever tool for CrewAI.

        You can either provide an existing RAGGuard retriever or create one
        by providing Qdrant client parameters.

        Args:
            retriever: Existing RAGGuard secure retriever (preferred)
            name: Tool name for agents
            description: Tool description for agent reasoning
            qdrant_client: QdrantClient instance (if not providing retriever)
            collection: Collection name (if not providing retriever)
            policy: Access control policy (if not providing retriever)
            embedding_function: Text to vector function (if not providing retriever)
            audit_logger: Optional audit logger
            **kwargs: Additional arguments
        """
        _check_crewai_available()

        # Set description
        if description:
            kwargs['description'] = description
        else:
            kwargs['description'] = self.description

        kwargs['name'] = name

        super().__init__(**kwargs)

        if retriever is not None:
            self._retriever = retriever
        elif qdrant_client and collection and policy and embedding_function:
            from ..retrievers import QdrantSecureRetriever
            self._retriever = QdrantSecureRetriever(
                client=qdrant_client,
                collection=collection,
                policy=policy,
                audit_logger=audit_logger,
                embed_fn=embedding_function
            )
            self._embed_fn = embedding_function
        else:
            raise ValueError(
                "Must provide either 'retriever' or "
                "(qdrant_client, collection, policy, embedding_function)"
            )

    def set_user(self, user: Dict[str, Any]) -> "SecureRetrieverTool":
        """
        Set the current user context for permission filtering.

        This must be called before the tool is used by an agent.
        The user context determines what documents are accessible.

        Args:
            user: User context dictionary with id, roles, department, etc.

        Returns:
            Self for method chaining

        Example:
            ```python
            tool.set_user({
                "id": "alice@company.com",
                "roles": ["engineer", "team-lead"],
                "department": "engineering"
            })
            ```
        """
        self._current_user = user
        return self

    def get_user(self) -> Optional[Dict[str, Any]]:
        """Get the current user context."""
        return self._current_user

    def _run(self, query: str, limit: int = 5) -> str:
        """
        Execute the search with permission filtering.

        This method is called by CrewAI when an agent uses the tool.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            Formatted string with search results

        Raises:
            RetrieverError: If user context not set or search fails
        """
        if self._current_user is None:
            raise RetrieverError(
                "User context not set. Call set_user() before using this tool. "
                "Example: tool.set_user({'id': 'user@company.com', 'roles': ['employee']})"
            )

        try:
            results = self._retriever.search(
                query=query,
                user=self._current_user,
                limit=limit
            )

            if not results:
                return "No documents found matching your query that you have access to."

            # Format results for agent consumption
            formatted = []
            for i, result in enumerate(results, 1):
                # Extract content from various result formats
                if hasattr(result, 'payload'):
                    content = result.payload.get('text') or result.payload.get('content') or str(result.payload)
                    metadata = {k: v for k, v in result.payload.items() if k not in ['text', 'content']}
                    score = getattr(result, 'score', 0.0)
                elif isinstance(result, dict):
                    content = result.get('text') or result.get('content') or result.get('metadata', {}).get('text', '')
                    metadata = result.get('metadata', {})
                    score = result.get('score', 0.0)
                else:
                    content = str(result)
                    metadata = {}
                    score = 0.0

                formatted.append(
                    f"[{i}] (relevance: {score:.2f})\n"
                    f"Content: {content[:500]}{'...' if len(str(content)) > 500 else ''}\n"
                    f"Metadata: {metadata}"
                )

            return f"Found {len(results)} document(s):\n\n" + "\n\n".join(formatted)

        except Exception as e:
            raise RetrieverError(f"Search failed: {e}")


class SecureRAGTool(SecureRetrieverTool):
    """
    Alias for SecureRetrieverTool for clearer naming in RAG contexts.

    Use this when building RAG (Retrieval-Augmented Generation) pipelines
    with CrewAI agents.
    """
    name: str = "secure_rag_search"
    description: str = (
        "Search the knowledge base for relevant information. "
        "Returns documents the user is authorized to access. "
        "Use this to find facts, policies, or reference material."
    )


def create_secure_retriever_tool(
    retriever: Any,
    name: str = "search_documents",
    description: str = "Search for documents with permission filtering",
    user: Optional[Dict[str, Any]] = None
) -> SecureRetrieverTool:
    """
    Create a secure retriever tool for CrewAI from an existing retriever.

    This is a convenience function for quickly creating tools.

    Args:
        retriever: Any RAGGuard secure retriever
        name: Tool name
        description: Tool description
        user: Optional user context to pre-set

    Returns:
        Configured SecureRetrieverTool

    Example:
        ```python
        from ragguard.integrations.crewai import create_secure_retriever_tool

        tool = create_secure_retriever_tool(
            retriever=qdrant_retriever,
            name="search_docs",
            description="Search internal documentation",
            user={"id": "alice", "roles": ["engineer"]}
        )
        ```
    """
    tool = SecureRetrieverTool(
        retriever=retriever,
        name=name,
        description=description
    )

    if user:
        tool.set_user(user)

    return tool


__all__ = [
    "SecureRetrieverTool",
    "SecureRAGTool",
    "SecureSearchInput",
    "create_secure_retriever_tool",
]
