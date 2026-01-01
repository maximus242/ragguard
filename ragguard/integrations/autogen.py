"""
Microsoft AutoGen integration for RAGGuard.

Provides secure retriever tools for AutoGen multi-agent workflows with
permission-aware document search.

AutoGen is Microsoft's framework for building multi-agent AI applications.
This integration allows AutoGen agents to search documents while respecting
access control policies.

Supports both AutoGen v0.2 (legacy) and AutoGen v0.4+ (current).

Example (AutoGen v0.4+):
    ```python
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from ragguard.integrations.autogen import create_secure_search_tool
    from ragguard import QdrantSecureRetriever, Policy

    # Create RAGGuard retriever
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}],
        "default": "deny"
    })

    retriever = QdrantSecureRetriever(
        client=qdrant_client,
        collection="documents",
        policy=policy,
        embed_fn=embed_function
    )

    # Create secure search tool
    search_tool = create_secure_search_tool(
        retriever=retriever,
        user={"id": "alice", "department": "engineering"}
    )

    # Create agent with the tool
    model_client = OpenAIChatCompletionClient(model="gpt-4")
    assistant = AssistantAgent(
        name="researcher",
        model_client=model_client,
        tools=[search_tool]
    )
    ```

Example (AutoGen v0.2 - Legacy):
    ```python
    from autogen import AssistantAgent, UserProxyAgent
    from ragguard.integrations.autogen import SecureRetrieverFunction

    # Create function for agents
    search_fn = SecureRetrieverFunction(
        retriever=retriever,
        user={"id": "alice", "department": "engineering"}
    )

    # Register with agent
    assistant = AssistantAgent(
        name="researcher",
        llm_config={"functions": [search_fn.function_schema]}
    )
    assistant.register_function(
        function_map={"search_documents": search_fn}
    )
    ```

Requirements:
    - autogen-agentchat>=0.4.0 (recommended)
    - OR autogen>=0.2.0 (legacy)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for AutoGen availability (try v0.4+ first, then v0.2)
AUTOGEN_V4_AVAILABLE = False
AUTOGEN_V2_AVAILABLE = False

try:
    from autogen_core import FunctionCall  # noqa: F401
    from autogen_core.tools import FunctionTool
    AUTOGEN_V4_AVAILABLE = True
except ImportError:
    pass

try:
    import autogen  # noqa: F401
    AUTOGEN_V2_AVAILABLE = True
except ImportError:
    pass

AUTOGEN_AVAILABLE = AUTOGEN_V4_AVAILABLE or AUTOGEN_V2_AVAILABLE


def _check_autogen_available():
    """Check if AutoGen is installed."""
    if not AUTOGEN_AVAILABLE:
        raise ImportError(
            "AutoGen integration requires autogen. "
            "Install with: pip install autogen-agentchat autogen-ext[openai] "
            "or for legacy: pip install autogen"
        )


class SecureSearchTool:
    """
    Secure search tool for AutoGen v0.4+ agents.

    This class provides a callable tool that AutoGen agents can use
    to search documents with permission filtering.

    Attributes:
        retriever: RAGGuard secure retriever instance
        user: User context for permission filtering
        max_results: Maximum results per search
        name: Tool name
        description: Tool description

    Example:
        ```python
        tool = SecureSearchTool(
            retriever=qdrant_retriever,
            user={"id": "alice", "department": "engineering"},
            max_results=5
        )

        # Use in AutoGen agent
        from autogen_agentchat.agents import AssistantAgent

        agent = AssistantAgent(
            name="researcher",
            model_client=model_client,
            tools=[tool.as_function_tool()]
        )
        ```
    """

    def __init__(
        self,
        retriever: Any,
        user: Dict[str, Any],
        max_results: int = 10,
        name: str = "search_documents",
        description: str = "Search documents with access control filtering"
    ):
        """
        Initialize secure search tool.

        Args:
            retriever: RAGGuard secure retriever instance
            user: User context for permission filtering
            max_results: Maximum number of results
            name: Tool name
            description: Tool description for agent reasoning
        """
        self.retriever = retriever
        self.user = user
        self.max_results = max_results
        self.name = name
        self.description = description

    def __call__(self, query: str, limit: Optional[int] = None) -> str:
        """
        Execute search with permission filtering.

        Args:
            query: Search query string
            limit: Maximum results (uses max_results if not provided)

        Returns:
            Formatted string with search results
        """
        result_limit = limit or self.max_results

        try:
            results = self.retriever.search(
                query=query,
                user=self.user,
                limit=result_limit
            )

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {e!s}"

    def _format_results(self, results: List[Any]) -> str:
        """Format search results for agent consumption."""
        if not results:
            return "No documents found matching your query that you have access to."

        formatted = []
        for i, result in enumerate(results, 1):
            if hasattr(result, 'payload'):
                content = result.payload.get('text') or result.payload.get('content', '')
                score = getattr(result, 'score', 0.0)
            elif isinstance(result, dict):
                content = result.get('text') or result.get('content', '')
                score = result.get('score', 0.0)
            else:
                content = str(result)
                score = 0.0

            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."

            formatted.append(f"[{i}] (score: {score:.3f}) {content}")

        return f"Found {len(results)} document(s):\n\n" + "\n\n".join(formatted)

    def set_user(self, user: Dict[str, Any]) -> "SecureSearchTool":
        """
        Update the user context.

        Args:
            user: New user context

        Returns:
            Self for method chaining
        """
        self.user = user
        return self

    def as_function_tool(self) -> Any:
        """
        Convert to AutoGen v0.4 FunctionTool.

        Returns:
            FunctionTool instance for use with AutoGen agents

        Raises:
            ImportError: If AutoGen v0.4+ is not installed
        """
        if not AUTOGEN_V4_AVAILABLE:
            raise ImportError(
                "AutoGen v0.4+ required for FunctionTool. "
                "Install with: pip install autogen-agentchat autogen-core"
            )

        return FunctionTool(
            func=self.__call__,
            name=self.name,
            description=self.description
        )


class SecureRetrieverFunction:
    """
    Secure retriever function for AutoGen v0.2 (legacy).

    This class provides a callable function that can be registered
    with AutoGen v0.2 agents for document retrieval.

    Example:
        ```python
        search_fn = SecureRetrieverFunction(
            retriever=qdrant_retriever,
            user={"id": "alice", "roles": ["engineer"]}
        )

        # Register with AssistantAgent
        assistant = AssistantAgent(
            name="researcher",
            llm_config={
                "functions": [search_fn.function_schema],
                "config_list": config_list
            }
        )
        assistant.register_function(
            function_map={"search_documents": search_fn}
        )
        ```
    """

    def __init__(
        self,
        retriever: Any,
        user: Dict[str, Any],
        max_results: int = 10,
        name: str = "search_documents",
        description: str = "Search documents with access control. Returns only documents the user can access."
    ):
        """
        Initialize secure retriever function.

        Args:
            retriever: RAGGuard secure retriever instance
            user: User context for permission filtering
            max_results: Maximum number of results
            name: Function name
            description: Function description
        """
        self.retriever = retriever
        self.user = user
        self.max_results = max_results
        self.name = name
        self.description = description

    def __call__(self, query: str, limit: int = None) -> str:
        """Execute the search."""
        result_limit = limit or self.max_results

        try:
            results = self.retriever.search(
                query=query,
                user=self.user,
                limit=result_limit
            )

            return self._format_results(results)

        except Exception as e:
            logger.error(f"AutoGen RAGGuardToolExecutor search failed: {e}")
            return f"Search error: {e!s}"

    def _format_results(self, results: List[Any]) -> str:
        """Format results for agent."""
        if not results:
            return "No accessible documents found."

        formatted = []
        for i, result in enumerate(results, 1):
            if hasattr(result, 'payload'):
                content = result.payload.get('text') or result.payload.get('content', '')
                score = getattr(result, 'score', 0.0)
            elif isinstance(result, dict):
                content = result.get('text') or result.get('content', '')
                score = result.get('score', 0.0)
            else:
                content = str(result)
                score = 0.0

            if len(content) > 500:
                content = content[:500] + "..."

            formatted.append(f"[{i}] (relevance: {score:.2f})\n{content}")

        return "\n\n".join(formatted)

    def set_user(self, user: Dict[str, Any]) -> "SecureRetrieverFunction":
        """Update user context."""
        self.user = user
        return self

    @property
    def function_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI function schema for AutoGen v0.2.

        Returns:
            Function schema dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": self.max_results
                    }
                },
                "required": ["query"]
            }
        }


def create_secure_search_tool(
    retriever: Any,
    user: Dict[str, Any],
    name: str = "search_documents",
    description: str = "Search documents with access control filtering",
    max_results: int = 10
) -> Any:
    """
    Create a secure search tool for AutoGen agents.

    This factory function creates the appropriate tool type based on
    the installed AutoGen version.

    Args:
        retriever: RAGGuard secure retriever instance
        user: User context for permission filtering
        name: Tool/function name
        description: Tool description
        max_results: Maximum results per search

    Returns:
        For AutoGen v0.4+: FunctionTool instance
        For AutoGen v0.2: SecureRetrieverFunction instance

    Example:
        ```python
        tool = create_secure_search_tool(
            retriever=qdrant_retriever,
            user={"id": "alice", "department": "engineering"}
        )

        # Use with agent (works with both v0.2 and v0.4)
        agent = AssistantAgent(
            name="researcher",
            tools=[tool]  # v0.4
            # OR llm_config={"functions": [tool.function_schema]}  # v0.2
        )
        ```
    """
    _check_autogen_available()

    tool = SecureSearchTool(
        retriever=retriever,
        user=user,
        name=name,
        description=description,
        max_results=max_results
    )

    if AUTOGEN_V4_AVAILABLE:
        return tool.as_function_tool()
    else:
        return SecureRetrieverFunction(
            retriever=retriever,
            user=user,
            name=name,
            description=description,
            max_results=max_results
        )


class SecureRAGAgent:
    """
    Pre-configured AutoGen agent with secure document retrieval.

    This is a convenience class that creates an AutoGen agent
    pre-configured with a secure search tool.

    Example:
        ```python
        rag_agent = SecureRAGAgent(
            name="researcher",
            retriever=qdrant_retriever,
            user={"id": "alice", "department": "engineering"},
            model="gpt-4",
            system_message="You are a research assistant."
        )

        # Get the configured agent
        agent = rag_agent.agent

        # Update user context for different users
        rag_agent.set_user({"id": "bob", "department": "sales"})
        ```
    """

    def __init__(
        self,
        name: str,
        retriever: Any,
        user: Dict[str, Any],
        model: str = "gpt-4",
        system_message: str = "You are a helpful assistant with access to a secure document search tool.",
        api_key: Optional[str] = None,
        **agent_kwargs
    ):
        """
        Initialize secure RAG agent.

        Args:
            name: Agent name
            retriever: RAGGuard secure retriever
            user: User context for permission filtering
            model: Model name (default: gpt-4)
            system_message: System message for the agent
            api_key: OpenAI API key (optional, uses env var if not provided)
            **agent_kwargs: Additional arguments for the agent
        """
        _check_autogen_available()

        self.retriever = retriever
        self._user = user
        self.search_tool = SecureSearchTool(
            retriever=retriever,
            user=user,
            name="search_documents",
            description="Search internal documents. Returns only documents you have access to."
        )

        if AUTOGEN_V4_AVAILABLE:
            self._init_v4_agent(name, model, system_message, api_key, **agent_kwargs)
        else:
            self._init_v2_agent(name, model, system_message, api_key, **agent_kwargs)

    def _init_v4_agent(self, name, model, system_message, api_key, **kwargs):
        """Initialize AutoGen v0.4+ agent."""
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            model_client = OpenAIChatCompletionClient(
                model=model,
                api_key=api_key
            ) if api_key else OpenAIChatCompletionClient(model=model)

            self.agent = AssistantAgent(
                name=name,
                model_client=model_client,
                tools=[self.search_tool.as_function_tool()],
                system_message=system_message,
                **kwargs
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to initialize AutoGen v0.4 agent: {e}. "
                "Install with: pip install autogen-agentchat autogen-ext[openai]"
            )

    def _init_v2_agent(self, name, model, system_message, api_key, **kwargs):
        """Initialize AutoGen v0.2 agent."""
        try:
            from autogen import AssistantAgent

            config_list = [{"model": model}]
            if api_key:
                config_list[0]["api_key"] = api_key

            search_fn = SecureRetrieverFunction(
                retriever=self.retriever,
                user=self._user
            )

            self.agent = AssistantAgent(
                name=name,
                system_message=system_message,
                llm_config={
                    "config_list": config_list,
                    "functions": [search_fn.function_schema]
                },
                **kwargs
            )
            self.agent.register_function(
                function_map={"search_documents": search_fn}
            )
            self._search_fn = search_fn
        except ImportError as e:
            raise ImportError(
                f"Failed to initialize AutoGen v0.2 agent: {e}. "
                "Install with: pip install autogen"
            )

    def set_user(self, user: Dict[str, Any]) -> "SecureRAGAgent":
        """
        Update the user context for permission filtering.

        Args:
            user: New user context

        Returns:
            Self for method chaining
        """
        self._user = user
        self.search_tool.set_user(user)

        if hasattr(self, '_search_fn'):
            self._search_fn.set_user(user)

        return self

    @property
    def user(self) -> Dict[str, Any]:
        """Get current user context."""
        return self._user


__all__ = [
    "SecureSearchTool",
    "SecureRetrieverFunction",
    "SecureRAGAgent",
    "create_secure_search_tool",
    "AUTOGEN_AVAILABLE",
    "AUTOGEN_V4_AVAILABLE",
    "AUTOGEN_V2_AVAILABLE",
]
