"""
Agent2Agent (A2A) Protocol integration for RAGGuard.

Provides an A2A-compliant agent server that exposes permission-aware retrieval
as agent skills, enabling secure document search in multi-agent workflows.

A2A is an open protocol by Google for agent-to-agent communication.
See: https://a2a-protocol.org

Usage:
    ```python
    from ragguard.integrations.a2a import create_a2a_server
    from ragguard.retrievers import QdrantSecureRetriever

    # Create your RAGGuard retriever
    retriever = QdrantSecureRetriever(
        client=qdrant_client,
        collection="documents",
        policy=policy,
        embed_fn=embed_fn
    )

    # Create A2A server
    server = create_a2a_server(
        retriever=retriever,
        name="RAGGuard Retriever",
        description="Permission-aware document retrieval"
    )

    # Run the server
    import asyncio
    asyncio.run(server.run(host="0.0.0.0", port=9999))
    ```

Requirements:
    pip install "a2a-sdk[http-server]"
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if A2A SDK is available
A2A_AVAILABLE = False
STARLETTE_AVAILABLE = False

try:
    from a2a.server.agent_execution import AgentExecutor as _AgentExecutor
    from a2a.server.agent_execution import RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentSkill,
    )
    from a2a.utils import new_agent_text_message

    # HTTP server support (optional)
    try:
        from a2a.server.apps import A2AStarletteApplication
        STARLETTE_AVAILABLE = True
    except ImportError:
        A2AStarletteApplication = None  # type: ignore

    A2A_AVAILABLE = True
except ImportError:
    _AgentExecutor = object  # Use object as base class when SDK not available
    RequestContext = None  # type: ignore
    EventQueue = None  # type: ignore
    AgentCard = None  # type: ignore
    AgentSkill = None  # type: ignore
    AgentCapabilities = None  # type: ignore
    DefaultRequestHandler = None  # type: ignore
    InMemoryTaskStore = None  # type: ignore
    A2AStarletteApplication = None  # type: ignore
    new_agent_text_message = None  # type: ignore


def _check_a2a_available():
    """Check if A2A SDK is installed."""
    if not A2A_AVAILABLE:
        raise ImportError(
            "A2A integration requires the a2a-sdk package. "
            "Install with: pip install 'a2a-sdk[http-server]'"
        )


class RAGGuardAgentExecutor(_AgentExecutor):
    """
    A2A AgentExecutor that performs permission-aware document retrieval.

    This executor handles incoming A2A tasks by:
    1. Extracting the query and user context from the message
    2. Performing permission-aware search using RAGGuard
    3. Returning results as agent messages
    """

    def __init__(
        self,
        retriever: Any,
        text_key: str = "text",
        include_score: bool = True,
        max_results: int = 10,
        default_user: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RAGGuard agent executor.

        Args:
            retriever: Any RAGGuard SecureRetriever instance
            text_key: Key in document metadata containing the text content
            include_score: Whether to include relevance scores in results
            max_results: Default maximum results to return
            default_user: Default user context if not provided in request.
                         For production, user context should come from
                         authentication/authorization layer.
        """
        self.retriever = retriever
        self.text_key = text_key
        self.include_score = include_score
        self.max_results = max_results
        self.default_user = default_user or {"id": "anonymous", "roles": []}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute a retrieval task.

        Args:
            context: Request context containing the message and metadata
            event_queue: Queue to send response events
        """
        try:
            # Extract query from the message
            query, user_context, limit = self._parse_request(context)

            if not query:
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        "Error: No query provided. Please send a search query."
                    )
                )
                return

            # Perform permission-aware search
            results = self.retriever.search(
                query=query,
                user=user_context,
                limit=limit
            )

            # Format and return results
            response = self._format_response(query, results, user_context)
            await event_queue.enqueue_event(new_agent_text_message(response))

        except Exception as e:
            logger.error(f"RAGGuard A2A execution error: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error performing search: {e!s}")
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Handle task cancellation.

        Args:
            context: Request context
            event_queue: Queue to send response events
        """
        await event_queue.enqueue_event(
            new_agent_text_message("Search task cancelled.")
        )

    def _parse_request(
        self,
        context: RequestContext
    ) -> tuple[str, Dict[str, Any], int]:
        """
        Parse the request to extract query, user context, and limit.

        The request can be:
        1. Plain text query (uses default user)
        2. JSON with query, user_id, user_roles, user_attributes, limit

        Args:
            context: Request context

        Returns:
            Tuple of (query, user_context, limit)
        """
        message = context.message
        query = ""
        user_context = dict(self.default_user)
        limit = self.max_results

        # Extract text from message parts
        if message and message.parts:
            for part in message.parts:
                if hasattr(part, 'text'):
                    text = part.text.strip()

                    # Try to parse as JSON
                    if text.startswith('{'):
                        try:
                            data = json.loads(text)
                            query = data.get('query', '')
                            limit = data.get('limit', self.max_results)

                            # Extract user context
                            if 'user_id' in data:
                                user_context['id'] = data['user_id']
                            if 'user_roles' in data:
                                user_context['roles'] = data['user_roles']
                            if 'user_attributes' in data:
                                user_context.update(data['user_attributes'])

                        except json.JSONDecodeError:
                            # Not JSON, treat as plain query
                            query = text
                    else:
                        query = text
                    break

                elif hasattr(part, 'data'):
                    # DataPart with structured data
                    try:
                        data = part.data if isinstance(part.data, dict) else json.loads(part.data)
                        query = data.get('query', '')
                        limit = data.get('limit', self.max_results)

                        if 'user_id' in data:
                            user_context['id'] = data['user_id']
                        if 'user_roles' in data:
                            user_context['roles'] = data['user_roles']
                        if 'user_attributes' in data:
                            user_context.update(data['user_attributes'])
                    except (json.JSONDecodeError, TypeError):
                        pass

        return query, user_context, limit

    def _format_response(
        self,
        query: str,
        results: List[Any],
        user_context: Dict[str, Any]
    ) -> str:
        """
        Format search results as a response string.

        Args:
            query: Original search query
            results: Search results from retriever
            user_context: User context used for search

        Returns:
            Formatted response string
        """
        formatted_results = []

        for result in results:
            # Handle different result formats
            if hasattr(result, 'payload'):
                # Qdrant ScoredPoint
                metadata = dict(result.payload) if result.payload else {}
                score = getattr(result, 'score', None)
            elif isinstance(result, dict):
                metadata = dict(result)
                score = metadata.pop('score', metadata.pop('_score', None))
            elif hasattr(result, 'metadata'):
                metadata = dict(result.metadata) if result.metadata else {}
                score = getattr(result, 'score', None)
            else:
                metadata = {"raw": str(result)}
                score = None

            # Extract text content
            text = ""
            for key in [self.text_key, "text", "content", "page_content", "body"]:
                if key in metadata:
                    text = str(metadata.pop(key, ""))
                    break

            formatted_result = {
                "content": text[:500] + "..." if len(text) > 500 else text,
                "metadata": metadata
            }

            if self.include_score and score is not None:
                formatted_result["score"] = round(score, 4)

            formatted_results.append(formatted_result)

        response = {
            "query": query,
            "user_id": user_context.get("id"),
            "total_results": len(results),
            "results": formatted_results
        }

        return json.dumps(response, indent=2, default=str)


class RAGGuardA2AServer:
    """
    A2A Server that exposes RAGGuard's permission-aware retrieval.

    This server provides:
    - Skill: secure_search - Search documents with permission filtering
    - Streaming support for real-time results
    - Authentication-aware extended agent cards

    Example:
        ```python
        from ragguard.integrations.a2a import RAGGuardA2AServer

        server = RAGGuardA2AServer(
            retriever=my_retriever,
            name="Document Search Agent"
        )

        # Run with Starlette/Uvicorn
        await server.run(host="0.0.0.0", port=9999)
        ```
    """

    def __init__(
        self,
        retriever: Any,
        name: str = "RAGGuard Retriever Agent",
        description: str = "Permission-aware document retrieval powered by RAGGuard",
        version: str = "1.0.0",
        url: str = "http://localhost:9999/",
        text_key: str = "text",
        include_score: bool = True,
        max_results: int = 10,
        default_user: Optional[Dict[str, Any]] = None,
        enable_streaming: bool = True,
        supports_extended_card: bool = False,
    ):
        """
        Initialize the RAGGuard A2A Server.

        Args:
            retriever: Any RAGGuard SecureRetriever instance
            name: Agent name for A2A identification
            description: Agent description
            version: Agent version
            url: Base URL where the agent is accessible
            text_key: Key in document metadata containing the text content
            include_score: Whether to include relevance scores in results
            max_results: Default maximum results to return
            default_user: Default user context if not provided in request
            enable_streaming: Whether to enable streaming responses
            supports_extended_card: Whether to support extended agent cards
                                   for authenticated users
        """
        _check_a2a_available()

        if not STARLETTE_AVAILABLE:
            raise ImportError(
                "A2A HTTP server requires Starlette. "
                "Install with: pip install 'a2a-sdk[http-server]'"
            )

        self.retriever = retriever
        self.name = name
        self.description = description
        self.version = version
        self.url = url
        self.text_key = text_key
        self.include_score = include_score
        self.max_results = max_results
        self.default_user = default_user
        self.enable_streaming = enable_streaming
        self.supports_extended_card = supports_extended_card

        # Create agent card and executor
        self._agent_card = self._create_agent_card()
        self._executor = RAGGuardAgentExecutor(
            retriever=retriever,
            text_key=text_key,
            include_score=include_score,
            max_results=max_results,
            default_user=default_user,
        )
        self._app = self._create_app()

    def _create_agent_card(self) -> AgentCard:
        """Create the A2A Agent Card."""
        # Get backend info from retriever
        backend_name = getattr(self.retriever, 'backend_name', 'unknown')
        collection_name = getattr(self.retriever, 'collection', 'documents')

        # Define the secure search skill
        search_skill = AgentSkill(
            id="secure_search",
            name="Permission-Aware Document Search",
            description=(
                f"Search documents in the '{collection_name}' collection with "
                f"automatic permission filtering based on user context. "
                f"Only returns documents the user has access to. "
                f"Backend: {backend_name}"
            ),
            tags=["search", "retrieval", "rag", "permissions", "documents"],
            examples=[
                "Search for quarterly reports",
                '{"query": "machine learning", "user_id": "alice", "limit": 5}',
                "Find documents about security policies",
            ],
        )

        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version,
            protocol_version="0.3",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(
                streaming=self.enable_streaming,
                push_notifications=False,
            ),
            skills=[search_skill],
            supports_authenticated_extended_card=self.supports_extended_card,
        )

    def _create_app(self) -> A2AStarletteApplication:
        """Create the Starlette application."""
        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=self._executor,
            task_store=task_store,
        )

        return A2AStarletteApplication(
            agent_card=self._agent_card,
            http_handler=request_handler,
        )

    @property
    def agent_card(self) -> AgentCard:
        """Get the agent card."""
        return self._agent_card

    @property
    def app(self) -> A2AStarletteApplication:
        """Get the Starlette application for custom deployment."""
        return self._app

    async def run(
        self,
        host: str = "0.0.0.0",  # nosec B104 - intentional for server binding
        port: int = 9999,
        log_level: str = "info",
    ) -> None:
        """
        Run the A2A server.

        Args:
            host: Host to bind to
            port: Port to listen on
            log_level: Uvicorn log level
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "Running the A2A server requires uvicorn. "
                "Install with: pip install uvicorn"
            )

        # Update URL in agent card
        self._agent_card = AgentCard(
            name=self._agent_card.name,
            description=self._agent_card.description,
            url=f"http://{host}:{port}/",
            version=self._agent_card.version,
            protocol_version=self._agent_card.protocol_version,
            default_input_modes=self._agent_card.default_input_modes,
            default_output_modes=self._agent_card.default_output_modes,
            capabilities=self._agent_card.capabilities,
            skills=self._agent_card.skills,
            supports_authenticated_extended_card=self._agent_card.supports_authenticated_extended_card,
        )

        # Recreate app with updated card
        self._app = self._create_app()

        config = uvicorn.Config(
            app=self._app.build(),
            host=host,
            port=port,
            log_level=log_level,
        )
        server = uvicorn.Server(config)
        await server.serve()


def create_a2a_server(
    retriever: Any,
    name: str = "RAGGuard Retriever Agent",
    description: str = "Permission-aware document retrieval powered by RAGGuard",
    **kwargs
) -> RAGGuardA2AServer:
    """
    Create an A2A server for RAGGuard retrieval.

    This is a convenience function to quickly create an A2A-compliant server
    that exposes permission-aware retrieval capabilities to other agents.

    Args:
        retriever: Any RAGGuard SecureRetriever instance
        name: Agent name
        description: Agent description
        **kwargs: Additional arguments passed to RAGGuardA2AServer

    Returns:
        RAGGuardA2AServer instance

    Example:
        ```python
        from ragguard.integrations.a2a import create_a2a_server
        from ragguard.retrievers import QdrantSecureRetriever

        retriever = QdrantSecureRetriever(...)
        server = create_a2a_server(retriever, name="docs-agent")

        # Run server
        import asyncio
        asyncio.run(server.run(port=9999))
        ```
    """
    return RAGGuardA2AServer(
        retriever=retriever,
        name=name,
        description=description,
        **kwargs
    )


def create_a2a_executor(
    retriever: Any,
    **kwargs
) -> RAGGuardAgentExecutor:
    """
    Create just the A2A executor for custom server setups.

    Use this when you want to integrate RAGGuard into an existing
    A2A server setup rather than using the full RAGGuardA2AServer.

    Args:
        retriever: Any RAGGuard SecureRetriever instance
        **kwargs: Additional arguments passed to RAGGuardAgentExecutor

    Returns:
        RAGGuardAgentExecutor instance

    Example:
        ```python
        from ragguard.integrations.a2a import create_a2a_executor
        from a2a.server.apps import A2AStarletteApplication

        executor = create_a2a_executor(retriever)

        # Use with custom A2A setup
        app = A2AStarletteApplication(
            agent_card=my_card,
            http_handler=DefaultRequestHandler(agent_executor=executor, ...),
        )
        ```
    """
    _check_a2a_available()
    return RAGGuardAgentExecutor(retriever=retriever, **kwargs)


__all__ = [
    "RAGGuardA2AServer",
    "RAGGuardAgentExecutor",
    "create_a2a_server",
    "create_a2a_executor",
]
