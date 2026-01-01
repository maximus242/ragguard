"""
Model Context Protocol (MCP) integration for RAGGuard.

NOTE: Uses `from __future__ import annotations` to allow class definition
even when MCP SDK is not installed (defers type hint evaluation).

Provides an MCP server that exposes permission-aware retrieval as tools
and document collections as resources.

MCP is an open protocol by Anthropic for connecting AI assistants to
external data sources. See: https://modelcontextprotocol.io

Usage:
    ```python
    from ragguard.integrations.mcp import create_mcp_server
    from ragguard.retrievers import ChromaDBSecureRetriever

    # Create your RAGGuard retriever
    retriever = ChromaDBSecureRetriever(
        client=chroma_client,
        collection="documents",
        policy=policy,
        embed_fn=embed_fn
    )

    # Create MCP server
    server = create_mcp_server(
        retriever=retriever,
        name="ragguard-retriever",
        description="Permission-aware document retrieval"
    )

    # Run the server (stdio transport)
    server.run()
    ```
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if MCP SDK is available
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        GetPromptResult,
        Prompt,
        PromptArgument,
        PromptMessage,
        Resource,
        ResourceTemplate,
        TextContent,
        Tool,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None  # type: ignore
    Tool = None  # type: ignore


def _check_mcp_available():
    """Check if MCP SDK is installed."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP integration requires the mcp package. "
            "Install with: pip install mcp"
        )


class RAGGuardMCPServer:
    """
    MCP Server that exposes RAGGuard's permission-aware retrieval.

    This server provides:
    - Tools: secure_search - Search documents with permission filtering
    - Resources: Document collections accessible based on user permissions
    - Prompts: Pre-configured retrieval prompts

    Example:
        ```python
        from ragguard.integrations.mcp import RAGGuardMCPServer

        server = RAGGuardMCPServer(
            retriever=my_retriever,
            name="docs-retriever"
        )

        # Run with stdio transport
        await server.run_stdio()
        ```
    """

    def __init__(
        self,
        retriever: Any,
        name: str = "ragguard",
        description: str = "Permission-aware document retrieval powered by RAGGuard",
        version: str = "1.0.0",
        user_context_extractor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        text_key: str = "text",
        include_score: bool = True,
        max_results: int = 10,
    ):
        """
        Initialize the RAGGuard MCP Server.

        Args:
            retriever: Any RAGGuard SecureRetriever instance
            name: Server name for MCP identification
            description: Server description
            version: Server version
            user_context_extractor: Function to extract user context from MCP request metadata.
                                   If None, user context must be passed in tool arguments.
            text_key: Key in document metadata containing the text content
            include_score: Whether to include relevance scores in results
            max_results: Default maximum results to return
        """
        _check_mcp_available()

        self.retriever = retriever
        self.name = name
        self.description = description
        self.version = version
        self.user_context_extractor = user_context_extractor
        self.text_key = text_key
        self.include_score = include_score
        self.max_results = max_results

        # Create MCP server
        self._server = Server(name)
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP request handlers."""

        @self._server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="secure_search",
                    description=(
                        "Search documents with permission-aware filtering. "
                        "Only returns documents the user has access to based on their roles and attributes."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": f"Maximum number of results (default: {self.max_results})",
                                "default": self.max_results
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User ID for permission checking"
                            },
                            "user_roles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "User roles for permission checking"
                            },
                            "user_attributes": {
                                "type": "object",
                                "description": "Additional user attributes (department, clearance, etc.)"
                            }
                        },
                        "required": ["query", "user_id"]
                    }
                ),
                Tool(
                    name="check_access",
                    description=(
                        "Check if a user has access to a specific document. "
                        "Returns whether the user can access the document based on policy rules."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "The document ID to check access for"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User ID for permission checking"
                            },
                            "user_roles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "User roles for permission checking"
                            },
                            "user_attributes": {
                                "type": "object",
                                "description": "Additional user attributes"
                            }
                        },
                        "required": ["document_id", "user_id"]
                    }
                )
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            if name == "secure_search":
                return await self._handle_secure_search(arguments)
            elif name == "check_access":
                return await self._handle_check_access(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

        @self._server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            # Get collection info from retriever if available
            collection_name = getattr(self.retriever, 'collection', 'documents')
            backend_name = getattr(self.retriever, 'backend_name', 'unknown')

            return [
                Resource(
                    uri=f"ragguard://{collection_name}",
                    name=f"Document Collection: {collection_name}",
                    description=f"Permission-aware document collection stored in {backend_name}",
                    mimeType="application/json"
                )
            ]

        @self._server.list_resource_templates()
        async def list_resource_templates() -> List[ResourceTemplate]:
            """List resource templates."""
            return [
                ResourceTemplate(
                    uriTemplate="ragguard://{collection}/document/{document_id}",
                    name="Document by ID",
                    description="Access a specific document by its ID (permission checked)"
                )
            ]

        @self._server.list_prompts()
        async def list_prompts() -> List[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name="search_and_summarize",
                    description="Search for documents and summarize the results",
                    arguments=[
                        PromptArgument(
                            name="query",
                            description="The search query",
                            required=True
                        ),
                        PromptArgument(
                            name="user_id",
                            description="User ID for permission checking",
                            required=True
                        )
                    ]
                ),
                Prompt(
                    name="qa_with_sources",
                    description="Answer a question using retrieved documents as sources",
                    arguments=[
                        PromptArgument(
                            name="question",
                            description="The question to answer",
                            required=True
                        ),
                        PromptArgument(
                            name="user_id",
                            description="User ID for permission checking",
                            required=True
                        )
                    ]
                )
            ]

        @self._server.get_prompt()
        async def get_prompt(name: str, arguments: Optional[Dict[str, str]]) -> GetPromptResult:
            """Get a prompt template."""
            if name == "search_and_summarize":
                query = arguments.get("query", "") if arguments else ""
                user_id = arguments.get("user_id", "") if arguments else ""
                return GetPromptResult(
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"""Search for documents related to: "{query}"

User ID for permission checking: {user_id}

After retrieving the documents, please:
1. Summarize the key findings
2. List the main topics covered
3. Note any important details or caveats"""
                            )
                        )
                    ]
                )
            elif name == "qa_with_sources":
                question = arguments.get("question", "") if arguments else ""
                user_id = arguments.get("user_id", "") if arguments else ""
                return GetPromptResult(
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"""Answer this question using retrieved documents: "{question}"

User ID for permission checking: {user_id}

Please:
1. Search for relevant documents
2. Answer the question based on the retrieved content
3. Cite your sources with document references"""
                            )
                        )
                    ]
                )
            else:
                raise ValueError(f"Unknown prompt: {name}")

    async def _handle_secure_search(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle secure_search tool call."""
        query = arguments.get("query", "")
        limit = arguments.get("limit", self.max_results)
        user_id = arguments.get("user_id")
        user_roles = arguments.get("user_roles", [])
        user_attributes = arguments.get("user_attributes", {})

        if not user_id:
            return [TextContent(
                type="text",
                text="Error: user_id is required for permission-aware search"
            )]

        # Build user context
        user = {
            "id": user_id,
            "roles": user_roles,
            **user_attributes
        }

        try:
            # Perform permission-aware search
            results = self.retriever.search(
                query=query,
                user=user,
                limit=limit
            )

            # Format results
            formatted_results = self._format_results(results)

            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "user_id": user_id,
                    "total_results": len(results),
                    "results": formatted_results
                }, indent=2)
            )]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return [TextContent(
                type="text",
                text=f"Error performing search: {e!s}"
            )]

    async def _handle_check_access(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle check_access tool call."""
        document_id = arguments.get("document_id")
        user_id = arguments.get("user_id")
        user_roles = arguments.get("user_roles", [])
        user_attributes = arguments.get("user_attributes", {})

        if not document_id or not user_id:
            return [TextContent(
                type="text",
                text="Error: document_id and user_id are required"
            )]

        user = {
            "id": user_id,
            "roles": user_roles,
            **user_attributes
        }

        try:
            # Check if retriever has a check_access method
            if hasattr(self.retriever, 'check_access'):
                has_access = self.retriever.check_access(
                    document_id=document_id,
                    user=user
                )
            else:
                # Fallback: try to retrieve the document
                # If it returns, user has access
                results = self.retriever.search(
                    query=f"id:{document_id}",
                    user=user,
                    limit=1
                )
                has_access = len(results) > 0

            return [TextContent(
                type="text",
                text=json.dumps({
                    "document_id": document_id,
                    "user_id": user_id,
                    "has_access": has_access
                }, indent=2)
            )]

        except Exception as e:
            logger.error(f"Access check error: {e}")
            return [TextContent(
                type="text",
                text=f"Error checking access: {e!s}"
            )]

    def _format_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Format search results for MCP response."""
        formatted = []

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
                "content": text,
                "metadata": metadata
            }

            if self.include_score and score is not None:
                formatted_result["score"] = score

            formatted.append(formatted_result)

        return formatted

    async def run_stdio(self):
        """Run the server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options()
            )

    def get_server(self) -> Server:
        """Get the underlying MCP Server instance."""
        return self._server


def create_mcp_server(
    retriever: Any,
    name: str = "ragguard",
    description: str = "Permission-aware document retrieval powered by RAGGuard",
    **kwargs
) -> RAGGuardMCPServer:
    """
    Create an MCP server for RAGGuard retrieval.

    This is a convenience function to quickly create an MCP server
    that exposes permission-aware retrieval capabilities.

    Args:
        retriever: Any RAGGuard SecureRetriever instance
        name: Server name
        description: Server description
        **kwargs: Additional arguments passed to RAGGuardMCPServer

    Returns:
        RAGGuardMCPServer instance

    Example:
        ```python
        from ragguard.integrations.mcp import create_mcp_server
        from ragguard.retrievers import QdrantSecureRetriever

        retriever = QdrantSecureRetriever(...)
        server = create_mcp_server(retriever, name="my-docs")

        # Run server
        import asyncio
        asyncio.run(server.run_stdio())
        ```
    """
    return RAGGuardMCPServer(
        retriever=retriever,
        name=name,
        description=description,
        **kwargs
    )


# Optional: FastMCP-style decorator support
class MCPRetrieverTool:
    """
    Decorator to expose a RAGGuard retriever as an MCP tool.

    Example:
        ```python
        @MCPRetrieverTool(name="search_docs")
        def get_retriever():
            return ChromaDBSecureRetriever(...)
        ```
    """

    def __init__(
        self,
        name: str = "secure_search",
        description: str = "Permission-aware document search"
    ):
        self.name = name
        self.description = description

    def __call__(self, retriever_factory: Callable[[], Any]):
        """Wrap retriever factory as MCP tool."""
        _check_mcp_available()

        retriever = retriever_factory()

        async def tool_handler(
            query: str,
            user_id: str,
            user_roles: Optional[List[str]] = None,
            limit: int = 10
        ) -> str:
            user = {"id": user_id, "roles": user_roles or []}
            results = retriever.search(query=query, user=user, limit=limit)
            return json.dumps([
                {"content": r.get("text", str(r)), "metadata": r}
                for r in results
            ])

        tool_handler.__name__ = self.name
        tool_handler.__doc__ = self.description

        return tool_handler
