"""
OpenAI Assistants API integration for RAGGuard.

Provides secure retrieval for OpenAI Assistants with permission-aware file search.

Note: The OpenAI Assistants API is deprecated and will be removed on August 26, 2026.
Consider using the Responses API for new projects.

This integration allows you to:
1. Use RAGGuard retrievers as an alternative to OpenAI's built-in file search
2. Apply permission policies to control what documents assistants can access
3. Inject secure context into assistant conversations

Example:
    ```python
    from openai import OpenAI
    from ragguard.integrations.openai_assistants import (
        SecureAssistantRetriever,
        create_secure_function_tool
    )
    from ragguard import QdrantSecureRetriever, Policy

    # Create RAGGuard retriever with policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "dept-access", "allow": {"conditions": ["user.department == document.department"]}}],
        "default": "deny"
    })

    retriever = QdrantSecureRetriever(
        client=qdrant_client,
        collection="documents",
        policy=policy,
        embed_fn=embed_function
    )

    # Create secure retriever for assistant
    secure_retriever = SecureAssistantRetriever(
        retriever=retriever,
        default_user={"id": "assistant", "department": "general"}
    )

    # Get function tool definition for assistant
    tool_def = create_secure_function_tool(
        name="search_documents",
        description="Search internal documents with access control"
    )

    # Create assistant with the tool
    client = OpenAI()
    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="Help users find information in documents.",
        tools=[tool_def],
        model="gpt-4-turbo-preview"
    )

    # Handle tool calls in your run loop
    # When assistant calls search_documents, use secure_retriever.search()
    ```

Requirements:
    - openai>=1.0.0
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Check if OpenAI is available
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None  # type: ignore
    OpenAI = None  # type: ignore


def _check_openai_available():
    """Check if OpenAI SDK is installed."""
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "OpenAI Assistants integration requires openai. "
            "Install with: pip install openai"
        )


class SecureAssistantRetriever:
    """
    Secure retriever for OpenAI Assistants with permission filtering.

    This class wraps a RAGGuard retriever and provides methods optimized
    for use with OpenAI Assistants API tool calls.

    The retriever enforces access control policies, ensuring assistants
    only receive documents the user is authorized to access.

    Attributes:
        retriever: RAGGuard secure retriever instance
        default_user: Default user context when none specified
        max_results: Maximum results per search (default 10)
        include_metadata: Whether to include metadata in results

    Example:
        ```python
        secure_retriever = SecureAssistantRetriever(
            retriever=qdrant_retriever,
            default_user={"id": "system", "roles": ["assistant"]},
            max_results=5
        )

        # Search with default user
        results = secure_retriever.search("quarterly report")

        # Search with specific user context
        results = secure_retriever.search(
            "quarterly report",
            user={"id": "alice", "department": "finance"}
        )
        ```
    """

    def __init__(
        self,
        retriever: Any,
        default_user: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        include_metadata: bool = True,
        format_for_context: bool = True
    ):
        """
        Initialize secure assistant retriever.

        Args:
            retriever: RAGGuard secure retriever instance
            default_user: Default user context for searches
            max_results: Maximum number of results to return
            include_metadata: Include document metadata in results
            format_for_context: Format results for assistant context window
        """
        self.retriever = retriever
        self.default_user = default_user or {"id": "assistant", "roles": ["assistant"]}
        self.max_results = max_results
        self.include_metadata = include_metadata
        self.format_for_context = format_for_context

    def search(
        self,
        query: str,
        user: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Search documents with permission filtering.

        Args:
            query: Search query string
            user: User context (uses default_user if not provided)
            limit: Max results (uses max_results if not provided)
            **kwargs: Additional search parameters

        Returns:
            If format_for_context=True: Formatted string for assistant context
            If format_for_context=False: List of result dictionaries
        """
        user_context = user or self.default_user
        result_limit = limit or self.max_results

        try:
            results = self.retriever.search(
                query=query,
                user=user_context,
                limit=result_limit,
                **kwargs
            )

            processed = self._process_results(results)

            if self.format_for_context:
                return self._format_for_context(query, processed)
            return processed

        except Exception as e:
            logger.error(f"Search failed: {e}")
            if self.format_for_context:
                return f"Search failed: {e!s}"
            raise

    def _process_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Process raw retriever results into standard format."""
        processed = []

        for result in results:
            if hasattr(result, 'payload'):
                # Qdrant-style result
                content = result.payload.get('text') or result.payload.get('content', '')
                metadata = {k: v for k, v in result.payload.items()
                           if k not in ['text', 'content', 'vector']}
                score = getattr(result, 'score', 0.0)
                doc_id = getattr(result, 'id', None)
            elif isinstance(result, dict):
                content = result.get('text') or result.get('content', '')
                metadata = result.get('metadata', {})
                score = result.get('score', 0.0)
                doc_id = result.get('id')
            else:
                content = str(result)
                metadata = {}
                score = 0.0
                doc_id = None

            item = {
                'content': content,
                'score': score
            }
            if doc_id:
                item['id'] = doc_id
            if self.include_metadata and metadata:
                item['metadata'] = metadata

            processed.append(item)

        return processed

    def _format_for_context(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Format results for assistant context window."""
        if not results:
            return f"No documents found for query: '{query}'"

        lines = [f"Search results for: '{query}'", ""]

        for i, result in enumerate(results, 1):
            lines.append(f"[Document {i}]")
            if 'id' in result:
                lines.append(f"ID: {result['id']}")
            lines.append(f"Relevance: {result['score']:.3f}")
            if 'metadata' in result:
                meta_str = ", ".join(f"{k}={v}" for k, v in result['metadata'].items())
                lines.append(f"Metadata: {meta_str}")
            lines.append(f"Content: {result['content']}")
            lines.append("")

        return "\n".join(lines)

    def handle_tool_call(
        self,
        tool_call: Any,
        user: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Handle an OpenAI Assistants tool call.

        This method parses the tool call arguments and performs the search.

        Args:
            tool_call: OpenAI tool call object from run steps
            user: User context for permission filtering

        Returns:
            Formatted search results string

        Example:
            ```python
            # In your run polling loop
            if run.status == "requires_action":
                tool_outputs = []
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    if tool_call.function.name == "search_documents":
                        output = secure_retriever.handle_tool_call(tool_call, user=current_user)
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": output
                        })

                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
            ```
        """
        try:
            args = json.loads(tool_call.function.arguments)
            query = args.get('query', '')
            limit = args.get('limit', self.max_results)

            return self.search(query=query, user=user, limit=limit)

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid tool call arguments: {e}")
            return f"Invalid tool call arguments: {e}"
        except Exception as e:
            logger.error(f"OpenAI Assistants tool call search failed: {e}")
            return f"Search error: {e}"


def create_secure_function_tool(
    name: str = "search_documents",
    description: str = "Search internal documents with access control filtering"
) -> Dict[str, Any]:
    """
    Create a function tool definition for OpenAI Assistants.

    This returns a tool definition that can be passed to the assistants.create()
    or assistants.update() API calls.

    Args:
        name: Function name
        description: Function description for the assistant

    Returns:
        Tool definition dictionary

    Example:
        ```python
        tool_def = create_secure_function_tool(
            name="search_knowledge_base",
            description="Search the company knowledge base for relevant documents"
        )

        assistant = client.beta.assistants.create(
            name="Research Assistant",
            tools=[tool_def],
            model="gpt-4-turbo-preview"
        )
        ```
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }


class AssistantRunHandler:
    """
    Helper class for handling OpenAI Assistant runs with secure retrieval.

    This class simplifies the run polling loop and automatically handles
    tool calls for secure document search.

    Example:
        ```python
        handler = AssistantRunHandler(
            client=openai_client,
            secure_retriever=secure_retriever,
            tool_name="search_documents"
        )

        # Run assistant and handle tool calls automatically
        messages = handler.run_and_wait(
            assistant_id=assistant.id,
            thread_id=thread.id,
            user={"id": "alice", "department": "engineering"}
        )

        # Get assistant's response
        for msg in messages:
            if msg.role == "assistant":
                print(msg.content[0].text.value)
        ```
    """

    def __init__(
        self,
        client: Any,
        secure_retriever: SecureAssistantRetriever,
        tool_name: str = "search_documents",
        poll_interval: float = 1.0
    ):
        """
        Initialize assistant run handler.

        Args:
            client: OpenAI client instance
            secure_retriever: SecureAssistantRetriever instance
            tool_name: Name of the search tool to handle
            poll_interval: Seconds between status checks
        """
        _check_openai_available()

        self.client = client
        self.secure_retriever = secure_retriever
        self.tool_name = tool_name
        self.poll_interval = poll_interval

    def run_and_wait(
        self,
        assistant_id: str,
        thread_id: str,
        user: Optional[Dict[str, Any]] = None,
        additional_instructions: Optional[str] = None,
        timeout: float = 300.0
    ) -> List[Any]:
        """
        Create a run and wait for completion, handling tool calls.

        Args:
            assistant_id: Assistant ID
            thread_id: Thread ID
            user: User context for permission filtering
            additional_instructions: Additional instructions for this run
            timeout: Maximum time to wait in seconds

        Returns:
            List of messages after run completion

        Raises:
            TimeoutError: If run doesn't complete within timeout
            RuntimeError: If run fails
        """
        import time

        # Create run
        run_params = {
            "assistant_id": assistant_id,
            "thread_id": thread_id
        }
        if additional_instructions:
            run_params["additional_instructions"] = additional_instructions

        run = self.client.beta.threads.runs.create(**run_params)

        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Run did not complete within {timeout} seconds")

            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )

            if run.status == "completed":
                break
            elif run.status == "failed":
                raise RuntimeError(f"Run failed: {run.last_error}")
            elif run.status == "cancelled":
                raise RuntimeError("Run was cancelled")
            elif run.status == "expired":
                raise RuntimeError("Run expired")
            elif run.status == "requires_action":
                self._handle_required_action(thread_id, run, user)
            else:
                time.sleep(self.poll_interval)

        # Get messages
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        return list(messages.data)

    def _handle_required_action(
        self,
        thread_id: str,
        run: Any,
        user: Optional[Dict[str, Any]]
    ):
        """Handle required action (tool calls)."""
        tool_outputs = []

        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            if tool_call.function.name == self.tool_name:
                output = self.secure_retriever.handle_tool_call(tool_call, user=user)
            else:
                output = f"Unknown tool: {tool_call.function.name}"

            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": output
            })

        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )


__all__ = [
    "SecureAssistantRetriever",
    "AssistantRunHandler",
    "create_secure_function_tool",
    "OPENAI_AVAILABLE",
]
