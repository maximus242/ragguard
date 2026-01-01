"""
Comprehensive tests for OpenAI Assistants integration to maximize coverage.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestCheckOpenAIAvailable:
    """Tests for _check_openai_available function."""

    def test_raises_when_not_available(self):
        """Test that function raises ImportError when OpenAI not installed."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            _check_openai_available,
        )

        if OPENAI_AVAILABLE:
            pytest.skip("OpenAI is available")

        with pytest.raises(ImportError, match="openai"):
            _check_openai_available()


class TestSecureAssistantRetriever:
    """Tests for SecureAssistantRetriever class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = [
            {"text": "Document 1", "score": 0.95, "id": "doc1"},
            {"text": "Document 2", "score": 0.85, "id": "doc2"},
        ]
        return retriever

    def test_init_default_values(self, mock_retriever):
        """Test initialization with default values."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(retriever=mock_retriever)

        assert retriever.default_user == {"id": "assistant", "roles": ["assistant"]}
        assert retriever.max_results == 10
        assert retriever.include_metadata is True
        assert retriever.format_for_context is True

    def test_init_custom_values(self, mock_retriever):
        """Test initialization with custom values."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            default_user={"id": "admin", "roles": ["admin"]},
            max_results=5,
            include_metadata=False,
            format_for_context=False
        )

        assert retriever.default_user["id"] == "admin"
        assert retriever.max_results == 5
        assert retriever.include_metadata is False
        assert retriever.format_for_context is False

    def test_search_uses_default_user(self, mock_retriever):
        """Test search uses default user when none provided."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            default_user={"id": "default"}
        )

        retriever.search("test query")

        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["user"]["id"] == "default"

    def test_search_uses_provided_user(self, mock_retriever):
        """Test search uses provided user."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(retriever=mock_retriever)

        retriever.search("test query", user={"id": "alice"})

        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["user"]["id"] == "alice"

    def test_search_uses_default_limit(self, mock_retriever):
        """Test search uses default limit."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            max_results=15
        )

        retriever.search("test query")

        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["limit"] == 15

    def test_search_uses_provided_limit(self, mock_retriever):
        """Test search uses provided limit."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(retriever=mock_retriever)

        retriever.search("test query", limit=3)

        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["limit"] == 3

    def test_search_formats_for_context(self, mock_retriever):
        """Test search formats results for context."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=True
        )

        result = retriever.search("test query")

        assert isinstance(result, str)
        assert "Search results" in result
        assert "Document 1" in result

    def test_search_returns_list(self, mock_retriever):
        """Test search returns list when format_for_context=False."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=False
        )

        result = retriever.search("test query")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_search_error_formatted(self, mock_retriever):
        """Test search error returns formatted string when format_for_context=True."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.side_effect = Exception("Search failed")

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=True
        )

        result = retriever.search("test query")

        assert "Search failed" in result

    def test_search_error_raises(self, mock_retriever):
        """Test search error raises when format_for_context=False."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.side_effect = Exception("Search failed")

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=False
        )

        with pytest.raises(Exception, match="Search failed"):
            retriever.search("test query")

    def test_process_results_qdrant_format(self, mock_retriever):
        """Test processing Qdrant-style results."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        qdrant_result = MagicMock()
        qdrant_result.payload = {"text": "Qdrant content", "author": "Alice"}
        qdrant_result.score = 0.9
        qdrant_result.id = "qdrant1"
        mock_retriever.search.return_value = [qdrant_result]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=False
        )

        result = retriever.search("test query")

        assert result[0]["content"] == "Qdrant content"
        assert result[0]["score"] == 0.9
        assert result[0]["id"] == "qdrant1"
        assert "author" in result[0]["metadata"]

    def test_process_results_qdrant_content_field(self, mock_retriever):
        """Test processing Qdrant results with content field."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        qdrant_result = MagicMock()
        qdrant_result.payload = {"content": "Content field", "source": "file.pdf"}
        qdrant_result.score = 0.8
        qdrant_result.id = "qdrant2"
        mock_retriever.search.return_value = [qdrant_result]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=False
        )

        result = retriever.search("test query")

        assert result[0]["content"] == "Content field"

    def test_process_results_dict_format(self, mock_retriever):
        """Test processing dict results."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.return_value = [
            {"text": "Dict content", "score": 0.7, "id": "dict1", "metadata": {"source": "web"}}
        ]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=False
        )

        result = retriever.search("test query")

        assert result[0]["content"] == "Dict content"
        assert result[0]["score"] == 0.7
        assert result[0]["id"] == "dict1"

    def test_process_results_dict_content_field(self, mock_retriever):
        """Test processing dict results with content field."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.return_value = [
            {"content": "Content field", "score": 0.6}
        ]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=False
        )

        result = retriever.search("test query")

        assert result[0]["content"] == "Content field"

    def test_process_results_plain_object(self, mock_retriever):
        """Test processing plain objects."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        class PlainResult:
            def __str__(self):
                return "Plain content"

        mock_retriever.search.return_value = [PlainResult()]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=False
        )

        result = retriever.search("test query")

        assert result[0]["content"] == "Plain content"
        assert result[0]["score"] == 0.0

    def test_process_results_excludes_metadata_when_disabled(self, mock_retriever):
        """Test that metadata is excluded when include_metadata=False."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.return_value = [
            {"text": "Content", "metadata": {"source": "web"}, "score": 0.8}
        ]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            include_metadata=False,
            format_for_context=False
        )

        result = retriever.search("test query")

        assert "metadata" not in result[0]

    def test_format_for_context_empty_results(self, mock_retriever):
        """Test formatting empty results for context."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.return_value = []

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=True
        )

        result = retriever.search("test query")

        assert "No documents found" in result

    def test_format_for_context_with_id(self, mock_retriever):
        """Test formatting results with ID."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.return_value = [
            {"text": "Content", "score": 0.9, "id": "doc123"}
        ]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=True
        )

        result = retriever.search("test query")

        assert "ID: doc123" in result

    def test_format_for_context_with_metadata(self, mock_retriever):
        """Test formatting results with metadata."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.return_value = [
            {"text": "Content", "score": 0.9, "metadata": {"source": "web", "author": "Alice"}}
        ]

        retriever = SecureAssistantRetriever(
            retriever=mock_retriever,
            format_for_context=True
        )

        result = retriever.search("test query")

        assert "Metadata:" in result
        assert "source=web" in result

    def test_handle_tool_call(self, mock_retriever):
        """Test handling OpenAI tool call."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.return_value = [{"text": "Result", "score": 0.9}]

        retriever = SecureAssistantRetriever(retriever=mock_retriever)

        tool_call = MagicMock()
        tool_call.function.arguments = json.dumps({"query": "test query", "limit": 5})

        result = retriever.handle_tool_call(tool_call, user={"id": "alice"})

        assert "Result" in result

    def test_handle_tool_call_invalid_json(self, mock_retriever):
        """Test handling tool call with invalid JSON."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        retriever = SecureAssistantRetriever(retriever=mock_retriever)

        tool_call = MagicMock()
        tool_call.function.arguments = "{invalid json"

        result = retriever.handle_tool_call(tool_call)

        assert "Invalid" in result

    def test_handle_tool_call_error(self, mock_retriever):
        """Test handling tool call error."""
        from ragguard.integrations.openai_assistants import SecureAssistantRetriever

        mock_retriever.search.side_effect = Exception("Search error")

        retriever = SecureAssistantRetriever(retriever=mock_retriever)

        tool_call = MagicMock()
        tool_call.function.arguments = json.dumps({"query": "test"})

        result = retriever.handle_tool_call(tool_call)

        assert "error" in result.lower()


class TestCreateSecureFunctionTool:
    """Tests for create_secure_function_tool function."""

    def test_default_values(self):
        """Test function tool with default values."""
        from ragguard.integrations.openai_assistants import create_secure_function_tool

        tool = create_secure_function_tool()

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "search_documents"
        assert "query" in tool["function"]["parameters"]["properties"]

    def test_custom_values(self):
        """Test function tool with custom values."""
        from ragguard.integrations.openai_assistants import create_secure_function_tool

        tool = create_secure_function_tool(
            name="custom_search",
            description="Custom search description"
        )

        assert tool["function"]["name"] == "custom_search"
        assert tool["function"]["description"] == "Custom search description"


class TestAssistantRunHandler:
    """Tests for AssistantRunHandler class."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        return client

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = []
        return retriever

    def test_init_raises_when_not_available(self, mock_client, mock_retriever):
        """Test init raises when OpenAI not available."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if OPENAI_AVAILABLE:
            pytest.skip("OpenAI is available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        with pytest.raises(ImportError, match="openai"):
            AssistantRunHandler(
                client=mock_client,
                secure_retriever=secure_retriever
            )

    def test_init_success(self, mock_client, mock_retriever):
        """Test successful initialization."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever,
            tool_name="custom_search",
            poll_interval=2.0
        )

        assert handler.tool_name == "custom_search"
        assert handler.poll_interval == 2.0

    def test_run_and_wait_completed(self, mock_client, mock_retriever):
        """Test run_and_wait when run completes successfully."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        run = MagicMock()
        run.status = "completed"
        run.id = "run_123"

        mock_client.beta.threads.runs.create.return_value = run
        mock_client.beta.threads.runs.retrieve.return_value = run
        mock_client.beta.threads.messages.list.return_value.data = []

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever
        )

        result = handler.run_and_wait(
            assistant_id="asst_123",
            thread_id="thread_123"
        )

        assert result == []

    def test_run_and_wait_timeout(self, mock_client, mock_retriever):
        """Test run_and_wait raises on timeout."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        run = MagicMock()
        run.status = "queued"
        run.id = "run_123"

        mock_client.beta.threads.runs.create.return_value = run
        mock_client.beta.threads.runs.retrieve.return_value = run

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever,
            poll_interval=0.01
        )

        with pytest.raises(TimeoutError):
            handler.run_and_wait(
                assistant_id="asst_123",
                thread_id="thread_123",
                timeout=0.05
            )

    def test_run_and_wait_failed(self, mock_client, mock_retriever):
        """Test run_and_wait raises on failure."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        run = MagicMock()
        run.status = "failed"
        run.last_error = "Something went wrong"
        run.id = "run_123"

        mock_client.beta.threads.runs.create.return_value = run
        mock_client.beta.threads.runs.retrieve.return_value = run

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever
        )

        with pytest.raises(RuntimeError, match="failed"):
            handler.run_and_wait(
                assistant_id="asst_123",
                thread_id="thread_123"
            )

    def test_run_and_wait_cancelled(self, mock_client, mock_retriever):
        """Test run_and_wait raises on cancellation."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        run = MagicMock()
        run.status = "cancelled"
        run.id = "run_123"

        mock_client.beta.threads.runs.create.return_value = run
        mock_client.beta.threads.runs.retrieve.return_value = run

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever
        )

        with pytest.raises(RuntimeError, match="cancelled"):
            handler.run_and_wait(
                assistant_id="asst_123",
                thread_id="thread_123"
            )

    def test_run_and_wait_expired(self, mock_client, mock_retriever):
        """Test run_and_wait raises on expiration."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        run = MagicMock()
        run.status = "expired"
        run.id = "run_123"

        mock_client.beta.threads.runs.create.return_value = run
        mock_client.beta.threads.runs.retrieve.return_value = run

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever
        )

        with pytest.raises(RuntimeError, match="expired"):
            handler.run_and_wait(
                assistant_id="asst_123",
                thread_id="thread_123"
            )

    def test_run_and_wait_handles_tool_calls(self, mock_client, mock_retriever):
        """Test run_and_wait handles tool calls."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        # First call returns requires_action, second returns completed
        tool_call = MagicMock()
        tool_call.function.name = "search_documents"
        tool_call.function.arguments = json.dumps({"query": "test"})
        tool_call.id = "call_123"

        run_requires_action = MagicMock()
        run_requires_action.status = "requires_action"
        run_requires_action.id = "run_123"
        run_requires_action.required_action.submit_tool_outputs.tool_calls = [tool_call]

        run_completed = MagicMock()
        run_completed.status = "completed"
        run_completed.id = "run_123"

        mock_client.beta.threads.runs.create.return_value = run_requires_action
        mock_client.beta.threads.runs.retrieve.side_effect = [run_requires_action, run_completed]
        mock_client.beta.threads.messages.list.return_value.data = []

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever
        )

        result = handler.run_and_wait(
            assistant_id="asst_123",
            thread_id="thread_123"
        )

        # Should have called submit_tool_outputs
        mock_client.beta.threads.runs.submit_tool_outputs.assert_called()

    def test_handle_unknown_tool(self, mock_client, mock_retriever):
        """Test handling unknown tool calls."""
        from ragguard.integrations.openai_assistants import (
            OPENAI_AVAILABLE,
            AssistantRunHandler,
            SecureAssistantRetriever,
        )

        if not OPENAI_AVAILABLE:
            pytest.skip("OpenAI not available")

        secure_retriever = SecureAssistantRetriever(retriever=mock_retriever)

        tool_call = MagicMock()
        tool_call.function.name = "unknown_tool"
        tool_call.id = "call_456"

        run_requires_action = MagicMock()
        run_requires_action.status = "requires_action"
        run_requires_action.id = "run_123"
        run_requires_action.required_action.submit_tool_outputs.tool_calls = [tool_call]

        run_completed = MagicMock()
        run_completed.status = "completed"
        run_completed.id = "run_123"

        mock_client.beta.threads.runs.create.return_value = run_requires_action
        mock_client.beta.threads.runs.retrieve.side_effect = [run_requires_action, run_completed]
        mock_client.beta.threads.messages.list.return_value.data = []

        handler = AssistantRunHandler(
            client=mock_client,
            secure_retriever=secure_retriever
        )

        handler.run_and_wait(
            assistant_id="asst_123",
            thread_id="thread_123"
        )

        # Should have submitted tool outputs with "Unknown tool" message
        call_args = mock_client.beta.threads.runs.submit_tool_outputs.call_args
        tool_outputs = call_args.kwargs["tool_outputs"]
        assert "Unknown tool" in tool_outputs[0]["output"]


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test all __all__ exports are accessible."""
        from ragguard.integrations import openai_assistants

        for name in openai_assistants.__all__:
            assert hasattr(openai_assistants, name), f"{name} not found"

    def test_openai_available_flag(self):
        """Test OPENAI_AVAILABLE flag is boolean."""
        from ragguard.integrations.openai_assistants import OPENAI_AVAILABLE
        assert isinstance(OPENAI_AVAILABLE, bool)
