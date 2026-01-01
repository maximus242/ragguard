"""
Comprehensive tests for A2A integration to maximize coverage.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestA2ACheckAvailable:
    """Tests for _check_a2a_available function."""

    def test_check_raises_when_not_available(self):
        """Test _check_a2a_available raises when SDK not installed."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, _check_a2a_available

        if A2A_AVAILABLE:
            pytest.skip("A2A SDK is available")

        with pytest.raises(ImportError, match="a2a-sdk"):
            _check_a2a_available()


class TestRAGGuardAgentExecutor:
    """Tests for RAGGuardAgentExecutor class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = [
            {"text": "Document 1", "score": 0.95, "id": 1},
            {"text": "Document 2", "score": 0.85, "id": 2},
        ]
        return retriever

    def test_executor_init_default_user(self, mock_retriever):
        """Test executor initialization with default user."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(
            retriever=mock_retriever
        )

        assert executor.default_user == {"id": "anonymous", "roles": []}

    def test_executor_init_custom_settings(self, mock_retriever):
        """Test executor initialization with custom settings."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(
            retriever=mock_retriever,
            text_key="content",
            include_score=False,
            max_results=20,
            default_user={"id": "admin", "roles": ["admin"]}
        )

        assert executor.text_key == "content"
        assert executor.include_score is False
        assert executor.max_results == 20
        assert executor.default_user["id"] == "admin"

    def test_parse_request_plain_text(self, mock_retriever):
        """Test parsing plain text request."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        # Create mock context with plain text message
        context = MagicMock()
        part = MagicMock()
        part.text = "Search for documents"
        context.message.parts = [part]

        query, user_context, limit = executor._parse_request(context)

        assert query == "Search for documents"
        assert user_context["id"] == "anonymous"
        assert limit == 10

    def test_parse_request_json(self, mock_retriever):
        """Test parsing JSON request."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        # Create mock context with JSON message
        context = MagicMock()
        part = MagicMock()
        part.text = json.dumps({
            "query": "machine learning",
            "user_id": "alice",
            "user_roles": ["engineer"],
            "user_attributes": {"department": "ml"},
            "limit": 5
        })
        context.message.parts = [part]

        query, user_context, limit = executor._parse_request(context)

        assert query == "machine learning"
        assert user_context["id"] == "alice"
        assert user_context["roles"] == ["engineer"]
        assert user_context["department"] == "ml"
        assert limit == 5

    def test_parse_request_invalid_json(self, mock_retriever):
        """Test parsing request with invalid JSON falls back to plain text."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        # Create mock context with invalid JSON (starts with { but not valid)
        context = MagicMock()
        part = MagicMock()
        part.text = "{not valid json"
        context.message.parts = [part]

        query, user_context, limit = executor._parse_request(context)

        assert query == "{not valid json"

    def test_parse_request_data_part(self, mock_retriever):
        """Test parsing DataPart request."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        # Create mock context with DataPart
        context = MagicMock()
        part = MagicMock(spec=['data'])  # Has data but not text
        del part.text  # Ensure no text attribute
        part.data = {
            "query": "data part query",
            "user_id": "bob",
            "limit": 3
        }
        context.message.parts = [part]

        query, user_context, limit = executor._parse_request(context)

        assert query == "data part query"
        assert user_context["id"] == "bob"
        assert limit == 3

    def test_parse_request_empty_message(self, mock_retriever):
        """Test parsing empty message."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        context = MagicMock()
        context.message.parts = []

        query, user_context, limit = executor._parse_request(context)

        assert query == ""

    def test_format_response_qdrant_result(self, mock_retriever):
        """Test formatting Qdrant-style results."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        qdrant_result = MagicMock()
        qdrant_result.payload = {"text": "Document content", "author": "Alice"}
        qdrant_result.score = 0.95

        response = executor._format_response(
            query="test",
            results=[qdrant_result],
            user_context={"id": "user1"}
        )

        response_dict = json.loads(response)
        assert response_dict["query"] == "test"
        assert response_dict["total_results"] == 1
        assert response_dict["results"][0]["score"] == 0.95

    def test_format_response_dict_result(self, mock_retriever):
        """Test formatting dict results."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        response = executor._format_response(
            query="test",
            results=[{"text": "Doc text", "score": 0.8, "id": 1}],
            user_context={"id": "user1"}
        )

        response_dict = json.loads(response)
        assert response_dict["results"][0]["score"] == 0.8

    def test_format_response_metadata_result(self, mock_retriever):
        """Test formatting results with metadata attribute."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        result = MagicMock()
        result.metadata = {"text": "Metadata text", "source": "file.pdf"}
        result.score = 0.75
        # Remove payload attribute
        del result.payload

        response = executor._format_response(
            query="test",
            results=[result],
            user_context={"id": "user1"}
        )

        response_dict = json.loads(response)
        assert response_dict["results"][0]["score"] == 0.75

    def test_format_response_raw_result(self, mock_retriever):
        """Test formatting raw results without known attributes."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        class RawResult:
            def __str__(self):
                return "raw result data"

        response = executor._format_response(
            query="test",
            results=[RawResult()],
            user_context={"id": "user1"}
        )

        response_dict = json.loads(response)
        assert response_dict["total_results"] == 1

    def test_format_response_without_score(self, mock_retriever):
        """Test formatting results with include_score=False."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(
            retriever=mock_retriever,
            include_score=False
        )

        response = executor._format_response(
            query="test",
            results=[{"text": "Doc", "score": 0.9}],
            user_context={"id": "user1"}
        )

        response_dict = json.loads(response)
        assert "score" not in response_dict["results"][0]

    def test_format_response_truncates_long_text(self, mock_retriever):
        """Test that long text is truncated."""
        from ragguard.integrations.a2a import RAGGuardAgentExecutor

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        long_text = "x" * 1000
        response = executor._format_response(
            query="test",
            results=[{"text": long_text}],
            user_context={"id": "user1"}
        )

        response_dict = json.loads(response)
        assert len(response_dict["results"][0]["content"]) < 600
        assert response_dict["results"][0]["content"].endswith("...")

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_retriever):
        """Test successful execute."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        from a2a.server.events import EventQueue
        from a2a.utils import new_agent_text_message

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        context = MagicMock()
        part = MagicMock()
        part.text = "test query"
        context.message.parts = [part]

        event_queue = AsyncMock(spec=EventQueue)

        await executor.execute(context, event_queue)

        event_queue.enqueue_event.assert_called()

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, mock_retriever):
        """Test execute with empty query."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        context = MagicMock()
        context.message.parts = []

        event_queue = AsyncMock()

        await executor.execute(context, event_queue)

        # Should send error message
        event_queue.enqueue_event.assert_called()
        call_args = event_queue.enqueue_event.call_args[0][0]
        # The message should mention no query provided

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, mock_retriever):
        """Test execute handles retriever errors."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        mock_retriever.search.side_effect = Exception("Search failed")
        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        context = MagicMock()
        part = MagicMock()
        part.text = "test query"
        context.message.parts = [part]

        event_queue = AsyncMock()

        await executor.execute(context, event_queue)

        event_queue.enqueue_event.assert_called()

    @pytest.mark.asyncio
    async def test_cancel(self, mock_retriever):
        """Test cancel method."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, RAGGuardAgentExecutor

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        executor = RAGGuardAgentExecutor(retriever=mock_retriever)

        context = MagicMock()
        event_queue = AsyncMock()

        await executor.cancel(context, event_queue)

        event_queue.enqueue_event.assert_called()


class TestRAGGuardA2AServer:
    """Tests for RAGGuardA2AServer class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test_collection"
        retriever.search.return_value = []
        return retriever

    def test_server_init_without_starlette(self, mock_retriever):
        """Test server initialization fails without Starlette."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, STARLETTE_AVAILABLE, RAGGuardA2AServer

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        if STARLETTE_AVAILABLE:
            pytest.skip("Starlette is available")

        with pytest.raises(ImportError, match="Starlette"):
            RAGGuardA2AServer(retriever=mock_retriever)

    def test_server_init_success(self, mock_retriever):
        """Test successful server initialization."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, STARLETTE_AVAILABLE, RAGGuardA2AServer

        if not A2A_AVAILABLE or not STARLETTE_AVAILABLE:
            pytest.skip("A2A SDK or Starlette not available")

        server = RAGGuardA2AServer(
            retriever=mock_retriever,
            name="Test Agent",
            description="Test description",
            version="2.0.0",
            url="http://example.com/",
            text_key="content",
            include_score=True,
            max_results=15,
            enable_streaming=True,
            supports_extended_card=True
        )

        assert server.name == "Test Agent"
        assert server.description == "Test description"
        assert server.version == "2.0.0"

    def test_agent_card_property(self, mock_retriever):
        """Test agent_card property."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, STARLETTE_AVAILABLE, RAGGuardA2AServer

        if not A2A_AVAILABLE or not STARLETTE_AVAILABLE:
            pytest.skip("A2A SDK or Starlette not available")

        server = RAGGuardA2AServer(retriever=mock_retriever)

        card = server.agent_card
        assert card is not None
        assert card.name == "RAGGuard Retriever Agent"

    def test_app_property(self, mock_retriever):
        """Test app property."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, STARLETTE_AVAILABLE, RAGGuardA2AServer

        if not A2A_AVAILABLE or not STARLETTE_AVAILABLE:
            pytest.skip("A2A SDK or Starlette not available")

        server = RAGGuardA2AServer(retriever=mock_retriever)

        app = server.app
        assert app is not None

    @pytest.mark.asyncio
    async def test_run_without_uvicorn(self, mock_retriever):
        """Test run fails without uvicorn."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, STARLETTE_AVAILABLE, RAGGuardA2AServer

        if not A2A_AVAILABLE or not STARLETTE_AVAILABLE:
            pytest.skip("A2A SDK or Starlette not available")

        server = RAGGuardA2AServer(retriever=mock_retriever)

        with patch.dict('sys.modules', {'uvicorn': None}):
            with pytest.raises(ImportError, match="uvicorn"):
                await server.run()


class TestCreateA2AServer:
    """Tests for create_a2a_server function."""

    def test_create_server(self):
        """Test create_a2a_server function."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, STARLETTE_AVAILABLE, create_a2a_server

        if not A2A_AVAILABLE or not STARLETTE_AVAILABLE:
            pytest.skip("A2A SDK or Starlette not available")

        mock_retriever = MagicMock()
        mock_retriever.backend_name = "qdrant"
        mock_retriever.collection = "docs"

        server = create_a2a_server(
            retriever=mock_retriever,
            name="Custom Agent",
            description="Custom description"
        )

        assert server.name == "Custom Agent"
        assert server.description == "Custom description"


class TestCreateA2AExecutor:
    """Tests for create_a2a_executor function."""

    def test_create_executor_without_sdk(self):
        """Test create_a2a_executor fails without SDK."""
        from ragguard.integrations.a2a import A2A_AVAILABLE, create_a2a_executor

        if A2A_AVAILABLE:
            pytest.skip("A2A SDK is available")

        mock_retriever = MagicMock()

        with pytest.raises(ImportError, match="a2a-sdk"):
            create_a2a_executor(retriever=mock_retriever)

    def test_create_executor_success(self):
        """Test create_a2a_executor success."""
        from ragguard.integrations.a2a import (
            A2A_AVAILABLE,
            RAGGuardAgentExecutor,
            create_a2a_executor,
        )

        if not A2A_AVAILABLE:
            pytest.skip("A2A SDK not available")

        mock_retriever = MagicMock()

        executor = create_a2a_executor(
            retriever=mock_retriever,
            text_key="body",
            max_results=25
        )

        assert isinstance(executor, RAGGuardAgentExecutor)
        assert executor.text_key == "body"
        assert executor.max_results == 25


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test that all __all__ exports exist."""
        from ragguard.integrations import a2a

        for name in a2a.__all__:
            assert hasattr(a2a, name), f"{name} not found in module"

    def test_a2a_available_flag(self):
        """Test A2A_AVAILABLE flag exists."""
        from ragguard.integrations.a2a import A2A_AVAILABLE
        assert isinstance(A2A_AVAILABLE, bool)

    def test_starlette_available_flag(self):
        """Test STARLETTE_AVAILABLE flag exists."""
        from ragguard.integrations.a2a import STARLETTE_AVAILABLE
        assert isinstance(STARLETTE_AVAILABLE, bool)
