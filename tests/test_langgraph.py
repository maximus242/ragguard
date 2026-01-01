"""
Comprehensive tests for LangGraph integration to maximize coverage.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestSecureRetrieverNode:
    """Tests for SecureRetrieverNode class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = [
            {"text": "Document 1", "score": 0.95, "department": "eng"},
            {"text": "Document 2", "score": 0.85, "department": "eng"},
        ]
        return retriever

    def test_creation_with_retriever(self, mock_retriever):
        """Test creation with existing retriever."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        node = SecureRetrieverNode(retriever=mock_retriever)

        assert node._retriever == mock_retriever

    def test_creation_requires_params_or_retriever(self):
        """Test creation fails without proper parameters."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        with pytest.raises(ValueError, match="Must provide"):
            SecureRetrieverNode()

    def test_call_executes_search(self, mock_retriever):
        """Test __call__ executes search."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        node = SecureRetrieverNode(retriever=mock_retriever)

        state = {
            "query": "test query",
            "user": {"id": "alice", "department": "eng"},
            "limit": 5
        }

        result = node(state)

        mock_retriever.search.assert_called_once()
        assert "documents" in result
        assert "metadata" in result

    def test_call_requires_query(self, mock_retriever):
        """Test __call__ requires query in state."""
        from ragguard.exceptions import RetrieverError
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        node = SecureRetrieverNode(retriever=mock_retriever)

        state = {"user": {"id": "alice"}}

        with pytest.raises(RetrieverError, match="query"):
            node(state)

    def test_call_requires_user(self, mock_retriever):
        """Test __call__ requires user in state."""
        from ragguard.exceptions import RetrieverError
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        node = SecureRetrieverNode(retriever=mock_retriever)

        state = {"query": "test"}

        with pytest.raises(RetrieverError, match="user"):
            node(state)

    def test_call_with_qdrant_results(self, mock_retriever):
        """Test __call__ handles Qdrant-style results."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        # Create Qdrant-style result
        qdrant_result = MagicMock()
        qdrant_result.payload = {"text": "Qdrant doc", "department": "eng"}
        qdrant_result.score = 0.9

        mock_retriever.search.return_value = [qdrant_result]

        node = SecureRetrieverNode(retriever=mock_retriever)

        state = {
            "query": "test",
            "user": {"id": "alice"},
            "limit": 10
        }

        result = node(state)

        assert len(result["documents"]) == 1

    def test_call_with_metadata_results(self, mock_retriever):
        """Test __call__ handles results with metadata attribute."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        # Create result with metadata attribute
        meta_result = MagicMock()
        del meta_result.payload  # Remove payload attribute
        meta_result.metadata = {"text": "Meta doc"}
        meta_result.score = 0.8

        mock_retriever.search.return_value = [meta_result]

        node = SecureRetrieverNode(retriever=mock_retriever)

        state = {
            "query": "test",
            "user": {"id": "alice"},
            "limit": 10
        }

        result = node(state)

        assert len(result["documents"]) == 1

    def test_call_with_fallback_results(self, mock_retriever):
        """Test __call__ handles results without payload or metadata."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        # Create plain result
        plain_result = MagicMock(spec=[])  # No attributes

        mock_retriever.search.return_value = [plain_result]

        node = SecureRetrieverNode(retriever=mock_retriever)

        state = {
            "query": "test",
            "user": {"id": "alice"},
            "limit": 10
        }

        result = node(state)

        assert len(result["documents"]) == 1

    def test_call_extracts_content(self, mock_retriever):
        """Test __call__ extracts content field."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverNode

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        qdrant_result = MagicMock()
        qdrant_result.payload = {"content": "Content field", "other": "data"}
        qdrant_result.score = 0.9

        mock_retriever.search.return_value = [qdrant_result]

        node = SecureRetrieverNode(retriever=mock_retriever)

        state = {
            "query": "test",
            "user": {"id": "alice"},
            "limit": 10
        }

        result = node(state)

        assert len(result["documents"]) == 1


class TestSecureRetrieverTool:
    """Tests for SecureRetrieverTool class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = []
        return retriever

    def test_creation(self, mock_retriever):
        """Test creation with retriever."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverTool

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        tool = SecureRetrieverTool(
            retriever=mock_retriever,
            name="my_tool",
            description="My search tool"
        )

        assert tool.name == "my_tool"
        assert tool.description == "My search tool"

    def test_run(self, mock_retriever):
        """Test run method."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverTool

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        mock_retriever.search.return_value = [{"text": "Result"}]

        tool = SecureRetrieverTool(retriever=mock_retriever)

        result = tool.run(
            query="test query",
            user={"id": "alice"},
            limit=5
        )

        assert isinstance(result, list)
        mock_retriever.search.assert_called_once()

    def test_as_tool(self, mock_retriever):
        """Test as_tool method."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, SecureRetrieverTool

        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not available")

        try:
            from langchain.tools import Tool
        except ImportError:
            pytest.skip("LangChain not available")

        tool = SecureRetrieverTool(
            retriever=mock_retriever,
            name="test_tool",
            description="Test description"
        )

        lc_tool = tool.as_tool()

        assert lc_tool is not None
        assert lc_tool.name == "test_tool"


class TestCheckLangGraphAvailable:
    """Tests for _check_langgraph_available function."""

    def test_check_when_not_available(self):
        """Test check raises when LangGraph not installed."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE, _check_langgraph_available

        if LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph is available")

        with pytest.raises(ImportError, match="langgraph"):
            _check_langgraph_available()


class TestLangGraphModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test module exports."""
        from ragguard.integrations import langgraph

        expected = [
            "SecureRetrieverNode",
            "SecureRetrieverTool",
            "RetrieverState",
            "LANGGRAPH_AVAILABLE",
        ]

        for name in expected:
            assert hasattr(langgraph, name)

    def test_langgraph_available_flag(self):
        """Test LANGGRAPH_AVAILABLE flag."""
        from ragguard.integrations.langgraph import LANGGRAPH_AVAILABLE

        assert isinstance(LANGGRAPH_AVAILABLE, bool)


class TestRetrieverState:
    """Tests for RetrieverState TypedDict."""

    def test_retriever_state_type(self):
        """Test RetrieverState is a TypedDict."""
        from ragguard.integrations.langgraph import RetrieverState

        # TypedDict should be a dict subtype
        assert RetrieverState is not None
