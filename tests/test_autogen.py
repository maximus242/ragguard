"""
Comprehensive tests for AutoGen integration to maximize coverage.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCheckAutogenAvailable:
    """Tests for _check_autogen_available function."""

    def test_raises_when_not_available(self):
        """Test that function raises ImportError when AutoGen not installed."""
        from ragguard.integrations.autogen import AUTOGEN_AVAILABLE, _check_autogen_available

        if AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen is available")

        with pytest.raises(ImportError, match="autogen"):
            _check_autogen_available()


class TestSecureSearchTool:
    """Tests for SecureSearchTool class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = [
            {"text": "Document 1", "score": 0.95},
            {"text": "Document 2", "score": 0.85},
        ]
        return retriever

    def test_init_default_values(self, mock_retriever):
        """Test initialization with default values."""
        from ragguard.integrations.autogen import SecureSearchTool

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        assert tool.max_results == 10
        assert tool.name == "search_documents"
        assert "access control" in tool.description

    def test_init_custom_values(self, mock_retriever):
        """Test initialization with custom values."""
        from ragguard.integrations.autogen import SecureSearchTool

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"},
            max_results=5,
            name="custom_search",
            description="Custom description"
        )

        assert tool.max_results == 5
        assert tool.name == "custom_search"
        assert tool.description == "Custom description"

    def test_call_success(self, mock_retriever):
        """Test successful search call."""
        from ragguard.integrations.autogen import SecureSearchTool

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")

        assert "Found 2 document(s)" in result
        assert "Document 1" in result
        assert "0.950" in result  # Score formatting

    def test_call_with_limit(self, mock_retriever):
        """Test search call with custom limit."""
        from ragguard.integrations.autogen import SecureSearchTool

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"},
            max_results=10
        )

        tool("test query", limit=5)

        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["limit"] == 5

    def test_call_error_handling(self, mock_retriever):
        """Test error handling in call."""
        from ragguard.integrations.autogen import SecureSearchTool

        mock_retriever.search.side_effect = Exception("Search failed")

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")
        assert "Search failed" in result

    def test_format_results_empty(self, mock_retriever):
        """Test formatting empty results."""
        from ragguard.integrations.autogen import SecureSearchTool

        mock_retriever.search.return_value = []

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")
        assert "No documents found" in result

    def test_format_results_qdrant_payload(self, mock_retriever):
        """Test formatting Qdrant-style results."""
        from ragguard.integrations.autogen import SecureSearchTool

        qdrant_result = MagicMock()
        qdrant_result.payload = {"text": "Qdrant content", "id": 1}
        qdrant_result.score = 0.9
        mock_retriever.search.return_value = [qdrant_result]

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")
        assert "Qdrant content" in result

    def test_format_results_qdrant_content_field(self, mock_retriever):
        """Test formatting Qdrant results with content field."""
        from ragguard.integrations.autogen import SecureSearchTool

        qdrant_result = MagicMock()
        qdrant_result.payload = {"content": "Content field", "id": 1}
        qdrant_result.score = 0.9
        mock_retriever.search.return_value = [qdrant_result]

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")
        assert "Content field" in result

    def test_format_results_dict_content(self, mock_retriever):
        """Test formatting dict results with content field."""
        from ragguard.integrations.autogen import SecureSearchTool

        mock_retriever.search.return_value = [
            {"content": "Dict content", "score": 0.8}
        ]

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")
        assert "Dict content" in result

    def test_format_results_plain_object(self, mock_retriever):
        """Test formatting plain objects."""
        from ragguard.integrations.autogen import SecureSearchTool

        class PlainResult:
            def __str__(self):
                return "Plain result"

        mock_retriever.search.return_value = [PlainResult()]

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")
        assert "Plain result" in result

    def test_format_results_truncates_long_content(self, mock_retriever):
        """Test that long content is truncated."""
        from ragguard.integrations.autogen import SecureSearchTool

        long_text = "x" * 1000
        mock_retriever.search.return_value = [{"text": long_text, "score": 0.9}]

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool("test query")
        assert "..." in result
        assert len(result) < len(long_text) + 200  # Allow for formatting

    def test_set_user_returns_self(self, mock_retriever):
        """Test set_user returns self for chaining."""
        from ragguard.integrations.autogen import SecureSearchTool

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = tool.set_user({"id": "bob"})
        assert result is tool
        assert tool.user["id"] == "bob"

    def test_as_function_tool_without_v4(self, mock_retriever):
        """Test as_function_tool fails without AutoGen v0.4."""
        from ragguard.integrations.autogen import AUTOGEN_V4_AVAILABLE, SecureSearchTool

        if AUTOGEN_V4_AVAILABLE:
            pytest.skip("AutoGen v0.4 is available")

        tool = SecureSearchTool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        with pytest.raises(ImportError, match="v0.4"):
            tool.as_function_tool()


class TestSecureRetrieverFunction:
    """Tests for SecureRetrieverFunction class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = []
        return retriever

    def test_init(self, mock_retriever):
        """Test initialization."""
        from ragguard.integrations.autogen import SecureRetrieverFunction

        fn = SecureRetrieverFunction(
            retriever=mock_retriever,
            user={"id": "alice"},
            max_results=15,
            name="custom_search",
            description="Custom desc"
        )

        assert fn.max_results == 15
        assert fn.name == "custom_search"
        assert fn.description == "Custom desc"

    def test_call_success(self, mock_retriever):
        """Test successful function call."""
        from ragguard.integrations.autogen import SecureRetrieverFunction

        mock_retriever.search.return_value = [{"text": "Result", "score": 0.9}]

        fn = SecureRetrieverFunction(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = fn("test query")
        assert "Result" in result

    def test_call_error(self, mock_retriever):
        """Test error handling in call."""
        from ragguard.integrations.autogen import SecureRetrieverFunction

        mock_retriever.search.side_effect = Exception("Search error")

        fn = SecureRetrieverFunction(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = fn("test query")
        assert "Search error" in result

    def test_format_results_empty(self, mock_retriever):
        """Test formatting empty results."""
        from ragguard.integrations.autogen import SecureRetrieverFunction

        fn = SecureRetrieverFunction(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = fn("test query")
        assert "No accessible documents found" in result

    def test_set_user(self, mock_retriever):
        """Test set_user method."""
        from ragguard.integrations.autogen import SecureRetrieverFunction

        fn = SecureRetrieverFunction(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = fn.set_user({"id": "bob"})
        assert result is fn
        assert fn.user["id"] == "bob"

    def test_function_schema(self, mock_retriever):
        """Test function_schema property."""
        from ragguard.integrations.autogen import SecureRetrieverFunction

        fn = SecureRetrieverFunction(
            retriever=mock_retriever,
            user={"id": "alice"},
            name="my_search",
            description="My search function"
        )

        schema = fn.function_schema

        assert schema["name"] == "my_search"
        assert schema["description"] == "My search function"
        assert "query" in schema["parameters"]["properties"]
        assert "limit" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]


class TestCreateSecureSearchTool:
    """Tests for create_secure_search_tool function."""

    def test_raises_when_not_available(self):
        """Test that function raises when AutoGen not available."""
        from ragguard.integrations.autogen import AUTOGEN_AVAILABLE, create_secure_search_tool

        if AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen is available")

        mock_retriever = MagicMock()

        with pytest.raises(ImportError, match="autogen"):
            create_secure_search_tool(
                retriever=mock_retriever,
                user={"id": "alice"}
            )

    def test_returns_function_tool_with_v4(self):
        """Test returns FunctionTool when v0.4 available."""
        from ragguard.integrations.autogen import AUTOGEN_V4_AVAILABLE, create_secure_search_tool

        if not AUTOGEN_V4_AVAILABLE:
            pytest.skip("AutoGen v0.4 not available")

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        result = create_secure_search_tool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        # Should be a FunctionTool
        from autogen_core.tools import FunctionTool
        assert isinstance(result, FunctionTool)

    def test_returns_function_with_v2(self):
        """Test returns SecureRetrieverFunction when only v0.2 available."""
        from ragguard.integrations.autogen import (
            AUTOGEN_V2_AVAILABLE,
            AUTOGEN_V4_AVAILABLE,
            SecureRetrieverFunction,
            create_secure_search_tool,
        )

        if AUTOGEN_V4_AVAILABLE:
            pytest.skip("AutoGen v0.4 is available")
        if not AUTOGEN_V2_AVAILABLE:
            pytest.skip("AutoGen v0.2 not available")

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        result = create_secure_search_tool(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        assert isinstance(result, SecureRetrieverFunction)


class TestSecureRAGAgent:
    """Tests for SecureRAGAgent class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = []
        return retriever

    def test_init_raises_when_not_available(self, mock_retriever):
        """Test init raises when AutoGen not available."""
        from ragguard.integrations.autogen import AUTOGEN_AVAILABLE, SecureRAGAgent

        if AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen is available")

        with pytest.raises(ImportError, match="autogen"):
            SecureRAGAgent(
                name="researcher",
                retriever=mock_retriever,
                user={"id": "alice"}
            )

    def test_set_user(self, mock_retriever):
        """Test set_user method."""
        from ragguard.integrations.autogen import AUTOGEN_AVAILABLE, SecureRAGAgent

        if not AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")

        # Patch the agent creation to avoid needing real OpenAI
        with patch('ragguard.integrations.autogen.SecureRAGAgent._init_v4_agent'), \
             patch('ragguard.integrations.autogen.SecureRAGAgent._init_v2_agent'):

            agent = SecureRAGAgent(
                name="researcher",
                retriever=mock_retriever,
                user={"id": "alice"}
            )
            agent.search_tool = MagicMock()

            result = agent.set_user({"id": "bob"})
            assert result is agent
            assert agent.user["id"] == "bob"

    def test_user_property(self, mock_retriever):
        """Test user property."""
        from ragguard.integrations.autogen import AUTOGEN_AVAILABLE, SecureRAGAgent

        if not AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")

        with patch('ragguard.integrations.autogen.SecureRAGAgent._init_v4_agent'), \
             patch('ragguard.integrations.autogen.SecureRAGAgent._init_v2_agent'):

            agent = SecureRAGAgent(
                name="researcher",
                retriever=mock_retriever,
                user={"id": "alice", "dept": "eng"}
            )

            assert agent.user["id"] == "alice"
            assert agent.user["dept"] == "eng"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test all __all__ exports are accessible."""
        from ragguard.integrations import autogen

        for name in autogen.__all__:
            assert hasattr(autogen, name), f"{name} not found"

    def test_availability_flags(self):
        """Test availability flags are booleans."""
        from ragguard.integrations.autogen import (
            AUTOGEN_AVAILABLE,
            AUTOGEN_V2_AVAILABLE,
            AUTOGEN_V4_AVAILABLE,
        )

        assert isinstance(AUTOGEN_AVAILABLE, bool)
        assert isinstance(AUTOGEN_V4_AVAILABLE, bool)
        assert isinstance(AUTOGEN_V2_AVAILABLE, bool)
        assert AUTOGEN_AVAILABLE == (AUTOGEN_V4_AVAILABLE or AUTOGEN_V2_AVAILABLE)
