"""
Tests for ragguard/integrations/crewai.py.

Tests the CrewAI integration for permission-aware document retrieval.
These tests mock the crewai dependency so they can run without crewai installed.
"""

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# Create mock crewai module before importing the integration
@pytest.fixture(autouse=True)
def mock_crewai():
    """Mock crewai module for testing without it installed."""
    # Create mock BaseTool class
    class MockBaseTool:
        name: str = ""
        description: str = ""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Create mock BaseModel
    class MockBaseModel:
        @classmethod
        def model_json_schema(cls):
            return {
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5}
                }
            }

    # Create mock Field
    def mock_field(*args, **kwargs):
        return kwargs.get('default', None)

    # Create mock modules
    mock_crewai = MagicMock()
    mock_crewai_tools = MagicMock()
    mock_crewai_tools.BaseTool = MockBaseTool
    mock_crewai.tools = mock_crewai_tools

    mock_pydantic = MagicMock()
    mock_pydantic.BaseModel = MockBaseModel
    mock_pydantic.Field = mock_field

    # Patch sys.modules
    with patch.dict(sys.modules, {
        'crewai': mock_crewai,
        'crewai.tools': mock_crewai_tools,
    }):
        # Reload the module to pick up the mocks
        if 'ragguard.integrations.crewai' in sys.modules:
            del sys.modules['ragguard.integrations.crewai']

        # Set CREWAI_AVAILABLE to True for testing
        yield


class TestCheckCrewAIAvailable:
    """Tests for _check_crewai_available function."""

    def test_crewai_available(self):
        """Test when CrewAI is available."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import _check_crewai_available
            # Should not raise
            _check_crewai_available()

    def test_crewai_not_available(self):
        """Test error when CrewAI is not installed."""
        # Force reload the module and test _check_crewai_available directly
        import ragguard.integrations.crewai as crewai_module

        # Save original value
        original_value = crewai_module.CREWAI_AVAILABLE

        try:
            # Set to False
            crewai_module.CREWAI_AVAILABLE = False

            with pytest.raises(ImportError) as exc:
                crewai_module._check_crewai_available()

            assert "crewai" in str(exc.value).lower()
        finally:
            # Restore original value
            crewai_module.CREWAI_AVAILABLE = original_value


class TestSecureRetrieverToolInit:
    """Tests for SecureRetrieverTool initialization."""

    def test_init_with_retriever(self):
        """Test initialization with existing retriever."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()

            tool = SecureRetrieverTool(
                retriever=mock_retriever,
                name="test_search",
                description="Test search tool"
            )

            assert tool.name == "test_search"
            assert tool._retriever is mock_retriever

    def test_init_with_qdrant_params(self):
        """Test initialization with Qdrant parameters."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        mock_client = MagicMock()
        mock_embed_fn = MagicMock(return_value=[0.1, 0.2, 0.3])

        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            # Patch at the retrievers module level where it's imported from
            with patch('ragguard.retrievers.QdrantSecureRetriever') as mock_qdrant:
                mock_qdrant.return_value = MagicMock()
                from ragguard.integrations.crewai import SecureRetrieverTool

                tool = SecureRetrieverTool(
                    qdrant_client=mock_client,
                    collection="test_collection",
                    policy=policy,
                    embedding_function=mock_embed_fn
                )

                mock_qdrant.assert_called_once()
                assert tool._embed_fn is mock_embed_fn

    def test_init_missing_params(self):
        """Test initialization with missing parameters raises error."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            with pytest.raises(ValueError) as exc:
                SecureRetrieverTool()

            assert "retriever" in str(exc.value).lower()

    def test_init_with_custom_description(self):
        """Test initialization with custom description."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()

            tool = SecureRetrieverTool(
                retriever=mock_retriever,
                description="Custom description"
            )

            assert tool.description == "Custom description"

    def test_init_with_audit_logger(self):
        """Test initialization with audit logger."""
        from ragguard import Policy
        from ragguard.audit import AuditLogger

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

        mock_logger = MagicMock(spec=AuditLogger)
        mock_client = MagicMock()
        mock_embed_fn = MagicMock(return_value=[0.1])

        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            # Patch at the retrievers module level where it's imported from
            with patch('ragguard.retrievers.QdrantSecureRetriever') as mock_qdrant:
                mock_qdrant.return_value = MagicMock()
                from ragguard.integrations.crewai import SecureRetrieverTool

                tool = SecureRetrieverTool(
                    qdrant_client=mock_client,
                    collection="test",
                    policy=policy,
                    embedding_function=mock_embed_fn,
                    audit_logger=mock_logger
                )

                # Verify audit_logger was passed
                call_kwargs = mock_qdrant.call_args.kwargs
                assert call_kwargs.get('audit_logger') is mock_logger


class TestSecureRetrieverToolUserContext:
    """Tests for user context management."""

    def test_set_user(self):
        """Test set_user method."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()
            tool = SecureRetrieverTool(retriever=mock_retriever)

            user = {"id": "alice", "roles": ["admin"]}
            result = tool.set_user(user)

            assert result is tool  # Returns self for chaining
            assert tool.get_user() == user

    def test_get_user_none(self):
        """Test get_user returns None when not set."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()
            tool = SecureRetrieverTool(retriever=mock_retriever)

            assert tool.get_user() is None

    def test_set_user_chaining(self):
        """Test set_user returns self for method chaining."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []

            tool = SecureRetrieverTool(retriever=mock_retriever)

            # Chain methods
            result = tool.set_user({"id": "bob"})
            assert result is tool


class TestSecureRetrieverToolRun:
    """Tests for _run method."""

    def test_run_without_user_raises(self):
        """Test _run raises error when user not set."""
        from ragguard.exceptions import RetrieverError

        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()
            tool = SecureRetrieverTool(retriever=mock_retriever)

            with pytest.raises(RetrieverError) as exc:
                tool._run("test query")

            assert "User context not set" in str(exc.value)

    def test_run_with_payload_result(self):
        """Test _run with result having payload attribute."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            # Mock result with payload
            mock_result = MagicMock()
            mock_result.payload = {"text": "Document content", "category": "test"}
            mock_result.score = 0.95

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "alice"})

            result = tool._run("test query", limit=5)

            assert "Found 1 document" in result
            assert "Document content" in result
            assert "0.95" in result
            mock_retriever.search.assert_called_once_with(
                query="test query",
                user={"id": "alice"},
                limit=5
            )

    def test_run_with_dict_result(self):
        """Test _run with dict result format."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_result = {
                "text": "Dict document",
                "metadata": {"source": "web"},
                "score": 0.8
            }

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "bob"})

            result = tool._run("query", limit=10)

            assert "Found 1 document" in result
            assert "Dict document" in result

    def test_run_with_content_field(self):
        """Test _run extracts from 'content' field when 'text' missing."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_result = MagicMock()
            mock_result.payload = {"content": "Content field data", "id": "doc1"}
            mock_result.score = 0.7

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("search")

            assert "Content field data" in result

    def test_run_with_empty_results(self):
        """Test _run with no results."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "alice"})

            result = tool._run("no matches")

            assert "No documents found" in result

    def test_run_with_fallback_result_format(self):
        """Test _run with result lacking standard attributes."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            # Mock result without payload or dict format
            mock_result = "Plain string result"

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("test")

            assert "Found 1 document" in result
            assert "Plain string result" in result

    def test_run_truncates_long_content(self):
        """Test _run truncates content over 500 chars."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            long_text = "A" * 600

            mock_result = MagicMock()
            mock_result.payload = {"text": long_text}
            mock_result.score = 0.9

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("test")

            assert "..." in result
            # Should have first 500 chars
            assert "A" * 500 in result

    def test_run_search_exception(self):
        """Test _run wraps exceptions in RetrieverError."""
        from ragguard.exceptions import RetrieverError

        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_retriever = MagicMock()
            mock_retriever.search.side_effect = Exception("Connection failed")

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "alice"})

            with pytest.raises(RetrieverError) as exc:
                tool._run("test")

            assert "Search failed" in str(exc.value)
            assert "Connection failed" in str(exc.value)

    def test_run_with_empty_payload(self):
        """Test _run with empty payload uses str representation."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_result = MagicMock()
            mock_result.payload = {}
            mock_result.score = 0.5

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("test")

            # Empty dict becomes "{}" as content
            assert "{}" in result

    def test_run_multiple_results(self):
        """Test _run formats multiple results correctly."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_result1 = MagicMock()
            mock_result1.payload = {"text": "First doc"}
            mock_result1.score = 0.9

            mock_result2 = MagicMock()
            mock_result2.payload = {"text": "Second doc"}
            mock_result2.score = 0.8

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result1, mock_result2]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("test")

            assert "Found 2 document" in result
            assert "[1]" in result
            assert "[2]" in result
            assert "First doc" in result
            assert "Second doc" in result


class TestSecureRetrieverToolDictResults:
    """Tests for dict result metadata extraction."""

    def test_dict_with_nested_metadata(self):
        """Test extracting from nested metadata.text."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_result = {
                "metadata": {"text": "Nested text"},
                "score": 0.75
            }

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("test")

            assert "Nested text" in result

    def test_dict_with_content_field(self):
        """Test extracting from 'content' field in dict."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_result = {
                "content": "Content from dict",
                "metadata": {},
                "score": 0.6
            }

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("test")

            assert "Content from dict" in result

    def test_dict_without_score(self):
        """Test dict result without score defaults to 0.0."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRetrieverTool

            mock_result = {
                "text": "Doc without score",
                "metadata": {}
            }

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]

            tool = SecureRetrieverTool(retriever=mock_retriever)
            tool.set_user({"id": "user1"})

            result = tool._run("test")

            assert "0.00" in result  # Default score


class TestSecureRAGTool:
    """Tests for SecureRAGTool alias class."""

    def test_default_name_and_description(self):
        """Test SecureRAGTool has correct defaults."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRAGTool

            mock_retriever = MagicMock()
            # SecureRAGTool uses its own default name/description
            tool = SecureRAGTool(retriever=mock_retriever, name="secure_rag_search")

            assert tool.name == "secure_rag_search"
            # Check the class-level description attribute
            assert "knowledge base" in SecureRAGTool.description.lower()

    def test_inherits_from_secure_retriever_tool(self):
        """Test SecureRAGTool inherits functionality."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureRAGTool, SecureRetrieverTool

            assert issubclass(SecureRAGTool, SecureRetrieverTool)


class TestCreateSecureRetrieverTool:
    """Tests for create_secure_retriever_tool function."""

    def test_basic_creation(self):
        """Test basic tool creation."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import create_secure_retriever_tool

            mock_retriever = MagicMock()

            tool = create_secure_retriever_tool(
                retriever=mock_retriever,
                name="custom_search",
                description="Custom description"
            )

            assert tool.name == "custom_search"
            assert tool.description == "Custom description"

    def test_creation_with_user(self):
        """Test tool creation with pre-set user."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import create_secure_retriever_tool

            mock_retriever = MagicMock()
            user = {"id": "alice", "roles": ["engineer"]}

            tool = create_secure_retriever_tool(
                retriever=mock_retriever,
                user=user
            )

            assert tool.get_user() == user

    def test_creation_without_user(self):
        """Test tool creation without user (user=None)."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import create_secure_retriever_tool

            mock_retriever = MagicMock()

            tool = create_secure_retriever_tool(
                retriever=mock_retriever
            )

            assert tool.get_user() is None

    def test_default_values(self):
        """Test default name and description."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import create_secure_retriever_tool

            mock_retriever = MagicMock()

            tool = create_secure_retriever_tool(retriever=mock_retriever)

            assert tool.name == "search_documents"
            assert "permission filtering" in tool.description.lower()


class TestSecureSearchInput:
    """Tests for SecureSearchInput schema."""

    def test_schema_fields(self):
        """Test SecureSearchInput has expected fields."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations.crewai import SecureSearchInput

            # Check the class exists and has expected attributes
            assert hasattr(SecureSearchInput, 'query') or hasattr(SecureSearchInput, '__annotations__')


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        with patch('ragguard.integrations.crewai.CREWAI_AVAILABLE', True):
            from ragguard.integrations import crewai

            expected = [
                "SecureRetrieverTool",
                "SecureRAGTool",
                "SecureSearchInput",
                "create_secure_retriever_tool",
            ]

            for name in expected:
                assert name in crewai.__all__
