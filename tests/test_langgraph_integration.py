"""
Comprehensive tests for langgraph integration coverage.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestSecureRetrieverNode:
    """Tests for SecureRetrieverNode."""

    def test_init_with_retriever(self):
        """Test initialization with existing retriever."""
        # Mock langraph availability
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                from ragguard.integrations.langgraph import SecureRetrieverNode

                mock_retriever = MagicMock()
                node = SecureRetrieverNode(retriever=mock_retriever)

                assert node._retriever == mock_retriever

    def test_init_missing_required_params(self):
        """Test initialization raises when missing required params."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                from ragguard.integrations.langgraph import SecureRetrieverNode

                with pytest.raises(ValueError, match="Must provide"):
                    SecureRetrieverNode()  # No params

    def test_call_missing_query(self):
        """Test __call__ raises when query missing."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                from ragguard.exceptions import RetrieverError
                from ragguard.integrations.langgraph import SecureRetrieverNode

                mock_retriever = MagicMock()
                node = SecureRetrieverNode(retriever=mock_retriever)

                with pytest.raises(RetrieverError, match="query"):
                    node({"user": {"id": "alice"}})

    def test_call_missing_user(self):
        """Test __call__ raises when user missing."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                from ragguard.exceptions import RetrieverError
                from ragguard.integrations.langgraph import SecureRetrieverNode

                mock_retriever = MagicMock()
                node = SecureRetrieverNode(retriever=mock_retriever)

                with pytest.raises(RetrieverError, match="user"):
                    node({"query": "test"})

    def test_call_with_qdrant_results(self):
        """Test __call__ with Qdrant-style results."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                with patch('ragguard.integrations.langgraph.Document') as MockDoc:
                    MockDoc.side_effect = lambda **kwargs: MagicMock(**kwargs)
                    from ragguard.integrations.langgraph import SecureRetrieverNode

                    mock_retriever = MagicMock()

                    # Create Qdrant-style result
                    mock_result = MagicMock()
                    mock_result.payload = {"text": "Test content", "id": "doc1"}
                    mock_result.score = 0.95
                    mock_retriever.search.return_value = [mock_result]

                    node = SecureRetrieverNode(retriever=mock_retriever)
                    result = node({
                        "query": "test",
                        "user": {"id": "alice"},
                        "limit": 5
                    })

                    assert "documents" in result
                    assert result["metadata"]["retrieved_count"] == 1

    def test_call_with_generic_results(self):
        """Test __call__ with generic-style results (metadata attribute)."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                with patch('ragguard.integrations.langgraph.Document') as MockDoc:
                    MockDoc.side_effect = lambda **kwargs: MagicMock(**kwargs)
                    from ragguard.integrations.langgraph import SecureRetrieverNode

                    mock_retriever = MagicMock()

                    # Create generic-style result with metadata (no payload)
                    mock_result = MagicMock(spec=['metadata', 'score'])
                    mock_result.metadata = {"content": "Test content", "id": "doc1"}
                    mock_result.score = 0.9
                    # Ensure hasattr(result, 'payload') returns False
                    delattr(mock_result, 'payload') if hasattr(mock_result, 'payload') else None
                    mock_retriever.search.return_value = [mock_result]

                    node = SecureRetrieverNode(retriever=mock_retriever)
                    result = node({
                        "query": "test",
                        "user": {"id": "alice"}
                    })

                    assert "documents" in result

    def test_call_with_fallback_results(self):
        """Test __call__ with fallback result format."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                with patch('ragguard.integrations.langgraph.Document') as MockDoc:
                    MockDoc.side_effect = lambda **kwargs: MagicMock(**kwargs)
                    from ragguard.integrations.langgraph import SecureRetrieverNode

                    mock_retriever = MagicMock()

                    # Create bare result without payload or metadata
                    mock_result = MagicMock(spec=[])  # No attributes
                    mock_retriever.search.return_value = [mock_result]

                    node = SecureRetrieverNode(retriever=mock_retriever)
                    result = node({
                        "query": "test",
                        "user": {"id": "alice"}
                    })

                    assert "documents" in result


class TestSecureRetrieverTool:
    """Tests for SecureRetrieverTool."""

    def test_tool_run(self):
        """Test tool run method."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                with patch('ragguard.integrations.langgraph.Document') as MockDoc:
                    MockDoc.side_effect = lambda **kwargs: MagicMock(**kwargs)
                    from ragguard.integrations.langgraph import SecureRetrieverTool

                    mock_retriever = MagicMock()
                    mock_retriever.search.return_value = []

                    tool = SecureRetrieverTool(
                        retriever=mock_retriever,
                        name="test_tool",
                        description="Test retriever tool"
                    )

                    result = tool.run(
                        query="test query",
                        user={"id": "alice"},
                        limit=5
                    )
                    assert result is not None
                    assert isinstance(result, list)


class TestCheckLangGraphAvailable:
    """Tests for _check_langgraph_available function."""

    def test_raises_when_not_available(self):
        """Test raises ImportError when LangGraph not available."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', False):
            # Need to reimport to get the function with mocked value
            import importlib

            import ragguard.integrations.langgraph as lg

            # Force the check to use our mocked value
            lg.LANGGRAPH_AVAILABLE = False

            with pytest.raises(ImportError, match="langgraph"):
                lg._check_langgraph_available()


class TestRetrieverState:
    """Tests for RetrieverState TypedDict."""

    def test_state_creation(self):
        """Test creating RetrieverState."""
        from ragguard.integrations.langgraph import RetrieverState

        state: RetrieverState = {
            "query": "test",
            "user": {"id": "alice"},
            "limit": 10
        }

        assert state["query"] == "test"
        assert state["user"]["id"] == "alice"


class TestSecureRetrieverToolAsTool:
    """Tests for SecureRetrieverTool.as_tool method."""

    def test_as_tool(self):
        """Test as_tool method returns LangChain tool."""
        with patch('ragguard.integrations.langgraph.LANGGRAPH_AVAILABLE', True):
            with patch('ragguard.integrations.langgraph._check_langgraph_available'):
                with patch('ragguard.integrations.langgraph.Document') as MockDoc:
                    MockDoc.side_effect = lambda **kwargs: MagicMock(**kwargs)

                    from ragguard.integrations.langgraph import SecureRetrieverTool

                    mock_retriever = MagicMock()
                    tool = SecureRetrieverTool(
                        retriever=mock_retriever,
                        name="test_tool",
                        description="Test tool"
                    )

                    # as_tool tries to use langchain.tools.tool decorator
                    # We just verify it exists and can be called
                    try:
                        langchain_tool = tool.as_tool()
                        assert langchain_tool is not None
                    except ImportError:
                        # Expected if langchain not installed
                        pass
                    except AttributeError:
                        # Expected if langchain.tools.tool not available
                        pass
