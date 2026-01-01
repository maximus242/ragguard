"""
Tests for integrations __init__ module to maximize coverage.
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest


class TestIsMissingDependency:
    """Tests for _is_missing_dependency function."""

    def test_no_module_named(self):
        """Test detection of 'no module named' error."""
        from ragguard.integrations import _is_missing_dependency

        error = ImportError("No module named 'some_package'")
        assert _is_missing_dependency(error) is True

    def test_cannot_import_name(self):
        """Test detection of 'cannot import name' error."""
        from ragguard.integrations import _is_missing_dependency

        error = ImportError("cannot import name 'SomeClass' from 'package'")
        assert _is_missing_dependency(error) is True

    def test_module_not_found(self):
        """Test detection of 'module not found' error."""
        from ragguard.integrations import _is_missing_dependency

        error = ImportError("module 'xyz' not found")
        assert _is_missing_dependency(error) is True

    def test_other_import_error(self):
        """Test non-missing-dependency error."""
        from ragguard.integrations import _is_missing_dependency

        error = ImportError("Some other import error without patterns")
        assert _is_missing_dependency(error) is False


class TestIntegrationsAllExports:
    """Tests for __all__ exports."""

    def test_all_is_list(self):
        """Test __all__ is a list."""
        from ragguard import integrations

        assert isinstance(integrations.__all__, list)

    def test_langchain_exports_when_available(self):
        """Test LangChain exports if available."""
        from ragguard import integrations

        # These may or may not be in __all__ depending on installation
        langchain_exports = [
            "LangChainSecureRetriever",
            "LangChainPgvectorSecureRetriever"
        ]

        for export in langchain_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_langgraph_exports_when_available(self):
        """Test LangGraph exports if available."""
        from ragguard import integrations

        langgraph_exports = [
            "RetrieverState",
            "SecureRetrieverNode",
            "SecureRetrieverTool",
            "create_secure_retriever_node",
            "create_secure_retriever_tool",
        ]

        for export in langgraph_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_llamaindex_exports_when_available(self):
        """Test LlamaIndex exports if available."""
        from ragguard import integrations

        llamaindex_exports = [
            "SecureLlamaIndexRetriever",
            "SecureQueryEngine",
            "wrap_retriever"
        ]

        for export in llamaindex_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_aws_bedrock_exports_when_available(self):
        """Test AWS Bedrock exports if available."""
        from ragguard import integrations

        bedrock_exports = [
            "BedrockKnowledgeBaseFilterBuilder",
            "BedrockKnowledgeBaseSecureRetriever"
        ]

        for export in bedrock_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_mcp_exports_when_available(self):
        """Test MCP exports if available."""
        from ragguard import integrations

        mcp_exports = [
            "MCPRetrieverTool",
            "RAGGuardMCPServer",
            "create_mcp_server"
        ]

        for export in mcp_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_crewai_exports_when_available(self):
        """Test CrewAI exports if available."""
        from ragguard import integrations

        crewai_exports = [
            "CrewAISecureRetrieverTool",
            "SecureRAGTool",
            "SecureSearchInput",
            "create_crewai_retriever_tool"
        ]

        for export in crewai_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_a2a_exports_when_available(self):
        """Test A2A exports if available."""
        from ragguard import integrations

        a2a_exports = [
            "RAGGuardA2AServer",
            "RAGGuardAgentExecutor",
            "create_a2a_server",
            "create_a2a_executor"
        ]

        for export in a2a_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_openai_assistants_exports_when_available(self):
        """Test OpenAI Assistants exports if available."""
        from ragguard import integrations

        openai_exports = [
            "SecureAssistantRetriever",
            "AssistantRunHandler",
            "create_secure_function_tool"
        ]

        for export in openai_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_autogen_exports_when_available(self):
        """Test AutoGen exports if available."""
        from ragguard import integrations

        autogen_exports = [
            "SecureSearchTool",
            "SecureRetrieverFunction",
            "SecureRAGAgent",
            "create_secure_search_tool"
        ]

        for export in autogen_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)

    def test_dspy_exports_when_available(self):
        """Test DSPy exports if available."""
        from ragguard import integrations

        dspy_exports = [
            "RAGGuardRM",
            "SecureRetrieve",
            "SecureRAG",
            "configure_ragguard_rm"
        ]

        for export in dspy_exports:
            if export in integrations.__all__:
                assert hasattr(integrations, export)


class TestIntegrationsModuleLogger:
    """Tests for module logger."""

    def test_logger_exists(self):
        """Test logger is defined."""
        from ragguard import integrations

        assert hasattr(integrations, 'logger')


class TestAllExportsAccessible:
    """Test all items in __all__ are accessible."""

    def test_all_exports_accessible(self):
        """Test every item in __all__ is accessible."""
        from ragguard import integrations

        for name in integrations.__all__:
            assert hasattr(integrations, name), f"Missing export: {name}"
            # Get the attribute to ensure it's not a broken import
            attr = getattr(integrations, name)
            assert attr is not None, f"Export is None: {name}"
