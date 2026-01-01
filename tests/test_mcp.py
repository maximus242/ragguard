"""
Tests for Model Context Protocol (MCP) integration.

Tests the MCP server that exposes permission-aware retrieval as tools
and document collections as resources.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from ragguard.policy.models import AllowConditions, Policy, Rule


def make_policy(rules, default="deny"):
    """Helper to create a policy from rules."""
    return Policy(
        version="1",
        rules=[
            Rule(
                name=f"rule_{i}",
                allow=AllowConditions(
                    roles=r.get("roles"),
                    everyone=r.get("everyone"),
                    conditions=r.get("conditions")
                ),
                match=r.get("match")
            )
            for i, r in enumerate(rules)
        ],
        default=default
    )


# ============================================================
# MCP Server Tests
# ============================================================

class TestRAGGuardMCPServer:
    """Tests for RAGGuardMCPServer class."""

    def test_mcp_not_available_error(self):
        """Test error when MCP SDK not installed."""
        with patch.dict('sys.modules', {'mcp': None, 'mcp.server': None}):
            import importlib

            import ragguard.integrations.mcp as mcp_mod

            # Force reimport
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import _check_mcp_available

            with pytest.raises(ImportError, match="mcp package"):
                _check_mcp_available()

    def test_server_initialization(self):
        """Test MCP server initialization."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock()
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            mock_retriever.collection = "test_docs"
            mock_retriever.backend_name = "test_backend"

            server = RAGGuardMCPServer(
                retriever=mock_retriever,
                name="test-server",
                description="Test RAGGuard MCP Server"
            )

            assert server.name == "test-server"
            assert server.description == "Test RAGGuard MCP Server"
            assert server.retriever == mock_retriever

    def test_create_mcp_server_function(self):
        """Test create_mcp_server convenience function."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock()
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer, create_mcp_server

            mock_retriever = MagicMock()

            server = create_mcp_server(
                retriever=mock_retriever,
                name="my-server",
                max_results=20
            )

            assert isinstance(server, RAGGuardMCPServer)
            assert server.max_results == 20


class TestMCPToolHandlers:
    """Tests for MCP tool handlers."""

    @pytest.mark.asyncio
    async def test_secure_search_handler(self):
        """Test secure_search tool handler."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [
                {"text": "Document 1", "category": "engineering"},
                {"text": "Document 2", "category": "engineering"}
            ]

            server = RAGGuardMCPServer(retriever=mock_retriever)

            # Call the search handler
            results = await server._handle_secure_search({
                "query": "test query",
                "user_id": "alice",
                "user_roles": ["engineer"],
                "limit": 5
            })

            # Verify search was called with correct params
            mock_retriever.search.assert_called_once()
            call_kwargs = mock_retriever.search.call_args[1]
            assert call_kwargs["query"] == "test query"
            assert call_kwargs["user"]["id"] == "alice"
            assert call_kwargs["user"]["roles"] == ["engineer"]
            assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_secure_search_missing_user_id(self):
        """Test secure_search with missing user_id."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            server = RAGGuardMCPServer(retriever=mock_retriever)

            results = await server._handle_secure_search({
                "query": "test query"
                # No user_id
            })

            # Should return error message
            assert len(results) == 1
            assert "user_id is required" in results[0]["text"]

    @pytest.mark.asyncio
    async def test_check_access_handler(self):
        """Test check_access tool handler."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            mock_retriever.check_access = MagicMock(return_value=True)

            server = RAGGuardMCPServer(retriever=mock_retriever)

            results = await server._handle_check_access({
                "document_id": "doc123",
                "user_id": "alice",
                "user_roles": ["admin"]
            })

            # Verify check_access was called
            mock_retriever.check_access.assert_called_once()

            # Parse result
            result_data = json.loads(results[0]["text"])
            assert result_data["document_id"] == "doc123"
            assert result_data["user_id"] == "alice"
            assert result_data["has_access"] is True

    @pytest.mark.asyncio
    async def test_check_access_fallback(self):
        """Test check_access fallback when check_access method not available."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock(spec=['search'])  # No check_access
            mock_retriever.search.return_value = [{"id": "doc123", "text": "content"}]

            server = RAGGuardMCPServer(retriever=mock_retriever)

            results = await server._handle_check_access({
                "document_id": "doc123",
                "user_id": "alice"
            })

            # Should use search fallback
            mock_retriever.search.assert_called_once()

            result_data = json.loads(results[0]["text"])
            assert result_data["has_access"] is True


class TestMCPResultFormatting:
    """Tests for result formatting."""

    def test_format_dict_results(self):
        """Test formatting dictionary results."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock()
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            server = RAGGuardMCPServer(retriever=mock_retriever)

            results = [
                {"text": "Document 1", "category": "eng", "score": 0.95},
                {"text": "Document 2", "category": "sales", "score": 0.85}
            ]

            formatted = server._format_results(results)

            assert len(formatted) == 2
            assert formatted[0]["content"] == "Document 1"
            assert formatted[0]["score"] == 0.95
            assert formatted[0]["metadata"]["category"] == "eng"
            assert "text" not in formatted[0]["metadata"]

    def test_format_qdrant_results(self):
        """Test formatting Qdrant ScoredPoint results."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock()
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            server = RAGGuardMCPServer(retriever=mock_retriever)

            # Create mock Qdrant-style results
            mock_result = MagicMock()
            mock_result.payload = {"text": "Qdrant document", "department": "engineering"}
            mock_result.score = 0.92

            formatted = server._format_results([mock_result])

            assert len(formatted) == 1
            assert formatted[0]["content"] == "Qdrant document"
            assert formatted[0]["score"] == 0.92
            assert formatted[0]["metadata"]["department"] == "engineering"

    def test_format_results_custom_text_key(self):
        """Test formatting with custom text key."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock()
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            server = RAGGuardMCPServer(retriever=mock_retriever, text_key="body")

            results = [
                {"body": "Custom body content", "type": "article"}
            ]

            formatted = server._format_results(results)

            assert formatted[0]["content"] == "Custom body content"


class TestMCPPermissionFiltering:
    """Tests for permission filtering through MCP."""

    @pytest.mark.asyncio
    async def test_role_based_filtering(self):
        """Test role-based permission filtering through MCP."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            # Mock retriever that filters by role
            def mock_search(query, user, limit=10, **kwargs):
                all_docs = [
                    {"text": "Admin doc", "required_role": "admin"},
                    {"text": "User doc", "required_role": "user"},
                    {"text": "Public doc", "required_role": None}
                ]
                user_roles = set(user.get("roles", []))
                return [
                    d for d in all_docs
                    if d["required_role"] is None or d["required_role"] in user_roles
                ]

            mock_retriever = MagicMock()
            mock_retriever.search.side_effect = mock_search

            server = RAGGuardMCPServer(retriever=mock_retriever)

            # Admin user should see all
            results = await server._handle_secure_search({
                "query": "test",
                "user_id": "admin_user",
                "user_roles": ["admin", "user"]
            })

            result_data = json.loads(results[0]["text"])
            assert result_data["total_results"] == 3

            # Regular user should see fewer
            results = await server._handle_secure_search({
                "query": "test",
                "user_id": "regular_user",
                "user_roles": ["user"]
            })

            result_data = json.loads(results[0]["text"])
            assert result_data["total_results"] == 2

    @pytest.mark.asyncio
    async def test_attribute_based_filtering(self):
        """Test attribute-based permission filtering through MCP."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            # Mock retriever that filters by department
            def mock_search(query, user, limit=10, **kwargs):
                all_docs = [
                    {"text": "Engineering doc", "department": "engineering"},
                    {"text": "Sales doc", "department": "sales"},
                    {"text": "HR doc", "department": "hr"}
                ]
                user_dept = user.get("department")
                return [d for d in all_docs if d["department"] == user_dept]

            mock_retriever = MagicMock()
            mock_retriever.search.side_effect = mock_search

            server = RAGGuardMCPServer(retriever=mock_retriever)

            # Engineering user
            results = await server._handle_secure_search({
                "query": "test",
                "user_id": "alice",
                "user_attributes": {"department": "engineering"}
            })

            result_data = json.loads(results[0]["text"])
            assert result_data["total_results"] == 1
            assert result_data["results"][0]["metadata"]["department"] == "engineering"


class TestMCPErrorHandling:
    """Tests for error handling in MCP integration."""

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test error handling during search."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            mock_retriever.search.side_effect = Exception("Database connection failed")

            server = RAGGuardMCPServer(retriever=mock_retriever)

            results = await server._handle_secure_search({
                "query": "test",
                "user_id": "alice"
            })

            assert "Error performing search" in results[0]["text"]
            assert "Database connection failed" in results[0]["text"]

    @pytest.mark.asyncio
    async def test_check_access_error_handling(self):
        """Test error handling during access check."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            mock_retriever.check_access = MagicMock(side_effect=Exception("Access check failed"))

            server = RAGGuardMCPServer(retriever=mock_retriever)

            results = await server._handle_check_access({
                "document_id": "doc123",
                "user_id": "alice"
            })

            assert "Error checking access" in results[0]["text"]

    @pytest.mark.asyncio
    async def test_missing_required_params(self):
        """Test handling of missing required parameters."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock(TextContent=lambda **kwargs: kwargs)
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import RAGGuardMCPServer

            mock_retriever = MagicMock()
            server = RAGGuardMCPServer(retriever=mock_retriever)

            # Missing document_id and user_id
            results = await server._handle_check_access({})

            assert "document_id and user_id are required" in results[0]["text"]


class TestMCPRetrieverTool:
    """Tests for MCPRetrieverTool decorator."""

    def test_decorator_wraps_retriever(self):
        """Test that decorator wraps retriever factory."""
        mock_mcp_server = MagicMock()

        with patch.dict('sys.modules', {
            'mcp': MagicMock(),
            'mcp.server': MagicMock(Server=MagicMock(return_value=mock_mcp_server)),
            'mcp.server.stdio': MagicMock(),
            'mcp.types': MagicMock()
        }):
            import importlib

            import ragguard.integrations.mcp as mcp_mod
            importlib.reload(mcp_mod)

            from ragguard.integrations.mcp import MCPRetrieverTool

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [{"text": "result"}]

            @MCPRetrieverTool(name="my_search", description="My search tool")
            def get_retriever():
                return mock_retriever

            # The decorator should return a callable
            assert callable(get_retriever)
            assert get_retriever.__name__ == "my_search"
