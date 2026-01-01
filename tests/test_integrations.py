"""Extended tests for RAGGuard integrations to improve coverage."""

from unittest.mock import MagicMock, Mock, patch

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
# LlamaIndex Integration Tests
# ============================================================

class TestSecureLlamaIndexRetriever:
    """Tests for SecureLlamaIndexRetriever class."""

    def test_missing_llamaindex_import(self):
        """Test error when llama-index not installed."""
        with patch.dict('sys.modules', {'llama_index': None, 'llama_index.core': None, 'llama_index.core.retrievers': None}):
            # Force reimport
            import importlib

            import ragguard.integrations.llamaindex as llamaindex_mod
            from ragguard.exceptions import RetrieverError

            with pytest.raises(RetrieverError, match="llama-index not installed"):
                # This should fail since llama_index is not available
                importlib.reload(llamaindex_mod)
                llamaindex_mod.SecureLlamaIndexRetriever(
                    base_retriever=Mock(),
                    policy=make_policy([{"everyone": True}])
                )

    def test_invalid_retriever_type(self):
        """Test error when invalid retriever type provided."""
        # Mock the llama_index imports
        mock_base_retriever = MagicMock()

        with patch.dict('sys.modules', {
            'llama_index': MagicMock(),
            'llama_index.core': MagicMock(),
            'llama_index.core.retrievers': MagicMock(BaseRetriever=MagicMock)
        }):
            from ragguard.exceptions import RetrieverError
            from ragguard.integrations.llamaindex import SecureLlamaIndexRetriever

            # Regular object is not instance of mocked BaseRetriever
            with pytest.raises(RetrieverError, match="must be a LlamaIndex BaseRetriever"):
                SecureLlamaIndexRetriever(
                    base_retriever="not a retriever",
                    policy=make_policy([{"everyone": True}])
                )

    def test_retrieve_with_permissions(self):
        """Test retrieve method with permission filtering."""
        mock_base_retriever_cls = MagicMock()
        mock_base_retriever = MagicMock()

        # Create mock nodes
        mock_node1 = MagicMock()
        mock_node1.node.metadata = {"department": "engineering", "category": "internal"}
        mock_node2 = MagicMock()
        mock_node2.node.metadata = {"department": "sales", "category": "internal"}

        mock_base_retriever.retrieve.return_value = [mock_node1, mock_node2]

        with patch.dict('sys.modules', {
            'llama_index': MagicMock(),
            'llama_index.core': MagicMock(),
            'llama_index.core.retrievers': MagicMock(BaseRetriever=type(mock_base_retriever))
        }):
            from ragguard.integrations.llamaindex import SecureLlamaIndexRetriever

            policy = make_policy([{
                "everyone": True,
                "conditions": ["user.department == document.department"]
            }])

            retriever = SecureLlamaIndexRetriever.__new__(SecureLlamaIndexRetriever)
            retriever.base_retriever = mock_base_retriever
            from ragguard.policy.engine import PolicyEngine
            retriever.policy_engine = PolicyEngine(policy)
            # Use a mock audit logger to avoid actual logging
            mock_audit = MagicMock()
            retriever.audit_logger = mock_audit

            results = retriever.retrieve(
                "test query",
                user_context={"id": "alice", "department": "engineering"}
            )

            # Should only return engineering document
            assert len(results) == 1
            assert results[0].node.metadata["department"] == "engineering"

    def test_retrieve_without_user_context(self):
        """Test retrieve method without user context."""
        mock_base_retriever = MagicMock()
        mock_node = MagicMock()
        mock_node.node.metadata = {"category": "public"}
        mock_base_retriever.retrieve.return_value = [mock_node]

        with patch.dict('sys.modules', {
            'llama_index': MagicMock(),
            'llama_index.core': MagicMock(),
            'llama_index.core.retrievers': MagicMock(BaseRetriever=type(mock_base_retriever))
        }):
            from ragguard.integrations.llamaindex import SecureLlamaIndexRetriever

            policy = make_policy([{"everyone": True}])

            retriever = SecureLlamaIndexRetriever.__new__(SecureLlamaIndexRetriever)
            retriever.base_retriever = mock_base_retriever
            from ragguard.policy.engine import PolicyEngine
            retriever.policy_engine = PolicyEngine(policy)
            mock_audit = MagicMock()
            retriever.audit_logger = mock_audit

            results = retriever.retrieve("test query")
            assert len(results) == 1

    def test_internal_retrieve_method(self):
        """Test _retrieve method for BaseRetriever compatibility."""
        mock_base_retriever = MagicMock()
        mock_query_bundle = MagicMock()
        mock_base_retriever._retrieve.return_value = [MagicMock()]

        with patch.dict('sys.modules', {
            'llama_index': MagicMock(),
            'llama_index.core': MagicMock(),
            'llama_index.core.retrievers': MagicMock(BaseRetriever=type(mock_base_retriever))
        }):
            from ragguard.integrations.llamaindex import SecureLlamaIndexRetriever

            retriever = SecureLlamaIndexRetriever.__new__(SecureLlamaIndexRetriever)
            retriever.base_retriever = mock_base_retriever

            results = retriever._retrieve(mock_query_bundle)

            mock_base_retriever._retrieve.assert_called_once_with(mock_query_bundle)


class TestSecureQueryEngine:
    """Tests for SecureQueryEngine class."""

    def test_missing_llamaindex_import(self):
        """Test error when llama-index not installed."""
        with patch.dict('sys.modules', {'llama_index': None, 'llama_index.core': None}):
            import importlib

            import ragguard.integrations.llamaindex as llamaindex_mod
            from ragguard.exceptions import RetrieverError

            with pytest.raises(RetrieverError, match="llama-index not installed"):
                importlib.reload(llamaindex_mod)
                llamaindex_mod.SecureQueryEngine(
                    index=Mock(),
                    policy=make_policy([{"everyone": True}])
                )

    def test_invalid_index_type(self):
        """Test error when invalid index type provided."""
        with patch.dict('sys.modules', {
            'llama_index': MagicMock(),
            'llama_index.core': MagicMock(VectorStoreIndex=MagicMock)
        }):
            from ragguard.exceptions import RetrieverError
            from ragguard.integrations.llamaindex import SecureQueryEngine

            with pytest.raises(RetrieverError, match="must be a LlamaIndex VectorStoreIndex"):
                SecureQueryEngine(
                    index="not an index",
                    policy=make_policy([{"everyone": True}])
                )

    def test_query_method(self):
        """Test query method with permission filtering."""
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        mock_node = MagicMock()
        mock_node.node.metadata = {"category": "public"}
        mock_node.node.get_content.return_value = "Test content"
        mock_retriever.retrieve.return_value = [mock_node]

        mock_response = MagicMock()
        mock_node_with_score = MagicMock()

        with patch.dict('sys.modules', {
            'llama_index': MagicMock(),
            'llama_index.core': MagicMock(VectorStoreIndex=type(mock_index)),
            'llama_index.core.retrievers': MagicMock(BaseRetriever=type(mock_retriever)),
            'llama_index.core.schema': MagicMock(NodeWithScore=mock_node_with_score),
            'llama_index.core.response': MagicMock(),
            'llama_index.core.response.schema': MagicMock(Response=lambda **kwargs: mock_response)
        }):
            from ragguard.integrations.llamaindex import SecureQueryEngine

            policy = make_policy([{"everyone": True}])

            engine = SecureQueryEngine.__new__(SecureQueryEngine)
            engine.index = mock_index
            engine.policy = policy
            engine.similarity_top_k = 10
            # Use mock audit logger
            mock_audit = MagicMock()
            engine.audit_logger = mock_audit
            engine.query_engine_kwargs = {}

            response = engine.query(
                "test query",
                user_context={"id": "alice"}
            )

            # Verify retriever was created
            mock_index.as_retriever.assert_called_once_with(similarity_top_k=10)


class TestWrapRetrieverFunction:
    """Tests for wrap_retriever convenience function."""

    def test_wrap_retriever(self):
        """Test wrap_retriever function."""
        mock_base_retriever = MagicMock()

        with patch.dict('sys.modules', {
            'llama_index': MagicMock(),
            'llama_index.core': MagicMock(),
            'llama_index.core.retrievers': MagicMock(BaseRetriever=type(mock_base_retriever))
        }):
            from ragguard.integrations.llamaindex import SecureLlamaIndexRetriever, wrap_retriever

            policy = make_policy([{"everyone": True}])

            secure_retriever = wrap_retriever(mock_base_retriever, policy)

            # Should return SecureLlamaIndexRetriever instance
            assert isinstance(secure_retriever, SecureLlamaIndexRetriever)


# ============================================================
# LangGraph Integration Tests
# ============================================================

class TestLangGraphIntegration:
    """Tests for LangGraph integration."""

    @staticmethod
    def _reload_langgraph_module():
        """Helper to properly reload the langgraph integration module."""
        import importlib
        import sys
        # Remove cached module to force fresh import
        if 'ragguard.integrations.langgraph' in sys.modules:
            del sys.modules['ragguard.integrations.langgraph']
        import ragguard.integrations.langgraph as langgraph_mod
        return langgraph_mod

    def test_secure_retriever_node_missing_langgraph(self):
        """Test error when langgraph not installed."""
        import sys
        # Save original modules
        original_modules = {k: v for k, v in sys.modules.items()
                          if k.startswith('langgraph') or k.startswith('langchain')}

        try:
            # Remove langgraph modules and set to None to simulate not installed
            for key in list(sys.modules.keys()):
                if key.startswith('langgraph') or key.startswith('langchain'):
                    del sys.modules[key]

            with patch.dict('sys.modules', {'langgraph': None, 'langgraph.graph': None}):
                langgraph_mod = self._reload_langgraph_module()
                # Module should load but LANGGRAPH_AVAILABLE should be False
                assert not langgraph_mod.LANGGRAPH_AVAILABLE
        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def test_create_secure_retriever_node_with_retriever(self):
        """Test creating a secure retriever node with an existing retriever."""
        import sys
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_state_graph = MagicMock()
        mock_document = MagicMock()

        # Remove cached module first
        if 'ragguard.integrations.langgraph' in sys.modules:
            del sys.modules['ragguard.integrations.langgraph']

        with patch.dict('sys.modules', {
            'langgraph': MagicMock(),
            'langgraph.graph': MagicMock(StateGraph=mock_state_graph),
            'langchain_core': MagicMock(),
            'langchain_core.documents': MagicMock(Document=mock_document)
        }):
            langgraph_mod = self._reload_langgraph_module()
            SecureRetrieverNode = langgraph_mod.SecureRetrieverNode

            node = SecureRetrieverNode(retriever=mock_retriever)

            # Should be callable
            assert callable(node)

    def test_secure_retriever_node_execution(self):
        """Test executing the secure retriever node."""
        import sys
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.payload = {"text": "Test content", "category": "public"}
        mock_result.score = 0.95
        mock_retriever.search.return_value = [mock_result]

        mock_state_graph = MagicMock()
        mock_document = MagicMock()

        # Remove cached module first
        if 'ragguard.integrations.langgraph' in sys.modules:
            del sys.modules['ragguard.integrations.langgraph']

        with patch.dict('sys.modules', {
            'langgraph': MagicMock(),
            'langgraph.graph': MagicMock(StateGraph=mock_state_graph),
            'langchain_core': MagicMock(),
            'langchain_core.documents': MagicMock(Document=mock_document)
        }):
            langgraph_mod = self._reload_langgraph_module()
            SecureRetrieverNode = langgraph_mod.SecureRetrieverNode

            node = SecureRetrieverNode(retriever=mock_retriever)

            # Execute node with state
            state = {
                "query": "test query",
                "user": {"id": "alice"},
                "documents": [],
                "limit": 10,
                "metadata": {}
            }

            result = node(state)

            # Should have retrieved documents
            assert "documents" in result
            mock_retriever.search.assert_called_once()

    def test_secure_retriever_node_missing_query(self):
        """Test node execution with missing query."""
        import sys
        mock_retriever = MagicMock()
        mock_state_graph = MagicMock()
        mock_document = MagicMock()

        # Remove cached module first
        if 'ragguard.integrations.langgraph' in sys.modules:
            del sys.modules['ragguard.integrations.langgraph']

        with patch.dict('sys.modules', {
            'langgraph': MagicMock(),
            'langgraph.graph': MagicMock(StateGraph=mock_state_graph),
            'langchain_core': MagicMock(),
            'langchain_core.documents': MagicMock(Document=mock_document)
        }):
            langgraph_mod = self._reload_langgraph_module()
            SecureRetrieverNode = langgraph_mod.SecureRetrieverNode
            from ragguard.exceptions import RetrieverError

            node = SecureRetrieverNode(retriever=mock_retriever)

            with pytest.raises(RetrieverError, match="query"):
                node({"user": {"id": "alice"}})

    def test_secure_retriever_node_missing_user(self):
        """Test node execution with missing user context."""
        import sys
        mock_retriever = MagicMock()
        mock_state_graph = MagicMock()
        mock_document = MagicMock()

        # Remove cached module first
        if 'ragguard.integrations.langgraph' in sys.modules:
            del sys.modules['ragguard.integrations.langgraph']

        with patch.dict('sys.modules', {
            'langgraph': MagicMock(),
            'langgraph.graph': MagicMock(StateGraph=mock_state_graph),
            'langchain_core': MagicMock(),
            'langchain_core.documents': MagicMock(Document=mock_document)
        }):
            langgraph_mod = self._reload_langgraph_module()
            SecureRetrieverNode = langgraph_mod.SecureRetrieverNode
            from ragguard.exceptions import RetrieverError

            node = SecureRetrieverNode(retriever=mock_retriever)

            with pytest.raises(RetrieverError, match="user"):
                node({"query": "test"})

    def test_secure_retriever_tool(self):
        """Test SecureRetrieverTool."""
        import sys
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_state_graph = MagicMock()
        mock_document = MagicMock()

        # Remove cached module first
        if 'ragguard.integrations.langgraph' in sys.modules:
            del sys.modules['ragguard.integrations.langgraph']

        with patch.dict('sys.modules', {
            'langgraph': MagicMock(),
            'langgraph.graph': MagicMock(StateGraph=mock_state_graph),
            'langchain_core': MagicMock(),
            'langchain_core.documents': MagicMock(Document=mock_document)
        }):
            langgraph_mod = self._reload_langgraph_module()
            SecureRetrieverTool = langgraph_mod.SecureRetrieverTool

            tool = SecureRetrieverTool(
                retriever=mock_retriever,
                name="my_tool",
                description="My search tool"
            )

            assert tool.name == "my_tool"
            assert tool.description == "My search tool"

            # Test run method
            docs = tool.run("test query", {"id": "alice"}, 5)
            mock_retriever.search.assert_called_once()


# ============================================================
# AWS Bedrock Integration Tests
# ============================================================

class TestAWSBedrockIntegration:
    """Tests for AWS Bedrock integration."""

    def test_bedrock_retriever_class_exists(self):
        """Test BedrockKnowledgeBaseSecureRetriever class exists."""
        from ragguard.integrations import aws_bedrock

        # Check for class existence
        assert hasattr(aws_bedrock, 'BedrockKnowledgeBaseSecureRetriever')
        assert hasattr(aws_bedrock, 'BedrockKnowledgeBaseFilterBuilder')


# ============================================================
# Integration Module Init Tests
# ============================================================

class TestIntegrationsInit:
    """Tests for integrations __init__ module."""

    def test_import_llamaindex_when_available(self):
        """Test that LlamaIndex imports when available."""
        mock_module = MagicMock()

        with patch.dict('sys.modules', {
            'llama_index': mock_module,
            'llama_index.core': mock_module,
            'llama_index.core.retrievers': mock_module
        }):
            import importlib

            import ragguard.integrations as integrations_mod
            importlib.reload(integrations_mod)

    def test_import_langgraph_when_available(self):
        """Test that LangGraph imports when available."""
        mock_module = MagicMock()

        with patch.dict('sys.modules', {
            'langgraph': mock_module,
            'langgraph.graph': mock_module
        }):
            import importlib

            import ragguard.integrations as integrations_mod
            importlib.reload(integrations_mod)
