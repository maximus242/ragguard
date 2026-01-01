"""
Tests for the generic LangChain wrapper and integration with major vector databases.

Tests permission-aware retrieval with:
- Generic wrapper functionality
- ChromaDB
- Pinecone
- Weaviate
- Milvus
- Elasticsearch
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if langchain is not installed
pytest.importorskip("langchain")

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
# Generic Wrapper Tests
# ============================================================

class TestLangChainRetrieverWrapper:
    """Tests for the generic LangChain wrapper."""

    def test_wrapper_initialization(self):
        """Test basic wrapper initialization."""
        mock_retriever = MagicMock()

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)

        assert wrapper.retriever == mock_retriever
        assert wrapper.text_key == "text"
        assert wrapper.current_user is None

    def test_wrapper_custom_text_key(self):
        """Test wrapper with custom text key."""
        mock_retriever = MagicMock()

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(
            retriever=mock_retriever,
            text_key="content"
        )

        assert wrapper.text_key == "content"

    def test_wrapper_custom_content_keys(self):
        """Test wrapper with custom content keys."""
        mock_retriever = MagicMock()

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(
            retriever=mock_retriever,
            content_keys=["body", "description", "text"]
        )

        assert wrapper.content_keys == ["body", "description", "text"]

    def test_set_user(self):
        """Test setting user context."""
        mock_retriever = MagicMock()

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        user = {"id": "alice", "department": "engineering"}

        result = wrapper.set_user(user)

        assert wrapper.current_user == user
        assert result == wrapper  # Check method chaining

    def test_get_relevant_documents_with_user_kwarg(self):
        """Test retrieval with user passed as kwarg."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "Document 1", "department": "engineering", "score": 0.95},
            {"text": "Document 2", "department": "engineering", "score": 0.85}
        ]

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)

        docs = wrapper.invoke(
            "test query",
            user={"id": "alice", "department": "engineering"}
        )

        assert len(docs) == 2
        assert docs[0].page_content == "Document 1"
        assert docs[0].metadata["department"] == "engineering"
        mock_retriever.search.assert_called_once()

    def test_get_relevant_documents_with_set_user(self):
        """Test retrieval with user set via set_user()."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "Document 1", "score": 0.9}
        ]

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "bob", "roles": ["admin"]})

        docs = wrapper.invoke("test query")

        assert len(docs) == 1
        mock_retriever.search.assert_called_once()

    def test_get_relevant_documents_missing_user(self):
        """Test that retrieval fails without user context."""
        mock_retriever = MagicMock()

        from ragguard.exceptions import RetrieverError
        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)

        with pytest.raises(RetrieverError, match="User context required"):
            wrapper.invoke("test query")

    def test_extract_text_from_dict_result(self):
        """Test text extraction from dict results."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "Hello world", "category": "greeting"}
        ]

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")

        assert docs[0].page_content == "Hello world"
        assert "text" not in docs[0].metadata  # text should be extracted

    def test_extract_text_fallback_keys(self):
        """Test text extraction with fallback keys."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"content": "Fallback content", "category": "test"}
        ]

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")

        assert docs[0].page_content == "Fallback content"

    def test_extract_text_from_qdrant_result(self):
        """Test text extraction from Qdrant ScoredPoint."""
        mock_retriever = MagicMock()

        # Simulate Qdrant ScoredPoint
        mock_result = MagicMock()
        mock_result.payload = {"text": "Qdrant document", "department": "eng"}
        mock_result.score = 0.95

        mock_retriever.search.return_value = [mock_result]

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")

        assert docs[0].page_content == "Qdrant document"
        assert docs[0].metadata["score"] == 0.95
        assert docs[0].metadata["department"] == "eng"

    def test_extract_text_from_object_with_metadata(self):
        """Test text extraction from object with metadata attribute."""
        mock_retriever = MagicMock()

        # Simulate object with metadata
        mock_result = MagicMock(spec=[])  # No payload attribute
        mock_result.metadata = {"text": "Object document", "category": "test"}
        mock_result.score = 0.8

        mock_retriever.search.return_value = [mock_result]

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")

        assert docs[0].page_content == "Object document"
        assert docs[0].metadata["score"] == 0.8

    def test_wrap_retriever_function(self):
        """Test the wrap_retriever convenience function."""
        mock_retriever = MagicMock()

        from ragguard.integrations.langchain import LangChainRetrieverWrapper, wrap_retriever

        wrapper = wrap_retriever(mock_retriever, text_key="body")

        assert isinstance(wrapper, LangChainRetrieverWrapper)
        assert wrapper.text_key == "body"

    def test_limit_parameter_k(self):
        """Test that 'k' parameter is passed as limit."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        wrapper.invoke("test", k=5)

        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_limit_parameter_explicit(self):
        """Test that explicit 'limit' parameter is used."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        wrapper.invoke("test", limit=15)

        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs["limit"] == 15


# ============================================================
# ChromaDB Integration Tests
# ============================================================

class TestChromaDBLangChainIntegration:
    """Tests for ChromaDB with LangChain wrapper."""

    def test_chromadb_wrapper_with_permissions(self):
        """Test ChromaDB retriever with permission filtering through LangChain."""
        # Create mock ChromaDB retriever
        mock_chromadb_retriever = MagicMock()

        # Simulate filtered results (engineering only)
        mock_chromadb_retriever.search.return_value = [
            {"text": "Engineering doc 1", "department": "engineering", "score": 0.9},
            {"text": "Engineering doc 2", "department": "engineering", "score": 0.85}
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_chromadb_retriever)
        wrapper.set_user({"id": "alice", "department": "engineering"})

        docs = wrapper.invoke("company policies")

        assert len(docs) == 2
        for doc in docs:
            assert doc.metadata["department"] == "engineering"

    def test_chromadb_real_retriever_mock(self):
        """Test wrapping a mocked ChromaDB SecureRetriever."""
        # This simulates what happens with a real ChromaDB retriever
        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.department == document.department"]
        }])

        # Mock the ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock query results - raw results before permission filtering
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Eng doc", "Sales doc", "Eng doc 2"]],
            "metadatas": [[
                {"department": "engineering", "text": "Engineering document"},
                {"department": "sales", "text": "Sales document"},
                {"department": "engineering", "text": "Another engineering doc"}
            ]],
            "distances": [[0.1, 0.2, 0.15]]
        }

        # Create mock retriever that applies permissions
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            # Simulate permission filtering
            all_docs = [
                {"text": "Engineering document", "department": "engineering", "score": 0.9},
                {"text": "Sales document", "department": "sales", "score": 0.8},
                {"text": "Another engineering doc", "department": "engineering", "score": 0.85}
            ]
            # Filter by user's department
            return [d for d in all_docs if d["department"] == user.get("department")][:limit]

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        # Engineering user should only see engineering docs
        wrapper.set_user({"id": "alice", "department": "engineering"})
        docs = wrapper.invoke("documents")

        assert len(docs) == 2
        for doc in docs:
            assert doc.metadata["department"] == "engineering"

        # Sales user should only see sales docs
        wrapper.set_user({"id": "bob", "department": "sales"})
        docs = wrapper.invoke("documents")

        assert len(docs) == 1
        assert docs[0].metadata["department"] == "sales"


# ============================================================
# Pinecone Integration Tests
# ============================================================

class TestPineconeLangChainIntegration:
    """Tests for Pinecone with LangChain wrapper."""

    def test_pinecone_wrapper_with_permissions(self):
        """Test Pinecone retriever with permission filtering through LangChain."""
        mock_pinecone_retriever = MagicMock()

        # Simulate Pinecone-style results
        mock_pinecone_retriever.search.return_value = [
            {"text": "Confidential report", "category": "internal", "score": 0.95},
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_pinecone_retriever)
        wrapper.set_user({"id": "alice", "roles": ["employee"]})

        docs = wrapper.invoke("reports")

        assert len(docs) == 1
        assert docs[0].page_content == "Confidential report"

    def test_pinecone_with_namespace_and_permissions(self):
        """Test Pinecone retriever with namespace and permissions."""
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            # Simulate namespace-based filtering + permissions
            results = [
                {"text": "Project A doc", "project": "project_a", "owner": "alice"},
                {"text": "Project B doc", "project": "project_b", "owner": "bob"}
            ]
            # Filter by owner
            return [r for r in results if r["owner"] == user.get("id")]

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        wrapper.set_user({"id": "alice"})
        docs = wrapper.invoke("project docs")

        assert len(docs) == 1
        assert docs[0].metadata["project"] == "project_a"


# ============================================================
# Weaviate Integration Tests
# ============================================================

class TestWeaviateLangChainIntegration:
    """Tests for Weaviate with LangChain wrapper."""

    def test_weaviate_wrapper_with_permissions(self):
        """Test Weaviate retriever with permission filtering through LangChain."""
        mock_weaviate_retriever = MagicMock()

        # Simulate Weaviate-style results
        mock_weaviate_retriever.search.return_value = [
            {"text": "Product documentation", "visibility": "public", "score": 0.88},
            {"text": "Internal guide", "visibility": "internal", "score": 0.82}
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_weaviate_retriever)
        wrapper.set_user({"id": "guest", "roles": ["viewer"]})

        docs = wrapper.invoke("documentation")

        assert len(docs) == 2

    def test_weaviate_tenant_isolation(self):
        """Test Weaviate with tenant-based isolation."""
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            # Simulate tenant isolation
            all_docs = [
                {"text": "Tenant A data", "tenant": "company_a"},
                {"text": "Tenant B data", "tenant": "company_b"}
            ]
            return [d for d in all_docs if d["tenant"] == user.get("tenant")]

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        wrapper.set_user({"id": "alice", "tenant": "company_a"})
        docs = wrapper.invoke("data")

        assert len(docs) == 1
        assert docs[0].metadata["tenant"] == "company_a"


# ============================================================
# Milvus Integration Tests
# ============================================================

class TestMilvusLangChainIntegration:
    """Tests for Milvus with LangChain wrapper."""

    def test_milvus_wrapper_with_permissions(self):
        """Test Milvus retriever with permission filtering through LangChain."""
        mock_milvus_retriever = MagicMock()

        mock_milvus_retriever.search.return_value = [
            {"text": "Research paper", "category": "research", "score": 0.92}
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_milvus_retriever)
        wrapper.set_user({"id": "researcher", "roles": ["scientist"]})

        docs = wrapper.invoke("papers")

        assert len(docs) == 1
        assert docs[0].page_content == "Research paper"

    def test_milvus_partition_permissions(self):
        """Test Milvus with partition-based permissions."""
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            # Simulate partition-based access
            partitions = {
                "public": [{"text": "Public doc", "partition": "public"}],
                "private": [{"text": "Private doc", "partition": "private"}]
            }

            results = partitions.get("public", [])
            if "admin" in user.get("roles", []):
                results.extend(partitions.get("private", []))

            return results

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        # Regular user sees only public
        wrapper.set_user({"id": "bob", "roles": ["user"]})
        docs = wrapper.invoke("docs")
        assert len(docs) == 1
        assert docs[0].metadata["partition"] == "public"

        # Admin sees all
        wrapper.set_user({"id": "alice", "roles": ["admin"]})
        docs = wrapper.invoke("docs")
        assert len(docs) == 2


# ============================================================
# Elasticsearch Integration Tests
# ============================================================

class TestElasticsearchLangChainIntegration:
    """Tests for Elasticsearch with LangChain wrapper."""

    def test_elasticsearch_wrapper_with_permissions(self):
        """Test Elasticsearch retriever with permission filtering through LangChain."""
        mock_es_retriever = MagicMock()

        mock_es_retriever.search.return_value = [
            {"text": "Log entry 1", "level": "info", "_score": 1.5},
            {"text": "Log entry 2", "level": "debug", "_score": 1.2}
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_es_retriever)
        wrapper.set_user({"id": "ops", "roles": ["operator"]})

        docs = wrapper.invoke("logs")

        assert len(docs) == 2

    def test_elasticsearch_index_permissions(self):
        """Test Elasticsearch with index-level permissions."""
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            # Simulate index-based access control
            user_indices = user.get("allowed_indices", ["public"])
            all_docs = [
                {"text": "Public log", "index": "public"},
                {"text": "Sensitive log", "index": "security"},
                {"text": "Audit log", "index": "audit"}
            ]
            return [d for d in all_docs if d["index"] in user_indices]

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        # Regular user
        wrapper.set_user({"id": "alice", "allowed_indices": ["public"]})
        docs = wrapper.invoke("logs")
        assert len(docs) == 1
        assert docs[0].metadata["index"] == "public"

        # Security admin
        wrapper.set_user({"id": "bob", "allowed_indices": ["public", "security", "audit"]})
        docs = wrapper.invoke("logs")
        assert len(docs) == 3


# ============================================================
# Permission Filtering Integration Tests
# ============================================================

class TestPermissionFilteringIntegration:
    """Tests for permission filtering across different scenarios."""

    def test_role_based_access(self):
        """Test role-based access control."""
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            docs = [
                {"text": "Admin doc", "required_role": "admin"},
                {"text": "User doc", "required_role": "user"},
                {"text": "Public doc", "required_role": None}
            ]

            user_roles = set(user.get("roles", []))
            return [
                d for d in docs
                if d["required_role"] is None or d["required_role"] in user_roles
            ]

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        # Admin sees all
        wrapper.set_user({"id": "alice", "roles": ["admin", "user"]})
        docs = wrapper.invoke("docs")
        assert len(docs) == 3

        # User sees user + public
        wrapper.set_user({"id": "bob", "roles": ["user"]})
        docs = wrapper.invoke("docs")
        assert len(docs) == 2

        # Guest sees only public
        wrapper.set_user({"id": "guest", "roles": []})
        docs = wrapper.invoke("docs")
        assert len(docs) == 1

    def test_attribute_based_access(self):
        """Test attribute-based access control."""
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            docs = [
                {"text": "HR policy", "department": "hr", "clearance": 1},
                {"text": "Engineering spec", "department": "engineering", "clearance": 2},
                {"text": "Executive brief", "department": "executive", "clearance": 3}
            ]

            user_dept = user.get("department")
            user_clearance = user.get("clearance", 0)

            return [
                d for d in docs
                if (d["department"] == user_dept or d["clearance"] <= user_clearance)
            ]

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        # HR employee with low clearance
        wrapper.set_user({"id": "alice", "department": "hr", "clearance": 1})
        docs = wrapper.invoke("policies")
        assert len(docs) == 1
        assert docs[0].metadata["department"] == "hr"

        # Executive with high clearance
        wrapper.set_user({"id": "ceo", "department": "executive", "clearance": 3})
        docs = wrapper.invoke("policies")
        assert len(docs) == 3

    def test_owner_based_access(self):
        """Test owner-based access control."""
        mock_retriever = MagicMock()

        def mock_search(query, user, limit=10, **kwargs):
            docs = [
                {"text": "Alice's notes", "owner": "alice", "shared_with": []},
                {"text": "Bob's notes", "owner": "bob", "shared_with": ["alice"]},
                {"text": "Public notes", "owner": "system", "shared_with": ["*"]}
            ]

            user_id = user.get("id")
            return [
                d for d in docs
                if (
                    d["owner"] == user_id or
                    user_id in d["shared_with"] or
                    "*" in d["shared_with"]
                )
            ]

        mock_retriever.search.side_effect = mock_search

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)

        # Alice sees her own + shared + public
        wrapper.set_user({"id": "alice"})
        docs = wrapper.invoke("notes")
        assert len(docs) == 3

        # Bob sees his own + public
        wrapper.set_user({"id": "bob"})
        docs = wrapper.invoke("notes")
        assert len(docs) == 2

        # Carol sees only public
        wrapper.set_user({"id": "carol"})
        docs = wrapper.invoke("notes")
        assert len(docs) == 1


# ============================================================
# Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results(self):
        """Test handling of empty results."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("no matches")

        assert docs == []

    def test_missing_text_field(self):
        """Test handling of results without text field."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"id": "doc1", "category": "test"}  # No text field
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")

        assert len(docs) == 1
        # Should handle gracefully

    def test_none_metadata_values(self):
        """Test handling of None values in metadata."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "Doc with nulls", "category": None, "score": None}
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")

        assert len(docs) == 1
        assert docs[0].page_content == "Doc with nulls"

    def test_special_characters_in_query(self):
        """Test handling of special characters in query."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [{"text": "Result", "score": 0.9}]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)
        wrapper.set_user({"id": "alice"})

        # Query with special characters
        docs = wrapper.invoke("test query with 'quotes' and \"double quotes\" and <brackets>")

        mock_retriever.search.assert_called_once()

    def test_unicode_content(self):
        """Test handling of unicode content."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "Unicode: 你好世界 🌍 Ñoño", "language": "mixed"}
        ]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("unicode test")

        assert len(docs) == 1
        assert "你好世界" in docs[0].page_content
        assert "🌍" in docs[0].page_content

    def test_large_metadata(self):
        """Test handling of large metadata objects."""
        mock_retriever = MagicMock()

        large_metadata = {f"field_{i}": f"value_{i}" for i in range(100)}
        large_metadata["text"] = "Document content"

        mock_retriever.search.return_value = [large_metadata]

        from ragguard.integrations.langchain import wrap_retriever

        wrapper = wrap_retriever(mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")

        assert len(docs) == 1
        assert len(docs[0].metadata) >= 99  # Many fields


# ============================================================
# LangChainSecureRetriever Tests (Qdrant-specific)
# ============================================================

class TestLangChainSecureRetriever:
    """Tests for the Qdrant-specific LangChain retriever."""

    def test_initialization(self):
        """Test LangChainSecureRetriever initialization."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            MockQdrantRetriever.return_value = MagicMock()

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            assert retriever.collection == "test_collection"
            assert retriever.policy == policy
            assert retriever.current_user is None

    def test_set_user(self):
        """Test setting user context."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            MockQdrantRetriever.return_value = MagicMock()

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            user = {"id": "alice", "roles": ["engineer"]}
            result = retriever.set_user(user)

            assert retriever.current_user == user
            assert result == retriever  # Check method chaining

    def test_get_relevant_documents_with_user(self):
        """Test retrieval with user context."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            mock_ragguard_retriever = MagicMock()
            MockQdrantRetriever.return_value = mock_ragguard_retriever

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            # Mock the underlying retriever search
            mock_result = MagicMock()
            mock_result.payload = {"text": "Test document", "category": "public"}
            mock_result.score = 0.95
            mock_ragguard_retriever.search.return_value = [mock_result]

            retriever.set_user({"id": "alice"})
            docs = retriever.invoke("test query")

            assert len(docs) == 1
            assert docs[0].page_content == "Test document"
            assert docs[0].metadata["score"] == 0.95

    def test_get_relevant_documents_missing_user(self):
        """Test retrieval fails without user context."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            MockQdrantRetriever.return_value = MagicMock()

            from ragguard.exceptions import RetrieverError
            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            with pytest.raises(RetrieverError, match="User context required"):
                retriever.invoke("test query")

    def test_handle_qdrant_scored_point(self):
        """Test handling Qdrant ScoredPoint results."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            mock_ragguard_retriever = MagicMock()
            MockQdrantRetriever.return_value = mock_ragguard_retriever

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            # Mock ScoredPoint result
            mock_result = MagicMock()
            mock_result.payload = {"text": "Document content", "department": "engineering"}
            mock_result.score = 0.92
            mock_ragguard_retriever.search.return_value = [mock_result]

            retriever.set_user({"id": "alice"})
            docs = retriever.invoke("engineering docs")

            assert len(docs) == 1
            assert docs[0].page_content == "Document content"
            assert docs[0].metadata["department"] == "engineering"
            assert docs[0].metadata["score"] == 0.92

    def test_handle_query_response_format(self):
        """Test handling newer QueryResponse format."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            mock_ragguard_retriever = MagicMock()
            MockQdrantRetriever.return_value = mock_ragguard_retriever

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            # Mock QueryResponse result (no payload, has metadata)
            mock_result = MagicMock(spec=['metadata', 'score'])
            mock_result.metadata = {"text": "New API document", "category": "api"}
            mock_result.score = 0.88
            mock_ragguard_retriever.search.return_value = [mock_result]

            retriever.set_user({"id": "alice"})
            docs = retriever.invoke("api docs")

            assert len(docs) == 1
            assert docs[0].page_content == "New API document"

    def test_handle_fallback_format(self):
        """Test handling unknown result format."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            mock_ragguard_retriever = MagicMock()
            MockQdrantRetriever.return_value = mock_ragguard_retriever

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            # Mock unknown result format (no payload, no metadata)
            mock_result = MagicMock(spec=[])
            mock_ragguard_retriever.search.return_value = [mock_result]

            retriever.set_user({"id": "alice"})
            docs = retriever.invoke("unknown format")

            assert len(docs) == 1
            # Should handle gracefully with empty content

    def test_content_field_extraction(self):
        """Test extraction of 'content' field as text."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            mock_ragguard_retriever = MagicMock()
            MockQdrantRetriever.return_value = mock_ragguard_retriever

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            # Mock result with 'content' instead of 'text'
            mock_result = MagicMock()
            mock_result.payload = {"content": "Content field document", "type": "article"}
            mock_result.score = 0.85
            mock_ragguard_retriever.search.return_value = [mock_result]

            retriever.set_user({"id": "alice"})
            docs = retriever.invoke("articles")

            assert len(docs) == 1
            assert docs[0].page_content == "Content field document"

    def test_k_parameter_for_limit(self):
        """Test that 'k' parameter is used for limit."""
        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            mock_ragguard_retriever = MagicMock()
            MockQdrantRetriever.return_value = mock_ragguard_retriever
            mock_ragguard_retriever.search.return_value = []

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn
            )

            retriever.set_user({"id": "alice"})
            retriever.invoke("test", k=5)

            call_kwargs = mock_ragguard_retriever.search.call_args[1]
            assert call_kwargs["limit"] == 5

    def test_with_audit_logger(self):
        """Test LangChainSecureRetriever with audit logger."""
        from ragguard.audit.logger import AuditLogger

        with patch('ragguard.integrations.langchain.QdrantSecureRetriever') as MockQdrantRetriever:
            MockQdrantRetriever.return_value = MagicMock()

            from ragguard.integrations.langchain import LangChainSecureRetriever

            mock_client = MagicMock()
            mock_audit_logger = MagicMock(spec=AuditLogger)
            policy = make_policy([{"everyone": True}])
            embed_fn = MagicMock(return_value=[0.1] * 128)

            retriever = LangChainSecureRetriever(
                qdrant_client=mock_client,
                collection="test_collection",
                policy=policy,
                embedding_function=embed_fn,
                audit_logger=mock_audit_logger
            )

            assert retriever.audit_logger == mock_audit_logger


# ============================================================
# Additional Edge Cases
# ============================================================

class TestLangChainAdditionalEdgeCases:
    """Additional edge case tests for LangChain integration."""

    def test_langchain_unavailable_error(self):
        """Test error when LangChain is not available."""
        from ragguard.integrations.langchain import LANGCHAIN_AVAILABLE, _check_langchain_available

        # If LangChain is available, this should not raise
        if LANGCHAIN_AVAILABLE:
            _check_langchain_available()  # Should not raise

    def test_wrapper_result_conversion_error(self):
        """Test handling of unconvertible result types."""
        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        mock_retriever = MagicMock()

        # Result that can't be converted to dict
        class UnconvertibleResult:
            pass

        mock_retriever.search.return_value = [UnconvertibleResult()]

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        # Should handle gracefully
        docs = wrapper.invoke("test")
        assert len(docs) == 1

    def test_wrapper_raw_result_fallback(self):
        """Test that raw_result is used as fallback text."""
        from ragguard.integrations.langchain import LangChainRetrieverWrapper

        mock_retriever = MagicMock()

        # Result with no text fields, just raw data
        mock_result = MagicMock(spec=[])

        mock_retriever.search.return_value = [mock_result]

        wrapper = LangChainRetrieverWrapper(retriever=mock_retriever)
        wrapper.set_user({"id": "alice"})

        docs = wrapper.invoke("test")
        assert len(docs) == 1
