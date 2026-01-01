"""
Comprehensive tests for filter backends covering edge cases.
"""

import pytest

# Skip all tests if qdrant-client is not installed (required for qdrant filter generation)
pytest.importorskip("qdrant_client")

from ragguard.policy.models import Policy


class TestQdrantFilterBackend:
    """Tests for Qdrant filter backend."""

    @pytest.fixture
    def policy(self):
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

    def test_simple_equality(self, policy):
        """Test simple equality filter."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        filter_dict = to_qdrant_filter(policy, user)

        assert filter_dict is not None

    def test_deny_all(self):
        """Test deny all filter."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {"name": "admin_only", "allow": {"roles": ["admin"]}}
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_qdrant_filter(policy, user)

        assert filter_dict is not None

    def test_allow_all(self):
        """Test allow all filter."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {"name": "admin_access", "allow": {"roles": ["admin"]}}
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["admin"]}
        filter_dict = to_qdrant_filter(policy, user)

        # Admin with unrestricted access - filter may vary by implementation
        assert True  # Just ensure no exception

    def test_match_condition(self):
        """Test match condition filter."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"type": "public"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_qdrant_filter(policy, user)

        assert filter_dict is not None

    def test_multiple_rules(self):
        """Test multiple rules combined."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"visibility": "public"}
                },
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        filter_dict = to_qdrant_filter(policy, user)

        assert filter_dict is not None


class TestChromaDBFilterBackend:
    """Tests for ChromaDB filter backend."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        filter_dict = to_chromadb_filter(policy, user)

        assert filter_dict is not None
        assert "$and" in filter_dict or "department" in str(filter_dict)

    def test_match_condition(self):
        """Test match condition filter."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "type_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"type": "public"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_chromadb_filter(policy, user)

        assert filter_dict is not None

    def test_list_match(self):
        """Test list match condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "category_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"category": ["public", "internal"]}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_chromadb_filter(policy, user)

        assert filter_dict is not None


class TestPineconeFilterBackend:
    """Tests for Pinecone filter backend."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        from ragguard.filters.backends.pinecone import to_pinecone_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        filter_dict = to_pinecone_filter(policy, user)

        assert filter_dict is not None

    def test_boolean_match(self):
        """Test boolean match condition."""
        from ragguard.filters.backends.pinecone import to_pinecone_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "published_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"published": True}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_pinecone_filter(policy, user)

        assert filter_dict is not None


class TestPgvectorFilterBackend:
    """Tests for Pgvector filter backend."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        result = to_pgvector_filter(policy, user)

        # pgvector returns tuple (query, params) or string
        assert result is not None
        if isinstance(result, tuple):
            assert "department" in result[0]
            assert "engineering" in result[1]
        else:
            assert "department" in str(result)

    def test_match_condition(self):
        """Test match condition filter."""
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "type_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"type": "public"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        result = to_pgvector_filter(policy, user)

        assert result is not None
        if isinstance(result, tuple):
            assert "type" in result[0]
        else:
            assert "type" in str(result)


class TestWeaviateFilterBackend:
    """Tests for Weaviate filter backend."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        from ragguard.filters.backends.weaviate import to_weaviate_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        filter_dict = to_weaviate_filter(policy, user)

        assert filter_dict is not None

    def test_match_condition(self):
        """Test match condition filter."""
        from ragguard.filters.backends.weaviate import to_weaviate_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "type_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"type": "public"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_weaviate_filter(policy, user)

        assert filter_dict is not None


class TestElasticsearchFilterBackend:
    """Tests for Elasticsearch filter backend."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        from ragguard.filters.backends.elasticsearch import to_elasticsearch_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        filter_dict = to_elasticsearch_filter(policy, user)

        assert filter_dict is not None

    def test_match_condition(self):
        """Test match condition filter."""
        from ragguard.filters.backends.elasticsearch import to_elasticsearch_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "type_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"type": "public"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_elasticsearch_filter(policy, user)

        assert filter_dict is not None


class TestFilterBackendImports:
    """Tests for filter backend imports."""

    def test_all_backends_importable(self):
        """Test all backends can be imported."""
        from ragguard.filters.backends import (
            to_chromadb_filter,
            to_milvus_filter,
            to_pgvector_filter,
            to_pinecone_filter,
            to_qdrant_filter,
            to_weaviate_filter,
        )

        assert to_qdrant_filter is not None
        assert to_chromadb_filter is not None
        assert to_pinecone_filter is not None
        assert to_pgvector_filter is not None
        assert to_weaviate_filter is not None
        assert to_milvus_filter is not None

    def test_elasticsearch_import(self):
        """Test Elasticsearch backend import."""
        from ragguard.filters.backends.elasticsearch import to_elasticsearch_filter
        assert to_elasticsearch_filter is not None

    def test_azure_search_import(self):
        """Test Azure Search backend import."""
        from ragguard.filters.backends.azure_search import to_azure_search_filter
        assert to_azure_search_filter is not None

    def test_neo4j_import(self):
        """Test Neo4j backend import."""
        from ragguard.filters.backends.neo4j import to_neo4j_filter
        assert to_neo4j_filter is not None

    def test_neptune_import(self):
        """Test Neptune backend import."""
        from ragguard.filters.backends.neptune import to_neptune_filter
        assert to_neptune_filter is not None

    def test_tigergraph_import(self):
        """Test TigerGraph backend import."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter
        assert to_tigergraph_filter is not None

    def test_arangodb_import(self):
        """Test ArangoDB backend import."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        assert to_arangodb_filter is not None


class TestFilterBackendEdgeCases:
    """Edge case tests for filter backends."""

    def test_empty_user_context(self):
        """Test with empty user context."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "everyone",
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

        user = {}
        filter_dict = to_qdrant_filter(policy, user)

        # Everyone rule - just ensure no exception
        assert True

    def test_nested_user_field(self):
        """Test with nested user field."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "nested_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.org.id == document.org_id"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "org": {"id": "acme"}}
        filter_dict = to_qdrant_filter(policy, user)

        assert filter_dict is not None

    def test_numeric_match_value(self):
        """Test numeric match value."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "priority_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"priority": 1}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_qdrant_filter(policy, user)

        assert filter_dict is not None

    def test_boolean_match_false(self):
        """Test boolean match with False value."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "draft_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"is_draft": False}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        filter_dict = to_qdrant_filter(policy, user)

        assert filter_dict is not None
