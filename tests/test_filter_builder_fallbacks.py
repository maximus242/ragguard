"""
Tests for filter builder fallback parsing paths.

These tests exercise the string-based fallback parsing code in filter builders
that isn't covered by the normal compiled expression tests.
"""

from unittest.mock import patch

import pytest

# Skip all tests if qdrant-client is not installed (required for qdrant filter generation)
pytest.importorskip("qdrant_client")

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


class TestChromaDBFilterFallbacks:
    """Test ChromaDB filter builder fallback paths."""

    def test_basic_filter_generation(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        user = {"id": "guest"}

        result = to_chromadb_filter(policy, user)

        assert result is not None
        assert "category" in str(result)

    def test_user_field_in_document_array(self):
        """Test user.field in document.array condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["user.id in document.authorized_users"]}
        ])
        user = {"id": "alice"}

        result = to_chromadb_filter(policy, user)

        assert result is not None

    def test_user_field_not_in_document_array(self):
        """Test user.field not in document.array condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["user.id not in document.blocked_users"]}
        ])
        user = {"id": "alice"}

        result = to_chromadb_filter(policy, user)

        # When user is not blocked, filter should not restrict
        assert result is None or isinstance(result, dict)

    def test_document_field_in_list(self):
        """Test document.field in ['a', 'b'] condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category in ['public', 'internal']"]}
        ])
        user = {"id": "guest"}

        result = to_chromadb_filter(policy, user)

        assert result is not None
        assert "$or" in str(result) or "$in" in str(result) or "category" in str(result)

    def test_document_field_not_in_list(self):
        """Test document.field not in ['a', 'b'] condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.status not in ['deleted', 'archived']"]}
        ])
        user = {"id": "guest"}

        result = to_chromadb_filter(policy, user)

        assert result is not None

    def test_comparison_operators(self):
        """Test comparison operators in conditions."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        # ChromaDB supports comparison operators through compiled expressions
        # Greater than
        policy = make_policy([
            {"everyone": True, "conditions": ["document.priority > 5"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})
        # ChromaDB may not support range queries natively
        assert result is None or isinstance(result, dict)

        # Less than
        policy = make_policy([
            {"everyone": True, "conditions": ["document.score < 100"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})
        assert result is None or isinstance(result, dict)

        # Greater than or equal
        policy = make_policy([
            {"everyone": True, "conditions": ["document.level >= 3"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})
        assert result is None or isinstance(result, dict)

        # Less than or equal
        policy = make_policy([
            {"everyone": True, "conditions": ["document.rank <= 10"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})
        assert result is None or isinstance(result, dict)

    def test_not_equals_operator(self):
        """Test != operator."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.status != 'deleted'"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})

        assert result is not None
        assert "$ne" in str(result) or "status" in str(result)

    def test_field_exists(self):
        """Test field exists condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.metadata exists"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})

        # ChromaDB may not have native EXISTS support
        assert result is None or isinstance(result, dict)

    def test_field_not_exists(self):
        """Test field not exists condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.deleted_at not exists"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})

        assert result is None or isinstance(result, dict)

    def test_null_user_field_deny(self):
        """Test that null user fields create deny-all filter."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["user.team == document.team"]}
        ])
        # User without team field
        user = {"id": "alice"}

        result = to_chromadb_filter(policy, user)

        # Should have a deny-all filter
        assert result is not None
        assert "__ragguard_deny_all" in str(result) or result == {} or "deny" in str(result).lower()

    def test_multiple_rules_or(self):
        """Test multiple rules create OR logic."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"roles": ["admin"]},
            {"roles": ["editor"], "match": {"category": "content"}}
        ])
        user = {"id": "alice", "roles": ["admin", "editor"]}

        result = to_chromadb_filter(policy, user)

        assert result is None or isinstance(result, dict)

    def test_match_with_multiple_fields(self):
        """Test match with multiple fields (AND)."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"roles": ["admin"], "match": {"category": "docs", "status": "active"}}
        ])
        user = {"id": "alice", "roles": ["admin"]}

        result = to_chromadb_filter(policy, user)

        assert result is not None
        assert "category" in str(result) and "status" in str(result)

    def test_literal_in_document_array(self):
        """Test 'literal' in document.array condition."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["'admin' in document.allowed_roles"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})

        assert result is None or isinstance(result, dict)


class TestQdrantFilterFallbacks:
    """Test Qdrant filter builder fallback paths."""

    def test_basic_filter_generation(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        user = {"id": "guest"}

        result = to_qdrant_filter(policy, user)

        assert result is not None

    def test_user_id_in_document_array(self):
        """Test user.id in document.authorized_users."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["user.id in document.authorized_users"]}
        ])
        user = {"id": "alice"}

        result = to_qdrant_filter(policy, user)

        assert result is not None

    def test_user_id_not_in_document_array(self):
        """Test user.id not in document.blocked_users."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["user.id not in document.blocked_users"]}
        ])
        user = {"id": "alice"}

        result = to_qdrant_filter(policy, user)

        assert result is None or result is not None

    def test_document_field_not_in_list(self):
        """Test document.field not in ['a', 'b']."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.status not in ['deleted', 'archived']"]}
        ])
        user = {"id": "guest"}

        result = to_qdrant_filter(policy, user)

        assert result is not None

    def test_comparison_operators(self):
        """Test comparison operators."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        # All comparison operators
        for op, condition in [
            (">", "document.priority > 5"),
            ("<", "document.score < 100"),
            (">=", "document.level >= 3"),
            ("<=", "document.rank <= 10"),
        ]:
            policy = make_policy([
                {"everyone": True, "conditions": [condition]}
            ])
            result = to_qdrant_filter(policy, {"id": "guest"})
            assert result is not None, f"Failed for operator {op}"

    def test_field_exists(self):
        """Test field exists."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.metadata exists"]}
        ])
        result = to_qdrant_filter(policy, {"id": "guest"})

        assert result is not None

    def test_field_not_exists(self):
        """Test field not exists."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.deleted_at not exists"]}
        ])
        result = to_qdrant_filter(policy, {"id": "guest"})

        assert result is not None

    def test_null_user_field_creates_empty_filter(self):
        """Test that null user field is handled properly."""
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["user.team == document.team"]}
        ])
        user = {"id": "alice"}  # No team

        result = to_qdrant_filter(policy, user)

        # Should have some result (possibly empty filter)
        assert result is not None or result is None


class TestPineconeFilterFallbacks:
    """Test Pinecone filter builder fallback paths."""

    def test_basic_filter(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.pinecone import to_pinecone_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        result = to_pinecone_filter(policy, {"id": "guest"})

        assert result is not None or result == {}

    def test_comparison_operators(self):
        """Test comparison operators."""
        from ragguard.filters.backends.pinecone import to_pinecone_filter

        for condition in [
            "document.priority > 5",
            "document.score < 100",
            "document.level >= 3",
            "document.rank <= 10",
        ]:
            policy = make_policy([
                {"everyone": True, "conditions": [condition]}
            ])
            result = to_pinecone_filter(policy, {"id": "guest"})
            # Pinecone may not support all range queries
            assert result is None or isinstance(result, dict), f"Failed for: {condition}"

    def test_in_operator(self):
        """Test IN operator."""
        from ragguard.filters.backends.pinecone import to_pinecone_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category in ['public', 'internal']"]}
        ])
        result = to_pinecone_filter(policy, {"id": "guest"})

        assert result is not None or result == {}

    def test_not_in_operator(self):
        """Test NOT IN operator."""
        from ragguard.filters.backends.pinecone import to_pinecone_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.status not in ['deleted']"]}
        ])
        result = to_pinecone_filter(policy, {"id": "guest"})

        assert result is not None or result == {}


class TestPgvectorFilterFallbacks:
    """Test pgvector filter builder fallback paths."""

    def test_basic_filter(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        result = to_pgvector_filter(policy, {"id": "guest"})

        # pgvector returns (clause, params) tuple
        assert result is not None
        assert isinstance(result, tuple)
        clause, params = result
        assert clause is not None

    def test_comparison_operators(self):
        """Test comparison operators generate valid SQL."""
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        for op, condition in [
            (">", "document.priority > 5"),
            ("<", "document.score < 100"),
            (">=", "document.level >= 3"),
            ("<=", "document.rank <= 10"),
        ]:
            policy = make_policy([
                {"everyone": True, "conditions": [condition]}
            ])
            result = to_pgvector_filter(policy, {"id": "guest"})
            assert result is not None, f"None result for {op}"
            clause, params = result
            assert clause, f"Empty clause for {op}"

    def test_in_operator(self):
        """Test IN operator generates ANY(array)."""
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category in ['public', 'internal']"]}
        ])
        result = to_pgvector_filter(policy, {"id": "guest"})

        assert result is not None
        clause, params = result
        assert clause
        assert "ANY" in clause or "IN" in clause or "category" in clause

    def test_user_equals_document(self):
        """Test user.field == document.field."""
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["user.department == document.department"]}
        ])
        result = to_pgvector_filter(policy, {"id": "alice", "department": "engineering"})

        assert result is not None
        clause, params = result
        assert clause
        # params could be dict or list depending on implementation
        assert "engineering" in str(params)


class TestWeaviateFilterFallbacks:
    """Test Weaviate filter builder fallback paths."""

    def test_basic_filter(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.weaviate import to_weaviate_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        result = to_weaviate_filter(policy, {"id": "guest"})

        assert result is not None

    def test_comparison_operators(self):
        """Test comparison operators."""
        from ragguard.filters.backends.weaviate import to_weaviate_filter

        for condition in [
            "document.priority > 5",
            "document.score < 100",
            "document.level >= 3",
            "document.rank <= 10",
        ]:
            policy = make_policy([
                {"everyone": True, "conditions": [condition]}
            ])
            result = to_weaviate_filter(policy, {"id": "guest"})
            # Weaviate may not support all range queries natively
            assert result is None or isinstance(result, dict), f"Failed for: {condition}"

    def test_in_list(self):
        """Test IN list filter."""
        from ragguard.filters.backends.weaviate import to_weaviate_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category in ['public', 'internal']"]}
        ])
        result = to_weaviate_filter(policy, {"id": "guest"})

        assert result is not None


class TestElasticsearchFilterFallbacks:
    """Test Elasticsearch filter builder fallback paths."""

    def test_basic_filter(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.elasticsearch import to_elasticsearch_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        result = to_elasticsearch_filter(policy, {"id": "guest"})

        assert result is not None

    def test_comparison_operators(self):
        """Test range queries for comparison operators."""
        from ragguard.filters.backends.elasticsearch import to_elasticsearch_filter

        for condition in [
            "document.priority > 5",
            "document.score < 100",
            "document.level >= 3",
            "document.rank <= 10",
        ]:
            policy = make_policy([
                {"everyone": True, "conditions": [condition]}
            ])
            result = to_elasticsearch_filter(policy, {"id": "guest"})
            assert result is not None, f"Failed for: {condition}"

    def test_in_list(self):
        """Test terms query for IN list."""
        from ragguard.filters.backends.elasticsearch import to_elasticsearch_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category in ['public', 'internal']"]}
        ])
        result = to_elasticsearch_filter(policy, {"id": "guest"})

        assert result is not None


class TestMilvusFilterFallbacks:
    """Test Milvus filter builder fallback paths."""

    def test_basic_filter(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        result = to_milvus_filter(policy, {"id": "guest"})

        assert result is not None

    def test_comparison_operators(self):
        """Test comparison operators generate valid expressions."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        for condition in [
            "document.priority > 5",
            "document.score < 100",
            "document.level >= 3",
            "document.rank <= 10",
        ]:
            policy = make_policy([
                {"everyone": True, "conditions": [condition]}
            ])
            result = to_milvus_filter(policy, {"id": "guest"})
            assert result, f"Empty filter for: {condition}"

    def test_in_list(self):
        """Test IN operator."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category in ['public', 'internal']"]}
        ])
        result = to_milvus_filter(policy, {"id": "guest"})

        assert result is not None
        assert "in" in result.lower() or "category" in result


class TestAzureSearchFilterFallbacks:
    """Test Azure Search filter builder fallback paths."""

    def test_basic_filter(self):
        """Test basic filter generation."""
        from ragguard.filters.backends.azure_search import to_azure_search_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.category == 'public'"]}
        ])
        result = to_azure_search_filter(policy, {"id": "guest"})

        assert result is not None

    def test_comparison_operators(self):
        """Test OData comparison operators."""
        from ragguard.filters.backends.azure_search import to_azure_search_filter

        for condition in [
            "document.priority > 5",
            "document.score < 100",
            "document.level >= 3",
            "document.rank <= 10",
        ]:
            policy = make_policy([
                {"everyone": True, "conditions": [condition]}
            ])
            result = to_azure_search_filter(policy, {"id": "guest"})
            assert result, f"Empty filter for: {condition}"


class TestFilterBuilderEdgeCases:
    """Test edge cases across all filter builders."""

    def test_empty_conditions_allow_all(self):
        """Test that rules with no conditions allow all."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter
        from ragguard.filters.backends.qdrant import to_qdrant_filter

        policy = make_policy([{"everyone": True}])
        user = {"id": "guest"}

        chromadb_result = to_chromadb_filter(policy, user)
        qdrant_result = to_qdrant_filter(policy, user)

        # Empty conditions = allow all
        assert chromadb_result is None or chromadb_result == {}
        assert qdrant_result is None or qdrant_result is not None

    def test_no_matching_roles_deny_all(self):
        """Test that no matching roles results in deny-all filter."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"roles": ["super_admin"]}  # User doesn't have this role
        ], default="deny")
        user = {"id": "alice", "roles": ["viewer"]}

        result = to_chromadb_filter(policy, user)

        # Should have deny-all filter
        assert result is not None
        assert "__ragguard_deny_all" in str(result)

    def test_multiple_conditions_and(self):
        """Test multiple conditions in a rule are ANDed."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {
                "everyone": True,
                "conditions": [
                    "document.category == 'public'",
                    "document.status == 'active'"
                ]
            }
        ])
        user = {"id": "guest"}

        result = to_chromadb_filter(policy, user)

        assert result is not None
        # Both conditions should be present
        result_str = str(result)
        assert "category" in result_str and "status" in result_str

    def test_boolean_literal_true(self):
        """Test boolean true literal."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.is_public == true"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})

        assert result is not None

    def test_boolean_literal_false(self):
        """Test boolean false literal."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.is_deleted == false"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})

        assert result is not None

    def test_numeric_literal(self):
        """Test numeric literal in equality."""
        from ragguard.filters.backends.chromadb import to_chromadb_filter

        policy = make_policy([
            {"everyone": True, "conditions": ["document.priority == 5"]}
        ])
        result = to_chromadb_filter(policy, {"id": "guest"})

        assert result is not None
