"""
Tests for filter builder edge cases and untested code paths.

Focuses on:
- EXISTS/NOT_EXISTS operators
- LITERAL_NONE handling
- USER_FIELD resolution in right values
- Edge cases across all backends
"""

import pytest

from ragguard.filters.builder import (
    to_azure_search_filter,
    to_chromadb_filter,
    to_elasticsearch_filter,
    to_milvus_filter,
    to_pgvector_filter,
    to_pinecone_filter,
    to_qdrant_filter,
    to_weaviate_filter,
)
from ragguard.policy import Policy

try:
    from qdrant_client import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class TestFieldExistenceOperators:
    """Test EXISTS and NOT_EXISTS operators across backends."""

    def test_exists_operator_qdrant(self):
        """Test field existence check in Qdrant."""
        if not QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "has_department",
                "allow": {
                    "conditions": ["document.department exists"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice"}
        filter_result = to_qdrant_filter(policy, user)

        # Should generate a filter checking for field existence
        assert filter_result is not None

    def test_not_exists_operator_qdrant(self):
        """Test field non-existence check in Qdrant."""
        if not QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "no_classification",
                "allow": {
                    "conditions": ["document.classification not exists"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice"}
        filter_result = to_qdrant_filter(policy, user)

        assert filter_result is not None


class TestLiteralNoneHandling:
    """Test LITERAL_NONE value handling."""

    def test_none_comparison_qdrant(self):
        """Test null/None handling in Qdrant."""
        if not QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

        # Qdrant doesn't support None in MatchValue
        # Use EXISTS operator instead for null checks
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "no_owner",
                "allow": {
                    "conditions": ["document.owner not exists"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice"}
        filter_result = to_qdrant_filter(policy, user)

        # Should generate filter for field non-existence
        assert filter_result is not None

    def test_none_in_condition_pgvector(self):
        """Test None/null field checks for pgvector."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "unassigned",
                "allow": {
                    "conditions": ["document.assignee not exists"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice"}
        where_clause, params = to_pgvector_filter(policy, user)

        # Should generate SQL - exact format depends on implementation
        assert where_clause is not None
        assert isinstance(params, list)


class TestUserFieldInRightValue:
    """Test user field resolution in right side of conditions."""

    def test_user_field_right_side_qdrant(self):
        """Test user.field on right side of comparison."""
        if not QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "same_department",
                "allow": {
                    "conditions": ["document.department == user.department"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice", "department": "engineering"}
        filter_result = to_qdrant_filter(policy, user)

        # Should resolve user.department to "engineering"
        assert filter_result is not None

    def test_user_field_right_side_pgvector(self):
        """Test user field on right side for pgvector."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "same_region",
                "allow": {
                    "conditions": ["document.region == user.region"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice", "region": "us-west"}
        where_clause, params = to_pgvector_filter(policy, user)

        # Should include "us-west" in parameters
        assert "us-west" in params


class TestComplexNestedConditions:
    """Test complex nested AND/OR conditions."""

    def test_nested_or_and_qdrant(self):
        """Test nested OR with AND conditions."""
        if not QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "complex_access",
                "allow": {
                    "conditions": [
                        "user.role == 'admin' OR (user.department == 'engineering' AND document.confidential == false)"
                    ]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice", "role": "user", "department": "engineering"}
        filter_result = to_qdrant_filter(policy, user)

        assert filter_result is not None

    def test_nested_and_or_pgvector(self):
        """Test nested AND with OR conditions."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "complex_access",
                "allow": {
                    "conditions": [
                        "user.clearance >= 3 AND (document.type == 'public' OR document.owner == user.id)"
                    ]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice", "clearance": 5}
        where_clause, params = to_pgvector_filter(policy, user)

        assert where_clause is not None
        assert len(params) > 0


class TestBackendSpecificEdgeCases:
    """Test backend-specific edge cases."""

    def test_weaviate_filter_generation(self):
        """Test Weaviate filter generation."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept_access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice", "department": "engineering"}
        filter_result = to_weaviate_filter(policy, user)

        # Weaviate filter should be a dict
        assert isinstance(filter_result, dict)

    def test_pinecone_filter_generation(self):
        """Test Pinecone filter generation."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "role_access",
                "match": {"type": "document"},  # Need match condition for filter
                "allow": {
                    "roles": ["admin", "editor"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["editor"]}
        filter_result = to_pinecone_filter(policy, user)

        # Pinecone filter should be a dict
        assert isinstance(filter_result, dict)

    def test_chromadb_filter_generation(self):
        """Test ChromaDB filter generation."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "visibility_access",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            }],
            "default": "deny"
        })

        user = {"id": "alice"}
        filter_result = to_chromadb_filter(policy, user)

        # ChromaDB filter should be a dict
        assert isinstance(filter_result, dict)

    def test_milvus_filter_generation(self):
        """Test Milvus filter generation."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept_access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        })

        user = {"id": "alice", "department": "engineering"}
        filter_result = to_milvus_filter(policy, user)

        # Milvus filter should be a string
        assert isinstance(filter_result, str)

    def test_elasticsearch_filter_generation(self):
        """Test Elasticsearch filter generation."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "status_access",
                "match": {"status": "published"},
                "allow": {"everyone": True}
            }],
            "default": "deny"
        })

        user = {"id": "alice"}
        filter_result = to_elasticsearch_filter(policy, user)

        # Elasticsearch filter should be a dict
        assert isinstance(filter_result, dict)

    def test_azure_search_filter_generation(self):
        """Test Azure Search filter generation."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "public_access",
                "match": {"is_public": True},
                "allow": {"everyone": True}
            }],
            "default": "deny"
        })

        user = {"id": "alice"}
        filter_result = to_azure_search_filter(policy, user)

        # Azure Search filter should be a string
        assert isinstance(filter_result, str)


class TestEmptyAndDenyAll:
    """Test edge cases with empty policies and deny-all scenarios."""

    def test_deny_all_policy_qdrant(self):
        """Test policy with single deny-all rule."""
        if not QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

        # Policy requires at least one rule - use a restrictive rule
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "admin_only",
                "allow": {
                    "roles": ["admin"]  # Very restrictive
                }
            }],
            "default": "deny"
        })

        # User is not admin
        user = {"id": "alice", "roles": ["user"]}
        filter_result = to_qdrant_filter(policy, user)

        # Should return a deny-all filter (matches nothing)
        # Implementation uses a sentinel value
        assert filter_result is not None
        assert isinstance(filter_result, qdrant_models.Filter)

    def test_empty_user_context(self):
        """Test with empty user context."""
        if not QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "public",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            }],
            "default": "deny"
        })

        user = {}  # Empty user context
        filter_result = to_qdrant_filter(policy, user)

        # Should still generate a filter (match on document fields)
        assert filter_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
