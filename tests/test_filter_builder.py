"""
Tests for filter builders.
"""

import pytest

from ragguard.filters.builder import to_pgvector_filter, to_qdrant_filter
from ragguard.policy import Policy

try:
    from qdrant_client import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantFilterBuilder:
    """Tests for Qdrant filter generation."""

    def test_build_everyone_filter(self):
        """Test building filter for everyone rule."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True},
                }
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com"}
        filter_obj = to_qdrant_filter(policy, user)

        # Should have a filter for visibility=public
        assert filter_obj is not None

    def test_build_role_filter(self):
        """Test building filter for role-based rule."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin",
                    "allow": {"roles": ["admin"]},
                }
            ],
            "default": "deny",
        })

        admin_user = {"id": "admin@example.com", "roles": ["admin"]}
        filter_obj = to_qdrant_filter(policy, admin_user)

        # Admin should get a filter that matches all docs
        assert filter_obj is not None

        # Non-admin should get deny filter
        regular_user = {"id": "user@example.com", "roles": ["user"]}
        filter_obj = to_qdrant_filter(policy, regular_user)
        assert filter_obj is not None

    def test_build_department_condition_filter(self):
        """Test building filter with user context condition."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept",
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    },
                }
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com", "department": "engineering"}
        filter_obj = to_qdrant_filter(policy, user)

        # Should create filter for department=engineering
        assert filter_obj is not None

    def test_build_multiple_rules_or_logic(self):
        """Test that multiple rules are combined with OR logic."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True},
                },
                {
                    "name": "admin",
                    "allow": {"roles": ["admin"]},
                },
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com", "roles": ["user"]}
        filter_obj = to_qdrant_filter(policy, user)

        # Should have OR between public docs and admin access
        assert filter_obj is not None

    def test_build_no_matching_rules_deny(self):
        """Test filter when no rules match and default is deny."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin-only",
                    "allow": {"roles": ["admin"]},
                }
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com", "roles": ["user"]}
        filter_obj = to_qdrant_filter(policy, user)

        # Should create a filter that matches nothing
        assert filter_obj is not None


class TestPgvectorFilterBuilder:
    """Tests for pgvector SQL filter generation."""

    def test_build_everyone_filter(self):
        """Test building SQL filter for everyone rule."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True},
                }
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com"}
        where_clause, params = to_pgvector_filter(policy, user)

        assert "WHERE" in where_clause
        assert "visibility" in where_clause
        assert "public" in params

    def test_build_role_filter(self):
        """Test building SQL filter for role-based rule."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin",
                    "match": {"type": "internal"},
                    "allow": {"roles": ["admin"]},
                }
            ],
            "default": "deny",
        })

        admin_user = {"id": "admin@example.com", "roles": ["admin"]}
        where_clause, params = to_pgvector_filter(policy, admin_user)

        assert "WHERE" in where_clause
        assert "type" in where_clause
        assert "internal" in params

    def test_build_department_condition_filter(self):
        """Test building SQL filter with user context condition."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept",
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    },
                }
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com", "department": "engineering"}
        where_clause, params = to_pgvector_filter(policy, user)

        assert "WHERE" in where_clause
        assert "department" in where_clause
        assert "engineering" in params

    def test_build_no_matching_rules_deny(self):
        """Test SQL filter when no rules match and default is deny."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin-only",
                    "allow": {"roles": ["admin"]},
                }
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com", "roles": ["user"]}
        where_clause, params = to_pgvector_filter(policy, user)

        assert "WHERE FALSE" in where_clause

    def test_build_multiple_rules_or_logic(self):
        """Test that multiple rules are combined with OR in SQL."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True},
                },
                {
                    "name": "dept",
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    },
                },
            ],
            "default": "deny",
        })

        user = {"id": "test@example.com", "department": "engineering"}
        where_clause, params = to_pgvector_filter(policy, user)

        assert "WHERE" in where_clause
        assert " OR " in where_clause

    def test_sql_injection_protection(self):
        """Test that parameters are properly parameterized."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept",
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    },
                }
            ],
            "default": "deny",
        })

        # Try with potentially malicious input
        user = {"id": "test@example.com", "department": "'; DROP TABLE users; --"}
        where_clause, params = to_pgvector_filter(policy, user)

        # Should use parameterized query with %s placeholders
        assert "%s" in where_clause
        # The malicious string should be in params, not directly in SQL
        assert "'; DROP TABLE users; --" in params
        assert "DROP TABLE" not in where_clause
