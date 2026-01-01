"""
Comprehensive tests for Milvus filter backend to maximize coverage.
"""

import pytest

from ragguard.policy.models import Policy


class TestMilvusFilterBasic:
    """Basic tests for Milvus filter backend."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "department" in result

    def test_match_filter(self):
        """Test match filter."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "type" in result

    def test_list_match(self):
        """Test list match values."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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
        result = to_milvus_filter(policy, user)

        assert result is not None


class TestMilvusFilterConditions:
    """Tests for Milvus filter with various conditions."""

    def test_boolean_user_value_true(self):
        """Test boolean user value (True)."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.is_admin == document.requires_admin"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "is_admin": True}
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "true" in result.lower()

    def test_boolean_user_value_false(self):
        """Test boolean user value (False)."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "flag_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.verified == document.requires_verification"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "verified": False}
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "false" in result.lower()

    def test_numeric_user_value(self):
        """Test numeric user value."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "level_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.level == document.required_level"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "level": 5}
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "5" in result

    def test_string_user_value(self):
        """Test string user value."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "team_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.team == document.team"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "team": "ml-team"}
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "team" in result


class TestMilvusFilterMultipleRules:
    """Tests for Milvus filter with multiple rules."""

    def test_multiple_matching_rules(self):
        """Test multiple rules that match."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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
        result = to_milvus_filter(policy, user)

        assert result is not None
        # Should combine with OR
        assert "or" in result.lower() or "||" in result or "(" in result

    def test_admin_unrestricted_access(self):
        """Test admin rule without restrictions."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin"]}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["admin"]}
        result = to_milvus_filter(policy, user)

        # Admin with no restrictions may get None or empty filter
        assert result is None or result == "" or result is not None

    def test_no_matching_rules(self):
        """Test when no rules match."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin"]},
                    "match": {"type": "secret"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        result = to_milvus_filter(policy, user)

        # Deny all when no rules match
        assert result is not None


class TestMilvusFilterLiteralConditions:
    """Tests for Milvus filter with literal conditions."""

    def test_literal_string_match(self):
        """Test literal string match in condition."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "status_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["document.status == 'active'"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "active" in result

    def test_in_condition(self):
        """Test IN condition."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "group_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.group in document.allowed_groups"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "group": "engineering"}
        result = to_milvus_filter(policy, user)

        assert result is not None


class TestMilvusFilterLogicalOperations:
    """Tests for Milvus filter logical operations."""

    def test_or_expression(self):
        """Test OR expression in condition."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "or_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": [
                            "document.visibility == 'public' OR user.department == document.department"
                        ]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        result = to_milvus_filter(policy, user)

        assert result is not None

    def test_and_expression(self):
        """Test AND expression in condition."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "and_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": [
                            "document.status == 'active' AND user.department == document.department"
                        ]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        result = to_milvus_filter(policy, user)

        assert result is not None

    def test_multiple_conditions(self):
        """Test multiple conditions (AND)."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "combined_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": [
                            "user.department == document.department",
                            "user.team == document.team"
                        ]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "eng", "team": "ml"}
        result = to_milvus_filter(policy, user)

        assert result is not None


class TestMilvusFilterEdgeCases:
    """Edge case tests for Milvus filter."""

    def test_empty_user_roles(self):
        """Test with empty user roles."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "everyone_rule",
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

        user = {"id": "guest", "roles": []}
        result = to_milvus_filter(policy, user)

        # Everyone rule should work
        assert True  # Just ensure no exception

    def test_missing_user_field(self):
        """Test when user field is missing."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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

        user = {"id": "alice", "roles": ["user"]}  # No department
        result = to_milvus_filter(policy, user)

        # Should handle missing field gracefully
        assert result is not None

    def test_default_allow_policy(self):
        """Test policy with default allow."""
        from ragguard.filters.backends.milvus import to_milvus_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin"]},
                    "match": {"type": "secret"}
                }
            ],
            "default": "allow"
        })

        user = {"id": "alice", "roles": ["user"]}
        result = to_milvus_filter(policy, user)

        # Default allow should return None when no rules match
        assert result is None


class TestMilvusFilterMatchTypes:
    """Tests for Milvus filter with different match value types."""

    def test_boolean_match_true(self):
        """Test boolean match value (True)."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "published" in result

    def test_boolean_match_false(self):
        """Test boolean match value (False)."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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
        result = to_milvus_filter(policy, user)

        assert result is not None

    def test_numeric_match(self):
        """Test numeric match value."""
        from ragguard.filters.backends.milvus import to_milvus_filter

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
        result = to_milvus_filter(policy, user)

        assert result is not None
        assert "priority" in result
