"""
Tests for FilterBuilderBase abstract class.

Tests the shared policy traversal logic used by all filter builders.
"""

from typing import Any, Dict, List, Optional

import pytest

from ragguard.filters.builder_base import DictFilterBuilder
from ragguard.policy.models import Policy


class ConcreteFilterBuilder(DictFilterBuilder):
    """Concrete implementation for testing using DictFilterBuilder."""

    backend_name = "test_backend"


class TestFilterBuilderBaseBasics:
    """Basic tests for FilterBuilderBase."""

    @pytest.fixture
    def builder(self):
        return ConcreteFilterBuilder()

    def test_backend_name(self, builder):
        """Test backend name property."""
        assert builder.backend_name == "test_backend"

    def test_simple_match_filter(self, builder):
        """Test simple match condition."""
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
        result = builder.build_filter(policy, user)

        assert result is not None
        # DictFilterBuilder uses $eq operator
        assert "type" in result or "$and" in result

    def test_deny_all_when_no_rules_match(self, builder):
        """Test deny all when no rules match user."""
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
        result = builder.build_filter(policy, user)

        # DictFilterBuilder returns a deny-all filter
        assert result is not None
        # The deny_all creates an impossible condition
        assert "__ragguard_deny_all__" in str(result) or "$eq" in str(result)

    def test_no_filter_when_default_allow(self, builder):
        """Test no filter when default is allow and no rules match."""
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
        result = builder.build_filter(policy, user)

        assert result is None

    def test_no_filter_when_unrestricted_access(self, builder):
        """Test no filter when rule grants unrestricted access."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_all",
                    "allow": {"roles": ["admin"]}
                    # No match or conditions
                }
            ],
            "default": "deny"
        })

        user = {"id": "admin_user", "roles": ["admin"]}
        result = builder.build_filter(policy, user)

        assert result is None


class TestFilterBuilderBaseMultipleRules:
    """Tests for multiple rule handling."""

    @pytest.fixture
    def builder(self):
        return ConcreteFilterBuilder()

    def test_multiple_rules_combined_with_or(self, builder):
        """Test multiple matching rules are OR'd together."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"visibility": "public"}
                },
                {
                    "name": "internal_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"visibility": "internal"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        result = builder.build_filter(policy, user)

        assert result is not None
        assert "$or" in result
        assert len(result["$or"]) == 2

    def test_single_matching_rule_not_wrapped(self, builder):
        """Test single matching rule is not wrapped in OR."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin"]},
                    "match": {"type": "secret"}
                },
                {
                    "name": "user_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"type": "public"}
                }
            ],
            "default": "deny"
        })

        # User only matches one rule
        user = {"id": "alice", "roles": ["user"]}
        result = builder.build_filter(policy, user)

        assert result is not None
        assert "$or" not in result
        # Should have the type filter
        assert "type" in result or "public" in str(result)


class TestFilterBuilderBaseConditions:
    """Tests for condition processing."""

    @pytest.fixture
    def builder(self):
        return ConcreteFilterBuilder()

    def test_multiple_conditions_combined_with_and(self, builder):
        """Test multiple conditions within rule are AND'd."""
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
                    },
                    "match": {"status": "active"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "eng", "team": "ml"}
        result = builder.build_filter(policy, user)

        assert result is not None
        assert "$and" in result

    def test_single_condition_not_wrapped(self, builder):
        """Test single condition is not wrapped in AND."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "simple_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        result = builder.build_filter(policy, user)

        assert result is not None
        # Should contain department filter
        assert "department" in str(result) or "engineering" in str(result)

    def test_or_condition_expression(self, builder):
        """Test OR condition within expression."""
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
        result = builder.build_filter(policy, user)

        assert result is not None


class TestFilterBuilderBaseOperators:
    """Tests for different comparison operators."""

    @pytest.fixture
    def builder(self):
        return ConcreteFilterBuilder()

    def test_equality_operator(self, builder):
        """Test equality comparison."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "level_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        result = builder.build_filter(policy, user)

        assert result is not None
        assert "engineering" in str(result)

    def test_literal_comparison(self, builder):
        """Test literal value comparison."""
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
        result = builder.build_filter(policy, user)

        assert result is not None
        assert "active" in str(result)

    def test_not_equal_operator(self, builder):
        """Test not equal comparison."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "exclude_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["document.status != 'deleted'"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}
        result = builder.build_filter(policy, user)

        assert result is not None


class TestFilterBuilderBaseInOperator:
    """Tests for IN operator support."""

    @pytest.fixture
    def builder(self):
        return ConcreteFilterBuilder()

    def test_in_with_user_list(self, builder):
        """Test IN operator with list from user context."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "group_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["document.group in user.groups"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "groups": ["eng", "research", "ml"]}
        result = builder.build_filter(policy, user)

        assert result is not None


class TestFilterBuilderBaseEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def builder(self):
        return ConcreteFilterBuilder()

    def test_no_matching_rules_deny_default(self, builder):
        """Test when no rules match user with deny default."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_only",
                    "allow": {"roles": ["admin"]},
                    "match": {"type": "secret"}
                }
            ],
            "default": "deny"
        })

        # User doesn't have admin role
        user = {"id": "alice", "roles": ["user"]}
        result = builder.build_filter(policy, user)

        # Should return a deny-all filter
        assert result is not None
        assert "__ragguard_deny_all__" in str(result)

    def test_no_matching_rules_allow_default(self, builder):
        """Test when no rules match user with allow default."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_only",
                    "allow": {"roles": ["admin"]},
                    "match": {"type": "secret"}
                }
            ],
            "default": "allow"
        })

        # User doesn't have admin role
        user = {"id": "alice", "roles": ["user"]}
        result = builder.build_filter(policy, user)

        # With default allow, no filter needed
        assert result is None

    def test_match_and_conditions_combined(self, builder):
        """Test rule with both match and conditions."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "combined",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    },
                    "match": {"status": "active"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        result = builder.build_filter(policy, user)

        assert result is not None
        # Should have both match and condition filters
        assert "$and" in result
