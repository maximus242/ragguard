#!/usr/bin/env python3
"""
Tests for native OR/AND support in pgvector filter builder.

Verifies that OR logic is pushed to PostgreSQL as native SQL (not post-filtering).
"""

import pytest

from ragguard import Policy
from ragguard.filters.builder import to_pgvector_filter


def test_pgvector_simple_or_filter():
    """Test that simple OR expression generates native SQL OR."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "published-or-reviewed",
            "allow": {
                "conditions": [
                    "(document.status == 'published' OR document.reviewed == true)"
                ]
            }
        }],
        "default": "deny"
    })

    user = {}
    sql, params = to_pgvector_filter(policy, user)

    # Verify it's native SQL OR
    assert "OR" in sql
    assert sql.count("OR") >= 1
    assert "status" in sql
    assert "reviewed" in sql
    assert len(params) == 2


def test_pgvector_simple_and_filter():
    """Test that simple AND expression generates native SQL AND."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "published-and-reviewed",
            "allow": {
                "conditions": [
                    "document.status == 'published' AND document.reviewed == true"
                ]
            }
        }],
        "default": "deny"
    })

    user = {}
    sql, params = to_pgvector_filter(policy, user)

    # Verify it's native SQL AND
    assert "AND" in sql
    assert "status" in sql
    assert "reviewed" in sql
    assert len(params) == 2


def test_pgvector_nested_or_and():
    """Test nested OR and AND expression."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex",
            "allow": {
                "conditions": [
                    "(document.category == 'tech' OR document.category == 'science') AND document.status == 'published'"
                ]
            }
        }],
        "default": "deny"
    })

    user = {}
    sql, params = to_pgvector_filter(policy, user)

    # Should have both OR and AND
    assert "OR" in sql
    assert "AND" in sql
    assert "category" in sql
    assert "status" in sql
    assert len(params) == 3


def test_pgvector_three_way_or():
    """Test OR with three conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "three-statuses",
            "allow": {
                "conditions": [
                    "document.status == 'published' OR document.status == 'reviewed' OR document.status == 'approved'"
                ]
            }
        }],
        "default": "deny"
    })

    user = {}
    sql, params = to_pgvector_filter(policy, user)

    # Should have 2 OR operators (3 conditions)
    assert sql.count("OR") >= 2
    assert len(params) == 3


def test_pgvector_or_with_user_context():
    """Test OR with user context substitution."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-or-public",
            "allow": {
                "conditions": [
                    "user.department == document.department OR document.visibility == 'public'"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"department": "engineering"}
    sql, params = to_pgvector_filter(policy, user)

    # Should substitute user.department value
    assert "OR" in sql
    assert "engineering" in params
    assert "public" in params


def test_pgvector_or_with_comparison():
    """Test OR with comparison operators."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "high-level-or-owner",
            "allow": {
                "conditions": [
                    "document.level >= 5 OR user.id == document.owner_id"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"id": "user123"}
    sql, params = to_pgvector_filter(policy, user)

    # Should have OR with >= operator
    assert "OR" in sql
    assert ">=" in sql
    assert 5 in params
    assert "user123" in params


def test_pgvector_or_with_array():
    """Test OR with array operations."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "shared-or-public",
            "allow": {
                "conditions": [
                    "user.id in document.shared_with OR document.visibility == 'public'"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}
    sql, params = to_pgvector_filter(policy, user)

    # Should have OR with ANY operator
    assert "OR" in sql
    assert "ANY" in sql or "visibility" in sql
    assert "alice" in params


def test_pgvector_backward_compat():
    """Test that simple conditions still work."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "simple",
            "allow": {
                "conditions": [
                    "user.department == document.department"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"department": "engineering"}
    sql, params = to_pgvector_filter(policy, user)

    # Should still work
    assert sql
    assert "department" in sql
    assert "engineering" in params


def test_pgvector_complex_nested():
    """Test deeply nested expression."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "deeply-nested",
            "allow": {
                "conditions": [
                    "((document.category == 'tech' OR document.category == 'science') AND document.status == 'published') OR user.id == document.owner_id"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"id": "user123"}
    sql, params = to_pgvector_filter(policy, user)

    # Should have nested OR and AND
    assert "OR" in sql
    assert "AND" in sql
    assert len(params) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
