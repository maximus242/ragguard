#!/usr/bin/env python3
"""
Tests for native OR/AND support in Qdrant filter builder.

Verifies that OR logic is pushed to Qdrant as native filters (not post-filtering).
"""

import pytest

# Skip all tests if qdrant-client is not installed
pytest.importorskip("qdrant_client")

from ragguard import Policy
from ragguard.filters.builder import to_qdrant_filter


def test_qdrant_simple_or_filter():
    """Test that simple OR expression generates native Qdrant filter."""
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
    filter_obj = to_qdrant_filter(policy, user)

    # Verify it's a native OR filter (should clause)
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')
    assert filter_obj.should is not None
    assert len(filter_obj.should) == 2


def test_qdrant_simple_and_filter():
    """Test that simple AND expression generates native Qdrant filter."""
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
    filter_obj = to_qdrant_filter(policy, user)

    # Verify it's a native AND filter (must clause)
    assert filter_obj is not None
    assert hasattr(filter_obj, 'must')
    assert filter_obj.must is not None
    assert len(filter_obj.must) == 2


def test_qdrant_nested_or_and_filter():
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
    filter_obj = to_qdrant_filter(policy, user)

    # Top level should be AND (must clause)
    assert filter_obj is not None
    assert hasattr(filter_obj, 'must')
    assert filter_obj.must is not None
    assert len(filter_obj.must) == 2

    # First child should be OR (should clause)
    first_child = filter_obj.must[0]
    assert hasattr(first_child, 'should')
    assert first_child.should is not None
    assert len(first_child.should) == 2


def test_qdrant_three_way_or():
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
    filter_obj = to_qdrant_filter(policy, user)

    # Should have 3 OR branches
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')
    assert filter_obj.should is not None
    assert len(filter_obj.should) == 3


def test_qdrant_or_with_user_context():
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
    filter_obj = to_qdrant_filter(policy, user)

    # Verify OR filter is created
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')
    assert filter_obj.should is not None
    assert len(filter_obj.should) == 2

    # First condition should have user.department value substituted
    # (Should be FieldCondition with department='engineering')


def test_qdrant_complex_nested():
    """Test deeply nested expression."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "deeply-nested",
            "allow": {
                "conditions": [
                    "((user.role == 'admin' OR user.role == 'manager') AND document.status == 'published') OR user.id == document.owner_id"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"role": "admin", "id": "user123"}
    filter_obj = to_qdrant_filter(policy, user)

    # Top level should be OR
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')
    assert filter_obj.should is not None
    assert len(filter_obj.should) == 2


def test_qdrant_backward_compat_simple_condition():
    """Test that simple conditions still work (no OR/AND)."""
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
    filter_obj = to_qdrant_filter(policy, user)

    # Should still work with old string parsing
    assert filter_obj is not None


def test_qdrant_or_with_field_existence():
    """Test OR combined with field existence."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-or-no-sensitivity",
            "allow": {
                "conditions": [
                    "document.public == true OR document.sensitivity not exists"
                ]
            }
        }],
        "default": "deny"
    })

    user = {}
    filter_obj = to_qdrant_filter(policy, user)

    # Should generate OR with field existence check
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')


def test_qdrant_or_with_comparison():
    """Test OR combined with comparison operators."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "high-level-or-owner",
            "allow": {
                "conditions": [
                    "user.level >= 8 OR user.id == document.owner_id"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"level": 9, "id": "user123"}
    filter_obj = to_qdrant_filter(policy, user)

    # Should generate OR filter
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')


def test_qdrant_or_with_array_operations():
    """Test OR combined with array operations."""
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
    filter_obj = to_qdrant_filter(policy, user)

    # Should generate OR filter
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')
    assert len(filter_obj.should) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
