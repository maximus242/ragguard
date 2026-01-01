#!/usr/bin/env python3
"""
Tests for native OR/AND support in Pinecone filter builder.

Verifies that OR logic is pushed to Pinecone as native filters (not post-filtering).
"""

import pytest

from ragguard import Policy
from ragguard.filters.builder import to_pinecone_filter


def test_pinecone_simple_or_filter():
    """Test that simple OR expression generates native Pinecone filter."""
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
    filter_obj = to_pinecone_filter(policy, user)

    # Verify it's a native OR filter
    assert filter_obj is not None
    assert "$or" in filter_obj
    assert len(filter_obj["$or"]) == 2


def test_pinecone_simple_and_filter():
    """Test that simple AND expression generates native Pinecone filter."""
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
    filter_obj = to_pinecone_filter(policy, user)

    # Verify it's a native AND filter
    assert filter_obj is not None
    assert "$and" in filter_obj
    assert len(filter_obj["$and"]) == 2


def test_pinecone_nested_or_and():
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
    filter_obj = to_pinecone_filter(policy, user)

    # Top level should be AND
    assert filter_obj is not None
    assert "$and" in filter_obj
    assert len(filter_obj["$and"]) == 2

    # First child should be OR
    first_child = filter_obj["$and"][0]
    assert "$or" in first_child
    assert len(first_child["$or"]) == 2


def test_pinecone_three_way_or():
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
    filter_obj = to_pinecone_filter(policy, user)

    # Should have 3 OR branches
    assert filter_obj is not None
    assert "$or" in filter_obj
    assert len(filter_obj["$or"]) == 3


def test_pinecone_or_with_user_context():
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
    filter_obj = to_pinecone_filter(policy, user)

    # Verify OR filter is created
    assert filter_obj is not None
    assert "$or" in filter_obj
    assert len(filter_obj["$or"]) == 2


def test_pinecone_complex_nested():
    """Test deeply nested expression."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "deeply-nested",
            "allow": {
                "conditions": [
                    "((document.category == 'tech' OR document.category == 'science') AND document.status == 'published') OR document.owner_id == user.id"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"id": "user123"}
    filter_obj = to_pinecone_filter(policy, user)

    # Top level should be OR
    assert filter_obj is not None
    assert "$or" in filter_obj
    assert len(filter_obj["$or"]) == 2


def test_pinecone_backward_compat_simple_condition():
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
    filter_obj = to_pinecone_filter(policy, user)

    # Should still work with old string parsing
    assert filter_obj is not None


def test_pinecone_or_with_field_existence():
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
    filter_obj = to_pinecone_filter(policy, user)

    # Should generate OR with field existence check
    assert filter_obj is not None
    assert "$or" in filter_obj


def test_pinecone_or_with_comparison():
    """Test OR combined with comparison operators."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "high-level-or-owner",
            "allow": {
                "conditions": [
                    "document.level >= 5 OR document.owner_id == user.id"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"id": "user123"}
    filter_obj = to_pinecone_filter(policy, user)

    # Should generate OR filter
    assert filter_obj is not None
    assert "$or" in filter_obj


def test_pinecone_or_with_array_operations():
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
    filter_obj = to_pinecone_filter(policy, user)

    # Should generate OR filter
    assert filter_obj is not None
    assert "$or" in filter_obj
    assert len(filter_obj["$or"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
