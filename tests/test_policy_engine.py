"""
Tests for policy evaluation engine.
"""

import pytest

from ragguard.policy import Policy, PolicyEngine


def test_evaluate_everyone_rule():
    """Test rule that allows everyone."""
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

    engine = PolicyEngine(policy)

    # Public document - should allow any user
    user = {"id": "test@example.com"}
    doc = {"visibility": "public", "text": "Hello"}

    assert engine.evaluate(user, doc) is True

    # Non-public document - should deny
    doc2 = {"visibility": "private", "text": "Secret"}
    assert engine.evaluate(user, doc2) is False


def test_evaluate_role_based():
    """Test role-based access control."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-access",
                "allow": {"roles": ["admin"]},
            }
        ],
        "default": "deny",
    })

    engine = PolicyEngine(policy)

    # Admin user - should allow
    admin_user = {"id": "admin@example.com", "roles": ["admin"]}
    doc = {"text": "Confidential"}
    assert engine.evaluate(admin_user, doc) is True

    # Regular user - should deny
    regular_user = {"id": "user@example.com", "roles": ["user"]}
    assert engine.evaluate(regular_user, doc) is False


def test_evaluate_department_condition():
    """Test condition-based evaluation."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "department-docs",
                "allow": {
                    "conditions": ["user.department == document.department"]
                },
            }
        ],
        "default": "deny",
    })

    engine = PolicyEngine(policy)

    # Same department - should allow
    user = {"id": "test@example.com", "department": "engineering"}
    doc = {"department": "engineering", "text": "Tech doc"}
    assert engine.evaluate(user, doc) is True

    # Different department - should deny
    doc2 = {"department": "finance", "text": "Finance doc"}
    assert engine.evaluate(user, doc2) is False


def test_evaluate_in_condition():
    """Test 'in' condition evaluation."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "shared-docs",
                "allow": {
                    "conditions": ["user.id in document.shared_with"]
                },
            }
        ],
        "default": "deny",
    })

    engine = PolicyEngine(policy)

    user = {"id": "alice@example.com"}

    # User in shared_with list - should allow
    doc = {"shared_with": ["alice@example.com", "bob@example.com"]}
    assert engine.evaluate(user, doc) is True

    # User not in shared_with list - should deny
    doc2 = {"shared_with": ["bob@example.com", "carol@example.com"]}
    assert engine.evaluate(user, doc2) is False


def test_evaluate_multiple_rules():
    """Test that multiple rules work with OR logic."""
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

    engine = PolicyEngine(policy)

    # Public doc - any user can access
    user = {"id": "user@example.com", "roles": ["user"]}
    doc = {"visibility": "public"}
    assert engine.evaluate(user, doc) is True

    # Private doc but admin user - can access
    admin_user = {"id": "admin@example.com", "roles": ["admin"]}
    doc2 = {"visibility": "private"}
    assert engine.evaluate(admin_user, doc2) is True

    # Private doc, regular user - cannot access
    assert engine.evaluate(user, doc2) is False


def test_evaluate_match_conditions():
    """Test that match conditions filter correctly."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-public",
                "match": {"type": "internal", "confidential": False},
                "allow": {
                    "conditions": ["user.department == document.department"]
                },
            }
        ],
        "default": "deny",
    })

    engine = PolicyEngine(policy)

    user = {"id": "test@example.com", "department": "engineering"}

    # Matches all conditions - should allow
    doc1 = {
        "type": "internal",
        "confidential": False,
        "department": "engineering"
    }
    assert engine.evaluate(user, doc1) is True

    # Doesn't match type - should deny
    doc2 = {
        "type": "external",
        "confidential": False,
        "department": "engineering"
    }
    assert engine.evaluate(user, doc2) is False

    # Confidential - should deny
    doc3 = {
        "type": "internal",
        "confidential": True,
        "department": "engineering"
    }
    assert engine.evaluate(user, doc3) is False


def test_evaluate_nested_attributes():
    """Test evaluation with nested attributes."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "team-docs",
                "allow": {
                    "conditions": ["user.team.id == document.metadata.team_id"]
                },
            }
        ],
        "default": "deny",
    })

    engine = PolicyEngine(policy)

    user = {
        "id": "test@example.com",
        "team": {"id": "team-123", "name": "Engineering"}
    }

    # Matching nested attributes - should allow
    doc = {
        "metadata": {"team_id": "team-123", "project": "X"}
    }
    assert engine.evaluate(user, doc) is True

    # Non-matching nested attributes - should deny
    doc2 = {
        "metadata": {"team_id": "team-456", "project": "Y"}
    }
    assert engine.evaluate(user, doc2) is False


def test_evaluate_default_allow():
    """Test that default allow works."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "restricted",
                "match": {"confidential": True},
                "allow": {"roles": ["admin"]},
            }
        ],
        "default": "allow",
    })

    engine = PolicyEngine(policy)

    # Non-confidential doc - should allow by default
    user = {"id": "user@example.com", "roles": ["user"]}
    doc = {"confidential": False}
    assert engine.evaluate(user, doc) is True

    # Confidential doc, non-admin - should deny
    doc2 = {"confidential": True}
    assert engine.evaluate(user, doc2) is False


def test_evaluate_combined_roles_and_conditions():
    """Test rule with both roles and conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "manager-dept-confidential",
                "match": {"confidential": True},
                "allow": {
                    "roles": ["manager"],
                    "conditions": ["user.department == document.department"]
                },
            }
        ],
        "default": "deny",
    })

    engine = PolicyEngine(policy)

    # Manager in same department - should allow
    manager = {
        "id": "manager@example.com",
        "roles": ["manager"],
        "department": "engineering"
    }
    doc = {"confidential": True, "department": "engineering"}
    assert engine.evaluate(manager, doc) is True

    # Manager in different department - should deny
    doc2 = {"confidential": True, "department": "finance"}
    assert engine.evaluate(manager, doc2) is False

    # Employee in same department - should deny
    employee = {
        "id": "employee@example.com",
        "roles": ["employee"],
        "department": "engineering"
    }
    assert engine.evaluate(employee, doc) is False
