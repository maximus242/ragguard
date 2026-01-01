#!/usr/bin/env python3
"""
Tests for policy explain mode.

Verifies that evaluate_with_explanation returns detailed debugging information.
"""

import pytest

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine


def test_explain_simple_allow():
    """Test explanation for a simple allow case."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "admin-access",
            "allow": {
                "roles": ["admin"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"id": "alice", "roles": ["admin"]}
    document = {"id": "doc1", "content": "secret"}

    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "allow"
    assert result["matched_rule"] == "admin-access"
    assert "granted access" in result["reason"]
    assert len(result["rules_evaluated"]) == 1

    rule_eval = result["rules_evaluated"][0]
    assert rule_eval["name"] == "admin-access"
    assert rule_eval["user_allowed"] == True
    assert rule_eval["allow_details"]["roles"]["matched"] == ["admin"]


def test_explain_simple_deny():
    """Test explanation for a simple deny case."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "admin-access",
            "allow": {
                "roles": ["admin"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"id": "bob", "roles": ["user"]}
    document = {"id": "doc1", "content": "secret"}

    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "deny"
    assert result["default_applied"] == True
    assert "default policy" in result["reason"]


def test_explain_or_condition():
    """Test explanation for OR logic in conditions."""
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

    engine = PolicyEngine(policy)
    user = {}

    # Test with published document
    document = {"status": "published", "reviewed": False}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "allow"
    assert result["matched_rule"] == "published-or-reviewed"

    condition_details = result["rules_evaluated"][0]["allow_details"]["conditions"]
    assert len(condition_details) == 1
    assert condition_details[0]["result"] == True

    # Test with reviewed document
    document = {"status": "draft", "reviewed": True}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "allow"

    # Test with neither published nor reviewed
    document = {"status": "draft", "reviewed": False}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "deny"
    assert result["default_applied"] == True


def test_explain_complex_nested():
    """Test explanation for complex nested OR/AND logic."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex-access",
            "allow": {
                "conditions": [
                    "((document.category == 'tech' OR document.category == 'science') AND document.level >= 3)"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {}

    # Should allow: tech category with level >= 3
    document = {"category": "tech", "level": 5}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "allow"

    # Should deny: tech category with level < 3
    document = {"category": "tech", "level": 2}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "deny"


def test_explain_explicit_rule_match():
    """Test explanation when explicit rule matches document."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "published-docs",
            "match": {"status": "published"},
            "allow": {
                "roles": ["viewer", "editor"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"roles": ["editor"]}

    # Document matches rule
    document = {"status": "published", "title": "Test"}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "allow"
    assert result["rules_evaluated"][0]["matched_document"] == True

    # Document doesn't match rule
    document = {"status": "draft", "title": "Test"}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "deny"
    assert result["rules_evaluated"][0]["matched_document"] == False
    assert result["rules_evaluated"][0]["skipped"] == True


def test_explain_multiple_rules():
    """Test explanation when multiple rules are evaluated."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-rule",
                "allow": {
                    "roles": ["admin"]
                }
            },
            {
                "name": "owner-rule",
                "allow": {
                    "conditions": ["user.id == document.owner"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"id": "alice", "roles": ["user"]}
    document = {"id": "doc1", "owner": "alice"}

    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "allow"
    assert result["matched_rule"] == "owner-rule"
    assert len(result["rules_evaluated"]) == 2


def test_explain_roles_and_conditions():
    """Test explanation when both roles and conditions are specified."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "editor-published",
            "allow": {
                "roles": ["editor"],
                "conditions": ["document.status == 'published'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"roles": ["editor"]}

    # User has role AND document is published
    document = {"status": "published"}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "allow"
    assert result["rules_evaluated"][0]["allow_details"]["logic"] == "user_allowed AND conditions_passed"

    # User has role but document is not published
    document = {"status": "draft"}
    result = engine.evaluate_with_explanation(user, document)

    assert result["decision"] == "deny"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
