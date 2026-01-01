"""
Tests for Query Explainer.

Tests the QueryExplainer class that helps debug why documents are allowed or denied.
"""

import pytest

from ragguard import Policy, QueryExplainer, QueryExplanation


def test_explainer_basic_allow():
    """Test explainer with simple allow condition."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice", "department": "engineering"}
    document = {"id": "doc1", "department": "engineering"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "ALLOW"
    assert explanation.document_id == "doc1"
    assert len(explanation.rule_evaluations) == 1
    assert explanation.rule_evaluations[0].matched is True
    assert explanation.rule_evaluations[0].passed is True


def test_explainer_basic_deny():
    """Test explainer with simple deny condition."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice", "department": "engineering"}
    document = {"id": "doc1", "department": "finance"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "DENY"
    assert explanation.document_id == "doc1"
    assert len(explanation.rule_evaluations) == 1
    assert explanation.rule_evaluations[0].matched is True
    assert explanation.rule_evaluations[0].passed is False


def test_explainer_role_based_allow():
    """Test explainer with role-based access."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "admin-access",
            "allow": {
                "roles": ["admin", "superuser"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice", "roles": ["admin"]}
    document = {"id": "doc1", "department": "finance"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "ALLOW"
    assert explanation.rule_evaluations[0].roles_matched is True


def test_explainer_role_based_deny():
    """Test explainer with role-based denial."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "admin-access",
            "allow": {
                "roles": ["admin", "superuser"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice", "roles": ["user"]}
    document = {"id": "doc1", "department": "finance"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "DENY"
    assert explanation.rule_evaluations[0].roles_matched is False


def test_explainer_everyone_access():
    """Test explainer with everyone flag."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-docs",
            "allow": {
                "everyone": True
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice"}
    document = {"id": "doc1", "classification": "public"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "ALLOW"
    assert explanation.rule_evaluations[0].everyone_matched is True


def test_explainer_match_condition():
    """Test explainer with match conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-only",
            "match": {"classification": "public"},
            "allow": {
                "everyone": True
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    # Document matches
    user = {"id": "alice"}
    document = {"id": "doc1", "classification": "public"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "ALLOW"
    assert explanation.rule_evaluations[0].matched is True

    # Document doesn't match
    document2 = {"id": "doc2", "classification": "confidential"}
    explanation2 = explainer.explain(user, document2)

    assert explanation2.final_decision == "DENY"
    assert explanation2.rule_evaluations[0].matched is False


def test_explainer_multiple_conditions():
    """Test explainer with multiple conditions (AND logic)."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "multi-check",
            "allow": {
                "conditions": [
                    "user.department == document.department",
                    "user.clearance >= document.clearance_level"
                ]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    # All conditions pass
    user = {"id": "alice", "department": "engineering", "clearance": 3}
    document = {"id": "doc1", "department": "engineering", "clearance_level": 2}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "ALLOW"
    assert len(explanation.rule_evaluations[0].condition_evaluations) == 2
    assert all(c.passed for c in explanation.rule_evaluations[0].condition_evaluations)

    # One condition fails
    user2 = {"id": "bob", "department": "engineering", "clearance": 1}
    explanation2 = explainer.explain(user2, document)

    assert explanation2.final_decision == "DENY"
    assert explanation2.rule_evaluations[0].condition_evaluations[0].passed is True
    assert explanation2.rule_evaluations[0].condition_evaluations[1].passed is False


def test_explainer_multiple_rules():
    """Test explainer with multiple rules."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-access",
                "allow": {"roles": ["admin"]}
            },
            {
                "name": "dept-access",
                "allow": {"conditions": ["user.department == document.department"]}
            }
        ],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    # First rule passes
    user = {"id": "alice", "roles": ["admin"], "department": "engineering"}
    document = {"id": "doc1", "department": "finance"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "ALLOW"
    assert explanation.rule_evaluations[0].passed is True
    assert explanation.rule_evaluations[1].passed is False

    # Second rule passes
    user2 = {"id": "bob", "roles": ["user"], "department": "finance"}
    explanation2 = explainer.explain(user2, document)

    assert explanation2.final_decision == "ALLOW"
    assert explanation2.rule_evaluations[0].passed is False
    assert explanation2.rule_evaluations[1].passed is True


def test_explainer_no_matching_rules():
    """Test explainer when no rules match the document."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "specific-doc",
            "match": {"doc_type": "confidential"},
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice"}
    document = {"id": "doc1", "doc_type": "public"}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "DENY"
    assert "No rules granted access" in explanation.reason


def test_explainer_string_formatting():
    """Test human-readable string formatting."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice", "department": "engineering"}
    document = {"id": "doc1", "department": "finance"}

    explanation = explainer.explain(user, document)
    output = str(explanation)

    # Check key parts are in output
    assert "Document 'doc1': DENY" in output
    assert "Rule 'dept-access'" in output
    assert "Default Policy: deny" in output


def test_explainer_comparison_operators():
    """Test explainer with various comparison operators."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "clearance-check",
            "allow": {
                "conditions": [
                    "user.clearance >= document.required_clearance"
                ]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    # Greater than (passes)
    user = {"id": "alice", "clearance": 5}
    document = {"id": "doc1", "required_clearance": 3}
    explanation = explainer.explain(user, document)
    assert explanation.final_decision == "ALLOW"

    # Equal (passes)
    user2 = {"id": "bob", "clearance": 3}
    explanation2 = explainer.explain(user2, document)
    assert explanation2.final_decision == "ALLOW"

    # Less than (fails)
    user3 = {"id": "charlie", "clearance": 2}
    explanation3 = explainer.explain(user3, document)
    assert explanation3.final_decision == "DENY"


def test_explainer_in_operator():
    """Test explainer with 'in' operator."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "team-access",
            "allow": {
                "conditions": ["user.id in document.allowed_users"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    # User in list (passes)
    user = {"id": "alice"}
    document = {"id": "doc1", "allowed_users": ["alice", "bob", "charlie"]}
    explanation = explainer.explain(user, document)
    assert explanation.final_decision == "ALLOW"

    # User not in list (fails)
    user2 = {"id": "eve"}
    explanation2 = explainer.explain(user2, document)
    assert explanation2.final_decision == "DENY"


def test_explainer_nested_fields():
    """Test explainer with nested field access."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "org-access",
            "allow": {
                "conditions": ["user.org.id == document.owner.org_id"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice", "org": {"id": "org123", "name": "ACME"}}
    document = {"id": "doc1", "owner": {"org_id": "org123"}}

    explanation = explainer.explain(user, document)

    assert explanation.final_decision == "ALLOW"


def test_explainer_document_id_override():
    """Test that document_id parameter overrides document.id."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice"}
    document = {"id": "internal_id", "name": "Test Doc"}

    # Use custom document ID
    explanation = explainer.explain(user, document, document_id="custom_id_123")

    assert explanation.document_id == "custom_id_123"


def test_explainer_value_extraction():
    """Test that condition evaluations extract and show values."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-check",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    user = {"id": "alice", "department": "engineering"}
    document = {"id": "doc1", "department": "finance"}

    explanation = explainer.explain(user, document)
    cond_eval = explanation.rule_evaluations[0].condition_evaluations[0]

    assert cond_eval.user_value == "engineering"
    assert cond_eval.document_value == "finance"


def test_explainer_no_allow_conditions():
    """Test explainer with rule that doesn't match any user."""
    # Use a policy with a specific role requirement that won't match
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "admin-only",
            "allow": {
                "roles": ["admin"]  # User doesn't have this role
            }
        }],
        "default": "deny"
    })

    explainer = QueryExplainer(policy)

    # User without the required role
    user = {"id": "alice", "roles": []}
    document = {"id": "doc1"}

    explanation = explainer.explain(user, document)

    # Should be DENY because user doesn't have admin role
    assert explanation.final_decision == "DENY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
