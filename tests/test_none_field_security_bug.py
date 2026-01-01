"""
Test for security bug: != operator with None fields.

This test demonstrates the security vulnerability where users without
certain fields can bypass != checks and gain unintended access.
"""

import pytest

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine


def test_not_equals_with_missing_field_should_deny():
    """
    SECURITY TEST: Users without a field should be DENIED when using !=.

    Scenario: Policy says "allow if user.role != 'guest'"
    Expected: Users without a role field should be DENIED (not allowed)
    Current Bug: Users without a role field are ALLOWED (security bypass)
    """
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "non-guests-only",
            "allow": {
                "conditions": ["user.role != 'guest'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    document = {"id": "doc1", "content": "test"}

    # Case 1: User with role='admin' - should be allowed
    user_admin = {"id": "alice", "role": "admin"}
    assert engine.evaluate(user_admin, document) == True, "Admin should be allowed"

    # Case 2: User with role='guest' - should be denied
    user_guest = {"id": "bob", "role": "guest"}
    assert engine.evaluate(user_guest, document) == False, "Guest should be denied"

    # Case 3: User WITHOUT role field - should be denied (SECURITY BUG)
    user_no_role = {"id": "charlie"}
    result = engine.evaluate(user_no_role, document)

    # THIS SHOULD BE FALSE (deny) but currently returns TRUE (allow) - SECURITY BUG
    assert result == False, (
        "SECURITY BUG: User without role field should be DENIED, not allowed! "
        "The != operator currently returns True when fields are missing, "
        "allowing unauthorized access."
    )


def test_not_equals_document_field_missing_should_deny():
    """
    SECURITY TEST: Documents without a field should fail != checks.

    Scenario: Policy says "allow if document.status != 'archived'"
    Expected: Documents without status field should be DENIED
    """
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "non-archived-docs",
            "allow": {
                "conditions": ["document.status != 'archived'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"id": "alice"}

    # Case 1: Document with status='active' - should be allowed
    doc_active = {"id": "doc1", "status": "active"}
    assert engine.evaluate(user, doc_active) == True, "Active doc should be allowed"

    # Case 2: Document with status='archived' - should be denied
    doc_archived = {"id": "doc2", "status": "archived"}
    assert engine.evaluate(user, doc_archived) == False, "Archived doc should be denied"

    # Case 3: Document WITHOUT status field - should be denied (SECURITY BUG)
    doc_no_status = {"id": "doc3"}
    result = engine.evaluate(user, doc_no_status)

    # THIS SHOULD BE FALSE (deny) but currently returns TRUE (allow) - SECURITY BUG
    assert result == False, (
        "SECURITY BUG: Document without status field should be DENIED! "
        "Missing fields should always deny access by default."
    )


def test_compiled_not_equals_with_missing_field():
    """
    Test that compiled conditions also have the same security bug.

    The compiled evaluator (used for performance) has the same issue.
    """
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "test",
            "allow": {
                "conditions": ["user.clearance != 'none'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    document = {"id": "doc1"}

    # User without clearance field - should be denied
    user_no_clearance = {"id": "eve"}
    result = engine.evaluate(user_no_clearance, document)

    assert result == False, (
        "SECURITY BUG: Compiled evaluator also has the != None vulnerability!"
    )


def test_comparison_operators_deny_on_none():
    """
    Verify that other comparison operators correctly deny when fields are missing.

    This test ensures ==, <, >, <=, >= all return False for None values (correct behavior).
    Only != has the bug.
    """
    policy_templates = [
        ("user.level == 5", "equals"),
        ("user.level > 3", "greater than"),
        ("user.level < 10", "less than"),
        ("user.level >= 5", "greater or equal"),
        ("user.level <= 8", "less or equal"),
    ]

    document = {"id": "doc1"}
    user_no_level = {"id": "alice"}  # Missing 'level' field

    for condition, operator_name in policy_templates:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {"conditions": [condition]}
            }],
            "default": "deny"
        })

        engine = PolicyEngine(policy)
        result = engine.evaluate(user_no_level, document)

        assert result == False, (
            f"Operator '{operator_name}' correctly denies when field is missing"
        )


def test_proper_not_equals_usage_with_field_check():
    """
    Show the CORRECT way to write policies that handle missing fields.

    If you want to allow documents without a field, use two separate rules:
    1. Allow documents where field doesn't exist
    2. Allow documents where field exists but != value
    """
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "allow-docs-without-status",
                "allow": {
                    "conditions": ["document.status not exists"]
                }
            },
            {
                "name": "allow-non-archived-docs",
                "allow": {
                    "conditions": [
                        "document.status exists",
                        "document.status != 'archived'"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"id": "alice"}

    # Document without status - should be allowed (first rule)
    doc_no_status = {"id": "doc1"}
    assert engine.evaluate(user, doc_no_status) == True

    # Document with status='active' - should be allowed (second rule)
    doc_active = {"id": "doc2", "status": "active"}
    assert engine.evaluate(user, doc_active) == True

    # Document with status='archived' - should be denied (no matching rule)
    doc_archived = {"id": "doc3", "status": "archived"}
    assert engine.evaluate(user, doc_archived) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
