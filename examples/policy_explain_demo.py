#!/usr/bin/env python3
"""
Policy Explain Mode Demo

Demonstrates how to use evaluate_with_explanation() to debug complex policies.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine


def print_explanation(result):
    """Pretty-print an explanation result."""
    print("\n" + "="*80)
    print(f"DECISION: {result['decision'].upper()}")
    print("="*80)
    print(f"\nReason: {result['reason']}")

    if result['matched_rule']:
        print(f"Matched Rule: {result['matched_rule']}")

    if result['default_applied']:
        print("Default policy was applied (no rules granted access)")

    print("\n" + "-"*80)
    print("RULES EVALUATED:")
    print("-"*80)

    for rule_eval in result['rules_evaluated']:
        print(f"\nRule: {rule_eval['name']} ({rule_eval['type']})")

        if rule_eval.get('skipped'):
            print(f"  Skipped: {rule_eval['skip_reason']}")
            continue

        if 'match_details' in rule_eval:
            print(f"  Document Match: {rule_eval['matched_document']}")
            if isinstance(rule_eval['match_details'], list):
                for detail in rule_eval['match_details']:
                    status = "✓" if detail['result'] else "✗"
                    print(f"    {status} {detail['field']}: expected {detail['expected']}, got {detail['actual']}")

        print(f"  User Allowed: {rule_eval['user_allowed']}")

        details = rule_eval['allow_details']

        if details.get('everyone'):
            print(f"    Everyone: {details['everyone']}")

        if details.get('roles'):
            roles = details['roles']
            print(f"    Roles:")
            print(f"      Required: {roles['required']}")
            print(f"      User Has: {roles['user_has']}")
            print(f"      Matched: {roles['matched']}")
            print(f"      Result: {roles['result']}")

        if details.get('conditions'):
            print(f"    Conditions:")
            for cond in details['conditions']:
                status = "✓" if cond['result'] else "✗"
                print(f"      {status} {cond['condition']}")

        if 'logic' in details:
            print(f"    Logic: {details['logic']}")

    print("\n" + "="*80 + "\n")


def demo_or_logic():
    """Demo: Debugging OR logic policies."""
    print("\n" + "╔"+"═"*78+"╗")
    print("║" + " "*25 + "OR LOGIC POLICY DEBUG" + " "*32 + "║")
    print("╚"+"═"*78+"╝")

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

    # Test 1: Published document
    print("\n📄 Test 1: Published document (status='published', reviewed=false)")
    user = {}
    document = {"status": "published", "reviewed": False}
    result = engine.evaluate_with_explanation(user, document)
    print_explanation(result)

    # Test 2: Reviewed document
    print("\n📄 Test 2: Reviewed document (status='draft', reviewed=true)")
    document = {"status": "draft", "reviewed": True}
    result = engine.evaluate_with_explanation(user, document)
    print_explanation(result)

    # Test 3: Neither published nor reviewed
    print("\n📄 Test 3: Draft document (status='draft', reviewed=false)")
    document = {"status": "draft", "reviewed": False}
    result = engine.evaluate_with_explanation(user, document)
    print_explanation(result)


def demo_complex_nested():
    """Demo: Debugging complex nested OR/AND logic."""
    print("\n" + "╔"+"═"*78+"╗")
    print("║" + " "*20 + "COMPLEX NESTED OR/AND POLICY DEBUG" + " "*24 + "║")
    print("╚"+"═"*78+"╝")

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-or-public-and-level",
            "allow": {
                "conditions": [
                    "((user.department == document.department OR document.visibility == 'public') AND document.level >= 3)"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: Department match with high level
    print("\n📄 Test 1: User in same dept, level=5")
    user = {"id": "alice", "department": "engineering"}
    document = {"department": "engineering", "visibility": "private", "level": 5}
    result = engine.evaluate_with_explanation(user, document)
    print_explanation(result)

    # Test 2: Department mismatch but public with high level
    print("\n📄 Test 2: Different dept, but public and level=4")
    user = {"id": "bob", "department": "sales"}
    document = {"department": "engineering", "visibility": "public", "level": 4}
    result = engine.evaluate_with_explanation(user, document)
    print_explanation(result)

    # Test 3: Department match but low level
    print("\n📄 Test 3: Same dept but level=2 (fails level check)")
    user = {"id": "alice", "department": "engineering"}
    document = {"department": "engineering", "visibility": "private", "level": 2}
    result = engine.evaluate_with_explanation(user, document)
    print_explanation(result)


def demo_multiple_rules():
    """Demo: Debugging when multiple rules are evaluated."""
    print("\n" + "╔"+"═"*78+"╗")
    print("║" + " "*25 + "MULTIPLE RULES DEBUG" + " "*34 + "║")
    print("╚"+"═"*78+"╝")

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-full-access",
                "allow": {
                    "roles": ["admin"]
                }
            },
            {
                "name": "owner-access",
                "allow": {
                    "conditions": ["user.id == document.owner"]
                }
            },
            {
                "name": "public-access",
                "allow": {
                    "conditions": ["document.visibility == 'public'"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test: Regular user accessing their own document
    print("\n📄 Test: Regular user accessing their own private document")
    user = {"id": "alice", "roles": ["user"]}
    document = {"id": "doc1", "owner": "alice", "visibility": "private"}
    result = engine.evaluate_with_explanation(user, document)
    print_explanation(result)


if __name__ == "__main__":
    print("\n" + "🔍 RAGGuard Policy Explain Mode - Debugging Demo")
    print("=" * 80)

    demo_or_logic()
    demo_complex_nested()
    demo_multiple_rules()

    print("\n✅ Demo complete! Use evaluate_with_explanation() to debug your policies.\n")
