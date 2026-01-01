#!/usr/bin/env python3
"""
Query Complexity Limits Tests for RAGGuard v0.2.0

Tests that complexity limits prevent DoS attacks from malicious policies.
"""

from ragguard import Policy
from ragguard.policy.models import PolicyLimits

print("=" * 80)
print("Query Complexity Limits Tests")
print("=" * 80)

tests_passed = 0
tests_failed = 0

def test(name, func):
    """Run a test and track results."""
    global tests_passed, tests_failed
    print(f"\n🔍 {name}")
    try:
        result = func()
        if result:
            print(f"   ✅ PASS")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

# ============================================================================
# COMPLEXITY LIMIT TESTS
# ============================================================================

def test_too_many_rules():
    """Test that policies with too many rules are rejected."""
    # Create a policy with MAX_RULES + 1 rules
    rules = []
    for i in range(PolicyLimits.MAX_RULES + 1):
        rules.append({
            "name": f"rule_{i}",
            "allow": {
                "everyone": True,
                "conditions": [f"document.id == '{i}'"]
            }
        })

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": rules,
            "default": "deny"
        })
        print(f"      Policy with {len(rules)} rules was accepted (should reject)")
        return False
    except ValueError as e:
        if "Too many rules" in str(e):
            print(f"      Correctly rejected policy with {len(rules)} rules")
            return True
        else:
            print(f"      Wrong error: {e}")
            return False

test("Too many rules rejected", test_too_many_rules)

def test_too_many_conditions():
    """Test that rules with too many conditions are rejected."""
    # Create a rule with MAX_CONDITIONS_PER_RULE + 1 conditions
    conditions = [f"document.field_{i} == 'value'" for i in range(PolicyLimits.MAX_CONDITIONS_PER_RULE + 1)]

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "too-many-conditions",
                "allow": {
                    "everyone": True,
                    "conditions": conditions
                }
            }],
            "default": "deny"
        })
        print(f"      Rule with {len(conditions)} conditions was accepted (should reject)")
        return False
    except ValueError as e:
        if "Too many conditions" in str(e):
            print(f"      Correctly rejected rule with {len(conditions)} conditions")
            return True
        else:
            print(f"      Wrong error: {e}")
            return False

test("Too many conditions rejected", test_too_many_conditions)

def test_large_list_literal():
    """Test that list literals with too many elements are rejected."""
    # Create a list with MAX_LIST_SIZE + 1 elements
    large_list = [f"value_{i}" for i in range(PolicyLimits.MAX_LIST_SIZE + 1)]

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "large-list",
                "allow": {
                    "everyone": True,
                    "conditions": [f"document.category in {large_list}"]
                }
            }],
            "default": "deny"
        })
        print(f"      List with {len(large_list)} elements was accepted (should reject)")
        return False
    except ValueError as e:
        if "List literal too large" in str(e):
            print(f"      Correctly rejected list with {len(large_list)} elements")
            return True
        else:
            print(f"      Wrong error: {e}")
            return False

test("Large list literal rejected", test_large_list_literal)

def test_policy_too_large():
    """Test that policies exceeding size limit are rejected."""
    # Create a huge policy by adding many large rules
    rules = []
    # Each rule adds roughly 200 bytes, so we need ~5000 rules to exceed 1MB
    # But we're limited to MAX_RULES, so let's create rules with huge strings
    num_rules = 100  # Within MAX_RULES limit
    huge_string = "x" * 12000  # 12KB per rule -> 1.2MB total

    for i in range(num_rules):
        rules.append({
            "name": f"rule_{i}_{huge_string}",
            "allow": {
                "everyone": True,
                "conditions": [f"document.id == '{huge_string}'"]
            }
        })

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": rules,
            "default": "deny"
        })
        print(f"      Large policy was accepted (should reject)")
        return False
    except ValueError as e:
        if "Policy too large" in str(e):
            print(f"      Correctly rejected oversized policy")
            return True
        else:
            print(f"      Wrong error: {e}")
            return False

test("Oversized policy rejected", test_policy_too_large)

def test_acceptable_limits():
    """Test that policies within limits are accepted."""
    # Create a policy with acceptable complexity
    rules = []
    for i in range(50):  # Half of MAX_RULES
        conditions = [f"document.field_{j} == 'value'" for j in range(10)]  # 10 conditions
        rules.append({
            "name": f"rule_{i}",
            "allow": {
                "everyone": True,
                "conditions": conditions
            }
        })

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": rules,
            "default": "deny"
        })
        print(f"      Policy with {len(rules)} rules, 10 conditions each accepted")
        return True
    except ValueError as e:
        print(f"      Acceptable policy rejected: {e}")
        return False

test("Acceptable policy accepted", test_acceptable_limits)

def test_edge_case_max_rules():
    """Test that exactly MAX_RULES is accepted."""
    rules = []
    for i in range(PolicyLimits.MAX_RULES):
        rules.append({
            "name": f"rule_{i}",
            "allow": {
                "everyone": True,
                "conditions": [f"document.id == '{i}'"]
            }
        })

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": rules,
            "default": "deny"
        })
        print(f"      Policy with exactly {PolicyLimits.MAX_RULES} rules accepted")
        return True
    except ValueError as e:
        print(f"      Policy with MAX_RULES rejected: {e}")
        return False

test("Edge case: exactly MAX_RULES accepted", test_edge_case_max_rules)

def test_edge_case_max_conditions():
    """Test that exactly MAX_CONDITIONS_PER_RULE is accepted."""
    conditions = [f"document.field_{i} == 'value'" for i in range(PolicyLimits.MAX_CONDITIONS_PER_RULE)]

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "max-conditions",
                "allow": {
                    "everyone": True,
                    "conditions": conditions
                }
            }],
            "default": "deny"
        })
        print(f"      Rule with exactly {PolicyLimits.MAX_CONDITIONS_PER_RULE} conditions accepted")
        return True
    except ValueError as e:
        print(f"      Rule with MAX_CONDITIONS_PER_RULE rejected: {e}")
        return False

test("Edge case: exactly MAX_CONDITIONS_PER_RULE accepted", test_edge_case_max_conditions)

def test_empty_list_not_counted():
    """Test that empty lists don't trigger size limit."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "empty-list",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.category in []"]
                }
            }],
            "default": "deny"
        })
        print(f"      Empty list accepted (correct)")
        return True
    except ValueError as e:
        print(f"      Empty list rejected: {e}")
        return False

test("Empty list not counted against limit", test_empty_list_not_counted)

def test_limit_values():
    """Test that limit values are reasonable."""
    print(f"      MAX_RULES: {PolicyLimits.MAX_RULES}")
    print(f"      MAX_CONDITIONS_PER_RULE: {PolicyLimits.MAX_CONDITIONS_PER_RULE}")
    print(f"      MAX_LIST_SIZE: {PolicyLimits.MAX_LIST_SIZE}")
    print(f"      MAX_POLICY_SIZE_BYTES: {PolicyLimits.MAX_POLICY_SIZE_BYTES}")
    print(f"      MAX_NESTING_DEPTH: {PolicyLimits.MAX_NESTING_DEPTH}")

    # Verify limits are reasonable
    checks = [
        PolicyLimits.MAX_RULES >= 10,
        PolicyLimits.MAX_CONDITIONS_PER_RULE >= 10,
        PolicyLimits.MAX_LIST_SIZE >= 100,
        PolicyLimits.MAX_POLICY_SIZE_BYTES >= 100_000,
        PolicyLimits.MAX_NESTING_DEPTH >= 5,
    ]

    if all(checks):
        print(f"      All limits are reasonable")
        return True
    else:
        print(f"      Some limits are too restrictive")
        return False

test("Limit values are reasonable", test_limit_values)

# Summary
print("\n" + "=" * 80)
print("QUERY COMPLEXITY LIMITS TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL COMPLEXITY LIMIT TESTS PASSED!")
    print("\nComplexity limits are working correctly to prevent DoS attacks.")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
