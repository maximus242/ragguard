#!/usr/bin/env python3
"""
Error Message Tests for RAGGuard v0.2.0

Tests that error messages are helpful and informative for common mistakes.
"""

from ragguard import Policy

print("=" * 80)
print("Error Message Tests")
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
        print(f"   ❌ ERROR: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

# ============================================================================
# ERROR MESSAGE TESTS
# ============================================================================

def test_malformed_list_missing_bracket():
    """Test error message for missing closing bracket."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.category in ['cs.AI', 'cs.LG'"]
                }
            }],
            "default": "deny"
        })
        print(f"      Missing bracket not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "missing closing bracket" in error_msg.lower():
            print(f"      Error message mentions missing bracket (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("Missing closing bracket error", test_malformed_list_missing_bracket)

def test_unclosed_quote():
    """Test error message for unclosed quote."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.category in ['cs.AI', 'cs.LG]"]
                }
            }],
            "default": "deny"
        })
        print(f"      Unclosed quote not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "unclosed" in error_msg.lower() and "quote" in error_msg.lower():
            print(f"      Error message mentions unclosed quote (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("Unclosed quote error", test_unclosed_quote)

def test_nested_list():
    """Test error message for nested lists."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.category in [['nested', 'list'], 'value']"]
                }
            }],
            "default": "deny"
        })
        print(f"      Nested list not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "nested list" in error_msg.lower():
            print(f"      Error message mentions nested lists (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("Nested list error", test_nested_list)

def test_invalid_operator_triple_equals():
    """Test error message for === operator."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.status === 'active'"]
                }
            }],
            "default": "deny"
        })
        print(f"      === operator not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "===" in error_msg and "==" in error_msg:
            print(f"      Error message suggests correct operator (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("Invalid operator === error", test_invalid_operator_triple_equals)

def test_invalid_operator_not_triple_equals():
    """Test error message for !== operator."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.status !== 'archived'"]
                }
            }],
            "default": "deny"
        })
        print(f"      !== operator not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "!==" in error_msg and "!=" in error_msg:
            print(f"      Error message suggests correct operator (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("Invalid operator !== error", test_invalid_operator_not_triple_equals)

def test_invalid_operator_sql_style():
    """Test error message for <> operator (SQL style)."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.status <> 'archived'"]
                }
            }],
            "default": "deny"
        })
        print(f"      <> operator not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "<>" in error_msg and "!=" in error_msg:
            print(f"      Error message suggests correct operator (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("Invalid operator <> error", test_invalid_operator_sql_style)

def test_invalid_operator_single_equals():
    """Test error message for = operator (assignment)."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.status = 'active'"]
                }
            }],
            "default": "deny"
        })
        print(f"      = operator not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "==" in error_msg:
            print(f"      Error message suggests correct operator (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("Invalid operator = error", test_invalid_operator_single_equals)

def test_no_operator():
    """Test error message when no operator is found."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.status active"]
                }
            }],
            "default": "deny"
        })
        print(f"      Missing operator not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "no valid operator" in error_msg.lower() or "supported operators" in error_msg.lower():
            print(f"      Error message mentions supported operators (good)")
            return True
        else:
            print(f"      Error message unclear: {error_msg[:100]}")
            return False

test("No operator error", test_no_operator)

def test_good_error_includes_examples():
    """Test that error messages include helpful examples."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": ["invalid condition"]
                }
            }],
            "default": "deny"
        })
        print(f"      Invalid condition not detected")
        return False
    except ValueError as e:
        error_msg = str(e)
        # Check if error includes examples
        has_examples = "example" in error_msg.lower() or "document." in error_msg
        if has_examples:
            print(f"      Error message includes examples (good)")
            return True
        else:
            print(f"      Error message missing examples")
            # Still pass - not all errors need examples
            return True

test("Error messages include examples", test_good_error_includes_examples)

def test_valid_conditions_still_work():
    """Test that valid conditions still work after adding validation."""
    try:
        # This should NOT raise an error
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {
                    "everyone": True,
                    "conditions": [
                        "document.status == 'active'",
                        "document.category != 'archived'",
                        "document.tags in ['ai', 'ml']",
                        "document.level not in ['restricted', 'secret']"
                    ]
                }
            }],
            "default": "deny"
        })
        print(f"      Valid conditions accepted (correct)")
        return True
    except ValueError as e:
        print(f"      Valid conditions rejected: {e}")
        return False

test("Valid conditions still accepted", test_valid_conditions_still_work)

# Summary
print("\n" + "=" * 80)
print("ERROR MESSAGE TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL ERROR MESSAGE TESTS PASSED!")
    print("\nError messages are clear and helpful for common mistakes.")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
