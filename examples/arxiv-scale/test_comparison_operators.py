#!/usr/bin/env python3
"""
Comparison Operators Tests for RAGGuard v0.2.0

Tests that RAGGuard handles comparison operators correctly.
"""

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine

print("=" * 80)
print("Comparison Operators Tests")
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
# COMPARISON OPERATOR TESTS
# ============================================================================

def test_greater_than():
    """Test document.field > value"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "high-priority",
            "allow": {
                "everyone": True,
                "conditions": ["document.priority > 5"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: priority > 5
    doc1 = {"id": "doc1", "priority": 10}
    if not engine.evaluate({}, doc1):
        print(f"      priority=10 should be > 5")
        return False

    # Test 2: priority = 5 (not > 5)
    doc2 = {"id": "doc2", "priority": 5}
    if engine.evaluate({}, doc2):
        print(f"      priority=5 should not be > 5")
        return False

    # Test 3: priority < 5
    doc3 = {"id": "doc3", "priority": 3}
    if engine.evaluate({}, doc3):
        print(f"      priority=3 should not be > 5")
        return False

    print(f"      Greater than operator works correctly")
    return True

test("document.field > value", test_greater_than)

def test_less_than():
    """Test document.field < value"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "low-score",
            "allow": {
                "everyone": True,
                "conditions": ["document.score < 0.5"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: score < 0.5
    doc1 = {"id": "doc1", "score": 0.3}
    if not engine.evaluate({}, doc1):
        print(f"      score=0.3 should be < 0.5")
        return False

    # Test 2: score = 0.5 (not < 0.5)
    doc2 = {"id": "doc2", "score": 0.5}
    if engine.evaluate({}, doc2):
        print(f"      score=0.5 should not be < 0.5")
        return False

    # Test 3: score > 0.5
    doc3 = {"id": "doc3", "score": 0.7}
    if engine.evaluate({}, doc3):
        print(f"      score=0.7 should not be < 0.5")
        return False

    print(f"      Less than operator works correctly")
    return True

test("document.field < value", test_less_than)

def test_greater_than_or_equal():
    """Test document.field >= value"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "min-confidence",
            "allow": {
                "everyone": True,
                "conditions": ["document.confidence >= 0.8"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: confidence > 0.8
    doc1 = {"id": "doc1", "confidence": 0.9}
    if not engine.evaluate({}, doc1):
        print(f"      confidence=0.9 should be >= 0.8")
        return False

    # Test 2: confidence = 0.8 (should pass)
    doc2 = {"id": "doc2", "confidence": 0.8}
    if not engine.evaluate({}, doc2):
        print(f"      confidence=0.8 should be >= 0.8")
        return False

    # Test 3: confidence < 0.8
    doc3 = {"id": "doc3", "confidence": 0.7}
    if engine.evaluate({}, doc3):
        print(f"      confidence=0.7 should not be >= 0.8")
        return False

    print(f"      Greater than or equal operator works correctly")
    return True

test("document.field >= value", test_greater_than_or_equal)

def test_less_than_or_equal():
    """Test document.field <= value"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "max-size",
            "allow": {
                "everyone": True,
                "conditions": ["document.size_mb <= 10"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: size < 10
    doc1 = {"id": "doc1", "size_mb": 5}
    if not engine.evaluate({}, doc1):
        print(f"      size=5 should be <= 10")
        return False

    # Test 2: size = 10 (should pass)
    doc2 = {"id": "doc2", "size_mb": 10}
    if not engine.evaluate({}, doc2):
        print(f"      size=10 should be <= 10")
        return False

    # Test 3: size > 10
    doc3 = {"id": "doc3", "size_mb": 15}
    if engine.evaluate({}, doc3):
        print(f"      size=15 should not be <= 10")
        return False

    print(f"      Less than or equal operator works correctly")
    return True

test("document.field <= value", test_less_than_or_equal)

def test_comparison_with_user_fields():
    """Test user.field > document.field"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "clearance-level",
            "allow": {
                "conditions": ["user.clearance_level >= document.required_clearance"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: User clearance >= required
    user1 = {"id": "alice", "clearance_level": 5}
    doc1 = {"id": "doc1", "required_clearance": 3}
    if not engine.evaluate(user1, doc1):
        print(f"      clearance 5 >= required 3 should allow")
        return False

    # Test 2: User clearance = required
    user2 = {"id": "bob", "clearance_level": 3}
    if not engine.evaluate(user2, doc1):
        print(f"      clearance 3 >= required 3 should allow")
        return False

    # Test 3: User clearance < required
    user3 = {"id": "charlie", "clearance_level": 2}
    if engine.evaluate(user3, doc1):
        print(f"      clearance 2 >= required 3 should deny")
        return False

    print(f"      Comparison with user fields works correctly")
    return True

test("user.field >= document.field", test_comparison_with_user_fields)

def test_comparison_combined_conditions():
    """Test comparison operators combined with other conditions"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.priority > 5",
                    "document.confidence >= 0.8",
                    "document.status == 'active'"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: All conditions met
    doc1 = {
        "id": "doc1",
        "priority": 10,
        "confidence": 0.9,
        "status": "active"
    }
    if not engine.evaluate({}, doc1):
        print(f"      All conditions met should allow")
        return False

    # Test 2: Low priority
    doc2 = {
        "id": "doc2",
        "priority": 3,
        "confidence": 0.9,
        "status": "active"
    }
    if engine.evaluate({}, doc2):
        print(f"      Low priority should deny")
        return False

    # Test 3: Low confidence
    doc3 = {
        "id": "doc3",
        "priority": 10,
        "confidence": 0.7,
        "status": "active"
    }
    if engine.evaluate({}, doc3):
        print(f"      Low confidence should deny")
        return False

    print(f"      Combined comparison conditions work correctly")
    return True

test("Comparison operators combined with other conditions", test_comparison_combined_conditions)

# Summary
print("\n" + "=" * 80)
print("COMPARISON OPERATORS TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL COMPARISON OPERATOR TESTS PASSED!")
    print("\n✅ RAGGuard now supports:")
    print("  - document.field > value")
    print("  - document.field < value")
    print("  - document.field >= value")
    print("  - document.field <= value")
    print("  - user.field >= document.field")
    print("  - Combined with other conditions")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
