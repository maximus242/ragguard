#!/usr/bin/env python3
"""
Field Existence Tests for RAGGuard v0.2.0

Tests that RAGGuard handles field existence checks correctly.
"""

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine

print("=" * 80)
print("Field Existence Tests")
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
# FIELD EXISTENCE TESTS
# ============================================================================

def test_field_exists():
    """Test document.field exists"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "has-reviewed-date",
            "allow": {
                "everyone": True,
                "conditions": ["document.reviewed_at exists"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: Document with reviewed_at field
    doc1 = {"id": "doc1", "reviewed_at": "2025-01-01"}
    if not engine.evaluate({}, doc1):
        print(f"      Document with reviewed_at should be allowed")
        return False

    # Test 2: Document without reviewed_at field
    doc2 = {"id": "doc2"}
    if engine.evaluate({}, doc2):
        print(f"      Document without reviewed_at should be denied")
        return False

    # Test 3: Document with reviewed_at = None (should be denied)
    doc3 = {"id": "doc3", "reviewed_at": None}
    if engine.evaluate({}, doc3):
        print(f"      Document with reviewed_at=None should be denied")
        return False

    print(f"      document.field exists works correctly")
    return True

test("document.field exists", test_field_exists)

def test_field_not_exists():
    """Test document.field not exists"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "no-draft-notes",
            "allow": {
                "everyone": True,
                "conditions": ["document.draft_notes not exists"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: Document without draft_notes field
    doc1 = {"id": "doc1", "title": "Final"}
    if not engine.evaluate({}, doc1):
        print(f"      Document without draft_notes should be allowed")
        return False

    # Test 2: Document with draft_notes field
    doc2 = {"id": "doc2", "draft_notes": "TODO"}
    if engine.evaluate({}, doc2):
        print(f"      Document with draft_notes should be denied")
        return False

    # Test 3: Document with draft_notes = None (should be allowed)
    doc3 = {"id": "doc3", "draft_notes": None}
    if not engine.evaluate({}, doc3):
        print(f"      Document with draft_notes=None should be allowed")
        return False

    print(f"      document.field not exists works correctly")
    return True

test("document.field not exists", test_field_not_exists)

def test_exists_combined_with_other_conditions():
    """Test field existence combined with other conditions"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.reviewed_at exists",
                    "document.status == 'published'",
                    "document.draft_notes not exists"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: All conditions met
    doc1 = {
        "id": "doc1",
        "reviewed_at": "2025-01-01",
        "status": "published"
    }
    if not engine.evaluate({}, doc1):
        print(f"      All conditions met should allow")
        return False

    # Test 2: Missing reviewed_at
    doc2 = {
        "id": "doc2",
        "status": "published"
    }
    if engine.evaluate({}, doc2):
        print(f"      Missing reviewed_at should deny")
        return False

    # Test 3: Wrong status
    doc3 = {
        "id": "doc3",
        "reviewed_at": "2025-01-01",
        "status": "draft"
    }
    if engine.evaluate({}, doc3):
        print(f"      Wrong status should deny")
        return False

    # Test 4: Has draft_notes (should deny)
    doc4 = {
        "id": "doc4",
        "reviewed_at": "2025-01-01",
        "status": "published",
        "draft_notes": "Some notes"
    }
    if engine.evaluate({}, doc4):
        print(f"      Has draft_notes should deny")
        return False

    print(f"      Combined conditions with exists work correctly")
    return True

test("Exists combined with other conditions", test_exists_combined_with_other_conditions)

def test_nested_field_exists():
    """Test nested field existence"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "has-metadata-timestamp",
            "allow": {
                "everyone": True,
                "conditions": ["document.metadata.timestamp exists"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: Document with nested metadata.timestamp
    doc1 = {
        "id": "doc1",
        "metadata": {
            "timestamp": "2025-01-01"
        }
    }
    if not engine.evaluate({}, doc1):
        print(f"      Document with metadata.timestamp should be allowed")
        return False

    # Test 2: Document with metadata but no timestamp
    doc2 = {
        "id": "doc2",
        "metadata": {
            "author": "Alice"
        }
    }
    if engine.evaluate({}, doc2):
        print(f"      Document without metadata.timestamp should be denied")
        return False

    # Test 3: Document without metadata
    doc3 = {"id": "doc3"}
    if engine.evaluate({}, doc3):
        print(f"      Document without metadata should be denied")
        return False

    print(f"      Nested field existence works correctly")
    return True

test("Nested field exists (document.metadata.timestamp exists)", test_nested_field_exists)

def test_user_field_exists():
    """Test user field existence"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "has-clearance",
            "allow": {
                "conditions": ["user.clearance_level exists"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: User with clearance_level
    user1 = {"id": "alice", "clearance_level": "secret"}
    doc = {"id": "doc1"}
    if not engine.evaluate(user1, doc):
        print(f"      User with clearance_level should be allowed")
        return False

    # Test 2: User without clearance_level
    user2 = {"id": "bob"}
    if engine.evaluate(user2, doc):
        print(f"      User without clearance_level should be denied")
        return False

    # Test 3: User with clearance_level = None (should be denied)
    user3 = {"id": "charlie", "clearance_level": None}
    if engine.evaluate(user3, doc):
        print(f"      User with clearance_level=None should be denied")
        return False

    print(f"      User field existence works correctly")
    return True

test("User field exists (user.clearance_level exists)", test_user_field_exists)

# Summary
print("\n" + "=" * 80)
print("FIELD EXISTENCE TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL FIELD EXISTENCE TESTS PASSED!")
    print("\n✅ RAGGuard now supports:")
    print("  - document.field exists")
    print("  - document.field not exists")
    print("  - user.field exists")
    print("  - Nested field existence (document.metadata.field exists)")
    print("  - Combined with other conditions")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
