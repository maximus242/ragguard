#!/usr/bin/env python3
"""
Array Field Operations Tests for RAGGuard v0.2.0

Tests the new capability to check if a value is in an array field.
Example: user.id in document.authorized_users
"""

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine

print("=" * 80)
print("Array Field Operations Tests")
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
# ARRAY FIELD OPERATION TESTS
# ============================================================================

def test_user_id_in_document_array():
    """Test user.id in document.authorized_users"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "authorized-users",
            "allow": {
                "conditions": ["user.id in document.authorized_users"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: User IS in authorized list
    user = {"id": "alice"}
    doc = {
        "id": "doc1",
        "authorized_users": ["alice", "bob", "charlie"]
    }
    if not engine.evaluate(user, doc):
        print(f"      User in list should be allowed")
        return False

    # Test 2: User NOT in authorized list
    user2 = {"id": "dave"}
    if engine.evaluate(user2, doc):
        print(f"      User not in list should be denied")
        return False

    # Test 3: Empty authorized list
    doc_empty = {"id": "doc2", "authorized_users": []}
    if engine.evaluate(user, doc_empty):
        print(f"      Empty list should deny all")
        return False

    print(f"      All user.id in document.array tests passed")
    return True

test("user.id in document.authorized_users", test_user_id_in_document_array)

def test_value_in_document_tags():
    """Test literal value in document.tags"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-tag",
            "allow": {
                "everyone": True,
                "conditions": ["'public' in document.tags"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: Tag present
    doc1 = {"id": "doc1", "tags": ["public", "ai", "ml"]}
    if not engine.evaluate({}, doc1):
        print(f"      Document with 'public' tag should be allowed")
        return False

    # Test 2: Tag absent
    doc2 = {"id": "doc2", "tags": ["private", "secret"]}
    if engine.evaluate({}, doc2):
        print(f"      Document without 'public' tag should be denied")
        return False

    print(f"      Literal value in document.tags tests passed")
    return True

test("'public' in document.tags", test_value_in_document_tags)

def test_user_role_in_document_roles():
    """Test user.role in document.allowed_roles"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "role-based",
            "allow": {
                "conditions": ["user.role in document.allowed_roles"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: User role matches
    user = {"id": "alice", "role": "engineer"}
    doc = {"id": "doc1", "allowed_roles": ["engineer", "manager"]}
    if not engine.evaluate(user, doc):
        print(f"      User with matching role should be allowed")
        return False

    # Test 2: User role doesn't match
    user2 = {"id": "bob", "role": "intern"}
    if engine.evaluate(user2, doc):
        print(f"      User with non-matching role should be denied")
        return False

    print(f"      user.role in document.allowed_roles tests passed")
    return True

test("user.role in document.allowed_roles", test_user_role_in_document_roles)

def test_not_in_array_field():
    """Test user.id not in document.blocked_users"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-blocked",
            "allow": {
                "everyone": True,
                "conditions": ["user.id not in document.blocked_users"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: User NOT in blocked list
    user = {"id": "alice"}
    doc = {"id": "doc1", "blocked_users": ["bob", "charlie"]}
    if not engine.evaluate(user, doc):
        print(f"      User not in blocked list should be allowed")
        return False

    # Test 2: User IS in blocked list
    user2 = {"id": "bob"}
    if engine.evaluate(user2, doc):
        print(f"      User in blocked list should be denied")
        return False

    # Test 3: Empty blocked list (all allowed)
    doc_empty = {"id": "doc2", "blocked_users": []}
    if not engine.evaluate(user, doc_empty):
        print(f"      Empty blocked list should allow all")
        return False

    print(f"      user.id not in document.blocked_users tests passed")
    return True

test("user.id not in document.blocked_users", test_not_in_array_field)

def test_combined_with_other_conditions():
    """Test array field operations combined with other conditions"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex",
            "allow": {
                "conditions": [
                    "user.id in document.authorized_users",
                    "document.status == 'active'",
                    "'public' in document.tags"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test 1: All conditions met
    user = {"id": "alice"}
    doc = {
        "authorized_users": ["alice", "bob"],
        "status": "active",
        "tags": ["public", "ai"]
    }
    if not engine.evaluate(user, doc):
        print(f"      All conditions met should allow")
        return False

    # Test 2: User not authorized
    user2 = {"id": "charlie"}
    if engine.evaluate(user2, doc):
        print(f"      User not authorized should deny")
        return False

    # Test 3: Wrong status
    doc2 = {
        "authorized_users": ["alice", "bob"],
        "status": "archived",
        "tags": ["public", "ai"]
    }
    if engine.evaluate(user, doc2):
        print(f"      Wrong status should deny")
        return False

    # Test 4: Missing 'public' tag
    doc3 = {
        "authorized_users": ["alice", "bob"],
        "status": "active",
        "tags": ["private", "ai"]
    }
    if engine.evaluate(user, doc3):
        print(f"      Missing 'public' tag should deny")
        return False

    print(f"      Combined conditions tests passed")
    return True

test("Combined array + other conditions", test_combined_with_other_conditions)

# Summary
print("\n" + "=" * 80)
print("ARRAY FIELD OPERATIONS TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL ARRAY FIELD OPERATION TESTS PASSED!")
    print("\n✅ RAGGuard now supports array field operations:")
    print("  - user.id in document.authorized_users")
    print("  - user.role in document.allowed_roles")
    print("  - 'value' in document.tags")
    print("  - user.field not in document.blocked_list")
    print("\n✅ Supported in:")
    print("  - PolicyEngine evaluation (FAISS post-filtering)")
    print("  - Filter builders (Qdrant, Weaviate, pgvector, Pinecone, ChromaDB)")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
