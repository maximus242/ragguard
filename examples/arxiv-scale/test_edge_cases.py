#!/usr/bin/env python3
"""
Comprehensive edge case testing for RAGGuard v0.2.0
"""

from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, Policy
from sentence_transformers import SentenceTransformer
from ragguard.policy.engine import PolicyEngine

print("=" * 70)
print("Edge Case Testing Suite")
print("=" * 70)

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
        print(f"   ❌ ERROR: {str(e)[:80]}")
        tests_failed += 1

# Test 1: Large list literals (100+ items)
def test_large_list():
    """Test list with 100+ items."""
    categories = [f"cs.{i:03d}" for i in range(100)]
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "many-categories",
            "allow": {
                "everyone": True,
                "conditions": [f"document.category in {categories}"]
            }
        }],
        "default": "deny"
    })

    retriever = QdrantSecureRetriever(
        client=client,
        collection="arxiv_2400_papers",
        policy=policy,
        embed_fn=model.encode
    )

    results = retriever.search("test", user={"id": "test"}, limit=5)
    print(f"      Large list (100 items): {len(results)} results")
    return True

test("Large list literal (100+ items)", test_large_list)

# Test 2: Multiple negations
def test_multiple_negations():
    """Test multiple != conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "multiple-negations",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.access_level != 'restricted'",
                    "document.status != 'archived'",
                    "document.type != 'draft'"
                ]
            }
        }],
        "default": "deny"
    })

    retriever = QdrantSecureRetriever(
        client=client,
        collection="arxiv_2400_papers",
        policy=policy,
        embed_fn=model.encode
    )

    results = retriever.search("test", user={"id": "test"}, limit=10)

    # Verify no restricted, archived, or draft docs
    for r in results:
        if r.payload.get("access_level") == "restricted":
            print(f"      Found restricted doc!")
            return False
        if r.payload.get("status") == "archived":
            print(f"      Found archived doc!")
            return False
        if r.payload.get("type") == "draft":
            print(f"      Found draft doc!")
            return False

    print(f"      Multiple negations: {len(results)} results, all valid")
    return True

test("Multiple negation operators", test_multiple_negations)

# Test 3: Mixed operators
def test_mixed_operators():
    """Test combining ==, !=, and in operators."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "mixed",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.category in ['cs.AI', 'cs.LG']",
                    "document.access_level != 'restricted'",
                    "document.type == 'paper'"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow
    doc_allowed = {"category": "cs.AI", "access_level": "public", "type": "paper"}
    if not engine.evaluate({}, doc_allowed):
        print(f"      Should allow but denied!")
        return False

    # Should deny (wrong category)
    doc_deny1 = {"category": "cs.CV", "access_level": "public", "type": "paper"}
    if engine.evaluate({}, doc_deny1):
        print(f"      Should deny (wrong category) but allowed!")
        return False

    # Should deny (restricted)
    doc_deny2 = {"category": "cs.AI", "access_level": "restricted", "type": "paper"}
    if engine.evaluate({}, doc_deny2):
        print(f"      Should deny (restricted) but allowed!")
        return False

    # Should deny (wrong type)
    doc_deny3 = {"category": "cs.AI", "access_level": "public", "type": "draft"}
    if engine.evaluate({}, doc_deny3):
        print(f"      Should deny (wrong type) but allowed!")
        return False

    print(f"      Mixed operators working correctly")
    return True

test("Mixed operators (==, !=, in)", test_mixed_operators)

# Test 4: Numeric comparisons with !=
def test_numeric_negation():
    """Test != with numbers."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-zero",
            "allow": {
                "everyone": True,
                "conditions": ["document.version != 0"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow
    doc1 = {"version": 1}
    if not engine.evaluate({}, doc1):
        return False

    # Should deny
    doc2 = {"version": 0}
    if engine.evaluate({}, doc2):
        return False

    print(f"      Numeric negation working")
    return True

test("Numeric negation (version != 0)", test_numeric_negation)

# Test 5: Boolean with !=
def test_boolean_negation():
    """Test != with booleans."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-deleted",
            "allow": {
                "everyone": True,
                "conditions": ["document.deleted != true"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow
    doc1 = {"deleted": False}
    if not engine.evaluate({}, doc1):
        return False

    # Should deny
    doc2 = {"deleted": True}
    if engine.evaluate({}, doc2):
        return False

    print(f"      Boolean negation working")
    return True

test("Boolean negation (deleted != true)", test_boolean_negation)

# Test 6: Empty list edge case
def test_empty_list():
    """Test behavior with empty list."""
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

    engine = PolicyEngine(policy)

    # Should deny everything (empty list matches nothing)
    doc = {"category": "cs.AI"}
    if engine.evaluate({}, doc):
        print(f"      Empty list should match nothing!")
        return False

    print(f"      Empty list correctly matches nothing")
    return True

test("Empty list behavior", test_empty_list)

# Test 7: Whitespace handling
def test_whitespace():
    """Test conditions with extra whitespace."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "whitespace",
            "allow": {
                "everyone": True,
                "conditions": [
                    "  document.field   !=   'value'  ",
                    "document.category  in  [ 'cs.AI' , 'cs.LG' ]"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    doc = {"field": "other", "category": "cs.AI"}
    if not engine.evaluate({}, doc):
        print(f"      Whitespace handling failed!")
        return False

    print(f"      Whitespace handled correctly")
    return True

test("Whitespace in conditions", test_whitespace)

# Test 8: Unicode in list literals
def test_unicode_lists():
    """Test list with unicode characters."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "unicode",
            "allow": {
                "everyone": True,
                "conditions": ["document.lang in ['中文', '日本語', 'العربية']"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    doc1 = {"lang": "中文"}
    if not engine.evaluate({}, doc1):
        print(f"      Unicode in list failed!")
        return False

    doc2 = {"lang": "English"}
    if engine.evaluate({}, doc2):
        print(f"      Unicode list should have excluded English!")
        return False

    print(f"      Unicode in lists working")
    return True

test("Unicode in list literals", test_unicode_lists)

# Test 9: Case sensitivity
def test_case_sensitivity():
    """Test that string comparisons are case-sensitive."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "case-test",
            "allow": {
                "everyone": True,
                "conditions": ["document.level != 'Public'"]  # Capital P
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # lowercase 'public' should not match 'Public'
    doc = {"level": "public"}
    if not engine.evaluate({}, doc):
        print(f"      Case sensitivity issue: 'public' matched 'Public'!")
        return False

    print(f"      Case sensitivity working correctly")
    return True

test("Case sensitivity", test_case_sensitivity)

# Test 10: Nested conditions
def test_nested_conditions():
    """Test deeply nested field access."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "nested",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.metadata.security.level != 'top-secret'"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    doc1 = {"metadata": {"security": {"level": "public"}}}
    if not engine.evaluate({}, doc1):
        print(f"      Nested field access failed!")
        return False

    doc2 = {"metadata": {"security": {"level": "top-secret"}}}
    if engine.evaluate({}, doc2):
        print(f"      Nested negation failed!")
        return False

    print(f"      Nested conditions working")
    return True

test("Nested field conditions", test_nested_conditions)

# Summary
print("\n" + "=" * 70)
print("EDGE CASE TEST SUMMARY")
print("=" * 70)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL EDGE CASE TESTS PASSED!")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 70)
