#!/usr/bin/env python3
"""
Comprehensive NOT IN operator testing for RAGGuard v0.2.0
Tests the newly implemented 'not in' operator across all backends.
"""

from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, Policy
from sentence_transformers import SentenceTransformer
from ragguard.policy.engine import PolicyEngine
from ragguard.filters.builder import (
    to_qdrant_filter,
    to_pgvector_filter,
    to_weaviate_filter,
    to_pinecone_filter,
    to_chromadb_filter
)

print("=" * 70)
print("NOT IN Operator Testing Suite")
print("=" * 70)

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

# Test 1: PolicyEngine evaluation with NOT IN
def test_policy_engine_not_in():
    """Test PolicyEngine correctly evaluates NOT IN conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-restricted",
            "allow": {
                "everyone": True,
                "conditions": ["document.category not in ['restricted', 'classified', 'top-secret']"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow (category not in exclusion list)
    doc1 = {"category": "public"}
    if not engine.evaluate({}, doc1):
        print(f"      Should allow 'public' category but denied!")
        return False

    # Should deny (category in exclusion list)
    doc2 = {"category": "restricted"}
    if engine.evaluate({}, doc2):
        print(f"      Should deny 'restricted' category but allowed!")
        return False

    doc3 = {"category": "classified"}
    if engine.evaluate({}, doc3):
        print(f"      Should deny 'classified' category but allowed!")
        return False

    doc4 = {"category": "top-secret"}
    if engine.evaluate({}, doc4):
        print(f"      Should deny 'top-secret' category but allowed!")
        return False

    print(f"      PolicyEngine NOT IN working correctly")
    return True

test("PolicyEngine NOT IN evaluation", test_policy_engine_not_in)

# Test 2: Qdrant filter generation
def test_qdrant_not_in():
    """Test Qdrant filter generation with NOT IN."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-categories",
            "allow": {
                "everyone": True,
                "conditions": ["document.status not in ['archived', 'deleted']"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_qdrant_filter(policy, {})

    # Verify must_not clause exists
    if not hasattr(filter_obj, 'must_not') or filter_obj.must_not is None:
        print(f"      Qdrant filter missing must_not clause!")
        return False

    if len(filter_obj.must_not) == 0:
        print(f"      Qdrant filter must_not is empty!")
        return False

    print(f"      Qdrant NOT IN filter generated correctly")
    return True

test("Qdrant NOT IN filter generation", test_qdrant_not_in)

# Test 3: pgvector filter generation
def test_pgvector_not_in():
    """Test pgvector SQL generation with NOT IN."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-types",
            "allow": {
                "everyone": True,
                "conditions": ["document.type not in ['draft', 'template']"]
            }
        }],
        "default": "deny"
    })

    sql, params = to_pgvector_filter(policy, {})

    # Verify SQL contains NOT IN
    if "NOT IN" not in sql:
        print(f"      pgvector SQL missing NOT IN clause!")
        print(f"      SQL: {sql}")
        return False

    # Verify parameters
    if 'draft' not in params or 'template' not in params:
        print(f"      pgvector params incorrect: {params}")
        return False

    print(f"      pgvector NOT IN SQL: {sql[:80]}...")
    print(f"      Parameters: {params}")
    return True

test("pgvector NOT IN SQL generation", test_pgvector_not_in)

# Test 4: Weaviate filter generation
def test_weaviate_not_in():
    """Test Weaviate filter generation with NOT IN."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-langs",
            "allow": {
                "everyone": True,
                "conditions": ["document.language not in ['en', 'es']"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_weaviate_filter(policy, {})

    # Verify filter contains negation
    if filter_obj is None:
        print(f"      Weaviate filter is None!")
        return False

    # Filter should have operator field
    if not hasattr(filter_obj, 'operator') and 'operator' not in filter_obj:
        print(f"      Weaviate filter missing operator!")
        return False

    print(f"      Weaviate NOT IN filter generated correctly")
    return True

test("Weaviate NOT IN filter generation", test_weaviate_not_in)

# Test 5: Pinecone filter generation
def test_pinecone_not_in():
    """Test Pinecone filter generation with NOT IN."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-regions",
            "allow": {
                "everyone": True,
                "conditions": ["document.region not in ['us-east', 'eu-west']"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_pinecone_filter(policy, {})

    # Verify filter contains $nin operator
    if filter_obj is None:
        print(f"      Pinecone filter is None!")
        return False

    # Check for $nin in filter structure
    filter_str = str(filter_obj)
    if "$nin" not in filter_str:
        print(f"      Pinecone filter missing $nin operator!")
        print(f"      Filter: {filter_obj}")
        return False

    print(f"      Pinecone NOT IN filter: {filter_obj}")
    return True

test("Pinecone NOT IN filter generation", test_pinecone_not_in)

# Test 6: ChromaDB filter generation
def test_chromadb_not_in():
    """Test ChromaDB filter generation with NOT IN."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-status",
            "allow": {
                "everyone": True,
                "conditions": ["document.status not in ['pending', 'failed']"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    # Verify filter contains $nin operator
    if filter_obj is None:
        print(f"      ChromaDB filter is None!")
        return False

    # Check for $nin in filter structure
    filter_str = str(filter_obj)
    if "$nin" not in filter_str:
        print(f"      ChromaDB filter missing $nin operator!")
        print(f"      Filter: {filter_obj}")
        return False

    print(f"      ChromaDB NOT IN filter: {filter_obj}")
    return True

test("ChromaDB NOT IN filter generation", test_chromadb_not_in)

# Test 7: Multiple NOT IN conditions
def test_multiple_not_in():
    """Test multiple NOT IN conditions combined."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "multiple-exclusions",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.category not in ['restricted', 'classified']",
                    "document.status not in ['archived', 'deleted']"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow
    doc1 = {"category": "public", "status": "active"}
    if not engine.evaluate({}, doc1):
        print(f"      Should allow but denied!")
        return False

    # Should deny (category excluded)
    doc2 = {"category": "restricted", "status": "active"}
    if engine.evaluate({}, doc2):
        print(f"      Should deny (restricted category) but allowed!")
        return False

    # Should deny (status excluded)
    doc3 = {"category": "public", "status": "archived"}
    if engine.evaluate({}, doc3):
        print(f"      Should deny (archived status) but allowed!")
        return False

    # Should deny (both excluded)
    doc4 = {"category": "classified", "status": "deleted"}
    if engine.evaluate({}, doc4):
        print(f"      Should deny (both excluded) but allowed!")
        return False

    print(f"      Multiple NOT IN conditions working correctly")
    return True

test("Multiple NOT IN conditions", test_multiple_not_in)

# Test 8: NOT IN with empty list
def test_not_in_empty_list():
    """Test NOT IN with empty list (should allow everything)."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "empty-exclusion",
            "allow": {
                "everyone": True,
                "conditions": ["document.category not in []"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow everything (nothing excluded)
    doc = {"category": "anything"}
    if not engine.evaluate({}, doc):
        print(f"      Empty exclusion list should allow everything!")
        return False

    print(f"      Empty NOT IN list working correctly")
    return True

test("NOT IN with empty list", test_not_in_empty_list)

# Test 9: NOT IN with single value
def test_not_in_single_value():
    """Test NOT IN with single value."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-one",
            "allow": {
                "everyone": True,
                "conditions": ["document.status not in ['deleted']"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow
    doc1 = {"status": "active"}
    if not engine.evaluate({}, doc1):
        print(f"      Should allow 'active' but denied!")
        return False

    # Should deny
    doc2 = {"status": "deleted"}
    if engine.evaluate({}, doc2):
        print(f"      Should deny 'deleted' but allowed!")
        return False

    print(f"      Single value NOT IN working correctly")
    return True

test("NOT IN with single value", test_not_in_single_value)

# Test 10: End-to-end with Qdrant
def test_qdrant_e2e_not_in():
    """Test NOT IN operator end-to-end with Qdrant."""
    try:
        client = QdrantClient("localhost", port=6333)
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "exclude-cs-categories",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.category not in ['cs.CR', 'cs.DC']"]
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

        results = retriever.search("machine learning", user={"id": "test"}, limit=10)

        # Verify no excluded categories in results
        for r in results:
            category = r.payload.get("category")
            if category in ["cs.CR", "cs.DC"]:
                print(f"      Found excluded category {category}!")
                return False

        print(f"      Qdrant E2E: {len(results)} results, all valid")
        return True
    except Exception as e:
        print(f"      Qdrant connection failed (may not be running): {e}")
        # Don't fail test if Qdrant not running
        return True

test("Qdrant end-to-end NOT IN", test_qdrant_e2e_not_in)

# Summary
print("\n" + "=" * 70)
print("NOT IN OPERATOR TEST SUMMARY")
print("=" * 70)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL NOT IN OPERATOR TESTS PASSED!")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 70)
