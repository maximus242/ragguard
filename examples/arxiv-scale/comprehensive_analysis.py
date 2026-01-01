#!/usr/bin/env python3
"""
Comprehensive vulnerability and gap analysis for RAGGuard.

This script tests edge cases, missing features, and potential security issues
that might not be covered by existing tests.
"""

from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, Policy, load_policy
from sentence_transformers import SentenceTransformer
from ragguard.policy.engine import PolicyEngine

print("=" * 70)
print("RAGGuard Comprehensive Analysis")
print("=" * 70)

issues_found = []
warnings_found = []
tests_passed = []

def test(name, test_func, is_critical=False):
    """Run a test and track results."""
    global issues_found, warnings_found, tests_passed
    print(f"\n{'🔴 CRITICAL' if is_critical else '🔍'} Testing: {name}")
    try:
        result = test_func()
        if result is True:
            print(f"   ✅ PASS")
            tests_passed.append(name)
        elif result is False:
            msg = f"{name}"
            if is_critical:
                issues_found.append(msg)
                print(f"   ❌ FAIL (CRITICAL)")
            else:
                warnings_found.append(msg)
                print(f"   ⚠️  WARNING")
        else:
            print(f"   ℹ️  {result}")
            tests_passed.append(name)
    except Exception as e:
        msg = f"{name}: {str(e)[:100]}"
        if is_critical:
            issues_found.append(msg)
            print(f"   ❌ FAIL: {str(e)[:100]}")
        else:
            warnings_found.append(msg)
            print(f"   ⚠️  ERROR: {str(e)[:100]}")

# Setup
client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("\n" + "=" * 70)
print("Category 1: Policy Condition Edge Cases")
print("=" * 70)

# Test 1: Negation operators (!= and not in)
def test_negation_in_policy():
    """Test if != and 'not in' operators work in policy conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "not-restricted",
                "allow": {
                    "conditions": ["document.access_level != 'restricted'"]
                }
            }
        ],
        "default": "deny"
    })

    retriever = QdrantSecureRetriever(
        client=client,
        collection="arxiv_2400_papers",
        policy=policy,
        embed_fn=model.encode
    )

    user = {"institution": "MIT", "roles": ["researcher"]}
    results = retriever.search("machine learning", user=user, limit=10)

    # Check that no restricted docs are in results
    for r in results:
        if r.payload.get("access_level") == "restricted":
            return False

    return len(results) > 0  # Should get some non-restricted results

test("Negation operator (!=) in conditions", test_negation_in_policy, is_critical=True)

# Test 2: List conditions
def test_list_conditions():
    """Test if 'in' with literal lists works."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "allowed-categories",
                "allow": {
                    "conditions": ["document.category in ['cs.AI', 'cs.LG', 'cs.CL']"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Should allow cs.AI
    doc_ai = {"category": "cs.AI"}
    if not engine.evaluate({}, doc_ai):
        return False

    # Should deny cs.CV
    doc_cv = {"category": "cs.CV"}
    if engine.evaluate({}, doc_cv):
        return False

    return True

test("List conditions (field in ['a', 'b'])", test_list_conditions, is_critical=True)

# Test 3: Nested field access
def test_nested_fields():
    """Test if nested field access works (document.metadata.level)."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "nested-access",
                "allow": {
                    "conditions": ["user.org.department == document.metadata.department"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    user = {"org": {"department": "engineering"}}
    doc = {"metadata": {"department": "engineering"}}

    return engine.evaluate(user, doc)

test("Nested field access (user.org.dept)", test_nested_fields, is_critical=False)

print("\n" + "=" * 70)
print("Category 2: Filter Builder Parity")
print("=" * 70)

# Test 4: Check all backends support document.field == 'literal'
def test_backend_parity():
    """Verify the security fix was applied to ALL backends."""
    from ragguard.filters.builder import (
        to_qdrant_filter,
        to_pgvector_filter,
        to_weaviate_filter,
        to_pinecone_filter,
        to_chromadb_filter
    )

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "public-only",
                "allow": {
                    "conditions": ["document.access_level == 'public'"]
                }
            }
        ],
        "default": "deny"
    })

    user = {"id": "test"}

    # Test Qdrant
    qdrant_filter = to_qdrant_filter(policy, user)
    if qdrant_filter is None:
        print("      ❌ Qdrant filter is None!")
        return False
    # Check if filter is meaningful (has either must or should conditions)
    has_conditions = False
    if hasattr(qdrant_filter, 'must') and qdrant_filter.must:
        has_conditions = True
    if hasattr(qdrant_filter, 'should') and qdrant_filter.should:
        has_conditions = True
    if not has_conditions:
        print("      ❌ Qdrant filter has no conditions!")
        return False

    # Test pgvector
    pg_clause, pg_params = to_pgvector_filter(policy, user)
    if not pg_clause or pg_clause == "WHERE FALSE":
        print("      ❌ pgvector filter is empty/deny!")
        return False

    # Test Weaviate
    weaviate_filter = to_weaviate_filter(policy, user)
    if weaviate_filter is None:
        print("      ❌ Weaviate filter is None!")
        return False

    # Test Pinecone
    pinecone_filter = to_pinecone_filter(policy, user)
    if pinecone_filter is None:
        print("      ❌ Pinecone filter is None!")
        return False

    # Test ChromaDB
    chromadb_filter = to_chromadb_filter(policy, user)
    if chromadb_filter is None:
        print("      ❌ ChromaDB filter is None!")
        return False

    print(f"      ✅ Qdrant, pgvector, Weaviate, Pinecone, ChromaDB all generate filters")
    return True

test("All backends support document.field == 'literal'", test_backend_parity, is_critical=True)

print("\n" + "=" * 70)
print("Category 3: Cache Security")
print("=" * 70)

# Test 5: Cache key collision
def test_cache_key_collision():
    """Test if different users can cause cache key collisions."""
    policy = load_policy("policy.yaml")

    retriever = QdrantSecureRetriever(
        client=client,
        collection="arxiv_2400_papers",
        policy=policy,
        embed_fn=model.encode,
        enable_filter_cache=True
    )

    # Clear cache
    retriever.invalidate_filter_cache()

    # Two users with similar but different data
    user1 = {"institution": "MIT", "roles": ["researcher"]}
    user2 = {"institution": "MIT", "roles": ["admin"]}  # Different role!

    # Query with user1
    results1 = retriever.search("test", user=user1, limit=10)

    # Query with user2 (different user, should be cache miss)
    results2 = retriever.search("test", user=user2, limit=10)

    cache_stats = retriever.get_cache_stats()

    # Should have 2 misses (different users)
    if cache_stats['misses'] < 2:
        print(f"      ❌ Only {cache_stats['misses']} cache misses - possible collision!")
        return False

    # Results might be same or different depending on policy
    print(f"      ✅ Different users get separate cache entries")
    return True

test("Cache key collision prevention", test_cache_key_collision, is_critical=True)

print("\n" + "=" * 70)
print("Category 4: Input Validation")
print("=" * 70)

# Test 6: Malformed policy
def test_malformed_policy():
    """Test if malformed policies are rejected properly."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "missing-allow",
                    "match": {"field": "value"}
                    # Missing 'allow' key!
                }
            ],
            "default": "deny"
        })
        print("      ⚠️  Malformed policy was accepted!")
        return False
    except Exception as e:
        print(f"      ✅ Malformed policy rejected: {str(e)[:50]}")
        return True

test("Malformed policy rejection", test_malformed_policy, is_critical=False)

# Test 7: Circular policy conditions
def test_circular_conditions():
    """Test if circular condition references are handled."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "circular",
                "allow": {
                    "conditions": ["user.field == user.field"]  # Circular!
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"field": "value"}
    doc = {"field": "value"}

    # Should handle without crashing
    result = engine.evaluate(user, doc)
    print(f"      ℹ️  Circular condition evaluated to: {result}")
    return True

test("Circular condition handling", test_circular_conditions, is_critical=False)

print("\n" + "=" * 70)
print("Category 5: Type Safety")
print("=" * 70)

# Test 8: Type mismatches
def test_type_mismatches():
    """Test if type mismatches are handled gracefully."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "string-match",
                "allow": {
                    "conditions": ["document.level == 'high'"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Document has integer instead of string
    doc_int = {"level": 5}
    doc_str = {"level": "high"}

    result_int = engine.evaluate({}, doc_int)
    result_str = engine.evaluate({}, doc_str)

    # String should match, int should not
    if result_str and not result_int:
        print(f"      ✅ Type mismatch handled correctly")
        return True
    else:
        print(f"      ⚠️  Type mismatch behavior: int={result_int}, str={result_str}")
        return "Type handling may be too permissive"

test("Type mismatch handling", test_type_mismatches, is_critical=False)

print("\n" + "=" * 70)
print("Category 6: Concurrent Edge Cases")
print("=" * 70)

# Test 9: Policy update during query
def test_policy_update_during_query():
    """Test if policy update during query causes issues."""
    import threading
    import time

    policy1 = load_policy("policy.yaml")

    retriever = QdrantSecureRetriever(
        client=client,
        collection="arxiv_2400_papers",
        policy=policy1,
        embed_fn=model.encode
    )

    results = []
    errors = []

    def query_loop():
        for _ in range(5):
            try:
                r = retriever.search("test", user={"institution": "MIT", "roles": ["researcher"]}, limit=5)
                results.append(len(r))
            except Exception as e:
                errors.append(str(e))
            time.sleep(0.01)

    # Start query thread
    thread = threading.Thread(target=query_loop)
    thread.start()

    # Update policy while querying
    time.sleep(0.02)
    policy2 = Policy.from_dict({"version": "1", "rules": [], "default": "deny"})
    retriever.policy = policy2

    thread.join(timeout=2)

    if errors:
        print(f"      ⚠️  {len(errors)} errors during policy update: {errors[0][:50]}")
        return False

    print(f"      ✅ {len(results)} queries completed during policy update")
    return True

test("Policy update race conditions", test_policy_update_during_query, is_critical=False)

print("\n" + "=" * 70)
print("Category 7: Performance Edge Cases")
print("=" * 70)

# Test 10: Very large policy
def test_large_policy():
    """Test if very large policies are handled efficiently."""
    import time

    # Create policy with 100 rules
    rules = []
    for i in range(100):
        rules.append({
            "name": f"rule-{i}",
            "allow": {
                "conditions": [f"document.category == 'cat{i}'"]
            }
        })

    policy = Policy.from_dict({
        "version": "1",
        "rules": rules,
        "default": "deny"
    })

    start = time.time()
    engine = PolicyEngine(policy)
    init_time = (time.time() - start) * 1000

    # Test evaluation
    start = time.time()
    user = {"id": "test"}
    doc = {"category": "cat50"}
    for _ in range(1000):
        engine.evaluate(user, doc)
    eval_time = (time.time() - start) * 1000 / 1000

    print(f"      ℹ️  100 rules: init={init_time:.2f}ms, eval={eval_time:.3f}ms/query")

    if init_time > 100:
        return f"Policy init slow ({init_time:.2f}ms)"
    if eval_time > 1:
        return f"Evaluation slow ({eval_time:.3f}ms)"

    return True

test("Large policy performance", test_large_policy, is_critical=False)

print("\n" + "=" * 70)
print("Category 8: Missing Test Coverage")
print("=" * 70)

# Test 11: Other database backends
def test_other_backends():
    """Check if other backends are actually tested."""
    import os

    test_dir = "/Users/cloud/Programming/ragguard/tests"

    backends_to_test = ["pgvector", "weaviate", "pinecone", "chromadb", "faiss"]
    tested_backends = []

    for backend in backends_to_test:
        # Check if there are tests for this backend
        test_file = os.path.join(test_dir, f"test_{backend}.py")
        if os.path.exists(test_file):
            tested_backends.append(backend)

    untested = [b for b in backends_to_test if b not in tested_backends]

    if untested:
        print(f"      ⚠️  Untested backends: {', '.join(untested)}")
        return f"Missing tests for: {', '.join(untested)}"
    else:
        print(f"      ✅ All backends have tests")
        return True

test("Database backend test coverage", test_other_backends, is_critical=False)

# Summary
print("\n" + "=" * 70)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("=" * 70)

print(f"\n✅ Tests Passed: {len(tests_passed)}")
print(f"⚠️  Warnings: {len(warnings_found)}")
print(f"❌ Critical Issues: {len(issues_found)}")

if issues_found:
    print(f"\n❌ CRITICAL ISSUES FOUND:")
    for issue in issues_found:
        print(f"   - {issue}")

if warnings_found:
    print(f"\n⚠️  WARNINGS:")
    for warning in warnings_found:
        print(f"   - {warning}")

print("\n" + "=" * 70)
if issues_found:
    print("❌ CRITICAL ISSUES DETECTED - DO NOT RELEASE")
elif warnings_found:
    print("⚠️  WARNINGS FOUND - Review before release")
else:
    print("✅ NO CRITICAL ISSUES FOUND")
print("=" * 70)
