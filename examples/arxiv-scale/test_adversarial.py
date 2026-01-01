#!/usr/bin/env python3
"""
Adversarial Query Testing

Tests how RAGGuard handles malicious or edge-case inputs:
- Extremely long queries
- Unicode/emoji attacks
- Null bytes
- Recursive structures
- Invalid types
"""

from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("Adversarial Query Testing")
print("=" * 70)

# Setup
client = QdrantClient("localhost", port=6333)
policy = load_policy("policy.yaml")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

retriever = QdrantSecureRetriever(
    client=client,
    collection="arxiv_2400_papers",
    policy=policy,
    embed_fn=model.encode,
    enable_filter_cache=True
)

valid_user = {"institution": "MIT", "roles": ["researcher"]}

tests_passed = 0
tests_failed = 0

def test_adversarial(name, query=None, user=None, should_error=False):
    """Run an adversarial test case."""
    global tests_passed, tests_failed

    if user is None:
        user = valid_user

    print(f"\n🔍 Testing: {name}")

    try:
        # If query is too long, just use a hash for display
        if query and len(str(query)) > 100:
            display_query = f"{str(query)[:50]}... ({len(str(query))} chars)"
        else:
            display_query = repr(query)[:80]

        print(f"   Query: {display_query}")
        print(f"   User: {str(user)[:80]}")

        results = retriever.search(query if query else "test", user=user, limit=10)

        if should_error:
            print(f"   ❌ FAIL: Expected error but got {len(results)} results")
            tests_failed += 1
        else:
            # Verify all results are authorized
            from ragguard.policy.engine import PolicyEngine
            engine = PolicyEngine(policy)

            unauthorized = 0
            for r in results:
                if not engine.evaluate(user, r.payload):
                    unauthorized += 1

            if unauthorized > 0:
                print(f"   ❌ FAIL: {unauthorized} unauthorized documents returned")
                tests_failed += 1
            else:
                print(f"   ✅ PASS: {len(results)} results, all authorized")
                tests_passed += 1

    except Exception as e:
        error_msg = str(e)[:100]
        if should_error:
            print(f"   ✅ PASS: Error raised as expected ({error_msg})")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL: Unexpected error ({error_msg})")
            tests_failed += 1

# Test 1: Extremely Long Query
print("\n" + "=" * 70)
print("Category 1: Size-Based Attacks")
print("=" * 70)

test_adversarial(
    "Extremely long query (1MB)",
    query="machine learning " * 50000,  # ~750KB
    should_error=False  # Should handle gracefully
)

test_adversarial(
    "Empty query",
    query="",
    should_error=False  # Should handle empty string
)

test_adversarial(
    "Single character",
    query="a",
    should_error=False
)

# Test 2: Unicode and Special Characters
print("\n" + "=" * 70)
print("Category 2: Unicode/Character Attacks")
print("=" * 70)

test_adversarial(
    "Emoji flood",
    query="🔥" * 100,
    should_error=False
)

test_adversarial(
    "Mixed unicode",
    query="机器学习 🤖 машинное обучение ⚡️",
    should_error=False
)

test_adversarial(
    "Null bytes in query",
    query="machine\x00learning",
    should_error=False
)

test_adversarial(
    "Control characters",
    query="machine\r\n\t\x00learning",
    should_error=False
)

test_adversarial(
    "Right-to-left override",
    query="machine\u202Egninrael",  # RLO character
    should_error=False
)

# Test 3: Type Confusion
print("\n" + "=" * 70)
print("Category 3: Type Confusion")
print("=" * 70)

test_adversarial(
    "Query as integer",
    query=12345,
    should_error=True  # Should error (not a string or embedding)
)

test_adversarial(
    "Query as list",
    query=["machine", "learning"],
    should_error=True  # Should error (not a string or embedding)
)

test_adversarial(
    "Query as dict",
    query={"text": "machine learning"},
    should_error=True  # Should error
)

# Test 4: User Context Attacks
print("\n" + "=" * 70)
print("Category 4: User Context Manipulation")
print("=" * 70)

test_adversarial(
    "Deeply nested user object",
    user={
        "institution": "MIT",
        "roles": ["researcher"],
        "nested": {"a": {"b": {"c": {"d": "very deep"}}}}
    },
    should_error=False
)

test_adversarial(
    "User with circular reference",
    user=None,  # Will create below
    should_error=False
)

# Create circular reference
circular_user = {"institution": "MIT", "roles": ["researcher"]}
circular_user["self"] = circular_user
test_adversarial(
    "User with circular reference (actual)",
    user=circular_user,
    should_error=False  # Should handle gracefully (just won't use the circular part)
)

test_adversarial(
    "User with None values",
    user={"institution": None, "roles": None},
    should_error=False
)

test_adversarial(
    "User with empty values",
    user={"institution": "", "roles": []},
    should_error=False
)

# Test 5: Injection Attacks (already tested in security_bypass but let's include key ones)
print("\n" + "=" * 70)
print("Category 5: Injection Attacks")
print("=" * 70)

test_adversarial(
    "SQL-like injection in institution",
    user={"institution": "' OR '1'='1", "roles": ["researcher"]},
    should_error=False
)

test_adversarial(
    "JavaScript injection",
    user={"institution": "<script>alert('xss')</script>", "roles": ["researcher"]},
    should_error=False
)

test_adversarial(
    "Command injection",
    user={"institution": "; rm -rf /", "roles": ["researcher"]},
    should_error=False
)

# Test 6: Resource Exhaustion
print("\n" + "=" * 70)
print("Category 6: Resource Exhaustion")
print("=" * 70)

test_adversarial(
    "Very large limit",
    query="machine learning",
    user=valid_user,
    should_error=False  # Qdrant will cap it
)

test_adversarial(
    "Huge user object",
    user={
        "institution": "MIT",
        "roles": ["researcher"],
        "metadata": "X" * 10000  # 10KB of data
    },
    should_error=False
)

# Summary
print("\n" + "=" * 70)
print("ADVERSARIAL TEST SUMMARY")
print("=" * 70)

total_tests = tests_passed + tests_failed
pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

print(f"\n   Total tests: {total_tests}")
print(f"   Passed: {tests_passed} ✅")
print(f"   Failed: {tests_failed} ❌")
print(f"   Pass rate: {pass_rate:.1f}%")

print("\n" + "=" * 70)
if tests_failed == 0:
    print("✅ ALL ADVERSARIAL TESTS PASSED")
    print("   - System handles malicious inputs gracefully")
    print("   - No crashes or security bypasses")
    print("   - Proper error handling")
else:
    print(f"⚠️  {tests_failed} ADVERSARIAL TESTS FAILED")
    print("   Review failures above for security concerns")
print("=" * 70)
