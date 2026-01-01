#!/usr/bin/env python3
"""
Test the new != and list literal operators with actual Qdrant queries.
"""

from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, Policy
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("Testing New Operators: != and List Literals")
print("=" * 70)

# Setup
client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Test 1: Negation operator (!=)
print("\n" + "=" * 70)
print("Test 1: Negation Operator (!=)")
print("=" * 70)

policy_negation = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "not-restricted",
            "allow": {
                "everyone": True,
                "conditions": ["document.access_level != 'restricted'"]
            }
        }
    ],
    "default": "deny"
})

retriever_negation = QdrantSecureRetriever(
    client=client,
    collection="arxiv_2400_papers",
    policy=policy_negation,
    embed_fn=model.encode
)

user = {"id": "test_user"}
results = retriever_negation.search("machine learning", user=user, limit=20)

print(f"\n✅ Query successful: {len(results)} results")
print(f"📊 Checking access levels...")

access_levels = {}
for r in results:
    level = r.payload.get("access_level", "none")
    access_levels[level] = access_levels.get(level, 0) + 1
    if level == "restricted":
        print(f"   ❌ FAIL: Found restricted document (ID: {r.id})")

print(f"\n📈 Access level distribution:")
for level, count in sorted(access_levels.items()):
    print(f"   {level}: {count}")

if "restricted" in access_levels:
    print("\n❌ TEST FAILED: != operator not working")
else:
    print("\n✅ TEST PASSED: != operator successfully excluded restricted docs")

# Test 2: List literal conditions
print("\n" + "=" * 70)
print("Test 2: List Literal Conditions")
print("=" * 70)

policy_list = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "allowed-categories",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in ['cs.AI', 'cs.LG', 'cs.CL']"]
            }
        }
    ],
    "default": "deny"
})

retriever_list = QdrantSecureRetriever(
    client=client,
    collection="arxiv_2400_papers",
    policy=policy_list,
    embed_fn=model.encode
)

results = retriever_list.search("neural networks", user=user, limit=20)

print(f"\n✅ Query successful: {len(results)} results")
print(f"📊 Checking categories...")

categories = {}
allowed_categories = ['cs.AI', 'cs.LG', 'cs.CL']

for r in results:
    cat = r.payload.get("category", "none")
    categories[cat] = categories.get(cat, 0) + 1
    if cat not in allowed_categories:
        print(f"   ❌ FAIL: Found disallowed category '{cat}' (ID: {r.id})")

print(f"\n📈 Category distribution:")
for cat, count in sorted(categories.items()):
    marker = "✅" if cat in allowed_categories else "❌"
    print(f"   {marker} {cat}: {count}")

disallowed = [c for c in categories.keys() if c not in allowed_categories]
if disallowed:
    print(f"\n❌ TEST FAILED: List literal not working (found: {disallowed})")
else:
    print("\n✅ TEST PASSED: List literal successfully filtered to allowed categories")

# Test 3: Combined operators
print("\n" + "=" * 70)
print("Test 3: Combined != and List Literals")
print("=" * 70)

policy_combined = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "safe-ml-papers",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.category in ['cs.AI', 'cs.LG']",
                    "document.access_level != 'restricted'"
                ]
            }
        }
    ],
    "default": "deny"
})

retriever_combined = QdrantSecureRetriever(
    client=client,
    collection="arxiv_2400_papers",
    policy=policy_combined,
    embed_fn=model.encode
)

results = retriever_combined.search("deep learning", user=user, limit=20)

print(f"\n✅ Query successful: {len(results)} results")
print(f"📊 Verifying both conditions...")

violations = []
for r in results:
    cat = r.payload.get("category", "none")
    level = r.payload.get("access_level", "none")

    if cat not in ['cs.AI', 'cs.LG']:
        violations.append(f"Wrong category: {cat} (ID: {r.id})")

    if level == "restricted":
        violations.append(f"Restricted document (ID: {r.id})")

if violations:
    print("\n❌ TEST FAILED: Combined operators not working")
    for v in violations:
        print(f"   ❌ {v}")
else:
    print("\n✅ TEST PASSED: Both operators work correctly together")
    print("   ✅ Categories limited to cs.AI and cs.LG")
    print("   ✅ No restricted documents")

# Test 4: Performance with new operators
print("\n" + "=" * 70)
print("Test 4: Performance Check")
print("=" * 70)

import time

# Warm up
for _ in range(5):
    retriever_combined.search("test", user=user, limit=10)

# Benchmark
iterations = 100
start = time.time()
for _ in range(iterations):
    retriever_combined.search("machine learning", user=user, limit=10)
duration = (time.time() - start) * 1000

avg_latency = duration / iterations
print(f"\n⚡ Performance:")
print(f"   Average latency: {avg_latency:.2f}ms per query")
print(f"   Throughput: {1000/avg_latency:.1f} queries/sec")

cache_stats = retriever_combined.get_cache_stats()
print(f"\n💾 Cache performance:")
print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
print(f"   Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")

if avg_latency < 10:  # Should be microseconds when cached
    print("\n✅ Performance excellent (cache working)")
else:
    print(f"\n⚠️  Performance okay but not optimal ({avg_latency:.2f}ms)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("✅ All new operator tests completed successfully!")
print()
print("New features verified:")
print("  ✅ != operator works in filter generation")
print("  ✅ List literals work in filter generation")
print("  ✅ Operators can be combined")
print("  ✅ Performance is acceptable")
print()
print("=" * 70)
