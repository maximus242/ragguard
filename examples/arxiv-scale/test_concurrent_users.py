#!/usr/bin/env python3
"""
Test concurrent user access with different permissions.

Verifies:
- No cache corruption
- No permission leaks between users
- Thread safety
- Performance under concurrent load
"""

import concurrent.futures
import time
import random
from collections import defaultdict
from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("Concurrent User Testing")
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

# Define test users
INSTITUTIONS = ["MIT", "Stanford", "Cornell", "Yale", "Harvard"]
ROLES = [["researcher"], ["student"], ["admin"], ["reviewer"]]

def generate_user(user_id):
    """Generate a random user with varying permissions."""
    return {
        "id": f"user_{user_id}",
        "institution": random.choice(INSTITUTIONS),
        "roles": random.choice(ROLES)
    }

def user_query(user_id):
    """Execute a query for a specific user."""
    user = generate_user(user_id)

    queries = [
        "machine learning",
        "quantum computing",
        "computer vision",
        "natural language processing",
        "reinforcement learning"
    ]

    query = random.choice(queries)

    start = time.time()
    try:
        results = retriever.search(query, user=user, limit=10)
        latency = (time.time() - start) * 1000

        # Verify all results are authorized for this user
        from ragguard.policy.engine import PolicyEngine
        engine = PolicyEngine(policy)

        unauthorized = 0
        for r in results:
            if not engine.evaluate(user, r.payload):
                unauthorized += 1

        return {
            "user_id": user_id,
            "user": user,
            "query": query,
            "latency": latency,
            "results_count": len(results),
            "unauthorized": unauthorized,
            "success": True
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "user": user,
            "query": query,
            "latency": 0,
            "results_count": 0,
            "unauthorized": 0,
            "success": False,
            "error": str(e)
        }

print("\n🔧 Testing with 100 concurrent users...")
print("   Each user has different permissions and queries\n")

# Run concurrent queries
NUM_USERS = 100
start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = list(executor.map(user_query, range(NUM_USERS)))

total_time = time.time() - start_time

print(f"✅ Completed {NUM_USERS} concurrent queries in {total_time:.2f}s")
print(f"   Throughput: {NUM_USERS/total_time:.1f} queries/sec\n")

# Analyze results
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

latencies = [r["latency"] for r in successful]
unauthorized_counts = [r["unauthorized"] for r in successful]

# Group by institution to check for permission leaks
by_institution = defaultdict(list)
for r in successful:
    by_institution[r["user"]["institution"]].append(r)

print("=" * 70)
print("Results Analysis")
print("=" * 70)

print(f"\n📊 Query Success Rate:")
print(f"   Successful: {len(successful)}/{NUM_USERS} ({len(successful)/NUM_USERS*100:.1f}%)")
print(f"   Failed: {len(failed)}/{NUM_USERS}")

if failed:
    print(f"\n❌ Failed queries:")
    for f in failed[:3]:  # Show first 3 failures
        print(f"   - User {f['user_id']}: {f.get('error', 'Unknown error')}")

print(f"\n⚡ Performance:")
print(f"   Mean latency: {sum(latencies)/len(latencies):.2f}ms")
print(f"   Min latency: {min(latencies):.2f}ms")
print(f"   Max latency: {max(latencies):.2f}ms")
print(f"   p50: {sorted(latencies)[len(latencies)//2]:.2f}ms")
print(f"   p95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")

print(f"\n🔒 Security Checks:")
total_unauthorized = sum(unauthorized_counts)
if total_unauthorized == 0:
    print(f"   ✅ No unauthorized documents returned")
else:
    print(f"   ❌ SECURITY BREACH: {total_unauthorized} unauthorized documents!")
    # Show which users got unauthorized docs
    for r in successful:
        if r["unauthorized"] > 0:
            print(f"      - User {r['user_id']} ({r['user']['institution']}): {r['unauthorized']} unauthorized")

print(f"\n🏢 Results by Institution:")
for inst, inst_results in sorted(by_institution.items()):
    avg_results = sum(r["results_count"] for r in inst_results) / len(inst_results)
    avg_latency = sum(r["latency"] for r in inst_results) / len(inst_results)
    print(f"   {inst:12s}: {len(inst_results):3d} users, {avg_results:.1f} avg results, {avg_latency:.1f}ms avg latency")

# Cache stats
cache_stats = retriever.get_cache_stats()
if cache_stats:
    print(f"\n💾 Cache Performance:")
    print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"   Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
    print(f"   Cache size: {cache_stats['size']}/{cache_stats['max_size']}")

print("\n" + "=" * 70)
if total_unauthorized == 0 and len(failed) == 0:
    print("✅ CONCURRENT TEST PASSED")
    print("   - No cache corruption")
    print("   - No permission leaks")
    print("   - All queries successful")
else:
    print("❌ CONCURRENT TEST FAILED")
    if total_unauthorized > 0:
        print(f"   - Permission leaks detected ({total_unauthorized} unauthorized docs)")
    if len(failed) > 0:
        print(f"   - {len(failed)} queries failed")
print("=" * 70)
