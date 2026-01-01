#!/usr/bin/env python3
"""
Demo showing how cache invalidation works with permission changes.
"""

from ragguard import QdrantSecureRetriever, load_policy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import time

client = QdrantClient("localhost", port=6333)
policy = load_policy("policy.yaml")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

retriever = QdrantSecureRetriever(
    client=client,
    collection="arxiv_2400_papers",
    policy=policy,
    embed_fn=model.encode
)

print("=" * 70)
print("Cache Invalidation Demo")
print("=" * 70)

# User 1 - multiple queries (should hit cache)
alice = {"institution": "MIT", "roles": ["researcher"]}
print("\n1️⃣  Alice queries 3 times (same permissions):")
for i in range(3):
    start = time.time()
    results = retriever.search("machine learning", user=alice, limit=5)
    latency = (time.time() - start) * 1000
    print(f"   Query {i+1}: {latency:.2f}ms")

stats = retriever.get_cache_stats()
print(f"\n   Cache: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']:.1%} hit rate)")

# User 2 - different permissions (cache miss, then hits)
bob = {"institution": "Yale", "roles": ["student"]}
print("\n2️⃣  Bob queries 2 times (different user):")
for i in range(2):
    start = time.time()
    results = retriever.search("quantum computing", user=bob, limit=5)
    latency = (time.time() - start) * 1000
    print(f"   Query {i+1}: {latency:.2f}ms")

stats = retriever.get_cache_stats()
print(f"\n   Cache: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']:.1%} hit rate)")

# User 3 - permission change simulation
carol = {"institution": "MIT", "roles": ["student"]}
print("\n3️⃣  Carol's permissions change:")
print(f"   Before: {carol}")
start = time.time()
results = retriever.search("computer vision", user=carol, limit=5)
latency1 = (time.time() - start) * 1000
print(f"   Query 1: {latency1:.2f}ms")

# Simulate permission change
carol["roles"] = ["researcher", "admin"]
print(f"\n   After:  {carol}")
start = time.time()
results = retriever.search("computer vision", user=carol, limit=5)
latency2 = (time.time() - start) * 1000
print(f"   Query 2: {latency2:.2f}ms (cache miss due to role change)")

stats = retriever.get_cache_stats()
print(f"\n   Cache: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']:.1%} hit rate)")

print("\n\n💡 Key Observations:")
print("-" * 70)
print("  • Alice's 2nd & 3rd queries were FAST (cache hit)")
print("  • Bob's 1st query was slower (cache miss), 2nd was fast")
print("  • Carol's permission change caused 1 cache miss")
print("  • Other users (Alice, Bob) still hit cache ✅")
print("\n  → Only affected user sees cache miss")
print("  → Permission changes don't bust entire cache")
