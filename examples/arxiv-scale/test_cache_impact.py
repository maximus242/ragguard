#!/usr/bin/env python3
"""
Quick test to compare RAGGuard performance with and without caching.
"""

import time
import numpy as np
from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("Testing Cache Impact on Real Query Performance")
print("=" * 70)

# Setup
client = QdrantClient("localhost", port=6333)
policy = load_policy("policy.yaml")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
collection = "arxiv_2400_papers"

# Test queries (same user, should benefit from cache)
queries = [
    "machine learning algorithms",
    "quantum computing applications",
    "computer vision deep learning",
    "natural language processing",
    "reinforcement learning"
] * 4  # 20 queries total

user = {"institution": "MIT", "roles": ["researcher"]}

print(f"\nCollection: {collection}")
print(f"Queries: {len(queries)}")
print(f"User: {user}")

def run_benchmark(enable_cache, label):
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    
    retriever = QdrantSecureRetriever(
        client=client,
        collection=collection,
        policy=policy,
        embed_fn=model.encode,
        enable_filter_cache=enable_cache
    )
    
    latencies = []
    
    for i, query in enumerate(queries):
        start = time.time()
        results = retriever.search(query=query, user=user, limit=10)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)
        
        if i < 3 or i >= len(queries) - 2:  # First 3 and last 2
            print(f"  Query {i+1:2d}: {latency:6.2f}ms")
        elif i == 3:
            print(f"  ...")
    
    # Statistics
    stats = retriever.get_cache_stats()
    
    print(f"\n📊 Results:")
    print(f"  p50: {np.percentile(latencies, 50):.2f}ms")
    print(f"  p95: {np.percentile(latencies, 95):.2f}ms")
    print(f"  p99: {np.percentile(latencies, 99):.2f}ms")
    print(f"  Mean: {np.mean(latencies):.2f}ms")
    
    if stats:
        print(f"\n🔧 Cache Stats:")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    else:
        print(f"\n🔧 Cache: Disabled")
    
    return latencies

# Run benchmarks
latencies_no_cache = run_benchmark(enable_cache=False, label="WITHOUT CACHE")
latencies_with_cache = run_benchmark(enable_cache=True, label="WITH CACHE")

# Comparison
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

p50_no_cache = np.percentile(latencies_no_cache, 50)
p50_with_cache = np.percentile(latencies_with_cache, 50)
improvement = ((p50_no_cache - p50_with_cache) / p50_no_cache) * 100

print(f"\nMedian (p50) Latency:")
print(f"  Without cache: {p50_no_cache:.2f}ms")
print(f"  With cache:    {p50_with_cache:.2f}ms")
print(f"  Difference:    {p50_no_cache - p50_with_cache:.2f}ms ({improvement:+.2f}%)")

if abs(improvement) < 1:
    print(f"\n💡 Cache impact is negligible (<1% difference)")
elif improvement > 0:
    print(f"\n✅ Cache provides {improvement:.1f}% improvement")
else:
    print(f"\n⚠️  Cache actually slows things down by {abs(improvement):.1f}%")

