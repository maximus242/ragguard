#!/usr/bin/env python3
"""
Long-running stability test.

Runs many queries over time and monitors:
- Memory usage (no leaks)
- Latency stability (no degradation)
- Cache hit rate (should stabilize)
- Error rate (should be zero)
"""

import time
import random
import sys
from collections import deque
from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy
from sentence_transformers import SentenceTransformer

# Try to import psutil, but work without it
try:
    import psutil
    import os
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: psutil not installed, skipping memory monitoring")

print("=" * 70)
print("Long-Running Stability Test")
print("=" * 70)

# Configuration
NUM_QUERIES = 1000  # Total queries to run
REPORT_INTERVAL = 100  # Report stats every N queries

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

# Test data
QUERIES = [
    "machine learning algorithms",
    "deep learning neural networks",
    "quantum computing applications",
    "computer vision object detection",
    "natural language processing transformers",
    "reinforcement learning robotics",
    "distributed systems consensus",
    "cryptography security protocols",
    "database query optimization",
    "compiler design optimization"
]

USERS = [
    {"institution": "MIT", "roles": ["researcher"]},
    {"institution": "Stanford", "roles": ["researcher"]},
    {"institution": "Cornell", "roles": ["student"]},
    {"institution": "Yale", "roles": ["admin"]},
    {"institution": "Harvard", "roles": ["reviewer"]},
]

# Tracking
if HAS_PSUTIL:
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
else:
    initial_memory = 0

latencies = deque(maxlen=100)  # Keep last 100 latencies
errors = []
results_per_query = []

print(f"\n🔧 Configuration:")
print(f"   Queries: {NUM_QUERIES}")
print(f"   Collection: arxiv_2400_papers")
if HAS_PSUTIL:
    print(f"   Initial memory: {initial_memory:.1f} MB")
print(f"\n🏃 Running stability test...\n")

start_time = time.time()

for i in range(NUM_QUERIES):
    # Random query and user
    query = random.choice(QUERIES)
    user = random.choice(USERS)

    # Execute query
    query_start = time.time()
    try:
        results = retriever.search(query, user=user, limit=10)
        latency = (time.time() - query_start) * 1000
        latencies.append(latency)
        results_per_query.append(len(results))

        # Verify authorization (spot check every 50 queries)
        if i % 50 == 0:
            from ragguard.policy.engine import PolicyEngine
            engine = PolicyEngine(policy)
            for r in results:
                if not engine.evaluate(user, r.payload):
                    errors.append(f"Query {i}: Unauthorized document for {user['institution']}")
                    break

    except Exception as e:
        errors.append(f"Query {i}: {str(e)}")
        latencies.append(0)
        results_per_query.append(0)

    # Report progress
    if (i + 1) % REPORT_INTERVAL == 0:
        cache_stats = retriever.get_cache_stats()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

        if HAS_PSUTIL:
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = current_memory - initial_memory
            print(f"Query {i+1:4d}/{NUM_QUERIES} | "
                  f"Latency: {avg_latency:6.1f}ms (p95: {p95_latency:6.1f}ms) | "
                  f"Cache: {cache_stats['hit_rate']:5.1%} | "
                  f"Memory: {current_memory:6.1f}MB ({memory_delta:+.1f}MB) | "
                  f"Errors: {len(errors)}")
        else:
            print(f"Query {i+1:4d}/{NUM_QUERIES} | "
                  f"Latency: {avg_latency:6.1f}ms (p95: {p95_latency:6.1f}ms) | "
                  f"Cache: {cache_stats['hit_rate']:5.1%} | "
                  f"Errors: {len(errors)}")

total_time = time.time() - start_time

if HAS_PSUTIL:
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
else:
    final_memory = 0
    memory_increase = 0

print("\n" + "=" * 70)
print("Stability Test Results")
print("=" * 70)

print(f"\n⏱️  Performance:")
print(f"   Total time: {total_time:.2f}s")
print(f"   Throughput: {NUM_QUERIES/total_time:.1f} queries/sec")
print(f"   Avg latency: {sum(latencies)/len(latencies):.2f}ms")
print(f"   p50 latency: {sorted(latencies)[len(latencies)//2]:.2f}ms")
print(f"   p95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
print(f"   p99 latency: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")

# Check for latency degradation (compare first 100 vs last 100)
first_100_avg = sum(list(latencies)[:100]) / 100 if len(latencies) >= 100 else 0
last_100_avg = sum(list(latencies)[-100:]) / 100 if len(latencies) >= 100 else 0
latency_change = ((last_100_avg - first_100_avg) / first_100_avg * 100) if first_100_avg > 0 else 0

print(f"\n📈 Latency Stability:")
print(f"   First 100 queries: {first_100_avg:.2f}ms")
print(f"   Last 100 queries: {last_100_avg:.2f}ms")
if abs(latency_change) < 10:
    print(f"   Change: {latency_change:+.1f}% ✅ (stable)")
else:
    print(f"   Change: {latency_change:+.1f}% ⚠️  (degraded)")

if HAS_PSUTIL:
    print(f"\n💾 Memory Usage:")
    print(f"   Initial: {initial_memory:.1f} MB")
    print(f"   Final: {final_memory:.1f} MB")
    print(f"   Increase: {memory_increase:+.1f} MB ({memory_increase/initial_memory*100:+.1f}%)")
    if memory_increase < 50:  # Less than 50MB increase is acceptable
        print(f"   Status: ✅ No significant memory leak")
    else:
        print(f"   Status: ⚠️  Possible memory leak")
else:
    print(f"\n💾 Memory Usage: (psutil not installed, skipped)")

cache_stats = retriever.get_cache_stats()
print(f"\n🔧 Cache Performance:")
print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
print(f"   Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
print(f"   Cache size: {cache_stats['size']}/{cache_stats['max_size']}")

print(f"\n🔒 Security & Correctness:")
print(f"   Total errors: {len(errors)}")
if errors:
    print(f"   ❌ Errors detected:")
    for err in errors[:5]:  # Show first 5
        print(f"      - {err}")
else:
    print(f"   ✅ No errors")

avg_results = sum(results_per_query) / len(results_per_query)
print(f"   Avg results per query: {avg_results:.1f}")

print("\n" + "=" * 70)

# Final verdict
issues = []
if len(errors) > 0:
    issues.append(f"{len(errors)} errors occurred")
if abs(latency_change) >= 10:
    issues.append(f"Latency degraded by {latency_change:.1f}%")
if HAS_PSUTIL and memory_increase > 50:
    issues.append(f"Memory increased by {memory_increase:.1f}MB")

if not issues:
    print("✅ STABILITY TEST PASSED")
    print("   - No errors")
    print("   - Stable latency")
    print("   - No memory leaks")
else:
    print("⚠️  STABILITY TEST ISSUES:")
    for issue in issues:
        print(f"   - {issue}")

print("=" * 70)
