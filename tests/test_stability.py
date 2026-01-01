"""
Stability tests for RAGGuard performance optimizations.

Tests for memory leaks, performance degradation, and long-running stability.

Run quick test with: pytest tests/test_stability.py -v
Run full test with: pytest tests/test_stability.py -v --stability-full
"""

import gc
import sys
import time

import pytest

# Skip all tests if qdrant-client is not installed (required for filter generation tests)
pytest.importorskip("qdrant_client")

from ragguard.policy import Policy, PolicyEngine


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except:
        # Fallback for systems without resource module
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0


def test_no_memory_leak_filter_generation():
    """Test that filter generation doesn't leak memory over time."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "test",
                "allow": {
                    "roles": ["admin"],
                    "conditions": ["user.dept == document.dept"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=100)

    # Create rotating users (more than cache size to test evictions)
    users = [
        {"id": f"user{i}", "dept": "eng", "roles": ["admin"]}
        for i in range(200)
    ]

    # Measure initial memory
    gc.collect()
    mem_start = get_memory_usage_mb()

    # Run many iterations
    iterations = 10000
    for i in range(iterations):
        user = users[i % len(users)]
        engine.to_filter(user, "qdrant")

        # Force garbage collection periodically
        if i % 1000 == 0:
            gc.collect()

    # Measure final memory
    gc.collect()
    mem_end = get_memory_usage_mb()

    mem_growth = mem_end - mem_start

    # Memory growth should be bounded (< 100MB for 10k iterations)
    # Some growth is expected from filter caching and Python's memory allocator
    if mem_start > 0:  # Only check if we can measure memory
        assert mem_growth < 100, (
            f"Memory leak detected: grew {mem_growth:.2f}MB over {iterations} iterations"
        )

        print(f"\n✓ Memory bounded: {mem_start:.2f}MB → {mem_end:.2f}MB (+{mem_growth:.2f}MB)")
    else:
        print("\n⚠ Could not measure memory (psutil/resource not available)")

    # Cache size should be bounded
    stats = engine.get_cache_stats()
    assert stats["size"] <= stats["max_size"], (
        f"Cache grew beyond max size: {stats['size']} > {stats['max_size']}"
    )

    print(f"✓ Cache bounded: {stats['size']}/{stats['max_size']}")


def test_no_memory_leak_policy_evaluation():
    """Test that policy evaluation doesn't leak memory."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "complex",
                "allow": {
                    "conditions": [
                        "user.metadata.team == document.metadata.team",
                        "user.level in document.allowed_levels"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    users = [
        {"metadata": {"team": f"team{i}"}, "level": "senior"}
        for i in range(50)
    ]

    documents = [
        {"metadata": {"team": f"team{i}"}, "allowed_levels": ["senior", "lead"]}
        for i in range(50)
    ]

    # Measure initial memory
    gc.collect()
    mem_start = get_memory_usage_mb()

    # Run many evaluations
    iterations = 10000
    for i in range(iterations):
        user = users[i % len(users)]
        doc = documents[i % len(documents)]
        engine.evaluate(user, doc)

        if i % 1000 == 0:
            gc.collect()

    # Measure final memory
    gc.collect()
    mem_end = get_memory_usage_mb()

    mem_growth = mem_end - mem_start

    if mem_start > 0:
        assert mem_growth < 5, (
            f"Memory leak in evaluation: grew {mem_growth:.2f}MB"
        )
        print(f"\n✓ Evaluation memory stable: +{mem_growth:.2f}MB over {iterations} iterations")


def test_performance_stability_over_time():
    """Test that performance doesn't degrade over time."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "test", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)
    user = {"roles": ["admin"]}

    # Measure performance at different points
    measurements = []

    for batch in range(5):  # 5 batches of 1000 iterations
        batch_start = time.time()

        for _ in range(1000):
            engine.to_filter(user, "qdrant")

        batch_time = (time.time() - batch_start) * 1000  # ms

        measurements.append(batch_time)

        # Small delay between batches
        time.sleep(0.01)

    # Performance should be stable (variance < 50%)
    avg_time = sum(measurements) / len(measurements)
    max_time = max(measurements)
    min_time = min(measurements)

    variance = (max_time - min_time) / avg_time

    assert variance < 0.5, (
        f"Performance unstable: variance {variance:.1%} (max: {max_time:.2f}ms, min: {min_time:.2f}ms)"
    )

    print(f"\n✓ Performance stable: {min_time:.2f}ms - {max_time:.2f}ms (variance: {variance:.1%})")


def test_cache_stability_under_load():
    """Test cache stability under concurrent-like load."""
    import random

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "test", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=100)

    # Simulate high load with "hot" users (frequently accessed) and "cold" users
    # This is a more realistic pattern than pure round-robin
    hot_users = [
        {"id": f"hot_user{i}", "roles": ["admin"]}
        for i in range(50)  # 50 hot users (fit in cache)
    ]
    cold_users = [
        {"id": f"cold_user{i}", "roles": ["admin"]}
        for i in range(200)  # 200 cold users (occasional access)
    ]

    # Run sustained load with 80% hot users, 20% cold users
    # This reflects realistic access patterns (Pareto principle)
    random.seed(42)  # Reproducible
    iterations = 5000
    for _ in range(iterations):
        if random.random() < 0.8:
            user = random.choice(hot_users)
        else:
            user = random.choice(cold_users)
        engine.to_filter(user, "qdrant")

    # Cache should still be functioning
    stats = engine.get_cache_stats()

    assert stats["size"] <= stats["max_size"], "Cache exceeded max size"
    assert stats["hits"] > 0, "No cache hits under load"
    assert stats["misses"] > 0, "No cache misses (unrealistic)"

    # With 80% hot user access and 50 hot users, we expect good hit rate
    # after initial warm-up (50 misses), subsequent hot accesses should mostly hit
    hit_rate = stats["hit_rate"]
    assert hit_rate > 0.5, f"Cache hit rate too low under load: {hit_rate:.1%}"

    print(f"\n✓ Cache stable under load: {hit_rate:.1%} hit rate")
    print(f"  {stats['hits']} hits, {stats['misses']} misses")


@pytest.mark.slow
def test_extended_stability(request):
    """
    Extended stability test (optional, run with --stability-full).

    This test runs for ~1 minute and exercises all components.
    """
    # Check if full stability test is requested
    if not request.config.getoption("--stability-full", default=False):
        pytest.skip("Skipping extended test (use --stability-full to run)")

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-docs",
                "allow": {
                    "roles": ["employee", "manager"],
                    "conditions": ["user.dept == document.dept"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=1000)

    users = [
        {"id": f"user{i}", "dept": f"dept{i%10}", "roles": ["employee"]}
        for i in range(100)
    ]

    documents = [
        {"id": f"doc{i}", "dept": f"dept{i%10}"}
        for i in range(100)
    ]

    # Track metrics
    start_time = time.time()
    operation_count = 0
    error_count = 0

    # Run for ~60 seconds
    while time.time() - start_time < 60:
        try:
            user = users[operation_count % len(users)]
            doc = documents[operation_count % len(documents)]

            # Alternate between filter generation and evaluation
            if operation_count % 2 == 0:
                engine.to_filter(user, "qdrant")
            else:
                engine.evaluate(user, doc)

            operation_count += 1

            # Periodic garbage collection
            if operation_count % 10000 == 0:
                gc.collect()

        except Exception:
            error_count += 1
            if error_count > 10:
                raise

    duration = time.time() - start_time

    # No errors should occur
    assert error_count == 0, f"{error_count} errors during extended test"

    # Cache should still be healthy
    stats = engine.get_cache_stats()
    assert stats["size"] <= stats["max_size"]
    assert stats["hit_rate"] > 0.8

    print("\n✓ Extended stability test passed")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Operations: {operation_count:,}")
    print(f"  Ops/sec: {operation_count/duration:.0f}")
    print(f"  Errors: {error_count}")
    print(f"  Cache hit rate: {stats['hit_rate']:.1%}")


if __name__ == "__main__":
    # Quick run
    pytest.main([__file__, "-v"])

    # Full run with: pytest tests/test_stability.py -v --stability-full
