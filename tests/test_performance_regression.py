"""
Performance regression tests for RAGGuard.

These tests ensure that optimizations don't regress and maintain
acceptable performance thresholds.

Run with: pytest tests/test_performance_regression.py -v
Skip benchmark tests: pytest tests/test_performance_regression.py -v -m "not benchmark"
Run only benchmarks: pytest tests/test_performance_regression.py -v -m benchmark
"""

import os
import time

import pytest

# Skip all tests if numpy is not installed
np = pytest.importorskip("numpy")

from ragguard.policy import Policy, PolicyEngine

# Skip performance tests in CI or when system is under load
# These tests are inherently flaky due to microsecond-level timing sensitivity
SKIP_PERF_TESTS = os.environ.get("CI") == "true" or os.environ.get("SKIP_PERF_TESTS") == "1"

# Performance thresholds (microseconds)
# These are generous to account for system variability - actual performance is typically 2-3x better
# Note: v0.3.0 added retry logic which adds ~2-3µs overhead for resilience
FILTER_GENERATION_COLD_THRESHOLD = 100.0  # µs (generous, typical is ~25-30µs)
FILTER_GENERATION_WARM_THRESHOLD = 30.0   # µs (generous, typical is ~4-8µs)
POLICY_EVALUATION_THRESHOLD = 20.0        # µs (generous, typical is ~2-4µs)
COMBINED_THRESHOLD = 50.0                 # µs (generous, typical is ~10-15µs)
CACHE_HIT_RATE_THRESHOLD = 0.95           # 95% minimum


def benchmark_function(func, iterations=1000, warmup=100):
    """
    Benchmark a function and return latency statistics.

    Args:
        func: Function to benchmark
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        dict with p50, p95, p99 latencies in microseconds
    """
    # Warm up
    for _ in range(warmup):
        func()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.time()
        func()
        latency = (time.time() - start) * 1000000  # Convert to µs
        latencies.append(latency)

    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
        "mean": np.mean(latencies)
    }


@pytest.fixture
def test_policy():
    """Create a test policy for benchmarking."""
    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-docs",
                "allow": {
                    "roles": ["admin", "manager", "employee"],
                    "conditions": [
                        "user.department == document.department",
                        "user.clearance_level in document.allowed_levels"
                    ]
                }
            },
            {
                "name": "public",
                "match": {"public": True},
                "allow": {"everyone": True}
            }
        ],
        "default": "deny"
    })


@pytest.fixture
def test_users():
    """Create test users."""
    return [
        {
            "id": "user1",
            "roles": ["employee"],
            "department": "engineering",
            "clearance_level": "standard"
        },
        {
            "id": "user2",
            "roles": ["manager"],
            "department": "sales",
            "clearance_level": "elevated"
        },
        {
            "id": "user3",
            "roles": ["admin"],
            "department": "operations",
            "clearance_level": "high"
        }
    ]


@pytest.fixture
def test_documents():
    """Create test documents."""
    return [
        {
            "id": "doc1",
            "department": "engineering",
            "allowed_levels": ["standard", "elevated"],
            "public": False
        },
        {
            "id": "doc2",
            "department": "sales",
            "allowed_levels": ["elevated", "high"],
            "public": False
        },
        {
            "id": "doc3",
            "department": "operations",
            "allowed_levels": ["high"],
            "public": False
        },
        {
            "id": "doc4",
            "public": True
        }
    ]


# ============================================================================
# Filter Generation Performance Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_filter_generation_cold_performance(test_policy, test_users):
    """Test filter generation performance without caching (cold)."""
    engine = PolicyEngine(test_policy, enable_filter_cache=False)
    user = test_users[0]

    stats = benchmark_function(
        lambda: engine.to_filter(user, "qdrant"),
        iterations=1000
    )

    # Verify performance threshold
    assert stats["p50"] < FILTER_GENERATION_COLD_THRESHOLD, (
        f"Cold filter generation too slow: {stats['p50']:.2f}µs "
        f"(threshold: {FILTER_GENERATION_COLD_THRESHOLD}µs)"
    )

    print(f"\n✓ Cold filter generation: {stats['p50']:.2f}µs (p50)")


@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_filter_generation_warm_performance(test_policy, test_users):
    """Test filter generation performance with caching (warm)."""
    engine = PolicyEngine(test_policy, enable_filter_cache=True)
    user = test_users[0]

    stats = benchmark_function(
        lambda: engine.to_filter(user, "qdrant"),
        iterations=1000
    )

    # Verify performance threshold
    assert stats["p50"] < FILTER_GENERATION_WARM_THRESHOLD, (
        f"Warm filter generation too slow: {stats['p50']:.2f}µs "
        f"(threshold: {FILTER_GENERATION_WARM_THRESHOLD}µs)"
    )

    # Verify cache hit rate
    cache_stats = engine.get_cache_stats()
    assert cache_stats["hit_rate"] > CACHE_HIT_RATE_THRESHOLD, (
        f"Cache hit rate too low: {cache_stats['hit_rate']:.1%} "
        f"(threshold: {CACHE_HIT_RATE_THRESHOLD:.1%})"
    )

    print(f"\n✓ Warm filter generation: {stats['p50']:.2f}µs (p50)")
    print(f"✓ Cache hit rate: {cache_stats['hit_rate']:.1%}")


@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_filter_generation_multi_user(test_policy, test_users):
    """Test filter generation with multiple users (realistic workload)."""
    engine = PolicyEngine(test_policy, enable_filter_cache=True)

    iteration = [0]

    def multi_user_query():
        user = test_users[iteration[0] % len(test_users)]
        iteration[0] += 1
        return engine.to_filter(user, "qdrant")

    stats = benchmark_function(multi_user_query, iterations=1000)

    # Should still be fast with multiple users
    assert stats["p50"] < FILTER_GENERATION_WARM_THRESHOLD * 1.5, (
        f"Multi-user filter generation too slow: {stats['p50']:.2f}µs"
    )

    # Cache should work across users
    cache_stats = engine.get_cache_stats()
    assert cache_stats["hit_rate"] > 0.90, (
        f"Multi-user cache hit rate too low: {cache_stats['hit_rate']:.1%}"
    )

    print(f"\n✓ Multi-user filter generation: {stats['p50']:.2f}µs (p50)")


# ============================================================================
# Policy Evaluation Performance Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_policy_evaluation_performance(test_policy, test_users, test_documents):
    """Test policy evaluation performance with compiled conditions."""
    engine = PolicyEngine(test_policy)
    user = test_users[0]
    doc = test_documents[0]

    stats = benchmark_function(
        lambda: engine.evaluate(user, doc),
        iterations=1000
    )

    # Verify performance threshold
    assert stats["p50"] < POLICY_EVALUATION_THRESHOLD, (
        f"Policy evaluation too slow: {stats['p50']:.2f}µs "
        f"(threshold: {POLICY_EVALUATION_THRESHOLD}µs)"
    )

    print(f"\n✓ Policy evaluation: {stats['p50']:.2f}µs (p50)")


@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_policy_evaluation_complex_conditions(test_users, test_documents):
    """Test policy evaluation with complex conditions."""
    complex_policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "complex",
                "allow": {
                    "conditions": [
                        "user.metadata.team.name == document.metadata.team.name",
                        "user.permissions.level in document.required_levels",
                        "user.organization.id == document.organization.id"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(complex_policy)

    user = {
        "metadata": {"team": {"name": "backend"}},
        "permissions": {"level": "senior"},
        "organization": {"id": "org123"}
    }

    doc = {
        "metadata": {"team": {"name": "backend"}},
        "required_levels": ["senior", "lead"],
        "organization": {"id": "org123"}
    }

    stats = benchmark_function(
        lambda: engine.evaluate(user, doc),
        iterations=1000
    )

    # Complex conditions should still be reasonably fast
    assert stats["p50"] < POLICY_EVALUATION_THRESHOLD * 2, (
        f"Complex policy evaluation too slow: {stats['p50']:.2f}µs"
    )

    print(f"\n✓ Complex evaluation: {stats['p50']:.2f}µs (p50)")


# ============================================================================
# Combined Performance Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_combined_overhead_performance(test_policy, test_users, test_documents):
    """Test combined filter generation + evaluation overhead."""
    engine = PolicyEngine(test_policy, enable_filter_cache=True)

    iteration = [0]

    def combined_operation():
        user = test_users[iteration[0] % len(test_users)]
        doc = test_documents[iteration[0] % len(test_documents)]
        iteration[0] += 1

        # Generate filter
        filter_obj = engine.to_filter(user, "qdrant")

        # Evaluate policy
        allowed = engine.evaluate(user, doc)

        return filter_obj, allowed

    stats = benchmark_function(combined_operation, iterations=1000)

    # Combined operations should meet threshold
    assert stats["p50"] < COMBINED_THRESHOLD, (
        f"Combined overhead too high: {stats['p50']:.2f}µs "
        f"(threshold: {COMBINED_THRESHOLD}µs)"
    )

    print(f"\n✓ Combined overhead: {stats['p50']:.2f}µs (p50)")


# ============================================================================
# Cache Performance Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_cache_hit_rate_realistic_workload(test_policy):
    """Test cache hit rate with realistic user patterns."""
    engine = PolicyEngine(test_policy, enable_filter_cache=True, filter_cache_size=1000)

    # Simulate 1000 queries with 50 unique users (realistic ratio)
    users = [
        {"id": f"user{i}", "roles": ["employee"], "department": "eng"}
        for i in range(50)
    ]

    # Make 1000 queries (users repeat)
    for i in range(1000):
        user = users[i % len(users)]
        engine.to_filter(user, "qdrant")

    # Check cache hit rate
    # With 50 unique users and 1000 queries: 50 misses + 950 hits = 95% exactly
    stats = engine.get_cache_stats()
    assert stats["hit_rate"] >= CACHE_HIT_RATE_THRESHOLD, (
        f"Realistic workload cache hit rate too low: {stats['hit_rate']:.1%} "
        f"(threshold: {CACHE_HIT_RATE_THRESHOLD:.1%})"
    )

    print(f"\n✓ Realistic cache hit rate: {stats['hit_rate']:.1%}")
    print(f"  Cache size: {stats['size']}/{stats['max_size']}")


@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_cache_lru_eviction_performance(test_policy):
    """Test that LRU eviction doesn't degrade performance."""
    # Small cache to force evictions
    engine = PolicyEngine(test_policy, enable_filter_cache=True, filter_cache_size=10)

    # Create 50 users (more than cache size)
    users = [
        {"id": f"user{i}", "roles": ["employee"], "department": "eng"}
        for i in range(50)
    ]

    iteration = [0]

    def query_with_eviction():
        user = users[iteration[0] % len(users)]
        iteration[0] += 1
        return engine.to_filter(user, "qdrant")

    stats = benchmark_function(query_with_eviction, iterations=500)

    # Performance should still be reasonable despite evictions
    assert stats["p50"] < FILTER_GENERATION_COLD_THRESHOLD, (
        f"Performance degraded with evictions: {stats['p50']:.2f}µs"
    )

    print(f"\n✓ LRU eviction overhead: {stats['p50']:.2f}µs (p50)")


# ============================================================================
# Compilation Performance Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_condition_compilation_time():
    """Test that condition compilation at initialization is fast."""
    # Create policy with many conditions
    rules = []
    for i in range(100):
        rules.append({
            "name": f"rule{i}",
            "allow": {
                "conditions": [
                    f"user.field{i} == document.field{i}",
                    f"user.other{i} in document.list{i}"
                ]
            }
        })

    policy_dict = {
        "version": "1",
        "rules": rules,
        "default": "deny"
    }

    # Measure compilation time
    start = time.time()
    policy = Policy.from_dict(policy_dict)
    engine = PolicyEngine(policy)
    compile_time = time.time() - start

    # Compilation should be fast (< 100ms for 100 rules)
    assert compile_time < 0.1, (
        f"Condition compilation too slow: {compile_time*1000:.2f}ms"
    )

    print(f"\n✓ Compiled 100 rules in {compile_time*1000:.2f}ms")


# ============================================================================
# Role Set Optimization Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_role_set_lookup_performance():
    """Test that role sets provide O(1) lookup performance."""
    # Create policy with many roles
    many_roles = [f"role_{i}" for i in range(1000)]

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "many-roles", "allow": {"roles": many_roles}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # User with role at end of list (worst case for list iteration)
    user = {"roles": ["role_999"]}
    doc = {"text": "test"}

    stats = benchmark_function(
        lambda: engine.evaluate(user, doc),
        iterations=1000
    )

    # Should still be fast even with 1000 roles (thanks to set lookup)
    assert stats["p50"] < POLICY_EVALUATION_THRESHOLD * 2, (
        f"Role set lookup too slow with 1000 roles: {stats['p50']:.2f}µs"
    )

    print(f"\n✓ 1000-role lookup: {stats['p50']:.2f}µs (p50)")


# ============================================================================
# Summary Test
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Skipping performance tests in CI")
def test_performance_summary(test_policy, test_users, test_documents):
    """Generate overall performance summary."""
    engine = PolicyEngine(test_policy, enable_filter_cache=True)

    # Warm up cache
    for user in test_users:
        engine.to_filter(user, "qdrant")

    # Measure components
    filter_stats = benchmark_function(
        lambda: engine.to_filter(test_users[0], "qdrant"),
        iterations=1000
    )

    eval_stats = benchmark_function(
        lambda: engine.evaluate(test_users[0], test_documents[0]),
        iterations=1000
    )

    cache_stats = engine.get_cache_stats()

    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Filter Generation (p50): {filter_stats['p50']:.2f}µs")
    print(f"Policy Evaluation (p50): {eval_stats['p50']:.2f}µs")
    print(f"Total Overhead (p50):    {filter_stats['p50'] + eval_stats['p50']:.2f}µs")
    print(f"Cache Hit Rate:          {cache_stats['hit_rate']:.1%}")
    print(f"Cache Size:              {cache_stats['size']}/{cache_stats['max_size']}")
    print("="*60)

    # All thresholds should be met
    total_overhead = filter_stats['p50'] + eval_stats['p50']
    assert total_overhead < COMBINED_THRESHOLD, (
        f"Total overhead exceeds threshold: {total_overhead:.2f}µs > {COMBINED_THRESHOLD}µs"
    )


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_performance_regression.py -v -s
    pytest.main([__file__, "-v", "-s"])
