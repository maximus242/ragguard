"""
Example: Using RAGGuard Enterprise Metrics

Demonstrates how to:
1. Enable/disable metrics collection
2. Export metrics in Prometheus format
3. Get metrics summary
4. Monitor cache performance

NOTE: Metrics features require ragguard-enterprise:
    pip install ragguard-enterprise
"""

import time


def main():
    print("=" * 60)
    print("RAGGuard Metrics Example")
    print("=" * 60)
    print()

    try:
        from ragguard_enterprise import (
            get_metrics_summary,
            export_metrics_prometheus,
            export_metrics_json,
            is_metrics_enabled,
        )
        from ragguard_enterprise.metrics import print_metrics_summary, get_metrics_collector
    except ImportError:
        print("This example requires ragguard-enterprise:")
        print("  pip install ragguard-enterprise")
        print()
        print("Enterprise metrics features include:")
        print("  - Query latency tracking (avg, p50, p95, p99)")
        print("  - Cache hit/miss rates")
        print("  - Per-backend statistics")
        print("  - Prometheus export format")
        print("  - JSON export format")
        print("  - Real-time metrics dashboard integration")
        return

    # Check if metrics are enabled (enabled by default)
    print(f"Metrics enabled: {is_metrics_enabled()}")
    print()

    # Simulate some queries
    print("Simulating 10 queries...")
    collector = get_metrics_collector()

    for i in range(10):
        # Simulate query with varying latency
        latency = 0.05 + (i * 0.01)  # 50ms to 140ms
        result_count = 5 + (i % 5)   # 5 to 9 results
        cache_hit = (i % 3 == 0)     # Every 3rd query is cache hit

        collector.record_query(
            duration=latency,
            backend="qdrant",
            result_count=result_count,
            cache_hit=cache_hit,
            user_id=f"user{i % 3}"  # 3 unique users
        )

        # Simulate filter build time
        collector.record_filter_build(0.005 + (i * 0.001))

        time.sleep(0.01)  # Small delay

    print("Done!")
    print()

    # Get human-readable summary
    print("-" * 60)
    print("Metrics Summary (Formatted)")
    print("-" * 60)
    print_metrics_summary()
    print()

    # Get programmatic summary
    print("-" * 60)
    print("Metrics Summary (Dict)")
    print("-" * 60)
    summary = get_metrics_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Export to Prometheus format
    print("-" * 60)
    print("Prometheus Format (first 20 lines)")
    print("-" * 60)
    prometheus_text = export_metrics_prometheus()
    lines = prometheus_text.split('\n')[:20]
    for line in lines:
        print(line)
    print(f"... ({len(prometheus_text.split(chr(10)))} total lines)")
    print()

    # Export to JSON
    print("-" * 60)
    print("JSON Format (summary only)")
    print("-" * 60)
    import json
    metrics_json = export_metrics_json()
    metrics = json.loads(metrics_json)
    print(json.dumps(metrics['summary'], indent=2))
    print()

    # Per-backend stats
    print("-" * 60)
    print("Per-Backend Stats")
    print("-" * 60)
    backend_stats = collector.get_backend_summary()
    for backend, stats in backend_stats.items():
        print(f"{backend}:")
        print(f"  Queries: {stats['total_queries']}")
        print(f"  Avg Latency: {stats['avg_latency_seconds'] * 1000:.2f}ms")
        print(f"  P95 Latency: {stats['p95_latency_seconds'] * 1000:.2f}ms")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
