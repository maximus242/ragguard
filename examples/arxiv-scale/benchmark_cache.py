#!/usr/bin/env python3
"""
Micro-benchmark to measure filter caching performance improvement.

Compares:
1. RAGGuard without filter caching (baseline)
2. RAGGuard with filter caching (optimized)

This isolates the filter generation overhead from database query time.
"""

import time
import statistics
from typing import List
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ragguard import load_policy
from ragguard.policy import PolicyEngine


console = Console()


def benchmark_filter_generation_uncached(policy, user, backend, iterations=1000):
    """Benchmark filter generation without caching."""
    engine = PolicyEngine(policy, enable_filter_cache=False)

    latencies = []
    for _ in track(range(iterations), description="[red]Without Cache"):
        start = time.time()
        filter_obj = engine.to_filter(user, backend)
        latency = (time.time() - start) * 1000 * 1000  # microseconds
        latencies.append(latency)

    return latencies


def benchmark_filter_generation_cached(policy, user, backend, iterations=1000):
    """Benchmark filter generation with caching."""
    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=1000)

    latencies = []
    for _ in track(range(iterations), description="[green]With Cache"):
        start = time.time()
        filter_obj = engine.to_filter(user, backend)
        latency = (time.time() - start) * 1000 * 1000  # microseconds
        latencies.append(latency)

    # Get cache stats
    stats = engine.get_cache_stats()

    return latencies, stats


def benchmark_filter_generation_multiple_users(policy, users, backend, iterations=100):
    """Benchmark with multiple users (realistic workload)."""
    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=1000)

    latencies = []
    for i in track(range(iterations), description="[cyan]Multi-User"):
        user = users[i % len(users)]
        start = time.time()
        filter_obj = engine.to_filter(user, backend)
        latency = (time.time() - start) * 1000 * 1000  # microseconds
        latencies.append(latency)

    stats = engine.get_cache_stats()
    return latencies, stats


def print_results(results):
    """Print benchmark results."""
    console.print("\n")
    console.print("=" * 80, style="bold")
    console.print("📊 Filter Caching Benchmark Results", style="bold cyan", justify="center")
    console.print("=" * 80, style="bold")
    console.print()

    # Latency comparison table
    table = Table(title="⚡ Filter Generation Performance (microseconds)", show_header=True)
    table.add_column("Scenario", style="cyan", width=30)
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")
    table.add_column("Speedup", justify="right")

    baseline_p50 = results["uncached"]["p50"]

    for name, data in results.items():
        speedup = ((baseline_p50 - data["p50"]) / baseline_p50) * 100
        speedup_str = f"{speedup:.1f}%" if speedup > 0 else f"{speedup:.1f}%"

        style = "green" if "cached" in name else "red"

        table.add_row(
            data["label"],
            f"{data['p50']:.1f}µs",
            f"{data['p95']:.1f}µs",
            f"{data['p99']:.1f}µs",
            speedup_str,
            style=style
        )

    console.print(table)
    console.print()

    # Cache statistics table
    table = Table(title="📊 Cache Performance", show_header=True)
    table.add_column("Scenario", style="cyan", width=30)
    table.add_column("Hit Rate", justify="right")
    table.add_column("Cache Size", justify="right")

    for name, data in results.items():
        if "cache_stats" in data and data["cache_stats"] is not None:
            stats = data["cache_stats"]
            hit_rate = f"{stats['hit_rate'] * 100:.1f}%"
            cache_size = f"{stats['size']}/{stats['max_size']}"

            table.add_row(
                data["label"],
                hit_rate,
                cache_size,
                style="green"
            )

    console.print(table)
    console.print()

    # Summary
    console.print("💡 Key Findings:", style="bold yellow")
    console.print()

    cached_p50 = results["cached"]["p50"]
    improvement = ((baseline_p50 - cached_p50) / baseline_p50) * 100

    console.print(f"  • Filter caching provides [green]{improvement:.1f}% reduction[/green] in filter generation time")
    console.print(f"  • Cold cache: {baseline_p50:.1f}µs → Warm cache: {cached_p50:.1f}µs")

    if "multi_user" in results:
        multi_hit_rate = results["multi_user"]["cache_stats"]["hit_rate"]
        console.print(f"  • Multi-user cache hit rate: [green]{multi_hit_rate * 100:.1f}%[/green]")

    # Check if we met the 30% improvement target
    if improvement >= 30:
        console.print(f"\n✅ [bold green]SUCCESS: Achieved {improvement:.1f}% improvement (target: 30%)[/bold green]")
    else:
        console.print(f"\n⚠️  [bold yellow]Target not met: {improvement:.1f}% improvement (target: 30%)[/bold yellow]")

    console.print()


def main():
    console.print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  RAGGuard Filter Caching Micro-Benchmark                     ║
║                                                              ║
║  Measuring filter generation performance improvement         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Load policy
    console.print("📋 Loading access control policy...")
    policy = load_policy("policy.yaml")

    # Define test users
    mit_user = {"institution": "MIT", "roles": ["researcher"]}
    stanford_user = {"institution": "Stanford", "roles": ["researcher"]}
    harvard_user = {"institution": "Harvard", "roles": ["student"]}
    users = [mit_user, stanford_user, harvard_user]

    backend = "qdrant"
    iterations = 1000

    console.print(f"\n🏃 Running benchmarks ({iterations} iterations)...\n")

    results = {}

    # 1. Without caching (baseline)
    console.print("1️⃣  Testing without cache...")
    lat_uncached = benchmark_filter_generation_uncached(policy, mit_user, backend, iterations)
    results["uncached"] = {
        "label": "Without Cache (Baseline)",
        "latencies": lat_uncached,
        "p50": np.percentile(lat_uncached, 50),
        "p95": np.percentile(lat_uncached, 95),
        "p99": np.percentile(lat_uncached, 99),
        "cache_stats": None
    }

    # 2. With caching (same user - maximum cache hit rate)
    console.print("\n2️⃣  Testing with cache (same user)...")
    lat_cached, stats_cached = benchmark_filter_generation_cached(policy, mit_user, backend, iterations)
    results["cached"] = {
        "label": "With Cache (Same User)",
        "latencies": lat_cached,
        "p50": np.percentile(lat_cached, 50),
        "p95": np.percentile(lat_cached, 95),
        "p99": np.percentile(lat_cached, 99),
        "cache_stats": stats_cached
    }

    # 3. With caching (multiple users - realistic workload)
    console.print("\n3️⃣  Testing with cache (multi-user workload)...")
    lat_multi, stats_multi = benchmark_filter_generation_multiple_users(policy, users, backend, iterations)
    results["multi_user"] = {
        "label": "With Cache (Multi-User)",
        "latencies": lat_multi,
        "p50": np.percentile(lat_multi, 50),
        "p95": np.percentile(lat_multi, 95),
        "p99": np.percentile(lat_multi, 99),
        "cache_stats": stats_multi
    }

    # Print results
    print_results(results)

    console.print("✨ Benchmark complete!", style="bold green")
    console.print()


if __name__ == "__main__":
    main()
