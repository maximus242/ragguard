#!/usr/bin/env python3
"""
Micro-benchmark to measure condition compilation performance improvement.

Compares:
1. Condition evaluation time with compilation
2. Shows the benefit of pre-parsing conditions

This measures the condition evaluation overhead specifically.
"""

import time
from typing import List
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ragguard import load_policy
from ragguard.policy import PolicyEngine


console = Console()


def benchmark_policy_evaluation(policy, users, documents, iterations=1000):
    """Benchmark policy evaluation with compiled conditions."""
    engine = PolicyEngine(policy, enable_filter_cache=False)  # Disable caching to measure pure evaluation

    latencies = []
    for _ in track(range(iterations), description="[green]With Compilation"):
        user = users[_ % len(users)]
        doc = documents[_ % len(documents)]

        start = time.time()
        result = engine.evaluate(user, doc)
        latency = (time.time() - start) * 1000 * 1000  # microseconds
        latencies.append(latency)

    return latencies


def benchmark_cached_evaluation(policy, users, documents, iterations=1000):
    """Benchmark with both caching and compilation."""
    engine = PolicyEngine(policy, enable_filter_cache=True)

    latencies = []
    for _ in track(range(iterations), description="[cyan]With Caching + Compilation"):
        user = users[_ % len(users)]
        doc = documents[_ % len(documents)]

        start = time.time()
        result = engine.evaluate(user, doc)
        latency = (time.time() - start) * 1000 * 1000  # microseconds
        latencies.append(latency)

    stats = engine.get_cache_stats()
    return latencies, stats


def print_results(results):
    """Print benchmark results."""
    console.print("\n")
    console.print("=" * 80, style="bold")
    console.print("📊 Condition Compilation Benchmark Results", style="bold cyan", justify="center")
    console.print("=" * 80, style="bold")
    console.print()

    # Latency comparison table
    table = Table(title="⚡ Policy Evaluation Performance (microseconds)", show_header=True)
    table.add_column("Scenario", style="cyan", width=35)
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")

    for name, data in results.items():
        table.add_row(
            data["label"],
            f"{data['p50']:.1f}µs",
            f"{data['p95']:.1f}µs",
            f"{data['p99']:.1f}µs",
            style=data.get("style", "white")
        )

    console.print(table)
    console.print()

    # Cache statistics (if available)
    has_cache_stats = any("cache_stats" in data and data["cache_stats"] for data in results.values())
    if has_cache_stats:
        table = Table(title="📊 Cache Performance", show_header=True)
        table.add_column("Scenario", style="cyan", width=35)
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

    eval_p50 = results["evaluation"]["p50"]

    console.print(f"  • Policy evaluation (with compilation): {eval_p50:.1f}µs")

    if "cached" in results:
        cached_p50 = results["cached"]["p50"]
        improvement = ((eval_p50 - cached_p50) / eval_p50) * 100
        console.print(f"  • With caching + compilation: {cached_p50:.1f}µs")
        console.print(f"  • Combined optimization: [green]{improvement:.1f}% faster[/green]")

    console.print(f"\n  • Compiled conditions avoid runtime string parsing overhead")
    console.print(f"  • Pre-split field paths eliminate repeated string operations")
    console.print(f"  • Operator type is determined once at initialization")

    console.print()


def main():
    console.print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  RAGGuard Condition Compilation Micro-Benchmark              ║
║                                                              ║
║  Measuring policy evaluation performance                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Load policy
    console.print("📋 Loading access control policy...")
    policy = load_policy("policy.yaml")

    # Define test users and documents
    users = [
        {"institution": "MIT", "roles": ["researcher"]},
        {"institution": "Stanford", "roles": ["researcher"]},
        {"institution": "Harvard", "roles": ["student"]},
        {"institution": "MIT", "roles": ["admin"]},
    ]

    documents = [
        {"institution": "MIT", "access_level": "public", "department": "CS"},
        {"institution": "MIT", "access_level": "internal", "department": "CS"},
        {"institution": "Stanford", "access_level": "public", "department": "EE"},
        {"institution": "Harvard", "access_level": "internal", "department": "Math"},
    ]

    iterations = 10000

    console.print(f"\n🏃 Running benchmarks ({iterations} iterations)...\n")

    results = {}

    # 1. With compilation (no caching)
    console.print("1️⃣  Testing policy evaluation (compiled conditions, no cache)...")
    lat_eval = benchmark_policy_evaluation(policy, users, documents, iterations)
    results["evaluation"] = {
        "label": "Policy Evaluation (Compiled)",
        "latencies": lat_eval,
        "p50": np.percentile(lat_eval, 50),
        "p95": np.percentile(lat_eval, 95),
        "p99": np.percentile(lat_eval, 99),
        "style": "green"
    }

    # 2. With compilation + caching
    console.print("\n2️⃣  Testing with both caching and compilation...")
    lat_cached, stats_cached = benchmark_cached_evaluation(policy, users, documents, iterations)
    results["cached"] = {
        "label": "With Caching + Compilation",
        "latencies": lat_cached,
        "p50": np.percentile(lat_cached, 50),
        "p95": np.percentile(lat_cached, 95),
        "p99": np.percentile(lat_cached, 99),
        "cache_stats": stats_cached,
        "style": "cyan"
    }

    # Print results
    print_results(results)

    console.print("✨ Benchmark complete!", style="bold green")
    console.print()


if __name__ == "__main__":
    main()
