#!/usr/bin/env python3
"""
Combined benchmark measuring total RAGGuard optimization improvements.

Measures:
1. Filter generation time (caching + compilation impact)
2. Policy evaluation time (compilation impact)
3. Combined end-to-end overhead

Shows the cumulative benefit of Phase 1 + Phase 2 optimizations.
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


def benchmark_combined(policy, users, backend, iterations=1000):
    """Benchmark both filter generation and policy evaluation together."""
    # This simulates the real workflow: generate filter + evaluate policy
    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=1000)

    documents = [
        {"institution": "MIT", "access_level": "public"},
        {"institution": "Stanford", "access_level": "internal"},
        {"institution": "Harvard", "access_level": "public"},
    ]

    filter_latencies = []
    eval_latencies = []
    total_latencies = []

    for _ in track(range(iterations), description="[cyan]Combined Optimization"):
        user = users[_ % len(users)]
        doc = documents[_ % len(documents)]

        # Measure filter generation
        start_filter = time.time()
        filter_obj = engine.to_filter(user, backend)
        filter_time = (time.time() - start_filter) * 1000 * 1000  # µs

        # Measure policy evaluation
        start_eval = time.time()
        result = engine.evaluate(user, doc)
        eval_time = (time.time() - start_eval) * 1000 * 1000  # µs

        total_time = filter_time + eval_time

        filter_latencies.append(filter_time)
        eval_latencies.append(eval_time)
        total_latencies.append(total_time)

    cache_stats = engine.get_cache_stats()

    return {
        "filter": filter_latencies,
        "eval": eval_latencies,
        "total": total_latencies,
        "cache_stats": cache_stats
    }


def print_results(results):
    """Print comprehensive benchmark results."""
    console.print("\n")
    console.print("=" * 90, style="bold")
    console.print("📊 RAGGuard Combined Optimization Benchmark", style="bold cyan", justify="center")
    console.print("=" * 90, style="bold")
    console.print()

    # Performance breakdown table
    table = Table(title="⚡ Performance Breakdown (microseconds, p50)", show_header=True)
    table.add_column("Component", style="cyan", width=30)
    table.add_column("Time (µs)", justify="right")
    table.add_column("% of Total", justify="right")

    filter_p50 = results["filter_p50"]
    eval_p50 = results["eval_p50"]
    total_p50 = results["total_p50"]

    table.add_row(
        "Filter Generation (Cached)",
        f"{filter_p50:.2f}µs",
        f"{(filter_p50/total_p50)*100:.1f}%",
        style="green"
    )
    table.add_row(
        "Policy Evaluation (Compiled)",
        f"{eval_p50:.2f}µs",
        f"{(eval_p50/total_p50)*100:.1f}%",
        style="blue"
    )
    table.add_row(
        "Total RAGGuard Overhead",
        f"{total_p50:.2f}µs",
        "100.0%",
        style="bold cyan"
    )

    console.print(table)
    console.print()

    # Percentile distribution table
    table = Table(title="📈 Latency Distribution (microseconds)", show_header=True)
    table.add_column("Component", style="cyan", width=30)
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")

    table.add_row(
        "Filter Generation",
        f"{results['filter_p50']:.2f}",
        f"{results['filter_p95']:.2f}",
        f"{results['filter_p99']:.2f}",
        style="green"
    )
    table.add_row(
        "Policy Evaluation",
        f"{results['eval_p50']:.2f}",
        f"{results['eval_p95']:.2f}",
        f"{results['eval_p99']:.2f}",
        style="blue"
    )
    table.add_row(
        "Total Overhead",
        f"{results['total_p50']:.2f}",
        f"{results['total_p95']:.2f}",
        f"{results['total_p99']:.2f}",
        style="bold cyan"
    )

    console.print(table)
    console.print()

    # Cache performance
    if results["cache_stats"]:
        stats = results["cache_stats"]
        table = Table(title="🎯 Cache Performance", show_header=True)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", justify="right")

        table.add_row("Hit Rate", f"{stats['hit_rate']*100:.1f}%", style="green")
        table.add_row("Cache Hits", f"{stats['hits']}", style="green")
        table.add_row("Cache Misses", f"{stats['misses']}", style="yellow")
        table.add_row("Cache Size", f"{stats['size']}/{stats['max_size']}", style="cyan")

        console.print(table)
        console.print()

    # Summary
    console.print("💡 Key Findings:", style="bold yellow")
    console.print()

    console.print(f"  ✅ Total RAGGuard overhead: [green]{total_p50:.2f}µs (p50)[/green]")
    console.print(f"     • Filter generation: {filter_p50:.2f}µs ({(filter_p50/total_p50)*100:.0f}%)")
    console.print(f"     • Policy evaluation: {eval_p50:.2f}µs ({(eval_p50/total_p50)*100:.0f}%)")
    console.print()

    if results["cache_stats"]:
        hit_rate = results["cache_stats"]["hit_rate"]
        console.print(f"  ✅ Filter cache hit rate: [green]{hit_rate*100:.1f}%[/green]")
        console.print(f"     • {results['cache_stats']['hits']} cache hits")
        console.print(f"     • {results['cache_stats']['misses']} cache misses (first time per user)")
        console.print()

    console.print("  🚀 Optimizations Applied:")
    console.print("     1. [green]Filter Caching[/green] - Eliminates repeated filter rebuilding")
    console.print("     2. [green]Condition Compilation[/green] - Avoids runtime string parsing")
    console.print("     3. [green]Pre-split Paths[/green] - Eliminates repeated string operations")
    console.print()

    # Comparison to pre-optimization baseline
    # Based on previous benchmark data: ~14.8µs for uncached filter generation
    # Plus ~2-3µs for string-parsed condition evaluation = ~17-18µs baseline
    estimated_baseline = 17.0  # Conservative estimate
    improvement = ((estimated_baseline - total_p50) / estimated_baseline) * 100

    console.print(f"  📊 Estimated Total Improvement:")
    console.print(f"     • Baseline (before optimizations): ~{estimated_baseline:.1f}µs")
    console.print(f"     • Optimized (current): {total_p50:.2f}µs")
    console.print(f"     • [bold green]Overall speedup: {improvement:.1f}%[/bold green]")
    console.print()


def main():
    console.print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  RAGGuard Combined Optimization Benchmark                    ║
║                                                              ║
║  Measuring cumulative impact of all optimizations            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Load policy
    console.print("📋 Loading access control policy...")
    policy = load_policy("policy.yaml")

    # Define test users (rotating to test cache behavior)
    users = [
        {"institution": "MIT", "roles": ["researcher"]},
        {"institution": "Stanford", "roles": ["researcher"]},
        {"institution": "Harvard", "roles": ["student"]},
    ]

    backend = "qdrant"
    iterations = 5000

    console.print(f"\n🏃 Running combined benchmark ({iterations} iterations)...\n")

    # Run benchmark
    benchmark_data = benchmark_combined(policy, users, backend, iterations)

    # Calculate percentiles
    results = {
        "filter_p50": np.percentile(benchmark_data["filter"], 50),
        "filter_p95": np.percentile(benchmark_data["filter"], 95),
        "filter_p99": np.percentile(benchmark_data["filter"], 99),
        "eval_p50": np.percentile(benchmark_data["eval"], 50),
        "eval_p95": np.percentile(benchmark_data["eval"], 95),
        "eval_p99": np.percentile(benchmark_data["eval"], 99),
        "total_p50": np.percentile(benchmark_data["total"], 50),
        "total_p95": np.percentile(benchmark_data["total"], 95),
        "total_p99": np.percentile(benchmark_data["total"], 99),
        "cache_stats": benchmark_data["cache_stats"]
    }

    # Print results
    print_results(results)

    console.print("✨ Benchmark complete!", style="bold green")
    console.print()


if __name__ == "__main__":
    main()
