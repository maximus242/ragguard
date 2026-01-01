#!/usr/bin/env python3
"""
RAGGuard Monitoring & Observability Example

Demonstrates how to add monitoring to RAGGuard applications:
- Prometheus metrics
- Custom metrics tracking
- Performance monitoring
- Access denial tracking
- Filter cache monitoring

Requirements:
    pip install prometheus-client

Usage:
    python examples/monitoring.py

    Then visit http://localhost:8000/metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from typing import Dict, Any, Optional

# Check prometheus availability
PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        start_http_server,
        CollectorRegistry,
        generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Mock classes for demo mode
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, amount=1): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def observe(self, amount): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, data): pass
    CollectorRegistry = None
    def start_http_server(port): pass
    def generate_latest(): return b""


class RAGGuardMetrics:
    """
    Prometheus metrics collector for RAGGuard.

    Tracks:
    - Query counts and latencies
    - Policy evaluations (allow/deny)
    - Filter generation performance
    - Cache hit rates
    - Error rates
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus registry (optional, uses default if None)
        """
        self.registry = registry

        # === Query Metrics ===

        self.queries_total = Counter(
            'ragguard_queries_total',
            'Total number of queries processed',
            ['backend', 'status'],  # Labels: qdrant/chromadb, success/error
            registry=registry
        )

        self.query_latency = Histogram(
            'ragguard_query_latency_seconds',
            'Query latency in seconds',
            ['backend'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=registry
        )

        # === Policy Evaluation Metrics ===

        self.policy_evaluations_total = Counter(
            'ragguard_policy_evaluations_total',
            'Total policy evaluations',
            ['policy_name', 'result'],  # result: allow/deny
            registry=registry
        )

        self.policy_evaluation_latency = Histogram(
            'ragguard_policy_evaluation_latency_seconds',
            'Policy evaluation latency',
            ['policy_name'],
            buckets=[0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001],
            registry=registry
        )

        # === Filter Generation Metrics ===

        self.filter_generation_total = Counter(
            'ragguard_filter_generation_total',
            'Total filter generations',
            ['backend'],
            registry=registry
        )

        self.filter_generation_latency = Histogram(
            'ragguard_filter_generation_latency_seconds',
            'Filter generation latency',
            ['backend'],
            buckets=[0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001],
            registry=registry
        )

        # === Cache Metrics ===

        self.cache_hits_total = Counter(
            'ragguard_cache_hits_total',
            'Filter cache hits',
            registry=registry
        )

        self.cache_misses_total = Counter(
            'ragguard_cache_misses_total',
            'Filter cache misses',
            registry=registry
        )

        self.cache_size = Gauge(
            'ragguard_cache_size',
            'Current cache size',
            registry=registry
        )

        # === Access Denial Metrics ===

        self.access_denied_total = Counter(
            'ragguard_access_denied_total',
            'Total access denials',
            ['user_id', 'policy_name', 'reason'],
            registry=registry
        )

        self.documents_filtered = Counter(
            'ragguard_documents_filtered_total',
            'Total documents filtered out',
            ['policy_name'],
            registry=registry
        )

        # === Results Metrics ===

        self.results_returned = Histogram(
            'ragguard_results_returned',
            'Number of results returned per query',
            ['backend'],
            buckets=[0, 1, 5, 10, 20, 50, 100],
            registry=registry
        )

        # === Error Metrics ===

        self.errors_total = Counter(
            'ragguard_errors_total',
            'Total errors',
            ['error_type', 'backend'],
            registry=registry
        )

        # === Info Metrics ===

        self.info = Info(
            'ragguard_build_info',
            'RAGGuard build information',
            registry=registry
        )
        self.info.info({
            'version': '0.2.0',
            'python_version': sys.version.split()[0]
        })

    # === Helper Methods ===

    def record_query(self, backend: str, latency_seconds: float, status: str = 'success', results_count: int = 0):
        """Record a query execution."""
        self.queries_total.labels(backend=backend, status=status).inc()
        self.query_latency.labels(backend=backend).observe(latency_seconds)
        self.results_returned.labels(backend=backend).observe(results_count)

    def record_policy_evaluation(self, policy_name: str, allowed: bool, latency_seconds: float):
        """Record a policy evaluation."""
        result = 'allow' if allowed else 'deny'
        self.policy_evaluations_total.labels(policy_name=policy_name, result=result).inc()
        self.policy_evaluation_latency.labels(policy_name=policy_name).observe(latency_seconds)

    def record_filter_generation(self, backend: str, latency_seconds: float):
        """Record filter generation."""
        self.filter_generation_total.labels(backend=backend).inc()
        self.filter_generation_latency.labels(backend=backend).observe(latency_seconds)

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits_total.inc()

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses_total.inc()

    def update_cache_size(self, size: int):
        """Update current cache size."""
        self.cache_size.set(size)

    def record_access_denied(self, user_id: str, policy_name: str, reason: str = 'policy_violation'):
        """Record an access denial."""
        self.access_denied_total.labels(
            user_id=user_id,
            policy_name=policy_name,
            reason=reason
        ).inc()

    def record_documents_filtered(self, policy_name: str, count: int):
        """Record number of documents filtered out."""
        self.documents_filtered.labels(policy_name=policy_name).inc(count)

    def record_error(self, error_type: str, backend: str):
        """Record an error."""
        self.errors_total.labels(error_type=error_type, backend=backend).inc()


class MonitoredSecureRetriever:
    """
    Example: Wrapper that adds monitoring to any SecureRetriever.

    This shows how to instrument your RAGGuard retrievers with metrics.
    """

    def __init__(self, retriever, metrics: RAGGuardMetrics, backend_name: str):
        """
        Args:
            retriever: Any SecureRetriever (QdrantSecure, ChromaSecure, etc.)
            metrics: RAGGuardMetrics instance
            backend_name: Backend name for metrics labels
        """
        self.retriever = retriever
        self.metrics = metrics
        self.backend_name = backend_name

    def search(self, query: str, user: Dict[str, Any], limit: int = 10):
        """
        Search with monitoring.

        All metrics are automatically recorded.
        """
        start_time = time.perf_counter()
        status = 'success'
        results = []

        try:
            # Execute search
            results = self.retriever.search(query=query, user=user, limit=limit)

            # Record success
            latency = time.perf_counter() - start_time
            self.metrics.record_query(
                backend=self.backend_name,
                latency_seconds=latency,
                status='success',
                results_count=len(results)
            )

            return results

        except Exception as e:
            # Record error
            latency = time.perf_counter() - start_time
            error_type = type(e).__name__
            self.metrics.record_error(error_type=error_type, backend=self.backend_name)
            self.metrics.record_query(
                backend=self.backend_name,
                latency_seconds=latency,
                status='error',
                results_count=0
            )
            raise


# ============================================================================
# Demo: Simulated Monitoring
# ============================================================================

def demo_monitoring():
    """Demonstrate RAGGuard monitoring."""

    print("=" * 80)
    print("RAGGuard Monitoring & Observability Demo")
    print("=" * 80)
    print()

    if not PROMETHEUS_AVAILABLE:
        print("Running in demo mode (prometheus-client not installed)")
        print("Install with: pip install prometheus-client")
        print()

        # Simulate metrics
        print("📊 Simulated Metrics:\n")
        print("ragguard_queries_total{backend=\"qdrant\",status=\"success\"} 1245")
        print("ragguard_query_latency_seconds{backend=\"qdrant\",quantile=\"0.5\"} 0.0063")
        print("ragguard_policy_evaluations_total{policy_name=\"dept-docs\",result=\"allow\"} 982")
        print("ragguard_cache_hits_total 745")
        print("ragguard_cache_misses_total 500")
        print("ragguard_access_denied_total{user_id=\"bob\",reason=\"policy_violation\"} 12")
        print()
        print("✅ In production, these metrics would be scraped by Prometheus")
        print("   and visualized in Grafana.")
        return

    # Initialize metrics
    metrics = RAGGuardMetrics()

    # Start Prometheus HTTP server
    print("🚀 Starting Prometheus metrics server on http://localhost:8000")
    start_http_server(8000)

    print("   Metrics available at: http://localhost:8000/metrics")
    print()

    # Simulate some queries
    print("📊 Simulating 100 queries with monitoring...\n")

    for i in range(100):
        # Simulate query
        backend = 'qdrant' if i % 2 == 0 else 'chromadb'
        latency = 0.005 + (i % 10) * 0.001  # 5-15ms
        status = 'success' if i % 10 != 0 else 'error'  # 10% error rate
        results_count = 10 if status == 'success' else 0

        # Record metrics
        metrics.record_query(
            backend=backend,
            latency_seconds=latency,
            status=status,
            results_count=results_count
        )

        # Policy evaluation
        allowed = i % 5 != 0  # 80% allow rate
        metrics.record_policy_evaluation(
            policy_name='dept-docs',
            allowed=allowed,
            latency_seconds=0.000003  # 3μs
        )

        if not allowed:
            metrics.record_access_denied(
                user_id=f"user{i}",
                policy_name='dept-docs',
                reason='department_mismatch'
            )

        # Filter generation
        metrics.record_filter_generation(
            backend=backend,
            latency_seconds=0.000011  # 11μs
        )

        # Cache
        if i % 3 == 0:
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

        # Update cache size
        metrics.update_cache_size(min(i, 50))

    print("✅ Simulation complete!")
    print()
    print("📈 Sample Metrics:")
    print()

    # Print some metrics
    print(f"   Total Queries:     ~100")
    print(f"   Success Rate:      ~90%")
    print(f"   Avg Latency:       ~10ms")
    print(f"   Allow Rate:        ~80%")
    print(f"   Cache Hit Rate:    ~33%")
    print()

    print("🌐 Visit http://localhost:8000/metrics to see all metrics")
    print()
    print("📊 Example Prometheus Queries:")
    print()
    print("   # Query rate per second")
    print("   rate(ragguard_queries_total[5m])")
    print()
    print("   # P95 query latency")
    print("   histogram_quantile(0.95, ragguard_query_latency_seconds)")
    print()
    print("   # Cache hit rate")
    print("   ragguard_cache_hits_total / (ragguard_cache_hits_total + ragguard_cache_misses_total)")
    print()
    print("   # Access denial rate")
    print("   rate(ragguard_access_denied_total[5m])")
    print()

    print("🎯 Integration with Grafana:")
    print("   1. Add Prometheus as data source")
    print("   2. Import dashboard from examples/grafana_dashboard.json")
    print("   3. View real-time RAGGuard metrics")
    print()

    print("⏸️  Press Ctrl+C to stop the metrics server")
    print()

    try:
        # Keep server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down metrics server...")


# ============================================================================
# Production Example
# ============================================================================

def create_monitored_retriever_example():
    """
    Example: How to create a monitored retriever in production.
    """
    print("\n" + "=" * 80)
    print("Production Example: Monitored Retriever")
    print("=" * 80)
    print()

    code = '''
# In your production application:

from ragguard import QdrantSecureRetriever, load_policy
from qdrant_client import QdrantClient
from examples.monitoring import RAGGuardMetrics, MonitoredSecureRetriever
from prometheus_client import start_http_server

# 1. Initialize metrics
metrics = RAGGuardMetrics()

# 2. Start metrics server (in a separate thread/process)
start_http_server(8000)

# 3. Create your RAGGuard retriever
client = QdrantClient("localhost", port=6333)
policy = load_policy("policy.yaml")

base_retriever = QdrantSecureRetriever(
    client=client,
    collection="documents",
    policy=policy,
    embed_fn=embeddings.embed_query
)

# 4. Wrap with monitoring
monitored_retriever = MonitoredSecureRetriever(
    retriever=base_retriever,
    metrics=metrics,
    backend_name="qdrant"
)

# 5. Use normally - metrics are automatic!
results = monitored_retriever.search(
    query="machine learning papers",
    user={"id": "alice", "department": "engineering"},
    limit=10
)

# All metrics are automatically recorded:
# - Query latency
# - Results count
# - Errors (if any)
'''

    print(code)
    print()


# ============================================================================
# Grafana Dashboard Example
# ============================================================================

def show_grafana_dashboard_config():
    """Show example Grafana dashboard configuration."""
    print("\n" + "=" * 80)
    print("Grafana Dashboard Configuration")
    print("=" * 80)
    print()

    print("📊 Key Panels to Add:")
    print()

    panels = [
        {
            "title": "Query Rate (QPS)",
            "query": "sum(rate(ragguard_queries_total[1m]))",
            "type": "Graph"
        },
        {
            "title": "P50/P95/P99 Latency",
            "query": "histogram_quantile(0.95, ragguard_query_latency_seconds)",
            "type": "Graph"
        },
        {
            "title": "Success Rate",
            "query": "sum(rate(ragguard_queries_total{status=\"success\"}[5m])) / sum(rate(ragguard_queries_total[5m]))",
            "type": "Gauge"
        },
        {
            "title": "Cache Hit Rate",
            "query": "ragguard_cache_hits_total / (ragguard_cache_hits_total + ragguard_cache_misses_total)",
            "type": "Gauge"
        },
        {
            "title": "Access Denials by User",
            "query": "topk(10, sum by (user_id) (rate(ragguard_access_denied_total[5m])))",
            "type": "Table"
        },
        {
            "title": "Error Rate by Type",
            "query": "sum by (error_type) (rate(ragguard_errors_total[5m]))",
            "type": "Graph"
        }
    ]

    for i, panel in enumerate(panels, 1):
        print(f"{i}. {panel['title']}")
        print(f"   Type: {panel['type']}")
        print(f"   Query: {panel['query']}")
        print()

    print("📁 Full dashboard config: examples/grafana_dashboard.json (to be created)")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run monitoring demo."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║  RAGGuard Monitoring & Observability Demo                   ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n")

    demo_monitoring()
    create_monitored_retriever_example()
    show_grafana_dashboard_config()

    print("=" * 80)
    print("✨ Monitoring Demo Complete!")
    print("=" * 80)
    print()
    print("📚 Learn More:")
    print("   - Prometheus: https://prometheus.io/docs/")
    print("   - Grafana: https://grafana.com/docs/")
    print("   - Production setup: See examples/monitoring.py")
    print()


if __name__ == "__main__":
    main()
