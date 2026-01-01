"""
Test concurrent access to the policy engine cache.

Verifies thread safety of the LRU cache implementation.
"""

import threading
from unittest.mock import Mock

import pytest

# Skip all tests if qdrant-client is not installed (required for filter generation)
pytest.importorskip("qdrant_client")

from ragguard.policy.models import Policy
from ragguard.retrievers.base import BaseSecureRetriever

# Metrics moved to enterprise - make import optional
try:
    from ragguard_enterprise.metrics import reset_metrics
except ImportError:
    def reset_metrics():
        pass  # No-op if enterprise not installed


class MockRetriever(BaseSecureRetriever):
    """Mock retriever for testing."""

    @property
    def backend_name(self) -> str:
        return "qdrant"

    def _execute_search(self, query, filter, limit, **kwargs):
        """Mock search that returns dummy results."""
        return [{"id": f"doc{i}", "score": 0.9 - (i * 0.1)} for i in range(limit)]

    def _check_backend_health(self):
        """Mock health check."""
        return {"status": "healthy"}


@pytest.fixture
def policy():
    """Create a simple test policy."""
    return Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "department_match",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })


@pytest.fixture
def mock_client():
    """Create a mock database client."""
    return Mock()


@pytest.fixture(autouse=True)
def reset_collector():
    """Reset metrics before each test."""
    reset_metrics()
    yield
    reset_metrics()


def test_concurrent_cache_access(policy, mock_client):
    """
    Test that concurrent cache access is thread-safe.

    Multiple threads should be able to access the cache simultaneously
    without race conditions or data corruption.
    """
    retriever = MockRetriever(
        client=mock_client,
        collection="test",
        policy=policy,
        enable_filter_cache=True,
        filter_cache_size=100
    )

    query = [0.1, 0.2, 0.3]
    num_threads = 10
    queries_per_thread = 20
    errors = []
    results_count = []

    def worker(thread_id: int):
        """Worker function that performs queries."""
        try:
            for i in range(queries_per_thread):
                # Each thread uses a mix of shared and unique users
                if i % 2 == 0:
                    # Shared user (should hit cache)
                    user = {"id": "shared", "department": f"dept{i % 3}"}
                else:
                    # Thread-specific user (cache miss)
                    user = {"id": f"user{thread_id}", "department": f"dept{thread_id}"}

                results = retriever.search(query, user, limit=5)
                results_count.append(len(results))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Launch threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify all queries returned results
    assert len(results_count) == num_threads * queries_per_thread
    assert all(count == 5 for count in results_count)

    # Verify cache stats are consistent
    cache_stats = retriever.policy_engine.get_cache_stats()
    assert cache_stats is not None
    assert cache_stats['hits'] + cache_stats['misses'] == num_threads * queries_per_thread


def test_concurrent_cache_updates(policy, mock_client):
    """
    Test that concurrent cache updates don't corrupt data.

    Multiple threads adding different entries should all succeed.
    """
    retriever = MockRetriever(
        client=mock_client,
        collection="test",
        policy=policy,
        enable_filter_cache=True,
        filter_cache_size=1000
    )

    query = [0.1, 0.2, 0.3]
    num_threads = 20
    unique_users_per_thread = 10
    errors = []

    def worker(thread_id: int):
        """Worker that creates unique cache entries."""
        try:
            for i in range(unique_users_per_thread):
                user = {"id": f"user_{thread_id}_{i}", "department": f"dept_{thread_id}"}
                retriever.search(query, user, limit=5)
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Launch threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify cache stats
    cache_stats = retriever.policy_engine.get_cache_stats()
    expected_queries = num_threads * unique_users_per_thread
    assert cache_stats['hits'] + cache_stats['misses'] == expected_queries


def test_concurrent_cache_eviction(policy, mock_client):
    """
    Test that cache eviction works correctly under concurrent load.

    When cache is full, LRU eviction should work without errors.
    """
    # Small cache to force eviction
    retriever = MockRetriever(
        client=mock_client,
        collection="test",
        policy=policy,
        enable_filter_cache=True,
        filter_cache_size=50  # Small cache
    )

    query = [0.1, 0.2, 0.3]
    num_threads = 10
    users_per_thread = 20  # 200 total users > 50 cache size
    errors = []

    def worker(thread_id: int):
        """Worker that creates more entries than cache can hold."""
        try:
            for i in range(users_per_thread):
                user = {"id": f"user_{thread_id}_{i}", "department": f"dept{i}"}
                retriever.search(query, user, limit=5)
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Launch threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify no errors occurred during eviction
    assert len(errors) == 0, f"Errors occurred during cache eviction: {errors}"

    # Verify cache stats are consistent
    cache_stats = retriever.policy_engine.get_cache_stats()
    assert cache_stats is not None
    # Cache size should be capped at max size
    # With 200 unique users and cache size of 50, eviction must have occurred


def test_cache_disabled_concurrent(policy, mock_client):
    """
    Test that cache disabled works correctly with concurrent access.

    All queries should be cache misses when cache is disabled.
    """
    retriever = MockRetriever(
        client=mock_client,
        collection="test",
        policy=policy,
        enable_filter_cache=False  # Cache disabled
    )

    query = [0.1, 0.2, 0.3]
    num_threads = 5
    queries_per_thread = 10
    errors = []

    def worker(thread_id: int):
        """Worker function."""
        try:
            for i in range(queries_per_thread):
                user = {"id": f"user{thread_id}", "department": "eng"}
                retriever.search(query, user, limit=5)
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Launch threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # With cache disabled, cache stats should be None or all zeros
    cache_stats = retriever.policy_engine.get_cache_stats()
    if cache_stats is not None:
        assert cache_stats['hits'] == 0
        assert cache_stats['cache_enabled'] is False
