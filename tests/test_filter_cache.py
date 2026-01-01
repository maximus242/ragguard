"""
Tests for filter caching infrastructure.

Tests the FilterCache class and its integration with PolicyEngine to ensure:
- Correct cache hit/miss behavior
- LRU eviction when cache is full
- Thread safety for concurrent access
- Cache invalidation on policy changes
- Efficient cache key generation
"""

import threading
import time

import pytest

# Skip all tests if qdrant-client is not installed (required for filter generation)
pytest.importorskip("qdrant_client")

from ragguard.filters.cache import (
    CacheKeyBuilder,
    FilterCache,
    compute_policy_hash,
    extract_user_fields_from_policy,
)
from ragguard.policy import Policy, PolicyEngine

# ============================================================================
# FilterCache Unit Tests
# ============================================================================

def test_filter_cache_basic_get_set():
    """Test basic cache get/set operations."""
    cache = FilterCache(max_size=10)

    # Cache miss
    assert cache.get("key1") is None

    # Set and get
    cache.set("key1", {"filter": "value1"})
    assert cache.get("key1") == {"filter": "value1"}

    # Stats
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_filter_cache_lru_eviction():
    """Test LRU eviction when cache reaches max size."""
    cache = FilterCache(max_size=3)

    # Fill cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    assert len(cache) == 3

    # Add 4th item - should evict key1 (least recently used)
    cache.set("key4", "value4")
    assert len(cache) == 3
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"


def test_filter_cache_lru_update_on_access():
    """Test that accessing an item updates its LRU position."""
    cache = FilterCache(max_size=3)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Access key1 to make it recently used
    cache.get("key1")

    # Add key4 - should evict key2 (now least recently used)
    cache.set("key4", "value4")

    assert cache.get("key1") == "value1"  # Still in cache
    assert cache.get("key2") is None      # Evicted
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"


def test_filter_cache_update_existing():
    """Test updating an existing cache entry."""
    cache = FilterCache(max_size=3)

    cache.set("key1", "value1")
    cache.set("key1", "value1_updated")

    assert cache.get("key1") == "value1_updated"
    assert len(cache) == 1  # Should not create duplicate


def test_filter_cache_invalidate():
    """Test invalidating specific cache entries."""
    cache = FilterCache(max_size=10)

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Invalidate key1
    assert cache.invalidate("key1") is True
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"

    # Try to invalidate non-existent key
    assert cache.invalidate("key3") is False


def test_filter_cache_invalidate_all():
    """Test clearing all cache entries."""
    cache = FilterCache(max_size=10)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Clear cache
    cache.invalidate_all()

    assert len(cache) == 0
    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert cache.get("key3") is None

    # Stats should be preserved
    stats = cache.get_stats()
    assert stats["hits"] > 0 or stats["misses"] > 0


def test_filter_cache_stats():
    """Test cache statistics tracking."""
    cache = FilterCache(max_size=10)

    # Initial stats
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0.0
    assert stats["size"] == 0
    assert stats["max_size"] == 10

    # Generate some hits and misses
    cache.get("key1")  # miss
    cache.set("key1", "value1")
    cache.get("key1")  # hit
    cache.get("key1")  # hit
    cache.get("key2")  # miss

    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 0.5
    assert stats["size"] == 1


def test_filter_cache_reset_stats():
    """Test resetting cache statistics."""
    cache = FilterCache(max_size=10)

    cache.set("key1", "value1")
    cache.get("key1")
    cache.get("key2")

    stats = cache.get_stats()
    assert stats["hits"] > 0 or stats["misses"] > 0

    cache.reset_stats()

    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_filter_cache_thread_safety():
    """Test thread-safe concurrent access to cache."""
    cache = FilterCache(max_size=100)
    errors = []

    def worker(thread_id):
        try:
            for i in range(50):
                key = f"key_{thread_id}_{i}"
                cache.set(key, f"value_{thread_id}_{i}")
                value = cache.get(key)
                assert value == f"value_{thread_id}_{i}"
        except Exception as e:
            errors.append(e)

    # Run 10 threads concurrently
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0

    # Cache should have entries (may be evicted due to LRU)
    stats = cache.get_stats()
    assert stats["size"] <= 100


def test_filter_cache_invalid_size():
    """Test that invalid cache sizes are rejected."""
    with pytest.raises(ValueError):
        FilterCache(max_size=0)

    with pytest.raises(ValueError):
        FilterCache(max_size=-1)


# ============================================================================
# CacheKeyBuilder Tests
# ============================================================================

def test_cache_key_builder_stable():
    """Test that cache keys are stable for same inputs."""
    user = {"id": "user1", "roles": ["admin"], "department": "eng"}
    relevant_fields = {"roles", "department"}

    key1 = CacheKeyBuilder.build_key("qdrant", "policy123", user, relevant_fields)
    key2 = CacheKeyBuilder.build_key("qdrant", "policy123", user, relevant_fields)

    assert key1 == key2


def test_cache_key_builder_different_backends():
    """Test that different backends produce different keys."""
    user = {"id": "user1", "roles": ["admin"]}
    relevant_fields = {"roles"}

    key_qdrant = CacheKeyBuilder.build_key("qdrant", "policy123", user, relevant_fields)
    key_pgvector = CacheKeyBuilder.build_key("pgvector", "policy123", user, relevant_fields)

    assert key_qdrant != key_pgvector


def test_cache_key_builder_different_policies():
    """Test that different policies produce different keys."""
    user = {"id": "user1", "roles": ["admin"]}
    relevant_fields = {"roles"}

    key1 = CacheKeyBuilder.build_key("qdrant", "policy123", user, relevant_fields)
    key2 = CacheKeyBuilder.build_key("qdrant", "policy456", user, relevant_fields)

    assert key1 != key2


def test_cache_key_builder_relevant_fields_only():
    """Test that only relevant user fields affect the key."""
    relevant_fields = {"roles"}

    user1 = {"id": "user1", "roles": ["admin"], "email": "user1@example.com"}
    user2 = {"id": "user2", "roles": ["admin"], "email": "user2@example.com"}

    # Different IDs and emails, but same roles
    key1 = CacheKeyBuilder.build_key("qdrant", "policy123", user1, relevant_fields)
    key2 = CacheKeyBuilder.build_key("qdrant", "policy123", user2, relevant_fields)

    # Keys should be the same because only "roles" is relevant
    assert key1 == key2


def test_cache_key_builder_different_relevant_values():
    """Test that different relevant field values produce different keys."""
    relevant_fields = {"roles"}

    user1 = {"roles": ["admin"]}
    user2 = {"roles": ["user"]}

    key1 = CacheKeyBuilder.build_key("qdrant", "policy123", user1, relevant_fields)
    key2 = CacheKeyBuilder.build_key("qdrant", "policy123", user2, relevant_fields)

    assert key1 != key2


def test_cache_key_builder_list_normalization():
    """Test that role lists are normalized for consistent hashing."""
    relevant_fields = {"roles"}

    user1 = {"roles": ["admin", "user"]}
    user2 = {"roles": ["user", "admin"]}  # Same roles, different order

    key1 = CacheKeyBuilder.build_key("qdrant", "policy123", user1, relevant_fields)
    key2 = CacheKeyBuilder.build_key("qdrant", "policy123", user2, relevant_fields)

    # Should produce the same key (lists are sorted)
    assert key1 == key2


def test_cache_key_builder_nested_fields():
    """Test cache key generation with nested fields."""
    relevant_fields = {"metadata.team", "roles"}

    user = {
        "id": "user1",
        "roles": ["admin"],
        "metadata": {"team": "backend"}
    }

    key = CacheKeyBuilder.build_key("qdrant", "policy123", user, relevant_fields)
    assert key is not None
    assert len(key) > 0


# ============================================================================
# Policy Hash Tests
# ============================================================================

def test_compute_policy_hash_stable():
    """Test that policy hashes are stable."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "test", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    hash1 = compute_policy_hash(policy)
    hash2 = compute_policy_hash(policy)

    assert hash1 == hash2
    assert len(hash1) == 16  # Should be 16 characters


def test_compute_policy_hash_different_policies():
    """Test that different policies produce different hashes."""
    policy1 = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test1", "allow": {"roles": ["admin"]}}],
        "default": "deny"
    })

    policy2 = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test2", "allow": {"roles": ["user"]}}],
        "default": "deny"
    })

    hash1 = compute_policy_hash(policy1)
    hash2 = compute_policy_hash(policy2)

    assert hash1 != hash2


# ============================================================================
# User Fields Extraction Tests
# ============================================================================

def test_extract_user_fields_roles():
    """Test that roles field is always extracted."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "test", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    fields = extract_user_fields_from_policy(policy)
    assert "roles" in fields


def test_extract_user_fields_conditions():
    """Test extracting user fields from conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "test",
                "allow": {
                    "conditions": [
                        "user.department == document.department",
                        "user.team in document.allowed_teams"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    fields = extract_user_fields_from_policy(policy)
    assert "roles" in fields
    assert "department" in fields
    assert "team" in fields


def test_extract_user_fields_nested():
    """Test extracting nested user fields."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "test",
                "allow": {
                    "conditions": [
                        "user.metadata.team == document.team"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    fields = extract_user_fields_from_policy(policy)
    assert "metadata.team" in fields


# ============================================================================
# PolicyEngine Integration Tests
# ============================================================================

def test_policy_engine_cache_hit():
    """Test that PolicyEngine cache hits work correctly."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=10)
    user = {"id": "user1", "roles": ["admin"]}

    # First call - cache miss
    filter1 = engine.to_filter(user, "qdrant")
    stats1 = engine.get_cache_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    # Second call - cache hit
    filter2 = engine.to_filter(user, "qdrant")
    stats2 = engine.get_cache_stats()
    assert stats2["hits"] == 1
    assert stats2["misses"] == 1

    # Filters should be the same object (from cache)
    assert filter1 is filter2


def test_policy_engine_cache_different_users():
    """Test that different users get different cached filters."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)

    user1 = {"id": "user1", "department": "eng"}
    user2 = {"id": "user2", "department": "sales"}

    filter1 = engine.to_filter(user1, "qdrant")
    filter2 = engine.to_filter(user2, "qdrant")

    # Different users should get different filters
    assert filter1 is not filter2

    stats = engine.get_cache_stats()
    assert stats["size"] == 2  # Two entries cached


def test_policy_engine_cache_disabled():
    """Test that caching can be disabled."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=False)
    user = {"id": "user1", "roles": ["admin"]}

    filter1 = engine.to_filter(user, "qdrant")
    filter2 = engine.to_filter(user, "qdrant")

    # Cache disabled - stats should be None
    assert engine.get_cache_stats() is None


def test_policy_engine_invalidate_cache():
    """Test cache invalidation."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)
    user = {"id": "user1", "roles": ["admin"]}

    # Generate cache entry
    engine.to_filter(user, "qdrant")
    stats = engine.get_cache_stats()
    assert stats["size"] == 1

    # Invalidate cache
    engine.invalidate_cache()

    stats = engine.get_cache_stats()
    assert stats["size"] == 0

    # Next call should be a cache miss
    engine.to_filter(user, "qdrant")
    stats = engine.get_cache_stats()
    assert stats["misses"] == 2  # Original miss + new miss after invalidation


def test_policy_engine_cache_different_backends():
    """Test that different backends get separate cache entries."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)
    user = {"id": "user1", "roles": ["admin"]}

    # Request filters for different backends
    filter_qdrant = engine.to_filter(user, "qdrant")
    filter_pgvector = engine.to_filter(user, "pgvector")

    # Should be different objects (different backends)
    assert filter_qdrant is not filter_pgvector

    stats = engine.get_cache_stats()
    assert stats["size"] == 2  # Two entries (one per backend)


def test_policy_engine_cache_lru_integration():
    """Test LRU eviction in PolicyEngine."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True, filter_cache_size=2)

    user1 = {"id": "user1", "roles": ["admin"]}
    user2 = {"id": "user2", "roles": ["user"]}
    user3 = {"id": "user3", "roles": ["guest"]}

    # Fill cache
    engine.to_filter(user1, "qdrant")
    engine.to_filter(user2, "qdrant")

    stats = engine.get_cache_stats()
    assert stats["size"] == 2

    # Add third user - should evict user1
    engine.to_filter(user3, "qdrant")

    stats = engine.get_cache_stats()
    assert stats["size"] == 2  # Size should stay at max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
