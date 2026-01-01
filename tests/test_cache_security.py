"""
Security-focused tests for filter caching.

These tests verify that caching doesn't introduce security vulnerabilities:
- Stale filters after policy changes
- Cached filters with outdated user permissions
- Cross-user filter leakage
"""

from unittest.mock import MagicMock

import pytest

# Skip all tests if qdrant-client is not installed
pytest.importorskip("qdrant_client")

from ragguard import QdrantSecureRetriever
from ragguard.policy import Policy, PolicyEngine


def test_policy_change_invalidates_cache_via_hash():
    """
    CRITICAL: Test that changing the policy invalidates cached filters.

    Security risk: If policy changes but cache isn't invalidated, users could
    get filters based on old policy rules.
    """
    policy1 = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin-only", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy1, enable_filter_cache=True)
    user = {"roles": ["user"]}

    # Generate filter with policy1 (should deny)
    filter1 = engine.to_filter(user, "qdrant")

    # Change policy to allow users
    policy2 = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "all-users", "allow": {"roles": ["admin", "user"]}}
        ],
        "default": "deny"
    })

    # Create NEW engine with new policy (recommended approach)
    engine2 = PolicyEngine(policy2, enable_filter_cache=True)

    # Generate filter with policy2 (should allow)
    filter2 = engine2.to_filter(user, "qdrant")

    # Filters should be different because policy changed
    assert filter1 != filter2

    # Verify: policy hashes should be different
    from ragguard.filters.cache import compute_policy_hash
    hash1 = compute_policy_hash(policy1)
    hash2 = compute_policy_hash(policy2)
    assert hash1 != hash2, "Policy hash should change when policy changes"


def test_user_role_change_gets_new_filter():
    """
    CRITICAL: Test that changing user roles results in a different filter.

    Security risk: If role changes don't affect cache key, user could get
    old filter with stale permissions.
    """
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)

    # User starts in engineering
    user_eng = {"id": "user123", "department": "engineering"}
    filter_eng = engine.to_filter(user_eng, "qdrant")

    # User moves to sales (same user ID, different department)
    user_sales = {"id": "user123", "department": "sales"}
    filter_sales = engine.to_filter(user_sales, "qdrant")

    # Filters MUST be different (department is a relevant field)
    assert filter_eng != filter_sales, "Filter should change when user department changes"

    # Cache should have 2 entries
    stats = engine.get_cache_stats()
    assert stats["size"] == 2, "Different user contexts should create separate cache entries"


def test_no_cross_user_cache_leakage():
    """
    CRITICAL: Test that users don't get each other's cached filters.

    Security risk: Cache key collision could leak filters between users.
    """
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)

    # Two users with different departments
    user1 = {"id": "alice", "department": "engineering"}
    user2 = {"id": "bob", "department": "sales"}

    # Generate filters
    filter1 = engine.to_filter(user1, "qdrant")
    filter2 = engine.to_filter(user2, "qdrant")

    # Get again (should hit cache)
    filter1_cached = engine.to_filter(user1, "qdrant")
    filter2_cached = engine.to_filter(user2, "qdrant")

    # Each user should get their own filter
    assert filter1 is filter1_cached, "User 1 should get cached filter"
    assert filter2 is filter2_cached, "User 2 should get cached filter"
    assert filter1 is not filter2, "Users should never share filters"


def test_irrelevant_user_fields_dont_break_cache():
    """
    Test that irrelevant user fields (like email, name) don't affect cache key.

    This is a performance optimization: users with same id/roles should
    share cache entries even if they have different emails/names.

    Note: The 'id' field IS relevant since it's commonly used in conditions
    like "user.id in document.authorized_users". So users with different IDs
    correctly get different cache entries.
    """
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)

    # Same user with different irrelevant fields (email, name not in policy)
    # Using the same ID since 'id' is a relevant field for security
    user1 = {"id": "admin-user", "roles": ["admin"], "email": "alice@example.com", "name": "Alice"}
    user2 = {"id": "admin-user", "roles": ["admin"], "email": "bob@example.com", "name": "Bob"}

    # Generate filters
    filter1 = engine.to_filter(user1, "qdrant")
    filter2 = engine.to_filter(user2, "qdrant")

    # Should be same object (cache hit) since id and roles are the same
    assert filter1 is filter2, "Users with same relevant fields (id, roles) should share cache entry"

    # Should only have 1 cache entry
    stats = engine.get_cache_stats()
    assert stats["size"] == 1
    assert stats["hit_rate"] == 0.5  # 1 hit, 1 miss


def test_manual_policy_update_is_now_safe():
    """
    SECURITY FIX: Policy updates via property setter now automatically recreate the engine.

    Previously, updating retriever.policy would leave stale filters in cache.
    Now, the policy property setter automatically recreates the PolicyEngine,
    which clears the cache and ensures correctness.
    """
    from qdrant_client import QdrantClient

    # Mock client
    client = MagicMock(spec=QdrantClient)

    policy1 = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "admin", "allow": {"roles": ["admin"]}}],
        "default": "deny"
    })

    retriever = QdrantSecureRetriever(
        client=client,
        collection="test",
        policy=policy1,
        enable_filter_cache=True
    )

    user = {"roles": ["user"]}

    # Generate filter with policy1 (should deny)
    filter1 = retriever.policy_engine.to_filter(user, "qdrant")

    # Update policy via property setter
    policy2 = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"roles": ["admin", "user"]}}],
        "default": "deny"
    })
    retriever.policy = policy2  # This now automatically recreates the engine

    # Get filter with policy2 (should allow)
    filter2 = retriever.policy_engine.to_filter(user, "qdrant")

    # Filters should be different (policy changed)
    assert filter1 != filter2, "Policy update should result in different filter"

    # Cache should be fresh (automatically cleared by property setter)
    stats = retriever.get_cache_stats()
    assert stats["size"] == 1, "New engine should have fresh cache with 1 entry"
    assert stats["misses"] == 1, "Should have had 1 cache miss (fresh engine)"

    # Verify the new policy is actually being used
    assert retriever.policy is policy2
    assert retriever.policy_engine.policy is policy2


def test_policy_hash_changes_on_rule_modification():
    """Test that ANY policy change (rules, default, conditions) changes hash."""
    from ragguard.filters.cache import compute_policy_hash

    policy1 = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"roles": ["admin"]}}],
        "default": "deny"
    })

    policy2 = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"roles": ["admin", "user"]}}],  # Added "user"
        "default": "deny"
    })

    policy3 = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"roles": ["admin"]}}],
        "default": "allow"  # Changed default
    })

    hash1 = compute_policy_hash(policy1)
    hash2 = compute_policy_hash(policy2)
    hash3 = compute_policy_hash(policy3)

    assert hash1 != hash2, "Hash should change when roles change"
    assert hash1 != hash3, "Hash should change when default changes"
    assert hash2 != hash3, "Different policies should have different hashes"


def test_cache_thread_safety_no_corruption():
    """
    Test that concurrent access doesn't corrupt cache or leak filters.

    Security risk: Race conditions could cause user A to get user B's filter.
    """
    import threading

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)

    results = {}
    errors = []

    def worker(user_id, department):
        try:
            user = {"id": user_id, "department": department}

            for _ in range(100):
                filter_obj = engine.to_filter(user, "qdrant")

                # Store first result for this user
                if user_id not in results:
                    results[user_id] = filter_obj
                else:
                    # Verify we always get the same filter for same user
                    assert results[user_id] == filter_obj, f"Filter changed for {user_id}!"

        except Exception as e:
            errors.append((user_id, e))

    # Run multiple users concurrently
    threads = []
    for i in range(10):
        dept = "engineering" if i % 2 == 0 else "sales"
        t = threading.Thread(target=worker, args=(f"user{i}", dept))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0, f"Thread safety errors: {errors}"

    # Each department should have different filter VALUES
    eng_filters = [results[f"user{i}"] for i in range(0, 10, 2)]
    sales_filters = [results[f"user{i}"] for i in range(1, 10, 2)]

    # All engineering users should have equivalent filters (same values, may be different objects)
    # Note: With security fix, each user ID gets its own cache entry, so filters may not be
    # the same object, but should have equivalent values for the same department
    first_eng = eng_filters[0]
    for f in eng_filters:
        assert f == first_eng, "Engineering users should have equivalent filters"

    # All sales users should have equivalent filters
    first_sales = sales_filters[0]
    for f in sales_filters:
        assert f == first_sales, "Sales users should have equivalent filters"

    # But engineering != sales (different department values in the filter)
    assert eng_filters[0] != sales_filters[0], "Different departments should have different filters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
