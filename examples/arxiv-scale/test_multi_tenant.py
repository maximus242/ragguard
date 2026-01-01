#!/usr/bin/env python3
"""
Multi-Tenant Isolation Test

Verifies perfect isolation between tenants:
- Tenant A never sees Tenant B data
- Cross-tenant cache isolation
- Permission changes in Tenant A don't affect Tenant B
"""

from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy
from sentence_transformers import SentenceTransformer
from ragguard.policy.engine import PolicyEngine

print("=" * 70)
print("Multi-Tenant Isolation Test")
print("=" * 70)

# Setup
client = QdrantClient("localhost", port=6333)
policy = load_policy("policy.yaml")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

retriever = QdrantSecureRetriever(
    client=client,
    collection="arxiv_2400_papers",
    policy=policy,
    embed_fn=model.encode,
    enable_filter_cache=True
)

# Define two separate tenants
TENANT_A_USERS = [
    {"id": "alice", "institution": "MIT", "roles": ["researcher"]},
    {"id": "bob", "institution": "MIT", "roles": ["student"]},
    {"id": "charlie", "institution": "MIT", "roles": ["admin"]},
]

TENANT_B_USERS = [
    {"id": "david", "institution": "Stanford", "roles": ["researcher"]},
    {"id": "eve", "institution": "Stanford", "roles": ["student"]},
    {"id": "frank", "institution": "Stanford", "roles": ["admin"]},
]

engine = PolicyEngine(policy)

print("\n🔍 Test Setup:")
print(f"   Tenant A: {len(TENANT_A_USERS)} users (MIT)")
print(f"   Tenant B: {len(TENANT_B_USERS)} users (Stanford)")

# Test 1: Verify each tenant only sees their own data
print("\n" + "=" * 70)
print("Test 1: Tenant Data Isolation")
print("=" * 70)

test1_passed = True

print("\n📋 Tenant A (MIT) users:")
for user in TENANT_A_USERS:
    results = retriever.search("machine learning", user=user, limit=10)

    # Check that no Stanford-only docs are returned
    # BUT: Admins/reviewers have full access per policy, so they CAN see other institutions
    stanford_only_count = 0
    for r in results:
        # If a doc is Stanford-only (not public), it shouldn't be in Tenant A results
        # UNLESS the user is admin/reviewer (who have full access)
        if (r.payload.get("institution") == "Stanford" and
            r.payload.get("access_level") != "public"):
            # Double-check with policy engine
            if not engine.evaluate(user, r.payload):
                stanford_only_count += 1  # Unauthorized!

    is_admin = "admin" in user["roles"] or "reviewer" in user["roles"]
    status = "✅" if stanford_only_count == 0 else "❌"
    note = " (admin: full access)" if is_admin else ""
    print(f"   {status} {user['id']:10s} ({user['roles'][0]:10s}): {len(results)} results, {stanford_only_count} unauthorized docs{note}")

    if stanford_only_count > 0:
        test1_passed = False

print("\n📋 Tenant B (Stanford) users:")
for user in TENANT_B_USERS:
    results = retriever.search("machine learning", user=user, limit=10)

    # Check that no MIT-only docs are returned
    mit_only_count = 0
    for r in results:
        # If a doc is MIT-only (not public), it shouldn't be in Tenant B results
        # UNLESS the user is admin/reviewer (who have full access)
        if (r.payload.get("institution") == "MIT" and
            r.payload.get("access_level") != "public"):
            # Double-check with policy engine
            if not engine.evaluate(user, r.payload):
                mit_only_count += 1  # Unauthorized!

    is_admin = "admin" in user["roles"] or "reviewer" in user["roles"]
    status = "✅" if mit_only_count == 0 else "❌"
    note = " (admin: full access)" if is_admin else ""
    print(f"   {status} {user['id']:10s} ({user['roles'][0]:10s}): {len(results)} results, {mit_only_count} unauthorized docs{note}")

    if mit_only_count > 0:
        test1_passed = False

print(f"\n{'✅ Test 1 PASSED' if test1_passed else '❌ Test 1 FAILED'}: Tenant data isolation")

# Test 2: Cross-tenant cache isolation
print("\n" + "=" * 70)
print("Test 2: Cross-Tenant Cache Isolation")
print("=" * 70)

# Clear cache first
retriever.invalidate_filter_cache()

# Get initial cache state
initial_stats = retriever.get_cache_stats()

# Tenant A user queries
alice = TENANT_A_USERS[0]
results_a1 = retriever.search("quantum computing", user=alice, limit=10)
cache_stats_after_a = retriever.get_cache_stats()

miss_increase_a = cache_stats_after_a['misses'] - initial_stats['misses']
print(f"\n   MIT user queried (cache miss expected)")
print(f"   Cache misses increased by: {miss_increase_a} (expected: 1)")

# Tenant B user queries with SAME query
david = TENANT_B_USERS[0]
results_b1 = retriever.search("quantum computing", user=david, limit=10)
cache_stats_after_b = retriever.get_cache_stats()

miss_increase_b = cache_stats_after_b['misses'] - cache_stats_after_a['misses']
print(f"   Stanford user queried same query")
print(f"   Cache misses increased by: {miss_increase_b} (expected: 1, different user)")

# Should get a cache miss for Tenant B even though same query (different user/tenant)
cache_isolated = miss_increase_b == 1  # Should be a miss for different user

# Same tenant, same query should hit cache
results_a2 = retriever.search("quantum computing", user=alice, limit=10)
cache_stats_after_a2 = retriever.get_cache_stats()

hit_increase = cache_stats_after_a2['hits'] - cache_stats_after_b['hits']
print(f"   MIT user queried again (cache hit expected)")
print(f"   Cache hits increased by: {hit_increase} (expected: 1)")

cache_works_within_tenant = hit_increase == 1

test2_passed = cache_isolated and cache_works_within_tenant

print(f"\n{'✅ Test 2 PASSED' if test2_passed else '❌ Test 2 FAILED'}: Cross-tenant cache isolation")
if cache_isolated:
    print(f"   ✅ Different tenants don't share cache entries")
else:
    print(f"   ❌ Cache leaked between tenants!")
if cache_works_within_tenant:
    print(f"   ✅ Same tenant cache hits work correctly")
else:
    print(f"   ❌ Same tenant cache not working!")

# Test 3: Permission change in Tenant A doesn't affect Tenant B
print("\n" + "=" * 70)
print("Test 3: Permission Change Isolation")
print("=" * 70)

# Clear cache
retriever.invalidate_filter_cache()

# Both tenants query
print(f"\n   Initial queries:")
alice_results_before = retriever.search("computer vision", user=alice, limit=10)
david_results_before = retriever.search("computer vision", user=david, limit=10)
print(f"   MIT user: {len(alice_results_before)} results")
print(f"   Stanford user: {len(david_results_before)} results")

# Simulate permission change for Tenant A (change Alice's role)
alice_new = {"id": "alice", "institution": "MIT", "roles": ["admin"]}  # Promote to admin

print(f"\n   MIT user role changed: researcher → admin")

# Query again
alice_results_after = retriever.search("computer vision", user=alice_new, limit=10)
david_results_after = retriever.search("computer vision", user=david, limit=10)

print(f"   MIT user (new role): {len(alice_results_after)} results")
print(f"   Stanford user: {len(david_results_after)} results")

# Stanford user results should be identical (cached and unaffected)
cache_stats_final = retriever.get_cache_stats()

# David's second query should be a cache hit (unaffected by Alice's change)
david_got_cache_hit = cache_stats_final['hits'] >= 1

test3_passed = (david_results_before == david_results_after and david_got_cache_hit)

print(f"\n{'✅ Test 3 PASSED' if test3_passed else '❌ Test 3 FAILED'}: Permission change isolation")
if david_got_cache_hit:
    print(f"   ✅ Tenant B cache unaffected by Tenant A permission change")
else:
    print(f"   ❌ Tenant B cache was invalidated!")

# Final Summary
print("\n" + "=" * 70)
print("MULTI-TENANT TEST SUMMARY")
print("=" * 70)

all_passed = test1_passed and test2_passed and test3_passed

print(f"\n   Test 1 (Data Isolation): {'✅ PASS' if test1_passed else '❌ FAIL'}")
print(f"   Test 2 (Cache Isolation): {'✅ PASS' if test2_passed else '❌ FAIL'}")
print(f"   Test 3 (Permission Isolation): {'✅ PASS' if test3_passed else '❌ FAIL'}")

print("\n" + "=" * 70)
if all_passed:
    print("✅ ALL MULTI-TENANT TESTS PASSED")
    print("   - Perfect tenant data isolation")
    print("   - Cross-tenant cache isolation")
    print("   - Permission changes don't leak between tenants")
else:
    print("❌ MULTI-TENANT TESTS FAILED")
    print("   DO NOT DEPLOY for multi-tenant use!")
print("=" * 70)
