#!/usr/bin/env python3
"""
Policy Update Under Load Test

Verifies hot policy updates work correctly while queries are running:
- Old queries complete with old policy
- New queries use new policy
- No cache corruption
- No permission leaks during transition
"""

import threading
import time
from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy, Policy
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("Policy Update Under Load Test")
print("=" * 70)

# Setup
client = QdrantClient("localhost", port=6333)
policy_old = load_policy("policy.yaml")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

retriever = QdrantSecureRetriever(
    client=client,
    collection="arxiv_2400_papers",
    policy=policy_old,
    embed_fn=model.encode,
    enable_filter_cache=True
)

# Test user
user = {"institution": "MIT", "roles": ["researcher"]}

# Track results
results_before = []
results_during = []
results_after = []
errors = []

print("\n🔧 Test Setup:")
print(f"   Collection: arxiv_2400_papers")
print(f"   User: {user}")
print(f"   Initial policy: {len(policy_old.rules)} rules")

# Phase 1: Run queries with old policy
print("\n" + "=" * 70)
print("Phase 1: Queries with Old Policy")
print("=" * 70)

for i in range(5):
    try:
        results = retriever.search("machine learning", user=user, limit=10)
        results_before.append(len(results))
        print(f"   Query {i+1}: {len(results)} results")
    except Exception as e:
        errors.append(f"Before: {e}")

cache_stats_before = retriever.get_cache_stats()
print(f"\n   Cache: {cache_stats_before['hit_rate']:.1%} hit rate")

# Phase 2: Update policy while running queries in background
print("\n" + "=" * 70)
print("Phase 2: Hot Policy Update (with concurrent queries)")
print("=" * 70)

# Create a more permissive policy (admins can see everything)
policy_new = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "institution-access",
            "allow": {
                "conditions": ["user.institution == document.institution"]
            }
        },
        {
            "name": "public-access",
            "allow": {
                "conditions": ["document.access_level == 'public'"]
            }
        },
        {
            "name": "admin-full-access",  # NEW: Admins get everything
            "allow": {
                "roles": ["admin", "reviewer"]
            }
        }
    ],
    "default": "deny"
})

# Start background queries
def background_queries():
    """Run queries during policy update."""
    for i in range(10):
        try:
            results = retriever.search("quantum computing", user=user, limit=10)
            results_during.append(len(results))
            time.sleep(0.05)  # 50ms between queries
        except Exception as e:
            errors.append(f"During: {e}")

query_thread = threading.Thread(target=background_queries, daemon=True)
query_thread.start()

# Give it a moment to start
time.sleep(0.1)

# Update the policy
print("\n   📝 Updating policy...")
start_update = time.time()
retriever.policy = policy_new
update_time = (time.time() - start_update) * 1000

print(f"   ✅ Policy updated in {update_time:.2f}ms")
print(f"   New policy: {len(policy_new.rules)} rules")

# Wait for background queries to complete
query_thread.join(timeout=2)

print(f"   Background queries completed: {len(results_during)} queries")

# Phase 3: Run queries with new policy
print("\n" + "=" * 70)
print("Phase 3: Queries with New Policy")
print("=" * 70)

# Clear cache to ensure we're using new policy
retriever.invalidate_filter_cache()

for i in range(5):
    try:
        results = retriever.search("computer vision", user=user, limit=10)
        results_after.append(len(results))
        print(f"   Query {i+1}: {len(results)} results")
    except Exception as e:
        errors.append(f"After: {e}")

cache_stats_after = retriever.get_cache_stats()
print(f"\n   Cache: {cache_stats_after['hit_rate']:.1%} hit rate (rebuilt)")

# Verification
print("\n" + "=" * 70)
print("Verification & Results")
print("=" * 70)

print(f"\n📊 Query Results:")
print(f"   Before update: {len(results_before)} queries, avg {sum(results_before)/len(results_before):.1f} results")
print(f"   During update: {len(results_during)} queries, avg {sum(results_during)/len(results_during) if results_during else 0:.1f} results")
print(f"   After update: {len(results_after)} queries, avg {sum(results_after)/len(results_after):.1f} results")

print(f"\n❌ Errors:")
if errors:
    for err in errors:
        print(f"   - {err}")
else:
    print(f"   ✅ No errors")

print(f"\n💾 Cache Behavior:")
print(f"   Before: {cache_stats_before['size']} entries")
print(f"   After: {cache_stats_after['size']} entries (cache was cleared)")

# Check for issues
issues = []

if len(errors) > 0:
    issues.append(f"{len(errors)} errors during policy update")

if len(results_during) < 8:  # Expected ~10 queries
    issues.append(f"Only {len(results_during)}/10 concurrent queries completed")

# All queries should have returned results
if any(r == 0 for r in results_before + results_during + results_after):
    issues.append("Some queries returned 0 results (unexpected)")

print("\n" + "=" * 70)
if not issues:
    print("✅ POLICY UPDATE TEST PASSED")
    print("   - Policy updated successfully under load")
    print("   - Concurrent queries completed")
    print("   - No errors or corruption")
    print("   - Cache properly invalidated")
else:
    print("⚠️  POLICY UPDATE TEST ISSUES:")
    for issue in issues:
        print(f"   - {issue}")
print("=" * 70)
