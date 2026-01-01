#!/usr/bin/env python3
"""
Debug version to see what filters are being generated.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ragguard import QdrantSecureRetriever, load_policy
from ragguard.filters.builder import to_qdrant_filter
from qdrant_client import QdrantClient
import json

# Initialize
client = QdrantClient(url="http://localhost:6333")
policy = load_policy("policy_simple.yaml")

print("Policy loaded. Rules:")
for i, rule in enumerate(policy.rules):
    print(f"  {i+1}. {rule.name}")
    if hasattr(rule, 'match') and rule.match:
        print(f"     Match: {rule.match}")
    if rule.allow and rule.allow.conditions:
        print(f"     Conditions: {rule.allow.conditions[:60]}...")

# Test users
users = [
    {"id": "guest", "role": "guest", "department": "none"},
    {"id": "eng-1", "role": "employee", "department": "engineering"},
    {"id": "admin", "role": "admin", "department": "engineering"}
]

print("\n" + "=" * 60)
print("Filter Generation Test")
print("=" * 60)

for user in users:
    print(f"\nUser: {json.dumps(user)}")

    # Generate filter
    qdrant_filter = to_qdrant_filter(policy, user)

    print(f"Filter: {qdrant_filter}")

    # Count matches
    results = client.scroll(
        collection_name="vercel_test",
        scroll_filter=qdrant_filter,
        limit=100
    )

    count = len(results[0])
    print(f"Matches: {count} documents")

    if count > 0 and count <= 5:
        for r in results[0]:
            dept = r.payload.get('department', '?')
            vis = r.payload.get('visibility', '?')
            title = r.payload.get('title', '?')
            print(f"  - [{dept}] {title} ({vis})")
