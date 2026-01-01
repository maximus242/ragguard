#!/usr/bin/env python3
"""
Test the RAGGuard search functionality.
This simulates what the Vercel API endpoint does.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ragguard import QdrantSecureRetriever, load_policy, RetryConfig
from qdrant_client import QdrantClient
import json

# Initialize client
client = QdrantClient(url="http://localhost:6333")
collection = "vercel_test"

# Load policy
policy = load_policy("policy.yaml")

# Configure retry for serverless
retry_config = RetryConfig(
    max_retries=2,
    initial_delay=0.05,
    max_delay=2.0
)

# Create retriever
retriever = QdrantSecureRetriever(
    client=client,
    collection=collection,
    policy=policy,
    retry_config=retry_config,
    enable_filter_cache=True
)

# Test scenarios
test_cases = [
    {
        "name": "Guest User (should see only public docs)",
        "user": {"id": "guest", "role": "guest", "department": "none"},
        "expected_min": 2,
        "expected_max": 2
    },
    {
        "name": "Engineering Employee (should see public + engineering)",
        "user": {"id": "eng-1", "role": "employee", "department": "engineering"},
        "expected_min": 4,
        "expected_max": 5
    },
    {
        "name": "Finance Employee (should see public + finance)",
        "user": {"id": "fin-1", "role": "employee", "department": "finance"},
        "expected_min": 3,
        "expected_max": 3
    },
    {
        "name": "Admin/Engineering (sees public + engineering)",
        "user": {"id": "admin", "role": "admin", "department": "engineering"},
        "expected_min": 4,
        "expected_max": 4
    }
]

print("=" * 60)
print("RAGGuard Vercel Example - Search Tests")
print("=" * 60)

# Create a query vector (random since we're using random embeddings)
import random
query_vector = [random.random() for _ in range(384)]

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   User: {json.dumps(test['user'])}")

    try:
        # Execute search
        results = retriever.search(
            query=query_vector,
            user=test['user'],
            limit=20  # High limit to get all accessible docs
        )

        count = len(results)
        expected = test['expected_min']

        # Check if count matches expectations
        if test['expected_min'] <= count <= test['expected_max']:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"   Results: {count} documents {status}")

        # Show document titles
        if count > 0:
            print(f"   Documents:")
            for r in results[:5]:  # Show first 5
                title = r.payload.get('title', 'Untitled')
                dept = r.payload.get('department', 'unknown')
                vis = r.payload.get('visibility', 'unknown')
                print(f"     - [{dept}] {title} ({vis})")
            if count > 5:
                print(f"     ... and {count - 5} more")

        # Detailed check
        if test['expected_min'] <= count <= test['expected_max']:
            print(f"   ✅ Expected {test['expected_min']}-{test['expected_max']}, got {count}")
        else:
            print(f"   ❌ Expected {test['expected_min']}-{test['expected_max']}, got {count}")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)

# Test with actual query (if we had embeddings)
print("\n🔍 Simulating actual search query:")
print("   Query: 'machine learning'")
print("   User: Engineering employee")

# Use the same random vector for consistency
results = retriever.search(
    query=query_vector,
    user={"id": "eng-1", "role": "employee", "department": "engineering"},
    limit=5
)

print(f"\n   Found {len(results)} results:")
for i, r in enumerate(results, 1):
    title = r.payload.get('title', 'Untitled')
    score = r.score
    text = r.payload.get('text', '')[:60] + '...'
    print(f"   {i}. [{score:.4f}] {title}")
    print(f"      {text}")

print("\n✅ All tests completed!")
print("\nNote: Using random embeddings for testing.")
print("In production, use actual embedding function for semantic search.")
