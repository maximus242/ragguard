#!/usr/bin/env python3
"""
Setup test data for Vercel example.
Creates a collection with sample documents for testing.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import random

# Connect to local Qdrant
client = QdrantClient(url="http://localhost:6333")
collection_name = "vercel_test"

# Delete collection if exists
try:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")
except:
    pass

# Create collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print(f"Created collection: {collection_name}")

# Sample documents with permissions
documents = [
    {
        "id": 1,
        "text": "Machine learning basics and introduction to neural networks",
        "department": "engineering",
        "visibility": "public",
        "title": "ML Introduction"
    },
    {
        "id": 2,
        "text": "Advanced deep learning techniques for computer vision",
        "department": "engineering",
        "visibility": "internal",
        "title": "Deep Learning Guide"
    },
    {
        "id": 3,
        "text": "Company financial report Q4 2024",
        "department": "finance",
        "visibility": "private",
        "title": "Q4 Financial Report"
    },
    {
        "id": 4,
        "text": "HR policies and employee benefits handbook",
        "department": "hr",
        "visibility": "internal",
        "title": "HR Handbook"
    },
    {
        "id": 5,
        "text": "Public product documentation and API reference",
        "department": "engineering",
        "visibility": "public",
        "title": "API Docs"
    },
    {
        "id": 6,
        "text": "Sales strategy and customer acquisition plan",
        "department": "sales",
        "visibility": "private",
        "title": "Sales Strategy"
    },
    {
        "id": 7,
        "text": "Engineering best practices and code review guidelines",
        "department": "engineering",
        "visibility": "internal",
        "title": "Engineering Guidelines"
    },
    {
        "id": 8,
        "text": "Marketing campaign results and analytics",
        "department": "marketing",
        "visibility": "internal",
        "title": "Marketing Analytics"
    }
]

# Insert documents with random vectors (in real use, use actual embeddings)
points = []
for doc in documents:
    vector = [random.random() for _ in range(384)]
    points.append(
        PointStruct(
            id=doc["id"],
            vector=vector,
            payload=doc
        )
    )

client.upsert(collection_name=collection_name, points=points)
print(f"\nInserted {len(points)} documents")

# Print summary
print("\nTest data created:")
print(f"  Collection: {collection_name}")
print(f"  Documents: {len(points)}")
print("\nSample documents:")
for doc in documents[:3]:
    print(f"  - [{doc['department']}] {doc['title']} ({doc['visibility']})")

print("\n✅ Ready to test! Try these user contexts:")
print("\n1. Public access:")
print('   {"id": "guest", "role": "guest", "department": "none"}')
print("   → Should see 2 public documents")

print("\n2. Engineering employee:")
print('   {"id": "eng-1", "role": "employee", "department": "engineering"}')
print("   → Should see public + engineering documents")

print("\n3. Admin:")
print('   {"id": "admin", "role": "admin", "department": "engineering"}')
print("   → Should see all documents")
