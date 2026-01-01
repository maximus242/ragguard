"""
RAGGuard Quickstart

This example shows how to add permission-aware search to an existing
Qdrant setup in under 5 minutes.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import sys
import os

# Add parent directory to path to import ragguard
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragguard import SecureRetriever, Policy

# 1. Setup: Create a Qdrant collection with some test documents
print("=" * 60)
print("RAGGuard Quickstart Demo")
print("=" * 60)
print("\n1. Setting up Qdrant with test documents...")

client = QdrantClient(":memory:")  # In-memory for demo

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Insert test documents with permission-relevant metadata
test_docs = [
    {
        "id": 1,
        "text": "Q3 revenue exceeded expectations by 15%",
        "department": "finance",
        "confidential": True,
    },
    {
        "id": 2,
        "text": "New product roadmap for 2026",
        "department": "product",
        "confidential": False,
    },
    {
        "id": 3,
        "text": "Engineering hiring plan Q1",
        "department": "engineering",
        "confidential": False,
    },
    {
        "id": 4,
        "text": "Executive compensation review",
        "department": "hr",
        "confidential": True,
    },
    {
        "id": 5,
        "text": "Company-wide holiday schedule",
        "visibility": "public",
        "confidential": False,
    },
]

# Fake embeddings for demo (in reality, use a real embedding model)
import random


def fake_embed(text):
    random.seed(hash(text) % 2**32)
    return [random.random() for _ in range(384)]


client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=doc["id"],
            vector=fake_embed(doc["text"]),
            payload=doc,
        )
        for doc in test_docs
    ],
)

print(f"   Inserted {len(test_docs)} test documents")

# 2. Define access policies
print("\n2. Defining access control policies...")

policy = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "public-docs",
            "match": {"visibility": "public"},
            "allow": {"everyone": True},
        },
        {
            "name": "department-docs",
            "match": {"confidential": False},
            "allow": {"conditions": ["user.department == document.department"]},
        },
        {
            "name": "own-department-confidential",
            "match": {"confidential": True},
            "allow": {
                "conditions": ["user.department == document.department"],
                "roles": ["manager", "director"],
            },
        },
        {
            "name": "admin-all-access",
            "allow": {"roles": ["admin"]},
        },
    ],
    "default": "deny",
})

print("   Policy loaded with 4 rules:")
print("   - Public documents: everyone")
print("   - Department docs: same department only")
print("   - Confidential: managers in same department")
print("   - Admin: full access")

# 3. Create secure retriever
print("\n3. Creating permission-aware retriever...")

retriever = SecureRetriever(
    client=client,
    collection="documents",
    policy=policy,
)

print("   Retriever initialized")

# 4. Test different users
print("\n" + "=" * 60)
print("Testing permission-aware search")
print("=" * 60)

# Alice: Engineer, can see engineering docs + public
print("\n4. Alice (engineer) searching for 'hiring plan':")
alice = {
    "id": "alice@company.com",
    "roles": ["engineer"],
    "department": "engineering"
}
results = retriever.search(query=fake_embed("hiring plan"), user=alice, limit=10)
print(f"   Found {len(results)} results:")
for r in results:
    print(f"     - {r.payload['text']}")
print(f"   Expected: Engineering docs + public (not finance/hr)")

# Bob: Finance manager, can see finance docs including confidential
print("\n5. Bob (finance manager) searching for 'revenue':")
bob = {
    "id": "bob@company.com",
    "roles": ["manager"],
    "department": "finance"
}
results = retriever.search(query=fake_embed("revenue"), user=bob, limit=10)
print(f"   Found {len(results)} results:")
for r in results:
    print(f"     - {r.payload['text']}")
print(f"   Expected: Finance docs including confidential")

# Carol: Admin, can see everything
print("\n6. Carol (admin) searching for 'compensation':")
carol = {
    "id": "carol@company.com",
    "roles": ["admin"],
    "department": "it"
}
results = retriever.search(query=fake_embed("compensation"), user=carol, limit=10)
print(f"   Found {len(results)} results:")
for r in results:
    print(f"     - {r.payload['text']}")
print(f"   Expected: All documents (admin has full access)")

# Dave: Intern with no special access, only sees public
print("\n7. Dave (intern) searching for 'schedule':")
dave = {
    "id": "dave@company.com",
    "roles": ["intern"],
    "department": "marketing"
}
results = retriever.search(query=fake_embed("schedule"), user=dave, limit=10)
print(f"   Found {len(results)} results:")
for r in results:
    print(f"     - {r.payload['text']}")
print(f"   Expected: Only public documents")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
print("\nKey takeaway: Each user sees different results for the same")
print("query based on their permissions. The filtering happens DURING")
print("vector search, not after, making it efficient and secure.")
