"""
RAGGuard + LangChain Integration Example

Shows how to use RAGGuard with LangChain for permission-aware RAG.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import random

# Import from ragguard
from ragguard.policy import Policy
from ragguard.integrations.langchain import LangChainSecureRetriever

print("=" * 70)
print("RAGGuard + LangChain Integration Demo")
print("=" * 70)

# 1. Setup Qdrant with test documents
print("\n1. Setting up Qdrant with test documents...")
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="company_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Fake embedding function for demo
def fake_embed(text):
    random.seed(hash(text) % 2**32)
    return [random.random() for _ in range(384)]

# Test documents with metadata
docs = [
    {
        "id": 1,
        "text": "Q4 financial results show 20% revenue growth",
        "department": "finance",
        "confidential": True,
    },
    {
        "id": 2,
        "text": "New product launch scheduled for next quarter",
        "department": "product",
        "confidential": False,
    },
    {
        "id": 3,
        "text": "Engineering team scaling to 50 people",
        "department": "engineering",
        "confidential": False,
    },
    {
        "id": 4,
        "text": "Company all-hands meeting on Friday at 2pm",
        "visibility": "public",
        "confidential": False,
    },
]

client.upsert(
    collection_name="company_docs",
    points=[
        PointStruct(id=doc["id"], vector=fake_embed(doc["text"]), payload=doc)
        for doc in docs
    ],
)
print(f"   Loaded {len(docs)} documents")

# 2. Define access policy
print("\n2. Setting up access control policy...")
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
            "name": "confidential-manager",
            "match": {"confidential": True},
            "allow": {
                "roles": ["manager"],
                "conditions": ["user.department == document.department"],
            },
        },
        {
            "name": "admin-all",
            "allow": {"roles": ["admin"]},
        },
    ],
    "default": "deny",
})
print("   Policy configured with role and department-based rules")

# 3. Create LangChain-compatible retriever
print("\n3. Creating LangChain-compatible retriever...")
retriever = LangChainSecureRetriever(
    qdrant_client=client,
    collection="company_docs",
    policy=policy,
    embedding_function=fake_embed,
)
print("   ✓ LangChain retriever ready")

# 4. Test with different users
print("\n" + "=" * 70)
print("Testing Permission-Aware Retrieval")
print("=" * 70)

# Engineer - can see engineering docs + public
print("\n4a. Alice (engineer) searching for 'team scaling':")
alice = {"id": "alice@company.com", "roles": ["engineer"], "department": "engineering"}
docs_alice = retriever.get_relevant_documents("team scaling", user=alice, k=10)
print(f"   Found {len(docs_alice)} documents:")
for doc in docs_alice:
    print(f"     - {doc.page_content}")
print(f"   ✓ Expected: Engineering docs + public")

# Finance manager - can see finance docs including confidential
print("\n4b. Bob (finance manager) searching for 'revenue':")
bob = {"id": "bob@company.com", "roles": ["manager"], "department": "finance"}
docs_bob = retriever.get_relevant_documents("revenue", user=bob, k=10)
print(f"   Found {len(docs_bob)} documents:")
for doc in docs_bob:
    print(f"     - {doc.page_content}")
print(f"   ✓ Expected: Finance docs including confidential")

# Admin - can see everything
print("\n4c. Carol (admin) searching for 'company':")
carol = {"id": "carol@company.com", "roles": ["admin"], "department": "it"}
docs_carol = retriever.get_relevant_documents("company", user=carol, k=10)
print(f"   Found {len(docs_carol)} documents:")
for doc in docs_carol:
    print(f"     - {doc.page_content}")
print(f"   ✓ Expected: All documents (admin access)")

# Intern - only public
print("\n4d. Dave (intern) searching for 'meeting':")
dave = {"id": "dave@company.com", "roles": ["intern"], "department": "marketing"}
docs_dave = retriever.get_relevant_documents("meeting", user=dave, k=10)
print(f"   Found {len(docs_dave)} documents:")
for doc in docs_dave:
    print(f"     - {doc.page_content}")
print(f"   ✓ Expected: Only public documents")

# 5. Alternative: Set user context on retriever
print("\n" + "=" * 70)
print("Alternative Usage: Set User Context")
print("=" * 70)

print("\n5. Setting user context on retriever instance...")
retriever.set_user(alice)
docs = retriever.get_relevant_documents("product launch")
print(f"   Alice sees {len(docs)} documents about 'product launch'")

retriever.set_user(bob)
docs = retriever.get_relevant_documents("product launch")
print(f"   Bob sees {len(docs)} documents about 'product launch'")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\nKey Points:")
print("  ✓ LangChain-compatible retriever")
print("  ✓ Permission-aware search")
print("  ✓ Easy to integrate into existing LangChain apps")
print("  ✓ Two ways to pass user context:")
print("    - retriever.get_relevant_documents(query, user={...})")
print("    - retriever.set_user({...}) then retrieve")
print("\nNext steps:")
print("  - Use in LangChain RetrievalQA chains")
print("  - Use in ConversationalRetrievalChain")
print("  - Use with LangChain agents")
