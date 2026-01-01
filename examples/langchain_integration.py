#!/usr/bin/env python3
"""
LangChain Integration Example for RAGGuard

Demonstrates how to add access control to a LangChain RAG application.

This example shows:
1. Setting up a secure vector store with RAGGuard
2. Building a RAG chain with user-specific access control
3. Multi-user scenarios with different access levels
4. Integration with LangChain's RetrievalQA chain

Requirements:
    pip install langchain langchain-community langchain-openai qdrant-client sentence-transformers

Usage:
    python examples/langchain_integration.py

Note: This is a standalone demo that works without external dependencies.
      For production use, install the packages above.
"""

import os
import sys

# Import RAGGuard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Dict, Any

# Try importing LangChain - if not available, use mock classes
try:
    from langchain.chains import RetrievalQA
    from langchain.schema import Document as LangChainDocument
    from langchain_community.llms.fake import FakeListLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.llms.fake import FakeListLLM
        from langchain.chains import RetrievalQA
        from langchain.schema import Document as LangChainDocument
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        # Mock classes for demo
        class LangChainDocument:
            def __init__(self, page_content: str, metadata: dict):
                self.page_content = page_content
                self.metadata = metadata

        class FakeListLLM:
            def __init__(self, responses):
                self.responses = responses
                self.i = 0
            def __call__(self, text):
                response = self.responses[self.i % len(self.responses)]
                self.i += 1
                return response

# Try importing embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        EMBEDDINGS_AVAILABLE = True
    except ImportError:
        EMBEDDINGS_AVAILABLE = False
        # Create a simple mock embedding function
        import random
        class HuggingFaceEmbeddings:
            def __init__(self, model_name: str = ""):
                self.model_name = model_name
            def embed_query(self, text: str) -> list:
                random.seed(hash(text) % 2**32)
                return [random.random() for _ in range(384)]
            def embed_documents(self, texts: list) -> list:
                return [self.embed_query(t) for t in texts]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️  Qdrant not installed. Install with: pip install qdrant-client")
    print("   Running in demo mode with mock data...\n")

from ragguard import Policy, QdrantSecureRetriever


def setup_demo_data(client: QdrantClient, collection_name: str = "company_docs"):
    """Create a demo collection with company documents."""
    print("📚 Setting up demo data...")

    # Create collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Sample company documents with different access levels
    documents = [
        {
            "id": 1,
            "text": "Q4 Financial Report: Revenue increased 25% year-over-year to $10M.",
            "department": "finance",
            "classification": "confidential",
            "shared_with": ["alice@company.com", "bob@company.com"]
        },
        {
            "id": 2,
            "text": "Engineering Roadmap: Launching new AI features in Q1 2025.",
            "department": "engineering",
            "classification": "internal",
            "shared_with": ["alice@company.com", "charlie@company.com"]
        },
        {
            "id": 3,
            "text": "Company Holiday Party: December 20th at the downtown office.",
            "department": "hr",
            "classification": "public",
            "shared_with": []
        },
        {
            "id": 4,
            "text": "Security Incident Response: New protocol for handling data breaches.",
            "department": "security",
            "classification": "confidential",
            "shared_with": ["bob@company.com"]
        },
        {
            "id": 5,
            "text": "Marketing Campaign Results: Email open rate improved to 35%.",
            "department": "marketing",
            "classification": "internal",
            "shared_with": ["alice@company.com", "dave@company.com"]
        },
    ]

    # Embed and upload documents
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    points = []
    for doc in documents:
        vector = embeddings.embed_query(doc["text"])
        points.append(
            PointStruct(
                id=doc["id"],
                vector=vector,
                payload={
                    "text": doc["text"],
                    "department": doc["department"],
                    "classification": doc["classification"],
                    "shared_with": doc["shared_with"]
                }
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"   ✅ Uploaded {len(documents)} documents\n")


def create_access_control_policy() -> Policy:
    """Create a multi-tenant access control policy."""
    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "public-docs",
                "match": {"classification": "public"},
                "allow": {"everyone": True}
            },
            {
                "name": "department-docs",
                "match": {"classification": "internal"},
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "shared-docs",
                "allow": {
                    "conditions": ["user.email in document.shared_with"]
                }
            },
            {
                "name": "admin-access",
                "allow": {"roles": ["admin"]}
            }
        ],
        "default": "deny"
    })


class SecureLangChainRetriever:
    """
    Wrapper that makes RAGGuard compatible with LangChain's retriever interface.

    This allows seamless integration with LangChain chains like RetrievalQA.
    """

    def __init__(self, ragguard_retriever: QdrantSecureRetriever, user: Dict[str, Any]):
        self.ragguard_retriever = ragguard_retriever
        self.user = user

    def get_relevant_documents(self, query: str) -> List[LangChainDocument]:
        """
        Retrieve documents with access control applied.

        This method is called by LangChain chains.
        """
        # Use RAGGuard to get filtered results
        results = self.ragguard_retriever.search(
            query=query,
            user=self.user,
            limit=5
        )

        # Convert to LangChain Document format
        # Results are ScoredPoint objects from Qdrant
        docs = []
        for result in results:
            # Handle both dict-style and ScoredPoint objects
            if hasattr(result, 'payload'):
                # ScoredPoint from Qdrant
                payload = result.payload
                score = result.score
            else:
                # Dict-style result
                payload = result
                score = result.get("score", 0.0)

            docs.append(LangChainDocument(
                page_content=payload.get("text", ""),
                metadata={
                    "department": payload.get("department"),
                    "classification": payload.get("classification"),
                    "score": score
                }
            ))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[LangChainDocument]:
        """Async version for async chains."""
        # For this demo, we just call the sync version
        # In production, you'd implement async search
        return self.get_relevant_documents(query)


def demo_scenario_1_basic_rag():
    """
    Scenario 1: Basic RAG with Access Control

    Shows how Alice (engineering) gets different results than Dave (marketing).
    """
    print("=" * 80)
    print("SCENARIO 1: Basic RAG with Access Control")
    print("=" * 80)
    print()

    # Setup
    client = QdrantClient(":memory:")  # In-memory for demo
    setup_demo_data(client)
    policy = create_access_control_policy()

    # Create RAGGuard retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ragguard_retriever = QdrantSecureRetriever(
        client=client,
        collection="company_docs",
        policy=policy,
        embed_fn=embeddings.embed_query
    )

    # User 1: Alice (Engineering)
    alice = {
        "email": "alice@company.com",
        "department": "engineering",
        "roles": ["employee"]
    }

    print("👤 Alice (Engineering) asks: 'What are the latest updates?'")
    print()

    alice_retriever = SecureLangChainRetriever(ragguard_retriever, alice)
    alice_docs = alice_retriever.get_relevant_documents("What are the latest updates?")

    print(f"   Retrieved {len(alice_docs)} documents:")
    for i, doc in enumerate(alice_docs, 1):
        dept = doc.metadata.get("department", "unknown")
        classification = doc.metadata.get("classification", "unknown")
        print(f"   {i}. [{dept}/{classification}] {doc.page_content[:60]}...")
    print()

    # User 2: Dave (Marketing)
    dave = {
        "email": "dave@company.com",
        "department": "marketing",
        "roles": ["employee"]
    }

    print("👤 Dave (Marketing) asks: 'What are the latest updates?'")
    print()

    dave_retriever = SecureLangChainRetriever(ragguard_retriever, dave)
    dave_docs = dave_retriever.get_relevant_documents("What are the latest updates?")

    print(f"   Retrieved {len(dave_docs)} documents:")
    for i, doc in enumerate(dave_docs, 1):
        dept = doc.metadata.get("department", "unknown")
        classification = doc.metadata.get("classification", "unknown")
        print(f"   {i}. [{dept}/{classification}] {doc.page_content[:60]}...")
    print()

    print("💡 Key Point: Alice sees engineering docs, Dave sees marketing docs!")
    print()


def demo_scenario_2_retrieval_qa_chain():
    """
    Scenario 2: Integration with LangChain RetrievalQA

    Shows how to use RAGGuard with LangChain's RetrievalQA chain.
    """
    print("=" * 80)
    print("SCENARIO 2: RetrievalQA Chain with Access Control")
    print("=" * 80)
    print()

    # Setup
    client = QdrantClient(":memory:")
    setup_demo_data(client)
    policy = create_access_control_policy()

    # Create RAGGuard retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ragguard_retriever = QdrantSecureRetriever(
        client=client,
        collection="company_docs",
        policy=policy,
        embed_fn=embeddings.embed_query
    )

    # Create a fake LLM for demo (in production, use OpenAI or another LLM)
    fake_llm = FakeListLLM(
        responses=[
            "Based on the financial report, revenue increased 25% to $10M in Q4.",
            "I don't have access to financial information.",
        ]
    )

    # User: Bob (Finance, has access to financial docs)
    bob = {
        "email": "bob@company.com",
        "department": "finance",
        "roles": ["employee"]
    }

    print("👤 Bob (Finance) asks about Q4 revenue...")
    print()

    bob_retriever = SecureLangChainRetriever(ragguard_retriever, bob)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=fake_llm,
        chain_type="stuff",
        retriever=bob_retriever
    )

    # Query
    result = qa_chain.invoke({"query": "What was Q4 revenue?"})
    print(f"   Answer: {result['result']}")
    print()

    # User: Charlie (Engineering, NO access to financial docs)
    charlie = {
        "email": "charlie@company.com",
        "department": "engineering",
        "roles": ["employee"]
    }

    print("👤 Charlie (Engineering) asks about Q4 revenue...")
    print()

    charlie_retriever = SecureLangChainRetriever(ragguard_retriever, charlie)

    qa_chain_charlie = RetrievalQA.from_chain_type(
        llm=fake_llm,
        chain_type="stuff",
        retriever=charlie_retriever
    )

    result = qa_chain_charlie.invoke({"query": "What was Q4 revenue?"})
    print(f"   Answer: {result['result']}")
    print()

    print("💡 Key Point: Access control prevents unauthorized data leakage to the LLM!")
    print()


def demo_scenario_3_admin_access():
    """
    Scenario 3: Admin Override

    Shows how admins can access all documents regardless of department.
    """
    print("=" * 80)
    print("SCENARIO 3: Admin Access Override")
    print("=" * 80)
    print()

    # Setup
    client = QdrantClient(":memory:")
    setup_demo_data(client)
    policy = create_access_control_policy()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ragguard_retriever = QdrantSecureRetriever(
        client=client,
        collection="company_docs",
        policy=policy,
        embed_fn=embeddings.embed_query
    )

    # Admin user
    admin = {
        "email": "admin@company.com",
        "department": "it",
        "roles": ["admin"]
    }

    print("👤 Admin asks: 'Show me all documents'")
    print()

    admin_retriever = SecureLangChainRetriever(ragguard_retriever, admin)
    admin_docs = admin_retriever.get_relevant_documents("company updates")

    print(f"   Retrieved {len(admin_docs)} documents (admin sees everything):")
    for i, doc in enumerate(admin_docs, 1):
        dept = doc.metadata.get("department", "unknown")
        classification = doc.metadata.get("classification", "unknown")
        print(f"   {i}. [{dept}/{classification}] {doc.page_content[:60]}...")
    print()

    print("💡 Key Point: Admins can override access controls when needed!")
    print()


def main():
    """Run all demo scenarios."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║  RAGGuard + LangChain Integration Demo                      ║")
    print("║  Secure RAG with User-Specific Access Control               ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n")

    demo_scenario_1_basic_rag()

    try:
        demo_scenario_2_retrieval_qa_chain()
    except Exception as e:
        print(f"\n⚠️  Scenario 2 skipped (requires full LangChain): {type(e).__name__}")
        print("   Install: pip install langchain langchain-community")

    try:
        demo_scenario_3_admin_access()
    except Exception as e:
        print(f"\n⚠️  Scenario 3 skipped: {type(e).__name__}")

    print("=" * 80)
    print("✨ Demo Complete!")
    print("=" * 80)
    print()
    print("📚 Learn More:")
    print("   - Documentation: https://github.com/maximus242/ragguard")
    print("   - Policy Language: See policy.yaml examples")
    print("   - Production Setup: Use OpenAI/Anthropic instead of FakeListLLM")
    print()


if __name__ == "__main__":
    main()
