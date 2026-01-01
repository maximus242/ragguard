#!/usr/bin/env python3
"""
RAGGuard Standalone Demo
========================

This demo runs with ZERO external dependencies beyond ragguard[chromadb].
No Docker, no servers, no configuration files needed.

Run with:
    pip install ragguard[chromadb]
    python standalone_demo.py

This demonstrates:
1. In-memory vector database setup
2. Inline policy definition
3. Permission-aware search in action
4. Different users seeing different documents
"""

import chromadb
from ragguard import ChromaDBSecureRetriever, Policy


def main():
    print("=" * 60)
    print("RAGGuard Demo: Permission-Aware RAG Search")
    print("=" * 60)
    print()

    # ==========================================================================
    # STEP 1: Create an in-memory vector database with sample documents
    # ==========================================================================
    print("[1/4] Setting up in-memory ChromaDB...")

    client = chromadb.Client()  # Ephemeral, in-memory
    collection = client.create_collection(
        name="company_docs",
        metadata={"description": "Company knowledge base"}
    )

    # Add documents with department-based access control metadata
    documents = [
        # Finance department documents
        {
            "id": "fin-001",
            "text": "Q3 2024 Revenue Report: Total revenue reached $50M, up 15% YoY.",
            "metadata": {"department": "finance", "confidential": True, "type": "report"}
        },
        {
            "id": "fin-002",
            "text": "Budget Planning 2025: Allocate $10M for R&D expansion.",
            "metadata": {"department": "finance", "confidential": True, "type": "planning"}
        },
        # Engineering department documents
        {
            "id": "eng-001",
            "text": "Technical Roadmap: Migrate to Kubernetes by Q2 2025.",
            "metadata": {"department": "engineering", "confidential": False, "type": "roadmap"}
        },
        {
            "id": "eng-002",
            "text": "Security Audit Results: All critical issues resolved.",
            "metadata": {"department": "engineering", "confidential": True, "type": "audit"}
        },
        # HR department documents
        {
            "id": "hr-001",
            "text": "Employee Benefits Update: New dental plan available.",
            "metadata": {"department": "hr", "confidential": False, "type": "policy"}
        },
        {
            "id": "hr-002",
            "text": "Salary Bands 2025: Engineering L5 range $180K-$220K.",
            "metadata": {"department": "hr", "confidential": True, "type": "compensation"}
        },
        # Public documents (accessible to everyone)
        {
            "id": "pub-001",
            "text": "Company Blog: We just launched our new product!",
            "metadata": {"department": "public", "confidential": False, "type": "blog"}
        },
        {
            "id": "pub-002",
            "text": "Press Release: Partnership with TechCorp announced.",
            "metadata": {"department": "public", "confidential": False, "type": "press"}
        },
    ]

    collection.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[d["metadata"] for d in documents]
    )
    print(f"   Added {len(documents)} documents to the database.")
    print()

    # ==========================================================================
    # STEP 2: Define access control policy
    # ==========================================================================
    print("[2/4] Defining access control policy...")

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            # Rule 1: Users can access their own department's documents
            {
                "name": "same-department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            # Rule 2: Non-confidential documents are public
            {
                "name": "public-documents",
                "match": {"confidential": False},
                "allow": {"everyone": True}
            },
            # Rule 3: Admins can access everything
            {
                "name": "admin-access",
                "allow": {"roles": ["admin"]}
            }
        ],
        "default": "deny"  # Deny by default if no rules match
    })

    print("   Policy rules:")
    print("   - Users see their own department's docs")
    print("   - Everyone sees non-confidential docs")
    print("   - Admins see everything")
    print()

    # ==========================================================================
    # STEP 3: Create permission-aware retriever
    # ==========================================================================
    print("[3/4] Creating secure retriever...")

    # Create an embedding function wrapper
    # ChromaDB's DefaultEmbeddingFunction takes a list and returns a list
    # RAGGuard expects: single string -> list of floats
    from chromadb.utils import embedding_functions
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_text(text: str) -> list:
        """Convert a single text string to an embedding vector."""
        result = default_ef([text])  # Pass as list
        return result[0].tolist()    # Return first result as list

    retriever = ChromaDBSecureRetriever(
        collection=collection,
        policy=policy,
        embed_fn=embed_text  # Wrapper that handles single strings
    )
    print("   Retriever ready!")
    print()

    # ==========================================================================
    # STEP 4: Demo different users searching
    # ==========================================================================
    print("[4/4] Running permission-aware searches...")
    print()

    # Define test users
    users = [
        {
            "name": "Alice",
            "context": {"id": "alice", "department": "finance", "roles": ["analyst"]},
            "description": "Finance Analyst"
        },
        {
            "name": "Bob",
            "context": {"id": "bob", "department": "engineering", "roles": ["engineer"]},
            "description": "Software Engineer"
        },
        {
            "name": "Carol",
            "context": {"id": "carol", "department": "hr", "roles": ["admin"]},
            "description": "HR Admin (has admin role)"
        },
        {
            "name": "Dave",
            "context": {"id": "dave", "department": "marketing", "roles": []},
            "description": "Marketing (no special access)"
        }
    ]

    query = "company report planning"
    print(f'Search query: "{query}"')
    print("-" * 60)

    for user in users:
        results = retriever.search(
            query=query,
            user=user["context"],
            limit=10
        )

        print(f"\n{user['name']} ({user['description']}):")
        print(f"   Can see {len(results)} documents:")

        for i, result in enumerate(results, 1):
            # ChromaDB returns results with metadata
            doc_id = result.get("id", "unknown")
            metadata = result.get("metadata", {})
            dept = metadata.get("department", "?")
            conf = "confidential" if metadata.get("confidential") else "public"
            print(f"   {i}. [{doc_id}] ({dept}, {conf})")

    print()
    print("=" * 60)
    print("Demo Complete!")
    print()
    print("Notice how:")
    print("- Alice (Finance) sees finance docs + public docs")
    print("- Bob (Engineering) sees engineering docs + public docs")
    print("- Carol (Admin) sees EVERYTHING (admin role)")
    print("- Dave (Marketing) only sees public docs (no special access)")
    print()
    print("This filtering happens DURING vector search, not after.")
    print("Zero unauthorized documents are ever retrieved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
