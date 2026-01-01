#!/usr/bin/env python3
"""
Real Database Integration Tests

Tests RAGGuard against actual vector databases running in Docker.

Requirements:
    1. Docker and Docker Compose installed
    2. Run: docker-compose up -d
    3. Wait for health checks to pass
    4. Run: pytest tests/integration/test_real_databases.py

This validates that:
- Native filtering actually works with real databases
- Filter generation produces valid syntax
- Results are correctly filtered
- Cross-backend consistency (same policy = same results)
"""

import os
import sys
import time

import pytest

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ragguard import Policy

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def test_policy():
    """Create a test policy for integration tests."""
    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-match",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "shared-with",
                "allow": {
                    "conditions": ["user.id in document.shared_with"]
                }
            }
        ],
        "default": "deny"
    })


@pytest.fixture(scope="module")
def test_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Engineering document about ML",
            "department": "engineering",
            "shared_with": ["alice", "bob"],
            "vector": [0.1] * 384
        },
        {
            "id": "doc2",
            "text": "Sales document about Q4 targets",
            "department": "sales",
            "shared_with": ["charlie"],
            "vector": [0.2] * 384
        },
        {
            "id": "doc3",
            "text": "Engineering document about infrastructure",
            "department": "engineering",
            "shared_with": ["alice"],
            "vector": [0.15] * 384
        },
        {
            "id": "doc4",
            "text": "HR document about benefits",
            "department": "hr",
            "shared_with": [],
            "vector": [0.3] * 384
        }
    ]


# ============================================================================
# Qdrant Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.qdrant
def test_qdrant_integration(test_policy, test_documents):
    """Test RAGGuard with real Qdrant database."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        from ragguard import QdrantSecureRetriever
    except ImportError:
        pytest.skip("qdrant-client not installed")

    # Connect to Qdrant
    client = QdrantClient("localhost", port=6333)

    # Wait for Qdrant to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            client.get_collections()
            break
        except Exception as e:
            if i == max_retries - 1:
                pytest.skip(f"Qdrant not available: {e}")
            time.sleep(1)

    collection_name = "test_ragguard_integration"

    try:
        # Create collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        # Insert test documents
        points = [
            PointStruct(
                id=i,
                vector=doc["vector"],
                payload={
                    "text": doc["text"],
                    "department": doc["department"],
                    "shared_with": doc["shared_with"]
                }
            )
            for i, doc in enumerate(test_documents, 1)
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Create secure retriever
        retriever = QdrantSecureRetriever(
            client=client,
            collection=collection_name,
            policy=test_policy
        )

        # Test 1: User from engineering department
        alice = {"id": "alice", "department": "engineering"}
        results = retriever.search(
            query=[0.12] * 384,  # Close to engineering docs
            user=alice,
            limit=10
        )

        # Alice should see engineering docs (doc1, doc3)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        departments = [r.payload["department"] for r in results]
        assert all(d == "engineering" for d in departments), "All results should be from engineering"

        # Test 2: User from sales department
        charlie = {"id": "charlie", "department": "sales"}
        results = retriever.search(
            query=[0.2] * 384,  # Close to sales doc
            user=charlie,
            limit=10
        )

        # Charlie should see sales doc (doc2)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0].payload["department"] == "sales"

        # Test 3: User with no access
        dave = {"id": "dave", "department": "hr"}
        results = retriever.search(
            query=[0.3] * 384,  # Close to HR doc
            user=dave,
            limit=10
        )

        # Dave should see HR doc (doc4) but it has no shared_with, so no access via that rule
        # Only has access via department match
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0].payload["department"] == "hr"

        print("✅ Qdrant integration test passed")

    finally:
        # Cleanup
        try:
            client.delete_collection(collection_name)
        except:
            pass


# ============================================================================
# ChromaDB Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.chromadb
def test_chromadb_integration(test_policy, test_documents):
    """Test RAGGuard with real ChromaDB database."""
    try:
        import chromadb

        from ragguard import ChromaDBSecureRetriever, Policy
    except ImportError:
        pytest.skip("chromadb not installed")

    # Connect to ChromaDB
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()  # Test connection
    except Exception as e:
        pytest.skip(f"ChromaDB not available: {e}")

    collection_name = "test_ragguard_integration"

    # ChromaDB doesn't support list values in metadata, so use a simpler policy
    # that only checks department (which is a string)
    chromadb_policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-match",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })

    try:
        # Delete collection if exists
        try:
            client.delete_collection(collection_name)
        except:
            pass

        # Create collection
        collection = client.create_collection(collection_name)

        # Insert test documents
        # Note: ChromaDB doesn't support list values in metadata
        collection.add(
            ids=[doc["id"] for doc in test_documents],
            embeddings=[doc["vector"] for doc in test_documents],
            metadatas=[
                {
                    "text": doc["text"],
                    "department": doc["department"]
                }
                for doc in test_documents
            ]
        )

        # Create secure retriever
        retriever = ChromaDBSecureRetriever(
            collection=collection,
            policy=chromadb_policy
        )

        # Test 1: User from engineering department
        alice = {"id": "alice", "department": "engineering"}
        results = retriever.search(
            query=[0.12] * 384,
            user=alice,
            limit=10
        )

        # Alice should see engineering docs
        assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
        departments = [r.get("metadata", {}).get("department") for r in results]
        assert all(d == "engineering" for d in departments if d), "All results should be from engineering"

        # Test 2: User from sales department
        charlie = {"id": "charlie", "department": "sales"}
        results = retriever.search(
            query=[0.2] * 384,
            user=charlie,
            limit=10
        )

        # Charlie should see sales doc
        assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
        assert any(r.get("metadata", {}).get("department") == "sales" for r in results)

        print("✅ ChromaDB integration test passed")

    finally:
        # Cleanup
        try:
            client.delete_collection(collection_name)
        except:
            pass


# ============================================================================
# pgvector Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.pgvector
def test_pgvector_integration(test_policy, test_documents):
    """Test RAGGuard with real PostgreSQL + pgvector."""
    try:
        import psycopg2
        from psycopg2.extras import Json

        from ragguard.filters.builder import to_pgvector_filter
    except ImportError:
        pytest.skip("psycopg2 not installed")

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="ragguard_test",
            user="ragguard",
            password="ragguard_test"
        )
        conn.autocommit = True
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")

    cursor = conn.cursor()

    try:
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create table
        cursor.execute("""
            DROP TABLE IF EXISTS test_documents
        """)
        cursor.execute("""
            CREATE TABLE test_documents (
                id TEXT PRIMARY KEY,
                embedding vector(384),
                metadata JSONB
            )
        """)

        # Insert test documents
        for doc in test_documents:
            cursor.execute(
                """
                INSERT INTO test_documents (id, embedding, metadata)
                VALUES (%s, %s, %s)
                """,
                (
                    doc["id"],
                    doc["vector"],
                    Json({
                        "text": doc["text"],
                        "department": doc["department"],
                        "shared_with": doc["shared_with"]
                    })
                )
            )

        # Test filter generation
        alice = {"id": "alice", "department": "engineering"}
        where_clause, params = to_pgvector_filter(test_policy, alice)

        # Query with filter
        # Note: where_clause already includes "WHERE" keyword
        query = f"""
            SELECT id, metadata
            FROM test_documents
            {where_clause}
            ORDER BY embedding <-> %s::vector
            LIMIT 10
        """

        query_vector = [0.12] * 384
        cursor.execute(query, params + [query_vector])
        results = cursor.fetchall()

        # Alice should see engineering docs
        assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
        for _, metadata in results:
            assert metadata["department"] == "engineering", "All results should be from engineering"

        print("✅ pgvector integration test passed")

    finally:
        # Cleanup
        try:
            cursor.execute("DROP TABLE IF EXISTS test_documents")
            cursor.close()
            conn.close()
        except:
            pass


# ============================================================================
# Cross-Backend Consistency Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.consistency
def test_cross_backend_consistency(test_policy):
    """
    Test that all backends return semantically equivalent results.

    This is a smoke test - we verify that filters are generated without errors
    and that the general structure is correct.
    """
    from ragguard.filters.builder import (
        to_chromadb_filter,
        to_pgvector_filter,
        to_pinecone_filter,
        to_qdrant_filter,
        to_weaviate_filter,
    )

    user = {"id": "alice", "department": "engineering"}

    # Generate filters for all backends
    filters = {
        "qdrant": to_qdrant_filter(test_policy, user),
        "pgvector": to_pgvector_filter(test_policy, user),
        "weaviate": to_weaviate_filter(test_policy, user),
        "pinecone": to_pinecone_filter(test_policy, user),
        "chromadb": to_chromadb_filter(test_policy, user)
    }

    # Verify all filters were generated
    for backend, filter_obj in filters.items():
        assert filter_obj is not None, f"{backend} filter should not be None"

        # pgvector returns (sql, params)
        if backend == "pgvector":
            sql, params = filter_obj
            assert len(sql) > 0, f"{backend} SQL should not be empty"
            assert "engineering" in params, f"{backend} params should contain 'engineering'"
        else:
            # Other backends return objects/dicts
            assert filter_obj, f"{backend} filter should not be empty"

    print("✅ Cross-backend consistency test passed")


# ============================================================================
# Helper: Run All Integration Tests
# ============================================================================

if __name__ == "__main__":
    """
    Run integration tests manually.

    Prerequisites:
        docker-compose up -d
        pip install qdrant-client chromadb psycopg2-binary

    Run:
        python tests/integration/test_real_databases.py
    """
    print("=" * 80)
    print("RAGGuard Real Database Integration Tests")
    print("=" * 80)
    print()

    # Create test data
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-match",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })

    documents = [
        {
            "id": "doc1",
            "text": "Engineering doc",
            "department": "engineering",
            "shared_with": ["alice"],
            "vector": [0.1] * 384
        },
        {
            "id": "doc2",
            "text": "Sales doc",
            "department": "sales",
            "shared_with": ["bob"],
            "vector": [0.2] * 384
        }
    ]

    # Run tests
    print("Testing Qdrant...")
    try:
        test_qdrant_integration(policy, documents)
    except Exception as e:
        print(f"❌ Qdrant test failed: {e}")

    print("\nTesting ChromaDB...")
    try:
        test_chromadb_integration(policy, documents)
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")

    print("\nTesting pgvector...")
    try:
        test_pgvector_integration(policy, documents)
    except Exception as e:
        print(f"❌ pgvector test failed: {e}")

    print("\nTesting cross-backend consistency...")
    try:
        test_cross_backend_consistency(policy)
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")

    print("\n" + "=" * 80)
    print("✅ Integration tests complete!")
    print("=" * 80)
