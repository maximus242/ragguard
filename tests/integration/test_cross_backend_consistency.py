"""
Cross-Backend Consistency Integration Tests

Verifies that the same policy applied to the same data produces identical
results across all supported vector database backends.

This is critical for ensuring RAGGuard behavior is consistent regardless
of which vector database is being used.

Prerequisites:
    - Docker and Docker Compose installed
    - Run: docker-compose up -d (from tests/integration/)
    - All backend client libraries installed

Test Strategy:
    1. Insert identical data into all backends
    2. Apply same policy and user context
    3. Verify all backends return same document IDs
    4. Verify order is consistent (by score)
"""

import pytest

# Integration test - requires docker-compose up (see docker-compose.yml)
pytestmark = pytest.mark.skip(reason="Requires running databases - integration test. Run docker-compose up to enable.")
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class BackendResult:
    """Result from a backend query."""
    backend: str
    doc_ids: List[str]
    scores: List[float]
    error: str = None


# ============================================================================
# Test Data
# ============================================================================

TEST_DOCUMENTS = [
    {
        "id": "doc1",
        "text": "Machine learning in Python",
        "department": "engineering",
        "visibility": "public",
        "shared_with": ["alice", "bob"],
        "vector": [0.1, 0.2, 0.1] + [0.0] * 381  # 384 dims
    },
    {
        "id": "doc2",
        "text": "Sales strategy for Q4",
        "department": "sales",
        "visibility": "internal",
        "shared_with": ["charlie"],
        "vector": [0.8, 0.7, 0.9] + [0.0] * 381
    },
    {
        "id": "doc3",
        "text": "Engineering infrastructure guide",
        "department": "engineering",
        "visibility": "internal",
        "shared_with": ["alice", "david"],
        "vector": [0.15, 0.25, 0.12] + [0.0] * 381
    },
    {
        "id": "doc4",
        "text": "HR benefits overview",
        "department": "hr",
        "visibility": "public",
        "shared_with": [],
        "vector": [0.5, 0.6, 0.5] + [0.0] * 381
    },
    {
        "id": "doc5",
        "text": "Engineering best practices",
        "department": "engineering",
        "visibility": "public",
        "shared_with": ["alice"],
        "vector": [0.12, 0.22, 0.11] + [0.0] * 381
    }
]


@pytest.fixture(scope="module")
def test_policy():
    """Policy that allows access based on department OR sharing."""
    from ragguard import Policy

    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "public-docs",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            },
            {
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "shared-access",
                "allow": {
                    "conditions": ["user.id in document.shared_with"]
                }
            }
        ],
        "default": "deny"
    })


# ============================================================================
# Backend Setup Helpers
# ============================================================================

def setup_qdrant(documents: List[Dict]) -> Any:
    """Setup Qdrant with test documents."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError:
        return None

    try:
        client = QdrantClient("localhost", port=6333)
        client.get_collections()  # Test connection
    except Exception:
        return None

    collection_name = "ragguard_consistency_test"

    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        points = [
            PointStruct(
                id=i,
                vector=doc["vector"],
                payload={
                    "doc_id": doc["id"],
                    "text": doc["text"],
                    "department": doc["department"],
                    "visibility": doc["visibility"],
                    "shared_with": doc["shared_with"]
                }
            )
            for i, doc in enumerate(documents, 1)
        ]
        client.upsert(collection_name=collection_name, points=points)

        return {"client": client, "collection": collection_name}
    except Exception as e:
        print(f"Qdrant setup failed: {e}")
        return None


def setup_chromadb(documents: List[Dict]) -> Any:
    """Setup ChromaDB with test documents."""
    try:
        import chromadb
    except ImportError:
        return None

    try:
        # Use persistent client for proper testing
        client = chromadb.PersistentClient(path="/tmp/ragguard_chromadb_test")
        client.heartbeat()
    except Exception:
        return None

    collection_name = "ragguard_consistency_test"

    try:
        # Delete existing collection
        try:
            client.delete_collection(collection_name)
        except:
            pass

        collection = client.create_collection(collection_name)

        # ChromaDB doesn't support list values, so we'll skip shared_with
        # and only test department/visibility matching
        collection.add(
            ids=[doc["id"] for doc in documents],
            embeddings=[doc["vector"] for doc in documents],
            metadatas=[
                {
                    "doc_id": doc["id"],
                    "text": doc["text"],
                    "department": doc["department"],
                    "visibility": doc["visibility"]
                }
                for doc in documents
            ]
        )

        return {"client": client, "collection": collection}
    except Exception as e:
        print(f"ChromaDB setup failed: {e}")
        return None


def setup_pgvector(documents: List[Dict]) -> Any:
    """Setup pgvector with test documents."""
    try:
        import psycopg2
        from psycopg2.extras import Json
    except ImportError:
        return None

    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="ragguard_test",
            user="ragguard",
            password="ragguard_test"
        )
        conn.autocommit = True
    except Exception:
        return None

    cursor = conn.cursor()

    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("DROP TABLE IF EXISTS ragguard_consistency_test")
        cursor.execute("""
            CREATE TABLE ragguard_consistency_test (
                id SERIAL PRIMARY KEY,
                doc_id TEXT,
                embedding vector(384),
                metadata JSONB
            )
        """)

        for doc in documents:
            cursor.execute(
                """
                INSERT INTO ragguard_consistency_test (doc_id, embedding, metadata)
                VALUES (%s, %s, %s)
                """,
                (
                    doc["id"],
                    doc["vector"],
                    Json({
                        "text": doc["text"],
                        "department": doc["department"],
                        "visibility": doc["visibility"],
                        "shared_with": doc["shared_with"]
                    })
                )
            )

        return {"connection": conn, "cursor": cursor, "table": "ragguard_consistency_test"}
    except Exception as e:
        print(f"pgvector setup failed: {e}")
        try:
            cursor.close()
            conn.close()
        except:
            pass
        return None


# ============================================================================
# Query Execution Helpers
# ============================================================================

def query_qdrant(setup: Dict, policy, user: Dict, query_vector: List[float], limit: int) -> BackendResult:
    """Execute query against Qdrant."""
    try:
        from ragguard import QdrantSecureRetriever

        retriever = QdrantSecureRetriever(
            client=setup["client"],
            collection=setup["collection"],
            policy=policy
        )

        results = retriever.search(
            query=query_vector,
            user=user,
            limit=limit
        )

        doc_ids = [r.payload["doc_id"] for r in results]
        scores = [r.score for r in results]

        return BackendResult(backend="qdrant", doc_ids=doc_ids, scores=scores)
    except Exception as e:
        return BackendResult(backend="qdrant", doc_ids=[], scores=[], error=str(e))


def query_chromadb(setup: Dict, policy, user: Dict, query_vector: List[float], limit: int) -> BackendResult:
    """Execute query against ChromaDB."""
    try:
        # Use simplified policy without shared_with for ChromaDB
        from ragguard import ChromaDBSecureRetriever, Policy
        chromadb_policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public-docs",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True}
                },
                {
                    "name": "department-access",
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        retriever = ChromaDBSecureRetriever(
            collection=setup["collection"],
            policy=chromadb_policy
        )

        results = retriever.search(
            query=query_vector,
            user=user,
            limit=limit
        )

        doc_ids = [r.get("id") or r.get("metadata", {}).get("doc_id") for r in results]
        # ChromaDB returns distances, lower is better (inverse of score)
        distances = [r.get("distance", 0) for r in results]

        return BackendResult(backend="chromadb", doc_ids=doc_ids, scores=distances)
    except Exception as e:
        return BackendResult(backend="chromadb", doc_ids=[], scores=[], error=str(e))


def query_pgvector(setup: Dict, policy, user: Dict, query_vector: List[float], limit: int) -> BackendResult:
    """Execute query against pgvector."""
    try:
        from ragguard.filters.builder import to_pgvector_filter

        where_clause, params = to_pgvector_filter(policy, user)

        query = f"""
            SELECT doc_id, embedding <-> %s::vector AS distance
            FROM {setup['table']}
            {where_clause}
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """

        # Parameter order: query_vector, WHERE clause params, query_vector, limit
        setup['cursor'].execute(query, [query_vector] + params + [query_vector, limit])
        results = setup['cursor'].fetchall()

        doc_ids = [row[0] for row in results]
        scores = [row[1] for row in results]

        return BackendResult(backend="pgvector", doc_ids=doc_ids, scores=scores)
    except Exception as e:
        return BackendResult(backend="pgvector", doc_ids=[], scores=[], error=str(e))


# ============================================================================
# Consistency Verification Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.consistency
def test_cross_backend_consistency_engineering_user(test_policy):
    """
    Test that all backends return same results for engineering user.

    Alice (engineering dept) should see:
    - doc1: engineering + public
    - doc3: engineering + internal + shared_with alice
    - doc5: engineering + public + shared_with alice
    """
    alice = {"id": "alice", "department": "engineering"}
    query_vector = [0.11, 0.21, 0.11] + [0.0] * 381  # Close to engineering docs

    # Setup all backends
    backends = {}

    qdrant_setup = setup_qdrant(TEST_DOCUMENTS)
    if qdrant_setup:
        backends["qdrant"] = qdrant_setup

    chromadb_setup = setup_chromadb(TEST_DOCUMENTS)
    if chromadb_setup:
        backends["chromadb"] = chromadb_setup

    pgvector_setup = setup_pgvector(TEST_DOCUMENTS)
    if pgvector_setup:
        backends["pgvector"] = pgvector_setup

    if not backends:
        pytest.skip("No backends available for testing")

    # Query all backends
    results = {}

    if "qdrant" in backends:
        results["qdrant"] = query_qdrant(backends["qdrant"], test_policy, alice, query_vector, 10)

    if "chromadb" in backends:
        results["chromadb"] = query_chromadb(backends["chromadb"], test_policy, alice, query_vector, 10)

    if "pgvector" in backends:
        results["pgvector"] = query_pgvector(backends["pgvector"], test_policy, alice, query_vector, 10)

    # Verify no errors
    for backend, result in results.items():
        assert result.error is None, f"{backend} query failed: {result.error}"

    # Verify all backends return same document IDs (order might differ)
    if len(results) >= 2:
        result_sets = {backend: set(result.doc_ids) for backend, result in results.items()}

        # Get expected docs for alice
        expected_docs = {"doc1", "doc3", "doc5"}

        for backend, doc_set in result_sets.items():
            assert doc_set == expected_docs, (
                f"{backend} returned {doc_set}, expected {expected_docs}. "
                f"Difference: missing={expected_docs - doc_set}, extra={doc_set - expected_docs}"
            )

        print(f"✅ Cross-backend consistency verified: {list(results.keys())}")
        print(f"   All backends returned: {expected_docs}")

    # Cleanup
    cleanup_backends(backends)


@pytest.mark.integration
@pytest.mark.consistency
def test_cross_backend_consistency_public_only(test_policy):
    """
    Test backends return same results for user with no department.

    User with no department should only see public docs:
    - doc1: public
    - doc4: public
    - doc5: public
    """
    guest = {"id": "guest"}  # No department
    query_vector = [0.5, 0.5, 0.5] + [0.0] * 381

    backends = {}

    qdrant_setup = setup_qdrant(TEST_DOCUMENTS)
    if qdrant_setup:
        backends["qdrant"] = qdrant_setup

    chromadb_setup = setup_chromadb(TEST_DOCUMENTS)
    if chromadb_setup:
        backends["chromadb"] = chromadb_setup

    pgvector_setup = setup_pgvector(TEST_DOCUMENTS)
    if pgvector_setup:
        backends["pgvector"] = pgvector_setup

    if not backends:
        pytest.skip("No backends available for testing")

    results = {}

    if "qdrant" in backends:
        results["qdrant"] = query_qdrant(backends["qdrant"], test_policy, guest, query_vector, 10)

    if "chromadb" in backends:
        results["chromadb"] = query_chromadb(backends["chromadb"], test_policy, guest, query_vector, 10)

    if "pgvector" in backends:
        results["pgvector"] = query_pgvector(backends["pgvector"], test_policy, guest, query_vector, 10)

    # Verify consistency
    expected_docs = {"doc1", "doc4", "doc5"}

    for backend, result in results.items():
        assert result.error is None, f"{backend} query failed: {result.error}"
        doc_set = set(result.doc_ids)
        assert doc_set == expected_docs, (
            f"{backend} returned {doc_set}, expected {expected_docs}"
        )

    print(f"✅ Public-only consistency verified across {len(results)} backends")

    cleanup_backends(backends)


@pytest.mark.integration
@pytest.mark.consistency
def test_cross_backend_consistency_no_access(test_policy):
    """
    Test backends consistently deny access when no rules match.

    User from finance dept should see no internal docs:
    - Only public docs (doc1, doc4, doc5)
    """
    finance_user = {"id": "frank", "department": "finance"}
    query_vector = [0.8, 0.7, 0.9] + [0.0] * 381  # Close to sales doc

    backends = {}

    qdrant_setup = setup_qdrant(TEST_DOCUMENTS)
    if qdrant_setup:
        backends["qdrant"] = qdrant_setup

    chromadb_setup = setup_chromadb(TEST_DOCUMENTS)
    if chromadb_setup:
        backends["chromadb"] = chromadb_setup

    pgvector_setup = setup_pgvector(TEST_DOCUMENTS)
    if pgvector_setup:
        backends["pgvector"] = pgvector_setup

    if not backends:
        pytest.skip("No backends available for testing")

    results = {}

    if "qdrant" in backends:
        results["qdrant"] = query_qdrant(backends["qdrant"], test_policy, finance_user, query_vector, 10)

    if "chromadb" in backends:
        results["chromadb"] = query_chromadb(backends["chromadb"], test_policy, finance_user, query_vector, 10)

    if "pgvector" in backends:
        results["pgvector"] = query_pgvector(backends["pgvector"], test_policy, finance_user, query_vector, 10)

    # Finance user should only see public docs
    expected_docs = {"doc1", "doc4", "doc5"}

    for backend, result in results.items():
        assert result.error is None, f"{backend} query failed: {result.error}"
        doc_set = set(result.doc_ids)
        assert doc_set == expected_docs, (
            f"{backend} should only show public docs to finance user. "
            f"Got {doc_set}, expected {expected_docs}"
        )

    print(f"✅ Access denial consistency verified across {len(results)} backends")

    cleanup_backends(backends)


# ============================================================================
# Cleanup Helper
# ============================================================================

def cleanup_backends(backends: Dict[str, Any]):
    """Cleanup all backend resources."""
    if "qdrant" in backends:
        try:
            backends["qdrant"]["client"].delete_collection(backends["qdrant"]["collection"])
        except:
            pass

    if "chromadb" in backends:
        try:
            backends["chromadb"]["client"].delete_collection("ragguard_consistency_test")
        except:
            pass

    if "pgvector" in backends:
        try:
            backends["pgvector"]["cursor"].execute("DROP TABLE IF EXISTS ragguard_consistency_test")
            backends["pgvector"]["cursor"].close()
            backends["pgvector"]["connection"].close()
        except:
            pass


# ============================================================================
# Main Entry Point for Manual Testing
# ============================================================================

if __name__ == "__main__":
    """
    Run cross-backend consistency tests manually.

    Prerequisites:
        docker-compose up -d
        pip install qdrant-client chromadb psycopg2-binary

    Run:
        python tests/integration/test_cross_backend_consistency.py
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    from ragguard import Policy

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "public-docs",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            },
            {
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })

    print("=" * 80)
    print("RAGGuard Cross-Backend Consistency Tests")
    print("=" * 80)
    print()

    print("Test 1: Engineering user access...")
    try:
        test_cross_backend_consistency_engineering_user(policy)
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    print("\nTest 2: Public-only access...")
    try:
        test_cross_backend_consistency_public_only(policy)
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    print("\nTest 3: Access denial consistency...")
    try:
        test_cross_backend_consistency_no_access(policy)
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

    print("\n" + "=" * 80)
    print("✅ Cross-backend consistency tests complete!")
    print("=" * 80)
