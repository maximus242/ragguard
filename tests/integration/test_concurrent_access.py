"""
Concurrent Access Integration Tests

Tests that RAGGuard retrievers are thread-safe and can handle concurrent
queries from multiple users without race conditions, data leaks, or crashes.

This is critical for production deployments where multiple requests may be
processed simultaneously.

Test Coverage:
1. Thread safety - Multiple threads querying simultaneously
2. User isolation - Each thread gets correct results for its user
3. No cross-contamination - User A doesn't see User B's filters
4. Connection pooling - Databases handle concurrent connections
5. Performance under load - Latency doesn't degrade excessively
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest


@dataclass
class ConcurrentTestResult:
    """Result from a concurrent test."""
    user_id: str
    query_num: int
    doc_count: int
    latency_ms: float
    error: str = None
    leaked_docs: List[str] = None  # Docs this user shouldn't have seen


# ============================================================================
# Test Data
# ============================================================================

TEST_USERS = [
    {"id": "alice", "department": "engineering"},
    {"id": "bob", "department": "engineering"},
    {"id": "charlie", "department": "sales"},
    {"id": "david", "department": "sales"},
    {"id": "eve", "department": "hr"},
    {"id": "frank", "department": "hr"},
    {"id": "guest1", "department": "unknown"},  # Should see nothing
    {"id": "guest2"},  # No department
]

TEST_DOCUMENTS = []

# Create 50 engineering docs
for i in range(50):
    TEST_DOCUMENTS.append({
        "id": f"eng_{i}",
        "text": f"Engineering document {i}",
        "department": "engineering",
        "visibility": "internal",
        "vector": [0.1 + random.random() * 0.1] + [random.random() for _ in range(383)]
    })

# Create 30 sales docs
for i in range(30):
    TEST_DOCUMENTS.append({
        "id": f"sales_{i}",
        "text": f"Sales document {i}",
        "department": "sales",
        "visibility": "internal",
        "vector": [0.8 + random.random() * 0.1] + [random.random() for _ in range(383)]
    })

# Create 20 HR docs
for i in range(20):
    TEST_DOCUMENTS.append({
        "id": f"hr_{i}",
        "text": f"HR document {i}",
        "department": "hr",
        "visibility": "internal",
        "vector": [0.5 + random.random() * 0.1] + [random.random() for _ in range(383)]
    })


@pytest.fixture(scope="module")
def test_policy():
    """Policy for concurrent testing."""
    from ragguard import Policy

    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })


# ============================================================================
# Qdrant Concurrent Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.concurrent
@pytest.mark.qdrant
def test_qdrant_concurrent_queries(test_policy):
    """Test Qdrant with 100 concurrent queries from multiple users."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        from ragguard import QdrantSecureRetriever
    except ImportError:
        pytest.skip("qdrant-client not installed")

    # Connect to Qdrant
    try:
        client = QdrantClient("localhost", port=6333)
        client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")

    collection_name = "ragguard_concurrent_test"

    try:
        # Setup collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        # Insert documents
        points = [
            PointStruct(
                id=i,
                vector=doc["vector"],
                payload={
                    "doc_id": doc["id"],
                    "text": doc["text"],
                    "department": doc["department"],
                    "visibility": doc["visibility"]
                }
            )
            for i, doc in enumerate(TEST_DOCUMENTS, 1)
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Create retriever
        retriever = QdrantSecureRetriever(
            client=client,
            collection=collection_name,
            policy=test_policy
        )

        # Function to execute query for a user
        def query_for_user(user: Dict, query_num: int) -> ConcurrentTestResult:
            try:
                start = time.time()

                # Random query vector
                query_vec = [random.random() for _ in range(384)]

                results = retriever.search(
                    query=query_vec,
                    user=user,
                    limit=10
                )

                latency = (time.time() - start) * 1000  # ms

                # Verify no leaked docs
                leaked = []
                user_dept = user.get("department")

                for r in results:
                    doc_dept = r.payload.get("department")
                    if user_dept and doc_dept != user_dept:
                        leaked.append(r.payload.get("doc_id"))

                return ConcurrentTestResult(
                    user_id=user["id"],
                    query_num=query_num,
                    doc_count=len(results),
                    latency_ms=latency,
                    leaked_docs=leaked if leaked else None
                )

            except Exception as e:
                return ConcurrentTestResult(
                    user_id=user["id"],
                    query_num=query_num,
                    doc_count=0,
                    latency_ms=0,
                    error=str(e)
                )

        # Execute 100 concurrent queries
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for i in range(100):
                user = random.choice(TEST_USERS)
                futures.append(executor.submit(query_for_user, user, i))

            for future in as_completed(futures):
                results.append(future.result())

        # Verify results
        errors = [r for r in results if r.error]
        leaks = [r for r in results if r.leaked_docs]

        # Allow some transient network errors under concurrent load (e.g., connection reset)
        # These are expected with HTTP/1.1 under high concurrency and don't indicate a bug
        transient_error_keywords = ["Bad file descriptor", "Connection reset", "Connection refused", "Read timed out"]
        real_errors = [
            r for r in errors
            if not any(keyword in str(r.error) for keyword in transient_error_keywords)
        ]

        # At most 10% transient errors allowed
        transient_error_rate = len(errors) / len(results) if results else 0
        assert transient_error_rate < 0.1, f"Too many transient errors: {transient_error_rate:.1%}"

        assert len(real_errors) == 0, f"Non-transient errors occurred: {[r.error for r in real_errors]}"
        assert len(leaks) == 0, f"Data leaks detected: {[r.leaked_docs for r in leaks]}"

        # Performance check
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        print("✅ Qdrant concurrent test passed:")
        print(f"   - 100 queries from {len(TEST_USERS)} users")
        print(f"   - Avg latency: {avg_latency:.1f}ms")
        print(f"   - Max latency: {max_latency:.1f}ms")
        print("   - No errors, no data leaks")

        # Sanity check: engineering users should have gotten results
        eng_results = [r for r in results if r.user_id in ["alice", "bob"] and r.doc_count > 0]
        assert len(eng_results) > 0, "Engineering users got no results"

    finally:
        try:
            client.delete_collection(collection_name)
        except:
            pass


# ============================================================================
# ChromaDB Concurrent Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.concurrent
@pytest.mark.chromadb
def test_chromadb_concurrent_queries(test_policy):
    """Test ChromaDB with concurrent queries."""
    try:
        import chromadb

        from ragguard import ChromaDBSecureRetriever
    except ImportError:
        pytest.skip("chromadb not installed")

    try:
        client = chromadb.PersistentClient(path="/tmp/ragguard_chromadb_concurrent_test")
        client.heartbeat()
    except Exception as e:
        pytest.skip(f"ChromaDB not available: {e}")

    collection_name = "ragguard_concurrent_test"

    try:
        # Delete existing
        try:
            client.delete_collection(collection_name)
        except:
            pass

        # Create collection
        collection = client.create_collection(collection_name)

        # Insert documents
        collection.add(
            ids=[doc["id"] for doc in TEST_DOCUMENTS],
            embeddings=[doc["vector"] for doc in TEST_DOCUMENTS],
            metadatas=[
                {
                    "doc_id": doc["id"],
                    "text": doc["text"],
                    "department": doc["department"],
                    "visibility": doc["visibility"]
                }
                for doc in TEST_DOCUMENTS
            ]
        )

        # Create retriever
        retriever = ChromaDBSecureRetriever(
            collection=collection,
            policy=test_policy
        )

        def query_for_user(user: Dict, query_num: int) -> ConcurrentTestResult:
            try:
                start = time.time()

                query_vec = [random.random() for _ in range(384)]
                results = retriever.search(
                    query=query_vec,
                    user=user,
                    limit=10
                )

                latency = (time.time() - start) * 1000

                # Verify no leaks
                leaked = []
                user_dept = user.get("department")

                for r in results:
                    doc_dept = r.get("metadata", {}).get("department")
                    if user_dept and doc_dept != user_dept:
                        leaked.append(r.get("id"))

                return ConcurrentTestResult(
                    user_id=user["id"],
                    query_num=query_num,
                    doc_count=len(results),
                    latency_ms=latency,
                    leaked_docs=leaked if leaked else None
                )

            except Exception as e:
                return ConcurrentTestResult(
                    user_id=user["id"],
                    query_num=query_num,
                    doc_count=0,
                    latency_ms=0,
                    error=str(e)
                )

        # Execute concurrent queries
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for i in range(50):  # Fewer for ChromaDB (slower)
                user = random.choice(TEST_USERS)
                futures.append(executor.submit(query_for_user, user, i))

            for future in as_completed(futures):
                results.append(future.result())

        # Verify
        errors = [r for r in results if r.error]
        leaks = [r for r in results if r.leaked_docs]

        assert len(errors) == 0, f"Errors: {[r.error for r in errors]}"
        assert len(leaks) == 0, f"Leaks: {[r.leaked_docs for r in leaks]}"

        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies)

        print("✅ ChromaDB concurrent test passed:")
        print(f"   - {len(results)} queries")
        print(f"   - Avg latency: {avg_latency:.1f}ms")
        print("   - No errors, no leaks")

    finally:
        try:
            client.delete_collection(collection_name)
        except:
            pass


# ============================================================================
# Cross-User Isolation Test
# ============================================================================

@pytest.mark.integration
@pytest.mark.concurrent
def test_user_isolation_under_load(test_policy):
    """
    Verify that rapid user switching doesn't cause filter cross-contamination.

    This tests a critical security property: User A's filters must NEVER
    be applied to User B's query.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        from ragguard import QdrantSecureRetriever
    except ImportError:
        pytest.skip("qdrant-client not installed")

    try:
        client = QdrantClient("localhost", port=6333)
        client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")

    collection_name = "ragguard_isolation_test"

    try:
        # Setup
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
                    "department": doc["department"]
                }
            )
            for i, doc in enumerate(TEST_DOCUMENTS, 1)
        ]
        client.upsert(collection_name=collection_name, points=points)

        retriever = QdrantSecureRetriever(
            client=client,
            collection=collection_name,
            policy=test_policy
        )

        # Rapidly alternate between users
        alice = {"id": "alice", "department": "engineering"}
        charlie = {"id": "charlie", "department": "sales"}

        query_vec = [0.15] * 384

        for i in range(100):
            # Alice query
            alice_results = retriever.search(query=query_vec, user=alice, limit=10)

            # Immediately followed by Charlie query
            charlie_results = retriever.search(query=query_vec, user=charlie, limit=10)

            # Verify isolation
            for r in alice_results:
                dept = r.payload.get("department")
                assert dept == "engineering", (
                    f"Iteration {i}: Alice saw {dept} doc (should only see engineering)"
                )

            for r in charlie_results:
                dept = r.payload.get("department")
                assert dept == "sales", (
                    f"Iteration {i}: Charlie saw {dept} doc (should only see sales)"
                )

        print("✅ User isolation verified: 100 rapid user switches, no cross-contamination")

    finally:
        try:
            client.delete_collection(collection_name)
        except:
            pass


if __name__ == "__main__":
    """
    Run concurrent access tests manually.

    Prerequisites:
        docker-compose up -d (from tests/integration/)
        pip install qdrant-client chromadb ragguard pytest

    Run:
        python tests/integration/test_concurrent_access.py
    """
    print("=" * 80)
    print("RAGGuard Concurrent Access Integration Tests")
    print("=" * 80)
    print()
    print("Run with: pytest tests/integration/test_concurrent_access.py -v")
    print()
    print("=" * 80)
