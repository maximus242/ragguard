"""
FAISS Post-Filtering Integration Tests

FAISS doesn't support native filtering like other vector databases.
RAGGuard implements post-fetch filtering for FAISS, which means:
1. Fetch more results than requested (to account for filtering)
2. Apply permission filtering in Python
3. Return only authorized results

This test suite verifies that post-filtering:
- Returns correct number of results
- Doesn't leak unauthorized documents
- Handles edge cases (all filtered out, partial filtering, etc.)
- Performs reasonably well
"""

from typing import Any, Dict, List

import numpy as np
import pytest

# Skip all tests if faiss is not installed
faiss = pytest.importorskip("faiss")


@pytest.fixture(scope="module")
def test_policy():
    """Policy for testing FAISS post-filtering."""
    from ragguard import Policy

    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "public-docs",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            }
        ],
        "default": "deny"
    })


@pytest.fixture(scope="module")
def faiss_index_with_documents():
    """Create FAISS index with test documents."""
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss not installed")

    from ragguard import FAISSSecureRetriever

    # Create test documents
    documents = []
    vectors = []

    # Engineering docs (10 docs)
    for i in range(10):
        vec = np.random.randn(384).astype('float32')
        vec[0] = 0.1  # Make them similar
        vectors.append(vec)
        documents.append({
            "id": f"eng_doc_{i}",
            "department": "engineering",
            "visibility": "internal",
            "text": f"Engineering document {i}"
        })

    # Sales docs (10 docs)
    for i in range(10):
        vec = np.random.randn(384).astype('float32')
        vec[0] = 0.8  # Different cluster
        vectors.append(vec)
        documents.append({
            "id": f"sales_doc_{i}",
            "department": "sales",
            "visibility": "internal",
            "text": f"Sales document {i}"
        })

    # Public docs (5 docs)
    for i in range(5):
        vec = np.random.randn(384).astype('float32')
        vec[0] = 0.5
        vectors.append(vec)
        documents.append({
            "id": f"public_doc_{i}",
            "department": "marketing",
            "visibility": "public",
            "text": f"Public document {i}"
        })

    # Create FAISS index
    dimension = 384
    index = faiss.IndexFlatL2(dimension)

    vectors_array = np.array(vectors).astype('float32')
    index.add(vectors_array)

    return {
        "index": index,
        "documents": documents,
        "vectors": vectors_array
    }


@pytest.mark.integration
@pytest.mark.faiss
def test_faiss_post_filtering_returns_correct_count(test_policy, faiss_index_with_documents):
    """Test that FAISS post-filtering returns requested number of results."""
    from ragguard import FAISSSecureRetriever

    index = faiss_index_with_documents["index"]
    documents = faiss_index_with_documents["documents"]

    retriever = FAISSSecureRetriever(
        index=index,
        metadata=documents,
        policy=test_policy
    )

    # Engineering user should see 10 engineering docs + 5 public docs = 15 total
    user = {"id": "alice", "department": "engineering"}
    query_vector = [0.1] + [0.0] * 383  # Similar to engineering docs

    results = retriever.search(
        query=query_vector,
        user=user,
        limit=10
    )

    # Should return exactly 10 results (as requested)
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    # All results should be authorized
    for result in results:
        metadata = result.get("metadata", {})
        dept = metadata.get("department")
        visibility = metadata.get("visibility")

        # Either from engineering or public
        assert dept == "engineering" or visibility == "public", (
            f"Unauthorized document returned: {metadata}"
        )

    print(f"✅ FAISS returned correct count: {len(results)}")


@pytest.mark.integration
@pytest.mark.faiss
def test_faiss_post_filtering_no_leakage(test_policy, faiss_index_with_documents):
    """Test that FAISS never returns unauthorized documents."""
    from ragguard import FAISSSecureRetriever

    index = faiss_index_with_documents["index"]
    documents = faiss_index_with_documents["documents"]

    retriever = FAISSSecureRetriever(
        index=index,
        metadata=documents,
        policy=test_policy
    )

    # Sales user should NEVER see engineering docs
    user = {"id": "charlie", "department": "sales"}

    # Query for engineering docs (but user is sales)
    query_vector = [0.1] + [0.0] * 383

    results = retriever.search(
        query=query_vector,
        user=user,
        limit=20  # Request more than available authorized docs
    )

    # Verify no engineering docs leaked
    for result in results:
        metadata = result.get("metadata", {})
        dept = metadata.get("department")
        visibility = metadata.get("visibility")

        assert dept != "engineering" or visibility == "public", (
            f"Engineering doc leaked to sales user: {metadata}"
        )

    # Should see sales docs (10) + public docs (5) = 15 max
    assert len(results) <= 15, f"Too many results: {len(results)}"

    print(f"✅ No unauthorized documents leaked: {len(results)} results, all authorized")


@pytest.mark.integration
@pytest.mark.faiss
def test_faiss_post_filtering_all_filtered_out(test_policy, faiss_index_with_documents):
    """Test FAISS when all nearby documents are filtered out."""
    from ragguard import FAISSSecureRetriever

    index = faiss_index_with_documents["index"]
    documents = faiss_index_with_documents["documents"]

    retriever = FAISSSecureRetriever(
        index=index,
        metadata=documents,
        policy=test_policy
    )

    # User from department with no documents
    user = {"id": "dave", "department": "hr"}

    # Query for engineering docs (user is HR, no access)
    query_vector = [0.1] + [0.0] * 383

    results = retriever.search(
        query=query_vector,
        user=user,
        limit=10
    )

    # HR user should only see public docs
    # Verify all returned docs are public
    for result in results:
        metadata = result.get("metadata", {})
        visibility = metadata.get("visibility")
        assert visibility == "public", f"Non-public doc returned to HR user: {metadata}"

    # Should return at most 5 public docs
    assert len(results) <= 5

    print(f"✅ Correctly filtered when most docs unauthorized: {len(results)} public docs")


@pytest.mark.integration
@pytest.mark.faiss
def test_faiss_post_filtering_guest_user(test_policy, faiss_index_with_documents):
    """Test FAISS post-filtering for guest user (public only)."""
    from ragguard import FAISSSecureRetriever

    index = faiss_index_with_documents["index"]
    documents = faiss_index_with_documents["documents"]

    retriever = FAISSSecureRetriever(
        index=index,
        metadata=documents,
        policy=test_policy
    )

    # Guest user with no department
    user = {"id": "guest"}

    results = retriever.search(
        query=[0.5] + [0.0] * 383,  # Query public docs cluster
        user=user,
        limit=10
    )

    # Should only see public docs
    for result in results:
        metadata = result.get("metadata", {})
        assert metadata.get("visibility") == "public", (
            f"Non-public doc shown to guest: {metadata}"
        )

    # Maximum 5 public docs
    assert len(results) <= 5

    print(f"✅ Guest user correctly limited to public docs: {len(results)} results")


@pytest.mark.integration
@pytest.mark.faiss
def test_faiss_post_filtering_performance(test_policy):
    """Test that FAISS post-filtering performs reasonably with large dataset."""
    import time

    import faiss

    from ragguard import FAISSSecureRetriever

    # Create larger dataset (1000 docs)
    documents = []
    vectors = []

    for i in range(1000):
        vec = np.random.randn(384).astype('float32')
        vectors.append(vec)

        # 50% engineering, 30% sales, 20% public
        if i < 500:
            dept = "engineering"
            visibility = "internal"
        elif i < 800:
            dept = "sales"
            visibility = "internal"
        else:
            dept = "marketing"
            visibility = "public"

        documents.append({
            "id": f"doc_{i}",
            "department": dept,
            "visibility": visibility,
            "text": f"Document {i}"
        })

    # Create index
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    vectors_array = np.array(vectors).astype('float32')
    index.add(vectors_array)

    retriever = FAISSSecureRetriever(
        index=index,
        metadata=documents,
        policy=test_policy
    )

    user = {"id": "alice", "department": "engineering"}

    # Measure performance
    start_time = time.time()

    for _ in range(100):  # 100 queries
        results = retriever.search(
            query=np.random.randn(384).tolist(),
            user=user,
            limit=10
        )

    elapsed = time.time() - start_time
    avg_latency = elapsed / 100

    # Should complete 100 queries in under 5 seconds (50ms avg)
    assert elapsed < 5.0, f"Performance issue: {elapsed:.2f}s for 100 queries (avg {avg_latency*1000:.1f}ms)"

    print(f"✅ Performance acceptable: {elapsed:.2f}s for 100 queries (avg {avg_latency*1000:.1f}ms per query)")


@pytest.mark.integration
@pytest.mark.faiss
def test_faiss_handles_empty_index(test_policy):
    """Test FAISS post-filtering with empty index."""
    import faiss

    from ragguard import FAISSSecureRetriever

    # Create empty index
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    documents = []

    retriever = FAISSSecureRetriever(
        index=index,
        metadata=documents,
        policy=test_policy
    )

    user = {"id": "alice", "department": "engineering"}

    results = retriever.search(
        query=[0.0] * 384,
        user=user,
        limit=10
    )

    assert len(results) == 0, "Empty index should return no results"

    print("✅ Empty index handled correctly")


@pytest.mark.integration
@pytest.mark.faiss
def test_faiss_over_fetching_strategy(test_policy, faiss_index_with_documents):
    """
    Test that FAISS over-fetches to account for filtering.

    If user requests 10 docs but 50% are filtered out, FAISS should
    fetch more than 10 initially to return 10 after filtering.
    """
    from ragguard import FAISSSecureRetriever

    index = faiss_index_with_documents["index"]
    documents = faiss_index_with_documents["documents"]

    retriever = FAISSSecureRetriever(
        index=index,
        metadata=documents,
        policy=test_policy
    )

    # User can see 15 out of 25 total docs (engineering + public)
    user = {"id": "alice", "department": "engineering"}

    # Request exactly 10 results
    results = retriever.search(
        query=[0.12] + [0.0] * 383,
        user=user,
        limit=10
    )

    # Should return exactly 10 (or close to it)
    # If it returns significantly fewer, over-fetching isn't working
    assert len(results) >= 8, (
        f"Over-fetching may not be working: only {len(results)} results "
        f"returned when 10 requested and 15 available"
    )

    print(f"✅ Over-fetching strategy working: returned {len(results)} of requested 10")


if __name__ == "__main__":
    """
    Run FAISS post-filtering tests manually.

    Prerequisites:
        pip install faiss-cpu  # or faiss-gpu
        pip install ragguard

    Run:
        python tests/integration/test_faiss_post_filtering.py
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    from ragguard import Policy

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "public-docs",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            }
        ],
        "default": "deny"
    })

    print("=" * 80)
    print("FAISS Post-Filtering Integration Tests")
    print("=" * 80)
    print()

    # Note: Manual testing requires running pytest
    print("Run with: pytest tests/integration/test_faiss_post_filtering.py -v")
    print()
    print("This will verify:")
    print("  ✓ Correct result counts")
    print("  ✓ No unauthorized document leakage")
    print("  ✓ Handling of fully filtered results")
    print("  ✓ Guest user restrictions")
    print("  ✓ Performance with large datasets")
    print("  ✓ Edge cases (empty index, over-fetching)")
    print()
    print("=" * 80)
