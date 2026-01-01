#!/usr/bin/env python3
"""
ChromaDB Integration Tests for RAGGuard v0.2.0

Tests the complete end-to-end flow with ChromaDB.

Requirements:
- pip install chromadb
"""

import sys
import numpy as np

print("=" * 80)
print("ChromaDB Integration Tests")
print("=" * 80)

# Check if ChromaDB is available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    print("\n⚠️  chromadb not installed. Install with: pip install chromadb")
    print("\n❌ Exiting with failure code (tests did not run)")
    CHROMADB_AVAILABLE = False
    sys.exit(1)  # Fail if dependency not available

from ragguard import Policy
from ragguard.filters.builder import to_chromadb_filter
from ragguard.policy.engine import PolicyEngine

tests_passed = 0
tests_failed = 0

# Create in-memory ChromaDB client
client = chromadb.Client(Settings(anonymized_telemetry=False))

def setup_database():
    """Create test collection and add documents."""
    global client

    try:
        # Delete existing collection if it exists
        try:
            client.delete_collection("test_documents")
        except:
            pass

        # Create collection
        collection = client.create_collection(
            name="test_documents",
            metadata={"description": "Test documents for RAGGuard"}
        )

        # Insert test data
        test_docs = [
            # AI papers - public
            ("doc1", "Deep Learning Overview", "cs.AI", "published", "public", "research", "alice", ["ai", "ml"]),
            ("doc2", "Neural Networks Intro", "cs.AI", "published", "public", "research", "bob", ["ai", "nn"]),

            # ML papers - public
            ("doc3", "Machine Learning Basics", "cs.LG", "published", "public", "research", "alice", ["ml"]),
            ("doc4", "Supervised Learning", "cs.LG", "published", "public", "research", "charlie", ["ml", "supervised"]),

            # Restricted papers
            ("doc5", "Secret AI Research", "cs.AI", "published", "restricted", "research", "alice", ["ai", "secret"]),
            ("doc6", "Classified ML Model", "cs.LG", "published", "classified", "security", "bob", ["ml", "classified"]),

            # Archived papers
            ("doc7", "Old AI Paper", "cs.AI", "archived", "public", "research", "alice", ["ai", "old"]),
            ("doc8", "Deprecated ML", "cs.LG", "archived", "public", "research", "bob", ["ml", "old"]),

            # Draft papers
            ("doc9", "Draft AI Work", "cs.AI", "draft", "public", "research", "charlie", ["ai", "draft"]),
            ("doc10", "WIP ML Paper", "cs.LG", "draft", "public", "research", "alice", ["ml", "wip"]),

            # Database papers (different category)
            ("doc11", "SQL Optimization", "cs.DB", "published", "public", "engineering", "bob", ["db", "sql"]),
            ("doc12", "NoSQL Systems", "cs.DB", "published", "public", "engineering", "charlie", ["db", "nosql"]),
        ]

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for doc_id, content, category, status, access_level, dept, created_by, tags in test_docs:
            ids.append(doc_id)
            embeddings.append(np.random.rand(384).tolist())
            metadatas.append({
                "category": category,
                "status": status,
                "access_level": access_level,
                "department": dept,
                "created_by": created_by,
                # ChromaDB doesn't support list values in metadata, so we skip tags
            })
            documents.append(content)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        print(f"✅ Database setup complete: {len(test_docs)} documents inserted")
        return collection

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise

collection = setup_database()

def test(name, func):
    """Run a test and track results."""
    global tests_passed, tests_failed
    print(f"\n🔍 {name}")
    try:
        result = func()
        if result:
            print(f"   ✅ PASS")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_basic_filter():
    """Test basic equality filter."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-only",
            "allow": {
                "everyone": True,
                "conditions": ["document.access_level == 'public'"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    # Query with filter
    query_embedding = np.random.rand(384).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where=filter_obj
    )

    # Should get 10 public documents (2 AI + 2 ML + 2 archived + 2 draft + 2 DB)
    num_results = len(results['ids'][0])
    if num_results != 10:
        print(f"      Expected 10 results, got {num_results}")
        return False

    # All should be public
    for metadata in results['metadatas'][0]:
        if metadata['access_level'] != 'public':
            print(f"      Found non-public document: {metadata}")
            return False

    print(f"      Retrieved {num_results} public documents (correct)")
    return True

test("Basic equality filter", test_basic_filter)

def test_negation_operator():
    """Test != operator."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-archived",
            "allow": {
                "everyone": True,
                "conditions": ["document.status != 'archived'"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    query_embedding = np.random.rand(384).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where=filter_obj
    )

    # Should get 10 non-archived (12 total - 2 archived)
    num_results = len(results['ids'][0])
    if num_results != 10:
        print(f"      Expected 10 results, got {num_results}")
        return False

    # None should be archived
    for metadata in results['metadatas'][0]:
        if metadata['status'] == 'archived':
            print(f"      Found archived document: {metadata}")
            return False

    print(f"      Retrieved {num_results} non-archived documents")
    return True

test("Negation operator (!=)", test_negation_operator)

def test_list_literal():
    """Test IN with list literal."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "ai-ml-only",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in ['cs.AI', 'cs.LG']"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    query_embedding = np.random.rand(384).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where=filter_obj
    )

    # Should get 10 AI/ML papers (5 AI + 5 ML)
    num_results = len(results['ids'][0])
    if num_results != 10:
        print(f"      Expected 10 results, got {num_results}")
        return False

    # All should be cs.AI or cs.LG
    for metadata in results['metadatas'][0]:
        if metadata['category'] not in ['cs.AI', 'cs.LG']:
            print(f"      Found wrong category: {metadata['category']}")
            return False

    print(f"      Retrieved {num_results} AI/ML documents")
    return True

test("List literal (IN)", test_list_literal)

def test_not_in_operator():
    """Test NOT IN operator."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-bad-status",
            "allow": {
                "everyone": True,
                "conditions": ["document.status not in ['archived', 'draft']"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    query_embedding = np.random.rand(384).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where=filter_obj
    )

    # Should get 8 published (12 total - 2 archived - 2 draft)
    num_results = len(results['ids'][0])
    if num_results != 8:
        print(f"      Expected 8 results, got {num_results}")
        return False

    # None should be archived or draft
    for metadata in results['metadatas'][0]:
        if metadata['status'] in ['archived', 'draft']:
            print(f"      Found excluded status: {metadata['status']}")
            return False

    print(f"      Retrieved {num_results} published documents")
    return True

test("NOT IN operator", test_not_in_operator)

def test_multiple_conditions():
    """Test multiple conditions with AND logic."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "ai-public-active",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.category in ['cs.AI', 'cs.LG']",
                    "document.access_level != 'restricted'",
                    "document.status not in ['archived', 'draft']"
                ]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    query_embedding = np.random.rand(384).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where=filter_obj
    )

    num_results = len(results['ids'][0])

    # Expected: 5 documents
    expected = 5
    if num_results != expected:
        print(f"      Expected {expected} results, got {num_results}")
        for metadata in results['metadatas'][0]:
            print(f"        - {metadata}")
        return False

    # Verify all match criteria
    for metadata in results['metadatas'][0]:
        if metadata['category'] not in ['cs.AI', 'cs.LG']:
            print(f"      Wrong category: {metadata['category']}")
            return False
        if metadata['access_level'] == 'restricted':
            print(f"      Found restricted: {metadata}")
            return False
        if metadata['status'] in ['archived', 'draft']:
            print(f"      Found excluded status: {metadata['status']}")
            return False

    print(f"      Retrieved {num_results} matching documents")
    return True

test("Multiple conditions (AND logic)", test_multiple_conditions)

def test_filter_object_structure():
    """Test that filter object has correct ChromaDB structure."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "test",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.category in ['cs.AI']",
                    "document.status != 'archived'"
                ]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    # ChromaDB filter should be a dict with $and operator
    if not isinstance(filter_obj, dict):
        print(f"      Filter should be dict, got {type(filter_obj)}")
        return False

    if "$and" not in filter_obj:
        print(f"      Filter should have $and operator, got {filter_obj.keys()}")
        return False

    print(f"      Filter structure correct: {filter_obj}")
    return True

test("Filter object structure", test_filter_object_structure)

def test_large_list():
    """Test performance with large list."""
    large_list = [f"category_{i}" for i in range(100)]
    large_list.extend(['cs.AI', 'cs.LG'])

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "large-list",
            "allow": {
                "everyone": True,
                "conditions": [f"document.category in {large_list}"]
            }
        }],
        "default": "deny"
    })

    import time
    start = time.time()
    filter_obj = to_chromadb_filter(policy, {})
    elapsed = time.time() - start

    if elapsed > 0.1:  # 100ms threshold
        print(f"      Filter generation slow: {elapsed*1000:.1f}ms")
        return False

    query_embedding = np.random.rand(384).tolist()
    query_start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where=filter_obj
    )
    query_elapsed = time.time() - query_start

    num_results = len(results['ids'][0])

    if query_elapsed > 0.5:  # 500ms threshold
        print(f"      Query slow: {query_elapsed*1000:.1f}ms")
        return False

    # Should get AI/ML docs
    if num_results != 10:
        print(f"      Expected 10 results, got {num_results}")
        return False

    print(f"      Large list (102 items): {elapsed*1000:.1f}ms filter, {query_elapsed*1000:.1f}ms query")
    return True

test("Large list performance", test_large_list)

def test_empty_list():
    """Test empty list behavior."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "empty-list",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in []"]
            }
        }],
        "default": "deny"
    })

    filter_obj = to_chromadb_filter(policy, {})

    query_embedding = np.random.rand(384).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where=filter_obj
    )

    num_results = len(results['ids'][0])

    # Empty list should match nothing
    if num_results != 0:
        print(f"      Empty list matched {num_results} documents")
        return False

    print(f"      Empty list correctly matches nothing")
    return True

test("Empty list behavior", test_empty_list)

# Summary
print("\n" + "=" * 80)
print("CHROMADB INTEGRATION TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL CHROMADB INTEGRATION TESTS PASSED!")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
