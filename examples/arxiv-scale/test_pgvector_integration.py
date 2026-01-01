#!/usr/bin/env python3
"""
pgvector Integration Tests for RAGGuard v0.2.0

Tests the complete end-to-end flow with a real PostgreSQL database.

Requirements:
- PostgreSQL with pgvector extension
- Database connection: postgresql://localhost/test_ragguard
- Run: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=test ankane/pgvector
"""

import os
import sys
import psycopg2
import numpy as np
from typing import Optional

print("=" * 80)
print("pgvector Integration Tests")
print("=" * 80)

# Check if pgvector is available
try:
    import psycopg2
    PGVECTOR_AVAILABLE = True
except ImportError:
    print("\n⚠️  psycopg2 not installed. Install with: pip install psycopg2-binary")
    PGVECTOR_AVAILABLE = False
    sys.exit(0)

# Try to connect to database
def get_connection() -> Optional[psycopg2.extensions.connection]:
    """Try to connect to test database."""
    connection_strings = [
        "postgresql://localhost/test_ragguard",
        "postgresql://postgres:test@localhost/postgres",
        "postgresql://localhost/postgres",
    ]

    for conn_str in connection_strings:
        try:
            conn = psycopg2.connect(conn_str)
            print(f"✅ Connected to: {conn_str}")
            return conn
        except Exception as e:
            continue

    return None

conn = get_connection()

if conn is None:
    print("\n⚠️  PostgreSQL not available. Integration tests SKIPPED.")
    print("\nTo run these tests:")
    print("  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=test ankane/pgvector")
    print("  createdb test_ragguard")
    print("\n❌ Exiting with failure code (tests did not run)")
    sys.exit(1)  # Fail if database not available

from ragguard import Policy
from ragguard.filters.builder import to_pgvector_filter
from ragguard.policy.engine import PolicyEngine

tests_passed = 0
tests_failed = 0

def setup_database():
    """Create test table with pgvector extension."""
    global conn
    cur = conn.cursor()

    try:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Drop existing table
        cur.execute("DROP TABLE IF EXISTS test_documents")

        # Create table with vector column
        cur.execute("""
            CREATE TABLE test_documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(384),
                category TEXT,
                status TEXT,
                access_level TEXT,
                department TEXT,
                created_by TEXT,
                tags TEXT[]
            )
        """)

        # Insert test data
        test_docs = [
            # AI papers - public
            ("Deep Learning Overview", np.random.rand(384).tolist(), "cs.AI", "published", "public", "research", "alice", ["ai", "ml"]),
            ("Neural Networks Intro", np.random.rand(384).tolist(), "cs.AI", "published", "public", "research", "bob", ["ai", "nn"]),

            # ML papers - public
            ("Machine Learning Basics", np.random.rand(384).tolist(), "cs.LG", "published", "public", "research", "alice", ["ml"]),
            ("Supervised Learning", np.random.rand(384).tolist(), "cs.LG", "published", "public", "research", "charlie", ["ml", "supervised"]),

            # Restricted papers
            ("Secret AI Research", np.random.rand(384).tolist(), "cs.AI", "published", "restricted", "research", "alice", ["ai", "secret"]),
            ("Classified ML Model", np.random.rand(384).tolist(), "cs.LG", "published", "classified", "security", "bob", ["ml", "classified"]),

            # Archived papers
            ("Old AI Paper", np.random.rand(384).tolist(), "cs.AI", "archived", "public", "research", "alice", ["ai", "old"]),
            ("Deprecated ML", np.random.rand(384).tolist(), "cs.LG", "archived", "public", "research", "bob", ["ml", "old"]),

            # Draft papers
            ("Draft AI Work", np.random.rand(384).tolist(), "cs.AI", "draft", "public", "research", "charlie", ["ai", "draft"]),
            ("WIP ML Paper", np.random.rand(384).tolist(), "cs.LG", "draft", "public", "research", "alice", ["ml", "wip"]),

            # Database papers (different category)
            ("SQL Optimization", np.random.rand(384).tolist(), "cs.DB", "published", "public", "engineering", "bob", ["db", "sql"]),
            ("NoSQL Systems", np.random.rand(384).tolist(), "cs.DB", "published", "public", "engineering", "charlie", ["db", "nosql"]),
        ]

        for content, embedding, category, status, access_level, dept, created_by, tags in test_docs:
            cur.execute("""
                INSERT INTO test_documents
                (content, embedding, category, status, access_level, department, created_by, tags)
                VALUES (%s, %s::vector, %s, %s, %s, %s, %s, %s)
            """, (content, embedding, category, status, access_level, dept, created_by, tags))

        conn.commit()
        print(f"✅ Database setup complete: {len(test_docs)} documents inserted")

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()

def teardown_database():
    """Clean up test database."""
    global conn
    cur = conn.cursor()
    try:
        cur.execute("DROP TABLE IF EXISTS test_documents")
        conn.commit()
    except:
        conn.rollback()
    finally:
        cur.close()

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

# Setup database
setup_database()

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

    sql, params = to_pgvector_filter(policy, {})

    # Query database
    cur = conn.cursor()
    try:
        query = f"""
            SELECT content, category, access_level
            FROM test_documents
            WHERE {sql}
            ORDER BY id
        """
        cur.execute(query, params)
        results = cur.fetchall()

        # Should get 8 public documents
        if len(results) != 8:
            print(f"      Expected 8 results, got {len(results)}")
            return False

        # All should be public
        for content, category, access_level in results:
            if access_level != 'public':
                print(f"      Found non-public document: {content}")
                return False

        print(f"      Retrieved {len(results)} public documents")
        return True
    finally:
        cur.close()

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

    sql, params = to_pgvector_filter(policy, {})

    cur = conn.cursor()
    try:
        query = f"SELECT content, status FROM test_documents WHERE {sql}"
        cur.execute(query, params)
        results = cur.fetchall()

        # Should get 10 non-archived (12 total - 2 archived)
        if len(results) != 10:
            print(f"      Expected 10 results, got {len(results)}")
            return False

        # None should be archived
        for content, status in results:
            if status == 'archived':
                print(f"      Found archived document: {content}")
                return False

        print(f"      Retrieved {len(results)} non-archived documents")
        return True
    finally:
        cur.close()

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

    sql, params = to_pgvector_filter(policy, {})

    cur = conn.cursor()
    try:
        query = f"SELECT content, category FROM test_documents WHERE {sql}"
        cur.execute(query, params)
        results = cur.fetchall()

        # Should get 10 AI/ML papers (5 AI + 5 ML)
        if len(results) != 10:
            print(f"      Expected 10 results, got {len(results)}")
            return False

        # All should be cs.AI or cs.LG
        for content, category in results:
            if category not in ['cs.AI', 'cs.LG']:
                print(f"      Found wrong category: {category}")
                return False

        print(f"      Retrieved {len(results)} AI/ML documents")
        return True
    finally:
        cur.close()

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

    sql, params = to_pgvector_filter(policy, {})

    cur = conn.cursor()
    try:
        query = f"SELECT content, status FROM test_documents WHERE {sql}"
        cur.execute(query, params)
        results = cur.fetchall()

        # Should get 8 published (12 total - 2 archived - 2 draft)
        if len(results) != 8:
            print(f"      Expected 8 results, got {len(results)}")
            return False

        # None should be archived or draft
        for content, status in results:
            if status in ['archived', 'draft']:
                print(f"      Found excluded status: {status}")
                return False

        print(f"      Retrieved {len(results)} published documents")
        return True
    finally:
        cur.close()

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

    sql, params = to_pgvector_filter(policy, {})

    cur = conn.cursor()
    try:
        query = f"SELECT content, category, access_level, status FROM test_documents WHERE {sql}"
        cur.execute(query, params)
        results = cur.fetchall()

        # Should get 4 documents (AI/ML, public/classified, published)
        # - Deep Learning Overview (AI, public, published) ✓
        # - Neural Networks Intro (AI, public, published) ✓
        # - Machine Learning Basics (LG, public, published) ✓
        # - Supervised Learning (LG, public, published) ✓
        # Excluded:
        # - Secret AI Research (restricted) ✗
        # - Classified ML Model (classified but != restricted, so ✓) - wait, classified != restricted

        # Let me recalculate:
        # AI: Deep Learning (public, published), Neural Networks (public, published), Secret (restricted - excluded), Old (archived - excluded), Draft (draft - excluded) = 2
        # LG: ML Basics (public, published), Supervised (public, published), Classified (classified, published - NOT restricted), Deprecated (archived - excluded), WIP (draft - excluded) = 3
        # Total: 5

        expected = 5
        if len(results) != expected:
            print(f"      Expected {expected} results, got {len(results)}")
            for content, category, access_level, status in results:
                print(f"        - {content}: {category}, {access_level}, {status}")
            return False

        # Verify all match criteria
        for content, category, access_level, status in results:
            if category not in ['cs.AI', 'cs.LG']:
                print(f"      Wrong category: {category}")
                return False
            if access_level == 'restricted':
                print(f"      Found restricted: {content}")
                return False
            if status in ['archived', 'draft']:
                print(f"      Found excluded status: {status}")
                return False

        print(f"      Retrieved {len(results)} matching documents")
        return True
    finally:
        cur.close()

test("Multiple conditions (AND logic)", test_multiple_conditions)

def test_sql_injection_protection():
    """Test SQL injection protection."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "injection-test",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in ['cs.AI', \"'; DROP TABLE test_documents--\"]"]
            }
        }],
        "default": "deny"
    })

    sql, params = to_pgvector_filter(policy, {})

    cur = conn.cursor()
    try:
        # This should treat the injection string as a literal value
        query = f"SELECT content, category FROM test_documents WHERE {sql}"
        cur.execute(query, params)
        results = cur.fetchall()

        # Should only match cs.AI (injection string won't match anything)
        for content, category in results:
            if category not in ['cs.AI']:
                print(f"      Injection might have worked: {category}")
                return False

        # Verify table still exists
        cur.execute("SELECT COUNT(*) FROM test_documents")
        count = cur.fetchone()[0]

        if count != 12:
            print(f"      Table was modified! Count: {count}")
            return False

        print(f"      SQL injection blocked, table intact ({count} docs)")
        return True
    finally:
        cur.close()

test("SQL injection protection", test_sql_injection_protection)

def test_large_list():
    """Test performance with large list."""
    large_list = [f"category_{i}" for i in range(100)]
    large_list.extend(['cs.AI', 'cs.LG'])  # Add some that match

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
    sql, params = to_pgvector_filter(policy, {})
    elapsed = time.time() - start

    if elapsed > 0.1:  # 100ms threshold
        print(f"      Filter generation slow: {elapsed*1000:.1f}ms")
        return False

    cur = conn.cursor()
    try:
        query_start = time.time()
        query = f"SELECT content, category FROM test_documents WHERE {sql}"
        cur.execute(query, params)
        results = cur.fetchall()
        query_elapsed = time.time() - query_start

        if query_elapsed > 0.5:  # 500ms threshold
            print(f"      Query slow: {query_elapsed*1000:.1f}ms")
            return False

        # Should get AI/ML docs
        if len(results) != 10:
            print(f"      Expected 10 results, got {len(results)}")
            return False

        print(f"      Large list (102 items): {elapsed*1000:.1f}ms filter, {query_elapsed*1000:.1f}ms query")
        return True
    finally:
        cur.close()

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

    sql, params = to_pgvector_filter(policy, {})

    cur = conn.cursor()
    try:
        query = f"SELECT content FROM test_documents WHERE {sql}"
        cur.execute(query, params)
        results = cur.fetchall()

        # Empty list should match nothing
        if len(results) != 0:
            print(f"      Empty list matched {len(results)} documents")
            return False

        print(f"      Empty list correctly matches nothing")
        return True
    finally:
        cur.close()

test("Empty list behavior", test_empty_list)

# Cleanup
teardown_database()
conn.close()

# Summary
print("\n" + "=" * 80)
print("PGVECTOR INTEGRATION TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL PGVECTOR INTEGRATION TESTS PASSED!")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
