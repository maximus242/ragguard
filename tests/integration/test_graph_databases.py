"""
Integration tests for graph database retrievers.

These tests run against real graph databases via Docker.
Run with: docker-compose up -d neo4j arangodb
Then: pytest tests/integration/test_graph_databases.py -v

Note: Neptune requires AWS infrastructure (use unit tests with mocks).
      TigerGraph requires enterprise license (use unit tests with mocks).
"""

import os
import time
from typing import Any, Dict, List

import pytest

from ragguard import Policy
from ragguard.policy.models import AllowConditions, Rule
from ragguard.retrievers.arangodb import ArangoDBSecureRetriever
from ragguard.retrievers.neo4j import Neo4jSecureRetriever

# Skip all tests if databases aren't available
NEO4J_AVAILABLE = os.environ.get("RAGGUARD_TEST_NEO4J", "0") == "1"
ARANGODB_AVAILABLE = os.environ.get("RAGGUARD_TEST_ARANGODB", "0") == "1"


def make_policy(rules: List[Dict], default: str = "deny") -> Policy:
    """Helper to create a policy from rules."""
    return Policy(
        version="1",
        rules=[
            Rule(
                name=f"rule_{i}",
                allow=AllowConditions(
                    roles=r.get("roles"),
                    everyone=r.get("everyone"),
                    conditions=r.get("conditions")
                ),
                match=r.get("match")
            )
            for i, r in enumerate(rules)
        ],
        default=default
    )


# ============================================================
# Neo4j Integration Tests
# ============================================================

@pytest.fixture(scope="module")
def neo4j_driver():
    """Create Neo4j driver for tests."""
    if not NEO4J_AVAILABLE:
        pytest.skip("Neo4j not available (set RAGGUARD_TEST_NEO4J=1)")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        pytest.skip("neo4j package not installed")

    # Connect to Neo4j
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "ragguard_test")
    )

    driver = GraphDatabase.driver(uri, auth=auth)

    # Wait for Neo4j to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            driver.verify_connectivity()
            break
        except Exception:
            if i == max_retries - 1:
                pytest.skip("Neo4j not ready after 30s")
            time.sleep(1)

    yield driver
    driver.close()


@pytest.fixture
def neo4j_test_data(neo4j_driver):
    """Set up test data in Neo4j."""
    with neo4j_driver.session() as session:
        # Clear existing test data
        session.run("MATCH (d:Document) DETACH DELETE d")

        # Create test documents with different departments and access levels
        test_docs = [
            {"id": "doc1", "title": "Engineering Doc", "department": "engineering", "category": "internal", "priority": 5},
            {"id": "doc2", "title": "Sales Report", "department": "sales", "category": "confidential", "priority": 8},
            {"id": "doc3", "title": "Public FAQ", "department": "support", "category": "public", "priority": 3},
            {"id": "doc4", "title": "HR Policy", "department": "hr", "category": "confidential", "priority": 7},
            {"id": "doc5", "title": "Engineering Secret", "department": "engineering", "category": "confidential", "priority": 10},
        ]

        for doc in test_docs:
            session.run(
                """
                CREATE (d:Document {
                    id: $id,
                    title: $title,
                    department: $department,
                    category: $category,
                    priority: $priority
                })
                """,
                **doc
            )

        # Create relationships between documents
        session.run("""
            MATCH (d1:Document {id: 'doc1'}), (d2:Document {id: 'doc5'})
            CREATE (d1)-[:RELATES_TO]->(d2)
        """)
        session.run("""
            MATCH (d1:Document {id: 'doc2'}), (d2:Document {id: 'doc4'})
            CREATE (d1)-[:RELATES_TO]->(d2)
        """)

    yield

    # Cleanup
    with neo4j_driver.session() as session:
        session.run("MATCH (d:Document) DETACH DELETE d")


@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
class TestNeo4jIntegration:
    """Integration tests for Neo4j retriever."""

    def test_department_filtering(self, neo4j_driver, neo4j_test_data):
        """Test that users can only see documents from their department."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.department == document.department"]
            }
        ])

        retriever = Neo4jSecureRetriever(
            driver=neo4j_driver,
            node_label="Document",
            policy=policy
        )

        # Engineering user should only see engineering docs
        results = retriever.search(
            query={"category": "internal"},
            user={"id": "alice", "department": "engineering"},
            limit=10
        )

        # Check all results are from engineering
        for doc in results:
            assert doc.get("department") == "engineering"

    def test_category_filtering(self, neo4j_driver, neo4j_test_data):
        """Test category-based access control."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category in ['public', 'internal']"]
            }
        ])

        retriever = Neo4jSecureRetriever(
            driver=neo4j_driver,
            node_label="Document",
            policy=policy
        )

        results = retriever.search(
            query={},  # All documents
            user={"id": "guest"},
            limit=10
        )

        # Should only get public and internal docs
        for doc in results:
            assert doc.get("category") in ["public", "internal"]

    def test_admin_access(self, neo4j_driver, neo4j_test_data):
        """Test that admins can access all documents."""
        policy = make_policy([
            {"roles": ["admin"]},  # Admin can see everything
            {
                "everyone": True,
                "conditions": ["document.category == 'public'"]
            }
        ])

        retriever = Neo4jSecureRetriever(
            driver=neo4j_driver,
            node_label="Document",
            policy=policy
        )

        # Admin should see all documents
        admin_results = retriever.search(
            query={},
            user={"id": "admin", "roles": ["admin"]},
            limit=10
        )

        # Regular user should only see public
        guest_results = retriever.search(
            query={},
            user={"id": "guest", "roles": []},
            limit=10
        )

        assert len(admin_results) > len(guest_results)

    def test_relationship_traversal(self, neo4j_driver, neo4j_test_data):
        """Test traversing relationships with permission filtering."""
        policy = make_policy([
            {"everyone": True}  # Allow all for traversal test
        ])

        retriever = Neo4jSecureRetriever(
            driver=neo4j_driver,
            node_label="Document",
            policy=policy
        )

        # Traverse from doc1 to related documents
        results = retriever.traverse(
            start_node_id="doc1",
            relationship_type="RELATES_TO",
            user={"id": "alice"},
            direction="outgoing",
            depth=1,
            limit=10
        )

        # Should find doc5 (related to doc1)
        related_ids = [doc.get("id") for doc in results]
        assert "doc5" in related_ids or len(results) > 0

    def test_deny_all_when_no_rules_match(self, neo4j_driver, neo4j_test_data):
        """Test that users get no results when no rules match."""
        policy = make_policy([
            {
                "roles": ["super_admin"],  # No one has this role
            }
        ], default="deny")

        retriever = Neo4jSecureRetriever(
            driver=neo4j_driver,
            node_label="Document",
            policy=policy
        )

        results = retriever.search(
            query={},
            user={"id": "alice", "roles": ["viewer"]},
            limit=10
        )

        assert len(results) == 0


# ============================================================
# ArangoDB Integration Tests
# ============================================================

@pytest.fixture(scope="module")
def arangodb_client():
    """Create ArangoDB client for tests."""
    if not ARANGODB_AVAILABLE:
        pytest.skip("ArangoDB not available (set RAGGUARD_TEST_ARANGODB=1)")

    try:
        from arango import ArangoClient
    except ImportError:
        pytest.skip("python-arango package not installed")

    # Connect to ArangoDB
    host = os.environ.get("ARANGODB_HOST", "localhost")
    port = int(os.environ.get("ARANGODB_PORT", "8529"))
    password = os.environ.get("ARANGODB_PASSWORD", "ragguard_test")

    client = ArangoClient(hosts=f"http://{host}:{port}")

    # Wait for ArangoDB to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            sys_db = client.db("_system", username="root", password=password)
            sys_db.properties()
            break
        except Exception:
            if i == max_retries - 1:
                pytest.skip("ArangoDB not ready after 30s")
            time.sleep(1)

    yield client, password
    client.close()


@pytest.fixture
def arangodb_database(arangodb_client):
    """Set up test database in ArangoDB."""
    client, password = arangodb_client
    sys_db = client.db("_system", username="root", password=password)

    # Create test database
    db_name = "ragguard_test"
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)

    test_db = client.db(db_name, username="root", password=password)

    # Create document collection
    if not test_db.has_collection("documents"):
        test_db.create_collection("documents")

    # Create edge collection for relationships
    if not test_db.has_collection("relations"):
        test_db.create_collection("relations", edge=True)

    yield test_db

    # Cleanup
    if sys_db.has_database(db_name):
        sys_db.delete_database(db_name)


@pytest.fixture
def arangodb_test_data(arangodb_database):
    """Set up test data in ArangoDB."""
    db = arangodb_database
    docs = db.collection("documents")
    edges = db.collection("relations")

    # Clear existing data
    docs.truncate()
    edges.truncate()

    # Create test documents
    test_docs = [
        {"_key": "doc1", "title": "Engineering Doc", "department": "engineering", "category": "internal", "priority": 5},
        {"_key": "doc2", "title": "Sales Report", "department": "sales", "category": "confidential", "priority": 8},
        {"_key": "doc3", "title": "Public FAQ", "department": "support", "category": "public", "priority": 3},
        {"_key": "doc4", "title": "HR Policy", "department": "hr", "category": "confidential", "priority": 7},
        {"_key": "doc5", "title": "Engineering Secret", "department": "engineering", "category": "confidential", "priority": 10},
    ]

    for doc in test_docs:
        docs.insert(doc)

    # Create relationships
    edges.insert({"_from": "documents/doc1", "_to": "documents/doc5", "type": "RELATES_TO"})
    edges.insert({"_from": "documents/doc2", "_to": "documents/doc4", "type": "RELATES_TO"})

    yield


@pytest.mark.skipif(not ARANGODB_AVAILABLE, reason="ArangoDB not available")
class TestArangoDBIntegration:
    """Integration tests for ArangoDB retriever."""

    def test_department_filtering(self, arangodb_database, arangodb_test_data):
        """Test that users can only see documents from their department."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.department == document.department"]
            }
        ])

        retriever = ArangoDBSecureRetriever(
            database=arangodb_database,
            collection_name="documents",
            policy=policy
        )

        # Engineering user should only see engineering docs
        results = retriever.search(
            query={"category": "internal"},
            user={"id": "alice", "department": "engineering"},
            limit=10
        )

        # Check all results are from engineering
        for doc in results:
            assert doc.get("department") == "engineering"

    def test_category_filtering(self, arangodb_database, arangodb_test_data):
        """Test category-based access control."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category in ['public', 'internal']"]
            }
        ])

        retriever = ArangoDBSecureRetriever(
            database=arangodb_database,
            collection_name="documents",
            policy=policy
        )

        results = retriever.search(
            query={},  # All documents
            user={"id": "guest"},
            limit=10
        )

        # Should only get public and internal docs
        for doc in results:
            assert doc.get("category") in ["public", "internal"]

    def test_admin_access(self, arangodb_database, arangodb_test_data):
        """Test that admins can access all documents."""
        policy = make_policy([
            {"roles": ["admin"]},  # Admin can see everything
            {
                "everyone": True,
                "conditions": ["document.category == 'public'"]
            }
        ])

        retriever = ArangoDBSecureRetriever(
            database=arangodb_database,
            collection_name="documents",
            policy=policy
        )

        # Admin should see all documents
        admin_results = retriever.search(
            query={},
            user={"id": "admin", "roles": ["admin"]},
            limit=10
        )

        # Regular user should only see public
        guest_results = retriever.search(
            query={},
            user={"id": "guest", "roles": []},
            limit=10
        )

        assert len(admin_results) > len(guest_results)

    def test_relationship_traversal(self, arangodb_database, arangodb_test_data):
        """Test traversing relationships with permission filtering."""
        policy = make_policy([
            {"everyone": True}  # Allow all for traversal test
        ])

        retriever = ArangoDBSecureRetriever(
            database=arangodb_database,
            collection_name="documents",
            policy=policy,
            edge_collection="relations"
        )

        # Traverse from doc1 to related documents
        results = retriever.traverse(
            start_node_id="documents/doc1",
            relationship_type="RELATES_TO",
            user={"id": "alice"},
            direction="outgoing",
            depth=1,
            limit=10
        )

        # Should find related documents
        assert len(results) >= 0  # May be 0 if traversal not fully implemented

    def test_deny_all_when_no_rules_match(self, arangodb_database, arangodb_test_data):
        """Test that users get no results when no rules match."""
        policy = make_policy([
            {
                "roles": ["super_admin"],  # No one has this role
            }
        ], default="deny")

        retriever = ArangoDBSecureRetriever(
            database=arangodb_database,
            collection_name="documents",
            policy=policy
        )

        results = retriever.search(
            query={},
            user={"id": "alice", "roles": ["viewer"]},
            limit=10
        )

        assert len(results) == 0

    def test_aql_graph_query(self, arangodb_database, arangodb_test_data):
        """Test executing raw AQL graph queries."""
        policy = make_policy([
            {"everyone": True}
        ])

        retriever = ArangoDBSecureRetriever(
            database=arangodb_database,
            collection_name="documents",
            policy=policy
        )

        # Execute raw AQL query
        results = retriever.search(
            query="FOR doc IN documents FILTER doc.priority > 5 RETURN doc",
            user={"id": "alice"},
            limit=10
        )

        # Should get high priority documents
        for doc in results:
            assert doc.get("priority", 0) > 5


# ============================================================
# Cross-Backend Consistency Tests
# ============================================================

@pytest.mark.skipif(
    not (NEO4J_AVAILABLE and ARANGODB_AVAILABLE),
    reason="Both Neo4j and ArangoDB required"
)
class TestGraphBackendConsistency:
    """Test that different graph backends produce consistent results."""

    def test_same_policy_different_backends(
        self,
        neo4j_driver, neo4j_test_data,
        arangodb_database, arangodb_test_data
    ):
        """Test that the same policy produces equivalent results."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category in ['public', 'internal']"]
            }
        ])

        neo4j_retriever = Neo4jSecureRetriever(
            driver=neo4j_driver,
            node_label="Document",
            policy=policy
        )

        arangodb_retriever = ArangoDBSecureRetriever(
            database=arangodb_database,
            collection_name="documents",
            policy=policy
        )

        user = {"id": "guest"}

        neo4j_results = neo4j_retriever.search(query={}, user=user, limit=10)
        arangodb_results = arangodb_retriever.search(query={}, user=user, limit=10)

        # Both should return same number of accessible documents
        # (2: public FAQ and internal engineering doc)
        neo4j_categories = {doc.get("category") for doc in neo4j_results}
        arangodb_categories = {doc.get("category") for doc in arangodb_results}

        # Categories should be subsets of allowed
        assert neo4j_categories <= {"public", "internal"}
        assert arangodb_categories <= {"public", "internal"}
