"""
Neo4j Graph Database Example with RAGGuard.

This example demonstrates how to use RAGGuard with Neo4j for permission-aware
graph-based retrieval. Neo4j is ideal for:
- Document graphs with relationships
- Knowledge graphs
- Multi-hop reasoning with access control
- Hybrid vector + graph search

Requirements:
    pip install ragguard[neo4j] neo4j

Example graph structure:
    (User)-[:BELONGS_TO]->(Department)
    (Document)-[:OWNED_BY]->(Department)
    (Document)-[:HAS_TAG]->(Tag)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragguard.policy import Policy

print("=" * 70)
print("RAGGuard + Neo4j: Permission-Aware Graph Retrieval")
print("=" * 70)


def example_basic_neo4j():
    """Basic Neo4j retriever example."""
    print("\n" + "-" * 70)
    print("Example 1: Basic Neo4j Retriever")
    print("-" * 70)

    # Define policy for document access
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-full-access",
                "allow": {"roles": ["admin"]}
            },
            {
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "public-documents",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            }
        ],
        "default": "deny"
    })

    print("\nPolicy defined:")
    print("  - Admins can access all documents")
    print("  - Users can access documents from their department")
    print("  - Everyone can access public documents")

    # In production, you would connect to Neo4j like this:
    print("\n# Production code:")
    print("""
    from neo4j import GraphDatabase
    from ragguard import Neo4jSecureRetriever

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )

    # Create secure retriever
    retriever = Neo4jSecureRetriever(
        driver=driver,
        database="neo4j",
        node_label="Document",
        policy=policy,
        text_property="content",
        embedding_property="embedding"
    )

    # Search as engineering user
    user = {"id": "alice", "department": "engineering", "roles": ["engineer"]}
    results = retriever.search(
        query="machine learning best practices",
        user=user,
        limit=5
    )

    # Results only include documents alice can access
    for doc in results:
        print(f"- {doc['title']} (score: {doc['score']:.2f})")
    """)


def example_cypher_filter():
    """Example showing Cypher filter generation."""
    print("\n" + "-" * 70)
    print("Example 2: Cypher Filter Generation")
    print("-" * 70)

    from ragguard.policy import PolicyEngine

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "owner-access",
                "allow": {
                    "conditions": ["user.id == document.owner_id"]
                }
            },
            {
                "name": "team-access",
                "allow": {
                    "conditions": ["user.team in document.allowed_teams"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Generate Cypher WHERE clause for different users
    users = [
        {"id": "alice", "team": "ml-team"},
        {"id": "bob", "team": "data-team"},
        {"id": "admin", "team": "admin-team", "roles": ["admin"]}
    ]

    print("\nGenerated Cypher filters:")
    for user in users:
        try:
            cypher_clause, params = engine.to_filter(user, "neo4j")
            print(f"\nUser: {user['id']} (team: {user['team']})")
            print(f"  WHERE clause: {cypher_clause}")
            print(f"  Parameters: {params}")
        except Exception as e:
            print(f"\nUser: {user['id']} - Error: {e}")


def example_knowledge_graph():
    """Example: Knowledge graph with permission-aware traversal."""
    print("\n" + "-" * 70)
    print("Example 3: Knowledge Graph Traversal")
    print("-" * 70)

    print("""
    # Graph Schema:
    #   (Document)-[:REFERENCES]->(Document)
    #   (Document)-[:ABOUT]->(Topic)
    #   (Document)-[:CLASSIFIED_AS]->(SecurityLevel)
    #   (User)-[:HAS_CLEARANCE]->(SecurityLevel)

    # Policy for multi-hop access control
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "clearance-based",
                "allow": {
                    "conditions": [
                        "user.clearance_level >= document.security_level"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    # Query that traverses relationships
    retriever = Neo4jSecureRetriever(...)

    # Find documents about "machine learning" that reference other documents
    # Only returns documents the user has clearance for
    results = retriever.search(
        query=\"\"\"
        MATCH (d:Document)-[:ABOUT]->(t:Topic {name: 'machine learning'})
        OPTIONAL MATCH (d)-[:REFERENCES]->(ref:Document)
        RETURN d, collect(ref) as references
        \"\"\",
        user={"id": "researcher", "clearance_level": 3},
        limit=10
    )
    """)


def example_hybrid_search():
    """Example: Hybrid vector + graph search."""
    print("\n" + "-" * 70)
    print("Example 4: Hybrid Vector + Graph Search")
    print("-" * 70)

    print("""
    # Neo4j 5.x supports vector indexes for hybrid search
    # Combine semantic similarity with graph relationships

    from ragguard import Neo4jSecureRetriever

    retriever = Neo4jSecureRetriever(
        driver=driver,
        database="neo4j",
        node_label="Document",
        policy=policy,
        # Vector search configuration
        vector_index="document_embeddings",
        embedding_property="embedding",
        # Enable hybrid search
        use_hybrid_search=True
    )

    # Search combines:
    # 1. Vector similarity (semantic match)
    # 2. Graph relationships (context)
    # 3. Permission filtering (access control)

    user = {"id": "alice", "department": "research"}

    results = retriever.search(
        query="neural network architectures",  # Semantic query
        user=user,
        limit=10,
        # Boost documents connected to user's previous queries
        graph_boost={
            "pattern": "(d)-[:CITED_BY]->(:Document)<-[:AUTHORED]-(:User {id: $user_id})",
            "weight": 0.3
        }
    )
    """)


def example_multi_tenant():
    """Example: Multi-tenant graph with RAGGuard."""
    print("\n" + "-" * 70)
    print("Example 5: Multi-Tenant Graph Database")
    print("-" * 70)

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "tenant-isolation",
                "allow": {
                    "conditions": ["user.tenant_id == document.tenant_id"]
                }
            },
            {
                "name": "shared-resources",
                "match": {"shared": True},
                "allow": {"everyone": True}
            }
        ],
        "default": "deny"
    })

    print(f"Policy: {policy}")
    print("""
    # Multi-tenant setup ensures complete data isolation

    # Tenant A user
    tenant_a_user = {
        "id": "alice",
        "tenant_id": "acme-corp",
        "roles": ["analyst"]
    }

    # Tenant B user
    tenant_b_user = {
        "id": "bob",
        "tenant_id": "globex-inc",
        "roles": ["analyst"]
    }

    # Same query, different results based on tenant
    retriever = Neo4jSecureRetriever(...)

    # Alice only sees Acme Corp documents
    alice_results = retriever.search(
        query="quarterly report",
        user=tenant_a_user
    )

    # Bob only sees Globex Inc documents
    bob_results = retriever.search(
        query="quarterly report",
        user=tenant_b_user
    )

    # No cross-tenant data leakage possible!
    """)


def example_audit_logging():
    """Example: Audit logging for compliance."""
    print("\n" + "-" * 70)
    print("Example 6: Audit Logging for Compliance")
    print("-" * 70)

    print("""
    from ragguard import Neo4jSecureRetriever, AuditLogger

    # Configure audit logger
    audit_logger = AuditLogger(
        backend="file",
        filepath="/var/log/ragguard/neo4j_access.log",
        include_query=True,
        include_results=False  # Don't log actual content
    )

    retriever = Neo4jSecureRetriever(
        driver=driver,
        policy=policy,
        audit_logger=audit_logger
    )

    # Every search is logged
    results = retriever.search(
        query="confidential project data",
        user={"id": "alice", "department": "engineering"}
    )

    # Audit log entry:
    # {
    #   "timestamp": "2024-01-15T10:30:00Z",
    #   "user_id": "alice",
    #   "action": "search",
    #   "backend": "neo4j",
    #   "query": "confidential project data",
    #   "result_count": 5,
    #   "allowed": true,
    #   "matched_rule": "department-access"
    # }
    """)


if __name__ == "__main__":
    example_basic_neo4j()
    example_cypher_filter()
    example_knowledge_graph()
    example_hybrid_search()
    example_multi_tenant()
    example_audit_logging()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nTo run with a real Neo4j instance:")
    print("  1. Start Neo4j: docker run -p 7474:7474 -p 7687:7687 neo4j")
    print("  2. Install: pip install ragguard[neo4j] neo4j")
    print("  3. Update connection settings in the code")
