"""
Amazon Neptune Graph Database Example with RAGGuard.

This example demonstrates how to use RAGGuard with Amazon Neptune for
permission-aware graph retrieval using Gremlin queries.

Neptune is AWS's managed graph database service, ideal for:
- Highly available graph workloads
- Knowledge graphs at scale
- Fraud detection with access control
- Identity graphs with permission filtering

Requirements:
    pip install ragguard gremlinpython

AWS Configuration:
    - Neptune cluster endpoint
    - IAM authentication or VPC access
    - Appropriate security groups
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragguard.policy import Policy

print("=" * 70)
print("RAGGuard + Amazon Neptune: Permission-Aware Gremlin Queries")
print("=" * 70)


def example_basic_neptune():
    """Basic Neptune retriever example."""
    print("\n" + "-" * 70)
    print("Example 1: Basic Neptune Setup")
    print("-" * 70)

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-access",
                "allow": {"roles": ["admin"]}
            },
            {
                "name": "owner-access",
                "allow": {
                    "conditions": ["user.id == document.owner"]
                }
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

    print("\nPolicy configured for:")
    print("  - Admin full access")
    print("  - Owner-based access")
    print("  - Department-based access")

    print("""
    # Production Neptune connection
    from ragguard import NeptuneSecureRetriever

    retriever = NeptuneSecureRetriever(
        endpoint="your-cluster.cluster-xxxxx.region.neptune.amazonaws.com",
        port=8182,
        policy=policy,
        use_iam_auth=True,  # Recommended for production
        region="us-east-1"
    )

    # Search with user context
    user = {
        "id": "alice@company.com",
        "department": "engineering",
        "roles": ["engineer", "team-lead"]
    }

    results = retriever.search(
        query="machine learning pipeline",
        user=user,
        limit=10
    )
    """)


def example_gremlin_filter():
    """Example showing Gremlin filter generation."""
    print("\n" + "-" * 70)
    print("Example 2: Gremlin Traversal Filter Generation")
    print("-" * 70)

    from ragguard.policy import PolicyEngine

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "classification-access",
                "allow": {
                    "conditions": [
                        "user.clearance_level >= document.classification"
                    ]
                }
            },
            {
                "name": "project-member",
                "allow": {
                    "conditions": ["user.project_id == document.project_id"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    users = [
        {"id": "analyst", "clearance_level": 2, "project_id": "proj-123"},
        {"id": "manager", "clearance_level": 4, "project_id": "proj-456"},
    ]

    print("\nGenerated Gremlin filters:")
    for user in users:
        try:
            # Neptune returns a list of filter specifications
            filters = engine.to_filter(user, "neptune")
            print(f"\nUser: {user['id']} (clearance: {user['clearance_level']})")
            print(f"  Filters: {filters}")
        except Exception as e:
            print(f"\nUser: {user['id']} - Error: {e}")


def example_fraud_detection():
    """Example: Fraud detection graph with access control."""
    print("\n" + "-" * 70)
    print("Example 3: Fraud Detection Knowledge Graph")
    print("-" * 70)

    print("""
    # Fraud detection graph schema:
    #   (Account)-[:TRANSFERRED_TO]->(Account)
    #   (Account)-[:OWNED_BY]->(Customer)
    #   (Transaction)-[:FROM]->(Account)
    #   (Transaction)-[:TO]->(Account)
    #   (Alert)-[:ABOUT]->(Transaction)

    # Policy: Analysts can only see accounts in their region
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "regional-access",
                "allow": {
                    "conditions": ["user.region == document.region"]
                }
            },
            {
                "name": "fraud-team-global",
                "allow": {"roles": ["fraud-investigator"]}
            }
        ],
        "default": "deny"
    })

    retriever = NeptuneSecureRetriever(
        endpoint=neptune_endpoint,
        policy=policy
    )

    # Regional analyst searches
    analyst = {
        "id": "analyst-1",
        "region": "us-east",
        "roles": ["analyst"]
    }

    # Only sees alerts for US-East accounts
    results = retriever.search(
        query="suspicious transfer patterns",
        user=analyst,
        vertex_label="Alert"
    )

    # Fraud investigator sees all regions
    investigator = {
        "id": "investigator-1",
        "region": "global",
        "roles": ["fraud-investigator"]
    }

    # Sees alerts from all regions
    results = retriever.search(
        query="suspicious transfer patterns",
        user=investigator,
        vertex_label="Alert"
    )
    """)


def example_identity_graph():
    """Example: Identity graph with RAGGuard."""
    print("\n" + "-" * 70)
    print("Example 4: Identity Graph")
    print("-" * 70)

    print("""
    # Identity graph for customer 360 view
    # Schema:
    #   (Person)-[:HAS_EMAIL]->(Email)
    #   (Person)-[:HAS_PHONE]->(Phone)
    #   (Person)-[:HAS_ADDRESS]->(Address)
    #   (Person)-[:CUSTOMER_OF]->(Organization)

    # Policy: Org admins see their customers only
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "org-admin-access",
                "allow": {
                    "roles": ["org-admin"],
                    "conditions": ["user.org_id == document.org_id"]
                }
            },
            {
                "name": "support-limited",
                "allow": {
                    "roles": ["support"],
                    "conditions": [
                        "user.org_id == document.org_id",
                        "document.pii_level <= 2"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    # Gremlin query with permission filtering
    retriever = NeptuneSecureRetriever(
        endpoint=neptune_endpoint,
        policy=policy
    )

    # Support agent - limited PII access
    support_user = {
        "id": "support-1",
        "org_id": "acme-corp",
        "roles": ["support"]
    }

    # Can find customers but won't see high-PII fields
    results = retriever.search(
        query="john.doe@example.com",
        user=support_user,
        traversal_depth=2  # Follow relationships up to 2 hops
    )
    """)


def example_serverless_neptune():
    """Example: Neptune Serverless with RAGGuard."""
    print("\n" + "-" * 70)
    print("Example 5: Neptune Serverless Configuration")
    print("-" * 70)

    print("""
    # Neptune Serverless auto-scales based on workload
    # Perfect for variable RAG query patterns

    from ragguard import NeptuneSecureRetriever

    # Serverless configuration
    retriever = NeptuneSecureRetriever(
        endpoint="your-serverless.cluster-xxxxx.region.neptune.amazonaws.com",
        port=8182,
        policy=policy,
        use_iam_auth=True,
        region="us-east-1",
        # Serverless-specific settings
        connection_timeout=30,  # Longer timeout for cold starts
        max_retry_attempts=3,
        # Connection pooling for efficiency
        pool_size=10
    )

    # The retriever handles:
    # 1. IAM Signature V4 authentication
    # 2. WebSocket connection management
    # 3. Automatic retry on transient failures
    # 4. Permission filtering in Gremlin queries
    """)


def example_neptune_ml():
    """Example: Neptune ML with RAGGuard."""
    print("\n" + "-" * 70)
    print("Example 6: Neptune ML Integration")
    print("-" * 70)

    print("""
    # Neptune ML enables ML on graphs (link prediction, classification)
    # RAGGuard ensures ML results respect access control

    from ragguard import NeptuneSecureRetriever

    retriever = NeptuneSecureRetriever(
        endpoint=neptune_endpoint,
        policy=policy,
        # Enable Neptune ML features
        use_neptune_ml=True,
        ml_endpoint="your-ml-endpoint.region.amazonaws.com"
    )

    # Example: Find similar documents using graph ML
    # Results are filtered by user permissions

    user = {"id": "researcher", "department": "ml-research"}

    # Neptune ML finds similar nodes based on graph structure
    # RAGGuard filters results based on policy
    similar_docs = retriever.search(
        query="similar_to:doc-123",  # Special query syntax
        user=user,
        use_ml_similarity=True,
        limit=20
    )

    # Only returns documents the researcher can access
    # Even if ML finds more similar documents in restricted areas
    """)


if __name__ == "__main__":
    example_basic_neptune()
    example_gremlin_filter()
    example_fraud_detection()
    example_identity_graph()
    example_serverless_neptune()
    example_neptune_ml()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nTo use with Amazon Neptune:")
    print("  1. Create Neptune cluster in AWS Console")
    print("  2. Configure VPC/security groups for access")
    print("  3. Install: pip install ragguard gremlinpython")
    print("  4. Set up IAM credentials (aws configure)")
    print("  5. Update endpoint in the code")
