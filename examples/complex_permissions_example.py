"""
Complex Real-World Permission Scenarios

This example shows how to handle advanced permission systems:
- Multi-tenant organizations
- Role hierarchies
- ACL-based permissions
- External permission systems
- Custom document schemas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import random

from ragguard.policy import Policy
from ragguard.policy.resolvers import (
    RoleHierarchyResolver,
    OrganizationResolver,
    CompositeResolver
)
from ragguard.filters.custom import (
    ACLFilterBuilder,
    LambdaFilterBuilder,
    HybridFilterBuilder
)

print("=" * 70)
print("Complex Real-World Permission Scenarios")
print("=" * 70)

# ============================================================================
# SCENARIO 1: Role Hierarchy with Organization Boundaries
# ============================================================================
print("\n" + "=" * 70)
print("SCENARIO 1: Multi-Tenant with Role Hierarchies")
print("=" * 70)

print("\n1. Setting up multi-tenant database...")

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="company_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def fake_embed(text):
    random.seed(hash(text) % 2**32)
    return [random.random() for _ in range(384)]

# Documents with organization and ACL
docs = [
    {
        "id": 1,
        "text": "Acme Corp Q4 Financial Results",
        "organization_id": "acme",
        "department": "finance",
        "acl": {
            "users": ["alice@acme.com"],
            "groups": ["finance-team", "executives"],
            "public": False
        }
    },
    {
        "id": 2,
        "text": "Acme Engineering Roadmap 2024",
        "organization_id": "acme",
        "department": "engineering",
        "acl": {
            "users": [],
            "groups": ["engineering-team"],
            "public": False
        }
    },
    {
        "id": 3,
        "text": "GloboCorp Product Launch Plan",
        "organization_id": "globocorp",
        "department": "product",
        "acl": {
            "users": ["bob@globocorp.com"],
            "groups": ["product-team"],
            "public": False
        }
    },
    {
        "id": 4,
        "text": "Industry Best Practices White Paper",
        "organization_id": None,
        "department": None,
        "acl": {
            "users": [],
            "groups": [],
            "public": True  # Public document
        }
    }
]

client.upsert(
    collection_name="company_docs",
    points=[
        PointStruct(id=doc["id"], vector=fake_embed(doc["text"]), payload=doc)
        for doc in docs
    ]
)

print(f"   Loaded {len(docs)} documents across 2 organizations")

# ============================================================================
# Simulated External Services
# ============================================================================

# Simulated organization membership database
USER_ORGANIZATIONS = {
    "alice@acme.com": ["acme"],
    "bob@globocorp.com": ["globocorp"],
    "charlie@acme.com": ["acme"],
    "diana@both.com": ["acme", "globocorp"],  # Works for both!
}

# Simulated group membership
USER_GROUPS = {
    "alice@acme.com": ["finance-team", "executives"],
    "bob@globocorp.com": ["product-team"],
    "charlie@acme.com": ["engineering-team"],
    "diana@both.com": ["finance-team", "product-team"],
}

# Role hierarchy: managers inherit employee permissions
ROLE_HIERARCHY = {
    "director": ["manager", "employee"],
    "manager": ["employee"],
    "employee": []
}

def get_user_organizations(user_id: str) -> list[str]:
    """Simulated API call to get user's organizations."""
    return USER_ORGANIZATIONS.get(user_id, [])

def get_user_groups(user: dict) -> list[str]:
    """Get user's groups."""
    return USER_GROUPS.get(user.get("id"), [])

print("\n2. Setting up permission resolvers...")

# Create resolvers
org_resolver = OrganizationResolver(
    get_user_organizations=get_user_organizations,
    org_field="organization_id"
)

role_resolver = RoleHierarchyResolver(hierarchy=ROLE_HIERARCHY)

# Combine: user must be in org AND have correct role
composite_resolver = CompositeResolver([org_resolver, role_resolver])

print("   ✓ Organization resolver")
print("   ✓ Role hierarchy resolver")
print("   ✓ Composite resolver (AND logic)")

# ============================================================================
# SCENARIO 1 Testing
# ============================================================================

print("\n3. Creating ACL-based filter builder...")

acl_builder = ACLFilterBuilder(
    acl_field="acl",
    get_user_groups=get_user_groups
)

# Note: We'd need to modify SecureRetriever to accept custom filter builder
# For now, demonstrate the filter building directly

print("\n4. Testing permission scenarios:")

# Test Alice (Acme, executive)
print("\n  a) Alice (Acme executive):")
alice = {
    "id": "alice@acme.com",
    "roles": ["manager"],
    "organization": "acme"
}
filter_alice = acl_builder.build_filter(None, alice, "qdrant")
print(f"     Can access: Acme docs + public + docs in her groups")
print(f"     Filter built: {type(filter_alice).__name__}")

# Test Bob (GloboCorp product)
print("\n  b) Bob (GloboCorp product manager):")
bob = {
    "id": "bob@globocorp.com",
    "roles": ["manager"],
    "organization": "globocorp"
}
filter_bob = acl_builder.build_filter(None, bob, "qdrant")
print(f"     Can access: GloboCorp docs + public")
print(f"     Cannot access: Acme docs (different org)")

# Test Diana (works for both!)
print("\n  c) Diana (consultant for both companies):")
diana = {
    "id": "diana@both.com",
    "roles": ["employee"],
    "organizations": ["acme", "globocorp"]
}
filter_diana = acl_builder.build_filter(None, diana, "qdrant")
print(f"     Can access: Docs from BOTH orgs + public")
print(f"     This is the multi-tenant scenario!")

# ============================================================================
# SCENARIO 2: Custom Lambda Filter for Complex Business Logic
# ============================================================================

print("\n" + "=" * 70)
print("SCENARIO 2: Custom Business Logic with Lambda Filters")
print("=" * 70)

print("\nScenario: Sales team can only see leads in their territory")

# Simulated territory assignments
SALES_TERRITORIES = {
    "sales1@company.com": ["US-West", "US-Southwest"],
    "sales2@company.com": ["US-East", "US-Southeast"],
    "sales3@company.com": ["Europe"],
}

def build_territory_filter(policy, user):
    """Custom filter for sales territory restrictions."""
    try:
        from qdrant_client import models
    except ImportError:
        return None

    # Get user's territories
    user_id = user.get("id")
    territories = SALES_TERRITORIES.get(user_id, [])

    if not territories:
        # No territories = no access
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="_impossible",
                    match=models.MatchValue(value="impossible")
                )
            ]
        )

    # Build filter: territory IN user's territories
    return models.Filter(
        should=[
            models.FieldCondition(
                key="territory",
                match=models.MatchValue(value=territory)
            )
            for territory in territories
        ]
    )

lambda_builder = LambdaFilterBuilder(
    qdrant=build_territory_filter
)

sales_user = {"id": "sales1@company.com", "role": "sales"}
territory_filter = lambda_builder.build_filter(None, sales_user, "qdrant")
print(f"\n✓ Territory filter built for sales1@company.com")
print(f"  Territories: US-West, US-Southwest")

# ============================================================================
# SCENARIO 3: Hybrid - Standard Policy + Custom Extensions
# ============================================================================

print("\n" + "=" * 70)
print("SCENARIO 3: Hybrid Filter (Standard + Custom)")
print("=" * 70)

print("\nScenario: Standard role-based policy + real-time security clearance check")

def add_security_clearance_filter(user):
    """Add filter based on user's security clearance level."""
    try:
        from qdrant_client import models
    except ImportError:
        return None

    clearance = user.get("security_clearance", 0)

    # User can only see docs at or below their clearance level
    return models.Filter(
        should=[
            models.FieldCondition(
                key="clearance_required",
                match=models.MatchValue(value=level)
            )
            for level in range(0, clearance + 1)
        ]
    )

hybrid_builder = HybridFilterBuilder(
    additional_filters={
        "qdrant": add_security_clearance_filter
    }
)

# Create a simple policy
simple_policy = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "employee-access",
            "allow": {"roles": ["employee", "manager"]}
        }
    ],
    "default": "deny"
})

clearance_user = {
    "id": "analyst@company.com",
    "roles": ["employee"],
    "security_clearance": 2  # Can see clearance 0, 1, 2
}

hybrid_filter = hybrid_builder.build_filter(simple_policy, clearance_user, "qdrant")
print(f"\n✓ Hybrid filter combines:")
print(f"  - Standard role check (employee)")
print(f"  - Custom clearance check (level ≤ 2)")

# ============================================================================
# SCENARIO 4: Field Mapping for Different Schema
# ============================================================================

print("\n" + "=" * 70)
print("SCENARIO 4: Field Mapping for Legacy Systems")
print("=" * 70)

print("""
Scenario: Your documents use different field names:
  Policy expects: 'department', 'confidential'
  Your docs have:  'dept_code', 'classification_level'

Solution: Use FieldMappingFilterBuilder
""")

print("""
from filters.custom import FieldMappingFilterBuilder

mapping = {
    "department": "dept_code",
    "confidential": lambda x: x == "SECRET",  # Transform boolean
    "visibility": "access_level"
}

mapped_builder = FieldMappingFilterBuilder(
    field_mapping=mapping
)

# Now your policy works with your schema!
""")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Real-World Permission Patterns")
print("=" * 70)

print("""
✓ Multi-Tenant Organizations
  - OrganizationResolver
  - Users belong to multiple orgs
  - Org boundaries enforced

✓ Role Hierarchies
  - RoleHierarchyResolver
  - directors > managers > employees
  - Inherited permissions

✓ ACL-Based Permissions
  - ACLFilterBuilder
  - Explicit user/group lists
  - Document-level ACLs

✓ Custom Business Logic
  - LambdaFilterBuilder
  - Territory restrictions
  - Time windows
  - Dynamic rules

✓ Hybrid Approaches
  - HybridFilterBuilder
  - Standard policy + custom extensions
  - Security clearances
  - Compliance checks

✓ Schema Flexibility
  - FieldMappingFilterBuilder
  - Map policy fields to your schema
  - Value transformations
  - Legacy system support
""")

print("\nKey Insight:")
print("  RAGGuard provides building blocks. Combine them to match")
print("  YOUR permission system, not the other way around!")

print("\n" + "=" * 70)
