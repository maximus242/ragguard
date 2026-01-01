"""
Google Workspace Integration Example

This example demonstrates how to integrate RAGGuard with Google Workspace
to enforce permissions based on:
- Google Groups membership
- Google Drive file permissions
- Organizational unit structure
- User attributes from Google Workspace Directory

Prerequisites:
    pip install ragguard[google,qdrant]

Setup:
    1. Create a Google Cloud project
    2. Enable APIs:
       - Admin SDK API
       - Groups Settings API
       - Google Drive API
    3. Create service account with domain-wide delegation OR OAuth credentials
    4. Download credentials JSON file
"""

from ragguard import SecureRetriever, Policy
from ragguard.integrations.google_workspace import (
    GoogleWorkspaceResolver,
    GoogleDrivePermissionBuilder
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Example 1: Service Account Authentication (Recommended for server-to-server)
# ============================================================================

def example_service_account():
    """
    Use a service account with domain-wide delegation.

    This is the recommended approach for server-to-server applications
    where you need to access Google Workspace data on behalf of users.
    """

    # Initialize Google Workspace resolver with service account
    resolver = GoogleWorkspaceResolver.from_service_account(
        service_account_file='path/to/service-account.json',
        delegated_user='admin@company.com',  # Domain admin for impersonation
        domain='company.com'
    )

    # Create a simple policy (Google resolver adds group filtering)
    policy = Policy.from_dict({
        'version': '1',
        'default': 'deny',
        'rules': [
            {
                'name': 'allow_authenticated',
                'allow': {
                    'roles': ['employee', 'contractor']
                }
            }
        ]
    })

    # Initialize Qdrant client
    client = QdrantClient(":memory:")

    # Create collection
    client.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Index documents with Google Groups metadata
    documents = [
        {
            "id": 1,
            "text": "Q3 Financial Report",
            "vector": [0.1] * 384,
            "groups": ["finance@company.com", "executives@company.com"],
            "department": "finance"
        },
        {
            "id": 2,
            "text": "Product Roadmap 2024",
            "vector": [0.2] * 384,
            "groups": ["product@company.com", "engineering@company.com"],
            "department": "product"
        },
        {
            "id": 3,
            "text": "Company Handbook",
            "vector": [0.3] * 384,
            "groups": ["everyone@company.com"],
            "department": "hr"
        }
    ]

    client.upsert(
        collection_name="documents",
        points=[
            PointStruct(
                id=doc["id"],
                vector=doc["vector"],
                payload={
                    "text": doc["text"],
                    "groups": doc["groups"],
                    "department": doc["department"]
                }
            )
            for doc in documents
        ]
    )

    # Create custom filter builder that checks Google Groups
    from ragguard.filters.custom import CustomFilterBuilder
    from qdrant_client import models

    class GoogleGroupsFilterBuilder(CustomFilterBuilder):
        def __init__(self, workspace_resolver):
            self.resolver = workspace_resolver

        def build_filter(self, policy, user, backend):
            email = user.get('id') or user.get('email')
            if not email:
                return self._build_deny_all(backend)

            # Get user's Google Groups
            user_groups = self.resolver.get_user_groups(email)

            if backend == 'qdrant':
                # Filter documents where user is in one of the allowed groups
                return models.Filter(
                    should=[
                        models.FieldCondition(
                            key='groups',
                            match=models.MatchAny(any=[group])
                        )
                        for group in user_groups
                    ]
                )
            elif backend == 'pgvector':
                if not user_groups:
                    return ("WHERE FALSE", [])
                group_conditions = " OR ".join(["groups @> %s::jsonb" for _ in user_groups])
                params = [f'["{group}"]' for group in user_groups]
                return (f"WHERE {group_conditions}", params)

        def _build_deny_all(self, backend):
            if backend == 'qdrant':
                return models.Filter(must=[
                    models.FieldCondition(
                        key='_impossible',
                        match=models.MatchValue(value='denied')
                    )
                ])
            elif backend == 'pgvector':
                return ("WHERE FALSE", [])

    # Create retriever with Google Groups filtering
    filter_builder = GoogleGroupsFilterBuilder(resolver)

    retriever = SecureRetriever(
        client=client,
        collection="documents",
        policy=policy,
        embed_fn=lambda text: [0.1] * 384,  # Use real embeddings in production
        custom_filter_builder=filter_builder
    )

    # Search as different users
    print("Example 1: Service Account Authentication")
    print("=" * 60)

    # User in finance group
    finance_user = {
        "id": "alice@company.com",
        "roles": ["employee"]
    }
    results = retriever.search(
        "financial report",
        user=finance_user,
        limit=5
    )
    print(f"\nAlice (finance@company.com): {len(results)} results")
    for r in results:
        print(f"  - {r.payload['text']}")

    # User in product group
    product_user = {
        "id": "bob@company.com",
        "roles": ["employee"]
    }
    results = retriever.search(
        "roadmap",
        user=product_user,
        limit=5
    )
    print(f"\nBob (product@company.com): {len(results)} results")
    for r in results:
        print(f"  - {r.payload['text']}")


# Example 2: Google Drive Permissions
# ===================================

def example_google_drive_permissions():
    """
    Use Google Drive permissions to control document access.

    This approach checks actual Google Drive file permissions to determine
    who can access documents in your RAG system.
    """

    # Initialize resolver
    resolver = GoogleWorkspaceResolver.from_service_account(
        service_account_file='path/to/service-account.json',
        delegated_user='admin@company.com',
        domain='company.com'
    )

    # Create policy
    policy = Policy.from_dict({
        'version': '1',
        'default': 'deny',
        'rules': [{'name': 'drive', 'allow': {}}]
    })

    # Initialize Qdrant
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="drive_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Index documents with Google Drive metadata
    # In production, you'd sync this from Drive API
    documents = [
        {
            "id": 1,
            "text": "Project Proposal",
            "vector": [0.1] * 384,
            "drive_file_id": "1abc123...",
            "drive_permissions": {
                "users": ["alice@company.com"],
                "groups": ["product@company.com"],
                "domain": None,
                "anyone": False
            }
        },
        {
            "id": 2,
            "text": "Marketing Plan",
            "vector": [0.2] * 384,
            "drive_file_id": "2def456...",
            "drive_permissions": {
                "users": [],
                "groups": ["marketing@company.com"],
                "domain": "company.com",  # Anyone in domain
                "anyone": False
            }
        },
        {
            "id": 3,
            "text": "Public Blog Post",
            "vector": [0.3] * 384,
            "drive_file_id": "3ghi789...",
            "drive_permissions": {
                "users": [],
                "groups": [],
                "domain": None,
                "anyone": True  # Public
            }
        }
    ]

    client.upsert(
        collection_name="drive_docs",
        points=[
            PointStruct(
                id=doc["id"],
                vector=doc["vector"],
                payload={
                    "text": doc["text"],
                    "drive_file_id": doc["drive_file_id"],
                    "drive_permissions": doc["drive_permissions"]
                }
            )
            for doc in documents
        ]
    )

    # Use the built-in GoogleDrivePermissionBuilder
    drive_builder = GoogleDrivePermissionBuilder(
        workspace_resolver=resolver,
        file_id_field='drive_file_id',
        fallback_to_groups=True
    )

    retriever = SecureRetriever(
        client=client,
        collection="drive_docs",
        policy=policy,
        embed_fn=lambda text: [0.1] * 384,
        custom_filter_builder=drive_builder
    )

    print("\n\nExample 2: Google Drive Permissions")
    print("=" * 60)

    # Search as user with direct access
    alice = {"id": "alice@company.com", "roles": []}
    results = retriever.search("proposal", user=alice, limit=5)
    print(f"\nAlice: {len(results)} results")
    for r in results:
        print(f"  - {r.payload['text']}")

    # Search as user with domain-wide access
    anyone_in_domain = {"id": "charlie@company.com", "roles": []}
    results = retriever.search("marketing", user=anyone_in_domain, limit=5)
    print(f"\nCharlie (company.com domain): {len(results)} results")
    for r in results:
        print(f"  - {r.payload['text']}")


# Example 3: Organization-Based Access
# ====================================

def example_org_unit_access():
    """
    Use Google Workspace organizational units to control access.

    This approach uses the org structure from Google Workspace Directory
    to enforce hierarchical permissions.
    """

    from ragguard.policy.resolvers import OrganizationResolver

    # Initialize Google Workspace resolver
    workspace_resolver = GoogleWorkspaceResolver.from_service_account(
        service_account_file='path/to/service-account.json',
        delegated_user='admin@company.com',
        domain='company.com'
    )

    # Create org resolver that gets org units from Google Workspace
    def get_user_org_units(user_id):
        """Get user's org unit path from Google Workspace."""
        user_info = workspace_resolver.get_user_info(user_id)
        if not user_info:
            return []

        # Return org unit and all parent org units
        org_path = user_info.get('orgUnitPath', '/')

        # Parse org path: /Engineering/Backend -> ['/Engineering/Backend', '/Engineering', '/']
        org_units = []
        parts = org_path.split('/')[1:]  # Skip leading /

        for i in range(len(parts), 0, -1):
            org_units.append('/' + '/'.join(parts[:i]))
        org_units.append('/')  # Root

        return org_units

    org_resolver = OrganizationResolver(
        get_user_organizations=get_user_org_units,
        org_field="org_unit"
    )

    print("\n\nExample 3: Organization Unit Access")
    print("=" * 60)
    print("\nThis example demonstrates org-based filtering.")
    print("In production, integrate org_resolver with custom filter builder.")


# Example 4: Combined Google + Custom Logic
# =========================================

def example_combined_permissions():
    """
    Combine Google Workspace permissions with custom business logic.

    This shows how to use HybridFilterBuilder to combine standard Google
    permissions with additional custom requirements.
    """

    from ragguard.filters.custom import HybridFilterBuilder
    from qdrant_client import models

    # Initialize resolver
    resolver = GoogleWorkspaceResolver.from_service_account(
        service_account_file='path/to/service-account.json',
        delegated_user='admin@company.com',
        domain='company.com'
    )

    # Custom compliance check
    def add_compliance_filters(user):
        """
        Add additional compliance requirements:
        - User must have completed security training
        - User account must not be suspended
        - User must be accessing from approved location
        """

        # Check if user info is available
        user_info = resolver.get_user_info(user.get('id') or user.get('email'))

        if not user_info:
            # No user info = deny all
            return models.Filter(must=[
                models.FieldCondition(
                    key='_impossible',
                    match=models.MatchValue(value='denied')
                )
            ])

        # Check if user is suspended
        if user_info.get('suspended', False):
            return models.Filter(must=[
                models.FieldCondition(
                    key='_impossible',
                    match=models.MatchValue(value='denied')
                )
            ])

        # Add data classification filter based on user's department
        department = user_info.get('department', 'unknown')

        # Users can see public + their department's data
        return models.Filter(
            should=[
                # Public documents
                models.FieldCondition(
                    key='classification',
                    match=models.MatchValue(value='public')
                ),
                # Department-specific documents
                models.FieldCondition(
                    key='department',
                    match=models.MatchValue(value=department)
                )
            ]
        )

    # Create hybrid filter builder
    hybrid = HybridFilterBuilder(
        additional_filters={
            "qdrant": add_compliance_filters
        }
    )

    print("\n\nExample 4: Combined Permissions")
    print("=" * 60)
    print("\nThis example demonstrates hybrid filtering with compliance checks.")


# OAuth2 Example
# ==============

def example_oauth2():
    """
    Use OAuth2 credentials for user-context applications.

    This approach is suitable for applications where users authenticate
    with their own Google accounts.
    """

    # Initialize with OAuth2 credentials
    # Note: You need to handle the OAuth2 flow separately to get the credentials
    resolver = GoogleWorkspaceResolver.from_oauth_credentials(
        credentials_file='path/to/oauth-credentials.json',
        domain='company.com'
    )

    print("\n\nOAuth2 Example")
    print("=" * 60)
    print("\nInitialized resolver with OAuth2 credentials.")
    print("In production, implement OAuth2 flow to get user credentials.")


if __name__ == "__main__":
    print("Google Workspace Integration Examples")
    print("=" * 60)
    print("\nNOTE: These examples require valid Google Workspace credentials.")
    print("Update the paths to your service account or OAuth credentials files.\n")

    # Uncomment to run examples (requires valid credentials)
    # example_service_account()
    # example_google_drive_permissions()
    # example_org_unit_access()
    # example_combined_permissions()
    # example_oauth2()

    print("\nTo run these examples:")
    print("1. Set up Google Cloud project and enable required APIs")
    print("2. Create service account or OAuth credentials")
    print("3. Update credential paths in the code")
    print("4. Uncomment the example functions above")
