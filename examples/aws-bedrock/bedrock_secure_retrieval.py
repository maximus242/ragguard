#!/usr/bin/env python3
"""
AWS Bedrock Knowledge Base Secure Retrieval Example

This example demonstrates how to use RAGGuard with AWS Bedrock Knowledge Bases
to implement policy-based access control for document retrieval.

Prerequisites:
    1. AWS account with Bedrock access
    2. Bedrock Knowledge Base created and configured
    3. AWS credentials configured (via ~/.aws/credentials or environment variables)
    4. boto3 installed: pip install boto3

Setup:
    1. Create a Bedrock Knowledge Base in AWS console
    2. Load documents with metadata (department, visibility, etc.)
    3. Note the Knowledge Base ID
    4. Configure AWS credentials

Run:
    export AWS_PROFILE=your-profile  # or use default
    python bedrock_secure_retrieval.py
"""

from ragguard import Policy
from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever
from ragguard.audit import AuditLogger
import json


def create_policy():
    """
    Create a RAGGuard policy for document access control.

    This policy allows access if:
    - User's department matches document's department, OR
    - Document is marked as public
    """
    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "department-access",
                "allow": {
                    "conditions": [
                        "user.department == document.department"
                    ]
                }
            },
            {
                "name": "public-documents",
                "allow": {
                    "conditions": [
                        "document.visibility == 'public'"
                    ]
                }
            },
            {
                "name": "admin-access",
                "allow": {
                    "roles": ["admin"]
                }
            }
        ],
        "default": "deny"
    })


def main():
    """Main example function."""

    # Configuration
    KNOWLEDGE_BASE_ID = "YOUR_KNOWLEDGE_BASE_ID"  # Replace with your KB ID
    AWS_REGION = "us-east-1"  # Replace with your region

    # Create policy
    print("Creating access control policy...")
    policy = create_policy()

    # Create audit logger
    audit_logger = AuditLogger(output="bedrock_audit.log")

    # Create secure retriever
    print(f"Connecting to Bedrock Knowledge Base: {KNOWLEDGE_BASE_ID}")
    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        region_name=AWS_REGION,
        policy=policy,
        audit_logger=audit_logger
    )

    # Example 1: Engineering employee searches
    print("\n" + "="*80)
    print("Example 1: Engineering Employee Search")
    print("="*80)

    engineering_user = {
        "id": "alice@company.com",
        "name": "Alice",
        "department": "engineering",
        "roles": ["employee"]
    }

    results = retriever.retrieve(
        query="What are the latest machine learning developments?",
        user=engineering_user,
        limit=5
    )

    print(f"\nUser: {engineering_user['name']} ({engineering_user['department']})")
    print(f"Results: {len(results)} documents")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['content'][:200]}...")
        print(f"   Metadata: {result['metadata']}")
        print(f"   Score: {result['score']:.4f}")

    # Example 2: HR employee searches (different department)
    print("\n" + "="*80)
    print("Example 2: HR Employee Search (Different Department)")
    print("="*80)

    hr_user = {
        "id": "bob@company.com",
        "name": "Bob",
        "department": "hr",
        "roles": ["employee"]
    }

    results = retriever.retrieve(
        query="What are the latest machine learning developments?",
        user=hr_user,
        limit=5
    )

    print(f"\nUser: {hr_user['name']} ({hr_user['department']})")
    print(f"Results: {len(results)} documents")
    print("(Only sees public documents or HR-specific docs)")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['content'][:200]}...")
        print(f"   Metadata: {result['metadata']}")
        print(f"   Visibility: {result['metadata'].get('visibility', 'department-specific')}")

    # Example 3: Admin user (full access)
    print("\n" + "="*80)
    print("Example 3: Admin User (Full Access)")
    print("="*80)

    admin_user = {
        "id": "admin@company.com",
        "name": "Admin",
        "department": "operations",
        "roles": ["admin"]
    }

    results = retriever.retrieve(
        query="What are the latest machine learning developments?",
        user=admin_user,
        limit=5
    )

    print(f"\nUser: {admin_user['name']} (role: admin)")
    print(f"Results: {len(results)} documents")
    print("(Sees all documents)")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['content'][:200]}...")
        print(f"   Department: {result['metadata'].get('department', 'N/A')}")

    # Example 4: Async retrieval
    print("\n" + "="*80)
    print("Example 4: Async Retrieval")
    print("="*80)

    async def async_search():
        results = await retriever.retrieve_async(
            query="What are best practices for security?",
            user=engineering_user,
            limit=3
        )
        return results

    import asyncio
    async_results = asyncio.run(async_search())

    print(f"Async results: {len(async_results)} documents")

    print("\n" + "="*80)
    print("Audit log written to: bedrock_audit.log")
    print("="*80)


def example_with_custom_retrieval_config():
    """
    Example showing custom Bedrock retrieval configuration.

    AWS Bedrock supports various retrieval configurations including:
    - Vector search configuration
    - Number of results
    - Override search type
    """

    KNOWLEDGE_BASE_ID = "YOUR_KNOWLEDGE_BASE_ID"
    policy = create_policy()

    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        region_name="us-east-1",
        policy=policy
    )

    user = {"id": "alice", "department": "engineering", "roles": ["employee"]}

    # Custom retrieval configuration
    retrieval_config = {
        "vectorSearchConfiguration": {
            "numberOfResults": 20,  # Retrieve more results before filtering
            "overrideSearchType": "HYBRID"  # Use hybrid search
        }
    }

    results = retriever.retrieve(
        query="machine learning deployment",
        user=user,
        limit=5,
        retrieval_configuration=retrieval_config
    )

    print(f"Retrieved {len(results)} documents with custom config")


def example_with_different_aws_credentials():
    """
    Example showing different AWS credential methods.
    """

    policy = create_policy()

    # Method 1: Use AWS profile
    retriever1 = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="YOUR_KB_ID",
        region_name="us-east-1",
        policy=policy,
        profile_name="my-aws-profile"
    )

    # Method 2: Use explicit credentials
    retriever2 = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="YOUR_KB_ID",
        region_name="us-east-1",
        policy=policy,
        aws_access_key_id="YOUR_ACCESS_KEY",
        aws_secret_access_key="YOUR_SECRET_KEY"
    )

    # Method 3: Use session token (for temporary credentials)
    retriever3 = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="YOUR_KB_ID",
        region_name="us-east-1",
        policy=policy,
        aws_access_key_id="YOUR_ACCESS_KEY",
        aws_secret_access_key="YOUR_SECRET_KEY",
        aws_session_token="YOUR_SESSION_TOKEN"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("1. Replace YOUR_KNOWLEDGE_BASE_ID with your actual KB ID")
        print("2. Configure AWS credentials")
        print("3. Install boto3: pip install boto3")
