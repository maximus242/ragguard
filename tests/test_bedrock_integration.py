"""
Tests for AWS Bedrock Knowledge Base integration.

These tests use mocked boto3 clients to avoid requiring actual AWS credentials.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def mock_bedrock_response():
    """Mock Bedrock retrieve API response."""
    return {
        "retrievalResults": [
            {
                "content": {"text": "Machine learning best practices for production"},
                "metadata": {"department": "engineering", "visibility": "internal"},
                "score": 0.95,
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/doc1.pdf"}}
            },
            {
                "content": {"text": "Company-wide ML deployment guidelines"},
                "metadata": {"department": "engineering", "visibility": "public"},
                "score": 0.89,
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/doc2.pdf"}}
            },
            {
                "content": {"text": "HR policies and procedures"},
                "metadata": {"department": "hr", "visibility": "internal"},
                "score": 0.45,
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/doc3.pdf"}}
            }
        ]
    }


@pytest.fixture
def test_policy():
    """Create test policy."""
    from ragguard import Policy

    return Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            },
            {
                "name": "public-access",
                "allow": {
                    "conditions": ["document.visibility == 'public'"]
                }
            }
        ],
        "default": "deny"
    })


def test_bedrock_retriever_import():
    """Test that Bedrock retriever can be imported."""
    try:
        from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever
        assert BedrockKnowledgeBaseSecureRetriever is not None
    except ImportError as e:
        # boto3 not installed is acceptable for this test
        if "boto3" not in str(e):
            raise


@patch('ragguard.integrations.aws_bedrock.boto3')
def test_bedrock_retriever_initialization(mock_boto3, test_policy):
    """Test Bedrock retriever initialization."""
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3 session and client
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client

    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=test_policy
    )

    assert retriever.knowledge_base_id == "test-kb-123"
    assert retriever.region_name == "us-east-1"
    assert retriever.policy == test_policy


@patch('ragguard.integrations.aws_bedrock.boto3')
def test_bedrock_retrieve_with_policy(mock_boto3, test_policy, mock_bedrock_response):
    """Test Bedrock retrieval with policy filtering."""
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3 session and client
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client

    # Mock retrieve call
    mock_client.retrieve.return_value = mock_bedrock_response

    # Create retriever
    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=test_policy
    )

    # Retrieve as engineering user
    engineering_user = {"id": "alice", "department": "engineering"}
    results = retriever.retrieve(
        query="ML best practices",
        user=engineering_user,
        limit=10
    )

    # Should see engineering docs (2 results: both engineering docs)
    assert len(results) == 2
    assert all(r["metadata"]["department"] == "engineering" for r in results)
    assert mock_client.retrieve.called


@patch('ragguard.integrations.aws_bedrock.boto3')
def test_bedrock_retrieve_different_department(mock_boto3, test_policy, mock_bedrock_response):
    """Test that users only see their department's docs or public docs."""
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client
    mock_client.retrieve.return_value = mock_bedrock_response

    # Create retriever
    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=test_policy
    )

    # Retrieve as HR user
    hr_user = {"id": "bob", "department": "hr"}
    results = retriever.retrieve(
        query="ML best practices",
        user=hr_user,
        limit=10
    )

    # Should see: 1 public engineering doc + 1 HR doc = 2 results
    assert len(results) == 2


@patch('ragguard.integrations.aws_bedrock.boto3')
def test_bedrock_retrieve_no_policy(mock_boto3, mock_bedrock_response):
    """Test Bedrock retrieval without policy (returns all results)."""
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client
    mock_client.retrieve.return_value = mock_bedrock_response

    # Create retriever without policy
    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=None  # No policy
    )

    user = {"id": "alice"}
    results = retriever.retrieve(
        query="test",
        user=user,
        limit=10
    )

    # Should return all 3 results (no filtering)
    assert len(results) == 3


@patch('ragguard.integrations.aws_bedrock.boto3')
def test_bedrock_retrieve_with_custom_config(mock_boto3, test_policy, mock_bedrock_response):
    """Test Bedrock retrieval with custom configuration."""
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client
    mock_client.retrieve.return_value = mock_bedrock_response

    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=test_policy
    )

    user = {"id": "alice", "department": "engineering"}

    # Custom retrieval configuration
    custom_config = {
        "vectorSearchConfiguration": {
            "numberOfResults": 20,
            "overrideSearchType": "HYBRID"
        }
    }

    results = retriever.retrieve(
        query="test",
        user=user,
        limit=5,
        retrieval_configuration=custom_config
    )

    # Verify retrieve was called with config
    call_args = mock_client.retrieve.call_args
    assert "retrievalConfiguration" in call_args[1]
    assert call_args[1]["retrievalConfiguration"] == custom_config


@patch('ragguard.integrations.aws_bedrock.boto3')
def test_bedrock_result_standardization(mock_boto3, test_policy, mock_bedrock_response):
    """Test that Bedrock results are standardized correctly."""
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client
    mock_client.retrieve.return_value = mock_bedrock_response

    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=test_policy
    )

    user = {"id": "alice", "department": "engineering"}
    results = retriever.retrieve(query="test", user=user, limit=10)

    # Check standardized format
    assert len(results) > 0
    for result in results:
        assert "content" in result
        assert "metadata" in result
        assert "score" in result
        assert "location" in result
        assert "source" in result
        assert result["source"] == "bedrock-kb"


@patch('ragguard.integrations.aws_bedrock.boto3')
def test_bedrock_audit_logging(mock_boto3, test_policy, mock_bedrock_response):
    """Test that audit logging works with Bedrock retriever."""
    from unittest.mock import MagicMock

    from ragguard.audit import AuditLogger
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client
    mock_client.retrieve.return_value = mock_bedrock_response

    # Create mock audit logger
    audit_logger = MagicMock(spec=AuditLogger)

    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=test_policy,
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    results = retriever.retrieve(query="test", user=user, limit=10)

    # Verify audit logger was called
    assert audit_logger.log.called
    call_args = audit_logger.log.call_args[1]
    assert call_args["user"] == user
    assert call_args["action"] == "bedrock_retrieve"
    assert call_args["resource"] == "test-kb-123"
    assert call_args["allowed"] is True


@patch('ragguard.integrations.aws_bedrock.boto3')
@pytest.mark.asyncio
async def test_bedrock_async_retrieve(mock_boto3, test_policy, mock_bedrock_response):
    """Test async retrieval."""
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Mock boto3
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_session.client.return_value = mock_client
    mock_client.retrieve.return_value = mock_bedrock_response

    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="test-kb-123",
        region_name="us-east-1",
        policy=test_policy
    )

    user = {"id": "alice", "department": "engineering"}
    results = await retriever.retrieve_async(query="test", user=user, limit=10)

    assert len(results) > 0
    assert mock_client.retrieve.called


def test_bedrock_missing_boto3():
    """Test that helpful error is raised when boto3 is not installed."""
    # Temporarily hide boto3
    import sys
    boto3_module = sys.modules.get('boto3')

    try:
        # Remove boto3 from modules
        if 'ragguard.integrations.aws_bedrock' in sys.modules:
            del sys.modules['ragguard.integrations.aws_bedrock']
        sys.modules['boto3'] = None

        # Try to import
        with pytest.raises(ImportError, match="boto3 is required"):
            from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever
            BedrockKnowledgeBaseSecureRetriever(
                knowledge_base_id="test",
                region_name="us-east-1"
            )

    finally:
        # Restore boto3
        if boto3_module:
            sys.modules['boto3'] = boto3_module
        elif 'boto3' in sys.modules:
            del sys.modules['boto3']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
