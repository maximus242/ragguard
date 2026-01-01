"""
Tests for AWS Bedrock Knowledge Bases integration.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from ragguard.policy.models import AllowConditions, Policy, Rule


@pytest.fixture
def sample_policy():
    """Create a sample policy for testing."""
    return Policy(
        version="1",
        default="deny",
        rules=[
            Rule(
                name="dept-access",
                allow=AllowConditions(
                    conditions=["user.department == document.department"]
                ),
            ),
            Rule(
                name="public-docs",
                match={"visibility": "public"},
                allow=AllowConditions(everyone=True),
            ),
        ],
    )


@pytest.fixture
def mock_bedrock_response():
    """Create a mock Bedrock retrieve response."""
    return {
        "retrievalResults": [
            {
                "content": {"text": "Document content 1"},
                "location": {"s3Location": {"uri": "s3://bucket/doc1.txt"}},
                "score": 0.95,
                "metadata": {
                    "department": "engineering",
                    "visibility": "internal",
                },
            },
            {
                "content": {"text": "Document content 2"},
                "location": {"s3Location": {"uri": "s3://bucket/doc2.txt"}},
                "score": 0.85,
                "metadata": {
                    "department": "marketing",
                    "visibility": "public",
                },
            },
            {
                "content": {"text": "Document content 3"},
                "location": {"s3Location": {"uri": "s3://bucket/doc3.txt"}},
                "score": 0.75,
                "metadata": {
                    "department": "engineering",
                    "visibility": "internal",
                },
            },
        ]
    }


class TestBedrockKnowledgeBaseRetriever:
    """Test AWS Bedrock Knowledge Base secure retriever."""

    def test_import_without_boto3(self):
        """Test that import works even without boto3."""
        # This should not raise
        from ragguard.integrations import aws_bedrock

        assert hasattr(aws_bedrock, "BedrockKnowledgeBaseSecureRetriever")

    @patch("ragguard.integrations.aws_bedrock.boto3")
    def test_initialization(self, mock_boto3, sample_policy):
        """Test retriever initialization."""
        from ragguard.integrations.aws_bedrock import (
            BedrockKnowledgeBaseSecureRetriever,
        )

        # Setup mock session and client
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client

        retriever = BedrockKnowledgeBaseSecureRetriever(
            knowledge_base_id="test-kb-id",
            region_name="us-east-1",
            policy=sample_policy,
        )

        assert retriever.knowledge_base_id == "test-kb-id"
        mock_boto3.Session.assert_called()

    @patch("ragguard.integrations.aws_bedrock.boto3")
    def test_retrieve_with_policy_filtering(
        self, mock_boto3, sample_policy, mock_bedrock_response
    ):
        """Test that retrieval applies policy filtering."""
        from ragguard.integrations.aws_bedrock import (
            BedrockKnowledgeBaseSecureRetriever,
        )

        # Setup mock session and client
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        mock_client.retrieve.return_value = mock_bedrock_response

        retriever = BedrockKnowledgeBaseSecureRetriever(
            knowledge_base_id="test-kb-id",
            region_name="us-east-1",
            policy=sample_policy,
        )

        # User from engineering department
        user = {"id": "alice", "department": "engineering"}

        results = retriever.retrieve(
            query="test query",
            user=user,
            limit=10,
        )

        # Should only return docs from engineering department or public docs
        # Doc 1: engineering - allowed (dept match)
        # Doc 2: marketing/public - allowed (public)
        # Doc 3: engineering - allowed (dept match)
        assert len(results) == 3  # All should be allowed

    @patch("ragguard.integrations.aws_bedrock.boto3")
    def test_retrieve_filters_unauthorized_docs(self, mock_boto3, sample_policy):
        """Test that unauthorized documents are filtered out."""
        from ragguard.integrations.aws_bedrock import (
            BedrockKnowledgeBaseSecureRetriever,
        )

        # Setup mock with only internal docs from other departments
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        mock_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Secret HR doc"},
                    "location": {"s3Location": {"uri": "s3://bucket/hr.txt"}},
                    "score": 0.95,
                    "metadata": {
                        "department": "hr",
                        "visibility": "internal",
                    },
                },
            ]
        }

        retriever = BedrockKnowledgeBaseSecureRetriever(
            knowledge_base_id="test-kb-id",
            region_name="us-east-1",
            policy=sample_policy,
        )

        # User from engineering - should not see HR internal docs
        user = {"id": "alice", "department": "engineering"}

        results = retriever.retrieve(
            query="test query",
            user=user,
            limit=10,
        )

        # HR internal doc should be filtered out
        assert len(results) == 0

    @patch("ragguard.integrations.aws_bedrock.boto3")
    def test_retrieve_without_policy(self, mock_boto3, mock_bedrock_response):
        """Test retrieval without policy returns all results."""
        from ragguard.integrations.aws_bedrock import (
            BedrockKnowledgeBaseSecureRetriever,
        )

        # Setup mock session and client
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        mock_client.retrieve.return_value = mock_bedrock_response

        retriever = BedrockKnowledgeBaseSecureRetriever(
            knowledge_base_id="test-kb-id",
            region_name="us-east-1",
            policy=None,  # No policy
        )

        user = {"id": "alice", "department": "engineering"}

        results = retriever.retrieve(
            query="test query",
            user=user,
            limit=10,
        )

        # All results should be returned
        assert len(results) == 3


class TestBedrockPolicyFiltering:
    """Test policy filtering logic used by Bedrock integration."""

    def test_filtering_by_department(self, sample_policy):
        """Test that documents are filtered by department."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(sample_policy)

        user = {"id": "alice", "department": "engineering"}

        # Engineering doc should be allowed
        doc_eng = {"department": "engineering", "visibility": "internal"}
        assert engine.evaluate(user, doc_eng) is True

        # Marketing doc should be denied
        doc_marketing = {"department": "marketing", "visibility": "internal"}
        assert engine.evaluate(user, doc_marketing) is False

    def test_public_docs_accessible(self, sample_policy):
        """Test that public docs are accessible to everyone."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(sample_policy)

        # Any user should be able to access public docs
        user = {"id": "bob", "department": "sales"}
        doc_public = {"visibility": "public", "department": "marketing"}

        assert engine.evaluate(user, doc_public) is True


class TestBedrockMetadataExtraction:
    """Test metadata extraction from Bedrock results."""

    @patch("ragguard.integrations.aws_bedrock.boto3")
    def test_extracts_metadata_correctly(self, mock_boto3, sample_policy):
        """Test that metadata is correctly extracted from results."""
        from ragguard.integrations.aws_bedrock import (
            BedrockKnowledgeBaseSecureRetriever,
        )

        # Setup mock
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        mock_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Test content"},
                    "location": {"s3Location": {"uri": "s3://bucket/doc.txt"}},
                    "score": 0.9,
                    "metadata": {
                        "department": "engineering",
                        "author": "alice",
                        "created": "2024-01-01",
                    },
                },
            ]
        }

        retriever = BedrockKnowledgeBaseSecureRetriever(
            knowledge_base_id="test-kb-id",
            region_name="us-east-1",
            policy=sample_policy,
        )

        user = {"id": "alice", "department": "engineering"}

        results = retriever.retrieve(query="test", user=user, limit=5)

        assert len(results) == 1
        # Verify result has expected structure
        result = results[0]
        assert "content" in result or "text" in result


class TestBedrockRetrieverAPICalls:
    """Test that the retriever makes correct API calls."""

    @patch("ragguard.integrations.aws_bedrock.boto3")
    def test_retrieve_api_call(self, mock_boto3, sample_policy):
        """Test that retrieve makes correct API call."""
        from ragguard.integrations.aws_bedrock import (
            BedrockKnowledgeBaseSecureRetriever,
        )

        # Setup mock
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        mock_client.retrieve.return_value = {"retrievalResults": []}

        retriever = BedrockKnowledgeBaseSecureRetriever(
            knowledge_base_id="test-kb-id",
            region_name="us-east-1",
            policy=sample_policy,
        )

        user = {"id": "alice", "department": "engineering"}

        retriever.retrieve(query="test query", user=user, limit=5)

        # Verify retrieve was called with correct parameters
        mock_client.retrieve.assert_called_once()
        call_args = mock_client.retrieve.call_args
        assert call_args.kwargs["knowledgeBaseId"] == "test-kb-id"
        assert call_args.kwargs["retrievalQuery"]["text"] == "test query"
