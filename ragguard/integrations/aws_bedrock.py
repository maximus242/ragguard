"""
AWS Bedrock Knowledge Bases Integration for RAGGuard.

This module provides secure retrieval for AWS Bedrock Knowledge Bases with
policy-based access control.

AWS Bedrock Knowledge Bases support multiple vector databases:
- Amazon OpenSearch Serverless
- Pinecone
- Amazon Aurora (PostgreSQL with pgvector)
- Redis Enterprise Cloud

Example:
    ```python
    from ragguard import Policy
    from ragguard.integrations.aws_bedrock import BedrockKnowledgeBaseSecureRetriever

    # Create policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })

    # Create secure retriever
    retriever = BedrockKnowledgeBaseSecureRetriever(
        knowledge_base_id="YOUR_KB_ID",
        region_name="us-east-1",
        policy=policy
    )

    # Search with user context
    user = {"id": "alice", "department": "engineering"}
    results = retriever.retrieve(
        query="What are the latest ML developments?",
        user=user,
        limit=5
    )
    ```

Requirements:
    - boto3>=1.28.0
    - botocore>=1.31.0
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None

from ..audit import AuditLogger
from ..policy import Policy

logger = logging.getLogger(__name__)


class BedrockKnowledgeBaseSecureRetriever:
    """
    Secure retriever for AWS Bedrock Knowledge Bases with policy-based access control.

    This retriever wraps AWS Bedrock's Knowledge Base retrieve API and applies
    RAGGuard policies to filter results based on user context.

    Attributes:
        knowledge_base_id: AWS Bedrock Knowledge Base ID
        region_name: AWS region (e.g., 'us-east-1')
        policy: RAGGuard policy for access control
        bedrock_agent: boto3 bedrock-agent-runtime client
        audit_logger: Optional audit logger
    """

    def __init__(
        self,
        knowledge_base_id: str,
        region_name: str = "us-east-1",
        policy: Optional[Policy] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        profile_name: Optional[str] = None,
        audit_logger: Optional[AuditLogger] = None,
        max_retries: int = 3
    ):
        """
        Initialize Bedrock Knowledge Base secure retriever.

        Args:
            knowledge_base_id: AWS Bedrock Knowledge Base ID
            region_name: AWS region name
            policy: RAGGuard policy for access control
            aws_access_key_id: AWS access key (optional, uses default credentials if not provided)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
            profile_name: AWS profile name (optional)
            audit_logger: Optional audit logger
            max_retries: Maximum number of retry attempts for API calls

        Raises:
            ImportError: If boto3 is not installed
        """
        if boto3 is None:
            raise ImportError(
                "boto3 is required for AWS Bedrock integration. "
                "Install it with: pip install boto3"
            )

        self.knowledge_base_id = knowledge_base_id
        self.region_name = region_name
        self.policy = policy
        self.audit_logger = audit_logger

        # Create boto3 session
        session_kwargs = {"region_name": region_name}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        elif aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                session_kwargs["aws_session_token"] = aws_session_token

        session = boto3.Session(**session_kwargs)

        # Create bedrock-agent-runtime client
        client_kwargs = {
            "service_name": "bedrock-agent-runtime",
            "region_name": region_name
        }

        if max_retries and boto3:  # Only configure retries if boto3 is available
            try:
                from botocore.config import Config
                client_kwargs["config"] = Config(
                    retries={"max_attempts": max_retries, "mode": "adaptive"}
                )
            except ImportError:
                # If botocore.config isn't available, proceed without retry config
                pass

        self.bedrock_agent = session.client(**client_kwargs)

    def retrieve(
        self,
        query: str,
        user: Dict[str, Any],
        limit: int = 10,
        retrieval_configuration: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from Bedrock Knowledge Base with policy enforcement.

        This method:
        1. Calls AWS Bedrock Knowledge Base retrieve API
        2. Applies RAGGuard policy to filter results
        3. Returns only documents the user is authorized to access

        Args:
            query: Natural language query string
            user: User context dictionary (e.g., {"id": "alice", "department": "eng"})
            limit: Maximum number of results to return
            retrieval_configuration: Optional Bedrock retrieval configuration
            **kwargs: Additional arguments passed to retrieve API

        Returns:
            List of authorized documents with metadata

        Example:
            >>> user = {"id": "alice", "department": "engineering", "roles": ["employee"]}
            >>> results = retriever.retrieve(
            ...     query="What are the ML best practices?",
            ...     user=user,
            ...     limit=5
            ... )
        """
        try:
            # Build retrieval request
            retrieve_params = {
                "knowledgeBaseId": self.knowledge_base_id,
                "retrievalQuery": {"text": query}
            }

            if retrieval_configuration:
                retrieve_params["retrievalConfiguration"] = retrieval_configuration

            # Merge additional kwargs
            retrieve_params.update(kwargs)

            # Call Bedrock retrieve API
            response = self.bedrock_agent.retrieve(**retrieve_params)

            # Extract results
            raw_results = response.get("retrievalResults", [])

            # Apply policy filtering if policy is provided
            if self.policy:
                filtered_results = self._filter_results(raw_results, user)
            else:
                filtered_results = raw_results

            # Limit results
            filtered_results = filtered_results[:limit]

            # Convert to standard format
            standardized_results = self._standardize_results(filtered_results)

            # Audit log
            if self.audit_logger:
                self.audit_logger.log(
                    user=user,
                    action="bedrock_retrieve",
                    resource=self.knowledge_base_id,
                    allowed=True,
                    metadata={
                        "query": query,
                        "total_results": len(raw_results),
                        "filtered_results": len(filtered_results)
                    }
                )

            return standardized_results

        except (ClientError, BotoCoreError) as e:
            logger.error(f"AWS Bedrock API error: {e}")

            if self.audit_logger:
                self.audit_logger.log(
                    user=user,
                    action="bedrock_retrieve",
                    resource=self.knowledge_base_id,
                    allowed=False,
                    metadata={"error": str(e)}
                )

            raise

    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        user: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter results based on RAGGuard policy.

        Args:
            results: Raw results from Bedrock
            user: User context

        Returns:
            Filtered results
        """
        from ..policy.engine import PolicyEngine

        engine = PolicyEngine(self.policy)
        filtered = []

        for result in results:
            # Extract metadata from result
            document_metadata = result.get("metadata", {})

            # Convert Bedrock metadata format to flat dict
            document = self._extract_metadata(document_metadata)

            # Evaluate policy
            if engine.evaluate(user, document):
                filtered.append(result)

        return filtered

    def _extract_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from Bedrock format to flat dictionary.

        AWS Bedrock returns metadata as a dictionary. This method
        converts it to the format expected by RAGGuard policies.

        Args:
            metadata: Bedrock metadata dictionary

        Returns:
            Flat metadata dictionary
        """
        # Bedrock metadata is already in dict format
        # Just return it as-is for policy evaluation
        return metadata or {}

    def _standardize_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert Bedrock results to standardized format.

        Args:
            results: Bedrock retrieval results

        Returns:
            Standardized result format
        """
        standardized = []

        for result in results:
            standardized.append({
                "content": result.get("content", {}).get("text", ""),
                "metadata": result.get("metadata", {}),
                "score": result.get("score", 0.0),
                "location": result.get("location", {}),
                "source": "bedrock-kb"
            })

        return standardized

    async def retrieve_async(
        self,
        query: str,
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Async version of retrieve.

        Note: This uses the synchronous boto3 client in a thread pool.
        For true async, consider using aioboto3.

        Args:
            query: Query string
            user: User context
            limit: Maximum results
            **kwargs: Additional arguments

        Returns:
            List of authorized documents
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.retrieve(query, user, limit, **kwargs)
            )


class BedrockKnowledgeBaseFilterBuilder:
    """
    Helper to build retrieval filters for Bedrock Knowledge Bases.

    Note: As of 2024, AWS Bedrock Knowledge Bases have limited support
    for metadata filtering. This builder helps construct filters where
    supported.
    """

    @staticmethod
    def build_metadata_filter(
        policy: Policy,
        user: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Build metadata filter for Bedrock retrieval configuration.

        Args:
            policy: RAGGuard policy
            user: User context

        Returns:
            Bedrock-compatible metadata filter (if supported)
        """
        # AWS Bedrock Knowledge Bases currently have limited metadata filtering
        # capabilities. RAGGuard applies post-retrieval filtering instead.
        # If AWS expands pre-retrieval filtering in the future, this method
        # can be updated to generate Bedrock-compatible filter expressions.
        logger.warning(
            "AWS Bedrock Knowledge Bases currently have limited metadata filtering support. "
            "RAGGuard will apply post-retrieval filtering."
        )
        return None


__all__ = [
    "BedrockKnowledgeBaseFilterBuilder",
    "BedrockKnowledgeBaseSecureRetriever"
]
