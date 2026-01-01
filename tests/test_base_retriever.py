"""
Tests for base retriever to improve coverage.

These tests cover error handling and edge cases.
"""

from unittest.mock import Mock

import pytest


def test_base_retriever_string_query_without_embed_fn():
    """Test base retriever raises error when string query but no embed_fn."""
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient

    from ragguard import Policy, QdrantSecureRetriever
    from ragguard.exceptions import RetrieverError

    client = QdrantClient(":memory:")

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    # Create retriever without embed_fn
    retriever = QdrantSecureRetriever(
        client=client,
        collection='test',
        policy=policy
        # No embed_fn provided
    )

    user = {'id': 'alice'}

    # Should raise error when query is string
    with pytest.raises(RetrieverError, match="Query is a string but no embed_fn"):
        retriever.search("test query", user=user)


def test_base_retriever_filter_build_error():
    """Test base retriever handles filter building errors."""
    pytest.importorskip("qdrant_client")

    from unittest.mock import patch

    from qdrant_client import QdrantClient

    from ragguard import Policy, QdrantSecureRetriever
    from ragguard.exceptions import RetrieverError

    client = QdrantClient(":memory:")

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    retriever = QdrantSecureRetriever(
        client=client,
        collection='test',
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3]
    )

    user = {'id': 'alice'}

    # Mock policy_engine.to_filter to raise an error
    with patch.object(retriever.policy_engine, 'to_filter', side_effect=Exception("Filter error")):
        with pytest.raises(RetrieverError, match="Failed to build permission filter"):
            retriever.search([0.1, 0.2, 0.3], user=user)


def test_base_retriever_audit_logging_failure(caplog):
    """Test base retriever handles audit logging failures gracefully."""
    pytest.importorskip("qdrant_client")

    import logging
    from unittest.mock import patch

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from ragguard import AuditLogger, Policy, QdrantSecureRetriever

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE)
    )

    # Add a test document
    client.upsert(
        collection_name="test",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3],
                payload={"text": "Test doc"}
            )
        ]
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    audit_logger = AuditLogger()

    retriever = QdrantSecureRetriever(
        client=client,
        collection='test',
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3],
        audit_logger=audit_logger
    )

    user = {'id': 'alice'}

    # Mock audit logger to raise error
    # Note: Using a simple assertion that the search succeeds despite logging failure
    # The actual logging failure is logged internally but we can't easily capture it in tests
    # due to the structured JSON logging configuration
    with patch.object(audit_logger, 'log', side_effect=Exception("Logging error")):
        # Should complete successfully despite logging error
        results = retriever.search("test", user=user)

        # Search should still succeed even when audit logging fails
        assert len(results) == 1


def test_base_retriever_with_custom_filter_builder():
    """Test base retriever uses custom filter builder when provided."""
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    from ragguard import Policy, QdrantSecureRetriever
    from ragguard.filters.custom import CustomFilterBuilder

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE)
    )

    client.upsert(
        collection_name="test",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3],
                payload={"text": "Test doc", "department": "engineering"}
            )
        ]
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    # Create a mock custom filter builder
    custom_builder = Mock(spec=CustomFilterBuilder)
    custom_builder.build_filter.return_value = Filter(
        must=[
            FieldCondition(
                key="department",
                match=MatchValue(value="engineering")
            )
        ]
    )

    retriever = QdrantSecureRetriever(
        client=client,
        collection='test',
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3],
        custom_filter_builder=custom_builder
    )

    user = {'id': 'alice', 'department': 'engineering'}
    results = retriever.search("test", user=user)

    # Verify custom filter builder was used
    custom_builder.build_filter.assert_called_once()
    assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
