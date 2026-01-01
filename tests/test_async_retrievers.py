"""
Tests for async retrievers.

Tests async search functionality, retry behavior, and async utilities.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Skip all tests if qdrant-client is not installed (required for async retrievers)
pytest.importorskip("qdrant_client")

from ragguard.policy import Policy
from ragguard.retrievers_async import (
    AsyncChromaDBSecureRetriever,
    AsyncFAISSSecureRetriever,
    AsyncPgvectorSecureRetriever,
    AsyncPineconeSecureRetriever,
    AsyncQdrantSecureRetriever,
    AsyncWeaviateSecureRetriever,
    batch_search_async,
    multi_user_search_async,
    run_sync_retriever_async,
)
from ragguard.retry import RetryConfig

# Test Policy
TEST_POLICY = Policy.from_dict({
    "version": "1",
    "rules": [{
        "name": "public_access",
        "match": {"visibility": "public"},
        "allow": {"everyone": True}
    }],
    "default": "deny"
})


class TestAsyncQdrantSecureRetriever:
    """Test AsyncQdrantSecureRetriever."""

    def test_init_requires_async_client(self):
        """Test initialization requires AsyncQdrantClient."""
        # Mock a non-async client
        mock_client = Mock()
        mock_client.__class__.__name__ = "QdrantClient"

        # Should raise TypeError
        with pytest.raises(TypeError, match="must be AsyncQdrantClient"):
            AsyncQdrantSecureRetriever(
                client=mock_client,
                collection="test",
                policy=TEST_POLICY
            )

    @pytest.mark.skipif(
        True,  # Skip by default - requires qdrant-client
        reason="Requires qdrant-client with AsyncQdrantClient"
    )
    async def test_async_search_with_vector(self):
        """Test async search with query vector."""
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import ScoredPoint
        except ImportError:
            pytest.skip("qdrant-client not installed")

        # Create mock async client
        mock_client = AsyncMock(spec=AsyncQdrantClient)
        mock_client.search = AsyncMock(return_value=[
            ScoredPoint(id="1", score=0.9, payload={"visibility": "public"}),
            ScoredPoint(id="2", score=0.8, payload={"visibility": "public"}),
        ])

        retriever = AsyncQdrantSecureRetriever(
            client=mock_client,
            collection="test_collection",
            policy=TEST_POLICY
        )

        query_vector = [0.1, 0.2, 0.3]
        results = await retriever.search(
            query=query_vector,
            user={"id": "alice"},
            limit=10
        )

        assert len(results) == 2
        assert mock_client.search.called

    async def test_async_search_with_text_requires_embed_fn(self):
        """Test async search with text requires embed_fn."""
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError:
            pytest.skip("qdrant-client not installed")

        mock_client = AsyncMock(spec=AsyncQdrantClient)

        retriever = AsyncQdrantSecureRetriever(
            client=mock_client,
            collection="test_collection",
            policy=TEST_POLICY,
            embed_fn=None  # No embed function
        )

        with pytest.raises(ValueError, match="embed_fn required"):
            await retriever.search(
                query="text query",  # String query
                user={"id": "alice"},
                limit=10
            )

    async def test_retry_disabled(self):
        """Test search with retry disabled."""
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError:
            pytest.skip("qdrant-client not installed")

        mock_client = AsyncMock(spec=AsyncQdrantClient)
        mock_client.search = AsyncMock(return_value=[])

        retriever = AsyncQdrantSecureRetriever(
            client=mock_client,
            collection="test_collection",
            policy=TEST_POLICY,
            enable_retry=False
        )

        results = await retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert results == []
        assert mock_client.search.called


@pytest.mark.asyncio
class TestAsyncChromaDBSecureRetriever:
    """Test AsyncChromaDBSecureRetriever."""

    async def test_async_search_with_text(self):
        """Test async search with text query."""
        # Create mock collection
        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1', 'doc2']],
            'metadatas': [[{'visibility': 'public'}, {'visibility': 'public'}]],
            'documents': [['text1', 'text2']],
            'distances': [[0.1, 0.2]]
        })

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY
        )

        results = await retriever.search(
            query="test query",
            user={"id": "alice"},
            limit=10
        )

        assert len(results) == 2
        assert results[0]['id'] == 'doc1'
        assert results[0]['metadata']['visibility'] == 'public'

    async def test_async_search_with_vector(self):
        """Test async search with query vector."""
        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'field': 'value'}]],
            'documents': [['text']],
            'distances': [[0.5]]
        })

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY
        )

        results = await retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=5
        )

        assert len(results) == 1
        # Verify query was called with query_embeddings
        assert mock_collection.query.called

    async def test_retry_config(self):
        """Test custom retry configuration."""
        mock_collection = Mock()
        mock_collection.query = Mock(return_value={'ids': [[]]})

        custom_retry = RetryConfig(
            max_retries=5,
            initial_delay=0.05,
            exponential_base=3
        )

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY,
            retry_config=custom_retry
        )

        results = await retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert results == []


@pytest.mark.asyncio
class TestAsyncPineconeSecureRetriever:
    """Test AsyncPineconeSecureRetriever."""

    async def test_async_search_with_vector(self):
        """Test async Pinecone search."""
        # Create mock index
        mock_index = Mock()
        mock_result = Mock()
        mock_result.matches = [
            Mock(id='doc1', score=0.9, metadata={'visibility': 'public'}),
            Mock(id='doc2', score=0.8, metadata={'visibility': 'public'}),
        ]
        mock_index.query = Mock(return_value=mock_result)

        retriever = AsyncPineconeSecureRetriever(
            index=mock_index,
            policy=TEST_POLICY,
            namespace="test-ns"
        )

        results = await retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert len(results) == 2
        assert mock_index.query.called

    async def test_text_query_requires_embed_fn(self):
        """Test text query requires embed function."""
        mock_index = Mock()

        retriever = AsyncPineconeSecureRetriever(
            index=mock_index,
            policy=TEST_POLICY,
            embed_fn=None
        )

        with pytest.raises(ValueError, match="embed_fn required"):
            await retriever.search(
                query="text query",
                user={"id": "alice"},
                limit=10
            )


@pytest.mark.asyncio
class TestAsyncWeaviateSecureRetriever:
    """Test AsyncWeaviateSecureRetriever."""

    async def test_async_search(self):
        """Test async Weaviate search."""
        # Create mock client with query builder pattern
        mock_client = Mock()
        mock_query = Mock()
        mock_query.get = Mock(return_value=mock_query)
        mock_query.with_near_vector = Mock(return_value=mock_query)
        mock_query.with_where = Mock(return_value=mock_query)
        mock_query.with_limit = Mock(return_value=mock_query)
        mock_query.do = Mock(return_value={
            'data': {
                'Get': {
                    'Documents': [
                        {'id': 'doc1', 'visibility': 'public'},
                        {'id': 'doc2', 'visibility': 'public'},
                    ]
                }
            }
        })
        mock_client.query = mock_query

        retriever = AsyncWeaviateSecureRetriever(
            client=mock_client,
            collection="Documents",
            policy=TEST_POLICY
        )

        results = await retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert len(results) == 2

    async def test_retry_enabled(self):
        """Test retry enabled by default."""
        mock_client = Mock()
        mock_query = Mock()
        mock_query.get = Mock(return_value=mock_query)
        mock_query.with_near_vector = Mock(return_value=mock_query)
        mock_query.with_where = Mock(return_value=mock_query)
        mock_query.with_limit = Mock(return_value=mock_query)
        # Return proper dict, not Mock
        mock_query.do = Mock(return_value={'data': {'Get': {'Documents': []}}})
        mock_client.query = mock_query

        retriever = AsyncWeaviateSecureRetriever(
            client=mock_client,
            collection="Documents",
            policy=TEST_POLICY,
            enable_retry=True
        )

        results = await retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert results == []


@pytest.mark.asyncio
class TestAsyncPgvectorSecureRetriever:
    """Test AsyncPgvectorSecureRetriever."""

    async def test_wraps_sync_retriever(self):
        """Test async pgvector wraps sync retriever."""
        # This uses run_in_executor to wrap sync retriever
        # We'll test that it delegates correctly

        mock_connection = Mock()

        # Create async retriever (should not raise)
        retriever = AsyncPgvectorSecureRetriever(
            connection=mock_connection,
            table="documents",
            policy=TEST_POLICY,
            embedding_column="embedding"
        )

        # Verify sync retriever was created
        assert retriever._sync_retriever is not None


@pytest.mark.asyncio
class TestAsyncFAISSSecureRetriever:
    """Test AsyncFAISSSecureRetriever."""

    async def test_wraps_sync_retriever(self):
        """Test async FAISS wraps sync retriever."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            pytest.skip("faiss not installed")

        # Create mock index
        mock_index = Mock()
        metadata = [
            {"id": "doc1", "visibility": "public"},
            {"id": "doc2", "visibility": "public"}
        ]

        # Create async retriever
        retriever = AsyncFAISSSecureRetriever(
            index=mock_index,
            metadata=metadata,
            policy=TEST_POLICY,
            over_fetch_factor=3
        )

        # Verify sync retriever was created
        assert retriever._sync_retriever is not None


@pytest.mark.asyncio
class TestAsyncUtilities:
    """Test async utility functions."""

    async def test_run_sync_retriever_async(self):
        """Test run_sync_retriever_async utility."""
        # Create mock sync retriever
        mock_retriever = Mock()
        mock_retriever.search = Mock(return_value=[
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8}
        ])

        results = await run_sync_retriever_async(
            retriever=mock_retriever,
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        mock_retriever.search.assert_called_once()

    async def test_batch_search_async(self):
        """Test batch_search_async utility."""
        # Create mock async retriever
        mock_retriever = AsyncMock()
        mock_retriever.search = AsyncMock(side_effect=[
            [{"id": "doc1"}],
            [{"id": "doc2"}],
            [{"id": "doc3"}]
        ])

        queries = ["query1", "query2", "query3"]
        result = await batch_search_async(
            retriever=mock_retriever,
            queries=queries,
            user={"id": "alice"},
            limit=10
        )

        # batch_search_async now returns BatchSearchResult
        assert result.success_count == 3
        assert result.all_succeeded is True
        assert result.results[0][0]["id"] == "doc1"
        assert result.results[1][0]["id"] == "doc2"
        assert result.results[2][0]["id"] == "doc3"
        assert mock_retriever.search.call_count == 3

    async def test_multi_user_search_async(self):
        """Test multi_user_search_async utility."""
        # Create mock async retriever
        mock_retriever = AsyncMock()
        mock_retriever.search = AsyncMock(side_effect=[
            [{"id": "doc1"}],  # alice results
            [{"id": "doc2"}],  # bob results
            []  # charlie results (denied)
        ])

        users = [
            {"id": "alice", "role": "admin"},
            {"id": "bob", "role": "user"},
            {"id": "charlie", "role": "guest"}
        ]

        result = await multi_user_search_async(
            retriever=mock_retriever,
            query="test query",
            users=users,
            limit=10
        )

        # multi_user_search_async now returns MultiUserSearchResult
        assert result.success_count == 3
        assert "alice" in result.results
        assert "bob" in result.results
        assert "charlie" in result.results
        assert len(result.results["alice"]) == 1
        assert len(result.results["bob"]) == 1
        assert len(result.results["charlie"]) == 0
        assert mock_retriever.search.call_count == 3


@pytest.mark.asyncio
class TestEmbedFunctionHandling:
    """Test embed function handling across async retrievers."""

    async def test_embed_fn_called_in_executor(self):
        """Test embed function is called in executor (non-blocking)."""
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError:
            pytest.skip("qdrant-client not installed")

        # Create mock embed function
        def mock_embed(text: str) -> List[float]:
            return [0.1, 0.2, 0.3]

        mock_client = AsyncMock(spec=AsyncQdrantClient)
        mock_client.search = AsyncMock(return_value=[])

        retriever = AsyncQdrantSecureRetriever(
            client=mock_client,
            collection="test",
            policy=TEST_POLICY,
            embed_fn=mock_embed
        )

        # This should work - embed_fn will be called in executor
        results = await retriever.search(
            query="text query",
            user={"id": "alice"},
            limit=10
        )

        assert results == []


@pytest.mark.asyncio
class TestRetryBehavior:
    """Test retry behavior in async retrievers."""

    async def test_retry_on_failure(self):
        """Test retry on transient failures."""
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError:
            pytest.skip("qdrant-client not installed")

        # Create mock that fails twice then succeeds
        mock_client = AsyncMock(spec=AsyncQdrantClient)
        call_count = [0]

        async def failing_search(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            return []

        mock_client.search = failing_search

        retriever = AsyncQdrantSecureRetriever(
            client=mock_client,
            collection="test",
            policy=TEST_POLICY,
            retry_config=RetryConfig(
                max_retries=3,
                initial_delay=0.01,
                jitter=False
            ),
            enable_retry=True
        )

        # Should succeed after 2 retries
        results = await retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert results == []
        assert call_count[0] == 3  # Initial + 2 retries

    async def test_no_retry_when_disabled(self):
        """Test no retry when retry is disabled."""
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError:
            pytest.skip("qdrant-client not installed")

        mock_client = AsyncMock(spec=AsyncQdrantClient)
        mock_client.search = AsyncMock(side_effect=ConnectionError("Network error"))

        retriever = AsyncQdrantSecureRetriever(
            client=mock_client,
            collection="test",
            policy=TEST_POLICY,
            enable_retry=False
        )

        # Should fail immediately without retry
        with pytest.raises(ConnectionError, match="Network error"):
            await retriever.search(
                query=[0.1, 0.2, 0.3],
                user={"id": "alice"},
                limit=10
            )

        # Should only be called once (no retries)
        assert mock_client.search.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
