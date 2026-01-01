"""
Tests to improve coverage for ragguard/retrievers_async/faiss.py to 95%+.

Focuses on:
- AsyncFAISSSecureRetriever initialization
- search method with async execution
- _execute_search NotImplementedError
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncFAISSSecureRetriever:
    """Tests for AsyncFAISSSecureRetriever class."""

    def create_mock_policy(self):
        """Create a mock policy for testing."""
        from ragguard import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

    def create_mock_index(self, dimension=768):
        """Create a mock FAISS index."""
        index = MagicMock()
        index.d = dimension
        index.ntotal = 100
        return index

    def test_init_basic(self):
        """Test basic initialization."""
        policy = self.create_mock_policy()
        index = self.create_mock_index()
        metadata = [{"id": str(i), "text": f"Doc {i}"} for i in range(100)]

        # Mock the FAISSSecureRetriever at the point where it's imported
        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever') as mock_faiss:
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy
            )

            assert retriever.index is index
            assert retriever.metadata is metadata
            assert retriever.backend_name == "faiss"

    def test_init_with_embed_fn(self):
        """Test initialization with embedding function."""
        policy = self.create_mock_policy()
        index = self.create_mock_index()
        metadata = []

        def mock_embed(text):
            return [0.1] * 768

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever'):
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy,
                embed_fn=mock_embed
            )

            assert retriever is not None

    def test_init_with_over_fetch_factor(self):
        """Test initialization with custom over-fetch factor."""
        policy = self.create_mock_policy()
        index = self.create_mock_index()
        metadata = []

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever') as mock_faiss:
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy,
                over_fetch_factor=5
            )

            # Verify over_fetch_factor was passed to sync retriever
            mock_faiss.assert_called_once()
            call_kwargs = mock_faiss.call_args.kwargs
            assert call_kwargs.get('over_fetch_factor') == 5

    @pytest.mark.asyncio
    async def test_search_async(self):
        """Test async search method."""
        policy = self.create_mock_policy()
        index = self.create_mock_index()
        metadata = [{"id": "doc1", "text": "Hello"}]

        # Create a mock sync retriever
        mock_sync_retriever = MagicMock()
        mock_sync_retriever.search.return_value = [
            {"id": "doc1", "score": 0.9, "metadata": {"text": "Hello"}}
        ]

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever', return_value=mock_sync_retriever):
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy
            )

            # Mock _log_audit
            retriever._log_audit = AsyncMock()

            results = await retriever.search(
                query=[0.1, 0.2, 0.3],
                user={"id": "alice"},
                limit=10
            )

            assert len(results) == 1
            assert results[0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_execute_search_not_implemented(self):
        """Test that _execute_search raises NotImplementedError."""
        policy = self.create_mock_policy()
        index = self.create_mock_index()
        metadata = []

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever'):
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy
            )

            with pytest.raises(NotImplementedError) as exc:
                await retriever._execute_search(
                    query_vector=[0.1, 0.2],
                    native_filter=None,
                    limit=10
                )

            assert "Use search() instead" in str(exc.value)

    def test_backend_name(self):
        """Test backend_name property."""
        policy = self.create_mock_policy()
        index = self.create_mock_index()
        metadata = []

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever'):
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy
            )

            assert retriever.backend_name == "faiss"


class TestAsyncFAISSWithRetry:
    """Tests for AsyncFAISSSecureRetriever with retry configuration."""

    def create_mock_policy(self):
        """Create a mock policy for testing."""
        from ragguard import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

    def test_init_with_retry_config(self):
        """Test initialization with retry configuration."""
        from ragguard.retry import RetryConfig

        policy = self.create_mock_policy()
        retry_config = RetryConfig(max_retries=5)
        index = MagicMock()
        metadata = []

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever') as mock_faiss:
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy,
                retry_config=retry_config,
                enable_retry=True
            )

            # Verify retry config was passed
            mock_faiss.assert_called_once()
            call_kwargs = mock_faiss.call_args.kwargs
            assert call_kwargs.get('retry_config') is retry_config

    def test_init_with_retry_disabled(self):
        """Test initialization with retry disabled."""
        policy = self.create_mock_policy()
        index = MagicMock()
        metadata = []

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever') as mock_faiss:
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy,
                enable_retry=False
            )

            mock_faiss.assert_called_once()
            call_kwargs = mock_faiss.call_args.kwargs
            assert call_kwargs.get('enable_retry') is False


class TestAsyncFAISSWithAuditLogger:
    """Tests for AsyncFAISSSecureRetriever with audit logging."""

    def create_mock_policy(self):
        """Create a mock policy for testing."""
        from ragguard import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "deny"
        })

    def test_init_with_audit_logger(self):
        """Test initialization with audit logger."""
        from ragguard.audit import AuditLogger

        policy = self.create_mock_policy()
        mock_audit_logger = MagicMock(spec=AuditLogger)
        index = MagicMock()
        metadata = []

        with patch('ragguard.retrievers.faiss.FAISSSecureRetriever') as mock_faiss:
            from ragguard.retrievers_async.faiss import AsyncFAISSSecureRetriever

            retriever = AsyncFAISSSecureRetriever(
                index=index,
                metadata=metadata,
                policy=policy,
                audit_logger=mock_audit_logger
            )

            mock_faiss.assert_called_once()
            call_kwargs = mock_faiss.call_args.kwargs
            assert call_kwargs.get('audit_logger') is mock_audit_logger
