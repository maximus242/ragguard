"""
Extended tests for AsyncSecureRetrieverBase.

Tests batch_search, multi_user_search, timeout handling, and validation.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragguard.exceptions import RetrieverTimeoutError
from ragguard.policy.models import Policy
from ragguard.retrievers_async.base import AsyncSecureRetrieverBase
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


class ConcreteAsyncRetriever(AsyncSecureRetrieverBase):
    """Concrete implementation for testing."""

    def __init__(self, policy, search_results=None, search_delay=0, **kwargs):
        super().__init__(policy, **kwargs)
        self._search_results = search_results or []
        self._search_delay = search_delay
        self._search_count = 0

    @property
    def backend_name(self) -> str:
        return "test_backend"

    async def search(
        self,
        query,
        user: Dict[str, Any],
        limit: int = 10,
        **kwargs
    ) -> List[Any]:
        if self._search_delay:
            await asyncio.sleep(self._search_delay)
        self._search_count += 1
        return self._search_results[:limit]

    async def _execute_search(
        self,
        query_vector: List[float],
        native_filter: Any,
        limit: int,
        **kwargs
    ) -> List[Any]:
        return self._search_results[:limit]


class TestAsyncSecureRetrieverBaseInit:
    """Tests for AsyncSecureRetrieverBase initialization."""

    @pytest.fixture
    def policy(self):
        """Create a test policy."""
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "default",
                    "allow": {"conditions": ["user.id == document.owner"]}
                }
            ],
            "default": "deny"
        })

    def test_init_with_defaults(self, policy):
        """Test initialization with default parameters."""
        retriever = ConcreteAsyncRetriever(policy)

        assert retriever.policy == policy
        assert retriever.embed_fn is None
        assert retriever.audit_logger is not None
        assert retriever._enable_retry is True
        assert retriever._enable_validation is True

    def test_init_with_custom_retry_config(self, policy):
        """Test initialization with custom retry config."""
        retry_config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0
        )

        retriever = ConcreteAsyncRetriever(
            policy,
            retry_config=retry_config,
            enable_retry=True
        )

        assert retriever._retry_config.max_retries == 5
        assert retriever._retry_config.initial_delay == 0.5

    def test_init_with_validation_disabled(self, policy):
        """Test initialization with validation disabled."""
        retriever = ConcreteAsyncRetriever(
            policy,
            enable_validation=False
        )

        assert retriever._enable_validation is False

    def test_init_with_embed_fn(self, policy):
        """Test initialization with embedding function."""
        embed_fn = MagicMock(return_value=[0.1, 0.2, 0.3])

        retriever = ConcreteAsyncRetriever(
            policy,
            embed_fn=embed_fn
        )

        assert retriever.embed_fn == embed_fn


class TestAsyncGetQueryVector:
    """Tests for _get_query_vector method."""

    @pytest.fixture
    def policy(self):
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"roles": ["user"]}}],
            "default": "deny"
        })

    @pytest.mark.asyncio
    async def test_get_query_vector_with_string(self, policy):
        """Test converting string query to vector."""
        embed_fn = MagicMock(return_value=[0.1, 0.2, 0.3])

        retriever = ConcreteAsyncRetriever(policy, embed_fn=embed_fn)

        with patch('ragguard.retrievers_async.base.run_in_executor_with_backpressure',
                   new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]):
            vector = await retriever._get_query_vector("test query")

        assert vector == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_query_vector_with_vector(self, policy):
        """Test passing through vector query."""
        retriever = ConcreteAsyncRetriever(policy)

        vector = await retriever._get_query_vector([0.1, 0.2, 0.3])

        assert vector == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_query_vector_string_without_embed_fn(self, policy):
        """Test error when string query but no embed_fn."""
        retriever = ConcreteAsyncRetriever(policy)

        with pytest.raises(ValueError, match="embed_fn required"):
            await retriever._get_query_vector("test query")


class TestAsyncBatchSearch:
    """Tests for batch_search method."""

    @pytest.fixture
    def policy(self):
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"roles": ["user"]}}],
            "default": "deny"
        })

    @pytest.mark.asyncio
    async def test_batch_search_empty_queries(self, policy):
        """Test batch search with empty queries list."""
        retriever = ConcreteAsyncRetriever(policy)

        results = await retriever.batch_search(
            queries=[],
            user={"id": "alice", "roles": ["user"]}
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_search_multiple_queries(self, policy):
        """Test batch search with multiple queries."""
        retriever = ConcreteAsyncRetriever(
            policy,
            search_results=[{"text": "result1"}, {"text": "result2"}]
        )

        results = await retriever.batch_search(
            queries=["query1", "query2", "query3"],
            user={"id": "alice", "roles": ["user"]},
            limit=5
        )

        assert len(results) == 3
        assert retriever._search_count == 3

    @pytest.mark.asyncio
    async def test_batch_search_concurrency_limit(self, policy):
        """Test batch search respects concurrency limit."""
        retriever = ConcreteAsyncRetriever(
            policy,
            search_results=[{"text": "result"}],
            search_delay=0.01
        )

        queries = [f"query{i}" for i in range(10)]

        results = await retriever.batch_search(
            queries=queries,
            user={"id": "alice", "roles": ["user"]},
            max_concurrent=3
        )

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_batch_search_timeout(self, policy):
        """Test batch search timeout."""
        retriever = ConcreteAsyncRetriever(
            policy,
            search_results=[{"text": "result"}],
            search_delay=1.0  # 1 second delay
        )

        with pytest.raises(RetrieverTimeoutError):
            await retriever.batch_search(
                queries=["query1", "query2"],
                user={"id": "alice", "roles": ["user"]},
                timeout=0.1  # Very short timeout
            )

    @pytest.mark.asyncio
    async def test_batch_search_validation_enabled(self, policy):
        """Test batch search with validation enabled."""
        retriever = ConcreteAsyncRetriever(
            policy,
            search_results=[{"text": "result"}],
            enable_validation=True
        )

        # This should work with valid user context
        results = await retriever.batch_search(
            queries=["query1", "query2"],
            user={"id": "alice", "roles": ["user"]}
        )

        assert len(results) == 2


class TestAsyncMultiUserSearch:
    """Tests for multi_user_search method."""

    @pytest.fixture
    def policy(self):
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"roles": ["user"]}}],
            "default": "deny"
        })

    @pytest.mark.asyncio
    async def test_multi_user_search_empty_users(self, policy):
        """Test multi-user search with empty users list."""
        retriever = ConcreteAsyncRetriever(policy)

        results = await retriever.multi_user_search(
            query="test query",
            users=[]
        )

        assert results == {}

    @pytest.mark.asyncio
    async def test_multi_user_search_multiple_users(self, policy):
        """Test multi-user search with multiple users."""
        retriever = ConcreteAsyncRetriever(
            policy,
            search_results=[{"text": "result"}]
        )

        users = [
            {"id": "alice", "roles": ["user"]},
            {"id": "bob", "roles": ["user"]},
            {"id": "carol", "roles": ["user"]}
        ]

        results = await retriever.multi_user_search(
            query="test query",
            users=users
        )

        assert len(results) == 3
        assert "alice" in results
        assert "bob" in results
        assert "carol" in results

    @pytest.mark.asyncio
    async def test_multi_user_search_timeout(self, policy):
        """Test multi-user search timeout."""
        retriever = ConcreteAsyncRetriever(
            policy,
            search_results=[{"text": "result"}],
            search_delay=1.0
        )

        users = [{"id": "alice"}, {"id": "bob"}]

        with pytest.raises(RetrieverTimeoutError):
            await retriever.multi_user_search(
                query="test",
                users=users,
                timeout=0.1
            )


class TestAsyncRunWithTimeout:
    """Tests for _run_with_timeout method."""

    @pytest.fixture
    def policy(self):
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"roles": ["user"]}}],
            "default": "deny"
        })

    @pytest.mark.asyncio
    async def test_run_with_timeout_success(self, policy):
        """Test successful execution within timeout."""
        retriever = ConcreteAsyncRetriever(policy)

        async def fast_operation():
            return "result"

        result = await retriever._run_with_timeout(
            fast_operation(),
            timeout=5.0,
            operation="test_op"
        )

        assert result == "result"

    @pytest.mark.asyncio
    async def test_run_with_timeout_exceeded(self, policy):
        """Test timeout exceeded."""
        retriever = ConcreteAsyncRetriever(policy)

        async def slow_operation():
            await asyncio.sleep(10)
            return "result"

        with pytest.raises(RetrieverTimeoutError) as exc_info:
            await retriever._run_with_timeout(
                slow_operation(),
                timeout=0.01,
                operation="slow_op"
            )

        assert exc_info.value.backend == "test_backend"
        assert exc_info.value.operation == "slow_op"


class TestAsyncLogAudit:
    """Tests for _log_audit method."""

    @pytest.fixture
    def policy(self):
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"roles": ["user"]}}],
            "default": "deny"
        })

    @pytest.mark.asyncio
    async def test_log_audit_with_logger(self, policy):
        """Test audit logging when logger is present."""
        audit_logger = MagicMock()

        retriever = ConcreteAsyncRetriever(policy, audit_logger=audit_logger)

        with patch('ragguard.retrievers_async.base.run_in_executor_with_backpressure',
                   new_callable=AsyncMock):
            await retriever._log_audit(
                user={"id": "alice"},
                query="test query",
                results=[{"text": "result"}]
            )

    @pytest.mark.asyncio
    async def test_log_audit_handles_error(self, policy):
        """Test audit logging handles errors gracefully."""
        audit_logger = MagicMock()

        retriever = ConcreteAsyncRetriever(policy, audit_logger=audit_logger)

        with patch('ragguard.retrievers_async.base.run_in_executor_with_backpressure',
                   new_callable=AsyncMock, side_effect=RuntimeError("Log error")):
            # Should not raise, just log warning
            await retriever._log_audit(
                user={"id": "alice"},
                query="test query",
                results=[]
            )
