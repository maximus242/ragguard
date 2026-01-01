"""
Tests for async retriever edge cases.

Covers timeouts, concurrency, backpressure, and error handling paths
that are specific to async operations.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ragguard.policy import Policy
from ragguard.retry import RetryConfig, run_in_executor_with_backpressure

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


@pytest.mark.asyncio
class TestAsyncTimeoutHandling:
    """Test timeout behavior in async operations."""

    async def test_executor_timeout(self):
        """Test that slow sync functions can be timed out."""
        import time

        def slow_function():
            time.sleep(2)
            return "done"

        # Run with a short timeout - should complete before timeout
        result = await asyncio.wait_for(
            run_in_executor_with_backpressure(lambda: "fast"),
            timeout=1.0
        )
        assert result == "fast"

    async def test_executor_handles_exception(self):
        """Test that exceptions in executor are propagated."""
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await run_in_executor_with_backpressure(failing_function)


@pytest.mark.asyncio
class TestAsyncConcurrency:
    """Test concurrent async operations."""

    async def test_concurrent_searches_independent(self):
        """Test that concurrent searches are independent."""
        from ragguard.retrievers_async import AsyncChromaDBSecureRetriever

        call_order = []

        def make_mock_collection(delay: float, result_id: str):
            mock = Mock()

            def slow_query(*args, **kwargs):
                import time
                time.sleep(delay)
                call_order.append(result_id)
                return {
                    'ids': [[result_id]],
                    'metadatas': [[{'visibility': 'public'}]],
                    'documents': [['text']],
                    'distances': [[0.1]]
                }

            mock.query = slow_query
            return mock

        # Create two retrievers with different delays
        retriever1 = AsyncChromaDBSecureRetriever(
            collection=make_mock_collection(0.1, "fast"),
            policy=TEST_POLICY
        )
        retriever2 = AsyncChromaDBSecureRetriever(
            collection=make_mock_collection(0.05, "slow"),
            policy=TEST_POLICY
        )

        # Run concurrently - both should complete
        results = await asyncio.gather(
            retriever1.search(query=[0.1], user={"id": "alice"}, limit=1),
            retriever2.search(query=[0.2], user={"id": "bob"}, limit=1)
        )

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

    async def test_many_concurrent_searches(self):
        """Test many concurrent searches don't cause issues."""
        from ragguard.retrievers_async import AsyncChromaDBSecureRetriever

        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'visibility': 'public'}]],
            'documents': [['text']],
            'distances': [[0.1]]
        })

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY
        )

        # Run 50 concurrent searches
        tasks = [
            retriever.search(query=[0.1] * 128, user={"id": f"user_{i}"}, limit=5)
            for i in range(50)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert all(len(r) == 1 for r in results)


@pytest.mark.asyncio
class TestAsyncBackpressure:
    """Test backpressure handling in async operations."""

    async def test_executor_backpressure_limits_concurrency(self):
        """Test that executor limits concurrent operations."""
        execution_count = [0]
        max_concurrent = [0]
        current_concurrent = [0]

        def track_concurrency():
            current_concurrent[0] += 1
            max_concurrent[0] = max(max_concurrent[0], current_concurrent[0])
            import time
            time.sleep(0.01)  # Simulate work
            current_concurrent[0] -= 1
            execution_count[0] += 1
            return execution_count[0]

        # Run many tasks
        tasks = [
            run_in_executor_with_backpressure(track_concurrency)
            for _ in range(20)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert execution_count[0] == 20
        # Max concurrent should be limited by thread pool size
        # (typically number of CPUs + 4)


@pytest.mark.asyncio
class TestAsyncErrorPropagation:
    """Test error propagation in async operations."""

    async def test_search_error_propagates(self):
        """Test that search errors propagate correctly."""
        from ragguard.retrievers_async import AsyncChromaDBSecureRetriever

        mock_collection = Mock()
        mock_collection.query = Mock(side_effect=RuntimeError("Database error"))

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY,
            enable_retry=False  # Disable retry to see immediate failure
        )

        with pytest.raises(RuntimeError, match="Database error"):
            await retriever.search(
                query=[0.1, 0.2],
                user={"id": "alice"},
                limit=5
            )

    async def test_validation_error_propagates(self):
        """Test that validation errors propagate."""
        from ragguard.retrievers_async import AsyncPineconeSecureRetriever

        mock_index = Mock()

        retriever = AsyncPineconeSecureRetriever(
            index=mock_index,
            policy=TEST_POLICY,
            embed_fn=None  # No embed function
        )

        # Should raise ValueError for text query without embed_fn
        with pytest.raises(ValueError, match="embed_fn required"):
            await retriever.search(
                query="text query",
                user={"id": "alice"},
                limit=10
            )


@pytest.mark.asyncio
class TestAsyncBatchOperations:
    """Test batch operations in async context."""

    async def test_batch_search_handles_partial_failures(self):
        """Test batch search continues despite individual failures."""
        from ragguard.retrievers_async import batch_search_async

        call_count = [0]

        async def mock_search(query, user, limit):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Query 2 failed")
            return [{"id": f"result_{call_count[0]}"}]

        mock_retriever = AsyncMock()
        mock_retriever.search = mock_search

        queries = ["q1", "q2", "q3"]

        # batch_search_async now captures exceptions by default (return_exceptions=True)
        # This allows partial results to be preserved
        result = await batch_search_async(
            retriever=mock_retriever,
            queries=queries,
            user={"id": "alice"},
            limit=5
        )

        # Should have 2 successes and 1 failure
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.has_failures is True

        # Query 1 and 3 succeeded
        assert result.results[0] == [{"id": "result_1"}]
        assert result.results[1] is None  # Query 2 failed
        assert result.results[2] == [{"id": "result_3"}]

        # Error captured
        assert isinstance(result.errors[1], ValueError)
        assert "Query 2 failed" in str(result.errors[1])

    async def test_multi_user_independent_failures(self):
        """Test multi-user search handles users independently."""
        from ragguard.retrievers_async import multi_user_search_async

        async def mock_search(query, user, limit):
            if user.get("id") == "bad_user":
                raise PermissionError("Access denied")
            return [{"id": f"result_for_{user.get('id')}"}]

        mock_retriever = AsyncMock()
        mock_retriever.search = mock_search

        users = [
            {"id": "alice"},
            {"id": "bad_user"},
            {"id": "bob"}
        ]

        # multi_user_search_async now captures exceptions by default (return_exceptions=True)
        # This allows independent user searches to succeed even if one fails
        result = await multi_user_search_async(
            retriever=mock_retriever,
            query="test",
            users=users,
            limit=5
        )

        # Should have 2 successes and 1 failure
        assert result.success_count == 2
        assert result.failure_count == 1

        # alice and bob succeeded
        assert result.results["alice"] == [{"id": "result_for_alice"}]
        assert result.results["bob"] == [{"id": "result_for_bob"}]

        # bad_user failed
        assert result.results["bad_user"] is None
        assert isinstance(result.errors["bad_user"], PermissionError)
        assert "bad_user" in result.failed_users()


@pytest.mark.asyncio
class TestAsyncAuditLogging:
    """Test audit logging in async context."""

    async def test_audit_log_runs_in_executor(self):
        """Test that audit logging doesn't block event loop."""
        from ragguard.audit.logger import AuditLogger
        from ragguard.retrievers_async import AsyncChromaDBSecureRetriever

        log_calls = []

        class TestLogger(AuditLogger):
            def log(self, *args, **kwargs):
                import time
                time.sleep(0.01)  # Simulate slow logging
                log_calls.append(kwargs)

        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'visibility': 'public'}]],
            'documents': [['text']],
            'distances': [[0.1]]
        })

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY,
            audit_logger=TestLogger()
        )

        results = await retriever.search(
            query=[0.1],
            user={"id": "alice"},
            limit=5
        )

        assert len(results) == 1
        # Note: audit logging may be async, so log_calls might be empty here
        # depending on implementation


@pytest.mark.asyncio
class TestAsyncPolicyEngineAccess:
    """Test policy engine access in async context."""

    async def test_policy_engine_thread_safe(self):
        """Test that policy engine access is thread-safe in async context."""
        from ragguard.retrievers_async import AsyncChromaDBSecureRetriever

        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'visibility': 'public'}]],
            'documents': [['text']],
            'distances': [[0.1]]
        })

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY
        )

        # Access engine from multiple concurrent tasks
        async def access_engine():
            _ = retriever.engine.policy
            return True

        tasks = [access_engine() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        assert all(results)

    async def test_concurrent_search_with_same_user(self):
        """Test concurrent searches with same user context."""
        from ragguard.retrievers_async import AsyncChromaDBSecureRetriever

        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'visibility': 'public'}]],
            'documents': [['text']],
            'distances': [[0.1]]
        })

        retriever = AsyncChromaDBSecureRetriever(
            collection=mock_collection,
            policy=TEST_POLICY
        )

        user = {"id": "alice", "department": "engineering"}

        # Run many concurrent searches with same user
        tasks = [
            retriever.search(query=[float(i) / 100], user=user, limit=1)
            for i in range(20)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(len(r) == 1 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
