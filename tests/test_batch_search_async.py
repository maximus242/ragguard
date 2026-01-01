"""
Tests for batch_search_async and multi_user_search_async with error handling.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from ragguard.retrievers_async.utils import (
    BatchSearchResult,
    MultiUserSearchResult,
    batch_search_async,
    multi_user_search_async,
)


class TestBatchSearchResult:
    """Test BatchSearchResult dataclass."""

    def test_success_count(self):
        """Test success_count property."""
        result = BatchSearchResult(
            results=[["doc1"], ["doc2"], None],
            errors=[None, None, ValueError("failed")]
        )
        assert result.success_count == 2
        assert result.failure_count == 1

    def test_all_succeeded(self):
        """Test all_succeeded property."""
        success = BatchSearchResult(
            results=[["doc1"], ["doc2"]],
            errors=[None, None]
        )
        assert success.all_succeeded is True

        partial = BatchSearchResult(
            results=[["doc1"], None],
            errors=[None, ValueError("failed")]
        )
        assert partial.all_succeeded is False

    def test_successful_results(self):
        """Test successful_results method."""
        result = BatchSearchResult(
            results=[["doc1"], None, ["doc2", "doc3"]],
            errors=[None, ValueError("failed"), None]
        )
        successful = result.successful_results()
        assert len(successful) == 2
        assert ["doc1"] in successful
        assert ["doc2", "doc3"] in successful

    def test_raise_on_any_failure(self):
        """Test raise_on_any_failure method."""
        result = BatchSearchResult(
            results=[["doc1"], None],
            errors=[None, ValueError("query 2 failed")]
        )

        with pytest.raises(ValueError) as exc_info:
            result.raise_on_any_failure()
        assert "query 2 failed" in str(exc_info.value)

    def test_no_raise_on_success(self):
        """Test raise_on_any_failure with no failures."""
        result = BatchSearchResult(
            results=[["doc1"], ["doc2"]],
            errors=[None, None]
        )
        # Should not raise
        result.raise_on_any_failure()


class TestMultiUserSearchResult:
    """Test MultiUserSearchResult dataclass."""

    def test_success_count(self):
        """Test success_count property."""
        result = MultiUserSearchResult(
            results={"alice": ["doc1"], "bob": None, "charlie": ["doc2"]},
            errors={"alice": None, "bob": ValueError("failed"), "charlie": None}
        )
        assert result.success_count == 2
        assert result.failure_count == 1

    def test_failed_users(self):
        """Test failed_users method."""
        result = MultiUserSearchResult(
            results={"alice": ["doc1"], "bob": None, "charlie": None},
            errors={"alice": None, "bob": ValueError("e1"), "charlie": ValueError("e2")}
        )
        failed = result.failed_users()
        assert "bob" in failed
        assert "charlie" in failed
        assert "alice" not in failed

    def test_successful_results(self):
        """Test successful_results method."""
        result = MultiUserSearchResult(
            results={"alice": ["doc1"], "bob": None},
            errors={"alice": None, "bob": ValueError("failed")}
        )
        successful = result.successful_results()
        assert "alice" in successful
        assert "bob" not in successful


class TestBatchSearchAsync:
    """Test batch_search_async function."""

    @pytest.mark.asyncio
    async def test_all_success(self):
        """Test batch search when all queries succeed."""
        retriever = AsyncMock()
        retriever.search = AsyncMock(side_effect=[
            ["doc1", "doc2"],
            ["doc3"],
            ["doc4", "doc5", "doc6"]
        ])

        result = await batch_search_async(
            retriever=retriever,
            queries=["q1", "q2", "q3"],
            user={"id": "alice"},
            limit=10
        )

        assert result.success_count == 3
        assert result.failure_count == 0
        assert result.all_succeeded is True
        assert len(result.successful_results()) == 3

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """Test batch search with some queries failing."""
        retriever = AsyncMock()
        retriever.search = AsyncMock(side_effect=[
            ["doc1"],
            ValueError("Query 2 failed"),
            ["doc2", "doc3"]
        ])

        result = await batch_search_async(
            retriever=retriever,
            queries=["q1", "q2", "q3"],
            user={"id": "alice"},
            limit=10
        )

        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.has_failures is True

        # Check specific results
        assert result.results[0] == ["doc1"]
        assert result.results[1] is None
        assert result.results[2] == ["doc2", "doc3"]

        # Check error
        assert isinstance(result.errors[1], ValueError)

    @pytest.mark.asyncio
    async def test_all_failure(self):
        """Test batch search when all queries fail."""
        retriever = AsyncMock()
        retriever.search = AsyncMock(side_effect=[
            ValueError("Error 1"),
            ValueError("Error 2"),
        ])

        result = await batch_search_async(
            retriever=retriever,
            queries=["q1", "q2"],
            user={"id": "alice"},
            limit=10
        )

        assert result.success_count == 0
        assert result.failure_count == 2
        assert len(result.successful_results()) == 0

    @pytest.mark.asyncio
    async def test_fail_fast_mode(self):
        """Test batch search with return_exceptions=False (fail fast)."""
        retriever = AsyncMock()
        retriever.search = AsyncMock(side_effect=[
            ["doc1"],
            ValueError("Should fail"),
            ["doc2"]  # Never reached
        ])

        with pytest.raises(ValueError):
            await batch_search_async(
                retriever=retriever,
                queries=["q1", "q2", "q3"],
                user={"id": "alice"},
                limit=10,
                return_exceptions=False
            )


class TestMultiUserSearchAsync:
    """Test multi_user_search_async function."""

    @pytest.mark.asyncio
    async def test_all_success(self):
        """Test multi-user search when all users succeed."""
        retriever = AsyncMock()
        retriever.search = AsyncMock(side_effect=[
            ["doc1"],
            ["doc2", "doc3"],
            ["doc4"]
        ])

        users = [
            {"id": "alice"},
            {"id": "bob"},
            {"id": "charlie"}
        ]

        result = await multi_user_search_async(
            retriever=retriever,
            query="test query",
            users=users,
            limit=10
        )

        assert result.success_count == 3
        assert result.failure_count == 0
        assert "alice" in result.results
        assert "bob" in result.results
        assert "charlie" in result.results

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """Test multi-user search with some users failing."""
        retriever = AsyncMock()
        retriever.search = AsyncMock(side_effect=[
            ["doc1"],
            RuntimeError("Bob's search failed"),
            ["doc2"]
        ])

        users = [
            {"id": "alice"},
            {"id": "bob"},
            {"id": "charlie"}
        ]

        result = await multi_user_search_async(
            retriever=retriever,
            query="test query",
            users=users,
            limit=10
        )

        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.results["alice"] == ["doc1"]
        assert result.results["bob"] is None
        assert result.results["charlie"] == ["doc2"]
        assert "bob" in result.failed_users()

    @pytest.mark.asyncio
    async def test_user_without_id(self):
        """Test that users without IDs get generated IDs."""
        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=["doc1"])

        users = [
            {"role": "admin"},  # No id
            {"role": "user"}    # No id
        ]

        result = await multi_user_search_async(
            retriever=retriever,
            query="test query",
            users=users,
            limit=10
        )

        assert result.success_count == 2
        assert "user_0" in result.results
        assert "user_1" in result.results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
