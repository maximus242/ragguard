"""
Tests for retry logic with exponential backoff.

Tests retry decorators, config, and context managers.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from ragguard.retry import (
    RetryableOperation,
    RetryConfig,
    async_retry_on_failure,
    retry_on_failure,
)


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 0.1
        assert config.max_delay == 10.0
        assert config.exponential_base == 2
        assert config.jitter is True
        assert ConnectionError in config.retry_on
        assert TimeoutError in config.retry_on
        assert OSError in config.retry_on

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=3,
            jitter=False
        )
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3
        assert config.jitter is False

    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2,
            max_delay=10.0,
            jitter=False
        )

        # Attempt 0: 1.0 * 2^0 = 1.0
        assert config.calculate_delay(0) == 1.0

        # Attempt 1: 1.0 * 2^1 = 2.0
        assert config.calculate_delay(1) == 2.0

        # Attempt 2: 1.0 * 2^2 = 4.0
        assert config.calculate_delay(2) == 4.0

        # Attempt 3: 1.0 * 2^3 = 8.0
        assert config.calculate_delay(3) == 8.0

        # Attempt 4: min(1.0 * 2^4, 10.0) = 10.0 (capped at max_delay)
        assert config.calculate_delay(4) == 10.0

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2,
            max_delay=10.0,
            jitter=True
        )

        # With jitter, delay should be between base_delay * 0.5 and base_delay
        for attempt in range(5):
            base_delay = min(1.0 * (2 ** attempt), 10.0)
            delay = config.calculate_delay(attempt)
            assert base_delay * 0.5 <= delay <= base_delay

    def test_custom_retry_exceptions(self):
        """Test custom retry exception types."""
        config = RetryConfig(
            retry_on=(ValueError, KeyError)
        )
        assert ValueError in config.retry_on
        assert KeyError in config.retry_on
        assert ConnectionError not in config.retry_on


class TestRetryOnFailureDecorator:
    """Test retry_on_failure decorator."""

    def test_successful_call_no_retry(self):
        """Test successful call requires no retries."""
        call_count = [0]

        @retry_on_failure(max_retries=3, initial_delay=0.01)
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count[0] == 1  # Only called once

    def test_retry_on_connection_error(self):
        """Test retry on ConnectionError."""
        call_count = [0]

        @retry_on_failure(max_retries=3, initial_delay=0.01, jitter=False)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count[0] == 3  # Retried twice, succeeded on 3rd attempt

    def test_max_retries_exceeded(self):
        """Test that max retries are respected."""
        call_count = [0]

        @retry_on_failure(max_retries=2, initial_delay=0.01)
        def always_failing_func():
            call_count[0] += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError, match="Always fails"):
            always_failing_func()

        # Called 3 times: initial + 2 retries
        assert call_count[0] == 3

    def test_non_retryable_exception_immediate_raise(self):
        """Test non-retryable exceptions are raised immediately."""
        call_count = [0]

        @retry_on_failure(max_retries=3, initial_delay=0.01)
        def func_with_value_error():
            call_count[0] += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError, match="Not retryable"):
            func_with_value_error()

        # Only called once (no retries for ValueError)
        assert call_count[0] == 1

    def test_retry_different_exception_types(self):
        """Test retry on different exception types."""
        call_count = [0]

        @retry_on_failure(max_retries=3, initial_delay=0.01, jitter=False)
        def func_with_mixed_errors():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Connection failed")
            elif call_count[0] == 2:
                raise TimeoutError("Timeout")
            else:
                return "success"

        result = func_with_mixed_errors()
        assert result == "success"
        assert call_count[0] == 3

    def test_custom_retry_exceptions(self):
        """Test custom retry exception configuration."""
        call_count = [0]

        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            jitter=False,
            retry_on=(ValueError,)
        )

        @retry_on_failure(config=config)
        def func_retry_value_error():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Retryable")
            return "success"

        result = func_retry_value_error()
        assert result == "success"
        assert call_count[0] == 3


@pytest.mark.asyncio
class TestAsyncRetryOnFailureDecorator:
    """Test async_retry_on_failure decorator."""

    async def test_async_successful_call_no_retry(self):
        """Test async successful call requires no retries."""
        call_count = [0]

        @async_retry_on_failure(max_retries=3, initial_delay=0.01)
        async def async_successful_func():
            call_count[0] += 1
            return "success"

        result = await async_successful_func()
        assert result == "success"
        assert call_count[0] == 1

    async def test_async_retry_on_connection_error(self):
        """Test async retry on ConnectionError."""
        call_count = [0]

        @async_retry_on_failure(max_retries=3, initial_delay=0.01, jitter=False)
        async def async_failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            return "success"

        result = await async_failing_func()
        assert result == "success"
        assert call_count[0] == 3

    async def test_async_max_retries_exceeded(self):
        """Test async max retries are respected."""
        call_count = [0]

        @async_retry_on_failure(max_retries=2, initial_delay=0.01)
        async def async_always_failing():
            call_count[0] += 1
            raise TimeoutError("Always fails")

        with pytest.raises(TimeoutError, match="Always fails"):
            await async_always_failing()

        assert call_count[0] == 3  # Initial + 2 retries

    async def test_async_non_retryable_exception(self):
        """Test async non-retryable exceptions."""
        call_count = [0]

        @async_retry_on_failure(max_retries=3, initial_delay=0.01)
        async def async_value_error():
            call_count[0] += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError, match="Not retryable"):
            await async_value_error()

        assert call_count[0] == 1  # No retries


class TestRetryableOperation:
    """Test RetryableOperation context manager."""

    def test_retryable_operation_success(self):
        """Test successful operation."""
        call_count = [0]

        with RetryableOperation(max_retries=3, initial_delay=0.01) as retry:
            while retry.should_continue():
                call_count[0] += 1
                retry.success()
                break

        assert call_count[0] == 1

    def test_retryable_operation_with_retries(self):
        """Test operation with retries."""
        call_count = [0]

        with RetryableOperation(max_retries=3, initial_delay=0.01, jitter=False) as retry:
            while retry.should_continue():
                call_count[0] += 1
                if call_count[0] < 3:
                    retry.failed(ConnectionError("Retry me"))
                else:
                    retry.success()
                    break

        assert call_count[0] == 3

    def test_retryable_operation_max_retries_exceeded(self):
        """Test max retries are enforced."""
        call_count = [0]

        with pytest.raises(ConnectionError):
            with RetryableOperation(max_retries=2, initial_delay=0.01) as retry:
                while retry.should_continue():
                    call_count[0] += 1
                    retry.failed(ConnectionError("Always fail"))

        assert call_count[0] == 3  # Initial + 2 retries

    def test_retryable_operation_non_retryable_error(self):
        """Test non-retryable errors."""
        call_count = [0]

        with pytest.raises(ValueError, match="Not retryable"):
            with RetryableOperation(max_retries=3, initial_delay=0.01) as retry:
                while retry.should_continue():
                    call_count[0] += 1
                    retry.failed(ValueError("Not retryable"))

        assert call_count[0] == 1  # No retries

    def test_retryable_operation_context_manager_suppression(self):
        """Test context manager exception suppression."""
        call_count = [0]
        config = RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)

        try:
            with RetryableOperation(config=config) as retry:
                call_count[0] += 1
                if call_count[0] == 1:
                    # First attempt fails, should be suppressed
                    raise ConnectionError("Attempt 1")
        except ConnectionError:
            # Should not reach here on first attempt
            pytest.fail("Exception should have been suppressed")

        assert call_count[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
