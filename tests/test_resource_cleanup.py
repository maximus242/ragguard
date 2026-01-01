"""
Tests for resource cleanup and lifecycle management.

Ensures proper cleanup of:
- Thread pool executors
- Filter caches
- Environment variable configuration
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from ragguard.config import (
    SecureRetrieverConfig,
    _env_bool,
    _env_float,
    _env_int,
    _env_str,
)
from ragguard.filters.cache import FilterCache
from ragguard.retry import (
    RetryConfig,
    _shared_executor,
    get_shared_executor,
    shutdown_executor,
)


class TestSharedExecutorLifecycle:
    """Test shared thread pool executor lifecycle management."""

    def setup_method(self):
        """Ensure clean state before each test."""
        shutdown_executor(wait=True)

    def teardown_method(self):
        """Clean up after each test."""
        shutdown_executor(wait=True)

    def test_get_shared_executor_creates_singleton(self):
        """Test that get_shared_executor creates a singleton."""
        executor1 = get_shared_executor()
        executor2 = get_shared_executor()

        assert executor1 is executor2
        assert isinstance(executor1, ThreadPoolExecutor)

    def test_get_shared_executor_default_workers(self):
        """Test default max_workers configuration."""
        executor = get_shared_executor()

        # ThreadPoolExecutor doesn't expose max_workers directly,
        # but we can verify it's created
        assert executor is not None
        assert executor._max_workers == 32  # Default value

    def test_get_shared_executor_custom_workers(self):
        """Test custom max_workers configuration."""
        executor = get_shared_executor(max_workers=16)

        # First call sets the workers, subsequent calls return same executor
        assert executor._max_workers == 16

    def test_shutdown_executor_cleans_up(self):
        """Test that shutdown_executor properly cleans up."""
        import ragguard.retry as retry_module

        executor = get_shared_executor()
        assert retry_module._shared_executor is not None

        shutdown_executor(wait=True)
        assert retry_module._shared_executor is None

    def test_shutdown_executor_allows_recreation(self):
        """Test that executor can be recreated after shutdown."""
        executor1 = get_shared_executor()
        shutdown_executor(wait=True)
        executor2 = get_shared_executor()

        assert executor1 is not executor2
        assert isinstance(executor2, ThreadPoolExecutor)

    def test_shutdown_idempotent(self):
        """Test that multiple shutdowns are safe."""
        get_shared_executor()
        shutdown_executor(wait=True)
        shutdown_executor(wait=True)  # Should not raise
        shutdown_executor(wait=True)  # Should not raise

    def test_executor_thread_naming(self):
        """Test that executor threads have correct naming prefix."""
        executor = get_shared_executor()

        # Submit a task and verify it runs
        result = executor.submit(lambda: "test")
        assert result.result(timeout=5) == "test"

        # Thread prefix is set in executor creation
        assert executor._thread_name_prefix == "ragguard_async_"

    @pytest.mark.asyncio
    async def test_executor_with_async_operation(self):
        """Test executor works correctly with async operations."""
        executor = get_shared_executor()
        loop = asyncio.get_running_loop()

        def sync_work():
            return 42

        result = await loop.run_in_executor(executor, sync_work)
        assert result == 42

    def test_executor_concurrent_tasks(self):
        """Test executor handles concurrent tasks properly."""
        executor = get_shared_executor()
        results = []

        def task(n):
            return n * 2

        futures = [executor.submit(task, i) for i in range(10)]
        results = [f.result(timeout=5) for f in futures]

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


class TestFilterCacheCleanup:
    """Test filter cache resource cleanup."""

    def test_cache_creation_with_max_size(self):
        """Test cache respects max_size configuration."""
        cache = FilterCache(max_size=10)

        # Fill cache beyond max_size
        for i in range(15):
            cache.set(f"key_{i}", f"value_{i}")

        # Cache should only have max_size items (LRU eviction)
        assert len(cache) <= 10

    def test_cache_clear(self):
        """Test cache can be cleared."""
        cache = FilterCache(max_size=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.get("key1") == "value1"

        cache.invalidate_all()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = FilterCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' to make it recently used
        cache.get("a")

        # Add new item, should evict 'b' (least recently used)
        cache.set("d", 4)

        assert cache.get("a") == 1  # Still present (recently used)
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4


class TestEnvironmentVariableConfiguration:
    """Test environment variable configuration helpers."""

    def test_env_bool_true_values(self):
        """Test _env_bool parses true values correctly."""
        for true_val in ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"TEST_BOOL": true_val}):
                assert _env_bool("TEST_BOOL", False) is True

    def test_env_bool_false_values(self):
        """Test _env_bool parses false values correctly."""
        for false_val in ["false", "FALSE", "0", "no", "off", "anything_else"]:
            with patch.dict(os.environ, {"TEST_BOOL": false_val}):
                assert _env_bool("TEST_BOOL", True) is False

    def test_env_bool_default(self):
        """Test _env_bool returns default when not set."""
        # Ensure env var is not set
        if "TEST_BOOL_UNSET" in os.environ:
            del os.environ["TEST_BOOL_UNSET"]

        assert _env_bool("TEST_BOOL_UNSET", True) is True
        assert _env_bool("TEST_BOOL_UNSET", False) is False

    def test_env_bool_empty_string(self):
        """Test _env_bool returns default for empty string."""
        with patch.dict(os.environ, {"TEST_BOOL": ""}):
            assert _env_bool("TEST_BOOL", True) is True
            assert _env_bool("TEST_BOOL", False) is False

    def test_env_int_valid(self):
        """Test _env_int parses valid integers."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert _env_int("TEST_INT", 0) == 42

        with patch.dict(os.environ, {"TEST_INT": "-10"}):
            assert _env_int("TEST_INT", 0) == -10

    def test_env_int_invalid(self):
        """Test _env_int returns default for invalid values."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert _env_int("TEST_INT", 99) == 99

        with patch.dict(os.environ, {"TEST_INT": "3.14"}):
            assert _env_int("TEST_INT", 99) == 99

    def test_env_int_default(self):
        """Test _env_int returns default when not set."""
        if "TEST_INT_UNSET" in os.environ:
            del os.environ["TEST_INT_UNSET"]

        assert _env_int("TEST_INT_UNSET", 123) == 123

    def test_env_float_valid(self):
        """Test _env_float parses valid floats."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert _env_float("TEST_FLOAT", 0.0) == 3.14

        with patch.dict(os.environ, {"TEST_FLOAT": "-2.5"}):
            assert _env_float("TEST_FLOAT", 0.0) == -2.5

        with patch.dict(os.environ, {"TEST_FLOAT": "42"}):
            assert _env_float("TEST_FLOAT", 0.0) == 42.0

    def test_env_float_invalid(self):
        """Test _env_float returns default for invalid values."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_number"}):
            assert _env_float("TEST_FLOAT", 1.5) == 1.5

    def test_env_float_default(self):
        """Test _env_float returns default when not set."""
        if "TEST_FLOAT_UNSET" in os.environ:
            del os.environ["TEST_FLOAT_UNSET"]

        assert _env_float("TEST_FLOAT_UNSET", 5.5) == 5.5

    def test_env_str_value(self):
        """Test _env_str gets string values."""
        with patch.dict(os.environ, {"TEST_STR": "hello"}):
            assert _env_str("TEST_STR", "default") == "hello"

    def test_env_str_default(self):
        """Test _env_str returns default when not set."""
        if "TEST_STR_UNSET" in os.environ:
            del os.environ["TEST_STR_UNSET"]

        assert _env_str("TEST_STR_UNSET", "default") == "default"


class TestSecureRetrieverConfigFromEnv:
    """Test SecureRetrieverConfig.from_env() method."""

    def test_from_env_default_values(self):
        """Test from_env uses defaults when no env vars set."""
        # Clear any existing env vars
        env_vars = [
            "RAGGUARD_ENABLE_VALIDATION",
            "RAGGUARD_ENABLE_RETRY",
            "RAGGUARD_MAX_RETRIES",
            "RAGGUARD_RETRY_INITIAL_DELAY",
            "RAGGUARD_RETRY_MAX_DELAY",
            "RAGGUARD_REQUEST_TIMEOUT",
            "RAGGUARD_TOTAL_TIMEOUT",
            "RAGGUARD_ENABLE_CACHE",
            "RAGGUARD_CACHE_SIZE",
            "RAGGUARD_ENABLE_AUDIT",
        ]

        with patch.dict(os.environ, {}, clear=True):
            # Re-add PATH and other essentials
            config = SecureRetrieverConfig.from_env()

            assert config.enable_validation is True
            assert config.enable_retry is True
            assert config.enable_filter_cache is True
            assert config.filter_cache_size == 1000
            assert config.enable_audit is False

            assert config.retry_config.max_retries == 3
            assert config.retry_config.initial_delay == 0.1
            assert config.retry_config.max_delay == 10.0
            assert config.retry_config.request_timeout == 30.0
            assert config.retry_config.total_timeout == 120.0

    def test_from_env_custom_values(self):
        """Test from_env reads custom env var values."""
        env_vars = {
            "RAGGUARD_ENABLE_VALIDATION": "false",
            "RAGGUARD_ENABLE_RETRY": "true",
            "RAGGUARD_MAX_RETRIES": "5",
            "RAGGUARD_RETRY_INITIAL_DELAY": "0.5",
            "RAGGUARD_RETRY_MAX_DELAY": "30.0",
            "RAGGUARD_REQUEST_TIMEOUT": "15.0",
            "RAGGUARD_TOTAL_TIMEOUT": "60.0",
            "RAGGUARD_ENABLE_CACHE": "true",
            "RAGGUARD_CACHE_SIZE": "5000",
            "RAGGUARD_ENABLE_AUDIT": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = SecureRetrieverConfig.from_env()

            assert config.enable_validation is False
            assert config.enable_retry is True
            assert config.enable_filter_cache is True
            assert config.filter_cache_size == 5000
            assert config.enable_audit is True

            assert config.retry_config.max_retries == 5
            assert config.retry_config.initial_delay == 0.5
            assert config.retry_config.max_delay == 30.0
            assert config.retry_config.request_timeout == 15.0
            assert config.retry_config.total_timeout == 60.0

    def test_from_env_validation_config(self):
        """Test from_env reads validation configuration."""
        env_vars = {
            "RAGGUARD_MAX_DICT_SIZE": "50",
            "RAGGUARD_MAX_STRING_LENGTH": "5000",
            "RAGGUARD_MAX_NESTING_DEPTH": "5",
            "RAGGUARD_MAX_ARRAY_LENGTH": "500",
        }

        with patch.dict(os.environ, env_vars):
            config = SecureRetrieverConfig.from_env()

            assert config.validation_config.max_dict_size == 50
            assert config.validation_config.max_string_length == 5000
            assert config.validation_config.max_nesting_depth == 5
            assert config.validation_config.max_array_length == 500

    def test_from_env_partial_override(self):
        """Test from_env with partial env var overrides."""
        env_vars = {
            "RAGGUARD_MAX_RETRIES": "10",
            "RAGGUARD_ENABLE_AUDIT": "true",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = SecureRetrieverConfig.from_env()

            # Overridden values
            assert config.retry_config.max_retries == 10
            assert config.enable_audit is True

            # Default values
            assert config.enable_validation is True
            assert config.retry_config.initial_delay == 0.1


class TestRetryConfigWithTimeouts:
    """Test RetryConfig timeout configuration."""

    def test_default_timeouts(self):
        """Test default timeout values."""
        config = RetryConfig()

        assert config.request_timeout == 30.0
        assert config.total_timeout == 120.0

    def test_custom_timeouts(self):
        """Test custom timeout configuration."""
        config = RetryConfig(
            request_timeout=10.0,
            total_timeout=60.0
        )

        assert config.request_timeout == 10.0
        assert config.total_timeout == 60.0

    def test_timeout_in_from_env_config(self):
        """Test timeouts are configured via from_env."""
        env_vars = {
            "RAGGUARD_REQUEST_TIMEOUT": "5.0",
            "RAGGUARD_TOTAL_TIMEOUT": "30.0",
        }

        with patch.dict(os.environ, env_vars):
            config = SecureRetrieverConfig.from_env()

            assert config.retry_config.request_timeout == 5.0
            assert config.retry_config.total_timeout == 30.0


class TestResourceCleanupIntegration:
    """Integration tests for resource cleanup across components."""

    def setup_method(self):
        """Clean state before each test."""
        shutdown_executor(wait=True)

    def teardown_method(self):
        """Clean up after each test."""
        shutdown_executor(wait=True)

    @pytest.mark.asyncio
    async def test_async_operations_cleanup(self):
        """Test that async operations clean up properly."""
        executor = get_shared_executor()
        loop = asyncio.get_running_loop()

        # Run several async operations
        tasks = []
        for i in range(5):
            tasks.append(
                loop.run_in_executor(executor, lambda x=i: x * 2)
            )

        results = await asyncio.gather(*tasks)
        assert results == [0, 2, 4, 6, 8]

        # Shutdown should complete without hanging
        shutdown_executor(wait=True)

    def test_cache_and_executor_independent(self):
        """Test cache and executor can be cleaned up independently."""
        # Create cache
        cache = FilterCache(max_size=100)
        cache.set("test", "value")

        # Create executor
        executor = get_shared_executor()
        executor.submit(lambda: "test").result()

        # Clear cache (executor still works)
        cache.invalidate_all()
        result = executor.submit(lambda: "after_cache_clear").result()
        assert result == "after_cache_clear"

        # Shutdown executor (cache still works)
        shutdown_executor(wait=True)
        cache.set("after_shutdown", "value")
        assert cache.get("after_shutdown") == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
