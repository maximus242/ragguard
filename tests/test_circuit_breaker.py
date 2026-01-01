"""
Tests for circuit breaker functionality.
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

# Skip all tests if qdrant-client is not installed (required for filter generation in integration tests)
pytest.importorskip("qdrant_client")

from ragguard.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerStats,
    CircuitState,
    _circuit_breakers,
    _registry_lock,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset all circuit breakers before each test."""
    reset_all_circuit_breakers()
    # Clear the registry for isolation between tests
    with _registry_lock:
        _circuit_breakers.clear()
    yield
    reset_all_circuit_breakers()
    with _registry_lock:
        _circuit_breakers.clear()


class TestCircuitBreakerBasic:
    """Basic circuit breaker functionality tests."""

    def test_initial_state_closed(self):
        """Circuit breaker should start in CLOSED state."""
        breaker = CircuitBreaker("test-backend")
        assert breaker.state == CircuitState.CLOSED

    def test_success_in_closed_keeps_closed(self):
        """Recording success in closed state should keep it closed."""
        breaker = CircuitBreaker("test-backend")
        breaker.record_success()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_failures_below_threshold_stays_closed(self):
        """Failures below threshold should keep circuit closed."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test-backend", config)

        for _ in range(4):  # 4 failures, threshold is 5
            breaker.record_failure()

        assert breaker.state == CircuitState.CLOSED

    def test_failures_at_threshold_opens_circuit(self):
        """Reaching failure threshold should open the circuit."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test-backend", config)

        for _ in range(5):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Success should reset failure count in closed state."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test-backend", config)

        # 4 failures
        for _ in range(4):
            breaker.record_failure()

        # Success resets count
        breaker.record_success()

        # 4 more failures should not open circuit (total would be 8 without reset)
        for _ in range(4):
            breaker.record_failure()

        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerOpen:
    """Tests for OPEN state behavior."""

    def test_check_raises_when_open(self):
        """check() should raise CircuitBreakerOpen when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker("test-backend", config)
        breaker.record_failure()  # Opens circuit

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            breaker.check()

        assert exc_info.value.backend == "test-backend"
        assert exc_info.value.time_until_retry > 0

    def test_stats_track_rejected_requests(self):
        """Statistics should track rejected requests."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker("test-backend", config)
        breaker.record_failure()

        # Try to make requests while open
        for _ in range(3):
            try:
                breaker.check()
            except CircuitBreakerOpen:
                pass

        stats = breaker.stats
        assert stats.total_rejected == 3


class TestCircuitBreakerHalfOpen:
    """Tests for HALF_OPEN state behavior."""

    def test_timeout_transitions_to_half_open(self):
        """After timeout, circuit should transition to HALF_OPEN."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=30.0)
        breaker = CircuitBreaker("test-backend", config)

        # Record failure at t=0
        with patch('ragguard.circuit_breaker.time') as mock_time:
            mock_time.time.return_value = 0.0
            breaker.record_failure()
            assert breaker.state == CircuitState.OPEN

            # Simulate time passing beyond timeout
            mock_time.time.return_value = 31.0

            # Next check should allow request and transition to HALF_OPEN
            breaker.check()  # Should not raise
            assert breaker.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        """Enough successes in half-open should close the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout=30.0
        )
        breaker = CircuitBreaker("test-backend", config)

        with patch('ragguard.circuit_breaker.time') as mock_time:
            mock_time.time.return_value = 0.0
            breaker.record_failure()

            # Simulate time passing and transition to half-open
            mock_time.time.return_value = 31.0
            breaker.check()

            # Record enough successes
            breaker.record_success()
            breaker.record_success()

            assert breaker.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self):
        """Single failure in half-open should reopen circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=30.0)
        breaker = CircuitBreaker("test-backend", config)

        with patch('ragguard.circuit_breaker.time') as mock_time:
            mock_time.time.return_value = 0.0
            breaker.record_failure()

            # Simulate time passing and transition to half-open
            mock_time.time.return_value = 31.0
            breaker.check()
            assert breaker.state == CircuitState.HALF_OPEN

            # Single failure reopens
            breaker.record_failure()
            assert breaker.state == CircuitState.OPEN

    def test_half_open_limits_concurrent_calls(self):
        """Half-open state should limit concurrent calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout=30.0,
            half_open_max_calls=2
        )
        breaker = CircuitBreaker("test-backend", config)

        with patch('ragguard.circuit_breaker.time') as mock_time:
            mock_time.time.return_value = 0.0
            breaker.record_failure()

            # Simulate time passing
            mock_time.time.return_value = 31.0

            # First two calls should be allowed
            breaker.check()  # Transitions to half-open and counts as first call
            breaker.check()  # Second call allowed

            # Third call should be rejected
            with pytest.raises(CircuitBreakerOpen):
                breaker.check()


class TestCircuitBreakerReset:
    """Tests for manual reset functionality."""

    def test_reset_closes_open_circuit(self):
        """reset() should close an open circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test-backend", config)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_reset_clears_counters(self):
        """reset() should clear failure and success counters."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test-backend", config)

        for _ in range(4):
            breaker.record_failure()

        breaker.reset()

        # After reset, 4 more failures should not open (count was reset)
        for _ in range(4):
            breaker.record_failure()

        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    """Tests for global circuit breaker registry."""

    def test_get_circuit_breaker_creates_new(self):
        """get_circuit_breaker should create new breaker if none exists."""
        breaker = get_circuit_breaker("new-backend")
        assert breaker is not None
        assert breaker.backend == "new-backend"

    def test_get_circuit_breaker_returns_same_instance(self):
        """get_circuit_breaker should return same instance for same backend."""
        breaker1 = get_circuit_breaker("shared-backend")
        breaker2 = get_circuit_breaker("shared-backend")
        assert breaker1 is breaker2

    def test_different_backends_get_different_breakers(self):
        """Different backends should get different circuit breakers."""
        breaker1 = get_circuit_breaker("backend-a")
        breaker2 = get_circuit_breaker("backend-b")
        assert breaker1 is not breaker2

    def test_reset_all_circuit_breakers(self):
        """reset_all_circuit_breakers should reset all breakers."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker1 = get_circuit_breaker("backend-1", config)
        breaker2 = get_circuit_breaker("backend-2", config)

        breaker1.record_failure()
        breaker2.record_failure()

        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN

        reset_all_circuit_breakers()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED


class TestCircuitBreakerStats:
    """Tests for statistics tracking."""

    def test_stats_snapshot_is_independent(self):
        """Stats property should return independent snapshot."""
        breaker = CircuitBreaker("test-backend")
        stats1 = breaker.stats
        breaker.record_failure()
        stats2 = breaker.stats

        # stats1 should not be affected by subsequent changes
        assert stats1.total_failures == 0
        assert stats2.total_failures == 1

    def test_total_successes_tracked(self):
        """Total successes should be tracked across state changes."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=30.0)
        breaker = CircuitBreaker("test-backend", config)

        with patch('ragguard.circuit_breaker.time') as mock_time:
            mock_time.time.return_value = 0.0
            breaker.record_success()
            breaker.record_success()
            breaker.record_failure()  # Opens circuit

            # Simulate time passing
            mock_time.time.return_value = 31.0
            breaker.check()  # Half-open
            breaker.record_success()

            assert breaker.stats.total_successes == 3


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_failures(self):
        """Circuit breaker should handle concurrent failures safely."""
        config = CircuitBreakerConfig(failure_threshold=100)
        breaker = CircuitBreaker("test-backend", config)

        def record_failures():
            for _ in range(50):
                breaker.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded all 200 failures
        assert breaker.stats.total_failures == 200

    def test_concurrent_check_and_record(self):
        """Circuit breaker should handle concurrent check and record safely."""
        config = CircuitBreakerConfig(failure_threshold=50, timeout=0.1)
        breaker = CircuitBreaker("test-backend", config)

        errors = []
        checks_passed = [0]

        def checker():
            for _ in range(100):
                try:
                    breaker.check()
                    checks_passed[0] += 1
                except CircuitBreakerOpen:
                    pass
                except Exception as e:
                    errors.append(e)

        def failer():
            for _ in range(100):
                breaker.record_failure()

        threads = [
            threading.Thread(target=checker),
            threading.Thread(target=checker),
            threading.Thread(target=failer),
            threading.Thread(target=failer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0


class TestCircuitBreakerIntegration:
    """Integration tests with retrievers using a concrete test retriever."""

    def _create_test_retriever(self, policy, enable_circuit_breaker=True,
                               circuit_breaker_config=None, enable_retry=True,
                               execute_search_side_effect=None):
        """Create a concrete test retriever subclass for testing."""
        from ragguard.retrievers.base import BaseSecureRetriever

        class TestRetriever(BaseSecureRetriever):
            def __init__(self, **kwargs):
                self._execute_search_side_effect = kwargs.pop('execute_search_side_effect', None)
                super().__init__(client=None, collection="test", **kwargs)

            @property
            def backend_name(self) -> str:
                # Use "qdrant" so filter generation works
                return "qdrant"

            def _execute_search(self, query, filter, limit, **kwargs):
                if self._execute_search_side_effect:
                    if callable(self._execute_search_side_effect):
                        return self._execute_search_side_effect()
                    raise self._execute_search_side_effect
                return []

            def _check_backend_health(self):
                return {"connection_alive": True}

        return TestRetriever(
            policy=policy,
            enable_circuit_breaker=enable_circuit_breaker,
            circuit_breaker_config=circuit_breaker_config,
            enable_retry=enable_retry,
            execute_search_side_effect=execute_search_side_effect
        )

    def test_retriever_with_circuit_breaker(self):
        """Test circuit breaker integration with retriever."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
            "default": "deny"
        })

        retriever = self._create_test_retriever(
            policy=policy,
            enable_circuit_breaker=True,
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2)
        )

        # Should have circuit breaker enabled
        stats = retriever.get_circuit_breaker_stats()
        assert stats is not None
        assert stats["state"] == "closed"

    def test_retriever_circuit_breaker_disabled(self):
        """Test retriever with circuit breaker disabled."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
            "default": "deny"
        })

        retriever = self._create_test_retriever(
            policy=policy,
            enable_circuit_breaker=False
        )

        # Should return None when disabled
        stats = retriever.get_circuit_breaker_stats()
        assert stats is None

    def test_retriever_opens_circuit_on_failures(self):
        """Test that retriever opens circuit after enough failures."""
        from ragguard import CircuitBreakerOpen, Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
            "default": "deny"
        })

        retriever = self._create_test_retriever(
            policy=policy,
            enable_circuit_breaker=True,
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
            enable_retry=False,
            execute_search_side_effect=ConnectionError("Backend down")
        )

        user = {"id": "test-user"}
        query = [0.1, 0.2, 0.3]

        # First two failures should open the circuit
        for _ in range(2):
            try:
                retriever.search(query, user)
            except Exception:
                pass

        # Circuit should now be open
        stats = retriever.get_circuit_breaker_stats()
        assert stats["state"] == "open"

        # Next request should be rejected immediately
        with pytest.raises(CircuitBreakerOpen):
            retriever.search(query, user)

    def test_retriever_reset_circuit_breaker(self):
        """Test manual circuit breaker reset."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
            "default": "deny"
        })

        retriever = self._create_test_retriever(
            policy=policy,
            enable_circuit_breaker=True,
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1),
            enable_retry=False,
            execute_search_side_effect=ConnectionError("Backend down")
        )

        user = {"id": "test-user"}
        query = [0.1, 0.2, 0.3]

        # Trigger circuit open
        try:
            retriever.search(query, user)
        except Exception:
            pass

        assert retriever.get_circuit_breaker_stats()["state"] == "open"

        # Reset the circuit breaker
        retriever.reset_circuit_breaker()

        assert retriever.get_circuit_breaker_stats()["state"] == "closed"

    def test_search_with_timeout_success(self):
        """Test search with timeout that succeeds within limit."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
            "default": "deny"
        })

        retriever = self._create_test_retriever(
            policy=policy,
            enable_circuit_breaker=False,
            enable_retry=False
        )

        user = {"id": "test-user"}
        query = [0.1, 0.2, 0.3]

        # Search with a generous timeout should succeed
        results = retriever.search(query, user, timeout=10.0)
        assert results == []  # Our test retriever returns empty list

    def test_search_with_timeout_expires(self):
        """Test search that exceeds timeout raises RetrieverTimeoutError."""
        import time as time_module

        from ragguard import Policy
        from ragguard.exceptions import RetrieverTimeoutError

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
            "default": "deny"
        })

        def slow_search():
            time_module.sleep(5)  # Sleep longer than timeout
            return []

        retriever = self._create_test_retriever(
            policy=policy,
            enable_circuit_breaker=False,
            enable_retry=False,
            execute_search_side_effect=slow_search
        )

        user = {"id": "test-user"}
        query = [0.1, 0.2, 0.3]

        # Search with a short timeout should raise TimeoutError
        with pytest.raises(RetrieverTimeoutError):
            retriever.search(query, user, timeout=0.1)

    def test_search_with_no_timeout(self):
        """Test search with timeout=0 (disabled) works."""
        from ragguard import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
            "default": "deny"
        })

        retriever = self._create_test_retriever(
            policy=policy,
            enable_circuit_breaker=False,
            enable_retry=False
        )

        user = {"id": "test-user"}
        query = [0.1, 0.2, 0.3]

        # Search with timeout=0 should disable timeout
        results = retriever.search(query, user, timeout=0)
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
