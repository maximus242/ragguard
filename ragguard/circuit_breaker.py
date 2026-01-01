"""
Circuit breaker implementation for RAGGuard.

Prevents cascading failures by fast-failing when a backend is experiencing issues.
Uses a three-state model: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing recovery).
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .exceptions import RetrieverError
from .logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing fast, rejecting requests
    HALF_OPEN = "half_open"  # Testing if backend recovered


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit (default: 5)
        success_threshold: Number of successes in half-open before closing (default: 2)
        timeout: Seconds to wait before transitioning from open to half-open (default: 30)
        half_open_max_calls: Max concurrent calls allowed in half-open state (default: 3)
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreakerOpen(RetrieverError):
    """Raised when circuit breaker is open and rejecting requests."""

    def __init__(self, backend: str, time_until_retry: float):
        self.backend = backend
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker OPEN for {backend}. "
            f"Retry in {time_until_retry:.1f}s. "
            f"Backend is experiencing failures."
        )


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker instance."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_failures: int = 0
    total_successes: int = 0
    total_rejected: int = 0


class CircuitBreaker:
    """
    Circuit breaker for a single backend.

    Tracks failures and automatically opens the circuit when failure_threshold
    is exceeded. After timeout seconds, allows limited requests through to
    test if the backend has recovered.

    Example:
        ```python
        breaker = CircuitBreaker("qdrant", CircuitBreakerConfig(
            failure_threshold=5,
            timeout=30.0
        ))

        # In retriever
        def search(...):
            breaker.check()  # Raises if circuit is open
            try:
                result = backend.search(...)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        ```
    """

    def __init__(self, backend: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker for a backend.

        Args:
            backend: Name of the backend (for logging/metrics)
            config: Circuit breaker configuration
        """
        self.backend = backend
        self.config = config or CircuitBreakerConfig()
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._stats.state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitBreakerStats(
                state=self._stats.state,
                failure_count=self._stats.failure_count,
                success_count=self._stats.success_count,
                last_failure_time=self._stats.last_failure_time,
                last_state_change=self._stats.last_state_change,
                total_failures=self._stats.total_failures,
                total_successes=self._stats.total_successes,
                total_rejected=self._stats.total_rejected
            )

    def check(self) -> None:
        """
        Check if request should be allowed through.

        Raises:
            CircuitBreakerOpen: If circuit is open and rejecting requests
        """
        with self._lock:
            state = self._stats.state

            if state == CircuitState.CLOSED:
                return  # Allow request

            if state == CircuitState.OPEN:
                # Check if timeout has passed
                time_since_open = time.time() - self._stats.last_state_change
                if time_since_open >= self.config.timeout:
                    # Transition to half-open
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 1
                    return  # Allow this request as test
                else:
                    # Still open, reject
                    self._stats.total_rejected += 1
                    time_until_retry = self.config.timeout - time_since_open
                    raise CircuitBreakerOpen(self.backend, time_until_retry)

            if state == CircuitState.HALF_OPEN:
                # Allow limited concurrent calls
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return  # Allow request
                else:
                    # Too many concurrent half-open calls
                    self._stats.total_rejected += 1
                    raise CircuitBreakerOpen(self.backend, 0.0)

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._stats.total_successes += 1
            state = self._stats.state

            if state == CircuitState.HALF_OPEN:
                self._stats.success_count += 1
                if self._stats.success_count >= self.config.success_threshold:
                    # Backend recovered, close circuit
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        f"Circuit breaker CLOSED for {self.backend} - backend recovered",
                        extra={"extra_fields": {"backend": self.backend}}
                    )

            elif state == CircuitState.CLOSED:
                # Reset failure count on success
                self._stats.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._stats.total_failures += 1
            self._stats.last_failure_time = time.time()
            state = self._stats.state

            if state == CircuitState.HALF_OPEN:
                # Single failure in half-open reopens circuit
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit breaker OPEN for {self.backend} - failure during recovery test",
                    extra={"extra_fields": {"backend": self.backend}}
                )

            elif state == CircuitState.CLOSED:
                self._stats.failure_count += 1
                if self._stats.failure_count >= self.config.failure_threshold:
                    # Too many failures, open circuit
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        f"Circuit breaker OPEN for {self.backend} - "
                        f"{self._stats.failure_count} failures exceeded threshold",
                        extra={"extra_fields": {
                            "backend": self.backend,
                            "failure_count": self._stats.failure_count,
                            "threshold": self.config.failure_threshold
                        }}
                    )

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            logger.info(
                f"Circuit breaker manually reset for {self.backend}",
                extra={"extra_fields": {"backend": self.backend}}
            )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._stats.state
        self._stats.state = new_state
        self._stats.last_state_change = time.time()
        self._stats.failure_count = 0
        self._stats.success_count = 0
        self._half_open_calls = 0

        logger.debug(
            f"Circuit breaker {self.backend}: {old_state.value} -> {new_state.value}",
            extra={"extra_fields": {
                "backend": self.backend,
                "old_state": old_state.value,
                "new_state": new_state.value
            }}
        )


# Global registry of circuit breakers per backend
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    backend: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a backend.

    Circuit breakers are shared across all retriever instances for the same backend,
    providing system-wide protection against failing backends.

    Args:
        backend: Backend name (e.g., "qdrant", "chromadb")
        config: Optional configuration (only used on first creation)

    Returns:
        CircuitBreaker instance for the backend
    """
    with _registry_lock:
        if backend not in _circuit_breakers:
            _circuit_breakers[backend] = CircuitBreaker(backend, config)
        return _circuit_breakers[backend]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state. Useful for testing."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()


def circuit_breaker_protected(backend: str):
    """
    Decorator to protect a function with circuit breaker.

    Args:
        backend: Backend name for circuit breaker

    Example:
        ```python
        @circuit_breaker_protected("qdrant")
        def search_qdrant(...):
            return client.search(...)
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            breaker = get_circuit_breaker(backend)
            breaker.check()  # May raise CircuitBreakerOpen
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except CircuitBreakerOpen:
                raise  # Don't record our own exception as failure
            except Exception:
                breaker.record_failure()
                raise
        return wrapper
    return decorator


def async_circuit_breaker_protected(backend: str):
    """
    Async decorator to protect a coroutine with circuit breaker.

    Args:
        backend: Backend name for circuit breaker

    Example:
        ```python
        @async_circuit_breaker_protected("qdrant")
        async def search_qdrant(...):
            return await client.search(...)
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            breaker = get_circuit_breaker(backend)
            breaker.check()  # May raise CircuitBreakerOpen
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except CircuitBreakerOpen:
                raise
            except Exception:
                breaker.record_failure()
                raise
        return wrapper
    return decorator
