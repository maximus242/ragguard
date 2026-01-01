"""
Retry logic with exponential backoff for RAGGuard retrievers.

Provides resilience against transient failures like:
- Network timeouts
- Database connection errors
- Rate limiting
- Temporary service unavailability
"""

import asyncio
import atexit
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

from .exceptions import (
    RETRYABLE_EXCEPTIONS,
)
from .logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_WORKERS = 32
_DEFAULT_MAX_QUEUE = 1000
_shared_executor: Optional[ThreadPoolExecutor] = None
_shared_semaphore: Optional[threading.Semaphore] = None
_executor_lock = threading.Lock()


def get_shared_executor(max_workers: int = _DEFAULT_MAX_WORKERS) -> ThreadPoolExecutor:
    """
    Get or create a shared thread pool executor for async operations.

    Using a shared, bounded executor prevents resource exhaustion under load.
    Thread-safe: uses a lock to prevent race conditions during creation.

    Args:
        max_workers: Maximum number of worker threads (default: 32)

    Returns:
        Shared ThreadPoolExecutor instance
    """
    global _shared_executor, _shared_semaphore
    if _shared_executor is not None:
        return _shared_executor

    with _executor_lock:
        if _shared_executor is None:
            _shared_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="ragguard_async_"
            )
            _shared_semaphore = threading.Semaphore(_DEFAULT_MAX_QUEUE)
            atexit.register(_cleanup_executor_on_exit)
    return _shared_executor


def get_executor_semaphore() -> threading.Semaphore:
    """
    Get the semaphore for backpressure on the shared executor.

    This semaphore limits the number of pending tasks in the queue.
    Callers should acquire before submitting work and release after completion.

    Returns:
        Semaphore for backpressure control

    Raises:
        RuntimeError: If semaphore could not be created
    """
    global _shared_semaphore
    if _shared_semaphore is None:
        get_shared_executor()
    # Verify semaphore was created (could fail if executor creation failed)
    if _shared_semaphore is None:
        raise RuntimeError(
            "Failed to create executor semaphore. "
            "The shared thread pool executor may have failed to initialize."
        )
    return _shared_semaphore


async def run_in_executor_with_backpressure(
    func: Callable,
    *args,
    timeout: Optional[float] = 30.0
) -> Any:
    """
    Run a sync function in the shared thread pool with backpressure.

    Prevents queue overflow by blocking when too many tasks are pending.
    Uses a semaphore to limit the number of queued tasks.

    Args:
        func: Synchronous function to run
        *args: Arguments to pass to the function
        timeout: Timeout for acquiring semaphore slot (default: 30s)

    Returns:
        Result of the function

    Raises:
        TimeoutError: If unable to acquire queue slot within timeout
    """
    loop = asyncio.get_running_loop()
    executor = get_shared_executor()
    semaphore = get_executor_semaphore()

    # Use non-blocking acquire with async polling to avoid blocking the event loop
    start_time = time.monotonic()
    while True:
        if semaphore.acquire(blocking=False):
            break
        if timeout is not None and time.monotonic() - start_time >= timeout:
            raise TimeoutError(
                f"Thread pool queue full: unable to submit task within {timeout}s. "
                f"Consider increasing queue size or reducing load."
            )
        # Yield to event loop before retrying (10ms poll interval)
        await asyncio.sleep(0.01)

    try:
        return await loop.run_in_executor(executor, func, *args)
    finally:
        semaphore.release()


def _cleanup_executor_on_exit() -> None:
    """Cleanup handler for interpreter shutdown."""
    global _shared_executor
    if _shared_executor is not None:
        try:
            _shared_executor.shutdown(wait=False)
        except Exception as e:
            # Log but don't raise - this is a cleanup handler during shutdown
            # Raising here could mask the original exception or cause issues
            logger.debug(
                "Failed to shutdown executor during cleanup",
                extra={"extra_fields": {"error": str(e)}}
            )


def shutdown_executor(wait: bool = True) -> None:
    """
    Shutdown the shared thread pool executor.

    Call this during application shutdown to properly clean up resources.
    Thread-safe: uses a lock to prevent race conditions.

    Args:
        wait: If True, wait for all pending tasks to complete
    """
    global _shared_executor, _shared_semaphore
    with _executor_lock:
        if _shared_executor is not None:
            _shared_executor.shutdown(wait=wait)
            _shared_executor = None
            _shared_semaphore = None


class RetryConfig:
    """
    Configuration for retry behavior with operation timeouts.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 0.1)
        max_delay: Maximum delay between retries in seconds (default: 10.0)
        exponential_base: Base for exponential backoff (default: 2)
        jitter: Add random jitter to prevent thundering herd (default: True)
        retry_on: Tuple of exception types to retry on
        request_timeout: Timeout for individual operations in seconds (default: 30.0)
        total_timeout: Total timeout for all retries in seconds (default: 120.0)

    Example:
        ```python
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=2,
            request_timeout=10.0,  # Each request times out after 10s
            total_timeout=60.0     # Give up after 60s total
        )
        ```
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        exponential_base: float = 2,
        jitter: bool = True,
        retry_on: Optional[Tuple[Type[Exception], ...]] = None,
        request_timeout: float = 30.0,
        total_timeout: float = 120.0
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.request_timeout = request_timeout
        self.total_timeout = total_timeout

        # Default exceptions to retry on - includes RAGGuard's retryable exceptions
        if retry_on is None:
            retry_on = (
                ConnectionError,
                TimeoutError,
                OSError,
            ) + RETRYABLE_EXCEPTIONS
        self.retry_on = retry_on

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt using exponential backoff.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            import random
            # Add jitter: random value between 0 and delay (not security-sensitive)
            delay = delay * (0.5 + random.random() * 0.5)  # nosec B311

        return delay


def retry_on_failure(
    config: Optional[RetryConfig] = None,
    **config_kwargs
) -> Callable:
    """
    Decorator to add retry logic with exponential backoff to a function.

    Args:
        config: RetryConfig instance (optional)
        **config_kwargs: Arguments to create RetryConfig if not provided

    Example:
        ```python
        @retry_on_failure(max_retries=5, initial_delay=0.5)
        def search(query, user, limit=10):
            # May fail with network errors
            return client.search(...)
        ```
    """
    if config is None:
        config = RetryConfig(**config_kwargs)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except config.retry_on as e:
                    last_exception = e

                    if attempt >= config.max_retries:
                        # Max retries reached, raise
                        logger.error(
                            "Function failed after max retries",
                            extra={
                                "extra_fields": {
                                    "function": func.__name__,
                                    "max_retries": config.max_retries,
                                    "error": str(e)
                                }
                            }
                        )
                        raise

                    # Calculate delay and retry
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        "Retrying after failure",
                        extra={
                            "extra_fields": {
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": config.max_retries + 1,
                                "delay_seconds": round(delay, 2),
                                "error": str(e)
                            }
                        }
                    )
                    time.sleep(delay)

                except Exception as e:
                    # Non-retryable exception, raise immediately
                    logger.error(
                        "Function failed with non-retryable error",
                        extra={
                            "extra_fields": {
                                "function": func.__name__,
                                "error": str(e),
                                "exception_type": type(e).__name__
                            }
                        },
                        exc_info=True  # Include full traceback for debugging
                    )
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return sync_wrapper

    return decorator


def async_retry_on_failure(
    config: Optional[RetryConfig] = None,
    **config_kwargs
) -> Callable:
    """
    Async decorator to add retry logic with exponential backoff.

    Args:
        config: RetryConfig instance (optional)
        **config_kwargs: Arguments to create RetryConfig if not provided

    Example:
        ```python
        @async_retry_on_failure(max_retries=5, initial_delay=0.5)
        async def search(query, user, limit=10):
            # May fail with network errors
            return await client.search(...)
        ```
    """
    if config is None:
        config = RetryConfig(**config_kwargs)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except config.retry_on as e:
                    last_exception = e

                    if attempt >= config.max_retries:
                        # Max retries reached, raise
                        logger.error(
                            "Async function failed after max retries",
                            extra={
                                "extra_fields": {
                                    "function": func.__name__,
                                    "max_retries": config.max_retries,
                                    "error": str(e)
                                }
                            }
                        )
                        raise

                    # Calculate delay and retry
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        "Retrying async function after failure",
                        extra={
                            "extra_fields": {
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": config.max_retries + 1,
                                "delay_seconds": round(delay, 2),
                                "error": str(e)
                            }
                        }
                    )
                    await asyncio.sleep(delay)

                except Exception as e:
                    # Non-retryable exception, raise immediately
                    logger.error(
                        "Async function failed with non-retryable error",
                        extra={
                            "extra_fields": {
                                "function": func.__name__,
                                "error": str(e),
                                "exception_type": type(e).__name__
                            }
                        }
                    )
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return async_wrapper

    return decorator


class RetryableOperation:
    """
    Context manager for retry operations.

    Useful when you want to wrap a block of code with retry logic
    instead of decorating a function.

    Example:
        ```python
        config = RetryConfig(max_retries=3)

        with RetryableOperation(config) as retry:
            while retry.should_continue():
                try:
                    result = client.search(query)
                    retry.success()
                    break
                except ConnectionError as e:
                    retry.failed(e)
        ```
    """

    def __init__(self, config: Optional[RetryConfig] = None, **config_kwargs):
        if config is None:
            config = RetryConfig(**config_kwargs)
        self.config = config
        self.attempt = 0
        self.last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Suppress retryable exceptions if we haven't exceeded max retries
        if exc_type and issubclass(exc_type, self.config.retry_on):
            if self.attempt < self.config.max_retries:
                return True  # Suppress exception
        return False  # Don't suppress

    def should_continue(self) -> bool:
        """Check if we should continue retrying."""
        return self.attempt <= self.config.max_retries

    def failed(self, exception: Exception):
        """Mark current attempt as failed."""
        self.last_exception = exception

        if isinstance(exception, self.config.retry_on):
            if self.attempt < self.config.max_retries:
                delay = self.config.calculate_delay(self.attempt)
                logger.warning(
                    "Retrying operation after failure",
                    extra={
                        "extra_fields": {
                            "attempt": self.attempt + 1,
                            "max_attempts": self.config.max_retries + 1,
                            "delay_seconds": round(delay, 2),
                            "error": str(exception)
                        }
                    }
                )
                time.sleep(delay)
                self.attempt += 1
            else:
                logger.error(
                    "Operation failed after max retries",
                    extra={
                        "extra_fields": {
                            "max_retries": self.config.max_retries,
                            "error": str(exception)
                        }
                    }
                )
                raise exception
        else:
            # Non-retryable exception
            logger.error(
                "Operation failed with non-retryable error",
                extra={
                    "extra_fields": {
                        "error": str(exception),
                        "exception_type": type(exception).__name__
                    }
                }
            )
            raise exception

    def success(self):
        """Mark operation as successful."""
        self.last_exception = None
