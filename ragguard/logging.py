"""
Structured logging for RAGGuard with JSON formatting and contextual information.

Example:
    ```python
    from ragguard.logging import get_logger, add_log_context

    logger = get_logger(__name__)

    # Add context for current request
    with add_log_context(request_id="req-123", user_id="alice"):
        logger.info("Processing search", query="test", limit=10)
        # Output: {"timestamp": "2024-01-15T10:30:00.000Z", "level": "INFO",
        #          "message": "Processing search", "request_id": "req-123",
        #          "user_id": "alice", "query": "test", "limit": 10}
    ```
"""

import json
import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

# Thread-local/async-local context storage
# Note: The empty dict default is safe because we always copy before modifying
_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})  # noqa: B039


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs as single-line JSON objects with:
    - timestamp: ISO 8601 format with timezone
    - level: Log level name
    - logger: Logger name
    - message: Log message
    - Any additional fields from extra dict or log context
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_logger_name: bool = True,
        include_thread_info: bool = False,
        timestamp_format: str = "iso"
    ):
        """
        Initialize the structured formatter.

        Args:
            include_timestamp: Include timestamp in output (default: True)
            include_logger_name: Include logger name in output (default: True)
            include_thread_info: Include thread/process info (default: False)
            timestamp_format: Format for timestamp - "iso" or "unix" (default: "iso")
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_logger_name = include_logger_name
        self.include_thread_info = include_thread_info
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string representation of log record
        """
        # Build base log entry
        log_entry: Dict[str, Any] = {}

        # Add timestamp
        if self.include_timestamp:
            if self.timestamp_format == "unix":
                log_entry["timestamp"] = record.created
            else:  # iso format
                dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
                log_entry["timestamp"] = dt.isoformat()

        # Add log level
        log_entry["level"] = record.levelname

        # Add logger name
        if self.include_logger_name:
            log_entry["logger"] = record.name

        # Add message
        log_entry["message"] = record.getMessage()

        # Add thread/process info if requested
        if self.include_thread_info:
            log_entry["thread_id"] = record.thread
            log_entry["thread_name"] = record.threadName
            log_entry["process_id"] = record.process

        # Add file/line info for debugging
        if record.levelno >= logging.WARNING:
            log_entry["file"] = record.pathname
            log_entry["line"] = record.lineno
            log_entry["function"] = record.funcName

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add stack trace if present
        if record.stack_info:
            log_entry["stack_trace"] = self.formatStack(record.stack_info)

        # Merge context from ContextVar
        context = _log_context.get({})
        if context:
            log_entry.update(context)

        # Merge extra fields from log call
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual fields to all log messages.

    This adapter automatically includes fields from the thread-local context
    in every log message.
    """

    def process(self, msg: str, kwargs: Any) -> tuple:
        """
        Process the logging call to add context.

        Args:
            msg: Log message
            kwargs: Keyword arguments

        Returns:
            Tuple of (msg, kwargs) with added context
        """
        # Get current context
        context = _log_context.get({})

        # Extract extra fields from kwargs
        extra = kwargs.get("extra", {})

        # Merge context with extra
        merged_extra = {**context, **extra}

        # Store merged fields in a special attribute
        if merged_extra:
            kwargs["extra"] = {"extra_fields": merged_extra}

        return msg, kwargs


def get_logger(
    name: str,
    level: Union[int, str] = logging.INFO,
    use_structured: bool = True,
    handlers: Optional[list] = None
) -> logging.LoggerAdapter:
    """
    Get or create a logger with optional structured formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        use_structured: Use structured JSON formatting (default: True)
        handlers: Optional list of handlers (default: StreamHandler to stdout)

    Returns:
        Configured logger instance

    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("User logged in", user_id="alice", ip="10.0.0.1")
        ```
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Set level
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

        # Create handlers
        if handlers is None:
            handler = logging.StreamHandler(sys.stdout)

            if use_structured:
                formatter: logging.Formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )

            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            for handler in handlers:
                logger.addHandler(handler)

        # Prevent propagation to root logger
        logger.propagate = False

    # Wrap in context adapter
    return ContextAdapter(logger, {})


@contextmanager
def add_log_context(**context):
    """
    Context manager to add fields to all logs within a scope.

    This is useful for adding request-scoped context like request_id,
    user_id, trace_id, etc.

    Args:
        **context: Key-value pairs to add to log context

    Example:
        ```python
        with add_log_context(request_id="req-123", user_id="alice"):
            logger.info("Processing request")  # Includes request_id and user_id
            process_data()
            logger.info("Request complete")  # Still includes context
        ```
    """
    # Get current context
    current_context = _log_context.get({})

    # Merge with new context
    new_context = {**current_context, **context}

    # Set new context
    token = _log_context.set(new_context)

    try:
        yield
    finally:
        # Restore previous context
        _log_context.reset(token)


def set_log_level(level: Union[int, str], logger_name: Optional[str] = None):
    """
    Set log level for a logger or all RAGGuard loggers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Specific logger name, or None for all RAGGuard loggers

    Example:
        ```python
        # Set all RAGGuard loggers to DEBUG
        set_log_level("DEBUG")

        # Set specific logger to WARNING
        set_log_level("WARNING", "ragguard.retrievers")
        ```
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    if logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    else:
        # Set level for all RAGGuard loggers
        ragguard_logger = logging.getLogger("ragguard")
        ragguard_logger.setLevel(level)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format: str = "structured",
    output: str = "stdout",
    include_timestamp: bool = True,
    include_logger_name: bool = True
):
    """
    Configure global logging settings for RAGGuard.

    Args:
        level: Default log level
        format: Log format - "structured" (JSON) or "text"
        output: Output destination - "stdout" or "stderr"
        include_timestamp: Include timestamp in logs
        include_logger_name: Include logger name in logs

    Example:
        ```python
        # Configure for production (JSON to stdout)
        configure_logging(level="INFO", format="structured")

        # Configure for development (text to stderr)
        configure_logging(level="DEBUG", format="text", output="stderr")
        ```
    """
    # Convert level string to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get root RAGGuard logger
    logger = logging.getLogger("ragguard")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    if output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.StreamHandler(sys.stdout)

    # Set formatter
    if format == "structured":
        formatter: logging.Formatter = StructuredFormatter(
            include_timestamp=include_timestamp,
            include_logger_name=include_logger_name
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False


# Convenience function to get context
def get_log_context() -> Dict[str, Any]:
    """
    Get current log context.

    Returns:
        Dictionary of current context fields
    """
    return _log_context.get({}).copy()


# Convenience function to clear context
def clear_log_context():
    """
    Clear all fields from log context.
    """
    _log_context.set({})


def generate_correlation_id() -> str:
    """
    Generate a unique correlation ID for request tracing.

    Returns:
        A unique identifier string (UUID4)

    Example:
        ```python
        correlation_id = generate_correlation_id()
        with add_log_context(correlation_id=correlation_id):
            logger.info("Starting request")
            # All logs in this context will include correlation_id
        ```
    """
    return str(uuid.uuid4())


@contextmanager
def request_context(
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra_context
):
    """
    Context manager for request-scoped logging with automatic correlation ID.

    This is a convenience wrapper around add_log_context that automatically
    generates a correlation ID if one is not provided.

    Args:
        correlation_id: Optional correlation ID (auto-generated if not provided)
        user_id: Optional user ID
        **extra_context: Additional context fields

    Example:
        ```python
        with request_context(user_id="alice"):
            logger.info("Processing search")  # Includes correlation_id and user_id
            results = retriever.search(query, user)
            logger.info("Search complete", result_count=len(results))
        ```
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()

    context = {"correlation_id": correlation_id, **extra_context}
    if user_id is not None:
        context["user_id"] = user_id

    with add_log_context(**context):
        yield correlation_id


__all__ = [
    "ContextAdapter",
    "StructuredFormatter",
    "add_log_context",
    "clear_log_context",
    "configure_logging",
    "generate_correlation_id",
    "get_log_context",
    "get_logger",
    "request_context",
    "set_log_level",
]
