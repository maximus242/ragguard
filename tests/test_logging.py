"""
Tests for structured logging functionality.
"""

import io
import json
import logging
import sys

import pytest

from ragguard.logging import (
    StructuredFormatter,
    add_log_context,
    clear_log_context,
    configure_logging,
    get_log_context,
    get_logger,
    set_log_level,
)


def test_get_logger():
    """Test getting a logger."""
    logger = get_logger(__name__)
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")


def test_structured_formatter():
    """Test structured JSON formatter."""
    # Create a string buffer to capture output
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    formatter = StructuredFormatter()
    handler.setFormatter(formatter)

    logger = logging.getLogger("test_logger")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Log a message
    logger.info("Test message", extra={"extra_fields": {"user_id": "alice", "count": 10}})

    # Get output
    log_output = output.getvalue()

    # Parse JSON
    log_entry = json.loads(log_output)

    # Verify structure
    assert log_entry["level"] == "INFO"
    assert log_entry["message"] == "Test message"
    assert log_entry["user_id"] == "alice"
    assert log_entry["count"] == 10
    assert "timestamp" in log_entry


def test_log_context():
    """Test adding log context."""
    # Clear any existing context
    clear_log_context()

    # Add context
    with add_log_context(request_id="req-123", user_id="alice"):
        context = get_log_context()
        assert context["request_id"] == "req-123"
        assert context["user_id"] == "alice"

    # Context should be cleared after exiting
    context = get_log_context()
    assert "request_id" not in context
    assert "user_id" not in context


def test_nested_log_context():
    """Test nested log contexts."""
    clear_log_context()

    with add_log_context(request_id="req-123"):
        assert get_log_context()["request_id"] == "req-123"

        with add_log_context(user_id="alice", action="search"):
            context = get_log_context()
            assert context["request_id"] == "req-123"
            assert context["user_id"] == "alice"
            assert context["action"] == "search"

        # Inner context removed
        context = get_log_context()
        assert context["request_id"] == "req-123"
        assert "user_id" not in context

    # All context cleared
    assert get_log_context() == {}


def test_configure_logging():
    """Test global logging configuration."""
    # Configure with structured format
    configure_logging(level="DEBUG", format="structured")

    logger = logging.getLogger("ragguard.test")
    assert logger.level <= logging.DEBUG


def test_set_log_level():
    """Test setting log level."""
    set_log_level("INFO")

    logger = logging.getLogger("ragguard")
    assert logger.level == logging.INFO

    set_log_level("DEBUG", "ragguard.test")
    test_logger = logging.getLogger("ragguard.test")
    assert test_logger.level == logging.DEBUG


def test_logging_with_extra_fields():
    """Test logging with additional fields."""
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    formatter = StructuredFormatter()
    handler.setFormatter(formatter)

    logger = logging.getLogger("test_extra")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Log with extra fields
    logger.info(
        "Search completed",
        extra={"extra_fields": {
            "user_id": "alice",
            "query": "test",
            "results": 10,
            "latency_ms": 45
        }}
    )

    log_entry = json.loads(output.getvalue())

    assert log_entry["message"] == "Search completed"
    assert log_entry["user_id"] == "alice"
    assert log_entry["query"] == "test"
    assert log_entry["results"] == 10
    assert log_entry["latency_ms"] == 45


def test_error_logging_includes_traceback():
    """Test that error logs include traceback."""
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    formatter = StructuredFormatter()
    handler.setFormatter(formatter)

    logger = logging.getLogger("test_error")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)

    # Log an error with exception info
    try:
        raise ValueError("Test error")
    except Exception:
        logger.error("An error occurred", exc_info=True)

    log_entry = json.loads(output.getvalue())

    assert log_entry["level"] == "ERROR"
    assert log_entry["message"] == "An error occurred"
    assert "exception" in log_entry
    assert "ValueError: Test error" in log_entry["exception"]


def test_clear_log_context():
    """Test clearing log context."""
    with add_log_context(user_id="alice", request_id="123"):
        assert len(get_log_context()) > 0

        clear_log_context()
        assert get_log_context() == {}


def test_timestamp_formats():
    """Test different timestamp formats."""
    # ISO format
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    formatter = StructuredFormatter(timestamp_format="iso")
    handler.setFormatter(formatter)

    logger = logging.getLogger("test_timestamp_iso")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Test")

    log_entry = json.loads(output.getvalue())
    assert "timestamp" in log_entry
    assert "T" in log_entry["timestamp"]  # ISO format includes T

    # Unix format
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    formatter = StructuredFormatter(timestamp_format="unix")
    handler.setFormatter(formatter)

    logger = logging.getLogger("test_timestamp_unix")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Test")

    log_entry = json.loads(output.getvalue())
    assert "timestamp" in log_entry
    assert isinstance(log_entry["timestamp"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
