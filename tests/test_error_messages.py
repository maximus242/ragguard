"""
Tests for improved error messages.
"""

import pytest

from ragguard import Policy
from ragguard.errors import (
    ErrorContext,
    connection_error,
    empty_user_context_error,
    field_not_found_error,
    filter_generation_error,
    missing_dependency_error,
    policy_compilation_error,
    unsupported_backend_error,
    validation_error,
)
from ragguard.exceptions import PolicyEvaluationError, RetrieverError
from ragguard.policy.engine import PolicyEngine
from ragguard.validation import InputValidator, ValidationConfig


def test_error_context_basic():
    """Test basic error context building."""
    error = ErrorContext("Test Error").build()

    assert "Test Error" in error
    assert "=" in error  # Check formatting


def test_error_context_with_attempted_value():
    """Test error with attempted value."""
    error = (
        ErrorContext("Invalid Option")
        .with_attempted_value("foobar")
        .build()
    )

    assert "foobar" in error
    assert "Attempted" in error


def test_error_context_with_valid_options():
    """Test error with valid options list."""
    error = (
        ErrorContext("Invalid Choice")
        .with_valid_options(["option1", "option2", "option3"])
        .build()
    )

    assert "option1" in error
    assert "option2" in error
    assert "option3" in error
    assert "Supported options" in error


def test_error_context_with_suggestions():
    """Test error with suggestions."""
    error = (
        ErrorContext("Configuration Error")
        .with_suggestion("Check your config file")
        .with_suggestion("Verify environment variables")
        .build()
    )

    assert "How to fix" in error
    assert "Check your config file" in error
    assert "Verify environment variables" in error


def test_error_context_with_context_info():
    """Test error with additional context."""
    error = (
        ErrorContext("Operation Failed")
        .with_context("operation", "search")
        .with_context("backend", "qdrant")
        .build()
    )

    assert "Context:" in error
    assert "operation: search" in error
    assert "backend: qdrant" in error


def test_error_context_with_doc_link():
    """Test error with documentation link."""
    error = (
        ErrorContext("Feature Not Available")
        .with_doc_link("https://docs.example.com/features")
        .build()
    )

    assert "Documentation:" in error
    assert "https://docs.example.com/features" in error


def test_unsupported_backend_error():
    """Test unsupported backend error message."""
    error = unsupported_backend_error(
        backend="unknown_db",
        supported_backends=["qdrant", "chromadb", "pgvector"],
        operation="search"
    )

    assert "unknown_db" in error
    assert "qdrant" in error
    assert "chromadb" in error
    assert "pgvector" in error
    assert "pip install" in error


def test_missing_dependency_error():
    """Test missing dependency error message."""
    error = missing_dependency_error(
        package="qdrant-client",
        backend="qdrant"
    )

    assert "qdrant-client" in error
    assert "pip install" in error
    assert "ragguard[qdrant]" in error


def test_validation_error():
    """Test validation error message."""
    error = validation_error(
        field_name="query_vector",
        value=[0.1] * 2000,
        constraint="max dimension: 1536",
        actual_value="2000 dimensions"
    )

    assert "query_vector" in error
    assert "max dimension: 1536" in error
    assert "2000 dimensions" in error


def test_policy_compilation_error():
    """Test policy compilation error message."""
    error = policy_compilation_error(
        condition="user.dept == document.dept",
        error_message="Field 'dept' does not exist",
        rule_name="department-access"
    )

    assert "user.dept == document.dept" in error
    assert "Field 'dept' does not exist" in error
    assert "department-access" in error
    assert "Valid operators" in error


def test_connection_error():
    """Test connection error message."""
    error = connection_error(
        backend="qdrant",
        connection_string="http://localhost:6333",
        original_error="Connection refused"
    )

    assert "qdrant" in error
    assert "http://localhost:6333" in error
    assert "Connection refused" in error
    assert "Verify the backend service is running" in error


def test_filter_generation_error():
    """Test filter generation error message."""
    error = filter_generation_error(
        backend="qdrant",
        policy_rules=5,
        user_context={"id": "alice", "department": "engineering"},
        original_error="Missing field: role"
    )

    assert "qdrant" in error
    assert "5" in error  # policy_rules count
    assert "id" in error
    assert "department" in error
    assert "Missing field: role" in error


def test_empty_user_context_error():
    """Test empty user context error message."""
    error = empty_user_context_error(require_id=True)

    assert "Empty User Context" in error
    assert "'id' field" in error
    assert "Example:" in error
    assert "alice" in error  # Example user


def test_field_not_found_error():
    """Test field not found error message."""
    error = field_not_found_error(
        field_path="user.department",
        context_type="user",
        available_fields=["id", "name", "email", "role"]
    )

    assert "user.department" in error
    assert "id" in error
    assert "name" in error
    assert "email" in error
    assert "role" in error
    assert "exists" in error  # Suggestion to use EXISTS operator


def test_policy_engine_unsupported_backend_error():
    """Test that policy engine raises improved error for unsupported backend."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"id": "alice"}

    with pytest.raises(PolicyEvaluationError) as exc_info:
        engine.to_filter(user, "unknown_backend")

    error_msg = str(exc_info.value)
    assert "unknown_backend" in error_msg
    assert "Supported options" in error_msg
    assert "qdrant" in error_msg
    assert "pip install" in error_msg


def test_validator_empty_user_context_error():
    """Test that validator raises improved error for empty user context."""
    validator = InputValidator(ValidationConfig())

    with pytest.raises(RetrieverError) as exc_info:
        validator.validate_user_context({})

    error_msg = str(exc_info.value)
    assert "Empty User Context" in error_msg
    assert "'id' field" in error_msg
    assert "Example:" in error_msg


def test_error_message_formatting():
    """Test that error messages are well formatted and readable."""
    error = (
        ErrorContext("Test Error")
        .with_attempted_value("bad_value")
        .with_valid_options(["good1", "good2"])
        .with_suggestion("Try using a valid option")
        .with_context("field", "backend")
        .build()
    )

    # Check it has proper structure
    assert error.startswith("\n")
    assert "=" * 70 in error
    assert "ERROR:" in error
    assert "Attempted:" in error
    assert "Supported options" in error
    assert "How to fix:" in error


def test_error_context_chaining():
    """Test that ErrorContext methods can be chained fluently."""
    # Should not raise any errors
    error = (
        ErrorContext("Chaining Test")
        .with_attempted_value("test")
        .with_valid_options(["a", "b"])
        .with_suggestion("First suggestion")
        .with_suggestion("Second suggestion")
        .with_context("key1", "value1")
        .with_context("key2", "value2")
        .with_doc_link("https://example.com")
        .build()
    )

    assert isinstance(error, str)
    assert len(error) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
