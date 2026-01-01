"""
Custom exceptions for RAGGuard.

Exception Hierarchy:
    RAGGuardError (base)
    ├── PolicyError
    │   ├── PolicyParseError (YAML/JSON parsing failed)
    │   └── PolicyValidationError (schema validation failed)
    ├── PolicyEvaluationError (runtime evaluation failed)
    ├── FilterBuildError
    │   └── UnsupportedConditionError (condition not supported by backend)
    ├── RetrieverError
    │   ├── RetrieverConnectionError (network/connection issues - RETRYABLE)
    │   ├── RetrieverTimeoutError (request timed out - RETRYABLE)
    │   ├── HealthCheckError (health check failed)
    │   ├── BackendError (backend-specific error)
    │   ├── RateLimitError (rate limited - RETRYABLE with backoff)
    │   ├── QuotaExceededError (quota exceeded - NOT RETRYABLE)
    │   └── PermissionError (access denied - NOT RETRYABLE)
    └── ConfigurationError (invalid configuration)

Retry Strategy:
    - RETRYABLE exceptions: RetrieverConnectionError, RetrieverTimeoutError, RateLimitError
    - NOT RETRYABLE: QuotaExceededError, PermissionError, ConfigurationError, PolicyError
"""

from typing import Any, Optional


class RAGGuardError(Exception):
    """Base exception for all RAGGuard errors."""

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        self.message = message
        super().__init__(message, *args)

    def __str__(self) -> str:
        return self.message or super().__str__()


# =============================================================================
# Policy Exceptions
# =============================================================================

class PolicyError(RAGGuardError):
    """Raised when there's an issue with policy definition or parsing."""
    pass


class PolicyParseError(PolicyError):
    """Raised when YAML/JSON policy parsing fails."""
    pass


class PolicyValidationError(PolicyError):
    """Raised when policy validation fails."""
    pass


class PolicyEvaluationError(RAGGuardError):
    """Raised when policy evaluation fails at runtime."""
    pass


class ConditionCompilationError(PolicyError):
    """
    Raised when a policy condition cannot be compiled.

    This typically indicates a syntax error or unsupported expression
    in a policy condition string.
    """

    def __init__(
        self,
        condition: str,
        rule_name: str = "",
        reason: str = "",
        cause: Optional[Exception] = None
    ) -> None:
        self.condition = condition
        self.rule_name = rule_name
        self.reason = reason
        self.cause = cause

        message = f"Failed to compile condition: '{condition}'"
        if rule_name:
            message += f" in rule '{rule_name}'"
        if reason:
            message += f": {reason}"
        if cause:
            message += f" (caused by: {type(cause).__name__}: {cause})"
        super().__init__(message)


# =============================================================================
# Filter Exceptions
# =============================================================================

class FilterBuildError(RAGGuardError):
    """Raised when building database-specific filters fails."""
    pass


class UnsupportedConditionError(FilterBuildError):
    """Raised when a condition is not supported by the backend."""

    def __init__(self, condition: str, backend: str, reason: str = ""):
        self.condition = condition
        self.backend = backend
        self.reason = reason
        message = f"Condition '{condition}' not supported by {backend}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


# =============================================================================
# Retriever Exceptions
# =============================================================================

class RetrieverError(RAGGuardError):
    """Base exception for retrieval operations."""
    pass


class RetrieverConnectionError(RetrieverError):
    """
    Raised when connection to vector database fails.

    This is a RETRYABLE error - the operation may succeed on retry.
    Common causes: network issues, server temporarily unavailable.
    """
    retryable = True

    def __init__(self, backend: str, message: str = "", cause: Optional[Exception] = None) -> None:
        self.backend = backend
        self.cause = cause
        full_message = f"{backend} connection failed"
        if message:
            full_message += f": {message}"
        if cause:
            full_message += f" (caused by: {type(cause).__name__}: {cause})"
        super().__init__(full_message)


class RetrieverTimeoutError(RetrieverError):
    """
    Raised when a retrieval operation times out.

    This is a RETRYABLE error - the operation may succeed with a longer timeout.
    """
    retryable = True

    def __init__(self, backend: str, operation: str = "operation", timeout: Optional[float] = None) -> None:
        self.backend = backend
        self.operation = operation
        self.timeout = timeout
        message = f"{backend} {operation} timed out"
        if timeout is not None:
            message += f" after {timeout}s"
        super().__init__(message)


class HealthCheckError(RetrieverError):
    """
    Raised when a health check fails.

    May be retryable depending on the underlying cause.
    """

    def __init__(self, backend: str, message: str = "", cause: Optional[Exception] = None) -> None:
        self.backend = backend
        self.cause = cause
        full_message = f"{backend} health check failed"
        if message:
            full_message += f": {message}"
        if cause:
            full_message += f" (caused by: {type(cause).__name__}: {cause})"
        super().__init__(full_message)


class BackendError(RetrieverError):
    """
    Raised for backend-specific errors.

    Wraps errors from the underlying vector database client.
    """

    def __init__(self, backend: str, message: str = "", cause: Optional[Exception] = None) -> None:
        self.backend = backend
        self.cause = cause
        full_message = f"{backend} error"
        if message:
            full_message += f": {message}"
        if cause:
            full_message += f" (caused by: {type(cause).__name__}: {cause})"
        super().__init__(full_message)


class RateLimitError(RetrieverError):
    """
    Raised when rate limited by the backend.

    This is a RETRYABLE error - retry with exponential backoff.
    """
    retryable = True

    def __init__(self, backend: str, retry_after: Optional[float] = None) -> None:
        self.backend = backend
        self.retry_after = retry_after
        message = f"{backend} rate limit exceeded"
        if retry_after is not None:
            message += f", retry after {retry_after}s"
        super().__init__(message)


class QuotaExceededError(RetrieverError):
    """
    Raised when quota is exceeded.

    This is NOT RETRYABLE - requires quota increase or billing action.
    """
    retryable = False

    def __init__(self, backend: str, message: str = ""):
        self.backend = backend
        full_message = f"{backend} quota exceeded"
        if message:
            full_message += f": {message}"
        super().__init__(full_message)


class RetrieverPermissionError(RetrieverError):
    """
    Raised when access is denied.

    This is NOT RETRYABLE - requires credential or permission changes.
    """
    retryable = False

    def __init__(self, backend: str, resource: str = "", message: str = ""):
        self.backend = backend
        self.resource = resource
        full_message = f"{backend} permission denied"
        if resource:
            full_message += f" for {resource}"
        if message:
            full_message += f": {message}"
        super().__init__(full_message)


# =============================================================================
# Audit Exceptions
# =============================================================================

class AuditLogError(RAGGuardError):
    """
    Raised when audit logging fails and fail_on_audit_error=True.

    This exception is only raised when the retriever is configured with
    fail_on_audit_error=True (fail-closed mode). By default, audit failures
    are logged as warnings and do not interrupt operations (fail-open mode).

    Use fail-closed mode for compliance-critical applications where
    missing audit logs are unacceptable.
    """

    def __init__(self, message: str = "", cause: Optional[Exception] = None) -> None:
        self.cause = cause
        full_message = "Audit logging failed"
        if message:
            full_message += f": {message}"
        if cause:
            full_message += f" (caused by: {type(cause).__name__}: {cause})"
        super().__init__(full_message)


# =============================================================================
# Configuration Exceptions
# =============================================================================

class ConfigurationError(RAGGuardError):
    """
    Raised when configuration is invalid.

    This is NOT RETRYABLE - requires configuration fix.
    """
    retryable = False

    def __init__(self, message: str = "", parameter: Optional[str] = None) -> None:
        self.parameter = parameter
        if parameter and not message:
            message = f"Invalid configuration for '{parameter}'"
        elif parameter:
            message = f"Configuration error for '{parameter}': {message}"
        super().__init__(message)


# =============================================================================
# Convenience sets for retry logic
# =============================================================================

# Exceptions that should trigger retry with backoff
RETRYABLE_EXCEPTIONS = (
    RetrieverConnectionError,
    RetrieverTimeoutError,
    RateLimitError,
)

# Exceptions that should NOT be retried
NON_RETRYABLE_EXCEPTIONS = (
    QuotaExceededError,
    RetrieverPermissionError,
    ConfigurationError,
    PolicyError,
    PolicyValidationError,
    PolicyParseError,
    ConditionCompilationError,
)
