"""
Enhanced error messages with context and suggestions.

This module provides helper functions to create informative error messages
that include:
- Clear description of what went wrong
- Context about what was attempted
- List of valid options
- Suggestions for how to fix
- Links to documentation (when applicable)
"""

from typing import Any, Dict, List, Optional


class ErrorContext:
    """
    Enhanced error message builder with context and suggestions.

    Example:
        raise RetrieverError(
            ErrorContext("Unsupported backend")
            .with_attempted_value("elasticsearch")
            .with_valid_options(["qdrant", "chromadb", "pgvector"])
            .with_suggestion("Install the backend: pip install ragguard[elasticsearch]")
            .build()
        )
    """

    def __init__(self, error_type: str):
        """
        Initialize error context.

        Args:
            error_type: Brief description of the error (e.g., "Unsupported backend")
        """
        self.error_type = error_type
        self.attempted_value: Optional[str] = None
        self.valid_options: Optional[List[str]] = None
        self.suggestions: List[str] = []
        self.context_info: Dict[str, Any] = {}
        self.doc_link: Optional[str] = None

    def with_attempted_value(self, value: str) -> "ErrorContext":
        """Add the value that was attempted."""
        self.attempted_value = value
        return self

    def with_valid_options(self, options: List[str]) -> "ErrorContext":
        """Add list of valid options."""
        self.valid_options = options
        return self

    def with_suggestion(self, suggestion: str) -> "ErrorContext":
        """Add a suggestion for how to fix."""
        self.suggestions.append(suggestion)
        return self

    def with_context(self, key: str, value: Any) -> "ErrorContext":
        """Add additional context information."""
        self.context_info[key] = value
        return self

    def with_doc_link(self, link: str) -> "ErrorContext":
        """Add documentation link."""
        self.doc_link = link
        return self

    def build(self) -> str:
        """Build the formatted error message."""
        parts = [f"\n{'='*70}"]
        parts.append(f"ERROR: {self.error_type}")
        parts.append('='*70)

        # Attempted value
        if self.attempted_value:
            parts.append(f"\nAttempted: '{self.attempted_value}'")

        # Valid options
        if self.valid_options:
            parts.append(f"\nSupported options ({len(self.valid_options)}):")
            for option in sorted(self.valid_options):
                parts.append(f"  - {option}")

        # Additional context
        if self.context_info:
            parts.append("\nContext:")
            for key, value in self.context_info.items():
                parts.append(f"  {key}: {value}")

        # Suggestions
        if self.suggestions:
            parts.append("\nHow to fix:")
            for i, suggestion in enumerate(self.suggestions, 1):
                parts.append(f"  {i}. {suggestion}")

        # Documentation link
        if self.doc_link:
            parts.append(f"\nDocumentation: {self.doc_link}")

        parts.append('='*70 + '\n')
        return '\n'.join(parts)


def unsupported_backend_error(
    backend: str,
    supported_backends: List[str],
    operation: str = "search"
) -> str:
    """
    Create error message for unsupported backend.

    Args:
        backend: The backend that was requested
        supported_backends: List of supported backends
        operation: What operation was being attempted

    Returns:
        Formatted error message
    """
    return (
        ErrorContext("Unsupported Backend")
        .with_attempted_value(backend)
        .with_valid_options(supported_backends)
        .with_context("operation", operation)
        .with_suggestion(
            f"Check if '{backend}' is the correct spelling (case-sensitive)"
        )
        .with_suggestion(
            "Verify that you have the required backend library installed"
        )
        .with_suggestion(
            f"Install with: pip install ragguard[{backend.lower()}]"
        )
        .with_doc_link("https://github.com/maximus242/ragguard#backends")
        .build()
    )


def missing_dependency_error(
    package: str,
    backend: str,
    install_command: Optional[str] = None
) -> str:
    """
    Create error message for missing dependencies.

    Args:
        package: Name of the missing package
        backend: Backend that requires this package
        install_command: Custom install command (optional)

    Returns:
        Formatted error message
    """
    if install_command is None:
        install_command = f"pip install ragguard[{backend}]"

    return (
        ErrorContext("Missing Dependency")
        .with_attempted_value(package)
        .with_context("required_for", backend)
        .with_suggestion(f"Install the package: {install_command}")
        .with_suggestion(
            "Or install all dependencies: pip install 'ragguard[all]'"
        )
        .build()
    )


def validation_error(
    field_name: str,
    value: Any,
    constraint: str,
    actual_value: Optional[str] = None
) -> str:
    """
    Create error message for validation failures.

    Args:
        field_name: Name of the field that failed validation
        value: The value that was provided
        constraint: Description of the constraint
        actual_value: Additional info about actual value (e.g., "actual size: 10MB")

    Returns:
        Formatted error message
    """
    ctx = (
        ErrorContext("Validation Failed")
        .with_context("field", field_name)
        .with_context("constraint", constraint)
    )

    if actual_value:
        ctx.with_context("actual", actual_value)

    return ctx.build()


def policy_compilation_error(
    condition: str,
    error_message: str,
    rule_name: Optional[str] = None
) -> str:
    """
    Create error message for policy compilation failures.

    Args:
        condition: The condition that failed to compile
        error_message: Description of what went wrong
        rule_name: Name of the rule containing this condition

    Returns:
        Formatted error message
    """
    ctx = ErrorContext("Policy Compilation Failed")

    if rule_name:
        ctx.with_context("rule", rule_name)

    ctx.with_context("condition", condition)
    ctx.with_context("error", error_message)

    ctx.with_suggestion("Check the condition syntax")
    ctx.with_suggestion("Ensure field names are correct (user.field, document.field)")
    ctx.with_suggestion(
        "Valid operators: ==, !=, <, >, <=, >=, in, not in, exists, not exists"
    )

    return ctx.build()


def connection_error(
    backend: str,
    connection_string: Optional[str] = None,
    original_error: Optional[str] = None
) -> str:
    """
    Create error message for connection failures.

    Args:
        backend: Backend type
        connection_string: Connection string (sanitized)
        original_error: Original error message

    Returns:
        Formatted error message
    """
    ctx = (
        ErrorContext("Connection Failed")
        .with_context("backend", backend)
    )

    if connection_string:
        ctx.with_context("connection", connection_string)

    if original_error:
        ctx.with_context("error", original_error)

    ctx.with_suggestion("Verify the backend service is running")
    ctx.with_suggestion("Check connection credentials and permissions")
    ctx.with_suggestion("Ensure network connectivity to the backend")
    ctx.with_suggestion("Review backend logs for more details")

    return ctx.build()


def filter_generation_error(
    backend: str,
    policy_rules: int,
    user_context: Dict[str, Any],
    original_error: Optional[str] = None
) -> str:
    """
    Create error message for filter generation failures.

    Args:
        backend: Backend type
        policy_rules: Number of policy rules
        user_context: User context (sanitized)
        original_error: Original error message

    Returns:
        Formatted error message
    """
    ctx = (
        ErrorContext("Filter Generation Failed")
        .with_context("backend", backend)
        .with_context("policy_rules", policy_rules)
        .with_context("user_fields", list(user_context.keys()))
    )

    if original_error:
        ctx.with_context("error", original_error)

    ctx.with_suggestion("Verify policy syntax is correct")
    ctx.with_suggestion("Check that user context contains required fields")
    ctx.with_suggestion("Ensure document fields referenced in policy exist")
    ctx.with_suggestion("Try simpler policy rules to isolate the issue")

    return ctx.build()


def empty_user_context_error(require_id: bool = True) -> str:
    """
    Create error message for empty user context.

    Args:
        require_id: Whether user ID is required

    Returns:
        Formatted error message
    """
    ctx = ErrorContext("Empty User Context")

    if require_id:
        ctx.with_suggestion("User context must include an 'id' field")

    ctx.with_suggestion("Provide user context as a dictionary: {'id': 'user123'}")
    ctx.with_suggestion("Include additional fields required by your policy")
    ctx.with_suggestion(
        "Example: {'id': 'alice', 'department': 'engineering', 'role': 'admin'}"
    )

    return ctx.build()


def field_not_found_error(
    field_path: str,
    context_type: str,  # "user" or "document"
    available_fields: List[str]
) -> str:
    """
    Create error message for missing field in context.

    Args:
        field_path: The field path that was accessed (e.g., "user.department")
        context_type: Whether this is user or document context
        available_fields: List of available fields in the context

    Returns:
        Formatted error message
    """
    return (
        ErrorContext("Field Not Found")
        .with_attempted_value(field_path)
        .with_context("context_type", context_type)
        .with_valid_options(available_fields)
        .with_suggestion(f"Check that {context_type} context includes this field")
        .with_suggestion("Use 'exists' operator if field is optional")
        .with_suggestion(
            f"Example: {field_path} exists AND {field_path} == 'value'"
        )
        .build()
    )


# Supported backends (keep this updated)
SUPPORTED_BACKENDS = [
    # Vector databases
    "qdrant",
    "chromadb",
    "pgvector",
    "weaviate",
    "pinecone",
    "faiss",
    "milvus",
    "elasticsearch",
    "opensearch",
    "azure_search",
    # Graph databases
    "neo4j",
    "neptune",
    "tigergraph",
    "arangodb",
]
